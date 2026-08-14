"""
main.py
=======

Forza Assistents
Orquestrador principal do pipeline ADAS.

Arquitetura:

    ScreenCapture
        │
        ▼
    YOLOP
        │
        ▼
    LaneTracker
        │
        ├──────────────► LaneGeometry
        │                    │
        │                    ▼
        │               LaneGeometryResult
        │
        ▼
    LaneModel
        │
        ▼
    LaneProjection
        │
        ▼
    LaneAssignment
        │
        ├──────────────► ADASStateEstimator
        │                    │
        │                    ▼
        │               ADASStateResult
        │
        ▼
    ADASDisplay

Princípios
==========

Este arquivo SOMENTE orquestra os módulos.

Não executa:

    - inferência;
    - processamento de máscara;
    - tracking;
    - fitting;
    - projeção;
    - classificação ADAS;
    - controle G29.

Segurança
=========

    MONITOR é o modo padrão.

    Controle físico permanece desabilitado.

    Nenhum comando é enviado ao G29.

Visualização
============

    visualization.adas_display.ADASDisplay

O display recebe resultados já calculados pelo pipeline.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np

from config import (
    CONFIG,
    CAPTURE,
    DEBUG,
    PERFORMANCE,
    SAFETY,
    VISUALIZATION,
    YOLOP,
    LANE_TRACKER,
    LANE_GEOMETRY,
    LANE_MODEL,
    LANE_PROJECTION,
    LANE_ASSIGNMENT,
    ADAS,
    ensure_directories,
    validate_config,
)

from capture.screen_capture import ScreenCapture

from vision.yolop_detector import (
    YOLOPLaneDetector,
    LaneDetectionResult,
    create_default_detector,
)

from vision.lane_tracker import (
    LaneTracker,
    LaneTrackingResult,
)

from vision.lane_geometry import (
    LaneGeometry,
    LaneGeometryResult,
)

from vision.lane_model import (
    LaneModel,
    build_lane_model,
)

from vision.lane_projection import (
    LaneProjectionEngine,
)

from vision.lane_assignment import (
    LaneAssignment,
    LaneAssignmentResult,
)

from vision.adas_state import (
    ADASStateEstimator,
    ADASStateResult,
)

from visualization.adas_display import (
    ADASDisplay,
    ADASDisplayConfig,
)


# =============================================================================
# LOGGING
# =============================================================================


def setup_logging() -> logging.Logger:
    """Inicializa o logger principal."""

    ensure_directories()

    logger = logging.getLogger(
        "forza_assistents"
    )

    if logger.handlers:
        return logger

    level = getattr(
        logging,
        CONFIG.logging.level.upper(),
        logging.INFO,
    )

    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    )

    if CONFIG.logging.log_to_console:

        console = logging.StreamHandler()
        console.setFormatter(formatter)

        logger.addHandler(console)

    if CONFIG.logging.log_to_file:

        path = (
            CONFIG.logging.directory
            / CONFIG.logging.filename
        )

        file_handler = logging.FileHandler(
            path,
            encoding="utf-8",
        )

        file_handler.setFormatter(
            formatter
        )

        logger.addHandler(
            file_handler
        )

    logger.propagate = False

    return logger


LOGGER = setup_logging()


# =============================================================================
# ESTATÍSTICAS
# =============================================================================


@dataclass
class RuntimeStatistics:
    """Métricas do pipeline em tempo real."""

    frame_index: int = 0
    total_frames: int = 0

    valid_detections: int = 0
    invalid_detections: int = 0

    valid_geometry: int = 0
    valid_assignment: int = 0
    valid_adas: int = 0

    lane_lost_frames: int = 0

    capture_time_ms: float = 0.0
    detection_time_ms: float = 0.0
    tracking_time_ms: float = 0.0
    geometry_time_ms: float = 0.0
    model_time_ms: float = 0.0
    projection_time_ms: float = 0.0
    assignment_time_ms: float = 0.0
    adas_time_ms: float = 0.0

    total_time_ms: float = 0.0

    fps: float = 0.0

    _last_timestamp: float = 0.0

    def update_fps(self) -> None:
        now = time.perf_counter()

        if self._last_timestamp > 0.0:

            delta = (
                now
                - self._last_timestamp
            )

            if delta > 0.0:

                instant_fps = 1.0 / delta

                if self.fps <= 0.0:
                    self.fps = instant_fps

                else:
                    self.fps = (
                        self.fps * 0.90
                        + instant_fps * 0.10
                    )

        self._last_timestamp = now


# =============================================================================
# RESULTADO DO PIPELINE
# =============================================================================


@dataclass
class PipelineResult:
    """Resultado completo de um frame."""

    frame: Optional[np.ndarray]

    detection: Optional[
        LaneDetectionResult
    ] = None

    tracking: Optional[
        LaneTrackingResult
    ] = None

    geometry: Optional[
        LaneGeometryResult
    ] = None

    models: list[LaneModel] = field(
        default_factory=list
    )

    projections: list[Any] = field(
        default_factory=list
    )

    assignment: Optional[
        LaneAssignmentResult
    ] = None

    adas: Optional[
        ADASStateResult
    ] = None

    valid: bool = False

    frame_index: int = 0

    timestamp: float = 0.0

    statistics: Optional[
        RuntimeStatistics
    ] = None


# =============================================================================
# APLICAÇÃO
# =============================================================================


class ForzaAssistents:
    """
    Orquestrador principal do Forza Assistents.

    Nenhum módulo de percepção é implementado aqui.
    """

    def __init__(self) -> None:

        validate_config()

        self.mode = CONFIG.runtime_mode

        self.running = False
        self._initialized = False

        self.frame_index = 0

        self.statistics = (
            RuntimeStatistics()
        )

        self.capture: Optional[
            ScreenCapture
        ] = None

        self.detector: Optional[
            YOLOPLaneDetector
        ] = None

        self.tracker: Optional[
            LaneTracker
        ] = None

        self.geometry: Optional[
            LaneGeometry
        ] = None

        self.projection: Optional[
            LaneProjectionEngine
        ] = None

        self.assignment: Optional[
            LaneAssignment
        ] = None

        self.adas: Optional[
            ADASStateEstimator
        ] = None

        self.display: Optional[
            ADASDisplay
        ] = None

        self.last_result: Optional[
            PipelineResult
        ] = None

        self._geometry_shape: Optional[
            tuple[int, int]
        ] = None

        self._display_enabled = bool(
            VISUALIZATION.enabled
        )

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def initialize(self) -> None:
        """Inicializa todos os componentes do pipeline."""

        if self._initialized:
            return

        LOGGER.info(
            "=========================================="
        )
        LOGGER.info(
            "FORZA ASSISTENTS"
        )
        LOGGER.info(
            "Initializing ADAS pipeline..."
        )
        LOGGER.info(
            "Runtime mode: %s",
            self.mode.value,
        )
        LOGGER.info(
            "Physical control: %s",
            (
                "ENABLED"
                if SAFETY.enable_control
                else "DISABLED"
            ),
        )

        # ---------------------------------------------------------------------
        # CAPTURE
        # ---------------------------------------------------------------------

        backend = str(
            CAPTURE.backend
        ).lower()

        if backend == "dxcam":
            backend = "dxgi"

        self.capture = ScreenCapture(
            region=(
                CAPTURE.roi
                if CAPTURE.use_roi
                else None
            ),
            target_fps=CAPTURE.target_fps,
            backend=backend,
            output_color=(
                CAPTURE.output_color_format
            ),
            max_buffer_size=(
                PERFORMANCE.max_frame_queue
            ),
        )

        if not self.capture.initialize():

            raise RuntimeError(
                "Falha ao inicializar "
                "ScreenCapture."
            )

        self.capture.start()

        LOGGER.info(
            "ScreenCapture: READY"
        )

        # ---------------------------------------------------------------------
        # YOLOP
        # ---------------------------------------------------------------------

        self.detector = (
            create_default_detector(
                model_path=YOLOP.model_path,
                input_width=YOLOP.input_width,
                input_height=YOLOP.input_height,
                lane_threshold=(
                    YOLOP.lane_confidence_threshold
                ),
                min_points_per_lane=(
                    YOLOP.minimum_lane_points
                ),
                max_lanes=YOLOP.max_lanes,
            )
        )

        if not self.detector.load_model():

            raise RuntimeError(
                "Falha ao carregar YOLOP: "
                f"{self.detector.last_error}"
            )

        LOGGER.info(
            "YOLOP: READY | device=%s",
            self.detector.get_device_name(),
        )

        # ---------------------------------------------------------------------
        # TRACKER
        #
        # O contrato atual do LaneTracker utiliza:
        #
        # max_lanes
        # history_size
        # min_points
        # match_distance
        # max_missed_frames
        # min_stable_frames
        # confidence_decay
        # ---------------------------------------------------------------------

        self.tracker = LaneTracker(
            max_lanes=(
                LANE_TRACKER.max_tracks
            ),
            history_size=(
                LANE_TRACKER.history_size
            ),
            min_points=max(
                1,
                YOLOP.minimum_lane_points,
            ),
            match_distance=(
                LANE_TRACKER.association_distance
            ),
            max_missed_frames=(
                LANE_TRACKER.max_lost_frames
            ),
            min_stable_frames=(
                LANE_TRACKER.min_stable_frames
            ),
            confidence_decay=(
                LANE_TRACKER.confidence_decay
            ),
        )

        LOGGER.info(
            "LaneTracker: READY"
        )

        # ---------------------------------------------------------------------
        # PROJECTION
        # ---------------------------------------------------------------------

        self.projection = (
            LaneProjectionEngine(
                min_points=(
                    LANE_PROJECTION.minimum_points
                ),
                min_confidence=(
                    LANE_PROJECTION.minimum_confidence
                ),
                max_projection_distance=(
                    LANE_PROJECTION.max_projection_distance
                ),
            )
        )

        LOGGER.info(
            "LaneProjection: READY"
        )

        # ---------------------------------------------------------------------
        # ASSIGNMENT
        #
        # Recebe LaneModel.
        # ---------------------------------------------------------------------

        self.assignment = LaneAssignment(
            max_lanes=max(
                2,
                LANE_TRACKER.max_tracks,
            ),
            min_lane_width_px=(
                LANE_GEOMETRY.min_lane_width
            ),
            max_lane_width_px=(
                LANE_GEOMETRY.max_lane_width
            ),
            vehicle_x_ratio=(
                LANE_ASSIGNMENT.center_reference_ratio
            ),
            expected_lane_width=(
                LANE_ASSIGNMENT.expected_lane_width
            ),
            lane_width_tolerance=(
                LANE_ASSIGNMENT.lane_width_tolerance
            ),
            minimum_confidence=(
                LANE_ASSIGNMENT.minimum_confidence
            ),
            maximum_lateral_offset_ratio=(
                LANE_ASSIGNMENT
                .maximum_lateral_offset_ratio
            ),
            enable_multi_lane_assignment=(
                LANE_ASSIGNMENT
                .enable_multi_lane_assignment
            ),
            max_left_lanes=(
                LANE_ASSIGNMENT.max_left_lanes
            ),
            max_right_lanes=(
                LANE_ASSIGNMENT.max_right_lanes
            ),
        )

        LOGGER.info(
            "LaneAssignment: READY"
        )

        # ---------------------------------------------------------------------
        # ADAS
        #
        # Assignment é passado explicitamente para o estimator.
        # ---------------------------------------------------------------------

        self.adas = ADASStateEstimator(
            warning_threshold=(
                ADAS.warning_threshold
            ),
            departure_threshold=(
                ADAS.critical_threshold
            ),
            heading_warning_threshold=(
                ADAS.heading_warning_threshold
            ),
            min_confidence=(
                ADAS.minimum_confidence
            ),
        )

        LOGGER.info(
            "ADASStateEstimator: READY"
        )

        # ---------------------------------------------------------------------
        # DISPLAY
        # ---------------------------------------------------------------------

        if self._display_enabled:

            display_config = (
                ADASDisplayConfig(
                    source_width=(
                        CAPTURE.roi[2]
                        - CAPTURE.roi[0]
                        if CAPTURE.use_roi
                        else 2560
                    ),
                    source_height=(
                        CAPTURE.roi[3]
                        - CAPTURE.roi[1]
                        if CAPTURE.use_roi
                        else 1600
                    ),
                    refresh_hz=30.0,
                )
            )

            self.display = ADASDisplay(
                config=display_config
            )

            try:

                self.display.start(
                    blocking=False
                )

                LOGGER.info(
                    "ADASDisplay: STARTED"
                )

            except Exception as exc:

                LOGGER.warning(
                    "ADASDisplay unavailable: %s",
                    exc,
                )

                self.display = None

        # ---------------------------------------------------------------------
        # GEOMETRY
        #
        # Criada somente após conhecer a resolução real do frame.
        # ---------------------------------------------------------------------

        self.geometry = None
        self._geometry_shape = None

        self._initialized = True

        LOGGER.info(
            "=========================================="
        )
        LOGGER.info(
            "Forza Assistents: INITIALIZED"
        )
        LOGGER.info(
            "=========================================="
        )

    # =========================================================================
    # GEOMETRY
    # =========================================================================

    def _ensure_geometry(
        self,
        frame: np.ndarray,
    ) -> LaneGeometry:

        height, width = frame.shape[:2]

        shape = (
            int(width),
            int(height),
        )

        if (
            self.geometry is not None
            and self._geometry_shape == shape
        ):
            return self.geometry

        LOGGER.info(
            "LaneGeometry: creating %dx%d",
            width,
            height,
        )

        self.geometry = LaneGeometry(
            screen_width=width,
            screen_height=height,
            roi=(
                0,
                0,
                width,
                height,
            ),
            detector_width=(
                self.detector.input_width
                if self.detector is not None
                else YOLOP.input_width
            ),
            detector_height=(
                self.detector.input_height
                if self.detector is not None
                else YOLOP.input_height
            ),
            near_weight=0.75,
            far_weight=0.25,
            min_points=(
                LANE_GEOMETRY.min_points
            ),
            samples=40,
            min_lane_width=(
                LANE_GEOMETRY.min_lane_width
            ),
            max_lane_width=(
                LANE_GEOMETRY.max_lane_width
            ),
            min_observed_span=(
                LANE_GEOMETRY.min_observed_span
            ),
            projection_min_span=180.0,
        )

        self._geometry_shape = shape

        return self.geometry

    # =========================================================================
    # FRAME
    # =========================================================================

    def _get_frame(
        self,
    ) -> Optional[np.ndarray]:

        if self.capture is None:
            return None

        start = time.perf_counter()

        frame = (
            self.capture.get_latest_frame()
        )

        self.statistics.capture_time_ms = (
            time.perf_counter() - start
        ) * 1000.0

        if frame is None:
            return None

        if not isinstance(
            frame,
            np.ndarray,
        ):
            return None

        if frame.ndim != 3:
            return None

        if frame.shape[2] != 3:
            return None

        if frame.size == 0:
            return None

        return frame

    # =========================================================================
    # DETECTION
    # =========================================================================

    def _detect(
        self,
        frame: np.ndarray,
    ) -> LaneDetectionResult:

        if self.detector is None:
            raise RuntimeError(
                "Detector não inicializado."
            )

        start = time.perf_counter()

        result = self.detector.detect(
            frame
        )

        self.statistics.detection_time_ms = (
            time.perf_counter() - start
        ) * 1000.0

        return result

    # =========================================================================
    # TRACKING
    # =========================================================================

    def _track(
        self,
        detection: LaneDetectionResult,
    ) -> LaneTrackingResult:

        if self.tracker is None:
            raise RuntimeError(
                "Tracker não inicializado."
            )

        start = time.perf_counter()

        result = self.tracker.update(
            detection,
            timestamp=time.monotonic(),
        )

        self.statistics.tracking_time_ms = (
            time.perf_counter() - start
        ) * 1000.0

        return result

    # =========================================================================
    # GEOMETRY
    # =========================================================================

    def _calculate_geometry(
        self,
        detection: LaneDetectionResult,
        frame: np.ndarray,
    ) -> LaneGeometryResult:

        geometry = self._ensure_geometry(
            frame
        )

        start = time.perf_counter()

        result = geometry.compute(
            detection
        )

        self.statistics.geometry_time_ms = (
            time.perf_counter() - start
        ) * 1000.0

        return result

    # =========================================================================
    # TRACK EXTRACTION
    # =========================================================================

    @staticmethod
    def _extract_tracks(
        tracking: LaneTrackingResult,
    ) -> list[Any]:

        if tracking is None:
            return []

        active = getattr(
            tracking,
            "active_lanes",
            None,
        )

        if active is not None:
            return list(active)

        lanes = getattr(
            tracking,
            "lanes",
            (),
        )

        return list(lanes)

    # =========================================================================
    # LANE MODELS
    # =========================================================================

    def _build_models(
        self,
        tracking: LaneTrackingResult,
    ) -> list[LaneModel]:

        start = time.perf_counter()

        models: list[LaneModel] = []

        tracks = self._extract_tracks(
            tracking
        )

        for track in tracks:

            if track is None:
                continue

            points = getattr(
                track,
                "points",
                None,
            )

            if not points:
                continue

            track_id = getattr(
                track,
                "track_id",
                0,
            )

            try:
                track_id = int(track_id)
            except (
                TypeError,
                ValueError,
            ):
                track_id = 0

            try:

                model = build_lane_model(
                    lane_id=track_id,
                    points=points,
                    min_points=(
                        LANE_MODEL.minimum_points
                    ),
                    min_confidence=(
                        LANE_MODEL.minimum_confidence
                    ),
                )

            except Exception as exc:

                LOGGER.debug(
                    "LaneModel failed "
                    "track=%s: %s",
                    track_id,
                    exc,
                )

                continue

            if model is None:
                continue

            # -----------------------------------------------------------------
            # Preserva estado temporal do tracker.
            # -----------------------------------------------------------------

            try:
                model.tracked = True
            except Exception:
                pass

            try:
                model.stable = bool(
                    getattr(
                        track,
                        "stable",
                        False,
                    )
                )
            except Exception:
                pass

            if bool(
                getattr(
                    model,
                    "valid",
                    False,
                )
            ):
                models.append(model)

        self.statistics.model_time_ms = (
            time.perf_counter() - start
        ) * 1000.0

        return models

    # =========================================================================
    # PROJECTION
    # =========================================================================

    def _project_models(
        self,
        models: list[LaneModel],
    ) -> list[Any]:

        start = time.perf_counter()

        projections: list[Any] = []

        if self.projection is None:

            self.statistics.projection_time_ms = (
                time.perf_counter() - start
            ) * 1000.0

            return projections

        for model in models:

            try:

                projection = (
                    self.projection.project(
                        model
                    )
                )

            except Exception as exc:

                LOGGER.debug(
                    "LaneProjection failed "
                    "lane=%s: %s",
                    getattr(
                        model,
                        "lane_id",
                        "?",
                    ),
                    exc,
                )

                continue

            if projection is not None:

                projections.append(
                    projection
                )

        self.statistics.projection_time_ms = (
            time.perf_counter() - start
        ) * 1000.0

        return projections

    # =========================================================================
    # ASSIGNMENT
    # =========================================================================

    def _assign(
        self,
        models: list[LaneModel],
        frame: np.ndarray,
    ) -> Optional[
        LaneAssignmentResult
    ]:

        if self.assignment is None:
            return None

        start = time.perf_counter()

        height, width = frame.shape[:2]

        try:

            result = (
                self.assignment.assign(
                    models,
                    frame_width=float(width),
                    frame_height=float(height),
                )
            )

        except Exception as exc:

            LOGGER.debug(
                "LaneAssignment failed: %s",
                exc,
            )

            result = None

        self.statistics.assignment_time_ms = (
            time.perf_counter() - start
        ) * 1000.0

        return result

    # =========================================================================
    # ADAS
    # =========================================================================

    def _adas_update(
        self,
        geometry: Optional[
            LaneGeometryResult
        ],
        assignment: Optional[
            LaneAssignmentResult
        ],
    ) -> Optional[
        ADASStateResult
    ]:

        if self.adas is None:
            return None

        start = time.perf_counter()

        try:

            result = self.adas.update(
                geometry,
                assignment=assignment,
                timestamp=time.monotonic(),
            )

        except Exception as exc:

            LOGGER.debug(
                "ADAS update failed: %s",
                exc,
            )

            result = None

        self.statistics.adas_time_ms = (
            time.perf_counter() - start
        ) * 1000.0

        return result

    # =========================================================================
    # VALIDATION
    # =========================================================================

    @staticmethod
    def _valid(
        value: Any,
    ) -> bool:

        if value is None:
            return False

        return bool(
            getattr(
                value,
                "valid",
                False,
            )
        )

    # =========================================================================
    # DISPLAY UPDATE
    # =========================================================================

    def _update_display(
        self,
        geometry: Optional[
            LaneGeometryResult
        ],
        adas: Optional[
            ADASStateResult
        ],
    ) -> None:

        if self.display is None:
            return

        try:

            self.display.update_from_pipeline(
                geometry=geometry,
                adas_state=adas,
                active=True,
            )

        except Exception as exc:

            LOGGER.debug(
                "ADASDisplay update failed: %s",
                exc,
            )

    # =========================================================================
    # FRAME PROCESSING
    # =========================================================================

    def process_frame(
        self,
        frame: np.ndarray,
    ) -> PipelineResult:

        pipeline_start = (
            time.perf_counter()
        )

        self.frame_index += 1

        self.statistics.frame_index = (
            self.frame_index
        )

        self.statistics.total_frames += 1

        self.statistics.update_fps()

        # ---------------------------------------------------------------------
        # GEOMETRY
        # ---------------------------------------------------------------------

        self._ensure_geometry(
            frame
        )

        # ---------------------------------------------------------------------
        # YOLOP
        # ---------------------------------------------------------------------

        detection = self._detect(
            frame
        )

        detection_valid = self._valid(
            detection
        )

        if detection_valid:
            self.statistics.valid_detections += 1
        else:
            self.statistics.invalid_detections += 1

        # ---------------------------------------------------------------------
        # TRACKER
        # ---------------------------------------------------------------------

        tracking = self._track(
            detection
        )

        # ---------------------------------------------------------------------
        # GEOMETRY
        #
        # Geometria trabalha com a observação atual.
        # Tracking temporal permanece responsabilidade do tracker.
        # ---------------------------------------------------------------------

        geometry = (
            self._calculate_geometry(
                detection,
                frame,
            )
        )

        geometry_valid = self._valid(
            geometry
        )

        if geometry_valid:
            self.statistics.valid_geometry += 1

        # ---------------------------------------------------------------------
        # LANE MODELS
        # ---------------------------------------------------------------------

        models = self._build_models(
            tracking
        )

        # ---------------------------------------------------------------------
        # PROJECTION
        # ---------------------------------------------------------------------

        projections = self._project_models(
            models
        )

        # ---------------------------------------------------------------------
        # ASSIGNMENT
        # ---------------------------------------------------------------------

        assignment = self._assign(
            models,
            frame,
        )

        assignment_valid = self._valid(
            assignment
        )

        if assignment_valid:
            self.statistics.valid_assignment += 1

        # ---------------------------------------------------------------------
        # ADAS
        #
        # IMPORTANTE:
        #
        # ADAS recebe:
        #
        #     LaneGeometryResult
        #     LaneAssignmentResult
        #
        # O Assignment é a fonte primária da posição lateral.
        # ---------------------------------------------------------------------

        adas = self._adas_update(
            geometry,
            assignment,
        )

        adas_valid = self._valid(
            adas
        )

        if adas_valid:
            self.statistics.valid_adas += 1

        # ---------------------------------------------------------------------
        # LANE LOST
        # ---------------------------------------------------------------------

        if adas is None:

            self.statistics.lane_lost_frames += 1

        else:

            state = getattr(
                adas,
                "state",
                None,
            )

            state_value = getattr(
                state,
                "value",
                "",
            )

            if state_value == "lane_lost":

                self.statistics.lane_lost_frames += 1

        # ---------------------------------------------------------------------
        # PIPELINE VALID
        #
        # Uma cadeia ADAS válida precisa de:
        #
        #     detection
        #     geometry
        #     assignment
        #     adas
        #
        # Tracking/model/projection podem estar parcialmente degradados
        # sem necessariamente invalidar toda a percepção.
        # ---------------------------------------------------------------------

        valid = bool(
            detection_valid
            and geometry_valid
            and assignment_valid
            and adas_valid
        )

        self.statistics.total_time_ms = (
            time.perf_counter()
            - pipeline_start
        ) * 1000.0

        result = PipelineResult(
            frame=frame,
            detection=detection,
            tracking=tracking,
            geometry=geometry,
            models=models,
            projections=projections,
            assignment=assignment,
            adas=adas,
            valid=valid,
            frame_index=self.frame_index,
            timestamp=time.monotonic(),
            statistics=self.statistics,
        )

        self.last_result = result

        # ---------------------------------------------------------------------
        # HUD
        # ---------------------------------------------------------------------

        self._update_display(
            geometry,
            adas,
        )

        return result

    # =========================================================================
    # OPENCV DEBUG
    # =========================================================================

    def _draw_debug_frame(
        self,
        result: PipelineResult,
    ) -> np.ndarray:

        if result.frame is None:

            return np.zeros(
                (720, 1280, 3),
                dtype=np.uint8,
            )

        frame = result.frame.copy()

        if not self._display_enabled:
            return frame

        # ---------------------------------------------------------------------
        # Header
        # ---------------------------------------------------------------------

        lines = []

        lines.append(
            f"FPS {self.statistics.fps:.1f}"
        )

        lines.append(
            f"{self.statistics.total_time_ms:.1f} ms"
        )

        device = "UNKNOWN"

        if self.detector is not None:

            try:
                device = (
                    self.detector
                    .get_device_name()
                )
            except Exception:
                pass

        lines.append(
            f"YOLOP {device}"
        )

        lines.append(
            f"MODE {self.mode.value.upper()}"
        )

        detection = result.detection

        lane_count = 0

        if detection is not None:

            lane_count = int(
                getattr(
                    detection,
                    "num_lanes_detected",
                    len(
                        getattr(
                            detection,
                            "lanes",
                            [],
                        )
                    ),
                )
            )

        lines.append(
            f"LANES {lane_count}"
        )

        tracking = result.tracking

        track_count = 0
        stable_count = 0

        if tracking is not None:

            track_count = len(
                getattr(
                    tracking,
                    "lanes",
                    (),
                )
            )

            stable_count = int(
                getattr(
                    tracking,
                    "stable_count",
                    0,
                )
            )

        lines.append(
            f"TRACKS {track_count} "
            f"STABLE {stable_count}"
        )

        geometry = result.geometry

        if geometry is not None:

            lines.append(
                "GEOMETRY "
                + (
                    "VALID"
                    if geometry.valid
                    else "INVALID"
                )
            )

            lines.append(
                f"LAT {geometry.lateral_error:+.3f} "
                f"HEAD {geometry.heading_error:+.3f}"
            )

            lines.append(
                f"WIDTH {geometry.lane_width:.1f} "
                f"CONF "
                f"{geometry.geometry_confidence:.2f}"
            )

        assignment = result.assignment

        if assignment is not None:

            lines.append(
                f"LANE "
                f"{assignment.current_lane_id}"
            )

            lines.append(
                f"OFFSET "
                f"{assignment.normalized_offset:+.3f}"
            )

            lines.append(
                f"ASSIGN "
                f"{assignment.confidence:.2f}"
            )

        adas = result.adas

        if adas is not None:

            state = getattr(
                adas,
                "state",
                None,
            )

            state_value = getattr(
                state,
                "value",
                "unknown",
            )

            lines.append(
                f"ADAS "
                f"{state_value.upper()}"
            )

            lines.append(
                f"ADAS CONF "
                f"{adas.confidence:.2f}"
            )

        lines.append(
            "PIPELINE "
            + (
                "VALID"
                if result.valid
                else "INVALID"
            )
        )

        # ---------------------------------------------------------------------
        # Render
        # ---------------------------------------------------------------------

        y = 28

        for text in lines:

            cv2.putText(
                frame,
                text,
                (18, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            y += 28

        return frame

    # =========================================================================
    # SUMMARY
    # =========================================================================

    def _print_summary(
        self,
        result: PipelineResult,
    ) -> None:

        detection_count = 0

        if result.detection is not None:

            detection_count = int(
                getattr(
                    result.detection,
                    "num_lanes_detected",
                    0,
                )
            )

        tracking_count = 0

        if result.tracking is not None:

            tracking_count = len(
                getattr(
                    result.tracking,
                    "lanes",
                    (),
                )
            )

        geometry_state = (
            "VALID"
            if self._valid(
                result.geometry
            )
            else "INVALID"
        )

        assignment_state = (
            "VALID"
            if self._valid(
                result.assignment
            )
            else "INVALID"
        )

        adas_state = "NONE"

        if result.adas is not None:

            adas_state = getattr(
                getattr(
                    result.adas,
                    "state",
                    None,
                ),
                "value",
                "unknown",
            )

        LOGGER.info(
            (
                "FRAME %d | "
                "YOLOP=%d | "
                "TRACKS=%d | "
                "GEOMETRY=%s | "
                "MODELS=%d | "
                "PROJECTIONS=%d | "
                "ASSIGNMENT=%s | "
                "ADAS=%s | "
                "%.1fms"
            ),
            result.frame_index,
            detection_count,
            tracking_count,
            geometry_state,
            len(result.models),
            len(result.projections),
            assignment_state,
            adas_state,
            self.statistics.total_time_ms,
        )

    # =========================================================================
    # RUN
    # =========================================================================

    def run(
        self,
        max_frames: Optional[int] = None,
    ) -> None:

        self.initialize()

        self.running = True

        processed = 0

        LOGGER.info(
            "Runtime started."
        )

        try:

            while self.running:

                frame = self._get_frame()

                if frame is None:

                    time.sleep(
                        0.001
                    )

                    continue

                try:

                    result = (
                        self.process_frame(
                            frame
                        )
                    )

                except Exception as exc:

                    LOGGER.exception(
                        "Pipeline exception: %s",
                        exc,
                    )

                    continue

                processed += 1

                # -------------------------------------------------------------
                # LOG
                # -------------------------------------------------------------

                if (
                    DEBUG.enabled
                    and DEBUG.print_pipeline_summary
                    and (
                        processed == 1
                        or (
                            processed
                            % DEBUG.debug_frame_interval
                            == 0
                        )
                    )
                ):

                    self._print_summary(
                        result
                    )

                # -------------------------------------------------------------
                # OpenCV debug window
                #
                # O HUD principal é ADASDisplay.
                # Esta janela existe apenas para debug rápido do pipeline.
                # -------------------------------------------------------------

                if (
                    self._display_enabled
                    and DEBUG.enabled
                ):

                    debug_frame = (
                        self._draw_debug_frame(
                            result
                        )
                    )

                    cv2.imshow(
                        VISUALIZATION.window_name,
                        debug_frame,
                    )

                    key = (
                        cv2.waitKey(
                            VISUALIZATION.wait_key_ms
                        )
                        & 0xFF
                    )

                    if key == 27:

                        LOGGER.info(
                            "ESC pressed."
                        )

                        break

                # -------------------------------------------------------------
                # FRAME LIMIT
                # -------------------------------------------------------------

                if (
                    max_frames is not None
                    and processed >= max_frames
                ):

                    LOGGER.info(
                        "Frame limit reached: %d",
                        max_frames,
                    )

                    break

        finally:

            self.shutdown()

    # =========================================================================
    # SHUTDOWN
    # =========================================================================

    def shutdown(self) -> None:

        if not self.running:

            # Mesmo quando initialize() falhou parcialmente,
            # tentamos liberar tudo com segurança.

            if self.display is not None:

                try:
                    self.display.stop()
                except Exception:
                    pass

            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

            return

        self.running = False

        LOGGER.info(
            "Shutting down Forza Assistents..."
        )

        # ---------------------------------------------------------------------
        # DISPLAY
        # ---------------------------------------------------------------------

        if self.display is not None:

            try:

                self.display.stop()

            except Exception as exc:

                LOGGER.debug(
                    "ADASDisplay shutdown error: %s",
                    exc,
                )

            self.display = None

        # ---------------------------------------------------------------------
        # CAPTURE
        # ---------------------------------------------------------------------

        if self.capture is not None:

            try:

                self.capture.stop()

            except Exception as exc:

                LOGGER.debug(
                    "ScreenCapture shutdown error: %s",
                    exc,
                )

            self.capture = None

        # ---------------------------------------------------------------------
        # WINDOWS
        # ---------------------------------------------------------------------

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        LOGGER.info(
            "Forza Assistents stopped."
        )

    # =========================================================================
    # STATUS
    # =========================================================================

    def get_status(self) -> dict[str, Any]:
        """
        Retorna estado operacional sem alterar o pipeline.
        """

        result = self.last_result

        status: dict[str, Any] = {
            "initialized": self._initialized,
            "running": self.running,
            "mode": self.mode.value,
            "physical_control": bool(
                SAFETY.enable_control
            ),
            "frame_index": self.frame_index,
            "fps": self.statistics.fps,
            "pipeline_latency_ms": (
                self.statistics.total_time_ms
            ),
        }

        if result is None:
            return status

        status.update(
            {
                "valid": result.valid,
                "lanes_detected": (
                    getattr(
                        result.detection,
                        "num_lanes_detected",
                        0,
                    )
                    if result.detection is not None
                    else 0
                ),
                "tracks": (
                    len(
                        getattr(
                            result.tracking,
                            "lanes",
                            (),
                        )
                    )
                    if result.tracking is not None
                    else 0
                ),
                "geometry_valid": (
                    self._valid(
                        result.geometry
                    )
                ),
                "assignment_valid": (
                    self._valid(
                        result.assignment
                    )
                ),
                "adas_valid": (
                    self._valid(
                        result.adas
                    )
                ),
            }
        )

        if result.assignment is not None:

            status[
                "current_lane"
            ] = result.assignment.current_lane_id

            status[
                "normalized_offset"
            ] = result.assignment.normalized_offset

        if result.adas is not None:

            status[
                "adas_state"
            ] = getattr(
                getattr(
                    result.adas,
                    "state",
                    None,
                ),
                "value",
                "unknown",
            )

            status[
                "adas_confidence"
            ] = result.adas.confidence

        return status


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Forza Assistents - "
            "ADAS runtime"
        )
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help=(
            "Número máximo de frames."
        ),
    )

    parser.add_argument(
        "--no-display",
        action="store_true",
        help=(
            "Desativa o display visual."
        ),
    )

    parser.add_argument(
        "--no-debug-window",
        action="store_true",
        help=(
            "Executa sem janela OpenCV "
            "de debug."
        ),
    )

    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:

    args = parse_args()

    application = (
        ForzaAssistents()
    )

    if args.no_display:

        application._display_enabled = False

    if args.no_debug_window:

        # O ADASDisplay continua funcionando.
        #
        # Somente a janela OpenCV de debug é desativada.
        #
        DEBUG.enabled = False

    try:

        application.run(
            max_frames=args.frames
        )

    except KeyboardInterrupt:

        LOGGER.info(
            "Interrupted by user."
        )

        application.shutdown()

    except Exception:

        LOGGER.exception(
            "Fatal application error."
        )

        application.shutdown()

        raise


if __name__ == "__main__":
    main()