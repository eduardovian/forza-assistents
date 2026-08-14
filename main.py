"""
main.py

Forza Assistents
================

Orquestrador principal do pipeline de percepção.

Pipeline:

    ScreenCapture
        ↓
    YOLOP
        ↓
    LaneTracker
        ↓
    LaneGeometry
        ↓
    LaneModel
        ↓
    LaneProjection
        ↓
    LaneAssignment
        ↓
    ADAS State
        ↓
    Monitor / Debug

IMPORTANTE
==========

Este arquivo somente orquestra os módulos.

Não executa:

    - inferência diretamente;
    - tracking diretamente;
    - fitting diretamente;
    - controle G29;
    - lógica de visão independente dos módulos.

O modo padrão permanece MONITOR.
Nenhum comando físico é enviado ao veículo.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
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
    RuntimeMode,
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


# ============================================================================
# LOGGING
# ============================================================================

def setup_logging() -> logging.Logger:
    """
    Configura logging global da aplicação.
    """

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

        console.setFormatter(
            formatter
        )

        logger.addHandler(
            console
        )

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

    return logger


LOGGER = setup_logging()


# ============================================================================
# ESTATÍSTICAS
# ============================================================================

@dataclass
class RuntimeStatistics:
    """
    Estatísticas do pipeline.
    """

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

                instant_fps = (
                    1.0 / delta
                )

                if self.fps <= 0.0:

                    self.fps = instant_fps

                else:

                    self.fps = (
                        self.fps * 0.90
                        + instant_fps * 0.10
                    )

        self._last_timestamp = now


# ============================================================================
# RESULTADO DO PIPELINE
# ============================================================================

@dataclass
class PipelineResult:

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

    models: list[LaneModel] | None = None

    projections: list[Any] | None = None

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

    def __post_init__(self) -> None:

        if self.models is None:
            self.models = []

        if self.projections is None:
            self.projections = []


# ============================================================================
# APLICAÇÃO
# ============================================================================

class ForzaAssistents:
    """
    Aplicação principal do Forza Assistents.
    """

    def __init__(self) -> None:

        validate_config()

        self.mode = (
            CONFIG.runtime_mode
        )

        self.running = False

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

        self.last_result: Optional[
            PipelineResult
        ] = None

        self._initialized = False

        self._geometry_shape: Optional[
            tuple[int, int]
        ] = None

        self._display_enabled = bool(
            VISUALIZATION.enabled
        )


    # ========================================================================
    # INITIALIZATION
    # ========================================================================

    def initialize(self) -> None:

        if self._initialized:
            return

        LOGGER.info(
            "Initializing Forza Assistents..."
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

        # --------------------------------------------------------------------
        # CAPTURE
        # --------------------------------------------------------------------

        backend = str(
            CAPTURE.backend
        ).lower()

        # ScreenCapture espera "dxgi" para DXCAM.
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
                "Não foi possível "
                "inicializar ScreenCapture."
            )

        self.capture.start()

        LOGGER.info(
            "Screen capture initialized."
        )

        # --------------------------------------------------------------------
        # YOLOP
        # --------------------------------------------------------------------

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
                "YOLOP não pôde ser carregado: "
                f"{self.detector.last_error}"
            )

        LOGGER.info(
            "YOLOP ready | device=%s",
            self.detector.get_device_name(),
        )

        # --------------------------------------------------------------------
        # TRACKER
        # --------------------------------------------------------------------

        self.tracker = LaneTracker(
            max_lanes=(
                LANE_TRACKER.max_tracks
            ),
            history_size=(
                LANE_TRACKER.history_size
            ),
            min_points=(
                LANE_TRACKER.min_confidence
                and LANE_TRACKER.max_lost_frames
                and 1
                or 1
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

        # --------------------------------------------------------------------
        # PROJECTION
        # --------------------------------------------------------------------

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

        # --------------------------------------------------------------------
        # ASSIGNMENT
        # --------------------------------------------------------------------

        self.assignment = LaneAssignment(
            max_lanes=(
                max(
                    1,
                    LANE_TRACKER.max_tracks,
                )
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
        )

        # --------------------------------------------------------------------
        # ADAS
        # --------------------------------------------------------------------

        self.adas = (
            ADASStateEstimator(
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
        )

        # --------------------------------------------------------------------
        # GEOMETRY
        #
        # Criada somente depois que o primeiro frame chegar.
        # --------------------------------------------------------------------

        self.geometry = None

        self._geometry_shape = None

        self._initialized = True

        LOGGER.info(
            "Forza Assistents initialized."
        )


    # ========================================================================
    # GEOMETRY
    # ========================================================================

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
            "Creating LaneGeometry: %dx%d",
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


    # ========================================================================
    # CAPTURE
    # ========================================================================

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


    # ========================================================================
    # DETECTOR
    # ========================================================================

    def _detect(
        self,
        frame: np.ndarray,
    ) -> LaneDetectionResult:

        if self.detector is None:
            raise RuntimeError(
                "Detector não inicializado."
            )

        start = time.perf_counter()

        result = (
            self.detector.detect(
                frame
            )
        )

        self.statistics.detection_time_ms = (
            time.perf_counter() - start
        ) * 1000.0

        return result


    # ========================================================================
    # TRACKER
    # ========================================================================

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


    # ========================================================================
    # GEOMETRY
    # ========================================================================

    def _calculate_geometry(
        self,
        detection: LaneDetectionResult,
        frame: np.ndarray,
    ) -> LaneGeometryResult:

        geometry = (
            self._ensure_geometry(
                frame
            )
        )

        start = time.perf_counter()

        result = geometry.compute(
            detection
        )

        self.statistics.geometry_time_ms = (
            time.perf_counter() - start
        ) * 1000.0

        return result


    # ========================================================================
    # MODELS
    # ========================================================================

    @staticmethod
    def _extract_tracks(
        tracking: LaneTrackingResult,
    ) -> list[Any]:

        if tracking is None:
            return []

        # Preferimos somente tracks ainda ativos.
        try:

            return list(
                tracking.active_lanes
            )

        except Exception:
            pass

        lanes = getattr(
            tracking,
            "lanes",
            (),
        )

        return list(lanes)


    def _build_models(
        self,
        tracking: LaneTrackingResult,
    ) -> list[LaneModel]:

        start = time.perf_counter()

        models: list[
            LaneModel
        ] = []

        tracks = (
            self._extract_tracks(
                tracking
            )
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
                track_id = int(
                    track_id
                )
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
                    "Lane model failed "
                    "for track %s: %s",
                    track_id,
                    exc,
                )

                continue

            if model is None:
                continue

            # Preserva informações temporais
            # do tracker.
            model.tracked = True

            model.stable = bool(
                getattr(
                    track,
                    "stable",
                    False,
                )
            )

            if model.valid:

                models.append(
                    model
                )

        self.statistics.model_time_ms = (
            time.perf_counter() - start
        ) * 1000.0

        return models


    # ========================================================================
    # PROJECTION
    # ========================================================================

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
                    "Projection failed "
                    "for lane %s: %s",
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


    # ========================================================================
    # ASSIGNMENT
    # ========================================================================

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

        height, width = (
            frame.shape[:2]
        )

        try:

            result = (
                self.assignment.assign(
                    models,
                    frame_width=float(
                        width
                    ),
                    frame_height=float(
                        height
                    ),
                )
            )

        except Exception as exc:

            LOGGER.debug(
                "Lane assignment failed: %s",
                exc,
            )

            result = None

        self.statistics.assignment_time_ms = (
            time.perf_counter() - start
        ) * 1000.0

        return result


    # ========================================================================
    # ADAS
    # ========================================================================

    def _adas_update(
        self,
        geometry: Optional[
            LaneGeometryResult
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


    # ========================================================================
    # VALIDAÇÃO
    # ========================================================================

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


    # ========================================================================
    # PROCESSAMENTO
    # ========================================================================

    def process_frame(
        self,
        frame: np.ndarray,
    ) -> PipelineResult:

        start = time.perf_counter()

        self.frame_index += 1

        self.statistics.frame_index = (
            self.frame_index
        )

        self.statistics.total_frames += 1

        self.statistics.update_fps()

        # --------------------------------------------------------------------
        # GEOMETRY CONFIG
        # --------------------------------------------------------------------

        self._ensure_geometry(
            frame
        )

        # --------------------------------------------------------------------
        # YOLOP
        # --------------------------------------------------------------------

        detection = self._detect(
            frame
        )

        detection_valid = (
            self._valid(
                detection
            )
        )

        if detection_valid:
            self.statistics.valid_detections += 1
        else:
            self.statistics.invalid_detections += 1

        # --------------------------------------------------------------------
        # TRACKER
        # --------------------------------------------------------------------

        tracking = self._track(
            detection
        )

        # --------------------------------------------------------------------
        # GEOMETRY
        #
        # A geometria usa a observação atual do YOLOP.
        # --------------------------------------------------------------------

        geometry = (
            self._calculate_geometry(
                detection,
                frame,
            )
        )

        geometry_valid = (
            self._valid(
                geometry
            )
        )

        if geometry_valid:
            self.statistics.valid_geometry += 1

        # --------------------------------------------------------------------
        # LANE MODELS
        # --------------------------------------------------------------------

        models = (
            self._build_models(
                tracking
            )
        )

        # --------------------------------------------------------------------
        # PROJECTION
        # --------------------------------------------------------------------

        projections = (
            self._project_models(
                models
            )
        )

        # --------------------------------------------------------------------
        # ASSIGNMENT
        #
        # IMPORTANTE:
        #
        # LaneAssignment recebe LaneModel.
        # Não recebe LaneProjectionResult.
        # --------------------------------------------------------------------

        assignment = (
            self._assign(
                models,
                frame,
            )
        )

        assignment_valid = (
            self._valid(
                assignment
            )
        )

        if assignment_valid:
            self.statistics.valid_assignment += 1

        # --------------------------------------------------------------------
        # ADAS
        #
        # ADAS utiliza LaneGeometryResult.
        # --------------------------------------------------------------------

        adas = (
            self._adas_update(
                geometry
            )
        )

        adas_valid = (
            self._valid(
                adas
            )
        )

        if adas_valid:
            self.statistics.valid_adas += 1

        if (
            adas is None
            or getattr(
                adas,
                "state",
                None,
            ) is None
        ):
            self.statistics.lane_lost_frames += 1

        else:

            state_name = getattr(
                getattr(
                    adas,
                    "state",
                    None,
                ),
                "value",
                "",
            )

            if state_name == "lane_lost":

                self.statistics.lane_lost_frames += 1

        # --------------------------------------------------------------------
        # VALIDADE FINAL
        # --------------------------------------------------------------------

        valid = bool(
            detection_valid
            and geometry_valid
            and assignment_valid
            and adas_valid
        )

        self.statistics.total_time_ms = (
            time.perf_counter() - start
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

        return result


    # ========================================================================
    # OVERLAY
    # ========================================================================

    def _draw_overlay(
        self,
        result: PipelineResult,
    ) -> np.ndarray:

        if result.frame is None:

            return np.zeros(
                (
                    720,
                    1280,
                    3,
                ),
                dtype=np.uint8,
            )

        output = result.frame.copy()

        if not self._display_enabled:
            return output

        # --------------------------------------------------------------------
        # FPS
        # --------------------------------------------------------------------

        cv2.putText(
            output,
            f"FPS: {self.statistics.fps:.1f}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # --------------------------------------------------------------------
        # LATÊNCIA
        # --------------------------------------------------------------------

        cv2.putText(
            output,
            (
                f"Pipeline: "
                f"{self.statistics.total_time_ms:.1f} ms"
            ),
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # --------------------------------------------------------------------
        # DEVICE
        # --------------------------------------------------------------------

        device = "UNKNOWN"

        if self.detector is not None:

            try:

                device = (
                    self.detector
                    .get_device_name()
                )

            except Exception:
                pass

        cv2.putText(
            output,
            f"YOLOP: {device}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # --------------------------------------------------------------------
        # MODE
        # --------------------------------------------------------------------

        cv2.putText(
            output,
            (
                f"MODE: "
                f"{self.mode.value.upper()}"
            ),
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # --------------------------------------------------------------------
        # DETECTION
        # --------------------------------------------------------------------

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

        cv2.putText(
            output,
            f"YOLOP lanes: {lane_count}",
            (20, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # --------------------------------------------------------------------
        # TRACKING
        # --------------------------------------------------------------------

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

        cv2.putText(
            output,
            (
                f"Tracks: {track_count} | "
                f"Stable: {stable_count}"
            ),
            (20, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # --------------------------------------------------------------------
        # GEOMETRY
        # --------------------------------------------------------------------

        geometry = result.geometry

        if geometry is not None:

            geometry_state = (
                "VALID"
                if geometry.valid
                else "INVALID"
            )

            cv2.putText(
                output,
                f"Geometry: {geometry_state}",
                (20, 210),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                output,
                (
                    f"Lat: "
                    f"{geometry.lateral_error:+.3f} | "
                    f"Head: "
                    f"{geometry.heading_error:+.3f}"
                ),
                (20, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                output,
                (
                    f"Width: "
                    f"{geometry.lane_width:.1f} | "
                    f"Conf: "
                    f"{geometry.geometry_confidence:.2f}"
                ),
                (20, 270),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # --------------------------------------------------------------------
        # MODELS
        # --------------------------------------------------------------------

        cv2.putText(
            output,
            (
                f"Models: "
                f"{len(result.models)} | "
                f"Projections: "
                f"{len(result.projections)}"
            ),
            (20, 300),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # --------------------------------------------------------------------
        # ASSIGNMENT
        # --------------------------------------------------------------------

        assignment = (
            result.assignment
        )

        if assignment is not None:

            lane_id = getattr(
                assignment,
                "current_lane_id",
                None,
            )

            offset = getattr(
                assignment,
                "normalized_offset",
                0.0,
            )

            confidence = getattr(
                assignment,
                "confidence",
                0.0,
            )

            cv2.putText(
                output,
                (
                    f"Lane: {lane_id} | "
                    f"Offset: "
                    f"{float(offset):+.3f}"
                ),
                (20, 330),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                output,
                (
                    f"Assignment: "
                    f"{float(confidence):.2f}"
                ),
                (20, 360),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # --------------------------------------------------------------------
        # ADAS
        # --------------------------------------------------------------------

        adas = result.adas

        if adas is not None:

            state = getattr(
                adas,
                "state",
                "UNKNOWN",
            )

            state_text = getattr(
                state,
                "value",
                str(state),
            )

            confidence = float(
                getattr(
                    adas,
                    "confidence",
                    0.0,
                )
            )

            cv2.putText(
                output,
                (
                    f"ADAS: "
                    f"{state_text.upper()} | "
                    f"Conf: "
                    f"{confidence:.2f}"
                ),
                (20, 390),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # --------------------------------------------------------------------
        # STATUS FINAL
        # --------------------------------------------------------------------

        status = (
            "VALID"
            if result.valid
            else "INVALID"
        )

        cv2.putText(
            output,
            f"PIPELINE: {status}",
            (20, 420),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return output


    # ========================================================================
    # SUMMARY
    # ========================================================================

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
            if result.geometry is not None
            and result.geometry.valid
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


    # ========================================================================
    # LOOP
    # ========================================================================

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
                        "Pipeline error: %s",
                        exc,
                    )

                    continue

                processed += 1

                # ------------------------------------------------------------
                # DEBUG
                # ------------------------------------------------------------

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

                # ------------------------------------------------------------
                # DISPLAY
                # ------------------------------------------------------------

                if self._display_enabled:

                    output = (
                        self._draw_overlay(
                            result
                        )
                    )

                    cv2.imshow(
                        VISUALIZATION.window_name,
                        output,
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

                # ------------------------------------------------------------
                # TEST LIMIT
                # ------------------------------------------------------------

                if (
                    max_frames is not None
                    and processed >= max_frames
                ):

                    break

        finally:

            self.shutdown()


    # ========================================================================
    # SHUTDOWN
    # ========================================================================

    def shutdown(self) -> None:

        if not self.running:

            # Mesmo se o loop não tiver iniciado,
            # destruímos janelas caso existam.
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

            return

        self.running = False

        LOGGER.info(
            "Shutting down Forza Assistents..."
        )

        if self.capture is not None:

            try:

                self.capture.stop()

            except Exception as exc:

                LOGGER.debug(
                    "Capture shutdown error: %s",
                    exc,
                )

        try:

            cv2.destroyAllWindows()

        except Exception:
            pass

        LOGGER.info(
            "Forza Assistents stopped."
        )


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Forza Assistents - "
            "YOLOP ADAS pipeline"
        )
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help=(
            "Número máximo de frames "
            "a processar."
        ),
    )

    parser.add_argument(
        "--no-display",
        action="store_true",
        help=(
            "Executa sem janela OpenCV."
        ),
    )

    return parser.parse_args()


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    args = parse_args()

    application = (
        ForzaAssistents()
    )

    if args.no_display:

        application._display_enabled = False

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