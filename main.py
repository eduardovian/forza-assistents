"""
main.py

Forza Assistents
================

Orquestrador principal do pipeline ADAS/LKA.

Arquitetura:

    ScreenCapture
        ↓
    YOLOPv2
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
    ADASState
        ↓
    Safety Gate
        ↓
    ADASDisplay

Princípios:

    - uma única fonte de configuração;
    - coordenadas preservadas pelo pipeline;
    - YOLOP não possui estado temporal;
    - tracking separado da inferência;
    - geometria observada separada de projeção;
    - nenhuma previsão autoriza controle;
    - qualquer falha bloqueia atuação;
    - controle físico permanece desabilitado por padrão.

IMPORTANTE
==========

O YOLOPDetector atual retorna LanePoint em coordenadas do
FRAME ORIGINAL recebido por detect().

O main.py NÃO deve:

    - redimensionar LanePoint;
    - aplicar ROI novamente;
    - converter coordenadas do detector;
    - fazer segundo mapeamento geométrico.
"""

from __future__ import annotations

import logging
import signal
import time
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np

import config

from capture.screen_capture import ScreenCapture

from vision.yolop_detector import (
    YOLOPDetector,
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
    validate_lane_model,
)

from vision.lane_projection import (
    LaneProjectionEngine,
)

from vision.lane_assignment import (
    LaneAssignment,
    LaneAssignmentResult,
)

from vision.adas_state import (
    ADASState,
    ADASStateEstimator,
    ADASStateResult,
)

try:
    from visualization.adas_display import (
        ADASDisplay,
        ADASDisplayConfig,
    )
except ImportError:
    ADASDisplay = None
    ADASDisplayConfig = None


# =============================================================================
# LOGGING
# =============================================================================

LOGGER = logging.getLogger("forza_assistents")


def setup_logging() -> None:
    """Inicializa logging uma única vez."""

    if LOGGER.handlers:
        return

    LOGGER.setLevel(logging.INFO)

    handler = logging.StreamHandler()

    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        )
    )

    LOGGER.addHandler(handler)
    LOGGER.propagate = False


# =============================================================================
# UTILITÁRIOS
# =============================================================================

def finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:

    try:
        value = float(value)

        if not np.isfinite(value):
            return default

        return value

    except (TypeError, ValueError):
        return default


def clamp01(value: Any) -> float:

    value = safe_float(value)

    return float(
        np.clip(
            value,
            0.0,
            1.0,
        )
    )


# =============================================================================
# ESTATÍSTICAS
# =============================================================================

@dataclass
class RuntimeStatistics:

    frame_index: int = 0

    fps: float = 0.0

    capture_ms: float = 0.0
    detection_ms: float = 0.0
    tracking_ms: float = 0.0
    geometry_ms: float = 0.0
    model_ms: float = 0.0
    projection_ms: float = 0.0
    assignment_ms: float = 0.0
    adas_ms: float = 0.0

    total_ms: float = 0.0

    dropped_frames: int = 0

    _last_timestamp: float = 0.0

    def update_fps(self) -> None:

        now = time.perf_counter()

        if self._last_timestamp > 0.0:

            dt = (
                now
                - self._last_timestamp
            )

            if dt > 1e-6:

                instantaneous = (
                    1.0 / dt
                )

                if self.fps <= 0.0:

                    self.fps = instantaneous

                else:

                    self.fps = (
                        self.fps * 0.90
                        + instantaneous * 0.10
                    )

        self._last_timestamp = now


# =============================================================================
# RESULTADO DO PIPELINE
# =============================================================================

@dataclass
class PipelineResult:

    frame_index: int

    timestamp: float

    detection: Optional[
        LaneDetectionResult
    ]

    tracking: Optional[
        LaneTrackingResult
    ]

    geometry: Optional[
        LaneGeometryResult
    ]

    models: tuple[Any, ...]

    projections: tuple[Any, ...]

    assignment: Optional[
        LaneAssignmentResult
    ]

    adas: Optional[
        ADASStateResult
    ]

    actuation_allowed: bool

    valid: bool

    processing_ms: float

    error: Optional[str] = None


# =============================================================================
# SAFETY GATE
# =============================================================================

class SafetyGate:
    """
    Gate central de segurança.

    Nenhuma camada acima pode simplesmente decidir atuar.

    Para permitir atuação é necessário:

        - controle explicitamente habilitado;
        - geometria real válida;
        - tracking seguro;
        - ADAS válido;
        - confiança suficiente;
        - estado ADAS não crítico;
        - ausência de erro no pipeline.
    """

    def __init__(
        self,
        minimum_confidence: float = 0.55,
        minimum_stable_frames: int = 4,
    ) -> None:

        self.minimum_confidence = (
            clamp01(
                minimum_confidence
            )
        )

        self.minimum_stable_frames = max(
            1,
            int(
                minimum_stable_frames
            ),
        )

        self._stable_frames = 0

    def reset(self) -> None:
        self._stable_frames = 0

    def evaluate(
        self,
        *,
        tracking: Optional[
            LaneTrackingResult
        ],
        geometry: Optional[
            LaneGeometryResult
        ],
        adas: Optional[
            ADASStateResult
        ],
    ) -> bool:

        if not bool(
            getattr(
                config.SAFETY,
                "enable_control",
                False,
            )
        ):
            self._stable_frames = 0
            return False

        if tracking is None:
            self._stable_frames = 0
            return False

        if not tracking.safe_for_control:
            self._stable_frames = 0
            return False

        if geometry is None:
            self._stable_frames = 0
            return False

        if not geometry.valid:
            self._stable_frames = 0
            return False

        if adas is None:
            self._stable_frames = 0
            return False

        if not bool(
            getattr(
                adas,
                "valid",
                False,
            )
        ):
            self._stable_frames = 0
            return False

        confidence = safe_float(
            getattr(
                adas,
                "confidence",
                0.0,
            )
        )

        if (
            confidence
            < self.minimum_confidence
        ):
            self._stable_frames = 0
            return False

        state = getattr(
            adas,
            "state",
            None,
        )

        blocked_states = {
            getattr(
                ADASState,
                "UNKNOWN",
                None,
            ),
            getattr(
                ADASState,
                "LANE_LOST",
                None,
            ),
        }

        if state in blocked_states:
            self._stable_frames = 0
            return False

        self._stable_frames += 1

        return (
            self._stable_frames
            >= self.minimum_stable_frames
        )


# =============================================================================
# APLICAÇÃO
# =============================================================================

class ForzaAssistents:

    def __init__(self) -> None:

        config.validate_config()

        self.running = False
        self.initialized = False

        self.frame_index = 0

        self.stats = RuntimeStatistics()

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

        self.display: Optional[Any] = None

        self.safety = SafetyGate(
            minimum_confidence=float(
                getattr(
                    config.ADAS,
                    "minimum_confidence",
                    0.55,
                )
            ),
            minimum_stable_frames=4,
        )

        self._last_frame_shape: Optional[
            tuple[int, int]
        ] = None

        self._last_result: Optional[
            PipelineResult
        ] = None

    # =========================================================================
    # INITIALIZAÇÃO
    # =========================================================================

    def initialize(self) -> None:

        if self.initialized:
            return

        LOGGER.info("=" * 72)
        LOGGER.info("FORZA ASSISTENTS")
        LOGGER.info(
            "Inicializando pipeline ADAS..."
        )

        # ---------------------------------------------------------------------
        # CONFIGURAÇÃO
        # ---------------------------------------------------------------------

        roi = config.ROI

        if not roi.enabled:
            raise RuntimeError(
                "ROI não está calibrado."
            )

        LOGGER.info(
            "ROI: "
            "(%d, %d) -> (%d, %d) | "
            "%dx%d",
            roi.left,
            roi.top,
            roi.right,
            roi.bottom,
            roi.width,
            roi.height,
        )

        # ---------------------------------------------------------------------
        # CAPTURE
        # ---------------------------------------------------------------------
        #
        # ScreenCapture atual:
        #
        #     ScreenCapture(monitor_index=...)
        #     start()
        #     read() -> FramePacket
        #
        # O ROI é aplicado internamente pelo ScreenCapture.
        # ---------------------------------------------------------------------

        self.capture = ScreenCapture(
            monitor_index=(
                config.CAPTURE.monitor_index
            ),
        )

        self.capture.start()

        LOGGER.info(
            "ScreenCapture: READY"
        )

        # ---------------------------------------------------------------------
        # YOLOPv2
        # ---------------------------------------------------------------------

        self.detector = (
            create_default_detector(
                model_path=(
                    config.YOLOP.model_path
                ),
                input_width=(
                    config.YOLOP.input_width
                ),
                input_height=(
                    config.YOLOP.input_height
                ),
                lane_threshold=(
                    config.YOLOP
                    .lane_confidence_threshold
                ),
                min_points_per_lane=(
                    config.YOLOP
                    .minimum_lane_points
                ),
                max_lanes=(
                    config.YOLOP.max_lanes
                ),
                use_fp16=True,
            )
        )

        if not self.detector.load_model():

            raise RuntimeError(
                "Falha ao carregar YOLOPv2: "
                f"{self.detector.last_error}"
            )

        LOGGER.info(
            "YOLOPv2: READY | device=%s | "
            "FP16=%s",
            self.detector.get_device_name(),
            getattr(
                self.detector,
                "fp16_active",
                False,
            ),
        )

        # ---------------------------------------------------------------------
        # TRACKER
        # ---------------------------------------------------------------------

        self.tracker = LaneTracker(
            max_lanes=(
                config.LANE_TRACKER
                .max_tracks
            ),
            history_size=(
                config.LANE_TRACKER
                .history_size
            ),
            min_fit_points=max(
                4,
                config.YOLOP
                .minimum_lane_points,
            ),
            max_missed_frames=(
                config.LANE_TRACKER
                .max_lost_frames
            ),
            confidence_decay=(
                config.LANE_TRACKER
                .confidence_decay
            ),
        )

        LOGGER.info(
            "LaneTracker: READY"
        )

        # ---------------------------------------------------------------------
        # GEOMETRY
        # ---------------------------------------------------------------------

        self.geometry = LaneGeometry(
            screen_width=(
                roi.width
            ),
            screen_height=(
                roi.height
            ),
            roi=(
                0,
                0,
                roi.width,
                roi.height,
            ),
            detector_width=(
                roi.width
            ),
            detector_height=(
                roi.height
            ),
            min_points=(
                config.LANE_GEOMETRY
                .min_points
            ),
            min_lane_width=(
                config.LANE_GEOMETRY
                .min_lane_width
            ),
            max_lane_width=(
                config.LANE_GEOMETRY
                .max_lane_width
            ),
            min_observed_span=(
                config.LANE_GEOMETRY
                .min_observed_span
            ),
        )

        LOGGER.info(
            "LaneGeometry: READY"
        )

        # ---------------------------------------------------------------------
        # PROJECTION
        # ---------------------------------------------------------------------

        self.projection = (
            LaneProjectionEngine(
                min_points=(
                    config.LANE_PROJECTION
                    .minimum_points
                ),
                min_confidence=(
                    config.LANE_PROJECTION
                    .minimum_confidence
                ),
                max_projection_distance=(
                    config.LANE_PROJECTION
                    .max_projection_distance
                ),
            )
        )

        LOGGER.info(
            "LaneProjection: READY"
        )

        # ---------------------------------------------------------------------
        # ASSIGNMENT
        # ---------------------------------------------------------------------

        self.assignment = LaneAssignment(
            max_lanes=(
                config.LANE_TRACKER
                .max_tracks
            ),
            min_lane_width_px=(
                config.LANE_GEOMETRY
                .min_lane_width
            ),
            max_lane_width_px=(
                config.LANE_GEOMETRY
                .max_lane_width
            ),
            vehicle_x_ratio=(
                config.LANE_ASSIGNMENT
                .center_reference_ratio
            ),
            expected_lane_width=(
                config.LANE_ASSIGNMENT
                .expected_lane_width
            ),
            lane_width_tolerance=(
                config.LANE_ASSIGNMENT
                .lane_width_tolerance
            ),
            minimum_confidence=(
                config.LANE_ASSIGNMENT
                .minimum_confidence
            ),
            maximum_lateral_offset_ratio=(
                config.LANE_ASSIGNMENT
                .maximum_lateral_offset_ratio
            ),
            enable_multi_lane_assignment=(
                config.LANE_ASSIGNMENT
                .enable_multi_lane_assignment
            ),
            max_left_lanes=(
                config.LANE_ASSIGNMENT
                .max_left_lanes
            ),
            max_right_lanes=(
                config.LANE_ASSIGNMENT
                .max_right_lanes
            ),
        )

        LOGGER.info(
            "LaneAssignment: READY"
        )

        # ---------------------------------------------------------------------
        # ADAS
        # ---------------------------------------------------------------------

        self.adas = (
            ADASStateEstimator(
                min_confidence=(
                    config.ADAS
                    .minimum_confidence
                ),
                warning_threshold=(
                    config.ADAS
                    .warning_threshold
                ),
                departure_threshold=(
                    config.ADAS
                    .critical_threshold
                ),
                heading_warning_threshold=(
                    config.ADAS
                    .heading_warning_threshold
                ),
            )
        )

        LOGGER.info(
            "ADASStateEstimator: READY"
        )

        # ---------------------------------------------------------------------
        # DISPLAY
        # ---------------------------------------------------------------------

        if (
            bool(
                getattr(
                    config.VISUALIZATION,
                    "enabled",
                    False,
                )
            )
            and ADASDisplay is not None
        ):

            try:

                self.display = (
                    ADASDisplay(
                        config=(
                            ADASDisplayConfig(
                                source_width=(
                                    roi.width
                                ),
                                source_height=(
                                    roi.height
                                ),
                                refresh_hz=30.0,
                            )
                        )
                    )
                )

                self.display.start(
                    blocking=False
                )

                LOGGER.info(
                    "ADASDisplay: READY"
                )

            except Exception as exc:

                LOGGER.warning(
                    "ADASDisplay "
                    "indisponível: %s",
                    exc,
                )

                self.display = None

        self.initialized = True

        LOGGER.info(
            "Pipeline inicializado."
        )

    # =========================================================================
    # FRAME
    # =========================================================================

    def get_frame(
        self,
    ) -> Optional[np.ndarray]:

        if self.capture is None:
            return None

        start = time.perf_counter()

        try:

            packet = self.capture.read()

        except Exception:

            LOGGER.exception(
                "Falha durante captura do frame."
            )

            self.stats.dropped_frames += 1

            return None

        self.stats.capture_ms = (
            time.perf_counter()
            - start
        ) * 1000.0

        if packet is None:

            self.stats.dropped_frames += 1

            return None

        frame = packet.frame

        if not isinstance(
            frame,
            np.ndarray,
        ):
            self.stats.dropped_frames += 1
            return None

        if frame.ndim != 3:
            self.stats.dropped_frames += 1
            return None

        if frame.shape[2] != 3:
            self.stats.dropped_frames += 1
            return None

        return frame

    # =========================================================================
    # PROCESSAMENTO
    # =========================================================================

    def process_frame(
        self,
        frame: np.ndarray,
    ) -> PipelineResult:

        started = time.perf_counter()

        timestamp = time.perf_counter()

        self.frame_index += 1

        frame_height, frame_width = (
            frame.shape[:2]
        )

        self.stats.frame_index = (
            self.frame_index
        )

        # ---------------------------------------------------------------------
        # 1. YOLOP
        # ---------------------------------------------------------------------

        detection: Optional[
            LaneDetectionResult
        ] = None

        try:

            if self.detector is None:
                raise RuntimeError(
                    "Detector não inicializado."
                )

            detection = (
                self.detector.detect(
                    frame
                )
            )

            self.stats.detection_ms = (
                safe_float(
                    self.detector
                    .last_diagnostics
                    .get(
                        "inference_ms",
                        0.0,
                    )
                )
            )

        except Exception as exc:

            LOGGER.exception(
                "Falha no YOLOPv2."
            )

            self._reset_temporal_state()

            return self._failure_result(
                timestamp,
                started,
                (
                    "YOLOP: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            )

        # ---------------------------------------------------------------------
        # CONTRATO DE COORDENADAS
        # ---------------------------------------------------------------------

        if not self._validate_detection_coordinates(
            detection,
            frame_width,
            frame_height,
        ):

            error = (
                "YOLOP retornou coordenadas "
                "fora do frame original."
            )

            LOGGER.error(error)

            self._reset_temporal_state()

            return self._failure_result(
                timestamp,
                started,
                error,
            )

        # ---------------------------------------------------------------------
        # 2. TRACKER
        # ---------------------------------------------------------------------

        tracking: Optional[
            LaneTrackingResult
        ] = None

        tracking_started = time.perf_counter()

        try:

            if self.tracker is None:
                raise RuntimeError(
                    "LaneTracker não inicializado."
                )

            tracking = (
                self.tracker.update(
                    detection,
                    frame_width=(
                        frame_width
                    ),
                    frame_height=(
                        frame_height
                    ),
                )
            )

        except Exception:

            LOGGER.exception(
                "Falha no LaneTracker."
            )

        self.stats.tracking_ms = (
            time.perf_counter()
            - tracking_started
        ) * 1000.0

        # ---------------------------------------------------------------------
        # 3. GEOMETRIA
        # ---------------------------------------------------------------------

        geometry: Optional[
            LaneGeometryResult
        ] = None

        geometry_started = time.perf_counter()

        try:

            if (
                self.geometry is not None
                and detection is not None
            ):

                geometry = (
                    self.geometry.compute(
                        detection
                    )
                )

        except Exception:

            LOGGER.exception(
                "Falha no LaneGeometry."
            )

        self.stats.geometry_ms = (
            time.perf_counter()
            - geometry_started
        ) * 1000.0

        # ---------------------------------------------------------------------
        # 4. LANE MODEL
        # ---------------------------------------------------------------------

        models = []

        model_started = time.perf_counter()

        if tracking is not None:

            models = (
                self._build_lane_models(
                    tracking
                )
            )

        self.stats.model_ms = (
            time.perf_counter()
            - model_started
        ) * 1000.0

        # ---------------------------------------------------------------------
        # 5. PROJECTION
        # ---------------------------------------------------------------------

        projections = []

        projection_started = time.perf_counter()

        if (
            self.projection is not None
            and models
        ):

            projections = (
                self._project_models(
                    models
                )
            )

        self.stats.projection_ms = (
            time.perf_counter()
            - projection_started
        ) * 1000.0

        # ---------------------------------------------------------------------
        # 6. ASSIGNMENT
        # ---------------------------------------------------------------------

        assignment: Optional[
            LaneAssignmentResult
        ] = None

        assignment_started = time.perf_counter()

        try:

            if (
                self.assignment is not None
                and models
            ):

                assignment = (
                    self.assignment.assign(
                        models,
                        frame_width=(
                            frame_width
                        ),
                        frame_height=(
                            frame_height
                        ),
                    )
                )

        except Exception:

            LOGGER.exception(
                "Falha no LaneAssignment."
            )

        self.stats.assignment_ms = (
            time.perf_counter()
            - assignment_started
        ) * 1000.0

        # ---------------------------------------------------------------------
        # 7. ADAS
        # ---------------------------------------------------------------------

        adas_result: Optional[
            ADASStateResult
        ] = None

        adas_started = time.perf_counter()

        try:

            if self.adas is not None:

                geometry_for_adas = (
                    geometry
                    if (
                        geometry is not None
                        and geometry.valid
                    )
                    else None
                )

                adas_result = (
                    self.adas.update(
                        geometry_for_adas,
                        timestamp,
                    )
                )

        except Exception:

            LOGGER.exception(
                "Falha no ADAS."
            )

        self.stats.adas_ms = (
            time.perf_counter()
            - adas_started
        ) * 1000.0

        # ---------------------------------------------------------------------
        # 8. SAFETY GATE
        # ---------------------------------------------------------------------

        actuation_allowed = (
            self.safety.evaluate(
                tracking=tracking,
                geometry=geometry,
                adas=adas_result,
            )
        )

        # ---------------------------------------------------------------------
        # RESULTADO
        # ---------------------------------------------------------------------

        processing_ms = (
            time.perf_counter()
            - started
        ) * 1000.0

        self.stats.total_ms = processing_ms

        self.stats.update_fps()

        result = PipelineResult(
            frame_index=self.frame_index,
            timestamp=timestamp,
            detection=detection,
            tracking=tracking,
            geometry=geometry,
            models=tuple(models),
            projections=tuple(projections),
            assignment=assignment,
            adas=adas_result,
            actuation_allowed=(
                actuation_allowed
            ),
            valid=bool(
                detection is not None
            ),
            processing_ms=processing_ms,
        )

        self._last_result = result

        return result

    # =========================================================================
    # MODELOS
    # =========================================================================

    @staticmethod
    def _build_lane_models(
        tracking: LaneTrackingResult,
    ) -> list[Any]:

        models = []

        for track in tracking.lanes:

            if not track.points:
                continue

            try:

                model = (
                    build_lane_model(
                        lane_id=(
                            track.lane_id
                        ),
                        points=(
                            track.points
                        ),
                        min_points=max(
                            4,
                            int(
                                getattr(
                                    config.LANE_MODEL,
                                    "minimum_points",
                                    6,
                                )
                            ),
                        ),
                    )
                )

            except Exception:

                continue

            if model is None:
                continue

            try:

                if not validate_lane_model(
                    model
                ):
                    continue

            except Exception:

                continue

            models.append(model)

        return models

    # =========================================================================
    # PROJEÇÃO
    # =========================================================================

    def _project_models(
        self,
        models: list[Any],
    ) -> list[Any]:

        projections = []

        for model in models:

            try:

                projection = (
                    self.projection
                    .project(model)
                )

            except Exception:

                continue

            if projection is None:
                continue

            if not getattr(
                projection,
                "valid",
                False,
            ):
                continue

            projections.append(
                projection
            )

        return projections

    # =========================================================================
    # VALIDAÇÃO DE COORDENADAS
    # =========================================================================

    @staticmethod
    def _validate_detection_coordinates(
        detection: LaneDetectionResult,
        frame_width: int,
        frame_height: int,
    ) -> bool:

        if detection is None:
            return False

        if (
            frame_width <= 0
            or frame_height <= 0
        ):
            return False

        coordinate_system = (
            detection.metadata.get(
                "coordinate_system"
            )
            if isinstance(
                detection.metadata,
                dict,
            )
            else None
        )

        if (
            coordinate_system is not None
            and coordinate_system
            != "original_frame"
        ):
            return False

        for lane in detection.lanes:

            for point in lane:

                if point is None:
                    return False

                if not point.valid:
                    continue

                if not (
                    finite(point.x)
                    and finite(point.y)
                ):
                    return False

                if (
                    point.x < 0.0
                    or point.x
                    > float(
                        frame_width - 1
                    )
                ):
                    return False

                if (
                    point.y < 0.0
                    or point.y
                    > float(
                        frame_height - 1
                    )
                ):
                    return False

        return True

    # =========================================================================
    # FAILURE
    # =========================================================================

    def _failure_result(
        self,
        timestamp: float,
        started: float,
        error: str,
    ) -> PipelineResult:

        processing_ms = (
            time.perf_counter()
            - started
        ) * 1000.0

        self.safety.reset()

        return PipelineResult(
            frame_index=self.frame_index,
            timestamp=timestamp,
            detection=None,
            tracking=None,
            geometry=None,
            models=(),
            projections=(),
            assignment=None,
            adas=None,
            actuation_allowed=False,
            valid=False,
            processing_ms=processing_ms,
            error=error,
        )

    # =========================================================================
    # RESET
    # =========================================================================

    def _reset_temporal_state(
        self,
    ) -> None:

        self.safety.reset()

        if self.tracker is not None:

            try:
                self.tracker.reset()
            except Exception:
                pass

        if self.adas is not None:

            try:
                self.adas.reset()
            except Exception:
                pass

    # =========================================================================
    # LOOP
    # =========================================================================

    def run(self) -> None:

        if not self.initialized:
            self.initialize()

        if self.capture is None:

            raise RuntimeError(
                "Capture não inicializado."
            )

        self.running = True

        LOGGER.info(
            "Runtime iniciado."
        )

        try:

            while self.running:

                frame = self.get_frame()

                if frame is None:

                    time.sleep(0.001)

                    continue

                result = (
                    self.process_frame(
                        frame
                    )
                )

                self._publish_result(
                    frame,
                    result,
                )

                if self._handle_keyboard():
                    break

        except KeyboardInterrupt:

            LOGGER.info(
                "Interrupção solicitada."
            )

        finally:

            self.shutdown()

    # =========================================================================
    # VISUALIZAÇÃO
    # =========================================================================

    def _publish_result(
        self,
        frame: np.ndarray,
        result: PipelineResult,
    ) -> None:

        # O ADASDisplay não precisa do frame.
        # Ele representa os resultados calculados
        # pelo pipeline.

        del frame

        if self.display is None:
            return

        try:

            self.display.update_from_pipeline(
                geometry=result.geometry,
                adas_state=result.adas,
                active=result.actuation_allowed,
            )

        except Exception:

            LOGGER.debug(
                "Falha ao atualizar ADASDisplay.",
                exc_info=True,
            )

    # =========================================================================
    # TECLADO
    # =========================================================================

    @staticmethod
    def _handle_keyboard() -> bool:

        try:

            key = cv2.waitKey(1) & 0xFF

        except Exception:

            return False

        if key == 27:
            return True

        return False

    # =========================================================================
    # SHUTDOWN
    # =========================================================================

    def shutdown(self) -> None:

        if (
            not self.running
            and self.capture is None
        ):
            return

        LOGGER.info(
            "Encerrando pipeline..."
        )

        self.running = False

        self.safety.reset()

        # ---------------------------------------------------------------------
        # DISPLAY
        # ---------------------------------------------------------------------

        if self.display is not None:

            try:

                self.display.stop()

            except Exception:

                LOGGER.debug(
                    "Erro ao parar display.",
                    exc_info=True,
                )

            self.display = None

        # ---------------------------------------------------------------------
        # CAPTURE
        # ---------------------------------------------------------------------

        if self.capture is not None:

            try:

                self.capture.stop()

            except Exception:

                LOGGER.debug(
                    "Erro ao parar capture.",
                    exc_info=True,
                )

            self.capture = None

        # ---------------------------------------------------------------------
        # OPENCV
        # ---------------------------------------------------------------------

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        self.initialized = False

        LOGGER.info(
            "Pipeline encerrado."
        )


# =============================================================================
# SIGNALS
# =============================================================================

_APP: Optional[
    ForzaAssistents
] = None


def _signal_handler(
    signum: int,
    frame: Any,
) -> None:

    del signum
    del frame

    global _APP

    if _APP is not None:
        _APP.running = False


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:

    global _APP

    setup_logging()

    signal.signal(
        signal.SIGINT,
        _signal_handler,
    )

    if hasattr(
        signal,
        "SIGTERM",
    ):

        signal.signal(
            signal.SIGTERM,
            _signal_handler,
        )

    try:

        _APP = ForzaAssistents()

        _APP.initialize()

        _APP.run()

        return 0

    except Exception:

        LOGGER.exception(
            "Falha fatal no Forza Assistents."
        )

        if _APP is not None:

            try:
                _APP.shutdown()
            except Exception:
                pass

        return 1

    finally:

        _APP = None


if __name__ == "__main__":
    raise SystemExit(
        main()
    )