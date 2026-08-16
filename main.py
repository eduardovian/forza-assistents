"""
main.py

Forza Assistents
================

Orquestrador principal do pipeline ADAS.

Pipeline:

    config
       ↓
    ScreenCapture
       ↓
      YOLOP
       ↓
   LaneTracker
       ↓
    LaneModel
       ↓
 LaneProjection
       ↓
   LaneGeometry
       ↓
 LaneAssignment
       ↓
   ADASState
       ↓
  ADASDisplay

Controle físico permanece desabilitado nesta etapa.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np

from config import (
    ROI,
    CAPTURE,
    YOLOP,
    LANE_TRACKER,
    LANE_GEOMETRY,
    LANE_MODEL,
    LANE_PROJECTION,
    LANE_ASSIGNMENT,
    ADAS,
    SAFETY,
    VISUALIZATION,
    validate_config,
)

from capture.screen_capture import ScreenCapture

from vision.yolop_detector import (
    YOLOPLaneDetector,
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

LOGGER = logging.getLogger("forza_assistents")


def setup_logging() -> None:
    """Inicializa o logging."""

    

    if LOGGER.handlers:
        return

    LOGGER.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)

    LOGGER.addHandler(console)

    LOGGER.propagate = False


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
    model_ms: float = 0.0
    projection_ms: float = 0.0
    geometry_ms: float = 0.0
    assignment_ms: float = 0.0
    adas_ms: float = 0.0

    total_ms: float = 0.0

    _last_time: float = 0.0

    def update_fps(self) -> None:

        now = time.perf_counter()

        if self._last_time > 0.0:

            dt = now - self._last_time

            if dt > 0.0:

                instant_fps = 1.0 / dt

                if self.fps <= 0.0:

                    self.fps = instant_fps

                else:

                    self.fps = (
                        self.fps * 0.9
                        + instant_fps * 0.1
                    )

        self._last_time = now


# =============================================================================
# APLICAÇÃO
# =============================================================================


class ForzaAssistents:

    def __init__(self) -> None:

        validate_config()

        self.running = False
        self.initialized = False

        self.frame_index = 0

        self.stats = RuntimeStatistics()

        self.capture: Optional[ScreenCapture] = None

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

        self.geometry_size: Optional[
            tuple[int, int]
        ] = None

    # =========================================================================
    # INITIALIZAÇÃO
    # =========================================================================

    def initialize(self) -> None:

        if self.initialized:
            return

        LOGGER.info("=" * 60)
        LOGGER.info("FORZA ASSISTENTS")
        LOGGER.info("Inicializando pipeline ADAS...")
        LOGGER.info("Runtime mode: MONITOR")
        LOGGER.info(
            "Controle físico: %s",
            "ENABLED"
            if SAFETY.enable_control
            else "DISABLED",
        )

        # ---------------------------------------------------------------------
        # ROI
        # ---------------------------------------------------------------------

        if not ROI.enabled:

            raise RuntimeError(
                "ROI não calibrado. "
                "Execute calibration/camera_calibration.py "
                "antes de iniciar o sistema."
            )

        LOGGER.info(
            "ROI: (%d, %d, %d, %d) | %dx%d",
            ROI.left,
            ROI.top,
            ROI.right,
            ROI.bottom,
            ROI.width,
            ROI.height,
        )

        # ---------------------------------------------------------------------
        # SCREEN CAPTURE
        #
        # O ScreenCapture atual NÃO recebe ROI no construtor.
        # Ele lê config.ROI diretamente.
        # ---------------------------------------------------------------------

        self.capture = ScreenCapture(
            monitor_index=CAPTURE.monitor_index,
        )

        self.capture.start()

        LOGGER.info(
            "ScreenCapture: READY | %d FPS",
            CAPTURE.target_fps,
        )

        # ---------------------------------------------------------------------
        # YOLOP
        # ---------------------------------------------------------------------

        self.detector = create_default_detector(
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
            providers=[
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
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
        # ---------------------------------------------------------------------

        self.tracker = LaneTracker(
            max_lanes=LANE_TRACKER.max_tracks,
            history_size=LANE_TRACKER.history_size,
            min_points=YOLOP.minimum_lane_points,
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

        self.projection = LaneProjectionEngine(
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

        LOGGER.info(
            "LaneProjection: READY"
        )

        # ---------------------------------------------------------------------
        # ASSIGNMENT
        # ---------------------------------------------------------------------

        self.assignment = LaneAssignment(
            max_lanes=LANE_TRACKER.max_tracks,
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
                LANE_ASSIGNMENT.maximum_lateral_offset_ratio
            ),
            enable_multi_lane_assignment=(
                LANE_ASSIGNMENT.enable_multi_lane_assignment
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

        if VISUALIZATION.enabled:

            self.display = ADASDisplay(
                config=ADASDisplayConfig(
                    source_width=ROI.width,
                    source_height=ROI.height,
                    refresh_hz=30.0,
                )
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
                    "ADASDisplay indisponível: %s",
                    exc,
                )

                self.display = None

        self.initialized = True

        LOGGER.info(
            "Forza Assistents: INITIALIZED"
        )

    # =========================================================================
    # GEOMETRY
    # =========================================================================

    def _get_geometry(
        self,
        frame: np.ndarray,
    ) -> LaneGeometry:

        height, width = frame.shape[:2]

        current_size = (
            width,
            height,
        )

        if (
            self.geometry is not None
            and self.geometry_size == current_size
        ):

            return self.geometry

        LOGGER.info(
            "LaneGeometry: %dx%d",
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
        )

        self.geometry_size = current_size

        return self.geometry

    # =========================================================================
    # CAPTURE
    # =========================================================================

    def get_frame(
        self,
    ) -> Optional[np.ndarray]:

        if self.capture is None:
            return None

        start = time.perf_counter()

        packet = self.capture.read()

        self.stats.capture_ms = (
            time.perf_counter() - start
        ) * 1000.0

        if packet is None:
            return None

        return packet.frame

    # =========================================================================
    # YOLOP
    # =========================================================================

    def detect(
        self,
        frame: np.ndarray,
    ) -> Any:

        if self.detector is None:
            return None

        start = time.perf_counter()

        result = self.detector.detect(
            frame
        )

        self.stats.detection_ms = (
            time.perf_counter() - start
        ) * 1000.0

        return result

    # =========================================================================
    # TRACKING
    # =========================================================================

    def track(
        self,
        detection: Any,
        timestamp: float,
    ) -> Optional[LaneTrackingResult]:

        if self.tracker is None:
            return None

        start = time.perf_counter()

        result = self.tracker.update(
            detection,
            timestamp=timestamp,
        )

        self.stats.tracking_ms = (
            time.perf_counter() - start
        ) * 1000.0

        return result

    # =========================================================================
    # LANE MODEL
    # =========================================================================

    def build_models(
        self,
        tracking: LaneTrackingResult,
    ) -> list:

        start = time.perf_counter()

        models = []

        for track in tracking.lanes:

            if not track.detected_this_frame:
                continue

            if track.missed_frames > 0:
                continue

            if not track.line.valid:
                continue

            points = track.line.points

            if not points:
                continue

            model = build_lane_model(
                lane_id=track.track_id,
                points=points,
                min_points=LANE_MODEL.minimum_points,
            )

            if model is None:
                continue

            model.tracked = True

            model.stable = track.is_stable(
                LANE_TRACKER.min_stable_frames
            )

            if not validate_lane_model(model):
                continue

            models.append(model)

        self.stats.model_ms = (
            time.perf_counter() - start
        ) * 1000.0

        return models

    # =========================================================================
    # PROJECTION
    # =========================================================================

    def project_models(
        self,
        models: list,
    ) -> list:

        if self.projection is None:
            return []

        start = time.perf_counter()

        projections = []

        if not LANE_PROJECTION.enabled:
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
                    "Projection failed: %s",
                    exc,
                )

                continue

            if projection.valid:

                projections.append(
                    projection
                )

        self.stats.projection_ms = (
            time.perf_counter() - start
        ) * 1000.0

        return projections

    # =========================================================================
    # GEOMETRY
    # =========================================================================

    def calculate_geometry(
        self,
        models: list,
        frame: np.ndarray,
    ) -> LaneGeometryResult:

        geometry = self._get_geometry(
            frame
        )

        height, width = frame.shape[:2]

        start = time.perf_counter()

        lane_points = [
            model.line.points
            for model in models
            if (
                model is not None
                and model.line is not None
                and model.line.valid
            )
        ]

        result = geometry.process(
            lane_points,
            image_width=width,
            image_height=height,
        )

        self.stats.geometry_ms = (
            time.perf_counter() - start
        ) * 1000.0

        return result

    # =========================================================================
    # ASSIGNMENT
    # =========================================================================

    def assign(
        self,
        models: list,
        frame: np.ndarray,
    ) -> Optional[LaneAssignmentResult]:

        if self.assignment is None:
            return None

        height, width = frame.shape[:2]

        start = time.perf_counter()

        result = self.assignment.assign(
            models,
            frame_width=width,
            frame_height=height,
        )

        self.stats.assignment_ms = (
            time.perf_counter() - start
        ) * 1000.0

        return result

    # =========================================================================
    # ADAS
    # =========================================================================

    def update_adas(
        self,
        geometry: LaneGeometryResult,
        assignment: Optional[
            LaneAssignmentResult
        ],
        timestamp: float,
    ) -> Optional[ADASStateResult]:

        if self.adas is None:
            return None

        start = time.perf_counter()

        result = self.adas.update(
            geometry=geometry,
            assignment=assignment,
            timestamp=timestamp,
        )

        self.stats.adas_ms = (
            time.perf_counter() - start
        ) * 1000.0

        return result

    # =========================================================================
    # DISPLAY
    # =========================================================================

    def update_display(
        self,
        geometry: LaneGeometryResult,
        adas_state: Optional[
            ADASStateResult
        ],
    ) -> None:

        if self.display is None:
            return

        try:

            self.display.update_from_pipeline(
                geometry=geometry,
                adas_state=adas_state,
                active=bool(
                    adas_state is not None
                    and adas_state.valid
                ),
            )

        except Exception as exc:

            LOGGER.debug(
                "Display update failed: %s",
                exc,
            )

    # =========================================================================
    # FRAME COMPLETO
    # =========================================================================

    def process_frame(
        self,
        frame: np.ndarray,
    ) -> Optional[ADASStateResult]:

        frame_start = time.perf_counter()

        timestamp = time.monotonic()

        # ---------------------------------------------------------------------
        # YOLOP
        # ---------------------------------------------------------------------

        detection = self.detect(
            frame
        )

        if detection is None:
            return None

        # ---------------------------------------------------------------------
        # TRACKER
        # ---------------------------------------------------------------------

        tracking = self.track(
            detection,
            timestamp,
        )

        if tracking is None:
            return None

        # ---------------------------------------------------------------------
        # MODEL
        # ---------------------------------------------------------------------

        models = self.build_models(
            tracking
        )

        # ---------------------------------------------------------------------
        # PROJECTION
        # ---------------------------------------------------------------------

        self.project_models(
            models
        )

        # ---------------------------------------------------------------------
        # GEOMETRY
        # ---------------------------------------------------------------------

        geometry = self.calculate_geometry(
            models,
            frame,
        )

        # ---------------------------------------------------------------------
        # ASSIGNMENT
        # ---------------------------------------------------------------------

        assignment = self.assign(
            models,
            frame,
        )

        # ---------------------------------------------------------------------
        # ADAS
        # ---------------------------------------------------------------------

        adas_state = self.update_adas(
            geometry,
            assignment,
            timestamp,
        )

        # ---------------------------------------------------------------------
        # DISPLAY
        # ---------------------------------------------------------------------

        self.update_display(
            geometry,
            adas_state,
        )

        # ---------------------------------------------------------------------
        # METRICS
        # ---------------------------------------------------------------------

        self.stats.total_ms = (
            time.perf_counter()
            - frame_start
        ) * 1000.0

        return adas_state

    # =========================================================================
    # DIAGNÓSTICO
    # =========================================================================

    def log_frame(
        self,
        frame: np.ndarray,
        adas_state: Optional[
            ADASStateResult
        ],
    ) -> None:

        # Mantemos o diagnóstico simples e independente
        # de um objeto CONFIG inexistente.

        if self.frame_index % 60 != 0:
            return

        state = (
            adas_state.state.value
            if adas_state is not None
            else "none"
        )

        confidence = (
            adas_state.confidence
            if adas_state is not None
            else 0.0
        )

        LOGGER.info(
            (
                "FRAME %d | "
                "FPS=%.1f | "
                "SIZE=%dx%d | "
                "CAP=%.2fms | "
                "DET=%.2fms | "
                "TRK=%.2fms | "
                "MODEL=%.2fms | "
                "PROJ=%.2fms | "
                "GEO=%.2fms | "
                "ASSIGN=%.2fms | "
                "ADAS=%.2fms | "
                "TOTAL=%.2fms | "
                "STATE=%s | "
                "CONF=%.2f"
            ),
            self.frame_index,
            self.stats.fps,
            frame.shape[1],
            frame.shape[0],
            self.stats.capture_ms,
            self.stats.detection_ms,
            self.stats.tracking_ms,
            self.stats.model_ms,
            self.stats.projection_ms,
            self.stats.geometry_ms,
            self.stats.assignment_ms,
            self.stats.adas_ms,
            self.stats.total_ms,
            state,
            confidence,
        )

    # =========================================================================
    # STOP
    # =========================================================================

    def stop(self) -> None:

        self.running = False

        if self.capture is not None:

            try:
                self.capture.stop()
            except Exception:
                LOGGER.exception(
                    "Erro ao parar captura."
                )

        if self.display is not None:

            try:
                self.display.stop()
            except Exception:
                LOGGER.exception(
                    "Erro ao parar display."
                )

        LOGGER.info(
            "Forza Assistents: STOPPED"
        )


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:

    setup_logging()

    app = ForzaAssistents()

    try:

        app.initialize()

        app.running = True

        LOGGER.info(
            "Runtime started."
        )

        while app.running:

            frame = app.get_frame()

            if frame is None:

                time.sleep(0.001)

                continue

            app.frame_index += 1

            app.stats.frame_index = (
                app.frame_index
            )

            app.stats.update_fps()

            app.process_frame(
                frame
            )

            app.log_frame(
                frame,
                None,
            )

            # ESC encerra.
            if (
                cv2.waitKey(1) & 0xFF
            ) == 27:

                break

    except KeyboardInterrupt:

        LOGGER.info(
            "Interrupted by user."
        )

    except Exception:

        LOGGER.exception(
            "Fatal runtime error."
        )

        raise

    finally:

        app.stop()


if __name__ == "__main__":
    main()