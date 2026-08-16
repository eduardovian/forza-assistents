"""
main.py

Forza Assistents
================

Orquestrador principal do pipeline ADAS.

Pipeline:

    Config
       ↓
    Screen Capture
       ↓
      YOLOP
       ↓
   Lane Tracker
       ↓
    Lane Model
       ↓
 Lane Projection
       ↓
   Lane Geometry
       ↓
 Lane Assignment
       ↓
   ADAS State
       ↓
  ADAS Display

Princípios
----------
- Toda configuração vem de config.py.
- Nenhum módulo de percepção possui estado global.
- LaneTracker é responsável pela memória temporal.
- LaneModel representa matematicamente cada lane observada.
- LaneProjection extrapola matematicamente a lane.
- LaneGeometry trabalha somente com geometria observada.
- LaneAssignment identifica o corredor atual.
- ADASState determina o estado sem controlar o veículo.
- Controle físico permanece desabilitado nesta etapa.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np

from config import (
    CONFIG,
    CAPTURE,
    YOLOP,
    LANE_TRACKER,
    LANE_GEOMETRY,
    LANE_PROJECTION,
    LANE_ASSIGNMENT,
    ADAS,
    SAFETY,
    VISUALIZATION,
    ensure_directories,
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
    """Inicializa o logging global."""

    ensure_directories()

    if LOGGER.handlers:
        return

    level = getattr(
        logging,
        CONFIG.logging.level.upper(),
        logging.INFO,
    )

    LOGGER.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    )

    if CONFIG.logging.log_to_console:
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        LOGGER.addHandler(console)

    if CONFIG.logging.log_to_file:
        logfile = (
            CONFIG.logging.directory
            / CONFIG.logging.filename
        )

        file_handler = logging.FileHandler(
            logfile,
            encoding="utf-8",
        )

        file_handler.setFormatter(
            formatter
        )

        LOGGER.addHandler(
            file_handler
        )

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

            dt = (
                now
                - self._last_time
            )

            if dt > 0.0:

                instant_fps = (
                    1.0 / dt
                )

                if self.fps <= 0.0:

                    self.fps = (
                        instant_fps
                    )

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

        self.stats = (
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

        self.geometry_size: Optional[
            tuple[int, int]
        ] = None

    # =========================================================================
    # INITIALIZAÇÃO
    # =========================================================================

    def initialize(self) -> None:

        if self.initialized:
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
            CONFIG.runtime_mode.value,
        )

        LOGGER.info(
            "Physical control: %s",
            (
                "ENABLED"
                if SAFETY.enable_control
                else "DISABLED"
            ),
        )

        # =====================================================================
        # CAPTURE
        # =====================================================================

        roi = (
            CAPTURE.roi
            if CAPTURE.use_roi
            else None
        )

        if roi is not None:

            LOGGER.info(
                "ROI: (%d, %d, %d, %d) | %dx%d",
                roi[0],
                roi[1],
                roi[2],
                roi[3],
                roi[2] - roi[0],
                roi[3] - roi[1],
            )

        else:

            LOGGER.info(
                "ROI: DISABLED — full screen"
            )

        self.capture = ScreenCapture(
            region=roi,
            target_fps=CAPTURE.target_fps,
            backend=CAPTURE.backend,
            output_color=(
                CAPTURE.output_color_format
            ),
            max_buffer_size=(
                CAPTURE.max_buffer_size
            ),
        )

        if not self.capture.initialize():

            raise RuntimeError(
                "Failed to initialize "
                "ScreenCapture."
            )

        self.capture.start()

        LOGGER.info(
            "ScreenCapture: READY"
        )

        # =====================================================================
        # YOLOP
        # =====================================================================

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
                providers=[
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider",
                ],
            )
        )

        if not self.detector.load_model():

            raise RuntimeError(
                "Failed to load YOLOP: "
                f"{self.detector.last_error}"
            )

        LOGGER.info(
            "YOLOP: READY | device=%s",
            self.detector.get_device_name(),
        )

        # =====================================================================
        # TRACKER
        # =====================================================================

        self.tracker = LaneTracker(
            max_lanes=(
                LANE_TRACKER.max_tracks
            ),
            history_size=(
                LANE_TRACKER.history_size
            ),
            min_points=(
                YOLOP.minimum_lane_points
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

        # =====================================================================
        # PROJECTION
        # =====================================================================

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

        # =====================================================================
        # ASSIGNMENT
        # =====================================================================

        self.assignment = LaneAssignment(
            max_lanes=(
                LANE_TRACKER.max_tracks
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

        # =====================================================================
        # ADAS
        # =====================================================================

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

        # =====================================================================
        # DISPLAY
        # =====================================================================

        if VISUALIZATION.enabled:

            source_width = (
                CAPTURE.roi[2]
                - CAPTURE.roi[0]
                if CAPTURE.use_roi
                else 2560
            )

            source_height = (
                CAPTURE.roi[3]
                - CAPTURE.roi[1]
                if CAPTURE.use_roi
                else 1600
            )

            self.display = ADASDisplay(
                config=ADASDisplayConfig(
                    source_width=source_width,
                    source_height=source_height,
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
                    "ADASDisplay unavailable: %s",
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

        height, width = (
            frame.shape[:2]
        )

        current_size = (
            width,
            height,
        )

        if (
            self.geometry is not None
            and self.geometry_size
            == current_size
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
        )

        self.geometry_size = (
            current_size
        )

        return self.geometry

    # =========================================================================
    # FRAME
    # =========================================================================

    def get_frame(
        self,
    ) -> Optional[np.ndarray]:

        if self.capture is None:
            return None

        start = (
            time.perf_counter()
        )

        frame = (
            self.capture.get_latest_frame()
        )

        self.stats.capture_ms = (
            time.perf_counter()
            - start
        ) * 1000.0

        return frame

    # =========================================================================
    # YOLOP
    # =========================================================================

    def detect(
        self,
        frame: np.ndarray,
    ) -> Any:

        if self.detector is None:
            return None

        start = (
            time.perf_counter()
        )

        result = (
            self.detector.detect(
                frame
            )
        )

        self.stats.detection_ms = (
            time.perf_counter()
            - start
        ) * 1000.0

        return result

    # =========================================================================
    # TRACKING
    # =========================================================================

    def track(
        self,
        detection: Any,
        timestamp: float,
    ) -> Optional[
        LaneTrackingResult
    ]:

        if self.tracker is None:
            return None

        start = (
            time.perf_counter()
        )

        result = self.tracker.update(
            detection,
            timestamp=timestamp,
        )

        self.stats.tracking_ms = (
            time.perf_counter()
            - start
        ) * 1000.0

        return result

    # =========================================================================
    # MODEL
    # =========================================================================

    def build_models(
        self,
        tracking: LaneTrackingResult,
    ) -> list:

        start = (
            time.perf_counter()
        )

        models = []

        for track in tracking.lanes:

            # ---------------------------------------------------------------
            # IMPORTANTE:
            #
            # Uma lane perdida não deve virar uma nova observação geométrica.
            #
            # O tracker pode manter a identidade temporal, mas o model
            # matemático deste frame somente usa uma detecção direta.
            # ---------------------------------------------------------------

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
                min_points=(
                    LANE_GEOMETRY.min_points
                ),
            )

            if model is None:
                continue

            model.tracked = True

            model.stable = (
                track.is_stable(
                    LANE_TRACKER.min_stable_frames
                )
            )

            if not validate_lane_model(
                model
            ):
                continue

            models.append(
                model
            )

        self.stats.model_ms = (
            time.perf_counter()
            - start
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

        start = (
            time.perf_counter()
        )

        projections = []

        for model in models:

            try:

                projection = (
                    self.projection.project(
                        model
                    )
                )

            except Exception:

                continue

            if projection.valid:

                projections.append(
                    projection
                )

        self.stats.projection_ms = (
            time.perf_counter()
            - start
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

        geometry = (
            self._get_geometry(
                frame
            )
        )

        height, width = (
            frame.shape[:2]
        )

        # ---------------------------------------------------------------------
        # Geometry recebe as lanes observadas.
        #
        # NÃO passamos as projeções como observações.
        # Isso preserva a separação entre:
        #
        # observação → geometria
        # extrapolação → projection
        # ---------------------------------------------------------------------

        start = (
            time.perf_counter()
        )

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
            time.perf_counter()
            - start
        ) * 1000.0

        return result

    # =========================================================================
    # ASSIGNMENT
    # =========================================================================

    def assign(
        self,
        models: list,
        frame: np.ndarray,
    ) -> Optional[
        LaneAssignmentResult
    ]:

        if self.assignment is None:
            return None

        height, width = (
            frame.shape[:2]
        )

        start = (
            time.perf_counter()
        )

        result = self.assignment.assign(
            models,
            frame_width=width,
            frame_height=height,
        )

        self.stats.assignment_ms = (
            time.perf_counter()
            - start
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
    ) -> Optional[
        ADASStateResult
    ]:

        if self.adas is None:
            return None

        start = (
            time.perf_counter()
        )

        result = self.adas.update(
            geometry=geometry,
            assignment=assignment,
            timestamp=timestamp,
        )

        self.stats.adas_ms = (
            time.perf_counter()
            - start
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
    # PIPELINE COMPLETO
    # =========================================================================

    def process_frame(
        self,
        frame: np.ndarray,
    ) -> Optional[
        ADASStateResult
    ]:

        frame_start = (
            time.perf_counter()
        )

        timestamp = (
            time.monotonic()
        )

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
        # LANE MODEL
        # ---------------------------------------------------------------------

        models = self.build_models(
            tracking
        )

        # ---------------------------------------------------------------------
        # PROJECTION
        # ---------------------------------------------------------------------

        #
        # A projeção é calculada para cada LaneModel.
        #
        # Ela permanece separada da geometria observada.
        #

        self.project_models(
            models
        )

        # ---------------------------------------------------------------------
        # GEOMETRY
        # ---------------------------------------------------------------------

        geometry = (
            self.calculate_geometry(
                models,
                frame,
            )
        )

        # ---------------------------------------------------------------------
        # ASSIGNMENT
        # ---------------------------------------------------------------------

        assignment = self.assign(
            models,
            frame,
        )

        # ---------------------------------------------------------------------
        # ADAS STATE
        # ---------------------------------------------------------------------

        adas_state = (
            self.update_adas(
                geometry,
                assignment,
                timestamp,
            )
        )

        # ---------------------------------------------------------------------
        # DISPLAY
        # ---------------------------------------------------------------------

        self.update_display(
            geometry,
            adas_state,
        )

        # ---------------------------------------------------------------------
        # TOTAL
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

        debug_interval = getattr(
            CONFIG,
            "debug_interval",
            60,
        )

        if debug_interval <= 0:
            return

        if (
            self.frame_index
            % debug_interval
            != 0
        ):
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
                pass

        if self.display is not None:

            try:
                self.display.stop()
            except Exception:
                pass

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

                time.sleep(
                    0.001
                )

                continue

            app.frame_index += 1

            app.stats.frame_index = (
                app.frame_index
            )

            app.stats.update_fps()

            # ---------------------------------------------------------------
            # PIPELINE
            # ---------------------------------------------------------------

            adas_state = (
                app.process_frame(
                    frame
                )
            )

            app.log_frame(
                frame,
                adas_state,
            )

            # ---------------------------------------------------------------
            # ESC
            # ---------------------------------------------------------------

            if (
                VISUALIZATION.enabled
                and (
                    cv2.waitKey(
                        max(
                            1,
                            VISUALIZATION.wait_key_ms,
                        )
                    )
                    & 0xFF
                )
                == 27
            ):

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