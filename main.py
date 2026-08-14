"""
Forza Assistents
================

Pipeline principal de percepção ADAS/LKA para Forza Horizon.

Arquitetura:

    SCREEN CAPTURE
          │
          ▼
        YOLOP
          │
          ▼
     LaneTracker
          │
          ▼
     LaneGeometry
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
          ▼
   ADASStateEstimator
          │
          ├──────────────► ADASDisplay
          │
          └──────────────► OpenCV Debug View

IMPORTANTE
----------

Este arquivo NÃO controla o veículo.

Não envia comandos para:
    - G29
    - teclado
    - mouse
    - acelerador
    - freio
    - volante

O sistema opera em MONITOR / VISION ONLY.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Any, Iterable, Optional

import cv2
import numpy as np
import torch

import config

from capture.screen_capture import ScreenCapture

from vision.yolop_detector import (
    YOLOPLaneDetector,
)

from vision.lane_tracker import (
    LaneTracker,
    LaneTrackingResult,
)

from vision.lane_geometry import (
    LaneGeometry,
    LaneGeometryResult,
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

from visualization.adas_display import (
    ADASDisplay,
)


# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=getattr(
        config,
        "LOG_LEVEL",
        "INFO",
    ),
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger("forza_assistents")


# ============================================================================
# CONSTANTES
# ============================================================================

WINDOW_NAME = "FORZA ASSISTENTS - ADAS"

DEFAULT_ROI = (
    0,
    279,
    1987,
    698,
)

DISPLAY_MAX_WIDTH = 1920

LOG_EVERY_N_FRAMES = 30

MONITOR_MODE = "monitor"

CONTROL_ENABLED = False


# ============================================================================
# UTILITÁRIOS
# ============================================================================

def _safe_len(value: Any) -> int:
    """
    Retorna len() com segurança.
    """
    if value is None:
        return 0

    try:
        return len(value)
    except Exception:
        return 0


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """
    Converte valor para float evitando NaN/inf.
    """
    try:
        result = float(value)

        if not np.isfinite(result):
            return default

        return result

    except (
        TypeError,
        ValueError,
    ):
        return default


def _state_name(
    result: Optional[ADASStateResult],
) -> str:
    """
    Nome textual seguro do estado ADAS.
    """
    if result is None:
        return "unknown"

    state = getattr(
        result,
        "state",
        None,
    )

    if state is None:
        return "unknown"

    value = getattr(
        state,
        "value",
        state,
    )

    return str(value)


def _get_lane_points(
    lane: Any,
) -> list:
    """
    Extrai pontos de diferentes representações de lane.

    Compatível com:

        LaneLine
        TrackedLane
        List[LanePoint]
        List[Point]
    """
    if lane is None:
        return []

    points = getattr(
        lane,
        "points",
        None,
    )

    if points is not None:
        return list(points)

    if isinstance(
        lane,
        (list, tuple),
    ):
        return list(lane)

    return []


def _point_xy(
    point: Any,
) -> Optional[tuple[float, float]]:
    """
    Extrai x/y de LanePoint ou tupla.
    """
    if point is None:
        return None

    if hasattr(point, "x") and hasattr(point, "y"):
        x = _safe_float(point.x)
        y = _safe_float(point.y)

        return x, y

    if isinstance(
        point,
        (list, tuple),
    ) and len(point) >= 2:

        x = _safe_float(point[0])
        y = _safe_float(point[1])

        return x, y

    return None


def _draw_polyline(
    image: np.ndarray,
    points: Iterable[Any],
    color: tuple[int, int, int],
    thickness: int = 3,
) -> None:
    """
    Desenha uma lane sem assumir uma classe específica.
    """
    xy = []

    for point in points:

        value = _point_xy(point)

        if value is None:
            continue

        x, y = value

        if not (
            np.isfinite(x)
            and np.isfinite(y)
        ):
            continue

        xy.append(
            (
                int(round(x)),
                int(round(y)),
            )
        )

    if len(xy) < 2:
        return

    pts = np.asarray(
        xy,
        dtype=np.int32,
    )

    cv2.polylines(
        image,
        [pts],
        False,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _draw_point(
    image: np.ndarray,
    point: Any,
    color: tuple[int, int, int],
    radius: int = 5,
) -> None:
    """
    Desenha ponto individual.
    """
    xy = _point_xy(point)

    if xy is None:
        return

    x, y = xy

    cv2.circle(
        image,
        (
            int(round(x)),
            int(round(y)),
        ),
        radius,
        color,
        -1,
        cv2.LINE_AA,
    )


# ============================================================================
# SISTEMA
# ============================================================================

class ForzaAssistents:
    """
    Orquestrador principal do sistema.
    """

    def __init__(
        self,
        video_path: Optional[str] = None,
        frame_limit: Optional[int] = None,
    ) -> None:

        self.video_path = video_path
        self.frame_limit = frame_limit

        self.mode = MONITOR_MODE

        self.control_enabled = CONTROL_ENABLED

        self.running = False

        self.processing_enabled = True

        self.initialized = False

        # ------------------------------------------------------------------
        # ROI
        # ------------------------------------------------------------------

        self.roi = self._load_roi()

        self._validate_roi()

        # ------------------------------------------------------------------
        # COMPONENTES
        # ------------------------------------------------------------------

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

        # ------------------------------------------------------------------
        # RESULTADOS
        # ------------------------------------------------------------------

        self.detection: Any = None

        self.tracking: Optional[
            LaneTrackingResult
        ] = None

        self.geometry_result: Optional[
            LaneGeometryResult
        ] = None

        self.models: list[Any] = []

        self.projections: list[Any] = []

        self.assignment_result: Optional[
            LaneAssignmentResult
        ] = None

        self.adas_result: Optional[
            ADASStateResult
        ] = None

        # ------------------------------------------------------------------
        # MÉTRICAS
        # ------------------------------------------------------------------

        self.frame_index = 0

        self.last_frame_time = time.perf_counter()

        self.fps = 0.0

        self.frame_time_ms = 0.0

        self.capture_time_ms = 0.0

        self.yolop_time_ms = 0.0

        self.pipeline_time_ms = 0.0

        # ------------------------------------------------------------------
        # SHUTDOWN
        # ------------------------------------------------------------------

        self._shutdown = False

    # ======================================================================
    # ROI
    # ======================================================================

    def _load_roi(self) -> tuple[int, int, int, int]:

        try:

            calibration = config.load_calibration()

            roi = (
                int(calibration["left"]),
                int(calibration["top"]),
                int(calibration["right"]),
                int(calibration["bottom"]),
            )

            logger.info(
                "ROI loaded from calibration: %s",
                roi,
            )

            return roi

        except Exception:

            logger.warning(
                "Calibration unavailable. "
                "Using working ROI: %s",
                DEFAULT_ROI,
            )

            return DEFAULT_ROI

    # ======================================================================

    def _validate_roi(self) -> None:

        left, top, right, bottom = self.roi

        if left < 0 or top < 0:
            raise ValueError(
                f"Invalid ROI: {self.roi}"
            )

        if right <= left:
            raise ValueError(
                f"Invalid ROI width: {self.roi}"
            )

        if bottom <= top:
            raise ValueError(
                f"Invalid ROI height: {self.roi}"
            )

        screen_width = int(
            getattr(
                config,
                "SCREEN_WIDTH",
                2560,
            )
        )

        screen_height = int(
            getattr(
                config,
                "SCREEN_HEIGHT",
                1600,
            )
        )

        if right > screen_width:
            raise ValueError(
                f"ROI exceeds screen width: {self.roi}"
            )

        if bottom > screen_height:
            raise ValueError(
                f"ROI exceeds screen height: {self.roi}"
            )

    # ==========================================================================
    # INITIALIZATION
    # ==========================================================================

    def initialize(self) -> bool:

        logger.info("=" * 42)
        logger.info("FORZA ASSISTENTS")
        logger.info("Initializing ADAS pipeline...")
        logger.info(
            "Runtime mode: %s",
            self.mode,
        )
        logger.info(
            "Physical control: %s",
            "ENABLED"
            if self.control_enabled
            else "DISABLED",
        )
        logger.info("=" * 42)

        # ------------------------------------------------------------------
        # CUDA
        # ------------------------------------------------------------------

        if torch.cuda.is_available():

            logger.info(
                "CUDA: READY | %s",
                torch.cuda.get_device_name(0),
            )

        else:

            logger.warning(
                "CUDA unavailable. "
                "Inference will use CPU."
            )

        # ------------------------------------------------------------------
        # CAPTURE
        # ------------------------------------------------------------------

        try:

            if self.video_path is not None:

                self.capture = ScreenCapture(
                    video_path=self.video_path,
                    target_fps=getattr(
                        config,
                        "CAPTURE_TARGET_FPS",
                        60,
                    ),
                    max_buffer_size=getattr(
                        config,
                        "MAX_FRAME_BUFFER_SIZE",
                        2,
                    ),
                )

            else:

                self.capture = ScreenCapture(
                    region=self.roi,
                    target_fps=getattr(
                        config,
                        "CAPTURE_TARGET_FPS",
                        60,
                    ),
                    backend=getattr(
                        config,
                        "CAPTURE_BACKEND",
                        "bettercam",
                    ),
                    output_color=getattr(
                        config,
                        "CAPTURE_OUTPUT_COLOR",
                        "BGR",
                    ),
                    max_buffer_size=getattr(
                        config,
                        "MAX_FRAME_BUFFER_SIZE",
                        2,
                    ),
                )

            if not self.capture.initialize():

                logger.error(
                    "ScreenCapture initialization failed."
                )

                return False

            self.capture.start()

            logger.info(
                "ScreenCapture: READY | ROI=%dx%d",
                self.roi[2] - self.roi[0],
                self.roi[3] - self.roi[1],
            )

        except Exception:

            logger.exception(
                "ScreenCapture initialization error."
            )

            return False

        # ------------------------------------------------------------------
        # YOLOP
        # ------------------------------------------------------------------

        try:

            self.detector = self._create_yolop()

            if hasattr(
                self.detector,
                "load_model",
            ):

                loaded = self.detector.load_model()

                if loaded is False:

                    logger.error(
                        "YOLOP model failed to load."
                    )

                    return False

            device = getattr(
                self.detector,
                "device",
                "CUDA"
                if torch.cuda.is_available()
                else "CPU",
            )

            logger.info(
                "YOLOP: READY | device=%s",
                device,
            )

        except Exception:

            logger.exception(
                "YOLOP initialization failed."
            )

            return False

        # ------------------------------------------------------------------
        # TRACKER
        # ------------------------------------------------------------------

        try:

            self.tracker = LaneTracker()

            logger.info(
                "LaneTracker: READY"
            )

        except Exception:

            logger.exception(
                "LaneTracker initialization failed."
            )

            return False

        # ------------------------------------------------------------------
        # GEOMETRY
        # ------------------------------------------------------------------

        try:

            roi_width = (
                self.roi[2]
                - self.roi[0]
            )

            roi_height = (
                self.roi[3]
                - self.roi[1]
            )

            logger.info(
                "LaneGeometry: creating %dx%d",
                roi_width,
                roi_height,
            )

            self.geometry = LaneGeometry(
                screen_width=roi_width,
                screen_height=roi_height,
                roi=(
                    0,
                    0,
                    roi_width,
                    roi_height,
                ),
                min_points=getattr(
                    config,
                    "MIN_POINTS_PER_LANE",
                    5,
                ),
            )

            logger.info(
                "LaneGeometry: READY"
            )

        except TypeError:

            # Compatibilidade com versões que
            # usam somente screen_width/height.

            try:

                self.geometry = LaneGeometry(
                    screen_width=(
                        self.roi[2]
                        - self.roi[0]
                    ),
                    screen_height=(
                        self.roi[3]
                        - self.roi[1]
                    ),
                    roi=(
                        0,
                        0,
                        self.roi[2]
                        - self.roi[0],
                        self.roi[3]
                        - self.roi[1],
                    ),
                )

                logger.info(
                    "LaneGeometry: READY"
                )

            except Exception:

                logger.exception(
                    "LaneGeometry initialization failed."
                )

                return False

        except Exception:

            logger.exception(
                "LaneGeometry initialization failed."
            )

            return False

        # ------------------------------------------------------------------
        # PROJECTION
        # ------------------------------------------------------------------

        try:

            self.projection = (
                LaneProjectionEngine()
            )

            logger.info(
                "LaneProjection: READY"
            )

        except Exception:

            logger.exception(
                "LaneProjection initialization failed."
            )

            return False

        # ------------------------------------------------------------------
        # ASSIGNMENT
        # ------------------------------------------------------------------

        try:

            self.assignment = LaneAssignment()

            logger.info(
                "LaneAssignment: READY"
            )

        except Exception:

            logger.exception(
                "LaneAssignment initialization failed."
            )

            return False

        # ------------------------------------------------------------------
        # ADAS
        # ------------------------------------------------------------------

        try:

            self.adas = ADASStateEstimator()

            logger.info(
                "ADASStateEstimator: READY"
            )

        except Exception:

            logger.exception(
                "ADASStateEstimator initialization failed."
            )

            return False

        # ------------------------------------------------------------------
        # DISPLAY
        # ------------------------------------------------------------------

        try:

            self.display = ADASDisplay()

            if hasattr(
                self.display,
                "show",
            ):

                self.display.show()

            logger.info(
                "ADASDisplay: STARTED"
            )

        except Exception:

            logger.warning(
                "ADASDisplay unavailable. "
                "Continuing with OpenCV visualization.",
                exc_info=True,
            )

            self.display = None

        # ------------------------------------------------------------------
        # OPENCV
        # ------------------------------------------------------------------

        cv2.namedWindow(
            WINDOW_NAME,
            cv2.WINDOW_NORMAL,
        )

        cv2.resizeWindow(
            WINDOW_NAME,
            min(
                DISPLAY_MAX_WIDTH,
                self.roi[2] - self.roi[0],
            ),
            min(
                1080,
                self.roi[3] - self.roi[1],
            ),
        )

        # ------------------------------------------------------------------
        # READY
        # ------------------------------------------------------------------

        self.running = True

        self.initialized = True

        logger.info("=" * 42)
        logger.info(
            "Forza Assistents: INITIALIZED"
        )
        logger.info("=" * 42)
        logger.info(
            "Runtime started."
        )
        logger.info(
            "F8 = processing ON/OFF"
        )
        logger.info(
            "ESC = shutdown"
        )

        return True

    # ==========================================================================
    # YOLOP FACTORY
    # ==========================================================================

    def _create_yolop(self) -> YOLOPLaneDetector:
        """
        Cria o detector usando as constantes disponíveis em config.py.

        A implementação aceita pequenas diferenças entre versões
        do detector sem modificar o próprio módulo YOLOP.
        """

        kwargs = {}

        candidates = {
            "model_path": (
                "YOLOP_MODEL_PATH",
                "MODEL_PATH",
            ),
            "device": (
                "YOLOP_DEVICE",
                "DEVICE",
            ),
            "confidence_threshold": (
                "YOLOP_CONFIDENCE_THRESHOLD",
                "CONFIDENCE_THRESHOLD",
            ),
            "input_width": (
                "YOLOP_INPUT_WIDTH",
                "INPUT_WIDTH",
            ),
            "input_height": (
                "YOLOP_INPUT_HEIGHT",
                "INPUT_HEIGHT",
            ),
        }

        for argument, names in candidates.items():

            for name in names:

                if hasattr(
                    config,
                    name,
                ):

                    value = getattr(
                        config,
                        name,
                    )

                    if value is not None:

                        kwargs[
                            argument
                        ] = value

                    break

        try:

            return YOLOPLaneDetector(
                **kwargs
            )

        except TypeError:

            logger.warning(
                "YOLOP constructor rejected "
                "extended configuration. "
                "Retrying with model_path only."
            )

            model_path = kwargs.get(
                "model_path"
            )

            if model_path is None:

                return YOLOPLaneDetector()

            return YOLOPLaneDetector(
                model_path=model_path
            )

    # ==========================================================================
    # FRAME PROCESSING
    # ==========================================================================

    def _process_frame(
        self,
        frame: np.ndarray,
    ) -> None:

        pipeline_start = (
            time.perf_counter()
        )

        # ------------------------------------------------------------------
        # YOLOP
        # ------------------------------------------------------------------

        yolop_start = (
            time.perf_counter()
        )

        detection = (
            self.detector.detect(
                frame
            )
        )

        self.yolop_time_ms = (
            time.perf_counter()
            - yolop_start
        ) * 1000.0

        self.detection = detection

        # ------------------------------------------------------------------
        # TRACKER
        # ------------------------------------------------------------------

        timestamp = time.perf_counter()

        tracking = (
            self.tracker.track(
                detection,
                timestamp=timestamp,
            )
        )

        self.tracking = tracking

        # ------------------------------------------------------------------
        # GEOMETRY
        # ------------------------------------------------------------------

        geometry_input = tracking

        try:

            geometry = self.geometry.compute(
                geometry_input
            )

        except Exception:

            # Alguns estados antigos do módulo
            # esperam LaneDetectionResult.

            geometry = self.geometry.compute(
                detection
            )

        self.geometry_result = geometry

        # ------------------------------------------------------------------
        # LANE MODELS
        # ------------------------------------------------------------------

        models = self._build_models(
            tracking
        )

        self.models = models

        # ------------------------------------------------------------------
        # PROJECTION
        # ------------------------------------------------------------------

        projections = self._project_models(
            models,
            geometry,
        )

        self.projections = projections

        # ------------------------------------------------------------------
        # ASSIGNMENT
        # ------------------------------------------------------------------

        assignment = self._assign_lanes(
            models,
            frame,
            geometry,
        )

        self.assignment_result = assignment

        # ------------------------------------------------------------------
        # ADAS
        # ------------------------------------------------------------------

        try:

            self.adas_result = (
                self.adas.update(
                    geometry
                )
            )

        except Exception:

            logger.exception(
                "ADAS state update failed."
            )

            self.adas_result = None

        # ------------------------------------------------------------------
        # LATENCY
        # ------------------------------------------------------------------

        self.pipeline_time_ms = (
            time.perf_counter()
            - pipeline_start
        ) * 1000.0

    # ==========================================================================
    # MODEL EXTRACTION
    # ==========================================================================

    def _build_models(
        self,
        tracking: Optional[LaneTrackingResult],
    ) -> list[Any]:

        if tracking is None:
            return []

        result = []

        lanes = getattr(
            tracking,
            "active_lanes",
            (),
        )

        for lane in lanes:

            points = _get_lane_points(
                lane
            )

            if len(points) < 2:
                continue

            model = self._make_lane_model(
                lane,
                points,
            )

            if model is not None:
                result.append(model)

        return result

    # ==========================================================================
    # LANE MODEL
    # ==========================================================================

    def _make_lane_model(
        self,
        lane: Any,
        points: list,
    ) -> Any:

        try:

            from vision.lane_types import (
                LaneModel,
                LanePoint,
            )

        except Exception:

            return lane

        lane_points = []

        for point in points:

            xy = _point_xy(point)

            if xy is None:
                continue

            x, y = xy

            confidence = _safe_float(
                getattr(
                    point,
                    "confidence",
                    getattr(
                        lane,
                        "confidence",
                        1.0,
                    ),
                ),
                1.0,
            )

            lane_points.append(
                LanePoint(
                    x=x,
                    y=y,
                    confidence=confidence,
                    valid=True,
                )
            )

        if not lane_points:
            return None

        # ------------------------------------------------------------------
        # Diferentes versões de LaneModel
        # ------------------------------------------------------------------

        constructors = (
            lambda: LaneModel(
                points=lane_points,
                confidence=_safe_float(
                    getattr(
                        lane,
                        "confidence",
                        0.0,
                    )
                ),
                valid=True,
            ),
            lambda: LaneModel(
                points=lane_points
            ),
        )

        for constructor in constructors:

            try:

                return constructor()

            except TypeError:

                continue

            except Exception:

                logger.exception(
                    "LaneModel creation failed."
                )

                return None

        return None

    # ==========================================================================
    # PROJECTION
    # ==========================================================================

    def _project_models(
        self,
        models: list[Any],
        geometry: Optional[LaneGeometryResult],
    ) -> list[Any]:

        if not models:
            return []

        if self.projection is None:
            return []

        results = []

        for model in models:

            try:

                result = self.projection.project(
                    model
                )

                if result is not None:
                    results.append(result)

            except TypeError:

                try:

                    result = self.projection.project(
                        lane=model
                    )

                    if result is not None:
                        results.append(result)

                except Exception:

                    logger.exception(
                        "Lane projection failed."
                    )

            except Exception:

                logger.exception(
                    "Lane projection failed."
                )

        return results

    # ==========================================================================
    # ASSIGNMENT
    # ==========================================================================

    def _assign_lanes(
        self,
        models: list[Any],
        frame: np.ndarray,
        geometry: Optional[LaneGeometryResult],
    ) -> Optional[LaneAssignmentResult]:

        if self.assignment is None:
            return None

        if not models:
            return None

        height, width = frame.shape[:2]

        vehicle_x = (
            geometry.image_center_x
            if geometry is not None
            else width * 0.5
        )

        reference_y = (
            geometry.lane_center_y
            if geometry is not None
            else height * 0.75
        )

        try:

            return self.assignment.assign(
                lanes=models,
                frame_width=float(width),
                frame_height=float(height),
                vehicle_x=float(vehicle_x),
                reference_y=float(reference_y),
            )

        except TypeError:

            try:

                return self.assignment.update(
                    lanes=models,
                    frame_width=float(width),
                    frame_height=float(height),
                    vehicle_x=float(vehicle_x),
                    reference_y=float(reference_y),
                )

            except Exception:

                logger.exception(
                    "Lane assignment failed."
                )

                return None

        except Exception:

            logger.exception(
                "Lane assignment failed."
            )

            return None

    # ==========================================================================
    # DEBUG DRAW
    # ==========================================================================

    def _draw_debug(
        self,
        frame: np.ndarray,
    ) -> np.ndarray:

        image = frame.copy()

        height, width = image.shape[:2]

        # ------------------------------------------------------------------
        # ROI BORDER
        # ------------------------------------------------------------------

        cv2.rectangle(
            image,
            (0, 0),
            (
                width - 1,
                height - 1,
            ),
            (255, 255, 0),
            2,
        )

        # ------------------------------------------------------------------
        # YOLOP LANES
        # ------------------------------------------------------------------

        detection = self.detection

        lanes = getattr(
            detection,
            "lanes",
            [],
        ) if detection is not None else []

        for index, lane in enumerate(lanes):

            color = (
                0,
                165,
                255,
            )

            if index == 0:

                color = (
                    255,
                    0,
                    255,
                )

            _draw_polyline(
                image,
                lane,
                color,
                2,
            )

        # ------------------------------------------------------------------
        # TRACKED LANES
        # ------------------------------------------------------------------

        tracking = self.tracking

        if tracking is not None:

            for lane in getattr(
                tracking,
                "active_lanes",
                (),
            ):

                points = _get_lane_points(
                    lane
                )

                _draw_polyline(
                    image,
                    points,
                    (
                        255,
                        255,
                        0,
                    ),
                    3,
                )

        # ------------------------------------------------------------------
        # GEOMETRY
        # ------------------------------------------------------------------

        geometry = self.geometry_result

        if geometry is not None:

            _draw_polyline(
                image,
                geometry.left_lane_screen,
                (
                    0,
                    165,
                    255,
                ),
                4,
            )

            _draw_polyline(
                image,
                geometry.right_lane_screen,
                (
                    255,
                    0,
                    255,
                ),
                4,
            )

            _draw_polyline(
                image,
                geometry.center_line,
                (
                    0,
                    255,
                    0,
                ),
                4,
            )

            # --------------------------------------------------------------
            # IMAGE CENTER
            # --------------------------------------------------------------

            center_x = int(
                round(
                    geometry.image_center_x
                )
            )

            cv2.line(
                image,
                (
                    center_x,
                    0,
                ),
                (
                    center_x,
                    height,
                ),
                (
                    0,
                    0,
                    255,
                ),
                2,
            )

            # --------------------------------------------------------------
            # LANE CENTER
            # --------------------------------------------------------------

            lane_center_x = int(
                round(
                    geometry.lane_center_x
                )
            )

            lane_center_y = int(
                round(
                    geometry.lane_center_y
                )
            )

            cv2.circle(
                image,
                (
                    lane_center_x,
                    lane_center_y,
                ),
                7,
                (
                    0,
                    255,
                    0,
                ),
                -1,
                cv2.LINE_AA,
            )

        # ------------------------------------------------------------------
        # TEXT
        # ------------------------------------------------------------------

        detection_count = (
            _safe_len(lanes)
        )

        tracking_count = (
            _safe_len(
                getattr(
                    tracking,
                    "active_lanes",
                    (),
                )
            )
            if tracking is not None
            else 0
        )

        geometry_valid = bool(
            geometry is not None
            and geometry.valid
        )

        assignment_valid = bool(
            self.assignment_result is not None
            and getattr(
                self.assignment_result,
                "valid",
                False,
            )
        )

        adas_state = _state_name(
            self.adas_result
        )

        lines = [
            (
                "FORZA ASSISTENTS | "
                f"MODE={self.mode.upper()}"
            ),
            (
                f"FPS={self.fps:.1f} | "
                f"FRAME={self.frame_index}"
            ),
            (
                f"YOLOP={detection_count} | "
                f"TRACKS={tracking_count}"
            ),
            (
                "GEOMETRY="
                + (
                    "VALID"
                    if geometry_valid
                    else "INVALID"
                )
            ),
            (
                f"MODELS={len(self.models)} | "
                f"PROJECTIONS={len(self.projections)}"
            ),
            (
                "ASSIGNMENT="
                + (
                    "VALID"
                    if assignment_valid
                    else "INVALID"
                )
            ),
            (
                f"ADAS={adas_state}"
            ),
            (
                f"YOLOP={self.yolop_time_ms:.1f}ms | "
                f"PIPELINE={self.pipeline_time_ms:.1f}ms"
            ),
        ]

        y = 28

        for line in lines:

            cv2.putText(
                image,
                line,
                (
                    15,
                    y,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (
                    255,
                    255,
                    255,
                ),
                2,
                cv2.LINE_AA,
            )

            y += 27

        # ------------------------------------------------------------------
        # LATERAL ERROR
        # ------------------------------------------------------------------

        if geometry is not None:

            text = (
                f"Lateral error: "
                f"{geometry.lateral_error:+.3f}"
            )

            cv2.putText(
                image,
                text,
                (
                    15,
                    height - 50,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (
                    0,
                    255,
                    0,
                ),
                2,
                cv2.LINE_AA,
            )

            text = (
                f"Heading: "
                f"{geometry.heading_error:+.3f}"
            )

            cv2.putText(
                image,
                text,
                (
                    15,
                    height - 20,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (
                    0,
                    255,
                    0,
                ),
                2,
                cv2.LINE_AA,
            )

        return image

    # ==========================================================================
    # DISPLAY
    # ==========================================================================

    def _update_adas_display(
        self,
    ) -> None:

        if self.display is None:
            return

        result = self.adas_result

        if result is None:

            try:

                self.display.update(
                    system_active=self.processing_enabled,
                    message="Aguardando detecção...",
                )

            except Exception:

                pass

            return

        state = getattr(
            result,
            "state",
            ADASState.UNKNOWN,
        )

        message = {
            ADASState.UNKNOWN:
                "Aguardando faixa.",

            ADASState.LANE_LOST:
                "Faixa não detectada.",

            ADASState.CENTERED:
                "Veículo centralizado.",

            ADASState.SLIGHT_LEFT:
                "Veículo deslocado à esquerda.",

            ADASState.SLIGHT_RIGHT:
                "Veículo deslocado à direita.",

            ADASState.LEFT_WARNING:
                "Atenção: aproximação da linha esquerda.",

            ADASState.RIGHT_WARNING:
                "Atenção: aproximação da linha direita.",

            ADASState.LEFT_DEPARTURE:
                "CRÍTICO: saída pela esquerda.",

            ADASState.RIGHT_DEPARTURE:
                "CRÍTICO: saída pela direita.",
        }.get(
            state,
            "Estado ADAS.",
        )

        kwargs = {
            "system_active": self.processing_enabled,
            "message": message,
            "lane_offset": _safe_float(
                getattr(
                    result,
                    "lateral_error",
                    0.0,
                )
            ),
            "lane_confidence": _safe_float(
                getattr(
                    result,
                    "confidence",
                    0.0,
                )
            ),
            "left_lane_detected": bool(
                self.geometry_result is not None
                and self.geometry_result.left_lane_screen
            ),
            "right_lane_detected": bool(
                self.geometry_result is not None
                and self.geometry_result.right_lane_screen
            ),
        }

        try:

            self.display.update(
                **kwargs
            )

        except TypeError:

            try:

                self.display.update(
                    system_active=self.processing_enabled,
                    message=message,
                )

            except Exception:

                logger.debug(
                    "ADASDisplay update failed.",
                    exc_info=True,
                )

        except Exception:

            logger.debug(
                "ADASDisplay update failed.",
                exc_info=True,
            )

    # ==========================================================================
    # FPS
    # ==========================================================================

    def _update_fps(
        self,
        frame_start: float,
    ) -> None:

        now = time.perf_counter()

        self.frame_time_ms = (
            now - frame_start
        ) * 1000.0

        if self.frame_time_ms > 0:

            instantaneous = (
                1000.0
                / self.frame_time_ms
            )

            if self.fps <= 0.0:

                self.fps = instantaneous

            else:

                self.fps = (
                    self.fps * 0.90
                    + instantaneous * 0.10
                )

    # ==========================================================================
    # MAIN LOOP
    # ==========================================================================

    def run(self) -> None:

        if not self.initialized:

            raise RuntimeError(
                "ForzaAssistents is not initialized."
            )

        logger.info(
            "Main loop started."
        )

        try:

            while self.running:

                frame_start = (
                    time.perf_counter()
                )

                # ----------------------------------------------------------
                # FRAME LIMIT
                # ----------------------------------------------------------

                if (
                    self.frame_limit is not None
                    and self.frame_index
                    >= self.frame_limit
                ):

                    logger.info(
                        "Frame limit reached: %d",
                        self.frame_limit,
                    )

                    break

                # ----------------------------------------------------------
                # CAPTURE
                # ----------------------------------------------------------

                capture_start = (
                    time.perf_counter()
                )

                try:

                    frame = (
                        self.capture.get_latest_frame()
                    )

                except Exception:

                    logger.exception(
                        "Capture frame retrieval failed."
                    )

                    frame = None

                self.capture_time_ms = (
                    time.perf_counter()
                    - capture_start
                ) * 1000.0

                if frame is None:

                    key = (
                        cv2.waitKey(1)
                        & 0xFF
                    )

                    if key == 27:
                        break

                    continue

                # ----------------------------------------------------------
                # FRAME VALIDATION
                # ----------------------------------------------------------

                if not isinstance(
                    frame,
                    np.ndarray,
                ):

                    continue

                if frame.ndim != 3:

                    continue

                if frame.shape[2] != 3:

                    continue

                # ----------------------------------------------------------
                # PROCESSING
                # ----------------------------------------------------------

                if self.processing_enabled:

                    try:

                        self._process_frame(
                            frame
                        )

                    except Exception:

                        logger.exception(
                            "Pipeline processing failed."
                        )

                        self.detection = None
                        self.tracking = None
                        self.geometry_result = None
                        self.models = []
                        self.projections = []
                        self.assignment_result = None

                # ----------------------------------------------------------
                # DEBUG
                # ----------------------------------------------------------

                display_frame = (
                    self._draw_debug(
                        frame
                    )
                )

                # ----------------------------------------------------------
                # ADAS DISPLAY
                # ----------------------------------------------------------

                self._update_adas_display()

                # ----------------------------------------------------------
                # RESIZE
                # ----------------------------------------------------------

                preview = display_frame

                if (
                    preview.shape[1]
                    > DISPLAY_MAX_WIDTH
                ):

                    scale = (
                        DISPLAY_MAX_WIDTH
                        / float(
                            preview.shape[1]
                        )
                    )

                    preview = cv2.resize(
                        preview,
                        (
                            int(
                                preview.shape[1]
                                * scale
                            ),
                            int(
                                preview.shape[0]
                                * scale
                            ),
                        ),
                        interpolation=cv2.INTER_AREA,
                    )

                # ----------------------------------------------------------
                # SHOW
                # ----------------------------------------------------------

                cv2.imshow(
                    WINDOW_NAME,
                    preview,
                )

                # ----------------------------------------------------------
                # KEYBOARD
                # ----------------------------------------------------------

                key = (
                    cv2.waitKey(1)
                    & 0xFF
                )

                # ESC
                if key == 27:

                    logger.info(
                        "ESC pressed."
                    )

                    break

                # F8
                elif key == 0x77:

                    self.processing_enabled = (
                        not self.processing_enabled
                    )

                    logger.info(
                        "Processing: %s",
                        (
                            "ENABLED"
                            if self.processing_enabled
                            else "DISABLED"
                        ),
                    )

                # ----------------------------------------------------------
                # FRAME ACCOUNTING
                # ----------------------------------------------------------

                self.frame_index += 1

                self._update_fps(
                    frame_start
                )

                # ----------------------------------------------------------
                # LOG
                # ----------------------------------------------------------

                if (
                    self.frame_index
                    == 1
                    or self.frame_index
                    % LOG_EVERY_N_FRAMES
                    == 0
                ):

                    tracks = (
                        _safe_len(
                            getattr(
                                self.tracking,
                                "active_lanes",
                                (),
                            )
                        )
                    )

                    geometry_state = (
                        "VALID"
                        if (
                            self.geometry_result
                            is not None
                            and self.geometry_result.valid
                        )
                        else "INVALID"
                    )

                    assignment_state = (
                        "VALID"
                        if (
                            self.assignment_result
                            is not None
                            and getattr(
                                self.assignment_result,
                                "valid",
                                False,
                            )
                        )
                        else "INVALID"
                    )

                    logger.info(
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
                        self.frame_index,
                        _safe_len(
                            getattr(
                                self.detection,
                                "lanes",
                                [],
                            )
                        ),
                        tracks,
                        geometry_state,
                        len(self.models),
                        len(self.projections),
                        assignment_state,
                        _state_name(
                            self.adas_result
                        ),
                        self.frame_time_ms,
                    )

        except KeyboardInterrupt:

            logger.info(
                "Interrupted by user."
            )

        except Exception:

            logger.exception(
                "Fatal error in main loop."
            )

        finally:

            self.shutdown()

    # ==========================================================================
    # SHUTDOWN
    # ==========================================================================

    def shutdown(self) -> None:

        if self._shutdown:
            return

        self._shutdown = True

        logger.info(
            "Shutting down Forza Assistents..."
        )

        self.running = False

        # ------------------------------------------------------------------
        # DISPLAY
        # ------------------------------------------------------------------

        if self.display is not None:

            try:

                if hasattr(
                    self.display,
                    "close",
                ):

                    self.display.close()

            except Exception:

                logger.debug(
                    "ADASDisplay shutdown error.",
                    exc_info=True,
                )

            self.display = None

        # ------------------------------------------------------------------
        # CAPTURE
        # ------------------------------------------------------------------

        if self.capture is not None:

            try:

                self.capture.stop()

            except Exception:

                logger.debug(
                    "Capture shutdown error.",
                    exc_info=True,
                )

            self.capture = None

        # ------------------------------------------------------------------
        # OPENCV
        # ------------------------------------------------------------------

        try:

            cv2.destroyAllWindows()

        except Exception:

            pass

        logger.info(
            "Forza Assistents stopped."
        )


# ============================================================================
# CLI
# ============================================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Forza Assistents "
            "ADAS/LKA perception pipeline"
        )
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help=(
            "Maximum number of frames "
            "to process."
        ),
    )

    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help=(
            "Optional video input "
            "instead of screen capture."
        ),
    )

    args = parser.parse_args()

    if (
        args.frames is not None
        and args.frames <= 0
    ):

        parser.error(
            "--frames must be greater than zero."
        )

    system = ForzaAssistents(
        video_path=args.video,
        frame_limit=args.frames,
    )

    try:

        if not system.initialize():

            logger.error(
                "System initialization failed."
            )

            system.shutdown()

            sys.exit(1)

        system.run()

    except Exception:

        logger.exception(
            "Fatal application error."
        )

        system.shutdown()

        sys.exit(1)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()