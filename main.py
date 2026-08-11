
"""
main.py

Forza Horizon 6 ADAS/LKA
Pipeline principal com YOLOP ONNX.

Arquitetura:

    ScreenCapture
          |
          v
        Frame
          |
          v
         ROI
          |
          v
        YOLOP
          |
          v
    LaneDetectionResult
          |
          v
     LaneGeometry
          |
          v
    EMATemporalFilter
          |
          v
   ADASStateEstimator
          |
          v
    Safety Gate
          |
          +----> VISÃO
          |
          +----> PAINEL ADAS

IMPORTANTE:

- YOLOP é responsável somente pela detecção.
- LaneGeometry calcula a geometria.
- EMATemporalFilter suaviza.
- ADASStateEstimator determina o estado.
- Predição de uma lane é somente visual.
- Predição nunca autoriza atuação.
- G29 permanece desabilitado pelo config.py.
"""

from __future__ import annotations

import argparse
import inspect
import logging
import math
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

import config

from vision.yolop_detector import (
    YOLOPLaneDetector,
    LaneDetectionResult,
    LanePoint,
)

from vision.lane_geometry import (
    LaneGeometry,
    LaneGeometryResult,
)

from vision.temporal_filter import (
    EMATemporalFilter,
)

from vision.adas_state import (
    ADASState,
    ADASStateEstimator,
    ADASStateResult,
)

try:
    from capture.screen_capture import ScreenCapture
except ImportError:
    ScreenCapture = None


# =============================================================================
# LOG
# =============================================================================

logger = logging.getLogger("forza_adas")


# =============================================================================
# CONSTANTES
# =============================================================================

SCREEN_WIDTH = int(config.SCREEN_WIDTH)
SCREEN_HEIGHT = int(config.SCREEN_HEIGHT)

ROI = (
    int(config.ROI_LEFT),
    int(config.ROI_TOP),
    int(config.ROI_RIGHT),
    int(config.ROI_BOTTOM),
)

ROI_LEFT, ROI_TOP, ROI_RIGHT, ROI_BOTTOM = ROI

ROI_WIDTH = ROI_RIGHT - ROI_LEFT
ROI_HEIGHT = ROI_BOTTOM - ROI_TOP

IMAGE_CENTER_X = SCREEN_WIDTH / 2.0
IMAGE_CENTER_Y = SCREEN_HEIGHT / 2.0

YOLOP_INPUT_WIDTH = int(
    getattr(config, "YOLOP_INPUT_WIDTH", 640)
)

YOLOP_INPUT_HEIGHT = int(
    getattr(config, "YOLOP_INPUT_HEIGHT", 640)
)

CONTROL_ENABLED = bool(
    getattr(config, "G29_CONTROL_ENABLED", False)
)

MIN_ADAS_CONFIDENCE = 0.55
MIN_STABLE_VALID_FRAMES = 4

PREDICTED_LANE_WIDTH = 650.0
MIN_PREDICTION_POINTS = 5

WINDOW_VISION = "Forza Horizon 6 - VISION"
WINDOW_ADAS = "Forza Horizon 6 - ADAS"

FONT = cv2.FONT_HERSHEY_SIMPLEX


# =============================================================================
# TIPOS
# =============================================================================

Point = Tuple[float, float]


# =============================================================================
# UTILITÁRIOS
# =============================================================================

def finite(value: float) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def clamp01(value: float) -> float:
    if not finite(value):
        return 0.0

    return float(np.clip(value, 0.0, 1.0))


def safe_float(value, default: float = 0.0) -> float:
    try:
        value = float(value)

        if not np.isfinite(value):
            return default

        return value

    except (TypeError, ValueError):
        return default


# =============================================================================
# RESULTADO DO PIPELINE
# =============================================================================

@dataclass
class PipelineResult:

    detection: Optional[LaneDetectionResult]

    geometry: Optional[LaneGeometryResult]

    adas: Optional[ADASStateResult]

    predicted_lane: List[Point]

    real_geometry: bool

    prediction_active: bool

    actuation_allowed: bool

    timestamp: float

    processing_ms: float


# =============================================================================
# PREDITOR DE UMA LANE
# =============================================================================

class SingleLanePredictor:

    def __init__(
        self,
        roi: Tuple[int, int, int, int],
        predicted_width: float = PREDICTED_LANE_WIDTH,
        samples: int = 40,
    ) -> None:

        self.roi_left = float(roi[0])
        self.roi_right = float(roi[2])

        self.predicted_width = float(
            predicted_width
        )

        self.samples = max(
            10,
            int(samples),
        )

    @staticmethod
    def _fit(
        points: Sequence[Point],
    ) -> Optional[Tuple[np.ndarray, float, float]]:

        if len(points) < MIN_PREDICTION_POINTS:
            return None

        arr = np.asarray(
            points,
            dtype=np.float64,
        )

        if arr.ndim != 2 or arr.shape[1] != 2:
            return None

        x = arr[:, 0]
        y = arr[:, 1]

        if not np.all(np.isfinite(x)):
            return None

        if not np.all(np.isfinite(y)):
            return None

        y_min = float(np.min(y))
        y_max = float(np.max(y))

        if y_max - y_min < 40.0:
            return None

        center = float(np.mean(y))
        scale = max(
            float(np.ptp(y)),
            1.0,
        )

        yn = (
            (y - center)
            / scale
        )

        degree = min(
            2,
            len(points) - 1,
        )

        try:

            coeff = np.polyfit(
                yn,
                x,
                degree,
            )

        except (
            np.linalg.LinAlgError,
            ValueError,
        ):

            return None

        return (
            np.asarray(
                coeff,
                dtype=np.float64,
            ),
            center,
            scale,
        )

    @staticmethod
    def _evaluate(
        model: Tuple[np.ndarray, float, float],
        y: np.ndarray,
    ) -> np.ndarray:

        coeff, center, scale = model

        yn = (
            (y - center)
            / max(scale, 1.0)
        )

        return np.polyval(
            coeff,
            yn,
        )

    def predict(
        self,
        visible_lane: Sequence[Point],
        visible_is_left: bool,
    ) -> List[Point]:

        model = self._fit(
            visible_lane
        )

        if model is None:
            return []

        arr = np.asarray(
            visible_lane,
            dtype=np.float64,
        )

        y_min = float(
            np.min(arr[:, 1])
        )

        y_max = float(
            np.max(arr[:, 1])
        )

        ys = np.linspace(
            y_min,
            y_max,
            self.samples,
        )

        visible_x = self._evaluate(
            model,
            ys,
        )

        normalized = (
            (ys - y_min)
            / max(
                y_max - y_min,
                1.0,
            )
        )

        width = (
            self.predicted_width
            * (
                0.55
                + 0.45
                * normalized
            )
        )

        if visible_is_left:
            predicted_x = (
                visible_x + width
            )
        else:
            predicted_x = (
                visible_x - width
            )

        predicted_x = np.clip(
            predicted_x,
            self.roi_left,
            self.roi_right,
        )

        return [
            (
                float(x),
                float(y),
            )
            for x, y in zip(
                predicted_x,
                ys,
            )
            if finite(x)
            and finite(y)
        ]


# =============================================================================
# CONVERSÃO DE LANE
# =============================================================================

def lane_to_screen(
    lane: Sequence[LanePoint],
    input_width: float,
    input_height: float,
    roi: Tuple[int, int, int, int],
) -> List[Point]:

    result: List[Point] = []

    if input_width <= 0 or input_height <= 0:
        return result

    roi_left, roi_top, roi_right, roi_bottom = (
        map(float, roi)
    )

    roi_width = (
        roi_right - roi_left
    )

    roi_height = (
        roi_bottom - roi_top
    )

    for point in lane:

        if not point.valid:
            continue

        if not finite(point.x):
            continue

        if not finite(point.y):
            continue

        x = float(point.x)
        y = float(point.y)

        if (
            x < 0.0
            or x > input_width
            or y < 0.0
            or y > input_height
        ):
            continue

        sx = (
            roi_left
            + (
                x / input_width
            )
            * roi_width
        )

        sy = (
            roi_top
            + (
                y / input_height
            )
            * roi_height
        )

        if finite(sx) and finite(sy):

            result.append(
                (
                    sx,
                    sy,
                )
            )

    result.sort(
        key=lambda p: p[1]
    )

    return result


# =============================================================================
# VISUALIZADOR
# =============================================================================

class Visualizer:

    @staticmethod
    def polyline(
        frame: np.ndarray,
        points: Sequence[Point],
        color: Tuple[int, int, int],
        thickness: int = 3,
        dashed: bool = False,
    ) -> None:

        if len(points) < 2:
            return

        pts = np.asarray(
            points,
            dtype=np.int32,
        ).reshape(-1, 1, 2)

        if not dashed:

            cv2.polylines(
                frame,
                [pts],
                False,
                color,
                thickness,
                cv2.LINE_AA,
            )

            return

        for i in range(
            len(pts) - 1
        ):

            if i % 2 == 0:

                cv2.line(
                    frame,
                    tuple(pts[i, 0]),
                    tuple(pts[i + 1, 0]),
                    color,
                    thickness,
                    cv2.LINE_AA,
                )

    def draw_vision(
        self,
        frame: np.ndarray,
        result: Optional[PipelineResult],
    ) -> np.ndarray:

        output = frame.copy()

        # -----------------------------------------------------------------
        # ROI
        # -----------------------------------------------------------------

        cv2.rectangle(
            output,
            (ROI_LEFT, ROI_TOP),
            (ROI_RIGHT, ROI_BOTTOM),
            getattr(
                config,
                "COLOR_ROI",
                (255, 255, 0),
            ),
            2,
        )

        # -----------------------------------------------------------------
        # Centro da imagem
        # -----------------------------------------------------------------

        cv2.line(
            output,
            (
                int(IMAGE_CENTER_X),
                ROI_TOP,
            ),
            (
                int(IMAGE_CENTER_X),
                ROI_BOTTOM,
            ),
            getattr(
                config,
                "COLOR_IMAGE_CENTER",
                (0, 0, 255),
            ),
            2,
        )

        if result is None:

            self._text(
                output,
                "INITIALIZING YOLOP...",
                30,
                45,
                0.8,
                (255, 255, 255),
            )

            return output

        geometry = result.geometry

        # -----------------------------------------------------------------
        # Predição
        # -----------------------------------------------------------------

        if result.predicted_lane:

            self.polyline(
                output,
                result.predicted_lane,
                (0, 255, 255),
                3,
                dashed=True,
            )

        # -----------------------------------------------------------------
        # Lanes reais
        # -----------------------------------------------------------------

        if geometry is not None:

            self.polyline(
                output,
                geometry.left_lane_screen,
                getattr(
                    config,
                    "COLOR_LEFT_LANE",
                    (0, 165, 255),
                ),
                4,
            )

            self.polyline(
                output,
                geometry.right_lane_screen,
                getattr(
                    config,
                    "COLOR_RIGHT_LANE",
                    (255, 0, 255),
                ),
                4,
            )

            self.polyline(
                output,
                geometry.center_line,
                getattr(
                    config,
                    "COLOR_LANE_CENTER",
                    (0, 255, 0),
                ),
                4,
            )

            for lane in geometry.additional_lanes_screen:

                self.polyline(
                    output,
                    lane,
                    (150, 150, 150),
                    2,
                )

            if geometry.valid:

                cv2.circle(
                    output,
                    (
                        int(
                            geometry.lane_center_x
                        ),
                        int(
                            geometry.lane_center_y
                        ),
                    ),
                    8,
                    (0, 255, 0),
                    -1,
                )

        # -----------------------------------------------------------------
        # Informações
        # -----------------------------------------------------------------

        self._draw_vision_status(
            output,
            result,
        )

        return output

    def _draw_vision_status(
        self,
        frame: np.ndarray,
        result: PipelineResult,
    ) -> None:

        if result.real_geometry:

            text = "LANE GEOMETRY: REAL"
            color = (0, 255, 0)

        elif result.prediction_active:

            text = "LANE: SINGLE-LANE PREDICTION"
            color = (0, 255, 255)

        else:

            text = "LANE GEOMETRY: LOST"
            color = (0, 0, 255)

        self._text(
            frame,
            text,
            30,
            45,
            0.72,
            color,
        )

        detection = result.detection

        lanes = (
            0
            if detection is None
            else detection.num_lanes_detected
        )

        self._text(
            frame,
            f"YOLOP LANES: {lanes}",
            30,
            80,
            0.62,
            (255, 255, 255),
        )

        self._text(
            frame,
            f"PROCESSING: {result.processing_ms:.1f} ms",
            30,
            115,
            0.62,
            (255, 255, 255),
        )

    def draw_adas_panel(
        self,
        result: Optional[PipelineResult],
    ) -> np.ndarray:

        width = 900
        height = 650

        panel = np.zeros(
            (
                height,
                width,
                3,
            ),
            dtype=np.uint8,
        )

        # -----------------------------------------------------------------
        # Título
        # -----------------------------------------------------------------

        self._text(
            panel,
            "FORZA HORIZON 6",
            40,
            55,
            1.0,
            (255, 255, 255),
            2,
        )

        self._text(
            panel,
            "ADAS / LKA MONITOR",
            40,
            95,
            0.72,
            (180, 180, 180),
            2,
        )

        # -----------------------------------------------------------------
        # Sem resultado
        # -----------------------------------------------------------------

        if result is None:

            self._text(
                panel,
                "SYSTEM INITIALIZING",
                40,
                180,
                0.9,
                (0, 255, 255),
                2,
            )

            return panel

        geometry = result.geometry
        adas = result.adas

        # -----------------------------------------------------------------
        # Estado
        # -----------------------------------------------------------------

        state = "LANE_LOST"

        if adas is not None:

            try:
                state = str(
                    adas.state.value
                ).upper()
            except Exception:
                state = str(
                    adas.state
                ).upper()

        state_color = (
            (0, 255, 0)
            if state not in {
                "LANE_LOST",
                "UNKNOWN",
            }
            else (0, 0, 255)
        )

        self._text(
            panel,
            f"ADAS: {state}",
            40,
            170,
            0.95,
            state_color,
            2,
        )

        # -----------------------------------------------------------------
        # Geometria
        # -----------------------------------------------------------------

        geometry_state = (
            "REAL"
            if result.real_geometry
            else (
                "PREDICTED"
                if result.prediction_active
                else "INVALID"
            )
        )

        self._text(
            panel,
            f"GEOMETRY: {geometry_state}",
            40,
            215,
            0.70,
            (
                0,
                255,
                0,
            )
            if result.real_geometry
            else (
                0,
                255,
                255,
            )
            if result.prediction_active
            else (
                0,
                0,
                255,
            ),
        )

        # -----------------------------------------------------------------
        # Confiança
        # -----------------------------------------------------------------

        confidence = 0.0

        if adas is not None:

            confidence = safe_float(
                getattr(
                    adas,
                    "confidence",
                    0.0,
                )
            )

        self._text(
            panel,
            f"CONFIDENCE: {confidence:.3f}",
            40,
            270,
            0.70,
            (255, 255, 255),
        )

        # Barra de confiança

        bar_x = 40
        bar_y = 290
        bar_w = 500
        bar_h = 24

        cv2.rectangle(
            panel,
            (
                bar_x,
                bar_y,
            ),
            (
                bar_x + bar_w,
                bar_y + bar_h,
            ),
            (100, 100, 100),
            2,
        )

        fill = int(
            bar_w
            * clamp01(confidence)
        )

        if fill > 0:

            cv2.rectangle(
                panel,
                (
                    bar_x,
                    bar_y,
                ),
                (
                    bar_x + fill,
                    bar_y + bar_h,
                ),
                (0, 255, 0),
                -1,
            )

        # -----------------------------------------------------------------
        # Erro lateral
        # -----------------------------------------------------------------

        lateral = 0.0
        heading = 0.0

        if adas is not None:

            lateral = safe_float(
                getattr(
                    adas,
                    "lateral_error",
                    0.0,
                )
            )

            heading = safe_float(
                getattr(
                    adas,
                    "heading_error",
                    0.0,
                )
            )

        self._text(
            panel,
            f"LATERAL ERROR: {lateral:+.4f}",
            40,
            365,
            0.68,
            (255, 255, 255),
        )

        self._text(
            panel,
            f"HEADING ERROR: {heading:+.4f}",
            40,
            405,
            0.68,
            (255, 255, 255),
        )

        # -----------------------------------------------------------------
        # Atuação
        # -----------------------------------------------------------------

        if result.actuation_allowed:

            control = "CONTROL: ARMED"
            control_color = (0, 255, 0)

        else:

            control = "CONTROL: BLOCKED"
            control_color = (0, 0, 255)

        self._text(
            panel,
            control,
            40,
            475,
            0.82,
            control_color,
            2,
        )

        # -----------------------------------------------------------------
        # Segurança
        # -----------------------------------------------------------------

        if not result.real_geometry:

            reason = (
                "BLOCK: REAL 2-LANE GEOMETRY REQUIRED"
            )

        elif confidence < MIN_ADAS_CONFIDENCE:

            reason = (
                "BLOCK: LOW ADAS CONFIDENCE"
            )

        elif not CONTROL_ENABLED:

            reason = (
                "BLOCK: G29 CONTROL DISABLED"
            )

        else:

            reason = "SAFETY GATE PASSED"

        self._text(
            panel,
            reason,
            40,
            530,
            0.55,
            (220, 220, 220),
        )

        self._text(
            panel,
            "ESC = EXIT    F8 = ENABLE/DISABLE",
            40,
            600,
            0.55,
            (160, 160, 160),
        )

        return panel

    @staticmethod
    def _text(
        frame: np.ndarray,
        text: str,
        x: int,
        y: int,
        scale: float,
        color: Tuple[int, int, int],
        thickness: int = 2,
    ) -> None:

        cv2.putText(
            frame,
            str(text),
            (int(x), int(y)),
            FONT,
            float(scale),
            color,
            thickness,
            cv2.LINE_AA,
        )


# =============================================================================
# PIPELINE
# =============================================================================

class ADASPipeline:

    def __init__(self) -> None:

        model_path = getattr(
            config,
            "YOLOP_MODEL_PATH",
            None,
        )

        if model_path:

            logger.info(
                "YOLOP model configurado: %s",
                model_path,
            )

            self.detector = (
                YOLOPLaneDetector(
                    model_path=model_path,
                    input_width=YOLOP_INPUT_WIDTH,
                    input_height=YOLOP_INPUT_HEIGHT,
                    lane_threshold=float(
                        getattr(
                            config,
                            "YOLOP_LANE_THRESHOLD",
                            0.50,
                        )
                    ),
                    min_points_per_lane=int(
                        getattr(
                            config,
                            "MIN_POINTS_PER_LANE",
                            5,
                        )
                    ),
                )
            )

        else:

            logger.info(
                "YOLOP_MODEL_PATH não existe no config.py."
            )

            logger.info(
                "Utilizando caminho padrão do detector."
            )

            self.detector = (
                YOLOPLaneDetector()
            )

        # -----------------------------------------------------------------
        # Geometria
        # -----------------------------------------------------------------

        self.geometry = LaneGeometry(
            screen_width=SCREEN_WIDTH,
            screen_height=SCREEN_HEIGHT,
            roi=ROI,
            ufld_width=YOLOP_INPUT_WIDTH,
            ufld_height=YOLOP_INPUT_HEIGHT,
            near_weight=float(
                getattr(
                    config,
                    "LANE_GEOMETRY_NEAR_WEIGHT",
                    0.7,
                )
            ),
            far_weight=float(
                getattr(
                    config,
                    "LANE_GEOMETRY_FAR_WEIGHT",
                    0.3,
                )
            ),
            min_points=int(
                getattr(
                    config,
                    "MIN_POINTS_PER_LANE",
                    5,
                )
            ),
        )

        # -----------------------------------------------------------------
        # Filtro
        # -----------------------------------------------------------------

        self.temporal_filter = (
            EMATemporalFilter(
                alpha=float(
                    getattr(
                        config,
                        "FILTER_ALPHA",
                        0.3,
                    )
                ),
                min_valid_points=int(
                    getattr(
                        config,
                        "MIN_POINTS_PER_LANE",
                        5,
                    )
                ),
            )
        )

        # -----------------------------------------------------------------
        # ADAS
        # -----------------------------------------------------------------

        self.adas = (
            ADASStateEstimator(
                min_confidence=MIN_ADAS_CONFIDENCE,
            )
        )

        # -----------------------------------------------------------------
        # Predição
        # -----------------------------------------------------------------

        self.predictor = (
            SingleLanePredictor(
                ROI
            )
        )

        self.stable_valid_frames = 0

    # =========================================================================
    # DETECÇÃO
    # =========================================================================

    @staticmethod
    def _valid_lane(
        lane: Sequence[LanePoint],
    ) -> bool:

        count = 0

        for point in lane:

            if not point.valid:
                continue

            if not finite(point.x):
                continue

            if not finite(point.y):
                continue

            count += 1

        return count >= 3

    def _lane_count(
        self,
        detection: LaneDetectionResult,
    ) -> int:

        if detection is None:
            return 0

        count = 0

        if self._valid_lane(
            detection.left_lane
        ):
            count += 1

        if self._valid_lane(
            detection.right_lane
        ):
            count += 1

        return count

    # =========================================================================
    # PREDIÇÃO
    # =========================================================================

    def _predict(
        self,
        detection: LaneDetectionResult,
    ) -> List[Point]:

        if detection is None:
            return []

        left = lane_to_screen(
            detection.left_lane,
            detection.input_width,
            detection.input_height,
            ROI,
        )

        right = lane_to_screen(
            detection.right_lane,
            detection.input_width,
            detection.input_height,
            ROI,
        )

        if (
            len(left) >= MIN_PREDICTION_POINTS
            and len(right) < MIN_PREDICTION_POINTS
        ):

            return self.predictor.predict(
                left,
                True,
            )

        if (
            len(right) >= MIN_PREDICTION_POINTS
            and len(left) < MIN_PREDICTION_POINTS
        ):

            return self.predictor.predict(
                right,
                False,
            )

        return []

    # =========================================================================
    # PROCESSAMENTO
    # =========================================================================

    def process(
        self,
        frame: np.ndarray,
    ) -> PipelineResult:

        start = time.perf_counter()

        timestamp = time.perf_counter()

        try:

            detection = self.detector.detect(
                frame
            )

        except Exception as exc:

            logger.exception(
                "Erro no YOLOP."
            )

            self.reset()

            return PipelineResult(
                detection=None,
                geometry=None,
                adas=None,
                predicted_lane=[],
                real_geometry=False,
                prediction_active=False,
                actuation_allowed=False,
                timestamp=timestamp,
                processing_ms=(
                    time.perf_counter()
                    - start
                )
                * 1000.0,
            )

        lane_count = self._lane_count(
            detection
        )

        # -----------------------------------------------------------------
        # Filtro
        # -----------------------------------------------------------------

        try:

            filtered_detection = (
                self.temporal_filter
                .filter_detection(
                    detection
                )
            )

        except Exception:

            logger.exception(
                "Erro no filtro de detecção."
            )

            filtered_detection = detection

        # -----------------------------------------------------------------
        # Geometria
        # -----------------------------------------------------------------

        try:

            geometry = (
                self.geometry.compute(
                    filtered_detection
                )
            )

        except Exception:

            logger.exception(
                "Erro no cálculo da geometria."
            )

            geometry = None

        # -----------------------------------------------------------------
        # Predição
        # -----------------------------------------------------------------

        predicted_lane = []

        if lane_count == 1:

            predicted_lane = (
                self._predict(
                    filtered_detection
                )
            )

        prediction_active = bool(
            predicted_lane
        )

        # -----------------------------------------------------------------
        # Geometria REAL
        # -----------------------------------------------------------------

        real_geometry = bool(
            geometry is not None
            and geometry.valid
            and lane_count >= 2
        )

        if real_geometry:

            self.stable_valid_frames += 1

        else:

            self.stable_valid_frames = 0

        # -----------------------------------------------------------------
        # Filtro geométrico
        # -----------------------------------------------------------------

        try:

            filtered_geometry = (
                self.temporal_filter
                .filter_geometry(
                    geometry
                )
            )

        except Exception:

            logger.exception(
                "Erro no filtro geométrico."
            )

            filtered_geometry = geometry

        # -----------------------------------------------------------------
        # Segurança:
        #
        # se não houver duas lanes reais,
        # a geometria NÃO pode ser usada pelo ADAS.
        # -----------------------------------------------------------------

        geometry_for_adas = (
            filtered_geometry
            if real_geometry
            else None
        )

        # -----------------------------------------------------------------
        # ADAS
        # -----------------------------------------------------------------

        try:

            adas_result = (
                self.adas.update(
                    geometry_for_adas,
                    timestamp,
                )
            )

        except Exception:

            logger.exception(
                "Erro no ADAS."
            )

            adas_result = None

        # -----------------------------------------------------------------
        # Gate
        # -----------------------------------------------------------------

        actuation_allowed = (
            self._can_actuate(
                filtered_geometry,
                adas_result,
                real_geometry,
            )
        )

        processing_ms = (
            time.perf_counter()
            - start
        ) * 1000.0

        return PipelineResult(
            detection=filtered_detection,
            geometry=filtered_geometry,
            adas=adas_result,
            predicted_lane=predicted_lane,
            real_geometry=real_geometry,
            prediction_active=prediction_active,
            actuation_allowed=actuation_allowed,
            timestamp=timestamp,
            processing_ms=processing_ms,
        )

    # =========================================================================
    # SAFETY GATE
    # =========================================================================

    def _can_actuate(
        self,
        geometry: Optional[LaneGeometryResult],
        adas: Optional[ADASStateResult],
        real_geometry: bool,
    ) -> bool:

        if not CONTROL_ENABLED:
            return False

        if not real_geometry:
            return False

        if geometry is None:
            return False

        if not geometry.valid:
            return False

        if adas is None:
            return False

        if not getattr(
            adas,
            "valid",
            False,
        ):
            return False

        confidence = safe_float(
            getattr(
                adas,
                "confidence",
                0.0,
            )
        )

        if confidence < MIN_ADAS_CONFIDENCE:
            return False

        if (
            self.stable_valid_frames
            < MIN_STABLE_VALID_FRAMES
        ):
            return False

        state = getattr(
            adas,
            "state",
            None,
        )

        if state in {
            ADASState.UNKNOWN,
            ADASState.LANE_LOST,
        }:

            return False

        return True

    # =========================================================================
    # RESET
    # =========================================================================

    def reset(self) -> None:

        try:
            self.temporal_filter.reset()
        except Exception:
            pass

        try:
            self.adas.reset()
        except Exception:
            pass

        self.stable_valid_frames = 0


# =============================================================================
# CAPTURA
# =============================================================================

class FrameSource:

    def __init__(
        self,
        mode: str = "screen",
        video_path: Optional[str] = None,
    ) -> None:

        self.mode = mode
        self.video_path = video_path
        self.capture = None

        if mode == "screen":

            self._create_screen_capture()

        elif mode == "video":

            if not video_path:
                raise ValueError(
                    "video_path é obrigatório."
                )

            self.capture = (
                cv2.VideoCapture(
                    video_path
                )
            )

            if not self.capture.isOpened():

                raise RuntimeError(
                    f"Não foi possível abrir: "
                    f"{video_path}"
                )

        else:

            raise ValueError(
                f"Modo desconhecido: {mode}"
            )

    # =========================================================================
    # SCREEN CAPTURE
    # =========================================================================

    def _create_screen_capture(
        self,
    ) -> None:

        if ScreenCapture is None:

            raise RuntimeError(
                "capture.screen_capture.ScreenCapture "
                "não pôde ser importado."
            )

        target_fps = int(
            getattr(
                config,
                "CAPTURE_TARGET_FPS",
                60,
            )
        )

        # -----------------------------------------------------------------
        # IMPORTANTE:
        #
        # A implementação instalada de ScreenCapture não aceita
        # necessariamente width/height/fps.
        #
        # Descobrimos os argumentos aceitos dinamicamente.
        # -----------------------------------------------------------------

        try:

            signature = inspect.signature(
                ScreenCapture
            )

            parameters = signature.parameters

            kwargs = {}

            if "width" in parameters:
                kwargs["width"] = SCREEN_WIDTH

            if "height" in parameters:
                kwargs["height"] = SCREEN_HEIGHT

            if "fps" in parameters:
                kwargs["fps"] = target_fps

            if "target_fps" in parameters:
                kwargs["target_fps"] = target_fps

            if "backend" in parameters:

                kwargs["backend"] = getattr(
                    config,
                    "CAPTURE_BACKEND",
                    "bettercam",
                )

            logger.info(
                "ScreenCapture args: %s",
                kwargs,
            )

            self.capture = ScreenCapture(
                **kwargs
            )

        except Exception:

            logger.exception(
                "Falha ao criar ScreenCapture."
            )

            raise

    # =========================================================================
    # READ
    # =========================================================================

    def read(
        self,
    ) -> Optional[np.ndarray]:

        if self.capture is None:
            return None

        try:

            if self.mode == "screen":

                frame = self.capture.read()

                if frame is None:
                    return None

            else:

                ok, frame = (
                    self.capture.read()
                )

                if not ok:
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

            # -------------------------------------------------------------
            # Garante resolução do pipeline.
            # -------------------------------------------------------------

            if (
                frame.shape[1]
                != SCREEN_WIDTH
                or frame.shape[0]
                != SCREEN_HEIGHT
            ):

                frame = cv2.resize(
                    frame,
                    (
                        SCREEN_WIDTH,
                        SCREEN_HEIGHT,
                    ),
                    interpolation=cv2.INTER_LINEAR,
                )

            return frame

        except Exception:

            logger.exception(
                "Erro ao capturar frame."
            )

            return None

    # =========================================================================
    # RELEASE
    # =========================================================================

    def release(self) -> None:

        if self.capture is None:
            return

        try:

            release = getattr(
                self.capture,
                "release",
                None,
            )

            if callable(release):
                release()

        except Exception:

            logger.exception(
                "Erro ao liberar captura."
            )


# =============================================================================
# APLICAÇÃO
# =============================================================================

class Application:

    def __init__(
        self,
        source: FrameSource,
    ) -> None:

        self.source = source

        self.pipeline = (
            ADASPipeline()
        )

        self.visualizer = (
            Visualizer()
        )

        self.running = True
        self.enabled = True

        self.latest_result: Optional[
            PipelineResult
        ] = None

        self.last_frame: Optional[
            np.ndarray
        ] = None

        self.last_error: Optional[
            str
        ] = None

    # =========================================================================
    # WINDOWS
    # =========================================================================

    def create_windows(self) -> None:

        logger.info(
            "Criando janela VISION..."
        )

        cv2.namedWindow(
            WINDOW_VISION,
            cv2.WINDOW_NORMAL,
        )

        cv2.resizeWindow(
            WINDOW_VISION,
            1280,
            800,
        )

        logger.info(
            "Criando janela ADAS..."
        )

        cv2.namedWindow(
            WINDOW_ADAS,
            cv2.WINDOW_NORMAL,
        )

        cv2.resizeWindow(
            WINDOW_ADAS,
            900,
            650,
        )

        # -----------------------------------------------------------------
        # Mostra imediatamente as duas janelas.
        # -----------------------------------------------------------------

        blank = np.zeros(
            (
                650,
                900,
                3,
            ),
            dtype=np.uint8,
        )

        cv2.putText(
            blank,
            "ADAS INITIALIZING...",
            (40, 100),
            FONT,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        vision = np.zeros(
            (
                800,
                1280,
                3,
            ),
            dtype=np.uint8,
        )

        cv2.putText(
            vision,
            "VISION INITIALIZING...",
            (40, 70),
            FONT,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(
            WINDOW_VISION,
            vision,
        )

        cv2.imshow(
            WINDOW_ADAS,
            blank,
        )

        cv2.waitKey(1)

        logger.info(
            "Janelas criadas."
        )

    # =========================================================================
    # FRAME
    # =========================================================================

    def capture_frame(
        self,
    ) -> Optional[np.ndarray]:

        return self.source.read()

    # =========================================================================
    # PROCESSAMENTO
    # =========================================================================

    def process_frame(
        self,
        frame: np.ndarray,
    ) -> None:

        if not self.enabled:
            return

        try:

            self.latest_result = (
                self.pipeline.process(
                    frame
                )
            )

            self.last_error = None

        except Exception as exc:

            self.last_error = (
                f"{type(exc).__name__}: {exc}"
            )

            logger.exception(
                "Erro no processamento."
            )

            self.pipeline.reset()

    # =========================================================================
    # DRAW
    # =========================================================================

    def draw(
        self,
        frame: np.ndarray,
    ) -> None:

        vision = (
            self.visualizer.draw_vision(
                frame,
                self.latest_result,
            )
        )

        adas = (
            self.visualizer.draw_adas_panel(
                self.latest_result,
            )
        )

        # -----------------------------------------------------------------
        # Redimensiona a visão para uma janela prática.
        # -----------------------------------------------------------------

        max_width = 1600

        if vision.shape[1] > max_width:

            scale = (
                max_width
                / vision.shape[1]
            )

            vision = cv2.resize(
                vision,
                (
                    max_width,
                    int(
                        vision.shape[0]
                        * scale
                    ),
                ),
                interpolation=cv2.INTER_AREA,
            )

        cv2.imshow(
            WINDOW_VISION,
            vision,
        )

        cv2.imshow(
            WINDOW_ADAS,
            adas,
        )

    # =========================================================================
    # KEYBOARD
    # =========================================================================

    def handle_key(
        self,
        key: int,
    ) -> None:

        if key == 27:

            logger.info(
                "ESC pressionado."
            )

            self.running = False

            return

        if key == getattr(
            config,
            "HOTKEY_TOGGLE",
            0x77,
        ):

            self.enabled = (
                not self.enabled
            )

            logger.info(
                "Pipeline: %s",
                (
                    "ENABLED"
                    if self.enabled
                    else "DISABLED"
                ),
            )

            if not self.enabled:

                self.pipeline.reset()

    # =========================================================================
    # LOOP
    # =========================================================================

    def run(self) -> None:

        self.create_windows()

        target_fps = float(
            getattr(
                config,
                "VISUALIZATION_FPS",
                30,
            )
        )

        interval = (
            1.0
            / max(
                target_fps,
                1.0,
            )
        )

        logger.info(
            "Loop principal iniciado."
        )

        while self.running:

            loop_start = (
                time.perf_counter()
            )

            # -------------------------------------------------------------
            # Captura
            # -------------------------------------------------------------

            frame = (
                self.capture_frame()
            )

            if frame is None:

                blank = np.zeros(
                    (
                        800,
                        1280,
                        3,
                    ),
                    dtype=np.uint8,
                )

                cv2.putText(
                    blank,
                    "WAITING FOR FRAME...",
                    (40, 70),
                    FONT,
                    1.0,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow(
                    WINDOW_VISION,
                    blank,
                )

                adas = (
                    self.visualizer
                    .draw_adas_panel(
                        self.latest_result
                    )
                )

                cv2.imshow(
                    WINDOW_ADAS,
                    adas,
                )

                key = (
                    cv2.waitKey(1)
                    & 0xFF
                )

                self.handle_key(
                    key
                )

                time.sleep(
                    0.005
                )

                continue

            self.last_frame = frame

            # -------------------------------------------------------------
            # YOLOP
            # -------------------------------------------------------------

            self.process_frame(
                frame
            )

            # -------------------------------------------------------------
            # Overlay
            # -------------------------------------------------------------

            self.draw(
                frame
            )

            # -------------------------------------------------------------
            # Keyboard
            # -------------------------------------------------------------

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            self.handle_key(
                key
            )

            # -------------------------------------------------------------
            # Limite de FPS
            # -------------------------------------------------------------

            elapsed = (
                time.perf_counter()
                - loop_start
            )

            remaining = (
                interval
                - elapsed
            )

            if remaining > 0:

                time.sleep(
                    min(
                        remaining,
                        0.01,
                    )
                )

    # =========================================================================
    # SHUTDOWN
    # =========================================================================

    def shutdown(self) -> None:

        logger.info(
            "Encerrando aplicação..."
        )

        self.running = False

        try:
            self.pipeline.reset()
        except Exception:
            pass

        try:
            self.source.release()
        except Exception:
            pass

        try:
            cv2.destroyWindow(
                WINDOW_VISION
            )
        except Exception:
            pass

        try:
            cv2.destroyWindow(
                WINDOW_ADAS
            )
        except Exception:
            pass

        cv2.destroyAllWindows()

        logger.info(
            "Aplicação encerrada."
        )


# =============================================================================
# ARGUMENTOS
# =============================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Forza Horizon 6 "
            "ADAS/LKA"
        )
    )

    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Arquivo de vídeo.",
    )

    return parser.parse_args()


# =============================================================================
# LOGGING
# =============================================================================

def configure_logging() -> None:

    level_name = str(
        getattr(
            config,
            "LOG_LEVEL",
            "INFO",
        )
    ).upper()

    level = getattr(
        logging,
        level_name,
        logging.INFO,
    )

    logging.basicConfig(
        level=level,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
    )


# =============================================================================
# STARTUP
# =============================================================================

def print_startup() -> None:

    logger.info(
        "========================================"
    )

    logger.info(
        "Forza Horizon 6 ADAS/LKA"
    )

    logger.info(
        "Detector: YOLOP ONNX"
    )

    logger.info(
        "Tela: %sx%s",
        SCREEN_WIDTH,
        SCREEN_HEIGHT,
    )

    logger.info(
        "ROI: (%s, %s) -> (%s, %s)",
        ROI_LEFT,
        ROI_TOP,
        ROI_RIGHT,
        ROI_BOTTOM,
    )

    logger.info(
        "YOLOP input: %sx%s",
        YOLOP_INPUT_WIDTH,
        YOLOP_INPUT_HEIGHT,
    )

    logger.info(
        "Controle G29: %s",
        (
            "ENABLED"
            if CONTROL_ENABLED
            else "DISABLED"
        ),
    )

    logger.info(
        "========================================"
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    configure_logging()

    print_startup()

    args = parse_args()

    source = None
    application = None

    try:

        # -------------------------------------------------------------
        # Fonte
        # -------------------------------------------------------------

        if args.video:

            logger.info(
                "Fonte: VIDEO"
            )

            source = FrameSource(
                mode="video",
                video_path=args.video,
            )

        else:

            logger.info(
                "Fonte: SCREEN"
            )

            source = FrameSource(
                mode="screen",
            )

        # -------------------------------------------------------------
        # Aplicação
        # -------------------------------------------------------------

        application = Application(
            source
        )

        application.run()

    except KeyboardInterrupt:

        logger.info(
            "Interrompido pelo usuário."
        )

    except Exception:

        logger.exception(
            "Falha fatal da aplicação."
        )

        raise

    finally:

        if application is not None:

            application.shutdown()

        elif source is not None:

            source.release()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
