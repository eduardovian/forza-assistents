"""
vision/lane_geometry.py

Forza Assistents
================

Núcleo geométrico da percepção de faixas.

Responsabilidade
----------------
Transformar observações de lanes em uma representação geométrica
determinística, robusta e independente do detector.

Fluxo:

    LaneDetectionResult
            |
            v
      LaneGeometry
            |
            v
    LaneGeometryResult
            |
            +----> LaneModel
            |
            +----> LaneAssignment
            |
            +----> ADAS

Este módulo NÃO:

    - executa inferência;
    - conhece YOLOP internamente;
    - conhece UFLD;
    - captura tela;
    - mantém estado temporal;
    - faz tracking;
    - extrapola lanes;
    - controla o veículo;
    - decide o estado do ADAS.

Princípio fundamental
---------------------
LaneGeometry representa somente a geometria que pode ser inferida
a partir das observações disponíveis no frame atual.

Ausência de observação não deve ser transformada em uma detecção
artificial.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from config import LANE_GEOMETRY, ROI

from .detection_types import LaneDetectionResult
from .lane_types import LaneLine, LanePoint


# =============================================================================
# TIPOS
# =============================================================================

Point = Tuple[float, float]


# =============================================================================
# CONSTANTES
# =============================================================================

DEFAULT_MIN_POINTS = 4
DEFAULT_MIN_OBSERVED_SPAN = 20.0

DEFAULT_MIN_LANE_WIDTH = 40.0
DEFAULT_MAX_LANE_WIDTH = 1200.0

DEFAULT_MIN_GEOMETRY_CONFIDENCE = 0.10

DEFAULT_HEADING_LOOKAHEAD = 80.0

DEFAULT_CURVATURE_EPSILON = 1e-9

DEFAULT_POLYNOMIAL_DEGREE = 3

DEFAULT_PAIR_SAMPLE_COUNT = 9


# =============================================================================
# RESULTADO
# =============================================================================

@dataclass(frozen=True, slots=True)
class LaneGeometryResult:
    """
    Resultado geométrico de um frame.

    Todas as coordenadas são expressas no mesmo sistema de coordenadas
    recebido pelo método process().

    Sign convention
    ---------------

    lateral_error:

        < 0  -> centro da faixa à esquerda do centro da imagem
        > 0  -> centro da faixa à direita do centro da imagem

    heading_error:

        < 0  -> orientação da faixa para a esquerda
        > 0  -> orientação da faixa para a direita

    curvature:

        < 0  -> curvatura para a esquerda
        > 0  -> curvatura para a direita
    """

    lane_center_x: float
    lane_center_y: float

    image_center_x: float
    image_center_y: float

    lateral_error: float
    heading_error: float

    lane_width: float
    curvature: float

    center_line: list[Point]

    valid: bool

    left_lane_screen: list[Point]
    right_lane_screen: list[Point]

    additional_lanes_screen: list[list[Point]]

    selected_left_index: int
    selected_right_index: int

    geometry_confidence: float = 0.0

    observed_y_min: float = 0.0
    observed_y_max: float = 0.0
    observed_span: float = 0.0

    enough_for_projection: bool = False


# =============================================================================
# CLASSE PRINCIPAL
# =============================================================================

class LaneGeometry:
    """
    Calculador puramente geométrico das lanes observadas.

    Características:

        - stateless;
        - determinístico;
        - independente do detector;
        - sem estado temporal;
        - sem extrapolação;
        - seleção robusta de pares;
        - cálculo do centro;
        - cálculo da largura;
        - heading observado;
        - curvatura observada;
        - confiança geométrica;
        - tratamento explícito de falhas.
    """

    # =========================================================================
    # CONSTRUÇÃO
    # =========================================================================

    def __init__(
        self,
        screen_width: Optional[float] = None,
        screen_height: Optional[float] = None,
        roi: Optional[
            Tuple[float, float, float, float]
        ] = None,
    ) -> None:
        """
        Cria o calculador geométrico.

        Parameters
        ----------
        screen_width:
            Largura da imagem/tela.

        screen_height:
            Altura da imagem/tela.

        roi:
            ROI opcional:

                (left, top, right, bottom)
        """

        self._screen_width = self._validate_dimension(
            screen_width,
            "screen_width",
        )

        self._screen_height = self._validate_dimension(
            screen_height,
            "screen_height",
        )

        self._roi = self._normalize_roi(
            roi
        )

        self._validate_configuration()

    # =========================================================================
    # CONFIGURAÇÃO
    # =========================================================================

    @staticmethod
    def _validate_dimension(
        value: Optional[float],
        name: str,
    ) -> Optional[float]:

        if value is None:
            return None

        try:
            value = float(value)
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                f"{name} must be numeric."
            ) from exc

        if not math.isfinite(value):
            raise ValueError(
                f"{name} must be finite."
            )

        if value <= 0.0:
            raise ValueError(
                f"{name} must be > 0."
            )

        return value

    @classmethod
    def _normalize_roi(
        cls,
        roi: Optional[
            Tuple[float, float, float, float]
        ],
    ) -> Optional[
        Tuple[float, float, float, float]
    ]:

        if roi is None:
            return None

        if len(roi) != 4:
            raise ValueError(
                "ROI must contain four values."
            )

        values = tuple(
            float(value)
            for value in roi
        )

        if not all(
            math.isfinite(value)
            for value in values
        ):
            raise ValueError(
                "ROI must contain finite values."
            )

        left, top, right, bottom = values

        if right <= left:
            raise ValueError(
                "ROI right must be greater than left."
            )

        if bottom <= top:
            raise ValueError(
                "ROI bottom must be greater than top."
            )

        return (
            left,
            top,
            right,
            bottom,
        )

    @staticmethod
    def _config_value(
        name: str,
        default: float,
    ) -> float:

        return float(
            getattr(
                LANE_GEOMETRY,
                name,
                default,
            )
        )

    def _validate_configuration(self) -> None:

        min_points = int(
            self._config_value(
                "min_points",
                DEFAULT_MIN_POINTS,
            )
        )

        if min_points < 2:
            raise ValueError(
                "LANE_GEOMETRY.min_points must be >= 2."
            )

        min_span = self._config_value(
            "min_observed_span",
            DEFAULT_MIN_OBSERVED_SPAN,
        )

        if min_span <= 0.0:
            raise ValueError(
                "LANE_GEOMETRY.min_observed_span must be > 0."
            )

        min_width = self._config_value(
            "min_lane_width",
            DEFAULT_MIN_LANE_WIDTH,
        )

        max_width = self._config_value(
            "max_lane_width",
            DEFAULT_MAX_LANE_WIDTH,
        )

        if min_width <= 0.0:
            raise ValueError(
                "LANE_GEOMETRY.min_lane_width must be > 0."
            )

        if max_width <= min_width:
            raise ValueError(
                "LANE_GEOMETRY.max_lane_width must be "
                "greater than min_lane_width."
            )

    # =========================================================================
    # UTILITÁRIOS
    # =========================================================================

    @staticmethod
    def _finite(
        value: object,
    ) -> bool:

        try:
            return math.isfinite(
                float(value)
            )
        except (
            TypeError,
            ValueError,
        ):
            return False

    @staticmethod
    def _safe_float(
        value: object,
        default: float = 0.0,
    ) -> float:

        try:
            result = float(value)
        except (
            TypeError,
            ValueError,
        ):
            return default

        if not math.isfinite(result):
            return default

        return result

    @staticmethod
    def _clip(
        value: float,
        lower: float,
        upper: float,
    ) -> float:

        return min(
            upper,
            max(
                lower,
                value,
            ),
        )

    @classmethod
    def _clip01(
        cls,
        value: float,
    ) -> float:

        return cls._clip(
            value,
            0.0,
            1.0,
        )

    # =========================================================================
    # ROI
    # =========================================================================

    def roi_bounds(
        self,
    ) -> Tuple[
        float,
        float,
        float,
        float,
    ]:

        if self._roi is not None:
            return self._roi

        left = self._safe_float(
            getattr(
                ROI,
                "left",
                0.0,
            )
        )

        top = self._safe_float(
            getattr(
                ROI,
                "top",
                0.0,
            )
        )

        right = self._safe_float(
            getattr(
                ROI,
                "right",
                self._screen_width
                if self._screen_width is not None
                else 1.0,
            )
        )

        bottom = self._safe_float(
            getattr(
                ROI,
                "bottom",
                self._screen_height
                if self._screen_height is not None
                else 1.0,
            )
        )

        if right <= left:
            right = left + 1.0

        if bottom <= top:
            bottom = top + 1.0

        return (
            left,
            top,
            right,
            bottom,
        )

    # =========================================================================
    # CONVERSÃO DE PONTOS
    # =========================================================================

    @staticmethod
    def _point_xy(
        point: object,
    ) -> Optional[Point]:
        """
        Converte uma representação de ponto em (x, y).

        Suporta:

            LanePoint
            tuple/list
            objetos com x/y
        """

        if isinstance(
            point,
            LanePoint,
        ):
            x = point.x
            y = point.y

        elif (
            hasattr(point, "x")
            and hasattr(point, "y")
        ):
            x = getattr(
                point,
                "x",
            )
            y = getattr(
                point,
                "y",
            )

        elif isinstance(
            point,
            (tuple, list),
        ) and len(point) >= 2:

            x = point[0]
            y = point[1]

        else:
            return None

        try:
            x = float(x)
            y = float(y)
        except (
            TypeError,
            ValueError,
        ):
            return None

        if not (
            math.isfinite(x)
            and math.isfinite(y)
        ):
            return None

        return (
            x,
            y,
        )

    @classmethod
    def _normalize_points(
        cls,
        points: Iterable[object],
    ) -> list[Point]:

        normalized: list[Point] = []

        for point in points:

            xy = cls._point_xy(
                point
            )

            if xy is None:
                continue

            normalized.append(
                xy
            )

        normalized.sort(
            key=lambda point: point[1]
        )

        return normalized

    # =========================================================================
    # EXTRAÇÃO DE LANES
    # =========================================================================

    @staticmethod
    def _extract_lanes(
        detection_result: LaneDetectionResult,
    ) -> list[LaneLine]:

        if detection_result is None:
            return []

        lanes = getattr(
            detection_result,
            "lanes",
            None,
        )

        if lanes is None:
            return []

        return [
            lane
            for lane in lanes
            if isinstance(
                lane,
                LaneLine,
            )
        ]

    # =========================================================================
    # LANE POINTS
    # =========================================================================

    @classmethod
    def _lane_points(
        cls,
        lane: LaneLine,
    ) -> list[Point]:

        points = getattr(
            lane,
            "points",
            (),
        )

        return cls._normalize_points(
            points
        )

    # =========================================================================
    # OBSERVAÇÃO
    # =========================================================================

    @classmethod
    def _observation_span(
        cls,
        points: Sequence[Point],
    ) -> float:

        if len(points) < 2:
            return 0.0

        ys = [
            point[1]
            for point in points
        ]

        return max(ys) - min(ys)

    @classmethod
    def _observation_bounds(
        cls,
        points: Sequence[Point],
    ) -> Tuple[
        float,
        float,
    ]:

        if not points:
            return (
                0.0,
                0.0,
            )

        ys = [
            point[1]
            for point in points
        ]

        return (
            min(ys),
            max(ys),
        )

    # =========================================================================
    # INTERPOLAÇÃO
    # =========================================================================

    @staticmethod
    def _interpolate_x(
        points: Sequence[Point],
        y: float,
    ) -> Optional[float]:
        """
        Interpola X para determinado Y.

        Fora da região observada, não extrapola.
        """

        if len(points) < 2:
            return None

        if not math.isfinite(y):
            return None

        ordered = sorted(
            points,
            key=lambda point: point[1],
        )

        if (
            y < ordered[0][1]
            or y > ordered[-1][1]
        ):
            return None

        for index in range(
            len(ordered) - 1
        ):

            x1, y1 = ordered[index]
            x2, y2 = ordered[index + 1]

            if y1 == y2:
                continue

            if y1 <= y <= y2:

                ratio = (
                    y - y1
                ) / (
                    y2 - y1
                )

                return (
                    x1
                    + ratio
                    * (x2 - x1)
                )

        return float(
            ordered[-1][0]
        )

    # =========================================================================
    # LARGURA
    # =========================================================================

    @classmethod
    def _lane_width_at_y(
        cls,
        left: Sequence[Point],
        right: Sequence[Point],
        y: float,
    ) -> Optional[float]:

        left_x = cls._interpolate_x(
            left,
            y,
        )

        right_x = cls._interpolate_x(
            right,
            y,
        )

        if (
            left_x is None
            or right_x is None
        ):
            return None

        width = (
            right_x
            - left_x
        )

        if not math.isfinite(width):
            return None

        return width

    # =========================================================================
    # CENTRO
    # =========================================================================

    @classmethod
    def _center_at_y(
        cls,
        left: Sequence[Point],
        right: Sequence[Point],
        y: float,
    ) -> Optional[float]:

        width = cls._lane_width_at_y(
            left,
            right,
            y,
        )

        if width is None:
            return None

        left_x = cls._interpolate_x(
            left,
            y,
        )

        if left_x is None:
            return None

        return (
            left_x
            + width / 2.0
        )

    # =========================================================================
    # CENTRO DA FAIXA
    # =========================================================================

    @classmethod
    def _build_center_line(
        cls,
        left: Sequence[Point],
        right: Sequence[Point],
        sample_count: int = DEFAULT_PAIR_SAMPLE_COUNT,
    ) -> list[Point]:

        if len(left) < 2 or len(right) < 2:
            return []

        left_y_min, left_y_max = (
            cls._observation_bounds(left)
        )

        right_y_min, right_y_max = (
            cls._observation_bounds(right)
        )

        y_min = max(
            left_y_min,
            right_y_min,
        )

        y_max = min(
            left_y_max,
            right_y_max,
        )

        if y_max <= y_min:
            return []

        count = max(
            2,
            int(sample_count),
        )

        y_values = np.linspace(
            y_min,
            y_max,
            count,
        )

        center_line: list[Point] = []

        for y in y_values:

            center_x = cls._center_at_y(
                left,
                right,
                float(y),
            )

            if center_x is None:
                continue

            center_line.append(
                (
                    float(center_x),
                    float(y),
                )
            )

        return center_line

    # =========================================================================
    # SELEÇÃO DO PAR
    # =========================================================================

    @classmethod
    def _pair_score(
        cls,
        left: LaneLine,
        right: LaneLine,
    ) -> float:

        left_points = cls._lane_points(
            left
        )

        right_points = cls._lane_points(
            right
        )

        if (
            len(left_points)
            < DEFAULT_MIN_POINTS
            or len(right_points)
            < DEFAULT_MIN_POINTS
        ):
            return -math.inf

        left_min, left_max = (
            cls._observation_bounds(
                left_points
            )
        )

        right_min, right_max = (
            cls._observation_bounds(
                right_points
            )
        )

        overlap_min = max(
            left_min,
            right_min,
        )

        overlap_max = min(
            left_max,
            right_max,
        )

        if overlap_max <= overlap_min:
            return -math.inf

        overlap = (
            overlap_max
            - overlap_min
        )

        min_width = cls._config_value(
            "min_lane_width",
            DEFAULT_MIN_LANE_WIDTH,
        )

        max_width = cls._config_value(
            "max_lane_width",
            DEFAULT_MAX_LANE_WIDTH,
        )

        sample_y = (
            overlap_min
            + 0.75
            * (
                overlap_max
                - overlap_min
            )
        )

        width = cls._lane_width_at_y(
            left_points,
            right_points,
            sample_y,
        )

        if width is None:
            return -math.inf

        if not (
            min_width
            <= width
            <= max_width
        ):
            return -math.inf

        left_confidence = cls._safe_lane_confidence(
            left
        )

        right_confidence = cls._safe_lane_confidence(
            right
        )

        confidence_score = (
            left_confidence
            + right_confidence
        ) / 2.0

        overlap_score = cls._clip01(
            overlap
            / max(
                1.0,
                overlap_max,
            )
        )

        width_center = (
            min_width
            + max_width
        ) / 2.0

        width_range = (
            max_width
            - min_width
        )

        width_score = 1.0 - (
            abs(
                width
                - width_center
            )
            / max(
                1.0,
                width_range / 2.0,
            )
        )

        width_score = cls._clip01(
            width_score
        )

        return (
            0.50
            * confidence_score
            + 0.30
            * overlap_score
            + 0.20
            * width_score
        )

    @staticmethod
    def _safe_lane_confidence(
        lane: LaneLine,
    ) -> float:

        confidence = getattr(
            lane,
            "confidence",
            0.0,
        )

        try:
            confidence = float(
                confidence
            )
        except (
            TypeError,
            ValueError,
        ):
            return 0.0

        if not math.isfinite(
            confidence
        ):
            return 0.0

        return max(
            0.0,
            min(
                1.0,
                confidence,
            ),
        )

    @classmethod
    def _select_pair(
        cls,
        lanes: Sequence[LaneLine],
    ) -> Tuple[
        Optional[int],
        Optional[int],
    ]:

        if len(lanes) < 2:
            return (
                None,
                None,
            )

        candidates: list[
            Tuple[
                float,
                int,
                int,
            ]
        ] = []

        for left_index in range(
            len(lanes)
        ):

            for right_index in range(
                len(lanes)
            ):

                if left_index == right_index:
                    continue

                left_points = cls._lane_points(
                    lanes[left_index]
                )

                right_points = cls._lane_points(
                    lanes[right_index]
                )

                if not (
                    left_points
                    and right_points
                ):
                    continue

                left_bottom = left_points[-1][0]
                right_bottom = right_points[-1][0]

                if left_bottom >= right_bottom:
                    continue

                score = cls._pair_score(
                    lanes[left_index],
                    lanes[right_index],
                )

                if math.isfinite(
                    score
                ):
                    candidates.append(
                        (
                            score,
                            left_index,
                            right_index,
                        )
                    )

        if not candidates:
            return (
                None,
                None,
            )

        candidates.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        _, left_index, right_index = (
            candidates[0]
        )

        return (
            left_index,
            right_index,
        )

    # =========================================================================
    # HEADING
    # =========================================================================

    @classmethod
    def _heading_error(
        cls,
        center_line: Sequence[Point],
    ) -> float:

        if len(center_line) < 2:
            return 0.0

        first = center_line[0]
        last = center_line[-1]

        dx = (
            last[0]
            - first[0]
        )

        dy = (
            last[1]
            - first[1]
        )

        if abs(dy) < 1e-9:
            return 0.0

        return math.atan2(
            dx,
            dy,
        )

    # =========================================================================
    # CURVATURA
    # =========================================================================

    @classmethod
    def _fit_center_polynomial(
        cls,
        center_line: Sequence[Point],
    ) -> Optional[
        Tuple[
            float,
            float,
            float,
            float,
        ]
    ]:

        if len(center_line) < 4:
            return None

        points = np.asarray(
            center_line,
            dtype=np.float64,
        )

        x = points[:, 0]
        y = points[:, 1]

        if not (
            np.all(
                np.isfinite(x)
            )
            and np.all(
                np.isfinite(y)
            )
        ):
            return None

        if (
            np.ptp(y)
            < DEFAULT_MIN_OBSERVED_SPAN
        ):
            return None

        try:
            coefficients = np.polyfit(
                y,
                x,
                DEFAULT_POLYNOMIAL_DEGREE,
            )
        except (
            np.linalg.LinAlgError,
            ValueError,
        ):
            return None

        if len(coefficients) != 4:
            return None

        values = tuple(
            float(value)
            for value in coefficients
        )

        if not all(
            math.isfinite(value)
            for value in values
        ):
            return None

        return values  # type: ignore[return-value]

    @classmethod
    def _curvature(
        cls,
        center_line: Sequence[Point],
    ) -> float:

        coefficients = (
            cls._fit_center_polynomial(
                center_line
            )
        )

        if coefficients is None:
            return 0.0

        a, b, _, _ = coefficients

        y = float(
            center_line[
                -1
            ][1]
        )

        dx_dy = (
            3.0 * a * y * y
            + 2.0 * b * y
        )

        d2x_dy2 = (
            6.0 * a * y
            + 2.0 * b
        )

        denominator = (
            1.0
            + dx_dy * dx_dy
        ) ** 1.5

        if denominator < (
            DEFAULT_CURVATURE_EPSILON
        ):
            return 0.0

        curvature = (
            d2x_dy2
            / denominator
        )

        if not math.isfinite(
            curvature
        ):
            return 0.0

        return float(
            curvature
        )

    # =========================================================================
    # CONFIANÇA
    # =========================================================================

    @classmethod
    def _geometry_confidence(
        cls,
        left: LaneLine,
        right: LaneLine,
        center_line: Sequence[Point],
        lane_width: float,
        observed_span: float,
    ) -> float:

        left_confidence = cls._safe_lane_confidence(
            left
        )

        right_confidence = cls._safe_lane_confidence(
            right
        )

        detection_confidence = (
            left_confidence
            + right_confidence
        ) / 2.0

        min_span = cls._config_value(
            "min_observed_span",
            DEFAULT_MIN_OBSERVED_SPAN,
        )

        span_score = cls._clip01(
            observed_span
            / max(
                min_span * 4.0,
                1.0,
            )
        )

        min_width = cls._config_value(
            "min_lane_width",
            DEFAULT_MIN_LANE_WIDTH,
        )

        max_width = cls._config_value(
            "max_lane_width",
            DEFAULT_MAX_LANE_WIDTH,
        )

        width_center = (
            min_width
            + max_width
        ) / 2.0

        width_score = 1.0 - (
            abs(
                lane_width
                - width_center
            )
            / max(
                width_center,
                1.0,
            )
        )

        width_score = cls._clip01(
            width_score
        )

        point_score = cls._clip01(
            len(center_line)
            / float(
                DEFAULT_PAIR_SAMPLE_COUNT
            )
        )

        confidence = (
            0.50
            * detection_confidence
            + 0.20
            * span_score
            + 0.20
            * width_score
            + 0.10
            * point_score
        )

        return cls._clip01(
            confidence
        )

    # =========================================================================
    # FALHA
    # =========================================================================

    @staticmethod
    def _invalid_result(
        *,
        image_center_x: float,
        image_center_y: float,
    ) -> LaneGeometryResult:

        return LaneGeometryResult(
            lane_center_x=image_center_x,
            lane_center_y=image_center_y,
            image_center_x=image_center_x,
            image_center_y=image_center_y,
            lateral_error=0.0,
            heading_error=0.0,
            lane_width=0.0,
            curvature=0.0,
            center_line=[],
            valid=False,
            left_lane_screen=[],
            right_lane_screen=[],
            additional_lanes_screen=[],
            selected_left_index=-1,
            selected_right_index=-1,
            geometry_confidence=0.0,
            observed_y_min=0.0,
            observed_y_max=0.0,
            observed_span=0.0,
            enough_for_projection=False,
        )

    # =========================================================================
    # PROCESSAMENTO
    # =========================================================================

    def process(
        self,
        detection_result: LaneDetectionResult,
        *,
        image_width: Optional[float] = None,
        image_height: Optional[float] = None,
    ) -> LaneGeometryResult:
        """
        Processa as lanes observadas em um frame.

        Não modifica detection_result.
        """

        width = (
            self._screen_width
            if image_width is None
            else self._validate_dimension(
                image_width,
                "image_width",
            )
        )

        height = (
            self._screen_height
            if image_height is None
            else self._validate_dimension(
                image_height,
                "image_height",
            )
        )

        if width is None:
            roi_left, _, roi_right, _ = (
                self.roi_bounds()
            )

            width = (
                roi_right
                - roi_left
            )

        if height is None:
            _, roi_top, _, roi_bottom = (
                self.roi_bounds()
            )

            height = (
                roi_bottom
                - roi_top
            )

        image_center_x = (
            float(width)
            / 2.0
        )

        image_center_y = (
            float(height)
            * 0.75
        )

        lanes = self._extract_lanes(
            detection_result
        )

        if len(lanes) < 2:

            return self._invalid_result(
                image_center_x=image_center_x,
                image_center_y=image_center_y,
            )

        left_index, right_index = (
            self._select_pair(
                lanes
            )
        )

        if (
            left_index is None
            or right_index is None
        ):

            return self._invalid_result(
                image_center_x=image_center_x,
                image_center_y=image_center_y,
            )

        left_lane = lanes[
            left_index
        ]

        right_lane = lanes[
            right_index
        ]

        left_points = self._lane_points(
            left_lane
        )

        right_points = self._lane_points(
            right_lane
        )

        center_line = (
            self._build_center_line(
                left_points,
                right_points,
            )
        )

        if len(center_line) < 2:

            return self._invalid_result(
                image_center_x=image_center_x,
                image_center_y=image_center_y,
            )

        center_y = center_line[-1][1]

        center_x = self._center_at_y(
            left_points,
            right_points,
            center_y,
        )

        if center_x is None:

            return self._invalid_result(
                image_center_x=image_center_x,
                image_center_y=image_center_y,
            )

        lane_width = self._lane_width_at_y(
            left_points,
            right_points,
            center_y,
        )

        if (
            lane_width is None
            or lane_width <= 0.0
        ):

            return self._invalid_result(
                image_center_x=image_center_x,
                image_center_y=image_center_y,
            )

        min_width = self._config_value(
            "min_lane_width",
            DEFAULT_MIN_LANE_WIDTH,
        )

        max_width = self._config_value(
            "max_lane_width",
            DEFAULT_MAX_LANE_WIDTH,
        )

        if not (
            min_width
            <= lane_width
            <= max_width
        ):

            return self._invalid_result(
                image_center_x=image_center_x,
                image_center_y=image_center_y,
            )

        observed_y_min = min(
            left_points[0][1],
            right_points[0][1],
        )

        observed_y_max = max(
            left_points[-1][1],
            right_points[-1][1],
        )

        observed_span = (
            observed_y_max
            - observed_y_min
        )

        min_span = self._config_value(
            "min_observed_span",
            DEFAULT_MIN_OBSERVED_SPAN,
        )

        heading_error = (
            self._heading_error(
                center_line
            )
        )

        curvature = (
            self._curvature(
                center_line
            )
        )

        geometry_confidence = (
            self._geometry_confidence(
                left_lane,
                right_lane,
                center_line,
                lane_width,
                observed_span,
            )
        )

        valid = (
            geometry_confidence
            >= DEFAULT_MIN_GEOMETRY_CONFIDENCE
            and observed_span
            >= min_span
        )

        additional_lanes: list[
            list[Point]
        ] = []

        for index, lane in enumerate(
            lanes
        ):

            if index in (
                left_index,
                right_index,
            ):
                continue

            points = self._lane_points(
                lane
            )

            if points:
                additional_lanes.append(
                    points
                )

        enough_for_projection = (
            valid
            and observed_span
            >= (
                min_span * 2.0
            )
            and len(center_line)
            >= DEFAULT_MIN_POINTS
        )

        lateral_error = (
            center_x
            - image_center_x
        )

        return LaneGeometryResult(
            lane_center_x=float(
                center_x
            ),
            lane_center_y=float(
                center_y
            ),
            image_center_x=float(
                image_center_x
            ),
            image_center_y=float(
                image_center_y
            ),
            lateral_error=float(
                lateral_error
            ),
            heading_error=float(
                heading_error
            ),
            lane_width=float(
                lane_width
            ),
            curvature=float(
                curvature
            ),
            center_line=[
                (
                    float(x),
                    float(y),
                )
                for x, y in center_line
            ],
            valid=bool(
                valid
            ),
            left_lane_screen=[
                (
                    float(x),
                    float(y),
                )
                for x, y in left_points
            ],
            right_lane_screen=[
                (
                    float(x),
                    float(y),
                )
                for x, y in right_points
            ],
            additional_lanes_screen=additional_lanes,
            selected_left_index=int(
                left_index
            ),
            selected_right_index=int(
                right_index
            ),
            geometry_confidence=float(
                geometry_confidence
            ),
            observed_y_min=float(
                observed_y_min
            ),
            observed_y_max=float(
                observed_y_max
            ),
            observed_span=float(
                observed_span
            ),
            enough_for_projection=bool(
                enough_for_projection
            ),
        )

    # =========================================================================
    # ALIASES DE COMPATIBILIDADE CONTROLADA
    # =========================================================================

    def compute(
        self,
        detection_result: LaneDetectionResult,
        *,
        image_width: Optional[float] = None,
        image_height: Optional[float] = None,
    ) -> LaneGeometryResult:
        """
        Alias semântico de process().
        """

        return self.process(
            detection_result,
            image_width=image_width,
            image_height=image_height,
        )

    def calculate(
        self,
        detection_result: LaneDetectionResult,
        *,
        image_width: Optional[float] = None,
        image_height: Optional[float] = None,
    ) -> LaneGeometryResult:
        """
        Alias semântico de process().
        """

        return self.process(
            detection_result,
            image_width=image_width,
            image_height=image_height,
        )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "Point",
    "LaneGeometryResult",
    "LaneGeometry",
]