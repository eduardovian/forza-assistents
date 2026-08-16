"""
vision/lane_geometry.py

Forza Assistents
================

Geometria observada das faixas.

Pipeline:

    YOLOP / LaneTracker
            ↓
    validação das observações
            ↓
    limpeza geométrica
            ↓
    ordenação espacial
            ↓
    avaliação de pares candidatos
            ↓
    seleção da faixa observada
            ↓
    centro da faixa
            ↓
    largura
            ↓
    erro lateral
            ↓
    heading
            ↓
    curvatura observada
            ↓
    confiança geométrica
            ↓
    LaneGeometryResult

RESPONSABILIDADE
----------------
Este módulo trabalha SOMENTE com geometria observada.

Não executa:

    - inferência;
    - captura;
    - definição de ROI;
    - tracking temporal;
    - extrapolação;
    - previsão;
    - associação temporal;
    - decisão ADAS;
    - controle do veículo.

A projeção/extrapolação pertence ao lane_projection.py.
O tracking pertence ao lane_tracker.py.
A associação semântica pertence ao lane_assignment.py.

CONFIGURAÇÃO
------------
Toda configuração vem exclusivamente de config.py:

    ROI
    YOLOP
    LANE_GEOMETRY

Nenhum ROI ou parâmetro equivalente é redefinido neste módulo.

COORDENADAS
-----------
O detector trabalha no frame recortado pelo ROI.

Portanto:

    detector coordinates
            ↓
        ROI coordinates
            ↓
        screen coordinates

A transformação utiliza exclusivamente o ROI calibrado.

SEGURANÇA
---------
Geometria inválida retorna:

    valid=False

e métricas numéricas seguras.

Nunca são produzidos NaN ou infinito como saída válida.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from config import LANE_GEOMETRY, ROI, YOLOP

from .lane_types import LanePoint

try:
    from .yolop_detector import LaneDetectionResult
except ImportError:  # pragma: no cover
    LaneDetectionResult = object


# =============================================================================
# TIPOS
# =============================================================================

Point = Tuple[float, float]


# =============================================================================
# RESULTADO
# =============================================================================


@dataclass(frozen=True, slots=True)
class LaneGeometryResult:
    """
    Resultado da geometria observada no frame atual.

    Todas as coordenadas são observadas.
    Nenhuma informação é extrapolada neste módulo.
    """

    lane_center_x: float
    lane_center_y: float

    image_center_x: float
    image_center_y: float

    lateral_error: float
    heading_error: float

    lane_width: float
    curvature: float

    center_line: List[Point]

    valid: bool

    left_lane_screen: List[Point]
    right_lane_screen: List[Point]

    additional_lanes_screen: List[List[Point]]

    selected_left_index: int
    selected_right_index: int

    geometry_confidence: float = 0.0

    observed_y_min: float = 0.0
    observed_y_max: float = 0.0
    observed_span: float = 0.0

    enough_for_projection: bool = False


# =============================================================================
# LANE GEOMETRY
# =============================================================================


class LaneGeometry:
    """
    Processador stateless da geometria observada.

    Princípios:

        - configuração centralizada;
        - nenhuma memória temporal;
        - nenhuma extrapolação;
        - múltiplas lanes;
        - seleção por pares;
        - validação geométrica;
        - robustez contra outliers;
        - confiança composta;
        - falha segura.
    """

    def __init__(self) -> None:
        self._validate_configuration()

    # =========================================================================
    # CONFIGURAÇÃO
    # =========================================================================

    @staticmethod
    def _validate_configuration() -> None:
        """
        Valida os contratos necessários para a geometria.
        """

        if not ROI.enabled:
            raise RuntimeError(
                "ROI calibrado não está habilitado. "
                "Execute a calibração antes de iniciar a geometria."
            )

        ROI.validate()

        if YOLOP.input_width <= 0:
            raise ValueError(
                "YOLOP.input_width deve ser > 0."
            )

        if YOLOP.input_height <= 0:
            raise ValueError(
                "YOLOP.input_height deve ser > 0."
            )

        if LANE_GEOMETRY.min_points < 2:
            raise ValueError(
                "LANE_GEOMETRY.min_points deve ser >= 2."
            )

        if LANE_GEOMETRY.min_observed_span <= 0:
            raise ValueError(
                "LANE_GEOMETRY.min_observed_span deve ser > 0."
            )

        if (
            LANE_GEOMETRY.min_lane_width
            <= 0.0
        ):
            raise ValueError(
                "LANE_GEOMETRY.min_lane_width deve ser > 0."
            )

        if (
            LANE_GEOMETRY.max_lane_width
            <= LANE_GEOMETRY.min_lane_width
        ):
            raise ValueError(
                "LANE_GEOMETRY.max_lane_width deve ser "
                "maior que min_lane_width."
            )

        if (
            LANE_GEOMETRY.polynomial_degree
            != 3
        ):
            raise ValueError(
                "LaneGeometry exige polynomial_degree=3."
            )

    # =========================================================================
    # NUMERIC
    # =========================================================================

    @staticmethod
    def _finite(value: object) -> bool:
        try:
            return math.isfinite(float(value))
        except (
            TypeError,
            ValueError,
        ):
            return False

    @staticmethod
    def _clip01(value: float) -> float:
        if not math.isfinite(float(value)):
            return 0.0

        return float(
            np.clip(
                value,
                0.0,
                1.0,
            )
        )

    # =========================================================================
    # COORDINATE TRANSFORM
    # =========================================================================

    @staticmethod
    def _detector_to_screen(
        x: float,
        y: float,
        detector_width: float,
        detector_height: float,
    ) -> Point:
        """
        Converte coordenadas do detector para coordenadas da tela.

        O ROI vem exclusivamente de config.py.
        """

        if detector_width <= 0.0:
            raise ValueError(
                "detector_width inválido."
            )

        if detector_height <= 0.0:
            raise ValueError(
                "detector_height inválido."
            )

        screen_x = (
            ROI.left
            + (
                float(x)
                / detector_width
            )
            * ROI.width
        )

        screen_y = (
            ROI.top
            + (
                float(y)
                / detector_height
            )
            * ROI.height
        )

        return (
            float(screen_x),
            float(screen_y),
        )

    @staticmethod
    def _detection_dimensions(
        detection: object,
    ) -> Tuple[float, float]:
        """
        Obtém as dimensões reais utilizadas pelo detector.

        YOLOP permanece como fallback oficial.
        """

        width = getattr(
            detection,
            "input_width",
            YOLOP.input_width,
        )

        height = getattr(
            detection,
            "input_height",
            YOLOP.input_height,
        )

        try:
            width = float(width)
        except (
            TypeError,
            ValueError,
        ):
            width = float(
                YOLOP.input_width
            )

        try:
            height = float(height)
        except (
            TypeError,
            ValueError,
        ):
            height = float(
                YOLOP.input_height
            )

        if (
            not math.isfinite(width)
            or width <= 0.0
        ):
            width = float(
                YOLOP.input_width
            )

        if (
            not math.isfinite(height)
            or height <= 0.0
        ):
            height = float(
                YOLOP.input_height
            )

        return (
            width,
            height,
        )

    # =========================================================================
    # LANE CONVERSION
    # =========================================================================

    def _convert_lane(
        self,
        lane: Sequence[LanePoint],
        detector_width: float,
        detector_height: float,
    ) -> List[Point]:
        """
        Valida e converte uma lane para coordenadas de tela.
        """

        if not lane:
            return []

        points: List[Point] = []

        for point in lane:

            if not isinstance(
                point,
                LanePoint,
            ):
                continue

            if not point.is_valid():
                continue

            if (
                point.confidence
                < LANE_GEOMETRY.min_lane_confidence
            ):
                continue

            x = float(point.x)
            y = float(point.y)

            if (
                x < 0.0
                or x > detector_width
                or y < 0.0
                or y > detector_height
            ):
                continue

            sx, sy = self._detector_to_screen(
                x=x,
                y=y,
                detector_width=detector_width,
                detector_height=detector_height,
            )

            if not (
                self._finite(sx)
                and self._finite(sy)
            ):
                continue

            points.append(
                (
                    sx,
                    sy,
                )
            )

        points.sort(
            key=lambda point: point[1]
        )

        return self._remove_duplicate_points(
            points
        )

    @staticmethod
    def _remove_duplicate_points(
        points: Sequence[Point],
    ) -> List[Point]:

        if not points:
            return []

        result: List[Point] = [
            points[0]
        ]

        for point in points[1:]:

            previous = result[-1]

            if abs(
                point[1]
                - previous[1]
            ) < 1e-6:
                continue

            result.append(point)

        return result

    # =========================================================================
    # OUTLIER REJECTION
    # =========================================================================

    def _remove_outliers(
        self,
        points: Sequence[Point],
    ) -> List[Point]:
        """
        Rejeição robusta baseada em residual MAD.

        A geometria continua observacional:
        nenhum ponto novo é criado.
        """

        if (
            not LANE_GEOMETRY.enable_outlier_rejection
            or len(points) < 5
        ):
            return list(points)

        array = np.asarray(
            points,
            dtype=np.float64,
        )

        if (
            array.ndim != 2
            or array.shape[1] != 2
            or not np.all(
                np.isfinite(array)
            )
        ):
            return list(points)

        x = array[:, 0]
        y = array[:, 1]

        y_center = float(
            np.mean(y)
        )

        y_scale = float(
            np.std(y)
        )

        if y_scale < 1e-6:
            return list(points)

        normalized_y = (
            y - y_center
        ) / y_scale

        try:
            coefficients = np.polyfit(
                normalized_y,
                x,
                2,
            )

            predicted = np.polyval(
                coefficients,
                normalized_y,
            )

        except (
            np.linalg.LinAlgError,
            ValueError,
            FloatingPointError,
        ):
            return list(points)

        residuals = np.abs(
            x - predicted
        )

        median = float(
            np.median(residuals)
        )

        mad = float(
            np.median(
                np.abs(
                    residuals - median
                )
            )
        )

        if mad < 1e-6:
            limit = max(
                5.0,
                median * 2.0,
            )
        else:
            robust_sigma = (
                1.4826 * mad
            )

            limit = (
                median
                + LANE_GEOMETRY.outlier_sigma
                * robust_sigma
            )

        filtered = [
            point
            for point, residual
            in zip(points, residuals)
            if residual <= limit
        ]

        if len(filtered) < (
            LANE_GEOMETRY.min_points
        ):
            return list(points)

        return filtered

    # =========================================================================
    # GEOMETRIC METRICS
    # =========================================================================

    @staticmethod
    def _lane_span(
        lane: Sequence[Point],
    ) -> float:

        if len(lane) < 2:
            return 0.0

        ys = [
            point[1]
            for point in lane
        ]

        return float(
            max(ys) - min(ys)
        )

    @staticmethod
    def _lane_mean_x(
        lane: Sequence[Point],
    ) -> float:

        if not lane:
            return 0.0

        return float(
            np.mean(
                [
                    point[0]
                    for point in lane
                ]
            )
        )

    @staticmethod
    def _interpolate_x(
        lane: Sequence[Point],
        y: float,
    ) -> Optional[float]:
        """
        Interpolação linear exclusivamente dentro
        da região observada.

        Não extrapola.
        """

        if len(lane) < 2:
            return None

        ordered = sorted(
            lane,
            key=lambda point: point[1],
        )

        ys = np.asarray(
            [point[1] for point in ordered],
            dtype=np.float64,
        )

        xs = np.asarray(
            [point[0] for point in ordered],
            dtype=np.float64,
        )

        if (
            y < ys[0]
            or y > ys[-1]
        ):
            return None

        return float(
            np.interp(
                y,
                ys,
                xs,
            )
        )

    def _common_observed_range(
        self,
        left: Sequence[Point],
        right: Sequence[Point],
    ) -> Optional[Tuple[float, float]]:

        if (
            len(left) < 2
            or len(right) < 2
        ):
            return None

        left_min = min(
            point[1]
            for point in left
        )

        left_max = max(
            point[1]
            for point in left
        )

        right_min = min(
            point[1]
            for point in right
        )

        right_max = max(
            point[1]
            for point in right
        )

        lower = max(
            left_min,
            right_min,
        )

        upper = min(
            left_max,
            right_max,
        )

        if upper <= lower:
            return None

        return (
            lower,
            upper,
        )

    def _sample_pair_geometry(
        self,
        left: Sequence[Point],
        right: Sequence[Point],
    ) -> Optional[
        Tuple[
            float,
            float,
            float,
            float,
            float,
        ]
    ]:
        """
        Avalia a geometria no intervalo realmente observado.

        Retorna:

            lane_center_x
            lane_center_y
            lane_width
            heading
            curvature
        """

        common = self._common_observed_range(
            left,
            right,
        )

        if common is None:
            return None

        y_min, y_max = common

        span = y_max - y_min

        if span < (
            LANE_GEOMETRY.min_observed_span
        ):
            return None

        sample_count = max(
            8,
            min(
                64,
                int(
                    LANE_GEOMETRY.min_points
                    * 4
                ),
            ),
        )

        ys = np.linspace(
            y_min,
            y_max,
            sample_count,
        )

        centers: List[float] = []
        widths: List[float] = []

        for y in ys:

            left_x = self._interpolate_x(
                left,
                float(y),
            )

            right_x = self._interpolate_x(
                right,
                float(y),
            )

            if (
                left_x is None
                or right_x is None
            ):
                continue

            width = (
                right_x - left_x
            )

            if width <= 0.0:
                continue

            if (
                width
                < LANE_GEOMETRY.min_lane_width
            ):
                continue

            if (
                width
                > LANE_GEOMETRY.max_lane_width
            ):
                continue

            centers.append(
                (
                    left_x
                    + right_x
                ) / 2.0
            )

            widths.append(width)

        if len(centers) < 3:
            return None

        center_array = np.asarray(
            centers,
            dtype=np.float64,
        )

        width_array = np.asarray(
            widths,
            dtype=np.float64,
        )

        # Região inferior possui maior peso.
        weights = np.linspace(
            LANE_GEOMETRY.far_weight
            if hasattr(
                LANE_GEOMETRY,
                "far_weight",
            )
            else 0.25,
            LANE_GEOMETRY.near_weight
            if hasattr(
                LANE_GEOMETRY,
                "near_weight",
            )
            else 0.75,
            len(center_array),
        )

        weights = np.asarray(
            weights,
            dtype=np.float64,
        )

        weights /= np.sum(weights)

        center_x = float(
            np.sum(
                center_array
                * weights
            )
        )

        lane_width = float(
            np.sum(
                width_array
                * weights
            )
        )

        # ---------------------------------------------------------------------
        # Heading
        # ---------------------------------------------------------------------

        try:
            center_coefficients = np.polyfit(
                ys[
                    :len(center_array)
                ],
                center_array,
                2,
            )

            derivative_coefficients = (
                np.polyder(
                    center_coefficients
                )
            )

            heading_slope = float(
                np.polyval(
                    derivative_coefficients,
                    y_max,
                )
            )

            heading = float(
                np.arctan(
                    heading_slope
                )
            )

        except (
            np.linalg.LinAlgError,
            ValueError,
            FloatingPointError,
        ):
            heading = 0.0

        # ---------------------------------------------------------------------
        # Curvatura
        # ---------------------------------------------------------------------

        curvature = 0.0

        try:
            if len(center_array) >= 5:

                polynomial = np.polyfit(
                    ys[
                        :len(center_array)
                    ],
                    center_array,
                    3,
                )

                first = np.polyval(
                    np.polyder(
                        polynomial,
                        1,
                    ),
                    y_max,
                )

                second = np.polyval(
                    np.polyder(
                        polynomial,
                        2,
                    ),
                    y_max,
                )

                denominator = (
                    1.0
                    + first * first
                ) ** 1.5

                if denominator > 1e-9:
                    curvature = float(
                        second
                        / denominator
                    )

        except (
            np.linalg.LinAlgError,
            ValueError,
            FloatingPointError,
        ):
            curvature = 0.0

        return (
            center_x,
            float(
                (y_min + y_max)
                / 2.0
            ),
            lane_width,
            heading,
            curvature,
        )

    # =========================================================================
    # PAIR SCORING
    # =========================================================================

    def _pair_score(
        self,
        left: Sequence[Point],
        right: Sequence[Point],
        left_confidence: float,
        right_confidence: float,
    ) -> Optional[
        Tuple[
            float,
            float,
            float,
            float,
            float,
            float,
        ]
    ]:
        """
        Avalia um par candidato.

        Retorna:

            score
            center_x
            center_y
            width
            heading
            curvature
        """

        metrics = self._sample_pair_geometry(
            left,
            right,
        )

        if metrics is None:
            return None

        (
            center_x,
            center_y,
            width,
            heading,
            curvature,
        ) = metrics

        span = min(
            self._lane_span(left),
            self._lane_span(right),
        )

        detection_confidence = (
            0.5 * self._clip01(
                left_confidence
            )
            + 0.5 * self._clip01(
                right_confidence
            )
        )

        span_score = self._clip01(
            span / 300.0
        )

        expected_width = (
            LANE_GEOMETRY.expected_lane_width
        )

        tolerance = max(
            1.0,
            LANE_GEOMETRY.lane_width_tolerance,
        )

        width_error = abs(
            width - expected_width
        ) / max(
            expected_width,
            1.0,
        )

        width_score = self._clip01(
            1.0
            - (
                width_error
                / tolerance
            )
        )

        heading_score = self._clip01(
            1.0
            - (
                abs(heading)
                / max(
                    LANE_GEOMETRY.max_heading_error,
                    1e-6,
                )
            )
        )

        curvature_score = self._clip01(
            1.0
            - (
                abs(curvature)
                / max(
                    LANE_GEOMETRY.max_curvature_score,
                    1e-6,
                )
            )
        )

        geometry_score = (
            0.60 * heading_score
            + 0.40 * curvature_score
        )

        score = _weighted_confidence(
            detection_confidence,
            span_score,
            width_score,
            geometry_score,
        )

        return (
            score,
            center_x,
            center_y,
            width,
            heading,
            curvature,
        )

    # =========================================================================
    # PAIR SELECTION
    # =========================================================================

    def _select_pair(
        self,
        lanes: Sequence[Sequence[Point]],
        confidences: Sequence[float],
    ) -> Optional[
        Tuple[
            int,
            int,
            float,
            float,
            float,
            float,
            float,
        ]
    ]:
        """
        Seleciona o melhor par esquerda/direita.

        O critério considera:

            - confiança;
            - extensão vertical;
            - largura;
            - heading;
            - curvatura;
            - proximidade do centro da imagem.
        """

        if len(lanes) < 2:
            return None

        image_center_x = (
            ROI.left
            + ROI.width / 2.0
        )

        candidates = []

        for left_index in range(
            len(lanes)
        ):

            left = lanes[
                left_index
            ]

            left_mean_x = (
                self._lane_mean_x(left)
            )

            if left_mean_x >= image_center_x:
                continue

            for right_index in range(
                len(lanes)
            ):

                if (
                    left_index
                    == right_index
                ):
                    continue

                right = lanes[
                    right_index
                ]

                right_mean_x = (
                    self._lane_mean_x(right)
                )

                if (
                    right_mean_x
                    <= image_center_x
                ):
                    continue

                pair = self._pair_score(
                    left,
                    right,
                    confidences[
                        left_index
                    ],
                    confidences[
                        right_index
                    ],
                )

                if pair is None:
                    continue

                (
                    score,
                    center_x,
                    center_y,
                    width,
                    heading,
                    curvature,
                ) = pair

                center_distance = (
                    abs(
                        center_x
                        - image_center_x
                    )
                    / max(
                        ROI.width,
                        1,
                    )
                )

                center_score = self._clip01(
                    1.0
                    - center_distance
                )

                final_score = (
                    0.85 * score
                    + 0.15 * center_score
                )

                candidates.append(
                    (
                        final_score,
                        left_index,
                        right_index,
                        center_x,
                        center_y,
                        width,
                        heading,
                        curvature,
                    )
                )

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        (
            score,
            left_index,
            right_index,
            center_x,
            center_y,
            width,
            heading,
            curvature,
        ) = candidates[0]

        return (
            left_index,
            right_index,
            score,
            center_x,
            center_y,
            width,
            heading,
        )

    # =========================================================================
    # CENTER LINE
    # =========================================================================

    def _build_center_line(
        self,
        left: Sequence[Point],
        right: Sequence[Point],
    ) -> List[Point]:
        """
        Cria centro somente onde ambas as lanes
        possuem observação.

        Não extrapola.
        """

        common = self._common_observed_range(
            left,
            right,
        )

        if common is None:
            return []

        y_min, y_max = common

        sample_count = max(
            8,
            min(
                64,
                int(
                    LANE_GEOMETRY.min_points
                    * 4
                ),
            ),
        )

        ys = np.linspace(
            y_min,
            y_max,
            sample_count,
        )

        result: List[Point] = []

        for y in ys:

            left_x = self._interpolate_x(
                left,
                float(y),
            )

            right_x = self._interpolate_x(
                right,
                float(y),
            )

            if (
                left_x is None
                or right_x is None
            ):
                continue

            center_x = (
                left_x
                + right_x
            ) / 2.0

            if not self._finite(
                center_x
            ):
                continue

            result.append(
                (
                    float(center_x),
                    float(y),
                )
            )

        return result

    # =========================================================================
    # ADDITIONAL LANES
    # =========================================================================

    @staticmethod
    def _additional_lanes(
        lanes: Sequence[Sequence[Point]],
        left_index: int,
        right_index: int,
    ) -> List[List[Point]]:

        return [
            list(lane)
            for index, lane in enumerate(lanes)
            if index not in {
                left_index,
                right_index,
            }
        ]

    # =========================================================================
    # CONFIDENCE
    # =========================================================================

    def _geometry_confidence(
        self,
        left_confidence: float,
        right_confidence: float,
        span: float,
        width: float,
        heading: float,
        curvature: float,
    ) -> float:

        detection_score = (
            0.5 * self._clip01(
                left_confidence
            )
            + 0.5 * self._clip01(
                right_confidence
            )
        )

        span_score = self._clip01(
            span / 300.0
        )

        expected_width = (
            LANE_GEOMETRY.expected_lane_width
        )

        width_error = abs(
            width - expected_width
        ) / max(
            expected_width,
            1.0,
        )

        width_score = self._clip01(
            1.0
            - width_error
        )

        heading_score = self._clip01(
            1.0
            - (
                abs(heading)
                / max(
                    LANE_GEOMETRY.max_heading_error,
                    1e-6,
                )
            )
        )

        curvature_score = self._clip01(
            1.0
            - (
                abs(curvature)
                / max(
                    LANE_GEOMETRY.max_curvature_score,
                    1e-6,
                )
            )
        )

        geometry_score = (
            0.65 * heading_score
            + 0.35 * curvature_score
        )

        return _weighted_confidence(
            detection_score,
            span_score,
            width_score,
            geometry_score,
        )

    # =========================================================================
    # INVALID RESULT
    # =========================================================================

    def _invalid_result(self) -> LaneGeometryResult:
        """
        Falha segura.
        """

        center_x = (
            ROI.left
            + ROI.width / 2.0
        )

        center_y = (
            ROI.top
            + ROI.height / 2.0
        )

        return LaneGeometryResult(
            lane_center_x=center_x,
            lane_center_y=center_y,
            image_center_x=center_x,
            image_center_y=center_y,
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
    # PUBLIC API
    # =========================================================================

    def compute(
        self,
        detection: LaneDetectionResult,
    ) -> LaneGeometryResult:
        """
        Calcula a geometria observada do frame.

        Não possui memória temporal.
        """

        if detection is None:
            return self._invalid_result()

        if not getattr(
            detection,
            "valid",
            False,
        ):
            return self._invalid_result()

        detector_width, detector_height = (
            self._detection_dimensions(
                detection
            )
        )

        raw_lanes = getattr(
            detection,
            "lanes",
            None,
        )

        confidences = getattr(
            detection,
            "lane_confidences",
            None,
        )

        # ---------------------------------------------------------------------
        # Compatibilidade com o formato antigo do detector.
        # ---------------------------------------------------------------------

        if not raw_lanes:

            raw_lanes = []

            left_lane = getattr(
                detection,
                "left_lane",
                [],
            )

            right_lane = getattr(
                detection,
                "right_lane",
                [],
            )

            if left_lane:
                raw_lanes.append(
                    left_lane
                )

            if right_lane:
                raw_lanes.append(
                    right_lane
                )

            additional = getattr(
                detection,
                "additional_lanes",
                [],
            )

            raw_lanes.extend(
                additional
            )

        if not raw_lanes:
            return self._invalid_result()

        if not confidences:

            confidences = [
                1.0
                for _ in raw_lanes
            ]

        confidences = list(
            confidences
        )

        while len(confidences) < len(
            raw_lanes
        ):
            confidences.append(
                LANE_GEOMETRY.min_lane_confidence
            )

        lanes: List[List[Point]] = []

        lane_confidences: List[float] = []

        original_indices: List[int] = []

        for index, raw_lane in enumerate(
            raw_lanes
        ):

            converted = self._convert_lane(
                raw_lane,
                detector_width,
                detector_height,
            )

            converted = self._remove_outliers(
                converted
            )

            if len(converted) < (
                LANE_GEOMETRY.min_points
            ):
                continue

            span = self._lane_span(
                converted
            )

            if span < (
                LANE_GEOMETRY.min_observed_span
            ):
                continue

            lanes.append(
                converted
            )

            lane_confidences.append(
                self._clip01(
                    confidences[index]
                )
            )

            original_indices.append(
                index
            )

        if len(lanes) < 2:
            return self._invalid_result()

        selected = self._select_pair(
            lanes,
            lane_confidences,
        )

        if selected is None:
            return self._invalid_result()

        (
            left_index,
            right_index,
            pair_score,
            lane_center_x,
            lane_center_y,
            lane_width,
            heading,
        ) = selected

        left_lane = lanes[
            left_index
        ]

        right_lane = lanes[
            right_index
        ]

        left_confidence = (
            lane_confidences[
                left_index
            ]
        )

        right_confidence = (
            lane_confidences[
                right_index
            ]
        )

        center_line = (
            self._build_center_line(
                left_lane,
                right_lane,
            )
        )

        if len(center_line) < 2:
            return self._invalid_result()

        observed_y_min = min(
            point[1]
            for point in center_line
        )

        observed_y_max = max(
            point[1]
            for point in center_line
        )

        observed_span = (
            observed_y_max
            - observed_y_min
        )

        # ---------------------------------------------------------------------
        # Curvatura observada
        # ---------------------------------------------------------------------

        curvature = 0.0

        if len(center_line) >= 5:

            center_array = np.asarray(
                center_line,
                dtype=np.float64,
            )

            try:

                coefficients = np.polyfit(
                    center_array[:, 1],
                    center_array[:, 0],
                    3,
                )

                y_eval = float(
                    observed_y_max
                )

                first = float(
                    np.polyval(
                        np.polyder(
                            coefficients,
                            1,
                        ),
                        y_eval,
                    )
                )

                second = float(
                    np.polyval(
                        np.polyder(
                            coefficients,
                            2,
                        ),
                        y_eval,
                    )
                )

                denominator = (
                    1.0
                    + first * first
                ) ** 1.5

                if denominator > 1e-9:
                    curvature = float(
                        second
                        / denominator
                    )

            except (
                np.linalg.LinAlgError,
                ValueError,
                FloatingPointError,
            ):
                curvature = 0.0

        # ---------------------------------------------------------------------
        # Lateral error
        # ---------------------------------------------------------------------

        image_center_x = (
            ROI.left
            + ROI.width / 2.0
        )

        image_center_y = (
            ROI.top
            + ROI.height / 2.0
        )

        # Erro normalizado pela largura do ROI.
        lateral_error = (
            image_center_x
            - lane_center_x
        ) / max(
            ROI.width,
            1,
        )

        lateral_error = float(
            np.clip(
                lateral_error,
                -1.0,
                1.0,
            )
        )

        geometry_confidence = (
            self._geometry_confidence(
                left_confidence,
                right_confidence,
                observed_span,
                lane_width,
                heading,
                curvature,
            )
        )

        # O score do par participa como limitador de confiança.
        geometry_confidence = min(
            geometry_confidence,
            self._clip01(
                pair_score
            ),
        )

        enough_for_projection = (
            observed_span
            >= LANE_GEOMETRY.min_observed_span
        )

        additional = (
            self._additional_lanes(
                lanes,
                left_index,
                right_index,
            )
        )

        return LaneGeometryResult(
            lane_center_x=float(
                lane_center_x
            ),
            lane_center_y=float(
                lane_center_y
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
                np.clip(
                    heading,
                    -LANE_GEOMETRY.max_heading_error,
                    LANE_GEOMETRY.max_heading_error,
                )
            ),
            lane_width=float(
                lane_width
            ),
            curvature=float(
                curvature
            ),
            center_line=center_line,
            valid=True,
            left_lane_screen=list(
                left_lane
            ),
            right_lane_screen=list(
                right_lane
            ),
            additional_lanes_screen=additional,
            selected_left_index=int(
                original_indices[
                    left_index
                ]
            ),
            selected_right_index=int(
                original_indices[
                    right_index
                ]
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


# =============================================================================
# HELPERS
# =============================================================================


def _weighted_confidence(
    detection: float,
    span: float,
    width: float,
    geometry: float,
) -> float:
    """
    Combina os componentes usando os pesos oficiais
    definidos em LANE_GEOMETRY.
    """

    weights = np.asarray(
        [
            LANE_GEOMETRY.confidence_weight_detection,
            LANE_GEOMETRY.confidence_weight_span,
            LANE_GEOMETRY.confidence_weight_width,
            LANE_GEOMETRY.confidence_weight_geometry,
        ],
        dtype=np.float64,
    )

    values = np.asarray(
        [
            detection,
            span,
            width,
            geometry,
        ],
        dtype=np.float64,
    )

    if not np.all(
        np.isfinite(weights)
    ):
        return 0.0

    if not np.all(
        np.isfinite(values)
    ):
        return 0.0

    weight_sum = float(
        np.sum(weights)
    )

    if weight_sum <= 1e-9:
        return 0.0

    return float(
        np.clip(
            np.sum(
                weights * values
            ) / weight_sum,
            0.0,
            1.0,
        )
    )


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    "LaneGeometry",
    "LaneGeometryResult",
]