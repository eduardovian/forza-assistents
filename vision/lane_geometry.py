"""
vision/lane_geometry.py

Forza Assistents
Geometria observada das faixas.

Responsabilidade:

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
    seleção da faixa atual
            ↓
    centro observado
            ↓
    largura
            ↓
    erro lateral
            ↓
    heading observado
            ↓
    curvatura observada
            ↓
    confiança geométrica
            ↓
    LaneGeometryResult

PRINCÍPIO FUNDAMENTAL
=====================

Este módulo trabalha SOMENTE com geometria observada.

Não:

    - executa inferência;
    - faz tracking temporal;
    - cria lanes;
    - extrapola lanes;
    - prevê trajetória;
    - inventa pontos ausentes;
    - usa memória temporal;
    - decide atuação ADAS.

A projeção/extrapolação pertence ao lane_projection.py.
A associação temporal pertence ao lane_tracker.py.
A associação semântica pertence ao lane_assignment.py.

COMPATIBILIDADE
===============

Aceita:

    LaneDetectionResult
    List[List[LanePoint]]

e mantém compatibilidade com consumidores legados que utilizam:

    ufld_width
    ufld_height

O detector atual YOLOP permanece como fonte primária.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .lane_types import LanePoint
from .yolop_detector import LaneDetectionResult


# ============================================================================
# TIPOS
# ============================================================================

Point = Tuple[float, float]


# ============================================================================
# RESULTADO
# ============================================================================


@dataclass
class LaneGeometryResult:
    """
    Resultado da geometria observada.

    Todas as coordenadas presentes neste objeto correspondem
    exclusivamente a informação observada no frame atual.
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


# ============================================================================
# GEOMETRIA
# ============================================================================


class LaneGeometry:
    """
    Processador industrial de geometria observada.

    Características:

        - aceita múltiplas lanes;
        - não depende de uma única coordenada Y;
        - seleção por pares candidatos;
        - largura avaliada em múltiplos níveis;
        - robustez contra outliers;
        - centro calculado somente na região realmente observada;
        - heading robusto;
        - curvatura observada;
        - confiança composta;
        - nenhuma extrapolação.

    A classe é deliberadamente stateless.

    Isso é importante para manter a arquitetura:

        detector
            ↓
        tracker
            ↓
        geometry
            ↓
        projection
            ↓
        assignment
            ↓
        ADAS
    """

    # ------------------------------------------------------------------------
    # DEFAULTS
    # ------------------------------------------------------------------------

    DEFAULT_SCREEN_WIDTH = 2560
    DEFAULT_SCREEN_HEIGHT = 1600

    DEFAULT_ROI = (
        300,
        700,
        2200,
        1600,
    )

    DEFAULT_DETECTOR_WIDTH = 640
    DEFAULT_DETECTOR_HEIGHT = 640

    DEFAULT_MIN_POINTS = 5
    DEFAULT_SAMPLES = 40

    DEFAULT_MIN_LANE_WIDTH = 180.0
    DEFAULT_MAX_LANE_WIDTH = 1400.0

    DEFAULT_MIN_OBSERVED_SPAN = 80.0
    DEFAULT_PROJECTION_MIN_SPAN = 180.0

    DEFAULT_OUTLIER_THRESHOLD = 100.0

    # Região inferior possui maior importância para controle lateral.
    DEFAULT_NEAR_WEIGHT = 0.75
    DEFAULT_FAR_WEIGHT = 0.25

    # ------------------------------------------------------------------------
    # CONSTRUÇÃO
    # ------------------------------------------------------------------------

    def __init__(
        self,
        screen_width: int = DEFAULT_SCREEN_WIDTH,
        screen_height: int = DEFAULT_SCREEN_HEIGHT,
        roi: Tuple[int, int, int, int] = DEFAULT_ROI,
        detector_width: int = DEFAULT_DETECTOR_WIDTH,
        detector_height: int = DEFAULT_DETECTOR_HEIGHT,
        near_weight: float = DEFAULT_NEAR_WEIGHT,
        far_weight: float = DEFAULT_FAR_WEIGHT,
        min_points: int = DEFAULT_MIN_POINTS,
        samples: int = DEFAULT_SAMPLES,
        min_lane_width: float = DEFAULT_MIN_LANE_WIDTH,
        max_lane_width: float = DEFAULT_MAX_LANE_WIDTH,
        min_observed_span: float = DEFAULT_MIN_OBSERVED_SPAN,
        outlier_threshold: float = DEFAULT_OUTLIER_THRESHOLD,
        projection_min_span: float = DEFAULT_PROJECTION_MIN_SPAN,
        ufld_width: Optional[int] = None,
        ufld_height: Optional[int] = None,
    ) -> None:

        self.screen_width = int(screen_width)
        self.screen_height = int(screen_height)

        if self.screen_width <= 0:
            raise ValueError(
                "screen_width deve ser maior que zero."
            )

        if self.screen_height <= 0:
            raise ValueError(
                "screen_height deve ser maior que zero."
            )

        (
            self.roi_left,
            self.roi_top,
            self.roi_right,
            self.roi_bottom,
        ) = map(int, roi)

        if self.roi_left < 0 or self.roi_top < 0:
            raise ValueError(
                "ROI não pode possuir coordenadas negativas."
            )

        if self.roi_right <= self.roi_left:
            raise ValueError(
                "ROI inválida: right <= left."
            )

        if self.roi_bottom <= self.roi_top:
            raise ValueError(
                "ROI inválida: bottom <= top."
            )

        if self.roi_right > self.screen_width:
            raise ValueError(
                "ROI excede a largura da tela."
            )

        if self.roi_bottom > self.screen_height:
            raise ValueError(
                "ROI excede a altura da tela."
            )

        self.roi_width = float(
            self.roi_right - self.roi_left
        )

        self.roi_height = float(
            self.roi_bottom - self.roi_top
        )

        # --------------------------------------------------------------------
        # Compatibilidade.
        #
        # YOLOP é a arquitetura atual.
        # ufld_* são aliases históricos.
        # --------------------------------------------------------------------

        if ufld_width is not None:
            detector_width = ufld_width

        if ufld_height is not None:
            detector_height = ufld_height

        self.detector_width = float(
            detector_width
        )

        self.detector_height = float(
            detector_height
        )

        if self.detector_width <= 0.0:
            raise ValueError(
                "detector_width deve ser maior que zero."
            )

        if self.detector_height <= 0.0:
            raise ValueError(
                "detector_height deve ser maior que zero."
            )

        self.near_weight = float(
            np.clip(
                near_weight,
                0.0,
                1.0,
            )
        )

        self.far_weight = float(
            np.clip(
                far_weight,
                0.0,
                1.0,
            )
        )

        if (
            self.near_weight
            + self.far_weight
            <= 0.0
        ):
            self.near_weight = 0.75
            self.far_weight = 0.25

        self.min_points = max(
            2,
            int(min_points),
        )

        self.samples = max(
            8,
            int(samples),
        )

        self.min_lane_width = float(
            min_lane_width
        )

        self.max_lane_width = float(
            max_lane_width
        )

        if self.min_lane_width <= 0.0:
            raise ValueError(
                "min_lane_width deve ser maior que zero."
            )

        if self.max_lane_width < self.min_lane_width:
            raise ValueError(
                "max_lane_width deve ser >= min_lane_width."
            )

        self.min_observed_span = max(
            1.0,
            float(min_observed_span),
        )

        self.projection_min_span = max(
            self.min_observed_span,
            float(projection_min_span),
        )

        self.outlier_threshold = max(
            1.0,
            float(outlier_threshold),
        )

        self.image_center_x = (
            self.screen_width / 2.0
        )

        self.image_center_y = (
            self.screen_height / 2.0
        )

    # =========================================================================
    # NUMERIC HELPERS
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
    # COORDENADAS
    # =========================================================================

    def _detector_to_screen(
        self,
        x: float,
        y: float,
        detector_width: Optional[float] = None,
        detector_height: Optional[float] = None,
    ) -> Point:

        width = (
            self.detector_width
            if detector_width is None
            else float(detector_width)
        )

        height = (
            self.detector_height
            if detector_height is None
            else float(detector_height)
        )

        if width <= 0.0 or height <= 0.0:
            raise ValueError(
                "Dimensões do detector inválidas."
            )

        screen_x = (
            self.roi_left
            + (float(x) / width)
            * self.roi_width
        )

        screen_y = (
            self.roi_top
            + (float(y) / height)
            * self.roi_height
        )

        return (
            float(screen_x),
            float(screen_y),
        )

    def _get_detection_dimensions(
        self,
        detection: object,
    ) -> Tuple[float, float]:

        width = getattr(
            detection,
            "input_width",
            self.detector_width,
        )

        height = getattr(
            detection,
            "input_height",
            self.detector_height,
        )

        try:
            width = float(width)
        except (
            TypeError,
            ValueError,
        ):
            width = self.detector_width

        try:
            height = float(height)
        except (
            TypeError,
            ValueError,
        ):
            height = self.detector_height

        if (
            not math.isfinite(width)
            or width <= 0.0
        ):
            width = self.detector_width

        if (
            not math.isfinite(height)
            or height <= 0.0
        ):
            height = self.detector_height

        return (
            width,
            height,
        )

    # =========================================================================
    # PONTOS
    # =========================================================================

    def _convert_lane(
        self,
        lane: Sequence[LanePoint],
        detector_width: Optional[float] = None,
        detector_height: Optional[float] = None,
    ) -> List[Point]:

        if lane is None:
            return []

        points: List[Point] = []

        width = (
            self.detector_width
            if detector_width is None
            else float(detector_width)
        )

        height = (
            self.detector_height
            if detector_height is None
            else float(detector_height)
        )

        if width <= 0.0 or height <= 0.0:
            return []

        for point in lane:

            if point is None:
                continue

            try:
                valid = bool(point.valid)
                x = float(point.x)
                y = float(point.y)
                confidence = float(
                    point.confidence
                )
            except (
                AttributeError,
                TypeError,
                ValueError,
            ):
                continue

            if not valid:
                continue

            if (
                not math.isfinite(x)
                or not math.isfinite(y)
                or not math.isfinite(confidence)
            ):
                continue

            if confidence <= 0.0:
                continue

            if (
                x < 0.0
                or x > width
                or y < 0.0
                or y > height
            ):
                continue

            sx, sy = self._detector_to_screen(
                x=x,
                y=y,
                detector_width=width,
                detector_height=height,
            )

            if (
                not math.isfinite(sx)
                or not math.isfinite(sy)
            ):
                continue

            if (
                sx < 0.0
                or sx > self.screen_width
                or sy < 0.0
                or sy > self.screen_height
            ):
                continue

            points.append(
                (
                    float(sx),
                    float(sy),
                )
            )

        points.sort(
            key=lambda point: point[1]
        )

        return points

    # =========================================================================
    # OUTLIERS
    # =========================================================================

    def _remove_outliers(
        self,
        points: Sequence[Point],
    ) -> List[Point]:

        if len(points) < 4:
            return list(points)

        arr = np.asarray(
            points,
            dtype=np.float64,
        )

        if (
            arr.ndim != 2
            or arr.shape[1] != 2
            or not np.all(
                np.isfinite(arr)
            )
        ):
            return list(points)

        y = arr[:, 1]
        x = arr[:, 0]

        if float(np.ptp(y)) < 1.0:
            return list(points)

        y_mean = float(
            np.mean(y)
        )

        y_std = float(
            np.std(y)
        )

        y_norm = (
            y - y_mean
        ) / max(
            y_std,
            1.0,
        )

        try:
            coeff = np.polyfit(
                y_norm,
                x,
                1,
            )

            predicted = np.polyval(
                coeff,
                y_norm,
            )

        except (
            np.linalg.LinAlgError,
            ValueError,
            FloatingPointError,
        ):
            return list(points)

        residual = np.abs(
            x - predicted
        )

        median = float(
            np.median(residual)
        )

        mad = float(
            np.median(
                np.abs(
                    residual - median
                )
            )
        )

        threshold = max(
            self.outlier_threshold,
            median
            + 4.0
            * max(
                mad,
                1.0,
            ),
        )

        keep = (
            residual <= threshold
        )

        filtered = [
            (
                float(arr[i, 0]),
                float(arr[i, 1]),
            )
            for i in range(
                len(arr)
            )
            if bool(keep[i])
        ]

        if len(filtered) < self.min_points:
            return list(points)

        return filtered

    # =========================================================================
    # PREPARAÇÃO
    # =========================================================================

    def _prepare_lane(
        self,
        lane: Sequence[Point],
    ) -> List[Point]:

        if lane is None:
            return []

        if len(lane) < self.min_points:
            return []

        clean = self._remove_outliers(
            lane
        )

        if len(clean) < self.min_points:
            return []

        # --------------------------------------------------------------------
        # Remove duplicatas de Y.
        #
        # Não cria informação.
        # Apenas evita que múltiplos pontos no mesmo nível
        # prejudiquem a interpolação.
        # --------------------------------------------------------------------

        by_y: dict[float, List[float]] = {}

        for x, y in clean:

            if (
                not math.isfinite(x)
                or not math.isfinite(y)
            ):
                continue

            by_y.setdefault(
                float(y),
                [],
            ).append(
                float(x)
            )

        result: List[Point] = []

        for y in sorted(by_y):

            xs = by_y[y]

            if not xs:
                continue

            result.append(
                (
                    float(
                        np.median(xs)
                    ),
                    float(y),
                )
            )

        if len(result) < self.min_points:
            return []

        return result

    # =========================================================================
    # INTERPOLAÇÃO
    # =========================================================================

    @staticmethod
    def _x_at_y(
        lane: Sequence[Point],
        y: float,
    ) -> Optional[float]:

        if len(lane) < 2:
            return None

        arr = np.asarray(
            lane,
            dtype=np.float64,
        )

        if (
            arr.ndim != 2
            or arr.shape[1] != 2
            or not np.all(
                np.isfinite(arr)
            )
        ):
            return None

        order = np.argsort(
            arr[:, 1]
        )

        ys = arr[
            order,
            1,
        ]

        xs = arr[
            order,
            0,
        ]

        if (
            y < ys[0]
            or y > ys[-1]
        ):
            return None

        return float(
            np.interp(
                float(y),
                ys,
                xs,
            )
        )

    @staticmethod
    def _lane_y_range(
        lane: Sequence[Point],
    ) -> Optional[Tuple[float, float]]:

        if not lane:
            return None

        ys = np.asarray(
            [
                point[1]
                for point in lane
            ],
            dtype=np.float64,
        )

        if (
            ys.size == 0
            or not np.all(
                np.isfinite(ys)
            )
        ):
            return None

        return (
            float(np.min(ys)),
            float(np.max(ys)),
        )

    # =========================================================================
    # ORDENAÇÃO
    # =========================================================================

    def _reference_y(
        self,
        lane: Optional[Sequence[Point]] = None,
    ) -> float:

        if lane:

            y_range = self._lane_y_range(
                lane
            )

            if y_range is not None:

                return float(
                    y_range[1]
                    - max(
                        5.0,
                        (
                            y_range[1]
                            - y_range[0]
                        )
                        * 0.10,
                    )
                )

        return float(
            self.roi_bottom
            - self.roi_height * 0.10
        )

    def _lane_position(
        self,
        lane: Sequence[Point],
    ) -> Optional[float]:

        y_range = self._lane_y_range(
            lane
        )

        if y_range is None:
            return None

        reference_y = self._reference_y(
            lane
        )

        x = self._x_at_y(
            lane,
            reference_y,
        )

        if x is not None:
            return x

        # Fallback estritamente observado.
        return float(
            lane[-1][0]
        )

    def _sort_lanes(
        self,
        lanes: Sequence[List[Point]],
    ) -> List[List[Point]]:

        positioned = []

        for index, lane in enumerate(lanes):

            x = self._lane_position(
                lane
            )

            if x is None:
                continue

            positioned.append(
                (
                    float(x),
                    index,
                )
            )

        positioned.sort(
            key=lambda item: item[0]
        )

        return [
            lanes[index]
            for _, index in positioned
        ]

    # =========================================================================
    # SOBREPOSIÇÃO
    # =========================================================================

    def _overlap_range(
        self,
        left: Sequence[Point],
        right: Sequence[Point],
    ) -> Optional[Tuple[float, float]]:

        left_range = self._lane_y_range(
            left
        )

        right_range = self._lane_y_range(
            right
        )

        if (
            left_range is None
            or right_range is None
        ):
            return None

        y_min = max(
            left_range[0],
            right_range[0],
        )

        y_max = min(
            left_range[1],
            right_range[1],
        )

        if y_max <= y_min:
            return None

        return (
            float(y_min),
            float(y_max),
        )

    # =========================================================================
    # LARGURA
    # =========================================================================

    def _sample_pair(
        self,
        left: Sequence[Point],
        right: Sequence[Point],
        samples: Optional[int] = None,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:

        overlap = self._overlap_range(
            left,
            right,
        )

        if overlap is None:
            return (
                np.empty(0),
                np.empty(0),
                np.empty(0),
            )

        y_min, y_max = overlap

        span = y_max - y_min

        if span < 1.0:
            return (
                np.empty(0),
                np.empty(0),
                np.empty(0),
            )

        count = (
            self.samples
            if samples is None
            else max(
                4,
                int(samples),
            )
        )

        ys = np.linspace(
            y_min,
            y_max,
            count,
        )

        left_xs = []
        right_xs = []
        valid_ys = []

        for y in ys:

            lx = self._x_at_y(
                left,
                float(y),
            )

            rx = self._x_at_y(
                right,
                float(y),
            )

            if (
                lx is None
                or rx is None
            ):
                continue

            if rx <= lx:
                continue

            left_xs.append(
                float(lx)
            )

            right_xs.append(
                float(rx)
            )

            valid_ys.append(
                float(y)
            )

        if not valid_ys:
            return (
                np.empty(0),
                np.empty(0),
                np.empty(0),
            )

        return (
            np.asarray(
                valid_ys,
                dtype=np.float64,
            ),
            np.asarray(
                left_xs,
                dtype=np.float64,
            ),
            np.asarray(
                right_xs,
                dtype=np.float64,
            ),
        )

    def _compute_lane_width(
        self,
        left: Sequence[Point],
        right: Sequence[Point],
    ) -> float:

        (
            _ys,
            left_x,
            right_x,
        ) = self._sample_pair(
            left,
            right,
            samples=24,
        )

        if left_x.size == 0:
            return 0.0

        widths = (
            right_x - left_x
        )

        widths = widths[
            np.isfinite(widths)
        ]

        widths = widths[
            (
                widths
                >= self.min_lane_width
            )
            & (
                widths
                <= self.max_lane_width
            )
        ]

        if widths.size == 0:
            return 0.0

        return float(
            np.median(widths)
        )

    # =========================================================================
    # CONSISTÊNCIA DE LARGURA
    # =========================================================================

    def _width_consistency(
        self,
        left: Sequence[Point],
        right: Sequence[Point],
    ) -> float:

        (
            _ys,
            left_x,
            right_x,
        ) = self._sample_pair(
            left,
            right,
            samples=24,
        )

        if left_x.size < 4:
            return 0.0

        widths = (
            right_x - left_x
        )

        if not np.all(
            np.isfinite(widths)
        ):
            return 0.0

        median = float(
            np.median(widths)
        )

        if median <= 0.0:
            return 0.0

        mad = float(
            np.median(
                np.abs(
                    widths - median
                )
            )
        )

        # 0 = inconsistente
        # 1 = praticamente constante
        normalized = (
            mad / median
        )

        return float(
            np.clip(
                1.0
                - normalized * 4.0,
                0.0,
                1.0,
            )
        )

    # =========================================================================
    # CENTRO
    # =========================================================================

    def _compute_center_line(
        self,
        left: Sequence[Point],
        right: Sequence[Point],
    ) -> List[Point]:

        (
            ys,
            left_x,
            right_x,
        ) = self._sample_pair(
            left,
            right,
            samples=self.samples,
        )

        if ys.size == 0:
            return []

        center_x = (
            left_x + right_x
        ) * 0.5

        result: List[Point] = []

        for x, y in zip(
            center_x,
            ys,
        ):

            if (
                not math.isfinite(
                    float(x)
                )
                or not math.isfinite(
                    float(y)
                )
            ):
                continue

            result.append(
                (
                    float(x),
                    float(y),
                )
            )

        return result

    # =========================================================================
    # SELEÇÃO DE PARES
    # =========================================================================

    def _pair_score(
        self,
        left: Sequence[Point],
        right: Sequence[Point],
    ) -> Optional[float]:

        (
            ys,
            left_x,
            right_x,
        ) = self._sample_pair(
            left,
            right,
            samples=24,
        )

        if ys.size < max(
            4,
            self.min_points,
        ):
            return None

        widths = (
            right_x - left_x
        )

        valid_widths = widths[
            (
                widths
                >= self.min_lane_width
            )
            & (
                widths
                <= self.max_lane_width
            )
        ]

        if valid_widths.size < 4:
            return None

        median_width = float(
            np.median(
                valid_widths
            )
        )

        width_consistency = (
            self._width_consistency(
                left,
                right,
            )
        )

        observed_span = float(
            ys[-1] - ys[0]
        )

        if observed_span <= 0.0:
            return None

        center_x = (
            left_x + right_x
        ) * 0.5

        # --------------------------------------------------------------------
        # Avaliamos mais fortemente a região inferior.
        # --------------------------------------------------------------------

        relative = (
            ys - ys[0]
        ) / max(
            observed_span,
            1.0,
        )

        weights = (
            self.far_weight
            + (
                self.near_weight
                - self.far_weight
            )
            * relative
        )

        weights = np.maximum(
            weights,
            1e-6,
        )

        weighted_center = float(
            np.average(
                center_x,
                weights=weights,
            )
        )

        center_distance = abs(
            weighted_center
            - self.image_center_x
        )

        # Normalização baseada na metade útil da tela.
        center_score = float(
            np.clip(
                1.0
                - center_distance
                / max(
                    self.roi_width * 0.5,
                    1.0,
                ),
                0.0,
                1.0,
            )
        )

        span_score = float(
            np.clip(
                observed_span
                / max(
                    self.roi_height * 0.60,
                    1.0,
                ),
                0.0,
                1.0,
            )
        )

        ideal_width = (
            self.min_lane_width
            + self.max_lane_width
        ) * 0.5

        width_range = (
            self.max_lane_width
            - self.min_lane_width
        )

        if width_range <= 0.0:
            width_score = 0.0
        else:
            width_score = float(
                np.clip(
                    1.0
                    - abs(
                        median_width
                        - ideal_width
                    )
                    / max(
                        width_range * 0.5,
                        1.0,
                    ),
                    0.0,
                    1.0,
                )
            )

        # --------------------------------------------------------------------
        # Score final.
        #
        # A proximidade ao veículo é importante, mas não pode dominar
        # completamente a geometria.
        # --------------------------------------------------------------------

        score = (
            center_score * 0.35
            + width_score * 0.20
            + width_consistency * 0.25
            + span_score * 0.20
        )

        return float(
            np.clip(
                score,
                0.0,
                1.0,
            )
        )

    def _select_current_lane(
        self,
        lanes: Sequence[List[Point]],
    ) -> Optional[Tuple[int, int]]:

        if len(lanes) < 2:
            return None

        candidates = []

        for index in range(
            len(lanes) - 1
        ):

            left = lanes[index]
            right = lanes[index + 1]

            score = self._pair_score(
                left,
                right,
            )

            if score is None:
                continue

            (
                ys,
                left_x,
                right_x,
            ) = self._sample_pair(
                left,
                right,
                samples=24,
            )

            if ys.size == 0:
                continue

            center_x = (
                left_x + right_x
            ) * 0.5

            relative = (
                ys - ys[0]
            ) / max(
                ys[-1] - ys[0],
                1.0,
            )

            weights = (
                self.far_weight
                + (
                    self.near_weight
                    - self.far_weight
                )
                * relative
            )

            center = float(
                np.average(
                    center_x,
                    weights=weights,
                )
            )

            distance = abs(
                center
                - self.image_center_x
            )

            candidates.append(
                (
                    score,
                    distance,
                    index,
                    index + 1,
                )
            )

        if not candidates:
            return None

        # --------------------------------------------------------------------
        # Score domina.
        # Distância ao centro desempata.
        # --------------------------------------------------------------------

        candidates.sort(
            key=lambda item: (
                -item[0],
                item[1],
            )
        )

        _, _, left_index, right_index = (
            candidates[0]
        )

        return (
            left_index,
            right_index,
        )

    # =========================================================================
    # ERRO LATERAL
    # =========================================================================

    def _compute_lateral_error(
        self,
        center_line: Sequence[Point],
    ) -> float:

        if not center_line:
            return 0.0

        # --------------------------------------------------------------------
        # Não utilizamos um Y fixo que pode estar fora da observação.
        #
        # Usamos a região inferior realmente observada.
        # --------------------------------------------------------------------

        arr = np.asarray(
            center_line,
            dtype=np.float64,
        )

        if (
            arr.ndim != 2
            or arr.shape[1] != 2
            or not np.all(
                np.isfinite(arr)
            )
        ):
            return 0.0

        y_max = float(
            np.max(arr[:, 1])
        )

        y_min = float(
            np.min(arr[:, 1])
        )

        observed_span = (
            y_max - y_min
        )

        if observed_span <= 0.0:
            center_x = float(
                arr[-1, 0]
            )
        else:
            # Região inferior observada.
            reference_y = (
                y_max
                - observed_span * 0.12
            )

            x = self._x_at_y(
                center_line,
                reference_y,
            )

            center_x = (
                float(x)
                if x is not None
                else float(arr[-1, 0])
            )

        error = (
            center_x
            - self.image_center_x
        )

        # --------------------------------------------------------------------
        # Normalização:
        #
        # ±1 corresponde aproximadamente a ±metade da região útil.
        # --------------------------------------------------------------------

        denominator = max(
            self.roi_width * 0.5,
            1.0,
        )

        return float(
            np.clip(
                error / denominator,
                -1.0,
                1.0,
            )
        )

    # =========================================================================
    # HEADING
    # =========================================================================

    def _compute_heading_error(
        self,
        center_line: Sequence[Point],
    ) -> float:

        if len(center_line) < 5:
            return 0.0

        arr = np.asarray(
            center_line,
            dtype=np.float64,
        )

        if (
            arr.ndim != 2
            or arr.shape[1] != 2
            or not np.all(
                np.isfinite(arr)
            )
        ):
            return 0.0

        # --------------------------------------------------------------------
        # Utilizamos a região inferior observada.
        # --------------------------------------------------------------------

        start = max(
            0,
            int(
                len(arr) * 0.55
            ),
        )

        local = arr[
            start:
        ]

        if len(local) < 3:
            return 0.0

        x = local[:, 0]
        y = local[:, 1]

        if float(np.ptp(y)) < 1.0:
            return 0.0

        # --------------------------------------------------------------------
        # Regressão robusta x=f(y).
        #
        # Isso é significativamente mais estável que calcular apenas
        # diferenças entre pontos consecutivos.
        # --------------------------------------------------------------------

        y_mean = float(
            np.mean(y)
        )

        y_scale = float(
            np.std(y)
        )

        if y_scale < 1e-6:
            return 0.0

        y_norm = (
            y - y_mean
        ) / y_scale

        try:
            coeff = np.polyfit(
                y_norm,
                x,
                1,
            )

        except (
            np.linalg.LinAlgError,
            ValueError,
            FloatingPointError,
        ):
            return 0.0

        slope_normalized = float(
            coeff[0]
        )

        # slope original:
        #
        # dx/dy = coeff / std(y)
        #
        slope = (
            slope_normalized
            / y_scale
        )

        if not math.isfinite(
            slope
        ):
            return 0.0

        angle = math.atan(
            slope
        )

        # ±45 graus -> ±1.
        max_angle = math.radians(
            45.0
        )

        return float(
            np.clip(
                angle / max_angle,
                -1.0,
                1.0,
            )
        )

    # =========================================================================
    # CURVATURA
    # =========================================================================

    def _compute_curvature(
        self,
        center_line: Sequence[Point],
    ) -> float:

        if len(center_line) < 7:
            return 0.0

        arr = np.asarray(
            center_line,
            dtype=np.float64,
        )

        if (
            arr.ndim != 2
            or arr.shape[1] != 2
            or not np.all(
                np.isfinite(arr)
            )
        ):
            return 0.0

        x = arr[:, 0]
        y = arr[:, 1]

        if float(
            np.ptp(y)
        ) < 20.0:
            return 0.0

        # --------------------------------------------------------------------
        # Remove pequenos problemas de espaçamento de Y.
        # --------------------------------------------------------------------

        order = np.argsort(y)

        y = y[order]
        x = x[order]

        unique_y, unique_indices = (
            np.unique(
                y,
                return_index=True,
            )
        )

        x = x[
            unique_indices
        ]

        y = unique_y

        if len(y) < 7:
            return 0.0

        try:

            dx_dy = np.gradient(
                x,
                y,
            )

            d2x_dy2 = np.gradient(
                dx_dy,
                y,
            )

        except (
            ValueError,
            FloatingPointError,
        ):
            return 0.0

        denominator = np.power(
            1.0
            + dx_dy * dx_dy,
            1.5,
        )

        valid = (
            denominator > 1e-9
        )

        if not np.any(valid):
            return 0.0

        values = (
            np.abs(
                d2x_dy2[valid]
            )
            / denominator[valid]
        )

        values = values[
            np.isfinite(values)
        ]

        if values.size == 0:
            return 0.0

        curvature = float(
            np.median(values)
        )

        # --------------------------------------------------------------------
        # Normalização visual.
        #
        # A unidade interna permanece estável entre frames.
        # --------------------------------------------------------------------

        return float(
            np.clip(
                curvature * 500.0,
                0.0,
                1.0,
            )
        )

    # =========================================================================
    # CENTRO REPRESENTATIVO
    # =========================================================================

    def _compute_weighted_center(
        self,
        center_line: Sequence[Point],
    ) -> Tuple[float, float]:

        if not center_line:
            return (
                self.image_center_x,
                self.image_center_y,
            )

        arr = np.asarray(
            center_line,
            dtype=np.float64,
        )

        if (
            arr.ndim != 2
            or arr.shape[1] != 2
            or not np.all(
                np.isfinite(arr)
            )
        ):
            return (
                self.image_center_x,
                self.image_center_y,
            )

        y = arr[:, 1]

        relative = (
            y - self.roi_top
        ) / max(
            self.roi_height,
            1.0,
        )

        relative = np.clip(
            relative,
            0.0,
            1.0,
        )

        weights = (
            self.far_weight
            + (
                self.near_weight
                - self.far_weight
            )
            * relative
        )

        weights = np.maximum(
            weights,
            1e-6,
        )

        total = float(
            np.sum(weights)
        )

        if total <= 0.0:
            weights = np.ones(
                len(arr),
                dtype=np.float64,
            )

        weights /= float(
            np.sum(weights)
        )

        center_x = float(
            np.average(
                arr[:, 0],
                weights=weights,
            )
        )

        center_y = float(
            np.average(
                arr[:, 1],
                weights=weights,
            )
        )

        return (
            center_x,
            center_y,
        )

    # =========================================================================
    # CONFIANÇA
    # =========================================================================

    def _compute_geometry_confidence(
        self,
        left: Sequence[Point],
        right: Sequence[Point],
        center: Sequence[Point],
        lane_width: float,
    ) -> float:

        if (
            len(left) < self.min_points
            or len(right) < self.min_points
            or len(center) < self.min_points
        ):
            return 0.0

        if not (
            self.min_lane_width
            <= lane_width
            <= self.max_lane_width
        ):
            return 0.0

        # --------------------------------------------------------------------
        # Quantidade de observações.
        # --------------------------------------------------------------------

        point_score = float(
            np.clip(
                min(
                    len(left),
                    len(right),
                )
                / max(
                    self.min_points * 3.0,
                    1.0,
                ),
                0.0,
                1.0,
            )
        )

        # --------------------------------------------------------------------
        # Extensão vertical.
        # --------------------------------------------------------------------

        y = np.asarray(
            [
                point[1]
                for point in center
            ],
            dtype=np.float64,
        )

        if y.size == 0:
            return 0.0

        observed_span = float(
            np.ptp(y)
        )

        span_score = float(
            np.clip(
                observed_span
                / max(
                    self.roi_height * 0.60,
                    1.0,
                ),
                0.0,
                1.0,
            )
        )

        # --------------------------------------------------------------------
        # Consistência da largura.
        # --------------------------------------------------------------------

        width_consistency = (
            self._width_consistency(
                left,
                right,
            )
        )

        # --------------------------------------------------------------------
        # Largura.
        #
        # Não penalizamos demais larguras legítimas próximas das bordas
        # do intervalo.
        # --------------------------------------------------------------------

        width_range = (
            self.max_lane_width
            - self.min_lane_width
        )

        if width_range <= 0.0:

            width_score = 1.0

        else:

            normalized = (
                lane_width
                - self.min_lane_width
            ) / width_range

            width_score = float(
                np.clip(
                    1.0
                    - abs(
                        normalized
                        - 0.5
                    ) * 0.8,
                    0.0,
                    1.0,
                )
            )

        confidence = (
            point_score * 0.20
            + span_score * 0.30
            + width_consistency * 0.30
            + width_score * 0.20
        )

        return float(
            np.clip(
                confidence,
                0.0,
                1.0,
            )
        )

    # =========================================================================
    # RESULTADO INVÁLIDO
    # =========================================================================

    def _invalid_result(
        self,
        additional_lanes: Optional[
            List[List[Point]]
        ] = None,
        left: Optional[
            List[Point]
        ] = None,
        right: Optional[
            List[Point]
        ] = None,
    ) -> LaneGeometryResult:

        return LaneGeometryResult(
            lane_center_x=self.image_center_x,
            lane_center_y=self.image_center_y,
            image_center_x=self.image_center_x,
            image_center_y=self.image_center_y,
            lateral_error=0.0,
            heading_error=0.0,
            lane_width=0.0,
            curvature=0.0,
            center_line=[],
            valid=False,
            left_lane_screen=(
                left
                if left is not None
                else []
            ),
            right_lane_screen=(
                right
                if right is not None
                else []
            ),
            additional_lanes_screen=(
                additional_lanes
                if additional_lanes is not None
                else []
            ),
            selected_left_index=-1,
            selected_right_index=-1,
            geometry_confidence=0.0,
            observed_y_min=0.0,
            observed_y_max=0.0,
            observed_span=0.0,
            enough_for_projection=False,
        )

    # =========================================================================
    # EXTRAÇÃO DAS LANES
    # =========================================================================

    def _extract_lanes(
        self,
        detection: object,
        detector_width: float,
        detector_height: float,
    ) -> List[List[Point]]:

        # --------------------------------------------------------------------
        # Prioridade:
        #
        # 1. detection.lanes
        #
        # Isso é importante porque o YOLOP atual entrega todas as lanes.
        # Não queremos perder as lanes adicionais.
        # --------------------------------------------------------------------

        raw_lanes = getattr(
            detection,
            "lanes",
            None,
        )

        candidates = []

        if raw_lanes:

            candidates.extend(
                raw_lanes
            )

        else:

            # Compatibilidade com resultados antigos.
            left = getattr(
                detection,
                "left_lane",
                [],
            )

            right = getattr(
                detection,
                "right_lane",
                [],
            )

            additional = getattr(
                detection,
                "additional_lanes",
                [],
            )

            if left:
                candidates.append(
                    left
                )

            if additional:
                candidates.extend(
                    additional
                )

            if right:
                candidates.append(
                    right
                )

        result = []

        for lane in candidates:

            converted = self._convert_lane(
                lane,
                detector_width,
                detector_height,
            )

            prepared = self._prepare_lane(
                converted
            )

            if len(prepared) < self.min_points:
                continue

            result.append(
                prepared
            )

        return result

    # =========================================================================
    # PIPELINE PRINCIPAL
    # =========================================================================

    def compute(
        self,
        detection: LaneDetectionResult,
    ) -> LaneGeometryResult:
        """
        Calcula a geometria observada.

        A seleção da faixa é feita por evidência geométrica do frame atual.
        """

        if detection is None:
            return self._invalid_result()

        detector_width, detector_height = (
            self._get_detection_dimensions(
                detection
            )
        )

        lanes = self._extract_lanes(
            detection,
            detector_width,
            detector_height,
        )

        if len(lanes) < 2:
            return self._invalid_result()

        # --------------------------------------------------------------------
        # Ordenação espacial.
        # --------------------------------------------------------------------

        ordered_lanes = self._sort_lanes(
            lanes
        )

        if len(ordered_lanes) < 2:
            return self._invalid_result(
                additional_lanes=ordered_lanes
            )

        # --------------------------------------------------------------------
        # Seleção do par atual.
        #
        # Diferentemente da implementação anterior, NÃO exigimos que todas
        # as lanes tenham observação no mesmo Y.
        # --------------------------------------------------------------------

        selected = self._select_current_lane(
            ordered_lanes
        )

        if selected is None:
            return self._invalid_result(
                additional_lanes=ordered_lanes
            )

        left_index, right_index = selected

        selected_left = ordered_lanes[
            left_index
        ]

        selected_right = ordered_lanes[
            right_index
        ]

        # --------------------------------------------------------------------
        # Centro.
        # --------------------------------------------------------------------

        center_line = self._compute_center_line(
            selected_left,
            selected_right,
        )

        if len(center_line) < self.min_points:
            return self._invalid_result(
                additional_lanes=ordered_lanes,
                left=selected_left,
                right=selected_right,
            )

        # --------------------------------------------------------------------
        # Largura.
        # --------------------------------------------------------------------

        lane_width = self._compute_lane_width(
            selected_left,
            selected_right,
        )

        if not (
            self.min_lane_width
            <= lane_width
            <= self.max_lane_width
        ):
            return self._invalid_result(
                additional_lanes=ordered_lanes,
                left=selected_left,
                right=selected_right,
            )

        # --------------------------------------------------------------------
        # Métricas.
        # --------------------------------------------------------------------

        lateral_error = (
            self._compute_lateral_error(
                center_line
            )
        )

        heading_error = (
            self._compute_heading_error(
                center_line
            )
        )

        curvature = (
            self._compute_curvature(
                center_line
            )
        )

        center_x, center_y = (
            self._compute_weighted_center(
                center_line
            )
        )

        # --------------------------------------------------------------------
        # Extensão realmente observada.
        # --------------------------------------------------------------------

        y_values = np.asarray(
            [
                point[1]
                for point in center_line
            ],
            dtype=np.float64,
        )

        observed_y_min = float(
            np.min(y_values)
        )

        observed_y_max = float(
            np.max(y_values)
        )

        observed_span = (
            observed_y_max
            - observed_y_min
        )

        # --------------------------------------------------------------------
        # Confiança.
        # --------------------------------------------------------------------

        geometry_confidence = (
            self._compute_geometry_confidence(
                selected_left,
                selected_right,
                center_line,
                lane_width,
            )
        )

        # --------------------------------------------------------------------
        # Pronto para projeção?
        #
        # Isto NÃO significa que a geometria foi projetada.
        # Apenas significa que existe informação observada suficiente.
        # --------------------------------------------------------------------

        enough_for_projection = bool(
            geometry_confidence >= 0.55
            and observed_span
            >= self.projection_min_span
            and len(center_line)
            >= self.min_points
        )

        # --------------------------------------------------------------------
        # Lanes restantes.
        # --------------------------------------------------------------------

        additional = [
            lane
            for index, lane in enumerate(
                ordered_lanes
            )
            if index
            not in (
                left_index,
                right_index,
            )
        ]

        # --------------------------------------------------------------------
        # A geometria é considerada válida quando existe um par observado
        # coerente. Não exigimos que projection já seja possível.
        # --------------------------------------------------------------------

        valid = bool(
            geometry_confidence >= 0.35
            and len(center_line)
            >= self.min_points
            and self.min_lane_width
            <= lane_width
            <= self.max_lane_width
        )

        return LaneGeometryResult(
            lane_center_x=center_x,
            lane_center_y=center_y,
            image_center_x=self.image_center_x,
            image_center_y=self.image_center_y,
            lateral_error=lateral_error,
            heading_error=heading_error,
            lane_width=lane_width,
            curvature=curvature,
            center_line=center_line,
            valid=valid,
            left_lane_screen=selected_left,
            right_lane_screen=selected_right,
            additional_lanes_screen=additional,
            selected_left_index=left_index,
            selected_right_index=right_index,
            geometry_confidence=geometry_confidence,
            observed_y_min=observed_y_min,
            observed_y_max=observed_y_max,
            observed_span=observed_span,
            enough_for_projection=(
                enough_for_projection
            ),
        )

    # =========================================================================
    # ALIASES DE COMPATIBILIDADE
    # =========================================================================

    def process(
        self,
        detection: LaneDetectionResult,
    ) -> LaneGeometryResult:
        """
        Alias semântico para compute().
        """

        return self.compute(
            detection
        )

    def calculate(
        self,
        detection: LaneDetectionResult,
    ) -> LaneGeometryResult:
        """
        Alias de compatibilidade.
        """

        return self.compute(
            detection
        )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "Point",
    "LaneGeometryResult",
    "LaneGeometry",
]