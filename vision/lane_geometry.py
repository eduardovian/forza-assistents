"""
vision/lane_geometry.py

Forza Horizon ADAS/LKA
Geometria OBSERVADA das faixas usando YOLOP.

Responsabilidade deste módulo:

    YOLOP LaneDetectionResult
            ↓
    coordenadas da tela
            ↓
    validação dos pontos
            ↓
    remoção de outliers
            ↓
    ordenação das lanes
            ↓
    seleção da faixa atual
            ↓
    centro OBSERVADO da faixa
            ↓
    largura
            ↓
    erro lateral
            ↓
    heading OBSERVADO
            ↓
    qualidade geométrica

IMPORTANTE
==========

Este módulo NÃO faz previsão da faixa.

Não deve:

- extrapolar uma lane além dos pontos observados;
- criar pontos onde não existem observações;
- ajustar polinômio para prever o restante da pista;
- decidir a trajetória futura;
- assumir uma lane desaparecida;
- utilizar memória temporal para inventar geometria.

A previsão/extrapolação pertence ao:

    vision/lane_projection.py

A geometria aqui representa somente aquilo que foi
efetivamente observado pelo YOLOP/tracker.
"""

from __future__ import annotations

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

    Todos os pontos representam informação observada.

    Nenhum ponto deste resultado representa extrapolação futura.
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
    Processa exclusivamente a geometria observada pelo YOLOP.

    Não realiza:

    - tracking;
    - previsão;
    - extrapolação;
    - identificação temporal da faixa;
    - controle do veículo.

    A associação temporal pertence ao LaneTracker.
    A projeção pertence ao LaneProjection.
    """

    def __init__(
        self,
        screen_width: int = 2560,
        screen_height: int = 1600,
        roi: Tuple[int, int, int, int] = (
            300,
            700,
            2200,
            1600,
        ),
        detector_width: int = 640,
        detector_height: int = 640,
        near_weight: float = 0.75,
        far_weight: float = 0.25,
        min_points: int = 5,
        samples: int = 40,
        min_lane_width: float = 180.0,
        max_lane_width: float = 1400.0,
        min_observed_span: float = 80.0,
        outlier_threshold: float = 100.0,
        projection_min_span: float = 180.0,
        # ------------------------------------------------------------------
        # Compatibilidade com testes/consumidores antigos.
        #
        # Estes nomes NÃO fazem parte da arquitetura YOLOP.
        # São apenas aliases para detector_width/height.
        # ------------------------------------------------------------------
        ufld_width: Optional[int] = None,
        ufld_height: Optional[int] = None,
    ) -> None:

        self.screen_width = int(screen_width)
        self.screen_height = int(screen_height)

        (
            self.roi_left,
            self.roi_top,
            self.roi_right,
            self.roi_bottom,
        ) = map(int, roi)

        if self.roi_right <= self.roi_left:
            raise ValueError(
                "ROI inválida: roi_right deve ser maior "
                "que roi_left."
            )

        if self.roi_bottom <= self.roi_top:
            raise ValueError(
                "ROI inválida: roi_bottom deve ser maior "
                "que roi_top."
            )

        self.roi_width = float(
            self.roi_right - self.roi_left
        )

        self.roi_height = float(
            self.roi_bottom - self.roi_top
        )

        # ------------------------------------------------------------------
        # YOLOP é a referência atual.
        #
        # Os aliases ufld_* só existem para não quebrar testes antigos.
        # ------------------------------------------------------------------

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
                "max_lane_width deve ser maior ou igual "
                "a min_lane_width."
            )

        self.min_observed_span = float(
            max(
                1.0,
                min_observed_span,
            )
        )

        self.outlier_threshold = float(
            max(
                1.0,
                outlier_threshold,
            )
        )

        self.projection_min_span = float(
            max(
                self.min_observed_span,
                projection_min_span,
            )
        )

        self.image_center_x = (
            self.screen_width / 2.0
        )

        self.image_center_y = (
            self.screen_height / 2.0
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
        """
        Converte coordenadas YOLOP para coordenadas da tela.

        Não extrapola nem modifica a observação.
        """

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

        screen_x = (
            self.roi_left
            + (x / width) * self.roi_width
        )

        screen_y = (
            self.roi_top
            + (y / height) * self.roi_height
        )

        return (
            float(screen_x),
            float(screen_y),
        )

    def _get_detection_dimensions(
        self,
        detection: LaneDetectionResult,
    ) -> Tuple[float, float]:
        """
        Obtém as dimensões utilizadas pelo YOLOP.

        O LaneDetectionResult atual carrega input_width/input_height.
        Caso algum objeto legado não possua esses atributos,
        utiliza as dimensões configuradas no geometry.
        """

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

        if not np.isfinite(width) or width <= 0.0:
            width = self.detector_width

        if not np.isfinite(height) or height <= 0.0:
            height = self.detector_height

        return width, height

    def _convert_lane(
        self,
        lane: Sequence[LanePoint],
        detector_width: Optional[float] = None,
        detector_height: Optional[float] = None,
    ) -> List[Point]:
        """
        Converte uma lane YOLOP para coordenadas da tela.

        Somente pontos válidos são utilizados.
        """

        points: List[Point] = []

        if lane is None:
            return points

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

        for point in lane:

            if point is None:
                continue

            try:
                valid = bool(point.valid)
            except AttributeError:
                continue

            if not valid:
                continue

            try:
                confidence = float(
                    point.confidence
                )
                x = float(point.x)
                y = float(point.y)
            except (
                AttributeError,
                TypeError,
                ValueError,
            ):
                continue

            if not np.isfinite(confidence):
                continue

            if confidence <= 0.0:
                continue

            if not np.isfinite(x):
                continue

            if not np.isfinite(y):
                continue

            if x < 0.0 or x > width:
                continue

            if y < 0.0 or y > height:
                continue

            sx, sy = self._detector_to_screen(
                x,
                y,
                detector_width=width,
                detector_height=height,
            )

            if not np.isfinite(sx):
                continue

            if not np.isfinite(sy):
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

        return points

    # =========================================================================
    # OUTLIERS
    # =========================================================================

    def _remove_outliers(
        self,
        points: Sequence[Point],
    ) -> List[Point]:
        """
        Remove pontos claramente incompatíveis.

        Utiliza somente uma tendência linear para limpeza.

        Não cria novos pontos.
        """

        if len(points) < 4:
            return list(points)

        arr = np.asarray(
            points,
            dtype=np.float64,
        )

        if arr.ndim != 2:
            return list(points)

        if arr.shape[1] != 2:
            return list(points)

        x = arr[:, 0]
        y = arr[:, 1]

        if not np.all(
            np.isfinite(arr)
        ):
            return list(points)

        if float(np.ptp(y)) < 1.0:
            return list(points)

        y_std = float(
            np.std(y)
        )

        y_norm = (
            (y - np.mean(y))
            / max(y_std, 1.0)
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
            + 4.0 * max(
                mad,
                1.0,
            ),
        )

        keep = (
            residual <= threshold
        )

        filtered = [
            (
                float(arr[index, 0]),
                float(arr[index, 1]),
            )
            for index in range(len(arr))
            if keep[index]
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
        """
        Normaliza uma lane observada.

        Não extrapola e não ajusta curva.
        """

        if len(lane) < self.min_points:
            return []

        clean = self._remove_outliers(
            lane
        )

        if len(clean) < self.min_points:
            return []

        by_y: dict[float, List[float]] = {}

        for x, y in clean:

            if not np.isfinite(x):
                continue

            if not np.isfinite(y):
                continue

            by_y.setdefault(
                float(y),
                [],
            ).append(
                float(x)
            )

        points: List[Point] = []

        for y in sorted(by_y):

            xs = by_y[y]

            if not xs:
                continue

            points.append(
                (
                    float(np.median(xs)),
                    float(y),
                )
            )

        if len(points) < self.min_points:
            return []

        return points

    # =========================================================================
    # INTERPOLAÇÃO
    # =========================================================================

    @staticmethod
    def _x_at_y(
        lane: Sequence[Point],
        y: float,
    ) -> Optional[float]:
        """
        Interpolação x=f(y).

        Nunca extrapola.
        """

        if len(lane) < 2:
            return None

        arr = np.asarray(
            lane,
            dtype=np.float64,
        )

        if arr.ndim != 2 or arr.shape[1] != 2:
            return None

        xs = arr[:, 0]
        ys = arr[:, 1]

        if not np.all(
            np.isfinite(arr)
        ):
            return None

        if np.any(
            np.diff(ys) < 0.0
        ):
            order = np.argsort(ys)
            ys = ys[order]
            xs = xs[order]

        if y < ys[0] or y > ys[-1]:
            return None

        return float(
            np.interp(
                y,
                ys,
                xs,
            )
        )

    # =========================================================================
    # ORDENAÇÃO
    # =========================================================================

    def _sort_lanes(
        self,
        lanes: Sequence[List[Point]],
    ) -> List[List[Point]]:
        """
        Ordena lanes espacialmente da esquerda para direita.
        """

        reference_y = (
            self.roi_bottom
            - self.roi_height * 0.12
        )

        positioned = []

        for index, lane in enumerate(lanes):

            x = self._x_at_y(
                lane,
                reference_y,
            )

            if x is None:

                if not lane:
                    continue

                # Somente fallback para uma observação
                # que realmente existe.
                x = lane[-1][0]

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
    # SELEÇÃO DA FAIXA ATUAL
    # =========================================================================

    def _select_current_lane(
        self,
        lanes: Sequence[List[Point]],
    ) -> Optional[Tuple[int, int]]:
        """
        Seleciona a faixa mais compatível com o centro da tela.

        A decisão usa somente a geometria observada no frame.
        """

        if len(lanes) < 2:
            return None

        reference_y = (
            self.roi_bottom
            - self.roi_height * 0.12
        )

        candidates = []

        for index in range(
            len(lanes) - 1
        ):

            left = lanes[index]
            right = lanes[index + 1]

            left_x = self._x_at_y(
                left,
                reference_y,
            )

            right_x = self._x_at_y(
                right,
                reference_y,
            )

            if (
                left_x is None
                or right_x is None
            ):
                continue

            if right_x <= left_x:
                continue

            width = (
                right_x - left_x
            )

            if not (
                self.min_lane_width
                <= width
                <= self.max_lane_width
            ):
                continue

            center = (
                left_x + right_x
            ) * 0.5

            distance = abs(
                center
                - self.image_center_x
            )

            candidates.append(
                (
                    distance,
                    index,
                    index + 1,
                )
            )

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: item[0]
        )

        _, left_index, right_index = (
            candidates[0]
        )

        return (
            left_index,
            right_index,
        )

    # =========================================================================
    # CENTRO
    # =========================================================================

    def _compute_center_line(
        self,
        left: Sequence[Point],
        right: Sequence[Point],
    ) -> List[Point]:
        """
        Calcula o centro somente na região observada
        simultaneamente pelas duas lanes.
        """

        if (
            len(left) < 2
            or len(right) < 2
        ):
            return []

        left_arr = np.asarray(
            left,
            dtype=np.float64,
        )

        right_arr = np.asarray(
            right,
            dtype=np.float64,
        )

        if (
            left_arr.ndim != 2
            or right_arr.ndim != 2
        ):
            return []

        left_y_min = float(
            np.min(left_arr[:, 1])
        )

        left_y_max = float(
            np.max(left_arr[:, 1])
        )

        right_y_min = float(
            np.min(right_arr[:, 1])
        )

        right_y_max = float(
            np.max(right_arr[:, 1])
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

        if (
            y_max - y_min
            < self.min_observed_span
        ):
            return []

        ys = np.linspace(
            y_min,
            y_max,
            self.samples,
        )

        center: List[Point] = []

        for y in ys:

            left_x = self._x_at_y(
                left,
                float(y),
            )

            right_x = self._x_at_y(
                right,
                float(y),
            )

            if (
                left_x is None
                or right_x is None
            ):
                continue

            if right_x <= left_x:
                continue

            center.append(
                (
                    float(
                        (left_x + right_x)
                        * 0.5
                    ),
                    float(y),
                )
            )

        return center

    # =========================================================================
    # LARGURA
    # =========================================================================

    def _compute_lane_width(
        self,
        left: Sequence[Point],
        right: Sequence[Point],
    ) -> float:
        """
        Calcula a largura média/mediana observada.
        """

        if not left or not right:
            return 0.0

        left_arr = np.asarray(
            left,
            dtype=np.float64,
        )

        right_arr = np.asarray(
            right,
            dtype=np.float64,
        )

        y_min = max(
            float(np.min(left_arr[:, 1])),
            float(np.min(right_arr[:, 1])),
        )

        y_max = min(
            float(np.max(left_arr[:, 1])),
            float(np.max(right_arr[:, 1])),
        )

        if y_max <= y_min:
            return 0.0

        ys = np.linspace(
            y_min,
            y_max,
            20,
        )

        widths = []

        for y in ys:

            left_x = self._x_at_y(
                left,
                float(y),
            )

            right_x = self._x_at_y(
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

            if (
                self.min_lane_width
                <= width
                <= self.max_lane_width
            ):
                widths.append(
                    float(width)
                )

        if not widths:
            return 0.0

        return float(
            np.median(widths)
        )

    # =========================================================================
    # ERRO LATERAL
    # =========================================================================

    def _compute_lateral_error(
        self,
        center_line: Sequence[Point],
    ) -> float:
        """
        Erro lateral normalizado.

        Positivo:
            centro da faixa está à direita
            do centro da tela.

        Negativo:
            centro da faixa está à esquerda.
        """

        if not center_line:
            return 0.0

        reference_y = (
            self.roi_bottom
            - self.roi_height * 0.12
        )

        center_x = self._x_at_y(
            center_line,
            reference_y,
        )

        if center_x is None:
            # Não extrapolar.
            center_x = center_line[-1][0]

        error = (
            center_x
            - self.image_center_x
        )

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
        """
        Calcula o heading observado.

        Não prevê a direção futura.
        """

        if len(center_line) < 5:
            return 0.0

        arr = np.asarray(
            center_line,
            dtype=np.float64,
        )

        if not np.all(
            np.isfinite(arr)
        ):
            return 0.0

        start = max(
            0,
            int(len(arr) * 0.55),
        )

        local = arr[start:]

        if len(local) < 3:
            return 0.0

        x = local[:, 0]
        y = local[:, 1]

        dy = np.diff(y)
        dx = np.diff(x)

        valid = (
            np.abs(dy) > 1e-6
        )

        if not np.any(valid):
            return 0.0

        slopes = (
            dx[valid]
            / dy[valid]
        )

        if not np.all(
            np.isfinite(slopes)
        ):
            return 0.0

        slope = float(
            np.median(slopes)
        )

        angle = np.arctan(
            slope
        )

        max_angle = np.deg2rad(
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
    # CURVATURA OBSERVADA
    # =========================================================================

    def _compute_curvature(
        self,
        center_line: Sequence[Point],
    ) -> float:
        """
        Mede curvatura local observada.

        Não é usada para extrapolação.
        """

        if len(center_line) < 7:
            return 0.0

        arr = np.asarray(
            center_line,
            dtype=np.float64,
        )

        if not np.all(
            np.isfinite(arr)
        ):
            return 0.0

        x = arr[:, 0]
        y = arr[:, 1]

        if float(np.ptp(y)) < 20.0:
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
            1.0 + dx_dy * dx_dy,
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

        if values.size == 0:
            return 0.0

        curvature = float(
            np.median(values)
        )

        return float(
            np.clip(
                curvature * 500.0,
                0.0,
                1.0,
            )
        )

    # =========================================================================
    # CENTRO PONDERADO
    # =========================================================================

    def _compute_weighted_center(
        self,
        center_line: Sequence[Point],
    ) -> Tuple[float, float]:
        """
        Calcula um ponto representativo do centro observado.

        Dá maior peso à região inferior da imagem.
        """

        if not center_line:
            return (
                self.image_center_x,
                self.image_center_y,
            )

        weights = []

        for _, y in center_line:

            relative_y = (
                (y - self.roi_top)
                / max(
                    self.roi_height,
                    1.0,
                )
            )

            relative_y = float(
                np.clip(
                    relative_y,
                    0.0,
                    1.0,
                )
            )

            weight = (
                self.near_weight
                * relative_y
                + self.far_weight
                * (1.0 - relative_y)
            )

            weights.append(
                weight
            )

        weights_np = np.asarray(
            weights,
            dtype=np.float64,
        )

        if (
            not np.all(
                np.isfinite(
                    weights_np
                )
            )
            or np.sum(weights_np) <= 0.0
        ):
            weights_np = np.ones(
                len(center_line),
                dtype=np.float64,
            )

        weights_np /= np.sum(
            weights_np
        )

        center_x = float(
            np.average(
                [
                    point[0]
                    for point in center_line
                ],
                weights=weights_np,
            )
        )

        center_y = float(
            np.average(
                [
                    point[1]
                    for point in center_line
                ],
                weights=weights_np,
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
        """
        Calcula confiança somente da geometria observada.
        """

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

        point_score = min(
            1.0,
            min(
                len(left),
                len(right),
            )
            / float(
                max(
                    self.min_points * 3,
                    1,
                )
            ),
        )

        y_values = [
            point[1]
            for point in center
        ]

        if not y_values:
            return 0.0

        observed_span = (
            max(y_values)
            - min(y_values)
        )

        span_score = float(
            np.clip(
                observed_span
                / max(
                    self.roi_height,
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
            width_score = 1.0 - min(
                1.0,
                abs(
                    lane_width
                    - ideal_width
                )
                / max(
                    width_range * 0.5,
                    1.0,
                ),
            )

        confidence = (
            point_score * 0.30
            + span_score * 0.45
            + width_score * 0.25
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
        left: Optional[List[Point]] = None,
        right: Optional[List[Point]] = None,
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
                left if left is not None else []
            ),
            right_lane_screen=(
                right if right is not None else []
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
    # PIPELINE PRINCIPAL
    # =========================================================================

    def compute(
        self,
        detection: LaneDetectionResult,
    ) -> LaneGeometryResult:
        """
        Processa uma detecção YOLOP.

        A geometria utiliza somente observações do frame atual.
        """

        if detection is None:
            return self._invalid_result()

        # ------------------------------------------------------------------
        # Dimensões reais utilizadas pelo YOLOP neste resultado.
        # ------------------------------------------------------------------

        detector_width, detector_height = (
            self._get_detection_dimensions(
                detection
            )
        )

        raw_lanes: List[List[Point]] = []

        # ------------------------------------------------------------------
        # Lane esquerda
        # ------------------------------------------------------------------

        left = self._prepare_lane(
            self._convert_lane(
                getattr(
                    detection,
                    "left_lane",
                    [],
                ),
                detector_width,
                detector_height,
            )
        )

        if len(left) >= self.min_points:
            raw_lanes.append(left)

        # ------------------------------------------------------------------
        # Lanes adicionais
        # ------------------------------------------------------------------

        additional_lanes = getattr(
            detection,
            "additional_lanes",
            [],
        )

        if additional_lanes is None:
            additional_lanes = []

        for lane in additional_lanes:

            prepared = self._prepare_lane(
                self._convert_lane(
                    lane,
                    detector_width,
                    detector_height,
                )
            )

            if len(prepared) >= self.min_points:
                raw_lanes.append(prepared)

        # ------------------------------------------------------------------
        # Lane direita
        # ------------------------------------------------------------------

        right = self._prepare_lane(
            self._convert_lane(
                getattr(
                    detection,
                    "right_lane",
                    [],
                ),
                detector_width,
                detector_height,
            )
        )

        if len(right) >= self.min_points:
            raw_lanes.append(right)

        # ------------------------------------------------------------------
        # Precisamos de duas linhas observadas.
        # ------------------------------------------------------------------

        if len(raw_lanes) < 2:
            return self._invalid_result()

        # ------------------------------------------------------------------
        # Ordenação espacial.
        # ------------------------------------------------------------------

        ordered_lanes = self._sort_lanes(
            raw_lanes
        )

        if len(ordered_lanes) < 2:
            return self._invalid_result()

        # ------------------------------------------------------------------
        # Seleção da faixa atual.
        # ------------------------------------------------------------------

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

        # ------------------------------------------------------------------
        # Centro observado.
        # ------------------------------------------------------------------

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

        # ------------------------------------------------------------------
        # Largura observada.
        # ------------------------------------------------------------------

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

        # ------------------------------------------------------------------
        # Métricas observadas.
        # ------------------------------------------------------------------

        lateral_error = self._compute_lateral_error(
            center_line
        )

        heading_error = self._compute_heading_error(
            center_line
        )

        curvature = self._compute_curvature(
            center_line
        )

        center_x, center_y = (
            self._compute_weighted_center(
                center_line
            )
        )

        # ------------------------------------------------------------------
        # Extensão observada.
        # ------------------------------------------------------------------

        observed_y_min = float(
            min(
                point[1]
                for point in center_line
            )
        )

        observed_y_max = float(
            max(
                point[1]
                for point in center_line
            )
        )

        observed_span = (
            observed_y_max
            - observed_y_min
        )

        # ------------------------------------------------------------------
        # Confiança.
        # ------------------------------------------------------------------

        geometry_confidence = (
            self._compute_geometry_confidence(
                selected_left,
                selected_right,
                center_line,
                lane_width,
            )
        )

        # ------------------------------------------------------------------
        # Indica somente se há informação suficiente
        # para o LaneProjection trabalhar.
        # ------------------------------------------------------------------

        enough_for_projection = bool(
            geometry_confidence >= 0.55
            and observed_span
            >= self.projection_min_span
            and len(center_line)
            >= self.min_points
        )

        # ------------------------------------------------------------------
        # Lanes não selecionadas.
        # ------------------------------------------------------------------

        remaining_lanes = [
            lane
            for index, lane in enumerate(
                ordered_lanes
            )
            if index not in (
                left_index,
                right_index,
            )
        ]

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
            valid=True,
            left_lane_screen=selected_left,
            right_lane_screen=selected_right,
            additional_lanes_screen=remaining_lanes,
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


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "Point",
    "LaneGeometryResult",
    "LaneGeometry",
]