"""
vision/lane_geometry.py

Forza Horizon ADAS/LKA
Geometria OBSERVADA das faixas.

Responsabilidade deste módulo:

    LaneDetectionResult
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
efetivamente observado pelo detector/tracker.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .yolop_detector import (
    LaneDetectionResult,
    LanePoint,
)

logger = logging.getLogger(__name__)


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

    Todos os pontos representam informação realmente
    observada no frame atual.

    Nenhum ponto deste resultado deve ser interpretado
    como extrapolação futura.
    """

    # ------------------------------------------------------------------
    # Centro observado da faixa atual
    # ------------------------------------------------------------------

    lane_center_x: float
    lane_center_y: float

    # ------------------------------------------------------------------
    # Centro da imagem
    # ------------------------------------------------------------------

    image_center_x: float
    image_center_y: float

    # ------------------------------------------------------------------
    # Métricas
    # ------------------------------------------------------------------

    lateral_error: float
    heading_error: float

    lane_width: float
    curvature: float

    # ------------------------------------------------------------------
    # Linha central observada
    # ------------------------------------------------------------------

    center_line: List[Point]

    # ------------------------------------------------------------------
    # Validade
    # ------------------------------------------------------------------

    valid: bool

    # ------------------------------------------------------------------
    # Limites da faixa atual
    # ------------------------------------------------------------------

    left_lane_screen: List[Point]
    right_lane_screen: List[Point]

    # ------------------------------------------------------------------
    # Outras linhas da pista
    # ------------------------------------------------------------------

    additional_lanes_screen: List[List[Point]]

    # ------------------------------------------------------------------
    # Índices das lanes
    # ------------------------------------------------------------------

    selected_left_index: int
    selected_right_index: int

    # ------------------------------------------------------------------
    # Qualidade da geometria observada
    # ------------------------------------------------------------------

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
    Processador da geometria observada das faixas.

    Este módulo não prevê.

    Pipeline:

        YOLOP / Tracker
              ↓
        conversão para tela
              ↓
        validação
              ↓
        remoção de outliers
              ↓
        ordenação
              ↓
        seleção da faixa atual
              ↓
        centro observado
              ↓
        métricas
              ↓
        qualidade geométrica

    A previsão fica exclusivamente no LaneProjection.
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
    ) -> None:

        self.screen_width = int(screen_width)
        self.screen_height = int(screen_height)

        (
            self.roi_left,
            self.roi_top,
            self.roi_right,
            self.roi_bottom,
        ) = map(int, roi)

        self.roi_width = float(
            self.roi_right - self.roi_left
        )

        self.roi_height = float(
            self.roi_bottom - self.roi_top
        )

        self.detector_width = float(
            detector_width
        )

        self.detector_height = float(
            detector_height
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

        self.min_observed_span = float(
            min_observed_span
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

    # ======================================================================
    # COORDENADAS
    # ======================================================================

    def _detector_to_screen(
        self,
        x: float,
        y: float,
    ) -> Point:
        """
        Converte coordenadas do detector para tela.

        O detector trabalha dentro da ROI.
        """

        screen_x = (
            self.roi_left
            + (
                x
                / self.detector_width
            )
            * self.roi_width
        )

        screen_y = (
            self.roi_top
            + (
                y
                / self.detector_height
            )
            * self.roi_height
        )

        return (
            float(screen_x),
            float(screen_y),
        )

    def _convert_lane(
        self,
        lane: Sequence[LanePoint],
    ) -> List[Point]:
        """
        Converte uma lane para coordenadas da tela.

        Somente pontos válidos entram na geometria.
        """

        points: List[Point] = []

        for point in lane:

            if point is None:
                continue

            if not point.valid:
                continue

            if not np.isfinite(
                point.confidence
            ):
                continue

            if point.confidence <= 0.0:
                continue

            if not np.isfinite(point.x):
                continue

            if not np.isfinite(point.y):
                continue

            if (
                point.x < 0.0
                or point.x > self.detector_width
            ):
                continue

            if (
                point.y < 0.0
                or point.y > self.detector_height
            ):
                continue

            sx, sy = (
                self._detector_to_screen(
                    point.x,
                    point.y,
                )
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
            key=lambda p: p[1]
        )

        return points

    # ======================================================================
    # OUTLIERS
    # ======================================================================

    def _remove_outliers(
        self,
        points: Sequence[Point],
    ) -> List[Point]:
        """
        Remove pontos claramente incompatíveis
        com a tendência local da lane.

        IMPORTANTE:

        Isto não cria uma curva.

        O objetivo é apenas impedir que um ponto
        completamente absurdo contamine a geometria.
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

        y_range = float(
            np.ptp(y)
        )

        if y_range < 1.0:
            return list(points)

        # --------------------------------------------------------------
        # Para remoção de outliers usamos apenas uma tendência
        # linear local.
        #
        # Não usamos grau 2/3 aqui porque isso começaria a misturar
        # limpeza com modelagem/predição.
        # --------------------------------------------------------------

        y_norm = (
            y - np.mean(y)
        ) / max(
            np.std(y),
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
                    residual
                    - median
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
            residual
            <= threshold
        )

        filtered = [
            (
                float(arr[i, 0]),
                float(arr[i, 1]),
            )
            for i in range(
                len(arr)
            )
            if keep[i]
        ]

        if len(filtered) < self.min_points:
            return list(points)

        return filtered

    # ======================================================================
    # NORMALIZAÇÃO DOS PONTOS
    # ======================================================================

    def _prepare_lane(
        self,
        lane: Sequence[Point],
    ) -> List[Point]:
        """
        Prepara uma lane observada.

        Não extrapola.
        Não ajusta curva.
        """

        if len(lane) < self.min_points:
            return []

        clean = self._remove_outliers(
            lane
        )

        if len(clean) < self.min_points:
            return []

        # --------------------------------------------------------------
        # Remove Y duplicado.
        #
        # Como a geometria é x=f(y), cada Y deve aparecer apenas uma
        # vez.
        # --------------------------------------------------------------

        by_y = {}

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

            x = float(
                np.median(xs)
            )

            points.append(
                (
                    x,
                    float(y),
                )
            )

        if len(points) < self.min_points:
            return []

        return points

    # ======================================================================
    # INTERPOLAÇÃO OBSERVADA
    # ======================================================================

    @staticmethod
    def _x_at_y(
        lane: Sequence[Point],
        y: float,
    ) -> Optional[float]:
        """
        Obtém X por interpolação.

        IMPORTANTE:

        A função NÃO extrapola.

        Se Y estiver fora da região observada,
        retorna None.
        """

        if len(lane) < 2:
            return None

        arr = np.asarray(
            lane,
            dtype=np.float64,
        )

        if arr.ndim != 2:
            return None

        ys = arr[:, 1]
        xs = arr[:, 0]

        if not np.all(
            np.isfinite(ys)
        ):
            return None

        if not np.all(
            np.isfinite(xs)
        ):
            return None

        if y < ys[0] or y > ys[-1]:
            return None

        return float(
            np.interp(
                y,
                ys,
                xs,
            )
        )

    # ======================================================================
    # ORDENAÇÃO DAS LANES
    # ======================================================================

    def _sort_lanes(
        self,
        lanes: Sequence[List[Point]],
    ) -> List[List[Point]]:
        """
        Ordena todas as lanes da esquerda para direita.

        A ordenação é feita em uma região próxima ao veículo.
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
                # Se não houver observação nesse Y,
                # usamos o ponto mais próximo do fundo.
                if not lane:
                    continue

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

    # ======================================================================
    # SELEÇÃO DA FAIXA ATUAL
    # ======================================================================

    def _select_current_lane(
        self,
        lanes: Sequence[List[Point]],
    ) -> Optional[Tuple[int, int]]:
        """
        Seleciona a faixa onde o veículo provavelmente está.

        Consideramos no máximo a informação geométrica
        observada neste frame.

        Procuramos duas linhas consecutivas cuja região
        central esteja próxima do centro da imagem.
        """

        if len(lanes) < 2:
            return None

        reference_y = (
            self.roi_bottom
            - self.roi_height * 0.12
        )

        candidates = []

        for i in range(
            len(lanes) - 1
        ):

            left = lanes[i]
            right = lanes[i + 1]

            lx = self._x_at_y(
                left,
                reference_y,
            )

            rx = self._x_at_y(
                right,
                reference_y,
            )

            if lx is None or rx is None:
                continue

            if rx <= lx:
                continue

            width = rx - lx

            if not (
                self.min_lane_width
                <= width
                <= self.max_lane_width
            ):
                continue

            center = (
                lx + rx
            ) * 0.5

            distance = abs(
                center
                - self.image_center_x
            )

            candidates.append(
                (
                    distance,
                    i,
                    i + 1,
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

    # ======================================================================
    # CENTRO OBSERVADO
    # ======================================================================

    def _compute_center_line(
        self,
        left: Sequence[Point],
        right: Sequence[Point],
    ) -> List[Point]:
        """
        Calcula o centro entre duas lanes observadas.

        Somente utiliza a região em que ambas as linhas
        realmente existem.
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

        left_y_min = float(
            left_arr[:, 1].min()
        )

        left_y_max = float(
            left_arr[:, 1].max()
        )

        right_y_min = float(
            right_arr[:, 1].min()
        )

        right_y_max = float(
            right_arr[:, 1].max()
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

            lx = self._x_at_y(
                left,
                float(y),
            )

            rx = self._x_at_y(
                right,
                float(y),
            )

            if lx is None or rx is None:
                continue

            if rx <= lx:
                continue

            center.append(
                (
                    float(
                        (lx + rx) * 0.5
                    ),
                    float(y),
                )
            )

        return center

    # ======================================================================
    # LARGURA
    # ======================================================================

    def _compute_lane_width(
        self,
        left: Sequence[Point],
        right: Sequence[Point],
    ) -> float:
        """
        Calcula a largura observada da faixa.

        Não extrapola.
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
            float(left_arr[:, 1].min()),
            float(right_arr[:, 1].min()),
        )

        y_max = min(
            float(left_arr[:, 1].max()),
            float(right_arr[:, 1].max()),
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

            lx = self._x_at_y(
                left,
                float(y),
            )

            rx = self._x_at_y(
                right,
                float(y),
            )

            if lx is None or rx is None:
                continue

            width = rx - lx

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

    # ======================================================================
    # ERRO LATERAL
    # ======================================================================

    def _compute_lateral_error(
        self,
        center_line: Sequence[Point],
    ) -> float:
        """
        Erro lateral normalizado.

        A referência é uma região próxima ao veículo.
        """

        if not center_line:
            return 0.0

        reference_y = (
            self.roi_bottom
            - self.roi_height * 0.12
        )

        x = self._x_at_y(
            center_line,
            reference_y,
        )

        if x is None:
            # Não extrapolar.
            # Utilizamos somente o ponto observado
            # mais próximo do fundo.
            x = center_line[-1][0]

        error = (
            x
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

    # ======================================================================
    # HEADING OBSERVADO
    # ======================================================================

    def _compute_heading_error(
        self,
        center_line: Sequence[Point],
    ) -> float:
        """
        Estima o heading observado da faixa.

        Não tenta prever a curva.

        Utiliza apenas a inclinação local
        da linha central observada.
        """

        if len(center_line) < 5:
            return 0.0

        arr = np.asarray(
            center_line,
            dtype=np.float64,
        )

        # Utilizamos a parte inferior da geometria,
        # onde a informação está mais próxima do veículo.
        start = max(
            0,
            int(
                len(arr) * 0.55
            ),
        )

        local = arr[start:]

        if len(local) < 3:
            return 0.0

        x = local[:, 0]
        y = local[:, 1]

        if not np.all(
            np.isfinite(x)
        ):
            return 0.0

        if not np.all(
            np.isfinite(y)
        ):
            return 0.0

        try:

            dx = np.diff(x)
            dy = np.diff(y)

            valid = (
                np.abs(dy)
                > 1e-6
            )

            if not np.any(valid):
                return 0.0

            slopes = (
                dx[valid]
                / dy[valid]
            )

            slope = float(
                np.median(slopes)
            )

        except (
            ValueError,
            FloatingPointError,
        ):
            return 0.0

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

    # ======================================================================
    # CURVATURA OBSERVADA
    # ======================================================================

    def _compute_curvature(
        self,
        center_line: Sequence[Point],
    ) -> float:
        """
        Mede a curvatura LOCAL observada.

        Atenção:

        Este valor não é utilizado para projetar
        a faixa.

        A projeção futura pertence ao LaneProjection.
        """

        if len(center_line) < 7:
            return 0.0

        arr = np.asarray(
            center_line,
            dtype=np.float64,
        )

        x = arr[:, 0]
        y = arr[:, 1]

        if not np.all(
            np.isfinite(x)
        ):
            return 0.0

        if not np.all(
            np.isfinite(y)
        ):
            return 0.0

        y_range = float(
            np.ptp(y)
        )

        if y_range < 20.0:
            return 0.0

        # Derivadas numéricas locais.
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

        curvature_values = (
            np.abs(
                d2x_dy2[valid]
            )
            / denominator[valid]
        )

        if curvature_values.size == 0:
            return 0.0

        curvature = float(
            np.median(
                curvature_values
            )
        )

        # Normalização conservadora.
        normalized = (
            curvature * 500.0
        )

        return float(
            np.clip(
                normalized,
                0.0,
                1.0,
            )
        )

    # ======================================================================
    # CENTRO PONDERADO
    # ======================================================================

    def _compute_weighted_center(
        self,
        center_line: Sequence[Point],
    ) -> Tuple[float, float]:
        """
        Calcula posição representativa do centro observado.

        Dá maior peso à região próxima ao veículo.
        """

        if not center_line:
            return (
                self.image_center_x,
                self.image_center_y,
            )

        weights = []

        for _, y in center_line:

            relative_y = (
                y - self.roi_top
            ) / max(
                self.roi_height,
                1.0,
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
            or np.sum(weights_np)
            <= 0.0
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
                    p[0]
                    for p
                    in center_line
                ],
                weights=weights_np,
            )
        )

        center_y = float(
            np.average(
                [
                    p[1]
                    for p
                    in center_line
                ],
                weights=weights_np,
            )
        )

        return (
            center_x,
            center_y,
        )

    # ======================================================================
    # QUALIDADE
    # ======================================================================

    def _compute_geometry_confidence(
        self,
        left: Sequence[Point],
        right: Sequence[Point],
        center: Sequence[Point],
        lane_width: float,
    ) -> float:
        """
        Calcula confiança da geometria OBSERVADA.

        Não representa confiança da previsão.
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

        # --------------------------------------------------------------
        # Quantidade de pontos.
        # --------------------------------------------------------------

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

        # --------------------------------------------------------------
        # Extensão observada.
        # --------------------------------------------------------------

        y_values = [
            p[1]
            for p in center
        ]

        observed_span = (
            max(y_values)
            - min(y_values)
            if y_values
            else 0.0
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

        # --------------------------------------------------------------
        # Largura plausível.
        # --------------------------------------------------------------

        ideal_width = (
            self.min_lane_width
            + self.max_lane_width
        ) * 0.5

        width_range = (
            self.max_lane_width
            - self.min_lane_width
        )

        if width_range <= 0:
            width_score = 0.0
        else:
            width_score = 1.0 - min(
                1.0,
                abs(
                    lane_width
                    - ideal_width
                )
                / (
                    width_range
                    * 0.5
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

    # ======================================================================
    # RESULTADO INVÁLIDO
    # ======================================================================

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
            left_lane_screen=left or [],
            right_lane_screen=right or [],
            additional_lanes_screen=(
                additional_lanes or []
            ),
            selected_left_index=-1,
            selected_right_index=-1,
            geometry_confidence=0.0,
            observed_y_min=0.0,
            observed_y_max=0.0,
            observed_span=0.0,
            enough_for_projection=False,
        )

    # ======================================================================
    # PIPELINE PRINCIPAL
    # ======================================================================

    def compute(
        self,
        detection: LaneDetectionResult,
    ) -> LaneGeometryResult:
        """
        Processa exclusivamente a geometria observada.

        Nenhuma extrapolação é realizada.
        """

        if detection is None:
            return self._invalid_result()

        raw_lanes: List[List[Point]] = []

        # --------------------------------------------------------------
        # Lane esquerda
        # --------------------------------------------------------------

        left = self._convert_lane(
            detection.left_lane
        )

        left = self._prepare_lane(
            left
        )

        if len(left) >= self.min_points:
            raw_lanes.append(left)

        # --------------------------------------------------------------
        # Lanes adicionais
        # --------------------------------------------------------------

        for lane in (
            detection.additional_lanes
        ):

            converted = (
                self._convert_lane(
                    lane
                )
            )

            prepared = (
                self._prepare_lane(
                    converted
                )
            )

            if (
                len(prepared)
                >= self.min_points
            ):
                raw_lanes.append(
                    prepared
                )

        # --------------------------------------------------------------
        # Lane direita
        # --------------------------------------------------------------

        right = self._convert_lane(
            detection.right_lane
        )

        right = self._prepare_lane(
            right
        )

        if len(right) >= self.min_points:
            raw_lanes.append(right)

        # --------------------------------------------------------------
        # Precisamos de pelo menos duas linhas observadas.
        # --------------------------------------------------------------

        if len(raw_lanes) < 2:
            return self._invalid_result()

        # --------------------------------------------------------------
        # Ordenação esquerda → direita.
        # --------------------------------------------------------------

        ordered_lanes = (
            self._sort_lanes(
                raw_lanes
            )
        )

        if len(ordered_lanes) < 2:
            return self._invalid_result()

        # --------------------------------------------------------------
        # Seleção da faixa atual.
        # --------------------------------------------------------------

        selected = (
            self._select_current_lane(
                ordered_lanes
            )
        )

        if selected is None:

            return self._invalid_result(
                additional_lanes=ordered_lanes
            )

        left_index, right_index = (
            selected
        )

        selected_left = (
            ordered_lanes[
                left_index
            ]
        )

        selected_right = (
            ordered_lanes[
                right_index
            ]
        )

        # --------------------------------------------------------------
        # Centro observado.
        # --------------------------------------------------------------

        center_line = (
            self._compute_center_line(
                selected_left,
                selected_right,
            )
        )

        if (
            len(center_line)
            < self.min_points
        ):

            return self._invalid_result(
                additional_lanes=ordered_lanes,
                left=selected_left,
                right=selected_right,
            )

        # --------------------------------------------------------------
        # Largura observada.
        # --------------------------------------------------------------

        lane_width = (
            self._compute_lane_width(
                selected_left,
                selected_right,
            )
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

        # --------------------------------------------------------------
        # Métricas observadas.
        # --------------------------------------------------------------

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

        # --------------------------------------------------------------
        # Extensão observada.
        # --------------------------------------------------------------

        observed_y_min = float(
            min(
                p[1]
                for p in center_line
            )
        )

        observed_y_max = float(
            max(
                p[1]
                for p in center_line
            )
        )

        observed_span = (
            observed_y_max
            - observed_y_min
        )

        # --------------------------------------------------------------
        # Qualidade.
        # --------------------------------------------------------------

        geometry_confidence = (
            self._compute_geometry_confidence(
                selected_left,
                selected_right,
                center_line,
                lane_width,
            )
        )

        # --------------------------------------------------------------
        # A geometria pode fornecer dados suficientes
        # para o futuro LaneProjection?
        #
        # Isto NÃO significa que a projeção já foi feita.
        # Apenas informamos que existe matéria-prima suficiente.
        # --------------------------------------------------------------

        enough_for_projection = bool(
            geometry_confidence >= 0.55
            and observed_span
            >= self.projection_min_span
            and len(center_line)
            >= self.min_points
        )

        # --------------------------------------------------------------
        # Lanes restantes.
        # --------------------------------------------------------------

        additional = [
            lane
            for index, lane
            in enumerate(
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


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "Point",
    "LaneGeometryResult",
    "LaneGeometry",
]