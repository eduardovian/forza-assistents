"""
Forza Horizon ADAS/LKA
Geometria robusta das faixas.

Objetivos:
- transformar pontos UFLD em coordenadas da tela;
- remover outliers;
- ajustar curvas suaves às marcações;
- evitar linhas em "cobra";
- selecionar a faixa onde o veículo está;
- calcular centro da faixa;
- calcular erro lateral;
- calcular heading;
- calcular largura;
- fornecer pontos estáveis para o overlay.

IMPORTANTE:
Este módulo não cria uma faixa a partir de pontos inválidos.
Quando a geometria não é confiável, valid=False.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .ufld_detector import LanePoint, LaneDetectionResult


logger = logging.getLogger(__name__)


Point = Tuple[float, float]


@dataclass
class LaneGeometryResult:
    """Resultado geométrico da detecção."""

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

    # Linhas adicionais detectadas.
    additional_lanes_screen: List[List[Point]]

    # Índices das linhas utilizadas como limites da faixa atual.
    selected_left_index: int
    selected_right_index: int


class LaneGeometry:
    """
    Processador robusto da geometria das faixas.

    A ideia é NÃO ligar diretamente os pontos do detector.

    Pipeline:

        UFLD points
            ↓
        conversão para tela
            ↓
        remoção de outliers
            ↓
        ordenação por Y
            ↓
        ajuste polinomial
            ↓
        amostragem uniforme
            ↓
        validação geométrica
            ↓
        seleção da faixa atual
            ↓
        centro da faixa
    """

    def __init__(
        self,
        screen_width: int = 2560,
        screen_height: int = 1600,
        roi: Tuple[int, int, int, int] = (300, 700, 2200, 1600),
        ufld_width: int = 800,
        ufld_height: int = 288,
        near_weight: float = 0.75,
        far_weight: float = 0.25,
        min_points: int = 5,
        polynomial_degree: int = 2,
        samples: int = 60,
        min_lane_width: float = 180.0,
        max_lane_width: float = 1400.0,
        max_slope: float = 2.5,
    ):
        self.screen_width = int(screen_width)
        self.screen_height = int(screen_height)

        (
            self.roi_left,
            self.roi_top,
            self.roi_right,
            self.roi_bottom,
        ) = roi

        self.roi_width = float(self.roi_right - self.roi_left)
        self.roi_height = float(self.roi_bottom - self.roi_top)

        self.ufld_width = float(ufld_width)
        self.ufld_height = float(ufld_height)

        self.near_weight = float(near_weight)
        self.far_weight = float(far_weight)

        self.min_points = int(min_points)
        self.polynomial_degree = int(polynomial_degree)
        self.samples = int(samples)

        self.min_lane_width = float(min_lane_width)
        self.max_lane_width = float(max_lane_width)

        self.max_slope = float(max_slope)

        self.image_center_x = self.screen_width / 2.0
        self.image_center_y = self.screen_height / 2.0

    # ==================================================================
    # COORDENADAS
    # ==================================================================

    def _ufld_to_screen(
        self,
        x: float,
        y: float,
    ) -> Point:
        """Converte coordenadas UFLD para coordenadas da tela."""

        screen_x = (
            self.roi_left
            + (x / self.ufld_width) * self.roi_width
        )

        screen_y = (
            self.roi_top
            + (y / self.ufld_height) * self.roi_height
        )

        return float(screen_x), float(screen_y)

    def _convert_lane(
        self,
        lane: Sequence[LanePoint],
    ) -> List[Point]:
        """
        Converte uma lane para coordenadas da tela.

        Somente pontos realmente válidos entram aqui.
        """

        points: List[Point] = []

        for point in lane:

            if not point.valid:
                continue

            if point.confidence <= 0.0:
                continue

            if not np.isfinite(point.x):
                continue

            if not np.isfinite(point.y):
                continue

            if point.x < 0.0 or point.x > self.ufld_width:
                continue

            if point.y < 0.0 or point.y > self.ufld_height:
                continue

            sx, sy = self._ufld_to_screen(
                point.x,
                point.y,
            )

            if not np.isfinite(sx) or not np.isfinite(sy):
                continue

            points.append((sx, sy))

        # Ordenação obrigatória: topo → baixo.
        points.sort(key=lambda p: p[1])

        return points

    # ==================================================================
    # OUTLIERS
    # ==================================================================

    def _remove_outliers(
        self,
        points: Sequence[Point],
    ) -> List[Point]:
        """
        Remove pontos geometricamente absurdos.

        O UFLD trabalha em anchors discretos.
        Um único erro pode produzir um salto enorme em X.

        Em vez de conectar esse salto diretamente, estimamos
        a tendência geral da lane e removemos pontos muito
        distantes dela.
        """

        if len(points) < 4:
            return list(points)

        arr = np.asarray(points, dtype=np.float64)

        x = arr[:, 0]
        y = arr[:, 1]

        # Normaliza Y para melhorar estabilidade numérica.
        y_norm = (y - np.mean(y)) / max(
            np.std(y),
            1.0,
        )

        try:
            coeff = np.polyfit(
                y_norm,
                x,
                min(2, len(points) - 1),
            )

            predicted = np.polyval(
                coeff,
                y_norm,
            )

            residual = np.abs(x - predicted)

            median = np.median(residual)

            mad = np.median(
                np.abs(residual - median)
            )

            # MAD muito pequeno significa que a curva está muito limpa.
            threshold = max(
                80.0,
                median + 4.0 * max(mad, 1.0),
            )

            keep = residual <= threshold

            filtered = [
                tuple(arr[i])
                for i in range(len(arr))
                if keep[i]
            ]

            if len(filtered) >= self.min_points:
                return filtered

        except (np.linalg.LinAlgError, ValueError):
            pass

        return list(points)

    # ==================================================================
    # CURVA
    # ==================================================================

    def _fit_lane(
        self,
        points: Sequence[Point],
    ) -> List[Point]:
        """
        Ajusta uma curva suave x=f(y).

        Esta é a principal correção da "cobra".

        Não desenhamos:

            P1 → P2 → P3 → P4

        diretamente.

        Ajustamos:

            x = f(y)

        e depois amostramos essa curva.
        """

        if len(points) < self.min_points:
            return []

        clean = self._remove_outliers(points)

        if len(clean) < self.min_points:
            return []

        arr = np.asarray(clean, dtype=np.float64)

        x = arr[:, 0]
        y = arr[:, 1]

        # Remove duplicatas de Y.
        unique_y, unique_indices = np.unique(
            y,
            return_index=True,
        )

        x = x[unique_indices]
        y = unique_y

        if len(x) < self.min_points:
            return []

        # Normalização.
        y_center = float(np.mean(y))
        y_scale = max(
            float(np.ptp(y)),
            1.0,
        )

        y_norm = (y - y_center) / y_scale

        degree = min(
            self.polynomial_degree,
            len(x) - 1,
        )

        try:
            coeff = np.polyfit(
                y_norm,
                x,
                degree,
            )
        except (np.linalg.LinAlgError, ValueError):
            return []

        # Amostragem uniforme.
        y_min = float(np.min(y))
        y_max = float(np.max(y))

        if y_max - y_min < 20.0:
            return []

        sample_y = np.linspace(
            y_min,
            y_max,
            self.samples,
        )

        sample_norm = (
            sample_y - y_center
        ) / y_scale

        sample_x = np.polyval(
            coeff,
            sample_norm,
        )

        # ==============================================================
        # LIMITAÇÃO DE SLOPE
        # ==============================================================

        # Evita uma curva que faça uma virada impossível.
        dx_dy = np.gradient(
            sample_x,
            sample_y,
        )

        dx_dy = np.clip(
            dx_dy,
            -self.max_slope,
            self.max_slope,
        )

        # Reintegra a curva usando a derivada limitada.
        smooth_x = np.empty_like(sample_x)
        smooth_x[0] = sample_x[0]

        for i in range(1, len(sample_x)):
            dy = sample_y[i] - sample_y[i - 1]

            smooth_x[i] = (
                smooth_x[i - 1]
                + dx_dy[i] * dy
            )

        # Pequena correção para manter a curva próxima
        # da regressão original.
        offset = (
            np.mean(sample_x)
            - np.mean(smooth_x)
        )

        smooth_x += offset

        # Limites físicos da ROI.
        smooth_x = np.clip(
            smooth_x,
            self.roi_left,
            self.roi_right,
        )

        result = [
            (
                float(px),
                float(py),
            )
            for px, py in zip(
                smooth_x,
                sample_y,
            )
        ]

        return result

    # ==================================================================
    # INTERPOLAÇÃO
    # ==================================================================

    @staticmethod
    def _x_at_y(
        lane: Sequence[Point],
        y: float,
    ) -> Optional[float]:
        """Obtém X de uma lane para determinado Y."""

        if len(lane) < 2:
            return None

        arr = np.asarray(lane, dtype=np.float64)

        ys = arr[:, 1]
        xs = arr[:, 0]

        if y < ys[0] or y > ys[-1]:
            return None

        return float(
            np.interp(
                y,
                ys,
                xs,
            )
        )

    # ==================================================================
    # CENTRO
    # ==================================================================

    def _compute_center_line(
        self,
        left: Sequence[Point],
        right: Sequence[Point],
    ) -> List[Point]:
        """Calcula uma linha central suave entre duas lanes."""

        if len(left) < 2 or len(right) < 2:
            return []

        left_arr = np.asarray(left)
        right_arr = np.asarray(right)

        y_min = max(
            float(left_arr[:, 1].min()),
            float(right_arr[:, 1].min()),
        )

        y_max = min(
            float(left_arr[:, 1].max()),
            float(right_arr[:, 1].max()),
        )

        if y_max <= y_min:
            return []

        ys = np.linspace(
            y_min,
            y_max,
            self.samples,
        )

        center: List[Point] = []

        for y in ys:

            lx = self._x_at_y(left, float(y))
            rx = self._x_at_y(right, float(y))

            if lx is None or rx is None:
                continue

            if rx <= lx:
                continue

            cx = (lx + rx) * 0.5

            center.append(
                (
                    float(cx),
                    float(y),
                )
            )

        return center

    # ==================================================================
    # LARGURA
    # ==================================================================

    def _compute_lane_width(
        self,
        left: Sequence[Point],
        right: Sequence[Point],
    ) -> float:
        """Calcula largura média da faixa."""

        if not left or not right:
            return 0.0

        left_arr = np.asarray(left)
        right_arr = np.asarray(right)

        y_min = max(
            left_arr[:, 1].min(),
            right_arr[:, 1].min(),
        )

        y_max = min(
            left_arr[:, 1].max(),
            right_arr[:, 1].max(),
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

            lx = self._x_at_y(left, float(y))
            rx = self._x_at_y(right, float(y))

            if lx is None or rx is None:
                continue

            width = rx - lx

            if (
                self.min_lane_width
                <= width
                <= self.max_lane_width
            ):
                widths.append(width)

        if not widths:
            return 0.0

        return float(np.median(widths))

    # ==================================================================
    # SELEÇÃO DA FAIXA
    # ==================================================================

    def _select_current_lane(
        self,
        lanes: Sequence[List[Point]],
    ) -> Optional[Tuple[int, int]]:
        """
        Seleciona a faixa onde o veículo está.

        Temos potencialmente:

            lane 0 | lane 1 | lane 2 | lane 3

        O veículo está aproximadamente no centro da tela.

        Procuramos duas linhas consecutivas que:
        - estejam em lados opostos do centro;
        - tenham largura plausível;
        - sejam próximas do centro da imagem.
        """

        if len(lanes) < 2:
            return None

        candidates = []

        reference_y = self.roi_bottom - (
            self.roi_height * 0.12
        )

        for i in range(len(lanes) - 1):

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

            center = (lx + rx) * 0.5

            distance = abs(
                center - self.image_center_x
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

        _, left_index, right_index = candidates[0]

        return left_index, right_index

    # ==================================================================
    # MÉTRICAS
    # ==================================================================

    def _compute_lateral_error(
        self,
        center_line: Sequence[Point],
    ) -> float:
        """Erro lateral normalizado."""

        if not center_line:
            return 0.0

        # Posição próxima ao veículo.
        y_reference = (
            self.roi_bottom
            - self.roi_height * 0.12
        )

        x = self._x_at_y(
            center_line,
            y_reference,
        )

        if x is None:
            x = center_line[-1][0]

        error = (
            x - self.image_center_x
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

    def _compute_heading_error(
        self,
        center_line: Sequence[Point],
    ) -> float:
        """
        Heading baseado na tangente da curva.

        Usa uma janela próxima ao veículo,
        evitando que a curvatura distante domine.
        """

        if len(center_line) < 5:
            return 0.0

        arr = np.asarray(
            center_line,
            dtype=np.float64,
        )

        # Usa aproximadamente os 35% inferiores.
        start = max(
            0,
            int(len(arr) * 0.55),
        )

        local = arr[start:]

        if len(local) < 3:
            return 0.0

        x = local[:, 0]
        y = local[:, 1]

        try:
            coeff = np.polyfit(
                y,
                x,
                1,
            )
        except np.linalg.LinAlgError:
            return 0.0

        slope = float(coeff[0])

        # A inclinação x/y representa a direção
        # horizontal da estrada.
        angle = np.arctan(slope)

        max_angle = np.deg2rad(45.0)

        return float(
            np.clip(
                angle / max_angle,
                -1.0,
                1.0,
            )
        )

    def _compute_curvature(
        self,
        center_line: Sequence[Point],
    ) -> float:
        """Curvatura normalizada."""

        if len(center_line) < 6:
            return 0.0

        arr = np.asarray(
            center_line,
            dtype=np.float64,
        )

        y = arr[:, 1]
        x = arr[:, 0]

        y_center = np.mean(y)
        y_scale = max(np.ptp(y), 1.0)

        yn = (y - y_center) / y_scale

        try:
            coeff = np.polyfit(
                yn,
                x,
                2,
            )
        except np.linalg.LinAlgError:
            return 0.0

        curvature = abs(float(coeff[0]))

        # Normalização conservadora.
        return float(
            np.clip(
                curvature * 2.0,
                0.0,
                1.0,
            )
        )

    # ==================================================================
    # RESULTADO INVÁLIDO
    # ==================================================================

    def _invalid_result(
        self,
        additional_lanes: Optional[List[List[Point]]] = None,
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
            left_lane_screen=left or [],
            right_lane_screen=right or [],
            additional_lanes_screen=additional_lanes or [],
            selected_left_index=-1,
            selected_right_index=-1,
        )

    # ==================================================================
    # PIPELINE PRINCIPAL
    # ==================================================================

    def compute(
        self,
        detection: LaneDetectionResult,
    ) -> LaneGeometryResult:
        """
        Executa todo o processamento geométrico.
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

        if len(left) >= self.min_points:
            raw_lanes.append(left)

        # --------------------------------------------------------------
        # Lanes adicionais
        # --------------------------------------------------------------

        for lane in detection.additional_lanes:

            converted = self._convert_lane(
                lane
            )

            if len(converted) >= self.min_points:
                raw_lanes.append(converted)

        # --------------------------------------------------------------
        # Lane direita
        # --------------------------------------------------------------

        right = self._convert_lane(
            detection.right_lane
        )

        if len(right) >= self.min_points:
            raw_lanes.append(right)

        if len(raw_lanes) < 2:
            return self._invalid_result()

        # --------------------------------------------------------------
        # Ajusta todas as lanes
        # --------------------------------------------------------------

        fitted_lanes: List[List[Point]] = []

        for lane in raw_lanes:

            fitted = self._fit_lane(lane)

            if len(fitted) >= self.min_points:
                fitted_lanes.append(fitted)

        if len(fitted_lanes) < 2:
            return self._invalid_result()

        # --------------------------------------------------------------
        # Ordena da esquerda para direita
        # --------------------------------------------------------------

        reference_y = self.roi_bottom - (
            self.roi_height * 0.12
        )

        lane_positions = []

        for index, lane in enumerate(fitted_lanes):

            x = self._x_at_y(
                lane,
                reference_y,
            )

            if x is not None:
                lane_positions.append(
                    (x, index)
                )

        if len(lane_positions) < 2:
            return self._invalid_result()

        lane_positions.sort(
            key=lambda item: item[0]
        )

        ordered_lanes = [
            fitted_lanes[index]
            for _, index in lane_positions
        ]

        # --------------------------------------------------------------
        # Seleciona a faixa atual
        # --------------------------------------------------------------

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

        # --------------------------------------------------------------
        # Centro
        # --------------------------------------------------------------

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

        # --------------------------------------------------------------
        # Largura
        # --------------------------------------------------------------

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

        # --------------------------------------------------------------
        # Métricas
        # --------------------------------------------------------------

        lateral_error = self._compute_lateral_error(
            center_line
        )

        heading_error = self._compute_heading_error(
            center_line
        )

        curvature = self._compute_curvature(
            center_line
        )

        # --------------------------------------------------------------
        # Centro ponderado
        # --------------------------------------------------------------

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

            weights.append(weight)

        weights_np = np.asarray(
            weights,
            dtype=np.float64,
        )

        if np.sum(weights_np) <= 0:
            weights_np = np.ones_like(
                weights_np
            )

        weights_np /= np.sum(weights_np)

        center_x = float(
            np.average(
                [p[0] for p in center_line],
                weights=weights_np,
            )
        )

        center_y = float(
            np.average(
                [p[1] for p in center_line],
                weights=weights_np,
            )
        )

        # --------------------------------------------------------------
        # Lanes restantes
        # --------------------------------------------------------------

        additional = [
            lane
            for index, lane
            in enumerate(ordered_lanes)
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
        )