"""
core/lane_selector.py

Identificação da faixa ocupada pelo veículo.

Responsabilidade
----------------
Receber múltiplas lanes detectadas/projetadas e determinar:

    - quais linhas formam cada faixa;
    - qual faixa é a faixa do veículo;
    - quais são as faixas laterais;
    - centro geométrico da faixa ocupada;
    - posição relativa do veículo dentro da faixa.

Este módulo NÃO:

    - faz inferência YOLOP;
    - rastreia temporalmente as lanes;
    - ajusta polinômios;
    - projeta lanes;
    - controla o volante;
    - decide intervenção ADAS.

A decisão de intervenção ficará nos módulos superiores.

Modelo
------
Para N linhas detectadas:

    L0   L1   L2   L3
     |    |    |    |
     ↓    ↓    ↓    ↓

    faixa 0 = L0 <-> L1
    faixa 1 = L1 <-> L2
    faixa 2 = L2 <-> L3

A faixa ocupada é determinada comparando a posição
horizontal estimada do veículo com os centros das faixas.

Importante
----------
A posição do veículo NÃO é assumida como sendo exatamente
o centro da imagem.

Ela será fornecida posteriormente pelo módulo de calibração
da câmera/ego-position.

Por enquanto, o centro óptico da imagem é utilizado como
referência configurável.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .lane_projection import (
    LaneProjectionResult,
    ProjectedLanePoint,
)


logger = logging.getLogger(__name__)


# ============================================================================
# ENUMERAÇÕES
# ============================================================================


class LaneSelectionStatus(str, Enum):
    INVALID = "invalid"
    VALID = "valid"
    AMBIGUOUS = "ambiguous"


# ============================================================================
# CONFIGURAÇÃO
# ============================================================================


@dataclass(frozen=True)
class LaneSelectorConfig:
    """
    Configuração do identificador de faixa.
    """

    # Y utilizado para avaliar a posição da faixa.
    evaluation_y_ratio: float = 0.88

    # Mínimo de largura da faixa, em relação à imagem.
    min_lane_width_ratio: float = 0.035

    # Máximo de largura plausível da faixa.
    max_lane_width_ratio: float = 0.75

    # Diferença máxima entre larguras em regiões diferentes.
    max_width_variation_ratio: float = 0.65

    # Distância máxima relativa para considerar duas linhas
    # como uma faixa plausível.
    max_center_jump_ratio: float = 0.40

    # Número mínimo de pontos para uma linha ser utilizada.
    min_points: int = 5

    # Confiança mínima da linha.
    min_confidence: float = 0.20

    # Quando a posição do veículo está muito próxima da fronteira
    # entre duas faixas, o resultado passa a ser ambíguo.
    boundary_ambiguity_ratio: float = 0.08

    # Tolerância para considerar uma linha aproximadamente vertical.
    minimum_valid_separation_ratio: float = 0.02


# ============================================================================
# ESTRUTURAS
# ============================================================================


@dataclass
class LaneLine:
    """
    Representação unificada de uma linha de faixa.

    points:
        Pontos observados/projetados.

    confidence:
        Confiança global da linha.

    identifier:
        Identificador estável fornecido pelo tracker/model.
    """

    points: List[ProjectedLanePoint]

    confidence: float

    identifier: Optional[int] = None

    valid: bool = True

    def x_at(self, y: float) -> Optional[float]:
        """
        Obtém X da linha na altura Y solicitada.

        Utiliza interpolação linear entre os pontos disponíveis.
        """

        if not self.points:
            return None

        valid_points = [
            point
            for point in self.points
            if point.valid
            and np.isfinite(point.x)
            and np.isfinite(point.y)
        ]

        if not valid_points:
            return None

        valid_points.sort(
            key=lambda point: point.y
        )

        ys = np.asarray(
            [point.y for point in valid_points],
            dtype=np.float64,
        )

        xs = np.asarray(
            [point.x for point in valid_points],
            dtype=np.float64,
        )

        y_value = float(y)

        if y_value <= ys[0]:
            return float(xs[0])

        if y_value >= ys[-1]:
            return float(xs[-1])

        return float(
            np.interp(
                y_value,
                ys,
                xs,
            )
        )


@dataclass
class LaneCorridor:
    """
    Faixa formada por duas linhas.

    left_line:
        Linha esquerda.

    right_line:
        Linha direita.

    center_x:
        Centro da faixa no Y de avaliação.

    width:
        Largura da faixa em pixels.

    confidence:
        Confiança da faixa.
    """

    index: int

    left_line: LaneLine

    right_line: LaneLine

    left_x: float

    right_x: float

    center_x: float

    width: float

    confidence: float

    valid: bool = True

    is_ego_lane: bool = False

    lateral_offset: Optional[float] = None

    lateral_offset_normalized: Optional[float] = None


@dataclass
class LaneSelectionResult:
    """
    Resultado da identificação das faixas.
    """

    lanes: List[LaneCorridor] = field(
        default_factory=list
    )

    ego_lane: Optional[LaneCorridor] = None

    left_adjacent_lane: Optional[LaneCorridor] = None

    right_adjacent_lane: Optional[LaneCorridor] = None

    vehicle_x: Optional[float] = None

    status: LaneSelectionStatus = (
        LaneSelectionStatus.INVALID
    )

    confidence: float = 0.0

    valid: bool = False

    reason: Optional[str] = None


# ============================================================================
# SELECTOR
# ============================================================================


class LaneSelector:
    """
    Determina quais pares de linhas formam faixas e identifica
    a faixa ocupada pelo veículo.
    """

    def __init__(
        self,
        config: Optional[
            LaneSelectorConfig
        ] = None,
    ) -> None:

        self.config = (
            config
            if config is not None
            else LaneSelectorConfig()
        )

        self.last_result: Optional[
            LaneSelectionResult
        ] = None

    # ========================================================================
    # API PRINCIPAL
    # ========================================================================

    def select(
        self,
        lane_results: Sequence[
            LaneProjectionResult
        ],
        image_width: int,
        image_height: int,
        vehicle_x: Optional[float] = None,
    ) -> LaneSelectionResult:
        """
        Identifica todas as faixas possíveis e determina a faixa ocupada.

        vehicle_x:
            Posição horizontal do veículo no frame.

            Se None:
                utiliza o centro óptico da imagem.

        Isso é propositalmente configurável porque futuramente
        substituiremos essa referência pela calibração da câmera.
        """

        result = LaneSelectionResult()

        if image_width <= 0 or image_height <= 0:

            result.reason = (
                "Dimensões da imagem inválidas."
            )

            self.last_result = result

            return result

        # --------------------------------------------------------------------
        # Posição do veículo.
        # --------------------------------------------------------------------

        if vehicle_x is None:

            vehicle_x = (
                image_width / 2.0
            )

        vehicle_x = float(vehicle_x)

        if not np.isfinite(vehicle_x):

            result.reason = (
                "Posição do veículo inválida."
            )

            self.last_result = result

            return result

        if (
            vehicle_x < 0.0
            or vehicle_x > image_width
        ):

            result.reason = (
                "Posição do veículo fora da imagem."
            )

            self.last_result = result

            return result

        result.vehicle_x = vehicle_x

        # --------------------------------------------------------------------
        # Converter resultados em linhas.
        # --------------------------------------------------------------------

        lines = self._build_lines(
            lane_results
        )

        if len(lines) < 2:

            result.reason = (
                "Menos de duas linhas válidas."
            )

            self.last_result = result

            return result

        # --------------------------------------------------------------------
        # Ordenar da esquerda para a direita.
        # --------------------------------------------------------------------

        evaluation_y = (
            image_height
            * self.config.evaluation_y_ratio
        )

        lines = self._sort_lines(
            lines,
            evaluation_y,
        )

        # --------------------------------------------------------------------
        # Criar corredores entre linhas consecutivas.
        # --------------------------------------------------------------------

        corridors = self._build_corridors(
            lines=lines,
            image_width=image_width,
            image_height=image_height,
            evaluation_y=evaluation_y,
        )

        if not corridors:

            result.reason = (
                "Nenhuma faixa geometricamente válida."
            )

            self.last_result = result

            return result

        result.lanes = corridors

        # --------------------------------------------------------------------
        # Encontrar faixa que contém o veículo.
        # --------------------------------------------------------------------

        ego_lane, ambiguity = (
            self._find_ego_lane(
                corridors,
                vehicle_x,
            )
        )

        if ego_lane is None:

            result.status = (
                LaneSelectionStatus.INVALID
            )

            result.reason = (
                "Não foi possível determinar "
                "a faixa ocupada."
            )

            result.valid = False

            self.last_result = result

            return result

        ego_lane.is_ego_lane = True

        # --------------------------------------------------------------------
        # Posição relativa dentro da faixa.
        #
        # -1 = extremo esquerdo
        #  0 = centro
        # +1 = extremo direito
        # --------------------------------------------------------------------

        normalized_offset = (
            vehicle_x
            - ego_lane.center_x
        ) / max(
            ego_lane.width / 2.0,
            1.0,
        )

        normalized_offset = float(
            np.clip(
                normalized_offset,
                -2.0,
                2.0,
            )
        )

        ego_lane.lateral_offset = (
            vehicle_x
            - ego_lane.center_x
        )

        ego_lane.lateral_offset_normalized = (
            normalized_offset
        )

        # --------------------------------------------------------------------
        # Faixas adjacentes.
        # --------------------------------------------------------------------

        ego_index = ego_lane.index

        left_candidates = [
            lane
            for lane in corridors
            if lane.index < ego_index
        ]

        right_candidates = [
            lane
            for lane in corridors
            if lane.index > ego_index
        ]

        if left_candidates:

            result.left_adjacent_lane = max(
                left_candidates,
                key=lambda lane: lane.index,
            )

        if right_candidates:

            result.right_adjacent_lane = min(
                right_candidates,
                key=lambda lane: lane.index,
            )

        result.ego_lane = ego_lane

        # --------------------------------------------------------------------
        # Estado.
        # --------------------------------------------------------------------

        if ambiguity:

            result.status = (
                LaneSelectionStatus.AMBIGUOUS
            )

        else:

            result.status = (
                LaneSelectionStatus.VALID
            )

        result.confidence = (
            self._calculate_selection_confidence(
                ego_lane=ego_lane,
                ambiguity=ambiguity,
            )
        )

        result.valid = (
            not ambiguity
        )

        if ambiguity:

            result.reason = (
                "Veículo próximo da fronteira "
                "entre duas faixas."
            )

        self.last_result = result

        return result

    # ========================================================================
    # CONVERSÃO
    # ========================================================================

    def _build_lines(
        self,
        lane_results: Sequence[
            LaneProjectionResult
        ],
    ) -> List[LaneLine]:

        lines = []

        for index, result in enumerate(
            lane_results
        ):

            if result is None:
                continue

            if not result.valid:
                continue

            if not result.points:
                continue

            confidence = float(
                np.clip(
                    result.confidence,
                    0.0,
                    1.0,
                )
            )

            if (
                confidence
                < self.config.min_confidence
            ):
                continue

            points = [
                point
                for point in result.points
                if point.valid
            ]

            if (
                len(points)
                < self.config.min_points
            ):
                continue

            lines.append(
                LaneLine(
                    points=points,
                    confidence=confidence,
                    identifier=index,
                    valid=True,
                )
            )

        return lines

    # ========================================================================
    # ORDENAÇÃO
    # ========================================================================

    @staticmethod
    def _sort_lines(
        lines: Sequence[LaneLine],
        evaluation_y: float,
    ) -> List[LaneLine]:

        def x_position(
            line: LaneLine,
        ) -> float:

            x = line.x_at(
                evaluation_y
            )

            if x is None:
                return float("inf")

            return x

        return sorted(
            lines,
            key=x_position,
        )

    # ========================================================================
    # CORREDORES
    # ========================================================================

    def _build_corridors(
        self,
        lines: Sequence[LaneLine],
        image_width: int,
        image_height: int,
        evaluation_y: float,
    ) -> List[LaneCorridor]:

        corridors = []

        for index in range(
            len(lines) - 1
        ):

            left = lines[index]
            right = lines[index + 1]

            left_x = left.x_at(
                evaluation_y
            )

            right_x = right.x_at(
                evaluation_y
            )

            if (
                left_x is None
                or right_x is None
            ):
                continue

            # ---------------------------------------------------------------
            # As linhas precisam estar corretamente ordenadas.
            # ---------------------------------------------------------------

            if right_x <= left_x:
                continue

            width = (
                right_x
                - left_x
            )

            width_ratio = (
                width
                / float(image_width)
            )

            if width_ratio < (
                self.config.min_lane_width_ratio
            ):
                continue

            if width_ratio > (
                self.config.max_lane_width_ratio
            ):
                continue

            # ---------------------------------------------------------------
            # Verificar largura em mais de uma profundidade.
            #
            # Isso impede que duas linhas se cruzando sejam tratadas
            # como uma faixa válida.
            # ---------------------------------------------------------------

            width_samples = []

            sample_ratios = (
                0.65,
                0.78,
                0.88,
                0.96,
            )

            for ratio in sample_ratios:

                y = (
                    image_height
                    * ratio
                )

                lx = left.x_at(y)
                rx = right.x_at(y)

                if (
                    lx is None
                    or rx is None
                ):
                    continue

                sample_width = (
                    rx - lx
                )

                if sample_width <= 0:
                    continue

                width_samples.append(
                    sample_width
                )

            if len(width_samples) < 2:
                continue

            width_min = min(
                width_samples
            )

            width_max = max(
                width_samples
            )

            width_variation = (
                width_max - width_min
            ) / max(
                width_max,
                1.0,
            )

            if width_variation > (
                self.config.max_width_variation_ratio
            ):
                continue

            center_x = (
                left_x
                + right_x
            ) / 2.0

            confidence = (
                left.confidence
                * right.confidence
            )

            confidence = float(
                np.sqrt(
                    max(
                        confidence,
                        0.0,
                    )
                )
            )

            corridors.append(
                LaneCorridor(
                    index=index,
                    left_line=left,
                    right_line=right,
                    left_x=float(left_x),
                    right_x=float(right_x),
                    center_x=float(center_x),
                    width=float(width),
                    confidence=confidence,
                    valid=True,
                )
            )

        return corridors

    # ========================================================================
    # FAIXA DO VEÍCULO
    # ========================================================================

    def _find_ego_lane(
        self,
        corridors: Sequence[LaneCorridor],
        vehicle_x: float,
    ) -> Tuple[
        Optional[LaneCorridor],
        bool,
    ]:
        """
        Determina qual corredor contém o veículo.

        Retorna:

            lane
            ambiguity

        A ambiguidade é importante:

        se o veículo estiver exatamente sobre uma marca,
        não devemos fingir que sabemos em qual faixa ele está.
        """

        containing = []

        for lane in corridors:

            if (
                lane.left_x
                <= vehicle_x
                <= lane.right_x
            ):

                containing.append(
                    lane
                )

        if not containing:

            return None, False

        # Normalmente apenas uma faixa contém o veículo.
        if len(containing) == 1:

            lane = containing[0]

            distance_to_left = (
                vehicle_x
                - lane.left_x
            )

            distance_to_right = (
                lane.right_x
                - vehicle_x
            )

            boundary_distance = min(
                distance_to_left,
                distance_to_right,
            )

            ambiguity_threshold = (
                lane.width
                * self.config.boundary_ambiguity_ratio
            )

            ambiguity = (
                boundary_distance
                <= ambiguity_threshold
            )

            return lane, ambiguity

        # Situação geometricamente inconsistente.
        #
        # Escolhemos a faixa cujo centro está mais próximo,
        # mas marcamos como ambígua.
        lane = min(
            containing,
            key=lambda item: abs(
                item.center_x
                - vehicle_x
            ),
        )

        return lane, True

    # ========================================================================
    # CONFIANÇA
    # ========================================================================

    @staticmethod
    def _calculate_selection_confidence(
        ego_lane: LaneCorridor,
        ambiguity: bool,
    ) -> float:

        confidence = (
            ego_lane.confidence
        )

        if ambiguity:
            confidence *= 0.35

        return float(
            np.clip(
                confidence,
                0.0,
                1.0,
            )
        )


# ============================================================================
# FACTORY
# ============================================================================


def create_default_lane_selector(
    **kwargs,
) -> LaneSelector:

    config = LaneSelectorConfig(
        **kwargs
    )

    return LaneSelector(
        config=config
    )


__all__ = [
    "LaneSelectionStatus",
    "LaneSelectorConfig",
    "LaneLine",
    "LaneCorridor",
    "LaneSelectionResult",
    "LaneSelector",
    "create_default_lane_selector",
]