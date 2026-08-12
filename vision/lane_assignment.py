"""
vision/lane_assignment.py

Atribuição das faixas da pista.

Responsabilidade:

    LaneProjection
          ↓
    linhas ordenadas espacialmente
          ↓
    corredores/faixas
          ↓
    faixa atual do veículo
          ↓
    faixas laterais

Este módulo NÃO:
    - executa YOLOP;
    - detecta pixels;
    - calcula geometria da faixa;
    - faz projeção da faixa;
    - controla o veículo.

Ele apenas interpreta a estrutura geométrica já produzida
pelas etapas anteriores.

Objetivos:

    1. Suportar até 3 faixas de rodagem.
    2. Suportar acostamento.
    3. Identificar a faixa ocupada pelo veículo.
    4. Identificar faixas à esquerda e à direita.
    5. Não depender da existência simultânea de todas as linhas.
    6. Permitir continuidade temporal quando uma linha desaparecer.
    7. Não trocar de faixa instantaneamente por ruído.
    8. Manter a estrutura preparada para futuras extensões.

Convenção:

    linha 0 ─────── limite esquerdo
             faixa 0
    linha 1 ───────
             faixa 1
    linha 2 ───────
             faixa 2
    linha 3 ─────── limite direito

O índice da faixa é determinado pelo corredor onde o
centro projetado do veículo se encontra.

Não assumimos que a câmera esteja perfeitamente centralizada.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

MAX_LANES = 3

DEFAULT_MIN_LANE_WIDTH_PX = 80.0
DEFAULT_MAX_LANE_WIDTH_PX = 1200.0

DEFAULT_BOUNDARY_SEARCH_MARGIN_PX = 140.0

# Número de atualizações consecutivas necessárias para aceitar
# uma troca de faixa.
DEFAULT_LANE_CHANGE_CONFIRMATIONS = 3

# Tolerância usada para evitar troca quando o veículo está
# praticamente sobre uma linha.
DEFAULT_BOUNDARY_EPSILON_PX = 8.0


# ============================================================================
# RESULTADOS
# ============================================================================


@dataclass(frozen=True)
class LaneBoundary:
    """
    Linha limite de uma faixa.

    A linha pode possuir vários pontos projetados.

    points:
        sequência de (x, y).

    confidence:
        confiança geral da linha.

    index:
        índice espacial da linha após ordenação.
    """

    points: Tuple[Tuple[float, float], ...] = field(
        default_factory=tuple
    )

    confidence: float = 0.0

    index: int = -1

    source_id: Optional[int] = None

    valid: bool = True

    @property
    def bottom_x(self) -> Optional[float]:
        """
        Retorna o X da linha na região inferior da imagem.

        Essa região é especialmente importante para determinar
        o corredor ocupado pelo veículo.
        """

        if not self.points:
            return None

        return float(
            max(
                self.points,
                key=lambda p: p[1],
            )[0]
        )

    @property
    def mean_x(self) -> Optional[float]:
        if not self.points:
            return None

        return float(
            np.mean(
                [
                    point[0]
                    for point in self.points
                ]
            )
        )


@dataclass(frozen=True)
class LaneCorridor:
    """
    Corredor delimitado por duas linhas.

    left_boundary:
        limite esquerdo.

    right_boundary:
        limite direito.

    index:
        índice da faixa, começando em zero.

    width:
        largura estimada na região utilizada.

    confidence:
        confiança combinada das duas linhas.

    is_shoulder:
        indica se o corredor foi classificado como acostamento.
    """

    index: int

    left_boundary: LaneBoundary

    right_boundary: LaneBoundary

    width: float

    confidence: float

    is_shoulder: bool = False

    valid: bool = True

    @property
    def left_x(self) -> Optional[float]:
        return self.left_boundary.bottom_x

    @property
    def right_x(self) -> Optional[float]:
        return self.right_boundary.bottom_x

    @property
    def center_x(self) -> Optional[float]:
        left = self.left_x
        right = self.right_x

        if left is None or right is None:
            return None

        return (left + right) * 0.5


@dataclass
class LaneAssignmentResult:
    """
    Resultado da atribuição das faixas.
    """

    lanes: List[LaneCorridor] = field(
        default_factory=list
    )

    current_lane_index: Optional[int] = None

    current_lane: Optional[LaneCorridor] = None

    left_lanes: List[LaneCorridor] = field(
        default_factory=list
    )

    right_lanes: List[LaneCorridor] = field(
        default_factory=list
    )

    left_boundary: Optional[LaneBoundary] = None

    right_boundary: Optional[LaneBoundary] = None

    vehicle_x: Optional[float] = None

    vehicle_inside_lane: bool = False

    vehicle_inside_shoulder: bool = False

    confidence: float = 0.0

    valid: bool = False

    stable: bool = False

    lane_change_pending: bool = False

    candidate_lane_index: Optional[int] = None

    boundary_distance_left: Optional[float] = None

    boundary_distance_right: Optional[float] = None

    error: Optional[str] = None

    @property
    def lane_count(self) -> int:
        return len(self.lanes)

    @property
    def current_lane_center_x(self) -> Optional[float]:
        if self.current_lane is None:
            return None

        return self.current_lane.center_x


# ============================================================================
# UTILITÁRIOS
# ============================================================================


def _finite(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _clamp_confidence(value: float) -> float:
    if not _finite(value):
        return 0.0

    return float(
        np.clip(
            value,
            0.0,
            1.0,
        )
    )


def _extract_bottom_x(
    boundary: LaneBoundary,
) -> Optional[float]:
    return boundary.bottom_x


# ============================================================================
# ASSIGNMENT
# ============================================================================


class LaneAssignment:
    """
    Determina qual faixa o veículo ocupa.

    O algoritmo trabalha sobre as linhas projetadas e não
    sobre a máscara original do YOLOP.

    Isso é importante porque a atribuição deve acontecer
    depois que as linhas foram estabilizadas/projetadas.
    """

    def __init__(
        self,
        max_lanes: int = MAX_LANES,
        min_lane_width_px: float = DEFAULT_MIN_LANE_WIDTH_PX,
        max_lane_width_px: float = DEFAULT_MAX_LANE_WIDTH_PX,
        boundary_search_margin_px: float = (
            DEFAULT_BOUNDARY_SEARCH_MARGIN_PX
        ),
        lane_change_confirmations: int = (
            DEFAULT_LANE_CHANGE_CONFIRMATIONS
        ),
        boundary_epsilon_px: float = (
            DEFAULT_BOUNDARY_EPSILON_PX
        ),
    ) -> None:

        self.max_lanes = max(
            1,
            int(max_lanes),
        )

        self.min_lane_width_px = max(
            1.0,
            float(min_lane_width_px),
        )

        self.max_lane_width_px = max(
            self.min_lane_width_px,
            float(max_lane_width_px),
        )

        self.boundary_search_margin_px = max(
            0.0,
            float(boundary_search_margin_px),
        )

        self.lane_change_confirmations = max(
            1,
            int(lane_change_confirmations),
        )

        self.boundary_epsilon_px = max(
            0.0,
            float(boundary_epsilon_px),
        )

        # Estado temporal.
        self._stable_lane_index: Optional[int] = None

        self._candidate_lane_index: Optional[int] = None

        self._candidate_count = 0

        self.last_result: Optional[
            LaneAssignmentResult
        ] = None

    # ========================================================================
    # NORMALIZAÇÃO
    # ========================================================================

    @staticmethod
    def normalize_boundaries(
        boundaries: Iterable[LaneBoundary],
    ) -> List[LaneBoundary]:
        """
        Ordena as linhas da esquerda para a direita.

        A ordenação é feita pela posição inferior da linha,
        pois é a região mais relevante para determinar
        a ocupação atual da pista.
        """

        valid: List[LaneBoundary] = []

        for boundary in boundaries:

            if not isinstance(
                boundary,
                LaneBoundary,
            ):
                continue

            if not boundary.valid:
                continue

            x = boundary.bottom_x

            if x is None:
                continue

            if not _finite(x):
                continue

            valid.append(boundary)

        valid.sort(
            key=lambda boundary: (
                boundary.bottom_x
                if boundary.bottom_x is not None
                else float("inf")
            )
        )

        normalized: List[LaneBoundary] = []

        for index, boundary in enumerate(valid):

            normalized.append(
                LaneBoundary(
                    points=boundary.points,
                    confidence=boundary.confidence,
                    index=index,
                    source_id=boundary.source_id,
                    valid=boundary.valid,
                )
            )

        return normalized

    # ========================================================================
    # CONVERSÃO DE FORMATOS
    # ========================================================================

    @staticmethod
    def from_points(
        points: Sequence[Sequence[Tuple[float, float]]],
        confidences: Optional[
            Sequence[float]
        ] = None,
    ) -> List[LaneBoundary]:
        """
        Constrói LaneBoundary a partir de listas de pontos.

        Exemplo:

            [
                [(100, 900), (110, 800)],
                [(500, 900), (490, 800)],
                ...
            ]
        """

        result: List[LaneBoundary] = []

        for index, lane_points in enumerate(points):

            normalized_points = []

            for point in lane_points:

                if len(point) < 2:
                    continue

                x = float(point[0])
                y = float(point[1])

                if not (
                    np.isfinite(x)
                    and np.isfinite(y)
                ):
                    continue

                normalized_points.append(
                    (x, y)
                )

            if not normalized_points:
                continue

            confidence = 1.0

            if (
                confidences is not None
                and index < len(confidences)
            ):
                confidence = _clamp_confidence(
                    confidences[index]
                )

            result.append(
                LaneBoundary(
                    points=tuple(
                        normalized_points
                    ),
                    confidence=confidence,
                    source_id=index,
                )
            )

        return LaneAssignment.normalize_boundaries(
            result
        )

    # ========================================================================
    # CONSTRUÇÃO DOS CORREDORES
    # ========================================================================

    def build_lanes(
        self,
        boundaries: Sequence[LaneBoundary],
    ) -> List[LaneCorridor]:
        """
        Constrói corredores entre linhas consecutivas.

        Se houver:

            L0 L1 L2 L3

        teremos:

            faixa 0 = L0 ↔ L1
            faixa 1 = L1 ↔ L2
            faixa 2 = L2 ↔ L3

        O número máximo de faixas é limitado por max_lanes.
        """

        ordered = self.normalize_boundaries(
            boundaries
        )

        corridors: List[LaneCorridor] = []

        for index in range(
            min(
                len(ordered) - 1,
                self.max_lanes,
            )
        ):

            left = ordered[index]
            right = ordered[index + 1]

            left_x = left.bottom_x
            right_x = right.bottom_x

            if (
                left_x is None
                or right_x is None
            ):
                continue

            width = (
                right_x
                - left_x
            )

            if width <= 0:
                continue

            if width < self.min_lane_width_px:
                logger.debug(
                    "[LANE_ASSIGNMENT] "
                    "Corredor rejeitado: "
                    "largura %.1f px.",
                    width,
                )
                continue

            if width > self.max_lane_width_px:
                logger.debug(
                    "[LANE_ASSIGNMENT] "
                    "Corredor muito largo: "
                    "%.1f px.",
                    width,
                )

            confidence = (
                left.confidence
                * right.confidence
            ) ** 0.5

            corridors.append(
                LaneCorridor(
                    index=len(corridors),
                    left_boundary=left,
                    right_boundary=right,
                    width=float(width),
                    confidence=float(
                        confidence
                    ),
                    is_shoulder=False,
                    valid=True,
                )
            )

            if len(corridors) >= self.max_lanes:
                break

        return corridors

    # ========================================================================
    # DETERMINAÇÃO DA FAIXA
    # ========================================================================

    @staticmethod
    def _contains_vehicle(
        lane: LaneCorridor,
        vehicle_x: float,
        epsilon: float,
    ) -> bool:

        left_x = lane.left_x
        right_x = lane.right_x

        if (
            left_x is None
            or right_x is None
        ):
            return False

        return (
            vehicle_x
            >= left_x - epsilon
            and vehicle_x
            <= right_x + epsilon
        )

    @staticmethod
    def _distance_to_lane(
        lane: LaneCorridor,
        vehicle_x: float,
    ) -> float:

        left_x = lane.left_x
        right_x = lane.right_x

        if (
            left_x is None
            or right_x is None
        ):
            return float("inf")

        if vehicle_x < left_x:
            return left_x - vehicle_x

        if vehicle_x > right_x:
            return vehicle_x - right_x

        return 0.0

    def _find_candidate_lane(
        self,
        lanes: Sequence[LaneCorridor],
        vehicle_x: float,
    ) -> Optional[int]:

        # Primeiro procuramos uma faixa que realmente
        # contenha o X do veículo.
        for lane in lanes:

            if self._contains_vehicle(
                lane,
                vehicle_x,
                self.boundary_epsilon_px,
            ):
                return lane.index

        # Se estiver ligeiramente fora de uma linha,
        # procuramos a faixa mais próxima.
        if not lanes:
            return None

        distances = [
            (
                self._distance_to_lane(
                    lane,
                    vehicle_x,
                ),
                lane.index,
            )
            for lane in lanes
        ]

        distance, index = min(
            distances,
            key=lambda item: item[0],
        )

        if (
            distance
            <= self.boundary_search_margin_px
        ):
            return index

        return None

    # ========================================================================
    # ESTABILIZAÇÃO TEMPORAL
    # ========================================================================

    def _stabilize_lane_index(
        self,
        candidate: Optional[int],
    ) -> Tuple[
        Optional[int],
        bool,
        bool,
    ]:
        """
        Evita trocar de faixa por uma única detecção
        instável.

        Retorna:

            stable_index
            stable
            lane_change_pending
        """

        if candidate is None:

            # Não apagamos imediatamente a faixa atual.
            #
            # Isso é importante quando uma linha desaparece
            # temporariamente por reflexo, chuva ou oclusão.
            return (
                self._stable_lane_index,
                self._stable_lane_index is not None,
                False,
            )

        if self._stable_lane_index is None:

            self._stable_lane_index = candidate

            self._candidate_lane_index = None
            self._candidate_count = 0

            return (
                candidate,
                True,
                False,
            )

        if candidate == self._stable_lane_index:

            self._candidate_lane_index = None
            self._candidate_count = 0

            return (
                self._stable_lane_index,
                True,
                False,
            )

        # Candidato diferente.
        if (
            self._candidate_lane_index
            != candidate
        ):

            self._candidate_lane_index = candidate
            self._candidate_count = 1

        else:

            self._candidate_count += 1

        if (
            self._candidate_count
            >= self.lane_change_confirmations
        ):

            self._stable_lane_index = candidate

            self._candidate_lane_index = None
            self._candidate_count = 0

            return (
                self._stable_lane_index,
                True,
                False,
            )

        return (
            self._stable_lane_index,
            True,
            True,
        )

    # ========================================================================
    # DISTÂNCIA DAS BORDAS
    # ========================================================================

    @staticmethod
    def _boundary_distances(
        lane: Optional[LaneCorridor],
        vehicle_x: Optional[float],
    ) -> Tuple[
        Optional[float],
        Optional[float],
    ]:

        if (
            lane is None
            or vehicle_x is None
        ):
            return None, None

        left_x = lane.left_x
        right_x = lane.right_x

        if (
            left_x is None
            or right_x is None
        ):
            return None, None

        return (
            vehicle_x - left_x,
            right_x - vehicle_x,
        )

    # ========================================================================
    # API PRINCIPAL
    # ========================================================================

    def assign(
        self,
        boundaries: Sequence[LaneBoundary],
        vehicle_x: Optional[float],
    ) -> LaneAssignmentResult:
        """
        Determina a faixa atual.

        vehicle_x:
            X do centro projetado do veículo na região
            de referência da pista.

        Importante:

        Se não houver elementos suficientes para determinar
        a faixa, o método não inventa uma posição.
        """

        try:

            if (
                vehicle_x is None
                or not _finite(vehicle_x)
            ):

                return LaneAssignmentResult(
                    valid=False,
                    stable=False,
                    error=(
                        "vehicle_x inválido."
                    ),
                )

            vehicle_x = float(vehicle_x)

            ordered = self.normalize_boundaries(
                boundaries
            )

            lanes = self.build_lanes(
                ordered
            )

            if not lanes:

                stable_index, stable, pending = (
                    self._stabilize_lane_index(
                        None
                    )
                )

                result = LaneAssignmentResult(
                    lanes=[],
                    current_lane_index=(
                        stable_index
                    ),
                    vehicle_x=vehicle_x,
                    vehicle_inside_lane=False,
                    confidence=0.0,
                    valid=False,
                    stable=stable,
                    lane_change_pending=pending,
                    candidate_lane_index=(
                        self._candidate_lane_index
                    ),
                    error=(
                        "Não existem "
                        "corredores suficientes."
                    ),
                )

                self.last_result = result

                return result

            candidate_index = (
                self._find_candidate_lane(
                    lanes,
                    vehicle_x,
                )
            )

            stable_index, stable, pending = (
                self._stabilize_lane_index(
                    candidate_index
                )
            )

            current_lane = None

            if stable_index is not None:

                for lane in lanes:

                    if (
                        lane.index
                        == stable_index
                    ):
                        current_lane = lane
                        break

            # Se a faixa estabilizada deixou de existir,
            # não usamos uma faixa aleatória.
            if current_lane is None:

                result = LaneAssignmentResult(
                    lanes=list(lanes),
                    current_lane_index=(
                        stable_index
                    ),
                    vehicle_x=vehicle_x,
                    vehicle_inside_lane=False,
                    confidence=0.0,
                    valid=False,
                    stable=stable,
                    lane_change_pending=pending,
                    candidate_lane_index=(
                        candidate_index
                    ),
                    error=(
                        "Faixa atual "
                        "não possui limites "
                        "suficientes."
                    ),
                )

                self.last_result = result

                return result

            vehicle_inside_lane = (
                candidate_index
                == stable_index
            )

            left_lanes = [
                lane
                for lane in lanes
                if lane.index < stable_index
            ]

            right_lanes = [
                lane
                for lane in lanes
                if lane.index > stable_index
            ]

            left_boundary = (
                current_lane.left_boundary
            )

            right_boundary = (
                current_lane.right_boundary
            )

            distance_left, distance_right = (
                self._boundary_distances(
                    current_lane,
                    vehicle_x,
                )
            )

            result = LaneAssignmentResult(
                lanes=list(lanes),
                current_lane_index=(
                    stable_index
                ),
                current_lane=current_lane,
                left_lanes=left_lanes,
                right_lanes=right_lanes,
                left_boundary=left_boundary,
                right_boundary=right_boundary,
                vehicle_x=vehicle_x,
                vehicle_inside_lane=(
                    vehicle_inside_lane
                ),
                vehicle_inside_shoulder=False,
                confidence=float(
                    current_lane.confidence
                ),
                valid=True,
                stable=stable,
                lane_change_pending=pending,
                candidate_lane_index=(
                    candidate_index
                ),
                boundary_distance_left=(
                    distance_left
                ),
                boundary_distance_right=(
                    distance_right
                ),
                error=None,
            )

            self.last_result = result

            return result

        except Exception as exc:

            logger.exception(
                "[LANE_ASSIGNMENT] "
                "Falha ao atribuir faixa."
            )

            result = LaneAssignmentResult(
                valid=False,
                stable=False,
                vehicle_x=vehicle_x,
                error=(
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            )

            self.last_result = result

            return result

    # ========================================================================
    # RESET
    # ========================================================================

    def reset(self) -> None:
        """
        Limpa o estado temporal.

        Deve ser chamado quando:
            - uma nova sessão começa;
            - a câmera muda;
            - o jogo reinicia;
            - a calibração muda.
        """

        self._stable_lane_index = None
        self._candidate_lane_index = None
        self._candidate_count = 0
        self.last_result = None


# ============================================================================
# FACTORY
# ============================================================================


def create_lane_assignment(
    **kwargs,
) -> LaneAssignment:
    """
    Factory padrão.
    """

    return LaneAssignment(
        **kwargs
    )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "LaneBoundary",
    "LaneCorridor",
    "LaneAssignmentResult",
    "LaneAssignment",
    "create_lane_assignment",
]