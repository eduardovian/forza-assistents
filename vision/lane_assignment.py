"""
vision/lane_assignment.py

Associação das linhas detectadas à estrutura da pista.

Fluxo:

    LaneModel[]
        ↓
    ordenação espacial
        ↓
    corredores
        ↓
    identificação da faixa atual
        ↓
    faixas à esquerda/direita

Este módulo NÃO:
    - executa YOLOP;
    - processa pixels;
    - ajusta polinômios;
    - calcula geometria;
    - calcula heading;
    - decide atuação ADAS.

O módulo trabalha exclusivamente com o contrato atual
definido em vision/lane_types.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .lane_types import LaneModel


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

DEFAULT_MAX_LANES = 3

DEFAULT_MIN_LANE_WIDTH_PX = 80.0
DEFAULT_MAX_LANE_WIDTH_PX = 1200.0

DEFAULT_VEHICLE_X_RATIO = 0.5

DEFAULT_BOUNDARY_EPSILON_PX = 8.0

DEFAULT_LANE_CHANGE_CONFIRMATIONS = 3


# =============================================================================
# RESULTADOS LOCAIS
# =============================================================================

@dataclass(frozen=True)
class LaneAssignmentResult:
    """
    Resultado da associação espacial das lanes.

    current_lane:
        LaneModel correspondente à faixa atual quando determinada.

    current_lane_id:
        ID da lane atual.

    left_lanes:
        Lanes localizadas à esquerda da faixa atual.

    right_lanes:
        Lanes localizadas à direita da faixa atual.

    normalized_offset:
        Posição normalizada do veículo dentro da faixa:

            -1.0 = limite esquerdo
             0.0 = centro
            +1.0 = limite direito

    lateral_offset:
        Distância em pixels entre o centro da faixa e o veículo.

    lane_width:
        Largura da faixa atual em pixels.

    confidence:
        Confiança da associação.

    valid:
        Indica se uma faixa atual válida foi encontrada.

    lane_change_pending:
        Indica que uma possível mudança de faixa ainda está
        aguardando confirmações temporais.
    """

    lanes: List[LaneModel]

    current_lane: Optional[LaneModel]

    current_lane_id: Optional[int]

    left_lanes: List[LaneModel]

    right_lanes: List[LaneModel]

    normalized_offset: float

    lateral_offset: float

    lane_width: float

    confidence: float

    valid: bool

    lane_change_pending: bool = False


# =============================================================================
# ESTRUTURAS INTERNAS
# =============================================================================

@dataclass(frozen=True)
class _LaneCandidate:
    left: LaneModel
    right: LaneModel

    left_x: float
    right_x: float

    width: float
    confidence: float
    center_x: float


# =============================================================================
# UTILITÁRIOS
# =============================================================================

def _finite(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _clamp01(value: float) -> float:
    if not _finite(value):
        return 0.0

    return float(
        np.clip(
            value,
            0.0,
            1.0,
        )
    )


def _lane_x_at(
    lane: LaneModel,
    y: Optional[float] = None,
) -> Optional[float]:
    """
    Obtém a posição X de uma LaneModel em um Y de referência.

    Prioridade:

        polynomial
            ↓
        último ponto válido
    """

    if lane is None or lane.line is None:
        return None

    polynomial = lane.polynomial

    if (
        polynomial is not None
        and polynomial.valid
    ):
        target_y = y

        if target_y is None:
            if _finite(polynomial.y_max):
                target_y = polynomial.y_max

        if target_y is not None:
            x = polynomial.evaluate(
                float(target_y)
            )

            if _finite(x):
                return float(x)

    points = [
        point
        for point in lane.line.points
        if point is not None
        and point.valid
        and _finite(point.x)
        and _finite(point.y)
    ]

    if not points:
        return None

    if y is None:
        point = max(
            points,
            key=lambda item: item.y,
        )

        return float(point.x)

    nearest = min(
        points,
        key=lambda item: abs(
            item.y - y
        ),
    )

    return float(nearest.x)


def _lane_confidence(
    lane: LaneModel,
) -> float:
    """
    Calcula a confiança da LaneModel usando somente
    campos existentes no contrato atual.
    """

    if lane is None or lane.line is None:
        return 0.0

    confidence = _clamp01(
        lane.line.confidence
    )

    if lane.polynomial is not None:
        confidence = (
            0.65 * confidence
            + 0.35
            * _clamp01(
                lane.polynomial.confidence
            )
        )

    if not lane.valid:
        confidence *= 0.5

    return _clamp01(confidence)


def _lane_center(
    left_x: float,
    right_x: float,
) -> float:
    return (
        float(left_x)
        + float(right_x)
    ) / 2.0


def _normalized_offset(
    vehicle_x: float,
    center_x: float,
    lane_width: float,
) -> float:
    """
    Converte o deslocamento lateral para [-1, +1].

    -1 = limite esquerdo
     0 = centro
    +1 = limite direito
    """

    if lane_width <= 0.0:
        return 0.0

    offset = (
        float(vehicle_x)
        - float(center_x)
    )

    normalized = (
        2.0 * offset / float(lane_width)
    )

    return float(
        np.clip(
            normalized,
            -1.0,
            1.0,
        )
    )


# =============================================================================
# CLASSE PRINCIPAL
# =============================================================================

class LaneAssignment:
    """
    Associa LaneModel à posição do veículo.

    A classe mantém somente o estado necessário para impedir
    mudanças instantâneas de faixa causadas por ruído.
    """

    def __init__(
        self,
        max_lanes: int = DEFAULT_MAX_LANES,
        min_lane_width_px: float = DEFAULT_MIN_LANE_WIDTH_PX,
        max_lane_width_px: float = DEFAULT_MAX_LANE_WIDTH_PX,
        vehicle_x_ratio: float = DEFAULT_VEHICLE_X_RATIO,
        boundary_epsilon_px: float = DEFAULT_BOUNDARY_EPSILON_PX,
        lane_change_confirmations: int = (
            DEFAULT_LANE_CHANGE_CONFIRMATIONS
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

        self.vehicle_x_ratio = float(
            np.clip(
                vehicle_x_ratio,
                0.0,
                1.0,
            )
        )

        self.boundary_epsilon_px = max(
            0.0,
            float(boundary_epsilon_px),
        )

        self.lane_change_confirmations = max(
            1,
            int(lane_change_confirmations),
        )

        self._stable_lane_id: Optional[int] = None

        self._candidate_lane_id: Optional[int] = None

        self._candidate_count = 0

        self.last_result: Optional[
            LaneAssignmentResult
        ] = None

    # =========================================================================
    # RESET
    # =========================================================================

    def reset(self) -> None:
        self._stable_lane_id = None
        self._candidate_lane_id = None
        self._candidate_count = 0
        self.last_result = None

    # =========================================================================
    # NORMALIZAÇÃO
    # =========================================================================

    def normalize_lanes(
        self,
        lanes: Iterable[LaneModel],
        reference_y: Optional[float] = None,
    ) -> List[LaneModel]:
        """
        Ordena as lanes da esquerda para a direita.

        O lane_id original é preservado.
        """

        samples: List[
            Tuple[float, LaneModel]
        ] = []

        for lane in lanes:

            if lane is None:
                continue

            if not lane.valid:
                continue

            x = _lane_x_at(
                lane,
                reference_y,
            )

            if x is None:
                continue

            samples.append(
                (
                    x,
                    lane,
                )
            )

        samples.sort(
            key=lambda item: item[0]
        )

        return [
            lane
            for _, lane in samples
        ]

    # =========================================================================
    # VEÍCULO
    # =========================================================================

    @staticmethod
    def estimate_vehicle_x(
        frame_width: float,
        vehicle_x: Optional[float] = None,
        vehicle_center_x: Optional[float] = None,
        vehicle_x_ratio: float = DEFAULT_VEHICLE_X_RATIO,
    ) -> float:
        """
        Determina o X utilizado para representar o veículo.

        Se nenhum valor externo for fornecido,
        utiliza o centro horizontal da câmera.
        """

        if (
            vehicle_center_x is not None
            and _finite(vehicle_center_x)
        ):
            return float(vehicle_center_x)

        if (
            vehicle_x is not None
            and _finite(vehicle_x)
        ):
            return float(vehicle_x)

        return float(
            frame_width
            * vehicle_x_ratio
        )

    # =========================================================================
    # CORREDORES
    # =========================================================================

    def _build_candidates(
        self,
        lanes: Sequence[LaneModel],
        reference_y: Optional[float],
    ) -> List[_LaneCandidate]:

        ordered = self.normalize_lanes(
            lanes,
            reference_y,
        )

        candidates: List[
            _LaneCandidate
        ] = []

        for index in range(
            len(ordered) - 1
        ):

            left = ordered[index]

            right = ordered[index + 1]

            left_x = _lane_x_at(
                left,
                reference_y,
            )

            right_x = _lane_x_at(
                right,
                reference_y,
            )

            if (
                left_x is None
                or right_x is None
            ):
                continue

            width = (
                right_x
                - left_x
            )

            if width <= 0.0:
                continue

            if width < self.min_lane_width_px:
                continue

            if width > self.max_lane_width_px:
                continue

            left_confidence = (
                _lane_confidence(left)
            )

            right_confidence = (
                _lane_confidence(right)
            )

            confidence = float(
                np.sqrt(
                    left_confidence
                    * right_confidence
                )
            )

            center_x = _lane_center(
                left_x,
                right_x,
            )

            candidates.append(
                _LaneCandidate(
                    left=left,
                    right=right,
                    left_x=float(left_x),
                    right_x=float(right_x),
                    width=float(width),
                    confidence=confidence,
                    center_x=float(center_x),
                )
            )

            if len(candidates) >= self.max_lanes:
                break

        return candidates

    # =========================================================================
    # FAIXA CANDIDATA
    # =========================================================================

    def _find_candidate(
        self,
        candidates: Sequence[_LaneCandidate],
        vehicle_x: float,
    ) -> Optional[int]:

        if not candidates:
            return None

        # Primeiro tenta encontrar um corredor que contenha
        # diretamente o veículo.

        for index, candidate in enumerate(
            candidates
        ):

            if (
                vehicle_x
                >= candidate.left_x
                - self.boundary_epsilon_px
                and
                vehicle_x
                <= candidate.right_x
                + self.boundary_epsilon_px
            ):
                return index

        # Caso o veículo esteja ligeiramente fora,
        # utiliza o corredor mais próximo.

        distances = []

        for index, candidate in enumerate(
            candidates
        ):

            if vehicle_x < candidate.left_x:

                distance = (
                    candidate.left_x
                    - vehicle_x
                )

            elif vehicle_x > candidate.right_x:

                distance = (
                    vehicle_x
                    - candidate.right_x
                )

            else:

                distance = 0.0

            distances.append(
                (
                    distance,
                    index,
                )
            )

        if not distances:
            return None

        _, nearest_index = min(
            distances,
            key=lambda item: item[0],
        )

        return nearest_index

    # =========================================================================
    # ESTABILIZAÇÃO
    # =========================================================================

    def _stabilize_lane(
        self,
        candidate_lane_id: Optional[int],
    ) -> Tuple[
        Optional[int],
        bool,
        bool,
    ]:
        """
        Retorna:

            stable_id
            stable
            change_pending
        """

        if candidate_lane_id is None:

            self._candidate_lane_id = None
            self._candidate_count = 0

            return (
                self._stable_lane_id,
                False,
                False,
            )

        if self._stable_lane_id is None:

            self._stable_lane_id = (
                candidate_lane_id
            )

            self._candidate_lane_id = None
            self._candidate_count = 0

            return (
                self._stable_lane_id,
                True,
                False,
            )

        if (
            candidate_lane_id
            == self._stable_lane_id
        ):

            self._candidate_lane_id = None
            self._candidate_count = 0

            return (
                self._stable_lane_id,
                True,
                False,
            )

        if (
            candidate_lane_id
            != self._candidate_lane_id
        ):

            self._candidate_lane_id = (
                candidate_lane_id
            )

            self._candidate_count = 1

        else:

            self._candidate_count += 1

        if (
            self._candidate_count
            >= self.lane_change_confirmations
        ):

            self._stable_lane_id = (
                candidate_lane_id
            )

            self._candidate_lane_id = None
            self._candidate_count = 0

            return (
                self._stable_lane_id,
                True,
                False,
            )

        return (
            self._stable_lane_id,
            True,
            True,
        )

    # =========================================================================
    # ASSOCIAÇÃO
    # =========================================================================

    def assign(
        self,
        lanes: Iterable[LaneModel],
        frame_width: float,
        frame_height: Optional[float] = None,
        vehicle_x: Optional[float] = None,
        vehicle_center_x: Optional[float] = None,
        reference_y: Optional[float] = None,
    ) -> LaneAssignmentResult:
        """
        Executa a associação completa.

        frame_height é utilizado para escolher a região inferior
        da imagem quando reference_y não é fornecido.
        """

        if (
            not _finite(frame_width)
            or frame_width <= 0.0
        ):
            raise ValueError(
                "frame_width inválido."
            )

        if (
            reference_y is None
            and frame_height is not None
            and _finite(frame_height)
            and frame_height > 0.0
        ):
            reference_y = (
                float(frame_height)
                * 0.90
            )

        estimated_vehicle_x = (
            self.estimate_vehicle_x(
                frame_width=frame_width,
                vehicle_x=vehicle_x,
                vehicle_center_x=vehicle_center_x,
                vehicle_x_ratio=(
                    self.vehicle_x_ratio
                ),
            )
        )

        normalized = self.normalize_lanes(
            lanes,
            reference_y,
        )

        candidates = self._build_candidates(
            normalized,
            reference_y,
        )

        if not candidates:

            result = LaneAssignmentResult(
                lanes=list(normalized),
                current_lane=None,
                current_lane_id=None,
                left_lanes=[],
                right_lanes=[],
                normalized_offset=0.0,
                lateral_offset=0.0,
                lane_width=0.0,
                confidence=0.0,
                valid=False,
                lane_change_pending=False,
            )

            self.last_result = result

            return result

        candidate_index = self._find_candidate(
            candidates,
            estimated_vehicle_x,
        )

        if candidate_index is None:

            result = LaneAssignmentResult(
                lanes=list(normalized),
                current_lane=None,
                current_lane_id=None,
                left_lanes=[],
                right_lanes=[],
                normalized_offset=0.0,
                lateral_offset=0.0,
                lane_width=0.0,
                confidence=0.0,
                valid=False,
                lane_change_pending=False,
            )

            self.last_result = result

            return result

        candidate = candidates[
            candidate_index
        ]

        candidate_lane_id = (
            candidate.left.lane_id
        )

        stable_lane_id, _, pending = (
            self._stabilize_lane(
                candidate_lane_id
            )
        )

        selected_candidate = candidate

        if stable_lane_id is not None:

            for item in candidates:

                if (
                    item.left.lane_id
                    == stable_lane_id
                ):
                    selected_candidate = item
                    break

        current_lane = selected_candidate.right

        current_lane_id = (
            selected_candidate.left.lane_id
        )

        normalized_offset = (
            _normalized_offset(
                vehicle_x=estimated_vehicle_x,
                center_x=selected_candidate.center_x,
                lane_width=selected_candidate.width,
            )
        )

        lateral_offset = (
            estimated_vehicle_x
            - selected_candidate.center_x
        )

        left_lanes = [
            lane
            for lane in normalized
            if _lane_x_at(
                lane,
                reference_y,
            )
            < selected_candidate.left_x
        ]

        right_lanes = [
            lane
            for lane in normalized
            if _lane_x_at(
                lane,
                reference_y,
            )
            > selected_candidate.right_x
        ]

        result = LaneAssignmentResult(
            lanes=list(normalized),
            current_lane=current_lane,
            current_lane_id=current_lane_id,
            left_lanes=left_lanes,
            right_lanes=right_lanes,
            normalized_offset=normalized_offset,
            lateral_offset=float(
                lateral_offset
            ),
            lane_width=float(
                selected_candidate.width
            ),
            confidence=float(
                selected_candidate.confidence
            ),
            valid=True,
            lane_change_pending=pending,
        )

        self.last_result = result

        return result

    # =========================================================================
    # ALIASES
    # =========================================================================

    def update(
        self,
        lanes: Iterable[LaneModel],
        frame_width: float,
        frame_height: Optional[float] = None,
        vehicle_x: Optional[float] = None,
        vehicle_center_x: Optional[float] = None,
        reference_y: Optional[float] = None,
    ) -> LaneAssignmentResult:
        """
        Alias semântico para assign().
        """

        return self.assign(
            lanes=lanes,
            frame_width=frame_width,
            frame_height=frame_height,
            vehicle_x=vehicle_x,
            vehicle_center_x=vehicle_center_x,
            reference_y=reference_y,
        )


# =============================================================================
# FACTORY
# =============================================================================

def create_default_lane_assignment(
    **kwargs,
) -> LaneAssignment:
    return LaneAssignment(
        **kwargs
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "LaneAssignmentResult",
    "LaneAssignment",
    "create_default_lane_assignment",
]