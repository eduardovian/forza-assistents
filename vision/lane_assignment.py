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
    - calcula geometria da faixa;
    - calcula heading;
    - decide atuação ADAS.

A estrutura de dados utilizada é a definida em lane_types.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .lane_types import (
    CurrentLane,
    LaneAssociationResult,
    LaneModel,
    LaneSide,
    calculate_lane_center,
    calculate_normalized_offset,
)

logger = logging.getLogger(__name__)


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
# ESTRUTURA INTERNA
# =============================================================================

@dataclass(frozen=True)
class _BoundarySample:
    lane: LaneModel
    x: float
    confidence: float


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
    Obtém a posição X da linha em um Y de referência.

    Prioridade:

        polynomial
        ↓
        último ponto válido
    """

    if lane is None:
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
            else:
                target_y = None

        if target_y is not None:

            x = polynomial.evaluate(
                float(target_y)
            )

            if _finite(x):
                return float(x)

    points = [
        point
        for point in lane.line.points
        if point.valid
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

    if lane is None:
        return 0.0

    values = [
        lane.valid,
        lane.line.valid,
        lane.line.confidence,
    ]

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
            LaneAssociationResult
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
        Ordena as linhas da esquerda para a direita.

        O lane_id original é preservado. A ordenação espacial
        é feita pela posição X no ponto de referência.
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

        Se nenhum centro externo for fornecido, assume o centro
        horizontal da câmera.
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

            center_x = calculate_lane_center(
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

        # Primeiro: veículo dentro do corredor.
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

        # Caso o veículo esteja ligeiramente fora
        # devido a erro de projeção.
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
    # CURRENT LANE
    # =========================================================================

    def _build_current_lane(
        self,
        candidate: _LaneCandidate,
        vehicle_x: float,
    ) -> CurrentLane:

        normalized_offset = (
            calculate_normalized_offset(
                vehicle_x=vehicle_x,
                center_x=candidate.center_x,
                lane_width=candidate.width,
            )
        )

        lateral_offset = (
            vehicle_x
            - candidate.center_x
        )

        return CurrentLane(
            left_boundary=candidate.left,
            right_boundary=candidate.right,
            center_x=candidate.center_x,
            lane_width=candidate.width,
            lateral_offset=float(
                lateral_offset
            ),
            normalized_offset=(
                normalized_offset
            ),
            confidence=candidate.confidence,
            valid=True,
        )

    # =========================================================================
    # RESULTADO
    # =========================================================================

    def assign(
        self,
        lanes: Iterable[LaneModel],
        frame_width: float,
        frame_height: Optional[float] = None,
        vehicle_x: Optional[float] = None,
        vehicle_center_x: Optional[float] = None,
        reference_y: Optional[float] = None,
    ) -> LaneAssociationResult:
        """
        Executa a associação completa.

        frame_height é utilizado para escolher a região inferior
        da imagem quando nenhum Y de referência é fornecido.
        """

        try:

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

                result = LaneAssociationResult(
                    lanes=list(normalized),
                    current_lane=None,
                    current_lane_id=None,
                    left_lanes=[],
                    right_lanes=[],
                    valid=False,
                    confidence=0.0,
                )

                self.last_result = result

                return result

            candidate_position = (
                self._find_candidate(
                    candidates,
                    estimated_vehicle_x,
                )
            )

            if candidate_position is None:

                result = LaneAssociationResult(
                    lanes=list(normalized),
                    valid=False,
                    confidence=0.0,
                )

                self.last_result = result

                return result

            candidate_lane = candidates[
                candidate_position
            ]

            candidate_lane_id = (
                candidate_lane.left.lane_id
            )

            (
                stable_lane_id,
                stable,
                change_pending,
            ) = self._stabilize_lane(
                candidate_lane_id
            )

            stable_position = None

            for index, candidate in enumerate(
                candidates
            ):

                if (
                    candidate.left.lane_id
                    == stable_lane_id
                ):
                    stable_position = index
                    break

            if stable_position is None:

                stable_position = (
                    candidate_position
                )

            current_candidate = candidates[
                stable_position
            ]

            current_lane = (
                self._build_current_lane(
                    current_candidate,
                    estimated_vehicle_x,
                )
            )

            left_lanes = []

            right_lanes = []

            for index, candidate in enumerate(
                candidates
            ):

                if index < stable_position:

                    left_lanes.extend(
                        [
                            candidate.left,
                        ]
                    )

                elif index > stable_position:

                    right_lanes.extend(
                        [
                            candidate.right,
                        ]
                    )

            # Adiciona linhas que estão mais externas
            # à estrutura de faixas identificada.
            ordered_lines = list(normalized)

            try:
                current_left_index = (
                    ordered_lines.index(
                        current_candidate.left
                    )
                )

                current_right_index = (
                    ordered_lines.index(
                        current_candidate.right
                    )
                )

                left_lanes = [
                    lane
                    for lane in ordered_lines[
                        :current_left_index
                    ]
                ]

                right_lanes = [
                    lane
                    for lane in ordered_lines[
                        current_right_index + 1:
                    ]
                ]

            except ValueError:
                pass

            confidence = (
                current_lane.confidence
            )

            if change_pending:
                confidence *= 0.85

            result = LaneAssociationResult(
                lanes=list(normalized),
                current_lane=current_lane,
                current_lane_id=stable_lane_id,
                left_lanes=left_lanes,
                right_lanes=right_lanes,
                valid=current_lane.valid,
                confidence=float(
                    np.clip(
                        confidence,
                        0.0,
                        1.0,
                    )
                ),
            )

            self.last_result = result

            return result

        except Exception as exc:

            logger.exception(
                "[LANE_ASSIGNMENT] "
                "Falha na associação."
            )

            result = LaneAssociationResult(
                valid=False,
                confidence=0.0,
            )

            self.last_result = result

            return result

    # =========================================================================
    # ALIASES DE COMPATIBILIDADE
    # =========================================================================

    def update(
        self,
        lanes: Iterable[LaneModel],
        frame_width: float,
        frame_height: Optional[float] = None,
        vehicle_x: Optional[float] = None,
        vehicle_center_x: Optional[float] = None,
        reference_y: Optional[float] = None,
    ) -> LaneAssociationResult:

        return self.assign(
            lanes=lanes,
            frame_width=frame_width,
            frame_height=frame_height,
            vehicle_x=vehicle_x,
            vehicle_center_x=vehicle_center_x,
            reference_y=reference_y,
        )

    def process(
        self,
        lanes: Iterable[LaneModel],
        frame_width: float,
        frame_height: Optional[float] = None,
        vehicle_x: Optional[float] = None,
        vehicle_center_x: Optional[float] = None,
        reference_y: Optional[float] = None,
    ) -> LaneAssociationResult:

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


__all__ = [
    "LaneAssignment",
    "create_default_lane_assignment",
]