"""
Forza Assistents
ADAS / LKA state estimation.

Responsabilidades:
- Determinar a posição do veículo dentro da faixa atual.
- Usar LaneAssignment como fonte primária da posição lateral.
- Detectar aproximação da linha esquerda/direita.
- Detectar saída iminente da faixa.
- Considerar erro lateral + heading.
- Aplicar histerese temporal.
- Produzir um estado ADAS estável.

Convenção obrigatória:

    normalized_offset < 0  -> esquerda
    normalized_offset = 0  -> centro
    normalized_offset > 0  -> direita

Este módulo NÃO controla o veículo.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from .lane_geometry import LaneGeometryResult
from .lane_assignment import LaneAssignmentResult


# =============================================================================
# ESTADOS
# =============================================================================


class ADASState(Enum):
    UNKNOWN = "unknown"
    LANE_LOST = "lane_lost"

    CENTERED = "centered"

    SLIGHT_LEFT = "slight_left"
    SLIGHT_RIGHT = "slight_right"

    LEFT_WARNING = "left_warning"
    RIGHT_WARNING = "right_warning"

    LEFT_DEPARTURE = "left_departure"
    RIGHT_DEPARTURE = "right_departure"


class LaneSide(Enum):
    NONE = "none"
    LEFT = "left"
    RIGHT = "right"


# =============================================================================
# RESULTADO
# =============================================================================


@dataclass(frozen=True)
class ADASStateResult:
    state: ADASState
    warning_side: LaneSide

    lateral_error: float
    heading_error: float

    left_distance: float
    right_distance: float

    left_approach_rate: float
    right_approach_rate: float

    confidence: float
    valid: bool

    timestamp: float


# =============================================================================
# ESTIMADOR
# =============================================================================


class ADASStateEstimator:
    """
    Estima o estado ADAS.

    Fonte da posição lateral:

    1. LaneAssignmentResult.normalized_offset
    2. fallback para LaneGeometryResult.lateral_error

    Convenção:

        negativo = esquerda
        positivo = direita
    """

    def __init__(
        self,
        warning_threshold: float = 0.55,
        departure_threshold: float = 0.82,
        centered_threshold: float = 0.20,
        slight_threshold: float = 0.38,
        heading_warning_threshold: float = 0.35,
        min_confidence: float = 0.35,
        state_hold_time: float = 0.12,
        lost_timeout: float = 0.35,
        approach_warning_rate: float = 0.70,
        approach_departure_rate: float = 1.20,
    ):
        self.warning_threshold = float(
            np.clip(warning_threshold, 0.0, 1.0)
        )

        self.departure_threshold = float(
            np.clip(
                departure_threshold,
                self.warning_threshold,
                1.0,
            )
        )

        self.centered_threshold = float(
            np.clip(
                centered_threshold,
                0.0,
                self.warning_threshold,
            )
        )

        self.slight_threshold = float(
            np.clip(
                slight_threshold,
                self.centered_threshold,
                self.warning_threshold,
            )
        )

        self.heading_warning_threshold = float(
            np.clip(
                heading_warning_threshold,
                0.0,
                1.0,
            )
        )

        self.min_confidence = float(
            np.clip(min_confidence, 0.0, 1.0)
        )

        self.state_hold_time = max(
            0.0,
            float(state_hold_time),
        )

        self.lost_timeout = max(
            0.0,
            float(lost_timeout),
        )

        self.approach_warning_rate = max(
            0.0,
            float(approach_warning_rate),
        )

        self.approach_departure_rate = max(
            self.approach_warning_rate,
            float(approach_departure_rate),
        )

        # Estado atual.
        self._state = ADASState.UNKNOWN
        self._warning_side = LaneSide.NONE

        # Histerese.
        self._candidate_state = ADASState.UNKNOWN
        self._candidate_since = 0.0

        # Tracking temporal.
        self._last_valid_time: Optional[float] = None

        self._previous_lateral_error: Optional[float] = None
        self._previous_timestamp: Optional[float] = None

        self._last_result: Optional[ADASStateResult] = None

    # =========================================================================
    # PROPRIEDADES
    # =========================================================================

    @property
    def state(self) -> ADASState:
        return self._state

    @property
    def warning_side(self) -> LaneSide:
        return self._warning_side

    @property
    def last_result(self) -> Optional[ADASStateResult]:
        return self._last_result

    # =========================================================================
    # UTILITÁRIOS
    # =========================================================================

    @staticmethod
    def _clip_error(value: float) -> float:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0.0

        if not np.isfinite(value):
            return 0.0

        return float(np.clip(value, -1.0, 1.0))

    @staticmethod
    def _safe_float(
        value: float,
        default: float = 0.0,
    ) -> float:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return default

        if not np.isfinite(value):
            return default

        return value

    # =========================================================================
    # POSIÇÃO LATERAL
    # =========================================================================

    def _get_lateral_error(
        self,
        geometry: LaneGeometryResult,
        assignment: Optional[LaneAssignmentResult],
    ) -> float:
        """
        Obtém a posição lateral do veículo.

        LaneAssignment é a fonte principal.

        Convenção:

            < 0 = esquerda
            = 0 = centro
            > 0 = direita
        """

        if assignment is not None:
            assignment_valid = bool(
                getattr(assignment, "valid", False)
            )

            if assignment_valid:
                value = getattr(
                    assignment,
                    "normalized_offset",
                    None,
                )

                if value is not None:
                    value = self._safe_float(
                        value,
                        default=0.0,
                    )

                    return self._clip_error(value)

        # Fallback para a geometria.
        return self._clip_error(
            getattr(
                geometry,
                "lateral_error",
                0.0,
            )
        )

    # =========================================================================
    # DISTÂNCIAS
    # =========================================================================

    def _compute_lane_distances(
        self,
        lateral_error: float,
    ) -> tuple[float, float]:
        """
        Converte posição lateral para distância normalizada.

        -1 = linha esquerda
         0 = centro
        +1 = linha direita

        left_distance:
            distância até a linha esquerda.

        right_distance:
            distância até a linha direita.
        """

        position = (
            self._clip_error(lateral_error) + 1.0
        ) / 2.0

        left_distance = position
        right_distance = 1.0 - position

        return (
            float(np.clip(left_distance, 0.0, 1.0)),
            float(np.clip(right_distance, 0.0, 1.0)),
        )

    # =========================================================================
    # APROXIMAÇÃO
    # =========================================================================

    def _compute_approach_rates(
        self,
        lateral_error: float,
        timestamp: float,
    ) -> tuple[float, float]:
        """
        Calcula a velocidade de aproximação das linhas.

        Movimento para esquerda:
            lateral_error diminui
            left_rate > 0

        Movimento para direita:
            lateral_error aumenta
            right_rate > 0
        """

        previous_error = self._previous_lateral_error
        previous_time = self._previous_timestamp

        self._previous_lateral_error = lateral_error
        self._previous_timestamp = timestamp

        if previous_error is None or previous_time is None:
            return 0.0, 0.0

        dt = timestamp - previous_time

        if dt <= 0.001 or dt > 1.0:
            return 0.0, 0.0

        rate = (
            lateral_error - previous_error
        ) / dt

        left_rate = max(
            0.0,
            -rate,
        )

        right_rate = max(
            0.0,
            rate,
        )

        return (
            float(left_rate),
            float(right_rate),
        )

    # =========================================================================
    # CONFIANÇA
    # =========================================================================

    def _compute_confidence(
        self,
        geometry: LaneGeometryResult,
        assignment: Optional[LaneAssignmentResult],
    ) -> float:
        """
        Calcula a confiança da estimativa.

        Quando o Assignment é válido, sua confiança tem prioridade.
        """

        geometry_valid = bool(
            getattr(
                geometry,
                "valid",
                False,
            )
        )

        if not geometry_valid:
            return 0.0

        geometry_confidence = self._safe_float(
            getattr(
                geometry,
                "confidence",
                1.0,
            ),
            default=1.0,
        )

        geometry_confidence = float(
            np.clip(
                geometry_confidence,
                0.0,
                1.0,
            )
        )

        if assignment is not None and bool(
            getattr(
                assignment,
                "valid",
                False,
            )
        ):
            assignment_confidence = self._safe_float(
                getattr(
                    assignment,
                    "confidence",
                    1.0,
                ),
                default=1.0,
            )

            assignment_confidence = float(
                np.clip(
                    assignment_confidence,
                    0.0,
                    1.0,
                )
            )

            return float(
                np.sqrt(
                    geometry_confidence
                    * assignment_confidence
                )
            )

        return geometry_confidence

    # =========================================================================
    # CLASSIFICAÇÃO
    # =========================================================================

    def _classify_state(
        self,
        lateral_error: float,
        heading_error: float,
        left_approach_rate: float,
        right_approach_rate: float,
    ) -> tuple[ADASState, LaneSide]:

        # ---------------------------------------------------------------------
        # DEPARTURE
        # ---------------------------------------------------------------------

        if lateral_error <= -self.departure_threshold:
            return (
                ADASState.LEFT_DEPARTURE,
                LaneSide.LEFT,
            )

        if lateral_error >= self.departure_threshold:
            return (
                ADASState.RIGHT_DEPARTURE,
                LaneSide.RIGHT,
            )

        # ---------------------------------------------------------------------
        # APROXIMAÇÃO MUITO RÁPIDA
        # ---------------------------------------------------------------------

        if (
            left_approach_rate
            >= self.approach_departure_rate
            and lateral_error < 0.0
        ):
            return (
                ADASState.LEFT_DEPARTURE,
                LaneSide.LEFT,
            )

        if (
            right_approach_rate
            >= self.approach_departure_rate
            and lateral_error > 0.0
        ):
            return (
                ADASState.RIGHT_DEPARTURE,
                LaneSide.RIGHT,
            )

        # ---------------------------------------------------------------------
        # WARNING
        # ---------------------------------------------------------------------

        if lateral_error <= -self.warning_threshold:
            return (
                ADASState.LEFT_WARNING,
                LaneSide.LEFT,
            )

        if lateral_error >= self.warning_threshold:
            return (
                ADASState.RIGHT_WARNING,
                LaneSide.RIGHT,
            )

        # Aproximação rápida.
        if (
            left_approach_rate
            >= self.approach_warning_rate
            and lateral_error
            <= -self.centered_threshold
        ):
            return (
                ADASState.LEFT_WARNING,
                LaneSide.LEFT,
            )

        if (
            right_approach_rate
            >= self.approach_warning_rate
            and lateral_error
            >= self.centered_threshold
        ):
            return (
                ADASState.RIGHT_WARNING,
                LaneSide.RIGHT,
            )

        # ---------------------------------------------------------------------
        # SLIGHT
        # ---------------------------------------------------------------------

        if lateral_error <= -self.slight_threshold:
            return (
                ADASState.SLIGHT_LEFT,
                LaneSide.NONE,
            )

        if lateral_error >= self.slight_threshold:
            return (
                ADASState.SLIGHT_RIGHT,
                LaneSide.NONE,
            )

        # ---------------------------------------------------------------------
        # HEADING
        # ---------------------------------------------------------------------

        if (
            lateral_error
            <= -self.centered_threshold
            and heading_error
            <= -self.heading_warning_threshold
        ):
            return (
                ADASState.SLIGHT_LEFT,
                LaneSide.NONE,
            )

        if (
            lateral_error
            >= self.centered_threshold
            and heading_error
            >= self.heading_warning_threshold
        ):
            return (
                ADASState.SLIGHT_RIGHT,
                LaneSide.NONE,
            )

        # ---------------------------------------------------------------------
        # CENTER
        # ---------------------------------------------------------------------

        return (
            ADASState.CENTERED,
            LaneSide.NONE,
        )

    # =========================================================================
    # HISTERESE
    # =========================================================================

    def _apply_state_hysteresis(
        self,
        proposed_state: ADASState,
        proposed_side: LaneSide,
        timestamp: float,
    ) -> tuple[ADASState, LaneSide]:

        immediate_states = {
            ADASState.LEFT_DEPARTURE,
            ADASState.RIGHT_DEPARTURE,
            ADASState.LEFT_WARNING,
            ADASState.RIGHT_WARNING,
            ADASState.LANE_LOST,
        }

        if self._state == ADASState.UNKNOWN:
            self._state = proposed_state
            self._warning_side = proposed_side

            self._candidate_state = proposed_state
            self._candidate_since = timestamp

            return (
                self._state,
                self._warning_side,
            )

        if proposed_state == self._state:
            self._candidate_state = proposed_state
            self._candidate_since = timestamp

            self._warning_side = proposed_side

            return (
                self._state,
                self._warning_side,
            )

        if proposed_state in immediate_states:
            self._state = proposed_state
            self._warning_side = proposed_side

            self._candidate_state = proposed_state
            self._candidate_since = timestamp

            return (
                self._state,
                self._warning_side,
            )

        if proposed_state != self._candidate_state:
            self._candidate_state = proposed_state
            self._candidate_since = timestamp

            return (
                self._state,
                self._warning_side,
            )

        if (
            timestamp - self._candidate_since
            >= self.state_hold_time
        ):
            self._state = proposed_state
            self._warning_side = proposed_side

        return (
            self._state,
            self._warning_side,
        )

    # =========================================================================
    # UPDATE
    # =========================================================================

    def update(
        self,
        geometry: Optional[LaneGeometryResult],
        assignment: Optional[LaneAssignmentResult] = None,
        timestamp: Optional[float] = None,
    ) -> ADASStateResult:

        now = (
            time.monotonic()
            if timestamp is None
            else float(timestamp)
        )

        # ---------------------------------------------------------------------
        # GEOMETRIA AUSENTE
        # ---------------------------------------------------------------------

        if geometry is None:
            return self._handle_invalid(
                now,
                confidence=0.0,
            )

        geometry_valid = bool(
            getattr(
                geometry,
                "valid",
                False,
            )
        )

        if not geometry_valid:
            return self._handle_invalid(
                now,
                confidence=0.0,
            )

        # ---------------------------------------------------------------------
        # POSIÇÃO LATERAL
        # ---------------------------------------------------------------------

        lateral_error = self._get_lateral_error(
            geometry,
            assignment,
        )

        # ---------------------------------------------------------------------
        # HEADING
        # ---------------------------------------------------------------------

        heading_error = self._clip_error(
            getattr(
                geometry,
                "heading_error",
                0.0,
            )
        )

        # ---------------------------------------------------------------------
        # CONFIANÇA
        # ---------------------------------------------------------------------

        confidence = self._compute_confidence(
            geometry,
            assignment,
        )

        # ---------------------------------------------------------------------
        # APROXIMAÇÃO
        # ---------------------------------------------------------------------

        left_approach_rate, right_approach_rate = (
            self._compute_approach_rates(
                lateral_error,
                now,
            )
        )

        # ---------------------------------------------------------------------
        # DISTÂNCIAS
        # ---------------------------------------------------------------------

        left_distance, right_distance = (
            self._compute_lane_distances(
                lateral_error,
            )
        )

        # ---------------------------------------------------------------------
        # VALIDADE
        # ---------------------------------------------------------------------

        valid = (
            confidence >= self.min_confidence
        )

        if not valid:
            return self._handle_invalid(
                now,
                confidence=confidence,
                lateral_error=lateral_error,
                heading_error=heading_error,
                left_distance=left_distance,
                right_distance=right_distance,
                left_approach_rate=left_approach_rate,
                right_approach_rate=right_approach_rate,
            )

        self._last_valid_time = now

        # ---------------------------------------------------------------------
        # CLASSIFICAÇÃO
        # ---------------------------------------------------------------------

        proposed_state, proposed_side = (
            self._classify_state(
                lateral_error=lateral_error,
                heading_error=heading_error,
                left_approach_rate=left_approach_rate,
                right_approach_rate=right_approach_rate,
            )
        )

        state, warning_side = (
            self._apply_state_hysteresis(
                proposed_state,
                proposed_side,
                now,
            )
        )

        result = ADASStateResult(
            state=state,
            warning_side=warning_side,

            lateral_error=lateral_error,
            heading_error=heading_error,

            left_distance=left_distance,
            right_distance=right_distance,

            left_approach_rate=left_approach_rate,
            right_approach_rate=right_approach_rate,

            confidence=confidence,
            valid=True,

            timestamp=now,
        )

        self._last_result = result

        return result

    # =========================================================================
    # INVALID / LANE LOST
    # =========================================================================

    def _handle_invalid(
        self,
        timestamp: float,
        confidence: float,
        lateral_error: float = 0.0,
        heading_error: float = 0.0,
        left_distance: float = 0.5,
        right_distance: float = 0.5,
        left_approach_rate: float = 0.0,
        right_approach_rate: float = 0.0,
    ) -> ADASStateResult:

        lost = (
            self._last_valid_time is None
            or (
                timestamp - self._last_valid_time
                >= self.lost_timeout
            )
        )

        if lost:
            self._state = ADASState.LANE_LOST
            self._warning_side = LaneSide.NONE

            self._candidate_state = ADASState.LANE_LOST
            self._candidate_since = timestamp

        result = ADASStateResult(
            state=self._state,
            warning_side=self._warning_side,

            lateral_error=self._clip_error(
                lateral_error
            ),
            heading_error=self._clip_error(
                heading_error
            ),

            left_distance=float(
                np.clip(
                    left_distance,
                    0.0,
                    1.0,
                )
            ),
            right_distance=float(
                np.clip(
                    right_distance,
                    0.0,
                    1.0,
                )
            ),

            left_approach_rate=max(
                0.0,
                float(left_approach_rate),
            ),
            right_approach_rate=max(
                0.0,
                float(right_approach_rate),
            ),

            confidence=float(
                np.clip(
                    confidence,
                    0.0,
                    1.0,
                )
            ),
            valid=False,

            timestamp=timestamp,
        )

        self._last_result = result

        return result

    # =========================================================================
    # RESET
    # =========================================================================

    def reset(self) -> None:
        """Reinicia completamente o estimador."""

        self._state = ADASState.UNKNOWN
        self._warning_side = LaneSide.NONE

        self._candidate_state = ADASState.UNKNOWN
        self._candidate_since = 0.0

        self._last_valid_time = None

        self._previous_lateral_error = None
        self._previous_timestamp = None

        self._last_result = None


__all__ = [
    "ADASState",
    "LaneSide",
    "ADASStateResult",
    "ADASStateEstimator",
]