"""
Forza Horizon 6 ADAS/LKA
Estimador de estado ADAS.

Responsabilidades:
- Determinar a posição do veículo dentro da faixa.
- Detectar aproximação da linha esquerda/direita.
- Detectar saída iminente da faixa.
- Considerar erro lateral + heading.
- Aplicar histerese e persistência temporal para evitar oscilações.
- Produzir um estado estável para a futura interface ADAS.

IMPORTANTE:
Este módulo NÃO controla o veículo.
Ele apenas estima o estado visual.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from .lane_geometry import LaneGeometryResult


# =============================================================================
# ESTADOS ADAS
# =============================================================================

class ADASState(Enum):
    """Estado visual atual do veículo em relação à faixa."""

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
    """Lado da faixa em relação ao veículo."""

    NONE = "none"
    LEFT = "left"
    RIGHT = "right"


# =============================================================================
# RESULTADO
# =============================================================================

@dataclass(frozen=True)
class ADASStateResult:
    """Resultado completo da estimativa ADAS."""

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
    Estima o estado ADAS a partir da geometria da faixa.

    lateral_error:
        [-1, +1]

        negativo = veículo deslocado para a esquerda
        positivo = veículo deslocado para a direita

    heading_error:
        [-1, +1]

        negativo = orientação para esquerda
        positivo = orientação para direita

    A saída utiliza histerese temporal para impedir que o estado fique
    alternando rapidamente entre CENTERED / WARNING.
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
        # ---------------------------------------------------------------------
        # Limites espaciais
        # ---------------------------------------------------------------------

        self.warning_threshold = float(
            np.clip(warning_threshold, 0.0, 1.0)
        )

        self.departure_threshold = float(
            np.clip(
                departure_threshold,
                self.warning_threshold,
                1.0
            )
        )

        self.centered_threshold = float(
            np.clip(centered_threshold, 0.0, self.warning_threshold)
        )

        self.slight_threshold = float(
            np.clip(
                slight_threshold,
                self.centered_threshold,
                self.warning_threshold
            )
        )

        self.heading_warning_threshold = float(
            np.clip(heading_warning_threshold, 0.0, 1.0)
        )

        self.min_confidence = float(
            np.clip(min_confidence, 0.0, 1.0)
        )

        # ---------------------------------------------------------------------
        # Filtros temporais
        # ---------------------------------------------------------------------

        self.state_hold_time = max(0.0, float(state_hold_time))
        self.lost_timeout = max(0.0, float(lost_timeout))

        # Taxa de aproximação:
        # valor positivo significa aproximação da respectiva linha.
        self.approach_warning_rate = max(
            0.0,
            float(approach_warning_rate)
        )

        self.approach_departure_rate = max(
            self.approach_warning_rate,
            float(approach_departure_rate)
        )

        # ---------------------------------------------------------------------
        # Estado interno
        # ---------------------------------------------------------------------

        self._state = ADASState.UNKNOWN
        self._warning_side = LaneSide.NONE

        self._candidate_state = ADASState.UNKNOWN
        self._candidate_since = 0.0

        self._last_valid_time: Optional[float] = None

        self._previous_lateral_error: Optional[float] = None
        self._previous_timestamp: Optional[float] = None

        self._last_result: Optional[ADASStateResult] = None

    # =========================================================================
    # UTILITÁRIOS
    # =========================================================================

    @staticmethod
    def _clip_error(value: float) -> float:
        """Limita um erro normalizado ao intervalo [-1, +1]."""
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0.0

        if not np.isfinite(value):
            return 0.0

        return float(np.clip(value, -1.0, 1.0))

    @staticmethod
    def _safe_float(value: float, default: float = 0.0) -> float:
        """Converte valores numéricos evitando NaN/inf."""
        try:
            value = float(value)
        except (TypeError, ValueError):
            return default

        if not np.isfinite(value):
            return default

        return value

    # =========================================================================
    # DISTÂNCIAS
    # =========================================================================

    def _compute_lane_distances(
        self,
        lateral_error: float
    ) -> tuple[float, float]:
        """
        Converte lateral_error em distância normalizada às bordas.

        lateral_error = -1:
            veículo extremamente próximo da esquerda.

        lateral_error = +1:
            veículo extremamente próximo da direita.

        Retorno:
            left_distance
            right_distance

        Valores:
            0.0 = borda
            1.0 = centro aproximado / maior distância
        """

        # Espaço normalizado entre -1 e +1.
        vehicle_position = (lateral_error + 1.0) / 2.0

        left_distance = vehicle_position
        right_distance = 1.0 - vehicle_position

        return (
            float(np.clip(left_distance, 0.0, 1.0)),
            float(np.clip(right_distance, 0.0, 1.0)),
        )

    # =========================================================================
    # TAXA DE APROXIMAÇÃO
    # =========================================================================

    def _compute_approach_rates(
        self,
        lateral_error: float,
        timestamp: float,
    ) -> tuple[float, float]:
        """
        Calcula velocidade de aproximação das linhas.

        Retorno:
            left_approach_rate
            right_approach_rate

        Valores positivos indicam aproximação.

        Exemplo:

            lateral_error = -0.20 -> -0.50

            veículo está indo para a esquerda.

            left_approach_rate > 0
        """

        if (
            self._previous_lateral_error is None
            or self._previous_timestamp is None
        ):
            self._previous_lateral_error = lateral_error
            self._previous_timestamp = timestamp
            return 0.0, 0.0

        dt = timestamp - self._previous_timestamp

        if dt <= 0.001 or dt > 1.0:
            self._previous_lateral_error = lateral_error
            self._previous_timestamp = timestamp
            return 0.0, 0.0

        delta = lateral_error - self._previous_lateral_error

        # delta positivo = movimento para direita
        # delta negativo = movimento para esquerda
        rate = delta / dt

        self._previous_lateral_error = lateral_error
        self._previous_timestamp = timestamp

        # Aproximação esquerda ocorre quando rate < 0.
        left_rate = max(0.0, -rate)

        # Aproximação direita ocorre quando rate > 0.
        right_rate = max(0.0, rate)

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
    ) -> float:
        """
        Estima confiança do estado geométrico.

        A confiança considera:
        - validade da geometria;
        - largura da faixa;
        - erro lateral;
        - presença da linha central.
        """

        if not geometry.valid:
            return 0.0

        score = 1.0

        # Erros extremamente grandes reduzem confiança.
        lateral = abs(
            self._clip_error(geometry.lateral_error)
        )

        score *= max(
            0.0,
            1.0 - 0.25 * lateral
        )

        # Centro da faixa precisa existir.
        if not geometry.center_line:
            score *= 0.5

        # Largura inválida reduz confiança.
        lane_width = self._safe_float(
            geometry.lane_width
        )

        if lane_width <= 1.0:
            score *= 0.4

        return float(np.clip(score, 0.0, 1.0))

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
        """Classifica o estado instantâneo."""

        abs_lateral = abs(lateral_error)
        abs_heading = abs(heading_error)

        # ---------------------------------------------------------------------
        # SAÍDA IMINENTE
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

        if left_approach_rate >= self.approach_departure_rate:
            if lateral_error <= -self.centered_threshold:
                return (
                    ADASState.LEFT_DEPARTURE,
                    LaneSide.LEFT,
                )

        if right_approach_rate >= self.approach_departure_rate:
            if lateral_error >= self.centered_threshold:
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

        # Aproximação rápida também pode gerar warning antes do limite
        # espacial, desde que o veículo esteja realmente deslocado.
        if (
            left_approach_rate >= self.approach_warning_rate
            and lateral_error < -self.centered_threshold
        ):
            return (
                ADASState.LEFT_WARNING,
                LaneSide.LEFT,
            )

        if (
            right_approach_rate >= self.approach_warning_rate
            and lateral_error > self.centered_threshold
        ):
            return (
                ADASState.RIGHT_WARNING,
                LaneSide.RIGHT,
            )

        # ---------------------------------------------------------------------
        # DESLOCAMENTO LEVE
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

        # Heading sozinho não deve provocar WARNING.
        # Ele só reforça um deslocamento lateral existente.
        if abs_lateral >= self.centered_threshold:
            if (
                lateral_error < 0.0
                and heading_error < -self.heading_warning_threshold
            ):
                return (
                    ADASState.SLIGHT_LEFT,
                    LaneSide.NONE,
                )

            if (
                lateral_error > 0.0
                and heading_error > self.heading_warning_threshold
            ):
                return (
                    ADASState.SLIGHT_RIGHT,
                    LaneSide.NONE,
                )

        # ---------------------------------------------------------------------
        # CENTRADO
        # ---------------------------------------------------------------------

        if abs_lateral <= self.centered_threshold:
            return (
                ADASState.CENTERED,
                LaneSide.NONE,
            )

        # Segurança: classificação residual.
        if lateral_error < 0:
            return (
                ADASState.SLIGHT_LEFT,
                LaneSide.NONE,
            )

        return (
            ADASState.SLIGHT_RIGHT,
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
        """
        Evita troca de estado em cada frame.

        Estados de emergência/departure entram imediatamente.
        Estados normais precisam permanecer estáveis durante
        state_hold_time.
        """

        immediate_states = {
            ADASState.LEFT_DEPARTURE,
            ADASState.RIGHT_DEPARTURE,
            ADASState.LEFT_WARNING,
            ADASState.RIGHT_WARNING,
            ADASState.LANE_LOST,
        }

        # Primeiro estado.
        if self._state == ADASState.UNKNOWN:
            self._state = proposed_state
            self._warning_side = proposed_side
            self._candidate_state = proposed_state
            self._candidate_since = timestamp
            return self._state, self._warning_side

        # Mesmo estado.
        if proposed_state == self._state:
            self._candidate_state = proposed_state
            self._candidate_since = timestamp
            self._warning_side = proposed_side
            return self._state, self._warning_side

        # Estados críticos entram imediatamente.
        if proposed_state in immediate_states:
            self._state = proposed_state
            self._warning_side = proposed_side
            self._candidate_state = proposed_state
            self._candidate_since = timestamp
            return self._state, self._warning_side

        # Novo candidato.
        if proposed_state != self._candidate_state:
            self._candidate_state = proposed_state
            self._candidate_since = timestamp
            return self._state, self._warning_side

        # Aguarda estabilidade.
        if timestamp - self._candidate_since >= self.state_hold_time:
            self._state = proposed_state
            self._warning_side = proposed_side

        return self._state, self._warning_side

    # =========================================================================
    # UPDATE
    # =========================================================================

    def update(
        self,
        geometry: Optional[LaneGeometryResult],
        timestamp: Optional[float] = None,
    ) -> ADASStateResult:
        """
        Atualiza o estado ADAS.

        Este é o método principal utilizado pelo pipeline.
        """

        now = (
            time.perf_counter()
            if timestamp is None
            else float(timestamp)
        )

        # ---------------------------------------------------------------------
        # GEOMETRIA AUSENTE
        # ---------------------------------------------------------------------

        if geometry is None or not geometry.valid:
            if self._last_valid_time is None:
                lost_duration = float("inf")
            else:
                lost_duration = now - self._last_valid_time

            # Mantém o último estado por um curto período.
            if (
                self._state not in {
                    ADASState.UNKNOWN,
                    ADASState.LANE_LOST,
                }
                and lost_duration < self.lost_timeout
            ):
                if self._last_result is not None:
                    return ADASStateResult(
                        state=self._state,
                        warning_side=self._warning_side,
                        lateral_error=self._last_result.lateral_error,
                        heading_error=self._last_result.heading_error,
                        left_distance=self._last_result.left_distance,
                        right_distance=self._last_result.right_distance,
                        left_approach_rate=self._last_result.left_approach_rate,
                        right_approach_rate=self._last_result.right_approach_rate,
                        confidence=0.0,
                        valid=False,
                        timestamp=now,
                    )

            self._state = ADASState.LANE_LOST
            self._warning_side = LaneSide.NONE
            self._candidate_state = ADASState.LANE_LOST
            self._candidate_since = now

            result = ADASStateResult(
                state=ADASState.LANE_LOST,
                warning_side=LaneSide.NONE,
                lateral_error=0.0,
                heading_error=0.0,
                left_distance=0.0,
                right_distance=0.0,
                left_approach_rate=0.0,
                right_approach_rate=0.0,
                confidence=0.0,
                valid=False,
                timestamp=now,
            )

            self._last_result = result
            return result

        # ---------------------------------------------------------------------
        # GEOMETRIA VÁLIDA
        # ---------------------------------------------------------------------

        self._last_valid_time = now

        lateral_error = self._clip_error(
            geometry.lateral_error
        )

        heading_error = self._clip_error(
            geometry.heading_error
        )

        left_distance, right_distance = (
            self._compute_lane_distances(lateral_error)
        )

        left_rate, right_rate = self._compute_approach_rates(
            lateral_error,
            now,
        )

        confidence = self._compute_confidence(
            geometry
        )

        # Confiança insuficiente.
        if confidence < self.min_confidence:
            self._state = ADASState.LANE_LOST
            self._warning_side = LaneSide.NONE

            result = ADASStateResult(
                state=ADASState.LANE_LOST,
                warning_side=LaneSide.NONE,
                lateral_error=lateral_error,
                heading_error=heading_error,
                left_distance=left_distance,
                right_distance=right_distance,
                left_approach_rate=left_rate,
                right_approach_rate=right_rate,
                confidence=confidence,
                valid=False,
                timestamp=now,
            )

            self._last_result = result
            return result

        # ---------------------------------------------------------------------
        # CLASSIFICAÇÃO
        # ---------------------------------------------------------------------

        proposed_state, proposed_side = self._classify_state(
            lateral_error=lateral_error,
            heading_error=heading_error,
            left_approach_rate=left_rate,
            right_approach_rate=right_rate,
        )

        state, warning_side = self._apply_state_hysteresis(
            proposed_state=proposed_state,
            proposed_side=proposed_side,
            timestamp=now,
        )

        result = ADASStateResult(
            state=state,
            warning_side=warning_side,
            lateral_error=lateral_error,
            heading_error=heading_error,
            left_distance=left_distance,
            right_distance=right_distance,
            left_approach_rate=left_rate,
            right_approach_rate=right_rate,
            confidence=confidence,
            valid=True,
            timestamp=now,
        )

        self._last_result = result
        return result

    # =========================================================================
    # RESET
    # =========================================================================

    def reset(self) -> None:
        """Reseta completamente o estimador."""

        self._state = ADASState.UNKNOWN
        self._warning_side = LaneSide.NONE

        self._candidate_state = ADASState.UNKNOWN
        self._candidate_since = 0.0

        self._last_valid_time = None

        self._previous_lateral_error = None
        self._previous_timestamp = None

        self._last_result = None

    # =========================================================================
    # PROPRIEDADES
    # =========================================================================

    @property
    def state(self) -> ADASState:
        """Estado atualmente confirmado."""
        return self._state

    @property
    def warning_side(self) -> LaneSide:
        """Lado atualmente em alerta."""
        return self._warning_side

    @property
    def last_result(self) -> Optional[ADASStateResult]:
        """Último resultado calculado."""
        return self._last_result