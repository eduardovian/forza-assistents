"""
core/safety_gate.py

Safety Gate da arquitetura ADAS/LKA.

Responsabilidade:

    ADASDecisionResult
            ↓
       Safety Gate
            ↓
      SafetyGateResult
            ↓
        atuação

O Safety Gate é a última barreira antes de qualquer atuação.

Este módulo NÃO:
    - detecta lanes;
    - calcula geometria;
    - toma a decisão ADAS;
    - envia comandos ao veículo.

Ele apenas determina se a decisão recebida está autorizada
a prosseguir para a camada de atuação.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


# =============================================================================
# ESTADOS
# =============================================================================


class SafetyGateState(str, Enum):
    """Estado do bloqueio de segurança."""

    BLOCKED = "blocked"
    ALLOWED = "allowed"
    DEGRADED = "degraded"


class SafetyBlockReason(str, Enum):
    """Motivo pelo qual uma atuação pode ser bloqueada."""

    NONE = "none"

    NO_DECISION = "no_decision"
    INVALID_DECISION = "invalid_decision"
    LOW_CONFIDENCE = "low_confidence"

    PERCEPTION_UNSAFE = "perception_unsafe"

    DEPARTURE_UNSAFE = "departure_unsafe"

    INVALID_OFFSET = "invalid_offset"
    INVALID_HEADING = "invalid_heading"

    INVALID_ACTION = "invalid_action"

    SYSTEM_DISABLED = "system_disabled"


# =============================================================================
# RESULTADO
# =============================================================================


@dataclass(frozen=True)
class SafetyGateResult:
    """
    Resultado da avaliação de segurança.

    allowed:
        Indica se a ação recebida pode prosseguir.

    state:
        Estado geral do Safety Gate.

    action:
        Ação liberada. Será None quando bloqueada.

    reason:
        Motivo textual para diagnóstico.
    """

    allowed: bool

    state: SafetyGateState

    reason_code: SafetyBlockReason

    reason: str

    action: Optional[object]

    confidence: float

    lateral_offset: float

    heading_error: float

    degraded: bool

    valid: bool


# =============================================================================
# SAFETY GATE
# =============================================================================


class SafetyGate:
    """
    Barreira final de segurança antes da atuação.

    Regra fundamental:

        decisão inválida
            → BLOCK

        confiança insuficiente
            → BLOCK

        percepção insegura
            → BLOCK

        valores matematicamente inválidos
            → BLOCK

        caso válido e seguro
            → ALLOW

    O Safety Gate nunca cria uma ação nova.
    """

    def __init__(
        self,
        enabled: bool = True,
        min_confidence: float = 0.70,
        max_abs_offset: float = 1.0,
        max_abs_heading: float = 1.0,
        allow_degraded: bool = False,
    ) -> None:

        self.enabled = bool(
            enabled
        )

        self.min_confidence = float(
            np.clip(
                min_confidence,
                0.0,
                1.0,
            )
        )

        self.max_abs_offset = float(
            np.clip(
                max_abs_offset,
                0.0,
                1.0,
            )
        )

        self.max_abs_heading = float(
            np.clip(
                max_abs_heading,
                0.0,
                1.0,
            )
        )

        self.allow_degraded = bool(
            allow_degraded
        )

        self.last_result: Optional[
            SafetyGateResult
        ] = None

    # =========================================================================
    # UTILITÁRIOS
    # =========================================================================

    @staticmethod
    def _safe_float(
        value: object,
        default: float = 0.0,
    ) -> float:

        try:
            value = float(
                value
            )
        except (
            TypeError,
            ValueError,
        ):
            return default

        if not np.isfinite(value):
            return default

        return value

    @staticmethod
    def _get(
        obj: object,
        *names: str,
        default=None,
    ):
        """
        Retorna o primeiro atributo existente.
        """

        if obj is None:
            return default

        for name in names:

            if hasattr(
                obj,
                name,
            ):

                value = getattr(
                    obj,
                    name,
                )

                if value is not None:
                    return value

        return default

    # =========================================================================
    # EXTRAÇÃO
    # =========================================================================

    def _extract_values(
        self,
        decision: object,
    ) -> tuple[
        bool,
        bool,
        bool,
        float,
        float,
        float,
        Optional[object],
    ]:
        """
        Extrai os campos relevantes da decisão.

        Retorno:

            valid
            safe
            departure
            confidence
            lateral_offset
            heading_error
            action
        """

        valid = bool(
            self._get(
                decision,
                "valid",
                default=False,
            )
        )

        safe = bool(
            self._get(
                decision,
                "safe",
                default=False,
            )
        )

        departure = bool(
            self._get(
                decision,
                "departure",
                default=False,
            )
        )

        confidence = self._safe_float(
            self._get(
                decision,
                "confidence",
                default=0.0,
            )
        )

        lateral_offset = self._safe_float(
            self._get(
                decision,
                "lateral_offset",
                default=0.0,
            )
        )

        heading_error = self._safe_float(
            self._get(
                decision,
                "heading_error",
                default=0.0,
            )
        )

        action = self._get(
            decision,
            "action",
            default=None,
        )

        return (
            valid,
            safe,
            departure,
            confidence,
            lateral_offset,
            heading_error,
            action,
        )

    # =========================================================================
    # RESULTADOS BLOQUEADOS
    # =========================================================================

    def _blocked(
        self,
        reason_code: SafetyBlockReason,
        reason: str,
        confidence: float = 0.0,
        lateral_offset: float = 0.0,
        heading_error: float = 0.0,
        action: Optional[object] = None,
        degraded: bool = False,
        valid: bool = False,
    ) -> SafetyGateResult:

        result = SafetyGateResult(
            allowed=False,
            state=(
                SafetyGateState.DEGRADED
                if degraded
                else SafetyGateState.BLOCKED
            ),
            reason_code=reason_code,
            reason=reason,
            action=None,
            confidence=confidence,
            lateral_offset=lateral_offset,
            heading_error=heading_error,
            degraded=degraded,
            valid=valid,
        )

        self.last_result = result

        return result

    # =========================================================================
    # AVALIAÇÃO
    # =========================================================================

    def evaluate(
        self,
        decision: object,
    ) -> SafetyGateResult:
        """
        Avalia uma decisão ADAS.

        Nenhuma ação é executada aqui.
        """

        # ---------------------------------------------------------------------
        # SISTEMA DESABILITADO
        # ---------------------------------------------------------------------

        if not self.enabled:

            return self._blocked(
                reason_code=SafetyBlockReason.SYSTEM_DISABLED,
                reason="Safety Gate desabilitado.",
            )

        # ---------------------------------------------------------------------
        # DECISÃO AUSENTE
        # ---------------------------------------------------------------------

        if decision is None:

            return self._blocked(
                reason_code=SafetyBlockReason.NO_DECISION,
                reason="Nenhuma decisão ADAS recebida.",
            )

        (
            valid,
            safe,
            departure,
            confidence,
            lateral_offset,
            heading_error,
            action,
        ) = self._extract_values(
            decision
        )

        # ---------------------------------------------------------------------
        # VALIDADE NUMÉRICA
        # ---------------------------------------------------------------------

        raw_offset = self._get(
            decision,
            "lateral_offset",
            default=0.0,
        )

        raw_heading = self._get(
            decision,
            "heading_error",
            default=0.0,
        )

        try:
            offset_finite = np.isfinite(
                float(raw_offset)
            )
        except (
            TypeError,
            ValueError,
        ):
            offset_finite = False

        try:
            heading_finite = np.isfinite(
                float(raw_heading)
            )
        except (
            TypeError,
            ValueError,
        ):
            heading_finite = False

        if not offset_finite:

            return self._blocked(
                reason_code=SafetyBlockReason.INVALID_OFFSET,
                reason="Erro lateral inválido.",
                confidence=confidence,
                lateral_offset=0.0,
                heading_error=heading_error,
                action=action,
            )

        if not heading_finite:

            return self._blocked(
                reason_code=SafetyBlockReason.INVALID_HEADING,
                reason="Erro de heading inválido.",
                confidence=confidence,
                lateral_offset=lateral_offset,
                heading_error=0.0,
                action=action,
            )

        # ---------------------------------------------------------------------
        # DECISÃO INVÁLIDA
        # ---------------------------------------------------------------------

        if not valid:

            return self._blocked(
                reason_code=SafetyBlockReason.INVALID_DECISION,
                reason="Decisão ADAS inválida.",
                confidence=confidence,
                lateral_offset=lateral_offset,
                heading_error=heading_error,
                action=action,
            )

        # ---------------------------------------------------------------------
        # AÇÃO INEXISTENTE
        # ---------------------------------------------------------------------

        if action is None:

            return self._blocked(
                reason_code=SafetyBlockReason.INVALID_ACTION,
                reason="Decisão não possui ação definida.",
                confidence=confidence,
                lateral_offset=lateral_offset,
                heading_error=heading_error,
            )

        # ---------------------------------------------------------------------
        # CONFIANÇA
        # ---------------------------------------------------------------------

        if confidence < self.min_confidence:

            return self._blocked(
                reason_code=SafetyBlockReason.LOW_CONFIDENCE,
                reason=(
                    "Confiança abaixo do limite "
                    "mínimo do Safety Gate."
                ),
                confidence=confidence,
                lateral_offset=lateral_offset,
                heading_error=heading_error,
                action=action,
                valid=True,
            )

        # ---------------------------------------------------------------------
        # LIMITES FÍSICOS / MATEMÁTICOS
        # ---------------------------------------------------------------------

        if (
            abs(lateral_offset)
            > self.max_abs_offset
        ):

            return self._blocked(
                reason_code=SafetyBlockReason.INVALID_OFFSET,
                reason=(
                    "Erro lateral excede o limite "
                    "permitido pelo Safety Gate."
                ),
                confidence=confidence,
                lateral_offset=lateral_offset,
                heading_error=heading_error,
                action=action,
                valid=True,
            )

        if (
            abs(heading_error)
            > self.max_abs_heading
        ):

            return self._blocked(
                reason_code=SafetyBlockReason.INVALID_HEADING,
                reason=(
                    "Heading excede o limite "
                    "permitido pelo Safety Gate."
                ),
                confidence=confidence,
                lateral_offset=lateral_offset,
                heading_error=heading_error,
                action=action,
                valid=True,
            )

        # ---------------------------------------------------------------------
        # PERCEPÇÃO NÃO SEGURA
        # ---------------------------------------------------------------------

        if not safe:

            if self.allow_degraded:

                result = SafetyGateResult(
                    allowed=False,
                    state=SafetyGateState.DEGRADED,
                    reason_code=SafetyBlockReason.PERCEPTION_UNSAFE,
                    reason=(
                        "Percepção válida, porém "
                        "não considerada segura."
                    ),
                    action=None,
                    confidence=confidence,
                    lateral_offset=lateral_offset,
                    heading_error=heading_error,
                    degraded=True,
                    valid=True,
                )

                self.last_result = result

                return result

            return self._blocked(
                reason_code=SafetyBlockReason.PERCEPTION_UNSAFE,
                reason=(
                    "Percepção não está liberada "
                    "para atuação."
                ),
                confidence=confidence,
                lateral_offset=lateral_offset,
                heading_error=heading_error,
                action=action,
                valid=True,
            )

        # ---------------------------------------------------------------------
        # DEPARTURE
        #
        # Uma decisão de departure não deve ser liberada automaticamente
        # para atuação por este módulo. Ela permanece bloqueada até que
        # uma política específica de atuação seja implementada.
        # ---------------------------------------------------------------------

        if departure:

            return self._blocked(
                reason_code=SafetyBlockReason.DEPARTURE_UNSAFE,
                reason=(
                    "Evento de departure requer "
                    "política específica de atuação."
                ),
                confidence=confidence,
                lateral_offset=lateral_offset,
                heading_error=heading_error,
                action=action,
                valid=True,
            )

        # ---------------------------------------------------------------------
        # LIBERAÇÃO
        # ---------------------------------------------------------------------

        result = SafetyGateResult(
            allowed=True,
            state=SafetyGateState.ALLOWED,
            reason_code=SafetyBlockReason.NONE,
            reason="Atuação liberada pelo Safety Gate.",
            action=action,
            confidence=confidence,
            lateral_offset=lateral_offset,
            heading_error=heading_error,
            degraded=False,
            valid=True,
        )

        self.last_result = result

        return result

    # =========================================================================
    # ALIASES
    # =========================================================================

    def check(
        self,
        decision: object,
    ) -> SafetyGateResult:
        """
        Alias para evaluate().
        """

        return self.evaluate(
            decision
        )

    def allow(
        self,
        decision: object,
    ) -> bool:
        """
        Retorna apenas se a decisão foi liberada.
        """

        return bool(
            self.evaluate(
                decision
            ).allowed
        )


# =============================================================================
# FACTORY
# =============================================================================


def create_default_safety_gate(
    **kwargs,
) -> SafetyGate:

    return SafetyGate(
        **kwargs
    )


__all__ = [
    "SafetyGateState",
    "SafetyBlockReason",
    "SafetyGateResult",
    "SafetyGate",
    "create_default_safety_gate",
]