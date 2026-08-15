"""
core/adas_decision.py

Camada de decisão ADAS/LKA.

Responsabilidade:

    LaneScene
        ↓
    ADAS Decision
        ↓
    ADASDecisionResult

Este módulo NÃO:
    - executa inferência;
    - detecta lanes;
    - altera a geometria;
    - controla diretamente o veículo;
    - envia comandos para teclado/volante.

A decisão é baseada exclusivamente no estado perceptivo já
consolidado pela camada vision.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


# =============================================================================
# ENUMERAÇÕES
# =============================================================================


class ADASDecisionState(str, Enum):
    """Estado de decisão do sistema ADAS."""

    UNKNOWN = "unknown"

    DISABLED = "disabled"

    NO_LANE = "no_lane"

    LOW_CONFIDENCE = "low_confidence"

    CENTERED = "centered"

    SLIGHT_LEFT = "slight_left"
    SLIGHT_RIGHT = "slight_right"

    LEFT_WARNING = "left_warning"
    RIGHT_WARNING = "right_warning"

    LEFT_DEPARTURE = "left_departure"
    RIGHT_DEPARTURE = "right_departure"


class ADASAction(str, Enum):
    """
    Ação lógica recomendada.

    A camada de decisão não executa a ação.
    """

    NONE = "none"

    HOLD = "hold"

    CORRECT_LEFT = "correct_left"
    CORRECT_RIGHT = "correct_right"

    EMERGENCY_LEFT = "emergency_left"
    EMERGENCY_RIGHT = "emergency_right"


# =============================================================================
# RESULTADO
# =============================================================================


@dataclass(frozen=True)
class ADASDecisionResult:
    """
    Resultado da decisão ADAS para um frame.
    """

    state: ADASDecisionState

    action: ADASAction

    lateral_offset: float

    heading_error: float

    confidence: float

    valid: bool

    safe: bool

    correction_requested: bool

    warning: bool

    departure: bool

    reason: str


# =============================================================================
# DECISOR
# =============================================================================


class ADASDecision:
    """
    Decide o estado lógico do ADAS a partir de LaneScene.

    O decisor é deliberadamente conservador:

        percepção inválida
            ↓
        NO_LANE / LOW_CONFIDENCE
            ↓
        ação NONE

    A atuação física deve acontecer somente depois do
    SafetyGate.
    """

    def __init__(
        self,
        enabled: bool = True,
        centered_threshold: float = 0.15,
        slight_threshold: float = 0.35,
        warning_threshold: float = 0.55,
        departure_threshold: float = 0.82,
        heading_warning_threshold: float = 0.35,
        min_confidence: float = 0.55,
        safe_confidence: float = 0.70,
    ) -> None:

        self.enabled = bool(enabled)

        self.centered_threshold = float(
            np.clip(
                centered_threshold,
                0.0,
                1.0,
            )
        )

        self.slight_threshold = float(
            np.clip(
                slight_threshold,
                self.centered_threshold,
                1.0,
            )
        )

        self.warning_threshold = float(
            np.clip(
                warning_threshold,
                self.slight_threshold,
                1.0,
            )
        )

        self.departure_threshold = float(
            np.clip(
                departure_threshold,
                self.warning_threshold,
                1.0,
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
            np.clip(
                min_confidence,
                0.0,
                1.0,
            )
        )

        self.safe_confidence = float(
            np.clip(
                safe_confidence,
                self.min_confidence,
                1.0,
            )
        )

        self.last_result: Optional[
            ADASDecisionResult
        ] = None

    # =========================================================================
    # UTILITÁRIOS
    # =========================================================================

    @staticmethod
    def _safe_float(
        value: object,
        default: float = 0.0,
    ) -> float:
        """
        Converte um valor para float sem permitir NaN/inf.
        """

        try:
            result = float(value)
        except (
            TypeError,
            ValueError,
        ):
            return default

        if not np.isfinite(result):
            return default

        return result

    @staticmethod
    def _clip_offset(
        value: object,
    ) -> float:
        """
        Limita offset lateral ao intervalo [-1, +1].
        """

        value = ADASDecision._safe_float(
            value
        )

        return float(
            np.clip(
                value,
                -1.0,
                1.0,
            )
        )

    @staticmethod
    def _get_attribute(
        obj: object,
        *names: str,
        default=None,
    ):
        """
        Obtém o primeiro atributo disponível.

        Mantém a camada tolerante a pequenas diferenças entre
        versões dos objetos de percepção.
        """

        if obj is None:
            return default

        for name in names:

            if hasattr(obj, name):

                value = getattr(
                    obj,
                    name,
                )

                if value is not None:
                    return value

        return default

    # =========================================================================
    # EXTRAÇÃO DA CENA
    # =========================================================================

    def _extract_scene_values(
        self,
        scene: object,
    ) -> tuple[
        float,
        float,
        float,
        bool,
        bool,
    ]:
        """
        Extrai:

            lateral_offset
            heading_error
            confidence
            perception_valid
            safe_for_adas
        """

        geometry = self._get_attribute(
            scene,
            "geometry",
        )

        confidence_object = self._get_attribute(
            scene,
            "confidence",
        )

        lateral_offset = self._get_attribute(
            scene,
            "lateral_offset",
            "normalized_offset",
            default=None,
        )

        if lateral_offset is None:

            lateral_offset = self._get_attribute(
                geometry,
                "normalized_offset",
                "lateral_error",
                "lateral_offset",
                default=0.0,
            )

        heading_error = self._get_attribute(
            scene,
            "heading_error",
            default=None,
        )

        if heading_error is None:

            heading_error = self._get_attribute(
                geometry,
                "heading_error",
                default=0.0,
            )

        confidence = self._get_attribute(
            scene,
            "confidence_value",
            default=None,
        )

        if confidence is None:

            confidence = self._get_attribute(
                confidence_object,
                "confidence",
                default=0.0,
            )

        perception_valid = bool(
            self._get_attribute(
                scene,
                "perception_valid",
                "valid",
                default=False,
            )
        )

        safe_for_adas = bool(
            self._get_attribute(
                scene,
                "safe_for_adas",
                default=False,
            )
        )

        return (
            self._clip_offset(
                lateral_offset
            ),
            self._clip_offset(
                heading_error
            ),
            float(
                np.clip(
                    self._safe_float(
                        confidence
                    ),
                    0.0,
                    1.0,
                )
            ),
            perception_valid,
            safe_for_adas,
        )

    # =========================================================================
    # CLASSIFICAÇÃO
    # =========================================================================

    def _classify(
        self,
        lateral_offset: float,
        heading_error: float,
    ) -> tuple[
        ADASDecisionState,
        ADASAction,
        bool,
        bool,
        bool,
        str,
    ]:
        """
        Classifica o estado instantâneo.

        Retorno:

            state
            action
            correction_requested
            warning
            departure
            reason
        """

        abs_offset = abs(
            lateral_offset
        )

        # ---------------------------------------------------------------------
        # DEPARTURE
        # ---------------------------------------------------------------------

        if lateral_offset <= -self.departure_threshold:

            return (
                ADASDecisionState.LEFT_DEPARTURE,
                ADASAction.EMERGENCY_LEFT,
                True,
                True,
                True,
                "Saída iminente pelo lado esquerdo.",
            )

        if lateral_offset >= self.departure_threshold:

            return (
                ADASDecisionState.RIGHT_DEPARTURE,
                ADASAction.EMERGENCY_RIGHT,
                True,
                True,
                True,
                "Saída iminente pelo lado direito.",
            )

        # ---------------------------------------------------------------------
        # WARNING
        # ---------------------------------------------------------------------

        if lateral_offset <= -self.warning_threshold:

            return (
                ADASDecisionState.LEFT_WARNING,
                ADASAction.CORRECT_LEFT,
                True,
                True,
                False,
                "Aproximação da linha esquerda.",
            )

        if lateral_offset >= self.warning_threshold:

            return (
                ADASDecisionState.RIGHT_WARNING,
                ADASAction.CORRECT_RIGHT,
                True,
                True,
                False,
                "Aproximação da linha direita.",
            )

        # ---------------------------------------------------------------------
        # DESLOCAMENTO LEVE
        # ---------------------------------------------------------------------

        if lateral_offset <= -self.slight_threshold:

            return (
                ADASDecisionState.SLIGHT_LEFT,
                ADASAction.CORRECT_LEFT,
                True,
                False,
                False,
                "Veículo deslocado para a esquerda.",
            )

        if lateral_offset >= self.slight_threshold:

            return (
                ADASDecisionState.SLIGHT_RIGHT,
                ADASAction.CORRECT_RIGHT,
                True,
                False,
                False,
                "Veículo deslocado para a direita.",
            )

        # ---------------------------------------------------------------------
        # HEADING
        # ---------------------------------------------------------------------

        if (
            abs_offset
            >= self.centered_threshold
        ):

            if (
                lateral_offset < 0.0
                and heading_error
                < -self.heading_warning_threshold
            ):

                return (
                    ADASDecisionState.SLIGHT_LEFT,
                    ADASAction.CORRECT_LEFT,
                    True,
                    False,
                    False,
                    "Deslocamento esquerdo com heading adverso.",
                )

            if (
                lateral_offset > 0.0
                and heading_error
                > self.heading_warning_threshold
            ):

                return (
                    ADASDecisionState.SLIGHT_RIGHT,
                    ADASAction.CORRECT_RIGHT,
                    True,
                    False,
                    False,
                    "Deslocamento direito com heading adverso.",
                )

        # ---------------------------------------------------------------------
        # CENTERED
        # ---------------------------------------------------------------------

        return (
            ADASDecisionState.CENTERED,
            ADASAction.HOLD,
            False,
            False,
            False,
            "Veículo dentro da região central da faixa.",
        )

    # =========================================================================
    # DECISÃO
    # =========================================================================

    def decide(
        self,
        scene: object,
    ) -> ADASDecisionResult:
        """
        Produz uma decisão ADAS a partir de LaneScene.
        """

        if not self.enabled:

            result = ADASDecisionResult(
                state=ADASDecisionState.DISABLED,
                action=ADASAction.NONE,
                lateral_offset=0.0,
                heading_error=0.0,
                confidence=0.0,
                valid=False,
                safe=False,
                correction_requested=False,
                warning=False,
                departure=False,
                reason="ADAS desabilitado.",
            )

            self.last_result = result

            return result

        if scene is None:

            result = ADASDecisionResult(
                state=ADASDecisionState.NO_LANE,
                action=ADASAction.NONE,
                lateral_offset=0.0,
                heading_error=0.0,
                confidence=0.0,
                valid=False,
                safe=False,
                correction_requested=False,
                warning=False,
                departure=False,
                reason="Cena de lanes inexistente.",
            )

            self.last_result = result

            return result

        (
            lateral_offset,
            heading_error,
            confidence,
            perception_valid,
            safe_for_adas,
        ) = self._extract_scene_values(
            scene
        )

        # ---------------------------------------------------------------------
        # PERCEPÇÃO AUSENTE
        # ---------------------------------------------------------------------

        if not perception_valid:

            result = ADASDecisionResult(
                state=ADASDecisionState.NO_LANE,
                action=ADASAction.NONE,
                lateral_offset=lateral_offset,
                heading_error=heading_error,
                confidence=confidence,
                valid=False,
                safe=False,
                correction_requested=False,
                warning=False,
                departure=False,
                reason="Percepção de lanes inválida.",
            )

            self.last_result = result

            return result

        # ---------------------------------------------------------------------
        # CONFIANÇA INSUFICIENTE
        # ---------------------------------------------------------------------

        if confidence < self.min_confidence:

            result = ADASDecisionResult(
                state=ADASDecisionState.LOW_CONFIDENCE,
                action=ADASAction.NONE,
                lateral_offset=lateral_offset,
                heading_error=heading_error,
                confidence=confidence,
                valid=True,
                safe=False,
                correction_requested=False,
                warning=False,
                departure=False,
                reason="Confiança insuficiente para decisão ADAS.",
            )

            self.last_result = result

            return result

        # ---------------------------------------------------------------------
        # CLASSIFICAÇÃO
        # ---------------------------------------------------------------------

        (
            state,
            action,
            correction_requested,
            warning,
            departure,
            reason,
        ) = self._classify(
            lateral_offset,
            heading_error,
        )

        # ---------------------------------------------------------------------
        # SEGURANÇA
        #
        # A decisão pode existir sem ser liberada para atuação.
        # Isso é importante:
        #
        #     DECISION != ACTUATION
        # ---------------------------------------------------------------------

        safe = (
            safe_for_adas
            and confidence
            >= self.safe_confidence
        )

        if not safe:

            action = ADASAction.NONE

            correction_requested = False

            reason = (
                f"{reason} "
                "Atuação bloqueada pela confiança."
            )

        result = ADASDecisionResult(
            state=state,
            action=action,
            lateral_offset=lateral_offset,
            heading_error=heading_error,
            confidence=confidence,
            valid=True,
            safe=safe,
            correction_requested=correction_requested,
            warning=warning,
            departure=departure,
            reason=reason,
        )

        self.last_result = result

        return result

    # =========================================================================
    # ALIASES
    # =========================================================================

    def update(
        self,
        scene: object,
    ) -> ADASDecisionResult:
        """
        Alias de decide() para integração com pipelines temporais.
        """

        return self.decide(
            scene
        )

    def evaluate(
        self,
        scene: object,
    ) -> ADASDecisionResult:
        """
        Alias de decide().
        """

        return self.decide(
            scene
        )


# =============================================================================
# FACTORY
# =============================================================================


def create_default_adas_decision(
    **kwargs,
) -> ADASDecision:

    return ADASDecision(
        **kwargs
    )


__all__ = [
    "ADASDecisionState",
    "ADASAction",
    "ADASDecisionResult",
    "ADASDecision",
    "create_default_adas_decision",
]