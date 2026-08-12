"""
vision/lane_confidence.py

Avaliação centralizada da confiança da percepção de lanes.

Responsabilidades:
    LaneModel(s)
        ↓
    qualidade geométrica
        +
    confiança das lanes
        +
    estabilidade temporal
        +
    observação direta/projeção
        ↓
    LaneConfidenceResult

Este módulo NÃO:
    - executa inferência;
    - rastreia lanes;
    - identifica a faixa atual;
    - calcula posição do veículo;
    - decide atuação ADAS;
    - controla o veículo.

A decisão de atuação será responsabilidade de:
    adas_decision.py
    safety_gate.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from .lane_types import LaneModel, LaneQuality


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

DEFAULT_MIN_SCORE = 0.55
DEFAULT_MIN_OBSERVED_LANES = 1
DEFAULT_MIN_STABLE_LANES = 1


# =============================================================================
# RESULTADO
# =============================================================================


@dataclass(frozen=True)
class LaneConfidenceResult:
    """
    Resultado consolidado da avaliação de confiança das lanes.

    score:
        Confiança agregada [0, 1].

    lane_count:
        Quantidade de LaneModel válidos considerados.

    stable_count:
        Quantidade de lanes consideradas estáveis.

    observed_count:
        Quantidade de lanes observadas diretamente.

    projected_count:
        Quantidade de lanes que possuem informação projetada.

    valid:
        Indica se a percepção possui qualidade mínima.

    safe_for_adas:
        Indica se a percepção possui qualidade suficiente para
        ser considerada pela camada de segurança.

    """

    score: float = 0.0

    lane_count: int = 0

    stable_count: int = 0

    observed_count: int = 0

    projected_count: int = 0

    valid: bool = False

    safe_for_adas: bool = False


# =============================================================================
# AVALIADOR
# =============================================================================


class LaneConfidenceEvaluator:
    """
    Avalia a confiabilidade global das lanes detectadas.

    A avaliação combina:

        1. confiança dos LaneModel;
        2. qualidade geométrica;
        3. estabilidade temporal;
        4. observação direta.

    A confiança resultante NÃO representa uma probabilidade estatística.
    É um score operacional para o pipeline ADAS.
    """

    def __init__(
        self,
        min_score: float = DEFAULT_MIN_SCORE,
        min_observed_lanes: int = DEFAULT_MIN_OBSERVED_LANES,
        min_stable_lanes: int = DEFAULT_MIN_STABLE_LANES,
    ) -> None:

        self.min_score = float(
            np.clip(
                min_score,
                0.0,
                1.0,
            )
        )

        self.min_observed_lanes = max(
            0,
            int(min_observed_lanes),
        )

        self.min_stable_lanes = max(
            0,
            int(min_stable_lanes),
        )

    # =========================================================================
    # HELPERS
    # =========================================================================

    @staticmethod
    def _clip01(
        value: float,
    ) -> float:
        """
        Limita um valor ao intervalo [0, 1].
        """

        try:
            value = float(value)
        except (
            TypeError,
            ValueError,
        ):
            return 0.0

        if not np.isfinite(value):
            return 0.0

        return float(
            np.clip(
                value,
                0.0,
                1.0,
            )
        )

    @staticmethod
    def _quality_score(
        quality: LaneQuality,
    ) -> float:
        """
        Converte qualidade geométrica em score numérico.
        """

        mapping = {
            LaneQuality.NONE: 0.0,
            LaneQuality.POOR: 0.25,
            LaneQuality.PARTIAL: 0.50,
            LaneQuality.GOOD: 0.78,
            LaneQuality.EXCELLENT: 1.0,
        }

        return mapping.get(
            quality,
            0.0,
        )

    @staticmethod
    def _is_directly_observed(
        lane: LaneModel,
    ) -> bool:
        """
        Determina se a lane possui observação direta.

        A implementação aceita as estruturas atuais de LaneModel
        sem depender de atributos opcionais inexistentes.
        """

        line = getattr(
            lane,
            "line",
            None,
        )

        if line is None:
            return False

        projected = bool(
            getattr(
                line,
                "projected",
                False,
            )
        )

        detected_directly = getattr(
            line,
            "detected_directly",
            None,
        )

        if detected_directly is not None:
            return bool(
                detected_directly
            )

        return not projected

    @staticmethod
    def _is_projected(
        lane: LaneModel,
    ) -> bool:
        """
        Determina se existe informação projetada.
        """

        line = getattr(
            lane,
            "line",
            None,
        )

        if line is not None and bool(
            getattr(
                line,
                "projected",
                False,
            )
        ):
            return True

        return getattr(
            lane,
            "projection",
            None,
        ) is not None

    @staticmethod
    def _is_stable(
        lane: LaneModel,
    ) -> bool:
        """
        Determina se a lane possui estabilidade temporal.

        O atributo é tratado de forma compatível com a estrutura
        atual do LaneModel.
        """

        return bool(
            getattr(
                lane,
                "stable",
                False,
            )
        )

    # =========================================================================
    # AVALIAÇÃO
    # =========================================================================

    def evaluate(
        self,
        lanes: Iterable[LaneModel],
        global_confidence: Optional[float] = None,
    ) -> LaneConfidenceResult:
        """
        Avalia um conjunto de LaneModel.

        Parameters
        ----------
        lanes:
            Lanes produzidas pela camada de modelagem.

        global_confidence:
            Confiança externa opcional, caso outra camada já tenha
            produzido uma estimativa global.

        Returns
        -------
        LaneConfidenceResult
        """

        valid_lanes = []

        for lane in lanes:

            if lane is None:
                continue

            if not bool(
                getattr(
                    lane,
                    "valid",
                    False,
                )
            ):
                continue

            valid_lanes.append(
                lane
            )

        if not valid_lanes:
            return LaneConfidenceResult()

        # ---------------------------------------------------------------------
        # Quantidades básicas
        # ---------------------------------------------------------------------

        observed_count = sum(
            1
            for lane in valid_lanes
            if self._is_directly_observed(
                lane
            )
        )

        projected_count = sum(
            1
            for lane in valid_lanes
            if self._is_projected(
                lane
            )
        )

        stable_count = sum(
            1
            for lane in valid_lanes
            if self._is_stable(
                lane
            )
        )

        # ---------------------------------------------------------------------
        # Confiança dos modelos
        # ---------------------------------------------------------------------

        model_confidences = []

        quality_scores = []

        for lane in valid_lanes:

            line = getattr(
                lane,
                "line",
                None,
            )

            if line is None:
                continue

            confidence = self._clip01(
                getattr(
                    line,
                    "confidence",
                    0.0,
                )
            )

            model_confidences.append(
                confidence
            )

            quality = getattr(
                line,
                "quality",
                LaneQuality.NONE,
            )

            quality_scores.append(
                self._quality_score(
                    quality
                )
            )

        if not model_confidences:
            return LaneConfidenceResult(
                lane_count=len(valid_lanes),
                stable_count=stable_count,
                observed_count=observed_count,
                projected_count=projected_count,
                valid=False,
                safe_for_adas=False,
            )

        confidence_score = float(
            np.mean(
                model_confidences
            )
        )

        quality_score = float(
            np.mean(
                quality_scores
            )
        )

        # ---------------------------------------------------------------------
        # Estabilidade
        # ---------------------------------------------------------------------

        stability_score = (
            stable_count
            / max(
                1,
                len(valid_lanes),
            )
        )

        stability_score = self._clip01(
            stability_score
        )

        # ---------------------------------------------------------------------
        # Observação direta
        #
        # Observação direta recebe peso maior que projeção porque
        # projeção é informação inferida.
        # ---------------------------------------------------------------------

        observation_score = (
            observed_count
            / max(
                1,
                len(valid_lanes),
            )
        )

        observation_score = self._clip01(
            observation_score
        )

        # ---------------------------------------------------------------------
        # Score final
        #
        # Confiança do modelo:
        #     35%
        #
        # Qualidade geométrica:
        #     30%
        #
        # Estabilidade:
        #     20%
        #
        # Observação direta:
        #     15%
        # ---------------------------------------------------------------------

        score = (
            0.35 * confidence_score
            + 0.30 * quality_score
            + 0.20 * stability_score
            + 0.15 * observation_score
        )

        # ---------------------------------------------------------------------
        # Confiança externa opcional
        # ---------------------------------------------------------------------

        if global_confidence is not None:

            external_score = self._clip01(
                global_confidence
            )

            score = (
                0.75 * score
                + 0.25 * external_score
            )

        score = self._clip01(
            score
        )

        # ---------------------------------------------------------------------
        # Validade
        # ---------------------------------------------------------------------

        valid = (
            observed_count
            >= self.min_observed_lanes
            and score
            >= self.min_score
        )

        safe_for_adas = (
            valid
            and stable_count
            >= self.min_stable_lanes
        )

        return LaneConfidenceResult(
            score=score,
            lane_count=len(valid_lanes),
            stable_count=stable_count,
            observed_count=observed_count,
            projected_count=projected_count,
            valid=valid,
            safe_for_adas=safe_for_adas,
        )


# =============================================================================
# API PÚBLICA
# =============================================================================


__all__ = [
    "LaneConfidenceResult",
    "LaneConfidenceEvaluator",
]