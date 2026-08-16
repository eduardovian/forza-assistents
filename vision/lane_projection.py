"""
vision/lane_projection.py

Forza Assistents
================

Projeção matemática das linhas de faixa.

Responsabilidade
----------------
Receber um LaneModel já construído e gerar uma LaneProjection
matematicamente consistente.

Pipeline:

    LaneModel
        ↓
    LanePolynomial
        ↓
    LaneProjectionEngine
        ↓
    LaneProjection
        ↓
    LaneAssignment

Este módulo NÃO:

    - captura imagens;
    - define ROI;
    - executa YOLOP;
    - realiza fitting;
    - executa tracking;
    - determina a faixa atual;
    - toma decisões ADAS;
    - envia comandos ao veículo.

COORDENADAS
-----------
Todo o cálculo ocorre no sistema de coordenadas do frame recebido.

ROI pertence exclusivamente a config.py/capture.

MODELO
------
O contrato matemático oficial é:

    x(y) = a*y³ + b*y² + c*y + d

Segurança:
- nunca altera o LanePolynomial original;
- rejeita modelos inválidos;
- rejeita valores não finitos;
- limita extrapolação;
- degrada confiança durante extrapolação;
- respeita o frame recebido;
- não cria parâmetros de ROI.
"""

from __future__ import annotations

from typing import Any, Optional
import math

from config import LANE_PROJECTION

from vision.lane_types import (
    LaneModel,
    LanePoint,
    LanePolynomial,
    LaneProjection,
    ProjectionQuality,
)


# =============================================================================
# NUMERIC UTILITIES
# =============================================================================


def _finite_float(
    value: Any,
) -> Optional[float]:
    """
    Retorna float somente quando o valor é finito.
    """

    try:
        result = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(result):
        return None

    return result


def _clip01(
    value: Any,
) -> float:
    """
    Limita valor ao intervalo [0, 1].
    """

    result = _finite_float(value)

    if result is None:
        return 0.0

    return max(
        0.0,
        min(1.0, result),
    )


# =============================================================================
# ENGINE
# =============================================================================


class LaneProjectionEngine:
    """
    Engine oficial de projeção.

    Todas as configurações operacionais são obtidas de:

        config.LANE_PROJECTION

    Nenhuma configuração paralela é mantida nesta classe.
    """

    def __init__(self) -> None:
        self.config = LANE_PROJECTION

    # =========================================================================
    # VALIDATION
    # =========================================================================

    def _validate_model(
        self,
        model: Any,
    ) -> bool:
        """
        Valida o contrato mínimo de LaneModel.
        """

        if not isinstance(
            model,
            LaneModel,
        ):
            return False

        try:
            if not model.is_valid():
                return False
        except Exception:
            return False

        polynomial = getattr(
            model,
            "polynomial",
            None,
        )

        if not isinstance(
            polynomial,
            LanePolynomial,
        ):
            return False

        if not polynomial.is_valid():
            return False

        if (
            self.config.reject_non_finite_points
            and not polynomial.is_finite()
        ):
            return False

        return True

    # =========================================================================
    # CONFIDENCE
    # =========================================================================

    @staticmethod
    def _model_confidence(
        model: LaneModel,
    ) -> float:
        """
        Obtém a confiança do LanePolynomial.
        """

        polynomial = model.polynomial

        if polynomial is None:
            return 0.0

        return _clip01(
            polynomial.confidence
        )

    # =========================================================================
    # OBSERVED DOMAIN
    # =========================================================================

    @staticmethod
    def _observed_range(
        model: LaneModel,
        polynomial: LanePolynomial,
    ) -> Optional[tuple[float, float]]:
        """
        Obtém o domínio vertical observado.

        Prioridade:

            LanePolynomial.y_min/y_max

        Fallback:

            LaneLine.points
        """

        y_min = _finite_float(
            polynomial.y_min
        )

        y_max = _finite_float(
            polynomial.y_max
        )

        if (
            y_min is not None
            and y_max is not None
            and y_max > y_min
        ):
            return (
                y_min,
                y_max,
            )

        line = getattr(
            model,
            "line",
            None,
        )

        if line is None:
            return None

        points = getattr(
            line,
            "points",
            None,
        )

        if not points:
            return None

        ys: list[float] = []

        for point in points:

            y = _finite_float(
                getattr(
                    point,
                    "y",
                    None,
                )
            )

            if y is not None:
                ys.append(y)

        if len(ys) < 2:
            return None

        minimum = min(ys)
        maximum = max(ys)

        if maximum <= minimum:
            return None

        return (
            minimum,
            maximum,
        )

    # =========================================================================
    # EVALUATION
    # =========================================================================

    @staticmethod
    def _evaluate(
        polynomial: LanePolynomial,
        y: float,
    ) -> Optional[float]:
        """
        Avalia x(y) utilizando exclusivamente o LanePolynomial real.
        """

        try:
            x = polynomial.evaluate(y)
        except (
            TypeError,
            ValueError,
            OverflowError,
        ):
            return None

        return _finite_float(x)

    # =========================================================================
    # CURVATURE
    # =========================================================================

    def _curvature(
        self,
        polynomial: LanePolynomial,
        y: float,
    ) -> Optional[float]:
        """
        Calcula a curvatura aproximada da função x(y):

                     |x''|
        k = -------------------------
            (1 + x'^2)^(3/2)
        """

        try:
            first = polynomial.derivative(y)
            second = polynomial.second_derivative(y)

        except (
            TypeError,
            ValueError,
            OverflowError,
        ):
            return None

        first = _finite_float(first)
        second = _finite_float(second)

        if first is None or second is None:
            return None

        denominator = (
            1.0 + first * first
        ) ** 1.5

        if denominator <= 1e-12:
            return None

        curvature = (
            abs(second)
            / denominator
        )

        return _finite_float(
            curvature
        )

    # =========================================================================
    # SAMPLE GENERATION
    # =========================================================================

    @staticmethod
    def _linspace(
        start: float,
        stop: float,
        count: int,
    ) -> list[float]:
        """
        Gera amostras uniformemente distribuídas.
        """

        count = max(
            2,
            int(count),
        )

        if stop <= start:
            return [
                start,
                stop,
            ]

        step = (
            stop - start
        ) / (
            count - 1
        )

        return [
            start + (
                step * index
            )
            for index in range(count)
        ]

    # =========================================================================
    # PROJECTION LIMIT
    # =========================================================================

    def _projection_end(
        self,
        y_max: float,
        frame_height: Optional[float],
    ) -> float:
        """
        Determina até onde a projeção pode avançar.
        """

        if not self.config.enable_extrapolation:
            return y_max

        end = (
            y_max
            + min(
                self.config.max_projection_distance,
                self.config.extrapolation_limit,
            )
        )

        if frame_height is not None:

            height = _finite_float(
                frame_height
            )

            if height is not None:
                end = min(
                    end,
                    height,
                )

        return max(
            y_max,
            end,
        )

    # =========================================================================
    # POINT CONFIDENCE
    # =========================================================================

    def _point_confidence(
        self,
        base_confidence: float,
        y: float,
        y_max: float,
    ) -> float:
        """
        Reduz progressivamente a confiança fora do domínio observado.
        """

        if y <= y_max:
            return base_confidence

        distance = (
            y - y_max
        )

        decay_distance = max(
            self.config.confidence_decay_distance,
            1e-6,
        )

        decay = max(
            0.0,
            min(
                1.0,
                1.0
                - (
                    distance
                    / decay_distance
                ),
            ),
        )

        return _clip01(
            base_confidence
            * decay
        )

    # =========================================================================
    # QUALITY
    # =========================================================================

    @staticmethod
    def _quality(
        confidence: float,
        extrapolation_distance: float,
    ) -> ProjectionQuality:
        """
        Determina qualidade da projeção.
        """

        confidence = _clip01(
            confidence
        )

        if extrapolation_distance > 0.0:

            confidence *= max(
                0.0,
                min(
                    1.0,
                    1.0
                    - (
                        extrapolation_distance
                        / 300.0
                    ),
                ),
            )

        if confidence >= 0.85:
            return ProjectionQuality.HIGH

        if confidence >= 0.65:
            return ProjectionQuality.MEDIUM

        if confidence >= 0.40:
            return ProjectionQuality.LOW

        return ProjectionQuality.NONE

    # =========================================================================
    # PROJECT
    # =========================================================================

    def project(
        self,
        model: LaneModel,
        frame_height: Optional[float] = None,
        *,
        horizon_y: Optional[float] = None,
    ) -> LaneProjection:
        """
        Projeta uma única LaneModel.
        """

        if not self.config.enabled:

            return LaneProjection(
                valid=False,
                quality=ProjectionQuality.NONE,
            )

        if not self._validate_model(model):

            return LaneProjection(
                valid=False,
                quality=ProjectionQuality.NONE,
            )

        polynomial = model.polynomial

        if polynomial is None:

            return LaneProjection(
                valid=False,
                quality=ProjectionQuality.NONE,
            )

        confidence = (
            self._model_confidence(
                model
            )
        )

        if (
            confidence
            < self.config.minimum_confidence
        ):

            return LaneProjection(
                polynomial=polynomial,
                valid=False,
                quality=ProjectionQuality.NONE,
            )

        line = getattr(
            model,
            "line",
            None,
        )

        if line is None:

            return LaneProjection(
                polynomial=polynomial,
                valid=False,
                quality=ProjectionQuality.NONE,
            )

        point_count = (
            line.valid_point_count()
        )

        if (
            point_count
            < self.config.minimum_points
        ):

            return LaneProjection(
                polynomial=polynomial,
                valid=False,
                quality=ProjectionQuality.NONE,
            )

        observed = (
            self._observed_range(
                model,
                polynomial,
            )
        )

        if observed is None:

            return LaneProjection(
                polynomial=polynomial,
                valid=False,
                quality=ProjectionQuality.NONE,
            )

        y_min, y_max = observed

        projection_end = (
            self._projection_end(
                y_max,
                frame_height,
            )
        )

        y_values = self._linspace(
            y_min,
            projection_end,
            self.config.samples,
        )

        points: list[LanePoint] = []

        for y in y_values:

            x = self._evaluate(
                polynomial,
                y,
            )

            if x is None:
                continue

            curvature = (
                self._curvature(
                    polynomial,
                    y,
                )
            )

            if curvature is None:
                continue

            if (
                curvature
                > self.config.maximum_curvature
            ):
                continue

            point_confidence = (
                self._point_confidence(
                    confidence,
                    y,
                    y_max,
                )
            )

            points.append(
                LanePoint(
                    x=x,
                    y=y,
                    confidence=point_confidence,
                    valid=True,
                )
            )

        if len(points) < 2:

            return LaneProjection(
                polynomial=polynomial,
                points=points,
                valid=False,
                quality=ProjectionQuality.NONE,
                extrapolated=(
                    projection_end > y_max
                ),
                horizon_y=horizon_y,
            )

        extrapolation_distance = max(
            0.0,
            projection_end - y_max,
        )

        quality = self._quality(
            confidence,
            extrapolation_distance,
        )

        return LaneProjection(
            polynomial=polynomial,
            points=points,
            quality=quality,
            extrapolated=(
                extrapolation_distance > 0.0
            ),
            valid=(
                quality
                != ProjectionQuality.NONE
            ),
            horizon_y=horizon_y,
        )

    # =========================================================================
    # BATCH
    # =========================================================================

    def project_many(
        self,
        models: list[LaneModel],
        frame_height: Optional[float] = None,
        *,
        horizon_y: Optional[float] = None,
    ) -> list[LaneProjection]:
        """
        Projeta múltiplas lanes preservando a ordem.
        """

        return [
            self.project(
                model,
                frame_height,
                horizon_y=horizon_y,
            )
            for model in models
        ]


# =============================================================================
# FACTORY
# =============================================================================


def create_lane_projection_engine() -> (
    LaneProjectionEngine
):
    """
    Factory oficial.
    """

    return LaneProjectionEngine()


# =============================================================================
# PUBLIC API
# =============================================================================


__all__ = [
    "LaneProjectionEngine",
    "create_lane_projection_engine",
]