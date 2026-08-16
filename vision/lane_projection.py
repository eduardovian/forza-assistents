"""
vision/lane_projection.py

Projeção e extrapolação matemática das linhas de faixa.

Responsabilidade
----------------

Receber LaneModel/LanePolynomial já construídos pela etapa de
modelagem e gerar uma LaneProjection consistente.

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

Este módulo NÃO executa:

    - captura de tela;
    - definição de ROI;
    - pré-processamento;
    - inferência YOLOP;
    - tracking;
    - fitting;
    - lane assignment;
    - decisões ADAS;
    - controle do veículo.

IMPORTANTE
----------

O sistema trabalha no sistema de coordenadas do frame recebido.

O ROI pertence exclusivamente ao config.py/captura.

Portanto, este módulo não possui qualquer configuração própria
de ROI.

Modelo matemático:

    x(y) = a*y³ + b*y² + c*y + d

A projeção é realizada avaliando o polinômio dentro do domínio
observado e, quando permitido e seguro, extrapolando além dele.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable, Optional, Sequence

import math

from vision.lane_types import (
    LaneModel,
    LanePoint,
    LanePolynomial,
    LaneProjection,
    ProjectionQuality,
)


# =============================================================================
# NUMERIC HELPERS
# =============================================================================


def _finite_float(
    value: Any,
    default: Optional[float] = None,
) -> Optional[float]:
    """
    Converte um valor para float finito.
    """

    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    if not math.isfinite(result):
        return default

    return result


def _clip01(value: Any) -> float:
    """
    Limita um valor ao intervalo [0, 1].
    """

    result = _finite_float(
        value,
        0.0,
    )

    if result is None:
        return 0.0

    return max(
        0.0,
        min(
            1.0,
            result,
        ),
    )


# =============================================================================
# LANE PROJECTION ENGINE
# =============================================================================


class LaneProjectionEngine:
    """
    Gera projeções matemáticas de LaneModel.

    O engine não modifica o modelo original.

    Exemplo:

        engine = LaneProjectionEngine(
            min_points=4,
            min_confidence=0.40,
            max_projection_distance=900.0,
        )

        projection = engine.project(
            model,
            frame_height=900,
        )
    """

    def __init__(
        self,
        min_points: int = 4,
        min_confidence: float = 0.40,
        max_projection_distance: float = 900.0,
        samples: int = 32,
        lookahead_distance: float = 500.0,
        near_distance: float = 100.0,
        far_distance: float = 700.0,
        enable_extrapolation: bool = True,
        extrapolation_limit: float = 300.0,
        **kwargs: Any,
    ) -> None:

        self.min_points = max(
            2,
            int(min_points),
        )

        self.min_confidence = _clip01(
            min_confidence
        )

        self.max_projection_distance = max(
            1.0,
            float(max_projection_distance),
        )

        self.samples = max(
            2,
            int(samples),
        )

        self.lookahead_distance = max(
            0.0,
            float(lookahead_distance),
        )

        self.near_distance = max(
            0.0,
            float(near_distance),
        )

        self.far_distance = max(
            self.near_distance,
            float(far_distance),
        )

        self.enable_extrapolation = bool(
            enable_extrapolation
        )

        self.extrapolation_limit = max(
            0.0,
            float(extrapolation_limit),
        )

    # =========================================================================
    # MODEL VALIDATION
    # =========================================================================

    @staticmethod
    def _model_is_valid(
        model: Any,
    ) -> bool:
        """
        Valida um LaneModel.
        """

        if model is None:
            return False

        method = getattr(
            model,
            "is_valid",
            None,
        )

        if callable(method):
            try:
                if not bool(method()):
                    return False
            except Exception:
                return False

        line = getattr(
            model,
            "line",
            None,
        )

        polynomial = getattr(
            model,
            "polynomial",
            None,
        )

        if line is None:
            return False

        if polynomial is None:
            return False

        if not bool(
            getattr(
                line,
                "valid",
                True,
            )
        ):
            return False

        if not bool(
            getattr(
                polynomial,
                "valid",
                True,
            )
        ):
            return False

        return True

    # =========================================================================
    # MODEL CONFIDENCE
    # =========================================================================

    @staticmethod
    def _model_confidence(
        model: Any,
    ) -> float:
        """
        Obtém a confiança do modelo.

        Prioridade:

            polynomial.confidence
            line.confidence
            model.confidence
        """

        polynomial = getattr(
            model,
            "polynomial",
            None,
        )

        if polynomial is not None:

            value = _finite_float(
                getattr(
                    polynomial,
                    "confidence",
                    None,
                )
            )

            if value is not None:
                return _clip01(
                    value
                )

        line = getattr(
            model,
            "line",
            None,
        )

        if line is not None:

            value = _finite_float(
                getattr(
                    line,
                    "confidence",
                    None,
                )
            )

            if value is not None:
                return _clip01(
                    value
                )

        return _clip01(
            getattr(
                model,
                "confidence",
                1.0,
            )
        )

    # =========================================================================
    # POLYNOMIAL
    # =========================================================================

    @staticmethod
    def _get_polynomial(
        model: Any,
    ) -> Optional[LanePolynomial]:
        """
        Retorna o LanePolynomial associado ao modelo.
        """

        polynomial = getattr(
            model,
            "polynomial",
            None,
        )

        if polynomial is None:
            return None

        if not isinstance(
            polynomial,
            LanePolynomial,
        ):
            return None

        if not polynomial.is_valid():
            return None

        return polynomial

    # =========================================================================
    # OBSERVED RANGE
    # =========================================================================

    @staticmethod
    def _observed_range(
        model: Any,
        polynomial: LanePolynomial,
    ) -> Optional[tuple[float, float]]:
        """
        Determina o intervalo vertical observado.

        Prioridade:

            polynomial.y_min / y_max
            LaneLine.points
        """

        y_min = _finite_float(
            getattr(
                polynomial,
                "y_min",
                None,
            )
        )

        y_max = _finite_float(
            getattr(
                polynomial,
                "y_max",
                None,
            )
        )

        if (
            y_min is not None
            and y_max is not None
            and y_max >= y_min
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

        points = getattr(
            line,
            "points",
            [],
        ) or []

        ys: list[float] = []

        for point in points:

            y = _finite_float(
                getattr(
                    point,
                    "y",
                    None,
                )
            )

            if y is None:
                continue

            ys.append(y)

        if len(ys) < 2:
            return None

        return (
            min(ys),
            max(ys),
        )

    # =========================================================================
    # POLYNOMIAL EVALUATION
    # =========================================================================

    @staticmethod
    def _evaluate(
        polynomial: LanePolynomial,
        y: float,
    ) -> Optional[float]:
        """
        Avalia x(y).
        """

        try:
            x = polynomial.evaluate(
                float(y)
            )
        except Exception:
            return None

        return _finite_float(
            x
        )

    # =========================================================================
    # POINT CREATION
    # =========================================================================

    @staticmethod
    def _make_point(
        polynomial: LanePolynomial,
        y: float,
        confidence: float,
    ) -> Optional[LanePoint]:
        """
        Cria um LanePoint a partir do polinômio.
        """

        x = LaneProjectionEngine._evaluate(
            polynomial,
            y,
        )

        if x is None:
            return None

        return LanePoint(
            x=x,
            y=float(y),
            confidence=_clip01(
                confidence
            ),
            valid=True,
        )

    # =========================================================================
    # SAMPLE RANGE
    # =========================================================================

    @staticmethod
    def _linspace(
        start: float,
        stop: float,
        count: int,
    ) -> list[float]:
        """
        Gera pontos uniformemente espaçados.
        """

        count = max(
            2,
            int(count),
        )

        if stop <= start:
            return [
                float(start),
                float(stop),
            ]

        step = (
            stop - start
        ) / float(
            count - 1
        )

        return [
            float(
                start
                + step * index
            )
            for index in range(count)
        ]

    # =========================================================================
    # PROJECTION LIMIT
    # =========================================================================

    def _projection_end(
        self,
        y_min: float,
        y_max: float,
        frame_height: Optional[float],
    ) -> float:
        """
        Determina o limite vertical da projeção.

        Nunca ultrapassa:

            y_max + max_projection_distance

        e, quando frame_height é conhecido:

            frame_height
        """

        end = (
            y_max
            + self.max_projection_distance
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
        Projeta uma única faixa.

        O trecho observado é sempre preservado.

        A extrapolação só ocorre quando:

            - o modelo é válido;
            - o polinômio é válido;
            - a confiança é suficiente;
            - existem pontos suficientes;
            - enable_extrapolation está ativo.
        """

        if not self._model_is_valid(
            model
        ):
            return LaneProjection(
                quality=ProjectionQuality.NONE,
                extrapolated=False,
                valid=False,
            )

        polynomial = self._get_polynomial(
            model
        )

        if polynomial is None:
            return LaneProjection(
                quality=ProjectionQuality.NONE,
                extrapolated=False,
                valid=False,
            )

        confidence = self._model_confidence(
            model
        )

        if confidence < self.min_confidence:
            return LaneProjection(
                polynomial=polynomial,
                quality=ProjectionQuality.LOW,
                extrapolated=False,
                valid=False,
            )

        line = getattr(
            model,
            "line",
            None,
        )

        point_count = 0

        if line is not None:

            try:
                point_count = int(
                    line.point_count()
                )
            except Exception:

                point_count = len(
                    getattr(
                        line,
                        "points",
                        [],
                    ) or []
                )

        if point_count < self.min_points:

            return LaneProjection(
                polynomial=polynomial,
                quality=ProjectionQuality.LOW,
                extrapolated=False,
                valid=False,
            )

        observed = self._observed_range(
            model,
            polynomial,
        )

        if observed is None:

            return LaneProjection(
                polynomial=polynomial,
                quality=ProjectionQuality.LOW,
                extrapolated=False,
                valid=False,
            )

        y_min, y_max = observed

        if y_max <= y_min:

            return LaneProjection(
                polynomial=polynomial,
                quality=ProjectionQuality.LOW,
                extrapolated=False,
                valid=False,
            )

        # ---------------------------------------------------------------------
        # Determine projection range.
        # ---------------------------------------------------------------------

        projection_end = y_max

        extrapolated = False

        if self.enable_extrapolation:

            projection_end = self._projection_end(
                y_min,
                y_max,
                frame_height,
            )

            allowed_end = (
                y_max
                + self.extrapolation_limit
            )

            projection_end = min(
                projection_end,
                allowed_end,
            )

            if projection_end > y_max:
                extrapolated = True

        # ---------------------------------------------------------------------
        # Generate points.
        # ---------------------------------------------------------------------

        ys = self._linspace(
            y_min,
            projection_end,
            self.samples,
        )

        points: list[LanePoint] = []

        for y in ys:

            point = self._make_point(
                polynomial,
                y,
                confidence,
            )

            if point is None:
                continue

            points.append(
                point
            )

        if len(points) < 2:

            return LaneProjection(
                polynomial=polynomial,
                points=points,
                quality=ProjectionQuality.LOW,
                extrapolated=extrapolated,
                valid=False,
                horizon_y=horizon_y,
            )

        # ---------------------------------------------------------------------
        # Projection quality.
        # ---------------------------------------------------------------------

        if confidence >= 0.85:
            quality = ProjectionQuality.HIGH

        elif confidence >= 0.65:
            quality = ProjectionQuality.MEDIUM

        else:
            quality = ProjectionQuality.LOW

        return LaneProjection(
            polynomial=polynomial,
            points=points,
            quality=quality,
            extrapolated=extrapolated,
            valid=True,
            horizon_y=(
                float(horizon_y)
                if horizon_y is not None
                else None
            ),
        )

    # =========================================================================
    # PROJECT MANY
    # =========================================================================

    def project_many(
        self,
        models: Iterable[LaneModel],
        frame_height: Optional[float] = None,
    ) -> list[LaneProjection]:
        """
        Projeta múltiplos LaneModel.
        """

        projections: list[
            LaneProjection
        ] = []

        for model in models:

            projection = self.project(
                model,
                frame_height=frame_height,
            )

            projections.append(
                projection
            )

        return projections

    # =========================================================================
    # APPLY TO MODEL
    # =========================================================================

    def apply(
        self,
        model: LaneModel,
        frame_height: Optional[float] = None,
        *,
        horizon_y: Optional[float] = None,
    ) -> LaneModel:
        """
        Gera a projeção e retorna uma cópia do LaneModel
        contendo o resultado.

        O modelo original não é modificado.
        """

        projection = self.project(
            model,
            frame_height=frame_height,
            horizon_y=horizon_y,
        )

        return replace(
            model,
            projection=projection,
        )

    # =========================================================================
    # APPLY MANY
    # =========================================================================

    def apply_many(
        self,
        models: Sequence[LaneModel],
        frame_height: Optional[float] = None,
    ) -> list[LaneModel]:
        """
        Aplica projeção a uma sequência de LaneModel.
        """

        return [
            self.apply(
                model,
                frame_height=frame_height,
            )
            for model in models
        ]


# =============================================================================
# FACTORY
# =============================================================================


def create_default_projection_engine(
    **kwargs: Any,
) -> LaneProjectionEngine:
    """
    Cria um LaneProjectionEngine.

    Os valores podem ser fornecidos diretamente pelo chamador,
    normalmente a partir de config.LANE_PROJECTION.
    """

    return LaneProjectionEngine(
        **kwargs
    )


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    "LaneProjectionEngine",
    "create_default_projection_engine",
]