"""
vision/lane_projection.py

Projeção e extrapolação das linhas de faixa.

Responsabilidade:

    LaneTracker / LaneGeometry
                ↓
        pontos confiáveis
                ↓
        LaneProjection
                ↓
        curva matemática da faixa
                ↓
        pontos projetados

Este módulo NÃO:
    - executa YOLOP
    - captura tela
    - calcula controle do volante
    - decide estado ADAS
    - aplica correção no veículo

Objetivo:

A partir de um trecho suficientemente confiável de uma
faixa, estimar a continuação dela dentro da região visível.

A projeção utiliza polinômio de até terceiro grau:

    x(y) = a*y³ + b*y² + c*y + d

O eixo principal é Y da imagem.

A projeção é limitada por:
    - quantidade mínima de pontos
    - distribuição vertical dos pontos
    - erro de ajuste
    - intervalo máximo de extrapolação
    - estabilidade do ajuste
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .lane_types import LanePoint


logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

DEFAULT_MIN_POINTS = 8

DEFAULT_MIN_VERTICAL_SPAN = 80.0

DEFAULT_MAX_FIT_ERROR = 18.0

DEFAULT_MAX_EXTRAPOLATION = 0.35

DEFAULT_SAMPLE_STEP = 8

DEFAULT_POLYNOMIAL_DEGREE = 3


# ============================================================================
# RESULTADO
# ============================================================================


@dataclass
class LaneProjectionResult:
    """
    Resultado da projeção de uma faixa.
    """

    points: List[LanePoint]

    coefficients: Tuple[float, ...]

    degree: int

    fitted: bool

    extrapolated: bool

    confidence: float

    fit_error: float

    source_y_min: float

    source_y_max: float

    projected_y_min: float

    projected_y_max: float

    valid: bool

    error: Optional[str] = None


# ============================================================================
# PROJETOR
# ============================================================================


class LaneProjection:
    """
    Projeta a continuação de uma linha de faixa.

    O ajuste é realizado em:

        x = f(y)

    em vez de:

        y = f(x)

    Isso é importante porque as linhas de faixa normalmente
    atravessam grande parte do eixo vertical da imagem.

    A ordem máxima utilizada é cúbica.
    """

    def __init__(
        self,
        min_points: int = DEFAULT_MIN_POINTS,
        min_vertical_span: float = DEFAULT_MIN_VERTICAL_SPAN,
        max_fit_error: float = DEFAULT_MAX_FIT_ERROR,
        max_extrapolation: float = DEFAULT_MAX_EXTRAPOLATION,
        sample_step: int = DEFAULT_SAMPLE_STEP,
        polynomial_degree: int = DEFAULT_POLYNOMIAL_DEGREE,
    ) -> None:

        self.min_points = max(
            4,
            int(min_points),
        )

        self.min_vertical_span = max(
            1.0,
            float(min_vertical_span),
        )

        self.max_fit_error = max(
            0.1,
            float(max_fit_error),
        )

        self.max_extrapolation = float(
            np.clip(
                max_extrapolation,
                0.0,
                2.0,
            )
        )

        self.sample_step = max(
            1,
            int(sample_step),
        )

        self.polynomial_degree = int(
            np.clip(
                polynomial_degree,
                1,
                3,
            )
        )

    # ========================================================================
    # FILTRAGEM
    # ========================================================================

    @staticmethod
    def _valid_points(
        points: Sequence[LanePoint],
    ) -> List[LanePoint]:

        valid = []

        for point in points:

            if not point.valid:
                continue

            if not np.isfinite(point.x):
                continue

            if not np.isfinite(point.y):
                continue

            valid.append(point)

        return valid

    # ========================================================================
    # PREPARAÇÃO
    # ========================================================================

    def _prepare_points(
        self,
        points: Sequence[LanePoint],
    ) -> Tuple[np.ndarray, np.ndarray]:

        valid = self._valid_points(points)

        if len(valid) < self.min_points:
            raise ValueError(
                "Pontos insuficientes para projeção."
            )

        # Ordenação vertical.
        valid = sorted(
            valid,
            key=lambda point: point.y,
        )

        x = np.asarray(
            [point.x for point in valid],
            dtype=np.float64,
        )

        y = np.asarray(
            [point.y for point in valid],
            dtype=np.float64,
        )

        # Remove Y duplicado mantendo a média de X.
        unique_y, inverse = np.unique(
            y,
            return_inverse=True,
        )

        if len(unique_y) != len(y):

            sums = np.zeros_like(
                unique_y,
                dtype=np.float64,
            )

            counts = np.zeros_like(
                unique_y,
                dtype=np.float64,
            )

            for index, group in enumerate(
                inverse
            ):
                sums[group] += x[index]
                counts[group] += 1.0

            x = sums / np.maximum(
                counts,
                1.0,
            )

            y = unique_y

        if len(y) < self.min_points:
            raise ValueError(
                "Pontos verticais insuficientes."
            )

        vertical_span = (
            float(np.max(y))
            - float(np.min(y))
        )

        if (
            vertical_span
            < self.min_vertical_span
        ):
            raise ValueError(
                "Trecho vertical insuficiente "
                "para projetar a faixa."
            )

        return x, y

    # ========================================================================
    # AJUSTE
    # ========================================================================

    def _fit_polynomial(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[
        np.ndarray,
        float,
        int,
    ]:

        degree = min(
            self.polynomial_degree,
            len(x) - 1,
        )

        if degree < 1:
            raise ValueError(
                "Não foi possível determinar "
                "o grau do polinômio."
            )

        # Normalizamos Y antes do ajuste para melhorar
        # a estabilidade numérica.
        y_center = float(
            np.mean(y)
        )

        y_scale = float(
            np.std(y)
        )

        if y_scale < 1e-6:
            raise ValueError(
                "Distribuição vertical inválida."
            )

        yn = (
            y - y_center
        ) / y_scale

        coefficients = np.polyfit(
            yn,
            x,
            degree,
        )

        predicted = np.polyval(
            coefficients,
            yn,
        )

        residuals = (
            predicted - x
        )

        rmse = float(
            np.sqrt(
                np.mean(
                    residuals ** 2
                )
            )
        )

        return (
            coefficients,
            rmse,
            degree,
        )

    # ========================================================================
    # AVALIAÇÃO
    # ========================================================================

    def _evaluate(
        self,
        coefficients: np.ndarray,
        y: np.ndarray,
        y_center: float,
        y_scale: float,
    ) -> np.ndarray:

        yn = (
            y - y_center
        ) / y_scale

        return np.polyval(
            coefficients,
            yn,
        )

    # ========================================================================
    # CONFIANÇA
    # ========================================================================

    def _calculate_confidence(
        self,
        point_count: int,
        vertical_span: float,
        fit_error: float,
    ) -> float:

        point_score = np.clip(
            point_count / 30.0,
            0.0,
            1.0,
        )

        span_score = np.clip(
            vertical_span / 400.0,
            0.0,
            1.0,
        )

        error_score = np.clip(
            1.0
            - (
                fit_error
                / self.max_fit_error
            ),
            0.0,
            1.0,
        )

        confidence = (
            0.30 * point_score
            + 0.30 * span_score
            + 0.40 * error_score
        )

        return float(
            np.clip(
                confidence,
                0.0,
                1.0,
            )
        )

    # ========================================================================
    # PROJEÇÃO
    # ========================================================================

    def project(
        self,
        points: Sequence[LanePoint],
        image_height: int,
        image_width: int,
    ) -> LaneProjectionResult:

        try:

            if image_height <= 0:
                raise ValueError(
                    "image_height inválido."
                )

            if image_width <= 0:
                raise ValueError(
                    "image_width inválido."
                )

            x, y = self._prepare_points(
                points
            )

            (
                coefficients,
                fit_error,
                degree,
            ) = self._fit_polynomial(
                x,
                y,
            )

            if (
                not np.isfinite(fit_error)
                or fit_error
                > self.max_fit_error
            ):

                raise ValueError(
                    "Erro do ajuste polinomial "
                    "acima do limite."
                )

            y_min = float(
                np.min(y)
            )

            y_max = float(
                np.max(y)
            )

            vertical_span = (
                y_max - y_min
            )

            # Projetamos para cima e para baixo.
            #
            # A quantidade de extrapolação é limitada
            # proporcionalmente ao trecho realmente observado.
            extrapolation = (
                vertical_span
                * self.max_extrapolation
            )

            projected_y_min = max(
                0.0,
                y_min - extrapolation,
            )

            projected_y_max = min(
                float(image_height - 1),
                y_max + extrapolation,
            )

            sample_count = max(
                2,
                int(
                    (
                        projected_y_max
                        - projected_y_min
                    )
                    / self.sample_step
                )
                + 1,
            )

            projected_y = np.linspace(
                projected_y_min,
                projected_y_max,
                sample_count,
            )

            # Normalização usada no ajuste.
            y_center = float(
                np.mean(y)
            )

            y_scale = float(
                np.std(y)
            )

            projected_x = self._evaluate(
                coefficients,
                projected_y,
                y_center,
                y_scale,
            )

            finite = (
                np.isfinite(projected_x)
                & np.isfinite(projected_y)
            )

            projected_x = projected_x[
                finite
            ]

            projected_y = projected_y[
                finite
            ]

            # Não permitir projeções absurdas.
            inside = (
                (projected_x >= 0.0)
                & (
                    projected_x
                    < float(image_width)
                )
            )

            projected_x = projected_x[
                inside
            ]

            projected_y = projected_y[
                inside
            ]

            if len(projected_x) < 2:
                raise ValueError(
                    "Projeção saiu dos limites "
                    "da imagem."
                )

            confidence = (
                self._calculate_confidence(
                    len(x),
                    vertical_span,
                    fit_error,
                )
            )

            result_points = [
                LanePoint(
                    x=float(px),
                    y=float(py),
                    confidence=confidence,
                    valid=True,
                )
                for px, py in zip(
                    projected_x,
                    projected_y,
                )
            ]

            coefficients_tuple = tuple(
                float(value)
                for value in coefficients
            )

            return LaneProjectionResult(
                points=result_points,
                coefficients=(
                    coefficients_tuple
                ),
                degree=degree,
                fitted=True,
                extrapolated=True,
                confidence=confidence,
                fit_error=fit_error,
                source_y_min=y_min,
                source_y_max=y_max,
                projected_y_min=float(
                    np.min(projected_y)
                ),
                projected_y_max=float(
                    np.max(projected_y)
                ),
                valid=True,
                error=None,
            )

        except Exception as exc:

            logger.debug(
                "[LANE PROJECTION] "
                "Projeção rejeitada: %s",
                exc,
            )

            return LaneProjectionResult(
                points=[],
                coefficients=tuple(),
                degree=0,
                fitted=False,
                extrapolated=False,
                confidence=0.0,
                fit_error=float("inf"),
                source_y_min=0.0,
                source_y_max=0.0,
                projected_y_min=0.0,
                projected_y_max=0.0,
                valid=False,
                error=(
                    f"{type(exc).__name__}: {exc}"
                ),
            )


# ============================================================================
# FACTORY
# ============================================================================


def create_default_projection(
    **kwargs,
) -> LaneProjection:

    return LaneProjection(
        **kwargs
    )


__all__ = [
    "LaneProjectionResult",
    "LaneProjection",
    "create_default_projection",
]