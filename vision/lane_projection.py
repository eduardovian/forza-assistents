"""
vision/lane_projection.py

Projeção matemática das linhas de faixa.

Responsabilidade:

    LaneModel
        ↓
    LaneProjectionEngine
        ↓
    projeção x(y)
        ↓
    LaneProjection

Este módulo NÃO:
    - executa YOLOP;
    - realiza tracking;
    - identifica a faixa atual;
    - calcula posição do veículo;
    - toma decisões ADAS.

A estrutura LaneProjection utilizada como resultado pertence
a vision.lane_types.
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .lane_types import (
    LaneModel,
    LanePoint,
    LanePolynomial,
    LaneProjection,
    ProjectionQuality,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

DEFAULT_MIN_POINTS = 8
DEFAULT_MIN_VERTICAL_SPAN = 80.0
DEFAULT_MAX_FIT_ERROR = 25.0
DEFAULT_MAX_EXTRAPOLATION = 0.35
DEFAULT_SAMPLE_STEP = 8
DEFAULT_POLYNOMIAL_DEGREE = 3
DEFAULT_MIN_CONFIDENCE = 0.45


# =============================================================================
# UTILITÁRIOS
# =============================================================================

def _clip01(value: float) -> float:
    return float(
        np.clip(
            value,
            0.0,
            1.0,
        )
    )


def _finite_array(
    values: np.ndarray,
) -> bool:
    return bool(
        np.all(
            np.isfinite(values)
        )
    )


# =============================================================================
# RESULTADO LEGACY
# =============================================================================

class LaneProjectionResult:
    """
    Resultado detalhado da projeção.

    Mantido para compatibilidade com testes e código
    que utilizavam a implementação anterior.
    """

    def __init__(
        self,
        points: Optional[List[LanePoint]] = None,
        coefficients: Tuple[float, ...] = tuple(),
        degree: int = 0,
        fitted: bool = False,
        extrapolated: bool = False,
        confidence: float = 0.0,
        fit_error: float = float("inf"),
        source_y_min: float = 0.0,
        source_y_max: float = 0.0,
        projected_y_min: float = 0.0,
        projected_y_max: float = 0.0,
        valid: bool = False,
        error: Optional[str] = None,
    ) -> None:

        self.points = (
            points
            if points is not None
            else []
        )

        self.coefficients = coefficients
        self.degree = int(degree)
        self.fitted = bool(fitted)
        self.extrapolated = bool(
            extrapolated
        )
        self.confidence = float(
            confidence
        )
        self.fit_error = float(
            fit_error
        )
        self.source_y_min = float(
            source_y_min
        )
        self.source_y_max = float(
            source_y_max
        )
        self.projected_y_min = float(
            projected_y_min
        )
        self.projected_y_max = float(
            projected_y_max
        )
        self.valid = bool(valid)
        self.error = error


# =============================================================================
# PROJETOR
# =============================================================================

class LaneProjectionEngine:
    """
    Motor de projeção das linhas de faixa.

    O modelo matemático utilizado é:

        x(y) = a*y³ + b*y² + c*y + d

    O ajuste é realizado em coordenadas Y normalizadas para
    melhorar a estabilidade numérica.
    """

    def __init__(
        self,
        min_points: int = DEFAULT_MIN_POINTS,
        min_vertical_span: float = DEFAULT_MIN_VERTICAL_SPAN,
        max_fit_error: float = DEFAULT_MAX_FIT_ERROR,
        max_extrapolation: float = DEFAULT_MAX_EXTRAPOLATION,
        sample_step: int = DEFAULT_SAMPLE_STEP,
        polynomial_degree: int = DEFAULT_POLYNOMIAL_DEGREE,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
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

        self.min_confidence = _clip01(
            min_confidence
        )

    # =========================================================================
    # FILTRAGEM
    # =========================================================================

    @staticmethod
    def _valid_points(
        points: Iterable[LanePoint],
    ) -> List[LanePoint]:

        result: List[LanePoint] = []

        for point in points:

            if not point.valid:
                continue

            if not (
                np.isfinite(point.x)
                and np.isfinite(point.y)
                and np.isfinite(
                    point.confidence
                )
            ):
                continue

            result.append(point)

        return result

    def _prepare_points(
        self,
        points: Sequence[LanePoint],
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:

        valid = self._valid_points(
            points
        )

        valid = [
            point
            for point in valid
            if point.confidence
            >= self.min_confidence
        ]

        if len(valid) < self.min_points:
            raise ValueError(
                "Pontos insuficientes "
                "para projeção."
            )

        valid.sort(
            key=lambda point: point.y
        )

        # Agrupa pontos com Y praticamente igual.
        groups: dict[float, List[LanePoint]] = {}

        for point in valid:

            key = round(
                float(point.y),
                3,
            )

            groups.setdefault(
                key,
                [],
            ).append(point)

        unique_points: List[
            LanePoint
        ] = []

        for group in groups.values():

            confidence_sum = sum(
                point.confidence
                for point in group
            )

            if confidence_sum <= 0.0:
                confidence_sum = float(
                    len(group)
                )

            x = sum(
                point.x
                * point.confidence
                for point in group
            ) / confidence_sum

            y = sum(
                point.y
                * point.confidence
                for point in group
            ) / confidence_sum

            confidence = max(
                point.confidence
                for point in group
            )

            unique_points.append(
                LanePoint(
                    x=float(x),
                    y=float(y),
                    confidence=float(
                        confidence
                    ),
                    valid=True,
                )
            )

        unique_points.sort(
            key=lambda point: point.y
        )

        if len(unique_points) < self.min_points:
            raise ValueError(
                "Pontos verticais insuficientes."
            )

        x = np.asarray(
            [
                point.x
                for point in unique_points
            ],
            dtype=np.float64,
        )

        y = np.asarray(
            [
                point.y
                for point in unique_points
            ],
            dtype=np.float64,
        )

        confidence = np.asarray(
            [
                point.confidence
                for point in unique_points
            ],
            dtype=np.float64,
        )

        if not (
            _finite_array(x)
            and _finite_array(y)
            and _finite_array(
                confidence
            )
        ):
            raise ValueError(
                "Pontos possuem valores "
                "não finitos."
            )

        vertical_span = float(
            np.max(y)
            - np.min(y)
        )

        if (
            vertical_span
            < self.min_vertical_span
        ):
            raise ValueError(
                "Extensão vertical insuficiente."
            )

        return (
            x,
            y,
            confidence,
        )

    # =========================================================================
    # AJUSTE
    # =========================================================================

    def _fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[
        np.ndarray,
        float,
        float,
        float,
        int,
    ]:

        degree = min(
            self.polynomial_degree,
            len(x) - 1,
        )

        if degree < 1:
            raise ValueError(
                "Não foi possível ajustar "
                "o polinômio."
            )

        y_center = float(
            np.mean(y)
        )

        y_scale = float(
            np.max(
                np.abs(
                    y - y_center
                )
            )
        )

        if y_scale < 1e-6:
            raise ValueError(
                "Escala vertical inválida."
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
            x - predicted
        )

        abs_residuals = np.abs(
            residuals
        )

        median_error = float(
            np.median(
                abs_residuals
            )
        )

        rmse = float(
            np.sqrt(
                np.mean(
                    residuals ** 2
                )
            )
        )

        fit_error = max(
            median_error,
            rmse * 0.75,
        )

        return (
            coefficients,
            fit_error,
            y_center,
            y_scale,
            degree,
        )

    # =========================================================================
    # CONVERSÃO PARA LanePolynomial
    # =========================================================================

    @staticmethod
    def _to_absolute_coefficients(
        coefficients: np.ndarray,
        center: float,
        scale: float,
    ) -> Tuple[
        float,
        float,
        float,
        float,
    ]:
        """
        Converte:

            x = p(z)
            z = (y-center)/scale

        para:

            x = a*y³+b*y²+c*y+d
        """

        if len(coefficients) == 4:

            an, bn, cn, dn = (
                float(value)
                for value in coefficients
            )

            a = (
                an
                / scale ** 3
            )

            b = (
                -3.0
                * an
                * center
                / scale ** 3
                + bn
                / scale ** 2
            )

            c = (
                3.0
                * an
                * center ** 2
                / scale ** 3
                - 2.0
                * bn
                * center
                / scale ** 2
                + cn
                / scale
            )

            d = (
                -an
                * center ** 3
                / scale ** 3
                + bn
                * center ** 2
                / scale ** 2
                - cn
                * center
                / scale
                + dn
            )

            return (
                float(a),
                float(b),
                float(c),
                float(d),
            )

        if len(coefficients) == 3:

            a2, b2, c2 = (
                float(value)
                for value in coefficients
            )

            a = 0.0

            b = (
                a2
                / scale ** 2
            )

            c = (
                -2.0
                * a2
                * center
                / scale ** 2
                + b2
                / scale
            )

            d = (
                a2
                * center ** 2
                / scale ** 2
                - b2
                * center
                / scale
                + c2
            )

            return (
                a,
                b,
                c,
                d,
            )

        if len(coefficients) == 2:

            a1, b1 = (
                float(value)
                for value in coefficients
            )

            return (
                0.0,
                0.0,
                a1 / scale,
                b1
                - a1
                * center
                / scale,
            )

        if len(coefficients) == 1:

            return (
                0.0,
                0.0,
                0.0,
                float(
                    coefficients[0]
                ),
            )

        raise ValueError(
            "Coeficientes inválidos."
        )

    # =========================================================================
    # CONFIANÇA
    # =========================================================================

    def _confidence(
        self,
        point_count: int,
        vertical_span: float,
        fit_error: float,
        point_confidence: float,
    ) -> float:

        count_score = _clip01(
            point_count / 25.0
        )

        span_score = _clip01(
            vertical_span / 400.0
        )

        if not np.isfinite(
            fit_error
        ):
            error_score = 0.0
        else:
            error_score = float(
                np.exp(
                    -fit_error / 18.0
                )
            )

        return _clip01(
            0.30 * count_score
            + 0.25 * span_score
            + 0.25 * error_score
            + 0.20 * point_confidence
        )

    # =========================================================================
    # PROJEÇÃO
    # =========================================================================

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

            (
                x,
                y,
                confidence_values,
            ) = self._prepare_points(
                points
            )

            (
                coefficients,
                fit_error,
                y_center,
                y_scale,
                degree,
            ) = self._fit(
                x,
                y,
            )

            if (
                not np.isfinite(
                    fit_error
                )
                or fit_error
                > self.max_fit_error
            ):
                raise ValueError(
                    "Erro do ajuste acima "
                    "do limite permitido."
                )

            source_y_min = float(
                np.min(y)
            )

            source_y_max = float(
                np.max(y)
            )

            vertical_span = (
                source_y_max
                - source_y_min
            )

            extrapolation = (
                vertical_span
                * self.max_extrapolation
            )

            projected_y_min = max(
                0.0,
                source_y_min
                - extrapolation,
            )

            projected_y_max = min(
                float(image_height - 1),
                source_y_max
                + extrapolation,
            )

            if (
                projected_y_max
                <= projected_y_min
            ):
                raise ValueError(
                    "Intervalo de projeção inválido."
                )

            projected_y = np.arange(
                projected_y_min,
                projected_y_max
                + self.sample_step,
                self.sample_step,
                dtype=np.float64,
            )

            if (
                projected_y.size == 0
                or projected_y[-1]
                < projected_y_max
            ):
                projected_y = np.append(
                    projected_y,
                    projected_y_max,
                )

            normalized_y = (
                projected_y
                - y_center
            ) / y_scale

            projected_x = np.polyval(
                coefficients,
                normalized_y,
            )

            finite = (
                np.isfinite(
                    projected_x
                )
                & np.isfinite(
                    projected_y
                )
            )

            projected_x = (
                projected_x[finite]
            )

            projected_y = (
                projected_y[finite]
            )

            inside = (
                (projected_x >= 0.0)
                & (
                    projected_x
                    <= float(
                        image_width - 1
                    )
                )
            )

            projected_x = (
                projected_x[inside]
            )

            projected_y = (
                projected_y[inside]
            )

            if len(projected_x) < 2:
                raise ValueError(
                    "A projeção não possui "
                    "pontos dentro da imagem."
                )

            mean_confidence = _clip01(
                float(
                    np.mean(
                        confidence_values
                    )
                )
            )

            confidence = self._confidence(
                len(x),
                vertical_span,
                fit_error,
                mean_confidence,
            )

            if confidence >= 0.80:
                quality = (
                    ProjectionQuality.HIGH
                )

            elif confidence >= 0.60:
                quality = (
                    ProjectionQuality.MEDIUM
                )

            elif confidence >= 0.40:
                quality = (
                    ProjectionQuality.LOW
                )

            else:
                quality = (
                    ProjectionQuality.NONE
                )

            polynomial = (
                LanePolynomial(
                    a=0.0,
                    b=0.0,
                    c=0.0,
                    d=0.0,
                    valid=True,
                    fit_error=fit_error,
                    sample_count=len(x),
                    confidence=confidence,
                    y_min=source_y_min,
                    y_max=source_y_max,
                )
            )

            (
                polynomial.a,
                polynomial.b,
                polynomial.c,
                polynomial.d,
            ) = self._to_absolute_coefficients(
                coefficients,
                y_center,
                y_scale,
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

            projection = LaneProjection(
                polynomial=polynomial,
                points=result_points,
                quality=quality,
                extrapolated=True,
                valid=(
                    quality
                    != ProjectionQuality.NONE
                ),
                horizon_y=(
                    projected_y_min
                ),
            )

            return LaneProjectionResult(
                points=result_points,
                coefficients=tuple(
                    float(value)
                    for value in coefficients
                ),
                degree=degree,
                fitted=True,
                extrapolated=True,
                confidence=confidence,
                fit_error=fit_error,
                source_y_min=source_y_min,
                source_y_max=source_y_max,
                projected_y_min=float(
                    np.min(projected_y)
                ),
                projected_y_max=float(
                    np.max(projected_y)
                ),
                valid=projection.valid,
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
                valid=False,
                error=(
                    f"{type(exc).__name__}: {exc}"
                ),
            )

    # =========================================================================
    # PROJEÇÃO A PARTIR DE LaneModel
    # =========================================================================

    def project_model(
        self,
        model: LaneModel,
        image_height: int,
        image_width: int,
    ) -> LaneProjectionResult:

        if model is None:
            return LaneProjectionResult(
                valid=False,
                error="LaneModel é None.",
            )

        return self.project(
            model.line.points,
            image_height,
            image_width,
        )

    # =========================================================================
    # PROJEÇÃO DIRETA DE POLINÔMIO
    # =========================================================================

    @staticmethod
    def evaluate_polynomial(
        polynomial: LanePolynomial,
        y: float,
    ) -> Optional[float]:

        if polynomial is None:
            return None

        if not polynomial.valid:
            return None

        value = polynomial.evaluate(
            float(y)
        )

        if not np.isfinite(value):
            return None

        return float(value)

    @staticmethod
    def sample_polynomial(
        polynomial: LanePolynomial,
        y_min: float,
        y_max: float,
        step: float = DEFAULT_SAMPLE_STEP,
        confidence: Optional[float] = None,
    ) -> List[LanePoint]:

        if polynomial is None:
            return []

        if not polynomial.valid:
            return []

        if y_max <= y_min:
            return []

        step = max(
            1.0,
            float(step),
        )

        ys = np.arange(
            y_min,
            y_max + step,
            step,
            dtype=np.float64,
        )

        if (
            ys.size == 0
            or ys[-1] < y_max
        ):
            ys = np.append(
                ys,
                y_max,
            )

        point_confidence = (
            polynomial.confidence
            if confidence is None
            else _clip01(
                confidence
            )
        )

        result = []

        for y in ys:

            x = polynomial.evaluate(
                float(y)
            )

            if not np.isfinite(x):
                continue

            result.append(
                LanePoint(
                    x=float(x),
                    y=float(y),
                    confidence=(
                        point_confidence
                    ),
                    valid=True,
                )
            )

        return result


# =============================================================================
# FUNÇÕES DE CONVENIÊNCIA
# =============================================================================

def project_lane(
    points: Sequence[LanePoint],
    image_height: int,
    image_width: int,
    **kwargs,
) -> LaneProjectionResult:

    projector = LaneProjectionEngine(
        **kwargs
    )

    return projector.project(
        points,
        image_height,
        image_width,
    )


def project_lane_model(
    model: LaneModel,
    image_height: int,
    image_width: int,
    **kwargs,
) -> LaneProjectionResult:

    projector = LaneProjectionEngine(
        **kwargs
    )

    return projector.project_model(
        model,
        image_height,
        image_width,
    )


def create_default_projection(
    **kwargs,
) -> LaneProjectionEngine:

    return LaneProjectionEngine(
        **kwargs
    )


# =============================================================================
# COMPATIBILIDADE
# =============================================================================

LaneProjector = LaneProjectionEngine


__all__ = [
    "LaneProjectionResult",
    "LaneProjectionEngine",
    "LaneProjector",
    "project_lane",
    "project_lane_model",
    "create_default_projection",
]