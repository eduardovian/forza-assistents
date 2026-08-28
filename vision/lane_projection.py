"""
vision/lane_projection.py

Forza Assistents
================

Projeção e extrapolação geométrica de lanes.

Responsabilidades
-----------------
Este módulo transforma um LanePolynomial observado em uma
representação LaneProjection que pode ser utilizada por etapas
posteriores da percepção.

A projeção é explicitamente diferente de uma detecção:

    DETECTED
        |
        v
    LaneLine
        |
        v
    LanePolynomial
        |
        v
    LaneProjection
        |
        v
    INFERRED / PROJECTED

Princípios:

- nunca alterar a observação original;
- nunca inventar uma detecção;
- limitar a distância de extrapolação;
- reduzir confiança conforme a extrapolação aumenta;
- rejeitar polinômios inválidos;
- manter o contrato definido em lane_types.py;
- não depender do detector YOLOP;
- não depender do tracker;
- não tomar decisões ADAS.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from .lane_types import (
    LaneLine,
    LanePoint,
    LanePolynomial,
    LaneProjection,
    LaneSource,
)


# =============================================================================
# CONSTANTES
# =============================================================================

POLYNOMIAL_DEGREE = 3

DEFAULT_MINIMUM_POINTS = 4

DEFAULT_POINT_SPACING = 10.0

DEFAULT_MAX_EXTRAPOLATION = 180.0

DEFAULT_MIN_EXTRAPOLATION = 5.0

DEFAULT_MIN_CONFIDENCE = 0.15

DEFAULT_CONFIDENCE_DECAY = 0.35

DEFAULT_MAX_SLOPE = 8.0

DEFAULT_MAX_SECOND_DERIVATIVE = 0.02

DEFAULT_VALIDATION_SAMPLES = 9


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

@dataclass(frozen=True, slots=True)
class ProjectionConfig:
    """
    Configuração imutável da projeção.

    Todos os valores estão em coordenadas de imagem.
    """

    max_extrapolation: float = (
        DEFAULT_MAX_EXTRAPOLATION
    )

    min_extrapolation: float = (
        DEFAULT_MIN_EXTRAPOLATION
    )

    point_spacing: float = (
        DEFAULT_POINT_SPACING
    )

    minimum_confidence: float = (
        DEFAULT_MIN_CONFIDENCE
    )

    confidence_decay: float = (
        DEFAULT_CONFIDENCE_DECAY
    )

    max_slope: float = (
        DEFAULT_MAX_SLOPE
    )

    max_second_derivative: float = (
        DEFAULT_MAX_SECOND_DERIVATIVE
    )

    validation_samples: int = (
        DEFAULT_VALIDATION_SAMPLES
    )

    def __post_init__(self) -> None:

        if self.max_extrapolation <= 0.0:
            raise ValueError(
                "max_extrapolation must be > 0"
            )

        if self.min_extrapolation < 0.0:
            raise ValueError(
                "min_extrapolation must be >= 0"
            )

        if (
            self.min_extrapolation
            > self.max_extrapolation
        ):
            raise ValueError(
                "min_extrapolation cannot exceed "
                "max_extrapolation"
            )

        if self.point_spacing <= 0.0:
            raise ValueError(
                "point_spacing must be > 0"
            )

        if not (
            0.0
            <= self.minimum_confidence
            <= 1.0
        ):
            raise ValueError(
                "minimum_confidence must be in [0, 1]"
            )

        if not (
            0.0
            <= self.confidence_decay
            <= 1.0
        ):
            raise ValueError(
                "confidence_decay must be in [0, 1]"
            )

        if self.max_slope <= 0.0:
            raise ValueError(
                "max_slope must be > 0"
            )

        if self.max_second_derivative <= 0.0:
            raise ValueError(
                "max_second_derivative must be > 0"
            )

        if self.validation_samples < 3:
            raise ValueError(
                "validation_samples must be >= 3"
            )


# =============================================================================
# RESULTADO
# =============================================================================

@dataclass(frozen=True, slots=True)
class ProjectionResult:
    """
    Resultado de uma operação de projeção.

    projection:
        LaneProjection quando a operação foi válida.

    source_model:
        Modelo matemático que originou a projeção.

    valid:
        Indica se o resultado pode ser consumido.

    reason:
        Motivo de falha quando valid=False.
    """

    projection: Optional[LaneProjection]

    source_model: Optional[LanePolynomial]

    valid: bool

    requested_distance: float

    actual_distance: float

    confidence: float

    reason: str = ""

    @property
    def projected(self) -> bool:
        return (
            self.valid
            and self.projection is not None
        )


# =============================================================================
# UTILITÁRIOS
# =============================================================================

def _is_finite(value: float) -> bool:
    """Retorna True somente para valores reais finitos."""

    try:
        return math.isfinite(
            float(value)
        )
    except (
        TypeError,
        ValueError,
    ):
        return False


def _clamp(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    """Limita um valor a um intervalo."""

    return max(
        minimum,
        min(
            maximum,
            value,
        ),
    )


def _as_coefficients(
    model: LanePolynomial,
) -> Optional[
    Tuple[float, float, float, float]
]:
    """
    Extrai e valida os coeficientes do polinômio.

    Contrato:

        coefficients = (a, b, c, d)

    com:

        x(y) = a*y³ + b*y² + c*y + d
    """

    coefficients = model.coefficients

    if len(coefficients) != 4:
        return None

    values = tuple(
        float(value)
        for value in coefficients
    )

    if not all(
        _is_finite(value)
        for value in values
    ):
        return None

    return values  # type: ignore[return-value]


# =============================================================================
# AVALIAÇÃO DO POLINÔMIO
# =============================================================================

def evaluate_polynomial(
    model: LanePolynomial,
    y: float,
) -> float:
    """
    Avalia:

        x(y) = a*y³ + b*y² + c*y + d

    usando Horner para estabilidade numérica.
    """

    coefficients = _as_coefficients(
        model
    )

    if coefficients is None:
        raise ValueError(
            "LanePolynomial has invalid coefficients."
        )

    if not _is_finite(y):
        raise ValueError(
            "y must be finite."
        )

    a, b, c, d = coefficients

    return (
        (
            (
                a * y
                + b
            )
            * y
            + c
        )
        * y
        + d
    )


def evaluate_many(
    model: LanePolynomial,
    y_values: Sequence[float],
) -> np.ndarray:
    """
    Avalia o polinômio para vários valores de Y.
    """

    coefficients = _as_coefficients(
        model
    )

    if coefficients is None:
        raise ValueError(
            "LanePolynomial has invalid coefficients."
        )

    y = np.asarray(
        y_values,
        dtype=np.float64,
    )

    if not np.all(
        np.isfinite(y)
    ):
        raise ValueError(
            "y_values contains non-finite values."
        )

    a, b, c, d = coefficients

    return (
        (
            (
                a * y
                + b
            )
            * y
            + c
        )
        * y
        + d
    )


# =============================================================================
# DERIVADAS
# =============================================================================

def polynomial_slope(
    model: LanePolynomial,
    y: float,
) -> float:
    """
    Calcula:

        dx/dy

    para:

        x(y) = a*y³ + b*y² + c*y + d
    """

    coefficients = _as_coefficients(
        model
    )

    if coefficients is None:
        raise ValueError(
            "LanePolynomial has invalid coefficients."
        )

    a, b, c, _ = coefficients

    return (
        3.0 * a * y * y
        + 2.0 * b * y
        + c
    )


def polynomial_second_derivative(
    model: LanePolynomial,
    y: float,
) -> float:
    """
    Calcula:

        d²x/dy²
    """

    coefficients = _as_coefficients(
        model
    )

    if coefficients is None:
        raise ValueError(
            "LanePolynomial has invalid coefficients."
        )

    a, b, _, _ = coefficients

    return (
        6.0 * a * y
        + 2.0 * b
    )


# =============================================================================
# VALIDAÇÃO DO MODELO
# =============================================================================

def validate_polynomial(
    model: LanePolynomial,
    config: Optional[ProjectionConfig] = None,
) -> Tuple[
    bool,
    str,
]:
    """
    Valida um LanePolynomial antes de extrapolá-lo.

    A validação é deliberadamente conservadora.
    """

    if config is None:
        config = ProjectionConfig()

    if not isinstance(
        model,
        LanePolynomial,
    ):
        return (
            False,
            "invalid_model_type",
        )

    coefficients = _as_coefficients(
        model
    )

    if coefficients is None:
        return (
            False,
            "invalid_coefficients",
        )

    if not (
        _is_finite(model.y_min)
        and _is_finite(model.y_max)
    ):
        return (
            False,
            "invalid_y_range",
        )

    if model.y_max <= model.y_min:
        return (
            False,
            "invalid_y_range",
        )

    if not _is_finite(
        model.confidence
    ):
        return (
            False,
            "invalid_confidence",
        )

    if not (
        0.0
        <= model.confidence
        <= 1.0
    ):
        return (
            False,
            "confidence_out_of_range",
        )

    y_values = np.linspace(
        model.y_min,
        model.y_max,
        config.validation_samples,
        dtype=np.float64,
    )

    try:
        x_values = evaluate_many(
            model,
            y_values,
        )

    except ValueError:
        return (
            False,
            "polynomial_evaluation_failed",
        )

    if not np.all(
        np.isfinite(x_values)
    ):
        return (
            False,
            "non_finite_evaluation",
        )

    for y in y_values:

        slope = polynomial_slope(
            model,
            float(y),
        )

        second_derivative = (
            polynomial_second_derivative(
                model,
                float(y),
            )
        )

        if not (
            _is_finite(slope)
            and _is_finite(
                second_derivative
            )
        ):
            return (
                False,
                "non_finite_derivative",
            )

        if (
            abs(slope)
            > config.max_slope
        ):
            return (
                False,
                "slope_limit_exceeded",
            )

        if (
            abs(second_derivative)
            > config.max_second_derivative
        ):
            return (
                False,
                "curvature_limit_exceeded",
            )

    return (
        True,
        "ok",
    )


# =============================================================================
# CONFIANÇA
# =============================================================================

def projected_confidence(
    base_confidence: float,
    distance: float,
    config: Optional[ProjectionConfig] = None,
) -> float:
    """
    Calcula a confiança da projeção.

    Quanto maior a extrapolação, menor a confiança.

    A confiança nunca ultrapassa a confiança original.
    """

    if config is None:
        config = ProjectionConfig()

    if not _is_finite(
        base_confidence
    ):
        return 0.0

    if not _is_finite(
        distance
    ):
        return 0.0

    base = _clamp(
        float(base_confidence),
        0.0,
        1.0,
    )

    distance = max(
        0.0,
        float(distance),
    )

    normalized_distance = (
        distance
        / config.max_extrapolation
    )

    decay = (
        normalized_distance
        * config.confidence_decay
    )

    confidence = (
        base
        * (
            1.0
            - decay
        )
    )

    return _clamp(
        confidence,
        0.0,
        base,
    )


# =============================================================================
# DISTÂNCIA DE PROJEÇÃO
# =============================================================================

def clamp_projection_distance(
    distance: float,
    config: Optional[ProjectionConfig] = None,
) -> float:
    """
    Limita a distância de projeção ao intervalo permitido.
    """

    if config is None:
        config = ProjectionConfig()

    if not _is_finite(
        distance
    ):
        return 0.0

    return _clamp(
        float(distance),
        0.0,
        config.max_extrapolation,
    )


# =============================================================================
# GERAÇÃO DE PONTOS
# =============================================================================

def generate_projected_points(
    model: LanePolynomial,
    *,
    y_start: float,
    y_end: float,
    spacing: float,
) -> Tuple[LanePoint, ...]:
    """
    Gera LanePoints ao longo do trecho projetado.

    Os pontos são produzidos em ordem crescente de Y.
    """

    if not (
        _is_finite(y_start)
        and _is_finite(y_end)
    ):
        return ()

    if y_end <= y_start:
        return ()

    if not _is_finite(
        spacing
    ) or spacing <= 0.0:
        raise ValueError(
            "spacing must be > 0"
        )

    distance = (
        y_end
        - y_start
    )

    count = max(
        2,
        int(
            math.ceil(
                distance
                / spacing
            )
        )
        + 1,
    )

    y_values = np.linspace(
        y_start,
        y_end,
        count,
        dtype=np.float64,
    )

    x_values = evaluate_many(
        model,
        y_values,
    )

    points = []

    for x, y in zip(
        x_values,
        y_values,
    ):

        x = float(x)
        y = float(y)

        if not (
            _is_finite(x)
            and _is_finite(y)
        ):
            continue

        points.append(
            LanePoint(
                x=x,
                y=y,
            )
        )

    return tuple(
        points
    )


# =============================================================================
# INTERVALO DE PROJEÇÃO
# =============================================================================

def projection_interval(
    model: LanePolynomial,
    distance: float,
) -> Tuple[
    float,
    float,
]:
    """
    Retorna o intervalo vertical extrapolado.

    A convenção atual considera Y crescente para baixo na imagem.

    Portanto:

        início = y_max observado
        fim    = y_max + distância
    """

    if distance <= 0.0:
        return (
            model.y_max,
            model.y_max,
        )

    y_start = float(
        model.y_max
    )

    y_end = (
        y_start
        + float(distance)
    )

    return (
        y_start,
        y_end,
    )


# =============================================================================
# PROJEÇÃO DE UM MODELO
# =============================================================================

def project_model(
    model: LanePolynomial,
    *,
    lane_id: int,
    distance: float,
    config: Optional[ProjectionConfig] = None,
) -> ProjectionResult:
    """
    Projeta um LanePolynomial.

    A projeção começa exatamente no limite observado
    `model.y_max`.
    """

    if config is None:
        config = ProjectionConfig()

    requested_distance = (
        float(distance)
        if _is_finite(distance)
        else 0.0
    )

    requested_distance = max(
        0.0,
        requested_distance,
    )

    valid_model, reason = (
        validate_polynomial(
            model,
            config,
        )
    )

    if not valid_model:

        return ProjectionResult(
            projection=None,
            source_model=model,
            valid=False,
            requested_distance=requested_distance,
            actual_distance=0.0,
            confidence=0.0,
            reason=reason,
        )

    if requested_distance < (
        config.min_extrapolation
    ):

        return ProjectionResult(
            projection=None,
            source_model=model,
            valid=False,
            requested_distance=requested_distance,
            actual_distance=0.0,
            confidence=model.confidence,
            reason="distance_below_minimum",
        )

    actual_distance = (
        clamp_projection_distance(
            requested_distance,
            config,
        )
    )

    confidence = projected_confidence(
        model.confidence,
        actual_distance,
        config,
    )

    if (
        confidence
        < config.minimum_confidence
    ):

        return ProjectionResult(
            projection=None,
            source_model=model,
            valid=False,
            requested_distance=requested_distance,
            actual_distance=actual_distance,
            confidence=confidence,
            reason="confidence_below_threshold",
        )

    y_start, y_end = (
        projection_interval(
            model,
            actual_distance,
        )
    )

    points = (
        generate_projected_points(
            model,
            y_start=y_start,
            y_end=y_end,
            spacing=config.point_spacing,
        )
    )

    if len(points) < (
        DEFAULT_MINIMUM_POINTS
    ):

        return ProjectionResult(
            projection=None,
            source_model=model,
            valid=False,
            requested_distance=requested_distance,
            actual_distance=actual_distance,
            confidence=confidence,
            reason="insufficient_projected_points",
        )

    projection = LaneProjection(
        lane_id=int(lane_id),
        points=points,
        confidence=confidence,
        extrapolated_distance=actual_distance,
        source=LaneSource.PROJECTED,
    )

    return ProjectionResult(
        projection=projection,
        source_model=model,
        valid=True,
        requested_distance=requested_distance,
        actual_distance=actual_distance,
        confidence=confidence,
        reason="ok",
    )


# =============================================================================
# PROJEÇÃO DE LANE
# =============================================================================

def project_lane(
    lane: LaneLine,
    *,
    distance: float,
    config: Optional[ProjectionConfig] = None,
) -> ProjectionResult:
    """
    Projeta uma LaneLine usando seu modelo matemático.

    A LaneLine original nunca é modificada.
    """

    if not isinstance(
        lane,
        LaneLine,
    ):
        return ProjectionResult(
            projection=None,
            source_model=None,
            valid=False,
            requested_distance=max(
                0.0,
                float(distance)
                if _is_finite(distance)
                else 0.0,
            ),
            actual_distance=0.0,
            confidence=0.0,
            reason="invalid_lane_type",
        )

    if lane.model is None:

        return ProjectionResult(
            projection=None,
            source_model=None,
            valid=False,
            requested_distance=max(
                0.0,
                float(distance)
                if _is_finite(distance)
                else 0.0,
            ),
            actual_distance=0.0,
            confidence=0.0,
            reason="lane_without_model",
        )

    return project_model(
        lane.model,
        lane_id=lane.lane_id,
        distance=distance,
        config=config,
    )


# =============================================================================
# PROJEÇÃO DE MÚLTIPLAS LANES
# =============================================================================

def project_lanes(
    lanes: Iterable[LaneLine],
    *,
    distance: float,
    config: Optional[ProjectionConfig] = None,
) -> Tuple[
    ProjectionResult,
    ...,
]:
    """
    Projeta todas as lanes independentemente.

    Uma lane inválida não invalida as demais.
    """

    if config is None:
        config = ProjectionConfig()

    results = []

    for lane in lanes:

        results.append(
            project_lane(
                lane,
                distance=distance,
                config=config,
            )
        )

    return tuple(
        results
    )


# =============================================================================
# PROJEÇÃO ATÉ Y-ALVO
# =============================================================================

def projection_distance_to_y(
    model: LanePolynomial,
    target_y: float,
    *,
    config: Optional[ProjectionConfig] = None,
) -> float:
    """
    Calcula a distância necessária para projetar até target_y.

    Retorna zero quando target_y já está dentro da região
    observada.
    """

    if config is None:
        config = ProjectionConfig()

    if not _is_finite(
        target_y
    ):
        return 0.0

    target_y = float(
        target_y
    )

    if target_y <= model.y_max:
        return 0.0

    distance = (
        target_y
        - model.y_max
    )

    return clamp_projection_distance(
        distance,
        config,
    )


def project_lane_to_y(
    lane: LaneLine,
    *,
    target_y: float,
    config: Optional[ProjectionConfig] = None,
) -> ProjectionResult:
    """
    Projeta uma lane até um Y específico.
    """

    if lane.model is None:

        return ProjectionResult(
            projection=None,
            source_model=None,
            valid=False,
            requested_distance=0.0,
            actual_distance=0.0,
            confidence=0.0,
            reason="lane_without_model",
        )

    distance = (
        projection_distance_to_y(
            lane.model,
            target_y,
            config=config,
        )
    )

    if distance <= 0.0:

        return ProjectionResult(
            projection=None,
            source_model=lane.model,
            valid=False,
            requested_distance=0.0,
            actual_distance=0.0,
            confidence=lane.model.confidence,
            reason="target_already_observed",
        )

    return project_lane(
        lane,
        distance=distance,
        config=config,
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "ProjectionConfig",
    "ProjectionResult",
    "evaluate_polynomial",
    "evaluate_many",
    "polynomial_slope",
    "polynomial_second_derivative",
    "validate_polynomial",
    "projected_confidence",
    "clamp_projection_distance",
    "generate_projected_points",
    "projection_interval",
    "project_model",
    "project_lane",
    "project_lanes",
    "projection_distance_to_y",
    "project_lane_to_y",
]