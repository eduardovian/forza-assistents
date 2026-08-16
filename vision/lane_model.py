"""
vision/lane_model.py

Modelagem matemática das linhas de faixa.

Responsabilidades
-----------------
LanePoint
    ↓
validação
    ↓
filtragem
    ↓
preparação
    ↓
ajuste polinomial cúbico
    ↓
rejeição robusta de outliers
    ↓
validação geométrica
    ↓
LanePolynomial
    ↓
LaneProjection
    ↓
LaneModel

Este módulo NÃO realiza:
    - inferência YOLOP;
    - captura de tela;
    - ROI;
    - tracking temporal;
    - associação semântica;
    - decisão ADAS;
    - controle do veículo.

Princípios
----------
- configuração centralizada em config.py;
- nenhuma configuração paralela;
- polinômio oficial cúbico;
- normalização numérica;
- rejeição robusta de outliers;
- comportamento determinístico;
- falha segura.
"""

from __future__ import annotations

import logging
import math
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from config import LANE_MODEL, LANE_PROJECTION

from .lane_types import (
    LaneLine,
    LaneModel,
    LanePoint,
    LanePolynomial,
    LaneProjection,
    LaneQuality,
    ProjectionQuality,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================


def _validate_configuration() -> None:
    """Valida os invariantes da configuração global."""

    if LANE_MODEL.polynomial_degree != 3:
        raise ValueError(
            "LANE_MODEL.polynomial_degree deve ser 3."
        )

    if LANE_MODEL.minimum_points < 4:
        raise ValueError(
            "LANE_MODEL.minimum_points deve ser >= 4."
        )

    if LANE_MODEL.minimum_y_span <= 0.0:
        raise ValueError(
            "LANE_MODEL.minimum_y_span deve ser > 0."
        )

    if LANE_MODEL.max_outlier_iterations < 0:
        raise ValueError(
            "LANE_MODEL.max_outlier_iterations não pode ser negativo."
        )

    if LANE_MODEL.outlier_threshold <= 0.0:
        raise ValueError(
            "LANE_MODEL.outlier_threshold deve ser > 0."
        )

    if not 0.0 <= LANE_MODEL.minimum_confidence <= 1.0:
        raise ValueError(
            "LANE_MODEL.minimum_confidence deve estar entre 0 e 1."
        )

    if LANE_PROJECTION.samples < 2:
        raise ValueError(
            "LANE_PROJECTION.samples deve ser >= 2."
        )

    if LANE_PROJECTION.minimum_points < 2:
        raise ValueError(
            "LANE_PROJECTION.minimum_points deve ser >= 2."
        )


_validate_configuration()


# =============================================================================
# NUMERIC UTILITIES
# =============================================================================


def _clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _is_finite(value: float) -> bool:
    return math.isfinite(float(value))


def _safe_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0

    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]

    if array.size == 0:
        return 0.0

    return float(np.mean(array))


def _safe_median(values: Sequence[float]) -> float:
    if not values:
        return 0.0

    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]

    if array.size == 0:
        return 0.0

    return float(np.median(array))


# =============================================================================
# POINT PREPARATION
# =============================================================================


def filter_lane_points(
    points: Iterable[LanePoint],
    min_confidence: Optional[float] = None,
) -> List[LanePoint]:
    """
    Remove pontos inválidos, não finitos e abaixo
    do limite de confiança.

    min_confidence existe como parâmetro de compatibilidade
    da API, mas seu valor padrão vem exclusivamente de config.py.
    """

    threshold = (
        LANE_MODEL.minimum_confidence
        if min_confidence is None
        else _clip01(min_confidence)
    )

    result: List[LanePoint] = []

    for point in points:

        if not isinstance(point, LanePoint):
            continue

        if not point.is_valid():
            continue

        if point.confidence < threshold:
            continue

        result.append(point)

    return result


def sort_lane_points(
    points: Iterable[LanePoint],
) -> List[LanePoint]:
    """Ordena pontos por Y crescente."""

    return sorted(
        points,
        key=lambda point: float(point.y),
    )


def remove_duplicate_y(
    points: Iterable[LanePoint],
) -> List[LanePoint]:
    """
    Remove pontos com o mesmo Y.

    Em caso de duplicidade, mantém o ponto
    de maior confiança.
    """

    best_by_y: dict[float, LanePoint] = {}

    for point in points:

        y = float(point.y)
        current = best_by_y.get(y)

        if (
            current is None
            or point.confidence > current.confidence
        ):
            best_by_y[y] = point

    return sorted(
        best_by_y.values(),
        key=lambda point: point.y,
    )


def limit_point_count(
    points: Sequence[LanePoint],
    max_points: Optional[int] = None,
) -> List[LanePoint]:
    """
    Limita a quantidade de pontos preservando
    a distribuição vertical.
    """

    if max_points is None:
        max_points = max(
            LANE_MODEL.minimum_points,
            LANE_PROJECTION.samples,
        )

    maximum = max(
        LANE_MODEL.minimum_points,
        int(max_points),
    )

    if len(points) <= maximum:
        return list(points)

    indices = np.linspace(
        0,
        len(points) - 1,
        maximum,
    ).astype(int)

    return [
        points[int(index)]
        for index in indices
    ]


def prepare_lane_points(
    points: Iterable[LanePoint],
    min_confidence: Optional[float] = None,
    max_points: Optional[int] = None,
) -> List[LanePoint]:
    """Pipeline determinístico de preparação."""

    filtered = filter_lane_points(
        points,
        min_confidence=min_confidence,
    )

    ordered = sort_lane_points(filtered)

    unique = remove_duplicate_y(ordered)

    return limit_point_count(
        unique,
        max_points=max_points,
    )


# =============================================================================
# STATISTICS
# =============================================================================


def lane_y_span(
    points: Sequence[LanePoint],
) -> float:

    if len(points) < 2:
        return 0.0

    ys = np.asarray(
        [point.y for point in points],
        dtype=np.float64,
    )

    return float(
        np.max(ys) - np.min(ys)
    )


def lane_x_span(
    points: Sequence[LanePoint],
) -> float:

    if len(points) < 2:
        return 0.0

    xs = np.asarray(
        [point.x for point in points],
        dtype=np.float64,
    )

    return float(
        np.max(xs) - np.min(xs)
    )


def lane_mean_confidence(
    points: Sequence[LanePoint],
) -> float:

    return _clip01(
        _safe_mean(
            [
                point.confidence
                for point in points
                if point.is_valid()
            ]
        )
    )


def lane_confidence_score(
    points: Sequence[LanePoint],
) -> float:
    """
    Confiança estrutural da observação.

    Combina:
        - confiança dos pontos;
        - quantidade de pontos;
        - extensão vertical.
    """

    if not points:
        return 0.0

    confidence = lane_mean_confidence(points)

    count_score = _clip01(
        len(points) / 20.0
    )

    span_score = _clip01(
        lane_y_span(points) / 300.0
    )

    return _clip01(
        0.55 * confidence
        + 0.25 * count_score
        + 0.20 * span_score
    )


# =============================================================================
# QUALITY
# =============================================================================


def classify_lane_quality(
    points: Sequence[LanePoint],
    polynomial: Optional[LanePolynomial] = None,
) -> LaneQuality:

    if not points:
        return LaneQuality.NONE

    count = len(points)
    confidence = lane_mean_confidence(points)
    span = lane_y_span(points)

    if (
        count < LANE_MODEL.minimum_points
        or span < LANE_MODEL.minimum_y_span
    ):
        return LaneQuality.POOR

    if confidence < 0.35:
        return LaneQuality.PARTIAL

    if polynomial is not None:

        if (
            polynomial.valid
            and polynomial.confidence >= 0.75
            and polynomial.fit_error <= 12.0
        ):
            return LaneQuality.EXCELLENT

        if (
            polynomial.valid
            and polynomial.confidence >= 0.55
        ):
            return LaneQuality.GOOD

    if (
        count >= 15
        and span >= 150.0
        and confidence >= 0.60
    ):
        return LaneQuality.GOOD

    return LaneQuality.PARTIAL


# =============================================================================
# NUMERIC NORMALIZATION
# =============================================================================


def _normalize_y(
    y: np.ndarray,
) -> Tuple[np.ndarray, float, float]:

    center = float(np.mean(y))

    scale = float(
        np.max(
            np.abs(y - center)
        )
    )

    if scale < 1e-9:
        scale = 1.0

    return (
        (y - center) / scale,
        center,
        scale,
    )


def _denormalize_coefficients(
    coefficients: Sequence[float],
    center: float,
    scale: float,
) -> Tuple[float, float, float, float]:
    """
    Converte:

        x = A*z³ + B*z² + C*z + D

    em:

        x = a*y³ + b*y² + c*y + d

    com:

        z = (y - center) / scale
    """

    A, B, C, D = (
        float(value)
        for value in coefficients
    )

    scale2 = scale * scale
    scale3 = scale2 * scale

    a = A / scale3

    b = (
        B / scale2
        - 3.0 * A * center / scale3
    )

    c = (
        C / scale
        - 2.0 * B * center / scale2
        + 3.0 * A * center * center / scale3
    )

    d = (
        D
        - C * center / scale
        + B * center * center / scale2
        - A * center * center * center / scale3
    )

    return (
        float(a),
        float(b),
        float(c),
        float(d),
    )


# =============================================================================
# FIT CONFIDENCE
# =============================================================================


def _calculate_fit_confidence(
    points: Sequence[LanePoint],
    fit_error: float,
) -> float:

    if not points:
        return 0.0

    observation_score = lane_mean_confidence(points)

    count_score = _clip01(
        len(points) / 20.0
    )

    span_score = _clip01(
        lane_y_span(points) / 300.0
    )

    if not _is_finite(fit_error):
        error_score = 0.0
    else:
        error_score = float(
            np.exp(-fit_error / 20.0)
        )

    return _clip01(
        0.40 * observation_score
        + 0.20 * count_score
        + 0.15 * span_score
        + 0.25 * error_score
    )


# =============================================================================
# POLYNOMIAL FIT
# =============================================================================


def fit_polynomial(
    points: Sequence[LanePoint],
    degree: int = 3,
    min_points: Optional[int] = None,
    max_fit_error: Optional[float] = None,
) -> LanePolynomial:
    """
    Ajusta o polinômio oficial cúbico:

        x(y) = a*y³ + b*y² + c*y + d

    degree é aceito para validar o contrato da API,
    mas somente grau 3 é permitido.
    """

    if degree != LANE_MODEL.polynomial_degree:
        raise ValueError(
            "lane_model utiliza exclusivamente polinômio "
            f"de grau {LANE_MODEL.polynomial_degree}."
        )

    minimum_points = (
        LANE_MODEL.minimum_points
        if min_points is None
        else int(min_points)
    )

    fit_error_limit = (
        LANE_MODEL.max_fit_error
        if max_fit_error is None
        else float(max_fit_error)
    )

    if minimum_points < 4:
        raise ValueError(
            "min_points deve ser >= 4."
        )

    if len(points) < minimum_points:

        return LanePolynomial(
            valid=False,
            sample_count=len(points),
        )

    xs = np.asarray(
        [point.x for point in points],
        dtype=np.float64,
    )

    ys = np.asarray(
        [point.y for point in points],
        dtype=np.float64,
    )

    if not (
        np.all(np.isfinite(xs))
        and np.all(np.isfinite(ys))
    ):
        return LanePolynomial(
            valid=False,
            sample_count=len(points),
        )

    y_min = float(np.min(ys))
    y_max = float(np.max(ys))

    if (
        y_max - y_min
        < LANE_MODEL.minimum_y_span
    ):
        return LanePolynomial(
            valid=False,
            sample_count=len(points),
            y_min=y_min,
            y_max=y_max,
        )

    normalized_y, center, scale = _normalize_y(ys)

    try:

        coefficients = np.polynomial.polynomial.polyfit(
            normalized_y,
            xs,
            deg=3,
        )

    except (
        np.linalg.LinAlgError,
        ValueError,
        FloatingPointError,
    ):

        return LanePolynomial(
            valid=False,
            sample_count=len(points),
            y_min=y_min,
            y_max=y_max,
        )

    if len(coefficients) != 4:
        return LanePolynomial(
            valid=False,
            sample_count=len(points),
            y_min=y_min,
            y_max=y_max,
        )

    a, b, c, d = _denormalize_coefficients(
        coefficients,
        center,
        scale,
    )

    polynomial = LanePolynomial(
        a=a,
        b=b,
        c=c,
        d=d,
        valid=True,
        sample_count=len(points),
        y_min=y_min,
        y_max=y_max,
    )

    predicted = np.asarray(
        [
            polynomial.evaluate(y)
            for y in ys
        ],
        dtype=np.float64,
    )

    residuals = np.abs(
        xs - predicted
    )

    fit_error = _safe_median(
        residuals.tolist()
    )

    polynomial.fit_error = fit_error

    polynomial.confidence = (
        _calculate_fit_confidence(
            points,
            fit_error,
        )
    )

    if (
        not _is_finite(fit_error)
        or fit_error > fit_error_limit
        or polynomial.confidence
        < LANE_MODEL.minimum_confidence
    ):
        polynomial.valid = False

    return polynomial


# =============================================================================
# OUTLIER REJECTION
# =============================================================================


def remove_polynomial_outliers(
    points: Sequence[LanePoint],
    polynomial: LanePolynomial,
    threshold: Optional[float] = None,
) -> List[LanePoint]:
    """
    Remove pontos incompatíveis com o modelo.

    Usa MAD (Median Absolute Deviation) para robustez
    contra outliers isolados.
    """

    if (
        not polynomial.valid
        or len(points) < 4
    ):
        return list(points)

    threshold_value = (
        LANE_MODEL.outlier_threshold
        if threshold is None
        else float(threshold)
    )

    residuals: List[float] = []

    for point in points:

        predicted = polynomial.evaluate(point.y)

        residuals.append(
            abs(point.x - predicted)
        )

    median = _safe_median(residuals)

    deviations = np.abs(
        np.asarray(residuals, dtype=np.float64)
        - median
    )

    mad = _safe_median(
        deviations.tolist()
    )

    if mad < 1e-6:

        limit = max(
            5.0,
            median * threshold_value,
        )

    else:

        limit = max(
            5.0,
            median
            + threshold_value * 1.4826 * mad,
        )

    return [
        point
        for point, residual
        in zip(points, residuals)
        if residual <= limit
    ]


# =============================================================================
# ROBUST LANE FIT
# =============================================================================


def fit_lane_model(
    points: Iterable[LanePoint],
    min_points: Optional[int] = None,
    max_points: Optional[int] = None,
    min_confidence: Optional[float] = None,
    max_fit_error: Optional[float] = None,
) -> Optional[LanePolynomial]:
    """
    Pipeline completo:

        filtragem
            ↓
        preparação
            ↓
        fitting inicial
            ↓
        rejeição de outliers
            ↓
        fitting final
    """

    prepared = prepare_lane_points(
        points,
        min_confidence=min_confidence,
        max_points=max_points,
    )

    minimum_points = (
        LANE_MODEL.minimum_points
        if min_points is None
        else int(min_points)
    )

    if len(prepared) < minimum_points:
        return None

    initial = fit_polynomial(
        prepared,
        min_points=minimum_points,
        max_fit_error=max_fit_error,
    )

    if not initial.valid:
        return None

    cleaned = remove_polynomial_outliers(
        prepared,
        initial,
    )

    if len(cleaned) < minimum_points:
        cleaned = prepared

    final = fit_polynomial(
        cleaned,
        min_points=minimum_points,
        max_fit_error=max_fit_error,
    )

    if not final.valid:
        return None

    return final


# =============================================================================
# PROJECTION
# =============================================================================


def project_lane(
    polynomial: Optional[LanePolynomial],
    points: Sequence[LanePoint],
    projection_step: Optional[float] = None,
    minimum_confidence: Optional[float] = None,
    horizon_y: Optional[float] = None,
) -> LaneProjection:
    """
    Gera pontos matemáticos da faixa.

    A projeção não é considerada observação direta.
    """

    if polynomial is None:

        return LaneProjection(
            quality=ProjectionQuality.NONE,
            valid=False,
        )

    if not polynomial.valid:

        return LaneProjection(
            polynomial=polynomial,
            quality=ProjectionQuality.NONE,
            valid=False,
        )

    minimum = (
        LANE_PROJECTION.minimum_confidence
        if minimum_confidence is None
        else _clip01(minimum_confidence)
    )

    if polynomial.confidence < minimum:

        return LaneProjection(
            polynomial=polynomial,
            quality=ProjectionQuality.LOW,
            valid=False,
        )

    if len(points) < LANE_PROJECTION.minimum_points:

        return LaneProjection(
            polynomial=polynomial,
            quality=ProjectionQuality.LOW,
            valid=False,
        )

    observed_y_min = min(
        float(point.y)
        for point in points
    )

    observed_y_max = max(
        float(point.y)
        for point in points
    )

    if horizon_y is None:
        horizon_y = observed_y_min

    horizon_y = float(horizon_y)

    if not _is_finite(horizon_y):
        return LaneProjection(
            polynomial=polynomial,
            quality=ProjectionQuality.NONE,
            valid=False,
        )

    samples = max(
        2,
        int(LANE_PROJECTION.samples),
    )

    ys = np.linspace(
        observed_y_min,
        observed_y_max,
        samples,
        dtype=np.float64,
    )

    projected_points: List[LanePoint] = []

    for y in ys:

        x = polynomial.evaluate(float(y))

        if not (
            _is_finite(x)
            and _is_finite(float(y))
        ):
            continue

        projected_points.append(
            LanePoint(
                x=float(x),
                y=float(y),
                confidence=polynomial.confidence,
                valid=True,
            )
        )

    if len(projected_points) < LANE_PROJECTION.minimum_points:

        return LaneProjection(
            polynomial=polynomial,
            points=projected_points,
            quality=ProjectionQuality.LOW,
            extrapolated=False,
            valid=False,
            horizon_y=horizon_y,
        )

    if polynomial.confidence >= 0.80:
        quality = ProjectionQuality.HIGH

    elif polynomial.confidence >= 0.65:
        quality = ProjectionQuality.MEDIUM

    else:
        quality = ProjectionQuality.LOW

    return LaneProjection(
        polynomial=polynomial,
        points=projected_points,
        quality=quality,
        extrapolated=False,
        valid=True,
        horizon_y=horizon_y,
    )


# =============================================================================
# MODEL CONSTRUCTION
# =============================================================================


def build_lane_model(
    lane_id: int,
    points: Iterable[LanePoint],
    *,
    min_points: Optional[int] = None,
    max_points: Optional[int] = None,
    min_confidence: Optional[float] = None,
    max_fit_error: Optional[float] = None,
) -> LaneModel:
    """
    Constrói um LaneModel completo.

    lane_id deve ser fornecido pelo sistema responsável
    pela identidade da faixa.
    """

    prepared = prepare_lane_points(
        points,
        min_confidence=min_confidence,
        max_points=max_points,
    )

    confidence = lane_confidence_score(
        prepared
    )

    line = LaneLine(
        lane_id=lane_id,
        points=prepared,
        confidence=confidence,
        quality=LaneQuality.NONE,
        detected_directly=True,
        projected=False,
        valid=bool(prepared),
    )

    polynomial = fit_lane_model(
        prepared,
        min_points=min_points,
        max_points=max_points,
        min_confidence=min_confidence,
        max_fit_error=max_fit_error,
    )

    line.quality = classify_lane_quality(
        prepared,
        polynomial,
    )

    if polynomial is None:

        return LaneModel(
            lane_id=lane_id,
            line=line,
            polynomial=None,
            projection=None,
            tracked=False,
            stable=False,
            valid=False,
        )

    projection = project_lane(
        polynomial,
        prepared,
    )

    model_confidence = _clip01(
        0.45 * line.confidence
        + 0.55 * polynomial.confidence
    )

    line.confidence = model_confidence

    return LaneModel(
        lane_id=lane_id,
        line=line,
        polynomial=polynomial,
        projection=projection,
        tracked=False,
        stable=False,
        valid=(
            polynomial.valid
            and polynomial.confidence
            >= LANE_MODEL.minimum_confidence
        ),
    )


# =============================================================================
# MODEL UPDATE
# =============================================================================


def update_lane_model(
    model: LaneModel,
    points: Iterable[LanePoint],
    *,
    min_points: Optional[int] = None,
    max_points: Optional[int] = None,
    min_confidence: Optional[float] = None,
    max_fit_error: Optional[float] = None,
) -> LaneModel:
    """
    Recalcula o modelo preservando sua identidade temporal.
    """

    updated = build_lane_model(
        lane_id=model.lane_id,
        points=points,
        min_points=min_points,
        max_points=max_points,
        min_confidence=min_confidence,
        max_fit_error=max_fit_error,
    )

    updated.tracked = model.tracked
    updated.stable = model.stable

    if updated.line is not None and model.line is not None:

        updated.line.age_frames = (
            model.line.age_frames + 1
        )

        updated.line.missed_frames = 0

    return updated


# =============================================================================
# VALIDATION
# =============================================================================


def validate_lane_model(
    model: Optional[LaneModel],
) -> bool:
    """Validação estrutural completa."""

    if model is None:
        return False

    if not model.valid:
        return False

    if model.line is None:
        return False

    if model.polynomial is None:
        return False

    if not model.line.valid:
        return False

    if not model.polynomial.valid:
        return False

    if model.line.point_count() < 2:
        return False

    if not model.is_finite():
        return False

    if not _is_finite(
        model.polynomial.confidence
    ):
        return False

    if not _is_finite(
        model.polynomial.fit_error
    ):
        return False

    return True


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    "filter_lane_points",
    "sort_lane_points",
    "remove_duplicate_y",
    "limit_point_count",
    "prepare_lane_points",
    "lane_y_span",
    "lane_x_span",
    "lane_mean_confidence",
    "lane_confidence_score",
    "classify_lane_quality",
    "fit_polynomial",
    "remove_polynomial_outliers",
    "fit_lane_model",
    "project_lane",
    "build_lane_model",
    "update_lane_model",
    "validate_lane_model",
]