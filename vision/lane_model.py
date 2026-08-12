"""
vision/lane_model.py

Construção do modelo matemático das linhas de faixa.

Responsabilidades:

    LanePoint
        ↓
    filtragem
        ↓
    preparação
        ↓
    ajuste polinomial
        ↓
    validação
        ↓
    LaneModel

Não realiza:
    - inferência YOLOP;
    - tracking temporal;
    - associação semântica;
    - posição do veículo;
    - decisão ADAS.
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

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
# CONFIGURAÇÃO
# =============================================================================

DEFAULT_MIN_POINTS = 8
DEFAULT_MAX_POINTS = 80
DEFAULT_MIN_Y_SPAN = 30.0
DEFAULT_MAX_FIT_ERROR = 35.0
DEFAULT_OUTLIER_THRESHOLD = 3.0
DEFAULT_PROJECTION_STEP = 10.0
DEFAULT_MIN_CONFIDENCE = 0.20
DEFAULT_MIN_POLYNOMIAL_CONFIDENCE = 0.45
DEFAULT_MIN_PROJECTION_CONFIDENCE = 0.55


# =============================================================================
# UTILITÁRIOS
# =============================================================================

def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _finite(value: float) -> bool:
    return bool(np.isfinite(value))


def _safe_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0

    array = np.asarray(
        values,
        dtype=np.float64,
    )

    array = array[np.isfinite(array)]

    if array.size == 0:
        return 0.0

    return float(np.mean(array))


def _safe_median(values: Sequence[float]) -> float:
    if not values:
        return 0.0

    array = np.asarray(
        values,
        dtype=np.float64,
    )

    array = array[np.isfinite(array)]

    if array.size == 0:
        return 0.0

    return float(np.median(array))


# =============================================================================
# PREPARAÇÃO DOS PONTOS
# =============================================================================

def filter_lane_points(
    points: Iterable[LanePoint],
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> List[LanePoint]:
    """
    Remove pontos inválidos, não finitos ou abaixo da confiança mínima.
    """

    threshold = _clip01(
        min_confidence
    )

    result: List[LanePoint] = []

    for point in points:

        if not point.valid:
            continue

        if not (
            _finite(point.x)
            and _finite(point.y)
            and _finite(point.confidence)
        ):
            continue

        if point.confidence < threshold:
            continue

        result.append(point)

    return result


def sort_lane_points(
    points: Iterable[LanePoint],
) -> List[LanePoint]:
    """
    Ordena os pontos por Y crescente.
    """

    return sorted(
        points,
        key=lambda point: point.y,
    )


def remove_duplicate_y(
    points: Iterable[LanePoint],
) -> List[LanePoint]:
    """
    Mantém somente um ponto por coordenada Y.

    Em caso de duplicidade, permanece o ponto
    com maior confiança.
    """

    best: dict[float, LanePoint] = {}

    for point in points:

        previous = best.get(
            point.y
        )

        if (
            previous is None
            or point.confidence
            > previous.confidence
        ):
            best[point.y] = point

    return sorted(
        best.values(),
        key=lambda point: point.y,
    )


def limit_point_count(
    points: Sequence[LanePoint],
    max_points: int = DEFAULT_MAX_POINTS,
) -> List[LanePoint]:
    """
    Reduz a quantidade de pontos preservando
    a distribuição vertical.
    """

    max_points = max(
        2,
        int(max_points),
    )

    if len(points) <= max_points:
        return list(points)

    indices = np.linspace(
        0,
        len(points) - 1,
        max_points,
    ).astype(int)

    return [
        points[int(index)]
        for index in indices
    ]


def prepare_lane_points(
    points: Iterable[LanePoint],
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    max_points: int = DEFAULT_MAX_POINTS,
) -> List[LanePoint]:
    """
    Pipeline completo de preparação.
    """

    filtered = filter_lane_points(
        points,
        min_confidence,
    )

    ordered = sort_lane_points(
        filtered
    )

    unique = remove_duplicate_y(
        ordered
    )

    return limit_point_count(
        unique,
        max_points,
    )


# =============================================================================
# ESTATÍSTICAS
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
        np.max(ys)
        - np.min(ys)
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
        np.max(xs)
        - np.min(xs)
    )


def lane_mean_confidence(
    points: Sequence[LanePoint],
) -> float:
    return _clip01(
        _safe_mean(
            [
                point.confidence
                for point in points
                if point.valid
            ]
        )
    )


def lane_confidence_score(
    points: Sequence[LanePoint],
) -> float:
    """
    Confiança estrutural da observação.

    Combina:
        - confiança média;
        - quantidade de pontos;
        - extensão vertical.
    """

    if not points:
        return 0.0

    confidence = lane_mean_confidence(
        points
    )

    count_score = _clip01(
        len(points) / 30.0
    )

    span_score = _clip01(
        lane_y_span(points) / 500.0
    )

    return _clip01(
        0.55 * confidence
        + 0.25 * count_score
        + 0.20 * span_score
    )


# =============================================================================
# QUALIDADE
# =============================================================================

def classify_lane_quality(
    points: Sequence[LanePoint],
    polynomial: Optional[LanePolynomial] = None,
) -> LaneQuality:

    if not points:
        return LaneQuality.NONE

    count = len(points)

    confidence = lane_mean_confidence(
        points
    )

    span = lane_y_span(
        points
    )

    if (
        count < 4
        or span < 20.0
    ):
        return LaneQuality.POOR

    if (
        count < 8
        or span < 50.0
        or confidence < 0.35
    ):
        return LaneQuality.PARTIAL

    if polynomial is not None:

        if (
            polynomial.valid
            and polynomial.fit_error <= 12.0
            and polynomial.confidence >= 0.70
        ):
            return LaneQuality.EXCELLENT

    if (
        count >= 15
        and span >= 150.0
        and confidence >= 0.60
    ):
        return LaneQuality.GOOD

    return LaneQuality.PARTIAL


# =============================================================================
# AJUSTE POLINOMIAL
# =============================================================================

def _normalize_y(
    y: np.ndarray,
) -> Tuple[
    np.ndarray,
    float,
    float,
]:
    center = float(
        np.mean(y)
    )

    scale = float(
        np.max(
            np.abs(
                y - center
            )
        )
    )

    if scale < 1e-9:
        scale = 1.0

    normalized = (
        y - center
    ) / scale

    return (
        normalized,
        center,
        scale,
    )


def _polynomial_from_coefficients(
    coefficients: Sequence[float],
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

        x = an*z³ + bn*z² + cn*z + dn

    onde:

        z = (y - center) / scale

    para:

        x = a*y³ + b*y² + c*y + d
    """

    an, bn, cn, dn = (
        float(value)
        for value in coefficients
    )

    scale2 = scale ** 2
    scale3 = scale ** 3

    a = an / scale3

    b = (
        -3.0 * an * center / scale3
        + bn / scale2
    )

    c = (
        3.0 * an * center ** 2 / scale3
        - 2.0 * bn * center / scale2
        + cn / scale
    )

    d = (
        -an * center ** 3 / scale3
        + bn * center ** 2 / scale2
        - cn * center / scale
        + dn
    )

    return (
        float(a),
        float(b),
        float(c),
        float(d),
    )


def _calculate_fit_confidence(
    points: Sequence[LanePoint],
    fit_error: float,
) -> float:
    """
    Confiança do ajuste matemático.

    Não representa probabilidade.
    """

    if not points:
        return 0.0

    observation_confidence = (
        lane_mean_confidence(points)
    )

    count_score = _clip01(
        len(points) / 20.0
    )

    span_score = _clip01(
        lane_y_span(points) / 300.0
    )

    if not _finite(fit_error):
        error_score = 0.0
    else:
        error_score = float(
            np.exp(
                -fit_error / 20.0
            )
        )

    return _clip01(
        0.40 * observation_confidence
        + 0.20 * count_score
        + 0.15 * span_score
        + 0.25 * error_score
    )


def fit_polynomial(
    points: Sequence[LanePoint],
    degree: int = 3,
    min_points: int = DEFAULT_MIN_POINTS,
    max_fit_error: float = DEFAULT_MAX_FIT_ERROR,
) -> LanePolynomial:
    """
    Ajusta:

        x(y) = a*y³ + b*y² + c*y + d

    aos pontos observados.
    """

    if degree != 3:
        raise ValueError(
            "lane_model utiliza polinômio de grau 3."
        )

    if len(points) < min_points:

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

    y_min = float(
        np.min(ys)
    )

    y_max = float(
        np.max(ys)
    )

    if (
        y_max - y_min
        < DEFAULT_MIN_Y_SPAN
    ):

        return LanePolynomial(
            valid=False,
            sample_count=len(points),
            y_min=y_min,
            y_max=y_max,
        )

    normalized_y, center, scale = (
        _normalize_y(ys)
    )

    try:

        coefficients = np.polyfit(
            normalized_y,
            xs,
            3,
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

    a, b, c, d = (
        _polynomial_from_coefficients(
            coefficients,
            center,
            scale,
        )
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
        fit_error > max_fit_error
        or polynomial.confidence
        < DEFAULT_MIN_POLYNOMIAL_CONFIDENCE
    ):

        polynomial.valid = False

    return polynomial


# =============================================================================
# REMOÇÃO DE OUTLIERS
# =============================================================================

def remove_polynomial_outliers(
    points: Sequence[LanePoint],
    polynomial: LanePolynomial,
    threshold: float = DEFAULT_OUTLIER_THRESHOLD,
) -> List[LanePoint]:
    """
    Remove pontos muito afastados do modelo inicial.
    """

    if (
        not polynomial.valid
        or len(points) < 4
    ):
        return list(points)

    residuals = []

    for point in points:

        predicted = polynomial.evaluate(
            point.y
        )

        residuals.append(
            abs(
                point.x
                - predicted
            )
        )

    median = _safe_median(
        residuals
    )

    deviations = np.abs(
        np.asarray(
            residuals,
            dtype=np.float64
        )
        - median
    )

    mad = _safe_median(
        deviations.tolist()
    )

    if mad < 1e-6:
        limit = max(
            5.0,
            median * threshold,
        )
    else:
        limit = max(
            5.0,
            median
            + threshold * 1.4826 * mad,
        )

    return [
        point
        for point, residual
        in zip(points, residuals)
        if residual <= limit
    ]


# =============================================================================
# AJUSTE ROBUSTO
# =============================================================================

def fit_lane_model(
    points: Iterable[LanePoint],
    min_points: int = DEFAULT_MIN_POINTS,
    max_points: int = DEFAULT_MAX_POINTS,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    max_fit_error: float = DEFAULT_MAX_FIT_ERROR,
) -> Optional[LanePolynomial]:
    """
    Pipeline robusto:

        filtragem
            ↓
        preparação
            ↓
        ajuste inicial
            ↓
        remoção de outliers
            ↓
        ajuste final
    """

    prepared = prepare_lane_points(
        points,
        min_confidence=min_confidence,
        max_points=max_points,
    )

    if len(prepared) < min_points:
        return None

    initial = fit_polynomial(
        prepared,
        min_points=min_points,
        max_fit_error=max_fit_error,
    )

    if not initial.valid:
        return None

    cleaned = remove_polynomial_outliers(
        prepared,
        initial,
    )

    if len(cleaned) < min_points:
        cleaned = prepared

    final = fit_polynomial(
        cleaned,
        min_points=min_points,
        max_fit_error=max_fit_error,
    )

    if not final.valid:
        return None

    return final


# =============================================================================
# PROJEÇÃO
# =============================================================================

def project_lane(
    polynomial: Optional[LanePolynomial],
    points: Sequence[LanePoint],
    projection_step: float = DEFAULT_PROJECTION_STEP,
    minimum_confidence: float = DEFAULT_MIN_PROJECTION_CONFIDENCE,
    horizon_y: Optional[float] = None,
) -> LaneProjection:
    """
    Gera uma continuação matemática da linha.

    A projeção nunca é marcada como observação direta.
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

    if polynomial.confidence < minimum_confidence:
        return LaneProjection(
            polynomial=polynomial,
            quality=ProjectionQuality.LOW,
            valid=False,
        )

    if not points:
        return LaneProjection(
            polynomial=polynomial,
            quality=ProjectionQuality.LOW,
            valid=False,
        )

    observed_y_min = min(
        point.y
        for point in points
    )

    observed_y_max = max(
        point.y
        for point in points
    )

    if horizon_y is None:
        horizon_y = observed_y_min

    horizon_y = float(
        max(
            horizon_y,
            observed_y_min,
        )
    )

    step = max(
        1.0,
        float(projection_step),
    )

    start = observed_y_min

    end = observed_y_max

    ys = np.arange(
        start,
        end + step,
        step,
        dtype=np.float64,
    )

    if ys.size == 0:
        ys = np.asarray(
            [start, end],
            dtype=np.float64,
        )

    projected_points = [
        LanePoint(
            x=float(
                polynomial.evaluate(y)
            ),
            y=float(y),
            confidence=float(
                polynomial.confidence
            ),
            valid=True,
        )
        for y in ys
        if (
            _finite(
                polynomial.evaluate(y)
            )
            and _finite(y)
        )
    ]

    if len(projected_points) < 2:
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
        extrapolated=True,
        valid=True,
        horizon_y=horizon_y,
    )


# =============================================================================
# CONSTRUÇÃO DO LANE MODEL
# =============================================================================

def build_lane_model(
    lane_id: int,
    points: Iterable[LanePoint],
    *,
    min_points: int = DEFAULT_MIN_POINTS,
    max_points: int = DEFAULT_MAX_POINTS,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    max_fit_error: float = DEFAULT_MAX_FIT_ERROR,
) -> LaneModel:
    """
    Constrói um LaneModel completo a partir dos pontos.

    O lane_id recebido deve ser fornecido pelo tracker.
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
            >= DEFAULT_MIN_POLYNOMIAL_CONFIDENCE
        ),
    )


# =============================================================================
# ATUALIZAÇÃO DE MODELO EXISTENTE
# =============================================================================

def update_lane_model(
    model: LaneModel,
    points: Iterable[LanePoint],
    *,
    min_points: int = DEFAULT_MIN_POINTS,
    max_points: int = DEFAULT_MAX_POINTS,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    max_fit_error: float = DEFAULT_MAX_FIT_ERROR,
) -> LaneModel:
    """
    Recalcula o modelo mantendo a identidade lane_id.
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

    updated.line.age_frames = (
        model.line.age_frames + 1
    )

    updated.line.missed_frames = 0

    return updated


# =============================================================================
# VALIDAÇÃO
# =============================================================================

def validate_lane_model(
    model: Optional[LaneModel],
) -> bool:
    """
    Validação estrutural do modelo.
    """

    if model is None:
        return False

    if not model.valid:
        return False

    if model.polynomial is None:
        return False

    if not model.polynomial.valid:
        return False

    if model.line.point_count() < 2:
        return False

    if not (
        _finite(
            model.polynomial.confidence
        )
        and _finite(
            model.polynomial.fit_error
        )
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