"""
vision/lane_model.py

Modelagem matemática oficial das linhas de faixa.

Responsabilidade
----------------
Este módulo transforma observações de LanePoint em um modelo
matemático cúbico estável:

    LanePoint
        ↓
    validação
        ↓
    filtragem
        ↓
    preparação
        ↓
    normalização
        ↓
    ajuste cúbico
        ↓
    rejeição robusta de outliers
        ↓
    ajuste final
        ↓
    LanePolynomial
        ↓
    LaneModel

Este módulo NÃO realiza:

    - inferência YOLOP;
    - captura de tela;
    - ROI;
    - tracking temporal;
    - associação de lanes;
    - extrapolação/projeção espacial;
    - decisão ADAS;
    - controle do veículo.

Modelo matemático oficial
-------------------------

A faixa é representada por:

    x(y) = a*y³ + b*y² + c*y + d

onde:

    y -> coordenada vertical da imagem
    x -> coordenada horizontal da faixa

O ajuste é realizado com coordenadas Y normalizadas para
reduzir problemas numéricos.

Princípios
----------

- somente polinômio cúbico;
- determinístico;
- numericamente estável;
- robusto contra outliers;
- somente dados finitos;
- falha segura;
- configuração centralizada em config.py;
- nenhuma extrapolação neste módulo.
"""

from __future__ import annotations

import logging
import math
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from config import LANE_MODEL

from .lane_types import (
    LaneLine,
    LaneModel,
    LanePoint,
    LanePolynomial,
    LaneQuality,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================


def _validate_configuration() -> None:
    """Valida os invariantes necessários para o modelo."""

    if LANE_MODEL.polynomial_degree != 3:
        raise ValueError(
            "LANE_MODEL.polynomial_degree deve ser exatamente 3."
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

    if LANE_MODEL.max_fit_error <= 0.0:
        raise ValueError(
            "LANE_MODEL.max_fit_error deve ser > 0."
        )


_validate_configuration()


# =============================================================================
# NUMERIC UTILITIES
# =============================================================================


def _clip01(value: float) -> float:
    """Limita um valor ao intervalo [0, 1]."""

    return float(
        np.clip(
            float(value),
            0.0,
            1.0,
        )
    )


def _is_finite(value: float) -> bool:
    """Retorna True quando o valor é finito."""

    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _safe_mean(values: Sequence[float]) -> float:
    """Média segura ignorando valores não finitos."""

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
    """Mediana segura ignorando valores não finitos."""

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
# POINT FILTERING
# =============================================================================


def filter_lane_points(
    points: Iterable[LanePoint],
    min_confidence: Optional[float] = None,
) -> List[LanePoint]:
    """
    Remove pontos inválidos, não finitos ou abaixo da confiança mínima.
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

        try:
            x = float(point.x)
            y = float(point.y)
            confidence = float(point.confidence)
        except (TypeError, ValueError):
            continue

        if not (
            math.isfinite(x)
            and math.isfinite(y)
            and math.isfinite(confidence)
        ):
            continue

        if not point.is_valid():
            continue

        if confidence < threshold:
            continue

        result.append(point)

    return result


def sort_lane_points(
    points: Iterable[LanePoint],
) -> List[LanePoint]:
    """Ordena pontos pela coordenada Y crescente."""

    return sorted(
        points,
        key=lambda point: float(point.y),
    )


def remove_duplicate_y(
    points: Iterable[LanePoint],
) -> List[LanePoint]:
    """
    Remove múltiplos pontos com o mesmo Y.

    Quando existem vários pontos no mesmo Y,
    somente o de maior confiança é preservado.
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
        key=lambda point: float(point.y),
    )


def limit_point_count(
    points: Sequence[LanePoint],
    max_points: Optional[int] = None,
) -> List[LanePoint]:
    """
    Limita a quantidade de pontos mantendo distribuição vertical.

    O primeiro e o último ponto são sempre preservados.
    """

    if max_points is None:
        return list(points)

    maximum = int(max_points)

    if maximum < LANE_MODEL.minimum_points:
        maximum = LANE_MODEL.minimum_points

    if len(points) <= maximum:
        return list(points)

    indices = np.linspace(
        0,
        len(points) - 1,
        maximum,
        dtype=np.int64,
    )

    return [
        points[int(index)]
        for index in indices
    ]


def prepare_lane_points(
    points: Iterable[LanePoint],
    min_confidence: Optional[float] = None,
    max_points: Optional[int] = None,
) -> List[LanePoint]:
    """
    Pipeline determinístico de preparação dos pontos.
    """

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
# GEOMETRIC STATISTICS
# =============================================================================


def lane_y_span(
    points: Sequence[LanePoint],
) -> float:
    """Extensão vertical observada."""

    if len(points) < 2:
        return 0.0

    ys = np.asarray(
        [point.y for point in points],
        dtype=np.float64,
    )

    if not np.all(np.isfinite(ys)):
        return 0.0

    return float(
        np.max(ys) - np.min(ys)
    )


def lane_x_span(
    points: Sequence[LanePoint],
) -> float:
    """Extensão horizontal observada."""

    if len(points) < 2:
        return 0.0

    xs = np.asarray(
        [point.x for point in points],
        dtype=np.float64,
    )

    if not np.all(np.isfinite(xs)):
        return 0.0

    return float(
        np.max(xs) - np.min(xs)
    )


def lane_mean_confidence(
    points: Sequence[LanePoint],
) -> float:
    """Confiança média dos pontos válidos."""

    values = [
        float(point.confidence)
        for point in points
        if point.is_valid()
        and _is_finite(point.confidence)
    ]

    return _clip01(
        _safe_mean(values)
    )


def lane_confidence_score(
    points: Sequence[LanePoint],
) -> float:
    """
    Calcula a qualidade estrutural da observação.

    Componentes:

        55% -> confiança média
        25% -> quantidade de pontos
        20% -> extensão vertical
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
    """Classifica a qualidade geométrica da faixa."""

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

    if polynomial is not None and polynomial.valid:

        if (
            polynomial.confidence >= 0.75
            and polynomial.fit_error <= 12.0
        ):
            return LaneQuality.EXCELLENT

        if polynomial.confidence >= 0.55:
            return LaneQuality.GOOD

    if (
        count >= 15
        and span >= 150.0
        and confidence >= 0.60
    ):
        return LaneQuality.GOOD

    if confidence >= 0.35:
        return LaneQuality.PARTIAL

    return LaneQuality.POOR


# =============================================================================
# NORMALIZATION
# =============================================================================


def _normalize_y(
    y: np.ndarray,
) -> Tuple[np.ndarray, float, float]:
    """
    Normaliza Y para:

        z = (y - center) / scale
    """

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

    if not math.isfinite(scale) or scale < 1e-12:
        scale = 1.0

    normalized = (
        y - center
    ) / scale

    return (
        normalized,
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

    para:

        x = a*y³ + b*y² + c*y + d
    """

    A = float(coefficients[3])
    B = float(coefficients[2])
    C = float(coefficients[1])
    D = float(coefficients[0])

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
# POLYNOMIAL FIT
# =============================================================================


def _invalid_polynomial(
    sample_count: int,
    y_min: float = 0.0,
    y_max: float = 0.0,
) -> LanePolynomial:
    """Cria um resultado de fitting inválido."""

    return LanePolynomial(
        valid=False,
        sample_count=int(sample_count),
        y_min=float(y_min),
        y_max=float(y_max),
    )


def _calculate_fit_confidence(
    points: Sequence[LanePoint],
    fit_error: float,
) -> float:
    """Calcula confiança do ajuste."""

    if not points:
        return 0.0

    observation_score = lane_mean_confidence(
        points
    )

    count_score = _clip01(
        len(points) / 20.0
    )

    span_score = _clip01(
        lane_y_span(points) / 300.0
    )

    if not math.isfinite(fit_error):
        error_score = 0.0
    else:
        error_score = float(
            np.exp(
                -fit_error / 20.0
            )
        )

    return _clip01(
        0.40 * observation_score
        + 0.20 * count_score
        + 0.15 * span_score
        + 0.25 * error_score
    )


def fit_polynomial(
    points: Sequence[LanePoint],
    degree: int = 3,
    min_points: Optional[int] = None,
    max_fit_error: Optional[float] = None,
) -> LanePolynomial:
    """
    Ajusta exclusivamente o polinômio cúbico oficial:

        x(y) = a*y³ + b*y² + c*y + d

    O ajuste é realizado em Y normalizado.
    """

    if degree != 3:
        raise ValueError(
            "lane_model suporta exclusivamente "
            "polinômio cúbico."
        )

    minimum = (
        LANE_MODEL.minimum_points
        if min_points is None
        else int(min_points)
    )

    if minimum < 4:
        raise ValueError(
            "min_points deve ser >= 4."
        )

    error_limit = (
        LANE_MODEL.max_fit_error
        if max_fit_error is None
        else float(max_fit_error)
    )

    if error_limit <= 0.0:
        raise ValueError(
            "max_fit_error deve ser > 0."
        )

    if len(points) < minimum:
        return _invalid_polynomial(
            len(points)
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
        return _invalid_polynomial(
            len(points)
        )

    y_min = float(
        np.min(ys)
    )

    y_max = float(
        np.max(ys)
    )

    y_span = y_max - y_min

    if y_span < LANE_MODEL.minimum_y_span:
        return _invalid_polynomial(
            len(points),
            y_min,
            y_max,
        )

    normalized_y, center, scale = _normalize_y(
        ys
    )

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

        return _invalid_polynomial(
            len(points),
            y_min,
            y_max,
        )

    if len(coefficients) != 4:
        return _invalid_polynomial(
            len(points),
            y_min,
            y_max,
        )

    a, b, c, d = _denormalize_coefficients(
        coefficients,
        center,
        scale,
    )

    if not all(
        math.isfinite(value)
        for value in (a, b, c, d)
    ):
        return _invalid_polynomial(
            len(points),
            y_min,
            y_max,
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

    predicted = (
        a * ys**3
        + b * ys**2
        + c * ys
        + d
    )

    if not np.all(
        np.isfinite(predicted)
    ):
        return _invalid_polynomial(
            len(points),
            y_min,
            y_max,
        )

    residuals = np.abs(
        xs - predicted
    )

    fit_error = _safe_median(
        residuals.tolist()
    )

    confidence = _calculate_fit_confidence(
        points,
        fit_error,
    )

    polynomial.fit_error = fit_error
    polynomial.confidence = confidence

    if (
        not math.isfinite(fit_error)
        or fit_error > error_limit
        or confidence < LANE_MODEL.minimum_confidence
    ):
        polynomial.valid = False

    return polynomial


# =============================================================================
# OUTLIER REJECTION
# =============================================================================


def _polynomial_residuals(
    points: Sequence[LanePoint],
    polynomial: LanePolynomial,
) -> np.ndarray:
    """Calcula resíduos absolutos."""

    if not points or not polynomial.valid:
        return np.empty(
            0,
            dtype=np.float64,
        )

    xs = np.asarray(
        [point.x for point in points],
        dtype=np.float64,
    )

    ys = np.asarray(
        [point.y for point in points],
        dtype=np.float64,
    )

    predicted = np.asarray(
        [
            polynomial.evaluate(float(y))
            for y in ys
        ],
        dtype=np.float64,
    )

    residuals = np.abs(
        xs - predicted
    )

    return residuals


def remove_polynomial_outliers(
    points: Sequence[LanePoint],
    polynomial: LanePolynomial,
    threshold: Optional[float] = None,
) -> List[LanePoint]:
    """
    Remove outliers utilizando MAD.

    MAD:

        median(|residual - median(residual)|)

    O método é robusto contra poucos pontos
    extremamente afastados da curva.
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

    if threshold_value <= 0.0:
        raise ValueError(
            "threshold deve ser > 0."
        )

    residuals = _polynomial_residuals(
        points,
        polynomial,
    )

    if residuals.size != len(points):
        return list(points)

    if not np.all(
        np.isfinite(residuals)
    ):
        return list(points)

    median = float(
        np.median(residuals)
    )

    deviations = np.abs(
        residuals - median
    )

    mad = float(
        np.median(deviations)
    )

    if mad < 1e-9:

        limit = max(
            5.0,
            median * threshold_value,
        )

    else:

        robust_sigma = (
            1.4826 * mad
        )

        limit = max(
            5.0,
            median
            + threshold_value
            * robust_sigma,
        )

    return [
        point
        for point, residual
        in zip(points, residuals)
        if float(residual) <= limit
    ]


# =============================================================================
# ROBUST FIT
# =============================================================================


def fit_lane_model(
    points: Iterable[LanePoint],
    min_points: Optional[int] = None,
    max_points: Optional[int] = None,
    min_confidence: Optional[float] = None,
    max_fit_error: Optional[float] = None,
) -> Optional[LanePolynomial]:
    """
    Pipeline oficial de modelagem:

        observações
            ↓
        filtragem
            ↓
        preparação
            ↓
        ajuste inicial
            ↓
        rejeição de outliers
            ↓
        ajuste final
    """

    prepared = prepare_lane_points(
        points,
        min_confidence=min_confidence,
        max_points=max_points,
    )

    minimum = (
        LANE_MODEL.minimum_points
        if min_points is None
        else int(min_points)
    )

    if minimum < 4:
        raise ValueError(
            "min_points deve ser >= 4."
        )

    if len(prepared) < minimum:
        return None

    polynomial = fit_polynomial(
        prepared,
        degree=3,
        min_points=minimum,
        max_fit_error=max_fit_error,
    )

    if not polynomial.valid:
        return None

    iterations = int(
        LANE_MODEL.max_outlier_iterations
    )

    cleaned = list(prepared)

    for _ in range(iterations):

        filtered = remove_polynomial_outliers(
            cleaned,
            polynomial,
        )

        if len(filtered) < minimum:
            break

        if len(filtered) == len(cleaned):
            break

        cleaned = filtered

        updated = fit_polynomial(
            cleaned,
            degree=3,
            min_points=minimum,
            max_fit_error=max_fit_error,
        )

        if not updated.valid:
            break

        polynomial = updated

    final = fit_polynomial(
        cleaned,
        degree=3,
        min_points=minimum,
        max_fit_error=max_fit_error,
    )

    if not final.valid:
        return None

    return final


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
    Constrói o modelo matemático de uma lane.

    A identidade lane_id é fornecida externamente.
    """

    prepared = prepare_lane_points(
        points,
        min_confidence=min_confidence,
        max_points=max_points,
    )

    observation_confidence = (
        lane_confidence_score(prepared)
    )

    line = LaneLine(
        lane_id=lane_id,
        points=prepared,
        confidence=observation_confidence,
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

    model_confidence = _clip01(
        0.45 * observation_confidence
        + 0.55 * polynomial.confidence
    )

    line.confidence = model_confidence

    return LaneModel(
        lane_id=lane_id,
        line=line,
        polynomial=polynomial,
        projection=None,
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
    Recalcula o modelo mantendo identidade temporal básica.

    Tracking propriamente dito continua pertencendo ao LaneTracker.
    """

    if model is None:
        raise ValueError(
            "model não pode ser None."
        )

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

    if (
        updated.line is not None
        and model.line is not None
    ):

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
    """Valida estruturalmente um LaneModel."""

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

    coefficients = (
        model.polynomial.a,
        model.polynomial.b,
        model.polynomial.c,
        model.polynomial.d,
    )

    if not all(
        _is_finite(value)
        for value in coefficients
    ):
        return False

    return True


# =============================================================================
# PUBLIC API
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
    "build_lane_model",
    "update_lane_model",
    "validate_lane_model",
]