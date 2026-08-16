"""
vision/lane_model.py

Modelagem matemática oficial das linhas de faixa.

Responsabilidade
----------------
Transforma observações LanePoint em um modelo cúbico estável:

    LanePoint
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
    - associação semântica;
    - projeção/extrapolação;
    - decisão ADAS;
    - controle do veículo.

Modelo oficial:

    x(y) = a*y³ + b*y² + c*y + d

Princípios:
    - somente polinômio cúbico;
    - configuração centralizada em config.py;
    - estabilidade numérica;
    - rejeição robusta de outliers;
    - comportamento determinístico;
    - somente dados finitos;
    - falha segura.
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
# NUMERIC UTILITIES
# =============================================================================


def _clip01(value: float) -> float:
    """Limita um valor ao intervalo [0, 1]."""

    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0

    if not math.isfinite(value):
        return 0.0

    return float(np.clip(value, 0.0, 1.0))


def _is_finite(value: object) -> bool:
    """Retorna True somente para valores numéricos finitos."""

    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _safe_mean(values: Sequence[float]) -> float:
    """Calcula média ignorando valores não finitos."""

    if not values:
        return 0.0

    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]

    if array.size == 0:
        return 0.0

    return float(np.mean(array))


def _safe_median(values: Sequence[float]) -> float:
    """Calcula mediana ignorando valores não finitos."""

    if not values:
        return 0.0

    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]

    if array.size == 0:
        return 0.0

    return float(np.median(array))


# =============================================================================
# CONFIGURATION
# =============================================================================


def _config_value(name: str, default: object = None) -> object:
    """
    Obtém uma configuração de forma segura.

    A configuração oficial continua sendo config.py.

    O acesso via getattr evita que uma propriedade opcional
    inexistente derrube o import do módulo.
    """

    return getattr(LANE_MODEL, name, default)


def _minimum_confidence() -> float:
    value = _config_value("minimum_confidence", 0.0)
    return _clip01(float(value))


def _max_fit_error() -> float:
    """
    Obtém o erro máximo permitido.

    Compatibilidade:
    - max_fit_error é preferencial;
    - fit_error_threshold é aceito caso exista;
    - fallback conservador = 25 px.
    """

    value = _config_value("max_fit_error", None)

    if value is None:
        value = _config_value("fit_error_threshold", 25.0)

    try:
        value = float(value)
    except (TypeError, ValueError):
        value = 25.0

    if not math.isfinite(value) or value <= 0.0:
        value = 25.0

    return value


def _validate_configuration() -> None:
    """Valida os invariantes utilizados pelo modelo."""

    degree = int(_config_value("polynomial_degree", 3))

    if degree != 3:
        raise ValueError(
            "LANE_MODEL.polynomial_degree deve ser exatamente 3."
        )

    minimum_points = int(
        _config_value("minimum_points", 4)
    )

    if minimum_points < 4:
        raise ValueError(
            "LANE_MODEL.minimum_points deve ser >= 4."
        )

    minimum_y_span = float(
        _config_value("minimum_y_span", 1.0)
    )

    if (
        not math.isfinite(minimum_y_span)
        or minimum_y_span <= 0.0
    ):
        raise ValueError(
            "LANE_MODEL.minimum_y_span deve ser > 0."
        )

    max_iterations = int(
        _config_value("max_outlier_iterations", 0)
    )

    if max_iterations < 0:
        raise ValueError(
            "LANE_MODEL.max_outlier_iterations não pode ser negativo."
        )

    outlier_threshold = float(
        _config_value("outlier_threshold", 2.5)
    )

    if (
        not math.isfinite(outlier_threshold)
        or outlier_threshold <= 0.0
    ):
        raise ValueError(
            "LANE_MODEL.outlier_threshold deve ser > 0."
        )

    confidence = _minimum_confidence()

    if not 0.0 <= confidence <= 1.0:
        raise ValueError(
            "LANE_MODEL.minimum_confidence deve estar entre 0 e 1."
        )

    if _max_fit_error() <= 0.0:
        raise ValueError(
            "O erro máximo do ajuste deve ser > 0."
        )


# IMPORTANTE:
# A validação ocorre somente depois de todas as funções auxiliares
# utilizadas por ela terem sido declaradas.
_validate_configuration()


# =============================================================================
# POINT FILTERING
# =============================================================================


def filter_lane_points(
    points: Iterable[LanePoint],
    min_confidence: Optional[float] = None,
) -> List[LanePoint]:
    """
    Remove pontos:

    - inexistentes;
    - inválidos;
    - não finitos;
    - abaixo da confiança mínima.
    """

    threshold = (
        _minimum_confidence()
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

        try:
            valid = bool(point.is_valid())
        except (AttributeError, TypeError, ValueError):
            valid = False

        if not valid:
            continue

        if confidence < threshold:
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
    Remove pontos com Y duplicado.

    Mantém o ponto de maior confiança.
    """

    best_by_y: dict[float, LanePoint] = {}

    for point in points:

        y = float(point.y)
        current = best_by_y.get(y)

        if (
            current is None
            or float(point.confidence)
            > float(current.confidence)
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
    Limita a quantidade de pontos preservando a distribuição vertical.

    O primeiro e o último ponto são preservados.
    """

    if max_points is None:
        return list(points)

    maximum = int(max_points)

    minimum = int(
        _config_value("minimum_points", 4)
    )

    maximum = max(
        minimum,
        maximum,
    )

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
    """Executa a preparação determinística dos pontos."""

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
    """Retorna a extensão vertical dos pontos."""

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
    """Retorna a extensão horizontal dos pontos."""

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
    """Calcula a confiança média dos pontos válidos."""

    values: List[float] = []

    for point in points:

        try:
            if not point.is_valid():
                continue

            confidence = float(point.confidence)

        except (AttributeError, TypeError, ValueError):
            continue

        if math.isfinite(confidence):
            values.append(confidence)

    return _clip01(
        _safe_mean(values)
    )


def lane_confidence_score(
    points: Sequence[LanePoint],
) -> float:
    """
    Confiança estrutural da observação.

    55% confiança média
    25% quantidade
    20% extensão vertical
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
    """Classifica a qualidade geométrica da lane."""

    if not points:
        return LaneQuality.NONE

    minimum_points = int(
        _config_value("minimum_points", 4)
    )

    minimum_y_span = float(
        _config_value("minimum_y_span", 1.0)
    )

    count = len(points)
    confidence = lane_mean_confidence(points)
    span = lane_y_span(points)

    if (
        count < minimum_points
        or span < minimum_y_span
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
    Normaliza Y:

        z = (y - center) / scale
    """

    center = float(np.mean(y))

    scale = float(
        np.max(
            np.abs(y - center)
        )
    )

    if (
        not math.isfinite(scale)
        or scale < 1e-12
    ):
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

    # np.polynomial.polynomial.polyfit retorna:
    #
    # [D, C, B, A]

    D = float(coefficients[0])
    C = float(coefficients[1])
    B = float(coefficients[2])
    A = float(coefficients[3])

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
    """Cria um LanePolynomial inválido."""

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
    """Calcula a confiança do ajuste cúbico."""

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
    Ajusta o polinômio cúbico oficial:

        x(y) = a*y³ + b*y² + c*y + d
    """

    configured_degree = int(
        _config_value("polynomial_degree", 3)
    )

    if degree != configured_degree or degree != 3:
        raise ValueError(
            "lane_model suporta exclusivamente "
            "polinômio cúbico."
        )

    minimum = (
        int(_config_value("minimum_points", 4))
        if min_points is None
        else int(min_points)
    )

    if minimum < 4:
        raise ValueError(
            "min_points deve ser >= 4."
        )

    error_limit = (
        _max_fit_error()
        if max_fit_error is None
        else float(max_fit_error)
    )

    if (
        not math.isfinite(error_limit)
        or error_limit <= 0.0
    ):
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

    y_min = float(np.min(ys))
    y_max = float(np.max(ys))

    minimum_y_span = float(
        _config_value("minimum_y_span", 1.0)
    )

    if (
        y_max - y_min
        < minimum_y_span
    ):
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

    try:
        a, b, c, d = _denormalize_coefficients(
            coefficients,
            center,
            scale,
        )
    except (
        ArithmeticError,
        ValueError,
        OverflowError,
        ZeroDivisionError,
    ):
        return _invalid_polynomial(
            len(points),
            y_min,
            y_max,
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

    minimum_confidence = _minimum_confidence()

    if (
        not math.isfinite(fit_error)
        or fit_error > error_limit
        or confidence < minimum_confidence
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
    """Calcula o resíduo absoluto de cada ponto."""

    if (
        not points
        or not polynomial.valid
    ):
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

    if not (
        np.all(np.isfinite(xs))
        and np.all(np.isfinite(ys))
    ):
        return np.empty(
            0,
            dtype=np.float64,
        )

    predicted = np.asarray(
        [
            polynomial.evaluate(float(y))
            for y in ys
        ],
        dtype=np.float64,
    )

    if not np.all(
        np.isfinite(predicted)
    ):
        return np.empty(
            0,
            dtype=np.float64,
        )

    return np.abs(
        xs - predicted
    )


def remove_polynomial_outliers(
    points: Sequence[LanePoint],
    polynomial: LanePolynomial,
    threshold: Optional[float] = None,
) -> List[LanePoint]:
    """
    Remove outliers usando MAD.

    O limite é:

        median + threshold * 1.4826 * MAD

    com piso de 5 px.
    """

    if (
        not polynomial.valid
        or len(points) < 4
    ):
        return list(points)

    threshold_value = (
        float(
            _config_value(
                "outlier_threshold",
                2.5,
            )
        )
        if threshold is None
        else float(threshold)
    )

    if (
        not math.isfinite(threshold_value)
        or threshold_value <= 0.0
    ):
        raise ValueError(
            "threshold deve ser > 0."
        )

    residuals = _polynomial_residuals(
        points,
        polynomial,
    )

    if (
        residuals.size != len(points)
        or not np.all(np.isfinite(residuals))
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
    Pipeline robusto oficial:

        filtragem
            ↓
        preparação
            ↓
        ajuste inicial
            ↓
        MAD
            ↓
        novo ajuste
            ↓
        validação final
    """

    prepared = prepare_lane_points(
        points,
        min_confidence=min_confidence,
        max_points=max_points,
    )

    minimum = (
        int(_config_value("minimum_points", 4))
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
        _config_value(
            "max_outlier_iterations",
            0,
        )
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
    Constrói um LaneModel.

    lane_id é fornecido pelo sistema responsável
    pela identidade da lane.
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
            >= _minimum_confidence()
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
    Recalcula o modelo preservando identidade temporal básica.

    O tracking completo permanece fora deste módulo.
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
    """Validação estrutural completa de um LaneModel."""

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

    try:
        if not model.is_finite():
            return False
    except (AttributeError, TypeError, ValueError):
        return False

    polynomial = model.polynomial

    if not _is_finite(polynomial.confidence):
        return False

    if not _is_finite(polynomial.fit_error):
        return False

    coefficients = (
        polynomial.a,
        polynomial.b,
        polynomial.c,
        polynomial.d,
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