"""
vision/lane_model.py

Modelagem matemática oficial das linhas de faixa.

Responsabilidade
----------------
Transformar observações LanePoint em um modelo polinomial cúbico:

    LanePoint
        ↓
    validação
        ↓
    filtragem
        ↓
    preparação
        ↓
    normalização numérica
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

Modelo oficial
--------------

    x(y) = a*y³ + b*y² + c*y + d

A variável independente é Y e a variável dependente é X.

O ajuste é realizado com Y normalizado para melhorar a estabilidade
numérica em resoluções elevadas.

Princípios
----------

- somente polinômio cúbico;
- configuração compatível com config.py;
- nenhum campo de configuração inexistente;
- determinístico;
- robusto contra outliers;
- somente dados finitos;
- falha segura;
- nenhuma mutação das observações de entrada;
- nenhuma extrapolação.
"""

from __future__ import annotations

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


# =============================================================================
# CONSTANTES LOCAIS
# =============================================================================

# O config.py atual não possui max_fit_error.
#
# Portanto, o limite de erro não é tratado como configuração obrigatória.
# O valor abaixo é deliberadamente local e serve somente como proteção
# geométrica do fitting. A API também permite substituí-lo explicitamente.
DEFAULT_MAX_FIT_ERROR = 25.0

POLYNOMIAL_DEGREE = 3

MIN_POLYNOMIAL_POINTS = 4

CONFIDENCE_COUNT_REFERENCE = 20.0
CONFIDENCE_SPAN_REFERENCE = 300.0

EXCELLENT_CONFIDENCE = 0.75
GOOD_CONFIDENCE = 0.55

EXCELLENT_FIT_ERROR = 12.0

MAD_SCALE = 1.4826

MIN_OUTLIER_LIMIT = 5.0


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================


def _configuration_value(
    name: str,
    default: object,
) -> object:
    """
    Obtém um valor da configuração sem assumir que o campo exista.

    Isso é importante porque config.py é a fonte oficial de configuração,
    mas versões diferentes do projeto podem possuir conjuntos diferentes
    de campos.
    """

    return getattr(
        LANE_MODEL,
        name,
        default,
    )


def _minimum_points() -> int:
    """Quantidade mínima de pontos para um modelo cúbico."""

    value = int(
        _configuration_value(
            "minimum_points",
            MIN_POLYNOMIAL_POINTS,
        )
    )

    return max(
        MIN_POLYNOMIAL_POINTS,
        value,
    )


def _minimum_y_span() -> float:
    """Extensão vertical mínima necessária."""

    value = float(
        _configuration_value(
            "minimum_y_span",
            1.0,
        )
    )

    if not math.isfinite(value):
        return 1.0

    return max(
        1e-9,
        value,
    )


def _minimum_confidence() -> float:
    """Confiança mínima global configurada."""

    value = float(
        _configuration_value(
            "minimum_confidence",
            0.0,
        )
    )

    if not math.isfinite(value):
        return 0.0

    return _clip01(value)


def _outlier_threshold() -> float:
    """Threshold MAD configurado."""

    value = float(
        _configuration_value(
            "outlier_threshold",
            3.5,
        )
    )

    if not math.isfinite(value):
        return 3.5

    return max(
        0.1,
        value,
    )


def _max_outlier_iterations() -> int:
    """Número máximo de iterações robustas."""

    value = int(
        _configuration_value(
            "max_outlier_iterations",
            3,
        )
    )

    return max(
        0,
        value,
    )


def _validate_configuration() -> None:
    """Valida somente campos realmente existentes/necessários."""

    degree = int(
        _configuration_value(
            "polynomial_degree",
            POLYNOMIAL_DEGREE,
        )
    )

    if degree != POLYNOMIAL_DEGREE:
        raise ValueError(
            "LANE_MODEL.polynomial_degree deve ser exatamente 3."
        )

    if _minimum_points() < MIN_POLYNOMIAL_POINTS:
        raise ValueError(
            "LANE_MODEL.minimum_points deve ser >= 4."
        )

    if _minimum_y_span() <= 0.0:
        raise ValueError(
            "LANE_MODEL.minimum_y_span deve ser > 0."
        )

    if _outlier_threshold() <= 0.0:
        raise ValueError(
            "LANE_MODEL.outlier_threshold deve ser > 0."
        )

    confidence = _minimum_confidence()

    if not 0.0 <= confidence <= 1.0:
        raise ValueError(
            "LANE_MODEL.minimum_confidence deve estar entre 0 e 1."
        )


_validate_configuration()


# =============================================================================
# UTILITÁRIOS NUMÉRICOS
# =============================================================================


def _clip01(value: float) -> float:
    """Limita valor ao intervalo [0, 1]."""

    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0

    if not math.isfinite(value):
        return 0.0

    return float(
        np.clip(
            value,
            0.0,
            1.0,
        )
    )


def _is_finite(value: object) -> bool:
    """Verifica se um escalar é finito."""

    try:
        return math.isfinite(
            float(value)
        )
    except (TypeError, ValueError):
        return False


def _safe_mean(
    values: Sequence[float],
) -> float:
    """Média segura."""

    if not values:
        return 0.0

    array = np.asarray(
        values,
        dtype=np.float64,
    )

    array = array[
        np.isfinite(array)
    ]

    if array.size == 0:
        return 0.0

    return float(
        np.mean(array)
    )


def _safe_median(
    values: Sequence[float],
) -> float:
    """Mediana segura."""

    if not values:
        return 0.0

    array = np.asarray(
        values,
        dtype=np.float64,
    )

    array = array[
        np.isfinite(array)
    ]

    if array.size == 0:
        return 0.0

    return float(
        np.median(array)
    )


# =============================================================================
# VALIDAÇÃO DE PONTOS
# =============================================================================


def _point_is_finite(
    point: LanePoint,
) -> bool:
    """Valida X, Y e confiança."""

    try:
        return (
            math.isfinite(float(point.x))
            and math.isfinite(float(point.y))
            and math.isfinite(
                float(point.confidence)
            )
        )
    except (
        AttributeError,
        TypeError,
        ValueError,
    ):
        return False


def filter_lane_points(
    points: Iterable[LanePoint],
    min_confidence: Optional[float] = None,
) -> List[LanePoint]:
    """
    Filtra observações inválidas.

    Remove:

    - None;
    - objetos que não sejam LanePoint;
    - pontos inválidos;
    - NaN;
    - infinito;
    - confiança abaixo do limite.
    """

    if points is None:
        return []

    threshold = (
        _minimum_confidence()
        if min_confidence is None
        else _clip01(
            min_confidence
        )
    )

    result: List[LanePoint] = []

    try:
        iterator = iter(points)
    except TypeError:
        return []

    for point in iterator:

        if not isinstance(
            point,
            LanePoint,
        ):
            continue

        if not _point_is_finite(
            point
        ):
            continue

        try:
            if not point.is_valid():
                continue
        except (
            AttributeError,
            TypeError,
        ):
            continue

        if (
            float(point.confidence)
            < threshold
        ):
            continue

        result.append(point)

    return result


# =============================================================================
# PREPARAÇÃO
# =============================================================================


def sort_lane_points(
    points: Iterable[LanePoint],
) -> List[LanePoint]:
    """Ordena pontos por Y crescente."""

    return sorted(
        list(points),
        key=lambda point: float(
            point.y
        ),
    )


def remove_duplicate_y(
    points: Iterable[LanePoint],
) -> List[LanePoint]:
    """
    Remove múltiplas observações no mesmo Y.

    Mantém a observação de maior confiança.
    """

    best: dict[float, LanePoint] = {}

    for point in points:

        y = float(point.y)

        previous = best.get(y)

        if (
            previous is None
            or float(point.confidence)
            > float(previous.confidence)
        ):
            best[y] = point

    return sorted(
        best.values(),
        key=lambda point: float(
            point.y
        ),
    )


def limit_point_count(
    points: Sequence[LanePoint],
    max_points: Optional[int] = None,
) -> List[LanePoint]:
    """
    Reduz a quantidade de pontos preservando distribuição vertical.

    Primeiro e último ponto permanecem representados.
    """

    if max_points is None:
        return list(points)

    maximum = max(
        _minimum_points(),
        int(max_points),
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
    """
    Pipeline determinístico:

        validação
            ↓
        filtragem
            ↓
        ordenação
            ↓
        remoção de Y duplicado
            ↓
        limitação opcional
    """

    filtered = filter_lane_points(
        points,
        min_confidence=min_confidence,
    )

    ordered = sort_lane_points(
        filtered
    )

    unique = remove_duplicate_y(
        ordered
    )

    return limit_point_count(
        unique,
        max_points=max_points,
    )


# =============================================================================
# ESTATÍSTICAS
# =============================================================================


def lane_y_span(
    points: Sequence[LanePoint],
) -> float:
    """Extensão vertical da observação."""

    if len(points) < 2:
        return 0.0

    try:
        ys = np.asarray(
            [point.y for point in points],
            dtype=np.float64,
        )
    except (
        TypeError,
        ValueError,
    ):
        return 0.0

    if not np.all(
        np.isfinite(ys)
    ):
        return 0.0

    return float(
        np.max(ys)
        - np.min(ys)
    )


def lane_x_span(
    points: Sequence[LanePoint],
) -> float:
    """Extensão horizontal da observação."""

    if len(points) < 2:
        return 0.0

    try:
        xs = np.asarray(
            [point.x for point in points],
            dtype=np.float64,
        )
    except (
        TypeError,
        ValueError,
    ):
        return 0.0

    if not np.all(
        np.isfinite(xs)
    ):
        return 0.0

    return float(
        np.max(xs)
        - np.min(xs)
    )


def lane_mean_confidence(
    points: Sequence[LanePoint],
) -> float:
    """Confiança média dos pontos válidos."""

    values: List[float] = []

    for point in points:

        try:
            if not point.is_valid():
                continue

            confidence = float(
                point.confidence
            )

        except (
            AttributeError,
            TypeError,
            ValueError,
        ):
            continue

        if math.isfinite(
            confidence
        ):
            values.append(
                confidence
            )

    return _clip01(
        _safe_mean(values)
    )


def lane_confidence_score(
    points: Sequence[LanePoint],
) -> float:
    """
    Confiança estrutural da observação.

    55% confiança dos pontos
    25% quantidade
    20% extensão vertical
    """

    if not points:
        return 0.0

    confidence = lane_mean_confidence(
        points
    )

    count_score = _clip01(
        len(points)
        / CONFIDENCE_COUNT_REFERENCE
    )

    span_score = _clip01(
        lane_y_span(points)
        / CONFIDENCE_SPAN_REFERENCE
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
    """Classifica a qualidade geométrica da lane."""

    if not points:
        return LaneQuality.NONE

    count = len(points)

    confidence = lane_mean_confidence(
        points
    )

    span = lane_y_span(
        points
    )

    minimum_points = _minimum_points()

    if (
        count < minimum_points
        or span < _minimum_y_span()
    ):
        return LaneQuality.POOR

    if (
        confidence < 0.35
    ):
        return LaneQuality.POOR

    if (
        polynomial is not None
        and polynomial.valid
    ):

        if (
            polynomial.confidence
            >= EXCELLENT_CONFIDENCE
            and polynomial.fit_error
            <= EXCELLENT_FIT_ERROR
        ):
            return LaneQuality.EXCELLENT

        if (
            polynomial.confidence
            >= GOOD_CONFIDENCE
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
# NORMALIZAÇÃO
# =============================================================================


def _normalize_y(
    y: np.ndarray,
) -> Tuple[
    np.ndarray,
    float,
    float,
]:
    """
    Normaliza Y:

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
) -> Tuple[
    float,
    float,
    float,
    float,
]:
    """
    Converte:

        x = A*z³ + B*z² + C*z + D

    para:

        x = a*y³ + b*y² + c*y + d
    """

    if len(coefficients) != 4:
        raise ValueError(
            "São necessários exatamente 4 coeficientes."
        )

    A = float(
        coefficients[3]
    )

    B = float(
        coefficients[2]
    )

    C = float(
        coefficients[1]
    )

    D = float(
        coefficients[0]
    )

    scale2 = scale * scale
    scale3 = scale2 * scale

    a = (
        A / scale3
    )

    b = (
        B / scale2
        - 3.0 * A * center / scale3
    )

    c = (
        C / scale
        - 2.0 * B * center / scale2
        + 3.0
        * A
        * center
        * center
        / scale3
    )

    d = (
        D
        - C * center / scale
        + B * center * center / scale2
        - A
        * center
        * center
        * center
        / scale3
    )

    return (
        float(a),
        float(b),
        float(c),
        float(d),
    )


# =============================================================================
# POLYNOMIAL
# =============================================================================


def _invalid_polynomial(
    sample_count: int,
    y_min: float = 0.0,
    y_max: float = 0.0,
) -> LanePolynomial:
    """Cria LanePolynomial inválido."""

    return LanePolynomial(
        valid=False,
        sample_count=int(
            sample_count
        ),
        y_min=float(
            y_min
        ),
        y_max=float(
            y_max
        ),
    )


def _calculate_fit_confidence(
    points: Sequence[LanePoint],
    fit_error: float,
) -> float:
    """Calcula confiança do ajuste."""

    if not points:
        return 0.0

    observation_score = (
        lane_mean_confidence(points)
    )

    count_score = _clip01(
        len(points)
        / CONFIDENCE_COUNT_REFERENCE
    )

    span_score = _clip01(
        lane_y_span(points)
        / CONFIDENCE_SPAN_REFERENCE
    )

    if not math.isfinite(
        fit_error
    ):
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
    degree: int = POLYNOMIAL_DEGREE,
    min_points: Optional[int] = None,
    max_fit_error: Optional[float] = None,
) -> LanePolynomial:
    """
    Ajusta o polinômio cúbico oficial:

        x(y) = a*y³ + b*y² + c*y + d

    O ajuste é feito em Y normalizado.

    `max_fit_error` existe como parâmetro explícito da API,
    mas NÃO depende de um campo inexistente em config.py.
    """

    if degree != POLYNOMIAL_DEGREE:
        raise ValueError(
            "lane_model suporta exclusivamente "
            "polinômio cúbico."
        )

    minimum = (
        _minimum_points()
        if min_points is None
        else max(
            MIN_POLYNOMIAL_POINTS,
            int(min_points),
        )
    )

    error_limit = (
        DEFAULT_MAX_FIT_ERROR
        if max_fit_error is None
        else float(
            max_fit_error
        )
    )

    if (
        not math.isfinite(
            error_limit
        )
        or error_limit <= 0.0
    ):
        raise ValueError(
            "max_fit_error deve ser finito e > 0."
        )

    if points is None:
        return _invalid_polynomial(
            0
        )

    if len(points) < minimum:
        return _invalid_polynomial(
            len(points)
        )

    xs = np.asarray(
        [
            float(point.x)
            for point in points
        ],
        dtype=np.float64,
    )

    ys = np.asarray(
        [
            float(point.y)
            for point in points
        ],
        dtype=np.float64,
    )

    if not (
        np.all(
            np.isfinite(xs)
        )
        and np.all(
            np.isfinite(ys)
        )
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

    if (
        y_max - y_min
        < _minimum_y_span()
    ):
        return _invalid_polynomial(
            len(points),
            y_min,
            y_max,
        )

    normalized_y, center, scale = (
        _normalize_y(ys)
    )

    try:

        coefficients = (
            np.polynomial.polynomial.polyfit(
                normalized_y,
                xs,
                deg=POLYNOMIAL_DEGREE,
            )
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

        a, b, c, d = (
            _denormalize_coefficients(
                coefficients,
                center,
                scale,
            )
        )

    except (
        ValueError,
        ZeroDivisionError,
        FloatingPointError,
    ):
        return _invalid_polynomial(
            len(points),
            y_min,
            y_max,
        )

    if not all(
        math.isfinite(value)
        for value in (
            a,
            b,
            c,
            d,
        )
    ):
        return _invalid_polynomial(
            len(points),
            y_min,
            y_max,
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

    if not np.all(
        np.isfinite(residuals)
    ):
        return _invalid_polynomial(
            len(points),
            y_min,
            y_max,
        )

    fit_error = float(
        np.median(residuals)
    )

    confidence = (
        _calculate_fit_confidence(
            points,
            fit_error,
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

    polynomial.fit_error = (
        fit_error
    )

    polynomial.confidence = (
        confidence
    )

    if (
        not math.isfinite(
            fit_error
        )
        or fit_error > error_limit
        or confidence
        < _minimum_confidence()
    ):
        polynomial.valid = False

    return polynomial


# =============================================================================
# RESÍDUOS
# =============================================================================


def _polynomial_residuals(
    points: Sequence[LanePoint],
    polynomial: LanePolynomial,
) -> np.ndarray:
    """Calcula erro absoluto de cada observação."""

    if (
        not points
        or polynomial is None
        or not polynomial.valid
    ):
        return np.empty(
            0,
            dtype=np.float64,
        )

    xs = np.asarray(
        [
            float(point.x)
            for point in points
        ],
        dtype=np.float64,
    )

    ys = np.asarray(
        [
            float(point.y)
            for point in points
        ],
        dtype=np.float64,
    )

    try:
        predicted = np.asarray(
            [
                polynomial.evaluate(
                    float(y)
                )
                for y in ys
            ],
            dtype=np.float64,
        )
    except (
        AttributeError,
        TypeError,
        ValueError,
        OverflowError,
    ):
        return np.empty(
            0,
            dtype=np.float64,
        )

    residuals = np.abs(
        xs - predicted
    )

    return residuals


# =============================================================================
# OUTLIERS
# =============================================================================


def remove_polynomial_outliers(
    points: Sequence[LanePoint],
    polynomial: LanePolynomial,
    threshold: Optional[float] = None,
) -> List[LanePoint]:
    """
    Remove outliers utilizando MAD.

    O limite é:

        median + threshold * 1.4826 * MAD

    com um piso absoluto de segurança.
    """

    if (
        polynomial is None
        or not polynomial.valid
        or len(points) < MIN_POLYNOMIAL_POINTS
    ):
        return list(points)

    threshold_value = (
        _outlier_threshold()
        if threshold is None
        else float(threshold)
    )

    if (
        not math.isfinite(
            threshold_value
        )
        or threshold_value <= 0.0
    ):
        raise ValueError(
            "threshold deve ser finito e > 0."
        )

    residuals = (
        _polynomial_residuals(
            points,
            polynomial,
        )
    )

    if residuals.size != len(
        points
    ):
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
            MIN_OUTLIER_LIMIT,
            median
            * threshold_value,
        )

    else:

        robust_sigma = (
            MAD_SCALE * mad
        )

        limit = max(
            MIN_OUTLIER_LIMIT,
            median
            + threshold_value
            * robust_sigma,
        )

    return [
        point
        for point, residual
        in zip(
            points,
            residuals,
        )
        if float(residual)
        <= limit
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
    Pipeline robusto completo.

        observações
            ↓
        filtragem
            ↓
        preparação
            ↓
        fitting inicial
            ↓
        MAD
            ↓
        refitting
            ↓
        LanePolynomial
    """

    prepared = prepare_lane_points(
        points,
        min_confidence=min_confidence,
        max_points=max_points,
    )

    minimum = (
        _minimum_points()
        if min_points is None
        else max(
            MIN_POLYNOMIAL_POINTS,
            int(min_points),
        )
    )

    if len(prepared) < minimum:
        return None

    polynomial = fit_polynomial(
        prepared,
        degree=POLYNOMIAL_DEGREE,
        min_points=minimum,
        max_fit_error=max_fit_error,
    )

    if not polynomial.valid:
        return None

    cleaned = list(
        prepared
    )

    iterations = (
        _max_outlier_iterations()
    )

    for _ in range(iterations):

        filtered = (
            remove_polynomial_outliers(
                cleaned,
                polynomial,
            )
        )

        if len(filtered) < minimum:
            break

        if len(filtered) == len(
            cleaned
        ):
            break

        cleaned = filtered

        updated = fit_polynomial(
            cleaned,
            degree=POLYNOMIAL_DEGREE,
            min_points=minimum,
            max_fit_error=max_fit_error,
        )

        if not updated.valid:
            break

        polynomial = updated

    final = fit_polynomial(
        cleaned,
        degree=POLYNOMIAL_DEGREE,
        min_points=minimum,
        max_fit_error=max_fit_error,
    )

    if not final.valid:
        return None

    return final


# =============================================================================
# LANE MODEL
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
    Constrói LaneModel a partir de uma observação.

    lane_id é responsabilidade do tracker/association layer.
    """

    prepared = prepare_lane_points(
        points,
        min_confidence=min_confidence,
        max_points=max_points,
    )

    observation_confidence = (
        lane_confidence_score(
            prepared
        )
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

    line.quality = (
        classify_lane_quality(
            prepared,
            polynomial,
        )
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

    line.confidence = (
        model_confidence
    )

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
# UPDATE
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

    Tracking completo permanece fora deste módulo.
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

    updated.tracked = (
        model.tracked
    )

    updated.stable = (
        model.stable
    )

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
# VALIDAÇÃO
# =============================================================================


def validate_lane_model(
    model: Optional[LaneModel],
) -> bool:
    """
    Validação estrutural e numérica completa.
    """

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

    if (
        model.line.point_count()
        < MIN_POLYNOMIAL_POINTS
    ):
        return False

    if not model.is_finite():
        return False

    polynomial = (
        model.polynomial
    )

    scalar_values = (
        polynomial.a,
        polynomial.b,
        polynomial.c,
        polynomial.d,
        polynomial.confidence,
        polynomial.fit_error,
        polynomial.y_min,
        polynomial.y_max,
    )

    if not all(
        _is_finite(value)
        for value in scalar_values
    ):
        return False

    if (
        polynomial.y_max
        < polynomial.y_min
    ):
        return False

    if (
        polynomial.sample_count
        < MIN_POLYNOMIAL_POINTS
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
    "build_lane_model",
    "update_lane_model",
    "validate_lane_model",
]