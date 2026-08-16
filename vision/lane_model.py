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
- nenhuma configuração paralela local;
- polinômio x(y) cúbico;
- normalização numérica antes do fitting;
- rejeição robusta de outliers;
- rejeição de dados não finitos;
- comportamento determinístico;
- falha segura;
- nenhuma mutação de configuração global.
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
# VALIDAÇÃO DE CONFIGURAÇÃO
# =============================================================================


def _validate_configuration() -> None:
    """
    Valida invariantes fundamentais da modelagem.

    O sistema utiliza exclusivamente polinômio cúbico.
    """

    if LANE_MODEL.polynomial_degree != 3:
        raise ValueError(
            "LANE_MODEL.polynomial_degree deve ser 3. "
            "LanePolynomial utiliza x(y) = a*y³ + b*y² + c*y + d."
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
# UTILITÁRIOS NUMÉRICOS
# =============================================================================


def _clip01(value: float) -> float:
    return float(
        np.clip(
            float(value),
            0.0,
            1.0,
        )
    )


def _is_finite(value: float) -> bool:
    return math.isfinite(float(value))


def _safe_mean(
    values: Sequence[float],
) -> float:

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
# PREPARAÇÃO
# =============================================================================


def filter_lane_points(
    points: Iterable[LanePoint],
) -> List[LanePoint]:
    """
    Filtra pontos inválidos, não finitos e abaixo
    da confiança configurada.
    """

    result: List[LanePoint] = []

    minimum_confidence = (
        LANE_MODEL.minimum_confidence
    )

    for point in points:

        if not isinstance(
            point,
            LanePoint,
        ):
            continue

        if not point.is_valid():
            continue

        if point.confidence < minimum_confidence:
            continue

        result.append(point)

    return result


def sort_lane_points(
    points: Iterable[LanePoint],
) -> List[LanePoint]:
    """
    Ordena por coordenada vertical.
    """

    return sorted(
        points,
        key=lambda point: float(point.y),
    )


def remove_duplicate_y(
    points: Iterable[LanePoint],
) -> List[LanePoint]:
    """
    Remove múltiplos pontos na mesma coordenada Y.

    Quando há colisão, preserva o ponto de maior
    confiança.
    """

    best_by_y: dict[float, LanePoint] = {}

    for point in points:

        y = float(point.y)

        current = best_by_y.get(y)

        if (
            current is None
            or point.confidence
            > current.confidence
        ):
            best_by_y[y] = point

    return sorted(
        best_by_y.values(),
        key=lambda point: point.y,
    )


def limit_point_count(
    points: Sequence[LanePoint],
) -> List[LanePoint]:
    """
    Limita a quantidade de pontos sem concentrá-los
    em uma determinada região vertical.

    A distribuição espacial é preservada por amostragem
    uniforme ao longo do conjunto ordenado.
    """

    maximum = max(
        LANE_MODEL.minimum_points,
        LANE_PROJECTION.samples,
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
) -> List[LanePoint]:
    """
    Pipeline determinístico de preparação.
    """

    filtered = filter_lane_points(
        points
    )

    ordered = sort_lane_points(
        filtered
    )

    unique = remove_duplicate_y(
        ordered
    )

    return limit_point_count(
        unique
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

        confiança dos pontos
        quantidade de observações
        extensão vertical
    """

    if not points:
        return 0.0

    confidence = (
        lane_mean_confidence(points)
    )

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
# QUALIDADE
# =============================================================================


def classify_lane_quality(
    points: Sequence[LanePoint],
    polynomial: Optional[LanePolynomial] = None,
) -> LaneQuality:

    if not points:
        return LaneQuality.NONE

    count = len(points)

    confidence = (
        lane_mean_confidence(points)
    )

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
# NORMALIZAÇÃO NUMÉRICA
# =============================================================================


def _normalize_y(
    y: np.ndarray,
) -> Tuple[
    np.ndarray,
    float,
    float,
]:
    """
    Normaliza Y para evitar problemas de condicionamento
    numérico no fitting cúbico.
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

    onde:

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
    """
    Calcula confiança estrutural do ajuste.

    Não representa probabilidade estatística.
    """

    if not points:
        return 0.0

    observation_score = (
        lane_mean_confidence(points)
    )

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


# =============================================================================
# FIT POLINOMIAL
# =============================================================================


def fit_polynomial(
    points: Sequence[LanePoint],
) -> LanePolynomial:
    """
    Ajusta o modelo oficial:

        x(y) = a*y³ + b*y² + c*y + d

    O grau NÃO pode ser alterado por chamada.
    """

    minimum_points = (
        LANE_MODEL.minimum_points
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

    y_min = float(
        np.min(ys)
    )

    y_max = float(
        np.max(ys)
    )

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

    normalized_y, center, scale = (
        _normalize_y(ys)
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

    # np.polynomial retorna:
    #
    # D + C*z + B*z² + A*z³
    #
    # portanto:
    A = coefficients[3]
    B = coefficients[2]
    C = coefficients[1]
    D = coefficients[0]

    a, b, c, d = _denormalize_coefficients(
        (A, B, C, D),
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

    if not np.all(
        np.isfinite(predicted)
    ):
        return LanePolynomial(
            valid=False,
            sample_count=len(points),
            y_min=y_min,
            y_max=y_max,
        )

    residuals = np.abs(
        xs - predicted
    )

    fit_error = _safe_median(
        residuals.tolist()
    )

    confidence = (
        _calculate_fit_confidence(
            points,
            fit_error,
        )
    )

    polynomial.fit_error = fit_error
    polynomial.confidence = confidence

    if (
        not _is_finite(fit_error)
        or not _is_finite(confidence)
        or fit_error > 35.0
        or confidence < LANE_MODEL.minimum_confidence
    ):
        polynomial.valid = False

    return polynomial


# =============================================================================
# OUTLIERS
# =============================================================================


def remove_polynomial_outliers(
    points: Sequence[LanePoint],
    polynomial: LanePolynomial,
) -> List[LanePoint]:
    """
    Remove outliers utilizando MAD
    (Median Absolute Deviation).

    Não utiliza limiar fixo de pixel como única regra.
    """

    if (
        not polynomial.valid
        or len(points) < 5
        or not LANE_MODEL.enable_outlier_rejection
    ):
        return list(points)

    residuals = np.asarray(
        [
            abs(
                point.x
                - polynomial.evaluate(point.y)
            )
            for point in points
        ],
        dtype=np.float64,
    )

    if not np.all(
        np.isfinite(residuals)
    ):
        return list(points)

    median = float(
        np.median(residuals)
    )

    mad = float(
        np.median(
            np.abs(
                residuals - median
            )
        )
    )

    if mad < 1e-6:
        limit = max(
            5.0,
            median * LANE_MODEL.outlier_threshold,
        )
    else:
        robust_sigma = (
            1.4826 * mad
        )

        limit = (
            median
            + LANE_MODEL.outlier_threshold
            * robust_sigma
        )

    return [
        point
        for point, residual
        in zip(points, residuals)
        if residual <= limit
    ]


# =============================================================================
# FIT ROBUSTO
# =============================================================================


def fit_lane_model(
    points: Iterable[LanePoint],
) -> Optional[LanePolynomial]:
    """
    Pipeline robusto de modelagem.

        observação
            ↓
        filtragem
            ↓
        fitting inicial
            ↓
        outlier rejection
            ↓
        refitting
            ↓
        validação final
    """

    prepared = prepare_lane_points(
        points
    )

    if len(prepared) < (
        LANE_MODEL.minimum_points
    ):
        return None

    current = prepared

    polynomial: Optional[
        LanePolynomial
    ] = None

    iterations = max(
        1,
        LANE_MODEL.max_outlier_iterations,
    )

    for _ in range(iterations):

        polynomial = fit_polynomial(
            current
        )

        if polynomial is None:
            return None

        if not polynomial.valid:
            return None

        if not LANE_MODEL.enable_outlier_rejection:
            break

        cleaned = remove_polynomial_outliers(
            current,
            polynomial,
        )

        if len(cleaned) < (
            LANE_MODEL.minimum_points
        ):
            break

        if len(cleaned) == len(current):
            break

        current = cleaned

    if polynomial is None:
        return None

    # Ajuste final utilizando somente os pontos
    # efetivamente aceitos.
    final = fit_polynomial(
        current
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
    *,
    horizon_y: Optional[float] = None,
) -> LaneProjection:
    """
    Gera representação amostrada da curva.

    A projeção é matemática e nunca deve ser interpretada
    como observação direta.
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

    if (
        polynomial.confidence
        < LANE_PROJECTION.minimum_confidence
    ):
        return LaneProjection(
            polynomial=polynomial,
            quality=ProjectionQuality.LOW,
            valid=False,
        )

    if len(points) < 2:
        return LaneProjection(
            polynomial=polynomial,
            quality=ProjectionQuality.LOW,
            valid=False,
        )

    valid_points = [
        point
        for point in points
        if point.is_valid()
    ]

    if len(valid_points) < 2:
        return LaneProjection(
            polynomial=polynomial,
            quality=ProjectionQuality.LOW,
            valid=False,
        )

    observed_y_min = min(
        point.y
        for point in valid_points
    )

    observed_y_max = max(
        point.y
        for point in valid_points
    )

    if horizon_y is None:
        horizon_y = observed_y_min

    horizon_y = float(horizon_y)

    # Não extrapolar acima do horizonte observado
    # quando a configuração não permite extrapolação.
    if not LANE_PROJECTION.enable_extrapolation:
        start_y = observed_y_min
        end_y = observed_y_max

    else:
        extrapolation = min(
            LANE_PROJECTION.extrapolation_limit,
            LANE_PROJECTION.max_projection_distance,
        )

        start_y = max(
            0.0,
            observed_y_min - extrapolation,
        )

        end_y = min(
            observed_y_max + extrapolation,
            observed_y_max
            + LANE_PROJECTION.max_projection_distance,
        )

    sample_count = max(
        2,
        LANE_PROJECTION.samples,
    )

    ys = np.linspace(
        start_y,
        end_y,
        sample_count,
        dtype=np.float64,
    )

    projected_points: List[
        LanePoint
    ] = []

    for y in ys:

        x = polynomial.evaluate(
            float(y)
        )

        if not (
            np.isfinite(x)
            and np.isfinite(y)
        ):
            if LANE_PROJECTION.reject_non_finite_points:
                continue

            continue

        distance_from_observation = max(
            0.0,
            float(
                observed_y_min - y
            ),
        )

        decay_distance = max(
            1.0,
            LANE_PROJECTION.confidence_decay_distance,
        )

        confidence_decay = float(
            np.exp(
                -distance_from_observation
                / decay_distance
            )
        )

        point_confidence = _clip01(
            polynomial.confidence
            * confidence_decay
        )

        projected_points.append(
            LanePoint(
                x=float(x),
                y=float(y),
                confidence=point_confidence,
                valid=True,
            )
        )

    if len(projected_points) < (
        LANE_PROJECTION.minimum_points
    ):
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

    elif polynomial.confidence >= 0.60:
        quality = ProjectionQuality.MEDIUM

    else:
        quality = ProjectionQuality.LOW

    return LaneProjection(
        polynomial=polynomial,
        points=projected_points,
        quality=quality,
        extrapolated=(
            LANE_PROJECTION.enable_extrapolation
        ),
        valid=True,
        horizon_y=horizon_y,
    )


# =============================================================================
# BUILD LANE MODEL
# =============================================================================


def build_lane_model(
    lane_id: int,
    points: Iterable[LanePoint],
) -> LaneModel:
    """
    Constrói um LaneModel completo.

    lane_id é preservado exatamente como recebido.
    """

    prepared = prepare_lane_points(
        points
    )

    line_confidence = (
        lane_confidence_score(
            prepared
        )
    )

    line = LaneLine(
        lane_id=int(lane_id),
        points=prepared,
        confidence=line_confidence,
        quality=LaneQuality.NONE,
        detected_directly=True,
        projected=False,
        valid=(
            len(prepared)
            >= LANE_MODEL.minimum_points
        ),
    )

    polynomial = fit_lane_model(
        prepared
    )

    line.quality = classify_lane_quality(
        prepared,
        polynomial,
    )

    if polynomial is None:

        return LaneModel(
            lane_id=int(lane_id),
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

    line.confidence = (
        model_confidence
    )

    valid = (
        polynomial.valid
        and polynomial.confidence
        >= LANE_MODEL.minimum_confidence
    )

    return LaneModel(
        lane_id=int(lane_id),
        line=line,
        polynomial=polynomial,
        projection=projection,
        tracked=False,
        stable=False,
        valid=valid,
    )


# =============================================================================
# UPDATE
# =============================================================================


def update_lane_model(
    model: LaneModel,
    points: Iterable[LanePoint],
) -> LaneModel:
    """
    Recalcula um modelo preservando sua identidade.

    O estado temporal do tracker permanece fora deste módulo.
    """

    if model is None:
        raise ValueError(
            "model não pode ser None."
        )

    updated = build_lane_model(
        lane_id=model.lane_id,
        points=points,
    )

    updated.tracked = model.tracked
    updated.stable = model.stable

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
    """
    Validação estrutural final.
    """

    if model is None:
        return False

    if not model.valid:
        return False

    if model.line is None:
        return False

    if not model.line.is_valid():
        return False

    if model.polynomial is None:
        return False

    if not model.polynomial.is_valid():
        return False

    if (
        model.polynomial.sample_count
        < LANE_MODEL.minimum_points
    ):
        return False

    if model.polynomial.fit_error < 0.0:
        return False

    if not (
        _is_finite(
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