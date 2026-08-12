"""
vision/lane_model.py

Construção do modelo matemático das linhas de faixa.

Responsabilidades:
    YOLOP LanePoint
        ↓
    filtragem
        ↓
    preparação dos pontos
        ↓
    ajuste polinomial
        ↓
    validação do ajuste
        ↓
    LaneModel

Este módulo NÃO:
    - identifica a faixa atual do veículo;
    - calcula o erro lateral do veículo;
    - decide se o ADAS deve atuar;
    - executa controle;
    - faz inferência YOLOP;
    - associa faixas entre frames.

A associação temporal será responsabilidade do LaneTracker.
A identificação da faixa ocupada será responsabilidade do LaneAssociation.
"""

from __future__ import annotations

import logging
from dataclasses import replace
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


# ============================================================================
# CONFIGURAÇÃO
# ============================================================================


DEFAULT_MIN_POINTS = 8

DEFAULT_MAX_POINTS = 80

DEFAULT_MIN_Y_SPAN = 30.0

DEFAULT_MAX_FIT_ERROR = 35.0

DEFAULT_OUTLIER_THRESHOLD = 3.0

DEFAULT_PROJECTION_STEP = 10.0

DEFAULT_MIN_CONFIDENCE = 0.20

DEFAULT_MIN_POLYNOMIAL_CONFIDENCE = 0.45

DEFAULT_MIN_PROJECTION_CONFIDENCE = 0.55


# ============================================================================
# HELPERS NUMÉRICOS
# ============================================================================


def _finite(value: float) -> bool:
    """
    Verifica se um valor é finito.
    """

    return bool(np.isfinite(value))


def _clip01(value: float) -> float:
    """
    Limita um valor ao intervalo [0, 1].
    """

    return float(
        np.clip(
            value,
            0.0,
            1.0,
        )
    )


def _safe_mean(
    values: Sequence[float],
) -> float:
    """
    Média segura.
    """

    if not values:
        return 0.0

    array = np.asarray(
        values,
        dtype=np.float64,
    )

    finite = array[
        np.isfinite(array)
    ]

    if finite.size == 0:
        return 0.0

    return float(
        np.mean(finite)
    )


def _safe_median(
    values: Sequence[float],
) -> float:
    """
    Mediana segura.
    """

    if not values:
        return 0.0

    array = np.asarray(
        values,
        dtype=np.float64,
    )

    finite = array[
        np.isfinite(array)
    ]

    if finite.size == 0:
        return 0.0

    return float(
        np.median(finite)
    )


# ============================================================================
# PONTOS
# ============================================================================


def filter_lane_points(
    points: Iterable[LanePoint],
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> List[LanePoint]:
    """
    Remove pontos inválidos ou numericamente impossíveis.

    Não altera os pontos originais.
    """

    result: List[LanePoint] = []

    threshold = _clip01(
        min_confidence
    )

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
    Remove duplicidades de Y.

    Para um mesmo Y, mantém o ponto com maior confiança.
    """

    best_by_y: dict[float, LanePoint] = {}

    for point in points:

        current = best_by_y.get(
            point.y
        )

        if (
            current is None
            or point.confidence
            > current.confidence
        ):
            best_by_y[point.y] = point

    return sorted(
        best_by_y.values(),
        key=lambda point: point.y,
    )


def limit_point_count(
    points: Sequence[LanePoint],
    max_points: int = DEFAULT_MAX_POINTS,
) -> List[LanePoint]:
    """
    Limita a quantidade de pontos preservando a distribuição vertical.

    Não simplesmente corta o início/fim da linha.
    """

    if len(points) <= max_points:
        return list(points)

    if max_points < 2:
        return [
            points[0]
        ]

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
    Pipeline de preparação dos pontos.
    """

    filtered = filter_lane_points(
        points,
        min_confidence=min_confidence,
    )

    sorted_points = sort_lane_points(
        filtered
    )

    unique_points = remove_duplicate_y(
        sorted_points
    )

    return limit_point_count(
        unique_points,
        max_points=max_points,
    )


# ============================================================================
# ESTATÍSTICAS DA LINHA
# ============================================================================


def lane_y_span(
    points: Sequence[LanePoint],
) -> float:
    """
    Extensão vertical observada da linha.
    """

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
    """
    Extensão horizontal observada da linha.
    """

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
    """
    Confiança média dos pontos.
    """

    return _safe_mean(
        [
            point.confidence
            for point in points
        ]
    )


def lane_confidence_score(
    points: Sequence[LanePoint],
) -> float:
    """
    Confiança estrutural da linha.

    Combina:
        - confiança dos pontos;
        - quantidade de pontos;
        - extensão vertical.

    O resultado é utilizado como indicador de qualidade,
    não como probabilidade estatística.
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

    score = (
        0.55 * confidence
        + 0.25 * count_score
        + 0.20 * span_score
    )

    return _clip01(score)


# ============================================================================
# CLASSIFICAÇÃO DA QUALIDADE
# ============================================================================


def classify_lane_quality(
    points: Sequence[LanePoint],
    polynomial: Optional[LanePolynomial] = None,
) -> LaneQuality:
    """
    Classifica a qualidade da linha.
    """

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
            and confidence >= 0.70
        ):
            return LaneQuality.EXCELLENT

    if (
        count >= 15
        and span >= 150.0
        and confidence >= 0.60
    ):
        return LaneQuality.GOOD

    return LaneQuality.PARTIAL


# ============================================================================
# NORMALIZAÇÃO PARA AJUSTE
# ============================================================================


def _normalize_y(
    y: np.ndarray,
) -> Tuple[
    np.ndarray,
    float,
    float,
]:
    """
    Normaliza Y para aproximadamente [-1, 1].

    Retorna:
        y_normalized
        center
        scale

    A normalização melhora a estabilidade numérica do ajuste cúbico.
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


# ============================================================================
# AJUSTE POLINOMIAL
# ============================================================================


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

    O ajuste é feito com Y normalizado para evitar problemas
    numéricos em resoluções elevadas.

    Depois os coeficientes são convertidos novamente para
    coordenadas originais.

    Se não houver informação suficiente, retorna um modelo inválido.
    """

    if degree != 3:
        raise ValueError(
            "Este módulo utiliza polinômio de terceiro grau."
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

    y_span = float(
        np.max(ys)
        - np.min(ys)
    )

    if y_span < DEFAULT_MIN_Y_SPAN:
        return LanePolynomial(
            valid=False,
            sample_count=len(points),
            y_min=float(np.min(ys)),
            y_max=float(np.max(ys)),
        )

    y_normalized, center, scale = (
        _normalize_y(ys)
    )

    try:

        coefficients = np.polyfit(
            y_normalized,
            xs,
            degree,
        )

    except (
        np.linalg.LinAlgError,
        ValueError,
        FloatingPointError,
    ):

        logger.debug(
            "Falha no ajuste polinomial."
        )

        return LanePolynomial(
            valid=False,
            sample_count=len(points),
            y_min=float(np.min(ys)),
            y_max=float(np.max(ys)),
        )

    if len(coefficients) != 4:
        return LanePolynomial(
            valid=False,
            sample_count=len(points),
        )

    a_n, b_n, c_n, d_n = (
        [
            float(value)
            for value in coefficients
        ]
    )

    # ----------------------------------------------------------------------
    # Conversão:
    #
    # z = (y - center) / scale
    #
    # x = a*z³ + b*z² + c*z + d
    #
    # Expandindo para x(y):
    #
    # a_original = a / scale³
    #
    # b_original =
    #     -3*a*center/scale³
    #     + b/scale²
    #
    # c_original =
    #     3*a*center²/scale³
    #     -2*b*center/scale²
    #     +c/scale
    #
    # d_original =
    #     -a*center³/scale³
    #     +b*center²/scale²
    #     -c*center/scale
    #     +d
    # ----------------------------------------------------------------------

    scale2 = scale ** 2
    scale3 = scale ** 3

    a = (
        a_n / scale3
    )

    b = (
        -3.0 * a_n * center / scale3
        + b_n / scale2
    )

    c = (
        3.0 * a_n * center ** 2 / scale3
        - 2.0 * b_n * center / scale2
        + c_n / scale
    )

    d = (
        -a_n * center ** 3 / scale3
        + b_n * center ** 2 / scale2
        - c_n * center / scale
        + d_n
    )

    polynomial = LanePolynomial(
        a=float(a),
        b=float(b),
        c=float(c),
        d=float(d),
        valid=True,
        sample_count=len(points),
        y_min=float(np.min(ys)),
        y_max=float(np.max(ys)),
    )

    # ----------------------------------------------------------------------
    # Erro residual.
    # ----------------------------------------------------------------------

    predicted = (
        polynomial.a * ys ** 3
        + polynomial.b * ys ** 2
        + polynomial.c * ys
        + polynomial.d
    )

    residuals = np.abs(
        xs - predicted
    )

    fit_error = float(
        np.median(residuals)
    )

    polynomial.fit_error = fit_error

    polynomial.confidence = _calculate_fit_confidence(
        points=points,
        fit_error=fit_error,
    )

    if fit_error > max_fit_error:
        polynomial.valid = False

    return polynomial


# ============================================================================
# CONFIANÇA DO AJUSTE
# ============================================================================


def _calculate_fit_confidence(
    points: Sequence[LanePoint],
    fit_error: float,
) -> float:
    """
    Calcula confiança do modelo matemático.

    Não é uma probabilidade.

    É um score de qualidade utilizado para decidir se a projeção
    pode ser confiável.
    """

    point_confidence = lane_mean_confidence(
        points
    )

    count_score = _clip01(
        len(points) / 25.0
    )

    error_score = float(
        np.exp(
            -fit_error / 20.0
        )
    )

    return _clip01(
        0.50 * point_confidence
        + 0.20 * count_score
        + 0.30 * error_score
    )


# ============================================================================
# REJEIÇÃO DE OUTLIERS
# ============================================================================


def remove_polynomial_outliers(
    points: Sequence[LanePoint],
    polynomial: LanePolynomial,
    threshold: float = DEFAULT_OUTLIER_THRESHOLD,
) -> List[LanePoint]:
    """
    Remove pontos muito distantes do modelo.

    O parâmetro threshold representa aproximadamente quantos
    desvios robustos serão tolerados.

    Não remove pontos quando o modelo é inválido.
    """

    if not polynomial.valid:
        return list(points)

    if len(points) < 6:
        return list(points)

    residuals = []

    for point in points:

        predicted_x = polynomial.evaluate(
            point.y
        )

        residuals.append(
            abs(
                point.x
                - predicted_x
            )
        )

    median = _safe_median(
        residuals
    )

    deviations = np.asarray(
        [
            abs(
                residual
                - median
            )
            for residual in residuals
        ],
        dtype=np.float64,
    )

    mad = float(
        np.median(
            deviations
        )
    )

    if mad < 1e-6:
        return list(points)

    robust_scale = (
        1.4826 * mad
    )

    limit = (
        median
        + threshold * robust_scale
    )

    return [
        point
        for point, residual in zip(
            points,
            residuals,
        )
        if residual <= limit
    ]


# ============================================================================
# AJUSTE ROBUSTO
# ============================================================================


def fit_polynomial_robust(
    points: Sequence[LanePoint],
    min_points: int = DEFAULT_MIN_POINTS,
    max_fit_error: float = DEFAULT_MAX_FIT_ERROR,
) -> LanePolynomial:
    """
    Ajuste em duas etapas:

        1. ajuste inicial;
        2. remoção de outliers;
        3. novo ajuste.

    Isso reduz o impacto de reflexos, ruído do YOLOP e pixels
    isolados que tenham entrado na linha.
    """

    prepared = prepare_lane_points(
        points
    )

    if len(prepared) < min_points:
        return LanePolynomial(
            valid=False,
            sample_count=len(prepared),
        )

    first = fit_polynomial(
        prepared,
        min_points=min_points,
        max_fit_error=max_fit_error,
    )

    if not first.valid:
        return first

    filtered = remove_polynomial_outliers(
        prepared,
        first,
    )

    if len(filtered) < min_points:
        return first

    second = fit_polynomial(
        filtered,
        min_points=min_points,
        max_fit_error=max_fit_error,
    )

    if not second.valid:
        return first

    return second


# ============================================================================
# CONSTRUÇÃO DE LaneModel
# ============================================================================


def build_lane_model(
    lane_id: int,
    points: Sequence[LanePoint],
    min_points: int = DEFAULT_MIN_POINTS,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    max_points: int = DEFAULT_MAX_POINTS,
    max_fit_error: float = DEFAULT_MAX_FIT_ERROR,
) -> LaneModel:
    """
    Constrói um LaneModel completo a partir dos pontos do YOLOP.
    """

    prepared = prepare_lane_points(
        points,
        min_confidence=min_confidence,
        max_points=max_points,
    )

    line = LaneLine(
        lane_id=lane_id,
        points=prepared,
        confidence=lane_confidence_score(
            prepared
        ),
        valid=bool(
            prepared
        ),
        detected_directly=True,
        projected=False,
        projection_quality=ProjectionQuality.NONE,
    )

    if len(prepared) < min_points:

        line.quality = (
            classify_lane_quality(
                prepared
            )
        )

        return LaneModel(
            lane_id=lane_id,
            line=line,
            polynomial=None,
            projection=None,
            tracked=False,
            stable=False,
            valid=False,
        )

    polynomial = fit_polynomial_robust(
        prepared,
        min_points=min_points,
        max_fit_error=max_fit_error,
    )

    line.quality = classify_lane_quality(
        prepared,
        polynomial,
    )

    if (
        polynomial.valid
        and polynomial.confidence
        >= DEFAULT_MIN_POLYNOMIAL_CONFIDENCE
    ):
        valid = True
    else:
        valid = (
            len(prepared)
            >= min_points
            and line.confidence
            >= 0.45
        )

    return LaneModel(
        lane_id=lane_id,
        line=line,
        polynomial=polynomial,
        projection=None,
        tracked=False,
        stable=False,
        valid=valid,
    )


# ============================================================================
# AVALIAÇÃO DO MODELO
# ============================================================================


def evaluate_lane_model(
    model: LaneModel,
    y_values: Sequence[float],
) -> List[LanePoint]:
    """
    Gera pontos previstos pelo polinômio.

    Estes pontos são matemáticos e NÃO devem ser confundidos
    com pontos observados pelo YOLOP.
    """

    if (
        model.polynomial is None
        or not model.polynomial.valid
    ):
        return []

    polynomial = model.polynomial

    result = []

    for y in y_values:

        if not _finite(float(y)):
            continue

        if (
            y < polynomial.y_min
            or y > polynomial.y_max
        ):
            continue

        x = polynomial.evaluate(
            float(y)
        )

        if not _finite(x):
            continue

        result.append(
            LanePoint(
                x=float(x),
                y=float(y),
                confidence=float(
                    polynomial.confidence
                ),
                valid=True,
            )
        )

    return result


# ============================================================================
# PROJEÇÃO
# ============================================================================


def project_lane(
    model: LaneModel,
    target_y_min: float,
    target_y_max: float,
    step: float = DEFAULT_PROJECTION_STEP,
    min_confidence: float = DEFAULT_MIN_PROJECTION_CONFIDENCE,
) -> LaneProjection:
    """
    Projeta uma linha para a região ainda não observada.

    Regra importante:

    A projeção somente é permitida quando:

        - o polinômio é válido;
        - possui confiança suficiente;
        - possui pontos suficientes;
        - existe extensão vertical real da observação.

    A função NÃO permite uma projeção arbitrariamente longa sem
    controle de qualidade.
    """

    polynomial = model.polynomial

    if (
        polynomial is None
        or not polynomial.valid
    ):
        return LaneProjection(
            polynomial=polynomial,
            valid=False,
        )

    if (
        polynomial.confidence
        < min_confidence
    ):
        return LaneProjection(
            polynomial=polynomial,
            valid=False,
        )

    if polynomial.sample_count < DEFAULT_MIN_POINTS:
        return LaneProjection(
            polynomial=polynomial,
            valid=False,
        )

    if (
        target_y_max
        <= target_y_min
    ):
        return LaneProjection(
            polynomial=polynomial,
            valid=False,
        )

    observed_max = polynomial.y_max

    observed_min = polynomial.y_min

    # ----------------------------------------------------------------------
    # A projeção pode continuar além da região observada,
    # mas nunca deve retroceder para dentro da região já observada.
    # ----------------------------------------------------------------------

    projection_start = max(
        target_y_min,
        observed_max,
    )

    projection_end = max(
        projection_start,
        target_y_max,
    )

    if (
        projection_end
        <= projection_start
    ):
        return LaneProjection(
            polynomial=polynomial,
            valid=False,
        )

    values = np.arange(
        projection_start,
        projection_end + step,
        step,
        dtype=np.float64,
    )

    projected_points = []

    for y in values:

        x = polynomial.evaluate(
            float(y)
        )

        if not _finite(x):
            continue

        projected_points.append(
            LanePoint(
                x=float(x),
                y=float(y),
                confidence=float(
                    polynomial.confidence
                    * 0.85
                ),
                valid=True,
            )
        )

    if len(projected_points) < 2:
        return LaneProjection(
            polynomial=polynomial,
            points=projected_points,
            valid=False,
        )

    quality = (
        ProjectionQuality.HIGH
        if polynomial.confidence >= 0.75
        else ProjectionQuality.MEDIUM
        if polynomial.confidence >= 0.60
        else ProjectionQuality.LOW
    )

    return LaneProjection(
        polynomial=polynomial,
        points=projected_points,
        quality=quality,
        extrapolated=True,
        valid=True,
        horizon_y=float(
            projected_points[-1].y
        ),
    )


# ============================================================================
# PROJEÇÃO CONTROLADA
# ============================================================================


def project_lane_safe(
    model: LaneModel,
    target_y_max: float,
    frame_height: float,
) -> LaneModel:
    """
    Adiciona projeção segura ao modelo.

    A projeção é limitada ao frame.

    Este método NÃO substitui os pontos observados.
    """

    if frame_height <= 0:
        return model

    if (
        model.polynomial is None
        or not model.polynomial.valid
    ):
        return model

    projection = project_lane(
        model=model,
        target_y_min=model.polynomial.y_max,
        target_y_max=min(
            target_y_max,
            frame_height,
        ),
    )

    model.projection = projection

    if projection.valid:

        model.line.projected = True

        model.line.projection_quality = (
            projection.quality
        )

    return model


# ============================================================================
# ESTABILIDADE GEOMÉTRICA
# ============================================================================


def compare_polynomials(
    previous: Optional[LanePolynomial],
    current: Optional[LanePolynomial],
    y_reference: float,
) -> Optional[float]:
    """
    Mede a diferença horizontal entre dois modelos no mesmo Y.

    Retorna None quando não existem modelos válidos.
    """

    if (
        previous is None
        or current is None
        or not previous.valid
        or not current.valid
    ):
        return None

    previous_x = previous.evaluate(
        y_reference
    )

    current_x = current.evaluate(
        y_reference
    )

    if not (
        _finite(previous_x)
        and _finite(current_x)
    ):
        return None

    return float(
        abs(
            current_x
            - previous_x
        )
    )


def polynomial_is_stable(
    previous: Optional[LanePolynomial],
    current: Optional[LanePolynomial],
    y_reference: float,
    max_shift_pixels: float = 80.0,
) -> bool:
    """
    Verifica se dois modelos consecutivos são geometricamente
    compatíveis.
    """

    difference = compare_polynomials(
        previous,
        current,
        y_reference,
    )

    if difference is None:
        return False

    return (
        difference
        <= max_shift_pixels
    )


# ============================================================================
# CONSTRUÇÃO DE MÚLTIPLAS LANES
# ============================================================================


def build_lane_models(
    lanes: Sequence[Sequence[LanePoint]],
    min_points: int = DEFAULT_MIN_POINTS,
) -> List[LaneModel]:
    """
    Constrói modelos para todas as linhas fornecidas.

    Nenhuma linha é descartada simplesmente porque está à esquerda
    ou à direita do centro da imagem.

    Isso é essencial para pistas com múltiplas faixas.
    """

    models: List[LaneModel] = []

    for lane_id, points in enumerate(
        lanes
    ):

        model = build_lane_model(
            lane_id=lane_id,
            points=points,
            min_points=min_points,
        )

        if model.valid:
            models.append(
                model
            )

    return models


# ============================================================================
# PROJEÇÃO DE TODAS AS LANES
# ============================================================================


def project_lane_models(
    models: Sequence[LaneModel],
    frame_height: int,
    horizon_y: Optional[float] = None,
) -> List[LaneModel]:
    """
    Projeta todas as linhas válidas.

    Não assume quantidade fixa de lanes.
    """

    if horizon_y is None:
        horizon_y = float(
            frame_height
        )

    result = []

    for model in models:

        copied = model

        project_lane_safe(
            copied,
            target_y_max=float(
                horizon_y
            ),
            frame_height=float(
                frame_height
            ),
        )

        result.append(
            copied
        )

    return result


# ============================================================================
# API DE ALTO NÍVEL
# ============================================================================


class LaneModelBuilder:
    """
    Interface orientada a objeto para construção dos modelos.

    Mantém parâmetros centralizados para que futuramente seja
    possível alterar o comportamento sem modificar o pipeline.
    """

    def __init__(
        self,
        min_points: int = DEFAULT_MIN_POINTS,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        max_points: int = DEFAULT_MAX_POINTS,
        max_fit_error: float = DEFAULT_MAX_FIT_ERROR,
    ) -> None:

        self.min_points = max(
            4,
            int(min_points),
        )

        self.min_confidence = _clip01(
            min_confidence
        )

        self.max_points = max(
            self.min_points,
            int(max_points),
        )

        self.max_fit_error = max(
            1.0,
            float(max_fit_error),
        )

    def build(
        self,
        lane_id: int,
        points: Sequence[LanePoint],
    ) -> LaneModel:
        """
        Constrói um único modelo.
        """

        return build_lane_model(
            lane_id=lane_id,
            points=points,
            min_points=self.min_points,
            min_confidence=self.min_confidence,
            max_points=self.max_points,
            max_fit_error=self.max_fit_error,
        )

    def build_all(
        self,
        lanes: Sequence[Sequence[LanePoint]],
    ) -> List[LaneModel]:
        """
        Constrói todos os modelos.
        """

        return [
            self.build(
                lane_id=index,
                points=points,
            )
            for index, points in enumerate(
                lanes
            )
        ]


# ============================================================================
# EXPORTS
# ============================================================================


__all__ = [
    "DEFAULT_MIN_POINTS",
    "DEFAULT_MAX_POINTS",
    "DEFAULT_MIN_Y_SPAN",
    "DEFAULT_MAX_FIT_ERROR",
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
    "fit_polynomial_robust",
    "build_lane_model",
    "evaluate_lane_model",
    "project_lane",
    "project_lane_safe",
    "compare_polynomials",
    "polynomial_is_stable",
    "build_lane_models",
    "project_lane_models",
    "LaneModelBuilder",
]