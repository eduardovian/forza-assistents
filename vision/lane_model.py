"""
vision/lane_model.py

Forza Assistents
================

Modelo matemático oficial das linhas de faixa.

Modelo:

    x(y) = a*y³ + b*y² + c*y + d

Responsabilidades:

    LaneLine
        ↓
    preparação dos pontos
        ↓
    normalização numérica
        ↓
    ajuste cúbico robusto
        ↓
    LanePolynomial

Este módulo não realiza:

    - captura;
    - inferência;
    - tracking;
    - associação de lanes;
    - decisão ADAS;
    - controle do volante.

O contrato de dados oficial é definido exclusivamente em
vision.lane_types.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from config import LANE_MODEL

from .lane_types import (
    LaneLine,
    LanePoint,
    LanePolynomial,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTES
# =============================================================================

POLYNOMIAL_DEGREE = 3

DEFAULT_MINIMUM_POINTS = 6
DEFAULT_MINIMUM_Y_SPAN = 20.0
DEFAULT_MAX_OUTLIER_ITERATIONS = 3
DEFAULT_OUTLIER_THRESHOLD = 2.5
DEFAULT_MINIMUM_CONFIDENCE = 0.35
DEFAULT_MAX_FIT_ERROR = 25.0


# =============================================================================
# RESULTADO
# =============================================================================

@dataclass(frozen=True, slots=True)
class LaneModelResult:
    """
    Resultado de um ajuste de modelo.

    O objeto contém somente informações matemáticas.
    """

    model: Optional[LanePolynomial]

    valid: bool

    point_count: int

    inlier_count: int

    fit_error: float

    confidence: float

    y_min: float

    y_max: float

    rejected_outliers: int

    reason: str = ""

    @property
    def has_model(self) -> bool:
        return self.model is not None and self.valid


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

def _config_value(
    name: str,
    default: object,
) -> object:
    """
    Obtém uma configuração centralizada.

    Não cria uma configuração paralela: somente fornece defaults
    defensivos para versões incompletas de config.py.
    """

    return getattr(
        LANE_MODEL,
        name,
        default,
    )


def _minimum_points() -> int:
    return max(
        4,
        int(
            _config_value(
                "minimum_points",
                DEFAULT_MINIMUM_POINTS,
            )
        ),
    )


def _minimum_y_span() -> float:
    return max(
        1.0,
        float(
            _config_value(
                "minimum_y_span",
                DEFAULT_MINIMUM_Y_SPAN,
            )
        ),
    )


def _max_outlier_iterations() -> int:
    return max(
        0,
        int(
            _config_value(
                "max_outlier_iterations",
                DEFAULT_MAX_OUTLIER_ITERATIONS,
            )
        ),
    )


def _outlier_threshold() -> float:
    return max(
        0.1,
        float(
            _config_value(
                "outlier_threshold",
                DEFAULT_OUTLIER_THRESHOLD,
            )
        ),
    )


def _minimum_confidence() -> float:
    return float(
        np.clip(
            _config_value(
                "minimum_confidence",
                DEFAULT_MINIMUM_CONFIDENCE,
            ),
            0.0,
            1.0,
        )
    )


def _max_fit_error() -> float:
    value = _config_value(
        "max_fit_error",
        _config_value(
            "fit_error_threshold",
            DEFAULT_MAX_FIT_ERROR,
        ),
    )

    try:
        value = float(value)
    except (TypeError, ValueError):
        return DEFAULT_MAX_FIT_ERROR

    if not math.isfinite(value) or value <= 0.0:
        return DEFAULT_MAX_FIT_ERROR

    return value


def _validate_configuration() -> None:
    degree = int(
        _config_value(
            "polynomial_degree",
            POLYNOMIAL_DEGREE,
        )
    )

    if degree != POLYNOMIAL_DEGREE:
        raise ValueError(
            "LANE_MODEL.polynomial_degree must be exactly 3."
        )

    if _minimum_points() < 4:
        raise ValueError(
            "LANE_MODEL.minimum_points must be >= 4."
        )

    if _minimum_y_span() <= 0.0:
        raise ValueError(
            "LANE_MODEL.minimum_y_span must be > 0."
        )

    if _outlier_threshold() <= 0.0:
        raise ValueError(
            "LANE_MODEL.outlier_threshold must be > 0."
        )


_validate_configuration()


# =============================================================================
# VALIDAÇÃO DOS PONTOS
# =============================================================================

def _finite_point(
    point: object,
) -> Optional[Tuple[float, float]]:
    """Extrai (x, y) de um LanePoint válido."""

    if not isinstance(
        point,
        LanePoint,
    ):
        return None

    try:
        x = float(point.x)
        y = float(point.y)
    except (TypeError, ValueError):
        return None

    if not (
        math.isfinite(x)
        and math.isfinite(y)
    ):
        return None

    return x, y


def prepare_points(
    points: Iterable[LanePoint],
) -> list[LanePoint]:
    """
    Prepara pontos para o fitting.

    Operações:

        1. validação;
        2. ordenação por Y;
        3. remoção de Y duplicado;
        4. preservação do melhor ponto duplicado.
    """

    valid: list[LanePoint] = []

    for point in points:

        if _finite_point(point) is not None:
            valid.append(point)

    valid.sort(
        key=lambda point: point.y
    )

    best_by_y: dict[float, LanePoint] = {}

    for point in valid:

        current = best_by_y.get(
            point.y
        )

        if current is None:
            best_by_y[
                point.y
            ] = point
            continue

        # O contrato atual não possui confiança por ponto.
        # Portanto, em caso de Y duplicado, preservamos o primeiro.
        continue

    return list(
        best_by_y.values()
    )


# =============================================================================
# NORMALIZAÇÃO
# =============================================================================

@dataclass(frozen=True, slots=True)
class _Normalization:
    y_center: float
    y_scale: float


def _normalization(
    y: np.ndarray,
) -> _Normalization:
    """
    Normalização robusta de Y.

    O domínio normalizado fica aproximadamente em [-1, 1].
    """

    y_min = float(
        np.min(y)
    )

    y_max = float(
        np.max(y)
    )

    center = (
        y_min
        + y_max
    ) * 0.5

    scale = (
        y_max
        - y_min
    ) * 0.5

    if scale <= 1e-9:
        scale = 1.0

    return _Normalization(
        y_center=center,
        y_scale=scale,
    )


def _normalize_y(
    y: np.ndarray,
    normalization: _Normalization,
) -> np.ndarray:

    return (
        y
        - normalization.y_center
    ) / normalization.y_scale


# =============================================================================
# FITTING
# =============================================================================

def _fit_cubic(
    x: np.ndarray,
    y_normalized: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Ajuste cúbico por mínimos quadrados.

    Retorna:

        [a, b, c, d]

    no domínio normalizado.
    """

    if len(x) < 4:
        return None

    try:

        coefficients = np.polynomial.polynomial.polyfit(
            y_normalized,
            x,
            deg=POLYNOMIAL_DEGREE,
        )

    except (
        np.linalg.LinAlgError,
        ValueError,
        FloatingPointError,
    ):

        return None

    coefficients = np.asarray(
        coefficients,
        dtype=np.float64,
    )

    if coefficients.shape != (4,):
        return None

    if not np.all(
        np.isfinite(coefficients)
    ):
        return None

    # polyfit retorna:
    #
    # d, c, b, a
    #
    # enquanto o contrato usa:
    #
    # a, b, c, d

    return coefficients[
        ::-1
    ]


def _evaluate_normalized(
    coefficients: np.ndarray,
    y_normalized: np.ndarray,
) -> np.ndarray:

    a, b, c, d = coefficients

    return (
        (
            (
                a * y_normalized
                + b
            )
            * y_normalized
            + c
        )
        * y_normalized
        + d
    )


# =============================================================================
# ROBUSTEZ
# =============================================================================

def _robust_fit(
    x: np.ndarray,
    y_normalized: np.ndarray,
) -> Tuple[
    Optional[np.ndarray],
    np.ndarray,
]:
    """
    Ajuste cúbico com rejeição iterativa de outliers.

    Retorna:

        coefficients
        inlier_mask
    """

    count = len(x)

    if count < 4:
        return None, np.zeros(
            count,
            dtype=bool,
        )

    mask = np.ones(
        count,
        dtype=bool,
    )

    threshold = _outlier_threshold()

    coefficients: Optional[np.ndarray] = None

    for _ in range(
        _max_outlier_iterations() + 1
    ):

        if int(
            np.count_nonzero(mask)
        ) < 4:
            break

        coefficients = _fit_cubic(
            x[mask],
            y_normalized[mask],
        )

        if coefficients is None:
            break

        prediction = _evaluate_normalized(
            coefficients,
            y_normalized,
        )

        residuals = np.abs(
            x - prediction
        )

        inlier_residuals = residuals[
            mask
        ]

        if len(
            inlier_residuals
        ) == 0:
            break

        median = float(
            np.median(
                inlier_residuals
            )
        )

        mad = float(
            np.median(
                np.abs(
                    inlier_residuals
                    - median
                )
            )
        )

        # Estimativa robusta do desvio.
        sigma = max(
            1e-6,
            1.4826 * mad,
        )

        limit = max(
            threshold,
            threshold * sigma,
        )

        new_mask = (
            residuals <= limit
        )

        if int(
            np.count_nonzero(new_mask)
        ) < 4:
            break

        if np.array_equal(
            new_mask,
            mask,
        ):
            mask = new_mask
            break

        mask = new_mask

    if coefficients is None:
        return (
            None,
            mask,
        )

    # Refit final somente nos inliers.
    final_coefficients = _fit_cubic(
        x[mask],
        y_normalized[mask],
    )

    if final_coefficients is None:
        return (
            None,
            mask,
        )

    return (
        final_coefficients,
        mask,
    )


# =============================================================================
# CONVERSÃO PARA COORDENADAS ORIGINAIS
# =============================================================================

def _denormalize_cubic(
    coefficients: np.ndarray,
    normalization: _Normalization,
) -> Tuple[
    float,
    float,
    float,
    float,
]:
    """
    Converte:

        x = A*z³ + B*z² + C*z + D

    onde:

        z = (y - center) / scale

    para:

        x = a*y³ + b*y² + c*y + d
    """

    A, B, C, D = coefficients

    center = normalization.y_center
    scale = normalization.y_scale

    if abs(scale) <= 1e-12:
        raise ValueError(
            "Normalization scale is too small."
        )

    # Expansão analítica.
    a = A / (
        scale ** 3
    )

    b = (
        B / (scale ** 2)
        - 3.0 * A * center
        / (scale ** 3)
    )

    c = (
        C / scale
        - 2.0 * B * center
        / (scale ** 2)
        + 3.0 * A * center ** 2
        / (scale ** 3)
    )

    d = (
        D
        - C * center / scale
        + B * center ** 2
        / (scale ** 2)
        - A * center ** 3
        / (scale ** 3)
    )

    values = (
        a,
        b,
        c,
        d,
    )

    if not all(
        math.isfinite(value)
        for value in values
    ):
        raise ValueError(
            "Denormalized polynomial contains "
            "non-finite coefficients."
        )

    return values


# =============================================================================
# MÉTRICAS
# =============================================================================

def _fit_error(
    x: np.ndarray,
    y_normalized: np.ndarray,
    coefficients: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Erro médio absoluto dos inliers."""

    if not np.any(mask):
        return float("inf")

    prediction = _evaluate_normalized(
        coefficients,
        y_normalized,
    )

    residuals = np.abs(
        x[mask]
        - prediction[mask]
    )

    if residuals.size == 0:
        return float("inf")

    return float(
        np.mean(residuals)
    )


def _confidence(
    *,
    fit_error: float,
    point_count: int,
    inlier_count: int,
    y_span: float,
) -> float:
    """
    Confiança matemática do modelo.

    Não representa confiança neural.
    """

    if point_count <= 0:
        return 0.0

    inlier_ratio = (
        inlier_count
        / point_count
    )

    error_limit = _max_fit_error()

    error_score = max(
        0.0,
        min(
            1.0,
            1.0
            - fit_error
            / error_limit,
        ),
    )

    point_score = max(
        0.0,
        min(
            1.0,
            point_count
            / 20.0,
        ),
    )

    span_score = max(
        0.0,
        min(
            1.0,
            y_span
            / (
                _minimum_y_span()
                * 5.0
            ),
        ),
    )

    confidence = (
        0.45 * error_score
        + 0.30 * inlier_ratio
        + 0.15 * point_score
        + 0.10 * span_score
    )

    return float(
        np.clip(
            confidence,
            0.0,
            1.0,
        )
    )


# =============================================================================
# AJUSTE PRINCIPAL
# =============================================================================

def fit_lane_polynomial(
    points: Iterable[LanePoint],
) -> LaneModelResult:
    """
    Ajusta um modelo cúbico a uma coleção de LanePoint.
    """

    prepared = prepare_points(
        points
    )

    point_count = len(
        prepared
    )

    if point_count < _minimum_points():

        return LaneModelResult(
            model=None,
            valid=False,
            point_count=point_count,
            inlier_count=0,
            fit_error=float("inf"),
            confidence=0.0,
            y_min=0.0,
            y_max=0.0,
            rejected_outliers=0,
            reason="insufficient_points",
        )

    x = np.asarray(
        [
            point.x
            for point in prepared
        ],
        dtype=np.float64,
    )

    y = np.asarray(
        [
            point.y
            for point in prepared
        ],
        dtype=np.float64,
    )

    if not (
        np.all(np.isfinite(x))
        and np.all(np.isfinite(y))
    ):

        return LaneModelResult(
            model=None,
            valid=False,
            point_count=point_count,
            inlier_count=0,
            fit_error=float("inf"),
            confidence=0.0,
            y_min=0.0,
            y_max=0.0,
            rejected_outliers=0,
            reason="non_finite_points",
        )

    y_min = float(
        np.min(y)
    )

    y_max = float(
        np.max(y)
    )

    y_span = (
        y_max
        - y_min
    )

    if y_span < _minimum_y_span():

        return LaneModelResult(
            model=None,
            valid=False,
            point_count=point_count,
            inlier_count=0,
            fit_error=float("inf"),
            confidence=0.0,
            y_min=y_min,
            y_max=y_max,
            rejected_outliers=0,
            reason="insufficient_y_span",
        )

    normalization = _normalization(
        y
    )

    y_normalized = _normalize_y(
        y,
        normalization,
    )

    coefficients, mask = _robust_fit(
        x,
        y_normalized,
    )

    if coefficients is None:

        return LaneModelResult(
            model=None,
            valid=False,
            point_count=point_count,
            inlier_count=0,
            fit_error=float("inf"),
            confidence=0.0,
            y_min=y_min,
            y_max=y_max,
            rejected_outliers=point_count,
            reason="fit_failed",
        )

    inlier_count = int(
        np.count_nonzero(mask)
    )

    error = _fit_error(
        x,
        y_normalized,
        coefficients,
        mask,
    )

    confidence = _confidence(
        fit_error=error,
        point_count=point_count,
        inlier_count=inlier_count,
        y_span=y_span,
    )

    rejected = (
        point_count
        - inlier_count
    )

    if (
        not math.isfinite(error)
        or error > _max_fit_error()
    ):

        return LaneModelResult(
            model=None,
            valid=False,
            point_count=point_count,
            inlier_count=inlier_count,
            fit_error=error,
            confidence=confidence,
            y_min=y_min,
            y_max=y_max,
            rejected_outliers=rejected,
            reason="fit_error_too_high",
        )

    if confidence < _minimum_confidence():

        return LaneModelResult(
            model=None,
            valid=False,
            point_count=point_count,
            inlier_count=inlier_count,
            fit_error=error,
            confidence=confidence,
            y_min=y_min,
            y_max=y_max,
            rejected_outliers=rejected,
            reason="low_confidence",
        )

    try:

        original_coefficients = (
            _denormalize_cubic(
                coefficients,
                normalization,
            )
        )

    except ValueError:

        return LaneModelResult(
            model=None,
            valid=False,
            point_count=point_count,
            inlier_count=inlier_count,
            fit_error=error,
            confidence=0.0,
            y_min=y_min,
            y_max=y_max,
            rejected_outliers=rejected,
            reason="denormalization_failed",
        )

    model = LanePolynomial(
        coefficients=original_coefficients,
        fit_error=error,
        confidence=confidence,
        y_min=y_min,
        y_max=y_max,
        normalized=False,
    )

    return LaneModelResult(
        model=model,
        valid=True,
        point_count=point_count,
        inlier_count=inlier_count,
        fit_error=error,
        confidence=confidence,
        y_min=y_min,
        y_max=y_max,
        rejected_outliers=rejected,
        reason="ok",
    )


# =============================================================================
# LANE MODEL CLASS
# =============================================================================

class LaneModel:
    """
    Interface orientada a objetos para o ajuste de lanes.
    """

    def __init__(
        self,
        minimum_points: Optional[int] = None,
        minimum_y_span: Optional[float] = None,
    ) -> None:

        self.minimum_points = (
            _minimum_points()
            if minimum_points is None
            else max(
                4,
                int(minimum_points),
            )
        )

        self.minimum_y_span = (
            _minimum_y_span()
            if minimum_y_span is None
            else max(
                1.0,
                float(minimum_y_span),
            )
        )

    def fit(
        self,
        points: Iterable[LanePoint],
    ) -> LaneModelResult:

        return fit_lane_polynomial(
            points
        )

    def fit_lane(
        self,
        lane: LaneLine,
    ) -> LaneModelResult:

        return self.fit(
            lane.points
        )


# =============================================================================
# API FUNCIONAL
# =============================================================================

def model_lane(
    points: Iterable[LanePoint],
) -> LaneModelResult:
    """API funcional para ajuste cúbico."""

    return fit_lane_polynomial(
        points
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "LaneModel",
    "LaneModelResult",
    "fit_lane_polynomial",
    "model_lane",
    "prepare_points",
]