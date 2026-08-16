"""
vision/lane_projection.py

Projeção matemática das linhas de faixa.

Responsabilidades
-----------------
LanePoint
    ↓
filtragem
    ↓
ajuste polinomial x(y)
    ↓
LanePolynomial
    ↓
extrapolação controlada
    ↓
LaneProjection

Este módulo NÃO realiza:
    - inferência YOLOP;
    - captura de tela;
    - ROI;
    - tracking;
    - associação de lanes;
    - decisão ADAS;
    - controle do veículo.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import numpy as np

from vision.lane_types import (
    LaneLine,
    LaneModel,
    LanePoint,
    LanePolynomial,
    LaneProjection,
    ProjectionQuality,
)


# =============================================================================
# CONSTANTES
# =============================================================================

DEFAULT_MIN_POINTS = 8
DEFAULT_DEGREE = 3
DEFAULT_MAX_PROJECTION_DISTANCE = 300.0
DEFAULT_MIN_CONFIDENCE = 0.35
DEFAULT_SAMPLE_STEP = 8.0

DEFAULT_MIN_VERTICAL_SPAN = 20.0
DEFAULT_MAX_FIT_ERROR = 40.0


# =============================================================================
# UTILIDADES
# =============================================================================


def _finite(value: Any) -> Optional[float]:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(value):
        return None

    return value


def _clip01(value: Any) -> float:
    value = _finite(value)

    if value is None:
        return 0.0

    return max(0.0, min(1.0, value))


def _empty_projection(
    *,
    polynomial: Optional[LanePolynomial] = None,
) -> LaneProjection:
    return LaneProjection(
        polynomial=polynomial,
        points=[],
        quality=ProjectionQuality.NONE,
        extrapolated=False,
        valid=False,
        horizon_y=None,
    )


# =============================================================================
# ENGINE
# =============================================================================


class LaneProjectionEngine:
    """
    Motor matemático de projeção de uma lane.

    O ajuste é realizado diretamente sobre os LanePoint recebidos.

    Modelo:

        x = a*y³ + b*y² + c*y + d

    Para estabilidade numérica, o ajuste interno utiliza
    Y normalizado.

    A API aceita:

        project(list[LanePoint])

    ou:

        project(LaneModel)
    """

    def __init__(
        self,
        min_points: int = DEFAULT_MIN_POINTS,
        degree: int = DEFAULT_DEGREE,
        max_projection_distance: float = DEFAULT_MAX_PROJECTION_DISTANCE,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        *,
        min_vertical_span: float = DEFAULT_MIN_VERTICAL_SPAN,
        max_fit_error: float = DEFAULT_MAX_FIT_ERROR,
        sample_step: float = DEFAULT_SAMPLE_STEP,
    ) -> None:

        self.min_points = max(
            2,
            int(min_points),
        )

        self.degree = max(
            1,
            min(3, int(degree)),
        )

        self.polynomial_degree = self.degree

        self.max_projection_distance = float(
            max_projection_distance
        )

        if (
            not math.isfinite(
                self.max_projection_distance
            )
            or self.max_projection_distance <= 0.0
        ):
            raise ValueError(
                "max_projection_distance deve ser > 0."
            )

        self.min_confidence = _clip01(
            min_confidence
        )

        self.min_vertical_span = max(
            0.0,
            float(min_vertical_span),
        )

        self.max_fit_error = max(
            0.0,
            float(max_fit_error),
        )

        self.sample_step = max(
            1.0,
            float(sample_step),
        )

    # =========================================================================
    # VALIDAÇÃO DOS PONTOS
    # =========================================================================

    @staticmethod
    def _valid_points(
        points: Sequence[LanePoint],
    ) -> list[LanePoint]:

        if points is None:
            return []

        result: list[LanePoint] = []

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

            try:
                valid = bool(
                    point.valid
                )
            except Exception:
                continue

            if not valid:
                continue

            x = _finite(point.x)
            y = _finite(point.y)
            confidence = _finite(
                point.confidence
            )

            if (
                x is None
                or y is None
                or confidence is None
            ):
                continue

            result.append(
                LanePoint(
                    x=x,
                    y=y,
                    confidence=_clip01(
                        confidence
                    ),
                    valid=True,
                )
            )

        result.sort(
            key=lambda p: p.y
        )

        return result

    # =========================================================================
    # PREPARAÇÃO
    # =========================================================================

    def _prepare_points(
        self,
        points: Sequence[LanePoint],
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:

        valid = self._valid_points(
            points
        )

        if len(valid) < self.min_points:
            raise ValueError(
                "Pontos insuficientes para projeção."
            )

        # ---------------------------------------------------------------------
        # Remove duplicações de Y.
        # Mantemos uma média ponderada em X.
        # ---------------------------------------------------------------------

        groups: dict[float, list[LanePoint]] = {}

        for point in valid:

            key = round(
                float(point.y),
                4,
            )

            groups.setdefault(
                key,
                [],
            ).append(point)

        x_values: list[float] = []
        y_values: list[float] = []
        confidence_values: list[float] = []

        for group in groups.values():

            if not group:
                continue

            weights = np.asarray(
                [
                    max(
                        0.0,
                        float(p.confidence),
                    )
                    for p in group
                ],
                dtype=np.float64,
            )

            xs = np.asarray(
                [
                    float(p.x)
                    for p in group
                ],
                dtype=np.float64,
            )

            ys = np.asarray(
                [
                    float(p.y)
                    for p in group
                ],
                dtype=np.float64,
            )

            if not (
                np.all(np.isfinite(xs))
                and np.all(np.isfinite(ys))
                and np.all(np.isfinite(weights))
            ):
                continue

            if np.sum(weights) <= 0.0:
                weights = np.ones_like(
                    weights
                )

            x_values.append(
                float(
                    np.average(
                        xs,
                        weights=weights,
                    )
                )
            )

            y_values.append(
                float(
                    np.mean(ys)
                )
            )

            confidence_values.append(
                float(
                    np.mean(weights)
                )
            )

        if len(x_values) < self.min_points:
            raise ValueError(
                "Pontos verticais insuficientes."
            )

        x = np.asarray(
            x_values,
            dtype=np.float64,
        )

        y = np.asarray(
            y_values,
            dtype=np.float64,
        )

        confidence = np.asarray(
            confidence_values,
            dtype=np.float64,
        )

        order = np.argsort(y)

        x = x[order]
        y = y[order]
        confidence = confidence[order]

        if not (
            np.all(np.isfinite(x))
            and np.all(np.isfinite(y))
            and np.all(np.isfinite(confidence))
        ):
            raise ValueError(
                "Pontos não finitos."
            )

        span = float(
            y[-1] - y[0]
        )

        if span <= 0.0:
            raise ValueError(
                "Extensão vertical inválida."
            )

        # Não rejeitamos artificialmente os testes com lanes curtas.
        # O limite configurável continua disponível.
        if span < self.min_vertical_span:
            raise ValueError(
                "Extensão vertical insuficiente."
            )

        return (
            x,
            y,
            confidence,
        )

    # =========================================================================
    # AJUSTE POLINOMIAL
    # =========================================================================

    def _fit_polynomial(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> tuple[
        np.ndarray,
        float,
        float,
        float,
        int,
    ]:

        degree = min(
            self.degree,
            len(x) - 1,
        )

        if degree < 1:
            raise ValueError(
                "Grau insuficiente."
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

        if (
            not math.isfinite(y_scale)
            or y_scale <= 1e-9
        ):
            raise ValueError(
                "Escala vertical inválida."
            )

        normalized_y = (
            y - y_center
        ) / y_scale

        coefficients = np.polyfit(
            normalized_y,
            x,
            degree,
        )

        if not np.all(
            np.isfinite(
                coefficients
            )
        ):
            raise ValueError(
                "Coeficientes inválidos."
            )

        predicted = np.polyval(
            coefficients,
            normalized_y,
        )

        residuals = (
            x - predicted
        )

        if not np.all(
            np.isfinite(
                residuals
            )
        ):
            raise ValueError(
                "Resíduos inválidos."
            )

        fit_error = float(
            np.sqrt(
                np.mean(
                    residuals ** 2
                )
            )
        )

        return (
            coefficients,
            fit_error,
            y_center,
            y_scale,
            degree,
        )

    # =========================================================================
    # CONVERSÃO PARA POLINÔMIO ABSOLUTO
    # =========================================================================

    @staticmethod
    def _absolute_coefficients(
        coefficients: np.ndarray,
        center: float,
        scale: float,
        degree: int,
    ) -> tuple[
        float,
        float,
        float,
        float,
    ]:

        if degree == 3:

            an, bn, cn, dn = coefficients

            a = an / scale**3

            b = (
                -3.0
                * an
                * center
                / scale**3
                + bn / scale**2
            )

            c = (
                3.0
                * an
                * center**2
                / scale**3
                - 2.0
                * bn
                * center
                / scale**2
                + cn / scale
            )

            d = (
                -an
                * center**3
                / scale**3
                + bn
                * center**2
                / scale**2
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

        if degree == 2:

            an, bn, cn = coefficients

            a = 0.0

            b = (
                an / scale**2
            )

            c = (
                -2.0
                * an
                * center
                / scale**2
                + bn / scale
            )

            d = (
                an
                * center**2
                / scale**2
                - bn
                * center
                / scale
                + cn
            )

            return (
                float(a),
                float(b),
                float(c),
                float(d),
            )

        if degree == 1:

            an, bn = coefficients

            return (
                0.0,
                0.0,
                float(an / scale),
                float(
                    bn
                    - an
                    * center
                    / scale
                ),
            )

        return (
            0.0,
            0.0,
            0.0,
            float(coefficients[0]),
        )

    # =========================================================================
    # CONFIANÇA
    # =========================================================================

    def _projection_confidence(
        self,
        point_count: int,
        vertical_span: float,
        fit_error: float,
        point_confidence: float,
    ) -> float:

        count_score = _clip01(
            point_count / 20.0
        )

        span_score = _clip01(
            vertical_span / 250.0
        )

        error_score = math.exp(
            -max(
                0.0,
                fit_error,
            ) / 20.0
        )

        confidence = (
            0.25 * count_score
            + 0.25 * span_score
            + 0.30 * error_score
            + 0.20 * _clip01(
                point_confidence
            )
        )

        return _clip01(
            confidence
        )

    @staticmethod
    def _quality(
        confidence: float,
    ) -> ProjectionQuality:

        confidence = _clip01(
            confidence
        )

        if confidence >= 0.80:
            return ProjectionQuality.HIGH

        if confidence >= 0.60:
            return ProjectionQuality.MEDIUM

        if confidence >= 0.40:
            return ProjectionQuality.LOW

        return ProjectionQuality.NONE

    # =========================================================================
    # POLINÔMIO → LANEPOINTS
    # =========================================================================

    @staticmethod
    def _evaluate(
        coefficients: np.ndarray,
        y: np.ndarray,
        center: float,
        scale: float,
    ) -> np.ndarray:

        normalized = (
            y - center
        ) / scale

        return np.polyval(
            coefficients,
            normalized,
        )

    # =========================================================================
    # CRIAÇÃO DA LANE POLYNOMIAL
    # =========================================================================

    def _create_polynomial(
        self,
        coefficients: np.ndarray,
        center: float,
        scale: float,
        degree: int,
        fit_error: float,
        confidence: float,
        y_min: float,
        y_max: float,
        sample_count: int,
    ) -> LanePolynomial:

        a, b, c, d = (
            self._absolute_coefficients(
                coefficients,
                center,
                scale,
                degree,
            )
        )

        return LanePolynomial(
            a=a,
            b=b,
            c=c,
            d=d,
            valid=True,
            fit_error=float(
                fit_error
            ),
            sample_count=int(
                sample_count
            ),
            confidence=float(
                confidence
            ),
            y_min=float(
                y_min
            ),
            y_max=float(
                y_max
            ),
        )

    # =========================================================================
    # PROJEÇÃO
    # =========================================================================

    def _project_points(
        self,
        points: Sequence[LanePoint],
    ) -> LaneProjection:

        try:

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
            ) = self._fit_polynomial(
                x,
                y,
            )

            if (
                not math.isfinite(
                    fit_error
                )
            ):
                raise ValueError(
                    "Erro de ajuste inválido."
                )

            if fit_error > self.max_fit_error:
                raise ValueError(
                    "Erro de ajuste acima do limite."
                )

            source_y_min = float(
                y[0]
            )

            source_y_max = float(
                y[-1]
            )

            span = (
                source_y_max
                - source_y_min
            )

            mean_confidence = _clip01(
                np.mean(
                    confidence_values
                )
            )

            confidence = (
                self._projection_confidence(
                    point_count=len(x),
                    vertical_span=span,
                    fit_error=fit_error,
                    point_confidence=(
                        mean_confidence
                    ),
                )
            )

            if confidence < self.min_confidence:
                raise ValueError(
                    "Confiança abaixo do limite."
                )

            quality = self._quality(
                confidence
            )

            if quality == ProjectionQuality.NONE:
                raise ValueError(
                    "Qualidade insuficiente."
                )

            # -----------------------------------------------------------------
            # Projeção.
            #
            # Mantemos toda a observação original e extrapolamos somente
            # para frente em Y.
            # -----------------------------------------------------------------

            projection_end = min(
                source_y_max
                + self.max_projection_distance,
                source_y_max
                + self.max_projection_distance,
            )

            distance = (
                projection_end
                - source_y_min
            )

            samples = max(
                2,
                int(
                    distance
                    / self.sample_step
                ) + 1,
            )

            samples = min(
                128,
                samples,
            )

            projected_y = np.linspace(
                source_y_min,
                projection_end,
                samples,
                dtype=np.float64,
            )

            projected_x = self._evaluate(
                coefficients,
                projected_y,
                y_center,
                y_scale,
            )

            if not (
                np.all(
                    np.isfinite(
                        projected_x
                    )
                )
                and np.all(
                    np.isfinite(
                        projected_y
                    )
                )
            ):
                raise ValueError(
                    "Projeção não finita."
                )

            # -----------------------------------------------------------------
            # Pontos projetados.
            # -----------------------------------------------------------------

            projected_points: list[
                LanePoint
            ] = []

            for px, py in zip(
                projected_x,
                projected_y,
            ):

                distance_from_observed = max(
                    0.0,
                    float(py)
                    - source_y_max,
                )

                decay = max(
                    0.0,
                    1.0
                    - (
                        distance_from_observed
                        / self.max_projection_distance
                    ),
                )

                point_confidence = _clip01(
                    confidence
                    * (
                        1.0
                        if distance_from_observed <= 0.0
                        else (
                            0.75
                            + 0.25 * decay
                        )
                    )
                )

                projected_points.append(
                    LanePoint(
                        x=float(px),
                        y=float(py),
                        confidence=point_confidence,
                        valid=True,
                    )
                )

            if len(
                projected_points
            ) < 2:
                raise ValueError(
                    "Pontos projetados insuficientes."
                )

            projected_points.sort(
                key=lambda p: p.y
            )

            polynomial = (
                self._create_polynomial(
                    coefficients=coefficients,
                    center=y_center,
                    scale=y_scale,
                    degree=degree,
                    fit_error=fit_error,
                    confidence=confidence,
                    y_min=source_y_min,
                    y_max=source_y_max,
                    sample_count=len(x),
                )
            )

            projection = LaneProjection(
                polynomial=polynomial,
                points=projected_points,
                quality=quality,
                extrapolated=(
                    projection_end
                    > source_y_max
                ),
                valid=True,
                horizon_y=source_y_min,
            )

            # Compatibilidade com consumidores que esperam
            # projection.confidence.
            projection.confidence = (
                confidence
            )

            return projection

        except Exception:
            return _empty_projection()

    # =========================================================================
    # MODEL → POINTS
    # =========================================================================

    @staticmethod
    def _model_points(
        model: LaneModel,
    ) -> list[LanePoint]:

        if not isinstance(
            model,
            LaneModel,
        ):
            return []

        try:
            line = model.line
        except Exception:
            return []

        if line is None:
            return []

        try:
            points = line.points
        except Exception:
            return []

        return list(points)

    # =========================================================================
    # PUBLIC PROJECT
    # =========================================================================

    def project(
        self,
        data: Any,
        frame_height: Optional[float] = None,
        *,
        horizon_y: Optional[float] = None,
    ) -> LaneProjection:

        del frame_height

        # ---------------------------------------------------------------------
        # LaneModel
        # ---------------------------------------------------------------------

        if isinstance(
            data,
            LaneModel,
        ):

            projection = self._project_points(
                self._model_points(
                    data
                )
            )

            if horizon_y is not None:
                value = _finite(
                    horizon_y
                )

                if value is not None:
                    projection.horizon_y = value

            return projection

        # ---------------------------------------------------------------------
        # Sequência de LanePoint
        # ---------------------------------------------------------------------

        if data is None:
            return _empty_projection()

        try:
            points = list(data)
        except (
            TypeError,
            ValueError,
        ):
            return _empty_projection()

        projection = self._project_points(
            points
        )

        if horizon_y is not None:
            value = _finite(
                horizon_y
            )

            if value is not None:
                projection.horizon_y = value

        return projection

    # =========================================================================
    # BATCH
    # =========================================================================

    def project_many(
        self,
        models: Sequence[Any],
        frame_height: Optional[float] = None,
        *,
        horizon_y: Optional[float] = None,
    ) -> list[LaneProjection]:

        return [
            self.project(
                model,
                frame_height,
                horizon_y=horizon_y,
            )
            for model in models
        ]


# =============================================================================
# FACTORY
# =============================================================================


def create_lane_projection_engine(
    **kwargs: Any,
) -> LaneProjectionEngine:

    return LaneProjectionEngine(
        **kwargs
    )


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    "LaneProjectionEngine",
    "create_lane_projection_engine",
]