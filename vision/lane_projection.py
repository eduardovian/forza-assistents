"""
vision/lane_projection.py

Projeção matemática das linhas de faixa.

Responsabilidades
-----------------
LanePoint
    ↓
filtragem
    ↓
LaneModel
    ↓
LanePolynomial
    ↓
projeção
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

from vision.lane_model import fit_lane_model


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


# =============================================================================
# ENGINE
# =============================================================================


class LaneProjectionEngine:
    """
    Engine de projeção.

    Compatível com a API atual dos testes e com LaneModel.

    Pode receber:

        project(list[LanePoint])

    ou:

        project(LaneModel)

    O modelo matemático continua sendo o LanePolynomial cúbico
    produzido por lane_model.py.
    """

    def __init__(
        self,
        min_points: int = 6,
        degree: int = 3,
        max_projection_distance: float = 300.0,
        min_confidence: float = 0.35,
    ) -> None:

        self.min_points = int(min_points)
        self.degree = int(degree)
        self.max_projection_distance = float(
            max_projection_distance
        )
        self.min_confidence = _clip01(
            min_confidence
        )

        if self.min_points < 4:
            raise ValueError(
                "min_points deve ser >= 4."
            )

        if self.degree < 1:
            raise ValueError(
                "degree deve ser >= 1."
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

    # =========================================================================
    # POINT VALIDATION
    # =========================================================================

    @staticmethod
    def _valid_points(
        points: Sequence[LanePoint],
    ) -> list[LanePoint]:

        result: list[LanePoint] = []

        for point in points:

            if not isinstance(
                point,
                LanePoint,
            ):
                continue

            try:
                if not point.is_valid():
                    continue
            except Exception:
                continue

            if not (
                math.isfinite(float(point.x))
                and math.isfinite(float(point.y))
                and math.isfinite(float(point.confidence))
            ):
                continue

            result.append(
                LanePoint(
                    x=float(point.x),
                    y=float(point.y),
                    confidence=_clip01(
                        point.confidence
                    ),
                    valid=True,
                )
            )

        result.sort(
            key=lambda point: point.y
        )

        return result

    # =========================================================================
    # MODEL CREATION
    # =========================================================================

    def _build_model(
        self,
        points: Sequence[LanePoint],
    ) -> Optional[LaneModel]:

        if len(points) < self.min_points:
            return None

        return fit_lane_model(
            points,
            min_points=self.min_points,
        )

    # =========================================================================
    # MODEL VALIDATION
    # =========================================================================

    @staticmethod
    def _validate_model(
        model: Any,
    ) -> bool:

        if not isinstance(
            model,
            LaneModel,
        ):
            return False

        try:
            if not model.is_valid():
                return False
        except Exception:
            return False

        polynomial = model.polynomial

        if not isinstance(
            polynomial,
            LanePolynomial,
        ):
            return False

        if not polynomial.valid:
            return False

        if not polynomial.is_finite():
            return False

        return True

    # =========================================================================
    # OBSERVED RANGE
    # =========================================================================

    @staticmethod
    def _observed_range(
        points: Sequence[LanePoint],
        polynomial: LanePolynomial,
    ) -> Optional[tuple[float, float]]:

        y_min = _finite(
            polynomial.y_min
        )

        y_max = _finite(
            polynomial.y_max
        )

        if (
            y_min is not None
            and y_max is not None
            and y_max > y_min
        ):
            return y_min, y_max

        if len(points) < 2:
            return None

        ys = [
            float(point.y)
            for point in points
            if math.isfinite(float(point.y))
        ]

        if len(ys) < 2:
            return None

        y_min = min(ys)
        y_max = max(ys)

        if y_max <= y_min:
            return None

        return y_min, y_max

    # =========================================================================
    # PROJECTION
    # =========================================================================

    def _project_model(
        self,
        model: LaneModel,
        points: Sequence[LanePoint],
    ) -> LaneProjection:

        if not self._validate_model(model):

            return LaneProjection(
                valid=False,
                quality=ProjectionQuality.NONE,
            )

        polynomial = model.polynomial

        if polynomial is None:

            return LaneProjection(
                valid=False,
                quality=ProjectionQuality.NONE,
            )

        confidence = _clip01(
            polynomial.confidence
        )

        if confidence < self.min_confidence:

            return LaneProjection(
                polynomial=polynomial,
                valid=False,
                quality=ProjectionQuality.NONE,
            )

        observed = self._observed_range(
            points,
            polynomial,
        )

        if observed is None:

            return LaneProjection(
                polynomial=polynomial,
                valid=False,
                quality=ProjectionQuality.NONE,
            )

        y_min, y_max = observed

        projection_end = (
            y_max
            + self.max_projection_distance
        )

        # A projeção deve representar a região observada
        # mais a extrapolação limitada.
        samples = max(
            2,
            min(
                64,
                int(
                    self.max_projection_distance / 10.0
                ) + 2,
            ),
        )

        ys = np.linspace(
            y_min,
            projection_end,
            samples,
            dtype=np.float64,
        )

        projected: list[LanePoint] = []

        for y in ys:

            y_value = float(y)

            try:
                x_value = float(
                    polynomial.evaluate(
                        y_value
                    )
                )
            except (
                TypeError,
                ValueError,
                OverflowError,
            ):
                continue

            if not (
                math.isfinite(x_value)
                and math.isfinite(y_value)
            ):
                continue

            distance = max(
                0.0,
                y_value - y_max,
            )

            # Degradação suave da confiança fora
            # do intervalo observado.
            decay = max(
                0.0,
                1.0
                - (
                    distance
                    / self.max_projection_distance
                ),
            )

            point_confidence = _clip01(
                confidence * (
                    1.0
                    if distance <= 0.0
                    else decay
                )
            )

            projected.append(
                LanePoint(
                    x=x_value,
                    y=y_value,
                    confidence=point_confidence,
                    valid=True,
                )
            )

        if len(projected) < 2:

            return LaneProjection(
                polynomial=polynomial,
                points=projected,
                valid=False,
                quality=ProjectionQuality.NONE,
                extrapolated=False,
            )

        # Confiança da projeção.
        extrapolated_distance = (
            projection_end - y_max
        )

        projection_confidence = _clip01(
            confidence
            * (
                1.0
                - 0.25
                * (
                    extrapolated_distance
                    / self.max_projection_distance
                )
            )
        )

        if projection_confidence >= 0.85:
            quality = ProjectionQuality.HIGH

        elif projection_confidence >= 0.65:
            quality = ProjectionQuality.MEDIUM

        elif projection_confidence >= 0.40:
            quality = ProjectionQuality.LOW

        else:
            quality = ProjectionQuality.NONE

        valid = (
            quality != ProjectionQuality.NONE
        )

        projection = LaneProjection(
            polynomial=polynomial,
            points=projected,
            quality=quality,
            extrapolated=(
                projection_end > y_max
            ),
            valid=valid,
            horizon_y=y_min,
        )

        # LaneProjection da versão atual de lane_types.py
        # não possui confidence explicitamente no dataclass.
        # Mantemos o atributo para compatibilidade com a API
        # pública esperada pelos testes.
        projection.confidence = (
            projection_confidence
        )

        return projection

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
        """
        Projeta uma lane.

        Aceita:

            list[LanePoint]
            LaneModel
        """

        # -------------------------------------------------------------
        # LaneModel
        # -------------------------------------------------------------

        if isinstance(
            data,
            LaneModel,
        ):

            points = []

            if data.line is not None:
                points = self._valid_points(
                    data.line.points
                )

            projection = self._project_model(
                data,
                points,
            )

            if horizon_y is not None:
                projection.horizon_y = (
                    float(horizon_y)
                )

            return projection

        # -------------------------------------------------------------
        # LanePoint sequence
        # -------------------------------------------------------------

        if data is None:

            return LaneProjection(
                valid=False,
                quality=ProjectionQuality.NONE,
            )

        try:
            points = self._valid_points(
                list(data)
            )
        except (
            TypeError,
            ValueError,
        ):

            return LaneProjection(
                valid=False,
                quality=ProjectionQuality.NONE,
            )

        if len(points) < self.min_points:

            return LaneProjection(
                valid=False,
                quality=ProjectionQuality.NONE,
            )

        model = self._build_model(
            points
        )

        if model is None:

            return LaneProjection(
                valid=False,
                quality=ProjectionQuality.NONE,
            )

        projection = self._project_model(
            model,
            points,
        )

        if horizon_y is not None:
            projection.horizon_y = (
                float(horizon_y)
            )

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