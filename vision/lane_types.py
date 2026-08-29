"""
vision/lane_types.py

Forza Assistents
================

Contratos canônicos do domínio de lanes.

Este módulo é a única fonte de verdade para:

    LanePoint
    LaneLine
    LanePolynomial
    LaneProjection
    LaneDetectionResult

Nenhum detector, tracker ou módulo geométrico deve criar versões
alternativas desses tipos.

Princípios
----------
- contratos simples e estáveis;
- validação na construção;
- imutabilidade;
- independência de OpenCV/PyTorch/YOLOP;
- coordenadas explicitamente definidas;
- compatibilidade controlada com a arquitetura existente.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


# =============================================================================
# CONSTANTES
# =============================================================================

Point = Tuple[float, float]

POLYNOMIAL_DEGREE = 3

CULANE_ROW_ANCHORS: Tuple[float, ...] = (
    0.42,
    0.45,
    0.48,
    0.51,
    0.54,
    0.57,
    0.60,
    0.63,
    0.66,
    0.69,
    0.72,
    0.75,
    0.78,
    0.81,
    0.84,
    0.87,
    0.90,
    0.93,
)


# =============================================================================
# ENUMS
# =============================================================================


class LaneSource(str, Enum):
    """
    Origem da informação da lane.
    """

    DETECTED = "detected"
    TRACKED = "tracked"
    PROJECTED = "projected"
    INFERRED = "inferred"


class LaneSide(str, Enum):
    """
    Lateralidade da lane.
    """

    LEFT = "left"
    RIGHT = "right"
    UNKNOWN = "unknown"


# =============================================================================
# VALIDADORES
# =============================================================================


def _finite_float(
    value: object,
    name: str,
) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be numeric."
        ) from exc

    if not math.isfinite(result):
        raise ValueError(
            f"{name} must be finite."
        )

    return result


def _confidence(
    value: object,
    name: str = "confidence",
) -> float:

    result = _finite_float(
        value,
        name,
    )

    if not 0.0 <= result <= 1.0:
        raise ValueError(
            f"{name} must be in [0, 1]."
        )

    return result


# =============================================================================
# LANE POINT
# =============================================================================


@dataclass(frozen=True, slots=True)
class LanePoint:
    """
    Ponto individual de uma lane.

    Coordenadas:

        x = horizontal
        y = vertical

    Convenção de imagem:

        origem no canto superior esquerdo
        x cresce para a direita
        y cresce para baixo
    """

    x: float
    y: float
    confidence: float = 1.0
    valid: bool = True

    def __post_init__(self) -> None:

        object.__setattr__(
            self,
            "x",
            _finite_float(
                self.x,
                "x",
            ),
        )

        object.__setattr__(
            self,
            "y",
            _finite_float(
                self.y,
                "y",
            ),
        )

        object.__setattr__(
            self,
            "confidence",
            _confidence(
                self.confidence
            ),
        )

        object.__setattr__(
            self,
            "valid",
            bool(self.valid),
        )

    def as_tuple(self) -> Point:
        return (
            self.x,
            self.y,
        )


# =============================================================================
# LANE LINE
# =============================================================================


@dataclass(frozen=True, slots=True)
class LaneLine:
    """
    Lane individual detectada ou rastreada.
    """

    lane_id: int
    points: Tuple[LanePoint, ...]
    confidence: float = 1.0
    source: LaneSource = LaneSource.DETECTED
    side: LaneSide = LaneSide.UNKNOWN
    model: Optional["LanePolynomial"] = None

    def __post_init__(self) -> None:

        normalized_points = []

        for point in self.points:

            if isinstance(
                point,
                LanePoint,
            ):
                normalized_points.append(
                    point
                )
                continue

            if (
                isinstance(point, Sequence)
                and len(point) >= 2
            ):
                normalized_points.append(
                    LanePoint(
                        x=point[0],
                        y=point[1],
                    )
                )
                continue

            raise TypeError(
                "LaneLine.points must contain "
                "LanePoint objects or (x, y) pairs."
            )

        try:
            source = (
                self.source
                if isinstance(
                    self.source,
                    LaneSource,
                )
                else LaneSource(
                    self.source
                )
            )
        except ValueError as exc:
            raise ValueError(
                f"Invalid LaneSource: {self.source!r}"
            ) from exc

        try:
            side = (
                self.side
                if isinstance(
                    self.side,
                    LaneSide,
                )
                else LaneSide(
                    self.side
                )
            )
        except ValueError as exc:
            raise ValueError(
                f"Invalid LaneSide: {self.side!r}"
            ) from exc

        object.__setattr__(
            self,
            "lane_id",
            int(self.lane_id),
        )

        object.__setattr__(
            self,
            "points",
            tuple(normalized_points),
        )

        object.__setattr__(
            self,
            "confidence",
            _confidence(
                self.confidence
            ),
        )

        object.__setattr__(
            self,
            "source",
            source,
        )

        object.__setattr__(
            self,
            "side",
            side,
        )

    @property
    def point_count(self) -> int:
        return len(self.points)

    @property
    def valid_points(
        self,
    ) -> Tuple[LanePoint, ...]:

        return tuple(
            point
            for point in self.points
            if point.valid
        )

    @property
    def x_values(self) -> Tuple[float, ...]:

        return tuple(
            point.x
            for point in self.valid_points
        )

    @property
    def y_values(self) -> Tuple[float, ...]:

        return tuple(
            point.y
            for point in self.valid_points
        )

    @property
    def x_min(self) -> Optional[float]:

        if not self.x_values:
            return None

        return min(
            self.x_values
        )

    @property
    def x_max(self) -> Optional[float]:

        if not self.x_values:
            return None

        return max(
            self.x_values
        )

    @property
    def y_min(self) -> Optional[float]:

        if not self.y_values:
            return None

        return min(
            self.y_values
        )

    @property
    def y_max(self) -> Optional[float]:

        if not self.y_values:
            return None

        return max(
            self.y_values
        )

    @property
    def observed_span(self) -> float:

        if (
            self.y_min is None
            or self.y_max is None
        ):
            return 0.0

        return (
            self.y_max
            - self.y_min
        )


# =============================================================================
# LANE POLYNOMIAL
# =============================================================================


@dataclass(frozen=True, slots=True)
class LanePolynomial:
    """
    Modelo cúbico:

        x(y) = a*y³ + b*y² + c*y + d
    """

    coefficients: Tuple[
        float,
        float,
        float,
        float,
    ]

    y_min: float
    y_max: float

    confidence: float = 1.0

    def __post_init__(self) -> None:

        if len(self.coefficients) != 4:
            raise ValueError(
                "LanePolynomial requires exactly "
                "four coefficients."
            )

        coefficients = tuple(
            _finite_float(
                value,
                "coefficient",
            )
            for value in self.coefficients
        )

        y_min = _finite_float(
            self.y_min,
            "y_min",
        )

        y_max = _finite_float(
            self.y_max,
            "y_max",
        )

        if y_max <= y_min:
            raise ValueError(
                "y_max must be greater than y_min."
            )

        object.__setattr__(
            self,
            "coefficients",
            coefficients,
        )

        object.__setattr__(
            self,
            "y_min",
            y_min,
        )

        object.__setattr__(
            self,
            "y_max",
            y_max,
        )

        object.__setattr__(
            self,
            "confidence",
            _confidence(
                self.confidence
            ),
        )

    @property
    def a(self) -> float:
        return self.coefficients[0]

    @property
    def b(self) -> float:
        return self.coefficients[1]

    @property
    def c(self) -> float:
        return self.coefficients[2]

    @property
    def d(self) -> float:
        return self.coefficients[3]

    def evaluate(
        self,
        y: float,
    ) -> float:

        y = _finite_float(
            y,
            "y",
        )

        a, b, c, d = (
            self.coefficients
        )

        return (
            (
                (
                    a * y
                    + b
                )
                * y
                + c
            )
            * y
            + d
        )

    def slope(
        self,
        y: float,
    ) -> float:

        y = _finite_float(
            y,
            "y",
        )

        return (
            3.0 * self.a * y * y
            + 2.0 * self.b * y
            + self.c
        )

    def second_derivative(
        self,
        y: float,
    ) -> float:

        y = _finite_float(
            y,
            "y",
        )

        return (
            6.0 * self.a * y
            + 2.0 * self.b
        )

    def contains_y(
        self,
        y: float,
    ) -> bool:

        y = _finite_float(
            y,
            "y",
        )

        return (
            self.y_min
            <= y
            <= self.y_max
        )


# =============================================================================
# LANE PROJECTION
# =============================================================================


@dataclass(frozen=True, slots=True)
class LaneProjection:
    """
    Lane extrapolada/projetada.
    """

    lane_id: int
    points: Tuple[LanePoint, ...]
    confidence: float
    extrapolated_distance: float
    source: LaneSource = LaneSource.PROJECTED

    def __post_init__(self) -> None:

        normalized_points = []

        for point in self.points:

            if isinstance(
                point,
                LanePoint,
            ):
                normalized_points.append(
                    point
                )
                continue

            if (
                isinstance(point, Sequence)
                and len(point) >= 2
            ):
                normalized_points.append(
                    LanePoint(
                        x=point[0],
                        y=point[1],
                    )
                )
                continue

            raise TypeError(
                "LaneProjection.points must contain "
                "LanePoint objects or (x, y) pairs."
            )

        source = (
            self.source
            if isinstance(
                self.source,
                LaneSource,
            )
            else LaneSource(
                self.source
            )
        )

        if source not in (
            LaneSource.PROJECTED,
            LaneSource.INFERRED,
        ):
            raise ValueError(
                "LaneProjection source must be "
                "PROJECTED or INFERRED."
            )

        distance = _finite_float(
            self.extrapolated_distance,
            "extrapolated_distance",
        )

        if distance < 0.0:
            raise ValueError(
                "extrapolated_distance must be >= 0."
            )

        object.__setattr__(
            self,
            "lane_id",
            int(self.lane_id),
        )

        object.__setattr__(
            self,
            "points",
            tuple(normalized_points),
        )

        object.__setattr__(
            self,
            "confidence",
            _confidence(
                self.confidence
            ),
        )

        object.__setattr__(
            self,
            "extrapolated_distance",
            distance,
        )

        object.__setattr__(
            self,
            "source",
            source,
        )

    @property
    def point_count(self) -> int:
        return len(self.points)

    @property
    def y_min(self) -> Optional[float]:

        if not self.points:
            return None

        return min(
            point.y
            for point in self.points
        )

    @property
    def y_max(self) -> Optional[float]:

        if not self.points:
            return None

        return max(
            point.y
            for point in self.points
        )


# =============================================================================
# LANE DETECTION RESULT
# =============================================================================


@dataclass(frozen=True, slots=True)
class LaneDetectionResult:
    """
    Resultado bruto da detecção de lanes.

    `lanes` aceita:

        - LaneLine
        - sequência de LanePoint

    Isso permite que o detector permaneça simples e que a geometria
    trabalhe sobre um contrato único.

    image_width / image_height:
        dimensões do frame de origem.

    frame_width / frame_height:
        aliases canônicos para consumidores que utilizam nomenclatura
        temporal/frame-based.

    valid:
        indica se o resultado global da detecção é utilizável.
    """

    lanes: Tuple[
        LaneLine | Tuple[LanePoint, ...],
        ...
    ]

    confidence: float = 0.0

    image_width: int = 0
    image_height: int = 0

    inference_ms: float = 0.0

    valid: bool = True

    def __post_init__(self) -> None:

        normalized = []

        for lane in self.lanes:

            if isinstance(
                lane,
                LaneLine,
            ):
                normalized.append(
                    lane
                )
                continue

            if isinstance(
                lane,
                Sequence,
            ):
                points = tuple(
                    point
                    if isinstance(
                        point,
                        LanePoint,
                    )
                    else LanePoint(
                        x=point[0],
                        y=point[1],
                        confidence=(
                            point[2]
                            if len(point) > 2
                            else 1.0
                        ),
                        valid=(
                            point[3]
                            if len(point) > 3
                            else True
                        ),
                    )
                    for point in lane
                )

                normalized.append(
                    points
                )
                continue

            raise TypeError(
                "lanes must contain LaneLine objects "
                "or sequences of LanePoint."
            )

        width = int(
            self.image_width
        )

        height = int(
            self.image_height
        )

        if width < 0:
            raise ValueError(
                "image_width must be >= 0."
            )

        if height < 0:
            raise ValueError(
                "image_height must be >= 0."
            )

        inference_ms = _finite_float(
            self.inference_ms,
            "inference_ms",
        )

        if inference_ms < 0.0:
            raise ValueError(
                "inference_ms must be >= 0."
            )

        object.__setattr__(
            self,
            "lanes",
            tuple(normalized),
        )

        object.__setattr__(
            self,
            "confidence",
            _confidence(
                self.confidence
            ),
        )

        object.__setattr__(
            self,
            "image_width",
            width,
        )

        object.__setattr__(
            self,
            "image_height",
            height,
        )

        object.__setattr__(
            self,
            "inference_ms",
            inference_ms,
        )

        object.__setattr__(
            self,
            "valid",
            bool(self.valid),
        )

    @property
    def frame_width(self) -> int:
        return self.image_width

    @property
    def frame_height(self) -> int:
        return self.image_height

    @property
    def lane_count(self) -> int:
        return len(self.lanes)

    @property
    def has_lanes(self) -> bool:
        return bool(
            self.lanes
        )


# =============================================================================
# CONVERSÕES
# =============================================================================


def points_from_xy(
    points: Iterable[
        Sequence[float]
    ],
    *,
    confidence: float = 1.0,
    valid: bool = True,
) -> Tuple[LanePoint, ...]:
    """
    Converte pares (x, y) em LanePoint.
    """

    confidence = _confidence(
        confidence
    )

    result = []

    for index, point in enumerate(
        points
    ):

        if len(point) < 2:
            raise ValueError(
                f"Point at index {index} must "
                "contain x and y."
            )

        result.append(
            LanePoint(
                x=point[0],
                y=point[1],
                confidence=confidence,
                valid=valid,
            )
        )

    return tuple(result)


def points_to_xy(
    points: Iterable[LanePoint],
) -> Tuple[Point, ...]:
    """
    Converte LanePoint em (x, y).
    """

    result = []

    for point in points:

        if not isinstance(
            point,
            LanePoint,
        ):
            raise TypeError(
                "points_to_xy expects LanePoint."
            )

        if point.valid:
            result.append(
                point.as_tuple()
            )

    return tuple(result)


# =============================================================================
# AJUSTE POLINOMIAL
# =============================================================================


def polynomial_from_points(
    points: Sequence[LanePoint],
    *,
    degree: int = POLYNOMIAL_DEGREE,
    confidence: Optional[float] = None,
) -> LanePolynomial:
    """
    Ajusta x(y) aos pontos observados.
    """

    if degree != POLYNOMIAL_DEGREE:
        raise ValueError(
            "Only cubic polynomials are supported."
        )

    valid_points = [
        point
        for point in points
        if isinstance(
            point,
            LanePoint,
        )
        and point.valid
    ]

    if len(valid_points) < (
        degree + 1
    ):
        raise ValueError(
            "At least four valid points are required."
        )

    x = np.asarray(
        [
            point.x
            for point in valid_points
        ],
        dtype=np.float64,
    )

    y = np.asarray(
        [
            point.y
            for point in valid_points
        ],
        dtype=np.float64,
    )

    if not (
        np.all(
            np.isfinite(x)
        )
        and np.all(
            np.isfinite(y)
        )
    ):
        raise ValueError(
            "Points must contain finite coordinates."
        )

    if np.ptp(y) <= 0.0:
        raise ValueError(
            "Points must span a non-zero Y range."
        )

    coefficients = np.polyfit(
        y,
        x,
        degree,
    )

    if confidence is None:
        confidence = sum(
            point.confidence
            for point in valid_points
        ) / len(valid_points)

    return LanePolynomial(
        coefficients=tuple(
            float(value)
            for value in coefficients
        ),  # type: ignore[arg-type]
        y_min=float(
            np.min(y)
        ),
        y_max=float(
            np.max(y)
        ),
        confidence=confidence,
    )


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    "Point",
    "POLYNOMIAL_DEGREE",
    "CULANE_ROW_ANCHORS",
    "LaneSource",
    "LaneSide",
    "LanePoint",
    "LaneLine",
    "LanePolynomial",
    "LaneProjection",
    "LaneDetectionResult",
    "points_from_xy",
    "points_to_xy",
    "polynomial_from_points",
]