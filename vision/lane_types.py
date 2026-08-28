"""
vision/lane_types.py

Forza Assistents
================

Contratos canônicos para representação de lanes.

Este módulo é a fonte única de verdade para os tipos básicos
utilizados pelo pipeline de percepção.

Princípios
----------
- tipos imutáveis sempre que possível;
- validação de invariantes na construção;
- nenhuma dependência de OpenCV;
- nenhuma dependência do detector;
- nenhuma lógica temporal;
- nenhuma lógica de controle;
- coordenadas sempre explicitamente documentadas;
- compatibilidade com CULane/UFLD quando necessária.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Iterable, Optional, Sequence, Tuple


# =============================================================================
# TIPOS BÁSICOS
# =============================================================================

Point = Tuple[float, float]

POLYNOMIAL_DEGREE = 3

# Anchors verticais utilizados pela representação CULane/UFLD.
#
# Os valores são normalizados em [0, 1], portanto não dependem da
# resolução do detector ou da tela.
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
    Origem da informação de uma lane.
    """

    DETECTED = "detected"
    TRACKED = "tracked"
    PROJECTED = "projected"
    INFERRED = "inferred"


class LaneSide(str, Enum):
    """
    Posição relativa da lane.

    UNKNOWN é utilizado quando a lateralidade não pode ser
    determinada de maneira confiável.
    """

    LEFT = "left"
    RIGHT = "right"
    UNKNOWN = "unknown"


# =============================================================================
# UTILITÁRIOS
# =============================================================================


def _finite_float(
    value: object,
    name: str,
) -> float:
    """
    Converte um valor para float e garante finitude.
    """

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a numeric value."
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
    """
    Valida confiança em [0, 1].
    """

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
    Ponto pertencente a uma lane.

    Coordenadas:

        x -> eixo horizontal da imagem
        y -> eixo vertical da imagem

    A convenção é a mesma da imagem digital:

        origem = canto superior esquerdo
        x cresce para a direita
        y cresce para baixo

    confidence e valid permanecem opcionais para preservar
    compatibilidade com consumidores antigos do pipeline.
    """

    x: float
    y: float
    confidence: float = 1.0
    valid: bool = True

    def __post_init__(self) -> None:

        x = _finite_float(
            self.x,
            "x",
        )

        y = _finite_float(
            self.y,
            "y",
        )

        confidence = _confidence(
            self.confidence
        )

        object.__setattr__(
            self,
            "x",
            x,
        )

        object.__setattr__(
            self,
            "y",
            y,
        )

        object.__setattr__(
            self,
            "confidence",
            confidence,
        )

        object.__setattr__(
            self,
            "valid",
            bool(self.valid),
        )

    def as_tuple(self) -> Point:
        """
        Retorna (x, y).
        """

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
    Representação de uma lane observada.

    lane_id:
        Identificador lógico da lane.

    points:
        Pontos observados em coordenadas de imagem.

    confidence:
        Confiança global da observação.

    source:
        Origem da informação.

    side:
        Lateralidade conhecida da lane.

    model:
        Modelo polinomial opcional associado à observação.
    """

    lane_id: int
    points: Tuple[LanePoint, ...]
    confidence: float = 1.0
    source: LaneSource = LaneSource.DETECTED
    side: LaneSide = LaneSide.UNKNOWN
    model: Optional["LanePolynomial"] = None

    def __post_init__(self) -> None:

        lane_id = int(self.lane_id)

        points = tuple(
            point
            if isinstance(
                point,
                LanePoint,
            )
            else LanePoint(
                x=point[0],
                y=point[1],
            )
            for point in self.points
        )

        confidence = _confidence(
            self.confidence
        )

        if not isinstance(
            self.source,
            LaneSource,
        ):
            source = LaneSource(
                self.source
            )
        else:
            source = self.source

        if not isinstance(
            self.side,
            LaneSide,
        ):
            side = LaneSide(
                self.side
            )
        else:
            side = self.side

        object.__setattr__(
            self,
            "lane_id",
            lane_id,
        )

        object.__setattr__(
            self,
            "points",
            points,
        )

        object.__setattr__(
            self,
            "confidence",
            confidence,
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
    def x_values(self) -> Tuple[float, ...]:
        return tuple(
            point.x
            for point in self.points
            if point.valid
        )

    @property
    def y_values(self) -> Tuple[float, ...]:
        return tuple(
            point.y
            for point in self.points
            if point.valid
        )

    @property
    def point_count(self) -> int:
        return len(self.points)

    @property
    def y_min(self) -> Optional[float]:
        if not self.y_values:
            return None
        return min(self.y_values)

    @property
    def y_max(self) -> Optional[float]:
        if not self.y_values:
            return None
        return max(self.y_values)

    @property
    def x_min(self) -> Optional[float]:
        if not self.x_values:
            return None
        return min(self.x_values)

    @property
    def x_max(self) -> Optional[float]:
        if not self.x_values:
            return None
        return max(self.x_values)

    @property
    def observed_span(self) -> float:
        if self.y_min is None or self.y_max is None:
            return 0.0

        return self.y_max - self.y_min


# =============================================================================
# LANE POLYNOMIAL
# =============================================================================


@dataclass(frozen=True, slots=True)
class LanePolynomial:
    """
    Modelo polinomial cúbico de uma lane.

    A representação é:

        x(y) = a*y³ + b*y² + c*y + d

    coefficients:

        (a, b, c, d)

    y_min / y_max definem explicitamente o domínio observado.

    Importante:
        este tipo representa um modelo matemático da observação.
        Ele não implica extrapolação.
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

        coefficients = tuple(
            _finite_float(
                value,
                "coefficient",
            )
            for value in self.coefficients
        )

        if len(coefficients) != 4:
            raise ValueError(
                "LanePolynomial requires exactly "
                "four coefficients."
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

        confidence = _confidence(
            self.confidence
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
            confidence,
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
        """
        Avalia x(y) utilizando Horner.
        """

        y = _finite_float(
            y,
            "y",
        )

        a, b, c, d = self.coefficients

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
        """
        Calcula dx/dy.
        """

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
        """
        Calcula d²x/dy².
        """

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
    Representação de uma lane projetada.

    Diferentemente de LaneLine, os pontos desta estrutura não são
    necessariamente observações diretas do detector.

    extrapolated_distance:
        distância além da região observada utilizada na projeção.
    """

    lane_id: int
    points: Tuple[LanePoint, ...]
    confidence: float
    extrapolated_distance: float
    source: LaneSource = LaneSource.PROJECTED

    def __post_init__(self) -> None:

        lane_id = int(
            self.lane_id
        )

        points = tuple(
            point
            if isinstance(
                point,
                LanePoint,
            )
            else LanePoint(
                x=point[0],
                y=point[1],
            )
            for point in self.points
        )

        confidence = _confidence(
            self.confidence
        )

        distance = _finite_float(
            self.extrapolated_distance,
            "extrapolated_distance",
        )

        if distance < 0.0:
            raise ValueError(
                "extrapolated_distance must be >= 0."
            )

        if self.source not in (
            LaneSource.PROJECTED,
            LaneSource.INFERRED,
        ):
            raise ValueError(
                "LaneProjection source must be "
                "PROJECTED or INFERRED."
            )

        object.__setattr__(
            self,
            "lane_id",
            lane_id,
        )

        object.__setattr__(
            self,
            "points",
            points,
        )

        object.__setattr__(
            self,
            "confidence",
            confidence,
        )

        object.__setattr__(
            self,
            "extrapolated_distance",
            distance,
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
# CONVERSÕES
# =============================================================================


def points_from_xy(
    points: Iterable[
        Sequence[float]
    ],
    *,
    confidence: float = 1.0,
) -> Tuple[LanePoint, ...]:
    """
    Converte pares (x, y) em LanePoint.

    Aceita qualquer iterable de sequências com pelo menos
    dois elementos.
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
                f"Point at index {index} "
                "must contain at least x and y."
            )

        result.append(
            LanePoint(
                x=point[0],
                y=point[1],
                confidence=confidence,
            )
        )

    return tuple(
        result
    )


def points_to_xy(
    points: Iterable[LanePoint],
) -> Tuple[Point, ...]:
    """
    Converte LanePoint em pares (x, y).
    """

    result = []

    for point in points:

        if not isinstance(
            point,
            LanePoint,
        ):
            raise TypeError(
                "points_to_xy expects LanePoint objects."
            )

        if not point.valid:
            continue

        result.append(
            point.as_tuple()
        )

    return tuple(
        result
    )


# =============================================================================
# POLINÔMIO
# =============================================================================


def polynomial_from_points(
    points: Sequence[LanePoint],
    *,
    degree: int = POLYNOMIAL_DEGREE,
    confidence: Optional[float] = None,
) -> LanePolynomial:
    """
    Ajusta um polinômio x(y) aos pontos.

    O grau máximo suportado pela API canônica é cúbico.
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
            "At least four valid points are required "
            "for a cubic polynomial."
        )

    x = [
        point.x
        for point in valid_points
    ]

    y = [
        point.y
        for point in valid_points
    ]

    # Import local para manter lane_types independente de numpy
    # quando usado apenas como módulo de contratos.
    import numpy as np

    coefficients = np.polyfit(
        np.asarray(y, dtype=np.float64),
        np.asarray(x, dtype=np.float64),
        degree,
    )

    values = tuple(
        float(value)
        for value in coefficients
    )

    if confidence is None:
        confidence = sum(
            point.confidence
            for point in valid_points
        ) / len(valid_points)

    return LanePolynomial(
        coefficients=values,  # type: ignore[arg-type]
        y_min=min(y),
        y_max=max(y),
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
    "points_from_xy",
    "points_to_xy",
    "polynomial_from_points",
]