"""
vision/lane_types.py

Tipos fundamentais do sistema de visão de faixas.

Responsabilidade:
    Definir enums e estruturas de dados compartilhadas
    entre detector, modelagem, projeção e tracking.

Este módulo NÃO executa:
    - inferência;
    - fitting;
    - projeção;
    - tracking;
    - decisões ADAS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import List, Optional


# =============================================================================
# ENUMS
# =============================================================================


class LaneQuality(Enum):
    """
    Qualidade estrutural de uma faixa.
    """

    NONE = "none"
    POOR = "poor"
    PARTIAL = "partial"
    GOOD = "good"
    EXCELLENT = "excellent"


class ProjectionQuality(Enum):
    """
    Qualidade de uma projeção matemática.
    """

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# =============================================================================
# LANE POINT
# =============================================================================


@dataclass
class LanePoint:
    """
    Ponto individual de uma linha de faixa.
    """

    x: float
    y: float
    confidence: float
    valid: bool = True

    def is_finite(self) -> bool:
        return (
            math.isfinite(float(self.x))
            and math.isfinite(float(self.y))
            and math.isfinite(float(self.confidence))
        )

    def is_valid(self) -> bool:
        return (
            bool(self.valid)
            and self.is_finite()
        )


# =============================================================================
# LANE POLYNOMIAL
# =============================================================================


@dataclass
class LanePolynomial:
    """
    Modelo cúbico:

        x(y) = a*y³ + b*y² + c*y + d
    """

    a: float = 0.0
    b: float = 0.0
    c: float = 0.0
    d: float = 0.0

    valid: bool = False

    sample_count: int = 0

    y_min: float = 0.0
    y_max: float = 0.0

    fit_error: float = float("inf")

    confidence: float = 0.0

    def evaluate(self, y: float) -> float:
        y = float(y)

        return float(
            self.a * y ** 3
            + self.b * y ** 2
            + self.c * y
            + self.d
        )

    def derivative(self, y: float) -> float:
        y = float(y)

        return float(
            3.0 * self.a * y ** 2
            + 2.0 * self.b * y
            + self.c
        )

    def second_derivative(self, y: float) -> float:
        y = float(y)

        return float(
            6.0 * self.a * y
            + 2.0 * self.b
        )

    def is_finite(self) -> bool:
        return all(
            math.isfinite(float(value))
            for value in (
                self.a,
                self.b,
                self.c,
                self.d,
                self.y_min,
                self.y_max,
                self.fit_error,
                self.confidence,
            )
        )

    def is_valid(self) -> bool:
        return (
            bool(self.valid)
            and self.sample_count >= 0
            and self.is_finite()
            and 0.0 <= self.confidence <= 1.0
            and self.y_max >= self.y_min
        )


# =============================================================================
# LANE LINE
# =============================================================================


@dataclass
class LaneLine:
    """
    Linha de faixa detectada.
    """

    lane_id: int

    points: List[LanePoint] = field(
        default_factory=list
    )

    confidence: float = 0.0

    quality: LaneQuality = LaneQuality.NONE

    detected_directly: bool = True

    projected: bool = False

    valid: bool = False

    age_frames: int = 0

    missed_frames: int = 0

    def point_count(self) -> int:
        return len(self.points)

    def valid_points(self) -> List[LanePoint]:
        return [
            point
            for point in self.points
            if point.is_valid()
        ]

    def valid_point_count(self) -> int:
        return len(
            self.valid_points()
        )

    def is_finite(self) -> bool:
        if not math.isfinite(
            float(self.confidence)
        ):
            return False

        return all(
            point.is_finite()
            for point in self.points
        )

    def is_valid(self) -> bool:
        return (
            bool(self.valid)
            and self.point_count() >= 1
            and self.is_finite()
            and 0.0 <= self.confidence <= 1.0
        )


# =============================================================================
# LANE PROJECTION
# =============================================================================


@dataclass
class LaneProjection:
    """
    Projeção matemática de uma faixa.
    """

    polynomial: Optional[LanePolynomial] = None

    points: List[LanePoint] = field(
        default_factory=list
    )

    quality: ProjectionQuality = (
        ProjectionQuality.NONE
    )

    extrapolated: bool = False

    valid: bool = False

    horizon_y: Optional[float] = None

    def point_count(self) -> int:
        return len(self.points)

    def is_finite(self) -> bool:

        if self.horizon_y is not None:
            if not math.isfinite(
                float(self.horizon_y)
            ):
                return False

        if self.polynomial is not None:
            if not self.polynomial.is_finite():
                return False

        return all(
            point.is_finite()
            for point in self.points
        )

    def is_valid(self) -> bool:
        return (
            bool(self.valid)
            and self.polynomial is not None
            and self.polynomial.valid
            and self.point_count() >= 2
            and self.is_finite()
        )


# =============================================================================
# LANE MODEL
# =============================================================================


@dataclass
class LaneModel:
    """
    Modelo matemático completo de uma faixa.
    """

    lane_id: int

    line: LaneLine

    polynomial: Optional[LanePolynomial] = None

    projection: Optional[LaneProjection] = None

    tracked: bool = False

    stable: bool = False

    valid: bool = False

    def point_count(self) -> int:
        if self.line is None:
            return 0

        return self.line.point_count()

    def is_finite(self) -> bool:

        if self.line is None:
            return False

        if not self.line.is_finite():
            return False

        if self.polynomial is not None:
            if not self.polynomial.is_finite():
                return False

        if self.projection is not None:
            if not self.projection.is_finite():
                return False

        return True

    def is_valid(self) -> bool:

        if not self.valid:
            return False

        if self.line is None:
            return False

        if not self.line.valid:
            return False

        if self.polynomial is None:
            return False

        if not self.polynomial.valid:
            return False

        return self.is_finite()


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    "LaneQuality",
    "ProjectionQuality",
    "LanePoint",
    "LanePolynomial",
    "LaneLine",
    "LaneProjection",
    "LaneModel",
]