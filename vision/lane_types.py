"""
vision/lane_types.py

Forza Assistents
================

Contratos canônicos de dados para percepção de faixas.

Este módulo é a única fonte de verdade para os tipos de dados
compartilhados entre:

    YOLOPv2
        ↓
    LaneTracker
        ↓
    LaneGeometry
        ↓
    LaneModel
        ↓
    LaneProjection
        ↓
    LaneAssignment
        ↓
    ADAS

Princípios:
    - contratos explícitos;
    - objetos imutáveis nas fronteiras dos módulos;
    - validação rigorosa;
    - somente valores numéricos finitos;
    - modelo de lane exclusivamente cúbico;
    - nenhuma dependência de OpenCV, PyTorch ou detector específico.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Iterable, Optional, Sequence, Tuple


# =============================================================================
# TIPOS BÁSICOS
# =============================================================================

Point = Tuple[float, float]


# =============================================================================
# ORIGEM DA LANE
# =============================================================================

class LaneSource(str, Enum):
    """
    Origem atual dos dados da lane.

    DETECTED:
        Diretamente observada pelo detector.

    TRACKED:
        Observação mantida/refinada pelo rastreador temporal.

    PROJECTED:
        Dados obtidos por projeção/extrapolação de um modelo anterior.

    FUSED:
        Dados resultantes da combinação de múltiplas fontes.
    """

    DETECTED = "detected"
    TRACKED = "tracked"
    PROJECTED = "projected"
    FUSED = "fused"


# =============================================================================
# LANE POINT
# =============================================================================

@dataclass(frozen=True, slots=True)
class LanePoint:
    """
    Ponto bidimensional pertencente a uma lane.

    Coordenadas:
        x -> posição horizontal em pixels.
        y -> posição vertical em pixels.

    As coordenadas devem sempre ser finitas.
    """

    x: float
    y: float

    def __post_init__(self) -> None:
        x = float(self.x)
        y = float(self.y)

        if not isfinite(x):
            raise ValueError(
                f"LanePoint.x must be finite, got {self.x!r}"
            )

        if not isfinite(y):
            raise ValueError(
                f"LanePoint.y must be finite, got {self.y!r}"
            )

        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)

    def as_tuple(self) -> Point:
        """Retorna o ponto como tupla (x, y)."""

        return self.x, self.y


# =============================================================================
# LANE POLYNOMIAL
# =============================================================================

@dataclass(frozen=True, slots=True)
class LanePolynomial:
    """
    Modelo cúbico de uma lane.

    A representação matemática oficial do projeto é:

        x(y) = a*y³ + b*y² + c*y + d

    Os coeficientes são armazenados na ordem:

        (a, b, c, d)

    Não são permitidos modelos lineares ou quadráticos neste contrato.
    """

    coefficients: Tuple[float, float, float, float]

    fit_error: float
    confidence: float

    y_min: float
    y_max: float

    normalized: bool = False

    def __post_init__(self) -> None:
        coefficients = tuple(
            float(value)
            for value in self.coefficients
        )

        if len(coefficients) != 4:
            raise ValueError(
                "LanePolynomial requires exactly four coefficients"
            )

        if not all(
            isfinite(value)
            for value in coefficients
        ):
            raise ValueError(
                "LanePolynomial coefficients must be finite"
            )

        fit_error = float(self.fit_error)
        confidence = float(self.confidence)
        y_min = float(self.y_min)
        y_max = float(self.y_max)

        if not isfinite(fit_error):
            raise ValueError(
                "LanePolynomial.fit_error must be finite"
            )

        if not isfinite(confidence):
            raise ValueError(
                "LanePolynomial.confidence must be finite"
            )

        if not isfinite(y_min):
            raise ValueError(
                "LanePolynomial.y_min must be finite"
            )

        if not isfinite(y_max):
            raise ValueError(
                "LanePolynomial.y_max must be finite"
            )

        if y_max < y_min:
            raise ValueError(
                "LanePolynomial.y_max must be >= y_min"
            )

        object.__setattr__(
            self,
            "coefficients",
            coefficients,
        )

        object.__setattr__(
            self,
            "fit_error",
            max(0.0, fit_error),
        )

        object.__setattr__(
            self,
            "confidence",
            min(1.0, max(0.0, confidence)),
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

    @property
    def degree(self) -> int:
        """Grau fixo do modelo."""

        return 3

    def evaluate(self, y: float) -> float:
        """
        Avalia x(y).

        Usa Horner para reduzir operações e erro numérico.
        """

        y = float(y)

        if not isfinite(y):
            raise ValueError(
                "y must be finite"
            )

        a, b, c, d = self.coefficients

        return (
            (a * y + b) * y + c
        ) * y + d

    def derivative(self, y: float) -> float:
        """
        Calcula dx/dy.

        Para:

            x(y) = a*y³ + b*y² + c*y + d

        temos:

            dx/dy = 3a*y² + 2b*y + c
        """

        y = float(y)

        if not isfinite(y):
            raise ValueError(
                "y must be finite"
            )

        a, b, c, _ = self.coefficients

        return (
            (3.0 * a * y + 2.0 * b) * y
            + c
        )


# =============================================================================
# LANE LINE
# =============================================================================

@dataclass(frozen=True, slots=True)
class LaneLine:
    """
    Representação canônica de uma borda de faixa.

    Uma LaneLine pode representar uma observação direta,
    uma lane rastreada ou uma lane projetada.

    lane_id:
        Identidade temporal atribuída pelo tracker.

    points:
        Pontos da borda da faixa.

    confidence:
        Confiança atual da percepção [0, 1].

    quality:
        Qualidade geométrica [0, 1].

    source:
        Origem dos dados.

    detected_directly:
        True quando existe observação direta no frame atual.

    projected:
        True quando parte ou toda a representação foi projetada.

    age:
        Quantidade de frames desde a criação da identidade.

    missed_frames:
        Quantidade de frames consecutivos sem observação direta.

    velocity_x / velocity_y:
        Velocidade estimada da representação da lane.

    model:
        Modelo cúbico opcional associado à lane.
    """

    lane_id: int

    points: Tuple[LanePoint, ...]

    confidence: float = 0.0
    quality: float = 0.0

    source: LaneSource = LaneSource.DETECTED

    detected_directly: bool = True
    projected: bool = False

    age: int = 0
    missed_frames: int = 0

    velocity_x: float = 0.0
    velocity_y: float = 0.0

    model: Optional[LanePolynomial] = None

    def __post_init__(self) -> None:

        lane_id = int(self.lane_id)

        if lane_id < 0:
            raise ValueError(
                "lane_id must be non-negative"
            )

        points = tuple(
            point
            if isinstance(point, LanePoint)
            else LanePoint(
                float(point[0]),
                float(point[1]),
            )
            for point in self.points
        )

        if not points:
            raise ValueError(
                "LaneLine requires at least one point"
            )

        confidence = float(self.confidence)
        quality = float(self.quality)

        if not isfinite(confidence):
            raise ValueError(
                "LaneLine.confidence must be finite"
            )

        if not isfinite(quality):
            raise ValueError(
                "LaneLine.quality must be finite"
            )

        velocity_x = float(self.velocity_x)
        velocity_y = float(self.velocity_y)

        if not isfinite(velocity_x):
            raise ValueError(
                "LaneLine.velocity_x must be finite"
            )

        if not isfinite(velocity_y):
            raise ValueError(
                "LaneLine.velocity_y must be finite"
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
            min(
                1.0,
                max(0.0, confidence),
            ),
        )

        object.__setattr__(
            self,
            "quality",
            min(
                1.0,
                max(0.0, quality),
            ),
        )

        object.__setattr__(
            self,
            "source",
            LaneSource(self.source),
        )

        object.__setattr__(
            self,
            "age",
            max(0, int(self.age)),
        )

        object.__setattr__(
            self,
            "missed_frames",
            max(
                0,
                int(self.missed_frames),
            ),
        )

        object.__setattr__(
            self,
            "velocity_x",
            velocity_x,
        )

        object.__setattr__(
            self,
            "velocity_y",
            velocity_y,
        )

    @property
    def is_projected(self) -> bool:
        """Indica se a lane contém dados projetados."""

        return (
            self.projected
            or self.source is LaneSource.PROJECTED
        )

    @property
    def span(self) -> float:
        """Extensão vertical observada da lane em pixels."""

        if len(self.points) < 2:
            return 0.0

        ys = [
            point.y
            for point in self.points
        ]

        return max(ys) - min(ys)


# =============================================================================
# LANE PROJECTION
# =============================================================================

@dataclass(frozen=True, slots=True)
class LaneProjection:
    """
    Resultado da projeção de uma lane.

    A projeção nunca altera a observação original.
    """

    lane_id: int

    points: Tuple[LanePoint, ...]

    confidence: float

    extrapolated_distance: float = 0.0

    source: LaneSource = LaneSource.PROJECTED

    def __post_init__(self) -> None:

        lane_id = int(self.lane_id)

        if lane_id < 0:
            raise ValueError(
                "lane_id must be non-negative"
            )

        points = tuple(
            point
            if isinstance(point, LanePoint)
            else LanePoint(
                float(point[0]),
                float(point[1]),
            )
            for point in self.points
        )

        if not points:
            raise ValueError(
                "LaneProjection requires at least one point"
            )

        confidence = float(self.confidence)
        extrapolated_distance = float(
            self.extrapolated_distance
        )

        if not isfinite(confidence):
            raise ValueError(
                "LaneProjection.confidence must be finite"
            )

        if not isfinite(extrapolated_distance):
            raise ValueError(
                "LaneProjection.extrapolated_distance "
                "must be finite"
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
            min(
                1.0,
                max(0.0, confidence),
            ),
        )

        object.__setattr__(
            self,
            "extrapolated_distance",
            max(
                0.0,
                extrapolated_distance,
            ),
        )

        object.__setattr__(
            self,
            "source",
            LaneSource(self.source),
        )


# =============================================================================
# DETECTION RESULT
# =============================================================================

@dataclass(frozen=True, slots=True)
class LaneDetectionResult:
    """
    Resultado canônico produzido pelo detector de lanes.

    Este objeto representa somente a percepção daquele frame.
    Não contém estado temporal.
    """

    lanes: Tuple[LaneLine, ...] = field(
        default_factory=tuple
    )

    frame_width: int = 0
    frame_height: int = 0

    inference_ms: float = 0.0

    confidence: float = 0.0

    def __post_init__(self) -> None:

        lanes = tuple(self.lanes)

        frame_width = int(self.frame_width)
        frame_height = int(self.frame_height)

        if frame_width < 0:
            raise ValueError(
                "frame_width must be non-negative"
            )

        if frame_height < 0:
            raise ValueError(
                "frame_height must be non-negative"
            )

        inference_ms = float(
            self.inference_ms
        )

        confidence = float(
            self.confidence
        )

        if not isfinite(inference_ms):
            raise ValueError(
                "inference_ms must be finite"
            )

        if not isfinite(confidence):
            raise ValueError(
                "confidence must be finite"
            )

        object.__setattr__(
            self,
            "lanes",
            lanes,
        )

        object.__setattr__(
            self,
            "frame_width",
            frame_width,
        )

        object.__setattr__(
            self,
            "frame_height",
            frame_height,
        )

        object.__setattr__(
            self,
            "inference_ms",
            max(
                0.0,
                inference_ms,
            ),
        )

        object.__setattr__(
            self,
            "confidence",
            min(
                1.0,
                max(0.0, confidence),
            ),
        )

    @property
    def lane_count(self) -> int:
        """Quantidade de lanes detectadas."""

        return len(self.lanes)


# =============================================================================
# HELPERS
# =============================================================================

def points_from_xy(
    points: Iterable[Sequence[float]],
) -> Tuple[LanePoint, ...]:
    """
    Converte sequências (x, y) para LanePoint.

    Útil na fronteira entre YOLOPv2/OpenCV e o domínio.
    """

    return tuple(
        LanePoint(
            float(point[0]),
            float(point[1]),
        )
        for point in points
    )


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    "Point",
    "LaneSource",
    "LanePoint",
    "LanePolynomial",
    "LaneLine",
    "LaneProjection",
    "LaneDetectionResult",
    "points_from_xy",
]