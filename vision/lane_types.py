"""
vision/lane_types.py

Tipos fundamentais utilizados pelo sistema de detecção e projeção
de faixas.

Responsabilidade deste módulo:

    LanePoint
        ↓
    LaneLine
        ↓
    LaneModel
        ↓
    LanePolynomial
        ↓
    LaneProjection

Este módulo NÃO:
    - executa inferência;
    - realiza tracking;
    - identifica a faixa atual;
    - calcula posição do veículo;
    - toma decisões ADAS.

Observação importante:
    Os dataclasses deste módulo são deliberadamente MUTÁVEIS.
    Os testes e alguns estágios do pipeline precisam poder alterar
    valores de pontos após sua criação.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, List, Optional, Sequence, Tuple

import math


# =============================================================================
# UTILITÁRIOS
# =============================================================================


def _finite(value: float) -> bool:
    """Retorna True quando o valor é numérico e finito."""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _clip01(value: float) -> float:
    """Limita um valor ao intervalo [0, 1]."""
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


# =============================================================================
# QUALIDADE DA PROJEÇÃO
# =============================================================================


class ProjectionQuality(Enum):
    """
    Qualidade estimada de uma projeção de faixa.
    """

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# =============================================================================
# PONTO DE FAIXA
# =============================================================================


@dataclass
class LanePoint:
    """
    Ponto pertencente a uma linha de faixa.

    IMPORTANTE:
        Esta classe NÃO é frozen.

    Os testes do projeto e estágios de processamento podem fazer:

        point.valid = False
        point.x = float("nan")
        point.y = float("inf")

    O projetor é responsável por rejeitar esses valores inválidos.
    """

    x: float
    y: float
    confidence: float = 0.0
    valid: bool = True

    def __post_init__(self) -> None:
        """
        Normalização mínima dos tipos.

        Não rejeitamos NaN/inf aqui.

        Isso é intencional: pontos inválidos precisam poder existir
        para que as camadas superiores possam filtrá-los.
        """

        try:
            self.x = float(self.x)
        except (TypeError, ValueError):
            self.x = float("nan")

        try:
            self.y = float(self.y)
        except (TypeError, ValueError):
            self.y = float("nan")

        try:
            self.confidence = float(self.confidence)
        except (TypeError, ValueError):
            self.confidence = 0.0

        self.valid = bool(self.valid)

    @property
    def is_finite(self) -> bool:
        """Indica se x, y e confidence são finitos."""
        return (
            _finite(self.x)
            and _finite(self.y)
            and _finite(self.confidence)
        )

    @property
    def is_valid(self) -> bool:
        """
        Indica se o ponto está marcado como válido e possui
        valores numéricos finitos.
        """
        return (
            self.valid
            and self.is_finite
        )

    def distance_to(
        self,
        other: "LanePoint",
    ) -> float:
        """Calcula a distância euclidiana até outro ponto."""

        if other is None:
            return float("inf")

        try:
            dx = float(self.x) - float(other.x)
            dy = float(self.y) - float(other.y)

            return float(
                math.hypot(dx, dy)
            )

        except (TypeError, ValueError):
            return float("inf")

    def copy(self) -> "LanePoint":
        """Retorna uma cópia independente do ponto."""

        return LanePoint(
            x=self.x,
            y=self.y,
            confidence=self.confidence,
            valid=self.valid,
        )


# =============================================================================
# LINHA DE FAIXA
# =============================================================================


@dataclass
class LaneLine:
    """
    Representa uma linha de faixa composta por vários pontos.
    """

    points: List[LanePoint] = field(
        default_factory=list
    )

    confidence: float = 0.0

    valid: bool = True

    lane_id: Optional[int] = None

    def __post_init__(self) -> None:

        if self.points is None:
            self.points = []

        else:
            self.points = list(self.points)

        try:
            self.confidence = float(
                self.confidence
            )
        except (TypeError, ValueError):
            self.confidence = 0.0

        self.valid = bool(
            self.valid
        )

    @property
    def point_count(self) -> int:
        """Quantidade de pontos da linha."""

        return len(self.points)

    @property
    def valid_points(self) -> List[LanePoint]:
        """Retorna somente pontos válidos."""

        result: List[LanePoint] = []

        for point in self.points:

            if point is None:
                continue

            if getattr(
                point,
                "is_valid",
                False,
            ):
                result.append(point)

        return result

    @property
    def y_min(self) -> Optional[float]:
        """Menor coordenada Y entre os pontos válidos."""

        points = self.valid_points

        if not points:
            return None

        return float(
            min(point.y for point in points)
        )

    @property
    def y_max(self) -> Optional[float]:
        """Maior coordenada Y entre os pontos válidos."""

        points = self.valid_points

        if not points:
            return None

        return float(
            max(point.y for point in points)
        )

    @property
    def x_min(self) -> Optional[float]:
        """Menor coordenada X entre os pontos válidos."""

        points = self.valid_points

        if not points:
            return None

        return float(
            min(point.x for point in points)
        )

    @property
    def x_max(self) -> Optional[float]:
        """Maior coordenada X entre os pontos válidos."""

        points = self.valid_points

        if not points:
            return None

        return float(
            max(point.x for point in points)
        )

    def add_point(
        self,
        point: LanePoint,
    ) -> None:
        """Adiciona um ponto à linha."""

        if point is None:
            return

        self.points.append(point)

    def copy(self) -> "LaneLine":
        """Cria uma cópia independente da linha."""

        return LaneLine(
            points=[
                point.copy()
                for point in self.points
                if point is not None
            ],
            confidence=self.confidence,
            valid=self.valid,
            lane_id=self.lane_id,
        )


# =============================================================================
# MODELO DE FAIXA
# =============================================================================


@dataclass
class LaneModel:
    """
    Modelo lógico de uma faixa detectada.

    A estrutura mantém a linha original separada das informações
    de confiança e identificação.
    """

    line: Optional[LaneLine] = None

    confidence: float = 0.0

    valid: bool = True

    lane_id: Optional[int] = None

    def __post_init__(self) -> None:

        try:
            self.confidence = float(
                self.confidence
            )
        except (TypeError, ValueError):
            self.confidence = 0.0

        self.valid = bool(
            self.valid
        )

    @property
    def points(self) -> List[LanePoint]:
        """Acesso conveniente aos pontos da linha."""

        if self.line is None:
            return []

        return self.line.points

    @property
    def point_count(self) -> int:
        """Quantidade de pontos do modelo."""

        return len(self.points)

    def copy(self) -> "LaneModel":
        """Cria uma cópia independente."""

        return LaneModel(
            line=(
                self.line.copy()
                if self.line is not None
                else None
            ),
            confidence=self.confidence,
            valid=self.valid,
            lane_id=self.lane_id,
        )


# =============================================================================
# POLINÔMIO DA FAIXA
# =============================================================================


@dataclass
class LanePolynomial:
    """
    Polinômio x(y):

        x(y) = a*y³ + b*y² + c*y + d

    Para polinômios de grau inferior, os coeficientes ausentes
    permanecem zero.

    Exemplo:

        linha reta:
            a = 0
            b = 0
            c = constante
            d = offset

        curva quadrática:
            a = 0
            b != 0
            c != 0
            d != 0
    """

    a: float = 0.0
    b: float = 0.0
    c: float = 0.0
    d: float = 0.0

    valid: bool = False

    fit_error: float = float("inf")

    sample_count: int = 0

    confidence: float = 0.0

    y_min: float = 0.0

    y_max: float = 0.0

    def __post_init__(self) -> None:

        for attribute in (
            "a",
            "b",
            "c",
            "d",
            "fit_error",
            "confidence",
            "y_min",
            "y_max",
        ):
            try:
                setattr(
                    self,
                    attribute,
                    float(
                        getattr(
                            self,
                            attribute,
                        )
                    ),
                )
            except (TypeError, ValueError):
                if attribute == "fit_error":
                    setattr(
                        self,
                        attribute,
                        float("inf"),
                    )
                else:
                    setattr(
                        self,
                        attribute,
                        0.0,
                    )

        try:
            self.sample_count = int(
                self.sample_count
            )
        except (TypeError, ValueError):
            self.sample_count = 0

        self.valid = bool(
            self.valid
        )

        self.confidence = _clip01(
            self.confidence
        )

    def evaluate(
        self,
        y: float,
    ) -> float:
        """
        Avalia x(y).
        """

        y = float(y)

        return float(
            (
                (
                    self.a * y
                    + self.b
                )
                * y
                + self.c
            )
            * y
            + self.d
        )

    def derivative(
        self,
        y: float,
    ) -> float:
        """
        Calcula dx/dy.
        """

        y = float(y)

        return float(
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

        y = float(y)

        return float(
            6.0 * self.a * y
            + 2.0 * self.b
        )

    def coefficients(
        self,
    ) -> Tuple[float, float, float, float]:
        """Retorna os coeficientes em ordem a,b,c,d."""

        return (
            float(self.a),
            float(self.b),
            float(self.c),
            float(self.d),
        )

    def copy(self) -> "LanePolynomial":
        """Cria uma cópia independente."""

        return LanePolynomial(
            a=self.a,
            b=self.b,
            c=self.c,
            d=self.d,
            valid=self.valid,
            fit_error=self.fit_error,
            sample_count=self.sample_count,
            confidence=self.confidence,
            y_min=self.y_min,
            y_max=self.y_max,
        )


# =============================================================================
# PROJEÇÃO
# =============================================================================


@dataclass
class LaneProjection:
    """
    Resultado estrutural da projeção de uma linha de faixa.
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

    def __post_init__(self) -> None:

        if self.points is None:
            self.points = []

        else:
            self.points = list(
                self.points
            )

        if not isinstance(
            self.quality,
            ProjectionQuality,
        ):
            try:
                self.quality = (
                    ProjectionQuality(
                        self.quality
                    )
                )
            except (
                TypeError,
                ValueError,
            ):
                self.quality = (
                    ProjectionQuality.NONE
                )

        self.extrapolated = bool(
            self.extrapolated
        )

        self.valid = bool(
            self.valid
        )

        if self.horizon_y is not None:
            try:
                self.horizon_y = float(
                    self.horizon_y
                )
            except (
                TypeError,
                ValueError,
            ):
                self.horizon_y = None

    @property
    def point_count(self) -> int:
        """Quantidade de pontos projetados."""

        return len(self.points)

    @property
    def confidence(self) -> float:
        """Confiança do polinômio associado."""

        if self.polynomial is None:
            return 0.0

        return float(
            self.polynomial.confidence
        )

    @property
    def fit_error(self) -> float:
        """Erro do ajuste polinomial."""

        if self.polynomial is None:
            return float("inf")

        return float(
            self.polynomial.fit_error
        )

    def evaluate(
        self,
        y: float,
    ) -> Optional[float]:
        """
        Avalia diretamente o polinômio da projeção.
        """

        if (
            self.polynomial is None
            or not self.polynomial.valid
        ):
            return None

        try:
            value = self.polynomial.evaluate(
                y
            )
        except (
            TypeError,
            ValueError,
            OverflowError,
        ):
            return None

        if not _finite(value):
            return None

        return float(value)

    def copy(self) -> "LaneProjection":
        """Cria uma cópia independente da projeção."""

        return LaneProjection(
            polynomial=(
                self.polynomial.copy()
                if self.polynomial is not None
                else None
            ),
            points=[
                point.copy()
                for point in self.points
                if point is not None
            ],
            quality=self.quality,
            extrapolated=self.extrapolated,
            valid=self.valid,
            horizon_y=self.horizon_y,
        )


# =============================================================================
# EXPORTAÇÕES
# =============================================================================


__all__ = [
    "ProjectionQuality",
    "LanePoint",
    "LaneLine",
    "LaneModel",
    "LanePolynomial",
    "LaneProjection",
]