"""
vision/lane_types.py

Tipos fundamentais do sistema de reconhecimento de faixas.

Este módulo NÃO executa:
    - inferência YOLOP
    - processamento de imagem
    - cálculo de geometria
    - controle do veículo
    - decisões ADAS

Ele define apenas os dados utilizados entre as diferentes
camadas do sistema.

Fluxo:

    YOLOP
      ↓
    LaneDetectionResult
      ↓
    LaneModel
      ↓
    LaneTracker
      ↓
    LaneAssociation
      ↓
    LaneGeometry
      ↓
    ADASState
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


# ============================================================================
# ENUMERAÇÕES
# ============================================================================


class LaneSide(str, Enum):
    """
    Relação de uma faixa com a faixa ocupada pelo veículo.
    """

    LEFT = "left"
    CURRENT_LEFT = "current_left"
    CURRENT = "current"
    CURRENT_RIGHT = "current_right"
    RIGHT = "right"
    UNKNOWN = "unknown"


class LaneType(str, Enum):
    """
    Tipo lógico de uma faixa detectada.

    Por enquanto o YOLOP não fornece diretamente essa informação.
    O sistema poderá inferi-la posteriormente a partir da geometria
    e do comportamento temporal da linha.
    """

    UNKNOWN = "unknown"
    SOLID = "solid"
    DASHED = "dashed"


class LaneQuality(str, Enum):
    """
    Qualidade atual da informação disponível sobre uma faixa.
    """

    NONE = "none"
    POOR = "poor"
    PARTIAL = "partial"
    GOOD = "good"
    EXCELLENT = "excellent"


class ProjectionQuality(str, Enum):
    """
    Qualidade da projeção da faixa para regiões que não foram
    diretamente observadas pelo detector.

    A projeção será fundamental para curvas, mas o ADAS só poderá
    utilizá-la quando houver informação suficiente para sustentá-la.
    """

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ============================================================================
# PONTO
# ============================================================================


@dataclass(frozen=True)
class LanePoint:
    """
    Ponto de uma faixa no frame original.

    x:
        Coordenada horizontal em pixels.

    y:
        Coordenada vertical em pixels.

    confidence:
        Confiança do detector nesse ponto.

    valid:
        Indica se o ponto pode ser utilizado pelos módulos posteriores.
    """

    x: float
    y: float
    confidence: float
    valid: bool = True


# ============================================================================
# LINHA DETECTADA
# ============================================================================


@dataclass
class LaneLine:
    """
    Representação de uma linha de faixa detectada.

    Uma LaneLine representa UMA marcação longitudinal da pista.

    Exemplo de uma pista com três faixas:

        |       |       |
        L0      L1      L2
        |       |       |
        L3      L4      L5

    O sistema não assume que duas linhas consecutivas são
    necessariamente a faixa ocupada pelo veículo.

    Essa associação será responsabilidade do LaneAssociation.
    """

    lane_id: Optional[int] = None

    points: List[LanePoint] = field(
        default_factory=list
    )

    confidence: float = 0.0

    quality: LaneQuality = LaneQuality.NONE

    lane_type: LaneType = LaneType.UNKNOWN

    side: LaneSide = LaneSide.UNKNOWN

    detected_directly: bool = True

    projected: bool = False

    projection_quality: ProjectionQuality = (
        ProjectionQuality.NONE
    )

    age_frames: int = 0

    missed_frames: int = 0

    valid: bool = False

    def point_count(self) -> int:
        """
        Quantidade de pontos válidos.
        """

        return sum(
            1
            for point in self.points
            if point.valid
        )

    def is_observable(self) -> bool:
        """
        Indica se existem pontos diretamente observados.
        """

        return (
            self.detected_directly
            and self.point_count() > 0
        )


# ============================================================================
# MODELO GEOMÉTRICO DA FAIXA
# ============================================================================


@dataclass
class LanePolynomial:
    """
    Modelo polinomial de uma linha de faixa.

    Representa:

        x(y) = a*y³ + b*y² + c*y + d

    O uso de x em função de y é intencional.

    Em uma imagem de câmera automotiva, y representa profundidade
    aproximada no plano da imagem. Isso permite modelar curvas
    sem obrigar o sistema a assumir que a faixa é uma reta.

    O modelo de terceiro grau será utilizado apenas quando houver
    pontos suficientes e estabilidade suficiente para justificar
    esse grau.
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

    def evaluate(self, y: float) -> float:
        """
        Avalia x(y).
        """

        return (
            self.a * y ** 3
            + self.b * y ** 2
            + self.c * y
            + self.d
        )

    def derivative(self, y: float) -> float:
        """
        Primeira derivada dx/dy.
        """

        return (
            3.0 * self.a * y ** 2
            + 2.0 * self.b * y
            + self.c
        )

    def second_derivative(self, y: float) -> float:
        """
        Segunda derivada d²x/dy².
        """

        return (
            6.0 * self.a * y
            + 2.0 * self.b
        )


# ============================================================================
# PROJEÇÃO
# ============================================================================


@dataclass
class LaneProjection:
    """
    Resultado da extrapolação de uma faixa.

    A projeção representa a continuação matemática de uma faixa
    parcialmente observada.

    Importante:

    Projeção NÃO significa que a faixa foi realmente detectada.

    O sistema deve distinguir:

        observado
        projetado
        observado + projetado

    para impedir que uma previsão matemática seja tratada como
    informação visual real.
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


# ============================================================================
# FAIXA COMPLETA
# ============================================================================


@dataclass
class LaneModel:
    """
    Modelo completo de uma linha de faixa.

    Junta:

        detecção
        histórico
        modelo matemático
        projeção
        identificação lógica
    """

    lane_id: int

    line: LaneLine = field(
        default_factory=LaneLine
    )

    polynomial: Optional[LanePolynomial] = None

    projection: Optional[LaneProjection] = None

    tracked: bool = False

    stable: bool = False

    valid: bool = False


# ============================================================================
# FAIXA OCUPADA PELO VEÍCULO
# ============================================================================


@dataclass
class CurrentLane:
    """
    Representa a faixa atualmente ocupada pelo veículo.

    left_boundary:
        Linha que limita a faixa pelo lado esquerdo.

    right_boundary:
        Linha que limita a faixa pelo lado direito.

    center_x:
        Centro estimado da faixa no ponto analisado.

    lateral_offset:
        Distância horizontal do centro do veículo para o centro
        da faixa.

    normalized_offset:
        Offset normalizado pela largura da faixa.

        -1.0 ≈ limite esquerdo
         0.0 ≈ centro
        +1.0 ≈ limite direito
    """

    left_boundary: Optional[LaneModel] = None

    right_boundary: Optional[LaneModel] = None

    center_x: Optional[float] = None

    lane_width: Optional[float] = None

    lateral_offset: Optional[float] = None

    normalized_offset: Optional[float] = None

    confidence: float = 0.0

    valid: bool = False


# ============================================================================
# RESULTADO DA ASSOCIAÇÃO
# ============================================================================


@dataclass
class LaneAssociationResult:
    """
    Resultado da identificação de todas as faixas em relação
    ao veículo.

    Exemplo:

        esquerda        atual        direita
           |              |             |
           |      CARRO   |             |
           |        ↓     |             |
        Lane 0          Lane 1        Lane 2

    Com no máximo três faixas + acostamento, esse objeto permitirá
    manter a identificação mesmo quando uma linha desaparecer
    temporariamente.
    """

    lanes: List[LaneModel] = field(
        default_factory=list
    )

    current_lane: Optional[CurrentLane] = None

    current_lane_id: Optional[int] = None

    left_lanes: List[LaneModel] = field(
        default_factory=list
    )

    right_lanes: List[LaneModel] = field(
        default_factory=list
    )

    valid: bool = False

    confidence: float = 0.0


# ============================================================================
# GEOMETRIA
# ============================================================================


@dataclass
class LaneGeometry:
    """
    Informações geométricas da faixa atual.

    Não representa apenas a posição atual.

    Também armazena informações necessárias para prever
    como a faixa continuará à frente.
    """

    current_center_x: Optional[float] = None

    lookahead_center_x: Optional[float] = None

    lane_width: Optional[float] = None

    lateral_offset: Optional[float] = None

    normalized_offset: Optional[float] = None

    heading_error: Optional[float] = None

    curvature: Optional[float] = None

    curvature_direction: Optional[float] = None

    valid: bool = False

    confidence: float = 0.0


# ============================================================================
# ESTADO VISUAL DA FAIXA
# ============================================================================


class LaneProximity(str, Enum):
    """
    Proximidade do veículo em relação ao centro da faixa.
    """

    CENTERED = "centered"
    APPROACHING_LEFT = "approaching_left"
    APPROACHING_RIGHT = "approaching_right"
    NEAR_LEFT = "near_left"
    NEAR_RIGHT = "near_right"
    VERY_NEAR_LEFT = "very_near_left"
    VERY_NEAR_RIGHT = "very_near_right"
    UNKNOWN = "unknown"


@dataclass
class LaneWarning:
    """
    Estado de advertência relacionado à posição lateral.
    """

    proximity: LaneProximity = LaneProximity.UNKNOWN

    warning_level: int = 0

    correction_allowed: bool = False

    confidence: float = 0.0

    valid: bool = False


# ============================================================================
# SNAPSHOT COMPLETO
# ============================================================================


@dataclass
class LaneFrame:
    """
    Snapshot completo de uma análise de faixas em um frame.

    Esse será o objeto principal transportado entre as camadas
    do sistema.
    """

    frame_index: int = 0

    timestamp: float = 0.0

    detected_lanes: List[LaneModel] = field(
        default_factory=list
    )

    association: Optional[
        LaneAssociationResult
    ] = None

    current_lane: Optional[
        CurrentLane
    ] = None

    geometry: Optional[
        LaneGeometry
    ] = None

    warning: Optional[
        LaneWarning
    ] = None

    valid: bool = False

    enough_information: bool = False

    safe_for_adas: bool = False

    error: Optional[str] = None


# ============================================================================
# HELPERS
# ============================================================================


def lane_points_to_xy(
    points: List[LanePoint],
) -> List[Tuple[float, float]]:
    """
    Converte LanePoint para pares (x, y).
    """

    return [
        (point.x, point.y)
        for point in points
        if point.valid
    ]


def calculate_lane_center(
    left_x: float,
    right_x: float,
) -> float:
    """
    Calcula o centro horizontal entre duas linhas.

        centro = (esquerda + direita) / 2
    """

    return (
        left_x + right_x
    ) / 2.0


def calculate_normalized_offset(
    vehicle_x: float,
    center_x: float,
    lane_width: float,
) -> Optional[float]:
    """
    Calcula o deslocamento lateral normalizado.

    Resultado:

        0.0  -> centro da faixa
        <0   -> esquerda do centro
        >0   -> direita do centro

    A normalização pela largura da faixa torna o valor
    independente da resolução da imagem.
    """

    if lane_width <= 0.0:
        return None

    return (
        vehicle_x - center_x
    ) / (
        lane_width / 2.0
    )


__all__ = [
    "LaneSide",
    "LaneType",
    "LaneQuality",
    "ProjectionQuality",
    "LaneProximity",
    "LanePoint",
    "LaneLine",
    "LanePolynomial",
    "LaneProjection",
    "LaneModel",
    "CurrentLane",
    "LaneAssociationResult",
    "LaneGeometry",
    "LaneWarning",
    "LaneFrame",
    "lane_points_to_xy",
    "calculate_lane_center",
    "calculate_normalized_offset",
]