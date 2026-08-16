"""
vision/detection_types.py

Forza Assistents
================

Contratos de dados compartilhados pelo pipeline de percepção.

Este módulo não depende de:
    - YOLOP
    - UFLD
    - OpenCV
    - ONNX Runtime
    - PyTorch

Responsabilidade:
    - representar pontos de faixa;
    - representar resultados de detecção;
    - normalizar entradas;
    - impedir que dados inválidos avancem para a geometria/modelagem.

Regra importante
----------------
NaN e infinito podem existir na entrada de um detector como representação
de uma observação inválida. Portanto, LanePoint NÃO deve lançar exceção
apenas por receber um valor não-finito.

Em vez disso:

    ponto não-finito -> valid=False

Isso permite que filtros temporais e módulos de limpeza removam o ponto
antes de qualquer operação matemática.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, hypot
from typing import Iterable, Mapping, Optional, Sequence, Tuple


# ============================================================================
# TIPOS BÁSICOS
# ============================================================================

Point2D = Tuple[float, float]


# ============================================================================
# CONSTANTES DE DETECÇÃO
# ============================================================================

# Âncoras verticais padrão do CULane.
#
# Mantidas aqui por compatibilidade com testes e módulos que ainda utilizam
# a convenção de 18 linhas verticais do UFLD/CULane.
#
# A arquitetura atual pode utilizar YOLOP, mas esse contrato permanece
# disponível porque LanePoint é um tipo neutro do pipeline.
CULANE_ROW_ANCHORS: Tuple[float, ...] = (
    240.0,
    230.0,
    220.0,
    210.0,
    200.0,
    190.0,
    180.0,
    170.0,
    160.0,
    150.0,
    140.0,
    130.0,
    120.0,
    110.0,
    100.0,
    90.0,
    80.0,
    70.0,
)


# ============================================================================
# FUNÇÕES INTERNAS
# ============================================================================


def _clamp_confidence(value: float) -> float:
    """
    Limita confiança ao intervalo [0, 1].

    Valores inválidos são tratados como 0.
    """

    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0

    if not isfinite(value):
        return 0.0

    return max(0.0, min(1.0, value))


def _safe_float(value: float) -> float:
    """
    Converte um valor para float.

    Não rejeita NaN/infinito.

    A decisão sobre validade pertence ao LanePoint.
    """

    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


# ============================================================================
# LANE POINT
# ============================================================================


@dataclass(frozen=True, slots=True)
class LanePoint:
    """
    Ponto de uma marcação de faixa.

    Parameters
    ----------
    x:
        Coordenada horizontal.

    y:
        Coordenada vertical.

    confidence:
        Confiança da observação, normalizada para [0, 1].

    valid:
        Indica se o ponto é matematicamente utilizável.

    Comportamento para dados inválidos
    -----------------------------------
    Caso x ou y sejam NaN/infinito, o ponto é preservado como objeto de
    entrada, mas automaticamente marcado como:

        valid=False

    Isso é intencional.

    O detector/filtro pode então receber observações inválidas sem quebrar
    a execução e removê-las antes da modelagem.
    """

    x: float
    y: float
    confidence: float = 1.0
    valid: bool = True

    def __post_init__(self) -> None:
        x = _safe_float(self.x)
        y = _safe_float(self.y)
        confidence = _clamp_confidence(self.confidence)

        requested_valid = bool(self.valid)

        # Um ponto não-finito jamais pode ser considerado matematicamente
        # válido.
        finite = isfinite(x) and isfinite(y)

        normalized_valid = requested_valid and finite

        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "valid", normalized_valid)

    # ------------------------------------------------------------------
    # Propriedades
    # ------------------------------------------------------------------

    @property
    def xy(self) -> Point2D:
        """Retorna `(x, y)`."""

        return self.x, self.y

    @property
    def finite(self) -> bool:
        """
        Indica se x e y são finitos.

        Não depende do campo `valid`.
        """

        return isfinite(self.x) and isfinite(self.y)

    @property
    def usable(self) -> bool:
        """
        Indica se o ponto pode entrar em operações matemáticas.
        """

        return self.valid and self.finite and self.confidence > 0.0

    def distance_to(self, other: "LanePoint") -> float:
        """
        Distância euclidiana até outro ponto.

        Levanta ValueError caso algum dos pontos seja não-finito.
        """

        if not isinstance(other, LanePoint):
            raise TypeError("other deve ser LanePoint.")

        if not self.finite or not other.finite:
            raise ValueError(
                "Não é possível calcular distância de ponto não-finito."
            )

        return hypot(
            self.x - other.x,
            self.y - other.y,
        )


# ============================================================================
# LANE DETECTION RESULT
# ============================================================================


@dataclass(slots=True)
class LaneDetectionResult:
    """
    Resultado neutro da detecção de faixas.

    O objeto é independente do detector utilizado.

    Attributes
    ----------
    lanes:
        Tupla de lanes, sendo cada lane uma sequência de LanePoint.

    confidence:
        Confiança global do resultado.

    image_width:
        Largura da imagem de origem.

    image_height:
        Altura da imagem de origem.

    valid:
        Validade global da detecção.

    frame_id:
        Identificador opcional do frame.

    timestamp:
        Timestamp opcional.

    metadata:
        Metadados adicionais.
    """

    lanes: Sequence[Sequence[LanePoint]] = field(default_factory=tuple)

    confidence: float = 0.0

    image_width: Optional[int] = None
    image_height: Optional[int] = None

    valid: bool = True

    frame_id: Optional[int] = None
    timestamp: Optional[float] = None

    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized_lanes = []

        for lane in self.lanes:
            normalized_lane = []

            for point in lane:
                if not isinstance(point, LanePoint):
                    raise TypeError(
                        "Cada ponto de lane deve ser LanePoint."
                    )

                normalized_lane.append(point)

            normalized_lanes.append(tuple(normalized_lane))

        self.lanes = tuple(normalized_lanes)

        self.confidence = _clamp_confidence(
            self.confidence
        )

        if self.image_width is not None:
            self.image_width = int(self.image_width)

        if self.image_height is not None:
            self.image_height = int(self.image_height)

        if self.frame_id is not None:
            self.frame_id = int(self.frame_id)

        if self.timestamp is not None:
            timestamp = float(self.timestamp)

            self.timestamp = (
                timestamp
                if isfinite(timestamp)
                else None
            )

        self.valid = bool(self.valid)

        if self.metadata is None:
            self.metadata = {}
        else:
            self.metadata = dict(self.metadata)

    # ------------------------------------------------------------------
    # Propriedades
    # ------------------------------------------------------------------

    @property
    def lane_count(self) -> int:
        """Número de lanes."""

        return len(self.lanes)

    @property
    def points(self) -> tuple[LanePoint, ...]:
        """Todos os pontos em uma sequência plana."""

        return tuple(
            point
            for lane in self.lanes
            for point in lane
        )

    @property
    def valid_points(self) -> tuple[LanePoint, ...]:
        """
        Todos os pontos matematicamente utilizáveis.
        """

        return tuple(
            point
            for point in self.points
            if point.usable
        )

    @property
    def is_valid(self) -> bool:
        """Alias semântico de `valid`."""

        return self.valid

    @property
    def has_valid_points(self) -> bool:
        """Indica se existe pelo menos um ponto utilizável."""

        return any(
            point.usable
            for point in self.points
        )

    # ------------------------------------------------------------------
    # Acesso
    # ------------------------------------------------------------------

    def get_lane(
        self,
        index: int,
    ) -> tuple[LanePoint, ...]:
        """Retorna uma lane pelo índice."""

        if index < 0 or index >= len(self.lanes):
            raise IndexError(
                f"Índice de lane fora do intervalo: {index}."
            )

        return tuple(self.lanes[index])

    def iter_lanes(
        self,
    ) -> Iterable[tuple[LanePoint, ...]]:
        """Itera pelas lanes."""

        return iter(self.lanes)

    def __len__(self) -> int:
        """Retorna a quantidade de lanes."""

        return len(self.lanes)


# ============================================================================
# FÁBRICAS
# ============================================================================


def make_lane_point(
    x: float,
    y: float,
    confidence: float = 1.0,
    valid: bool = True,
) -> LanePoint:
    """
    Cria um LanePoint normalizado.
    """

    return LanePoint(
        x=x,
        y=y,
        confidence=confidence,
        valid=valid,
    )


def make_detection_result(
    lanes: Sequence[Sequence[LanePoint]],
    *,
    confidence: float = 0.0,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
    valid: bool = True,
    frame_id: Optional[int] = None,
    timestamp: Optional[float] = None,
    metadata: Optional[Mapping] = None,
) -> LaneDetectionResult:
    """
    Cria um LaneDetectionResult.
    """

    return LaneDetectionResult(
        lanes=lanes,
        confidence=confidence,
        image_width=image_width,
        image_height=image_height,
        valid=valid,
        frame_id=frame_id,
        timestamp=timestamp,
        metadata={} if metadata is None else dict(metadata),
    )


# ============================================================================
# EXPORTS
# ============================================================================


__all__ = [
    "Point2D",
    "CULANE_ROW_ANCHORS",
    "LanePoint",
    "LaneDetectionResult",
    "make_lane_point",
    "make_detection_result",
]