"""
vision/detection_types.py

Forza Assistents
================

Tipos de dados compartilhados pelo pipeline de percepção.

Este módulo é deliberadamente independente de:

- YOLOP
- UFLD
- OpenCV
- ONNX Runtime
- PyTorch

A finalidade é fornecer contratos estáveis para os módulos posteriores
do pipeline:

    Detector
        ↓
    LaneDetectionResult
        ↓
    LaneGeometry
        ↓
    LaneModel
        ↓
    LaneTracker / LaneAssignment
        ↓
    ADAS

IMPORTANTE
----------
Este módulo não deve importar nenhum detector.

Isso permite que os testes matemáticos e geométricos sejam executados
sem carregar CUDA, OpenCV, ONNX Runtime ou qualquer backend de inferência.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Tipos básicos
# ---------------------------------------------------------------------------

Point2D = Tuple[float, float]


# ---------------------------------------------------------------------------
# LanePoint
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class LanePoint:
    """
    Ponto pertencente a uma marcação de faixa.

    Attributes
    ----------
    x:
        Coordenada horizontal em pixels.

    y:
        Coordenada vertical em pixels.

    confidence:
        Confiança da detecção no intervalo [0, 1].

    valid:
        Indica se o ponto deve ser considerado válido pelo pipeline.
    """

    x: float
    y: float
    confidence: float = 1.0
    valid: bool = True

    def __post_init__(self) -> None:
        """Normaliza e valida os valores básicos."""

        x = float(self.x)
        y = float(self.y)
        confidence = float(self.confidence)

        if not all(
            value == value and abs(value) != float("inf")
            for value in (x, y, confidence)
        ):
            raise ValueError("LanePoint não pode conter NaN ou infinito.")

        confidence = max(0.0, min(1.0, confidence))

        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "valid", bool(self.valid))

    @property
    def xy(self) -> Point2D:
        """Retorna o ponto como `(x, y)`."""

        return self.x, self.y

    def distance_to(self, other: "LanePoint") -> float:
        """Distância euclidiana até outro ponto."""

        if not isinstance(other, LanePoint):
            raise TypeError("other deve ser LanePoint.")

        dx = self.x - other.x
        dy = self.y - other.y

        return (dx * dx + dy * dy) ** 0.5


# ---------------------------------------------------------------------------
# LaneDetectionResult
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class LaneDetectionResult:
    """
    Resultado neutro da etapa de detecção de faixas.

    O detector pode ser YOLOP ou qualquer outro detector futuro.
    Nenhuma parte posterior do pipeline precisa conhecer a implementação
    interna do detector.

    Attributes
    ----------
    lanes:
        Sequência de faixas detectadas.

        Cada faixa é representada por uma sequência de LanePoint.

    confidence:
        Confiança global da detecção.

    image_width:
        Largura da imagem utilizada na detecção.

    image_height:
        Altura da imagem utilizada na detecção.

    valid:
        Indica se o resultado é utilizável.

    frame_id:
        Identificador opcional do frame.

    timestamp:
        Timestamp opcional associado ao frame.

    metadata:
        Informações adicionais não obrigatórias.
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
        """Normaliza o resultado sem introduzir dependências externas."""

        normalized_lanes = []

        for lane in self.lanes:
            normalized_lane = []

            for point in lane:
                if isinstance(point, LanePoint):
                    normalized_lane.append(point)
                else:
                    raise TypeError(
                        "Cada ponto de lane deve ser uma instância de LanePoint."
                    )

            normalized_lanes.append(tuple(normalized_lane))

        self.lanes = tuple(normalized_lanes)

        self.confidence = max(
            0.0,
            min(1.0, float(self.confidence)),
        )

        if self.image_width is not None:
            self.image_width = int(self.image_width)

        if self.image_height is not None:
            self.image_height = int(self.image_height)

        if self.frame_id is not None:
            self.frame_id = int(self.frame_id)

        if self.timestamp is not None:
            self.timestamp = float(self.timestamp)

        self.valid = bool(self.valid)

        if self.metadata is None:
            self.metadata = {}
        else:
            self.metadata = dict(self.metadata)

    # ------------------------------------------------------------------
    # Compatibilidade / acesso
    # ------------------------------------------------------------------

    @property
    def lane_count(self) -> int:
        """Número de faixas detectadas."""

        return len(self.lanes)

    @property
    def points(self) -> tuple[LanePoint, ...]:
        """
        Retorna todos os pontos de todas as faixas em uma única sequência.

        Útil para módulos que não precisam preservar a identidade da faixa.
        """

        return tuple(
            point
            for lane in self.lanes
            for point in lane
        )

    @property
    def is_valid(self) -> bool:
        """Alias semântico para `valid`."""

        return self.valid

    def get_lane(self, index: int) -> tuple[LanePoint, ...]:
        """Retorna uma faixa pelo índice."""

        if index < 0 or index >= len(self.lanes):
            raise IndexError(
                f"Índice de lane fora do intervalo: {index}."
            )

        return tuple(self.lanes[index])

    def iter_lanes(self) -> Iterable[tuple[LanePoint, ...]]:
        """Itera pelas faixas detectadas."""

        return iter(self.lanes)

    def __len__(self) -> int:
        """Permite `len(result)` para obter o número de faixas."""

        return len(self.lanes)


# ---------------------------------------------------------------------------
# Fábricas auxiliares
# ---------------------------------------------------------------------------

def make_lane_point(
    x: float,
    y: float,
    confidence: float = 1.0,
    valid: bool = True,
) -> LanePoint:
    """
    Cria um LanePoint.

    Função pequena para manter criação de pontos consistente.
    """

    return LanePoint(
        x=float(x),
        y=float(y),
        confidence=float(confidence),
        valid=bool(valid),
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
    metadata: Optional[dict] = None,
) -> LaneDetectionResult:
    """
    Cria um LaneDetectionResult de forma explícita.
    """

    return LaneDetectionResult(
        lanes=lanes,
        confidence=confidence,
        image_width=image_width,
        image_height=image_height,
        valid=valid,
        frame_id=frame_id,
        timestamp=timestamp,
        metadata={} if metadata is None else metadata,
    )


__all__ = [
    "Point2D",
    "LanePoint",
    "LaneDetectionResult",
    "make_lane_point",
    "make_detection_result",
]