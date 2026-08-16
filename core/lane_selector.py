"""
core/lane_selector.py

Seleção das lanes relevantes para o ADAS.

Responsabilidades:
    - receber lanes detectadas/tracked;
    - filtrar detecções inválidas;
    - ordenar lanes espacialmente;
    - identificar candidatos à faixa atual;
    - preservar todas as lanes válidas para as etapas seguintes.

IMPORTANTE:
    ROI não é tratado aqui.
    O ROI pertence exclusivamente ao pipeline de captura/config.py.

Este módulo não:
    - captura tela;
    - executa YOLOP;
    - altera coordenadas;
    - controla o G29;
    - decide o estado final do ADAS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple


# ============================================================================
# RESULTADOS
# ============================================================================

@dataclass(frozen=True)
class LaneSelectionResult:
    """
    Resultado da seleção espacial das lanes.
    """

    lanes: Tuple[Any, ...]
    current_candidates: Tuple[Any, ...]
    left_candidates: Tuple[Any, ...]
    right_candidates: Tuple[Any, ...]
    valid: bool

    @property
    def count(self) -> int:
        return len(self.lanes)


# ============================================================================
# HELPERS
# ============================================================================

def _get_value(obj: Any, *names: str, default: Any = None) -> Any:
    """
    Obtém um atributo de forma compatível com diferentes versões dos
    modelos LaneLine/LaneTrack.
    """

    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)

        if isinstance(obj, dict) and name in obj:
            return obj[name]

    return default


def _lane_points(lane: Any) -> Sequence[Any]:
    """
    Obtém os pontos de uma lane.
    """

    points = _get_value(
        lane,
        "points",
        "lane_points",
        "screen_points",
        "pixels",
        default=(),
    )

    if points is None:
        return ()

    return points


def _point_xy(point: Any) -> Optional[Tuple[float, float]]:
    """
    Converte um ponto para (x, y).
    """

    if point is None:
        return None

    if isinstance(point, dict):
        x = point.get("x")
        y = point.get("y")
    else:
        x = getattr(point, "x", None)
        y = getattr(point, "y", None)

        if x is None and isinstance(point, (tuple, list)) and len(point) >= 2:
            x = point[0]
            y = point[1]

    if x is None or y is None:
        return None

    try:
        return float(x), float(y)
    except (TypeError, ValueError):
        return None


def _lane_reference_x(lane: Any) -> Optional[float]:
    """
    Obtém o X representativo da lane.

    Preferência:
        1. atributo explícito de posição;
        2. ponto mais baixo da lane.

    O ponto mais baixo é utilizado porque representa melhor a posição
    da faixa próxima ao veículo.
    """

    explicit_x = _get_value(
        lane,
        "reference_x",
        "center_x",
        "bottom_x",
        "x",
        default=None,
    )

    if explicit_x is not None:
        try:
            return float(explicit_x)
        except (TypeError, ValueError):
            pass

    points = _lane_points(lane)

    best: Optional[Tuple[float, float]] = None

    for point in points:
        xy = _point_xy(point)

        if xy is None:
            continue

        if best is None or xy[1] > best[1]:
            best = xy

    if best is None:
        return None

    return best[0]


def _lane_confidence(lane: Any) -> float:
    """
    Obtém confiança da lane.
    """

    value = _get_value(
        lane,
        "confidence",
        "score",
        "probability",
        default=0.0,
    )

    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _is_valid_lane(lane: Any, minimum_confidence: float) -> bool:
    """
    Valida uma lane sem modificar seus dados.
    """

    if lane is None:
        return False

    points = _lane_points(lane)

    if points is None or len(points) < 2:
        return False

    x = _lane_reference_x(lane)

    if x is None:
        return False

    return _lane_confidence(lane) >= minimum_confidence


# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

@dataclass(frozen=True)
class LaneSelectorConfig:
    """
    Configuração do seletor.

    Não contém ROI.
    """

    minimum_confidence: float = 0.35

    minimum_lane_separation: float = 40.0

    center_reference_ratio: float = 0.50

    maximum_candidates: int = 16

    enable_confidence_filter: bool = True


# ============================================================================
# SELECTOR
# ============================================================================

class LaneSelector:
    """
    Seleciona e organiza lanes para as etapas posteriores do ADAS.

    O seletor trabalha somente com as coordenadas que recebe.

    Nenhuma dimensão de tela ou ROI é assumida internamente.
    """

    def __init__(
        self,
        config: Optional[LaneSelectorConfig] = None,
    ) -> None:

        self.config = config or LaneSelectorConfig()

        if self.config.minimum_confidence < 0.0:
            raise ValueError(
                "minimum_confidence must be >= 0"
            )

        if self.config.minimum_confidence > 1.0:
            raise ValueError(
                "minimum_confidence must be <= 1"
            )

        if self.config.minimum_lane_separation < 0.0:
            raise ValueError(
                "minimum_lane_separation must be >= 0"
            )

        if self.config.maximum_candidates <= 0:
            raise ValueError(
                "maximum_candidates must be > 0"
            )

    # ----------------------------------------------------------------------
    # PUBLIC
    # ----------------------------------------------------------------------

    def select(
        self,
        lanes: Optional[Iterable[Any]],
        image_width: Optional[float] = None,
    ) -> LaneSelectionResult:
        """
        Seleciona lanes válidas.

        image_width é opcional e serve apenas para determinar o centro
        espacial da imagem recebida.

        Não representa ROI nem altera coordenadas.
        """

        if lanes is None:
            return LaneSelectionResult(
                lanes=(),
                current_candidates=(),
                left_candidates=(),
                right_candidates=(),
                valid=False,
            )

        valid_lanes: List[Any] = []

        for lane in lanes:
            if self.config.enable_confidence_filter:
                if not _is_valid_lane(
                    lane,
                    self.config.minimum_confidence,
                ):
                    continue
            else:
                if lane is None or _lane_reference_x(lane) is None:
                    continue

            valid_lanes.append(lane)

        valid_lanes.sort(
            key=lambda lane: _lane_reference_x(lane)
            if _lane_reference_x(lane) is not None
            else float("inf")
        )

        valid_lanes = valid_lanes[
            : self.config.maximum_candidates
        ]

        if not valid_lanes:
            return LaneSelectionResult(
                lanes=(),
                current_candidates=(),
                left_candidates=(),
                right_candidates=(),
                valid=False,
            )

        center_x = self._calculate_reference_center(
            valid_lanes,
            image_width,
        )

        left: List[Any] = []
        right: List[Any] = []
        current: List[Any] = []

        for lane in valid_lanes:
            x = _lane_reference_x(lane)

            if x is None:
                continue

            if abs(x - center_x) <= self.config.minimum_lane_separation:
                current.append(lane)
            elif x < center_x:
                left.append(lane)
            else:
                right.append(lane)

        return LaneSelectionResult(
            lanes=tuple(valid_lanes),
            current_candidates=tuple(current),
            left_candidates=tuple(left),
            right_candidates=tuple(right),
            valid=True,
        )

    # ----------------------------------------------------------------------
    # CENTER
    # ----------------------------------------------------------------------

    def _calculate_reference_center(
        self,
        lanes: Sequence[Any],
        image_width: Optional[float],
    ) -> float:
        """
        Calcula o centro de referência.

        Se a largura da imagem estiver disponível, usa-a.

        Caso contrário, utiliza o centro espacial das lanes detectadas.
        """

        if image_width is not None:
            try:
                width = float(image_width)

                if width > 0.0:
                    return (
                        width
                        * self.config.center_reference_ratio
                    )
            except (TypeError, ValueError):
                pass

        xs = [
            _lane_reference_x(lane)
            for lane in lanes
            if _lane_reference_x(lane) is not None
        ]

        if not xs:
            return 0.0

        return (min(xs) + max(xs)) * 0.5

    # ----------------------------------------------------------------------
    # STATIC HELPERS
    # ----------------------------------------------------------------------

    @staticmethod
    def sort_by_position(
        lanes: Iterable[Any],
    ) -> Tuple[Any, ...]:
        """
        Ordena lanes da esquerda para a direita.
        """

        valid = [
            lane
            for lane in lanes
            if _lane_reference_x(lane) is not None
        ]

        valid.sort(
            key=lambda lane: _lane_reference_x(lane)
        )

        return tuple(valid)

    @staticmethod
    def find_nearest_to_center(
        lanes: Iterable[Any],
        center_x: float,
    ) -> Optional[Any]:
        """
        Retorna a lane mais próxima do centro fornecido.
        """

        best_lane = None
        best_distance = float("inf")

        for lane in lanes:
            x = _lane_reference_x(lane)

            if x is None:
                continue

            distance = abs(x - center_x)

            if distance < best_distance:
                best_distance = distance
                best_lane = lane

        return best_lane


# ============================================================================
# FACTORY
# ============================================================================

def create_lane_selector(
    config: Optional[LaneSelectorConfig] = None,
) -> LaneSelector:
    """
    Cria o seletor padrão.
    """

    return LaneSelector(config=config)


__all__ = [
    "LaneSelector",
    "LaneSelectorConfig",
    "LaneSelectionResult",
    "create_lane_selector",
]