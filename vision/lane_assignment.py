"""
vision/lane_assignment.py

Associação semântica das linhas de faixa.

Responsabilidade:

    LaneProjection
        ↓
    ordenação espacial das linhas
        ↓
    construção dos corredores
        ↓
    identificação da faixa atual
        ↓
    identificação das faixas adjacentes
        ↓
    LaneAssignmentResult

IMPORTANTE
----------
YOLOP / Tracker / Geometry / Projection identificam e modelam LINHAS.

Este módulo identifica FAIXAS.

Exemplo:

    L0        L1        L2        L3
     │         │         │         │
     │   C0    │   C1    │   C2    │
     │         │         │         │

Com 4 linhas existem 3 corredores possíveis.

O veículo ocupa um desses corredores.

Este módulo NÃO:
- executa YOLOP;
- realiza tracking;
- ajusta polinômios;
- projeta linhas;
- toma decisões ADAS;
- controla o veículo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# =============================================================================
# RESULTADO
# =============================================================================


@dataclass(frozen=True)
class LaneAssignmentResult:
    """
    Resultado da associação semântica das linhas.

    lanes:
        Linhas ordenadas da esquerda para a direita.

    current_lane_index:
        Índice da faixa atual no conjunto de corredores.

        Exemplo com 4 linhas:

            L0 | L1 | L2 | L3

        corredores:

            0 = L0-L1
            1 = L1-L2
            2 = L2-L3

    left_boundary_index:
        Índice da linha esquerda da faixa atual.

    right_boundary_index:
        Índice da linha direita da faixa atual.

    left_lanes:
        Linhas/corredores à esquerda da faixa atual.

    right_lanes:
        Linhas/corredores à direita da faixa atual.

    lane_center_x:
        Centro da faixa atual na referência usada.

    vehicle_x:
        Centro/referência lateral do veículo.

    lateral_offset:
        Offset lateral normalizado.

        < 0 = esquerda
        > 0 = direita

    lane_width:
        Distância entre as duas linhas que delimitam a faixa.

    confidence:
        Confiança da associação.

    valid:
        Indica se a associação é utilizável.
    """

    lanes: Tuple[Any, ...] = ()

    current_lane_index: int = -1

    left_boundary_index: int = -1
    right_boundary_index: int = -1

    left_lanes: Tuple[Any, ...] = ()
    right_lanes: Tuple[Any, ...] = ()

    lane_center_x: float = 0.0
    vehicle_x: float = 0.0

    lateral_offset: float = 0.0
    normalized_offset: float = 0.0

    lane_width: float = 0.0

    confidence: float = 0.0

    valid: bool = False

    reason: Optional[str] = None

    @property
    def current_lane(self) -> Optional[Any]:
        """Retorna a faixa atual quando disponível."""
        if self.current_lane_index < 0:
            return None

        if self.current_lane_index >= len(self.lanes) - 1:
            return None

        return (
            self.lanes[self.current_lane_index],
            self.lanes[self.current_lane_index + 1],
        )

    @property
    def left_boundary(self) -> Optional[Any]:
        """Linha esquerda da faixa atual."""
        if self.left_boundary_index < 0:
            return None

        if self.left_boundary_index >= len(self.lanes):
            return None

        return self.lanes[self.left_boundary_index]

    @property
    def right_boundary(self) -> Optional[Any]:
        """Linha direita da faixa atual."""
        if self.right_boundary_index < 0:
            return None

        if self.right_boundary_index >= len(self.lanes):
            return None

        return self.lanes[self.right_boundary_index]

    @property
    def lane_count(self) -> int:
        """Quantidade de linhas disponíveis."""
        return len(self.lanes)

    @property
    def corridor_count(self) -> int:
        """Quantidade de corredores possíveis."""
        return max(0, len(self.lanes) - 1)


# =============================================================================
# UTILITÁRIOS
# =============================================================================


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """Converte valor para float sem permitir NaN/inf."""

    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    if not np.isfinite(result):
        return default

    return result


def _clip01(value: float) -> float:
    """Limita valor ao intervalo [0, 1]."""

    return float(
        np.clip(
            _safe_float(value),
            0.0,
            1.0,
        )
    )


def _get_attr(
    obj: Any,
    names: Sequence[str],
    default: Any = None,
) -> Any:
    """
    Obtém o primeiro atributo disponível.

    Isso mantém o assignment compatível com diferentes versões
    de LaneProjection / LaneModel.
    """

    if obj is None:
        return default

    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)

            if value is not None:
                return value

    return default


def _extract_points(obj: Any) -> list[Any]:
    """Obtém pontos de um objeto de projeção/modelo."""

    points = _get_attr(
        obj,
        (
            "points",
            "projected_points",
            "projection_points",
        ),
        None,
    )

    if points is None:
        return []

    try:
        return list(points)
    except TypeError:
        return []


def _point_x(point: Any) -> Optional[float]:
    """Obtém X de um ponto."""

    value = _get_attr(
        point,
        (
            "x",
            "screen_x",
            "image_x",
        ),
        None,
    )

    if value is None:
        return None

    result = _safe_float(
        value,
        float("nan"),
    )

    if not np.isfinite(result):
        return None

    return result


def _point_y(point: Any) -> Optional[float]:
    """Obtém Y de um ponto."""

    value = _get_attr(
        point,
        (
            "y",
            "screen_y",
            "image_y",
        ),
        None,
    )

    if value is None:
        return None

    result = _safe_float(
        value,
        float("nan"),
    )

    if not np.isfinite(result):
        return None

    return result


# =============================================================================
# ENGINE
# =============================================================================


class LaneAssignmentEngine:
    """
    Motor de associação semântica.

    O algoritmo principal é:

        1. obter posição representativa de cada linha;
        2. ordenar esquerda → direita;
        3. formar corredores entre linhas adjacentes;
        4. determinar qual corredor contém o veículo;
        5. calcular offset relativo ao centro;
        6. identificar corredores vizinhos;
        7. aplicar continuidade temporal.

    A posição do veículo é definida por ``vehicle_x``.

    Se não for fornecida explicitamente, o engine utiliza
    ``image_center_x``.
    """

    def __init__(
        self,
        expected_lane_width: float = 312.0,
        lane_width_tolerance: float = 0.45,
        minimum_confidence: float = 0.40,
        minimum_lane_separation: float = 80.0,
        maximum_lane_separation: float = 900.0,
        maximum_lateral_offset_ratio: float = 1.25,
        center_reference_ratio: float = 0.50,
        max_left_lanes: int = 8,
        max_right_lanes: int = 8,
        enable_multi_lane_assignment: bool = True,
    ) -> None:

        self.expected_lane_width = max(
            1.0,
            float(expected_lane_width),
        )

        self.lane_width_tolerance = max(
            0.0,
            float(lane_width_tolerance),
        )

        self.minimum_confidence = _clip01(
            minimum_confidence
        )

        self.minimum_lane_separation = max(
            0.0,
            float(minimum_lane_separation),
        )

        self.maximum_lane_separation = max(
            self.minimum_lane_separation,
            float(maximum_lane_separation),
        )

        self.maximum_lateral_offset_ratio = max(
            0.1,
            float(maximum_lateral_offset_ratio),
        )

        self.center_reference_ratio = float(
            np.clip(
                center_reference_ratio,
                0.0,
                1.0,
            )
        )

        self.max_left_lanes = max(
            0,
            int(max_left_lanes),
        )

        self.max_right_lanes = max(
            0,
            int(max_right_lanes),
        )

        self.enable_multi_lane_assignment = bool(
            enable_multi_lane_assignment
        )

        # ------------------------------------------------------------------
        # Histórico
        # ------------------------------------------------------------------

        self._previous_lane_index: Optional[int] = None

    # =========================================================================
    # POSIÇÃO DA LINHA
    # =========================================================================

    def _line_x(
        self,
        lane: Any,
        reference_y: Optional[float] = None,
    ) -> Optional[float]:
        """
        Obtém a posição X representativa da linha.

        Prioridade:

        1. método/atributo explícito em reference_x;
        2. ponto próximo ao reference_y;
        3. ponto inferior da projeção;
        4. centro médio dos pontos;
        5. atributos x/center_x.
        """

        # ------------------------------------------------------------------
        # X explícito
        # ------------------------------------------------------------------

        explicit_x = _get_attr(
            lane,
            (
                "reference_x",
                "bottom_x",
                "near_x",
                "x_at_bottom",
                "center_x",
            ),
            None,
        )

        if explicit_x is not None:
            value = _safe_float(
                explicit_x,
                float("nan"),
            )

            if np.isfinite(value):
                return value

        # ------------------------------------------------------------------
        # Pontos
        # ------------------------------------------------------------------

        points = _extract_points(lane)

        valid_points: list[tuple[float, float]] = []

        for point in points:
            x = _point_x(point)
            y = _point_y(point)

            if x is None or y is None:
                continue

            valid_points.append(
                (x, y)
            )

        if valid_points:

            # --------------------------------------------------------------
            # Procurar ponto mais próximo da referência inferior
            # --------------------------------------------------------------

            if reference_y is not None:

                try:
                    target_y = float(reference_y)

                    nearest = min(
                        valid_points,
                        key=lambda item: abs(
                            item[1] - target_y
                        ),
                    )

                    return float(nearest[0])

                except (TypeError, ValueError):
                    pass

            # --------------------------------------------------------------
            # Sem referência:
            # utilizar o ponto de maior Y.
            # --------------------------------------------------------------

            bottom = max(
                valid_points,
                key=lambda item: item[1],
            )

            return float(bottom[0])

        # ------------------------------------------------------------------
        # Fallbacks
        # ------------------------------------------------------------------

        value = _get_attr(
            lane,
            (
                "x",
                "screen_x",
            ),
            None,
        )

        if value is not None:
            result = _safe_float(
                value,
                float("nan"),
            )

            if np.isfinite(result):
                return result

        return None

    # =========================================================================
    # CONFIANÇA DA LINHA
    # =========================================================================

    def _line_confidence(
        self,
        lane: Any,
    ) -> float:
        """Obtém confiança da linha."""

        value = _get_attr(
            lane,
            (
                "confidence",
                "score",
                "lane_confidence",
            ),
            1.0,
        )

        return _clip01(
            _safe_float(
                value,
                1.0,
            )
        )

    # =========================================================================
    # ORDENAÇÃO
    # =========================================================================

    def _prepare_lanes(
        self,
        lanes: Iterable[Any],
        reference_y: Optional[float],
    ) -> list[tuple[Any, float, float]]:
        """
        Converte as linhas para:

            (lane, x, confidence)

        e ordena esquerda → direita.
        """

        prepared = []

        if lanes is None:
            return prepared

        try:
            iterator = iter(lanes)
        except TypeError:
            return prepared

        for lane in iterator:

            if lane is None:
                continue

            x = self._line_x(
                lane,
                reference_y,
            )

            if x is None:
                continue

            confidence = self._line_confidence(
                lane
            )

            prepared.append(
                (
                    lane,
                    float(x),
                    confidence,
                )
            )

        prepared.sort(
            key=lambda item: item[1]
        )

        return prepared

    # =========================================================================
    # VEÍCULO
    # =========================================================================

    @staticmethod
    def _vehicle_x(
        vehicle_x: Optional[float],
        image_center_x: Optional[float],
    ) -> Optional[float]:

        if vehicle_x is not None:
            value = _safe_float(
                vehicle_x,
                float("nan"),
            )

            if np.isfinite(value):
                return value

        if image_center_x is not None:
            value = _safe_float(
                image_center_x,
                float("nan"),
            )

            if np.isfinite(value):
                return value

        return None

    # =========================================================================
    # CORREDORES
    # =========================================================================

    def _build_corridors(
        self,
        prepared: list[tuple[Any, float, float]],
    ) -> list[dict[str, Any]]:
        """
        Constrói corredores entre linhas adjacentes.

        Exemplo:

            L0 L1 L2 L3

        produz:

            C0 = L0-L1
            C1 = L1-L2
            C2 = L2-L3
        """

        corridors: list[dict[str, Any]] = []

        if len(prepared) < 2:
            return corridors

        for index in range(
            len(prepared) - 1
        ):

            left_lane, left_x, left_conf = (
                prepared[index]
            )

            right_lane, right_x, right_conf = (
                prepared[index + 1]
            )

            width = right_x - left_x

            if width <= 0:
                continue

            confidence = min(
                left_conf,
                right_conf,
            )

            width_error = abs(
                width - self.expected_lane_width
            ) / self.expected_lane_width

            width_score = max(
                0.0,
                1.0
                - (
                    width_error
                    / max(
                        self.lane_width_tolerance,
                        1e-6,
                    )
                ),
            )

            corridors.append(
                {
                    "index": index,
                    "left_index": index,
                    "right_index": index + 1,
                    "left_lane": left_lane,
                    "right_lane": right_lane,
                    "left_x": left_x,
                    "right_x": right_x,
                    "center_x": (
                        left_x
                        + (
                            width
                            * self.center_reference_ratio
                        )
                    ),
                    "width": width,
                    "confidence": confidence,
                    "width_score": _clip01(
                        width_score
                    ),
                }
            )

        return corridors

    # =========================================================================
    # SELEÇÃO DO CORREDOR
    # =========================================================================

    def _select_corridor(
        self,
        corridors: list[dict[str, Any]],
        vehicle_x: float,
    ) -> Optional[dict[str, Any]]:
        """
        Seleciona o corredor que contém o veículo.

        Regra principal:

            left_x <= vehicle_x <= right_x

        Isso é fundamental.

        Não escolhemos a linha mais próxima.

        Escolhemos o INTERVALO entre duas linhas.
        """

        containing = []

        for corridor in corridors:

            left_x = corridor["left_x"]
            right_x = corridor["right_x"]

            if (
                left_x
                <= vehicle_x
                <= right_x
            ):
                containing.append(
                    corridor
                )

        if containing:

            # Em condições normais haverá somente um.
            # Se houver sobreposição, prefere o corredor
            # cujo centro esteja mais próximo do veículo.

            return min(
                containing,
                key=lambda corridor: abs(
                    corridor["center_x"]
                    - vehicle_x
                ),
            )

        # ------------------------------------------------------------------
        # Fallback:
        # veículo ligeiramente fora de uma faixa.
        # ------------------------------------------------------------------

        if not corridors:
            return None

        nearest = min(
            corridors,
            key=lambda corridor: min(
                abs(
                    vehicle_x
                    - corridor["left_x"]
                ),
                abs(
                    vehicle_x
                    - corridor["right_x"]
                ),
            ),
        )

        width = nearest["width"]

        if width <= 0:
            return None

        offset_ratio = abs(
            vehicle_x
            - nearest["center_x"]
        ) / (
            width * 0.5
        )

        if (
            offset_ratio
            <= self.maximum_lateral_offset_ratio
        ):
            return nearest

        return None

    # =========================================================================
    # CONTINUIDADE TEMPORAL
    # =========================================================================

    def _apply_temporal_continuity(
        self,
        selected: Optional[dict[str, Any]],
        corridors: list[dict[str, Any]],
        vehicle_x: float,
    ) -> Optional[dict[str, Any]]:
        """
        Mantém a faixa atual estável entre frames.

        O histórico só é utilizado quando ainda existe
        um corredor plausível.

        Não força uma faixa antiga quando ela claramente
        não contém mais o veículo.
        """

        if selected is None:
            self._previous_lane_index = None
            return None

        current_index = int(
            selected["index"]
        )

        previous = self._previous_lane_index

        if previous is None:
            self._previous_lane_index = current_index
            return selected

        # --------------------------------------------------------------
        # Mesma faixa
        # --------------------------------------------------------------

        if current_index == previous:
            self._previous_lane_index = current_index
            return selected

        # --------------------------------------------------------------
        # Mudança para corredor adjacente:
        # permitido.
        # --------------------------------------------------------------

        if abs(
            current_index - previous
        ) <= 1:
            self._previous_lane_index = current_index
            return selected

        # --------------------------------------------------------------
        # Mudança grande:
        # procurar o corredor anterior.
        # --------------------------------------------------------------

        previous_corridor = None

        for corridor in corridors:
            if (
                int(corridor["index"])
                == previous
            ):
                previous_corridor = corridor
                break

        if previous_corridor is not None:

            left = previous_corridor["left_x"]
            right = previous_corridor["right_x"]

            if (
                left
                <= vehicle_x
                <= right
            ):
                return previous_corridor

        self._previous_lane_index = current_index

        return selected

    # =========================================================================
    # UPDATE
    # =========================================================================

    def assign(
        self,
        lanes: Iterable[Any],
        vehicle_x: Optional[float] = None,
        image_center_x: Optional[float] = None,
        reference_y: Optional[float] = None,
    ) -> LaneAssignmentResult:
        """
        Executa a associação.

        Parâmetros
        ----------
        lanes:
            Linhas/projeções.

        vehicle_x:
            Centro lateral do veículo.

        image_center_x:
            Centro da imagem caso vehicle_x não seja fornecido.

        reference_y:
            Y utilizado para determinar a posição das linhas.
        """

        prepared = self._prepare_lanes(
            lanes,
            reference_y,
        )

        ordered_lanes = tuple(
            item[0]
            for item in prepared
        )

        # ------------------------------------------------------------------
        # Linhas insuficientes
        # ------------------------------------------------------------------

        if len(prepared) < 2:

            self._previous_lane_index = None

            return LaneAssignmentResult(
                lanes=ordered_lanes,
                valid=False,
                reason=(
                    "São necessárias pelo menos "
                    "duas linhas para formar uma faixa."
                ),
            )

        # ------------------------------------------------------------------
        # Veículo
        # ------------------------------------------------------------------

        resolved_vehicle_x = self._vehicle_x(
            vehicle_x,
            image_center_x,
        )

        if resolved_vehicle_x is None:

            return LaneAssignmentResult(
                lanes=ordered_lanes,
                valid=False,
                reason=(
                    "Não foi possível determinar "
                    "a posição lateral do veículo."
                ),
            )

        # ------------------------------------------------------------------
        # Corredores
        # ------------------------------------------------------------------

        corridors = self._build_corridors(
            prepared
        )

        if not corridors:

            self._previous_lane_index = None

            return LaneAssignmentResult(
                lanes=ordered_lanes,
                vehicle_x=resolved_vehicle_x,
                valid=False,
                reason=(
                    "Nenhum corredor válido "
                    "foi formado entre as linhas."
                ),
            )

        # ------------------------------------------------------------------
        # Seleção
        # ------------------------------------------------------------------

        selected = self._select_corridor(
            corridors,
            resolved_vehicle_x,
        )

        selected = self._apply_temporal_continuity(
            selected,
            corridors,
            resolved_vehicle_x,
        )

        if selected is None:

            return LaneAssignmentResult(
                lanes=ordered_lanes,
                vehicle_x=resolved_vehicle_x,
                valid=False,
                reason=(
                    "O veículo não está dentro "
                    "de nenhum corredor plausível."
                ),
            )

        # ------------------------------------------------------------------
        # Índice
        # ------------------------------------------------------------------

        lane_index = int(
            selected["index"]
        )

        left_boundary_index = int(
            selected["left_index"]
        )

        right_boundary_index = int(
            selected["right_index"]
        )

        # ------------------------------------------------------------------
        # Offset
        # ------------------------------------------------------------------

        lane_width = float(
            selected["width"]
        )

        lane_center_x = float(
            selected["center_x"]
        )

        lateral_offset = (
            resolved_vehicle_x
            - lane_center_x
        )

        half_width = max(
            lane_width * 0.5,
            1.0,
        )

        normalized_offset = float(
            np.clip(
                lateral_offset
                / half_width,
                -1.0,
                1.0,
            )
        )

        # ------------------------------------------------------------------
        # Confiança
        # ------------------------------------------------------------------

        confidence = (
            float(selected["confidence"])
            * (
                0.65
                + 0.35
                * float(
                    selected["width_score"]
                )
            )
        )

        confidence = _clip01(
            confidence
        )

        valid = (
            confidence
            >= self.minimum_confidence
        )

        # ------------------------------------------------------------------
        # Linhas à esquerda/direita
        # ------------------------------------------------------------------

        left_lanes = ordered_lanes[
            :left_boundary_index
        ]

        right_lanes = ordered_lanes[
            right_boundary_index + 1:
        ]

        # ------------------------------------------------------------------
        # Limitação
        # ------------------------------------------------------------------

        left_lanes = left_lanes[
            -self.max_left_lanes:
        ]

        right_lanes = right_lanes[
            :self.max_right_lanes
        ]

        return LaneAssignmentResult(
            lanes=ordered_lanes,

            current_lane_index=lane_index,

            left_boundary_index=(
                left_boundary_index
            ),

            right_boundary_index=(
                right_boundary_index
            ),

            left_lanes=tuple(
                left_lanes
            ),

            right_lanes=tuple(
                right_lanes
            ),

            lane_center_x=lane_center_x,

            vehicle_x=resolved_vehicle_x,

            lateral_offset=lateral_offset,

            normalized_offset=normalized_offset,

            lane_width=lane_width,

            confidence=confidence,

            valid=valid,

            reason=(
                None
                if valid
                else (
                    "Confiança da associação "
                    "abaixo do mínimo."
                )
            ),
        )

    # =========================================================================
    # COMPATIBILIDADE
    # =========================================================================

    def update(
        self,
        lanes: Iterable[Any],
        vehicle_x: Optional[float] = None,
        image_center_x: Optional[float] = None,
        reference_y: Optional[float] = None,
    ) -> LaneAssignmentResult:
        """
        Alias de assign().

        Mantido para compatibilidade com chamadas existentes.
        """

        return self.assign(
            lanes=lanes,
            vehicle_x=vehicle_x,
            image_center_x=image_center_x,
            reference_y=reference_y,
        )


# =============================================================================
# FACTORY
# =============================================================================


def create_default_lane_assignment(
    config: Optional[Any] = None,
) -> LaneAssignmentEngine:
    """
    Cria o assignment utilizando a configuração global.

    Aceita config opcional para manter compatibilidade
    com diferentes versões do projeto.
    """

    if config is None:

        try:
            from config import LANE_ASSIGNMENT

            config = LANE_ASSIGNMENT

        except ImportError:
            config = None

    if config is None:

        return LaneAssignmentEngine()

    return LaneAssignmentEngine(
        expected_lane_width=_safe_float(
            getattr(
                config,
                "expected_lane_width",
                312.0,
            ),
            312.0,
        ),

        lane_width_tolerance=_safe_float(
            getattr(
                config,
                "lane_width_tolerance",
                0.45,
            ),
            0.45,
        ),

        minimum_confidence=_safe_float(
            getattr(
                config,
                "minimum_confidence",
                0.40,
            ),
            0.40,
        ),

        minimum_lane_separation=_safe_float(
            getattr(
                config,
                "minimum_lane_separation",
                80.0,
            ),
            80.0,
        ),

        maximum_lane_separation=_safe_float(
            getattr(
                config,
                "maximum_lane_separation",
                900.0,
            ),
            900.0,
        ),

        maximum_lateral_offset_ratio=_safe_float(
            getattr(
                config,
                "maximum_lateral_offset_ratio",
                1.25,
            ),
            1.25,
        ),

        center_reference_ratio=_safe_float(
            getattr(
                config,
                "center_reference_ratio",
                0.50,
            ),
            0.50,
        ),

        max_left_lanes=int(
            getattr(
                config,
                "max_left_lanes",
                8,
            )
        ),

        max_right_lanes=int(
            getattr(
                config,
                "max_right_lanes",
                8,
            )
        ),

        enable_multi_lane_assignment=bool(
            getattr(
                config,
                "enable_multi_lane_assignment",
                True,
            )
        ),
    )


# =============================================================================
# FUNÇÃO DE CONVENIÊNCIA
# =============================================================================


def assign_lanes(
    lanes: Iterable[Any],
    vehicle_x: Optional[float] = None,
    image_center_x: Optional[float] = None,
    reference_y: Optional[float] = None,
    engine: Optional[LaneAssignmentEngine] = None,
) -> LaneAssignmentResult:
    """
    Função simples para executar o assignment.

    Exemplo:

        result = assign_lanes(
            projections,
            image_center_x=960,
        )

    """

    if engine is None:
        engine = create_default_lane_assignment()

    return engine.assign(
        lanes=lanes,
        vehicle_x=vehicle_x,
        image_center_x=image_center_x,
        reference_y=reference_y,
    )


__all__ = [
    "LaneAssignment",
    "LaneAssignmentResult",
    "LaneAssignmentEngine",
    "create_default_lane_assignment",
    "assign_lanes",
]


class LaneAssignment(LaneAssignmentEngine):
    """
    Interface compatível com o main.py existente.
    """

    pass
