"""
core/lane_selector.py

Forza Assistents
================

Seleção determinística de lanes.

Responsabilidades
-----------------
- Validar candidatos.
- Filtrar lanes inválidas ou de baixa confiança.
- Ordenar lanes espacialmente.
- Identificar a lane mais provável à esquerda/direita do centro.
- Evitar seleção baseada em ROI.
- Preservar informação original dos candidatos.
- Produzir um resultado determinístico para as camadas seguintes.

Este módulo NÃO:
- captura frames;
- define ROI;
- modifica ROI;
- executa YOLOP;
- calcula geometria de pista;
- executa tracking;
- decide estado ADAS;
- envia comandos ao volante.

Arquitetura
-----------

    YOLOP / Detector
           │
           ▼
      Lane candidates
           │
           ▼
      LaneSelector
           │
           ▼
    SelectedLaneSet
           │
           ├── LaneGeometry
           ├── LaneTracker
           └── LaneAssignment

PRINCÍPIO
---------

O seletor trabalha exclusivamente nas coordenadas do frame
que recebe.

ROI pertence exclusivamente a config.py/capture.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from typing import Any, Iterable, Sequence

from config import LANE_SELECTOR


LOGGER = logging.getLogger(__name__)


# =============================================================================
# RESULT TYPES
# =============================================================================


@dataclass(frozen=True, slots=True)
class LaneCandidate:
    """
    Representação normalizada de uma lane candidata.

    O seletor não exige que o detector use exatamente esta classe.
    Objetos externos podem ser normalizados através de
    LaneSelector._normalize_candidate().
    """

    lane_id: int

    points: tuple[tuple[float, float], ...]

    confidence: float

    source: Any = None

    @property
    def point_count(self) -> int:
        return len(self.points)

    @property
    def bottom_x(self) -> float:
        """
        Estima a posição horizontal da lane no ponto mais baixo
        observado.
        """

        if not self.points:
            return math.nan

        return max(
            self.points,
            key=lambda point: point[1],
        )[0]

    @property
    def top_x(self) -> float:
        """
        Estima a posição horizontal da lane no ponto mais alto
        observado.
        """

        if not self.points:
            return math.nan

        return min(
            self.points,
            key=lambda point: point[1],
        )[0]

    @property
    def y_span(self) -> float:
        if len(self.points) < 2:
            return 0.0

        ys = [
            point[1]
            for point in self.points
        ]

        return max(ys) - min(ys)

    @property
    def is_valid(self) -> bool:
        return (
            self.point_count
            >= LANE_SELECTOR.minimum_confidence * 0
            and math.isfinite(
                self.confidence
            )
            and 0.0
            <= self.confidence
            <= 1.0
            and all(
                math.isfinite(x)
                and math.isfinite(y)
                for x, y in self.points
            )
        )


@dataclass(frozen=True, slots=True)
class SelectedLaneSet:
    """
    Resultado do processo de seleção.
    """

    lanes: tuple[LaneCandidate, ...]

    left_lanes: tuple[LaneCandidate, ...]

    right_lanes: tuple[LaneCandidate, ...]

    center_lane: LaneCandidate | None

    frame_width: float

    frame_height: float

    rejected_count: int

    confidence: float

    valid: bool

    reason: str = ""

    @property
    def lane_count(self) -> int:
        return len(self.lanes)

    @property
    def has_center_lane(self) -> bool:
        return self.center_lane is not None

    @property
    def has_left_lane(self) -> bool:
        return bool(self.left_lanes)

    @property
    def has_right_lane(self) -> bool:
        return bool(self.right_lanes)


# =============================================================================
# SELECTOR
# =============================================================================


class LaneSelector:
    """
    Selecionador determinístico de lanes.

    O seletor utiliza somente:

        candidates
        frame_width
        frame_height

    Não possui qualquer conhecimento de ROI.
    """

    def __init__(
        self,
        *,
        frame_width: float,
        frame_height: float,
    ) -> None:

        if frame_width <= 0:
            raise ValueError(
                "frame_width deve ser > 0."
            )

        if frame_height <= 0:
            raise ValueError(
                "frame_height deve ser > 0."
            )

        self.frame_width = float(
            frame_width
        )

        self.frame_height = float(
            frame_height
        )

        self._center_x = (
            self.frame_width
            * LANE_SELECTOR.center_reference_ratio
        )

    # =========================================================================
    # PUBLIC
    # =========================================================================

    def select(
        self,
        candidates: Iterable[Any],
    ) -> SelectedLaneSet:
        """
        Seleciona e classifica os candidatos.

        Processo:

            1. normalização
            2. validação
            3. filtragem
            4. ordenação espacial
            5. classificação esquerda/direita
            6. determinação da lane central
            7. cálculo da confiança
        """

        normalized: list[LaneCandidate] = []

        rejected = 0

        for index, candidate in enumerate(
            candidates
        ):

            try:

                lane = self._normalize_candidate(
                    candidate,
                    fallback_id=index,
                )

            except (
                TypeError,
                ValueError,
            ):

                rejected += 1

                continue

            if not self._is_candidate_valid(
                lane
            ):

                rejected += 1

                continue

            normalized.append(
                lane
            )

        if not normalized:

            return SelectedLaneSet(
                lanes=(),
                left_lanes=(),
                right_lanes=(),
                center_lane=None,
                frame_width=self.frame_width,
                frame_height=self.frame_height,
                rejected_count=rejected,
                confidence=0.0,
                valid=False,
                reason="no_valid_lanes",
            )

        ordered = self._sort_lanes(
            normalized
        )

        left_lanes, right_lanes = (
            self._classify_lanes(
                ordered
            )
        )

        center_lane = self._select_center_lane(
            ordered
        )

        confidence = (
            self._calculate_selection_confidence(
                ordered,
                center_lane,
            )
        )

        valid = (
            confidence
            >= LANE_SELECTOR.minimum_confidence
        )

        reason = (
            "valid"
            if valid
            else "low_selection_confidence"
        )

        return SelectedLaneSet(
            lanes=tuple(ordered),
            left_lanes=tuple(left_lanes),
            right_lanes=tuple(right_lanes),
            center_lane=center_lane,
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            rejected_count=rejected,
            confidence=confidence,
            valid=valid,
            reason=reason,
        )

    # =========================================================================
    # NORMALIZATION
    # =========================================================================

    @staticmethod
    def _normalize_candidate(
        candidate: Any,
        *,
        fallback_id: int,
    ) -> LaneCandidate:
        """
        Converte uma lane externa para LaneCandidate.

        Suporta objetos com:

            .points
            .confidence
            .lane_id

        ou dicionários equivalentes.
        """

        if isinstance(
            candidate,
            LaneCandidate,
        ):

            return candidate

        if isinstance(
            candidate,
            dict,
        ):

            points = candidate.get(
                "points"
            )

            confidence = candidate.get(
                "confidence",
                0.0,
            )

            lane_id = candidate.get(
                "lane_id",
                fallback_id,
            )

            source = candidate

        else:

            points = getattr(
                candidate,
                "points",
                None,
            )

            confidence = getattr(
                candidate,
                "confidence",
                0.0,
            )

            lane_id = getattr(
                candidate,
                "lane_id",
                fallback_id,
            )

            source = candidate

        if points is None:

            raise ValueError(
                "Lane não possui points."
            )

        normalized_points = []

        for point in points:

            if len(point) != 2:

                raise ValueError(
                    "Cada ponto deve possuir "
                    "(x, y)."
                )

            x, y = point

            normalized_points.append(
                (
                    float(x),
                    float(y),
                )
            )

        return LaneCandidate(
            lane_id=int(
                lane_id
            ),
            points=tuple(
                normalized_points
            ),
            confidence=float(
                confidence
            ),
            source=source,
        )

    # =========================================================================
    # VALIDATION
    # =========================================================================

    def _is_candidate_valid(
        self,
        lane: LaneCandidate,
    ) -> bool:

        if not lane.points:
            return False

        if lane.point_count < 2:
            return False

        if (
            lane.confidence
            < LANE_SELECTOR.minimum_confidence
        ):
            return False

        if not (
            0.0
            <= lane.confidence
            <= 1.0
        ):
            return False

        if not math.isfinite(
            lane.bottom_x
        ):
            return False

        if not math.isfinite(
            lane.y_span
        ):
            return False

        if lane.y_span <= 0.0:
            return False

        for x, y in lane.points:

            if not (
                math.isfinite(x)
                and math.isfinite(y)
            ):
                return False

            if x < 0.0:
                return False

            if y < 0.0:
                return False

            if x > self.frame_width:
                return False

            if y > self.frame_height:
                return False

        return True

    # =========================================================================
    # SORTING
    # =========================================================================

    @staticmethod
    def _sort_lanes(
        lanes: Sequence[LaneCandidate],
    ) -> list[LaneCandidate]:
        """
        Ordena lanes da esquerda para a direita.

        A posição inferior é preferida porque representa melhor
        a posição próxima ao veículo.
        """

        return sorted(
            lanes,
            key=lambda lane: (
                lane.bottom_x,
                -lane.confidence,
                lane.lane_id,
            ),
        )

    # =========================================================================
    # CLASSIFICATION
    # =========================================================================

    def _classify_lanes(
        self,
        lanes: Sequence[LaneCandidate],
    ) -> tuple[
        list[LaneCandidate],
        list[LaneCandidate],
    ]:

        left: list[LaneCandidate] = []
        right: list[LaneCandidate] = []

        for lane in lanes:

            x = lane.bottom_x

            if (
                x
                < self._center_x
            ):

                left.append(
                    lane
                )

            elif (
                x
                > self._center_x
            ):

                right.append(
                    lane
                )

        left.sort(
            key=lambda lane: (
                self._center_x
                - lane.bottom_x
            )
        )

        right.sort(
            key=lambda lane: (
                lane.bottom_x
                - self._center_x
            )
        )

        return left, right

    # =========================================================================
    # CENTER LANE
    # =========================================================================

    def _select_center_lane(
        self,
        lanes: Sequence[LaneCandidate],
    ) -> LaneCandidate | None:
        """
        Seleciona a lane mais próxima do centro da imagem.

        Esta função não afirma que essa lane é a faixa atual.
        A identificação da faixa atual pertence às camadas
        de geometria/assignment.
        """

        if not lanes:
            return None

        return min(
            lanes,
            key=lambda lane: (
                abs(
                    lane.bottom_x
                    - self._center_x
                ),
                -lane.confidence,
                -lane.y_span,
            ),
        )

    # =========================================================================
    # CONFIDENCE
    # =========================================================================

    @staticmethod
    def _calculate_selection_confidence(
        lanes: Sequence[LaneCandidate],
        center_lane: LaneCandidate | None,
    ) -> float:

        if not lanes:
            return 0.0

        detection_confidence = sum(
            lane.confidence
            for lane in lanes
        ) / len(lanes)

        if center_lane is None:

            center_confidence = 0.0

        else:

            center_confidence = (
                center_lane.confidence
            )

        confidence = (
            0.70
            * detection_confidence
            + 0.30
            * center_confidence
        )

        return max(
            0.0,
            min(
                1.0,
                confidence,
            ),
        )


# =============================================================================
# FACTORY
# =============================================================================


def create_lane_selector(
    *,
    frame_width: float,
    frame_height: float,
) -> LaneSelector:

    return LaneSelector(
        frame_width=frame_width,
        frame_height=frame_height,
    )


# =============================================================================
# PUBLIC API
# =============================================================================


__all__ = [
    "LaneCandidate",
    "SelectedLaneSet",
    "LaneSelector",
    "create_lane_selector",
]