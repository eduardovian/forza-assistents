"""
vision/vehicle_position.py

Estimativa da posição lateral do veículo dentro da faixa.

Responsabilidades:
    - localizar a faixa ocupada;
    - calcular centro da faixa;
    - calcular erro lateral;
    - normalizar o erro pela largura;
    - classificar a posição lateral;
    - fornecer confiança;
    - rejeitar estimativas insuficientes.

Não executa:
    - inferência;
    - tracking;
    - projeção;
    - associação;
    - decisão ADAS.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .lane_types import LanePoint

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

DEFAULT_EVALUATION_Y_RATIO = 0.82

DEFAULT_MIN_LANE_WIDTH = 30.0
DEFAULT_MAX_LANE_WIDTH = 1800.0

DEFAULT_CENTER_TOLERANCE = 0.10
DEFAULT_WARNING_TOLERANCE = 0.22
DEFAULT_CRITICAL_TOLERANCE = 0.38

DEFAULT_MIN_CONFIDENCE = 0.45
DEFAULT_MIN_POINTS = 4


# =============================================================================
# ESTADO
# =============================================================================

class VehiclePositionState(str, Enum):
    UNKNOWN = "unknown"

    CENTERED = "centered"

    LEFT = "left"
    RIGHT = "right"

    APPROACHING_LEFT = "approaching_left"
    APPROACHING_RIGHT = "approaching_right"

    WARNING_LEFT = "warning_left"
    WARNING_RIGHT = "warning_right"

    CRITICAL_LEFT = "critical_left"
    CRITICAL_RIGHT = "critical_right"


# =============================================================================
# RESULTADO
# =============================================================================

@dataclass
class VehiclePositionResult:
    lane_index: Optional[int]

    lane_center_x: float
    vehicle_center_x: float

    lateral_error: float
    normalized_error: float

    lane_width: float

    left_distance: float
    right_distance: float

    state: VehiclePositionState

    confidence: float
    valid: bool

    evaluation_y: float

    error: Optional[str] = None

    @property
    def offset(self) -> float:
        return self.normalized_error

    @property
    def error_pixels(self) -> float:
        return self.lateral_error

    @property
    def is_centered(self) -> bool:
        return (
            self.valid
            and self.state
            == VehiclePositionState.CENTERED
        )


# =============================================================================
# VEHICLE POSITION
# =============================================================================

class VehiclePosition:
    """
    Calcula a posição lateral do veículo dentro da faixa.

    Convenção:

        lateral_error > 0
            veículo à direita.

        lateral_error < 0
            veículo à esquerda.

        normalized_error:
            erro / (largura_da_faixa / 2)

        portanto:

            -1 = limite esquerdo
             0 = centro
            +1 = limite direito
    """

    def __init__(
        self,
        evaluation_y_ratio: float = DEFAULT_EVALUATION_Y_RATIO,
        min_lane_width: float = DEFAULT_MIN_LANE_WIDTH,
        max_lane_width: float = DEFAULT_MAX_LANE_WIDTH,
        center_tolerance: float = DEFAULT_CENTER_TOLERANCE,
        warning_tolerance: float = DEFAULT_WARNING_TOLERANCE,
        critical_tolerance: float = DEFAULT_CRITICAL_TOLERANCE,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        min_points: int = DEFAULT_MIN_POINTS,
        vehicle_x_offset: float = 0.0,
    ) -> None:

        self.evaluation_y_ratio = float(
            np.clip(
                evaluation_y_ratio,
                0.0,
                1.0,
            )
        )

        self.min_lane_width = max(
            1.0,
            float(min_lane_width),
        )

        self.max_lane_width = max(
            self.min_lane_width,
            float(max_lane_width),
        )

        self.center_tolerance = float(
            np.clip(
                center_tolerance,
                0.0,
                1.0,
            )
        )

        self.warning_tolerance = float(
            np.clip(
                warning_tolerance,
                self.center_tolerance,
                1.0,
            )
        )

        self.critical_tolerance = float(
            np.clip(
                critical_tolerance,
                self.warning_tolerance,
                1.0,
            )
        )

        self.min_confidence = float(
            np.clip(
                min_confidence,
                0.0,
                1.0,
            )
        )

        self.min_points = max(
            2,
            int(min_points),
        )

        self.vehicle_x_offset = float(
            vehicle_x_offset
        )

        self.last_result: Optional[
            VehiclePositionResult
        ] = None

    # =========================================================================
    # UTILIDADES
    # =========================================================================

    @staticmethod
    def _clip01(value: float) -> float:
        if not np.isfinite(value):
            return 0.0

        return float(
            np.clip(
                value,
                0.0,
                1.0,
            )
        )

    @staticmethod
    def _valid_points(
        points: Sequence[LanePoint],
    ) -> List[LanePoint]:

        result = []

        for point in points:

            if not point.valid:
                continue

            if not np.isfinite(point.x):
                continue

            if not np.isfinite(point.y):
                continue

            result.append(point)

        return result

    def _has_enough_points(
        self,
        points: Sequence[LanePoint],
    ) -> bool:

        return len(
            self._valid_points(points)
        ) >= self.min_points

    @classmethod
    def _interpolate_x(
        cls,
        points: Sequence[LanePoint],
        y: float,
    ) -> Optional[float]:

        valid = cls._valid_points(points)

        if len(valid) < 2:
            return None

        valid.sort(
            key=lambda point: point.y
        )

        ys = np.asarray(
            [point.y for point in valid],
            dtype=np.float64,
        )

        xs = np.asarray(
            [point.x for point in valid],
            dtype=np.float64,
        )

        unique_y, indices = np.unique(
            ys,
            return_index=True,
        )

        ys = unique_y
        xs = xs[indices]

        if len(ys) < 2:
            return None

        if y < ys[0] or y > ys[-1]:
            return None

        return float(
            np.interp(
                y,
                ys,
                xs,
            )
        )

    @classmethod
    def _lane_confidence(
        cls,
        points: Sequence[LanePoint],
    ) -> float:

        valid = cls._valid_points(points)

        if not valid:
            return 0.0

        values = []

        for point in valid:

            value = float(
                point.confidence
            )

            if np.isfinite(value):
                values.append(
                    np.clip(
                        value,
                        0.0,
                        1.0,
                    )
                )

        if not values:
            return 0.0

        return float(
            np.mean(values)
        )

    # =========================================================================
    # ESCOLHA DA FAIXA
    # =========================================================================

    def _find_current_lane(
        self,
        lanes: Sequence[
            Sequence[LanePoint]
        ],
        vehicle_x: float,
        evaluation_y: float,
    ) -> Tuple[
        Optional[int],
        Optional[float],
        Optional[float],
        float,
    ]:

        if len(lanes) < 2:
            return None, None, None, 0.0

        candidates = []

        for index in range(
            len(lanes) - 1
        ):

            left_lane = lanes[index]
            right_lane = lanes[index + 1]

            if not self._has_enough_points(
                left_lane
            ):
                continue

            if not self._has_enough_points(
                right_lane
            ):
                continue

            left_x = self._interpolate_x(
                left_lane,
                evaluation_y,
            )

            right_x = self._interpolate_x(
                right_lane,
                evaluation_y,
            )

            if (
                left_x is None
                or right_x is None
            ):
                continue

            if right_x < left_x:
                left_x, right_x = (
                    right_x,
                    left_x,
                )

            width = right_x - left_x

            if (
                width < self.min_lane_width
                or width > self.max_lane_width
            ):
                continue

            center_x = (
                left_x + right_x
            ) / 2.0

            confidence = (
                self._lane_confidence(
                    left_lane
                )
                + self._lane_confidence(
                    right_lane
                )
            ) / 2.0

            distance_to_interval = 0.0

            if vehicle_x < left_x:
                distance_to_interval = (
                    left_x - vehicle_x
                )
            elif vehicle_x > right_x:
                distance_to_interval = (
                    vehicle_x - right_x
                )

            distance_to_center = abs(
                vehicle_x - center_x
            )

            inside = (
                left_x
                <= vehicle_x
                <= right_x
            )

            candidates.append(
                (
                    0 if inside else 1,
                    distance_to_interval,
                    distance_to_center,
                    -confidence,
                    index,
                    left_x,
                    right_x,
                    confidence,
                )
            )

        if not candidates:
            return None, None, None, 0.0

        best = min(candidates)

        return (
            best[4],
            best[5],
            best[6],
            best[7],
        )

    # =========================================================================
    # CLASSIFICAÇÃO
    # =========================================================================

    def _classify_state(
        self,
        normalized_error: float,
    ) -> VehiclePositionState:

        magnitude = abs(
            normalized_error
        )

        if (
            magnitude
            <= self.center_tolerance
        ):
            return VehiclePositionState.CENTERED

        if normalized_error < 0.0:

            if (
                magnitude
                >= self.critical_tolerance
            ):
                return (
                    VehiclePositionState.CRITICAL_LEFT
                )

            if (
                magnitude
                >= self.warning_tolerance
            ):
                return (
                    VehiclePositionState.WARNING_LEFT
                )

            return (
                VehiclePositionState.APPROACHING_LEFT
            )

        if (
            magnitude
            >= self.critical_tolerance
        ):
            return (
                VehiclePositionState.CRITICAL_RIGHT
            )

        if (
            magnitude
            >= self.warning_tolerance
        ):
            return (
                VehiclePositionState.WARNING_RIGHT
            )

        return (
            VehiclePositionState.APPROACHING_RIGHT
        )

    # =========================================================================
    # CONFIANÇA
    # =========================================================================

    def _calculate_confidence(
        self,
        left_lane: Sequence[LanePoint],
        right_lane: Sequence[LanePoint],
        lane_width: float,
        image_width: int,
    ) -> float:

        left_confidence = (
            self._lane_confidence(
                left_lane
            )
        )

        right_confidence = (
            self._lane_confidence(
                right_lane
            )
        )

        detection_confidence = (
            left_confidence
            + right_confidence
        ) / 2.0

        left_count = len(
            self._valid_points(
                left_lane
            )
        )

        right_count = len(
            self._valid_points(
                right_lane
            )
        )

        point_confidence = self._clip01(
            min(
                left_count,
                right_count,
            )
            / max(
                self.min_points * 2,
                1,
            )
        )

        expected_width = (
            image_width * 0.35
        )

        if expected_width <= 0.0:
            width_confidence = 0.0
        else:
            width_ratio = (
                lane_width
                / expected_width
            )

            width_confidence = self._clip01(
                1.0
                - abs(
                    1.0 - width_ratio
                )
            )

        confidence = (
            0.65 * detection_confidence
            + 0.20 * point_confidence
            + 0.15 * width_confidence
        )

        return self._clip01(
            confidence
        )

    # =========================================================================
    # INVALID
    # =========================================================================

    @staticmethod
    def _invalid_result(
        vehicle_x: float,
        evaluation_y: float,
        error: str,
    ) -> VehiclePositionResult:

        return VehiclePositionResult(
            lane_index=None,
            lane_center_x=float("nan"),
            vehicle_center_x=float(
                vehicle_x
            ),
            lateral_error=float("nan"),
            normalized_error=float("nan"),
            lane_width=0.0,
            left_distance=float("nan"),
            right_distance=float("nan"),
            state=VehiclePositionState.UNKNOWN,
            confidence=0.0,
            valid=False,
            evaluation_y=float(
                evaluation_y
            ),
            error=error,
        )

    # =========================================================================
    # API
    # =========================================================================

    def estimate(
        self,
        lanes: Sequence[
            Sequence[LanePoint]
        ],
        image_width: int,
        image_height: int,
        vehicle_x: Optional[float] = None,
        evaluation_y: Optional[float] = None,
    ) -> VehiclePositionResult:

        if image_width <= 0:
            result = self._invalid_result(
                0.0,
                0.0,
                "image_width inválido.",
            )
            self.last_result = result
            return result

        if image_height <= 0:
            result = self._invalid_result(
                0.0,
                0.0,
                "image_height inválido.",
            )
            self.last_result = result
            return result

        if vehicle_x is None:
            vehicle_x = (
                image_width / 2.0
                + self.vehicle_x_offset
            )

        vehicle_x = float(vehicle_x)

        if not np.isfinite(vehicle_x):
            result = self._invalid_result(
                image_width / 2.0,
                0.0,
                "vehicle_x inválido.",
            )
            self.last_result = result
            return result

        if evaluation_y is None:
            evaluation_y = (
                image_height
                * self.evaluation_y_ratio
            )

        evaluation_y = float(
            evaluation_y
        )

        if not np.isfinite(evaluation_y):
            result = self._invalid_result(
                vehicle_x,
                0.0,
                "evaluation_y inválido.",
            )
            self.last_result = result
            return result

        evaluation_y = float(
            np.clip(
                evaluation_y,
                0.0,
                image_height - 1,
            )
        )

        try:

            lane_index, left_x, right_x, confidence = (
                self._find_current_lane(
                    lanes,
                    vehicle_x,
                    evaluation_y,
                )
            )

            if (
                lane_index is None
                or left_x is None
                or right_x is None
            ):
                result = self._invalid_result(
                    vehicle_x,
                    evaluation_y,
                    "Não foi possível determinar "
                    "a faixa atual.",
                )
                self.last_result = result
                return result

            lane_width = (
                right_x - left_x
            )

            if lane_width <= 0.0:
                result = self._invalid_result(
                    vehicle_x,
                    evaluation_y,
                    "Largura da faixa inválida.",
                )
                self.last_result = result
                return result

            lane_center_x = (
                left_x + right_x
            ) / 2.0

            lateral_error = (
                vehicle_x
                - lane_center_x
            )

            normalized_error = (
                lateral_error
                / (lane_width / 2.0)
            )

            normalized_error = float(
                np.clip(
                    normalized_error,
                    -1.5,
                    1.5,
                )
            )

            left_distance = (
                vehicle_x - left_x
            )

            right_distance = (
                right_x - vehicle_x
            )

            confidence = (
                self._calculate_confidence(
                    lanes[lane_index],
                    lanes[lane_index + 1],
                    lane_width,
                    image_width,
                )
            )

            # Mantém a confiança calculada internamente
            # como fonte principal.
            confidence = min(
                confidence,
                self._clip01(
                    confidence
                ),
            )

            state = self._classify_state(
                normalized_error
            )

            valid = (
                confidence
                >= self.min_confidence
            )

            if not valid:
                state = (
                    VehiclePositionState.UNKNOWN
                )

            result = VehiclePositionResult(
                lane_index=lane_index,
                lane_center_x=float(
                    lane_center_x
                ),
                vehicle_center_x=float(
                    vehicle_x
                ),
                lateral_error=float(
                    lateral_error
                ),
                normalized_error=float(
                    normalized_error
                ),
                lane_width=float(
                    lane_width
                ),
                left_distance=float(
                    left_distance
                ),
                right_distance=float(
                    right_distance
                ),
                state=state,
                confidence=float(
                    confidence
                ),
                valid=valid,
                evaluation_y=float(
                    evaluation_y
                ),
                error=None,
            )

            self.last_result = result

            return result

        except Exception as exc:

            error = (
                f"{type(exc).__name__}: {exc}"
            )

            logger.exception(
                "[VEHICLE_POSITION] "
                "Falha na estimativa."
            )

            result = self._invalid_result(
                vehicle_x,
                evaluation_y,
                error,
            )

            self.last_result = result

            return result

    # =========================================================================
    # COMPATIBILIDADE
    # =========================================================================

    def update(
        self,
        lanes: Sequence[
            Sequence[LanePoint]
        ],
        image_width: int,
        image_height: int,
        vehicle_x: Optional[float] = None,
        evaluation_y: Optional[float] = None,
    ) -> VehiclePositionResult:

        return self.estimate(
            lanes=lanes,
            image_width=image_width,
            image_height=image_height,
            vehicle_x=vehicle_x,
            evaluation_y=evaluation_y,
        )

    def process(
        self,
        lanes: Sequence[
            Sequence[LanePoint]
        ],
        image_width: int,
        image_height: int,
        vehicle_x: Optional[float] = None,
        evaluation_y: Optional[float] = None,
    ) -> VehiclePositionResult:

        return self.estimate(
            lanes=lanes,
            image_width=image_width,
            image_height=image_height,
            vehicle_x=vehicle_x,
            evaluation_y=evaluation_y,
        )


# =============================================================================
# FACTORY
# =============================================================================

def create_default_vehicle_position(
    **kwargs,
) -> VehiclePosition:

    return VehiclePosition(**kwargs)


__all__ = [
    "VehiclePositionState",
    "VehiclePositionResult",
    "VehiclePosition",
    "create_default_vehicle_position",
]