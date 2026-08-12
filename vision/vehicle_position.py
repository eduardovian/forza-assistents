"""
vision/vehicle_position.py

Estimativa da posição lateral do veículo dentro da faixa.

Pipeline:

    LaneTracker
         ↓
    LaneProjection
         ↓
    VehiclePosition
         ↓
    posição lateral do veículo

Responsabilidades deste módulo:

- identificar a faixa ocupada pelo veículo;
- calcular o centro da faixa;
- calcular o erro lateral em pixels;
- normalizar o erro pela largura da faixa;
- determinar a posição relativa do veículo;
- fornecer confiança da estimativa;
- rejeitar estimativas sem evidência suficiente.

Este módulo NÃO:

- executa YOLOP;
- captura tela;
- faz tracking temporal;
- projeta lanes;
- calcula controle do volante;
- decide estado ADAS.

Convenção:

    erro_lateral > 0
        veículo está à direita do centro.

    erro_lateral < 0
        veículo está à esquerda do centro.

    erro_lateral = 0
        veículo está no centro.

A posição do veículo na imagem é representada pelo
centro horizontal da imagem.

Isso é apropriado para a câmera interna enquanto a
calibração da câmera não fornecer um offset específico
do veículo em relação ao eixo óptico.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .lane_types import LanePoint


logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

DEFAULT_EVALUATION_Y_RATIO = 0.82

DEFAULT_MIN_LANE_WIDTH = 30.0

DEFAULT_MAX_LANE_WIDTH = 1800.0

DEFAULT_MIN_VALID_WIDTH = 10.0

DEFAULT_CENTER_TOLERANCE = 0.10

DEFAULT_WARNING_TOLERANCE = 0.22

DEFAULT_CRITICAL_TOLERANCE = 0.38

DEFAULT_MIN_CONFIDENCE = 0.45

DEFAULT_MIN_POINTS = 4


# ============================================================================
# ESTADO DA POSIÇÃO
# ============================================================================


class VehiclePositionState(str, Enum):
    """
    Estado lateral do veículo dentro da faixa.
    """

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


# ============================================================================
# RESULTADO
# ============================================================================


@dataclass
class VehiclePositionResult:
    """
    Resultado da estimativa da posição do veículo.

    lane_center_x:
        Centro horizontal da faixa atual.

    vehicle_center_x:
        Centro horizontal estimado do veículo/câmera.

    lateral_error:
        Erro lateral em pixels.

        Positivo = veículo à direita.
        Negativo = veículo à esquerda.

    normalized_error:
        Erro normalizado pela largura da faixa.

        Aproximadamente:

            -1.0 = extremo esquerdo
             0.0 = centro
            +1.0 = extremo direito

    lane_width:
        Largura estimada da faixa no ponto de avaliação.

    left_distance:
        Distância do veículo à linha esquerda.

    right_distance:
        Distância do veículo à linha direita.

    state:
        Estado lateral do veículo.

    confidence:
        Confiança da estimativa.

    valid:
        Indica se a estimativa é suficientemente confiável.
    """

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


# ============================================================================
# VEHICLE POSITION
# ============================================================================


class VehiclePosition:
    """
    Calcula a posição do veículo dentro da faixa.

    A classe aceita:

        left_lane
        right_lane

    ou uma lista ordenada de lanes.

    O cálculo é realizado em uma linha horizontal da imagem,
    normalmente próxima da região inferior, onde a posição
    lateral do veículo é mais representativa.

    Exemplo:

        esquerda = x=700
        direita  = x=1200

        centro = 950

        veículo = x=1000

        erro = +50

    Portanto:

        veículo está 50 px à direita do centro.
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

    # ========================================================================
    # UTILIDADES
    # ========================================================================

    @staticmethod
    def _valid_points(
        points: Sequence[LanePoint],
    ) -> List[LanePoint]:

        result: List[LanePoint] = []

        for point in points:

            if not point.valid:
                continue

            if not np.isfinite(point.x):
                continue

            if not np.isfinite(point.y):
                continue

            result.append(point)

        return result

    @staticmethod
    def _interpolate_x(
        points: Sequence[LanePoint],
        y: float,
    ) -> Optional[float]:
        """
        Obtém X da lane para um determinado Y.

        Utiliza interpolação linear entre os pontos observados.

        Não extrapola fora do intervalo observado.
        """

        valid = VehiclePosition._valid_points(
            points
        )

        if len(valid) < 2:
            return None

        ordered = sorted(
            valid,
            key=lambda point: point.y,
        )

        ys = np.asarray(
            [point.y for point in ordered],
            dtype=np.float64,
        )

        xs = np.asarray(
            [point.x for point in ordered],
            dtype=np.float64,
        )

        if y < ys[0] or y > ys[-1]:
            return None

        return float(
            np.interp(
                y,
                ys,
                xs,
            )
        )

    @staticmethod
    def _lane_confidence(
        points: Sequence[LanePoint],
    ) -> float:

        valid = VehiclePosition._valid_points(
            points
        )

        if not valid:
            return 0.0

        values = []

        for point in valid:

            confidence = float(
                point.confidence
            )

            if np.isfinite(confidence):
                values.append(
                    np.clip(
                        confidence,
                        0.0,
                        1.0,
                    )
                )

        if not values:
            return 0.0

        return float(
            np.mean(values)
        )

    # ========================================================================
    # ESCOLHA DA FAIXA
    # ========================================================================

    def _find_current_lane(
        self,
        lanes: Sequence[Sequence[LanePoint]],
        vehicle_x: float,
        evaluation_y: float,
    ) -> Tuple[
        Optional[int],
        Optional[float],
        Optional[float],
        float,
    ]:
        """
        Procura a faixa que contém o veículo.

        Para cada par de lanes consecutivas:

            left_lane
            right_lane

        calcula:

            left_x
            right_x
            center_x

        e verifica se o veículo está dentro desse intervalo.

        Retorna:

            lane_index
            left_x
            right_x
            confidence
        """

        if len(lanes) < 2:
            return (
                None,
                None,
                None,
                0.0,
            )

        best_candidate = None

        for index in range(
            len(lanes) - 1
        ):

            left_lane = lanes[index]
            right_lane = lanes[index + 1]

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

            width = (
                right_x - left_x
            )

            if (
                width < self.min_lane_width
                or width > self.max_lane_width
            ):
                continue

            if (
                vehicle_x < left_x
                or vehicle_x > right_x
            ):
                continue

            center_x = (
                left_x + right_x
            ) / 2.0

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

            confidence = (
                left_confidence
                + right_confidence
            ) / 2.0

            distance_from_center = abs(
                vehicle_x - center_x
            )

            candidate = (
                distance_from_center,
                index,
                left_x,
                right_x,
                confidence,
            )

            if (
                best_candidate is None
                or candidate[0]
                < best_candidate[0]
            ):
                best_candidate = candidate

        if best_candidate is None:

            return (
                None,
                None,
                None,
                0.0,
            )

        (
            _,
            index,
            left_x,
            right_x,
            confidence,
        ) = best_candidate

        return (
            index,
            left_x,
            right_x,
            confidence,
        )

    # ========================================================================
    # ESTADO
    # ========================================================================

    def _classify_state(
        self,
        normalized_error: float,
    ) -> VehiclePositionState:
        """
        Classifica a posição lateral.

        O valor absoluto do erro determina a severidade.
        O sinal determina o lado.
        """

        magnitude = abs(
            normalized_error
        )

        if (
            magnitude
            <= self.center_tolerance
        ):
            return (
                VehiclePositionState.CENTERED
            )

        if (
            normalized_error < 0
        ):

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

    # ========================================================================
    # RESULTADO INVÁLIDO
    # ========================================================================

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

    # ========================================================================
    # API PRINCIPAL
    # ========================================================================

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
        """
        Estima a posição do veículo.

        Parameters
        ----------
        lanes:
            Lista ordenada das linhas da pista.

            Exemplo:

                [
                    acostamento_esquerdo,
                    faixa_1_esquerda,
                    faixa_1_direita,
                    faixa_2_direita,
                    ...
                ]

            Na arquitetura atual, o ideal é passar somente
            as lanes de marcação que representam limites
            consecutivos da pista.

        image_width:
            Largura do frame.

        image_height:
            Altura do frame.

        vehicle_x:
            Posição X do veículo.

            Se não fornecida, utiliza o centro da imagem.

        evaluation_y:
            Altura onde a posição será calculada.

            Se não fornecida:

                image_height * evaluation_y_ratio
        """

        try:

            if image_width <= 0:
                return self._invalid_result(
                    0.0,
                    0.0,
                    "image_width inválido.",
                )

            if image_height <= 0:
                return self._invalid_result(
                    0.0,
                    0.0,
                    "image_height inválido.",
                )

            if vehicle_x is None:

                vehicle_x = (
                    image_width / 2.0
                    + self.vehicle_x_offset
                )

            else:

                vehicle_x = float(
                    vehicle_x
                )

            if evaluation_y is None:

                evaluation_y = (
                    image_height
                    * self.evaluation_y_ratio
                )

            else:

                evaluation_y = float(
                    evaluation_y
                )

            evaluation_y = float(
                np.clip(
                    evaluation_y,
                    0.0,
                    image_height - 1,
                )
            )

            if len(lanes) < 2:

                return self._invalid_result(
                    vehicle_x,
                    evaluation_y,
                    "Menos de duas lanes disponíveis.",
                )

            lane_index, left_x, right_x, confidence = (
                self._find_current_lane(
                    lanes,
                    vehicle_x,
                    evaluation_y,
                )
            )

            if lane_index is None:

                return self._invalid_result(
                    vehicle_x,
                    evaluation_y,
                    "Não foi possível identificar "
                    "a faixa ocupada pelo veículo.",
                )

            if confidence < self.min_confidence:

                return self._invalid_result(
                    vehicle_x,
                    evaluation_y,
                    (
                        "Confiança das lanes abaixo "
                        "do limite mínimo."
                    ),
                )

            assert left_x is not None
            assert right_x is not None

            lane_width = (
                right_x - left_x
            )

            if lane_width < self.min_lane_width:

                return self._invalid_result(
                    vehicle_x,
                    evaluation_y,
                    "Largura da faixa inválida.",
                )

            lane_center_x = (
                left_x + right_x
            ) / 2.0

            lateral_error = (
                vehicle_x
                - lane_center_x
            )

            half_width = (
                lane_width / 2.0
            )

            normalized_error = (
                lateral_error
                / half_width
            )

            normalized_error = float(
                np.clip(
                    normalized_error,
                    -2.0,
                    2.0,
                )
            )

            left_distance = (
                vehicle_x
                - left_x
            )

            right_distance = (
                right_x
                - vehicle_x
            )

            state = self._classify_state(
                normalized_error
            )

            return VehiclePositionResult(
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
                valid=True,
                evaluation_y=float(
                    evaluation_y
                ),
                error=None,
            )

        except Exception as exc:

            logger.exception(
                "[VEHICLE POSITION] "
                "Falha na estimativa."
            )

            return self._invalid_result(
                (
                    float(vehicle_x)
                    if vehicle_x is not None
                    else image_width / 2.0
                ),
                (
                    float(evaluation_y)
                    if evaluation_y is not None
                    else image_height
                    * self.evaluation_y_ratio
                ),
                f"{type(exc).__name__}: {exc}",
            )


# ============================================================================
# FACTORY
# ============================================================================


def create_default_vehicle_position(
    **kwargs,
) -> VehiclePosition:

    return VehiclePosition(
        **kwargs
    )


# ============================================================================
# EXPORTS
# ============================================================================


__all__ = [
    "VehiclePositionState",
    "VehiclePositionResult",
    "VehiclePosition",
    "create_default_vehicle_position",
]