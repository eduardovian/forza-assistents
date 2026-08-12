"""
vision/lane_center.py

Determinação do centro da faixa atual.

Responsabilidade:

    lane esquerda
          +
    lane direita
          ↓
    centro da faixa

Também calcula:
    - posição do veículo em relação ao centro
    - erro lateral normalizado
    - largura estimada da faixa
    - centro previsto da faixa

Este módulo NÃO:
    - executa YOLOP
    - captura imagens
    - faz controle do volante
    - decide quando corrigir o veículo
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .lane_types import LanePoint


# ============================================================================
# RESULTADO
# ============================================================================


@dataclass
class LaneCenterPoint:
    """Ponto do centro da faixa."""

    x: float
    y: float
    confidence: float
    valid: bool = True


@dataclass
class LaneCenterResult:
    """
    Resultado do cálculo do centro da faixa atual.
    """

    center_points: List[LaneCenterPoint]

    center_x_near: Optional[float]
    center_x_far: Optional[float]

    vehicle_x: float

    lateral_error_pixels: Optional[float]

    lateral_error_normalized: Optional[float]

    lane_width_near: Optional[float]
    lane_width_far: Optional[float]

    confidence: float

    valid: bool

    left_available: bool
    right_available: bool

    error: Optional[str] = None


# ============================================================================
# CALCULADOR
# ============================================================================


class LaneCenterEstimator:
    """
    Calcula o centro da faixa a partir das linhas esquerda/direita.

    O centro é calculado horizontalmente para cada altura Y:

        center_x = (left_x + right_x) / 2

    Isso permite acompanhar faixas curvas sem assumir que
    o centro da imagem representa o centro da pista.
    """

    def __init__(
        self,
        near_y_ratio: float = 0.85,
        far_y_ratio: float = 0.45,
        min_lane_width_pixels: float = 20.0,
        max_lane_width_pixels: float = 1500.0,
    ) -> None:

        self.near_y_ratio = float(
            np.clip(
                near_y_ratio,
                0.0,
                1.0,
            )
        )

        self.far_y_ratio = float(
            np.clip(
                far_y_ratio,
                0.0,
                1.0,
            )
        )

        self.min_lane_width_pixels = max(
            1.0,
            float(min_lane_width_pixels),
        )

        self.max_lane_width_pixels = max(
            self.min_lane_width_pixels,
            float(max_lane_width_pixels),
        )

    # ========================================================================
    # INTERPOLAÇÃO
    # ========================================================================

    @staticmethod
    def _interpolate_lane(
        lane: Sequence[LanePoint],
        y: float,
    ) -> Optional[float]:

        valid = [
            point
            for point in lane
            if point.valid
            and np.isfinite(point.x)
            and np.isfinite(point.y)
        ]

        if len(valid) < 2:
            return None

        valid = sorted(
            valid,
            key=lambda point: point.y,
        )

        ys = np.asarray(
            [point.y for point in valid],
            dtype=np.float64,
        )

        xs = np.asarray(
            [point.x for point in valid],
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

    # ========================================================================
    # PONTOS COMUNS
    # ========================================================================

    @staticmethod
    def _common_y_range(
        left_lane: Sequence[LanePoint],
        right_lane: Sequence[LanePoint],
    ) -> Optional[Tuple[float, float]]:

        left = [
            point.y
            for point in left_lane
            if point.valid
            and np.isfinite(point.y)
        ]

        right = [
            point.y
            for point in right_lane
            if point.valid
            and np.isfinite(point.y)
        ]

        if not left or not right:
            return None

        lower = max(
            min(left),
            min(right),
        )

        upper = min(
            max(left),
            max(right),
        )

        if upper <= lower:
            return None

        return (
            float(lower),
            float(upper),
        )

    # ========================================================================
    # CONSTRUÇÃO DO CENTRO
    # ========================================================================

    def _build_center_points(
        self,
        left_lane: Sequence[LanePoint],
        right_lane: Sequence[LanePoint],
        samples: int = 30,
    ) -> List[LaneCenterPoint]:

        y_range = self._common_y_range(
            left_lane,
            right_lane,
        )

        if y_range is None:
            return []

        y_min, y_max = y_range

        ys = np.linspace(
            y_min,
            y_max,
            max(2, samples),
        )

        result = []

        for y in ys:

            left_x = self._interpolate_lane(
                left_lane,
                float(y),
            )

            right_x = self._interpolate_lane(
                right_lane,
                float(y),
            )

            if (
                left_x is None
                or right_x is None
            ):
                continue

            width = (
                right_x - left_x
            )

            if (
                width
                < self.min_lane_width_pixels
                or width
                > self.max_lane_width_pixels
            ):
                continue

            center_x = (
                left_x + right_x
            ) / 2.0

            result.append(
                LaneCenterPoint(
                    x=float(center_x),
                    y=float(y),
                    confidence=1.0,
                    valid=True,
                )
            )

        return result

    # ========================================================================
    # CENTRO EM UMA ALTURA
    # ========================================================================

    def _center_at_y(
        self,
        points: Sequence[LaneCenterPoint],
        y: float,
    ) -> Optional[float]:

        valid = [
            point
            for point in points
            if point.valid
            and np.isfinite(point.x)
            and np.isfinite(point.y)
        ]

        if len(valid) < 2:
            return None

        valid = sorted(
            valid,
            key=lambda point: point.y,
        )

        ys = np.asarray(
            [point.y for point in valid],
            dtype=np.float64,
        )

        xs = np.asarray(
            [point.x for point in valid],
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

    # ========================================================================
    # CONFIANÇA
    # ========================================================================

    @staticmethod
    def _confidence(
        points: Sequence[LaneCenterPoint],
        lane_widths: Sequence[float],
    ) -> float:

        if not points:
            return 0.0

        point_score = np.clip(
            len(points) / 25.0,
            0.0,
            1.0,
        )

        if lane_widths:

            widths = np.asarray(
                lane_widths,
                dtype=np.float64,
            )

            mean_width = float(
                np.mean(widths)
            )

            std_width = float(
                np.std(widths)
            )

            if mean_width > 1e-6:

                consistency = np.clip(
                    1.0
                    - (
                        std_width
                        / mean_width
                    ),
                    0.0,
                    1.0,
                )

            else:
                consistency = 0.0

        else:
            consistency = 0.0

        return float(
            np.clip(
                0.6 * point_score
                + 0.4 * consistency,
                0.0,
                1.0,
            )
        )

    # ========================================================================
    # API
    # ========================================================================

    def estimate(
        self,
        left_lane: Sequence[LanePoint],
        right_lane: Sequence[LanePoint],
        image_width: int,
        image_height: int,
        vehicle_x: Optional[float] = None,
    ) -> LaneCenterResult:

        if image_width <= 0:
            return self._invalid_result(
                vehicle_x or 0.0,
                "image_width inválido.",
            )

        if image_height <= 0:
            return self._invalid_result(
                vehicle_x or 0.0,
                "image_height inválido.",
            )

        if vehicle_x is None:
            vehicle_x = (
                image_width / 2.0
            )

        try:

            center_points = (
                self._build_center_points(
                    left_lane,
                    right_lane,
                )
            )

            if len(center_points) < 2:
                return self._invalid_result(
                    vehicle_x,
                    "Não existem pontos "
                    "suficientes para determinar "
                    "o centro da faixa.",
                )

            near_y = (
                image_height
                * self.near_y_ratio
            )

            far_y = (
                image_height
                * self.far_y_ratio
            )

            center_near = (
                self._center_at_y(
                    center_points,
                    near_y,
                )
            )

            center_far = (
                self._center_at_y(
                    center_points,
                    far_y,
                )
            )

            # Se o ponto exato não estiver disponível,
            # usamos os extremos válidos.
            if center_near is None:

                center_near = float(
                    center_points[-1].x
                )

            if center_far is None:

                center_far = float(
                    center_points[0].x
                )

            left_near = (
                self._interpolate_lane(
                    left_lane,
                    near_y,
                )
            )

            right_near = (
                self._interpolate_lane(
                    right_lane,
                    near_y,
                )
            )

            left_far = (
                self._interpolate_lane(
                    left_lane,
                    far_y,
                )
            )

            right_far = (
                self._interpolate_lane(
                    right_lane,
                    far_y,
                )
            )

            lane_width_near = None
            lane_width_far = None

            if (
                left_near is not None
                and right_near is not None
            ):

                lane_width_near = (
                    right_near
                    - left_near
                )

            if (
                left_far is not None
                and right_far is not None
            ):

                lane_width_far = (
                    right_far
                    - left_far
                )

            widths = [
                width
                for width in (
                    lane_width_near,
                    lane_width_far,
                )
                if width is not None
                and width > 0
            ]

            lateral_error = (
                float(vehicle_x)
                - float(center_near)
            )

            if lane_width_near:
                lateral_error_normalized = (
                    lateral_error
                    / (
                        lane_width_near
                        / 2.0
                    )
                )
            else:
                lateral_error_normalized = None

            confidence = self._confidence(
                center_points,
                widths,
            )

            return LaneCenterResult(
                center_points=center_points,
                center_x_near=float(
                    center_near
                ),
                center_x_far=float(
                    center_far
                ),
                vehicle_x=float(
                    vehicle_x
                ),
                lateral_error_pixels=(
                    lateral_error
                ),
                lateral_error_normalized=(
                    lateral_error_normalized
                ),
                lane_width_near=(
                    lane_width_near
                ),
                lane_width_far=(
                    lane_width_far
                ),
                confidence=confidence,
                valid=True,
                left_available=bool(
                    left_lane
                ),
                right_available=bool(
                    right_lane
                ),
                error=None,
            )

        except Exception as exc:

            return self._invalid_result(
                vehicle_x,
                f"{type(exc).__name__}: {exc}",
            )

    # ========================================================================
    # RESULTADO INVÁLIDO
    # ========================================================================

    @staticmethod
    def _invalid_result(
        vehicle_x: float,
        error: str,
    ) -> LaneCenterResult:

        return LaneCenterResult(
            center_points=[],
            center_x_near=None,
            center_x_far=None,
            vehicle_x=float(vehicle_x),
            lateral_error_pixels=None,
            lateral_error_normalized=None,
            lane_width_near=None,
            lane_width_far=None,
            confidence=0.0,
            valid=False,
            left_available=False,
            right_available=False,
            error=error,
        )


# ============================================================================
# FACTORY
# ============================================================================


def create_default_lane_center_estimator(
    **kwargs,
) -> LaneCenterEstimator:

    return LaneCenterEstimator(
        **kwargs
    )


__all__ = [
    "LaneCenterPoint",
    "LaneCenterResult",
    "LaneCenterEstimator",
    "create_default_lane_center_estimator",
]