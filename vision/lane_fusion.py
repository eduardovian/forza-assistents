"""
vision/lane_fusion.py

Fusão das linhas esquerda/direita em um corredor de faixa.

Responsabilidades:
    - combinar as duas bordas;
    - utilizar projeções quando disponíveis;
    - calcular o centro da faixa;
    - validar largura;
    - avaliar consistência;
    - produzir pontos centrais;
    - produzir confiança da fusão.

Não faz:
    - inferência YOLOP;
    - rastreamento temporal;
    - associação de faixas;
    - cálculo de posição do veículo;
    - decisão ADAS.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .lane_types import LanePoint
from .lane_projection import LaneProjectionResult

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

DEFAULT_MIN_WIDTH = 40.0
DEFAULT_MAX_WIDTH = 1600.0

DEFAULT_MIN_WIDTH_RATIO = 0.05
DEFAULT_MAX_WIDTH_RATIO = 0.90

DEFAULT_MAX_WIDTH_VARIATION = 0.45

DEFAULT_MIN_SAMPLES = 5
DEFAULT_SAMPLE_STEP = 10


# =============================================================================
# RESULTADOS
# =============================================================================

@dataclass
class LaneCenterPoint:
    x: float
    y: float

    left_x: float
    right_x: float

    width: float

    confidence: float

    valid: bool = True


@dataclass
class LaneFusionResult:
    center_points: List[LaneCenterPoint]
    left_points: List[LanePoint]
    right_points: List[LanePoint]

    lane_widths: List[float]

    confidence: float
    valid: bool

    left_available: bool
    right_available: bool
    both_sides_available: bool

    projected_left: bool
    projected_right: bool

    error: Optional[str] = None

    @property
    def center_lane(self) -> List[LanePoint]:
        return [
            LanePoint(
                x=point.x,
                y=point.y,
                confidence=point.confidence,
                valid=point.valid,
            )
            for point in self.center_points
        ]

    @property
    def has_center(self) -> bool:
        return bool(self.center_points)


# =============================================================================
# UTILITÁRIOS
# =============================================================================

def _clip01(value: float) -> float:
    if not np.isfinite(value):
        return 0.0

    return float(
        np.clip(value, 0.0, 1.0)
    )


def _prepare_points(
    points: Sequence[LanePoint],
) -> Tuple[np.ndarray, np.ndarray]:

    valid = [
        point
        for point in points
        if point.valid
        and np.isfinite(point.x)
        and np.isfinite(point.y)
    ]

    if len(valid) < 2:
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )

    valid.sort(
        key=lambda point: point.y
    )

    y = np.asarray(
        [point.y for point in valid],
        dtype=np.float64,
    )

    x = np.asarray(
        [point.x for point in valid],
        dtype=np.float64,
    )

    unique_y, indices = np.unique(
        y,
        return_index=True,
    )

    return (
        unique_y,
        x[indices],
    )


def _points_confidence(
    points: Sequence[LanePoint],
) -> float:

    values = [
        float(point.confidence)
        for point in points
        if point.valid
        and np.isfinite(point.confidence)
    ]

    if not values:
        return 0.0

    return _clip01(
        float(np.mean(values))
    )


# =============================================================================
# FUSÃO
# =============================================================================

class LaneFusion:
    """
    Constrói o corredor central a partir das duas bordas.

    Para cada Y:

        center_x = (left_x + right_x) / 2

        width = right_x - left_x
    """

    def __init__(
        self,
        min_width: float = DEFAULT_MIN_WIDTH,
        max_width: float = DEFAULT_MAX_WIDTH,
        min_width_ratio: float = DEFAULT_MIN_WIDTH_RATIO,
        max_width_ratio: float = DEFAULT_MAX_WIDTH_RATIO,
        max_width_variation: float = (
            DEFAULT_MAX_WIDTH_VARIATION
        ),
        min_samples: int = DEFAULT_MIN_SAMPLES,
        sample_step: int = DEFAULT_SAMPLE_STEP,
    ) -> None:

        self.min_width = max(
            1.0,
            float(min_width),
        )

        self.max_width = max(
            self.min_width,
            float(max_width),
        )

        self.min_width_ratio = float(
            np.clip(
                min_width_ratio,
                0.001,
                1.0,
            )
        )

        self.max_width_ratio = float(
            np.clip(
                max_width_ratio,
                self.min_width_ratio,
                1.0,
            )
        )

        self.max_width_variation = max(
            0.01,
            float(max_width_variation),
        )

        self.min_samples = max(
            2,
            int(min_samples),
        )

        self.sample_step = max(
            1,
            int(sample_step),
        )

        self.last_result: Optional[
            LaneFusionResult
        ] = None

    # =========================================================================
    # PROJEÇÃO
    # =========================================================================

    @staticmethod
    def _projection_points(
        projection: Optional[LaneProjectionResult],
    ) -> List[LanePoint]:

        if projection is None:
            return []

        if not projection.valid:
            return []

        return [
            point
            for point in projection.points
            if point.valid
        ]

    def _select_points(
        self,
        points: Sequence[LanePoint],
        projection: Optional[LaneProjectionResult],
    ) -> Tuple[List[LanePoint], bool]:

        projected = self._projection_points(
            projection
        )

        if projected:
            return projected, True

        return list(points), False

    # =========================================================================
    # AMOSTRAGEM
    # =========================================================================

    def _build_axis(
        self,
        left_y: np.ndarray,
        right_y: np.ndarray,
    ) -> np.ndarray:

        if (
            left_y.size < 2
            or right_y.size < 2
        ):
            return np.empty(
                0,
                dtype=np.float64,
            )

        y_min = max(
            float(np.min(left_y)),
            float(np.min(right_y)),
        )

        y_max = min(
            float(np.max(left_y)),
            float(np.max(right_y)),
        )

        if y_max <= y_min:
            return np.empty(
                0,
                dtype=np.float64,
            )

        count = max(
            self.min_samples,
            int(
                (y_max - y_min)
                / self.sample_step
            ) + 1,
        )

        return np.linspace(
            y_min,
            y_max,
            count,
            dtype=np.float64,
        )

    # =========================================================================
    # VALIDAÇÃO
    # =========================================================================

    def _validate_widths(
        self,
        widths: np.ndarray,
        image_width: int,
    ) -> Tuple[bool, float]:

        if widths.size < self.min_samples:
            return False, 0.0

        if not np.all(
            np.isfinite(widths)
        ):
            return False, 0.0

        minimum = max(
            self.min_width,
            image_width
            * self.min_width_ratio,
        )

        maximum = min(
            self.max_width,
            image_width
            * self.max_width_ratio,
        )

        if np.any(widths < minimum):
            return False, 0.0

        if np.any(widths > maximum):
            return False, 0.0

        mean_width = float(
            np.mean(widths)
        )

        if mean_width <= 0.0:
            return False, 0.0

        variation = float(
            np.std(widths)
            / mean_width
        )

        if (
            variation
            > self.max_width_variation
        ):
            return False, 0.0

        stability = _clip01(
            1.0
            - (
                variation
                / self.max_width_variation
            )
        )

        return True, stability

    # =========================================================================
    # CONFIANÇA
    # =========================================================================

    @staticmethod
    def _calculate_confidence(
        left_confidence: float,
        right_confidence: float,
        width_stability: float,
        sample_count: int,
    ) -> float:

        line_confidence = (
            _clip01(left_confidence)
            + _clip01(right_confidence)
        ) / 2.0

        sample_score = _clip01(
            sample_count / 30.0
        )

        return _clip01(
            0.50 * line_confidence
            + 0.35 * width_stability
            + 0.15 * sample_score
        )

    # =========================================================================
    # API PRINCIPAL
    # =========================================================================

    def fuse(
        self,
        left_points: Sequence[LanePoint],
        right_points: Sequence[LanePoint],
        image_width: int,
        image_height: int,
        left_projection: Optional[
            LaneProjectionResult
        ] = None,
        right_projection: Optional[
            LaneProjectionResult
        ] = None,
    ) -> LaneFusionResult:

        try:

            if image_width <= 0:
                raise ValueError(
                    "image_width inválido."
                )

            if image_height <= 0:
                raise ValueError(
                    "image_height inválido."
                )

            effective_left, projected_left = (
                self._select_points(
                    left_points,
                    left_projection,
                )
            )

            effective_right, projected_right = (
                self._select_points(
                    right_points,
                    right_projection,
                )
            )

            left_y, left_x = _prepare_points(
                effective_left
            )

            right_y, right_x = _prepare_points(
                effective_right
            )

            if (
                left_y.size < 2
                or right_y.size < 2
            ):
                raise ValueError(
                    "Linhas insuficientes para fusão."
                )

            y = self._build_axis(
                left_y,
                right_y,
            )

            if y.size < self.min_samples:
                raise ValueError(
                    "Região comum insuficiente."
                )

            sampled_left = np.interp(
                y,
                left_y,
                left_x,
            )

            sampled_right = np.interp(
                y,
                right_y,
                right_x,
            )

            # Nunca aceitar linhas cruzadas.
            if np.any(
                sampled_left
                >= sampled_right
            ):
                raise ValueError(
                    "Linhas esquerda/direita "
                    "cruzadas."
                )

            widths = (
                sampled_right
                - sampled_left
            )

            width_valid, width_stability = (
                self._validate_widths(
                    widths,
                    image_width,
                )
            )

            if not width_valid:
                raise ValueError(
                    "Largura da faixa "
                    "inconsistente."
                )

            center_x = (
                sampled_left
                + sampled_right
            ) / 2.0

            if np.any(
                center_x < 0.0
            ) or np.any(
                center_x >= image_width
            ):
                raise ValueError(
                    "Centro da faixa "
                    "fora da imagem."
                )

            left_confidence = (
                _points_confidence(
                    effective_left
                )
            )

            right_confidence = (
                _points_confidence(
                    effective_right
                )
            )

            confidence = (
                self._calculate_confidence(
                    left_confidence,
                    right_confidence,
                    width_stability,
                    len(y),
                )
            )

            center_points = [
                LaneCenterPoint(
                    x=float(center_x[i]),
                    y=float(y[i]),
                    left_x=float(
                        sampled_left[i]
                    ),
                    right_x=float(
                        sampled_right[i]
                    ),
                    width=float(widths[i]),
                    confidence=confidence,
                    valid=True,
                )
                for i in range(len(y))
            ]

            result = LaneFusionResult(
                center_points=center_points,
                left_points=effective_left,
                right_points=effective_right,
                lane_widths=[
                    float(width)
                    for width in widths
                ],
                confidence=confidence,
                valid=True,
                left_available=bool(
                    effective_left
                ),
                right_available=bool(
                    effective_right
                ),
                both_sides_available=True,
                projected_left=projected_left,
                projected_right=projected_right,
                error=None,
            )

            self.last_result = result

            return result

        except Exception as exc:

            error = (
                f"{type(exc).__name__}: {exc}"
            )

            logger.warning(
                "[LANE_FUSION] %s",
                error,
            )

            result = LaneFusionResult(
                center_points=[],
                left_points=list(left_points),
                right_points=list(right_points),
                lane_widths=[],
                confidence=0.0,
                valid=False,
                left_available=bool(left_points),
                right_available=bool(right_points),
                both_sides_available=(
                    bool(left_points)
                    and bool(right_points)
                ),
                projected_left=False,
                projected_right=False,
                error=error,
            )

            self.last_result = result

            return result

    # =========================================================================
    # COMPATIBILIDADE
    # =========================================================================

    def update(
        self,
        left_points: Sequence[LanePoint],
        right_points: Sequence[LanePoint],
        image_width: int,
        image_height: int,
        left_projection: Optional[
            LaneProjectionResult
        ] = None,
        right_projection: Optional[
            LaneProjectionResult
        ] = None,
    ) -> LaneFusionResult:

        return self.fuse(
            left_points=left_points,
            right_points=right_points,
            image_width=image_width,
            image_height=image_height,
            left_projection=left_projection,
            right_projection=right_projection,
        )

    def process(
        self,
        left_points: Sequence[LanePoint],
        right_points: Sequence[LanePoint],
        image_width: int,
        image_height: int,
        left_projection: Optional[
            LaneProjectionResult
        ] = None,
        right_projection: Optional[
            LaneProjectionResult
        ] = None,
    ) -> LaneFusionResult:

        return self.fuse(
            left_points=left_points,
            right_points=right_points,
            image_width=image_width,
            image_height=image_height,
            left_projection=left_projection,
            right_projection=right_projection,
        )


# =============================================================================
# FACTORY
# =============================================================================

def create_default_lane_fusion(
    **kwargs,
) -> LaneFusion:

    return LaneFusion(**kwargs)


__all__ = [
    "LaneCenterPoint",
    "LaneFusionResult",
    "LaneFusion",
    "create_default_lane_fusion",
]