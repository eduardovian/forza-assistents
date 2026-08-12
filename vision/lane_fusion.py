"""
vision/lane_fusion.py

Fusão das linhas de faixa detectadas/projetadas.

Pipeline:

    YOLOP
      ↓
    LaneTracker
      ↓
    LaneGeometry
      ↓
    LaneProjection
      ↓
    LaneFusion
      ↓
    corredor da faixa
      ↓
    centro previsto da faixa

Responsabilidades:
    - combinar esquerda e direita;
    - aceitar linhas parcialmente projetadas;
    - calcular o centro da faixa;
    - verificar a largura da faixa;
    - verificar consistência geométrica;
    - produzir o centro ao longo de vários pontos Y;
    - informar a qualidade da estimativa.

Não faz:
    - inferência YOLOP;
    - captura de tela;
    - controle do volante;
    - decisão de correção ADAS;
    - classificação de estado ADAS.

A ideia é que o ADAS só receba um corredor considerado
geometricamente confiável.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .lane_types import LanePoint
from .lane_projection import LaneProjectionResult


logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

DEFAULT_MIN_WIDTH = 40.0
DEFAULT_MAX_WIDTH = 1600.0

DEFAULT_MIN_WIDTH_RATIO = 0.05
DEFAULT_MAX_WIDTH_RATIO = 0.90

DEFAULT_WIDTH_VARIATION = 0.45

DEFAULT_MIN_SAMPLES = 5

DEFAULT_SAMPLE_STEP = 10


# ============================================================================
# RESULTADO
# ============================================================================


@dataclass
class LaneCenterPoint:
    """
    Ponto do centro previsto da faixa.
    """

    x: float
    y: float

    left_x: float
    right_x: float

    width: float

    confidence: float

    valid: bool = True


@dataclass
class LaneFusionResult:
    """
    Resultado da fusão das linhas.

    center_points:
        Centro previsto da faixa ao longo do eixo Y.

    left_points:
        Linha esquerda utilizada.

    right_points:
        Linha direita utilizada.

    valid:
        Indica se existe um corredor confiável.

    confidence:
        Confiança global da fusão.
    """

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
        """
        Compatibilidade com estruturas que trabalham
        diretamente com LanePoint.
        """

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
        return bool(
            self.center_points
        )


# ============================================================================
# FUSÃO
# ============================================================================


class LaneFusion:
    """
    Constrói o corredor da faixa a partir das duas linhas.

    A operação principal é:

        centro(Y) = (esquerda(Y) + direita(Y)) / 2

    A largura também é monitorada:

        largura(Y) = direita(Y) - esquerda(Y)

    O sistema não aceita simplesmente qualquer par de linhas.
    A largura precisa permanecer fisicamente plausível e
    relativamente estável.
    """

    def __init__(
        self,
        min_width: float = DEFAULT_MIN_WIDTH,
        max_width: float = DEFAULT_MAX_WIDTH,
        min_width_ratio: float = DEFAULT_MIN_WIDTH_RATIO,
        max_width_ratio: float = DEFAULT_MAX_WIDTH_RATIO,
        max_width_variation: float = DEFAULT_WIDTH_VARIATION,
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

        self.max_width_variation = float(
            np.clip(
                max_width_variation,
                0.01,
                2.0,
            )
        )

        self.min_samples = max(
            2,
            int(min_samples),
        )

        self.sample_step = max(
            1,
            int(sample_step),
        )

    # ========================================================================
    # INTERPOLAÇÃO
    # ========================================================================

    @staticmethod
    def _prepare_line(
        points: Sequence[LanePoint],
    ) -> tuple[np.ndarray, np.ndarray]:

        valid = [
            point
            for point in points
            if point.valid
            and np.isfinite(point.x)
            and np.isfinite(point.y)
        ]

        if len(valid) < 2:
            return (
                np.empty(0),
                np.empty(0),
            )

        valid = sorted(
            valid,
            key=lambda point: point.y,
        )

        y_values = np.asarray(
            [point.y for point in valid],
            dtype=np.float64,
        )

        x_values = np.asarray(
            [point.x for point in valid],
            dtype=np.float64,
        )

        unique_y, indices = np.unique(
            y_values,
            return_index=True,
        )

        x_values = x_values[
            indices
        ]

        return (
            unique_y,
            x_values,
        )

    @staticmethod
    def _interpolate(
        y: np.ndarray,
        line_y: np.ndarray,
        line_x: np.ndarray,
    ) -> np.ndarray:

        return np.interp(
            y,
            line_y,
            line_x,
        )

    # ========================================================================
    # PONTOS DE PROJEÇÃO
    # ========================================================================

    @staticmethod
    def _projection_points(
        projection: Optional[
            LaneProjectionResult
        ],
    ) -> List[LanePoint]:

        if projection is None:
            return []

        if not projection.valid:
            return []

        return list(
            projection.points
        )

    # ========================================================================
    # CONSTRUÇÃO DO EIXO Y
    # ========================================================================

    def _build_sampling_axis(
        self,
        left_y: np.ndarray,
        right_y: np.ndarray,
    ) -> np.ndarray:

        if (
            left_y.size == 0
            or right_y.size == 0
        ):
            return np.empty(0)

        y_min = max(
            float(np.min(left_y)),
            float(np.min(right_y)),
        )

        y_max = min(
            float(np.max(left_y)),
            float(np.max(right_y)),
        )

        if y_max <= y_min:
            return np.empty(0)

        count = max(
            self.min_samples,
            int(
                (
                    y_max - y_min
                )
                / self.sample_step
            )
            + 1,
        )

        return np.linspace(
            y_min,
            y_max,
            count,
        )

    # ========================================================================
    # LARGURA
    # ========================================================================

    def _validate_widths(
        self,
        widths: np.ndarray,
        image_width: int,
    ) -> tuple[bool, float]:

        if widths.size < self.min_samples:
            return False, 0.0

        if not np.all(
            np.isfinite(widths)
        ):
            return False, 0.0

        minimum_allowed = max(
            self.min_width,
            image_width
            * self.min_width_ratio,
        )

        maximum_allowed = min(
            self.max_width,
            image_width
            * self.max_width_ratio,
        )

        if np.any(
            widths < minimum_allowed
        ):
            return False, 0.0

        if np.any(
            widths > maximum_allowed
        ):
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

        stability = float(
            np.clip(
                1.0
                - (
                    variation
                    / self.max_width_variation
                ),
                0.0,
                1.0,
            )
        )

        return True, stability

    # ========================================================================
    # CONFIANÇA
    # ========================================================================

    @staticmethod
    def _confidence(
        left_confidence: float,
        right_confidence: float,
        width_stability: float,
        sample_count: int,
    ) -> float:

        line_confidence = (
            float(
                np.clip(
                    left_confidence,
                    0.0,
                    1.0,
                )
            )
            +
            float(
                np.clip(
                    right_confidence,
                    0.0,
                    1.0,
                )
            )
        ) / 2.0

        sample_score = float(
            np.clip(
                sample_count / 30.0,
                0.0,
                1.0,
            )
        )

        confidence = (
            0.45 * line_confidence
            + 0.35 * width_stability
            + 0.20 * sample_score
        )

        return float(
            np.clip(
                confidence,
                0.0,
                1.0,
            )
        )

    # ========================================================================
    # FUSÃO PRINCIPAL
    # ========================================================================

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

            # ---------------------------------------------------------------
            # Se houver projeção válida, ela passa a representar a
            # continuação da linha.
            # ---------------------------------------------------------------

            effective_left = (
                self._projection_points(
                    left_projection
                )
                if left_projection is not None
                and left_projection.valid
                else list(left_points)
            )

            effective_right = (
                self._projection_points(
                    right_projection
                )
                if right_projection is not None
                and right_projection.valid
                else list(right_points)
            )

            left_y, left_x = (
                self._prepare_line(
                    effective_left
                )
            )

            right_y, right_x = (
                self._prepare_line(
                    effective_right
                )
            )

            if (
                left_y.size < 2
                or right_y.size < 2
            ):
                raise ValueError(
                    "Não existem duas linhas "
                    "suficientemente definidas."
                )

            y = self._build_sampling_axis(
                left_y,
                right_y,
            )

            if y.size < self.min_samples:
                raise ValueError(
                    "Trecho comum entre as linhas "
                    "é insuficiente."
                )

            sampled_left = self._interpolate(
                y,
                left_y,
                left_x,
            )

            sampled_right = self._interpolate(
                y,
                right_y,
                right_x,
            )

            # ---------------------------------------------------------------
            # A esquerda precisa permanecer à esquerda.
            # ---------------------------------------------------------------

            if np.any(
                sampled_left
                >= sampled_right
            ):
                raise ValueError(
                    "Linhas esquerda/direita "
                    "cruzaram."
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
                    "geometricamente inconsistente."
                )

            # ---------------------------------------------------------------
            # Centro geométrico.
            # ---------------------------------------------------------------

            center_x = (
                sampled_left
                + sampled_right
            ) / 2.0

            if np.any(
                center_x < 0.0
            ) or np.any(
                center_x
                >= image_width
            ):
                raise ValueError(
                    "Centro da faixa "
                    "fora da imagem."
                )

            left_confidence = (
                float(
                    np.mean(
                        [
                            p.confidence
                            for p in effective_left
                            if p.valid
                        ]
                    )
                )
                if effective_left
                else 0.0
            )

            right_confidence = (
                float(
                    np.mean(
                        [
                            p.confidence
                            for p in effective_right
                            if p.valid
                        ]
                    )
                )
                if effective_right
                else 0.0
            )

            confidence = self._confidence(
                left_confidence,
                right_confidence,
                width_stability,
                len(y),
            )

            center_points = []

            for index in range(
                len(y)
            ):

                center_points.append(
                    LaneCenterPoint(
                        x=float(
                            center_x[index]
                        ),
                        y=float(
                            y[index]
                        ),
                        left_x=float(
                            sampled_left[index]
                        ),
                        right_x=float(
                            sampled_right[index]
                        ),
                        width=float(
                            widths[index]
                        ),
                        confidence=confidence,
                        valid=True,
                    )
                )

            return LaneFusionResult(
                center_points=center_points,
                left_points=list(
                    effective_left
                ),
                right_points=list(
                    effective_right
                ),
                lane_widths=[
                    float(width)
                    for width in widths
                ],
                confidence=confidence,
                valid=True,
                left_available=True,
                right_available=True,
                both_sides_available=True,
                projected_left=(
                    left_projection is not None
                    and left_projection.valid
                ),
                projected_right=(
                    right_projection is not None
                    and right_projection.valid
                ),
                error=None,
            )

        except Exception as exc:

            logger.debug(
                "[LANE FUSION] "
                "Fusão rejeitada: %s",
                exc,
            )

            return LaneFusionResult(
                center_points=[],
                left_points=[],
                right_points=[],
                lane_widths=[],
                confidence=0.0,
                valid=False,
                left_available=False,
                right_available=False,
                both_sides_available=False,
                projected_left=False,
                projected_right=False,
                error=(
                    f"{type(exc).__name__}: {exc}"
                ),
            )


# ============================================================================
# FACTORY
# ============================================================================


def create_default_fusion(
    **kwargs,
) -> LaneFusion:

    return LaneFusion(
        **kwargs
    )


__all__ = [
    "LaneCenterPoint",
    "LaneFusionResult",
    "LaneFusion",
    "create_default_fusion",
]