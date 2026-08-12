"""
vision/lane_confidence.py

Avaliação de confiança da percepção de lanes.

Responsabilidades:
    - avaliar qualidade das linhas;
    - avaliar qualidade geométrica;
    - avaliar estabilidade temporal;
    - avaliar qualidade da projeção;
    - produzir confiança global;
    - determinar se a percepção é segura para ADAS.

Não executa:
    - inferência;
    - tracking;
    - projeção;
    - associação;
    - decisão ADAS.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Sequence

import numpy as np

from .lane_types import (
    LaneGeometry,
    LaneLine,
    LaneModel,
    LaneQuality,
    ProjectionQuality,
)


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

DEFAULT_MIN_CONFIDENCE = 0.55
DEFAULT_SAFE_CONFIDENCE = 0.70
DEFAULT_HIGH_CONFIDENCE = 0.85

DEFAULT_MIN_POINTS = 5
DEFAULT_GOOD_POINTS = 12
DEFAULT_EXCELLENT_POINTS = 24

DEFAULT_MAX_FIT_ERROR = 30.0
DEFAULT_GOOD_FIT_ERROR = 15.0
DEFAULT_EXCELLENT_FIT_ERROR = 7.5

DEFAULT_MAX_MISSED_FRAMES = 5
DEFAULT_MAX_AGE_FRAMES = 120

DEFAULT_MIN_LANE_WIDTH = 30.0
DEFAULT_MAX_LANE_WIDTH = 1800.0

DEFAULT_MAX_OFFSET = 1.25


# =============================================================================
# NÍVEIS
# =============================================================================

class ConfidenceLevel(IntEnum):
    INVALID = 0
    POOR = 1
    PARTIAL = 2
    GOOD = 3
    HIGH = 4


# =============================================================================
# RESULTADOS
# =============================================================================

@dataclass
class LaneConfidenceResult:
    confidence: float
    level: ConfidenceLevel

    detection_score: float
    geometry_score: float
    tracking_score: float
    projection_score: float
    consistency_score: float

    valid: bool
    safe_for_adas: bool

    reason: Optional[str] = None


@dataclass
class SceneConfidenceResult:
    confidence: float
    level: ConfidenceLevel

    lane_score: float
    geometry_score: float
    tracking_score: float
    projection_score: float
    consistency_score: float

    valid: bool
    safe_for_adas: bool

    reason: Optional[str] = None


# =============================================================================
# UTILITÁRIOS
# =============================================================================

def _clip01(value: float) -> float:
    if not np.isfinite(value):
        return 0.0

    return float(
        np.clip(value, 0.0, 1.0)
    )


def _mean_confidence(
    points,
) -> float:

    values = []

    for point in points:

        if not point.valid:
            continue

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


# =============================================================================
# AVALIADOR
# =============================================================================

class LaneConfidence:
    """
    Calcula a confiança da percepção de lanes.

    A confiança é deliberadamente conservadora.

    Uma detecção pode ser visualmente plausível, mas ainda assim
    não ser considerada segura para atuação do ADAS.
    """

    def __init__(
        self,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        safe_confidence: float = DEFAULT_SAFE_CONFIDENCE,
        high_confidence: float = DEFAULT_HIGH_CONFIDENCE,
        min_points: int = DEFAULT_MIN_POINTS,
        good_points: int = DEFAULT_GOOD_POINTS,
        excellent_points: int = DEFAULT_EXCELLENT_POINTS,
        max_fit_error: float = DEFAULT_MAX_FIT_ERROR,
        good_fit_error: float = DEFAULT_GOOD_FIT_ERROR,
        excellent_fit_error: float = DEFAULT_EXCELLENT_FIT_ERROR,
        max_missed_frames: int = DEFAULT_MAX_MISSED_FRAMES,
        max_age_frames: int = DEFAULT_MAX_AGE_FRAMES,
        min_lane_width: float = DEFAULT_MIN_LANE_WIDTH,
        max_lane_width: float = DEFAULT_MAX_LANE_WIDTH,
        max_offset: float = DEFAULT_MAX_OFFSET,
    ) -> None:

        self.min_confidence = float(
            np.clip(
                min_confidence,
                0.0,
                1.0,
            )
        )

        self.safe_confidence = float(
            np.clip(
                safe_confidence,
                self.min_confidence,
                1.0,
            )
        )

        self.high_confidence = float(
            np.clip(
                high_confidence,
                self.safe_confidence,
                1.0,
            )
        )

        self.min_points = max(
            1,
            int(min_points),
        )

        self.good_points = max(
            self.min_points,
            int(good_points),
        )

        self.excellent_points = max(
            self.good_points,
            int(excellent_points),
        )

        self.max_fit_error = max(
            0.001,
            float(max_fit_error),
        )

        self.good_fit_error = max(
            0.001,
            min(
                float(good_fit_error),
                self.max_fit_error,
            ),
        )

        self.excellent_fit_error = max(
            0.001,
            min(
                float(excellent_fit_error),
                self.good_fit_error,
            ),
        )

        self.max_missed_frames = max(
            0,
            int(max_missed_frames),
        )

        self.max_age_frames = max(
            1,
            int(max_age_frames),
        )

        self.min_lane_width = max(
            1.0,
            float(min_lane_width),
        )

        self.max_lane_width = max(
            self.min_lane_width,
            float(max_lane_width),
        )

        self.max_offset = max(
            0.01,
            float(max_offset),
        )

        self.last_result: Optional[
            SceneConfidenceResult
        ] = None

    # =========================================================================
    # LINHA
    # =========================================================================

    def _point_score(
        self,
        line: LaneLine,
    ) -> float:

        count = line.point_count()

        if count < self.min_points:
            return 0.0

        if count >= self.excellent_points:
            return 1.0

        if count >= self.good_points:

            ratio = (
                count
                - self.good_points
            ) / max(
                1,
                self.excellent_points
                - self.good_points,
            )

            return 0.75 + (
                0.25 * _clip01(ratio)
            )

        ratio = (
            count
            - self.min_points
        ) / max(
            1,
            self.good_points
            - self.min_points,
        )

        return 0.50 + (
            0.25 * _clip01(ratio)
        )

    def _direct_detection_score(
        self,
        line: LaneLine,
    ) -> float:

        if not line.valid:
            return 0.0

        confidence = _clip01(
            float(line.confidence)
        )

        point_score = self._point_score(
            line
        )

        direct_score = (
            1.0
            if line.detected_directly
            else 0.55
        )

        return _clip01(
            0.60 * confidence
            + 0.25 * point_score
            + 0.15 * direct_score
        )

    # =========================================================================
    # MODELO
    # =========================================================================

    def _fit_score(
        self,
        model: LaneModel,
    ) -> float:

        polynomial = model.polynomial

        if polynomial is None:
            return 0.0

        if not polynomial.valid:
            return 0.0

        error = float(
            polynomial.fit_error
        )

        if not np.isfinite(error):
            return 0.0

        if error <= self.excellent_fit_error:
            return 1.0

        if error <= self.good_fit_error:

            ratio = (
                error
                - self.excellent_fit_error
            ) / (
                self.good_fit_error
                - self.excellent_fit_error
            )

            return 1.0 - (
                0.25 * _clip01(ratio)
            )

        if error <= self.max_fit_error:

            ratio = (
                error
                - self.good_fit_error
            ) / (
                self.max_fit_error
                - self.good_fit_error
            )

            return 0.75 - (
                0.50 * _clip01(ratio)
            )

        return 0.0

    # =========================================================================
    # TRACKING
    # =========================================================================

    def _tracking_score(
        self,
        line: LaneLine,
    ) -> float:

        if not line.valid:
            return 0.0

        missed = max(
            0,
            int(line.missed_frames),
        )

        age = max(
            0,
            int(line.age_frames),
        )

        if missed > self.max_missed_frames:
            return 0.0

        missed_score = (
            1.0
            - (
                missed
                / max(
                    1,
                    self.max_missed_frames + 1,
                )
            )
        )

        age_score = _clip01(
            age
            / max(
                1,
                self.max_age_frames,
            )
        )

        if age == 0:
            age_score = 0.5

        tracked_score = (
            1.0
            if line.valid
            else 0.0
        )

        return _clip01(
            0.55 * tracked_score
            + 0.30 * missed_score
            + 0.15 * age_score
        )

    # =========================================================================
    # PROJEÇÃO
    # =========================================================================

    def _projection_score(
        self,
        model: LaneModel,
    ) -> float:

        if not model.line.valid:
            return 0.0

        projection = model.projection

        if projection is None:
            return 0.0

        if not projection.valid:
            return 0.0

        quality = projection.quality

        values = {
            "none": 0.0,
            "low": 0.35,
            "medium": 0.65,
            "high": 0.90,
        }

        return values.get(
            getattr(
                quality,
                "value",
                str(quality),
            ),
            0.0,
        )

    # =========================================================================
    # LINHA COMPLETA
    # =========================================================================

    def evaluate_lane(
        self,
        lane: LaneModel,
    ) -> LaneConfidenceResult:

        if lane is None:
            return LaneConfidenceResult(
                confidence=0.0,
                level=ConfidenceLevel.INVALID,
                detection_score=0.0,
                geometry_score=0.0,
                tracking_score=0.0,
                projection_score=0.0,
                consistency_score=0.0,
                valid=False,
                safe_for_adas=False,
                reason="Lane inexistente.",
            )

        detection_score = (
            self._direct_detection_score(
                lane.line
            )
        )

        geometry_score = (
            self._fit_score(
                lane
            )
        )

        tracking_score = (
            self._tracking_score(
                lane.line
            )
        )

        projection_score = (
            self._projection_score(
                lane
            )
        )

        consistency_score = _clip01(
            (
                detection_score
                + geometry_score
                + tracking_score
            ) / 3.0
        )

        confidence = _clip01(
            0.45 * detection_score
            + 0.25 * geometry_score
            + 0.20 * tracking_score
            + 0.10 * projection_score
        )

        valid = (
            lane.line.valid
            and lane.line.point_count()
            >= self.min_points
            and confidence
            >= self.min_confidence
        )

        if confidence >= self.high_confidence:
            level = ConfidenceLevel.HIGH

        elif confidence >= self.safe_confidence:
            level = ConfidenceLevel.GOOD

        elif confidence >= self.min_confidence:
            level = ConfidenceLevel.PARTIAL

        elif confidence > 0.0:
            level = ConfidenceLevel.POOR

        else:
            level = ConfidenceLevel.INVALID

        return LaneConfidenceResult(
            confidence=confidence,
            level=level,
            detection_score=detection_score,
            geometry_score=geometry_score,
            tracking_score=tracking_score,
            projection_score=projection_score,
            consistency_score=consistency_score,
            valid=valid,
            safe_for_adas=(
                valid
                and confidence
                >= self.safe_confidence
            ),
            reason=(
                None
                if valid
                else "Confiança insuficiente."
            ),
        )

    # =========================================================================
    # GEOMETRIA
    # =========================================================================

    def _geometry_score(
        self,
        geometry: Optional[LaneGeometry],
    ) -> float:

        if geometry is None:
            return 0.0

        if not geometry.valid:
            return 0.0

        score = 1.0

        width = geometry.lane_width

        if width is None:
            return 0.50

        width = float(width)

        if not np.isfinite(width):
            return 0.0

        if (
            width < self.min_lane_width
            or width > self.max_lane_width
        ):
            return 0.0

        normalized_offset = (
            geometry.normalized_offset
        )

        if normalized_offset is not None:

            normalized_offset = float(
                normalized_offset
            )

            if not np.isfinite(
                normalized_offset
            ):
                return 0.0

            if abs(
                normalized_offset
            ) > self.max_offset:
                score *= 0.50

        if geometry.heading_error is not None:

            heading = abs(
                float(
                    geometry.heading_error
                )
            )

            if np.isfinite(heading):

                # Até ~10° é considerado
                # geometricamente confortável.
                score *= _clip01(
                    1.0 - heading / 20.0
                )

        if geometry.confidence > 0.0:
            score *= _clip01(
                float(
                    geometry.confidence
                )
            )

        return _clip01(score)

    # =========================================================================
    # CENA
    # =========================================================================

    def evaluate_scene(
        self,
        lanes: Sequence[LaneModel],
        geometry: Optional[LaneGeometry] = None,
        require_two_lanes: bool = True,
    ) -> SceneConfidenceResult:

        valid_lanes = [
            lane
            for lane in lanes
            if lane is not None
        ]

        if not valid_lanes:

            result = SceneConfidenceResult(
                confidence=0.0,
                level=ConfidenceLevel.INVALID,
                lane_score=0.0,
                geometry_score=0.0,
                tracking_score=0.0,
                projection_score=0.0,
                consistency_score=0.0,
                valid=False,
                safe_for_adas=False,
                reason="Nenhuma lane disponível.",
            )

            self.last_result = result

            return result

        lane_results = [
            self.evaluate_lane(
                lane
            )
            for lane in valid_lanes
        ]

        lane_score = float(
            np.mean(
                [
                    result.confidence
                    for result in lane_results
                ]
            )
        )

        geometry_score = float(
            np.mean(
                [
                    result.geometry_score
                    for result in lane_results
                ]
            )
        )

        tracking_score = float(
            np.mean(
                [
                    result.tracking_score
                    for result in lane_results
                ]
            )
        )

        projection_score = float(
            np.mean(
                [
                    result.projection_score
                    for result in lane_results
                ]
            )
        )

        geometry_global = (
            self._geometry_score(
                geometry
            )
        )

        if geometry is not None:
            geometry_score = (
                0.65 * geometry_score
                + 0.35 * geometry_global
            )

        confidence_values = [
            result.confidence
            for result in lane_results
        ]

        consistency_score = _clip01(
            1.0
            - (
                np.std(
                    confidence_values
                )
                if len(
                    confidence_values
                ) > 1
                else 0.0
            )
        )

        confidence = _clip01(
            0.40 * lane_score
            + 0.25 * geometry_score
            + 0.15 * tracking_score
            + 0.10 * projection_score
            + 0.10 * consistency_score
        )

        sufficient_lanes = (
            len(valid_lanes) >= 2
            if require_two_lanes
            else len(valid_lanes) >= 1
        )

        valid = (
            sufficient_lanes
            and lane_score
            >= self.min_confidence
        )

        safe_for_adas = (
            valid
            and confidence
            >= self.safe_confidence
            and geometry_global
            >= self.min_confidence
        )

        if confidence >= self.high_confidence:
            level = ConfidenceLevel.HIGH

        elif confidence >= self.safe_confidence:
            level = ConfidenceLevel.GOOD

        elif confidence >= self.min_confidence:
            level = ConfidenceLevel.PARTIAL

        elif confidence > 0.0:
            level = ConfidenceLevel.POOR

        else:
            level = ConfidenceLevel.INVALID

        reason = None

        if not sufficient_lanes:
            reason = (
                "Quantidade insuficiente de lanes."
            )

        elif not valid:
            reason = (
                "Confiança das lanes insuficiente."
            )

        elif not safe_for_adas:
            reason = (
                "Percepção válida, porém "
                "não suficientemente confiável "
                "para ADAS."
            )

        result = SceneConfidenceResult(
            confidence=confidence,
            level=level,
            lane_score=lane_score,
            geometry_score=geometry_score,
            tracking_score=tracking_score,
            projection_score=projection_score,
            consistency_score=consistency_score,
            valid=valid,
            safe_for_adas=safe_for_adas,
            reason=reason,
        )

        self.last_result = result

        return result

    # =========================================================================
    # API DE COMPATIBILIDADE
    # =========================================================================

    def evaluate(
        self,
        lanes: Sequence[LaneModel],
        geometry: Optional[LaneGeometry] = None,
        require_two_lanes: bool = True,
    ) -> SceneConfidenceResult:

        return self.evaluate_scene(
            lanes=lanes,
            geometry=geometry,
            require_two_lanes=require_two_lanes,
        )

    def calculate(
        self,
        lanes: Sequence[LaneModel],
        geometry: Optional[LaneGeometry] = None,
        require_two_lanes: bool = True,
    ) -> SceneConfidenceResult:

        return self.evaluate_scene(
            lanes=lanes,
            geometry=geometry,
            require_two_lanes=require_two_lanes,
        )


# =============================================================================
# FACTORY
# =============================================================================

def create_default_lane_confidence(
    **kwargs,
) -> LaneConfidence:

    return LaneConfidence(**kwargs)


__all__ = [
    "ConfidenceLevel",
    "LaneConfidenceResult",
    "SceneConfidenceResult",
    "LaneConfidence",
    "create_default_lane_confidence",
]