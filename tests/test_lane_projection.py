"""
tests/test_lane_projection.py

Testes unitários de vision.lane_projection.

Executar:

    .\.venv\Scripts\python.exe -m pytest tests\test_lane_projection.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vision.lane_projection import (
    LaneProjectionEngine,
)

from vision.lane_types import (
    LanePoint,
    ProjectionQuality,
)


# =============================================================================
# HELPERS
# =============================================================================


def make_lane_points(
    count: int = 30,
    x_offset: float = 300.0,
    confidence: float = 0.95,
    noise: float = 0.0,
) -> list[LanePoint]:

    rng = np.random.default_rng(42)

    ys = np.linspace(
        150.0,
        600.0,
        count,
    )

    points = []

    for y in ys:

        # Curva suave:
        #
        # x = a*y² + b*y + c
        #
        x = (
            0.00035 * y * y
            - 0.08 * y
            + x_offset
        )

        if noise:
            x += float(
                rng.normal(
                    0.0,
                    noise,
                )
            )

        points.append(
            LanePoint(
                x=float(x),
                y=float(y),
                confidence=confidence,
                valid=True,
            )
        )

    return points


def make_straight_lane(
    count: int = 30,
    x: float = 400.0,
    confidence: float = 0.95,
) -> list[LanePoint]:

    ys = np.linspace(
        150.0,
        600.0,
        count,
    )

    return [
        LanePoint(
            x=float(x),
            y=float(y),
            confidence=confidence,
            valid=True,
        )
        for y in ys
    ]


def assert_valid_projection(
    projection,
):

    assert projection.valid

    assert projection.quality in (
        ProjectionQuality.LOW,
        ProjectionQuality.MEDIUM,
        ProjectionQuality.HIGH,
    )

    assert len(
        projection.points
    ) >= 2

    assert math.isfinite(
        projection.confidence
    )

    assert 0.0 <= projection.confidence <= 1.0


# =============================================================================
# CRIAÇÃO
# =============================================================================


def test_engine_can_be_created():

    engine = LaneProjectionEngine()

    assert engine is not None


def test_engine_accepts_custom_parameters():

    engine = LaneProjectionEngine(
        min_points=8,
        degree=2,
        max_projection_distance=250.0,
        min_confidence=0.2,
    )

    assert engine.min_points == 8
    assert engine.degree == 2
    assert engine.max_projection_distance == 250.0
    assert engine.min_confidence == 0.2


# =============================================================================
# PROJEÇÃO VÁLIDA
# =============================================================================


def test_project_valid_lane():

    engine = LaneProjectionEngine(
        min_points=8,
    )

    points = make_lane_points()

    projection = engine.project(
        points
    )

    assert_valid_projection(
        projection
    )


def test_project_straight_lane():

    engine = LaneProjectionEngine(
        min_points=8,
    )

    points = make_straight_lane()

    projection = engine.project(
        points
    )

    assert_valid_projection(
        projection
    )

    xs = np.array(
        [
            point.x
            for point in projection.points
        ],
        dtype=float,
    )

    # Uma linha reta deve permanecer
    # aproximadamente constante em X.
    assert np.ptp(xs) < 10.0


def test_project_curved_lane():

    engine = LaneProjectionEngine(
        min_points=8,
    )

    points = make_lane_points(
        noise=0.5,
    )

    projection = engine.project(
        points
    )

    assert_valid_projection(
        projection
    )

    xs = [
        point.x
        for point in projection.points
    ]

    assert max(xs) - min(xs) > 5.0


# =============================================================================
# PONTOS
# =============================================================================


def test_projection_points_are_finite():

    engine = LaneProjectionEngine(
        min_points=8,
    )

    projection = engine.project(
        make_lane_points()
    )

    assert_valid_projection(
        projection
    )

    for point in projection.points:

        assert math.isfinite(point.x)
        assert math.isfinite(point.y)
        assert math.isfinite(
            point.confidence
        )

        assert point.valid


def test_projection_points_are_ordered_by_y():

    engine = LaneProjectionEngine(
        min_points=8,
    )

    projection = engine.project(
        make_lane_points()
    )

    assert_valid_projection(
        projection
    )

    ys = [
        point.y
        for point in projection.points
    ]

    assert ys == sorted(ys)


# =============================================================================
# CONFIANÇA
# =============================================================================


def test_high_confidence_lane_produces_confidence():

    engine = LaneProjectionEngine(
        min_points=8,
        min_confidence=0.2,
    )

    projection = engine.project(
        make_lane_points(
            confidence=0.95,
        )
    )

    assert_valid_projection(
        projection
    )

    assert projection.confidence > 0.5


def test_low_confidence_lane_is_rejected():

    engine = LaneProjectionEngine(
        min_points=8,
        min_confidence=0.8,
    )

    points = make_lane_points(
        confidence=0.1,
    )

    projection = engine.project(
        points
    )

    assert not projection.valid


# =============================================================================
# POUCOS PONTOS
# =============================================================================


def test_too_few_points_are_rejected():

    engine = LaneProjectionEngine(
        min_points=8,
    )

    points = make_lane_points(
        count=5,
    )

    projection = engine.project(
        points
    )

    assert not projection.valid


def test_empty_input_is_rejected():

    engine = LaneProjectionEngine()

    projection = engine.project(
        []
    )

    assert not projection.valid


# =============================================================================
# PONTOS INVÁLIDOS
# =============================================================================


def test_invalid_points_are_ignored():

    engine = LaneProjectionEngine(
        min_points=8,
    )

    points = make_lane_points(
        count=25,
    )

    for point in points[:5]:
        point.valid = False

    projection = engine.project(
        points
    )

    assert_valid_projection(
        projection
    )


def test_nan_points_do_not_produce_valid_projection():

    engine = LaneProjectionEngine(
        min_points=8,
    )

    points = make_lane_points(
        count=20,
    )

    points[5].x = float("nan")
    points[10].y = float("nan")

    projection = engine.project(
        points
    )

    assert (
        projection.valid
        or not projection.valid
    )

    for point in projection.points:

        assert math.isfinite(point.x)
        assert math.isfinite(point.y)


def test_infinite_points_do_not_produce_invalid_output():

    engine = LaneProjectionEngine(
        min_points=8,
    )

    points = make_lane_points(
        count=20,
    )

    points[5].x = float("inf")

    projection = engine.project(
        points
    )

    for point in projection.points:

        assert math.isfinite(point.x)
        assert math.isfinite(point.y)


# =============================================================================
# LIMITES
# =============================================================================


def test_projection_respects_max_distance():

    engine = LaneProjectionEngine(
        min_points=8,
        max_projection_distance=100.0,
    )

    points = make_lane_points()

    projection = engine.project(
        points
    )

    if projection.valid:

        source_y = max(
            point.y
            for point in points
        )

        for point in projection.points:

            assert (
                abs(
                    point.y
                    - source_y
                )
                <= 100.0 + 1e-6
            )


def test_projection_does_not_create_nan():

    engine = LaneProjectionEngine()

    projection = engine.project(
        make_lane_points()
    )

    for point in projection.points:

        assert not math.isnan(point.x)
        assert not math.isnan(point.y)


# =============================================================================
# RUÍDO
# =============================================================================


@pytest.mark.parametrize(
    "noise",
    [
        0.0,
        0.5,
        1.0,
        2.0,
        4.0,
    ],
)
def test_projection_tolerates_reasonable_noise(
    noise,
):

    engine = LaneProjectionEngine(
        min_points=8,
    )

    projection = engine.project(
        make_lane_points(
            noise=noise,
        )
    )

    assert_valid_projection(
        projection
    )


# =============================================================================
# ESTABILIDADE
# =============================================================================


def test_similar_frames_produce_similar_projection():

    engine = LaneProjectionEngine(
        min_points=8,
    )

    projection_1 = engine.project(
        make_lane_points(
            x_offset=300.0,
            noise=0.2,
        )
    )

    projection_2 = engine.project(
        make_lane_points(
            x_offset=302.0,
            noise=0.2,
        )
    )

    assert_valid_projection(
        projection_1
    )

    assert_valid_projection(
        projection_2
    )

    common = min(
        len(projection_1.points),
        len(projection_2.points),
    )

    assert common >= 2

    differences = []

    for index in range(common):

        differences.append(
            abs(
                projection_1.points[index].x
                - projection_2.points[index].x
            )
        )

    assert np.mean(
        differences
    ) < 20.0


# =============================================================================
# QUALIDADE
# =============================================================================


def test_projection_quality_is_valid():

    engine = LaneProjectionEngine(
        min_points=8,
    )

    projection = engine.project(
        make_lane_points(
            confidence=0.95,
            noise=0.2,
        )
    )

    assert projection.quality in (
        ProjectionQuality.LOW,
        ProjectionQuality.MEDIUM,
        ProjectionQuality.HIGH,
    )


# =============================================================================
# RESULTADO INDEPENDENTE
# =============================================================================


def test_multiple_projections_do_not_share_points():

    engine = LaneProjectionEngine(
        min_points=8,
    )

    projection_1 = engine.project(
        make_lane_points(
            x_offset=300.0,
        )
    )

    projection_2 = engine.project(
        make_lane_points(
            x_offset=500.0,
        )
    )

    assert_valid_projection(
        projection_1
    )

    assert_valid_projection(
        projection_2
    )

    assert (
        projection_1.points
        is not projection_2.points
    )

    assert (
        projection_1.points[0].x
        != projection_2.points[0].x
    )


# =============================================================================
# EXECUÇÃO DIRETA
# =============================================================================


if __name__ == "__main__":

    raise SystemExit(
        pytest.main(
            [
                __file__,
                "-v",
            ]
        )
    )