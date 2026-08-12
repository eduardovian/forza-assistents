"""
tests/test_lane_model.py

Testes unitários do vision.lane_model.

Executar:

    .\.venv\Scripts\python.exe -m pytest tests\test_lane_model.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vision.lane_model import (
    build_lane_model,
    classify_lane_quality,
    filter_lane_points,
    fit_lane_model,
    fit_polynomial,
    lane_confidence_score,
    lane_x_span,
    lane_y_span,
    prepare_lane_points,
    project_lane,
    remove_duplicate_y,
    remove_polynomial_outliers,
    sort_lane_points,
    update_lane_model,
    validate_lane_model,
)

from vision.lane_types import (
    LaneLine,
    LaneModel,
    LanePoint,
    LanePolynomial,
    LaneQuality,
    ProjectionQuality,
)


# =============================================================================
# HELPERS
# =============================================================================


def make_points(
    count: int = 20,
    x_offset: float = 300.0,
    confidence: float = 0.9,
    noise: float = 0.0,
) -> list[LanePoint]:
    """
    Gera uma lane sintética:

        x(y) = 0.0005*y² - 0.10*y + x_offset

    com pequena curvatura.
    """

    rng = np.random.default_rng(42)

    ys = np.linspace(
        100.0,
        500.0,
        count,
    )

    points = []

    for y in ys:

        x = (
            0.0005 * y * y
            - 0.10 * y
            + x_offset
        )

        if noise > 0.0:
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


def make_straight_points(
    count: int = 20,
    x: float = 400.0,
    confidence: float = 0.9,
) -> list[LanePoint]:

    ys = np.linspace(
        100.0,
        500.0,
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


# =============================================================================
# PREPARAÇÃO
# =============================================================================


def test_filter_lane_points_removes_invalid_points():

    points = make_points(
        count=10,
    )

    points.append(
        LanePoint(
            x=100.0,
            y=200.0,
            confidence=0.05,
            valid=True,
        )
    )

    points.append(
        LanePoint(
            x=100.0,
            y=250.0,
            confidence=0.9,
            valid=False,
        )
    )

    result = filter_lane_points(
        points,
        min_confidence=0.2,
    )

    assert len(result) == 10

    assert all(
        point.valid
        for point in result
    )

    assert all(
        point.confidence >= 0.2
        for point in result
    )


def test_sort_lane_points_orders_by_y():

    points = [
        LanePoint(
            x=300.0,
            y=500.0,
            confidence=0.9,
        ),
        LanePoint(
            x=300.0,
            y=100.0,
            confidence=0.9,
        ),
        LanePoint(
            x=300.0,
            y=300.0,
            confidence=0.9,
        ),
    ]

    result = sort_lane_points(points)

    assert [
        point.y
        for point in result
    ] == [
        100.0,
        300.0,
        500.0,
    ]


def test_remove_duplicate_y_keeps_highest_confidence():

    points = [
        LanePoint(
            x=300.0,
            y=100.0,
            confidence=0.4,
        ),
        LanePoint(
            x=305.0,
            y=100.0,
            confidence=0.9,
        ),
        LanePoint(
            x=400.0,
            y=200.0,
            confidence=0.8,
        ),
    ]

    result = remove_duplicate_y(points)

    assert len(result) == 2

    y100 = next(
        point
        for point in result
        if point.y == 100.0
    )

    assert y100.x == 305.0
    assert y100.confidence == 0.9


def test_prepare_lane_points():

    points = make_points(
        count=100,
    )

    result = prepare_lane_points(
        points,
        min_confidence=0.2,
        max_points=30,
    )

    assert len(result) <= 30
    assert len(result) >= 2

    assert all(
        result[i].y <= result[i + 1].y
        for i in range(len(result) - 1)
    )


# =============================================================================
# ESTATÍSTICAS
# =============================================================================


def test_lane_spans():

    points = make_points(
        count=20,
    )

    y_span = lane_y_span(points)
    x_span = lane_x_span(points)

    assert y_span > 300.0
    assert x_span > 0.0


def test_lane_confidence_score_is_bounded():

    points = make_points(
        count=30,
        confidence=0.9,
    )

    confidence = lane_confidence_score(
        points
    )

    assert math.isfinite(confidence)
    assert 0.0 <= confidence <= 1.0


# =============================================================================
# AJUSTE POLINOMIAL
# =============================================================================


def test_fit_polynomial_creates_valid_model():

    points = make_points(
        count=25,
        noise=0.5,
    )

    polynomial = fit_polynomial(
        points,
        min_points=8,
    )

    assert isinstance(
        polynomial,
        LanePolynomial,
    )

    assert polynomial.valid
    assert polynomial.sample_count == 25

    assert math.isfinite(
        polynomial.fit_error
    )

    assert math.isfinite(
        polynomial.confidence
    )

    assert 0.0 <= polynomial.confidence <= 1.0


def test_polynomial_reproduces_lane():

    points = make_points(
        count=30,
        noise=0.2,
    )

    polynomial = fit_polynomial(
        points,
        min_points=8,
    )

    assert polynomial.valid

    errors = []

    for point in points:

        predicted = polynomial.evaluate(
            point.y
        )

        errors.append(
            abs(
                predicted
                - point.x
            )
        )

    median_error = float(
        np.median(errors)
    )

    assert median_error < 5.0


def test_fit_polynomial_rejects_too_few_points():

    points = make_points(
        count=5,
    )

    polynomial = fit_polynomial(
        points,
        min_points=8,
    )

    assert not polynomial.valid
    assert polynomial.sample_count == 5


def test_fit_polynomial_rejects_small_y_span():

    points = [
        LanePoint(
            x=300.0 + index,
            y=100.0 + index,
            confidence=0.9,
        )
        for index in range(10)
    ]

    polynomial = fit_polynomial(
        points,
        min_points=8,
    )

    assert not polynomial.valid


def test_fit_polynomial_rejects_wrong_degree():

    points = make_points(
        count=20,
    )

    with pytest.raises(ValueError):

        fit_polynomial(
            points,
            degree=2,
        )


# =============================================================================
# OUTLIERS
# =============================================================================


def test_remove_polynomial_outliers():

    points = make_points(
        count=30,
        noise=0.2,
    )

    polynomial = fit_polynomial(
        points,
        min_points=8,
    )

    assert polynomial.valid

    points_with_outlier = list(points)

    points_with_outlier.append(
        LanePoint(
            x=1500.0,
            y=300.0,
            confidence=0.9,
        )
    )

    cleaned = remove_polynomial_outliers(
        points_with_outlier,
        polynomial,
    )

    assert len(cleaned) < len(
        points_with_outlier
    )


# =============================================================================
# MODELO ROBUSTO
# =============================================================================


def test_fit_lane_model():

    points = make_points(
        count=30,
        noise=0.5,
    )

    model = fit_lane_model(
        points,
        min_points=8,
    )

    assert model is not None
    assert model.valid
    assert model.sample_count >= 8


def test_fit_lane_model_returns_none_for_invalid_data():

    points = [
        LanePoint(
            x=300.0,
            y=100.0,
            confidence=0.1,
        )
        for _ in range(10)
    ]

    model = fit_lane_model(
        points,
        min_points=8,
        min_confidence=0.2,
    )

    assert model is None


# =============================================================================
# PROJEÇÃO
# =============================================================================


def test_project_lane():

    points = make_points(
        count=30,
    )

    polynomial = fit_polynomial(
        points,
        min_points=8,
    )

    assert polynomial.valid

    projection = project_lane(
        polynomial,
        points,
    )

    assert projection.valid
    assert projection.extrapolated

    assert len(
        projection.points
    ) >= 2

    assert projection.quality in (
        ProjectionQuality.LOW,
        ProjectionQuality.MEDIUM,
        ProjectionQuality.HIGH,
    )


def test_project_lane_rejects_invalid_polynomial():

    polynomial = LanePolynomial(
        valid=False,
    )

    points = make_points(
        count=20,
    )

    projection = project_lane(
        polynomial,
        points,
    )

    assert not projection.valid
    assert (
        projection.quality
        == ProjectionQuality.NONE
    )


def test_project_lane_rejects_none_polynomial():

    points = make_points(
        count=20,
    )

    projection = project_lane(
        None,
        points,
    )

    assert not projection.valid
    assert (
        projection.quality
        == ProjectionQuality.NONE
    )


# =============================================================================
# BUILD LANE MODEL
# =============================================================================


def test_build_lane_model():

    points = make_points(
        count=30,
        noise=0.5,
    )

    model = build_lane_model(
        lane_id=7,
        points=points,
    )

    assert isinstance(
        model,
        LaneModel,
    )

    assert model.lane_id == 7
    assert model.line.lane_id == 7

    assert model.polynomial is not None
    assert model.projection is not None

    assert model.valid

    assert model.line.detected_directly
    assert not model.line.projected


def test_build_lane_model_invalid_points():

    points = [
        LanePoint(
            x=300.0,
            y=100.0,
            confidence=0.05,
        )
        for _ in range(20)
    ]

    model = build_lane_model(
        lane_id=3,
        points=points,
    )

    assert not model.valid
    assert model.polynomial is None


# =============================================================================
# VALIDAÇÃO
# =============================================================================


def test_validate_lane_model():

    points = make_points(
        count=30,
    )

    model = build_lane_model(
        lane_id=1,
        points=points,
    )

    assert validate_lane_model(
        model
    )


def test_validate_lane_model_rejects_none():

    assert not validate_lane_model(
        None
    )


def test_validate_lane_model_rejects_invalid():

    points = make_points(
        count=30,
    )

    model = build_lane_model(
        lane_id=1,
        points=points,
    )

    model.valid = False

    assert not validate_lane_model(
        model
    )


# =============================================================================
# ATUALIZAÇÃO
# =============================================================================


def test_update_lane_model_preserves_lane_id():

    points_1 = make_points(
        count=30,
        x_offset=300.0,
    )

    model = build_lane_model(
        lane_id=42,
        points=points_1,
    )

    model.tracked = True
    model.stable = True

    points_2 = make_points(
        count=30,
        x_offset=320.0,
    )

    updated = update_lane_model(
        model,
        points_2,
    )

    assert updated.lane_id == 42
    assert updated.line.lane_id == 42

    assert updated.tracked
    assert updated.stable

    assert (
        updated.line.age_frames
        == model.line.age_frames + 1
    )

    assert updated.line.missed_frames == 0


# =============================================================================
# QUALIDADE
# =============================================================================


def test_classify_lane_quality_none():

    quality = classify_lane_quality(
        []
    )

    assert quality == LaneQuality.NONE


def test_classify_lane_quality_poor():

    points = make_points(
        count=3,
    )

    quality = classify_lane_quality(
        points
    )

    assert quality == LaneQuality.POOR


def test_classify_lane_quality_good_or_better():

    points = make_points(
        count=25,
        confidence=0.9,
        noise=0.2,
    )

    polynomial = fit_polynomial(
        points,
        min_points=8,
    )

    quality = classify_lane_quality(
        points,
        polynomial,
    )

    assert quality in (
        LaneQuality.GOOD,
        LaneQuality.EXCELLENT,
    )