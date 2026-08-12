"""
tests/test_lane_tracker.py

Testes unitários do LaneTracker.

Executar:

    .\.venv\Scripts\python.exe -m pytest tests/test_lane_tracker.py -v
"""

from __future__ import annotations

import math

from vision.lane_tracker import LaneTracker
from vision.lane_types import LaneLine, LanePoint


# =============================================================================
# HELPERS
# =============================================================================


def make_lane(
    x: float,
    confidence: float = 0.9,
    lane_id: int | None = None,
) -> LaneLine:
    """Cria uma lane artificial vertical."""

    points = [
        LanePoint(
            x=x,
            y=480.0,
            confidence=confidence,
            valid=True,
        ),
        LanePoint(
            x=x - 5.0,
            y=420.0,
            confidence=confidence,
            valid=True,
        ),
        LanePoint(
            x=x - 10.0,
            y=360.0,
            confidence=confidence,
            valid=True,
        ),
        LanePoint(
            x=x - 15.0,
            y=300.0,
            confidence=confidence,
            valid=True,
        ),
        LanePoint(
            x=x - 20.0,
            y=240.0,
            confidence=confidence,
            valid=True,
        ),
    ]

    return LaneLine(
        lane_id=lane_id,
        points=points,
        confidence=confidence,
        valid=True,
    )


def get_track_ids(result):
    return [lane.track_id for lane in result.lanes]


def get_track_by_x(result, x: float):
    return min(
        result.lanes,
        key=lambda lane: abs(
            lane.current_center_x - x
        ),
    )


# =============================================================================
# TESTE 1 — CRIAÇÃO
# =============================================================================


def test_tracker_creates_tracks():

    tracker = LaneTracker(
        min_points=4,
    )

    detections = [
        make_lane(300.0),
        make_lane(700.0),
    ]

    result = tracker.update(
        detections,
        timestamp=1.0,
    )

    assert result.valid
    assert result.detected_count == 2
    assert len(result.lanes) == 2

    assert len(set(get_track_ids(result))) == 2

    for track in result.lanes:
        assert track.age == 1
        assert track.missed_frames == 0
        assert track.detected_this_frame
        assert track.current_center_x is not None


# =============================================================================
# TESTE 2 — IDENTIDADE ENTRE FRAMES
# =============================================================================


def test_tracker_preserves_identity():

    tracker = LaneTracker(
        min_points=4,
        match_distance=90.0,
    )

    frame_1 = [
        make_lane(300.0),
        make_lane(700.0),
    ]

    result_1 = tracker.update(
        frame_1,
        timestamp=1.0,
    )

    left_id = get_track_by_x(
        result_1,
        300.0,
    ).track_id

    right_id = get_track_by_x(
        result_1,
        700.0,
    ).track_id

    frame_2 = [
        make_lane(310.0),
        make_lane(690.0),
    ]

    result_2 = tracker.update(
        frame_2,
        timestamp=2.0,
    )

    left_track = get_track_by_x(
        result_2,
        310.0,
    )

    right_track = get_track_by_x(
        result_2,
        690.0,
    )

    assert left_track.track_id == left_id
    assert right_track.track_id == right_id

    assert left_track.missed_frames == 0
    assert right_track.missed_frames == 0


# =============================================================================
# TESTE 3 — MOVIMENTO CONTÍNUO
# =============================================================================


def test_tracker_follows_lane_motion():

    tracker = LaneTracker(
        min_points=4,
        match_distance=100.0,
    )

    positions = [
        300.0,
        310.0,
        320.0,
        330.0,
        340.0,
    ]

    track_ids = []

    for index, x in enumerate(positions):

        result = tracker.update(
            [make_lane(x)],
            timestamp=float(index),
        )

        assert len(result.lanes) == 1

        track = result.lanes[0]

        track_ids.append(
            track.track_id
        )

        assert track.current_center_x is not None

    assert len(set(track_ids)) == 1


# =============================================================================
# TESTE 4 — ESTABILIDADE
# =============================================================================


def test_tracker_becomes_stable():

    tracker = LaneTracker(
        min_points=4,
        min_stable_frames=3,
    )

    for frame in range(1, 5):

        result = tracker.update(
            [make_lane(300.0)],
            timestamp=float(frame),
        )

    track = result.lanes[0]

    assert track.stable_frames >= 3
    assert track.is_stable(3)
    assert track.stable

    assert result.stable_count >= 1


# =============================================================================
# TESTE 5 — PERDA TEMPORÁRIA
# =============================================================================


def test_tracker_handles_temporary_loss():

    tracker = LaneTracker(
        min_points=4,
        max_missed_frames=3,
    )

    result_1 = tracker.update(
        [make_lane(300.0)],
        timestamp=1.0,
    )

    original_id = result_1.lanes[0].track_id

    result_2 = tracker.update(
        [],
        timestamp=2.0,
    )

    assert len(result_2.lanes) == 1

    lost_track = result_2.lanes[0]

    assert lost_track.track_id == original_id
    assert lost_track.missed_frames == 1
    assert not lost_track.detected_this_frame

    assert len(result_2.active_lanes) == 1


# =============================================================================
# TESTE 6 — RECUPERAÇÃO
# =============================================================================


def test_tracker_recovers_same_track():

    tracker = LaneTracker(
        min_points=4,
        max_missed_frames=3,
        match_distance=90.0,
    )

    result_1 = tracker.update(
        [make_lane(300.0)],
        timestamp=1.0,
    )

    original_id = result_1.lanes[0].track_id

    tracker.update(
        [],
        timestamp=2.0,
    )

    tracker.update(
        [],
        timestamp=3.0,
    )

    result_4 = tracker.update(
        [make_lane(305.0)],
        timestamp=4.0,
    )

    recovered = result_4.lanes[0]

    assert recovered.track_id == original_id
    assert recovered.missed_frames == 0
    assert recovered.detected_this_frame


# =============================================================================
# TESTE 7 — EXPIRAÇÃO
# =============================================================================


def test_tracker_expires_lost_track():

    tracker = LaneTracker(
        min_points=4,
        max_missed_frames=2,
    )

    result_1 = tracker.update(
        [make_lane(300.0)],
        timestamp=1.0,
    )

    original_id = result_1.lanes[0].track_id

    tracker.update(
        [],
        timestamp=2.0,
    )

    tracker.update(
        [],
        timestamp=3.0,
    )

    result_4 = tracker.update(
        [],
        timestamp=4.0,
    )

    assert all(
        track.track_id != original_id
        for track in result_4.active_lanes
    )


# =============================================================================
# TESTE 8 — NÃO TROCAR ESQUERDA/DIREITA
# =============================================================================


def test_tracker_does_not_swap_lanes():

    tracker = LaneTracker(
        min_points=4,
        match_distance=100.0,
    )

    result_1 = tracker.update(
        [
            make_lane(300.0),
            make_lane(700.0),
        ],
        timestamp=1.0,
    )

    left_id = get_track_by_x(
        result_1,
        300.0,
    ).track_id

    right_id = get_track_by_x(
        result_1,
        700.0,
    ).track_id

    result_2 = tracker.update(
        [
            make_lane(340.0),
            make_lane(660.0),
        ],
        timestamp=2.0,
    )

    left = get_track_by_x(
        result_2,
        340.0,
    )

    right = get_track_by_x(
        result_2,
        660.0,
    )

    assert left.track_id == left_id
    assert right.track_id == right_id


# =============================================================================
# TESTE 9 — CONFIANÇA
# =============================================================================


def test_tracker_confidence_is_valid():

    tracker = LaneTracker(
        min_points=4,
    )

    result = tracker.update(
        [
            make_lane(
                300.0,
                confidence=0.8,
            )
        ],
        timestamp=1.0,
    )

    track = result.lanes[0]

    assert math.isfinite(
        track.confidence
    )

    assert 0.0 <= track.confidence <= 1.0


# =============================================================================
# TESTE 10 — HISTÓRICO
# =============================================================================


def test_tracker_history_is_limited():

    tracker = LaneTracker(
        min_points=4,
        history_size=3,
    )

    for frame in range(10):

        tracker.update(
            [make_lane(300.0 + frame)],
            timestamp=float(frame),
        )

    track = tracker.tracks[0]

    assert len(track.history) <= 3


# =============================================================================
# TESTE 11 — MÚLTIPLAS LANES
# =============================================================================


def test_tracker_supports_multiple_lanes():

    tracker = LaneTracker(
        max_lanes=4,
        min_points=4,
    )

    detections = [
        make_lane(200.0),
        make_lane(400.0),
        make_lane(600.0),
        make_lane(800.0),
    ]

    result = tracker.update(
        detections,
        timestamp=1.0,
    )

    assert len(result.lanes) == 4
    assert result.detected_count == 4

    ids = get_track_ids(result)

    assert len(ids) == 4
    assert len(set(ids)) == 4


# =============================================================================
# TESTE 12 — ENTRADA VAZIA
# =============================================================================


def test_tracker_handles_empty_input():

    tracker = LaneTracker()

    result = tracker.update(
        [],
        timestamp=1.0,
    )

    assert result.detected_count == 0
    assert result.valid is False


# =============================================================================
# EXECUÇÃO DIRETA
# =============================================================================


if __name__ == "__main__":

    import pytest

    raise SystemExit(
        pytest.main(
            [
                __file__,
                "-v",
            ]
        )
    )