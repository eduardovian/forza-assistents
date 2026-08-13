"""
tests/test_yolop_pipeline.py

Teste de integração da pipeline YOLOP/ADAS.

Fluxo:

    yolop_test.png
        ↓
       ROI
        ↓
      YOLOP
        ↓
LaneDetectionResult
        ├──────────────→ LaneGeometry
        │                    ↓
        │              LaneGeometryResult
        │                    ↓
        │              ADASStateEstimator
        │
        ↓
   LaneTracker
        ↓
   TrackedLane
        ↓
   LaneModel
        ↓
 LaneProjectionEngine
        ↓
 LaneAssignment
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np


# =============================================================================
# PATH
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IMAGE_PATH = ROOT / "yolop_test.png"


# =============================================================================
# ROI
# =============================================================================
#
# ROI aplicada sobre a imagem original.
#
# Atualmente:
#   X = 0% -> 100%
#   Y = 40% -> 100%
#
# Substitua somente estes valores quando quiser colocar exatamente
# a ROI definitiva utilizada pelo sistema.
#

ROI_X1_RATIO = 0.00
ROI_Y1_RATIO = 0.40

ROI_X2_RATIO = 1.00
ROI_Y2_RATIO = 1.00


# =============================================================================
# IMAGE LOADING
# =============================================================================

def load_image(path: Path) -> np.ndarray:
    """
    Carrega uma imagem usando bytes + imdecode.

    Isso evita problemas do OpenCV com caminhos Unicode no Windows,
    como:

        Área de Trabalho
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Imagem não encontrada: {path}"
        )

    data = np.frombuffer(
        path.read_bytes(),
        dtype=np.uint8,
    )

    image = cv2.imdecode(
        data,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise RuntimeError(
            f"Não foi possível decodificar a imagem: {path}"
        )

    return image


# =============================================================================
# ROI
# =============================================================================

def crop_roi(
    image: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    Recorta a ROI definida acima.

    Retorna:

        roi,
        (x1, y1, x2, y2)
    """

    height, width = image.shape[:2]

    x1 = int(
        width * ROI_X1_RATIO
    )

    y1 = int(
        height * ROI_Y1_RATIO
    )

    x2 = int(
        width * ROI_X2_RATIO
    )

    y2 = int(
        height * ROI_Y2_RATIO
    )

    x1 = max(
        0,
        min(
            x1,
            width - 1,
        ),
    )

    y1 = max(
        0,
        min(
            y1,
            height - 1,
        ),
    )

    x2 = max(
        x1 + 1,
        min(
            x2,
            width,
        ),
    )

    y2 = max(
        y1 + 1,
        min(
            y2,
            height,
        ),
    )

    roi = image[
        y1:y2,
        x1:x2,
    ]

    return (
        roi,
        (
            x1,
            y1,
            x2,
            y2,
        ),
    )


# =============================================================================
# PRINT HELPERS
# =============================================================================

def print_header(
    title: str,
) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def print_detection_result(
    result,
) -> None:

    print()
    print("YOLOP RESULT")
    print("-" * 72)

    print(
        f"valid: {getattr(result, 'valid', None)}"
    )

    print(
        "num_lanes_detected: "
        f"{getattr(result, 'num_lanes_detected', None)}"
    )

    lanes = getattr(
        result,
        "lanes",
        None,
    )

    if lanes is None:
        print("lanes: None")
        return

    print(
        f"lanes: {len(lanes)}"
    )

    for lane_index, lane in enumerate(
        lanes
    ):

        print(
            f"  lane[{lane_index}]: "
            f"{len(lane)} pontos"
        )

        for point in lane[:3]:

            print(
                "    "
                f"x={point.x:.2f}, "
                f"y={point.y:.2f}, "
                f"confidence={point.confidence:.3f}, "
                f"valid={point.valid}"
            )


def print_tracking_result(
    result,
) -> None:

    print()
    print("LANE TRACKER")
    print("-" * 72)

    print(
        f"valid: {result.valid}"
    )

    print(
        f"frame_index: {result.frame_index}"
    )

    print(
        f"detected_count: {result.detected_count}"
    )

    print(
        f"stable_count: {result.stable_count}"
    )

    print(
        f"lost_count: {result.lost_count}"
    )

    for lane in result.lanes:

        print(
            f"  track_id={lane.track_id} | "
            f"points={len(lane.points)} | "
            f"confidence={lane.confidence:.3f} | "
            f"stable={lane.stable} | "
            f"missed={lane.missed_frames}"
        )


# =============================================================================
# TEST
# =============================================================================

def test_yolop_full_pipeline() -> None:

    print_header(
        "YOLOP FULL PIPELINE TEST"
    )

    # =========================================================================
    # 1. IMAGE
    # =========================================================================

    image = load_image(
        IMAGE_PATH
    )

    image_height, image_width = (
        image.shape[:2]
    )

    print()
    print("IMAGE")
    print("-" * 72)

    print(
        f"path: {IMAGE_PATH}"
    )

    print(
        f"size: "
        f"{image_width}x{image_height}"
    )

    assert image.size > 0

    # =========================================================================
    # 2. ROI
    # =========================================================================

    roi, roi_rect = crop_roi(
        image
    )

    roi_x1, roi_y1, roi_x2, roi_y2 = (
        roi_rect
    )

    print()
    print("ROI")
    print("-" * 72)

    print(
        f"x1={roi_x1}, "
        f"y1={roi_y1}, "
        f"x2={roi_x2}, "
        f"y2={roi_y2}"
    )

    print(
        f"size: "
        f"{roi.shape[1]}x{roi.shape[0]}"
    )

    assert roi.size > 0

    # =========================================================================
    # 3. YOLOP
    # =========================================================================

    from vision.yolop_detector import (
        create_default_detector,
    )

    print()
    print("YOLOP")
    print("-" * 72)

    detector = create_default_detector()

    print(
        "detector: OK"
    )

    detection = detector.detect(
        roi
    )

    assert detection is not None

    print_detection_result(
        detection
    )

    # =========================================================================
    # 4. LANE TRACKER
    # =========================================================================

    from vision.lane_tracker import (
        LaneTracker,
    )

    print()
    print("LANE TRACKER")
    print("-" * 72)

    tracker = LaneTracker()

    tracking = tracker.update(
        detection,
        timestamp=0.0,
    )

    assert tracking is not None

    print_tracking_result(
        tracking
    )

    # =========================================================================
    # 5. LANE GEOMETRY
    # =========================================================================

    from vision.lane_geometry import (
        LaneGeometry,
    )

    print()
    print("LANE GEOMETRY")
    print("-" * 72)

    geometry = LaneGeometry(
        screen_width=image_width,
        screen_height=image_height,
        roi=(
            roi_x1,
            roi_y1,
            roi_x2,
            roi_y2,
        ),
    )

    geometry_result = geometry.compute(
        detection
    )

    assert geometry_result is not None

    print(
        f"valid: "
        f"{geometry_result.valid}"
    )

    print(
        f"lane_width: "
        f"{geometry_result.lane_width:.2f}"
    )

    print(
        f"lateral_error: "
        f"{geometry_result.lateral_error:.4f}"
    )

    print(
        f"heading_error: "
        f"{geometry_result.heading_error:.4f}"
    )

    print(
        f"curvature: "
        f"{geometry_result.curvature:.6f}"
    )

    print(
        f"geometry_confidence: "
        f"{geometry_result.geometry_confidence:.4f}"
    )

    print(
        f"observed_span: "
        f"{geometry_result.observed_span:.2f}"
    )

    print(
        f"enough_for_projection: "
        f"{geometry_result.enough_for_projection}"
    )

    # =========================================================================
    # 6. LANE MODELS
    # =========================================================================

    from vision.lane_model import (
        build_lane_model,
    )

    print()
    print("LANE MODELS")
    print("-" * 72)

    lane_models = []

    for track in tracking.active_lanes:

        model = build_lane_model(
            lane_id=track.track_id,
            points=track.points,
        )

        if model.valid:
            model.tracked = True
            model.stable = track.is_stable(
                tracker.min_stable_frames
            )

            lane_models.append(
                model
            )

        print(
            f"track_id={track.track_id} | "
            f"valid={model.valid} | "
            f"points={len(track.points)} | "
            f"polynomial="
            f"{model.polynomial is not None}"
        )

    # =========================================================================
    # 7. LANE PROJECTION
    # =========================================================================

    from vision.lane_projection import (
        LaneProjectionEngine,
    )

    print()
    print("LANE PROJECTION")
    print("-" * 72)

    projection_engine = (
        LaneProjectionEngine()
    )

    projections = []

    for model in lane_models:

        projection = (
            projection_engine.project(
                model.line
            )
        )

        projections.append(
            projection
        )

        print(
            f"lane_id={model.lane_id} | "
            f"valid={projection.valid} | "
            f"confidence={projection.confidence:.3f} | "
            f"quality={projection.quality}"
        )

    # =========================================================================
    # 8. LANE ASSIGNMENT
    # =========================================================================

    from vision.lane_assignment import (
        LaneAssignment,
    )

    print()
    print("LANE ASSIGNMENT")
    print("-" * 72)

    assignment = LaneAssignment()

    assignment_result = assignment.assign(
        lanes=lane_models,
        frame_width=roi.shape[1],
        frame_height=roi.shape[0],
    )

    assert assignment_result is not None

    print(
        f"valid: "
        f"{assignment_result.valid}"
    )

    print(
        f"current_lane_id: "
        f"{assignment_result.current_lane_id}"
    )

    print(
        f"lane_width: "
        f"{assignment_result.lane_width:.2f}"
    )

    print(
        f"lateral_offset: "
        f"{assignment_result.lateral_offset:.2f}"
    )

    print(
        f"normalized_offset: "
        f"{assignment_result.normalized_offset:.4f}"
    )

    print(
        f"confidence: "
        f"{assignment_result.confidence:.4f}"
    )

    print(
        f"left_lanes: "
        f"{len(assignment_result.left_lanes)}"
    )

    print(
        f"right_lanes: "
        f"{len(assignment_result.right_lanes)}"
    )

    # =========================================================================
    # 9. ADAS STATE
    # =========================================================================

    from vision.adas_state import (
        ADASStateEstimator,
    )

    print()
    print("ADAS STATE")
    print("-" * 72)

    state_estimator = (
        ADASStateEstimator()
    )

    adas_result = (
        state_estimator.update(
            geometry_result,
            timestamp=0.0,
        )
    )

    assert adas_result is not None

    print(
        f"state: "
        f"{adas_result.state}"
    )

    print(
        f"warning_side: "
        f"{adas_result.warning_side}"
    )

    print(
        f"lateral_error: "
        f"{adas_result.lateral_error:.4f}"
    )

    print(
        f"heading_error: "
        f"{adas_result.heading_error:.4f}"
    )

    print(
        f"confidence: "
        f"{adas_result.confidence:.4f}"
    )

    print(
        f"valid: "
        f"{adas_result.valid}"
    )

    # =========================================================================
    # 10. FINAL RESULT
    # =========================================================================

    print_header(
        "FINAL RESULT"
    )

    print(
        "YOLOP            : OK"
    )

    print(
        "LaneTracker      : OK"
    )

    print(
        "LaneGeometry     : "
        f"{'VALID' if geometry_result.valid else 'INVALID'}"
    )

    print(
        f"LaneModels       : "
        f"{len(lane_models)}"
    )

    print(
        f"LaneProjection   : "
        f"{len(projections)}"
    )

    print(
        "LaneAssignment   : "
        f"{'VALID' if assignment_result.valid else 'INVALID'}"
    )

    print(
        "ADAS             : "
        f"{adas_result.state.value}"
    )

    print("=" * 72)