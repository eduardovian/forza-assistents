"""
vision/lane_assignment.py

Semantic lane assignment.

YOLOP / Tracker / Geometry / LaneModel detect and model lane boundaries.

This module determines which corridor between adjacent lane boundaries
contains the vehicle.

Example with four detected boundaries:

    L0        L1        L2        L3
     |         |         |         |
     |   C0    |   C1    |   C2    |
     |         |         |         |

Four boundaries create three possible lanes/corridors:

    C0 = L0-L1
    C1 = L1-L2
    C2 = L2-L3

The current lane is the corridor containing the vehicle center.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

import numpy as np


# ============================================================================
# RESULT
# ============================================================================


@dataclass(frozen=True)
class LaneAssignmentResult:
    """
    Semantic assignment result.

    current_lane_id:
        Zero-based corridor index.

        With four boundaries:

            L0-L1 -> 0
            L1-L2 -> 1
            L2-L3 -> 2

    left_boundary_id:
        Index of the left boundary of the current lane.

    right_boundary_id:
        Index of the right boundary of the current lane.
    """

    valid: bool = False

    current_lane_id: Optional[int] = None

    left_boundary_id: Optional[int] = None
    right_boundary_id: Optional[int] = None

    lane_center_x: float = 0.0
    vehicle_x: float = 0.0

    lane_width: float = 0.0

    lateral_offset: float = 0.0
    normalized_offset: float = 0.0

    confidence: float = 0.0

    lane_count: int = 0

    left_lane_count: int = 0
    right_lane_count: int = 0

    reason: Optional[str] = None

    @property
    def corridor_count(self) -> int:
        """Number of possible corridors."""
        return max(0, self.lane_count - 1)

    @property
    def current_lane_index(self) -> int:
        """Compatibility alias."""
        if self.current_lane_id is None:
            return -1

        return int(self.current_lane_id)

    @property
    def left_boundary_index(self) -> int:
        """Compatibility alias."""
        if self.left_boundary_id is None:
            return -1

        return int(self.left_boundary_id)

    @property
    def right_boundary_index(self) -> int:
        """Compatibility alias."""
        if self.right_boundary_id is None:
            return -1

        return int(self.right_boundary_id)

    @property
    def is_valid(self) -> bool:
        """Compatibility alias."""
        return bool(self.valid)


# ============================================================================
# INTERNAL HELPERS
# ============================================================================


def _finite_float(
    value: Any,
    default: Optional[float] = None,
) -> Optional[float]:
    """Convert a value to a finite float."""

    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    if not np.isfinite(result):
        return default

    return result


def _clip01(value: Any) -> float:
    """Clamp value to [0, 1]."""

    result = _finite_float(value, 0.0)

    if result is None:
        return 0.0

    return float(
        np.clip(
            result,
            0.0,
            1.0,
        )
    )


def _first_attr(
    obj: Any,
    names: Sequence[str],
    default: Any = None,
) -> Any:
    """Return the first existing non-None attribute."""

    if obj is None:
        return default

    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)

            if value is not None:
                return value

    return default


def _extract_xy(
    point: Any,
) -> Optional[tuple[float, float]]:
    """Extract an (x, y) point from common representations."""

    if point is None:
        return None

    if isinstance(point, np.ndarray):
        point = point.tolist()

    if isinstance(point, (list, tuple)):

        if len(point) < 2:
            return None

        x = _finite_float(point[0])
        y = _finite_float(point[1])

        if x is None or y is None:
            return None

        return x, y

    x = _first_attr(
        point,
        (
            "x",
            "screen_x",
            "image_x",
        ),
    )

    y = _first_attr(
        point,
        (
            "y",
            "screen_y",
            "image_y",
        ),
    )

    x = _finite_float(x)
    y = _finite_float(y)

    if x is None or y is None:
        return None

    return x, y


# ============================================================================
# ENGINE
# ============================================================================


class LaneAssignment:
    """
    Assign detected lane boundaries to semantic lane corridors.

    API intentionally matches the current main.py:

        LaneAssignment(
            max_lanes=...,
            min_lane_width_px=...,
            max_lane_width_px=...,
            vehicle_x_ratio=...,
        )

    and:

        assign(
            models,
            frame_width=...,
            frame_height=...,
        )
    """

    def __init__(
        self,
        max_lanes: int = 8,
        min_lane_width_px: float = 80.0,
        max_lane_width_px: float = 900.0,
        vehicle_x_ratio: float = 0.5,
        **kwargs: Any,
    ) -> None:

        self.max_lanes = max(
            2,
            int(max_lanes),
        )

        self.min_lane_width_px = max(
            1.0,
            float(min_lane_width_px),
        )

        self.max_lane_width_px = max(
            self.min_lane_width_px,
            float(max_lane_width_px),
        )

        self.vehicle_x_ratio = float(
            np.clip(
                vehicle_x_ratio,
                0.0,
                1.0,
            )
        )

        # Optional compatibility parameters.
        self.expected_lane_width = _finite_float(
            kwargs.get(
                "expected_lane_width",
            )
        )

        self.lane_width_tolerance = _finite_float(
            kwargs.get(
                "lane_width_tolerance",
            ),
            0.45,
        )

        self.minimum_confidence = _clip01(
            kwargs.get(
                "minimum_confidence",
                0.35,
            )
        )

        self.maximum_lateral_offset_ratio = max(
            0.5,
            float(
                kwargs.get(
                    "maximum_lateral_offset_ratio",
                    1.25,
                )
            ),
        )

        # Temporal state.
        self._previous_lane_id: Optional[int] = None
        self._previous_center_x: Optional[float] = None

    # =========================================================================
    # MODEL X EXTRACTION
    # =========================================================================

    def _model_points(
        self,
        model: Any,
    ) -> list[Any]:
        """
        Extract points from LaneModel.

        Supports several representations used by the project.
        """

        points = _first_attr(
            model,
            (
                "points",
                "lane_points",
                "samples",
                "projected_points",
                "projection_points",
            ),
            None,
        )

        if points is None:
            return []

        try:
            return list(points)
        except TypeError:
            return []

    def _model_x_at_y(
        self,
        model: Any,
        target_y: float,
    ) -> Optional[float]:
        """
        Evaluate a lane model at target_y.

        Supports explicit model evaluation methods and
        point-based models.
        """

        # ------------------------------------------------------------------
        # Explicit evaluation methods.
        # ------------------------------------------------------------------

        for method_name in (
            "x_at_y",
            "evaluate_x",
            "get_x",
            "predict_x",
        ):

            method = getattr(
                model,
                method_name,
                None,
            )

            if callable(method):

                try:
                    value = method(
                        target_y
                    )
                except TypeError:
                    try:
                        value = method(
                            y=target_y
                        )
                    except Exception:
                        continue
                except Exception:
                    continue

                value = _finite_float(value)

                if value is not None:
                    return value

        # ------------------------------------------------------------------
        # Polynomial coefficients.
        # ------------------------------------------------------------------

        coefficients = _first_attr(
            model,
            (
                "coefficients",
                "coeffs",
                "poly_coeffs",
                "polynomial",
            ),
            None,
        )

        if coefficients is not None:

            try:
                coefficients = list(
                    coefficients
                )

                if coefficients:

                    value = np.polyval(
                        np.asarray(
                            coefficients,
                            dtype=float,
                        ),
                        target_y,
                    )

                    value = _finite_float(
                        value
                    )

                    if value is not None:
                        return value

            except (
                TypeError,
                ValueError,
            ):
                pass

        # ------------------------------------------------------------------
        # Point interpolation.
        # ------------------------------------------------------------------

        points = self._model_points(
            model
        )

        xy_points: list[
            tuple[float, float]
        ] = []

        for point in points:

            xy = _extract_xy(
                point
            )

            if xy is not None:
                xy_points.append(
                    xy
                )

        if xy_points:

            xy_points.sort(
                key=lambda item: item[1]
            )

            xs = np.asarray(
                [
                    item[0]
                    for item in xy_points
                ],
                dtype=float,
            )

            ys = np.asarray(
                [
                    item[1]
                    for item in xy_points
                ],
                dtype=float,
            )

            if len(xs) == 1:
                return float(xs[0])

            try:

                value = np.interp(
                    target_y,
                    ys,
                    xs,
                )

                value = _finite_float(
                    value
                )

                if value is not None:
                    return value

            except Exception:
                pass

        # ------------------------------------------------------------------
        # Explicit bottom/reference coordinates.
        # ------------------------------------------------------------------

        for name in (
            "bottom_x",
            "reference_x",
            "near_x",
            "x_at_bottom",
            "center_x",
            "x",
        ):

            value = _finite_float(
                getattr(
                    model,
                    name,
                    None,
                )
            )

            if value is not None:
                return value

        return None

    # =========================================================================
    # MODEL CONFIDENCE
    # =========================================================================

    def _model_confidence(
        self,
        model: Any,
    ) -> float:

        value = _first_attr(
            model,
            (
                "confidence",
                "score",
                "lane_confidence",
                "model_confidence",
            ),
            1.0,
        )

        return _clip01(
            value
        )

    # =========================================================================
    # MODEL SORTING
    # =========================================================================

    def _prepare_models(
        self,
        models: Iterable[Any],
        reference_y: float,
    ) -> list[dict[str, Any]]:

        prepared: list[
            dict[str, Any]
        ] = []

        if models is None:
            return prepared

        try:
            iterator = iter(models)
        except TypeError:
            return prepared

        for model in iterator:

            if model is None:
                continue

            x = self._model_x_at_y(
                model,
                reference_y,
            )

            if x is None:
                continue

            confidence = (
                self._model_confidence(
                    model
                )
            )

            prepared.append(
                {
                    "model": model,
                    "x": float(x),
                    "confidence": confidence,
                }
            )

        # Spatial order: left → right.
        prepared.sort(
            key=lambda item: item["x"]
        )

        # Prevent pathological duplicate lines.
        filtered: list[
            dict[str, Any]
        ] = []

        for item in prepared:

            if not filtered:
                filtered.append(item)
                continue

            previous_x = filtered[-1]["x"]

            if abs(
                item["x"]
                - previous_x
            ) < 1.0:
                # Keep the more confident duplicate.
                if (
                    item["confidence"]
                    > filtered[-1]["confidence"]
                ):
                    filtered[-1] = item

                continue

            filtered.append(item)

        return filtered[
            : self.max_lanes
        ]

    # =========================================================================
    # CORRIDORS
    # =========================================================================

    def _build_corridors(
        self,
        prepared: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:

        corridors: list[
            dict[str, Any]
        ] = []

        if len(prepared) < 2:
            return corridors

        for index in range(
            len(prepared) - 1
        ):

            left = prepared[index]
            right = prepared[
                index + 1
            ]

            left_x = float(
                left["x"]
            )

            right_x = float(
                right["x"]
            )

            width = (
                right_x
                - left_x
            )

            # The two lines must be spatially ordered.
            if width <= 0:
                continue

            # Reject obviously impossible corridors.
            if width < self.min_lane_width_px:
                continue

            if width > self.max_lane_width_px:
                continue

            left_conf = float(
                left["confidence"]
            )

            right_conf = float(
                right["confidence"]
            )

            confidence = min(
                left_conf,
                right_conf,
            )

            center_x = (
                left_x
                + right_x
            ) * 0.5

            corridors.append(
                {
                    "lane_id": index,
                    "left_id": index,
                    "right_id": index + 1,
                    "left_x": left_x,
                    "right_x": right_x,
                    "center_x": center_x,
                    "width": width,
                    "confidence": confidence,
                }
            )

        return corridors

    # =========================================================================
    # CURRENT LANE
    # =========================================================================

    def _select_current_lane(
        self,
        corridors: list[dict[str, Any]],
        vehicle_x: float,
    ) -> Optional[dict[str, Any]]:
        """
        Select the corridor containing the vehicle.

        This is the critical semantic operation.

        With:

            L0 L1 L2 L3

        and vehicle_x between L1 and L2:

            current_lane_id = 1

        NOT the closest boundary.
        """

        if not corridors:
            return None

        containing: list[
            dict[str, Any]
        ] = []

        for corridor in corridors:

            left_x = corridor[
                "left_x"
            ]

            right_x = corridor[
                "right_x"
            ]

            if (
                left_x
                <= vehicle_x
                <= right_x
            ):
                containing.append(
                    corridor
                )

        if containing:

            # Normally exactly one corridor contains
            # the vehicle. If numerical overlap occurs,
            # use the closest center.
            return min(
                containing,
                key=lambda item: abs(
                    item["center_x"]
                    - vehicle_x
                ),
            )

        # ------------------------------------------------------------------
        # Vehicle just outside a boundary.
        #
        # We allow a limited extrapolation so that a temporary
        # detection error does not immediately invalidate assignment.
        # ------------------------------------------------------------------

        nearest = min(
            corridors,
            key=lambda item: min(
                abs(
                    vehicle_x
                    - item["left_x"]
                ),
                abs(
                    vehicle_x
                    - item["right_x"]
                ),
            ),
        )

        half_width = (
            nearest["width"]
            * 0.5
        )

        if half_width <= 0:
            return None

        offset_ratio = abs(
            vehicle_x
            - nearest["center_x"]
        ) / half_width

        if (
            offset_ratio
            <= self.maximum_lateral_offset_ratio
        ):
            return nearest

        return None

    # =========================================================================
    # TEMPORAL CONTINUITY
    # =========================================================================

    def _apply_temporal_continuity(
        self,
        selected: Optional[dict[str, Any]],
        corridors: list[dict[str, Any]],
        vehicle_x: float,
    ) -> Optional[dict[str, Any]]:
        """
        Avoid unnecessary lane-ID jumps.

        A jump of one corridor is allowed because it can represent
        an actual lane change.

        Large jumps require the previous corridor to remain plausible.
        """

        if selected is None:
            return None

        current_id = int(
            selected["lane_id"]
        )

        previous_id = (
            self._previous_lane_id
        )

        if previous_id is None:

            self._previous_lane_id = (
                current_id
            )

            self._previous_center_x = (
                float(
                    selected["center_x"]
                )
            )

            return selected

        if current_id == previous_id:

            self._previous_center_x = (
                float(
                    selected["center_x"]
                )
            )

            return selected

        # Adjacent lane transition is legitimate.
        if abs(
            current_id
            - previous_id
        ) <= 1:

            self._previous_lane_id = (
                current_id
            )

            self._previous_center_x = (
                float(
                    selected["center_x"]
                )
            )

            return selected

        # For a larger jump, verify that the old lane is
        # still physically plausible.
        previous_corridor = None

        for corridor in corridors:

            if (
                int(corridor["lane_id"])
                == previous_id
            ):
                previous_corridor = (
                    corridor
                )
                break

        if previous_corridor is not None:

            if (
                previous_corridor[
                    "left_x"
                ]
                <= vehicle_x
                <= previous_corridor[
                    "right_x"
                ]
            ):
                return previous_corridor

        self._previous_lane_id = (
            current_id
        )

        self._previous_center_x = (
            float(
                selected["center_x"]
            )
        )

        return selected

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def assign(
        self,
        models: Iterable[Any],
        frame_width: float,
        frame_height: float,
    ) -> LaneAssignmentResult:
        """
        Assign semantic lane information.

        Parameters
        ----------
        models:
            LaneModel objects generated by the current pipeline.

        frame_width:
            Current frame width in pixels.

        frame_height:
            Current frame height in pixels.

        Returns
        -------
        LaneAssignmentResult
        """

        width = _finite_float(
            frame_width
        )

        height = _finite_float(
            frame_height
        )

        if width is None or width <= 0:
            return LaneAssignmentResult(
                valid=False,
                reason="Invalid frame width.",
            )

        if height is None or height <= 0:
            return LaneAssignmentResult(
                valid=False,
                reason="Invalid frame height.",
            )

        # ------------------------------------------------------------------
        # Reference point.
        #
        # We evaluate lanes close to the bottom of the image because
        # this is where lateral lane separation is most useful.
        # ------------------------------------------------------------------

        reference_y = (
            height * 0.90
        )

        vehicle_x = (
            width
            * self.vehicle_x_ratio
        )

        prepared = self._prepare_models(
            models,
            reference_y,
        )

        lane_count = len(
            prepared
        )

        if lane_count < 2:

            self._previous_lane_id = None
            self._previous_center_x = None

            return LaneAssignmentResult(
                valid=False,
                vehicle_x=vehicle_x,
                lane_count=lane_count,
                reason=(
                    "At least two lane "
                    "boundaries are required."
                ),
            )

        corridors = (
            self._build_corridors(
                prepared
            )
        )

        if not corridors:

            return LaneAssignmentResult(
                valid=False,
                vehicle_x=vehicle_x,
                lane_count=lane_count,
                reason=(
                    "No valid corridor could "
                    "be constructed."
                ),
            )

        selected = (
            self._select_current_lane(
                corridors,
                vehicle_x,
            )
        )

        if selected is None:

            self._previous_lane_id = None

            return LaneAssignmentResult(
                valid=False,
                vehicle_x=vehicle_x,
                lane_count=lane_count,
                reason=(
                    "Vehicle could not be "
                    "associated with a lane."
                ),
            )

        selected = (
            self._apply_temporal_continuity(
                selected,
                corridors,
                vehicle_x,
            )
        )

        if selected is None:

            return LaneAssignmentResult(
                valid=False,
                vehicle_x=vehicle_x,
                lane_count=lane_count,
                reason=(
                    "Temporal lane association "
                    "failed."
                ),
            )

        # ------------------------------------------------------------------
        # Geometry of selected corridor.
        # ------------------------------------------------------------------

        lane_id = int(
            selected["lane_id"]
        )

        left_id = int(
            selected["left_id"]
        )

        right_id = int(
            selected["right_id"]
        )

        lane_center_x = float(
            selected["center_x"]
        )

        lane_width = float(
            selected["width"]
        )

        lateral_offset = (
            vehicle_x
            - lane_center_x
        )

        half_width = max(
            lane_width * 0.5,
            1.0,
        )

        normalized_offset = float(
            lateral_offset
            / half_width
        )

        normalized_offset = float(
            np.clip(
                normalized_offset,
                -1.0,
                1.0,
            )
        )

        # ------------------------------------------------------------------
        # Confidence.
        #
        # Confidence depends on:
        #   - both boundary confidences;
        #   - whether the vehicle is inside the corridor;
        #   - how far the vehicle is from the center.
        # ------------------------------------------------------------------

        boundary_confidence = float(
            selected["confidence"]
        )

        inside = (
            selected["left_x"]
            <= vehicle_x
            <= selected["right_x"]
        )

        if inside:

            position_confidence = 1.0

        else:

            position_confidence = max(
                0.0,
                1.0
                - (
                    abs(
                        normalized_offset
                    )
                    - 1.0
                ),
            )

        confidence = (
            boundary_confidence
            * position_confidence
        )

        confidence = _clip01(
            confidence
        )

        valid = (
            confidence
            >= self.minimum_confidence
        )

        # ------------------------------------------------------------------
        # Adjacent lanes.
        #
        # lane_id represents a corridor:
        #
        #   0 = L0-L1
        #   1 = L1-L2
        #   2 = L2-L3
        #
        # Therefore:
        #
        #   lanes left  = corridors [0 ... lane_id-1]
        #   lanes right = corridors [lane_id+1 ...]
        # ------------------------------------------------------------------

        left_lane_count = max(
            0,
            lane_id,
        )

        right_lane_count = max(
            0,
            len(corridors)
            - lane_id
            - 1,
        )

        return LaneAssignmentResult(
            valid=valid,

            current_lane_id=lane_id,

            left_boundary_id=left_id,
            right_boundary_id=right_id,

            lane_center_x=lane_center_x,
            vehicle_x=vehicle_x,

            lane_width=lane_width,

            lateral_offset=lateral_offset,
            normalized_offset=normalized_offset,

            confidence=confidence,

            lane_count=lane_count,

            left_lane_count=left_lane_count,
            right_lane_count=right_lane_count,

            reason=(
                None
                if valid
                else (
                    "Assignment confidence "
                    "below threshold."
                )
            ),
        )

    # =========================================================================
    # OPTIONAL COMPATIBILITY ALIAS
    # =========================================================================

    def update(
        self,
        models: Iterable[Any],
        frame_width: float,
        frame_height: float,
    ) -> LaneAssignmentResult:
        """Compatibility alias for assign()."""

        return self.assign(
            models=models,
            frame_width=frame_width,
            frame_height=frame_height,
        )

    def reset(self) -> None:
        """Reset temporal assignment state."""

        self._previous_lane_id = None
        self._previous_center_x = None


# ============================================================================
# ENGINE COMPATIBILITY
# ============================================================================


class LaneAssignmentEngine(LaneAssignment):
    """
    Backward-compatible alias.

    Older code may import LaneAssignmentEngine.
    """

    pass


# ============================================================================
# FACTORY
# ============================================================================


def create_default_lane_assignment(
    config: Optional[Any] = None,
) -> LaneAssignment:
    """
    Create LaneAssignment using project configuration when available.
    """

    if config is None:

        try:
            from config import LANE_ASSIGNMENT

            config = LANE_ASSIGNMENT

        except (
            ImportError,
            AttributeError,
        ):
            config = None

    if config is None:

        return LaneAssignment()

    # Current main.py API.
    max_lanes = getattr(
        config,
        "max_lanes",
        8,
    )

    min_width = getattr(
        config,
        "min_lane_width_px",
        getattr(
            config,
            "minimum_lane_separation",
            80.0,
        ),
    )

    max_width = getattr(
        config,
        "max_lane_width_px",
        getattr(
            config,
            "maximum_lane_separation",
            900.0,
        ),
    )

    vehicle_ratio = getattr(
        config,
        "vehicle_x_ratio",
        0.5,
    )

    return LaneAssignment(
        max_lanes=int(
            max_lanes
        ),

        min_lane_width_px=float(
            min_width
        ),

        max_lane_width_px=float(
            max_width
        ),

        vehicle_x_ratio=float(
            vehicle_ratio
        ),

        expected_lane_width=getattr(
            config,
            "expected_lane_width",
            None,
        ),

        lane_width_tolerance=getattr(
            config,
            "lane_width_tolerance",
            0.45,
        ),

        minimum_confidence=getattr(
            config,
            "minimum_confidence",
            0.35,
        ),

        maximum_lateral_offset_ratio=getattr(
            config,
            "maximum_lateral_offset_ratio",
            1.25,
        ),
    )


# ============================================================================
# FUNCTIONAL API
# ============================================================================


def assign_lanes(
    models: Iterable[Any],
    frame_width: float,
    frame_height: float,
    assignment: Optional[LaneAssignment] = None,
) -> LaneAssignmentResult:
    """
    Convenience wrapper.
    """

    if assignment is None:
        assignment = create_default_lane_assignment()

    return assignment.assign(
        models=models,
        frame_width=frame_width,
        frame_height=frame_height,
    )


__all__ = [
    "LaneAssignment",
    "LaneAssignmentEngine",
    "LaneAssignmentResult",
    "create_default_lane_assignment",
    "assign_lanes",
]