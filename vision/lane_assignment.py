"""
vision/lane_assignment.py

Semantic assignment of lane boundaries into drivable lane corridors.

The perception pipeline produces lane BOUNDARIES:

    L0 | L1 | L2 | L3

The assignment stage converts them into CORRIDORS:

    C0 = L0 <-> L1
    C1 = L1 <-> L2
    C2 = L2 <-> L3

and determines which corridor currently contains the vehicle.

This module does NOT perform:
    - detection;
    - tracking;
    - polynomial fitting;
    - projection;
    - ADAS decisions.

It only performs semantic spatial assignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

import math


# =============================================================================
# RESULT
# =============================================================================


@dataclass(frozen=True)
class LaneAssignmentResult:
    """
    Result of semantic lane assignment.

    Boundary indices are spatial indices after sorting from left to right.

    Example:

        L0 | L1 | L2 | L3

    If the vehicle is between L1 and L2:

        current_lane_id = 1
        left_boundary_id = 1
        right_boundary_id = 2
    """

    valid: bool = False

    current_lane_id: Optional[int] = None

    left_boundary_id: Optional[int] = None
    right_boundary_id: Optional[int] = None

    lane_count: int = 0
    corridor_count: int = 0

    left_lane_count: int = 0
    right_lane_count: int = 0

    vehicle_x: float = 0.0

    lane_center_x: float = 0.0
    lane_width: float = 0.0

    lateral_offset: float = 0.0
    normalized_offset: float = 0.0

    confidence: float = 0.0

    reference_y: float = 0.0

    reason: Optional[str] = None

    # -------------------------------------------------------------------------
    # Compatibility properties
    # -------------------------------------------------------------------------

    @property
    def is_valid(self) -> bool:
        return self.valid

    @property
    def current_lane_index(self) -> int:
        if self.current_lane_id is None:
            return -1

        return int(self.current_lane_id)

    @property
    def left_boundary_index(self) -> int:
        if self.left_boundary_id is None:
            return -1

        return int(self.left_boundary_id)

    @property
    def right_boundary_index(self) -> int:
        if self.right_boundary_id is None:
            return -1

        return int(self.right_boundary_id)


# =============================================================================
# INTERNAL REPRESENTATION
# =============================================================================


@dataclass(frozen=True)
class _Boundary:
    """
    Internal representation of one lane boundary.
    """

    original_index: int

    model: Any

    x: float

    confidence: float

    y_min: float
    y_max: float

    valid: bool = True


@dataclass(frozen=True)
class _Corridor:
    """
    Internal representation of a corridor between two boundaries.
    """

    lane_id: int

    left_index: int
    right_index: int

    left_x: float
    right_x: float

    center_x: float
    width: float

    confidence: float


# =============================================================================
# NUMERIC HELPERS
# =============================================================================


def _finite_float(
    value: Any,
    default: Optional[float] = None,
) -> Optional[float]:
    """
    Convert value to finite float.
    """

    try:
        value = float(value)
    except (TypeError, ValueError):
        return default

    if not math.isfinite(value):
        return default

    return value


def _clip01(value: Any) -> float:
    """
    Clamp confidence to [0, 1].
    """

    value = _finite_float(value, 0.0)

    if value is None:
        return 0.0

    return max(
        0.0,
        min(
            1.0,
            value,
        ),
    )


def _get_attr(
    obj: Any,
    names: Sequence[str],
    default: Any = None,
) -> Any:
    """
    Return first existing non-None attribute.
    """

    if obj is None:
        return default

    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)

            if value is not None:
                return value

    return default


# =============================================================================
# LANE ASSIGNMENT
# =============================================================================


class LaneAssignment:
    """
    Assign LaneModel boundaries into semantic lane corridors.

    Public API expected by main.py:

        LaneAssignment(
            max_lanes=...,
            min_lane_width_px=...,
            max_lane_width_px=...,
            vehicle_x_ratio=...,
        )

    followed by:

        assign(
            models,
            frame_width=...,
            frame_height=...,
        )
    """

    def __init__(
        self,
        max_lanes: int = 16,
        min_lane_width_px: float = 80.0,
        max_lane_width_px: float = 900.0,
        vehicle_x_ratio: float = 0.50,
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

        self.vehicle_x_ratio = max(
            0.0,
            min(
                1.0,
                float(vehicle_x_ratio),
            ),
        )

        # ---------------------------------------------------------------------
        # Optional configuration values.
        # ---------------------------------------------------------------------

        self.expected_lane_width = _finite_float(
            kwargs.get(
                "expected_lane_width",
                None,
            )
        )

        self.lane_width_tolerance = max(
            0.0,
            float(
                kwargs.get(
                    "lane_width_tolerance",
                    0.50,
                )
            ),
        )

        self.minimum_confidence = _clip01(
            kwargs.get(
                "minimum_confidence",
                0.40,
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

        self.enable_multi_lane_assignment = bool(
            kwargs.get(
                "enable_multi_lane_assignment",
                True,
            )
        )

        self.max_left_lanes = max(
            0,
            int(
                kwargs.get(
                    "max_left_lanes",
                    8,
                )
            ),
        )

        self.max_right_lanes = max(
            0,
            int(
                kwargs.get(
                    "max_right_lanes",
                    8,
                )
            ),
        )

        # ---------------------------------------------------------------------
        # Temporal state.
        # ---------------------------------------------------------------------

        self._previous_lane_id: Optional[int] = None
        self._previous_center_x: Optional[float] = None

    # =========================================================================
    # MODEL VALIDATION
    # =========================================================================

    @staticmethod
    def _model_is_valid(
        model: Any,
    ) -> bool:
        """
        Validate a real LaneModel from vision/lane_types.py.

        Expected structure:

            LaneModel
                line
                polynomial
                projection
        """

        if model is None:
            return False

        # Prefer native is_valid().
        method = getattr(
            model,
            "is_valid",
            None,
        )

        if callable(method):
            try:
                if not bool(method()):
                    return False
            except Exception:
                return False

        # LaneModel.line
        line = getattr(
            model,
            "line",
            None,
        )

        if line is None:
            return False

        line_valid = getattr(
            line,
            "valid",
            True,
        )

        if not bool(line_valid):
            return False

        # LaneModel.polynomial
        polynomial = getattr(
            model,
            "polynomial",
            None,
        )

        if polynomial is None:
            return False

        polynomial_valid = getattr(
            polynomial,
            "valid",
            True,
        )

        if not bool(polynomial_valid):
            return False

        return True

    # =========================================================================
    # MODEL CONFIDENCE
    # =========================================================================

    @staticmethod
    def _model_confidence(
        model: Any,
    ) -> float:
        """
        Obtain confidence from LaneModel.

        Priority:

            polynomial.confidence
            line.confidence
            model.confidence
        """

        polynomial = getattr(
            model,
            "polynomial",
            None,
        )

        if polynomial is not None:

            value = _finite_float(
                getattr(
                    polynomial,
                    "confidence",
                    None,
                )
            )

            if value is not None:
                return _clip01(
                    value
                )

        line = getattr(
            model,
            "line",
            None,
        )

        if line is not None:

            value = _finite_float(
                getattr(
                    line,
                    "confidence",
                    None,
                )
            )

            if value is not None:
                return _clip01(
                    value
                )

        value = _finite_float(
            getattr(
                model,
                "confidence",
                None,
            ),
            1.0,
        )

        return _clip01(
            value
        )

    # =========================================================================
    # MODEL Y RANGE
    # =========================================================================

    @staticmethod
    def _model_y_range(
        model: Any,
    ) -> tuple[float, float]:
        """
        Determine the valid vertical range of a LaneModel.

        Priority:

            polynomial.y_min / y_max
            line.points
            projection points
        """

        polynomial = getattr(
            model,
            "polynomial",
            None,
        )

        if polynomial is not None:

            y_min = _finite_float(
                getattr(
                    polynomial,
                    "y_min",
                    None,
                )
            )

            y_max = _finite_float(
                getattr(
                    polynomial,
                    "y_max",
                    None,
                )
            )

            if (
                y_min is not None
                and y_max is not None
                and y_max >= y_min
            ):
                return (
                    y_min,
                    y_max,
                )

        # ---------------------------------------------------------------------
        # LaneLine points.
        # ---------------------------------------------------------------------

        line = getattr(
            model,
            "line",
            None,
        )

        points = []

        if line is not None:
            points = getattr(
                line,
                "points",
                [],
            ) or []

        ys: list[float] = []

        for point in points:

            y = _finite_float(
                getattr(
                    point,
                    "y",
                    None,
                )
            )

            if y is not None:
                ys.append(y)

        if ys:
            return (
                min(ys),
                max(ys),
            )

        # ---------------------------------------------------------------------
        # Projection points.
        # ---------------------------------------------------------------------

        projection = getattr(
            model,
            "projection",
            None,
        )

        if projection is not None:

            projection_points = getattr(
                projection,
                "points",
                [],
            ) or []

            ys = []

            for point in projection_points:

                y = _finite_float(
                    getattr(
                        point,
                        "y",
                        None,
                    )
                )

                if y is not None:
                    ys.append(y)

            if ys:
                return (
                    min(ys),
                    max(ys),
                )

        return (
            0.0,
            float("inf"),
        )

    # =========================================================================
    # POLYNOMIAL EVALUATION
    # =========================================================================

    @staticmethod
    def _evaluate_model(
        model: Any,
        y: float,
    ) -> Optional[float]:
        """
        Evaluate the real LanePolynomial:

            x(y) = a*y^3 + b*y^2 + c*y + d

        through:

            model.polynomial.evaluate(y)
        """

        polynomial = getattr(
            model,
            "polynomial",
            None,
        )

        if polynomial is None:
            return None

        evaluate = getattr(
            polynomial,
            "evaluate",
            None,
        )

        if callable(evaluate):

            try:
                x = evaluate(y)
            except Exception:
                return None

            return _finite_float(
                x
            )

        # ---------------------------------------------------------------------
        # Defensive fallback for compatible polynomial objects.
        # ---------------------------------------------------------------------

        coefficients = getattr(
            polynomial,
            "coefficients",
            None,
        )

        if coefficients is None:

            values = (
                getattr(
                    polynomial,
                    "a",
                    None,
                ),
                getattr(
                    polynomial,
                    "b",
                    None,
                ),
                getattr(
                    polynomial,
                    "c",
                    None,
                ),
                getattr(
                    polynomial,
                    "d",
                    None,
                ),
            )

            if all(
                value is not None
                for value in values
            ):
                a, b, c, d = values

                try:
                    x = (
                        float(a) * y ** 3
                        + float(b) * y ** 2
                        + float(c) * y
                        + float(d)
                    )

                    return _finite_float(
                        x
                    )

                except (
                    TypeError,
                    ValueError,
                ):
                    return None

            return None

        try:

            coefficients = list(
                coefficients
            )

            if not coefficients:
                return None

            result = 0.0

            for coefficient in coefficients:
                result = (
                    result * y
                    + float(coefficient)
                )

            return _finite_float(
                result
            )

        except (
            TypeError,
            ValueError,
        ):
            return None

    # =========================================================================
    # REFERENCE Y
    # =========================================================================

    def _reference_y(
        self,
        models: Sequence[Any],
        frame_height: float,
    ) -> Optional[float]:
        """
        Find a Y position shared by the models.

        We deliberately avoid blindly evaluating at:

            frame_height * 0.90

        because the polynomial may not be valid there.

        Instead, use the common vertical range of the models.
        """

        ranges: list[
            tuple[float, float]
        ] = []

        for model in models:

            y_min, y_max = (
                self._model_y_range(
                    model
                )
            )

            if not math.isfinite(
                y_max
            ):
                y_max = float(
                    frame_height
                )

            if not math.isfinite(
                y_min
            ):
                continue

            if y_max < y_min:
                continue

            ranges.append(
                (
                    y_min,
                    y_max,
                )
            )

        if not ranges:
            return None

        common_min = max(
            item[0]
            for item in ranges
        )

        common_max = min(
            item[1]
            for item in ranges
        )

        if common_max < common_min:
            return None

        # Prefer the lower part of the common observed range.
        #
        # 85% of the common interval gives strong lateral separation
        # without aggressively extrapolating.
        reference_y = (
            common_min
            + (
                common_max
                - common_min
            )
            * 0.85
        )

        # Keep inside image.
        reference_y = max(
            0.0,
            min(
                float(frame_height),
                reference_y,
            ),
        )

        return reference_y

    # =========================================================================
    # PREPARE BOUNDARIES
    # =========================================================================

    def _prepare_boundaries(
        self,
        models: Sequence[Any],
        reference_y: float,
    ) -> list[_Boundary]:
        """
        Evaluate and spatially sort valid lane boundaries.
        """

        boundaries: list[
            _Boundary
        ] = []

        for original_index, model in enumerate(
            models
        ):

            if not self._model_is_valid(
                model
            ):
                continue

            y_min, y_max = (
                self._model_y_range(
                    model
                )
            )

            # Reference point must be inside model range.
            if reference_y < y_min:
                continue

            if reference_y > y_max:
                continue

            x = self._evaluate_model(
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

            boundaries.append(
                _Boundary(
                    original_index=(
                        original_index
                    ),
                    model=model,
                    x=x,
                    confidence=confidence,
                    y_min=y_min,
                    y_max=y_max,
                    valid=True,
                )
            )

        # ---------------------------------------------------------------------
        # Sort spatially from left to right.
        # ---------------------------------------------------------------------

        boundaries.sort(
            key=lambda boundary: boundary.x
        )

        # ---------------------------------------------------------------------
        # Remove essentially duplicate boundaries.
        # ---------------------------------------------------------------------

        result: list[
            _Boundary
        ] = []

        duplicate_distance = max(
            2.0,
            self.min_lane_width_px * 0.05,
        )

        for boundary in boundaries:

            if not result:
                result.append(
                    boundary
                )
                continue

            previous = result[-1]

            if (
                abs(
                    boundary.x
                    - previous.x
                )
                < duplicate_distance
            ):

                # Keep the stronger boundary.
                if (
                    boundary.confidence
                    > previous.confidence
                ):
                    result[-1] = (
                        boundary
                    )

                continue

            result.append(
                boundary
            )

        # max_lanes refers to boundaries in the existing main.py.
        return result[
            : self.max_lanes
        ]

    # =========================================================================
    # BUILD CORRIDORS
    # =========================================================================

    def _build_corridors(
        self,
        boundaries: Sequence[_Boundary],
    ) -> list[_Corridor]:
        """
        Convert adjacent boundaries into corridors.
        """

        corridors: list[
            _Corridor
        ] = []

        if len(boundaries) < 2:
            return corridors

        for index in range(
            len(boundaries) - 1
        ):

            left = boundaries[index]
            right = boundaries[
                index + 1
            ]

            width = (
                right.x
                - left.x
            )

            if width <= 0.0:
                continue

            # Hard physical limits.
            if width < self.min_lane_width_px:
                continue

            if width > self.max_lane_width_px:
                continue

            # -----------------------------------------------------------------
            # Expected width is a confidence criterion, NOT a hard rejection.
            #
            # This is important because perspective, curves and screen
            # geometry can make apparent width differ from expected width.
            # -----------------------------------------------------------------

            width_confidence = 1.0

            if (
                self.expected_lane_width
                is not None
                and self.expected_lane_width > 0.0
            ):

                width_error = abs(
                    width
                    - self.expected_lane_width
                ) / self.expected_lane_width

                tolerance = max(
                    0.01,
                    self.lane_width_tolerance,
                )

                if width_error <= tolerance:

                    width_confidence = max(
                        0.0,
                        1.0
                        - (
                            width_error
                            / tolerance
                        ),
                    )

                else:

                    # Do not automatically reject.
                    #
                    # Give progressively lower confidence.
                    width_confidence = max(
                        0.0,
                        1.0
                        - width_error,
                    )

            boundary_confidence = min(
                left.confidence,
                right.confidence,
            )

            confidence = (
                boundary_confidence
                * width_confidence
            )

            center_x = (
                left.x
                + right.x
            ) * 0.5

            corridors.append(
                _Corridor(
                    lane_id=index,
                    left_index=index,
                    right_index=index + 1,
                    left_x=left.x,
                    right_x=right.x,
                    center_x=center_x,
                    width=width,
                    confidence=confidence,
                )
            )

        return corridors

    # =========================================================================
    # CURRENT CORRIDOR
    # =========================================================================

    def _select_corridor(
        self,
        corridors: Sequence[_Corridor],
        vehicle_x: float,
    ) -> Optional[_Corridor]:
        """
        Select the corridor that actually contains the vehicle.

        This is the central semantic operation.

        Example:

            L0   L1   L2   L3
                 |    |
                 | 🚗 |
                 |    |

        returns:

            corridor 1 = L1-L2
        """

        if not corridors:
            return None

        # ---------------------------------------------------------------------
        # Primary rule: vehicle is physically between the boundaries.
        # ---------------------------------------------------------------------

        containing: list[
            _Corridor
        ] = []

        for corridor in corridors:

            if (
                corridor.left_x
                <= vehicle_x
                <= corridor.right_x
            ):
                containing.append(
                    corridor
                )

        if containing:

            # In a valid ordered boundary set there should normally
            # be exactly one.
            return max(
                containing,
                key=lambda corridor: (
                    corridor.confidence,
                    -abs(
                        corridor.center_x
                        - vehicle_x
                    ),
                ),
            )

        # ---------------------------------------------------------------------
        # Secondary rule: vehicle slightly outside due to imperfect geometry.
        # ---------------------------------------------------------------------

        best: Optional[
            _Corridor
        ] = None

        best_score = float(
            "inf"
        )

        for corridor in corridors:

            half_width = (
                corridor.width
                * 0.5
            )

            if half_width <= 0.0:
                continue

            normalized_distance = (
                abs(
                    vehicle_x
                    - corridor.center_x
                )
                / half_width
            )

            if (
                normalized_distance
                > self.maximum_lateral_offset_ratio
            ):
                continue

            score = (
                normalized_distance
                - corridor.confidence
                * 0.15
            )

            if score < best_score:

                best_score = score
                best = corridor

        return best

    # =========================================================================
    # TEMPORAL STABILITY
    # =========================================================================

    def _stabilize(
        self,
        selected: Optional[_Corridor],
        corridors: Sequence[_Corridor],
        vehicle_x: float,
    ) -> Optional[_Corridor]:
        """
        Preserve lane identity without preventing legitimate lane changes.
        """

        if selected is None:
            return None

        current_id = (
            selected.lane_id
        )

        previous_id = (
            self._previous_lane_id
        )

        if previous_id is None:

            self._previous_lane_id = (
                current_id
            )

            self._previous_center_x = (
                selected.center_x
            )

            return selected

        if current_id == previous_id:

            self._previous_center_x = (
                selected.center_x
            )

            return selected

        # A one-lane transition is completely legitimate.
        if abs(
            current_id
            - previous_id
        ) <= 1:

            self._previous_lane_id = (
                current_id
            )

            self._previous_center_x = (
                selected.center_x
            )

            return selected

        # ---------------------------------------------------------------------
        # Large jump.
        #
        # If the previous corridor is still physically containing the
        # vehicle, prefer it. Otherwise accept the new corridor.
        # ---------------------------------------------------------------------

        previous_corridor = None

        for corridor in corridors:

            if (
                corridor.lane_id
                == previous_id
            ):
                previous_corridor = (
                    corridor
                )
                break

        if previous_corridor is not None:

            if (
                previous_corridor.left_x
                <= vehicle_x
                <= previous_corridor.right_x
            ):

                self._previous_center_x = (
                    previous_corridor.center_x
                )

                return previous_corridor

        self._previous_lane_id = (
            current_id
        )

        self._previous_center_x = (
            selected.center_x
        )

        return selected

    # =========================================================================
    # PUBLIC ASSIGN
    # =========================================================================

    def assign(
        self,
        models: Iterable[Any],
        frame_width: float,
        frame_height: float,
    ) -> LaneAssignmentResult:
        """
        Assign lane boundaries to the vehicle's current corridor.
        """

        width = _finite_float(
            frame_width
        )

        height = _finite_float(
            frame_height
        )

        if (
            width is None
            or width <= 0.0
        ):

            return LaneAssignmentResult(
                valid=False,
                reason="Invalid frame width.",
            )

        if (
            height is None
            or height <= 0.0
        ):

            return LaneAssignmentResult(
                valid=False,
                reason="Invalid frame height.",
            )

        # ---------------------------------------------------------------------
        # Materialize iterable.
        # ---------------------------------------------------------------------

        try:
            model_list = [
                model
                for model in models
                if model is not None
            ]
        except TypeError:

            return LaneAssignmentResult(
                valid=False,
                reason="Models is not iterable.",
            )

        # ---------------------------------------------------------------------
        # Vehicle reference.
        # ---------------------------------------------------------------------

        vehicle_x = (
            width
            * self.vehicle_x_ratio
        )

        # ---------------------------------------------------------------------
        # We need at least two valid LaneModels.
        # ---------------------------------------------------------------------

        valid_models = [
            model
            for model in model_list
            if self._model_is_valid(
                model
            )
        ]

        if len(valid_models) < 2:

            return LaneAssignmentResult(
                valid=False,
                lane_count=len(
                    valid_models
                ),
                vehicle_x=vehicle_x,
                reason=(
                    "At least two valid "
                    "LaneModels are required."
                ),
            )

        # ---------------------------------------------------------------------
        # Determine common vertical reference.
        # ---------------------------------------------------------------------

        reference_y = self._reference_y(
            valid_models,
            height,
        )

        if reference_y is None:

            return LaneAssignmentResult(
                valid=False,
                lane_count=len(
                    valid_models
                ),
                vehicle_x=vehicle_x,
                reason=(
                    "No common vertical "
                    "range between lane models."
                ),
            )

        # ---------------------------------------------------------------------
        # Evaluate and sort boundaries.
        # ---------------------------------------------------------------------

        boundaries = (
            self._prepare_boundaries(
                valid_models,
                reference_y,
            )
        )

        if len(boundaries) < 2:

            return LaneAssignmentResult(
                valid=False,
                lane_count=len(
                    boundaries
                ),
                vehicle_x=vehicle_x,
                reference_y=reference_y,
                reason=(
                    "Fewer than two valid "
                    "lane boundaries at reference Y."
                ),
            )

        # ---------------------------------------------------------------------
        # Build corridors.
        # ---------------------------------------------------------------------

        corridors = (
            self._build_corridors(
                boundaries
            )
        )

        if not corridors:

            return LaneAssignmentResult(
                valid=False,
                lane_count=len(
                    boundaries
                ),
                corridor_count=0,
                vehicle_x=vehicle_x,
                reference_y=reference_y,
                reason=(
                    "No valid lane corridor "
                    "could be constructed."
                ),
            )

        # ---------------------------------------------------------------------
        # Select corridor containing vehicle.
        # ---------------------------------------------------------------------

        selected = (
            self._select_corridor(
                corridors,
                vehicle_x,
            )
        )

        if selected is None:

            return LaneAssignmentResult(
                valid=False,
                lane_count=len(
                    boundaries
                ),
                corridor_count=len(
                    corridors
                ),
                vehicle_x=vehicle_x,
                reference_y=reference_y,
                reason=(
                    "Vehicle is not associated "
                    "with any lane corridor."
                ),
            )

        # ---------------------------------------------------------------------
        # Temporal stabilization.
        # ---------------------------------------------------------------------

        selected = self._stabilize(
            selected,
            corridors,
            vehicle_x,
        )

        if selected is None:

            return LaneAssignmentResult(
                valid=False,
                lane_count=len(
                    boundaries
                ),
                corridor_count=len(
                    corridors
                ),
                vehicle_x=vehicle_x,
                reference_y=reference_y,
                reason=(
                    "Temporal lane "
                    "stabilization failed."
                ),
            )

        # ---------------------------------------------------------------------
        # Lateral geometry.
        # ---------------------------------------------------------------------

        lane_center_x = (
            selected.center_x
        )

        lane_width = (
            selected.width
        )

        lateral_offset = (
            vehicle_x
            - lane_center_x
        )

        half_width = max(
            lane_width * 0.5,
            1.0,
        )

        normalized_offset = (
            lateral_offset
            / half_width
        )

        # Keep semantic output bounded.
        normalized_offset = max(
            -1.0,
            min(
                1.0,
                normalized_offset,
            ),
        )

        # ---------------------------------------------------------------------
        # Confidence.
        # ---------------------------------------------------------------------

        confidence = _clip01(
            selected.confidence
        )

        # Position inside corridor is important,
        # but must not destroy a valid assignment.
        if (
            selected.left_x
            <= vehicle_x
            <= selected.right_x
        ):

            position_confidence = 1.0

        else:

            distance_ratio = abs(
                normalized_offset
            )

            position_confidence = max(
                0.0,
                1.0
                - max(
                    0.0,
                    distance_ratio
                    - 1.0,
                ),
            )

        confidence *= (
            0.75
            + 0.25
            * position_confidence
        )

        confidence = _clip01(
            confidence
        )

        # ---------------------------------------------------------------------
        # Assignment validity.
        #
        # IMPORTANT:
        #
        # We do NOT require confidence to be >= minimum_confidence
        # to construct the semantic assignment.
        #
        # The result remains useful to ADAS, which can decide whether
        # the confidence is sufficient for control.
        #
        # This prevents:
        #
        #     MODELS=2
        #     ASSIGNMENT=INVALID
        #
        # merely because confidence is slightly below threshold.
        # ---------------------------------------------------------------------

        valid = (
            confidence > 0.0
        )

        lane_id = int(
            selected.lane_id
        )

        # ---------------------------------------------------------------------
        # Number of adjacent corridors.
        # ---------------------------------------------------------------------

        left_lane_count = min(
            lane_id,
            self.max_left_lanes,
        )

        right_lane_count = min(
            max(
                0,
                len(corridors)
                - lane_id
                - 1,
            ),
            self.max_right_lanes,
        )

        return LaneAssignmentResult(
            valid=valid,

            current_lane_id=lane_id,

            left_boundary_id=(
                selected.left_index
            ),

            right_boundary_id=(
                selected.right_index
            ),

            lane_count=len(
                boundaries
            ),

            corridor_count=len(
                corridors
            ),

            left_lane_count=(
                left_lane_count
            ),

            right_lane_count=(
                right_lane_count
            ),

            vehicle_x=vehicle_x,

            lane_center_x=(
                lane_center_x
            ),

            lane_width=(
                lane_width
            ),

            lateral_offset=(
                lateral_offset
            ),

            normalized_offset=(
                normalized_offset
            ),

            confidence=(
                confidence
            ),

            reference_y=(
                reference_y
            ),

            reason=None,
        )

    # =========================================================================
    # COMPATIBILITY API
    # =========================================================================

    def update(
        self,
        models: Iterable[Any],
        frame_width: float,
        frame_height: float,
    ) -> LaneAssignmentResult:
        """
        Compatibility alias.
        """

        return self.assign(
            models=models,
            frame_width=frame_width,
            frame_height=frame_height,
        )

    def reset(self) -> None:
        """
        Reset temporal state.
        """

        self._previous_lane_id = None
        self._previous_center_x = None


# =============================================================================
# ENGINE COMPATIBILITY
# =============================================================================


class LaneAssignmentEngine(
    LaneAssignment
):
    """
    Backward-compatible name.
    """

    pass


# =============================================================================
# FACTORY
# =============================================================================


def create_default_lane_assignment(
    config: Optional[Any] = None,
) -> LaneAssignment:
    """
    Create LaneAssignment from project configuration.
    """

    if config is None:

        try:
            from config import (
                LANE_ASSIGNMENT,
            )

            config = LANE_ASSIGNMENT

        except (
            ImportError,
            AttributeError,
        ):

            config = None

    if config is None:

        return LaneAssignment()

    return LaneAssignment(
        max_lanes=max(
            2,
            int(
                getattr(
                    config,
                    "max_left_lanes",
                    8,
                )
            )
            + int(
                getattr(
                    config,
                    "max_right_lanes",
                    8,
                )
            )
            + 1,
        ),

        min_lane_width_px=float(
            getattr(
                config,
                "minimum_lane_separation",
                80.0,
            )
        ),

        max_lane_width_px=float(
            getattr(
                config,
                "maximum_lane_separation",
                900.0,
            )
        ),

        vehicle_x_ratio=float(
            getattr(
                config,
                "center_reference_ratio",
                0.50,
            )
        ),

        expected_lane_width=getattr(
            config,
            "expected_lane_width",
            312.0,
        ),

        lane_width_tolerance=getattr(
            config,
            "lane_width_tolerance",
            0.50,
        ),

        minimum_confidence=getattr(
            config,
            "minimum_confidence",
            0.40,
        ),

        maximum_lateral_offset_ratio=getattr(
            config,
            "maximum_lateral_offset_ratio",
            1.25,
        ),

        enable_multi_lane_assignment=getattr(
            config,
            "enable_multi_lane_assignment",
            True,
        ),

        max_left_lanes=getattr(
            config,
            "max_left_lanes",
            8,
        ),

        max_right_lanes=getattr(
            config,
            "max_right_lanes",
            8,
        ),
    )


# =============================================================================
# FUNCTIONAL API
# =============================================================================


def assign_lanes(
    models: Iterable[Any],
    frame_width: float,
    frame_height: float,
    assignment: Optional[
        LaneAssignment
    ] = None,
) -> LaneAssignmentResult:
    """
    Functional convenience API.
    """

    if assignment is None:
        assignment = (
            create_default_lane_assignment()
        )

    return assignment.assign(
        models=models,
        frame_width=frame_width,
        frame_height=frame_height,
    )


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    "LaneAssignment",
    "LaneAssignmentEngine",
    "LaneAssignmentResult",
    "create_default_lane_assignment",
    "assign_lanes",
]