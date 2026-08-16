"""
visualization/yolop_display.py

Visualização da saída bruta do YOLOPv2.

Esta janela representa SOMENTE a percepção da rede.

Não executa:
    - tracking
    - geometria
    - lane assignment
    - ADAS
    - controle
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import cv2
import numpy as np


LOGGER = logging.getLogger(__name__)

WINDOW_NAME = "Forza Assistents - YOLOPv2"

DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720

LANE_THICKNESS = 3
POINT_RADIUS = 3
OBJECT_THICKNESS = 2

TEXT_SCALE = 0.55
TEXT_THICKNESS = 1

DRIVABLE_ALPHA = 0.22


@dataclass
class YOLOPDisplayStats:

    frames_displayed: int = 0
    start_time: float = 0.0

    @property
    def fps(self) -> float:

        if self.start_time <= 0.0:
            return 0.0

        elapsed = (
            time.monotonic()
            - self.start_time
        )

        if elapsed <= 0.0:
            return 0.0

        return (
            self.frames_displayed
            / elapsed
        )


class YOLOPDisplay:

    def __init__(
        self,
        *,
        enabled: bool = True,
        window_name: str = WINDOW_NAME,
        window_width: int = DEFAULT_WIDTH,
        window_height: int = DEFAULT_HEIGHT,
        wait_ms: int = 1,
        show_lane_points: bool = True,
        show_lane_confidence: bool = True,
        show_objects: bool = True,
        show_drivable_area: bool = True,
        show_info: bool = True,
    ) -> None:

        self.enabled = bool(enabled)

        self.window_name = str(
            window_name
        )

        self.window_width = max(
            640,
            int(window_width),
        )

        self.window_height = max(
            360,
            int(window_height),
        )

        self.wait_ms = max(
            1,
            int(wait_ms),
        )

        self.show_lane_points = (
            bool(show_lane_points)
        )

        self.show_lane_confidence = (
            bool(show_lane_confidence)
        )

        self.show_objects = (
            bool(show_objects)
        )

        self.show_drivable_area = (
            bool(show_drivable_area)
        )

        self.show_info = bool(
            show_info
        )

        self._window_created = False
        self._closed = False

        self._stats = (
            YOLOPDisplayStats()
        )

        self._last_frame: Optional[
            np.ndarray
        ] = None

    # ------------------------------------------------------------------
    # LIFECYCLE
    # ------------------------------------------------------------------

    def start(
        self,
        blocking: bool = False,
    ) -> None:

        del blocking

        if not self.enabled:
            return

        if self._closed:
            return

        if self._window_created:
            return

        cv2.namedWindow(
            self.window_name,
            cv2.WINDOW_NORMAL,
        )

        cv2.resizeWindow(
            self.window_name,
            self.window_width,
            self.window_height,
        )

        self._window_created = True

        if self._stats.start_time <= 0.0:
            self._stats.start_time = (
                time.monotonic()
            )

        LOGGER.info(
            "YOLOPDisplay: READY"
        )

    def stop(self) -> None:

        if not self._window_created:
            self._closed = True
            return

        try:
            cv2.destroyWindow(
                self.window_name
            )
            cv2.waitKey(1)

        except Exception:
            LOGGER.debug(
                "Falha ao fechar YOLOPDisplay.",
                exc_info=True,
            )

        finally:
            self._window_created = False
            self._closed = True

    close = stop

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def _float(
        value: Any,
        default: float = 0.0,
    ) -> float:

        try:
            result = float(value)
        except (
            TypeError,
            ValueError,
        ):
            return default

        if not math.isfinite(result):
            return default

        return result

    @staticmethod
    def _clip01(
        value: Any,
    ) -> float:

        return float(
            np.clip(
                YOLOPDisplay._float(value),
                0.0,
                1.0,
            )
        )

    @staticmethod
    def _point(
        point: Any,
    ) -> Optional[tuple[int, int]]:

        try:

            x = float(
                getattr(point, "x")
            )

            y = float(
                getattr(point, "y")
            )

        except (
            TypeError,
            ValueError,
            AttributeError,
        ):
            return None

        if not (
            math.isfinite(x)
            and math.isfinite(y)
        ):
            return None

        return (
            int(round(x)),
            int(round(y)),
        )

    @staticmethod
    def _lane_points(
        lane: Any,
    ) -> Sequence[Any]:

        points = getattr(
            lane,
            "points",
            None,
        )

        if points is not None:
            return points

        if isinstance(
            lane,
            Sequence,
        ) and not isinstance(
            lane,
            (str, bytes),
        ):
            return lane

        return ()

    # ------------------------------------------------------------------
    # DRIVABLE AREA
    # ------------------------------------------------------------------

    def _draw_drivable(
        self,
        frame: np.ndarray,
        detection: Any,
    ) -> None:

        if not self.show_drivable_area:
            return

        mask = getattr(
            detection,
            "drivable_area_mask",
            None,
        )

        if not isinstance(
            mask,
            np.ndarray,
        ):
            return

        if mask.size == 0:
            return

        height, width = frame.shape[:2]

        try:

            if mask.ndim == 3:

                if mask.shape[2] == 1:
                    mask = mask[:, :, 0]

                else:
                    mask = cv2.cvtColor(
                        mask,
                        cv2.COLOR_BGR2GRAY,
                    )

            if (
                mask.shape[0] != height
                or mask.shape[1] != width
            ):

                mask = cv2.resize(
                    mask,
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )

            mask = mask.astype(
                np.float32,
                copy=False,
            )

            if mask.max() > 1.0:
                mask /= 255.0

            binary = (
                mask >= 0.15
            ).astype(
                np.uint8
            ) * 255

            if not np.any(binary):
                return

            overlay = np.zeros_like(
                frame
            )

            overlay[:, :, 1] = binary

            cv2.addWeighted(
                overlay,
                DRIVABLE_ALPHA,
                frame,
                1.0,
                0.0,
                dst=frame,
            )

        except Exception:

            LOGGER.debug(
                "Falha ao desenhar drivable area.",
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # LANES
    # ------------------------------------------------------------------

    @staticmethod
    def _lane_color(
        index: int,
    ) -> tuple[int, int, int]:

        palette = (
            (255, 120, 0),
            (0, 100, 255),
            (0, 220, 255),
            (255, 255, 255),
            (255, 0, 255),
            (0, 255, 100),
        )

        return palette[
            index % len(palette)
        ]

    def _draw_lanes(
        self,
        frame: np.ndarray,
        detection: Any,
    ) -> None:

        lanes = getattr(
            detection,
            "lanes",
            (),
        )

        if not lanes:
            return

        confidences = getattr(
            detection,
            "lane_confidences",
            (),
        )

        for index, lane in enumerate(
            lanes
        ):

            points = self._lane_points(
                lane
            )

            valid = []

            for point in points:

                xy = self._point(
                    point
                )

                if xy is not None:
                    valid.append(xy)

            if len(valid) < 2:
                continue

            color = self._lane_color(
                index
            )

            for first, second in zip(
                valid[:-1],
                valid[1:],
            ):

                cv2.line(
                    frame,
                    first,
                    second,
                    color,
                    LANE_THICKNESS,
                    cv2.LINE_AA,
                )

            if self.show_lane_points:

                for xy in valid:

                    cv2.circle(
                        frame,
                        xy,
                        POINT_RADIUS,
                        color,
                        -1,
                        cv2.LINE_AA,
                    )

            confidence = 0.0

            if (
                isinstance(
                    confidences,
                    Sequence,
                )
                and index < len(confidences)
            ):

                confidence = self._clip01(
                    confidences[index]
                )

            else:

                confidence = self._clip01(
                    getattr(
                        lane,
                        "confidence",
                        0.0,
                    )
                )

            if self.show_lane_confidence:

                x, y = valid[0]

                self._text(
                    frame,
                    f"LANE {index}  "
                    f"{confidence:.2f}",
                    (x + 8, max(22, y - 8)),
                    color,
                )

    # ------------------------------------------------------------------
    # OBJECTS
    # ------------------------------------------------------------------

    def _draw_objects(
        self,
        frame: np.ndarray,
        detection: Any,
    ) -> None:

        if not self.show_objects:
            return

        objects = getattr(
            detection,
            "objects",
            (),
        )

        for obj in objects:

            try:

                x1 = int(float(obj.x1))
                y1 = int(float(obj.y1))
                x2 = int(float(obj.x2))
                y2 = int(float(obj.y2))

            except (
                TypeError,
                ValueError,
                AttributeError,
            ):
                continue

            confidence = self._clip01(
                getattr(
                    obj,
                    "confidence",
                    0.0,
                )
            )

            name = getattr(
                obj,
                "class_name",
                f"class_{getattr(obj, 'class_id', '?')}",
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 165, 255),
                OBJECT_THICKNESS,
                cv2.LINE_AA,
            )

            self._text(
                frame,
                f"{name} {confidence:.2f}",
                (x1, max(18, y1 - 5)),
                (0, 165, 255),
            )

    # ------------------------------------------------------------------
    # TEXT
    # ------------------------------------------------------------------

    @staticmethod
    def _text(
        frame: np.ndarray,
        text: str,
        position: tuple[int, int],
        color: tuple[int, int, int],
    ) -> None:

        x, y = position

        (
            size,
            baseline,
        ) = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            TEXT_SCALE,
            TEXT_THICKNESS,
        )

        w, h = size

        cv2.rectangle(
            frame,
            (x - 3, y - h - 3),
            (x + w + 3, y + baseline + 3),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            frame,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            TEXT_SCALE,
            color,
            TEXT_THICKNESS,
            cv2.LINE_AA,
        )

    # ------------------------------------------------------------------
    # RENDER
    # ------------------------------------------------------------------

    def render(
        self,
        frame: np.ndarray,
        detection: Any = None,
    ) -> np.ndarray:

        if not isinstance(
            frame,
            np.ndarray,
        ):
            raise TypeError(
                "Frame inválido."
            )

        if frame.ndim != 3:
            raise ValueError(
                "Frame deve ser HxWx3."
            )

        output = frame.copy()

        if detection is not None:

            self._draw_drivable(
                output,
                detection,
            )

            self._draw_lanes(
                output,
                detection,
            )

            self._draw_objects(
                output,
                detection,
            )

        self._text(
            output,
            "YOLOPv2 - RAW PERCEPTION",
            (12, 25),
            (255, 255, 255),
        )

        return output

    # ------------------------------------------------------------------
    # UPDATE
    # ------------------------------------------------------------------

    def update(
        self,
        frame: np.ndarray,
        detection: Any = None,
        fps: float = 0.0,
    ) -> bool:

        del fps

        if not self.enabled:
            return True

        if self._closed:
            return False

        if not self._window_created:
            self.start()

        output = self.render(
            frame,
            detection,
        )

        self._last_frame = output

        self._stats.frames_displayed += 1

        try:

            cv2.imshow(
                self.window_name,
                output,
            )

            key = (
                cv2.waitKey(
                    self.wait_ms
                )
                & 0xFF
            )

        except Exception:

            LOGGER.exception(
                "Falha no YOLOPDisplay."
            )

            return False

        if key in (
            27,
            ord("q"),
            ord("Q"),
        ):

            self.stop()
            return False

        return True

    show = update

    @property
    def is_open(self) -> bool:

        return (
            self.enabled
            and self._window_created
            and not self._closed
        )

    @property
    def fps(self) -> float:
        return self._stats.fps

    @property
    def last_frame(
        self,
    ) -> Optional[np.ndarray]:
        return self._last_frame


def create_yolop_display(
    **kwargs: Any,
) -> YOLOPDisplay:

    return YOLOPDisplay(
        **kwargs
    )


__all__ = [
    "YOLOPDisplay",
    "YOLOPDisplayStats",
    "create_yolop_display",
]