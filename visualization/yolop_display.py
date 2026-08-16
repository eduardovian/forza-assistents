"""
visualization/yolop_display.py

Forza Assistents
================

Visualização da percepção bruta do YOLOPv2.

Responsabilidades
-----------------
- Exibir o frame capturado pelo ScreenCapture.
- Desenhar lanes detectadas pelo YOLOPv2.
- Desenhar drivable area quando disponível.
- Desenhar objetos detectados.
- Mostrar confiança das lanes.
- Mostrar FPS e informações básicas da inferência.
- Permitir ativação/desativação da janela.
- Encerrar a janela de forma segura.

Este módulo NÃO:

- executa inferência;
- altera LaneDetectionResult;
- executa tracking;
- calcula geometria;
- calcula LaneAssignment;
- decide estado ADAS;
- envia comandos para o veículo.

Fluxo:

    ScreenCapture
          ↓
        Frame
          ↓
       YOLOPv2
          ↓
    LaneDetectionResult
          ↓
    YOLOPDisplay
          ↓
       OpenCV
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


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

WINDOW_NAME = "Forza Assistents - YOLOPv2"

DEFAULT_WINDOW_WIDTH = 1280
DEFAULT_WINDOW_HEIGHT = 720

LANE_THICKNESS = 3
LANE_POINT_RADIUS = 3

OBJECT_THICKNESS = 2

TEXT_SCALE = 0.55
TEXT_THICKNESS = 1

OVERLAY_ALPHA = 0.28

DRIVABLE_ALPHA = 0.22

MIN_CONFIDENCE = 0.0


# =============================================================================
# ESTATÍSTICAS
# =============================================================================


@dataclass
class YOLOPDisplayStats:
    """
    Estatísticas da janela de visualização.
    """

    frames_displayed: int = 0

    start_time: float = 0.0

    last_timestamp: float = 0.0

    @property
    def fps(self) -> float:
        if self.start_time <= 0.0:
            return 0.0

        elapsed = time.monotonic() - self.start_time

        if elapsed <= 0.0:
            return 0.0

        return self.frames_displayed / elapsed


# =============================================================================
# DISPLAY
# =============================================================================


class YOLOPDisplay:
    """
    Janela de visualização da percepção YOLOPv2.

    Exemplo:

        display = YOLOPDisplay()

        display.show(
            frame,
            detection_result,
        )

        display.close()
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        window_name: str = WINDOW_NAME,
        window_width: int = DEFAULT_WINDOW_WIDTH,
        window_height: int = DEFAULT_WINDOW_HEIGHT,
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
            320,
            int(window_width),
        )

        self.window_height = max(
            240,
            int(window_height),
        )

        self.wait_ms = max(
            1,
            int(wait_ms),
        )

        self.show_lane_points = bool(
            show_lane_points
        )

        self.show_lane_confidence = bool(
            show_lane_confidence
        )

        self.show_objects = bool(
            show_objects
        )

        self.show_drivable_area = bool(
            show_drivable_area
        )

        self.show_info = bool(
            show_info
        )

        self._window_created = False

        self._closed = False

        self._stats = YOLOPDisplayStats()

        self._last_frame: Optional[
            np.ndarray
        ] = None

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    def start(self) -> None:
        """
        Cria a janela de visualização.
        """

        if not self.enabled:
            return

        if self._closed:
            return

        if self._window_created:
            return

        cv2.namedWindow(
            self.window_name,
            cv2.WINDOW_NORMAL,
            # Não utilizar WINDOW_AUTOSIZE:
            # a janela precisa ser redimensionável.
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

    def close(self) -> None:
        """
        Fecha a janela de maneira segura.
        """

        if not self._window_created:
            self._closed = True
            return

        try:
            cv2.destroyWindow(
                self.window_name
            )

            cv2.waitKey(1)

        except Exception:
            LOGGER.exception(
                "Falha ao fechar "
                "YOLOPDisplay."
            )

        finally:
            self._window_created = False
            self._closed = True

    def stop(self) -> None:
        """
        Alias semântico para close().
        """

        self.close()

    def __enter__(self) -> "YOLOPDisplay":

        self.start()

        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:

        self.close()

    # =========================================================================
    # UTILITÁRIOS
    # =========================================================================

    @staticmethod
    def _safe_float(
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

        value = YOLOPDisplay._safe_float(
            value
        )

        return float(
            np.clip(
                value,
                0.0,
                1.0,
            )
        )

    @staticmethod
    def _point_xy(
        point: Any,
    ) -> Optional[tuple[int, int]]:

        if point is None:
            return None

        try:
            x = float(
                getattr(
                    point,
                    "x",
                )
            )

            y = float(
                getattr(
                    point,
                    "y",
                )
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

        if lane is None:
            return ()

        if hasattr(
            lane,
            "points",
        ):
            points = getattr(
                lane,
                "points",
                None,
            )

            if points is None:
                return ()

            return points

        if isinstance(
            lane,
            Sequence,
        ) and not isinstance(
            lane,
            (
                str,
                bytes,
            ),
        ):
            return lane

        return ()

    # =========================================================================
    # FRAME
    # =========================================================================

    @staticmethod
    def _validate_frame(
        frame: np.ndarray,
    ) -> None:

        if not isinstance(
            frame,
            np.ndarray,
        ):
            raise TypeError(
                "YOLOPDisplay recebeu um "
                "frame que não é numpy.ndarray."
            )

        if frame.ndim != 3:
            raise ValueError(
                "Frame deve possuir formato "
                "(height, width, channels)."
            )

        if frame.shape[2] != 3:
            raise ValueError(
                "Frame deve possuir 3 canais."
            )

    @staticmethod
    def _prepare_frame(
        frame: np.ndarray,
    ) -> np.ndarray:

        YOLOPDisplay._validate_frame(
            frame
        )

        if frame.dtype != np.uint8:

            frame = np.clip(
                frame,
                0,
                255,
            ).astype(
                np.uint8
            )

        return frame.copy()

    # =========================================================================
    # DRIVABLE AREA
    # =========================================================================

    def _draw_drivable_area(
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

        if mask is None:
            return

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
                    (
                        width,
                        height,
                    ),
                    interpolation=cv2.INTER_NEAREST,
                )

            mask = mask.astype(
                np.float32
            )

            # Normalização para 0..1.
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

            # Verde para área dirigível.
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
                "Não foi possível desenhar "
                "drivable area.",
                exc_info=True,
            )

    # =========================================================================
    # LANES
    # =========================================================================

    def _lane_color(
        self,
        index: int,
        lane_count: int,
    ) -> tuple[int, int, int]:

        # Cores BGR.
        #
        # Primeira lane:
        #   azul
        #
        # Segunda:
        #   vermelho
        #
        # Demais:
        #   amarelo/branco alternado.

        if lane_count == 2:

            if index == 0:
                return (
                    255,
                    120,
                    0,
                )

            return (
                0,
                100,
                255,
            )

        palette = (
            (
                255,
                120,
                0,
            ),
            (
                0,
                100,
                255,
            ),
            (
                0,
                220,
                255,
            ),
            (
                255,
                255,
                255,
            ),
            (
                255,
                0,
                255,
            ),
            (
                0,
                255,
                100,
            ),
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
            None,
        )

        if lanes is None:
            return

        if not isinstance(
            lanes,
            Sequence,
        ):
            return

        lane_confidences = getattr(
            detection,
            "lane_confidences",
            [],
        )

        lane_count = len(
            lanes
        )

        for index, lane in enumerate(
            lanes
        ):

            points = self._lane_points(
                lane
            )

            if not points:
                continue

            valid_points = []

            for point in points:

                xy = self._point_xy(
                    point
                )

                if xy is None:
                    continue

                valid_points.append(
                    (
                        point,
                        xy,
                    )
                )

            if len(
                valid_points
            ) < 2:

                continue

            confidence = 0.0

            if (
                isinstance(
                    lane_confidences,
                    Sequence,
                )
                and index
                < len(lane_confidences)
            ):

                confidence = (
                    self._clip01(
                        lane_confidences[
                            index
                        ]
                    )
                )

            elif hasattr(
                lane,
                "confidence",
            ):

                confidence = (
                    self._clip01(
                        getattr(
                            lane,
                            "confidence",
                            0.0,
                        )
                    )
                )

            color = self._lane_color(
                index,
                lane_count,
            )

            xy_points = [
                xy
                for _point, xy
                in valid_points
            ]

            # -----------------------------------------------------------------
            # Linha principal
            # -----------------------------------------------------------------

            for first, second in zip(
                xy_points[:-1],
                xy_points[1:],
            ):

                cv2.line(
                    frame,
                    first,
                    second,
                    color,
                    LANE_THICKNESS,
                    cv2.LINE_AA,
                )

            # -----------------------------------------------------------------
            # Pontos
            # -----------------------------------------------------------------

            if self.show_lane_points:

                for point, xy in valid_points:

                    point_confidence = (
                        self._safe_float(
                            getattr(
                                point,
                                "confidence",
                                confidence,
                            ),
                            confidence,
                        )
                    )

                    point_confidence = (
                        self._clip01(
                            point_confidence
                        )
                    )

                    radius = (
                        LANE_POINT_RADIUS
                    )

                    cv2.circle(
                        frame,
                        xy,
                        radius,
                        color,
                        -1,
                        cv2.LINE_AA,
                    )

            # -----------------------------------------------------------------
            # Confidence
            # -----------------------------------------------------------------

            if (
                self.show_lane_confidence
                and xy_points
            ):

                label_x, label_y = (
                    xy_points[0]
                )

                text = (
                    f"Lane {index}"
                    f"  {confidence:.2f}"
                )

                self._draw_text(
                    frame,
                    text,
                    (
                        label_x + 8,
                        max(
                            20,
                            label_y - 8,
                        ),
                    ),
                    color=color,
                )

    # =========================================================================
    # OBJECTS
    # =========================================================================

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
            None,
        )

        if not objects:
            return

        height, width = frame.shape[:2]

        for obj in objects:

            try:

                x1 = int(
                    round(
                        float(
                            obj.x1
                        )
                    )
                )

                y1 = int(
                    round(
                        float(
                            obj.y1
                        )
                    )
                )

                x2 = int(
                    round(
                        float(
                            obj.x2
                        )
                    )
                )

                y2 = int(
                    round(
                        float(
                            obj.y2
                        )
                    )
                )

            except (
                TypeError,
                ValueError,
                AttributeError,
            ):
                continue

            x1 = max(
                0,
                min(
                    width - 1,
                    x1,
                ),
            )

            x2 = max(
                0,
                min(
                    width - 1,
                    x2,
                ),
            )

            y1 = max(
                0,
                min(
                    height - 1,
                    y1,
                ),
            )

            y2 = max(
                0,
                min(
                    height - 1,
                    y2,
                ),
            )

            if x2 <= x1 or y2 <= y1:
                continue

            confidence = self._clip01(
                getattr(
                    obj,
                    "confidence",
                    0.0,
                )
            )

            class_name = getattr(
                obj,
                "class_name",
                f"class_{getattr(obj, 'class_id', '?')}",
            )

            color = (
                0,
                165,
                255,
            )

            cv2.rectangle(
                frame,
                (
                    x1,
                    y1,
                ),
                (
                    x2,
                    y2,
                ),
                color,
                OBJECT_THICKNESS,
                cv2.LINE_AA,
            )

            label = (
                f"{class_name} "
                f"{confidence:.2f}"
            )

            self._draw_text(
                frame,
                label,
                (
                    x1,
                    max(
                        18,
                        y1 - 6,
                    ),
                ),
                color=color,
            )

    # =========================================================================
    # INFO
    # =========================================================================

    def _draw_info(
        self,
        frame: np.ndarray,
        detection: Any,
    ) -> None:

        if not self.show_info:
            return

        height, width = frame.shape[:2]

        detector_fps = 0.0

        metadata = getattr(
            detection,
            "metadata",
            None,
        )

        if isinstance(
            metadata,
            dict,
        ):

            for key in (
                "fps",
                "inference_fps",
                "detector_fps",
            ):

                if key in metadata:

                    detector_fps = (
                        self._safe_float(
                            metadata[key]
                        )
                    )

                    break

        num_lanes = getattr(
            detection,
            "num_lanes_detected",
            0,
        )

        valid = bool(
            getattr(
                detection,
                "valid",
                False,
            )
        )

        vehicle_count = getattr(
            detection,
            "vehicle_count",
            None,
        )

        if vehicle_count is None:

            objects = getattr(
                detection,
                "objects",
                [],
            )

            vehicle_count = len(
                objects
            ) if objects else 0

        lines = [
            "YOLOPv2",
            f"Lanes: {num_lanes}",
            f"Vehicles: {vehicle_count}",
            f"Valid: {'YES' if valid else 'NO'}",
            f"Display FPS: {self._stats.fps:.1f}",
        ]

        if detector_fps > 0.0:

            lines.append(
                f"Inference FPS: "
                f"{detector_fps:.1f}"
            )

        y = 25

        for line in lines:

            self._draw_text(
                frame,
                line,
                (
                    12,
                    y,
                ),
                color=(
                    255,
                    255,
                    255,
                ),
            )

            y += 21

        # ---------------------------------------------------------------------
        # Resolução
        # ---------------------------------------------------------------------

        resolution = (
            f"{width}x{height}"
        )

        self._draw_text(
            frame,
            resolution,
            (
                width - 95,
                22,
            ),
            color=(
                255,
                255,
                255,
            ),
        )

    @staticmethod
    def _draw_text(
        frame: np.ndarray,
        text: str,
        position: tuple[int, int],
        *,
        color: tuple[int, int, int],
    ) -> None:

        x, y = position

        # Fundo discreto para melhorar leitura.
        (
            text_width,
            text_height,
        ), baseline = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            TEXT_SCALE,
            TEXT_THICKNESS,
        )

        cv2.rectangle(
            frame,
            (
                x - 3,
                y - text_height - 3,
            ),
            (
                x + text_width + 3,
                y + baseline + 3,
            ),
            (
                0,
                0,
                0,
            ),
            -1,
        )

        cv2.putText(
            frame,
            text,
            (
                x,
                y,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            TEXT_SCALE,
            color,
            TEXT_THICKNESS,
            cv2.LINE_AA,
        )

    # =========================================================================
    # RENDER
    # =========================================================================

    def render(
        self,
        frame: np.ndarray,
        detection: Any = None,
    ) -> np.ndarray:
        """
        Renderiza a visualização sem abrir a janela.

        Útil para testes e para integração com outros displays.
        """

        output = self._prepare_frame(
            frame
        )

        if detection is not None:

            self._draw_drivable_area(
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

            self._draw_info(
                output,
                detection,
            )

        return output

    def show(
        self,
        frame: np.ndarray,
        detection: Any = None,
    ) -> bool:
        """
        Renderiza e mostra o frame.

        Returns
        -------

        bool
            False quando o usuário solicita encerramento.

        Teclas:

            ESC / Q
                encerra a visualização.

        """

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

        self._stats.last_timestamp = (
            time.monotonic()
        )

        try:

            cv2.imshow(
                self.window_name,
                output,
            )

            key = cv2.waitKey(
                self.wait_ms
            ) & 0xFF

        except Exception:

            LOGGER.exception(
                "Falha ao atualizar "
                "YOLOPDisplay."
            )

            return False

        if key in (
            27,      # ESC
            ord("q"),
            ord("Q"),
        ):

            self.close()

            return False

        return True

    # =========================================================================
    # PROPERTIES
    # =========================================================================

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

    @property
    def stats(
        self,
    ) -> YOLOPDisplayStats:

        return self._stats


# =============================================================================
# FACTORY
# =============================================================================


def create_yolop_display(
    **kwargs: Any,
) -> YOLOPDisplay:

    return YOLOPDisplay(
        **kwargs
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "YOLOPDisplay",
    "YOLOPDisplayStats",
    "create_yolop_display",
]