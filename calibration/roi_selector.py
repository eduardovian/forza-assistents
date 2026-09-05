"""
calibration/roi_selector.py

Seletor interativo de ROI trapezoidal para o Forza Assistents.

A ROI possui quatro pontos independentes:

    P0 ---------------- P1
      \                /
       \              /
        \            /
         P3 -------- P2

P0 = superior esquerdo
P1 = superior direito
P2 = inferior direito
P3 = inferior esquerdo

Controles
---------
Mouse esquerdo:
    - Arrastar P0/P1/P2/P3 -> mover vértice.
    - Arrastar dentro do trapézio -> mover ROI inteira.
    - Arrastar fora -> criar nova ROI.

ENTER / S:
    Confirma.

R / BACKSPACE:
    Reseta.

ESC:
    Cancela.

As coordenadas armazenadas são sempre relativas
ao frame original.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


# ============================================================================
# MODELOS
# ============================================================================


@dataclass
class Point:
    """Ponto 2D em coordenadas do frame original."""

    x: int
    y: int

    def as_tuple(self) -> tuple[int, int]:
        return self.x, self.y


@dataclass
class TrapezoidROI:
    """
    ROI trapezoidal.

    Ordem:

        P0 ---------------- P1
          \                /
           \              /
            \            /
             P3 -------- P2
    """

    p0: Point
    p1: Point
    p2: Point
    p3: Point

    def points(self) -> list[Point]:
        return [
            self.p0,
            self.p1,
            self.p2,
            self.p3,
        ]

    def as_array(self) -> np.ndarray:
        return np.array(
            [
                self.p0.as_tuple(),
                self.p1.as_tuple(),
                self.p2.as_tuple(),
                self.p3.as_tuple(),
            ],
            dtype=np.float32,
        )

    def center(self) -> Point:
        points = self.as_array()

        return Point(
            int(np.mean(points[:, 0])),
            int(np.mean(points[:, 1])),
        )

    def bounding_box(self) -> tuple[int, int, int, int]:
        points = self.as_array()

        x_min = int(np.min(points[:, 0]))
        y_min = int(np.min(points[:, 1]))
        x_max = int(np.max(points[:, 0]))
        y_max = int(np.max(points[:, 1]))

        return (
            x_min,
            y_min,
            x_max - x_min,
            y_max - y_min,
        )

    def area(self) -> float:
        """Área do trapézio em pixels²."""

        return abs(
            cv2.contourArea(
                self.as_array()
            )
        )

    def is_valid(
        self,
        frame_width: int,
        frame_height: int,
        minimum_width: int = 20,
        minimum_height: int = 20,
    ) -> bool:
        """
        Verifica se a ROI é geometricamente válida.

        A ROI precisa:
        - possuir quatro pontos;
        - estar dentro do frame;
        - possuir tamanho mínimo;
        - possuir área mínima;
        - ser convexa;
        - não possuir pontos degenerados.
        """

        points = self.points()

        if len(points) != 4:
            return False

        # ------------------------------------------------------------------
        # Limites do frame
        # ------------------------------------------------------------------

        for point in points:

            if not (
                0 <= point.x < frame_width
                and 0 <= point.y < frame_height
            ):
                return False

        # ------------------------------------------------------------------
        # Bounding box
        # ------------------------------------------------------------------

        _, _, width, height = self.bounding_box()

        if width < minimum_width:
            return False

        if height < minimum_height:
            return False

        # ------------------------------------------------------------------
        # Convexidade
        # ------------------------------------------------------------------

        polygon = self.as_array()

        if not cv2.isContourConvex(
            polygon.astype(np.float32)
        ):
            return False

        # ------------------------------------------------------------------
        # Área
        # ------------------------------------------------------------------

        minimum_area = (
            minimum_width
            * minimum_height
        )

        if self.area() < minimum_area:
            return False

        # ------------------------------------------------------------------
        # Verificação das arestas
        # ------------------------------------------------------------------

        for index in range(4):

            p1 = polygon[index]

            p2 = polygon[
                (index + 1) % 4
            ]

            distance = np.linalg.norm(
                p2 - p1
            )

            if distance < 5:
                return False

        return True


# ============================================================================
# SELETOR
# ============================================================================


class ROISelector:
    """Editor interativo de ROI trapezoidal."""

    WINDOW_NAME = (
        "Forza Assistents - "
        "Trapezoidal ROI Calibration"
    )

    HANDLE_RADIUS = 12

    MIN_ROI_WIDTH = 30
    MIN_ROI_HEIGHT = 30

    def __init__(
        self,
        window_width: int = 1400,
        window_height: int = 900,
        perspective_width: int = 1000,
        perspective_height: int = 600,
    ) -> None:

        self.window_width = window_width
        self.window_height = window_height

        self.perspective_width = (
            perspective_width
        )

        self.perspective_height = (
            perspective_height
        )

        self.frame: Optional[np.ndarray] = None

        self.frame_width = 0
        self.frame_height = 0

        self.roi: Optional[
            TrapezoidROI
        ] = None

        # ------------------------------------------------------------------
        # Conversão frame <-> janela
        # ------------------------------------------------------------------

        self.display_scale = 1.0

        self.display_offset_x = 0
        self.display_offset_y = 0

        # ------------------------------------------------------------------
        # Estado do mouse
        # ------------------------------------------------------------------

        self.dragging = False

        self.drag_mode: Optional[str] = None

        self.active_point: Optional[int] = None

        self.drag_start: Optional[Point] = None

        self.original_roi: Optional[
            TrapezoidROI
        ] = None

    # ========================================================================
    # COORDENADAS
    # ========================================================================

    def _frame_to_display(
        self,
        x: int,
        y: int,
    ) -> tuple[int, int]:

        display_x = int(
            x * self.display_scale
            + self.display_offset_x
        )

        display_y = int(
            y * self.display_scale
            + self.display_offset_y
        )

        return (
            display_x,
            display_y,
        )

    def _display_to_frame(
        self,
        x: int,
        y: int,
    ) -> Point:

        frame_x = int(
            (
                x
                - self.display_offset_x
            )
            / self.display_scale
        )

        frame_y = int(
            (
                y
                - self.display_offset_y
            )
            / self.display_scale
        )

        frame_x = max(
            0,
            min(
                self.frame_width - 1,
                frame_x,
            ),
        )

        frame_y = max(
            0,
            min(
                self.frame_height - 1,
                frame_y,
            ),
        )

        return Point(
            frame_x,
            frame_y,
        )

    # ========================================================================
    # ROI
    # ========================================================================

    def _create_default_roi(
        self,
    ) -> TrapezoidROI:

        width = self.frame_width
        height = self.frame_height

        return TrapezoidROI(

            # Topo
            p0=Point(
                int(width * 0.35),
                int(height * 0.52),
            ),

            p1=Point(
                int(width * 0.65),
                int(height * 0.52),
            ),

            # Baixo
            p2=Point(
                int(width * 0.95),
                int(height * 0.98),
            ),

            p3=Point(
                int(width * 0.05),
                int(height * 0.98),
            ),
        )

    def _reset(self) -> None:

        self.roi = self._create_default_roi()

        self.dragging = False
        self.drag_mode = None
        self.active_point = None

        self.drag_start = None
        self.original_roi = None

    # ========================================================================
    # PONTOS
    # ========================================================================

    def _find_point(
        self,
        x: int,
        y: int,
    ) -> Optional[int]:

        if self.roi is None:
            return None

        for index, point in enumerate(
            self.roi.points()
        ):

            px, py = (
                self._frame_to_display(
                    point.x,
                    point.y,
                )
            )

            distance = np.sqrt(
                (x - px) ** 2
                + (y - py) ** 2
            )

            if (
                distance
                <= self.HANDLE_RADIUS * 1.8
            ):
                return index

        return None

    def _point_inside_roi(
        self,
        x: int,
        y: int,
    ) -> bool:

        if self.roi is None:
            return False

        polygon = (
            self.roi.as_array()
        )

        result = cv2.pointPolygonTest(
            polygon,
            (
                float(x),
                float(y),
            ),
            False,
        )

        return result >= 0

    # ========================================================================
    # MOVIMENTO DE PONTO
    # ========================================================================

    def _move_point(
        self,
        point_index: int,
        point: Point,
    ) -> None:

        if self.roi is None:
            return

        points = self.roi.points()

        point.x = max(
            0,
            min(
                self.frame_width - 1,
                point.x,
            ),
        )

        point.y = max(
            0,
            min(
                self.frame_height - 1,
                point.y,
            ),
        )

        points[point_index] = Point(
            point.x,
            point.y,
        )

        self.roi = TrapezoidROI(
            p0=points[0],
            p1=points[1],
            p2=points[2],
            p3=points[3],
        )

    # ========================================================================
    # MOVIMENTO DA ROI INTEIRA
    # ========================================================================

    def _move_roi(
        self,
        current: Point,
    ) -> None:

        if (
            self.original_roi is None
            or self.drag_start is None
        ):
            return

        dx = (
            current.x
            - self.drag_start.x
        )

        dy = (
            current.y
            - self.drag_start.y
        )

        original_points = (
            self.original_roi.points()
        )

        new_points = [
            Point(
                point.x + dx,
                point.y + dy,
            )
            for point in original_points
        ]

        # ------------------------------------------------------------------
        # Corrige X
        # ------------------------------------------------------------------

        min_x = min(
            point.x
            for point in new_points
        )

        max_x = max(
            point.x
            for point in new_points
        )

        if min_x < 0:

            correction = -min_x

            for point in new_points:
                point.x += correction

        elif max_x >= self.frame_width:

            correction = (
                self.frame_width
                - 1
                - max_x
            )

            for point in new_points:
                point.x += correction

        # ------------------------------------------------------------------
        # Corrige Y
        # ------------------------------------------------------------------

        min_y = min(
            point.y
            for point in new_points
        )

        max_y = max(
            point.y
            for point in new_points
        )

        if min_y < 0:

            correction = -min_y

            for point in new_points:
                point.y += correction

        elif max_y >= self.frame_height:

            correction = (
                self.frame_height
                - 1
                - max_y
            )

            for point in new_points:
                point.y += correction

        self.roi = TrapezoidROI(
            p0=new_points[0],
            p1=new_points[1],
            p2=new_points[2],
            p3=new_points[3],
        )

    # ========================================================================
    # CRIAÇÃO DE ROI POR ARRASTE
    # ========================================================================

    def _create_roi_from_drag(
        self,
        start: Point,
        current: Point,
    ) -> None:

        x1 = min(
            start.x,
            current.x,
        )

        x2 = max(
            start.x,
            current.x,
        )

        y1 = min(
            start.y,
            current.y,
        )

        y2 = max(
            start.y,
            current.y,
        )

        width = x2 - x1
        height = y2 - y1

        if width < self.MIN_ROI_WIDTH:
            return

        if height < self.MIN_ROI_HEIGHT:
            return

        # ------------------------------------------------------------------
        # Cria trapézio.
        #
        # A margem superior determina o quanto
        # a parte distante da pista é estreita.
        # ------------------------------------------------------------------

        top_margin = int(
            width * 0.25
        )

        p0 = Point(
            x1 + top_margin,
            y1,
        )

        p1 = Point(
            x2 - top_margin,
            y1,
        )

        p2 = Point(
            x2,
            y2,
        )

        p3 = Point(
            x1,
            y2,
        )

        self.roi = TrapezoidROI(
            p0=p0,
            p1=p1,
            p2=p2,
            p3=p3,
        )

    # ========================================================================
    # MOUSE
    # ========================================================================

    def _mouse_callback(
        self,
        event: int,
        x: int,
        y: int,
        _flags: int,
        _param,
    ) -> None:

        if self.frame is None:
            return

        point = self._display_to_frame(
            x,
            y,
        )

        # ------------------------------------------------------------------
        # Clique
        # ------------------------------------------------------------------

        if event == cv2.EVENT_LBUTTONDOWN:

            selected_point = (
                self._find_point(x, y)
            )

            # --------------------------------------------------------------
            # Arrastar vértice
            # --------------------------------------------------------------

            if selected_point is not None:

                self.dragging = True

                self.drag_mode = "point"

                self.active_point = (
                    selected_point
                )

                self.drag_start = point

                return

            # --------------------------------------------------------------
            # Arrastar ROI inteira
            # --------------------------------------------------------------

            if self._point_inside_roi(
                point.x,
                point.y,
            ):

                self.dragging = True

                self.drag_mode = "move"

                self.drag_start = point

                if self.roi is not None:

                    self.original_roi = (
                        TrapezoidROI(
                            p0=Point(
                                self.roi.p0.x,
                                self.roi.p0.y,
                            ),
                            p1=Point(
                                self.roi.p1.x,
                                self.roi.p1.y,
                            ),
                            p2=Point(
                                self.roi.p2.x,
                                self.roi.p2.y,
                            ),
                            p3=Point(
                                self.roi.p3.x,
                                self.roi.p3.y,
                            ),
                        )
                    )

                return

            # --------------------------------------------------------------
            # Criar nova ROI
            # --------------------------------------------------------------

            self.dragging = True

            self.drag_mode = "create"

            self.active_point = None

            self.drag_start = point

            self.original_roi = None

            self.roi = TrapezoidROI(
                p0=point,
                p1=point,
                p2=point,
                p3=point,
            )

            return

        # ------------------------------------------------------------------
        # Movimento
        # ------------------------------------------------------------------

        if event == cv2.EVENT_MOUSEMOVE:

            if not self.dragging:
                return

            if self.drag_mode == "point":

                if self.active_point is not None:

                    self._move_point(
                        self.active_point,
                        point,
                    )

            elif self.drag_mode == "move":

                self._move_roi(point)

            elif self.drag_mode == "create":

                if self.drag_start is not None:

                    self._create_roi_from_drag(
                        self.drag_start,
                        point,
                    )

            return

        # ------------------------------------------------------------------
        # Soltou mouse
        # ------------------------------------------------------------------

        if event == cv2.EVENT_LBUTTONUP:

            self.dragging = False

            self.drag_mode = None

            self.active_point = None

            self.drag_start = None

            self.original_roi = None

    # ========================================================================
    # DISPLAY
    # ========================================================================

    def _calculate_display_geometry(
        self,
    ) -> None:

        scale_x = (
            self.window_width
            / self.frame_width
        )

        scale_y = (
            self.window_height
            / self.frame_height
        )

        self.display_scale = min(
            scale_x,
            scale_y,
        )

        display_width = int(
            self.frame_width
            * self.display_scale
        )

        display_height = int(
            self.frame_height
            * self.display_scale
        )

        self.display_offset_x = (
            self.window_width
            - display_width
        ) // 2

        self.display_offset_y = (
            self.window_height
            - display_height
        ) // 2

    def _create_canvas(
        self,
    ) -> np.ndarray:

        if self.frame is None:
            raise RuntimeError(
                "Frame não definido."
            )

        canvas = np.zeros(
            (
                self.window_height,
                self.window_width,
                3,
            ),
            dtype=np.uint8,
        )

        resized = cv2.resize(
            self.frame,
            None,
            fx=self.display_scale,
            fy=self.display_scale,
            interpolation=cv2.INTER_AREA,
        )

        height, width = resized.shape[:2]

        x = self.display_offset_x
        y = self.display_offset_y

        canvas[
            y:y + height,
            x:x + width,
        ] = resized

        return canvas

    # ========================================================================
    # DESENHO DA ROI
    # ========================================================================

    def _draw_roi(
        self,
        canvas: np.ndarray,
    ) -> None:

        if self.roi is None:
            return

        display_points = []

        for point in self.roi.points():

            display_points.append(
                self._frame_to_display(
                    point.x,
                    point.y,
                )
            )

        polygon = np.array(
            display_points,
            dtype=np.int32,
        )

        # ------------------------------------------------------------------
        # Escurece exterior
        # ------------------------------------------------------------------

        mask = np.zeros(
            canvas.shape[:2],
            dtype=np.uint8,
        )

        cv2.fillPoly(
            mask,
            [polygon],
            255,
        )

        darkened = (
            canvas.astype(np.float32)
            * 0.30
        ).astype(np.uint8)

        canvas[:] = np.where(
            mask[:, :, None] == 255,
            canvas,
            darkened,
        )

        # ------------------------------------------------------------------
        # Preenchimento da ROI
        # ------------------------------------------------------------------

        overlay = canvas.copy()

        cv2.fillPoly(
            overlay,
            [polygon],
            (0, 120, 0),
        )

        canvas[:] = cv2.addWeighted(
            overlay,
            0.20,
            canvas,
            0.80,
            0,
        )

        # ------------------------------------------------------------------
        # Borda
        # ------------------------------------------------------------------

        cv2.polylines(
            canvas,
            [polygon],
            True,
            (0, 255, 0),
            4,
            cv2.LINE_AA,
        )

        # ------------------------------------------------------------------
        # Pontos
        # ------------------------------------------------------------------

        labels = [
            "P0",
            "P1",
            "P2",
            "P3",
        ]

        for index, (
            display_point,
            label,
        ) in enumerate(
            zip(
                display_points,
                labels,
            )
        ):

            px, py = display_point

            cv2.circle(
                canvas,
                (px, py),
                self.HANDLE_RADIUS,
                (0, 255, 255),
                -1,
            )

            cv2.circle(
                canvas,
                (px, py),
                self.HANDLE_RADIUS + 2,
                (0, 0, 0),
                2,
            )

            cv2.putText(
                canvas,
                label,
                (
                    px + 15,
                    py - 10,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    # ========================================================================
    # COORDENADAS
    # ========================================================================

    def _draw_coordinates(
        self,
        canvas: np.ndarray,
    ) -> None:

        if self.roi is None:
            return

        lines = []

        for index, point in enumerate(
            self.roi.points()
        ):

            lines.append(
                f"P{index}: "
                f"({point.x}, {point.y})"
            )

        center = self.roi.center()

        lines.append(
            f"Centro: "
            f"({center.x}, {center.y})"
        )

        lines.append(
            f"Area: "
            f"{self.roi.area():.0f}px²"
        )

        x = 20
        y = 35

        width = 360
        height = (
            20
            + len(lines) * 27
            + 15
        )

        cv2.rectangle(
            canvas,
            (10, 10),
            (width, height),
            (0, 0, 0),
            -1,
        )

        for line in lines:

            cv2.putText(
                canvas,
                line,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            y += 27

    # ========================================================================
    # INSTRUÇÕES
    # ========================================================================

    def _draw_instructions(
        self,
        canvas: np.ndarray,
    ) -> None:

        lines = [
            "ARRASTE PONTOS: ajustar vertices",
            "ARRASTE INTERIOR: mover ROI",
            "FORA DA ROI: criar nova",
            "ENTER / S: confirmar",
            "R / BACKSPACE: resetar",
            "ESC: cancelar",
        ]

        background_height = (
            len(lines) * 25
            + 25
        )

        cv2.rectangle(
            canvas,
            (
                10,
                self.window_height
                - background_height
                - 10,
            ),
            (
                500,
                self.window_height - 10,
            ),
            (0, 0, 0),
            -1,
        )

        y = (
            self.window_height
            - background_height
            + 15
        )

        for line in lines:

            cv2.putText(
                canvas,
                line,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            y += 25

    # ========================================================================
    # STATUS
    # ========================================================================

    def _draw_status(
        self,
        canvas: np.ndarray,
    ) -> None:

        if self.frame is None:
            return

        text = (
            f"Frame: "
            f"{self.frame_width}x"
            f"{self.frame_height}"
        )

        cv2.putText(
            canvas,
            text,
            (
                self.window_width - 260,
                self.window_height - 20,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # ========================================================================
    # PERSPECTIVE TRANSFORM
    # ========================================================================

    def perspective_transform(
        self,
        frame: np.ndarray,
        roi: Optional[
            TrapezoidROI
        ] = None,
    ) -> np.ndarray:
        """
        Converte a ROI trapezoidal em um retângulo.

        Isso cria uma visão aproximada de bird's-eye view,
        útil posteriormente para análise geométrica da pista.
        """

        if roi is None:
            roi = self.roi

        if roi is None:
            raise ValueError(
                "Nenhuma ROI foi selecionada."
            )

        source = roi.as_array()

        destination = np.array(
            [
                [0, 0],

                [
                    self.perspective_width - 1,
                    0,
                ],

                [
                    self.perspective_width - 1,
                    self.perspective_height - 1,
                ],

                [
                    0,
                    self.perspective_height - 1,
                ],
            ],
            dtype=np.float32,
        )

        matrix = cv2.getPerspectiveTransform(
            source,
            destination,
        )

        return cv2.warpPerspective(
            frame,
            matrix,
            (
                self.perspective_width,
                self.perspective_height,
            ),
            flags=cv2.INTER_LINEAR,
        )

    # ========================================================================
    # API PÚBLICA
    # ========================================================================

    def select(
        self,
        frame: np.ndarray,
    ) -> Optional[TrapezoidROI]:
        """
        Abre o editor de ROI.

        Returns:
            TrapezoidROI selecionada ou None.
        """

        if frame is None:
            raise ValueError(
                "Frame não pode ser None."
            )

        if frame.ndim != 3:
            raise ValueError(
                "Frame deve possuir três canais."
            )

        if frame.shape[2] != 3:
            raise ValueError(
                "Frame deve possuir três canais."
            )

        self.frame = frame.copy()

        self.frame_height, self.frame_width = (
            self.frame.shape[:2]
        )

        if (
            self.frame_width <= 0
            or self.frame_height <= 0
        ):
            raise ValueError(
                "Dimensões inválidas do frame."
            )

        self._calculate_display_geometry()

        self._reset()

        cv2.namedWindow(
            self.WINDOW_NAME,
            cv2.WINDOW_NORMAL,
        )

        cv2.resizeWindow(
            self.WINDOW_NAME,
            self.window_width,
            self.window_height,
        )

        cv2.setMouseCallback(
            self.WINDOW_NAME,
            self._mouse_callback,
        )

        while True:

            canvas = self._create_canvas()

            self._draw_roi(canvas)

            self._draw_coordinates(
                canvas
            )

            self._draw_instructions(
                canvas
            )

            self._draw_status(
                canvas
            )

            cv2.imshow(
                self.WINDOW_NAME,
                canvas,
            )

            key = (
                cv2.waitKey(16)
                & 0xFF
            )

            # --------------------------------------------------------------
            # ESC
            # --------------------------------------------------------------

            if key == 27:

                cv2.destroyWindow(
                    self.WINDOW_NAME
                )

                return None

            # --------------------------------------------------------------
            # RESET
            # --------------------------------------------------------------

            if key in (
                ord("r"),
                ord("R"),
                8,
            ):

                self._reset()

                continue

            # --------------------------------------------------------------
            # CONFIRMAR
            # --------------------------------------------------------------

            if key in (
                13,
                ord("s"),
                ord("S"),
            ):

                if self.roi is None:
                    continue

                if not self.roi.is_valid(
                    self.frame_width,
                    self.frame_height,
                    self.MIN_ROI_WIDTH,
                    self.MIN_ROI_HEIGHT,
                ):
                    continue

                result = self.roi

                cv2.destroyWindow(
                    self.WINDOW_NAME
                )

                return result