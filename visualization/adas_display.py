"""
Forza Assistents
ADAS HUD Display

Responsabilidade:
    Apenas visualização dos resultados produzidos pelo pipeline ADAS.

Arquitetura:

    YOLOP
        ↓
    LaneDetectionResult
        ↓
    LaneTracker
        ↓
    LaneModel
        ↓
    LaneGeometry
        ↓
    LaneProjection
        ↓
    LaneAssignment
        ↓
    ADASStateEstimator
        ↓
    ADASDisplay
        ↓
      HUD

Este módulo NÃO:
    - executa inferência;
    - calcula geometria;
    - calcula offset;
    - calcula heading;
    - calcula curvatura;
    - faz tracking;
    - decide estado ADAS;
    - controla o volante.

O HUD apenas representa os resultados recebidos.

Compatibilidade:
    - coordenadas de imagem;
    - LaneGeometryResult;
    - ADASStateResult;
    - atualização thread-safe;
    - Tkinter;
    - funcionamento independente do main.py;
    - modo demo independente.
"""

from __future__ import annotations

import math
import threading
import time
import tkinter as tk
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence


# =============================================================================
# TIPOS
# =============================================================================

Point = tuple[float, float]


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================


@dataclass
class ADASDisplayConfig:
    # -------------------------------------------------------------------------
    # Janela
    # -------------------------------------------------------------------------

    width: int = 900
    height: int = 720

    title: str = "FORZA ASSISTENTS — ADAS"

    always_on_top: bool = True
    resizable: bool = True

    min_width: int = 700
    min_height: int = 560

    # -------------------------------------------------------------------------
    # Imagem de origem
    # -------------------------------------------------------------------------
    #
    # As lanes produzidas pelo pipeline estão em coordenadas da imagem.
    #
    # O HUD possui resolução independente.
    #

    source_width: int = 2560
    source_height: int = 1600

    # -------------------------------------------------------------------------
    # Tema
    # -------------------------------------------------------------------------

    background: str = "#080b10"
    panel_background: str = "#0d1219"

    text: str = "#e8eef5"
    text_secondary: str = "#7f8b99"
    text_muted: str = "#4e5966"

    grid: str = "#18212b"

    green: str = "#20e890"
    yellow: str = "#ffd34d"
    red: str = "#ff4057"
    gray: str = "#586472"

    vehicle: str = "#f2f6fa"

    # -------------------------------------------------------------------------
    # Limites visuais
    # -------------------------------------------------------------------------

    warning_threshold: float = 0.55
    critical_threshold: float = 0.82

    # -------------------------------------------------------------------------
    # Suavização visual
    # -------------------------------------------------------------------------

    smoothing_alpha: float = 0.18

    # -------------------------------------------------------------------------
    # Atualização
    # -------------------------------------------------------------------------

    refresh_hz: float = 30.0

    # -------------------------------------------------------------------------
    # Geometria visual
    # -------------------------------------------------------------------------

    horizon_ratio: float = 0.23
    road_bottom_margin: float = 40.0

    lane_width_visual: float = 180.0

    # -------------------------------------------------------------------------
    # Linhas
    # -------------------------------------------------------------------------

    lane_width: int = 6
    lane_dash_width: int = 3

    # -------------------------------------------------------------------------
    # Fonte
    # -------------------------------------------------------------------------

    font_family: str = "Segoe UI"


# =============================================================================
# ESTADO INTERNO
# =============================================================================


@dataclass
class _DisplayState:
    left_lane: Optional[list[Point]] = None
    right_lane: Optional[list[Point]] = None

    lateral_offset: float = 0.0
    heading_error: float = 0.0

    confidence: Optional[float] = None
    curvature: Optional[float] = None

    left_distance: Optional[float] = None
    right_distance: Optional[float] = None

    active: bool = False

    state: str = "LANE LOST"
    warning_side: str = "NONE"

    valid: bool = False

    timestamp: float = field(
        default_factory=time.monotonic
    )


# =============================================================================
# DISPLAY
# =============================================================================


class ADASDisplay:
    """
    HUD visual independente para o sistema ADAS.

    O pipeline pode atualizar o HUD através de:

        display.update_from_pipeline(
            geometry,
            adas_state,
        )

    ou, para integração genérica:

        display.update(...)

    A renderização permanece na thread do Tkinter.
    """

    def __init__(
        self,
        config: Optional[ADASDisplayConfig] = None,
    ) -> None:

        self.config = (
            config
            if config is not None
            else ADASDisplayConfig()
        )

        self._lock = threading.Lock()

        self._state = _DisplayState()

        self._smoothed_offset = 0.0
        self._smoothed_heading = 0.0
        self._smoothed_curvature = 0.0

        self._running = False

        self._thread: Optional[
            threading.Thread
        ] = None

        self._root: Optional[tk.Tk] = None
        self._canvas: Optional[tk.Canvas] = None

        self._closed_event = threading.Event()

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def start(
        self,
        blocking: bool = False,
    ) -> None:
        """
        Inicia o HUD.

        blocking=True:
            executa Tkinter na thread atual.

        blocking=False:
            executa o HUD em thread própria.
        """

        if self._running:
            return

        self._running = True
        self._closed_event.clear()

        if blocking:
            self._run_tk()
            return

        self._thread = threading.Thread(
            target=self._run_tk,
            name="ADASDisplay",
            daemon=True,
        )

        self._thread.start()

    def stop(self) -> None:
        """Solicita encerramento do HUD."""

        self._running = False

        root = self._root

        if root is not None:
            try:
                root.after(
                    0,
                    root.destroy,
                )
            except Exception:
                pass

        self._closed_event.set()

    def wait(self) -> None:
        """Espera o HUD terminar."""

        self._closed_event.wait()

    # =========================================================================
    # PIPELINE API
    # =========================================================================

    def update_from_pipeline(
        self,
        geometry: Any = None,
        adas_state: Any = None,
        active: bool = True,
    ) -> None:
        """
        Atualiza o HUD diretamente a partir do pipeline atual.

        geometry:
            LaneGeometryResult.

        adas_state:
            ADASStateResult.

        Nenhum cálculo ADAS é realizado aqui.
        Apenas os campos já calculados pelo pipeline são consumidos.
        """

        left_lane = None
        right_lane = None
        curvature = None

        if geometry is not None:

            left_lane = getattr(
                geometry,
                "left_lane_screen",
                None,
            )

            right_lane = getattr(
                geometry,
                "right_lane_screen",
                None,
            )

            curvature = getattr(
                geometry,
                "curvature",
                None,
            )

        if adas_state is not None:

            lateral_offset = getattr(
                adas_state,
                "lateral_error",
                0.0,
            )

            heading_error = getattr(
                adas_state,
                "heading_error",
                0.0,
            )

            confidence = getattr(
                adas_state,
                "confidence",
                None,
            )

            left_distance = getattr(
                adas_state,
                "left_distance",
                None,
            )

            right_distance = getattr(
                adas_state,
                "right_distance",
                None,
            )

            valid = bool(
                getattr(
                    adas_state,
                    "valid",
                    False,
                )
            )

            state_obj = getattr(
                adas_state,
                "state",
                None,
            )

            state = self._enum_value(
                state_obj,
                "LANE LOST",
            )

            warning_side_obj = getattr(
                adas_state,
                "warning_side",
                None,
            )

            warning_side = self._enum_value(
                warning_side_obj,
                "NONE",
            )

        else:

            lateral_offset = getattr(
                geometry,
                "lateral_error",
                0.0,
            ) if geometry is not None else 0.0

            heading_error = getattr(
                geometry,
                "heading_error",
                0.0,
            ) if geometry is not None else 0.0

            confidence = getattr(
                geometry,
                "geometry_confidence",
                None,
            ) if geometry is not None else None

            left_distance = None
            right_distance = None

            valid = bool(
                getattr(
                    geometry,
                    "valid",
                    False,
                )
            ) if geometry is not None else False

            state = (
                "LANE LOST"
                if not valid
                else self._infer_state(
                    lateral_offset
                )
            )

            warning_side = "NONE"

        self.update(
            left_lane=left_lane,
            right_lane=right_lane,
            lateral_offset=lateral_offset,
            heading_error=heading_error,
            confidence=confidence,
            curvature=curvature,
            left_distance=left_distance,
            right_distance=right_distance,
            active=active,
            state=state,
            warning_side=warning_side,
            valid=valid,
        )

    # =========================================================================
    # GENERIC UPDATE
    # =========================================================================

    def update(
        self,
        left_lane: Optional[
            Iterable[Sequence[float]]
        ] = None,
        right_lane: Optional[
            Iterable[Sequence[float]]
        ] = None,
        lateral_offset: float = 0.0,
        heading_error: float = 0.0,
        confidence: Optional[float] = None,
        curvature: Optional[float] = None,
        left_distance: Optional[float] = None,
        right_distance: Optional[float] = None,
        active: bool = True,
        state: Optional[str] = None,
        warning_side: Optional[str] = None,
        valid: bool = False,
    ) -> None:
        """
        Atualização genérica do HUD.

        Convenção obrigatória:

            lateral_offset < 0 -> esquerda
            lateral_offset = 0 -> centro
            lateral_offset > 0 -> direita
        """

        left = self._normalize_points(
            left_lane
        )

        right = self._normalize_points(
            right_lane
        )

        offset = self._clip(
            self._safe_float(
                lateral_offset,
                0.0,
            ),
            -1.0,
            1.0,
        )

        heading = self._clip(
            self._safe_float(
                heading_error,
                0.0,
            ),
            -1.0,
            1.0,
        )

        confidence_value = None

        if confidence is not None:
            confidence_value = self._clip(
                self._safe_float(
                    confidence,
                    0.0,
                ),
                0.0,
                1.0,
            )

        curvature_value = None

        if curvature is not None:
            curvature_value = self._safe_float(
                curvature,
                0.0,
            )

        left_distance_value = (
            self._safe_float(
                left_distance,
                0.0,
            )
            if left_distance is not None
            else None
        )

        right_distance_value = (
            self._safe_float(
                right_distance,
                0.0,
            )
            if right_distance is not None
            else None
        )

        with self._lock:

            alpha = self._clip(
                self.config.smoothing_alpha,
                0.0,
                1.0,
            )

            self._smoothed_offset = (
                alpha * offset
                + (1.0 - alpha)
                * self._smoothed_offset
            )

            self._smoothed_heading = (
                alpha * heading
                + (1.0 - alpha)
                * self._smoothed_heading
            )

            if curvature_value is not None:
                self._smoothed_curvature = (
                    alpha * curvature_value
                    + (1.0 - alpha)
                    * self._smoothed_curvature
                )

            resolved_state = (
                str(state).upper()
                if state
                else self._infer_state(
                    self._smoothed_offset
                )
            )

            self._state = _DisplayState(
                left_lane=left,
                right_lane=right,
                lateral_offset=self._smoothed_offset,
                heading_error=self._smoothed_heading,
                confidence=confidence_value,
                curvature=(
                    self._smoothed_curvature
                    if curvature is not None
                    else None
                ),
                left_distance=left_distance_value,
                right_distance=right_distance_value,
                active=bool(active),
                state=resolved_state,
                warning_side=(
                    str(warning_side).upper()
                    if warning_side
                    else "NONE"
                ),
                valid=bool(valid),
                timestamp=time.monotonic(),
            )

    def set_active(
        self,
        active: bool,
    ) -> None:

        with self._lock:
            self._state.active = bool(
                active
            )

    # =========================================================================
    # TKINTER
    # =========================================================================

    def _run_tk(self) -> None:

        try:
            root = tk.Tk()

            self._root = root

            root.title(
                self.config.title
            )

            root.configure(
                bg=self.config.background
            )

            root.geometry(
                f"{self.config.width}"
                f"x"
                f"{self.config.height}"
            )

            root.minsize(
                self.config.min_width,
                self.config.min_height,
            )

            root.resizable(
                self.config.resizable,
                self.config.resizable,
            )

            if self.config.always_on_top:
                root.attributes(
                    "-topmost",
                    True,
                )

            canvas = tk.Canvas(
                root,
                bg=self.config.background,
                highlightthickness=0,
            )

            canvas.pack(
                fill="both",
                expand=True,
            )

            self._canvas = canvas

            root.protocol(
                "WM_DELETE_WINDOW",
                self.stop,
            )

            self._schedule_render()

            root.mainloop()

        except Exception:
            self._running = False
            self._closed_event.set()
            raise

        finally:
            self._root = None
            self._canvas = None

            self._running = False
            self._closed_event.set()

    def _schedule_render(self) -> None:

        if not self._running:
            return

        root = self._root

        if root is None:
            return

        interval = max(
            1,
            int(
                1000.0
                / max(
                    1.0,
                    self.config.refresh_hz,
                )
            ),
        )

        try:
            root.after(
                interval,
                self._render,
            )
        except tk.TclError:
            pass

    # =========================================================================
    # RENDER
    # =========================================================================

    def _render(self) -> None:

        if not self._running:
            return

        canvas = self._canvas

        if canvas is None:
            return

        with self._lock:
            state = self._copy_state(
                self._state
            )

        width = max(
            1,
            canvas.winfo_width(),
        )

        height = max(
            1,
            canvas.winfo_height(),
        )

        canvas.delete("all")

        self._draw_background(
            canvas,
            width,
            height,
        )

        self._draw_header(
            canvas,
            width,
            state,
        )

        self._draw_lane_view(
            canvas,
            width,
            height,
            state,
        )

        self._draw_offset_indicator(
            canvas,
            width,
            height,
            state,
        )

        self._draw_bottom_info(
            canvas,
            width,
            height,
            state,
        )

        self._schedule_render()

    # =========================================================================
    # BACKGROUND
    # =========================================================================

    def _draw_background(
        self,
        canvas: tk.Canvas,
        width: int,
        height: int,
    ) -> None:

        canvas.create_rectangle(
            0,
            0,
            width,
            height,
            fill=self.config.background,
            outline="",
        )

        self._rounded_panel(
            canvas,
            18,
            18,
            width - 18,
            height - 18,
            self.config.panel_background,
            radius=18,
        )

        horizon = int(
            height
            * self.config.horizon_ratio
        )

        for y in range(
            horizon,
            height,
            45,
        ):

            canvas.create_line(
                30,
                y,
                width - 30,
                y,
                fill=self.config.grid,
                width=1,
            )

    # =========================================================================
    # HEADER
    # =========================================================================

    def _draw_header(
        self,
        canvas: tk.Canvas,
        width: int,
        state: _DisplayState,
    ) -> None:

        status_color = (
            self.config.green
            if state.active
            else self.config.gray
        )

        canvas.create_text(
            42,
            42,
            anchor="w",
            text="ADAS",
            fill=self.config.text,
            font=(
                self.config.font_family,
                22,
                "bold",
            ),
        )

        canvas.create_text(
            42,
            69,
            anchor="w",
            text="LANE KEEPING ASSIST",
            fill=self.config.text_secondary,
            font=(
                self.config.font_family,
                9,
                "bold",
            ),
        )

        status_x = width - 45

        canvas.create_oval(
            status_x - 12,
            35,
            status_x,
            47,
            fill=status_color,
            outline="",
        )

        canvas.create_text(
            status_x - 20,
            41,
            anchor="e",
            text=(
                "ACTIVE"
                if state.active
                else "INACTIVE"
            ),
            fill=status_color,
            font=(
                self.config.font_family,
                11,
                "bold",
            ),
        )

    # =========================================================================
    # LANE VIEW
    # =========================================================================

    def _draw_lane_view(
        self,
        canvas: tk.Canvas,
        width: int,
        height: int,
        state: _DisplayState,
    ) -> None:

        top = int(
            height
            * self.config.horizon_ratio
        )

        bottom = (
            height
            - 175
        )

        center_x = width / 2.0

        left = state.left_lane
        right = state.right_lane

        # ---------------------------------------------------------------------
        # Nenhuma lane
        # ---------------------------------------------------------------------

        if not left and not right:

            self._draw_lane_lost(
                canvas,
                width,
                height,
            )

            self._draw_vehicle(
                canvas,
                center_x,
                bottom - 30,
                self.config.red,
            )

            return

        # ---------------------------------------------------------------------
        # Esquerda
        # ---------------------------------------------------------------------

        if left:

            self._draw_lane_curve(
                canvas,
                left,
                width,
                height,
                self._lane_color(
                    state,
                    "left",
                ),
            )

        else:

            self._draw_missing_lane(
                canvas,
                center_x
                - self.config.lane_width_visual,
                top,
                bottom,
                "left",
            )

        # ---------------------------------------------------------------------
        # Direita
        # ---------------------------------------------------------------------

        if right:

            self._draw_lane_curve(
                canvas,
                right,
                width,
                height,
                self._lane_color(
                    state,
                    "right",
                ),
            )

        else:

            self._draw_missing_lane(
                canvas,
                center_x
                + self.config.lane_width_visual,
                top,
                bottom,
                "right",
            )

        # ---------------------------------------------------------------------
        # Centro
        # ---------------------------------------------------------------------

        self._draw_center_reference(
            canvas,
            left,
            right,
            width,
            height,
        )

        # ---------------------------------------------------------------------
        # Veículo
        # ---------------------------------------------------------------------

        vehicle_x = self._vehicle_x(
            center_x,
            state.lateral_offset,
            left,
            right,
            width,
            height,
        )

        vehicle_color = (
            self._status_color(
                state.lateral_offset,
                state.active,
            )
        )

        self._draw_vehicle(
            canvas,
            vehicle_x,
            bottom - 25,
            vehicle_color,
        )

        self._draw_state_badge(
            canvas,
            width,
            top + 28,
            state,
        )

    # =========================================================================
    # LANE CURVE
    # =========================================================================

    def _draw_lane_curve(
        self,
        canvas: tk.Canvas,
        points: list[Point],
        width: int,
        height: int,
        color: str,
    ) -> None:

        if len(points) < 2:
            return

        transformed = [
            self._transform_lane_point(
                point,
                width,
                height,
            )
            for point in points
        ]

        transformed = self._smooth_points(
            transformed
        )

        flat: list[float] = []

        for x, y in transformed:
            flat.extend(
                (x, y)
            )

        canvas.create_line(
            *flat,
            fill=color,
            width=self.config.lane_width,
            smooth=True,
            splinesteps=12,
        )

        canvas.create_line(
            *flat,
            fill=color,
            width=2,
            smooth=True,
            splinesteps=12,
        )

    # =========================================================================
    # MISSING LANE
    # =========================================================================

    def _draw_missing_lane(
        self,
        canvas: tk.Canvas,
        x: float,
        top: float,
        bottom: float,
        side: str,
    ) -> None:

        segments = 10

        total = bottom - top
        segment = total / segments

        for i in range(segments):

            if i % 2 == 0:

                y1 = (
                    top
                    + i * segment
                )

                y2 = (
                    top
                    + (i + 0.65)
                    * segment
                )

                canvas.create_line(
                    x,
                    y1,
                    x,
                    y2,
                    fill=self.config.gray,
                    width=self.config.lane_dash_width,
                )

        if side == "left":

            label = (
                "FAIXA ESQUERDA\n"
                "NÃO DETECTADA"
            )

            anchor = "e"
            text_x = x - 18

        else:

            label = (
                "FAIXA DIREITA\n"
                "NÃO DETECTADA"
            )

            anchor = "w"
            text_x = x + 18

        canvas.create_text(
            text_x,
            top + 45,
            anchor=anchor,
            text=label,
            fill=self.config.yellow,
            font=(
                self.config.font_family,
                8,
                "bold",
            ),
        )

    # =========================================================================
    # CENTER REFERENCE
    # =========================================================================

    def _draw_center_reference(
        self,
        canvas: tk.Canvas,
        left: Optional[list[Point]],
        right: Optional[list[Point]],
        width: int,
        height: int,
    ) -> None:

        if not left or not right:
            return

        count = min(
            len(left),
            len(right),
        )

        if count < 2:
            return

        points: list[Point] = []

        for i in range(count):

            lx, ly = (
                self._transform_lane_point(
                    left[i],
                    width,
                    height,
                )
            )

            rx, ry = (
                self._transform_lane_point(
                    right[i],
                    width,
                    height,
                )
            )

            points.append(
                (
                    (lx + rx) / 2.0,
                    (ly + ry) / 2.0,
                )
            )

        flat: list[float] = []

        for x, y in points:
            flat.extend(
                (x, y)
            )

        canvas.create_line(
            *flat,
            fill=self.config.text_muted,
            width=1,
            dash=(3, 8),
            smooth=True,
            splinesteps=8,
        )

    # =========================================================================
    # VEHICLE
    # =========================================================================

    def _draw_vehicle(
        self,
        canvas: tk.Canvas,
        x: float,
        y: float,
        color: str,
    ) -> None:

        vehicle_width = 30
        vehicle_height = 58

        canvas.create_oval(
            x - 25,
            y + 20,
            x + 25,
            y + 42,
            fill="#030508",
            outline="",
        )

        canvas.create_polygon(
            x - vehicle_width / 2,
            y + vehicle_height / 2,

            x - vehicle_width / 2 + 4,
            y - vehicle_height / 2 + 13,

            x - vehicle_width / 2 + 11,
            y - vehicle_height / 2,

            x + vehicle_width / 2 - 11,
            y - vehicle_height / 2,

            x + vehicle_width / 2 - 4,
            y - vehicle_height / 2 + 13,

            x + vehicle_width / 2,
            y + vehicle_height / 2,

            fill=color,
            outline="",
        )

        canvas.create_polygon(
            x - 9,
            y - 18,
            x + 9,
            y - 18,
            x + 12,
            y - 4,
            x - 12,
            y - 4,
            fill=self.config.background,
            outline="",
        )

        canvas.create_rectangle(
            x - 3,
            y + 8,
            x + 3,
            y + 17,
            fill=self.config.background,
            outline="",
        )

    # =========================================================================
    # OFFSET
    # =========================================================================

    def _draw_offset_indicator(
        self,
        canvas: tk.Canvas,
        width: int,
        height: int,
        state: _DisplayState,
    ) -> None:

        y = height - 115

        left = 90
        right = width - 90

        center = (
            left + right
        ) / 2.0

        span = right - left

        # ---------------------------------------------------------------------
        # Base
        # ---------------------------------------------------------------------

        canvas.create_line(
            left,
            y,
            right,
            y,
            fill=self.config.grid,
            width=8,
        )

        # ---------------------------------------------------------------------
        # Warning zone
        # ---------------------------------------------------------------------

        warning_half = (
            span
            * self.config.warning_threshold
            / 2.0
        )

        canvas.create_line(
            center - warning_half,
            y,
            center + warning_half,
            y,
            fill=self.config.yellow,
            width=8,
        )

        # ---------------------------------------------------------------------
        # Critical zone
        # ---------------------------------------------------------------------

        critical_half = (
            span
            * self.config.critical_threshold
            / 2.0
        )

        canvas.create_line(
            center - critical_half,
            y,
            center + critical_half,
            y,
            fill=self.config.red,
            width=8,
        )

        # ---------------------------------------------------------------------
        # Centro
        # ---------------------------------------------------------------------

        canvas.create_line(
            center - 35,
            y,
            center + 35,
            y,
            fill=self.config.green,
            width=8,
        )

        # ---------------------------------------------------------------------
        # Marcador
        # ---------------------------------------------------------------------

        marker_x = (
            center
            + state.lateral_offset
            * span
            / 2.0
        )

        marker_x = max(
            left,
            min(
                right,
                marker_x,
            ),
        )

        marker_color = (
            self._status_color(
                state.lateral_offset,
                state.active,
            )
        )

        canvas.create_oval(
            marker_x - 9,
            y - 9,
            marker_x + 9,
            y + 9,
            fill=marker_color,
            outline=self.config.text,
            width=2,
        )

        # ---------------------------------------------------------------------
        # Offset
        # ---------------------------------------------------------------------

        offset_percent = (
            state.lateral_offset
            * 100.0
        )

        canvas.create_text(
            center,
            y - 25,
            text=self._format_offset(
                offset_percent
            ),
            fill=self.config.text,
            font=(
                self.config.font_family,
                12,
                "bold",
            ),
        )

        canvas.create_text(
            left,
            y + 24,
            anchor="w",
            text="ESQUERDA",
            fill=self.config.text_muted,
            font=(
                self.config.font_family,
                8,
                "bold",
            ),
        )

        canvas.create_text(
            right,
            y + 24,
            anchor="e",
            text="DIREITA",
            fill=self.config.text_muted,
            font=(
                self.config.font_family,
                8,
                "bold",
            ),
        )

    # =========================================================================
    # STATE BADGE
    # =========================================================================

    def _draw_state_badge(
        self,
        canvas: tk.Canvas,
        width: int,
        y: float,
        state: _DisplayState,
    ) -> None:

        color = (
            self._state_color(
                state
            )
        )

        text = (
            state.state
            .replace("_", " ")
            .upper()
        )

        canvas.create_text(
            width / 2,
            y,
            text=text,
            fill=color,
            font=(
                self.config.font_family,
                13,
                "bold",
            ),
        )

    # =========================================================================
    # BOTTOM INFO
    # =========================================================================

    def _draw_bottom_info(
        self,
        canvas: tk.Canvas,
        width: int,
        height: int,
        state: _DisplayState,
    ) -> None:

        y = height - 52

        left_detected = bool(
            state.left_lane
        )

        right_detected = bool(
            state.right_lane
        )

        self._info_item(
            canvas,
            45,
            y,
            "LEFT",
            "OK"
            if left_detected
            else "LOST",
            self.config.green
            if left_detected
            else self.config.yellow,
        )

        self._info_item(
            canvas,
            180,
            y,
            "RIGHT",
            "OK"
            if right_detected
            else "LOST",
            self.config.green
            if right_detected
            else self.config.yellow,
        )

        confidence_text = "--"

        if state.confidence is not None:

            confidence_text = (
                f"{state.confidence * 100:.0f}%"
            )

        self._info_item(
            canvas,
            width / 2 - 65,
            y,
            "CONF",
            confidence_text,
            self.config.text,
        )

        curvature_text = "--"

        if state.curvature is not None:

            curvature_text = (
                f"{state.curvature:+.3f}"
            )

        self._info_item(
            canvas,
            width - 175,
            y,
            "CURV",
            curvature_text,
            self.config.text,
        )

    def _info_item(
        self,
        canvas: tk.Canvas,
        x: float,
        y: float,
        label: str,
        value: str,
        color: str,
    ) -> None:

        canvas.create_text(
            x,
            y - 10,
            anchor="w",
            text=label,
            fill=self.config.text_muted,
            font=(
                self.config.font_family,
                7,
                "bold",
            ),
        )

        canvas.create_text(
            x,
            y + 8,
            anchor="w",
            text=value,
            fill=color,
            font=(
                self.config.font_family,
                10,
                "bold",
            ),
        )

    # =========================================================================
    # LANE LOST
    # =========================================================================

    def _draw_lane_lost(
        self,
        canvas: tk.Canvas,
        width: int,
        height: int,
    ) -> None:

        center = width / 2.0
        y = height * 0.48

        pulse = (
            math.sin(
                time.monotonic() * 4.0
            )
            + 1.0
        ) / 2.0

        color = (
            self.config.red
            if pulse > 0.5
            else "#8f2534"
        )

        canvas.create_text(
            center,
            y,
            text="!",
            fill=color,
            font=(
                self.config.font_family,
                36,
                "bold",
            ),
        )

        canvas.create_text(
            center,
            y + 48,
            text="SEM DETECÇÃO DE FAIXA",
            fill=color,
            font=(
                self.config.font_family,
                14,
                "bold",
            ),
        )

    # =========================================================================
    # TRANSFORMAÇÃO
    # =========================================================================

    def _transform_lane_point(
        self,
        point: Point,
        width: int,
        height: int,
    ) -> Point:
        """
        Converte coordenadas da imagem para coordenadas do HUD.

        Importante:
            width/height são dimensões do Canvas.

        As coordenadas de entrada usam:
            config.source_width
            config.source_height
        """

        x, y = point

        source_width = max(
            1.0,
            float(
                self.config.source_width
            ),
        )

        source_height = max(
            1.0,
            float(
                self.config.source_height
            ),
        )

        x_norm = (
            x / source_width
        )

        y_norm = (
            y / source_height
        )

        x_norm = max(
            0.0,
            min(
                1.0,
                x_norm,
            ),
        )

        y_norm = max(
            0.0,
            min(
                1.0,
                y_norm,
            ),
        )

        horizon = (
            height
            * self.config.horizon_ratio
        )

        bottom = (
            height
            - self.config.road_bottom_margin
        )

        screen_y = (
            horizon
            + y_norm
            * (bottom - horizon)
        )

        # Pequena correção de perspectiva:
        #
        # próximo do horizonte:
        #     menor separação visual
        #
        # próximo do veículo:
        #     maior separação visual
        #
        center_x = width / 2.0

        perspective = (
            0.72
            + 0.28 * y_norm
        )

        screen_x = (
            center_x
            + (
                x_norm * width
                - center_x
            )
            * perspective
        )

        return (
            screen_x,
            screen_y,
        )

    # =========================================================================
    # VEHICLE POSITION
    # =========================================================================

    def _vehicle_x(
        self,
        center_x: float,
        offset: float,
        left: Optional[list[Point]],
        right: Optional[list[Point]],
        width: int,
        height: int,
    ) -> float:

        if left and right:

            left_x = (
                self._transform_lane_point(
                    left[-1],
                    width,
                    height,
                )[0]
            )

            right_x = (
                self._transform_lane_point(
                    right[-1],
                    width,
                    height,
                )[0]
            )

            lane_center = (
                left_x + right_x
            ) / 2.0

            lane_half = (
                right_x - left_x
            ) / 2.0

            if lane_half > 10.0:

                return (
                    lane_center
                    + offset * lane_half
                )

        return (
            center_x
            + offset
            * self.config.lane_width_visual
            / 2.0
        )

    # =========================================================================
    # COLORS
    # =========================================================================

    def _lane_color(
        self,
        state: _DisplayState,
        side: str,
    ) -> str:

        offset = state.lateral_offset

        if side == "left":

            proximity = max(
                0.0,
                -offset,
            )

        else:

            proximity = max(
                0.0,
                offset,
            )

        if proximity >= (
            self.config.critical_threshold
        ):
            return self.config.red

        if proximity >= (
            self.config.warning_threshold
        ):
            return self.config.yellow

        return self.config.green

    def _status_color(
        self,
        offset: float,
        active: bool,
    ) -> str:

        if not active:
            return self.config.gray

        magnitude = abs(
            offset
        )

        if magnitude >= (
            self.config.critical_threshold
        ):
            return self.config.red

        if magnitude >= (
            self.config.warning_threshold
        ):
            return self.config.yellow

        return self.config.green

    def _state_color(
        self,
        state: _DisplayState,
    ) -> str:

        if not state.active:
            return self.config.gray

        name = (
            state.state.upper()
        )

        if (
            "DEPARTURE"
            in name
        ):
            return self.config.red

        if (
            "WARNING"
            in name
        ):
            return self.config.yellow

        if name == "LANE LOST":
            return self.config.red

        if name == "UNKNOWN":
            return self.config.gray

        return self.config.green

    # =========================================================================
    # FALLBACK STATE
    # =========================================================================

    def _infer_state(
        self,
        offset: float,
    ) -> str:

        if (
            offset
            <= -self.config.critical_threshold
        ):
            return "LEFT_DEPARTURE"

        if (
            offset
            >= self.config.critical_threshold
        ):
            return "RIGHT_DEPARTURE"

        if (
            offset
            <= -self.config.warning_threshold
        ):
            return "LEFT_WARNING"

        if (
            offset
            >= self.config.warning_threshold
        ):
            return "RIGHT_WARNING"

        if offset < -0.20:
            return "SLIGHT_LEFT"

        if offset > 0.20:
            return "SLIGHT_RIGHT"

        return "CENTERED"

    # =========================================================================
    # HELPERS
    # =========================================================================

    @staticmethod
    def _enum_value(
        value: Any,
        default: str,
    ) -> str:

        if value is None:
            return default

        enum_value = getattr(
            value,
            "value",
            None,
        )

        if enum_value is not None:
            return str(
                enum_value
            ).upper()

        return str(
            value
        ).upper()

    @staticmethod
    def _clip(
        value: float,
        minimum: float,
        maximum: float,
    ) -> float:

        return max(
            minimum,
            min(
                maximum,
                value,
            ),
        )

    @staticmethod
    def _safe_float(
        value: Any,
        default: float = 0.0,
    ) -> float:

        try:
            value = float(value)
        except (
            TypeError,
            ValueError,
        ):
            return default

        if not math.isfinite(value):
            return default

        return value

    @staticmethod
    def _normalize_points(
        points: Any,
    ) -> Optional[list[Point]]:

        if points is None:
            return None

        result: list[Point] = []

        try:
            iterator = iter(points)
        except TypeError:
            return None

        for point in iterator:

            try:

                if hasattr(
                    point,
                    "x",
                ) and hasattr(
                    point,
                    "y",
                ):

                    x = float(
                        point.x
                    )

                    y = float(
                        point.y
                    )

                else:

                    if len(point) < 2:
                        continue

                    x = float(
                        point[0]
                    )

                    y = float(
                        point[1]
                    )

                if not (
                    math.isfinite(x)
                    and math.isfinite(y)
                ):
                    continue

                result.append(
                    (
                        x,
                        y,
                    )
                )

            except (
                TypeError,
                ValueError,
                IndexError,
            ):
                continue

        if len(result) < 2:
            return None

        return result

    @staticmethod
    def _smooth_points(
        points: list[Point],
    ) -> list[Point]:

        if len(points) <= 2:
            return points

        result = [
            points[0]
        ]

        for i in range(
            1,
            len(points) - 1,
        ):

            px, py = (
                points[i - 1]
            )

            x, y = (
                points[i]
            )

            nx, ny = (
                points[i + 1]
            )

            result.append(
                (
                    (
                        px
                        + 2.0 * x
                        + nx
                    ) / 4.0,

                    (
                        py
                        + 2.0 * y
                        + ny
                    ) / 4.0,
                )
            )

        result.append(
            points[-1]
        )

        return result

    @staticmethod
    def _format_offset(
        value: float,
    ) -> str:

        if abs(value) < 0.05:
            return "0%"

        sign = (
            "+"
            if value > 0
            else ""
        )

        return (
            f"{sign}{value:.1f}%"
        )

    @staticmethod
    def _copy_state(
        state: _DisplayState,
    ) -> _DisplayState:

        return _DisplayState(
            left_lane=(
                list(state.left_lane)
                if state.left_lane
                else None
            ),
            right_lane=(
                list(state.right_lane)
                if state.right_lane
                else None
            ),
            lateral_offset=(
                state.lateral_offset
            ),
            heading_error=(
                state.heading_error
            ),
            confidence=(
                state.confidence
            ),
            curvature=(
                state.curvature
            ),
            left_distance=(
                state.left_distance
            ),
            right_distance=(
                state.right_distance
            ),
            active=state.active,
            state=state.state,
            warning_side=(
                state.warning_side
            ),
            valid=state.valid,
            timestamp=state.timestamp,
        )

    @staticmethod
    def _rounded_panel(
        canvas: tk.Canvas,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        fill: str,
        radius: float,
    ) -> None:

        canvas.create_rectangle(
            x1 + radius,
            y1,
            x2 - radius,
            y2,
            fill=fill,
            outline="",
        )

        canvas.create_rectangle(
            x1,
            y1 + radius,
            x2,
            y2 - radius,
            fill=fill,
            outline="",
        )

        canvas.create_oval(
            x1,
            y1,
            x1 + 2 * radius,
            y1 + 2 * radius,
            fill=fill,
            outline="",
        )

        canvas.create_oval(
            x2 - 2 * radius,
            y1,
            x2,
            y1 + 2 * radius,
            fill=fill,
            outline="",
        )

        canvas.create_oval(
            x1,
            y2 - 2 * radius,
            x1 + 2 * radius,
            y2,
            fill=fill,
            outline="",
        )

        canvas.create_oval(
            x2 - 2 * radius,
            y2 - 2 * radius,
            x2,
            y2,
            fill=fill,
            outline="",
        )


# =============================================================================
# DEMO
# =============================================================================


def run_demo() -> None:
    """
    Executa o HUD sem depender do Forza ou do pipeline.
    """

    display = ADASDisplay()

    display.start(
        blocking=True
    )


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    "ADASDisplay",
    "ADASDisplayConfig",
    "run_demo",
]


if __name__ == "__main__":
    run_demo()