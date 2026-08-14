"""
Forza Assistents
ADAS HUD Display

Painel visual independente para ADAS / Lane Keeping.

Características:
- HUD automotivo em Tkinter.
- Vista em perspectiva das faixas.
- Suporte a linhas retas e curvas.
- Suporte a apenas uma faixa detectada.
- Estado sem detecção.
- Indicador de posição lateral.
- Offset numérico.
- Confiança.
- Curvatura.
- Status ADAS.
- Semáforo verde/amarelo/vermelho.
- Suavização temporal.
- Atualização thread-safe.
- Não depende do main.py.
"""

from __future__ import annotations

import math
import threading
import time
import tkinter as tk
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence


# =============================================================================
# TIPOS
# =============================================================================

Point = tuple[float, float]


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
    # Thresholds
    # -------------------------------------------------------------------------

    # Offset normalizado.
    #
    # -1 = extrema esquerda
    #  0 = centro
    # +1 = extrema direita

    warning_threshold: float = 0.55
    critical_threshold: float = 0.82

    # -------------------------------------------------------------------------
    # Suavização
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


@dataclass
class _DisplayState:
    left_lane: Optional[list[Point]] = None
    right_lane: Optional[list[Point]] = None

    lateral_offset: float = 0.0

    confidence: Optional[float] = None
    curvature: Optional[float] = None

    active: bool = False

    state: str = "LANE LOST"

    timestamp: float = field(
        default_factory=time.monotonic
    )


# =============================================================================
# DISPLAY
# =============================================================================


class ADASDisplay:
    """
    HUD visual para o sistema ADAS.

    O painel pode ser atualizado por qualquer thread:

        display.update(...)

    A renderização permanece na thread do Tkinter.

    ---------------------------------------------------------------------------
    Exemplo
    ---------------------------------------------------------------------------

        display = ADASDisplay()

        display.start()

        display.update(
            left_lane=[...],
            right_lane=[...],
            lateral_offset=0.12,
            confidence=0.94,
            curvature=0.02,
            active=True,
            state="CENTERED",
        )
    """

    def __init__(
        self,
        config: Optional[ADASDisplayConfig] = None,
    ):
        self.config = (
            config
            if config is not None
            else ADASDisplayConfig()
        )

        self._lock = threading.Lock()

        self._state = _DisplayState()

        self._smoothed_offset = 0.0

        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._root: Optional[tk.Tk] = None
        self._canvas: Optional[tk.Canvas] = None

        self._closed_event = threading.Event()

        self._last_render = 0.0

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def start(
        self,
        blocking: bool = False,
    ) -> None:
        """
        Inicia o painel.

        blocking=False:
            recomendado para integração com o sistema principal.

        blocking=True:
            útil para testes independentes.
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
        """Fecha o painel."""

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
        """Espera o painel terminar."""

        self._closed_event.wait()

    def update(
        self,
        left_lane: Optional[
            Iterable[Sequence[float]]
        ] = None,
        right_lane: Optional[
            Iterable[Sequence[float]]
        ] = None,
        lateral_offset: float = 0.0,
        confidence: Optional[float] = None,
        curvature: Optional[float] = None,
        active: bool = True,
        state: Optional[str] = None,
    ) -> None:
        """
        Atualiza os dados do painel.

        left_lane/right_lane:

            lista de pontos:

                [(x1, y1), (x2, y2), ...]

        lateral_offset:

            preferencialmente normalizado:

                -1 = esquerda
                 0 = centro
                +1 = direita

            Também pode ser usado como percentual/valor normalizado.

        confidence:

            0.0 até 1.0.

        curvature:

            valor opcional da curvatura.

        state:

            Ex.:

                CENTERED
                SLIGHT_LEFT
                SLIGHT_RIGHT
                LEFT_WARNING
                RIGHT_WARNING
                LEFT_DEPARTURE
                RIGHT_DEPARTURE
                LANE_LOST
        """

        left = self._normalize_points(left_lane)
        right = self._normalize_points(right_lane)

        offset = self._safe_float(
            lateral_offset,
            0.0,
        )

        offset = max(
            -1.0,
            min(1.0, offset),
        )

        confidence_value = None

        if confidence is not None:
            confidence_value = max(
                0.0,
                min(
                    1.0,
                    self._safe_float(
                        confidence,
                        0.0,
                    ),
                ),
            )

        curvature_value = None

        if curvature is not None:
            curvature_value = self._safe_float(
                curvature,
                0.0,
            )

        with self._lock:
            self._state = _DisplayState(
                left_lane=left,
                right_lane=right,
                lateral_offset=offset,
                confidence=confidence_value,
                curvature=curvature_value,
                active=bool(active),
                state=(
                    str(state).upper()
                    if state
                    else self._infer_state(
                        offset,
                        left,
                        right,
                        active,
                    )
                ),
                timestamp=time.monotonic(),
            )

    def set_active(
        self,
        active: bool,
    ) -> None:
        with self._lock:
            self._state.active = bool(active)

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
                700,
                560,
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
            self._closed_event.set()
            self._running = False
            raise

        finally:
            self._root = None
            self._canvas = None
            self._closed_event.set()
            self._running = False

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

        root.after(
            interval,
            self._render,
        )

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
            state = self._state

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
            height,
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

        # Painel principal.
        canvas.create_round_rect = (
            getattr(
                canvas,
                "create_round_rect",
                None,
            )
        )

        # Tkinter padrão não possui create_round_rect.
        # Usamos polygon + rectangles para compatibilidade.
        self._rounded_panel(
            canvas,
            18,
            18,
            width - 18,
            height - 18,
            self.config.panel_background,
            radius=18,
        )

        # Grade discreta.
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
        height: int,
        state: _DisplayState,
    ) -> None:

        active_color = (
            self.config.green
            if state.active
            else self.config.red
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
            fill=active_color,
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
            fill=active_color,
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
        # Sem detecção
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
        # Faixa esquerda
        # ---------------------------------------------------------------------

        if left:
            self._draw_lane_curve(
                canvas,
                left,
                width,
                height,
                self._lane_color(
                    state,
                    side="left",
                ),
                solid=True,
            )
        else:
            self._draw_missing_lane(
                canvas,
                center_x
                - self.config.lane_width_visual,
                top,
                bottom,
                side="left",
            )

        # ---------------------------------------------------------------------
        # Faixa direita
        # ---------------------------------------------------------------------

        if right:
            self._draw_lane_curve(
                canvas,
                right,
                width,
                height,
                self._lane_color(
                    state,
                    side="right",
                ),
                solid=True,
            )
        else:
            self._draw_missing_lane(
                canvas,
                center_x
                + self.config.lane_width_visual,
                top,
                bottom,
                side="right",
            )

        # ---------------------------------------------------------------------
        # Centro estimado
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
        )

        vehicle_color = self._status_color(
            state.lateral_offset,
            state.active,
        )

        self._draw_vehicle(
            canvas,
            vehicle_x,
            bottom - 25,
            vehicle_color,
        )

        # ---------------------------------------------------------------------
        # Estado
        # ---------------------------------------------------------------------

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
        solid: bool = True,
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
            transformed,
        )

        flat = []

        for x, y in transformed:
            flat.extend(
                [
                    x,
                    y,
                ]
            )

        if solid:
            canvas.create_line(
                *flat,
                fill=color,
                width=self.config.lane_width,
                smooth=True,
                splinesteps=12,
            )

            # brilho.
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

        for i in range(
            segments
        ):
            if i % 2 == 0:
                y1 = top + i * segment
                y2 = top + (i + 0.65) * segment

                canvas.create_line(
                    x,
                    y1,
                    x,
                    y2,
                    fill=self.config.gray,
                    width=self.config.lane_dash_width,
                )

        label = (
            "FAIXA ESQUERDA\nNÃO DETECTADA"
            if side == "left"
            else
            "FAIXA DIREITA\nNÃO DETECTADA"
        )

        anchor = (
            "e"
            if side == "left"
            else "w"
        )

        text_x = (
            x - 18
            if side == "left"
            else x + 18
        )

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

        points = []

        for i in range(count):
            lx, ly = self._transform_lane_point(
                left[i],
                width,
                height,
            )

            rx, ry = self._transform_lane_point(
                right[i],
                width,
                height,
            )

            points.append(
                (
                    (lx + rx) / 2.0,
                    (ly + ry) / 2.0,
                )
            )

        flat = []

        for x, y in points:
            flat.extend(
                [
                    x,
                    y,
                ]
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

        width = 30
        height = 58

        # Sombra.
        canvas.create_oval(
            x - 25,
            y + 20,
            x + 25,
            y + 42,
            fill="#030508",
            outline="",
        )

        # Corpo.
        canvas.create_polygon(
            x - width / 2,
            y + height / 2,
            x - width / 2 + 4,
            y - height / 2 + 13,
            x - width / 2 + 11,
            y - height / 2,
            x + width / 2 - 11,
            y - height / 2,
            x + width / 2 - 4,
            y - height / 2 + 13,
            x + width / 2,
            y + height / 2,
            fill=color,
            outline="",
        )

        # Para-brisa.
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

        # Luz central.
        canvas.create_rectangle(
            x - 3,
            y + 8,
            x + 3,
            y + 17,
            fill=self.config.background,
            outline="",
        )

    # =========================================================================
    # OFFSET INDICATOR
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

        # Linha principal.
        canvas.create_line(
            left,
            y,
            right,
            y,
            fill=self.config.grid,
            width=8,
        )

        # Zonas.
        canvas.create_line(
            center
            - (right - left)
            * self.config.warning_threshold
            / 2.0,
            y,
            center
            + (right - left)
            * self.config.warning_threshold
            / 2.0,
            y,
            fill=self.config.yellow,
            width=8,
        )

        canvas.create_line(
            center
            - (right - left)
            * self.config.critical_threshold
            / 2.0,
            y,
            center
            + (right - left)
            * self.config.critical_threshold
            / 2.0,
            y,
            fill=self.config.red,
            width=8,
        )

        # Centro verde.
        canvas.create_line(
            center
            - (right - left)
            * self.config.centered_width
            if hasattr(
                self.config,
                "centered_width",
            )
            else center - 35,
            y,
            center
            + (
                (right - left)
                * self.config.centered_width
                if hasattr(
                    self.config,
                    "centered_width",
                )
                else 35
            ),
            y,
            fill=self.config.green,
            width=8,
        )

        # Marcador.
        marker_x = (
            center
            + state.lateral_offset
            * (right - left)
            / 2.0
        )

        marker_x = max(
            left,
            min(
                right,
                marker_x,
            ),
        )

        color = self._status_color(
            state.lateral_offset,
            state.active,
        )

        canvas.create_oval(
            marker_x - 9,
            y - 9,
            marker_x + 9,
            y + 9,
            fill=color,
            outline=self.config.text,
            width=2,
        )

        # Offset.
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

        color = self._status_color(
            state.lateral_offset,
            state.active,
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

        # Faixas.
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

        # Confiança.
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

        # Curvatura.
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
            text="⚠",
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
    # HELPERS
    # =========================================================================

    def _vehicle_x(
        self,
        center_x: float,
        offset: float,
        left: Optional[list[Point]],
        right: Optional[list[Point]],
        width: int,
    ) -> float:

        if left and right:
            left_x = self._transform_lane_point(
                left[-1],
                width,
                self.config.height,
            )[0]

            right_x = self._transform_lane_point(
                right[-1],
                width,
                self.config.height,
            )[0]

            lane_center = (
                left_x + right_x
            ) / 2.0

            lane_half = (
                right_x - left_x
            ) / 2.0

            if lane_half > 10:
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

        if proximity >= self.config.critical_threshold:
            return self.config.red

        if proximity >= self.config.warning_threshold:
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

        if magnitude >= self.config.critical_threshold:
            return self.config.red

        if magnitude >= self.config.warning_threshold:
            return self.config.yellow

        return self.config.green

    def _infer_state(
        self,
        offset: float,
        left: Optional[list[Point]],
        right: Optional[list[Point]],
        active: bool,
    ) -> str:

        if not active:
            return "ADAS INACTIVE"

        if not left and not right:
            return "LANE LOST"

        if offset <= -self.config.critical_threshold:
            return "LEFT DEPARTURE"

        if offset >= self.config.critical_threshold:
            return "RIGHT DEPARTURE"

        if offset <= -self.config.warning_threshold:
            return "LEFT WARNING"

        if offset >= self.config.warning_threshold:
            return "RIGHT WARNING"

        if offset < -0.20:
            return "SLIGHT LEFT"

        if offset > 0.20:
            return "SLIGHT RIGHT"

        return "CENTERED"

    def _format_offset(
        self,
        value: float,
    ) -> str:

        if abs(value) < 0.05:
            return "0%"

        sign = "+" if value > 0 else ""

        return f"{sign}{value:.1f}%"

    @staticmethod
    def _safe_float(
        value,
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
        points,
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
                if len(point) < 2:
                    continue

                x = float(point[0])
                y = float(point[1])

                if not (
                    math.isfinite(x)
                    and math.isfinite(y)
                ):
                    continue

                result.append(
                    (x, y)
                )

            except (
                TypeError,
                ValueError,
            ):
                continue

        if len(result) < 2:
            return None

        return result

    def _transform_lane_point(
        self,
        point: Point,
        width: int,
        height: int,
    ) -> Point:

        x, y = point

        # Normalização automática.
        #
        # O detector normalmente trabalha em coordenadas de imagem.
        # O painel transforma a coordenada vertical em profundidade.
        #
        # Se y aumenta para baixo:
        #
        #       horizonte
        #          ↓
        #       estrada
        #          ↓
        #       veículo

        # Tentativa de normalização.
        #
        # Para coordenadas de imagem 0..W / 0..H,
        # assumimos que o eixo x/y recebido já pertence à imagem.
        #
        # A escala é ajustada posteriormente pelo bounding visual.

        x_norm = x / max(
            1.0,
            width,
        )

        y_norm = y / max(
            1.0,
            height,
        )

        # Mantém proporção dentro do HUD.
        horizon = (
            height
            * self.config.horizon_ratio
        )

        bottom = (
            height
            - self.config.road_bottom_margin
        )

        # Perspectiva simples.
        depth = max(
            0.0,
            min(
                1.0,
                y_norm,
            ),
        )

        screen_y = (
            horizon
            + depth
            * (bottom - horizon)
        )

        screen_x = (
            x_norm
            * width
        )

        return (
            screen_x,
            screen_y,
        )

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
            px, py = points[i - 1]
            x, y = points[i]
            nx, ny = points[i + 1]

            result.append(
                (
                    (
                        px
                        + 2.0 * x
                        + nx
                    )
                    / 4.0,
                    (
                        py
                        + 2.0 * y
                        + ny
                    )
                    / 4.0,
                )
            )

        result.append(
            points[-1]
        )

        return result

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

        # Centro.
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

        # Cantos.
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
    Executa o HUD sem o Forza.

    Útil para testar a interface antes da integração.
    """

    display = ADASDisplay()

    display.start()

    t0 = time.monotonic()

    def demo_loop():
        t = time.monotonic() - t0

        # Curva senoidal suave.
        curve = math.sin(t * 0.45)

        left = []
        right = []

        for i in range(30):
            y = 150 + i * 45

            curve_offset = (
                math.sin(
                    i / 29.0 * math.pi
                )
                * 80
                * curve
            )

            left.append(
                (
                    780 + curve_offset,
                    y,
                )
            )

            right.append(
                (
                    1320 + curve_offset,
                    y,
                )
            )

        # Normaliza para uma imagem de exemplo 1900x900.
        left = [
            (
                x,
                y,
            )
            for x, y in left
        ]

        right = [
            (
                x,
                y,
            )
            for x, y in right
        ]

        offset = (
            math.sin(t * 0.65)
            * 0.45
        )

        confidence = (
            0.93
            + 0.03
            * math.sin(t)
        )

        display.update(
            left_lane=left,
            right_lane=right,
            lateral_offset=offset,
            confidence=confidence,
            curvature=curve * 0.03,
            active=True,
        )

        if display._running:
            threading.Timer(
                1.0 / 30.0,
                demo_loop,
            ).start()

    demo_loop()

    display.wait()


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