"""
Painel ADAS independente para o Forza Assistant.

O painel roda em uma janela Tkinter separada e pode receber
atualizações do pipeline principal de forma segura.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import threading
import tkinter as tk
from typing import Optional


class ADASDisplayState(Enum):
    UNKNOWN = "unknown"
    CENTERED = "centered"
    LEFT = "left"
    RIGHT = "right"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class ADASDisplayData:
    state: ADASDisplayState = ADASDisplayState.UNKNOWN

    lane_offset: Optional[float] = None
    lane_confidence: Optional[float] = None

    left_lane_detected: bool = False
    right_lane_detected: bool = False

    vehicle_detected: bool = False
    vehicle_distance: Optional[float] = None

    system_active: bool = True
    message: str = "Sistema iniciando..."


class ADASDisplay:
    """
    Painel ADAS independente.

    A interface Tkinter possui uma thread própria.
    O pipeline pode usar update_async() sem manipular
    diretamente os widgets Tkinter.
    """

    WINDOW_TITLE = "Forza Assistant - ADAS"
    WINDOW_WIDTH = 900
    WINDOW_HEIGHT = 600

    BG_COLOR = "#101010"
    PANEL_COLOR = "#181818"
    ROAD_COLOR = "#292929"

    TEXT_COLOR = "#EAEAEA"
    MUTED_COLOR = "#888888"

    CENTERED_COLOR = "#35D07F"
    WARNING_COLOR = "#F5C542"
    CRITICAL_COLOR = "#FF4D4D"
    INACTIVE_COLOR = "#666666"

    def __init__(
        self,
        master: Optional[tk.Misc] = None,
        width: int = WINDOW_WIDTH,
        height: int = WINDOW_HEIGHT,
    ) -> None:
        self.master = master
        self.width = width
        self.height = height

        self.window: Optional[tk.Misc] = None

        self.data = ADASDisplayData()

        self.status_label: Optional[tk.Label] = None
        self.message_label: Optional[tk.Label] = None
        self.confidence_label: Optional[tk.Label] = None
        self.offset_label: Optional[tk.Label] = None
        self.distance_label: Optional[tk.Label] = None
        self.system_label: Optional[tk.Label] = None

        self.vehicle_canvas: Optional[tk.Canvas] = None

        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._closed = threading.Event()

        self._data_lock = threading.Lock()

        self._pending_data: Optional[ADASDisplayData] = None

    # ==================================================================
    # THREAD / CICLO DE VIDA
    # ==================================================================

    def start(self) -> None:
        """
        Inicia a janela em uma thread dedicada.

        Retorna imediatamente depois que a janela foi criada.
        """

        if self.is_running():
            return

        self._ready.clear()
        self._closed.clear()

        self._thread = threading.Thread(
            target=self._thread_main,
            name="ADAS-Display",
            daemon=True,
        )

        self._thread.start()

        # Aguarda a criação da janela.
        self._ready.wait(timeout=3.0)

    def _thread_main(self) -> None:
        """Thread principal do Tkinter."""

        try:
            if self.master is not None:
                self.window = tk.Toplevel(self.master)
            else:
                self.window = tk.Tk()

            self.window.title(self.WINDOW_TITLE)
            self.window.geometry(
                f"{self.width}x{self.height}"
            )
            self.window.minsize(700, 450)
            self.window.configure(
                bg=self.BG_COLOR
            )

            self.window.protocol(
                "WM_DELETE_WINDOW",
                self.close,
            )

            self._build_ui()
            self._refresh_ui()

            self._ready.set()

            self.window.mainloop()

        except Exception:
            self._ready.set()
            raise

        finally:
            self._closed.set()
            self.window = None

    def show(self) -> None:
        """
        Compatibilidade com a API anterior.

        Se o painel ainda não estiver iniciado,
        inicia a thread Tkinter.
        """

        self.start()

    def run(self) -> None:
        """
        Executa o painel de forma independente.

        Usado principalmente para testes manuais.
        """

        if self.is_running():
            return

        self._ready.clear()
        self._closed.clear()

        self._thread_main()

    def close(self) -> None:
        """Fecha a janela."""

        window = self.window

        if window is None:
            return

        try:
            window.after(
                0,
                self._destroy_window,
            )
        except tk.TclError:
            pass

    def _destroy_window(self) -> None:
        """Destrói a janela dentro da thread Tkinter."""

        if self.window is None:
            return

        try:
            self.window.destroy()
        except tk.TclError:
            pass

    def is_visible(self) -> bool:
        """Retorna True se a janela existir."""

        window = self.window

        if window is None:
            return False

        try:
            return bool(
                window.winfo_exists()
            )
        except tk.TclError:
            return False

    def is_running(self) -> bool:
        """Retorna True enquanto a thread do painel estiver ativa."""

        return (
            self._thread is not None
            and self._thread.is_alive()
        )

    # ==================================================================
    # ATUALIZAÇÃO THREAD-SAFE
    # ==================================================================

    def update_async(
        self,
        state: Optional[ADASDisplayState] = None,
        lane_offset: Optional[float] = None,
        lane_confidence: Optional[float] = None,
        left_lane_detected: Optional[bool] = None,
        right_lane_detected: Optional[bool] = None,
        vehicle_detected: Optional[bool] = None,
        vehicle_distance: Optional[float] = None,
        system_active: Optional[bool] = None,
        message: Optional[str] = None,
        data: Optional[ADASDisplayData] = None,
    ) -> None:
        """
        Atualiza o painel de forma thread-safe.

        Este é o método que o main.py deverá usar.
        """

        with self._data_lock:

            if data is not None:
                self._pending_data = ADASDisplayData(
                    state=data.state,
                    lane_offset=data.lane_offset,
                    lane_confidence=data.lane_confidence,
                    left_lane_detected=data.left_lane_detected,
                    right_lane_detected=data.right_lane_detected,
                    vehicle_detected=data.vehicle_detected,
                    vehicle_distance=data.vehicle_distance,
                    system_active=data.system_active,
                    message=data.message,
                )

            else:
                current = self.data

                self._pending_data = ADASDisplayData(
                    state=(
                        state
                        if state is not None
                        else current.state
                    ),
                    lane_offset=(
                        lane_offset
                        if lane_offset is not None
                        else current.lane_offset
                    ),
                    lane_confidence=(
                        lane_confidence
                        if lane_confidence is not None
                        else current.lane_confidence
                    ),
                    left_lane_detected=(
                        left_lane_detected
                        if left_lane_detected is not None
                        else current.left_lane_detected
                    ),
                    right_lane_detected=(
                        right_lane_detected
                        if right_lane_detected is not None
                        else current.right_lane_detected
                    ),
                    vehicle_detected=(
                        vehicle_detected
                        if vehicle_detected is not None
                        else current.vehicle_detected
                    ),
                    vehicle_distance=(
                        vehicle_distance
                        if vehicle_distance is not None
                        else current.vehicle_distance
                    ),
                    system_active=(
                        system_active
                        if system_active is not None
                        else current.system_active
                    ),
                    message=(
                        message
                        if message is not None
                        else current.message
                    ),
                )

        window = self.window

        if window is None:
            return

        try:
            window.after(
                0,
                self._apply_pending_update,
            )
        except tk.TclError:
            pass

    def _apply_pending_update(self) -> None:
        """Aplica uma atualização dentro da thread Tkinter."""

        with self._data_lock:
            if self._pending_data is None:
                return

            self.data = self._pending_data
            self._pending_data = None

        self._refresh_ui()

    def update(
        self,
        state: Optional[ADASDisplayState] = None,
        lane_offset: Optional[float] = None,
        lane_confidence: Optional[float] = None,
        left_lane_detected: Optional[bool] = None,
        right_lane_detected: Optional[bool] = None,
        vehicle_detected: Optional[bool] = None,
        vehicle_distance: Optional[float] = None,
        system_active: Optional[bool] = None,
        message: Optional[str] = None,
        data: Optional[ADASDisplayData] = None,
    ) -> None:
        """
        Atualização compatível com a API anterior.

        Para integração com o main.py, prefira update_async().
        """

        self.update_async(
            state=state,
            lane_offset=lane_offset,
            lane_confidence=lane_confidence,
            left_lane_detected=left_lane_detected,
            right_lane_detected=right_lane_detected,
            vehicle_detected=vehicle_detected,
            vehicle_distance=vehicle_distance,
            system_active=system_active,
            message=message,
            data=data,
        )

    def set_state(
        self,
        state: ADASDisplayState,
        message: Optional[str] = None,
    ) -> None:
        """Atualiza somente o estado."""

        self.update_async(
            state=state,
            message=message,
        )

    # ==================================================================
    # INTERFACE
    # ==================================================================

    def _build_ui(self) -> None:
        """Constrói a interface."""

        if self.window is None:
            return

        header = tk.Frame(
            self.window,
            bg=self.PANEL_COLOR,
            height=70,
        )
        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(
            header,
            text="ADAS",
            bg=self.PANEL_COLOR,
            fg=self.TEXT_COLOR,
            font=("Segoe UI", 24, "bold"),
        ).pack(
            side="left",
            padx=25,
        )

        self.system_label = tk.Label(
            header,
            text="SISTEMA ATIVO",
            bg=self.PANEL_COLOR,
            fg=self.CENTERED_COLOR,
            font=("Segoe UI", 11, "bold"),
        )

        self.system_label.pack(
            side="right",
            padx=25,
        )

        content = tk.Frame(
            self.window,
            bg=self.BG_COLOR,
        )

        content.pack(
            fill="both",
            expand=True,
            padx=20,
            pady=20,
        )

        left_panel = tk.Frame(
            content,
            bg=self.PANEL_COLOR,
            width=260,
        )

        left_panel.pack(
            side="left",
            fill="y",
            padx=(0, 15),
        )

        left_panel.pack_propagate(False)

        tk.Label(
            left_panel,
            text="STATUS",
            bg=self.PANEL_COLOR,
            fg=self.MUTED_COLOR,
            font=("Segoe UI", 10, "bold"),
        ).pack(
            pady=(25, 5),
        )

        self.status_label = tk.Label(
            left_panel,
            text="UNKNOWN",
            bg=self.PANEL_COLOR,
            fg=self.TEXT_COLOR,
            font=("Segoe UI", 25, "bold"),
        )

        self.status_label.pack(
            pady=(0, 20),
        )

        self.message_label = tk.Label(
            left_panel,
            text="Sistema iniciando...",
            bg=self.PANEL_COLOR,
            fg=self.TEXT_COLOR,
            font=("Segoe UI", 12),
            wraplength=220,
            justify="center",
        )

        self.message_label.pack(
            padx=20,
            pady=10,
        )

        info_frame = tk.Frame(
            left_panel,
            bg=self.PANEL_COLOR,
        )

        info_frame.pack(
            fill="x",
            padx=20,
            pady=25,
        )

        self.confidence_label = self._create_info_row(
            info_frame,
            "Confiança",
        )

        self.offset_label = self._create_info_row(
            info_frame,
            "Offset",
        )

        self.distance_label = self._create_info_row(
            info_frame,
            "Distância",
        )

        road_panel = tk.Frame(
            content,
            bg=self.PANEL_COLOR,
        )

        road_panel.pack(
            side="left",
            fill="both",
            expand=True,
        )

        tk.Label(
            road_panel,
            text="VISÃO ADAS",
            bg=self.PANEL_COLOR,
            fg=self.MUTED_COLOR,
            font=("Segoe UI", 10, "bold"),
        ).pack(
            pady=(15, 5),
        )

        self.vehicle_canvas = tk.Canvas(
            road_panel,
            bg="#202020",
            highlightthickness=0,
        )

        self.vehicle_canvas.pack(
            fill="both",
            expand=True,
            padx=20,
            pady=(5, 20),
        )

        self.vehicle_canvas.bind(
            "<Configure>",
            lambda _event: self._draw_road(),
        )

    def _create_info_row(
        self,
        parent: tk.Misc,
        title: str,
    ) -> tk.Label:
        """Cria uma linha de informação."""

        frame = tk.Frame(
            parent,
            bg=self.PANEL_COLOR,
        )

        frame.pack(
            fill="x",
            pady=5,
        )

        tk.Label(
            frame,
            text=title,
            bg=self.PANEL_COLOR,
            fg=self.MUTED_COLOR,
            font=("Segoe UI", 10),
        ).pack(
            side="left",
        )

        value = tk.Label(
            frame,
            text="--",
            bg=self.PANEL_COLOR,
            fg=self.TEXT_COLOR,
            font=("Segoe UI", 10, "bold"),
        )

        value.pack(
            side="right",
        )

        return value

    # ==================================================================
    # REFRESH
    # ==================================================================

    def _refresh_ui(self) -> None:
        """Atualiza os elementos visuais."""

        if not self.is_visible():
            return

        if self.status_label is None:
            return

        if self.message_label is None:
            return

        if self.confidence_label is None:
            return

        if self.offset_label is None:
            return

        if self.distance_label is None:
            return

        if self.system_label is None:
            return

        state = self.data.state

        self.status_label.configure(
            text=state.name,
            fg=self._state_color(state),
        )

        self.message_label.configure(
            text=self.data.message,
        )

        if self.data.lane_confidence is None:
            confidence = "--"
        else:
            confidence = (
                f"{self.data.lane_confidence * 100:.1f}%"
            )

        self.confidence_label.configure(
            text=confidence,
        )

        if self.data.lane_offset is None:
            offset = "--"
        else:
            offset = f"{self.data.lane_offset:+.3f}"

        self.offset_label.configure(
            text=offset,
        )

        if self.data.vehicle_distance is None:
            distance = "--"
        else:
            distance = (
                f"{self.data.vehicle_distance:.1f} m"
            )

        self.distance_label.configure(
            text=distance,
        )

        if self.data.system_active:

            self.system_label.configure(
                text="SISTEMA ATIVO",
                fg=self.CENTERED_COLOR,
            )

        else:

            self.system_label.configure(
                text="SISTEMA INATIVO",
                fg=self.INACTIVE_COLOR,
            )

        self._draw_road()

    # ==================================================================
    # ROAD VIEW
    # ==================================================================

    def _draw_road(self) -> None:
        """Desenha a representação da estrada."""

        canvas = self.vehicle_canvas

        if canvas is None:
            return

        try:
            width = canvas.winfo_width()
            height = canvas.winfo_height()

        except tk.TclError:
            return

        if width <= 1 or height <= 1:
            return

        canvas.delete("all")

        center_x = width / 2

        road_left = width * 0.18
        road_right = width * 0.82

        canvas.create_polygon(
            road_left,
            height,
            width * 0.38,
            0,
            width * 0.62,
            0,
            road_right,
            height,
            fill=self.ROAD_COLOR,
            outline="",
        )

        left_x_bottom = (
            road_left + width * 0.20
        )

        left_x_top = width * 0.43

        right_x_bottom = (
            road_right - width * 0.20
        )

        right_x_top = width * 0.57

        left_color = (
            self.CENTERED_COLOR
            if self.data.left_lane_detected
            else self.INACTIVE_COLOR
        )

        right_color = (
            self.CENTERED_COLOR
            if self.data.right_lane_detected
            else self.INACTIVE_COLOR
        )

        canvas.create_line(
            left_x_bottom,
            height,
            left_x_top,
            0,
            fill=left_color,
            width=5,
        )

        canvas.create_line(
            right_x_bottom,
            height,
            right_x_top,
            0,
            fill=right_color,
            width=5,
        )

        vehicle_width = max(
            40,
            width * 0.08,
        )

        vehicle_height = max(
            60,
            height * 0.12,
        )

        vehicle_y = height * 0.78

        offset_pixels = 0.0

        if self.data.lane_offset is not None:

            normalized_offset = max(
                -1.0,
                min(
                    1.0,
                    self.data.lane_offset,
                ),
            )

            offset_pixels = (
                normalized_offset
                * width
                * 0.12
            )

        vehicle_x = (
            center_x + offset_pixels
        )

        vehicle_color = self._state_color(
            self.data.state,
        )

        canvas.create_rectangle(
            vehicle_x - vehicle_width / 2,
            vehicle_y - vehicle_height / 2,
            vehicle_x + vehicle_width / 2,
            vehicle_y + vehicle_height / 2,
            fill=vehicle_color,
            outline="",
        )

        if self.data.vehicle_detected:

            target_y = height * 0.30

            target_width = (
                vehicle_width * 0.75
            )

            target_height = (
                vehicle_height * 0.70
            )

            canvas.create_rectangle(
                center_x - target_width / 2,
                target_y - target_height / 2,
                center_x + target_width / 2,
                target_y + target_height / 2,
                fill=self.CRITICAL_COLOR,
                outline="",
            )

            if self.data.vehicle_distance is not None:

                distance_text = (
                    f"{self.data.vehicle_distance:.1f} m"
                )

            else:

                distance_text = "VEÍCULO"

            canvas.create_text(
                center_x,
                target_y - target_height,
                text=distance_text,
                fill=self.TEXT_COLOR,
                font=("Segoe UI", 10, "bold"),
            )

        canvas.create_line(
            center_x,
            height * 0.05,
            center_x,
            height * 0.15,
            fill="#555555",
            width=2,
        )

    # ==================================================================
    # ESTADO
    # ==================================================================

    def _state_color(
        self,
        state: ADASDisplayState,
    ) -> str:

        if state == ADASDisplayState.CENTERED:
            return self.CENTERED_COLOR

        if state in (
            ADASDisplayState.LEFT,
            ADASDisplayState.RIGHT,
            ADASDisplayState.WARNING,
        ):
            return self.WARNING_COLOR

        if state == ADASDisplayState.CRITICAL:
            return self.CRITICAL_COLOR

        return self.INACTIVE_COLOR


# ======================================================================
# TESTE INDEPENDENTE
# ======================================================================

def main() -> None:
    """Teste independente do painel."""

    display = ADASDisplay()

    display.start()

    display.update_async(
        state=ADASDisplayState.CENTERED,
        lane_offset=0.02,
        lane_confidence=0.96,
        left_lane_detected=True,
        right_lane_detected=True,
        vehicle_detected=False,
        system_active=True,
        message="Veículo centralizado na faixa.",
    )

    if display._thread is not None:
        display._thread.join()


if __name__ == "__main__":
    main()