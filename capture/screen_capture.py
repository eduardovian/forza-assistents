"""
capture/screen_capture.py

Forza Assistents
================

Camada de captura de tela de baixa latência.

Responsabilidades
-----------------
- Capturar frames da tela.
- Utilizar DXGI/DXCam quando disponível.
- Aplicar exclusivamente o ROI definido em config.py.
- Garantir que todos os frames tenham formato consistente.
- Fornecer timestamp monotônico e métricas básicas.
- Detectar perda de frames.
- Encerrar o backend de forma segura.

Arquitetura
-----------

    Screen
      │
      ▼
    DXGI / DXCam
      │
      ▼
    Full Frame
      │
      ▼
    config.ROI
      │
      ▼
    ROI Frame
      │
      ▼
    Vision Pipeline

PRINCÍPIO
---------

ROI NÃO é configurado neste módulo.

A única fonte é:

    from config import ROI

Nenhum outro módulo deve possuir coordenadas duplicadas de ROI.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Final

import numpy as np

from config import CAPTURE, ROI


# =============================================================================
# LOGGING
# =============================================================================

LOGGER = logging.getLogger(
    __name__
)


# =============================================================================
# CONSTANTS
# =============================================================================

NANOSECONDS_PER_SECOND: Final[int] = 1_000_000_000

MIN_FRAME_DIMENSION: Final[int] = 32


# =============================================================================
# TYPES
# =============================================================================


@dataclass(frozen=True, slots=True)
class FramePacket:
    """
    Frame capturado pelo sistema.

    O frame retornado já está no sistema de coordenadas
    do ROI.

    Attributes
    ----------
    frame:
        Imagem BGR/RGB conforme config.CAPTURE.

    timestamp_ns:
        Timestamp monotônico de aquisição.

    sequence:
        Número sequencial do frame.

    source_width:
        Largura do frame original.

    source_height:
        Altura do frame original.

    roi_applied:
        Indica se o ROI foi aplicado.

    capture_latency_ns:
        Tempo gasto para obter o frame.
    """

    frame: np.ndarray

    timestamp_ns: int

    sequence: int

    source_width: int

    source_height: int

    roi_applied: bool

    capture_latency_ns: int

    @property
    def width(self) -> int:
        return int(
            self.frame.shape[1]
        )

    @property
    def height(self) -> int:
        return int(
            self.frame.shape[0]
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return self.frame.shape


@dataclass(frozen=True, slots=True)
class CaptureMetrics:
    """
    Métricas acumuladas da captura.
    """

    frames_captured: int

    frames_dropped: int

    consecutive_failures: int

    elapsed_seconds: float

    effective_fps: float

    average_latency_ms: float

    last_frame_timestamp_ns: int


# =============================================================================
# BACKEND
# =============================================================================


class _DXCamBackend:
    """
    Backend DXCam encapsulado.

    A implementação concreta de DXCam fica isolada desta classe
    para que o restante do sistema não dependa diretamente
    da API da biblioteca.
    """

    def __init__(
        self,
        monitor_index: int,
    ) -> None:

        try:
            import dxcam
        except ImportError as exc:

            raise RuntimeError(
                "DXCam não está instalado."
            ) from exc

        self._camera = dxcam.create(
            output_idx=monitor_index,
            output_color=CAPTURE.output_color_format,
        )

        if self._camera is None:

            raise RuntimeError(
                "DXCam não conseguiu criar "
                "o dispositivo de captura."
            )

        self._started = False

    def start(
        self,
        target_fps: int,
    ) -> None:

        if self._started:
            return

        self._camera.start(
            target_fps=target_fps,
            video_mode=False,
        )

        self._started = True

    def grab(self) -> np.ndarray | None:

        if not self._started:

            raise RuntimeError(
                "Backend DXCam não foi iniciado."
            )

        frame = self._camera.get_latest_frame()

        if frame is None:
            return None

        return frame

    def stop(self) -> None:

        if not self._started:
            return

        try:
            self._camera.stop()
        finally:
            self._started = False


# =============================================================================
# SCREEN CAPTURE
# =============================================================================


class ScreenCapture:
    """
    Capturador principal do Forza Assistents.

    Características
    ---------------

    - baixa latência;
    - backend DXGI/DXCam;
    - ROI centralizado em config.py;
    - sem cópias desnecessárias;
    - métricas temporais;
    - detecção de falhas;
    - shutdown seguro.

    Exemplo
    -------

        capture = ScreenCapture()

        capture.start()

        packet = capture.read()

        if packet is not None:
            frame = packet.frame

        capture.stop()
    """

    def __init__(
        self,
        *,
        monitor_index: int | None = None,
    ) -> None:

        self._monitor_index = (
            CAPTURE.monitor_index
            if monitor_index is None
            else monitor_index
        )

        self._backend: _DXCamBackend | None = None

        self._started = False

        self._sequence = 0

        self._frames_captured = 0

        self._frames_dropped = 0

        self._consecutive_failures = 0

        self._total_latency_ns = 0

        self._start_time_ns: int | None = None

        self._last_frame_timestamp_ns = 0

        self._last_source_shape: (
            tuple[int, int] | None
        ) = None

        self._validate_configuration()

    # =========================================================================
    # VALIDATION
    # =========================================================================

    @staticmethod
    def _validate_configuration() -> None:
        """
        Valida a configuração antes de inicializar o backend.
        """

        CAPTURE.validate()

        if not ROI.enabled:

            raise RuntimeError(
                "ROI não calibrado. "
                "Execute calibration/camera_calibration.py "
                "antes de iniciar o sistema."
            )

        ROI.validate()

        if ROI.width < MIN_FRAME_DIMENSION:

            raise RuntimeError(
                "ROI possui largura inválida."
            )

        if ROI.height < MIN_FRAME_DIMENSION:

            raise RuntimeError(
                "ROI possui altura inválida."
            )

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    def start(self) -> None:
        """
        Inicializa o backend de captura.
        """

        if self._started:
            return

        self._validate_configuration()

        backend = _DXCamBackend(
            self._monitor_index
        )

        backend.start(
            CAPTURE.target_fps
        )

        self._backend = backend

        self._started = True

        self._start_time_ns = (
            time.monotonic_ns()
        )

        self._sequence = 0

        self._frames_captured = 0

        self._frames_dropped = 0

        self._consecutive_failures = 0

        self._total_latency_ns = 0

        self._last_frame_timestamp_ns = 0

        LOGGER.info(
            "ScreenCapture iniciado: "
            "monitor=%d fps=%d ROI=%s",
            self._monitor_index,
            CAPTURE.target_fps,
            ROI.rectangle,
        )

    def stop(self) -> None:
        """
        Encerra o backend de forma segura.
        """

        backend = self._backend

        self._backend = None

        self._started = False

        if backend is not None:

            try:
                backend.stop()

            except Exception:

                LOGGER.exception(
                    "Erro ao encerrar backend "
                    "de captura."
                )

        LOGGER.info(
            "ScreenCapture encerrado."
        )

    def __enter__(self) -> ScreenCapture:

        self.start()

        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:

        self.stop()

    # =========================================================================
    # CAPTURE
    # =========================================================================

    def read(
        self,
    ) -> FramePacket | None:
        """
        Captura o frame mais recente.

        Returns
        -------
        FramePacket | None

            Frame válido ou None quando não há frame novo.

        Notes
        -----

        Não reutilizamos silenciosamente o frame anterior.
        Um frame antigo não deve ser tratado como observação
        atual pelo pipeline temporal.
        """

        if not self._started:

            raise RuntimeError(
                "ScreenCapture não foi iniciado."
            )

        if self._backend is None:

            raise RuntimeError(
                "Backend de captura indisponível."
            )

        capture_start_ns = (
            time.monotonic_ns()
        )

        frame = self._backend.grab()

        capture_end_ns = (
            time.monotonic_ns()
        )

        latency_ns = (
            capture_end_ns
            - capture_start_ns
        )

        if frame is None:

            self._frames_dropped += 1

            self._consecutive_failures += 1

            return None

        self._consecutive_failures = 0

        self._validate_frame(
            frame
        )

        source_height, source_width = (
            frame.shape[:2]
        )

        self._last_source_shape = (
            source_width,
            source_height,
        )

        cropped = self._apply_roi(
            frame
        )

        if CAPTURE.copy_frame:

            cropped = cropped.copy()

        timestamp_ns = (
            capture_end_ns
        )

        sequence = (
            self._sequence
        )

        self._sequence += 1

        self._frames_captured += 1

        self._total_latency_ns += (
            latency_ns
        )

        self._last_frame_timestamp_ns = (
            timestamp_ns
        )

        return FramePacket(
            frame=cropped,
            timestamp_ns=timestamp_ns,
            sequence=sequence,
            source_width=source_width,
            source_height=source_height,
            roi_applied=True,
            capture_latency_ns=latency_ns,
        )

    # =========================================================================
    # ROI
    # =========================================================================

    @staticmethod
    def _apply_roi(
        frame: np.ndarray,
    ) -> np.ndarray:
        """
        Aplica exclusivamente config.ROI.

        O ROI é definido em coordenadas absolutas da tela.

        O frame retornado passa a utilizar coordenadas locais:

            (0, 0)
                │
                └── canto superior esquerdo do ROI
        """

        left = ROI.left
        top = ROI.top
        right = ROI.right
        bottom = ROI.bottom

        frame_height, frame_width = (
            frame.shape[:2]
        )

        if right > frame_width:

            raise RuntimeError(
                "ROI excede a largura real "
                "do frame capturado: "
                f"ROI.right={right}, "
                f"frame_width={frame_width}."
            )

        if bottom > frame_height:

            raise RuntimeError(
                "ROI excede a altura real "
                "do frame capturado: "
                f"ROI.bottom={bottom}, "
                f"frame_height={frame_height}."
            )

        cropped = frame[
            top:bottom,
            left:right,
        ]

        if cropped.size == 0:

            raise RuntimeError(
                "ROI produziu um frame vazio."
            )

        return cropped

    # =========================================================================
    # FRAME VALIDATION
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
                "Backend retornou um objeto "
                "que não é numpy.ndarray."
            )

        if frame.ndim != 3:

            raise ValueError(
                "Frame deve possuir dimensão "
                "(height, width, channels)."
            )

        if frame.shape[2] != 3:

            raise ValueError(
                "Frame deve possuir 3 canais."
            )

        if frame.shape[0] < MIN_FRAME_DIMENSION:

            raise ValueError(
                "Frame possui altura inválida."
            )

        if frame.shape[1] < MIN_FRAME_DIMENSION:

            raise ValueError(
                "Frame possui largura inválida."
            )

        if frame.dtype != np.uint8:

            raise ValueError(
                "Frame deve utilizar dtype uint8."
            )

    # =========================================================================
    # METRICS
    # =========================================================================

    def metrics(
        self,
    ) -> CaptureMetrics:

        if (
            self._start_time_ns is None
        ):

            elapsed_seconds = 0.0

        else:

            elapsed_seconds = (
                time.monotonic_ns()
                - self._start_time_ns
            ) / NANOSECONDS_PER_SECOND

        if elapsed_seconds > 0:

            fps = (
                self._frames_captured
                / elapsed_seconds
            )

        else:

            fps = 0.0

        if self._frames_captured > 0:

            average_latency_ms = (
                self._total_latency_ns
                / self._frames_captured
                / 1_000_000
            )

        else:

            average_latency_ms = 0.0

        return CaptureMetrics(
            frames_captured=(
                self._frames_captured
            ),
            frames_dropped=(
                self._frames_dropped
            ),
            consecutive_failures=(
                self._consecutive_failures
            ),
            elapsed_seconds=(
                elapsed_seconds
            ),
            effective_fps=fps,
            average_latency_ms=(
                average_latency_ms
            ),
            last_frame_timestamp_ns=(
                self._last_frame_timestamp_ns
            ),
        )

    # =========================================================================
    # STATE
    # =========================================================================

    @property
    def is_running(self) -> bool:

        return self._started

    @property
    def sequence(self) -> int:

        return self._sequence

    @property
    def roi(self):
        """
        Retorna o ROI centralizado.

        Não permite alteração.
        """

        return ROI

    @property
    def source_shape(
        self,
    ) -> tuple[int, int] | None:

        return self._last_source_shape

    @property
    def frame_size(
        self,
    ) -> tuple[int, int]:

        return (
            ROI.width,
            ROI.height,
        )


# =============================================================================
# FACTORY
# =============================================================================


def create_screen_capture(
    *,
    monitor_index: int | None = None,
) -> ScreenCapture:
    """
    Factory oficial utilizada pelo main.py.
    """

    return ScreenCapture(
        monitor_index=monitor_index
    )


# =============================================================================
# SELF TEST
# =============================================================================


def _self_test() -> None:
    """
    Teste básico da camada de captura.

    Não inicia automaticamente quando o módulo é importado.
    """

    print(
        "ScreenCapture self-test"
    )

    print(
        f"ROI: {ROI.rectangle}"
    )

    print(
        f"ROI size: "
        f"{ROI.width}x{ROI.height}"
    )

    capture = ScreenCapture()

    try:

        capture.start()

        packet = capture.read()

        if packet is None:

            raise RuntimeError(
                "Nenhum frame foi capturado."
            )

        print(
            "Frame:"
            f" {packet.width}x"
            f"{packet.height}"
        )

        print(
            f"Sequence: "
            f"{packet.sequence}"
        )

        print(
            f"Latency: "
            f"{packet.capture_latency_ns / 1_000_000:.2f} ms"
        )

    finally:

        capture.stop()


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    "FramePacket",
    "CaptureMetrics",
    "ScreenCapture",
    "create_screen_capture",
]


if __name__ == "__main__":
    _self_test()