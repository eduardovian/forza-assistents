"""
capture/screen_capture.py

Forza Assistents
================

Camada profissional de captura de tela de baixa latência.

Arquitetura:

    Windows Desktop
          │
          ├── DXCam / DXGI
          │       │
          │       └── Backend primário
          │
          └── Windows GDI / PIL
                  │
                  └── Fallback robusto
                          │
                          ▼
                     FramePacket
                          │
                          ▼
                         ROI
                          │
                          ▼
                    Vision Pipeline


PRINCÍPIOS
----------

1. DXCam é o backend preferencial.
2. Falha do DXCam NÃO deve derrubar o pipeline.
3. O fallback mantém o mesmo contrato público.
4. ROI existe exclusivamente em config.py.
5. Nenhum frame antigo é reutilizado silenciosamente.
6. Coordenadas retornadas pelo capture são locais ao ROI.
7. O restante do pipeline não precisa saber qual backend
   está sendo utilizado.
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

LOGGER = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

NANOSECONDS_PER_SECOND: Final[int] = 1_000_000_000
NANOSECONDS_PER_MILLISECOND: Final[int] = 1_000_000

MIN_FRAME_DIMENSION: Final[int] = 32

DEFAULT_DXCAM_RECOVERY_INTERVAL: Final[int] = 300
DEFAULT_MAX_BACKEND_FAILURES: Final[int] = 3

MAX_CAPTURE_FAILURES_BEFORE_FALLBACK: Final[int] = 2


# =============================================================================
# TYPES
# =============================================================================


@dataclass(frozen=True, slots=True)
class FramePacket:
    """
    Frame capturado pelo sistema.

    O frame retornado já está no sistema de coordenadas do ROI.
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
        return int(self.frame.shape[1])

    @property
    def height(self) -> int:
        return int(self.frame.shape[0])

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

    backend: str = "unknown"

    backend_switches: int = 0


# =============================================================================
# BACKEND ERROR
# =============================================================================


class CaptureBackendError(RuntimeError):
    """
    Erro controlado de backend de captura.
    """


# =============================================================================
# DXCAM BACKEND
# =============================================================================


class _DXCamBackend:
    """
    Backend DXCam / DXGI.

    Não deixa detalhes da biblioteca vazarem para o restante
    do sistema.
    """

    name = "dxcam"

    def __init__(
        self,
        monitor_index: int,
    ) -> None:

        self._camera = None
        self._started = False

        try:
            import dxcam

        except ImportError as exc:

            raise CaptureBackendError(
                "DXCam não está instalado."
            ) from exc

        try:

            self._camera = dxcam.create(
                output_idx=int(monitor_index),
                output_color=CAPTURE.output_color_format,
            )

        except Exception as exc:

            raise CaptureBackendError(
                "Falha ao inicializar "
                f"DXCam/Desktop Duplication: {exc}"
            ) from exc

        if self._camera is None:

            raise CaptureBackendError(
                "DXCam retornou uma câmera inválida."
            )

    def start(
        self,
        target_fps: int,
    ) -> None:

        if self._started:
            return

        if self._camera is None:

            raise CaptureBackendError(
                "Câmera DXCam indisponível."
            )

        try:

            self._camera.start(
                target_fps=max(
                    1,
                    int(target_fps),
                ),
                video_mode=False,
            )

            self._started = True

        except Exception as exc:

            raise CaptureBackendError(
                f"Falha ao iniciar DXCam: {exc}"
            ) from exc

    def grab(self) -> np.ndarray | None:

        if not self._started:

            raise CaptureBackendError(
                "DXCam não foi iniciado."
            )

        try:

            frame = (
                self._camera.get_latest_frame()
            )

        except Exception as exc:

            raise CaptureBackendError(
                f"Falha durante captura DXCam: {exc}"
            ) from exc

        if frame is None:
            return None

        return frame

    def stop(self) -> None:

        camera = self._camera

        self._camera = None
        self._started = False

        if camera is None:
            return

        try:

            camera.stop()

        except Exception:

            LOGGER.debug(
                "Erro ao parar DXCam.",
                exc_info=True,
            )


# =============================================================================
# WINDOWS GDI / PIL BACKEND
# =============================================================================


class _GDIBackend:
    """
    Backend de fallback baseado no Windows GDI através de PIL.ImageGrab.

    É mais lento que DXCam, porém evita depender de Desktop Duplication.

    Este backend captura a tela inteira e o ROI é aplicado posteriormente
    pelo ScreenCapture, mantendo o contrato arquitetural.
    """

    name = "windows-gdi"

    def __init__(
        self,
        monitor_index: int,
    ) -> None:

        del monitor_index

        try:
            from PIL import ImageGrab

        except ImportError as exc:

            raise CaptureBackendError(
                "Pillow não está instalado. "
                "Instale com: pip install pillow"
            ) from exc

        self._image_grab = ImageGrab
        self._started = False

    def start(
        self,
        target_fps: int,
    ) -> None:

        del target_fps

        if self._started:
            return

        self._started = True

    def grab(self) -> np.ndarray | None:

        if not self._started:

            raise CaptureBackendError(
                "Windows GDI não foi iniciado."
            )

        try:

            image = self._image_grab.grab(
                all_screens=False
            )

        except Exception as exc:

            raise CaptureBackendError(
                f"Falha durante captura GDI: {exc}"
            ) from exc

        if image is None:
            return None

        frame = np.asarray(
            image,
            dtype=np.uint8,
        )

        if frame.ndim != 3:
            raise CaptureBackendError(
                "GDI retornou imagem inválida."
            )

        if frame.shape[2] == 4:

            frame = frame[:, :, :3]

        # PIL -> numpy produz RGB.
        # O contrato do projeto normalmente utiliza BGR.
        output_format = str(
            getattr(
                CAPTURE,
                "output_color_format",
                "BGR",
            )
        ).upper()

        if output_format == "BGR":

            frame = np.ascontiguousarray(
                frame[:, :, ::-1]
            )

        else:

            frame = np.ascontiguousarray(
                frame
            )

        return frame

    def stop(self) -> None:

        self._started = False


# =============================================================================
# SCREEN CAPTURE
# =============================================================================


class ScreenCapture:
    """
    Capturador principal do Forza Assistents.

    Backend:

        DXCam -> preferencial

        Windows GDI -> fallback

    A interface pública permanece independente do backend.
    """

    def __init__(
        self,
        *,
        monitor_index: int | None = None,
        dxcam_recovery_interval: int = (
            DEFAULT_DXCAM_RECOVERY_INTERVAL
        ),
        max_backend_failures: int = (
            DEFAULT_MAX_BACKEND_FAILURES
        ),
    ) -> None:

        self._monitor_index = (
            CAPTURE.monitor_index
            if monitor_index is None
            else int(monitor_index)
        )

        self._dxcam_recovery_interval = max(
            1,
            int(dxcam_recovery_interval),
        )

        self._max_backend_failures = max(
            1,
            int(max_backend_failures),
        )

        self._backend = None

        self._backend_name = "none"

        self._started = False

        self._sequence = 0

        self._frames_captured = 0

        self._frames_dropped = 0

        self._consecutive_failures = 0

        self._backend_failures = 0

        self._backend_switches = 0

        self._frames_since_dxcam_attempt = 0

        self._total_latency_ns = 0

        self._start_time_ns: int | None = None

        self._last_frame_timestamp_ns = 0

        self._last_source_shape: (
            tuple[int, int] | None
        ) = None

        self._last_frame_signature: bytes | None = None

        self._validate_configuration()

    # =========================================================================
    # VALIDATION
    # =========================================================================

    @staticmethod
    def _validate_configuration() -> None:

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
    # BACKEND CREATION
    # =========================================================================

    def _create_dxcam(self):

        try:

            backend = _DXCamBackend(
                self._monitor_index
            )

            backend.start(
                CAPTURE.target_fps
            )

            return backend

        except Exception as exc:

            LOGGER.warning(
                "Backend DXCAM indisponível: %s",
                exc,
            )

            return None

    def _create_fallback(self):

        try:

            backend = _GDIBackend(
                self._monitor_index
            )

            backend.start(
                CAPTURE.target_fps
            )

            return backend

        except Exception as exc:

            LOGGER.error(
                "Backend de fallback indisponível: %s",
                exc,
            )

            return None

    def _select_initial_backend(self) -> None:
        """
        Seleciona DXCam primeiro e GDI como fallback.
        """

        backend = self._create_dxcam()

        if backend is not None:

            self._backend = backend
            self._backend_name = backend.name

            LOGGER.info(
                "ScreenCapture backend: DXCam/DXGI"
            )

            return

        backend = self._create_fallback()

        if backend is not None:

            self._backend = backend
            self._backend_name = backend.name

            LOGGER.warning(
                "ScreenCapture usando fallback "
                "Windows GDI."
            )

            return

        raise RuntimeError(
            "Nenhum backend de captura disponível."
        )

    # =========================================================================
    # BACKEND MANAGEMENT
    # =========================================================================

    def _stop_backend(self) -> None:

        backend = self._backend

        self._backend = None
        self._backend_name = "none"

        if backend is None:
            return

        try:

            backend.stop()

        except Exception:

            LOGGER.debug(
                "Erro ao encerrar backend.",
                exc_info=True,
            )

    def _switch_to_fallback(self) -> None:
        """
        Troca imediatamente para o backend de fallback.
        """

        LOGGER.warning(
            "Falha persistente no backend %s. "
            "Ativando fallback.",
            self._backend_name,
        )

        self._stop_backend()

        backend = self._create_fallback()

        if backend is None:

            raise RuntimeError(
                "DXCam falhou e o backend "
                "de fallback também não pôde ser iniciado."
            )

        self._backend = backend

        self._backend_name = backend.name

        self._backend_switches += 1

        self._backend_failures = 0

        self._frames_since_dxcam_attempt = 0

        LOGGER.warning(
            "Backend de captura alterado para: %s",
            self._backend_name,
        )

    def _try_recover_dxcam(self) -> None:
        """
        Tenta recuperar DXCam periodicamente.

        Isso permite que o sistema volte ao backend de baixa
        latência depois que o problema do Desktop Duplication
        for resolvido.
        """

        if self._backend_name != "windows-gdi":
            return

        self._frames_since_dxcam_attempt += 1

        if (
            self._frames_since_dxcam_attempt
            < self._dxcam_recovery_interval
        ):
            return

        self._frames_since_dxcam_attempt = 0

        LOGGER.info(
            "Tentando recuperar backend DXCam..."
        )

        backend = self._create_dxcam()

        if backend is None:
            return

        old_backend = self._backend

        self._backend = backend
        self._backend_name = backend.name

        self._backend_failures = 0
        self._backend_switches += 1

        if old_backend is not None:

            try:
                old_backend.stop()
            except Exception:
                LOGGER.debug(
                    "Erro ao encerrar fallback.",
                    exc_info=True,
                )

        LOGGER.info(
            "DXCam recuperado com sucesso."
        )

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    def start(self) -> None:

        if self._started:
            return

        self._validate_configuration()

        self._select_initial_backend()

        self._started = True

        self._start_time_ns = (
            time.monotonic_ns()
        )

        self._sequence = 0
        self._frames_captured = 0
        self._frames_dropped = 0
        self._consecutive_failures = 0
        self._backend_failures = 0
        self._backend_switches = 0
        self._frames_since_dxcam_attempt = 0
        self._total_latency_ns = 0
        self._last_frame_timestamp_ns = 0
        self._last_source_shape = None
        self._last_frame_signature = None

        LOGGER.info(
            "ScreenCapture iniciado: "
            "monitor=%d fps=%d ROI=%s backend=%s",
            self._monitor_index,
            CAPTURE.target_fps,
            ROI.rectangle,
            self._backend_name,
        )

    def stop(self) -> None:

        self._stop_backend()

        self._started = False

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

    def _grab_backend(self) -> np.ndarray | None:

        if self._backend is None:

            raise CaptureBackendError(
                "Backend de captura indisponível."
            )

        return self._backend.grab()

    def read(
        self,
    ) -> FramePacket | None:
        """
        Captura um novo frame.

        Nunca reutiliza silenciosamente o frame anterior.
        """

        if not self._started:

            raise RuntimeError(
                "ScreenCapture não foi iniciado."
            )

        capture_start_ns = (
            time.monotonic_ns()
        )

        try:

            frame = self._grab_backend()

        except CaptureBackendError as exc:

            self._backend_failures += 1
            self._consecutive_failures += 1

            LOGGER.warning(
                "Falha de captura [%s]: %s",
                self._backend_name,
                exc,
            )

            if (
                self._backend_name == "dxcam"
                and self._backend_failures
                >= MAX_CAPTURE_FAILURES_BEFORE_FALLBACK
            ):

                self._switch_to_fallback()

            elif (
                self._backend_failures
                >= self._max_backend_failures
            ):

                self._switch_to_fallback()

            self._frames_dropped += 1

            return None

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

        self._backend_failures = 0
        self._consecutive_failures = 0

        self._validate_frame(frame)

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

        else:

            cropped = np.ascontiguousarray(
                cropped
            )

        timestamp_ns = capture_end_ns

        sequence = self._sequence

        self._sequence += 1

        self._frames_captured += 1

        self._total_latency_ns += latency_ns

        self._last_frame_timestamp_ns = (
            timestamp_ns
        )

        self._try_recover_dxcam()

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

        left = int(ROI.left)
        top = int(ROI.top)
        right = int(ROI.right)
        bottom = int(ROI.bottom)

        frame_height, frame_width = (
            frame.shape[:2]
        )

        if left < 0 or top < 0:

            raise RuntimeError(
                "ROI possui coordenadas negativas."
            )

        if right <= left:

            raise RuntimeError(
                "ROI possui largura inválida."
            )

        if bottom <= top:

            raise RuntimeError(
                "ROI possui altura inválida."
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
                "Frame deve possuir exatamente 3 canais."
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

        if self._start_time_ns is None:

            elapsed_seconds = 0.0

        else:

            elapsed_seconds = (
                time.monotonic_ns()
                - self._start_time_ns
            ) / NANOSECONDS_PER_SECOND

        if elapsed_seconds > 0.0:

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
                / NANOSECONDS_PER_MILLISECOND
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
            backend=self._backend_name,
            backend_switches=(
                self._backend_switches
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
    def backend_name(self) -> str:

        return self._backend_name

    @property
    def roi(self):

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

    @property
    def backend_available(self) -> bool:

        return (
            self._backend is not None
            and self._started
        )

    # =========================================================================
    # RESET
    # =========================================================================

    def reset_metrics(self) -> None:

        self._sequence = 0
        self._frames_captured = 0
        self._frames_dropped = 0
        self._consecutive_failures = 0
        self._backend_failures = 0
        self._total_latency_ns = 0
        self._last_frame_timestamp_ns = 0
        self._start_time_ns = (
            time.monotonic_ns()
            if self._started
            else None
        )


# =============================================================================
# FACTORY
# =============================================================================


def create_screen_capture(
    *,
    monitor_index: int | None = None,
) -> ScreenCapture:

    return ScreenCapture(
        monitor_index=monitor_index
    )


# =============================================================================
# SELF TEST
# =============================================================================


def _self_test() -> None:

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

        print(
            f"Backend: "
            f"{capture.backend_name}"
        )

        packet = None

        for _ in range(30):

            packet = capture.read()

            if packet is not None:
                break

            time.sleep(0.01)

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
            f"Source: "
            f"{packet.source_width}x"
            f"{packet.source_height}"
        )

        print(
            f"Sequence: "
            f"{packet.sequence}"
        )

        print(
            f"Latency: "
            f"{packet.capture_latency_ns / 1_000_000:.2f} ms"
        )

        metrics = capture.metrics()

        print(
            f"FPS: "
            f"{metrics.effective_fps:.2f}"
        )

        print(
            f"Backend: "
            f"{metrics.backend}"
        )

    finally:

        capture.stop()


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    "FramePacket",
    "CaptureMetrics",
    "CaptureBackendError",
    "ScreenCapture",
    "create_screen_capture",
]


if __name__ == "__main__":
    _self_test()