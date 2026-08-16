"""
capture/screen_capture.py

Captura de tela do Forza Assistents.

Responsabilidades:
- capturar a tela;
- aplicar exclusivamente a região recebida pelo chamador;
- manter captura assíncrona;
- fornecer o frame mais recente;
- não possuir configuração própria de ROI;
- não conter lógica de visão.

O ROI deve vir exclusivamente do config.py.

Fluxo:

    config.py
        ↓
    main.py
        ↓
    ScreenCapture(region=...)
        ↓
    frame
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional, Tuple

import numpy as np

try:
    import dxcam
except ImportError:
    dxcam = None


LOGGER = logging.getLogger("forza_assistents.capture")


# ============================================================================
# TYPES
# ============================================================================

ROI = Tuple[int, int, int, int]


# ============================================================================
# SCREEN CAPTURE
# ============================================================================

class ScreenCapture:
    """
    Capturador assíncrono de tela.

    O ROI é recebido externamente.

    Não existe ROI padrão neste módulo.

    Parameters
    ----------
    region:
        Região da tela no formato:

            (left, top, right, bottom)

        None significa tela inteira.

    target_fps:
        Frequência desejada da captura.

    backend:
        Backend de captura. Atualmente DXCam.

    output_color:
        Formato de saída. DXCam fornece BGR.

    max_buffer_size:
        Número máximo de frames mantidos internamente.
    """

    def __init__(
        self,
        *,
        region: Optional[ROI] = None,
        target_fps: int = 60,
        backend: str = "dxcam",
        output_color: str = "BGR",
        max_buffer_size: int = 2,
    ) -> None:

        self.region = self._normalize_region(region)

        self.target_fps = max(
            1,
            int(target_fps),
        )

        self.backend = backend.lower().strip()

        self.output_color = (
            output_color.upper().strip()
        )

        self.max_buffer_size = max(
            1,
            int(max_buffer_size),
        )

        self._camera = None

        self._running = False
        self._initialized = False

        self._thread: Optional[
            threading.Thread
        ] = None

        self._lock = threading.Lock()

        self._latest_frame: Optional[
            np.ndarray
        ] = None

        self._frame_counter = 0

        self._last_capture_time = 0.0

        self._capture_fps = 0.0

        self._last_error: Optional[str] = None

    # ========================================================================
    # ROI
    # ========================================================================

    @staticmethod
    def _normalize_region(
        region: Optional[ROI],
    ) -> Optional[ROI]:

        if region is None:
            return None

        if len(region) != 4:
            raise ValueError(
                "ROI must contain exactly "
                "(left, top, right, bottom)."
            )

        left, top, right, bottom = map(
            int,
            region,
        )

        if left < 0 or top < 0:
            raise ValueError(
                f"ROI coordinates cannot be negative: "
                f"{region}"
            )

        if right <= left:
            raise ValueError(
                f"ROI right must be greater than left: "
                f"{region}"
            )

        if bottom <= top:
            raise ValueError(
                f"ROI bottom must be greater than top: "
                f"{region}"
            )

        return (
            left,
            top,
            right,
            bottom,
        )

    # ========================================================================
    # INITIALIZE
    # ========================================================================

    def initialize(self) -> bool:
        """
        Inicializa o backend de captura.

        Returns
        -------
        bool
            True quando inicializado corretamente.
        """

        if self._initialized:
            return True

        if self.backend != "dxcam":
            self._last_error = (
                f"Unsupported capture backend: "
                f"{self.backend}"
            )

            LOGGER.error(
                self._last_error
            )

            return False

        if dxcam is None:
            self._last_error = (
                "DXCam is not installed."
            )

            LOGGER.error(
                self._last_error
            )

            return False

        try:

            self._camera = dxcam.create(
                output_color="BGR",
            )

            if self._camera is None:
                raise RuntimeError(
                    "dxcam.create() returned None."
                )

            self._initialized = True

            if self.region is None:

                LOGGER.info(
                    "ScreenCapture: READY | "
                    "ROI=FULL SCREEN | "
                    "FPS=%d",
                    self.target_fps,
                )

            else:

                left, top, right, bottom = (
                    self.region
                )

                LOGGER.info(
                    "ScreenCapture: READY | "
                    "ROI=(%d,%d,%d,%d) | "
                    "SIZE=%dx%d | FPS=%d",
                    left,
                    top,
                    right,
                    bottom,
                    right - left,
                    bottom - top,
                    self.target_fps,
                )

            return True

        except Exception as exc:

            self._last_error = str(exc)

            LOGGER.exception(
                "Failed to initialize ScreenCapture."
            )

            self._camera = None
            self._initialized = False

            return False

    # ========================================================================
    # START
    # ========================================================================

    def start(self) -> bool:
        """
        Inicia a captura assíncrona.
        """

        if not self._initialized:

            if not self.initialize():
                return False

        if self._running:
            return True

        if self._camera is None:
            self._last_error = (
                "Capture backend is not initialized."
            )
            return False

        try:

            self._running = True

            self._thread = threading.Thread(
                target=self._capture_loop,
                name="ForzaAssistents-Capture",
                daemon=True,
            )

            self._thread.start()

            LOGGER.info(
                "ScreenCapture: STARTED"
            )

            return True

        except Exception as exc:

            self._last_error = str(exc)

            self._running = False

            LOGGER.exception(
                "Failed to start ScreenCapture."
            )

            return False

    # ========================================================================
    # CAPTURE LOOP
    # ========================================================================

    def _capture_loop(self) -> None:
        """
        Loop interno de captura.

        O DXCam recebe diretamente a região configurada.
        """

        if self._camera is None:
            return

        frame_interval = (
            1.0 / self.target_fps
        )

        last_time = time.perf_counter()

        try:

            self._camera.start(
                region=self.region,
                target_fps=self.target_fps,
                video_mode=True,
            )

            while self._running:

                frame = (
                    self._camera.get_latest_frame()
                )

                if frame is None:

                    time.sleep(0.001)
                    continue

                now = time.perf_counter()

                elapsed = (
                    now - last_time
                )

                last_time = now

                if elapsed > 0.0:

                    instant_fps = (
                        1.0 / elapsed
                    )

                    if self._capture_fps <= 0.0:

                        self._capture_fps = (
                            instant_fps
                        )

                    else:

                        self._capture_fps = (
                            self._capture_fps * 0.9
                            + instant_fps * 0.1
                        )

                with self._lock:

                    self._latest_frame = frame
                    self._frame_counter += 1

                remaining = (
                    frame_interval
                    - (
                        time.perf_counter()
                        - now
                    )
                )

                if remaining > 0.0:

                    time.sleep(
                        min(
                            remaining,
                            0.002,
                        )
                    )

        except Exception as exc:

            self._last_error = str(exc)

            LOGGER.exception(
                "ScreenCapture loop failed."
            )

        finally:

            try:
                self._camera.stop()
            except Exception:
                pass

    # ========================================================================
    # FRAME
    # ========================================================================

    def get_latest_frame(
        self,
    ) -> Optional[np.ndarray]:
        """
        Retorna o frame mais recente.

        O frame pertence à captura atual.
        """

        with self._lock:

            if self._latest_frame is None:
                return None

            return self._latest_frame

    # ========================================================================
    # COPY FRAME
    # ========================================================================

    def get_latest_frame_copy(
        self,
    ) -> Optional[np.ndarray]:
        """
        Retorna uma cópia independente do frame.
        """

        with self._lock:

            if self._latest_frame is None:
                return None

            return self._latest_frame.copy()

    # ========================================================================
    # STATUS
    # ========================================================================

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def frame_count(self) -> int:
        return self._frame_counter

    @property
    def capture_fps(self) -> float:
        return self._capture_fps

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    @property
    def frame_size(
        self,
    ) -> Optional[Tuple[int, int]]:

        frame = self.get_latest_frame()

        if frame is None:
            return None

        height, width = frame.shape[:2]

        return (
            width,
            height,
        )

    # ========================================================================
    # ROI INFORMATION
    # ========================================================================

    @property
    def roi(self) -> Optional[ROI]:
        return self.region

    @property
    def roi_size(
        self,
    ) -> Optional[Tuple[int, int]]:

        if self.region is None:
            return None

        left, top, right, bottom = (
            self.region
        )

        return (
            right - left,
            bottom - top,
        )

    # ========================================================================
    # STOP
    # ========================================================================

    def stop(self) -> None:
        """
        Para a captura de forma segura.
        """

        if not self._running:
            return

        self._running = False

        if self._thread is not None:

            self._thread.join(
                timeout=2.0
            )

            self._thread = None

        try:

            if self._camera is not None:
                self._camera.stop()

        except Exception:
            pass

        LOGGER.info(
            "ScreenCapture: STOPPED"
        )

    # ========================================================================
    # RELEASE
    # ========================================================================

    def release(self) -> None:
        """
        Libera completamente o backend.
        """

        self.stop()

        self._camera = None
        self._initialized = False

        with self._lock:

            self._latest_frame = None

        LOGGER.info(
            "ScreenCapture: RELEASED"
        )

    # ========================================================================
    # CONTEXT MANAGER
    # ========================================================================

    def __enter__(self) -> "ScreenCapture":

        if not self.initialize():
            raise RuntimeError(
                self._last_error
                or "Failed to initialize capture."
            )

        self.start()

        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:

        self.release()