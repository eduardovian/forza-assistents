
"""
Forza Horizon 6 ADAS/LKA - Captura de Tela de Baixa Latência

Backends suportados:

1. dxcam (DXGI Desktop Duplication) - recomendado
2. mss (fallback)
3. OpenCV VideoCapture - modo vídeo/arquivo

Para jogos fullscreen, o dxcam captura a tela inteira e o ROI,
quando configurado, é aplicado posteriormente via NumPy.
"""

import logging
import time
import threading
from typing import Optional, Tuple
from collections import deque

import cv2
import numpy as np


logger = logging.getLogger(__name__)


class ScreenCapture:
    """
    Captura de tela de baixa latência com suporte a:

    - DXGI Desktop Duplication via dxcam
    - MSS
    - OpenCV VideoCapture

    O consumidor sempre recebe o frame mais recente disponível.
    """

    def __init__(
        self,
        region: Optional[Tuple[int, int, int, int]] = None,
        target_fps: int = 60,
        backend: str = "dxgi",
        output_color: str = "BGR",
        video_path: Optional[str] = None,
        max_buffer_size: int = 2,
    ):
        self.region = region
        self.target_fps = target_fps
        self.backend_name = backend
        self.output_color = output_color
        self.video_path = video_path
        self.max_buffer_size = max(1, max_buffer_size)

        self._camera = None
        self._cap = None

        self._running = False
        self._capture_thread = None

        self._frame_buffer = deque(maxlen=self.max_buffer_size)
        self._lock = threading.Lock()

        # Último frame válido.
        # Mantemos este frame para evitar FRAME: NONE quando
        # o produtor e o consumidor estão em velocidades diferentes.
        self._last_frame = None

        self._last_capture_time = 0.0

        # Estatísticas
        self.capture_count = 0
        self.capture_latency_ms = 0.0
        self.dropped_frames = 0

        # Identifica explicitamente o backend ativo.
        self._active_backend = None

    def initialize(self) -> bool:
        """Inicializa o backend de captura."""

        if self.video_path is not None:
            return self._init_video()

        # ---------------------------------------------------------
        # 1. DXCAM
        # ---------------------------------------------------------
        if self.backend_name in ("dxgi", "winrt"):
            try:
                import dxcam

                self._camera = dxcam.create(
                    backend=self.backend_name,
                    output_color=self.output_color,
                )

                self._active_backend = "dxcam"

                logger.info(
                    f"[CAPTURE] dxcam inicializado ({self.backend_name})"
                )

                width = getattr(self._camera, "width", None)
                height = getattr(self._camera, "height", None)

                if width is not None and height is not None:
                    logger.info(
                        f"[CAPTURE] Resolução do monitor: "
                        f"{width}x{height}"
                    )

                return True

            except Exception as e:
                logger.warning(f"[CAPTURE] dxcam falhou: {e}")
                self._camera = None
                self._active_backend = None

        # ---------------------------------------------------------
        # 2. MSS
        # ---------------------------------------------------------
        try:
            import mss

            self._camera = mss.mss()
            self._active_backend = "mss"

            logger.info("[CAPTURE] mss inicializado (fallback)")

            return True

        except Exception as e:
            logger.warning(f"[CAPTURE] mss falhou: {e}")

        logger.error("[CAPTURE] Nenhum backend disponível")
        return False

    def _init_video(self) -> bool:
        """Inicializa captura de vídeo via OpenCV."""

        self._cap = cv2.VideoCapture(self.video_path)

        if not self._cap.isOpened():
            logger.error(
                f"[CAPTURE] Não foi possível abrir vídeo: "
                f"{self.video_path}"
            )
            return False

        self._active_backend = "video"

        logger.info(
            f"[CAPTURE] Vídeo aberto: {self.video_path}"
        )

        return True

    def start(self):
        """Inicia captura contínua em thread separada."""

        if self._running:
            return

        if self._cap is None and self._camera is None:
            logger.error(
                "[CAPTURE] start() chamado antes de initialize()"
            )
            return

        self._running = True

        if self._active_backend == "video":
            target = self._video_loop
        else:
            target = self._capture_loop

        self._capture_thread = threading.Thread(
            target=target,
            name="ScreenCapture",
            daemon=True,
        )

        self._capture_thread.start()

        logger.info("[CAPTURE] Thread de captura iniciada")

    def _capture_loop(self):
        """
        Loop principal para dxcam/mss.

        Para dxcam usamos exclusivamente:

            start() -> get_latest_frame()

        Não misturamos get_latest_frame() com grab(), pois isso
        pode causar comportamento inconsistente no Desktop Duplication.
        """

        # ---------------------------------------------------------
        # DXCAM
        # ---------------------------------------------------------
        if self._active_backend == "dxcam":

            try:
                self._camera.start(
                    target_fps=self.target_fps,
                    region=None,
                )

                logger.info(
                    f"[CAPTURE] dxcam streaming iniciado "
                    f"({self.target_fps} FPS)"
                )

            except Exception as e:
                logger.error(
                    f"[CAPTURE] Não foi possível iniciar dxcam: {e}"
                )
                self._running = False
                return

            while self._running:
                t0 = time.perf_counter()

                try:
                    frame = self._camera.get_latest_frame()
                except Exception as e:
                    logger.debug(
                        f"[CAPTURE] get_latest_frame falhou: {e}"
                    )
                    frame = None

                t1 = time.perf_counter()

                if frame is not None:
                    frame = self._apply_region(frame)

                    if frame is not None:
                        self._push_frame(
                            frame,
                            (t1 - t0) * 1000.0,
                        )
                else:
                    # Não há frame novo neste instante.
                    # Não consideramos isso como erro.
                    time.sleep(0.001)

            try:
                self._camera.stop()
            except Exception:
                pass

            return

        # ---------------------------------------------------------
        # MSS
        # ---------------------------------------------------------
        if self._active_backend == "mss":

            interval = 1.0 / max(1, self.target_fps)

            while self._running:
                t0 = time.perf_counter()

                frame = self._grab_mss_frame()

                t1 = time.perf_counter()

                if frame is not None:
                    self._push_frame(
                        frame,
                        (t1 - t0) * 1000.0,
                    )

                elapsed = t1 - t0
                sleep_time = max(0.0, interval - elapsed)

                if sleep_time > 0:
                    time.sleep(sleep_time)

    def _grab_mss_frame(self) -> Optional[np.ndarray]:
        """Captura um frame usando MSS."""

        if self._camera is None:
            return None

        try:
            if self.region is not None:
                left, top, right, bottom = self.region

                width = max(1, right - left)
                height = max(1, bottom - top)

                monitor = {
                    "left": max(0, left),
                    "top": max(0, top),
                    "width": width,
                    "height": height,
                }
            else:
                monitors = self._camera.monitors

                if len(monitors) > 1:
                    monitor = monitors[1]
                else:
                    monitor = monitors[0]

            screenshot = self._camera.grab(monitor)

            if screenshot is None:
                return None

            frame = np.asarray(screenshot)

            if frame.ndim != 3:
                return None

            if frame.shape[2] == 4:
                frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGRA2BGR,
                )

            return frame

        except Exception as e:
            logger.debug(
                f"[CAPTURE] MSS grab falhou: {e}"
            )
            return None

    def _video_loop(self):
        """Loop de leitura contínua de vídeo via OpenCV."""

        interval = 1.0 / max(1, self.target_fps)

        while self._running:

            t0 = time.perf_counter()

            ret, frame = self._cap.read()

            t1 = time.perf_counter()

            if not ret:
                self._cap.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    0,
                )

                ret, frame = self._cap.read()

                if not ret:
                    time.sleep(0.01)
                    continue

            if self.region is not None:
                frame = self._apply_region(frame)

            if frame is not None:
                self._push_frame(
                    frame,
                    (t1 - t0) * 1000.0,
                )

            elapsed = t1 - t0
            sleep_time = max(0.0, interval - elapsed)

            if sleep_time > 0:
                time.sleep(sleep_time)

    def _apply_region(
        self,
        frame: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Aplica ROI ao frame."""

        if frame is None:
            return None

        if self.region is None:
            return frame

        try:
            left, top, right, bottom = self.region

            h, w = frame.shape[:2]

            left = max(0, min(left, w))
            right = max(0, min(right, w))

            top = max(0, min(top, h))
            bottom = max(0, min(bottom, h))

            if right <= left or bottom <= top:
                logger.warning(
                    "[CAPTURE] ROI inválido: "
                    f"{self.region} para frame {w}x{h}"
                )
                return None

            return frame[top:bottom, left:right]

        except Exception as e:
            logger.debug(
                f"[CAPTURE] Aplicação de ROI falhou: {e}"
            )
            return None

    def _push_frame(
        self,
        frame: np.ndarray,
        latency_ms: float,
    ):
        """Adiciona um frame ao buffer e atualiza estatísticas."""

        if frame is None:
            return

        with self._lock:

            if len(self._frame_buffer) >= self.max_buffer_size:
                self._frame_buffer.popleft()
                self.dropped_frames += 1

            self._frame_buffer.append(frame)

            # Sempre mantém referência ao frame mais recente.
            self._last_frame = frame

        self.capture_count += 1
        self.capture_latency_ms = latency_ms
        self._last_capture_time = time.perf_counter()

    def get_latest_frame(
        self,
    ) -> Optional[np.ndarray]:
        """
        Retorna o frame mais recente.

        Diferente da implementação anterior, não limpa o buffer
        e mantém o último frame válido disponível.
        """

        with self._lock:

            if self._frame_buffer:
                frame = self._frame_buffer[-1]
                self._last_frame = frame
                return frame

            if self._last_frame is not None:
                return self._last_frame

        return None

    def get_frame_copy(
        self,
    ) -> Optional[np.ndarray]:
        """Retorna uma cópia do frame mais recente."""

        frame = self.get_latest_frame()

        if frame is not None:
            return frame.copy()

        return None

    def stop(self):
        """Para a captura e libera recursos."""

        self._running = False

        if self._capture_thread is not None:
            self._capture_thread.join(timeout=2.0)
            self._capture_thread = None

        if self._active_backend == "dxcam":
            if self._camera is not None:
                try:
                    self._camera.stop()
                except Exception:
                    pass

        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass

            self._cap = None

        if self._camera is not None:
            if self._active_backend == "mss":
                try:
                    self._camera.close()
                except Exception:
                    pass

            elif hasattr(self._camera, "release"):
                try:
                    self._camera.release()
                except Exception:
                    pass

            self._camera = None

        with self._lock:
            self._frame_buffer.clear()
            self._last_frame = None

        self._active_backend = None

        logger.info("[CAPTURE] Recursos liberados")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def frame_shape(
        self,
    ) -> Optional[Tuple[int, ...]]:
        """Retorna a forma do frame mais recente."""

        with self._lock:

            if self._frame_buffer:
                return self._frame_buffer[-1].shape

            if self._last_frame is not None:
                return self._last_frame.shape

        return None

