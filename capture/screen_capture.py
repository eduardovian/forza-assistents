"""
Captura de tela para o Forza Assistents.

Primeira versão:
- Captura fullscreen do monitor principal.
- Usa DXGI via dxcam quando disponível.
- Retorna frames como numpy.ndarray no formato BGR.
"""

from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np

try:
    import dxcam
except ImportError:
    dxcam = None


class ScreenCapture:
    """Capturador de tela de baixa latência."""

    def __init__(self, target_fps: int = 60) -> None:
        if dxcam is None:
            raise ImportError(
                "DXCam não está instalado. "
                "Instale com: pip install dxcam"
            )

        self.target_fps = target_fps
        self.camera = dxcam.create(
            output_color="BGR",
            output_idx=0,
        )

        self._last_time = time.perf_counter()
        self._fps = 0.0

    def start(self) -> None:
        """Inicia a captura contínua."""
        self.camera.start(
            target_fps=self.target_fps,
            video_mode=True,
        )

    def read(self) -> Optional[np.ndarray]:
        """
        Obtém o frame mais recente.

        Returns:
            Frame BGR ou None caso ainda não exista um frame.
        """
        frame = self.camera.get_latest_frame()

        if frame is None:
            return None

        now = time.perf_counter()
        dt = now - self._last_time

        if dt > 0:
            instant_fps = 1.0 / dt
            self._fps = (
                0.9 * self._fps +
                0.1 * instant_fps
            )

        self._last_time = now

        return frame

    @property
    def fps(self) -> float:
        """FPS aproximado da captura."""
        return self._fps

    def stop(self) -> None:
        """Encerra a captura."""
        self.camera.stop()

    def __enter__(self) -> "ScreenCapture":
        self.start()
        return self

    def __exit__(self, *_args) -> None:
        self.stop()
        cv2.destroyAllWindows()