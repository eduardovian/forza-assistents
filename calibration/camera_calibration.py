"""
Forza Horizon 6 ADAS/LKA - Ferramenta de Calibração da ROI

Permite ajustar visualmente a região de interesse (ROI) usando
a captura de tela em tempo real.

Uso:
    python -m calibration.camera_calibration

Controles:
    WASD / Setas: mover ROI
    Q/E: ajustar largura
    R/F: ajustar altura
    +/-: ajustar fine
    S: salvar
    ESC: sair
"""
import sys
import os
import logging

# Adiciona o diretório pai ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from config import (
    SCREEN_WIDTH, SCREEN_HEIGHT,
    ROI_LEFT, ROI_TOP, ROI_RIGHT, ROI_BOTTOM,
    save_calibration, CALIBRATION_FILE
)
from capture.screen_capture import ScreenCapture

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logger.info("[CALIBRATION] Iniciando ferramenta de calibração da ROI")
    logger.info("[CALIBRATION] Controles:")
    logger.info("  WASD / Setas: mover ROI")
    logger.info("  Q/E: ajustar largura")
    logger.info("  R/F: ajustar altura")
    logger.info("  +/-: ajuste fino")
    logger.info("  S: salvar calibração")
    logger.info("  ESC: sair sem salvar")

    # Carrega calibração atual
    left = ROI_LEFT
    top = ROI_TOP
    right = ROI_RIGHT
    bottom = ROI_BOTTOM

    step = 10
    fine_step = 1

    # Inicializa captura
    capture = ScreenCapture(
        region=None,  # captura tela inteira
        target_fps=30,
        backend="dxgi",
        output_color="BGR"
    )

    if not capture.initialize():
        logger.error("[CALIBRATION] Falha ao inicializar captura")
        return

    capture.start()

    window_name = "Forza ADAS - Calibração de ROI"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    running = True
    saved = False

    try:
        while running:
            frame = capture.get_latest_frame()
            if frame is None:
                cv2.waitKey(1)
                continue

            # Desenha ROI no frame
            display = frame.copy()
            cv2.rectangle(display, (left, top), (right, bottom), (0, 255, 0), 3)

            # Info
            info_texts = [
                f"ROI: left={left}, top={top}, right={right}, bottom={bottom}",
                f"Tamanho: {right-left}x{bottom-top}",
                "WASD/Setas: mover | Q/E: largura | R/F: altura",
                "+/-: fine | S: salvar | ESC: sair"
            ]
            for i, text in enumerate(info_texts):
                cv2.putText(
                    display, text,
                    (10, 30 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2
                )

            # Preview da ROI
            roi_preview = display[top:bottom, left:right]
            if roi_preview.size > 0:
                preview_h = 200
                preview_w = int(preview_h * (right - left) / (bottom - top))
                roi_preview = cv2.resize(roi_preview, (preview_w, preview_h))
                # Coloca preview no canto superior direito
                ph, pw = roi_preview.shape[:2]
                display[10:10+ph, display.shape[1]-pw-10:display.shape[1]-10] = roi_preview
                cv2.rectangle(
                    display,
                    (display.shape[1]-pw-10, 10),
                    (display.shape[1]-10, 10+ph),
                    (255, 255, 0), 2
                )

            cv2.imshow(window_name, display)

            key = cv2.waitKey(30) & 0xFF

            if key == 27:  # ESC
                running = False
            elif key == ord('s') or key == ord('S'):
                data = {
                    "left": left,
                    "top": top,
                    "right": right,
                    "bottom": bottom
                }
                save_calibration(data)
                saved = True
                logger.info(f"[CALIBRATION] Salvo em {CALIBRATION_FILE}")
            elif key == ord('w') or key == 82:  # W / Seta cima
                top = max(0, top - step)
                bottom = max(top + 100, bottom - step)
            elif key == ord('s') or key == 84:  # S / Seta baixo
                top = min(SCREEN_HEIGHT - 100, top + step)
                bottom = min(SCREEN_HEIGHT, bottom + step)
            elif key == ord('a') or key == 81:  # A / Seta esquerda
                left = max(0, left - step)
                right = max(left + 100, right - step)
            elif key == ord('d') or key == 83:  # D / Seta direita
                left = min(SCREEN_WIDTH - 100, left + step)
                right = min(SCREEN_WIDTH, right + step)
            elif key == ord('q'):  # Q: diminuir largura
                right = max(left + 100, right - step)
            elif key == ord('e'):  # E: aumentar largura
                right = min(SCREEN_WIDTH, right + step)
            elif key == ord('r'):  # R: aumentar altura
                bottom = min(SCREEN_HEIGHT, bottom + step)
            elif key == ord('f'):  # F: diminuir altura
                bottom = max(top + 100, bottom - step)
            elif key == ord('+') or key == ord('='):  # +: fine step
                step = fine_step
            elif key == ord('-') or key == ord('_'):  # -: coarse step
                step = 10

    except KeyboardInterrupt:
        pass
    finally:
        capture.stop()
        cv2.destroyAllWindows()

    if saved:
        logger.info(f"[CALIBRATION] Calibração salva com sucesso em {CALIBRATION_FILE}")
    else:
        logger.info("[CALIBRATION] Saindo sem salvar")


if __name__ == "__main__":
    main()