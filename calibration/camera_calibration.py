"""
Forza Assistents
Ferramenta de calibração visual da ROI.

A calibração:
    - captura a tela inteira;
    - permite selecionar/mover/redimensionar o ROI;
    - utiliza exclusivamente o config.py como fonte de configuração;
    - salva o ROI através do config.py;
    - não contém parâmetros fixos de resolução.

Execução:

    python -m calibration.camera_calibration

Controles:

    W / ↑       mover para cima
    S / ↓       mover para baixo
    A / ←       mover para esquerda
    D / →       mover para direita

    Q           diminuir largura
    E           aumentar largura

    R           aumentar altura
    F           diminuir altura

    + / =       movimento fino
    -           movimento normal

    ENTER       salvar ROI
    ESC         sair sem salvar
"""

from __future__ import annotations

import logging

import cv2

from config import (
    CALIBRATION_FILE,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    ROI_LEFT,
    ROI_TOP,
    ROI_RIGHT,
    ROI_BOTTOM,
    save_calibration,
)

from capture.screen_capture import ScreenCapture


# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("forza_assistents.calibration")


# ============================================================================
# CONSTANTS
# ============================================================================

WINDOW_NAME = "Forza Assistents - ROI Calibration"

MIN_ROI_WIDTH = 160
MIN_ROI_HEIGHT = 100

NORMAL_STEP = 10
FINE_STEP = 1

CAPTURE_FPS = 30


# ============================================================================
# HELPERS
# ============================================================================

def clamp_roi(
    left: int,
    top: int,
    right: int,
    bottom: int,
) -> tuple[int, int, int, int]:

    width = max(MIN_ROI_WIDTH, right - left)
    height = max(MIN_ROI_HEIGHT, bottom - top)

    width = min(width, SCREEN_WIDTH)
    height = min(height, SCREEN_HEIGHT)

    left = max(0, min(left, SCREEN_WIDTH - width))
    top = max(0, min(top, SCREEN_HEIGHT - height))

    right = left + width
    bottom = top + height

    return left, top, right, bottom


def draw_text(
    image,
    text: str,
    position: tuple[int, int],
    scale: float = 0.6,
    thickness: int = 2,
) -> None:

    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 255, 0),
        thickness,
        cv2.LINE_AA,
    )


def draw_overlay(
    frame,
    left: int,
    top: int,
    right: int,
    bottom: int,
    step: int,
):
    display = frame.copy()

    # ------------------------------------------------------------------
    # ROI
    # ------------------------------------------------------------------

    cv2.rectangle(
        display,
        (left, top),
        (right, bottom),
        (0, 255, 0),
        3,
    )

    # ------------------------------------------------------------------
    # Centro do ROI
    # ------------------------------------------------------------------

    center_x = (left + right) // 2
    center_y = (top + bottom) // 2

    cv2.line(
        display,
        (center_x, top),
        (center_x, bottom),
        (0, 255, 255),
        1,
    )

    cv2.line(
        display,
        (left, center_y),
        (right, center_y),
        (0, 255, 255),
        1,
    )

    # ------------------------------------------------------------------
    # Informações
    # ------------------------------------------------------------------

    roi_width = right - left
    roi_height = bottom - top

    texts = [
        "FORZA ASSISTENTS - ROI CALIBRATION",
        (
            f"ROI: "
            f"({left}, {top}) -> ({right}, {bottom})"
        ),
        f"SIZE: {roi_width} x {roi_height}",
        f"SCREEN: {SCREEN_WIDTH} x {SCREEN_HEIGHT}",
        f"STEP: {step}px",
        "",
        "W/S/A/D or ARROWS : MOVE",
        "Q/E               : WIDTH",
        "R/F               : HEIGHT",
        "+/-               : FINE/NORMAL",
        "ENTER             : SAVE",
        "ESC               : EXIT",
    ]

    y = 30

    for index, text in enumerate(texts):

        if not text:
            y += 10
            continue

        scale = 0.65 if index == 0 else 0.55

        draw_text(
            display,
            text,
            (15, y),
            scale=scale,
            thickness=2,
        )

        y += 27

    return display


# ============================================================================
# CALIBRATION
# ============================================================================

def main() -> None:

    logger.info("=" * 60)
    logger.info("FORZA ASSISTENTS - ROI CALIBRATION")
    logger.info("=" * 60)

    logger.info(
        "Screen: %dx%d",
        SCREEN_WIDTH,
        SCREEN_HEIGHT,
    )

    logger.info(
        "Initial ROI: (%d, %d) -> (%d, %d)",
        ROI_LEFT,
        ROI_TOP,
        ROI_RIGHT,
        ROI_BOTTOM,
    )

    # ------------------------------------------------------------------
    # ROI inicial vindo EXCLUSIVAMENTE do config.py
    # ------------------------------------------------------------------

    left = ROI_LEFT
    top = ROI_TOP
    right = ROI_RIGHT
    bottom = ROI_BOTTOM

    left, top, right, bottom = clamp_roi(
        left,
        top,
        right,
        bottom,
    )

    # ------------------------------------------------------------------
    # Captura da tela inteira
    # ------------------------------------------------------------------

    capture = ScreenCapture(
        region=None,
        target_fps=CAPTURE_FPS,
        backend="dxgi",
        output_color="BGR",
    )

    if not capture.initialize():

        logger.error(
            "Failed to initialize screen capture."
        )

        return

    capture.start()

    logger.info(
        "Screen capture started."
    )

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL,
    )

    cv2.resizeWindow(
        WINDOW_NAME,
        min(SCREEN_WIDTH, 1600),
        min(SCREEN_HEIGHT, 900),
    )

    step = NORMAL_STEP

    saved = False

    try:

        while True:

            frame = capture.get_latest_frame()

            if frame is None:

                cv2.waitKey(1)
                continue

            # ----------------------------------------------------------
            # Garante que o ROI nunca saia da tela
            # ----------------------------------------------------------

            left, top, right, bottom = clamp_roi(
                left,
                top,
                right,
                bottom,
            )

            # ----------------------------------------------------------
            # Desenha interface
            # ----------------------------------------------------------

            display = draw_overlay(
                frame,
                left,
                top,
                right,
                bottom,
                step,
            )

            cv2.imshow(
                WINDOW_NAME,
                display,
            )

            key = cv2.waitKey(1) & 0xFF

            # ==========================================================
            # EXIT
            # ==========================================================

            if key == 27:
                logger.info(
                    "Calibration cancelled."
                )
                break

            # ==========================================================
            # SAVE
            # ==========================================================

            if key in (13, 10):

                data = {
                    "left": int(left),
                    "top": int(top),
                    "right": int(right),
                    "bottom": int(bottom),
                }

                save_calibration(data)

                saved = True

                logger.info(
                    "ROI saved: %s",
                    data,
                )

                logger.info(
                    "Calibration file: %s",
                    CALIBRATION_FILE,
                )

                break

            # ==========================================================
            # STEP
            # ==========================================================

            if key in (ord("+"), ord("=")):

                step = FINE_STEP

            elif key in (ord("-"), ord("_")):

                step = NORMAL_STEP

            # ==========================================================
            # MOVE UP
            # ==========================================================

            elif key in (ord("w"), ord("W"), 82):

                top -= step
                bottom -= step

            # ==========================================================
            # MOVE DOWN
            # ==========================================================

            elif key in (ord("s"), ord("S"), 84):

                top += step
                bottom += step

            # ==========================================================
            # MOVE LEFT
            # ==========================================================

            elif key in (ord("a"), ord("A"), 81):

                left -= step
                right -= step

            # ==========================================================
            # MOVE RIGHT
            # ==========================================================

            elif key in (ord("d"), ord("D"), 83):

                left += step
                right += step

            # ==========================================================
            # WIDTH
            # ==========================================================

            elif key in (ord("q"), ord("Q")):

                right -= step

            elif key in (ord("e"), ord("E")):

                right += step

            # ==========================================================
            # HEIGHT
            # ==========================================================

            elif key in (ord("r"), ord("R")):

                bottom += step

            elif key in (ord("f"), ord("F")):

                bottom -= step

    except KeyboardInterrupt:

        logger.info(
            "Calibration interrupted."
        )

    finally:

        capture.stop()

        cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    # Result
    # ------------------------------------------------------------------

    if saved:

        logger.info("=" * 60)
        logger.info("ROI CALIBRATION SAVED")
        logger.info(
            "ROI = (%d, %d, %d, %d)",
            left,
            top,
            right,
            bottom,
        )
        logger.info("=" * 60)

    else:

        logger.info(
            "ROI calibration finished without changes."
        )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()