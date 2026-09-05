"""
main.py

Forza Assistents
================

Primeiro protótipo de percepção:

    Captura da tela
          ↓
    Seleção manual da ROI trapezoidal
          ↓
    Perspective Transform
          ↓
    Visualização da ROI
          ↓
    Medição de FPS

Ainda não existe:
- detecção de faixas;
- tracking;
- controle;
- G29.
"""

from __future__ import annotations

import time

import cv2

from calibration.roi_selector import ROISelector, TrapezoidROI
from capture.screen_capture import ScreenCapture


# ================================================================
# Configuração
# ================================================================

TARGET_FPS = 60

WINDOW_NAME = "Forza Assistents - Perception"

PERSPECTIVE_WIDTH = 1000
PERSPECTIVE_HEIGHT = 600


# ================================================================
# Funções auxiliares
# ================================================================

def crop_roi(
    frame,
    roi: TrapezoidROI,
):
    """
    Aplica a ROI trapezoidal diretamente sobre o frame.

    Pixels fora do trapézio ficam pretos.
    """

    mask = frame.copy()
    mask[:] = 0

    points = roi.as_array().astype("int32")

    cv2.fillPoly(
        mask,
        [points],
        (255, 255, 255),
    )

    return cv2.bitwise_and(
        frame,
        mask,
    )


def draw_roi_overlay(
    frame,
    roi: TrapezoidROI,
):
    """
    Desenha o trapézio e seus quatro pontos sobre o frame.
    """

    output = frame.copy()

    points = roi.as_array().astype("int32")

    # Contorno.
    cv2.polylines(
        output,
        [points],
        True,
        (0, 255, 0),
        3,
        cv2.LINE_AA,
    )

    # Pontos.
    labels = [
        "P0",
        "P1",
        "P2",
        "P3",
    ]

    for index, point in enumerate(points):

        x, y = point

        cv2.circle(
            output,
            (x, y),
            12,
            (0, 255, 255),
            -1,
        )

        cv2.circle(
            output,
            (x, y),
            14,
            (0, 0, 0),
            2,
        )

        cv2.putText(
            output,
            labels[index],
            (x + 15, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return output


# ================================================================
# Main
# ================================================================

def main() -> None:

    print("=" * 64)
    print("FORZA ASSISTENTS")
    print("Perception Prototype")
    print("=" * 64)

    capture = ScreenCapture(
        target_fps=TARGET_FPS,
    )

    try:

        # ----------------------------------------------------------
        # Inicialização da captura
        # ----------------------------------------------------------

        print("\nIniciando captura da tela...")

        capture.start()

        frame = None

        while frame is None:

            frame = capture.read()

            time.sleep(0.01)

        frame_height, frame_width = frame.shape[:2]

        print(
            f"Resolução detectada: "
            f"{frame_width}x{frame_height}"
        )

        # ----------------------------------------------------------
        # Seleção da ROI
        # ----------------------------------------------------------

        print("\nAbrindo seletor de ROI...")

        print(
            "\nControles:"
            "\n  Arraste os pontos P0-P3"
            "\n  Arraste dentro da ROI para mover"
            "\n  Clique fora para criar outra"
            "\n  ENTER/S = confirmar"
            "\n  R = resetar"
            "\n  ESC = cancelar"
        )

        selector = ROISelector(
            window_width=1400,
            window_height=900,
            perspective_width=PERSPECTIVE_WIDTH,
            perspective_height=PERSPECTIVE_HEIGHT,
        )

        roi = selector.select(frame)

        if roi is None:

            print("\nSeleção cancelada.")

            return

        # ----------------------------------------------------------
        # Exibe coordenadas
        # ----------------------------------------------------------

        print("\nROI selecionada:")

        for index, point in enumerate(
            roi.points()
        ):

            print(
                f"  P{index}: "
                f"({point.x}, {point.y})"
            )

        # ----------------------------------------------------------
        # Janela principal
        # ----------------------------------------------------------

        cv2.namedWindow(
            WINDOW_NAME,
            cv2.WINDOW_NORMAL,
        )

        cv2.resizeWindow(
            WINDOW_NAME,
            1400,
            900,
        )

        print(
            "\nPercepção iniciada."
            "\nPressione ESC para sair."
        )

        # ----------------------------------------------------------
        # FPS
        # ----------------------------------------------------------

        last_time = time.perf_counter()

        fps = 0.0

        # ----------------------------------------------------------
        # Loop
        # ----------------------------------------------------------

        while True:

            frame = capture.read()

            if frame is None:
                continue

            # ------------------------------------------------------
            # FPS
            # ------------------------------------------------------

            current_time = time.perf_counter()

            delta = current_time - last_time

            if delta > 0:

                instant_fps = 1.0 / delta

                fps = (
                    fps * 0.9
                    + instant_fps * 0.1
                )

            last_time = current_time

            # ------------------------------------------------------
            # ROI trapezoidal
            # ------------------------------------------------------

            roi_frame = crop_roi(
                frame,
                roi,
            )

            # ------------------------------------------------------
            # Perspective Transform
            # ------------------------------------------------------

            perspective = selector.perspective_transform(
                frame,
                roi,
            )

            # ------------------------------------------------------
            # Overlay no frame original
            # ------------------------------------------------------

            original_view = draw_roi_overlay(
                frame,
                roi,
            )

            cv2.putText(
                original_view,
                f"FPS: {fps:.1f}",
                (25, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                original_view,
                "ROI trapezoidal",
                (25, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # ------------------------------------------------------
            # Overlay na perspectiva
            # ------------------------------------------------------

            perspective_view = perspective.copy()

            cv2.putText(
                perspective_view,
                "Perspective Transform",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # ------------------------------------------------------
            # Mostrar
            # ------------------------------------------------------

            # Redimensionamos para visualizar as duas imagens
            # simultaneamente.
            original_display = cv2.resize(
                original_view,
                (900, 562),
                interpolation=cv2.INTER_AREA,
            )

            perspective_display = cv2.resize(
                perspective_view,
                (500, 300),
                interpolation=cv2.INTER_AREA,
            )

            # ROI isolada.
            roi_display = cv2.resize(
                roi_frame,
                (500, 300),
                interpolation=cv2.INTER_AREA,
            )

            # Canvas final.
            canvas = (
                __import__("numpy").zeros(
                    (600, 900, 3),
                    dtype="uint8",
                )
            )

            # Frame original.
            canvas[
                0:562,
                0:900,
            ] = original_display

            # Perspective.
            canvas[
                562:600,
                0:500,
            ] = perspective_display[
                0:38,
                0:500,
            ]

            # ------------------------------------------------------
            # Exibição principal
            # ------------------------------------------------------

            cv2.imshow(
                WINDOW_NAME,
                canvas,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break

    finally:

        capture.stop()

        cv2.destroyAllWindows()

        print("\nCaptura encerrada.")


# ================================================================
# Entry point
# ================================================================

if __name__ == "__main__":
    main()