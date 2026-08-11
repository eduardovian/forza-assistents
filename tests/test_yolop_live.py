
"""
tests/test_yolop_live.py

Teste de integração:

ScreenCapture -> YOLOP -> LaneDetectionResult

Execução recomendada:

    python -m tests.test_yolop_live

Não altera nenhum módulo do sistema.
"""

from __future__ import annotations

import logging
import time

from capture.screen_capture import ScreenCapture
from vision.yolop_detector import create_default_detector


# ----------------------------------------------------------------------
# CONFIGURAÇÃO
# ----------------------------------------------------------------------

NUM_FRAMES = 30
TARGET_FPS = 60
BACKEND = "dxgi"


# ----------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


# ----------------------------------------------------------------------
# TESTE
# ----------------------------------------------------------------------

def main() -> int:
    print()
    print("=" * 70)
    print("YOLOP LIVE INTEGRATION TEST")
    print("=" * 70)
    print()

    # --------------------------------------------------------------
    # 1. CAPTURA
    # --------------------------------------------------------------

    print("Inicializando captura...")

    capture = ScreenCapture(
        target_fps=TARGET_FPS,
        backend=BACKEND,
    )

    if not capture.initialize():
        print("ERRO: não foi possível inicializar a captura.")
        return 1

    print("Captura: OK")
    print()

    # --------------------------------------------------------------
    # 2. YOLOP
    # --------------------------------------------------------------

    print("Criando detector YOLOP...")

    try:
        detector = create_default_detector()
    except Exception as exc:
        print(f"ERRO ao criar detector YOLOP: {exc}")
        capture.stop()
        return 1

    print("YOLOP: OK")

    provider = getattr(detector, "provider", None)

    if provider is None:
        provider = getattr(detector, "_provider", "desconhecido")

    print(f"Provider: {provider}")
    print()

    # --------------------------------------------------------------
    # 3. INICIAR CAPTURA
    # --------------------------------------------------------------

    print("Iniciando captura...")

    capture.start()

    # Dá tempo para o buffer receber frames.
    time.sleep(0.2)

    print("Processando frames...")
    print()

    # --------------------------------------------------------------
    # 4. ESTATÍSTICAS
    # --------------------------------------------------------------

    requested = NUM_FRAMES
    processed = 0

    valid_count = 0
    detected_count = 0
    two_lane_count = 0

    inference_times = []
    pipeline_times = []

    last_result = None

    # --------------------------------------------------------------
    # 5. LOOP
    # --------------------------------------------------------------

    try:
        for index in range(1, NUM_FRAMES + 1):

            pipeline_start = time.perf_counter()

            frame = capture.get_latest_frame()

            if frame is None:
                print(
                    f"[{index:02d}/{NUM_FRAMES}] "
                    "FRAME: NONE"
                )

                time.sleep(0.005)
                continue

            infer_start = time.perf_counter()

            try:
                result = detector.detect(frame)
            except Exception as exc:
                infer_end = time.perf_counter()

                infer_ms = (
                    infer_end - infer_start
                ) * 1000.0

                print(
                    f"[{index:02d}/{NUM_FRAMES}] "
                    f"DETECT ERROR | "
                    f"infer={infer_ms:.1f}ms | "
                    f"{exc}"
                )

                continue

            infer_end = time.perf_counter()

            infer_ms = (
                infer_end - infer_start
            ) * 1000.0

            pipeline_end = time.perf_counter()

            pipeline_ms = (
                pipeline_end - pipeline_start
            ) * 1000.0

            processed += 1

            inference_times.append(infer_ms)
            pipeline_times.append(pipeline_ms)

            last_result = result

            valid = bool(
                getattr(result, "valid", False)
            )

            lanes = int(
                getattr(
                    result,
                    "num_lanes_detected",
                    0,
                )
            )

            left_lane = getattr(
                result,
                "left_lane",
                [],
            )

            right_lane = getattr(
                result,
                "right_lane",
                [],
            )

            left_count = len(left_lane)
            right_count = len(right_lane)

            if valid:
                valid_count += 1

            if lanes > 0:
                detected_count += 1

            if lanes >= 2:
                two_lane_count += 1

            print(
                f"[{index:02d}/{NUM_FRAMES}] "
                f"frame={frame.shape} | "
                f"valid={valid} | "
                f"lanes={lanes} | "
                f"L={left_count} | "
                f"R={right_count} | "
                f"infer={infer_ms:.1f}ms"
            )

    finally:
        capture.stop()

    # --------------------------------------------------------------
    # 6. RESULTADOS
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("RESULTADO")
    print("=" * 70)

    print()

    print(
        f"Frames solicitados:     {requested}"
    )

    print(
        f"Frames processados:     {processed}"
    )

    if processed > 0:

        valid_pct = (
            valid_count / processed
        ) * 100.0

        detected_pct = (
            detected_count / processed
        ) * 100.0

        two_lane_pct = (
            two_lane_count / processed
        ) * 100.0

        print(
            f"Valid=True:             "
            f"{valid_count}/{processed} "
            f"({valid_pct:.1f}%)"
        )

        print(
            f"Detected=True:          "
            f"{detected_count}/{processed} "
            f"({detected_pct:.1f}%)"
        )

        print(
            f"2 lanes:                "
            f"{two_lane_count}/{processed} "
            f"({two_lane_pct:.1f}%)"
        )

    else:
        print("Nenhum frame foi processado.")

    # --------------------------------------------------------------
    # 7. INFERÊNCIA
    # --------------------------------------------------------------

    if inference_times:

        avg_infer = (
            sum(inference_times)
            / len(inference_times)
        )

        min_infer = min(inference_times)
        max_infer = max(inference_times)

        fps_yolop = (
            1000.0 / avg_infer
            if avg_infer > 0
            else 0.0
        )

        print(
            f"Inferência média:      "
            f"{avg_infer:.2f} ms"
        )

        print(
            f"Inferência mínima:     "
            f"{min_infer:.2f} ms"
        )

        print(
            f"Inferência máxima:     "
            f"{max_infer:.2f} ms"
        )

        print(
            f"FPS YOLOP aproximado:   "
            f"{fps_yolop:.2f}"
        )

    if pipeline_times:

        avg_pipeline = (
            sum(pipeline_times)
            / len(pipeline_times)
        )

        print(
            f"Pipeline médio:        "
            f"{avg_pipeline:.2f} ms"
        )

    # --------------------------------------------------------------
    # 8. CAPTURA
    # --------------------------------------------------------------

    print(
        f"Capturas realizadas:   "
        f"{capture.capture_count}"
    )

    print(
        f"Frames descartados:     "
        f"{capture.dropped_frames}"
    )

    print(
        f"Última latência captura:"
        f" {capture.capture_latency_ms:.2f} ms"
    )

    # --------------------------------------------------------------
    # 9. ÚLTIMOS PONTOS
    # --------------------------------------------------------------

    if last_result is not None:

        left_lane = getattr(
            last_result,
            "left_lane",
            [],
        )

        right_lane = getattr(
            last_result,
            "right_lane",
            [],
        )

        print()
        print("Últimos pontos LEFT:")

        if left_lane:

            for point in left_lane[-5:]:
                x = getattr(point, "x", None)
                y = getattr(point, "y", None)
                conf = getattr(
                    point,
                    "confidence",
                    getattr(point, "conf", 1.0),
                )

                if x is not None and y is not None:
                    print(
                        f"x={x:.1f}, "
                        f"y={y:.1f}, "
                        f"conf={conf:.3f}"
                    )

        else:
            print("(nenhum)")

        print()
        print("Últimos pontos RIGHT:")

        if right_lane:

            for point in right_lane[-5:]:
                x = getattr(point, "x", None)
                y = getattr(point, "y", None)
                conf = getattr(
                    point,
                    "confidence",
                    getattr(point, "conf", 1.0),
                )

                if x is not None and y is not None:
                    print(
                        f"x={x:.1f}, "
                        f"y={y:.1f}, "
                        f"conf={conf:.3f}"
                    )

        else:
            print("(nenhum)")

    print()
    print("=" * 70)
    print("TESTE FINALIZADO")
    print("=" * 70)
    print()

    return 0


# ----------------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------------

if __name__ == "__main__":
    raise SystemExit(main())

