"""Gera imagens de diagnóstico da leitura YOLOPv2, sem usar captura do jogo.

Uso:
    .\\.venv-yolopv2\\Scripts\\python.exe tests\\visualize_yolopv2.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vision.yolop_detector import YOLOPLaneDetector, create_default_detector


INPUTS = (ROOT / "teste_yolopv2.png", ROOT / "teste_yolopv21.png")
OUTPUT_DIR = ROOT / "visualization" / "yolopv2"
COLORS = (
    (0, 255, 0),
    (0, 210, 255),
    (255, 80, 40),
    (255, 0, 255),
    (0, 255, 255),
    (255, 255, 0),
)


def draw_result(image: np.ndarray, result: object) -> np.ndarray:
    """Sobrepõe área dirigível e pontos de faixa em coordenadas originais."""
    overlay = image.copy()
    drivable = getattr(result, "drivable_area_mask", None)

    if isinstance(drivable, np.ndarray):
        overlay[drivable.astype(bool)] = (40, 130, 40)
        overlay = cv2.addWeighted(image, 0.72, overlay, 0.28, 0.0)

    for lane_index, lane in enumerate(getattr(result, "lanes", ())):
        color = COLORS[lane_index % len(COLORS)]
        points = np.asarray([(round(point.x), round(point.y)) for point in lane])

        if len(points) > 1:
            cv2.polylines(overlay, [points], False, color, 3, cv2.LINE_AA)

        for x, y in points:
            cv2.circle(overlay, (int(x), int(y)), 4, color, -1, cv2.LINE_AA)

    label = f"YOLOPv2 | lanes={result.num_lanes_detected}"
    cv2.putText(overlay, label, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 5, cv2.LINE_AA)
    cv2.putText(overlay, label, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return overlay


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detector = create_default_detector()

    if not detector.load_model():
        raise RuntimeError(detector.last_error or "Falha ao carregar YOLOPv2.")

    for input_path in INPUTS:
        image = YOLOPLaneDetector.load_image(input_path)
        result = detector.detect(image)
        output_path = OUTPUT_DIR / f"{input_path.stem}_yolopv2.png"
        YOLOPLaneDetector.save_image(output_path, draw_result(image, result))
        print(f"{input_path.name}: {result.num_lanes_detected} faixas -> {output_path}")


if __name__ == "__main__":
    main()
