
from pathlib import Path
import sys

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from vision.ufld_detector import create_default_detector


INPUT_PATH = PROJECT_ROOT / "ufld_test_input.png"
OUTPUT_PATH = PROJECT_ROOT / "ufld_test_output.png"


def point_xy(point):
    if hasattr(point, "x") and hasattr(point, "y"):
        return float(point.x), float(point.y)

    if hasattr(point, "x_px") and hasattr(point, "y_px"):
        return float(point.x_px), float(point.y_px)

    raise TypeError(
        f"LanePoint incompatível: {type(point).__name__}"
    )


def draw_lane(image, lane, lane_index):
    points = []

    for point in lane:
        x, y = point_xy(point)
        points.append((int(round(x)), int(round(y))))

    if not points:
        return

    for x, y in points:
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            cv2.circle(
                image,
                (x, y),
                4,
                (0, 255, 0),
                -1,
                cv2.LINE_AA,
            )

    for p1, p2 in zip(points, points[1:]):
        cv2.line(
            image,
            p1,
            p2,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    x0, y0 = points[0]

    cv2.putText(
        image,
        f"LANE {lane_index}",
        (x0 + 8, y0),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def main():
    print("=" * 60)
    print("UFLD VISUAL TEST")
    print("=" * 60)

    print(f"INPUT : {INPUT_PATH}")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Imagem de entrada não encontrada:\n{INPUT_PATH}"
        )

    # Leitura robusta para caminhos Unicode no Windows.
    data = INPUT_PATH.read_bytes()

    frame = cv2.imdecode(
        np.frombuffer(data, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )

    if frame is None:
        raise RuntimeError(
            f"OpenCV não conseguiu carregar a imagem:\n{INPUT_PATH}"
        )

    print(f"IMAGE : {frame.shape[1]}x{frame.shape[0]}")

    detector = create_default_detector()

    print("MODEL : loading...")
    detector.load_model()
    print("MODEL : loaded")

    preprocessed = detector.preprocess(frame)
    print("PREPROCESS: OK")

    raw = detector.infer(preprocessed)
    print("INFERENCE: OK")

    if hasattr(raw, "shape"):
        print(f"RAW SHAPE: {tuple(raw.shape)}")
    else:
        print(f"RAW TYPE : {type(raw).__name__}")

    result = detector.decode(raw)

    print("DECODE: OK")
    print()

    print(f"LANES   : {len(result.lanes)}")
    print(f"CONF    : {result.lane_confidences}")
    print(f"CURRENT : {result.current_lane_index}")
    print(f"LEFT    : {len(result.left_lane)}")
    print(f"RIGHT   : {len(result.right_lane)}")
    print(f"VALID   : {result.valid}")
    print(f"ERROR   : {result.error}")

    output = frame.copy()

    for lane_index, lane in enumerate(result.lanes):
        draw_lane(output, lane, lane_index)

    cv2.rectangle(
        output,
        (10, 10),
        (390, 115),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        output,
        f"Lanes: {len(result.lanes)}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        output,
        f"Current: {result.current_lane_index}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        output,
        f"Valid: {result.valid}",
        (20, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Gravação robusta para caminhos Unicode no Windows.
    success, encoded = cv2.imencode(
        ".png",
        output,
    )

    if not success:
        raise RuntimeError(
            "OpenCV não conseguiu codificar a imagem PNG."
        )

    OUTPUT_PATH.write_bytes(encoded.tobytes())

    if not OUTPUT_PATH.exists():
        raise RuntimeError(
            f"O arquivo não foi criado:\n{OUTPUT_PATH}"
        )

    print()
    print("=" * 60)
    print(f"OUTPUT  : {OUTPUT_PATH}")
    print(f"EXISTS  : {OUTPUT_PATH.exists()}")
    print(f"SIZE    : {OUTPUT_PATH.stat().st_size} bytes")
    print("=" * 60)


if __name__ == "__main__":
    main()

