
"""
tests/test_yolop_detector.py

Teste de integração:

    imagem
       ↓
    YOLOP detector
       ↓
    LaneDetectionResult
"""

from pathlib import Path
import sys

import cv2
import numpy as np


# ============================================================================
# PROJECT ROOT
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# IMPORT
# ============================================================================

from vision.yolop_detector import create_default_detector


# ============================================================================
# PATHS
# ============================================================================

IMAGE_PATH = PROJECT_ROOT / "ufld_test_input.png"


# ============================================================================
# IMAGE LOADING
# ============================================================================

def load_image_unicode(path: Path):
    """
    Carrega uma imagem corretamente em Windows,
    inclusive quando o caminho possui caracteres Unicode.
    """

    if not path.is_file():
        raise FileNotFoundError(
            f"Imagem não encontrada:\n{path}"
        )

    data = np.fromfile(
        str(path),
        dtype=np.uint8,
    )

    image = cv2.imdecode(
        data,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise RuntimeError(
            f"OpenCV não conseguiu decodificar:\n{path}"
        )

    return image


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 70)
    print("YOLOP -> LaneDetectionResult TEST")
    print("=" * 70)

    # ------------------------------------------------------------------------
    # Project
    # ------------------------------------------------------------------------

    print()
    print("Project root:")
    print(PROJECT_ROOT)

    # ------------------------------------------------------------------------
    # Image
    # ------------------------------------------------------------------------

    print()
    print("Image:")
    print(IMAGE_PATH)

    print()
    print("Loading image...")

    image = load_image_unicode(
        IMAGE_PATH
    )

    print(
        "Original:",
        f"{image.shape[1]}x{image.shape[0]}"
    )

    # ------------------------------------------------------------------------
    # Detector
    # ------------------------------------------------------------------------

    print()
    print("Creating YOLOP detector...")

    detector = create_default_detector()

    print(
        "Detector:",
        type(detector).__name__,
    )

    # ------------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------------

    print()
    print("Loading YOLOP...")

    loaded = detector.load_model()

    print(
        "Load:",
        loaded,
    )

    if not loaded:

        error = getattr(
            detector,
            "last_error",
            None,
        )

        raise RuntimeError(
            error
            or "Não foi possível carregar o modelo YOLOP."
        )

    # ------------------------------------------------------------------------
    # Provider
    # ------------------------------------------------------------------------

    provider_method = getattr(
        detector,
        "get_device_name",
        None,
    )

    if callable(provider_method):

        print(
            "Provider:",
            provider_method(),
        )

    # ------------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------------

    print()
    print("Running YOLOP detection...")

    result = detector.detect(
        image
    )

    # ------------------------------------------------------------------------
    # Result
    # ------------------------------------------------------------------------

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)

    print(
        "Type:",
        type(result).__name__,
    )

    print(
        "Valid:",
        getattr(result, "valid", None),
    )

    print(
        "Detected:",
        getattr(result, "detected", None),
    )

    print(
        "Number of lanes:",
        getattr(
            result,
            "num_lanes_detected",
            None,
        ),
    )

    # ------------------------------------------------------------------------
    # Left lane
    # ------------------------------------------------------------------------

    left_lane = getattr(
        result,
        "left_lane",
        [],
    )

    print()
    print(
        "Left points:",
        len(left_lane),
    )

    left_confidence = getattr(
        result,
        "left_confidence",
        None,
    )

    if left_confidence is not None:

        print(
            "Left confidence:",
            f"{left_confidence:.3f}",
        )

    # ------------------------------------------------------------------------
    # Right lane
    # ------------------------------------------------------------------------

    right_lane = getattr(
        result,
        "right_lane",
        [],
    )

    print(
        "Right points:",
        len(right_lane),
    )

    right_confidence = getattr(
        result,
        "right_confidence",
        None,
    )

    if right_confidence is not None:

        print(
            "Right confidence:",
            f"{right_confidence:.3f}",
        )

    # ------------------------------------------------------------------------
    # Model output
    # ------------------------------------------------------------------------

    model_output_shape = getattr(
        result,
        "model_output_shape",
        None,
    )

    print(
        "Model output:",
        model_output_shape,
    )

    # ------------------------------------------------------------------------
    # Error
    # ------------------------------------------------------------------------

    error = getattr(
        result,
        "error",
        None,
    )

    if error:

        print()
        print(
            "ERROR:",
            error,
        )

        raise RuntimeError(
            error
        )

    # ------------------------------------------------------------------------
    # Lane samples
    # ------------------------------------------------------------------------

    if left_lane:

        print()
        print("Left lane sample:")

        for point in left_lane[:5]:

            print(
                f"  x={point.x:.1f} "
                f"y={point.y:.1f} "
                f"confidence={point.confidence:.3f}"
            )

    else:

        print()
        print("Left lane: NONE")

    if right_lane:

        print()
        print("Right lane sample:")

        for point in right_lane[:5]:

            print(
                f"  x={point.x:.1f} "
                f"y={point.y:.1f} "
                f"confidence={point.confidence:.3f}"
            )

    else:

        print()
        print("Right lane: NONE")

    # ------------------------------------------------------------------------
    # Diagnostic
    # ------------------------------------------------------------------------

    print()
    print("=" * 70)
    print("DIAGNOSTIC")
    print("=" * 70)

    print(
        "LEFT lane:",
        "OK" if left_lane else "NOT DETECTED",
    )

    print(
        "RIGHT lane:",
        "OK" if right_lane else "NOT DETECTED",
    )

    print(
        "PAIR:",
        "OK" if left_lane and right_lane else "INCOMPLETE",
    )

    # ------------------------------------------------------------------------
    # Final
    # ------------------------------------------------------------------------

    print()
    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

