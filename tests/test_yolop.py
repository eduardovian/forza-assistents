
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "weights" / "yolop-640-640.onnx"
IMAGE_PATH = PROJECT_ROOT / "ufld_test_input.png"


def main():
    print("=" * 70)
    print("YOLOP TEST")
    print("=" * 70)

    print(f"MODEL: {MODEL_PATH}")
    print(f"IMAGE: {IMAGE_PATH}")
    print()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado:\n{MODEL_PATH}"
        )

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Imagem não encontrada:\n{IMAGE_PATH}"
        )

    providers = ort.get_available_providers()

    print("AVAILABLE PROVIDERS:")

    for provider in providers:
        print(f"  - {provider}")

    print()

    # Temporariamente usamos CPU.
    # Depois vamos configurar CUDA corretamente.
    selected = ["CPUExecutionProvider"]

    print("USING:")
    print(f"  {selected}")
    print()

    session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=selected,
    )

    print("MODEL INPUTS:")

    for inp in session.get_inputs():
        print(
            f"  name={inp.name}, "
            f"shape={inp.shape}, "
            f"type={inp.type}"
        )

    print()

    print("MODEL OUTPUTS:")

    for out in session.get_outputs():
        print(
            f"  name={out.name}, "
            f"shape={out.shape}, "
            f"type={out.type}"
        )

    print()

    # Leitura compatível com caminhos Unicode do Windows.
    image_data = np.fromfile(
        str(IMAGE_PATH),
        dtype=np.uint8,
    )

    image = cv2.imdecode(
        image_data,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise RuntimeError(
            "OpenCV não conseguiu carregar a imagem."
        )

    print(
        f"ORIGINAL IMAGE: "
        f"{image.shape[1]}x{image.shape[0]}"
    )

    resized = cv2.resize(
        image,
        (640, 640),
        interpolation=cv2.INTER_LINEAR,
    )

    rgb = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2RGB,
    )

    tensor = rgb.astype(np.float32) / 255.0

    tensor = np.transpose(
        tensor,
        (2, 0, 1),
    )

    tensor = np.expand_dims(
        tensor,
        axis=0,
    )

    tensor = np.ascontiguousarray(
        tensor,
        dtype=np.float32,
    )

    input_name = session.get_inputs()[0].name

    print(
        f"INPUT TENSOR: {tensor.shape}"
    )

    print()
    print("RUNNING INFERENCE...")

    outputs = session.run(
        None,
        {
            input_name: tensor,
        },
    )

    print("INFERENCE OK")
    print()

    for index, output in enumerate(outputs):
        print(
            f"OUTPUT {index}: "
            f"shape={output.shape}, "
            f"dtype={output.dtype}"
        )

    print()
    print("=" * 70)
    print("YOLOP IS WORKING")
    print("=" * 70)


if __name__ == "__main__":
    main()

