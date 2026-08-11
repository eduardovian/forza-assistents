
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


# ============================================================
# CONFIGURAÇÃO
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "weights" / "yolop-640-640.onnx"
IMAGE_PATH = PROJECT_ROOT / "images" / "image.png"
OUTPUT_DIR = PROJECT_ROOT / "visualization" / "yolop"

OUTPUT_ORIGINAL = OUTPUT_DIR / "original.png"
OUTPUT_LANE_MASK = OUTPUT_DIR / "lane_mask.png"
OUTPUT_DRIVABLE_MASK = OUTPUT_DIR / "drivable_mask.png"
OUTPUT_OVERLAY = OUTPUT_DIR / "yolop_overlay.png"
OUTPUT_COMBINED = OUTPUT_DIR / "yolop_combined.png"


# ============================================================
# FUNÇÕES
# ============================================================

def load_image_unicode(path: Path):
    """
    Carrega imagem corretamente mesmo quando o caminho
    contém caracteres Unicode, como 'Área de Trabalho'.
    """

    
        str(path),
        dtype=np.uint8,
    )

    image = cv2.imdecode(
        data,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise RuntimeError(
            f"Não foi possível carregar:\n{path}"
        )

    return image


def save_image_unicode(path: Path, image):
    """
    Salva imagem corretamente em caminhos Unicode.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    extension = path.suffix

    success, encoded = cv2.imencode(
        extension,
        image,
    )

    if not success:
        raise RuntimeError(
            f"Não foi possível codificar:\n{path}"
        )

    encoded.tofile(str(path))


def preprocess(image):
    """
    Pré-processamento do YOLOP.
    """

    resized = cv2.resize(
        image,
        (640, 640),
        interpolation=cv2.INTER_LINEAR,
    )

    rgb = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2RGB,
    )

    tensor = (
        rgb.astype(np.float32)
        / 255.0
    )

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

    return tensor


def create_binary_mask(output):
    """
    Converte uma saída de segmentação [1, 2, H, W]
    em uma máscara binária.

    Canal 0 = background
    Canal 1 = classe detectada.
    """

    output = output[0]

    # Classe com maior probabilidade.
    class_map = np.argmax(
        output,
        axis=0,
    )

    mask = (
        class_map == 1
    ).astype(np.uint8) * 255

    return mask


def colorize_mask(mask, color):
    """
    Cria uma imagem colorida a partir da máscara.
    """

    result = np.zeros(
        (
            mask.shape[0],
            mask.shape[1],
            3,
        ),
        dtype=np.uint8,
    )

    result[mask > 0] = color

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("YOLOP VISUALIZATION TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Verificações
    # --------------------------------------------------------

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado:\n{MODEL_PATH}"
        )

   
        

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Carregar imagem
    # --------------------------------------------------------

    print()
    print("Loading image...")

    image = load_image_unicode(
        IMAGE_PATH
    )

    print(
        f"Original: "
        f"{image.shape[1]}x{image.shape[0]}"
    )

    # --------------------------------------------------------
    # ONNX Runtime
    # --------------------------------------------------------

    print()
    print("Loading YOLOP...")

    session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=[
            "CPUExecutionProvider"
        ],
    )

    print(
        "Provider:",
        session.get_providers(),
    )

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    tensor = preprocess(
        image
    )

    print(
        "Input:",
        tensor.shape,
    )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    print()
    print("Running inference...")

    input_name = (
        session
        .get_inputs()[0]
        .name
    )

    outputs = session.run(
        None,
        {
            input_name: tensor
        },
    )

    det_out = outputs[0]

    drive_area_seg = outputs[1]

    lane_line_seg = outputs[2]

    print()
    print("Inference OK")

    print(
        "det_out:",
        det_out.shape,
    )

    print(
        "drive_area_seg:",
        drive_area_seg.shape,
    )

    print(
        "lane_line_seg:",
        lane_line_seg.shape,
    )

    # --------------------------------------------------------
    # Criar máscaras
    # --------------------------------------------------------

    print()
    print("Creating masks...")

    lane_mask = create_binary_mask(
        lane_line_seg
    )

    drivable_mask = create_binary_mask(
        drive_area_seg
    )

    print(
        "Lane pixels:",
        int(
            np.count_nonzero(
                lane_mask
            )
        ),
    )

    print(
        "Drivable pixels:",
        int(
            np.count_nonzero(
                drivable_mask
            )
        ),
    )

    # --------------------------------------------------------
    # Colorização
    # --------------------------------------------------------

    # Lane = vermelho
    lane_color = colorize_mask(
        lane_mask,
        (0, 0, 255),
    )

    # Drivable area = verde
    drivable_color = colorize_mask(
        drivable_mask,
        (0, 255, 0),
    )

    # --------------------------------------------------------
    # Resize imagem original
    # --------------------------------------------------------

    base = cv2.resize(
        image,
        (640, 640),
        interpolation=cv2.INTER_LINEAR,
    )

    # --------------------------------------------------------
    # Overlay das lanes
    # --------------------------------------------------------

    lane_overlay = cv2.addWeighted(
        base,
        0.70,
        lane_color,
        0.90,
        0,
    )

    # --------------------------------------------------------
    # Overlay área dirigível
    # --------------------------------------------------------

    combined_color = cv2.addWeighted(
        lane_color,
        0.85,
        drivable_color,
        0.45,
        0,
    )

    combined = cv2.addWeighted(
        base,
        0.65,
        combined_color,
        0.80,
        0,
    )

    # --------------------------------------------------------
    # Informações na imagem
    # --------------------------------------------------------

    cv2.putText(
        lane_overlay,
        "YOLOP - LANE LINE",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        combined,
        "YOLOP - LANE + DRIVABLE AREA",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # --------------------------------------------------------
    # Salvar resultados
    # --------------------------------------------------------

    save_image_unicode(
        OUTPUT_ORIGINAL,
        base,
    )

    save_image_unicode(
        OUTPUT_LANE_MASK,
        lane_mask,
    )

    save_image_unicode(
        OUTPUT_DRIVABLE_MASK,
        drivable_mask,
    )

    save_image_unicode(
        OUTPUT_OVERLAY,
        lane_overlay,
    )

    save_image_unicode(
        OUTPUT_COMBINED,
        combined,
    )

    print()
    print("Files saved:")
    print(
        OUTPUT_ORIGINAL
    )
    print(
        OUTPUT_LANE_MASK
    )
    print(
        OUTPUT_DRIVABLE_MASK
    )
    print(
        OUTPUT_OVERLAY
    )
    print(
        OUTPUT_COMBINED
    )

    # --------------------------------------------------------
    # Mostrar resultado
    # --------------------------------------------------------

    cv2.imshow(
        "YOLOP - Lane Detection",
        lane_overlay,
    )

    cv2.imshow(
        "YOLOP - Lane + Drivable Area",
        combined,
    )

    print()
    print(
        "Press any key in an image window to exit."
    )

    cv2.waitKey(0)

    cv2.destroyAllWindows()

    print()
    print("=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

