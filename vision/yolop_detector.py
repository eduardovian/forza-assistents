
"""
vision/yolop_detector.py

Forza Assistents
================

Detector de percepção baseado no YOLOPv2 oficial em TorchScript.

Arquitetura:

    frame BGR
        │
        ▼
    validação
        │
        ▼
    canonicalização YOLOPv2
        │
        ▼
    letterbox 640x640
        │
        ▼
    TorchScript / CUDA
        │
        ├───────────────┐
        │               │
        ▼               ▼
    detection        segmentation
        │               │
        │        ┌──────┴──────┐
        │        ▼             ▼
        │    drivable         lanes
        │        │             │
        │        │        probabilidade
        │        │             │
        │        │          máscara
        │        │             │
        │        │      componentes/segmentos
        │        │             │
        │        │      associação espacial
        │        │             │
        │        │       LanePoint[]
        │        │             │
        └────────┴─────────────┤
                               ▼
                    LaneDetectionResult

Este módulo NÃO executa:

    - tracking temporal entre frames;
    - fitting polinomial;
    - geometria da faixa;
    - projeção;
    - identificação da faixa atual;
    - LaneAssignment;
    - decisão ADAS;
    - controle do veículo.

Essas responsabilidades pertencem às camadas posteriores.

Princípios:

    - TorchScript nativo;
    - CUDA preferencial;
    - FP16 opcional em CUDA;
    - CPU fallback;
    - fail-soft;
    - nenhuma dependência de ONNX Runtime;
    - compatibilidade com o contrato atual do projeto;
    - preservação das lanes adicionais;
    - uso da área dirigível como evidência auxiliar;
    - detecção de objetos preservada para evolução futura;
    - confiança explícita;
    - diagnóstico por frame;
    - nenhuma identidade temporal dentro do detector.

Compatibilidade pública:

    YOLOPDetector
    YOLOPLaneDetector
    create_default_detector
    LaneDetectionResult
    LanePoint
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torchvision

from .lane_types import LanePoint


logger = logging.getLogger(__name__)


# =============================================================================
# PATHS / MODEL CONTRACT
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "weights"
    / "yolopv2.pt"
)

LEGACY_MODEL_PATH = (
    PROJECT_ROOT
    / "weights"
    / "yolop-640-640.onnx"
)

# Contrato do modelo oficial.
YOLOPV2_INPUT_WIDTH = 640
YOLOPV2_INPUT_HEIGHT = 640

# O demo oficial trabalha sobre uma representação 1280x720 antes
# do letterbox para 640x640.
CANONICAL_WIDTH = 1280
CANONICAL_HEIGHT = 720

YOLOPV2_STRIDE = 32

# Pós-processamento oficial da saída de lane-line segmentation.
LANE_OUTPUT_TOP_CROP = 12
LANE_OUTPUT_BOTTOM = 372

# =============================================================================
# DEFAULTS
# =============================================================================

DEFAULT_LANE_THRESHOLD = 0.50
DEFAULT_OBJECT_CONFIDENCE = 0.30
DEFAULT_OBJECT_IOU = 0.45

DEFAULT_MIN_POINTS_PER_LANE = 4
DEFAULT_ROW_STEP = 6
DEFAULT_MIN_PIXELS_PER_SEGMENT = 1

DEFAULT_MAX_LANES = 16

DEFAULT_MAX_TRACKING_JUMP = 55.0
DEFAULT_MIN_LANE_VERTICAL_SPAN = 35.0
DEFAULT_MIN_LANE_HORIZONTAL_SPAN = 1.0

DEFAULT_MORPH_KERNEL = 3
DEFAULT_MIN_COMPONENT_AREA = 3

DEFAULT_SAMPLE_BOTTOM_ROWS = 10

DEFAULT_USE_FP16 = True

# Para não transformar o detector em um filtro excessivamente agressivo.
DEFAULT_DRIVABLE_GATE_MIN = 0.15
DEFAULT_DRIVABLE_CONFIDENCE_BONUS = 0.10

# Curvatura vertical máxima permitida entre observações consecutivas
# durante a reconstrução dentro de um único frame.
DEFAULT_MAX_SLOPE_CHANGE = 1.8


# =============================================================================
# COCO NAMES
# =============================================================================
#
# O modelo TorchScript oficial utiliza a cabeça de detecção com 85 valores
# por âncora, isto é:
#
#     4 bbox + 1 objectness + 80 classes
#
# O YOLOPv2 oficial, portanto, segue o contrato de 80 classes da cabeça
# YOLO utilizada no projeto original.
#
# Mantemos os nomes aqui apenas para diagnóstico e futura camada de veículos.
# O pipeline de lanes não depende desses nomes.
#

COCO_CLASS_NAMES: Tuple[str, ...] = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)


# Classes relevantes para a futura camada de percepção de tráfego.
VEHICLE_CLASS_IDS = frozenset(
    {
        2,   # car
        3,   # motorcycle
        5,   # bus
        7,   # truck
    }
)

ROAD_USER_CLASS_IDS = frozenset(
    {
        0,   # person
        1,   # bicycle
        2,   # car
        3,   # motorcycle
        5,   # bus
        7,   # truck
    }
)


# =============================================================================
# NUMERIC UTILITIES
# =============================================================================


def _finite(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(result):
        return None

    return result


def _clip01(value: Any) -> float:
    result = _finite(value)

    if result is None:
        return 0.0

    return max(0.0, min(1.0, result))


def _finite_array(
    values: np.ndarray,
) -> np.ndarray:
    return np.nan_to_num(
        values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


# =============================================================================
# RESULT TYPES
# =============================================================================


@dataclass(frozen=True)
class ObjectDetection:
    """
    Detecção de objeto produzida pela cabeça YOLOPv2.

    Não participa diretamente do pipeline de LaneModel.

    Serve como contrato de percepção para a futura camada
    de risco/distância/veículos.
    """

    class_id: int

    confidence: float

    x1: float
    y1: float
    x2: float
    y2: float

    frame_width: int
    frame_height: int

    @property
    def class_name(self) -> str:
        if 0 <= self.class_id < len(COCO_CLASS_NAMES):
            return COCO_CLASS_NAMES[self.class_id]

        return f"class_{self.class_id}"

    @property
    def is_vehicle(self) -> bool:
        return self.class_id in VEHICLE_CLASS_IDS

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) * 0.5

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) * 0.5

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def bottom_y(self) -> float:
        return self.y2

    @property
    def normalized_center_x(self) -> float:
        if self.frame_width <= 0:
            return 0.5

        return _clip01(
            self.center_x
            / float(self.frame_width)
        )

    @property
    def normalized_bottom_y(self) -> float:
        if self.frame_height <= 0:
            return 1.0

        return _clip01(
            self.bottom_y
            / float(self.frame_height)
        )


@dataclass
class LaneDetectionResult:
    """
    Resultado público do detector.

    lanes:
        Todas as linhas de faixa detectadas.

    left_lane / right_lane:
        Apenas classificação espacial.
        Não significa lane atual.

    additional_lanes:
        Demais linhas preservadas.

    objects:
        Detecções de tráfego da cabeça de objetos.

    drivable_area_mask:
        Máscara binária da área dirigível, já convertida
        para o frame original.

    metadata:
        Diagnóstico adicional.

    O resultado mantém os campos utilizados pelo pipeline atual,
    evitando alterações desnecessárias nos módulos downstream.
    """

    lanes: List[List[LanePoint]] = field(
        default_factory=list
    )

    lane_confidences: List[float] = field(
        default_factory=list
    )

    current_lane_index: Optional[int] = None

    left_lane: List[LanePoint] = field(
        default_factory=list
    )

    right_lane: List[LanePoint] = field(
        default_factory=list
    )

    additional_lanes: List[List[LanePoint]] = field(
        default_factory=list
    )

    left_confidence: float = 0.0

    right_confidence: float = 0.0

    valid: bool = False

    num_lanes_detected: int = 0

    input_width: int = YOLOPV2_INPUT_WIDTH

    input_height: int = YOLOPV2_INPUT_HEIGHT

    model_output_shape: Tuple[int, ...] = field(
        default_factory=tuple
    )

    error: Optional[str] = None

    objects: List[ObjectDetection] = field(
        default_factory=list
    )

    drivable_area_mask: Optional[np.ndarray] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def detected(self) -> bool:
        return self.num_lanes_detected > 0

    @property
    def has_current_lane(self) -> bool:
        return (
            self.current_lane_index is not None
            and bool(self.left_lane)
            and bool(self.right_lane)
        )

    @property
    def vehicle_detections(
        self,
    ) -> List[ObjectDetection]:
        return [
            detection
            for detection in self.objects
            if detection.is_vehicle
        ]

    @property
    def object_count(self) -> int:
        return len(self.objects)

    @property
    def vehicle_count(self) -> int:
        return len(self.vehicle_detections)


# =============================================================================
# INTERNAL ROW SEGMENT
# =============================================================================


@dataclass
class _RowSegment:
    y: int
    x_min: float
    x_max: float
    x_center: float
    confidence: float
    road_confidence: float
    pixel_count: int

    @property
    def width(self) -> float:
        return self.x_max - self.x_min


# =============================================================================
# INTERNAL FRAME TRACK
# =============================================================================


@dataclass
class _LaneTrack:
    points: List[Tuple[int, float, float]] = field(
        default_factory=list
    )

    last_x: float = 0.0
    last_y: int = 0

    previous_x: Optional[float] = None
    previous_y: Optional[int] = None

    confidence_sum: float = 0.0
    pixel_sum: int = 0
    road_confidence_sum: float = 0.0

    def predicted_x(self) -> float:
        if (
            self.previous_x is None
            or self.previous_y is None
        ):
            return self.last_x

        dy = self.last_y - self.previous_y

        if dy == 0:
            return self.last_x

        dx = self.last_x - self.previous_x

        return (
            self.last_x
            + dx / float(dy)
        )


# =============================================================================
# PREPROCESS METADATA
# =============================================================================


@dataclass(frozen=True)
class _PreprocessMeta:
    """
    Transformação usada para passar do frame do projeto
    ao tensor aceito pelo YOLOPv2.

    A representação canônica é mantida explícita para
    que a máscara possa retornar ao frame original sem
    depender de escala mágica.
    """

    original_width: int
    original_height: int

    canonical_width: int
    canonical_height: int

    ratio: float

    pad_left: int
    pad_top: int

    pad_right: int
    pad_bottom: int


# =============================================================================
# DETECTOR
# =============================================================================


class YOLOPLaneDetector:
    """
    Detector YOLOPv2 em TorchScript.

    A classe preserva o nome utilizado pelo pipeline atual.

    Public API:

        load_model()
        get_device_name()
        preprocess()
        infer()
        detect()

    O detector é stateless entre frames no sentido temporal:
    nenhum histórico de lane é mantido aqui.

    Isso é deliberado.

    A identidade temporal pertence ao LaneTracker.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        input_width: int = YOLOPV2_INPUT_WIDTH,
        input_height: int = YOLOPV2_INPUT_HEIGHT,
        lane_threshold: float = DEFAULT_LANE_THRESHOLD,
        min_points_per_lane: int = DEFAULT_MIN_POINTS_PER_LANE,
        row_step: int = DEFAULT_ROW_STEP,
        min_lane_pixels_per_row: int = (
            DEFAULT_MIN_PIXELS_PER_SEGMENT
        ),
        max_lanes: int = DEFAULT_MAX_LANES,
        max_tracking_jump: Optional[float] = None,
        min_lane_span: float = 25.0,
        min_lane_vertical_span: float = (
            DEFAULT_MIN_LANE_VERTICAL_SPAN
        ),
        morph_kernel: int = DEFAULT_MORPH_KERNEL,
        min_component_area: int = (
            DEFAULT_MIN_COMPONENT_AREA
        ),
        providers: Optional[List[str]] = None,
        use_fp16: bool = DEFAULT_USE_FP16,
        object_confidence: float = (
            DEFAULT_OBJECT_CONFIDENCE
        ),
        object_iou: float = DEFAULT_OBJECT_IOU,
        canonical_width: int = CANONICAL_WIDTH,
        canonical_height: int = CANONICAL_HEIGHT,
    ) -> None:

        self.requested_model_path = Path(
            model_path
        )

        self.model_path = (
            self._resolve_model_path(
                self.requested_model_path
            )
        )

        self.input_width = max(
            1,
            int(input_width),
        )

        self.input_height = max(
            1,
            int(input_height),
        )

        self.lane_threshold = _clip01(
            lane_threshold
        )

        self.min_points_per_lane = max(
            2,
            int(min_points_per_lane),
        )

        self.row_step = max(
            1,
            int(row_step),
        )

        self.min_lane_pixels_per_row = max(
            1,
            int(min_lane_pixels_per_row),
        )

        self.max_lanes = max(
            1,
            int(max_lanes),
        )

        self.max_tracking_jump = (
            float(max_tracking_jump)
            if max_tracking_jump is not None
            else max(
                DEFAULT_MAX_TRACKING_JUMP,
                self.input_width
                * 0.14,
            )
        )

        self.min_lane_span = max(
            0.0,
            float(min_lane_span),
        )

        self.min_lane_vertical_span = max(
            0.0,
            float(min_lane_vertical_span),
        )

        self.morph_kernel = max(
            1,
            int(morph_kernel),
        )

        if self.morph_kernel % 2 == 0:
            self.morph_kernel += 1

        self.min_component_area = max(
            1,
            int(min_component_area),
        )

        self.providers = (
            list(providers)
            if providers is not None
            else None
        )

        self.use_fp16 = bool(
            use_fp16
        )

        self.object_confidence = _clip01(
            object_confidence
        )

        self.object_iou = _clip01(
            object_iou
        )

        self.canonical_width = max(
            64,
            int(canonical_width),
        )

        self.canonical_height = max(
            64,
            int(canonical_height),
        )

        # ---------------------------------------------------------------------
        # Runtime state
        # ---------------------------------------------------------------------

        self.model: Optional[
            torch.jit.ScriptModule
        ] = None

        self.device = torch.device(
            "cpu"
        )

        self.loaded = False

        self.fp16_active = False

        self.last_error: Optional[str] = None

        self.last_result: Optional[
            LaneDetectionResult
        ] = None

        self.last_output_shape: Tuple[
            int, ...
        ] = tuple()

        self.last_output_shapes: List[
            Tuple[int, ...]
        ] = []

        self.last_lane_probability: Optional[
            np.ndarray
        ] = None

        self.last_lane_mask: Optional[
            np.ndarray
        ] = None

        self.last_drivable_mask: Optional[
            np.ndarray
        ] = None

        self.last_diagnostics: Dict[
            str,
            Any,
        ] = {}

        self.last_preprocess_meta: Optional[
            _PreprocessMeta
        ] = None

        self._warmup_done = False

    # =========================================================================
    # MODEL PATH
    # =========================================================================

    @staticmethod
    def _resolve_model_path(
        requested: Path,
    ) -> Path:

        if requested.is_file():
            return requested

        if (
            requested == LEGACY_MODEL_PATH
            or requested.suffix.lower() == ".onnx"
        ):
            if DEFAULT_MODEL_PATH.is_file():
                logger.warning(
                    "[YOLOPv2] Caminho legado solicitado (%s), "
                    "usando YOLOPv2: %s",
                    requested,
                    DEFAULT_MODEL_PATH,
                )

                return DEFAULT_MODEL_PATH

        return requested

    def model_exists(self) -> bool:
        return self.model_path.is_file()

    # =========================================================================
    # DEVICE
    # =========================================================================

    def _select_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device(
                "cuda:0"
            )

        return torch.device(
            "cpu"
        )

    def get_device_name(self) -> str:

        if not self.loaded:
            return "NOT_LOADED"

        if self.device.type == "cuda":

            try:
                return str(
                    torch.cuda.get_device_name(
                        self.device
                    )
                )

            except Exception:
                return "CUDA"

        return "CPU"

    # =========================================================================
    # LOAD
    # =========================================================================

    def load_model(self) -> bool:
        """
        Carrega o TorchScript YOLOPv2.

        CUDA é preferencial.

        CPU permanece como fallback.

        A assinatura aceita `providers` apenas por
        compatibilidade com o main/config legado.
        """

        if (
            self.loaded
            and self.model is not None
        ):
            return True

        self.last_error = None

        if not self.model_exists():

            self.last_error = (
                "Modelo YOLOPv2 não encontrado: "
                f"{self.model_path}"
            )

            logger.error(
                "[YOLOPv2] %s",
                self.last_error,
            )

            return False

        if self.model_path.suffix.lower() != ".pt":

            self.last_error = (
                "O detector atual requer um checkpoint "
                "TorchScript .pt do YOLOPv2. "
                f"Arquivo recebido: {self.model_path}"
            )

            logger.error(
                "[YOLOPv2] %s",
                self.last_error,
            )

            return False

        target_device = self._select_device()

        try:
            logger.info(
                "[YOLOPv2] Carregando TorchScript: %s",
                self.model_path,
            )

            model = torch.jit.load(
                str(self.model_path),
                map_location=target_device,
            )

            model = model.to(
                target_device
            )

            model.eval()

            fp16_active = (
                self.use_fp16
                and target_device.type
                == "cuda"
            )

            if fp16_active:
                model = model.half()

            self.model = model
            self.device = target_device
            self.fp16_active = fp16_active
            self.loaded = True
            self._warmup_done = False

            logger.info(
                "[YOLOPv2] READY | device=%s | fp16=%s",
                self.get_device_name(),
                self.fp16_active,
            )

            self._warmup()

            return True

        except Exception as exc:

            self.model = None
            self.loaded = False
            self.fp16_active = False

            self.last_error = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            logger.exception(
                "[YOLOPv2] Falha ao carregar modelo."
            )

            # Última chance: CPU.
            if target_device.type == "cuda":

                try:
                    logger.warning(
                        "[YOLOPv2] Tentando fallback CPU."
                    )

                    model = torch.jit.load(
                        str(self.model_path),
                        map_location="cpu",
                    )

                    model = model.to(
                        torch.device("cpu")
                    )

                    model.eval()

                    self.model = model
                    self.device = torch.device(
                        "cpu"
                    )
                    self.fp16_active = False
                    self.loaded = True
                    self._warmup_done = False

                    logger.warning(
                        "[YOLOPv2] CPU fallback ativo."
                    )

                    self._warmup()

                    return True

                except Exception as cpu_exc:

                    self.last_error = (
                        f"CUDA: {self.last_error}; "
                        f"CPU: "
                        f"{type(cpu_exc).__name__}: "
                        f"{cpu_exc}"
                    )

                    self.model = None
                    self.loaded = False

            return False

    # =========================================================================
    # WARMUP
    # =========================================================================

    def _warmup(self) -> None:

        if (
            not self.loaded
            or self.model is None
            or self._warmup_done
        ):
            return

        try:

            dtype = (
                torch.float16
                if self.fp16_active
                else torch.float32
            )

            tensor = torch.zeros(
                (
                    1,
                    3,
                    self.input_height,
                    self.input_width,
                ),
                device=self.device,
                dtype=dtype,
            )

            with torch.inference_mode():
                _ = self.model(
                    tensor
                )

            if self.device.type == "cuda":
                torch.cuda.synchronize(
                    self.device
                )

            self._warmup_done = True

        except Exception as exc:

            logger.warning(
                "[YOLOPv2] Warmup falhou: %s",
                exc,
            )

    # =========================================================================
    # FRAME VALIDATION
    # =========================================================================

    @staticmethod
    def _validate_frame(
        frame: np.ndarray,
    ) -> Tuple[int, int]:

        if frame is None:
            raise ValueError(
                "Frame é None."
            )

        if not isinstance(
            frame,
            np.ndarray,
        ):
            raise TypeError(
                "Frame deve ser numpy.ndarray."
            )

        if frame.size == 0:
            raise ValueError(
                "Frame vazio."
            )

        if frame.ndim != 3:
            raise ValueError(
                "Frame deve ser HxWxC."
            )

        if frame.shape[2] != 3:
            raise ValueError(
                "Frame deve possuir 3 canais BGR."
            )

        height, width = frame.shape[:2]

        if (
            width <= 0
            or height <= 0
        ):
            raise ValueError(
                "Dimensões inválidas."
            )

        return int(width), int(height)

    # =========================================================================
    # LETTERBOX
    # =========================================================================

    def _letterbox(
        self,
        image: np.ndarray,
    ) -> Tuple[
        np.ndarray,
        float,
        Tuple[int, int],
    ]:
        """
        Letterbox determinístico para 640x640.

        Retorna:

            image
            scale
            (pad_left, pad_top)
        """

        target_w = self.input_width
        target_h = self.input_height

        src_h, src_w = image.shape[:2]

        scale = min(
            target_w / float(src_w),
            target_h / float(src_h),
        )

        resized_w = max(
            1,
            int(round(src_w * scale)),
        )

        resized_h = max(
            1,
            int(round(src_h * scale)),
        )

        resized = cv2.resize(
            image,
            (
                resized_w,
                resized_h,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

        pad_x = target_w - resized_w
        pad_y = target_h - resized_h

        left = pad_x // 2
        right = pad_x - left

        top = pad_y // 2
        bottom = pad_y - top

        output = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )

        return (
            output,
            float(scale),
            (
                int(left),
                int(top),
            ),
        )

    # =========================================================================
    # PREPROCESS
    # =========================================================================

    def preprocess(
        self,
        frame: np.ndarray,
    ) -> np.ndarray:

        width, height = (
            self._validate_frame(
                frame
            )
        )

        # ---------------------------------------------------------------------
        # O modelo original foi treinado com a cadeia:
        #
        # imagem -> 1280x720 -> letterbox -> 640x640
        #
        # Mantemos essa convenção para maximizar compatibilidade com o
        # checkpoint oficial.
        # ---------------------------------------------------------------------

        canonical = cv2.resize(
            frame,
            (
                self.canonical_width,
                self.canonical_height,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

        letterboxed, scale, pad = (
            self._letterbox(
                canonical
            )
        )

        rgb = cv2.cvtColor(
            letterboxed,
            cv2.COLOR_BGR2RGB,
        )

        tensor = rgb.astype(
            np.float32
        )

        tensor *= (
            1.0 / 255.0
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

        self.last_preprocess_meta = (
            _PreprocessMeta(
                original_width=width,
                original_height=height,
                canonical_width=self.canonical_width,
                canonical_height=self.canonical_height,
                ratio=scale,
                pad_left=pad[0],
                pad_top=pad[1],
                pad_right=(
                    self.input_width
                    - int(
                        round(
                            self.canonical_width
                            * scale
                        )
                    )
                    - pad[0]
                ),
                pad_bottom=(
                    self.input_height
                    - int(
                        round(
                            self.canonical_height
                            * scale
                        )
                    )
                    - pad[1]
                ),
            )
        )

        return tensor

    # =========================================================================
    # TENSOR INFERENCE
    # =========================================================================

    def _prepare_tensor(
        self,
        tensor: np.ndarray,
    ) -> torch.Tensor:

        if tensor.ndim != 4:
            raise ValueError(
                "Tensor de entrada deve possuir "
                "formato NCHW."
            )

        torch_tensor = torch.from_numpy(
            tensor
        ).to(
            self.device,
            non_blocking=True,
        )

        if self.fp16_active:
            torch_tensor = (
                torch_tensor.half()
            )
        else:
            torch_tensor = (
                torch_tensor.float()
            )

        return torch_tensor

    @staticmethod
    def _shape_tuple(
        value: Any,
    ) -> Tuple[int, ...]:

        try:
            return tuple(
                int(v)
                for v in value.shape
            )
        except Exception:
            return tuple()

    def infer(
        self,
        frame: np.ndarray,
    ) -> Tuple[Any, Any, Any]:

        if not self.loaded:

            if not self.load_model():
                raise RuntimeError(
                    self.last_error
                    or "YOLOPv2 não carregado."
                )

        if self.model is None:
            raise RuntimeError(
                "Modelo YOLOPv2 inexistente."
            )

        input_array = self.preprocess(
            frame
        )

        tensor = self._prepare_tensor(
            input_array
        )

        with torch.inference_mode():

            outputs = self.model(
                tensor
            )

        if self.device.type == "cuda":
            torch.cuda.synchronize(
                self.device
            )

        if not isinstance(
            outputs,
            (tuple, list),
        ):
            raise RuntimeError(
                "YOLOPv2 deveria retornar uma tupla/lista "
                "com 3 elementos."
            )

        if len(outputs) != 3:
            raise RuntimeError(
                "Saída YOLOPv2 incompatível: "
                f"esperado 3 elementos, recebido {len(outputs)}."
            )

        prediction_output = outputs[0]
        drivable_output = outputs[1]
        lane_output = outputs[2]

        self.last_output_shapes = [
            self._shape_tuple(
                value
            )
            for value in (
                prediction_output
                if isinstance(
                    prediction_output,
                    (tuple, list),
                )
                else outputs
            )
        ]

        return (
            prediction_output,
            drivable_output,
            lane_output,
        )

    # =========================================================================
    # TRACE MODEL DETECTION DECODE
    # =========================================================================

    @staticmethod
    def _make_grid(
        nx: int,
        ny: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:

        y = torch.arange(
            ny,
            device=device,
            dtype=dtype,
        )

        x = torch.arange(
            nx,
            device=device,
            dtype=dtype,
        )

        try:
            yy, xx = torch.meshgrid(
                y,
                x,
                indexing="ij",
            )
        except TypeError:
            yy, xx = torch.meshgrid(
                y,
                x,
            )

        return torch.stack(
            (
                xx,
                yy,
            ),
            dim=2,
        ).view(
            1,
            1,
            ny,
            nx,
            2,
        )

    def _split_for_trace_model(
        self,
        prediction: Any,
        anchor_grid: Any,
    ) -> torch.Tensor:

        if not isinstance(
            prediction,
            (tuple, list),
        ):
            raise RuntimeError(
                "Cabeça de detecção YOLOPv2 inválida."
            )

        if not isinstance(
            anchor_grid,
            (tuple, list),
        ):
            raise RuntimeError(
                "anchor_grid YOLOPv2 inválido."
            )

        if len(prediction) != 3:
            raise RuntimeError(
                "A cabeça YOLOPv2 deve possuir "
                "3 escalas de detecção."
            )

        if len(anchor_grid) != 3:
            raise RuntimeError(
                "anchor_grid YOLOPv2 deveria possuir "
                "3 escalas."
            )

        chunks: List[torch.Tensor] = []

        strides = (
            8,
            16,
            32,
        )

        for i in range(3):

            feature = prediction[i]

            if feature.ndim != 4:
                raise RuntimeError(
                    "Mapa de detecção inválido: "
                    f"{tuple(feature.shape)}"
                )

            batch, channels, ny, nx = (
                feature.shape
            )

            if channels % 3 != 0:
                raise RuntimeError(
                    "Número de canais incompatível "
                    "com 3 anchors."
                )

            attributes = (
                channels // 3
            )

            reshaped = feature.view(
                batch,
                3,
                attributes,
                ny,
                nx,
            ).permute(
                0,
                1,
                3,
                4,
                2,
            ).contiguous()

            activated = (
                reshaped.sigmoid()
            )

            grid = self._make_grid(
                nx=nx,
                ny=ny,
                device=feature.device,
                dtype=feature.dtype,
            )

            stride = float(
                strides[i]
            )

            activated[..., 0:2] = (
                activated[..., 0:2]
                * 2.0
                - 0.5
                + grid
            ) * stride

            activated[..., 2:4] = (
                activated[..., 2:4]
                * 2.0
            ) ** 2 * anchor_grid[i]

            chunks.append(
                activated.view(
                    batch,
                    -1,
                    attributes,
                )
            )

        return torch.cat(
            chunks,
            dim=1,
        )

    # =========================================================================
    # NMS
    # =========================================================================

    @staticmethod
    def _xywh_to_xyxy(
        boxes: torch.Tensor,
    ) -> torch.Tensor:

        output = boxes.clone()

        output[:, 0] = (
            boxes[:, 0]
            - boxes[:, 2] * 0.5
        )

        output[:, 1] = (
            boxes[:, 1]
            - boxes[:, 3] * 0.5
        )

        output[:, 2] = (
            boxes[:, 0]
            + boxes[:, 2] * 0.5
        )

        output[:, 3] = (
            boxes[:, 1]
            + boxes[:, 3] * 0.5
        )

        return output

    def _decode_objects(
        self,
        prediction_output: Any,
        frame_width: int,
        frame_height: int,
    ) -> List[ObjectDetection]:
        """
        Decodifica a cabeça de objetos.

        Falha nesta etapa não invalida a percepção das lanes.
        """

        if not isinstance(
            prediction_output,
            (tuple, list),
        ):
            return []

        if len(prediction_output) != 2:
            return []

        raw_prediction = prediction_output[0]
        anchor_grid = prediction_output[1]

        try:

            prediction = (
                self._split_for_trace_model(
                    raw_prediction,
                    anchor_grid,
                )
            )

            if prediction.ndim != 3:
                return []

            if prediction.shape[0] < 1:
                return []

            data = prediction[0]

            if data.shape[1] < 6:
                return []

            objectness = (
                data[:, 4]
            )

            class_scores = data[:, 5:]

            class_confidence, class_id = (
                torch.max(
                    class_scores,
                    dim=1,
                )
            )

            confidence = (
                objectness
                * class_confidence
            )

            keep = (
                confidence
                >= self.object_confidence
            )

            if not bool(
                keep.any()
            ):
                return []

            boxes = (
                self._xywh_to_xyxy(
                    data[keep, :4]
                )
            )

            scores = (
                confidence[keep]
            )

            classes = (
                class_id[keep]
            )

            if (
                self.device.type == "cuda"
            ):
                boxes_cpu = (
                    boxes.float().cpu()
                )
                scores_cpu = (
                    scores.float().cpu()
                )
                classes_cpu = (
                    classes.long().cpu()
                )
            else:
                boxes_cpu = (
                    boxes.float()
                )
                scores_cpu = (
                    scores.float()
                )
                classes_cpu = (
                    classes.long()
                )

            keep_indices = (
                torchvision.ops.nms(
                    boxes_cpu,
                    scores_cpu,
                    self.object_iou,
                )
            )

            objects: List[
                ObjectDetection
            ] = []

            # -----------------------------------------------------------------
            # O decoder oficial opera nas coordenadas da imagem de inferência
            # 640x640. Para o objeto, removemos o letterbox e devolvemos
            # coordenadas no frame original.
            # -----------------------------------------------------------------

            meta = (
                self.last_preprocess_meta
            )

            if meta is None:
                return []

            boxes_numpy = (
                boxes_cpu.numpy()
            )

            scores_numpy = (
                scores_cpu.numpy()
            )

            classes_numpy = (
                classes_cpu.numpy()
            )

            for index in keep_indices.numpy():

                box = boxes_numpy[index]

                x1, y1, x2, y2 = (
                    self._map_inference_box_to_frame(
                        box,
                        frame_width,
                        frame_height,
                    )
                )

                confidence_value = (
                    _clip01(
                        scores_numpy[index]
                    )
                )

                class_value = int(
                    classes_numpy[index]
                )

                if (
                    x2 <= x1
                    or y2 <= y1
                ):
                    continue

                objects.append(
                    ObjectDetection(
                        class_id=class_value,
                        confidence=confidence_value,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        frame_width=frame_width,
                        frame_height=frame_height,
                    )
                )

            objects.sort(
                key=lambda detection:
                detection.confidence,
                reverse=True,
            )

            return objects

        except Exception as exc:

            self.last_diagnostics[
                "object_decode_error"
            ] = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            logger.debug(
                "[YOLOPv2] Falha no decode de objetos: %s",
                exc,
            )

            return []

    # =========================================================================
    # INFERENCE COORDINATE MAPPING
    # =========================================================================

    def _map_inference_point_to_frame(
        self,
        x: float,
        y: float,
        frame_width: int,
        frame_height: int,
    ) -> Tuple[float, float]:

        meta = (
            self.last_preprocess_meta
        )

        if meta is None:
            return (
                float(x),
                float(y),
            )

        # 640 -> canonical.
        canonical_x = (
            float(x)
            - float(meta.pad_left)
        ) / max(
            meta.ratio,
            1e-9,
        )

        canonical_y = (
            float(y)
            - float(meta.pad_top)
        ) / max(
            meta.ratio,
            1e-9,
        )

        canonical_x = float(
            np.clip(
                canonical_x,
                0.0,
                float(
                    meta.canonical_width
                ),
            )
        )

        canonical_y = float(
            np.clip(
                canonical_y,
                0.0,
                float(
                    meta.canonical_height
                ),
            )
        )

        frame_x = (
            canonical_x
            * float(frame_width)
            / float(
                meta.canonical_width
            )
        )

        frame_y = (
            canonical_y
            * float(frame_height)
            / float(
                meta.canonical_height
            )
        )

        return (
            float(
                np.clip(
                    frame_x,
                    0.0,
                    float(frame_width - 1),
                )
            ),
            float(
                np.clip(
                    frame_y,
                    0.0,
                    float(frame_height - 1),
                )
            ),
        )

    def _map_inference_box_to_frame(
        self,
        box: Sequence[float],
        frame_width: int,
        frame_height: int,
    ) -> Tuple[float, float, float, float]:

        x1, y1 = (
            self._map_inference_point_to_frame(
                box[0],
                box[1],
                frame_width,
                frame_height,
            )
        )

        x2, y2 = (
            self._map_inference_point_to_frame(
                box[2],
                box[3],
                frame_width,
                frame_height,
            )
        )

        return (
            min(x1, x2),
            min(y1, y2),
            max(x1, x2),
            max(y1, y2),
        )

    # =========================================================================
    # SEGMENTATION NORMALIZATION
    # =========================================================================

    @staticmethod
    def _as_tensor(
        value: Any,
        device: torch.device,
    ) -> torch.Tensor:

        if isinstance(
            value,
            torch.Tensor,
        ):
            return value

        if isinstance(
            value,
            np.ndarray,
        ):
            return torch.from_numpy(
                value
            ).to(device)

        raise TypeError(
            "Saída de segmentação incompatível."
        )

    def _decode_lane_output(
        self,
        lane_output: Any,
    ) -> np.ndarray:
        """
        Replica o contrato oficial lane_line_mask()
        do YOLOPv2, mas devolve HxW uint8/probability sem
        depender do código do repositório externo.
        """

        tensor = self._as_tensor(
            lane_output,
            self.device,
        )

        if tensor.ndim != 4:
            raise RuntimeError(
                "Saída lane do YOLOPv2 inválida: "
                f"{tuple(tensor.shape)}"
            )

        # Contrato oficial:
        #
        #   ll[:, :, 12:372, :]
        #
        # seguido por interpolação x2 e round.
        if (
            tensor.shape[2]
            >= LANE_OUTPUT_BOTTOM
            and tensor.shape[3]
            == self.input_width
        ):

            tensor = tensor[
                :,
                :,
                LANE_OUTPUT_TOP_CROP:
                LANE_OUTPUT_BOTTOM,
                :,
            ]

            tensor = torch.nn.functional.interpolate(
                tensor,
                scale_factor=2.0,
                mode="bilinear",
                align_corners=False,
            )

            tensor = torch.round(
                tensor
            )

        else:

            # Fallback defensivo para variantes exportadas.
            tensor = torch.sigmoid(
                tensor
            )

        while tensor.ndim > 2:
            if tensor.shape[1] == 1:
                tensor = tensor[:, 0]

            elif tensor.shape[0] == 1:
                tensor = tensor[0]

            else:
                tensor = tensor[0]

        result = tensor.float().detach().cpu().numpy()

        result = _finite_array(
            result
        )

        if result.ndim != 2:
            raise RuntimeError(
                "Máscara de lane deveria ser HxW: "
                f"{result.shape}"
            )

        result = np.clip(
            result,
            0.0,
            1.0,
        ).astype(
            np.float32,
            copy=False,
        )

        return result

    def _decode_drivable_output(
        self,
        drivable_output: Any,
    ) -> np.ndarray:
        """
        Replica driving_area_mask() do YOLOPv2.
        """

        tensor = self._as_tensor(
            drivable_output,
            self.device,
        )

        if tensor.ndim != 4:
            raise RuntimeError(
                "Saída drivable do YOLOPv2 inválida: "
                f"{tuple(tensor.shape)}"
            )

        if (
            tensor.shape[2]
            >= LANE_OUTPUT_BOTTOM
            and tensor.shape[3]
            == self.input_width
        ):

            tensor = tensor[
                :,
                :,
                LANE_OUTPUT_TOP_CROP:
                LANE_OUTPUT_BOTTOM,
                :,
            ]

            tensor = torch.nn.functional.interpolate(
                tensor,
                scale_factor=2.0,
                mode="bilinear",
                align_corners=False,
            )

        channels = tensor.shape[1]

        if channels < 1:
            raise RuntimeError(
                "Saída drivable sem canais."
            )

        if channels == 1:

            mask = torch.sigmoid(
                tensor
            )[:, 0]

        else:

            mask = torch.argmax(
                tensor,
                dim=1,
            ).float()

        mask = mask[0]

        result = (
            mask.float()
            .detach()
            .cpu()
            .numpy()
        )

        result = _finite_array(
            result
        )

        return (
            result > 0.5
        ).astype(
            np.uint8
        )

    # =========================================================================
    # MASK MAPPING
    # =========================================================================

    def _map_canonical_mask_to_frame(
        self,
        canonical_mask: np.ndarray,
        frame_width: int,
        frame_height: int,
    ) -> np.ndarray:

        canonical = np.asarray(
            canonical_mask
        ).astype(
            np.uint8,
            copy=False,
        )

        canonical = cv2.resize(
            canonical,
            (
                self.canonical_width,
                self.canonical_height,
            ),
            interpolation=cv2.INTER_NEAREST,
        )

        frame_mask = cv2.resize(
            canonical,
            (
                frame_width,
                frame_height,
            ),
            interpolation=cv2.INTER_NEAREST,
        )

        return frame_mask.astype(
            np.uint8,
            copy=False,
        )

    # =========================================================================
    # MASK CLEANING
    # =========================================================================

    def _clean_mask(
        self,
        mask: np.ndarray,
    ) -> np.ndarray:

        result = np.asarray(
            mask
        ).astype(
            np.uint8,
            copy=True,
        )

        if result.ndim != 2:
            raise ValueError(
                "Máscara deve ser 2D."
            )

        if self.morph_kernel > 1:

            kernel = np.ones(
                (
                    self.morph_kernel,
                    self.morph_kernel,
                ),
                dtype=np.uint8,
            )

            # Fechamento conservador.
            result = cv2.morphologyEx(
                result,
                cv2.MORPH_CLOSE,
                kernel,
                iterations=1,
            )

        if self.min_component_area > 1:

            labels_count, labels, stats, _ = (
                cv2.connectedComponentsWithStats(
                    result,
                    connectivity=8,
                )
            )

            cleaned = np.zeros_like(
                result
            )

            for label in range(
                1,
                labels_count,
            ):

                area = int(
                    stats[
                        label,
                        cv2.CC_STAT_AREA,
                    ]
                )

                if area >= self.min_component_area:

                    cleaned[
                        labels == label
                    ] = 1

            result = cleaned

        return result

    # =========================================================================
    # DRIVABLE GATE
    # =========================================================================

    def _drivable_probability_at(
        self,
        drivable_mask: Optional[np.ndarray],
        x: float,
        y: int,
    ) -> float:

        if drivable_mask is None:
            return 1.0

        height, width = (
            drivable_mask.shape
        )

        if (
            width <= 0
            or height <= 0
        ):
            return 1.0

        xi = int(
            np.clip(
                round(x),
                0,
                width - 1,
            )
        )

        yi = int(
            np.clip(
                y,
                0,
                height - 1,
            )
        )

        return float(
            drivable_mask[
                yi,
                xi,
            ]
        )

    # =========================================================================
    # ROW SEGMENTS
    # =========================================================================

    @staticmethod
    def _split_row_segments(
        xs: np.ndarray,
    ) -> List[np.ndarray]:

        if xs.size == 0:
            return []

        gaps = np.diff(
            xs
        )

        indices = (
            np.where(
                gaps > 2
            )[0]
            + 1
        )

        return list(
            np.split(
                xs,
                indices,
            )
        )

    def _extract_row_segments(
        self,
        probability: np.ndarray,
        lane_mask: np.ndarray,
        drivable_mask: Optional[np.ndarray],
    ) -> List[List[_RowSegment]]:

        height, width = (
            lane_mask.shape
        )

        rows: List[
            List[_RowSegment]
        ] = []

        for y in range(
            height - 1,
            -1,
            -self.row_step,
        ):

            xs = np.flatnonzero(
                lane_mask[y] > 0
            )

            if (
                xs.size
                < self.min_lane_pixels_per_row
            ):
                continue

            segments = (
                self._split_row_segments(
                    xs
                )
            )

            row_segments: List[
                _RowSegment
            ] = []

            for segment in segments:

                if (
                    segment.size
                    < self.min_lane_pixels_per_row
                ):
                    continue

                x_min = float(
                    segment[0]
                )

                x_max = float(
                    segment[-1]
                )

                x_center = float(
                    np.mean(
                        segment
                    )
                )

                lane_confidence = float(
                    np.mean(
                        probability[
                            y,
                            segment,
                        ]
                    )
                )

                if drivable_mask is None:

                    road_confidence = 1.0

                else:

                    road_confidence = float(
                        np.mean(
                            drivable_mask[
                                y,
                                segment,
                            ]
                        )
                    )

                row_segments.append(
                    _RowSegment(
                        y=int(y),
                        x_min=x_min,
                        x_max=x_max,
                        x_center=x_center,
                        confidence=_clip01(
                            lane_confidence
                        ),
                        road_confidence=_clip01(
                            road_confidence
                        ),
                        pixel_count=int(
                            segment.size
                        ),
                    )
                )

            if row_segments:

                row_segments.sort(
                    key=lambda item:
                    item.x_center
                )

                rows.append(
                    row_segments
                )

        return rows

    # =========================================================================
    # FRAME-LOCAL LANE ASSOCIATION
    # =========================================================================

    def _segment_distance(
        self,
        track: _LaneTrack,
        segment: _RowSegment,
    ) -> float:

        predicted = (
            track.predicted_x()
        )

        return abs(
            segment.x_center
            - predicted
        )

    def _associate_segments(
        self,
        rows: List[List[_RowSegment]],
    ) -> List[_LaneTrack]:

        tracks: List[
            _LaneTrack
        ] = []

        for segments in rows:

            used_tracks: set[int] = set()
            used_segments: set[int] = set()

            matches: List[
                Tuple[float, int, int]
            ] = []

            for segment_index, segment in enumerate(
                segments
            ):

                for track_index, track in enumerate(
                    tracks
                ):

                    if (
                        track_index
                        in used_tracks
                    ):
                        continue

                    distance = (
                        self._segment_distance(
                            track,
                            segment,
                        )
                    )

                    jump = (
                        self.max_tracking_jump
                        + min(
                            20.0,
                            segment.width,
                        )
                    )

                    if distance <= jump:

                        matches.append(
                            (
                                distance,
                                segment_index,
                                track_index,
                            )
                        )

            matches.sort(
                key=lambda item:
                item[0]
            )

            for (
                _distance,
                segment_index,
                track_index,
            ) in matches:

                if (
                    segment_index
                    in used_segments
                ):
                    continue

                if (
                    track_index
                    in used_tracks
                ):
                    continue

                segment = segments[
                    segment_index
                ]

                track = tracks[
                    track_index
                ]

                # Rejeição de saltos de inclinação absurdos.
                if (
                    track.previous_x is not None
                    and track.previous_y is not None
                ):

                    dy = (
                        track.last_y
                        - segment.y
                    )

                    if abs(dy) > 0:

                        previous_dx = (
                            track.last_x
                            - track.previous_x
                        )

                        new_dx = (
                            segment.x_center
                            - track.last_x
                        )

                        previous_slope = (
                            previous_dx
                            / max(
                                1.0,
                                abs(
                                    track.last_y
                                    - track.previous_y
                                ),
                            )
                        )

                        new_slope = (
                            new_dx
                            / max(
                                1.0,
                                abs(dy)
                            )
                        )

                        if (
                            abs(
                                new_slope
                                - previous_slope
                            )
                            > DEFAULT_MAX_SLOPE_CHANGE
                        ):
                            continue

                track.previous_x = (
                    track.last_x
                )

                track.previous_y = (
                    track.last_y
                )

                track.last_x = (
                    segment.x_center
                )

                track.last_y = (
                    segment.y
                )

                track.points.append(
                    (
                        segment.y,
                        segment.x_center,
                        segment.confidence,
                    )
                )

                track.confidence_sum += (
                    segment.confidence
                )

                track.road_confidence_sum += (
                    segment.road_confidence
                )

                track.pixel_sum += (
                    segment.pixel_count
                )

                used_segments.add(
                    segment_index
                )

                used_tracks.add(
                    track_index
                )

            # -----------------------------------------------------------------
            # Novas lanes.
            # -----------------------------------------------------------------

            for segment_index, segment in enumerate(
                segments
            ):

                if (
                    segment_index
                    in used_segments
                ):
                    continue

                if (
                    len(tracks)
                    >= self.max_lanes
                ):
                    break

                tracks.append(
                    _LaneTrack(
                        points=[
                            (
                                segment.y,
                                segment.x_center,
                                segment.confidence,
                            )
                        ],
                        last_x=segment.x_center,
                        last_y=segment.y,
                        previous_x=None,
                        previous_y=None,
                        confidence_sum=(
                            segment.confidence
                        ),
                        pixel_sum=(
                            segment.pixel_count
                        ),
                        road_confidence_sum=(
                            segment.road_confidence
                        ),
                    )
                )

        return tracks

    # =========================================================================
    # TRACK VALIDATION
    # =========================================================================

    def _validate_track(
        self,
        track: _LaneTrack,
    ) -> bool:

        if (
            len(track.points)
            < self.min_points_per_lane
        ):
            return False

        ys = np.asarray(
            [
                value[0]
                for value in track.points
            ],
            dtype=np.float32,
        )

        xs = np.asarray(
            [
                value[1]
                for value in track.points
            ],
            dtype=np.float32,
        )

        if not (
            np.all(
                np.isfinite(ys)
            )
            and np.all(
                np.isfinite(xs)
            )
        ):
            return False

        vertical_span = float(
            np.max(ys)
            - np.min(ys)
        )

        horizontal_span = float(
            np.max(xs)
            - np.min(xs)
        )

        if (
            vertical_span
            < self.min_lane_vertical_span
        ):
            return False

        if (
            horizontal_span
            < self.min_lane_span
            and vertical_span
            < self.min_lane_vertical_span
        ):
            return False

        mean_confidence = (
            track.confidence_sum
            / max(
                1,
                len(track.points),
            )
        )

        mean_road_confidence = (
            track.road_confidence_sum
            / max(
                1,
                len(track.points),
            )
        )

        structural_confidence = (
            0.75 * _clip01(
                mean_confidence
            )
            + 0.25 * _clip01(
                mean_road_confidence
            )
        )

        # Não eliminar linhas legítimas apenas porque a rede
        # classificou parcialmente sua região como estrada.
        return (
            structural_confidence
            >= max(
                0.15,
                self.lane_threshold * 0.50,
            )
        )

    # =========================================================================
    # LANE CONFIDENCE
    # =========================================================================

    @staticmethod
    def _lane_confidence(
        lane: Sequence[LanePoint],
    ) -> float:

        if not lane:
            return 0.0

        values = [
            _clip01(
                point.confidence
            )
            for point in lane
            if point.valid
        ]

        if not values:
            return 0.0

        confidence = float(
            np.mean(
                values
            )
        )

        count_score = _clip01(
            len(values)
            / 24.0
        )

        ys = np.asarray(
            [
                point.y
                for point in lane
                if point.valid
            ],
            dtype=np.float64,
        )

        if ys.size >= 2:

            span_score = _clip01(
                (
                    float(
                        np.max(ys)
                        - np.min(ys)
                    )
                )
                / 400.0
            )

        else:
            span_score = 0.0

        return _clip01(
            0.60 * confidence
            + 0.20 * count_score
            + 0.20 * span_score
        )

    # =========================================================================
    # TRACK -> LANE
    # =========================================================================

    def _track_to_lane(
        self,
        track: _LaneTrack,
        frame_width: int,
        frame_height: int,
    ) -> List[LanePoint]:

        points: List[
            LanePoint
        ] = []

        meta = (
            self.last_preprocess_meta
        )

        if meta is None:
            return points

        for (
            canonical_y,
            canonical_x,
            confidence,
        ) in track.points:

            frame_x, frame_y = (
                self._map_inference_point_to_frame(
                    canonical_x,
                    canonical_y,
                    frame_width,
                    frame_height,
                )
            )

            points.append(
                LanePoint(
                    x=frame_x,
                    y=frame_y,
                    confidence=_clip01(
                        confidence
                    ),
                    valid=True,
                )
            )

        points.sort(
            key=lambda point:
            point.y
        )

        # Remove exact/near duplicates.
        deduplicated: List[
            LanePoint
        ] = []

        last_y: Optional[float] = None

        for point in points:

            if (
                last_y is not None
                and abs(
                    point.y - last_y
                )
                < 0.5
            ):
                continue

            deduplicated.append(
                point
            )

            last_y = point.y

        return deduplicated

    # =========================================================================
    # LANE REFERENCE
    # =========================================================================

    @staticmethod
    def _lane_reference_x(
        lane: Sequence[LanePoint],
    ) -> float:

        valid = [
            point
            for point in lane
            if point.valid
            and math.isfinite(
                point.x
            )
            and math.isfinite(
                point.y
            )
        ]

        if not valid:
            return float(
                "inf"
            )

        valid.sort(
            key=lambda point:
            point.y,
            reverse=True,
        )

        sample = valid[
            : min(
                DEFAULT_SAMPLE_BOTTOM_ROWS,
                len(valid),
            )
        ]

        return float(
            np.mean(
                [
                    point.x
                    for point in sample
                ]
            )
        )

    # =========================================================================
    # PRIMARY LEFT / RIGHT
    # =========================================================================

    def _classify_primary_lanes(
        self,
        lanes: List[List[LanePoint]],
        frame_width: int,
    ) -> Tuple[
        List[LanePoint],
        List[LanePoint],
        List[List[LanePoint]],
    ]:

        if not lanes:
            return (
                [],
                [],
                [],
            )

        center_x = (
            float(frame_width)
            * 0.5
        )

        ordered = sorted(
            lanes,
            key=self._lane_reference_x,
        )

        left_candidates = [
            lane
            for lane in ordered
            if self._lane_reference_x(lane)
            < center_x
        ]

        right_candidates = [
            lane
            for lane in ordered
            if self._lane_reference_x(lane)
            >= center_x
        ]

        left_lane: List[
            LanePoint
        ] = []

        right_lane: List[
            LanePoint
        ] = []

        if left_candidates:

            left_lane = max(
                left_candidates,
                key=self._lane_reference_x,
            )

        if right_candidates:

            right_lane = min(
                right_candidates,
                key=self._lane_reference_x,
            )

        primary_ids = {
            id(left_lane)
            if left_lane
            else None,
            id(right_lane)
            if right_lane
            else None,
        }

        additional = [
            lane
            for lane in ordered
            if id(lane)
            not in primary_ids
        ]

        return (
            left_lane,
            right_lane,
            additional,
        )

    # =========================================================================
    # DRIVABLE AREA CONFIDENCE
    # =========================================================================

    def _calculate_lane_confidence_adjustment(
        self,
        lane: Sequence[LanePoint],
        drivable_mask: Optional[np.ndarray],
    ) -> float:

        if (
            not lane
            or drivable_mask is None
        ):
            return 1.0

        height, width = (
            drivable_mask.shape
        )

        values: List[
            float
        ] = []

        for point in lane:

            xi = int(
                np.clip(
                    round(point.x),
                    0,
                    width - 1,
                )
            )

            yi = int(
                np.clip(
                    round(point.y),
                    0,
                    height - 1,
                )
            )

            values.append(
                float(
                    drivable_mask[
                        yi,
                        xi,
                    ]
                )
            )

        if not values:
            return 1.0

        road_ratio = float(
            np.mean(
                values
            )
        )

        # Apenas como ajuste suave, nunca como veto absoluto.
        return _clip01(
            1.0
            + (
                road_ratio
                - 0.5
            )
            * 2.0
            * DEFAULT_DRIVABLE_CONFIDENCE_BONUS
        )

    # =========================================================================
    # RESULT
    # =========================================================================

    def _build_result(
        self,
        lanes: List[List[LanePoint]],
        objects: List[ObjectDetection],
        drivable_mask: Optional[np.ndarray],
        frame_width: int,
        frame_height: int,
    ) -> LaneDetectionResult:

        valid_lanes: List[
            List[LanePoint]
        ] = []

        for lane in lanes:

            points = [
                point
                for point in lane
                if point.valid
                and point.is_finite()
            ]

            if (
                len(points)
                < self.min_points_per_lane
            ):
                continue

            points.sort(
                key=lambda point:
                point.y
            )

            adjustment = (
                self._calculate_lane_confidence_adjustment(
                    points,
                    drivable_mask,
                )
            )

            if adjustment != 1.0:

                adjusted_points: List[
                    LanePoint
                ] = []

                for point in points:

                    adjusted_confidence = (
                        _clip01(
                            point.confidence
                            * adjustment
                        )
                    )

                    adjusted_points.append(
                        LanePoint(
                            x=point.x,
                            y=point.y,
                            confidence=(
                                adjusted_confidence
                            ),
                            valid=True,
                        )
                    )

                points = adjusted_points

            valid_lanes.append(
                points
            )

        valid_lanes.sort(
            key=self._lane_reference_x
        )

        (
            left_lane,
            right_lane,
            additional_lanes,
        ) = self._classify_primary_lanes(
            valid_lanes,
            frame_width,
        )

        confidences = [
            self._lane_confidence(
                lane
            )
            for lane in valid_lanes
        ]

        left_confidence = (
            self._lane_confidence(
                left_lane
            )
            if left_lane
            else 0.0
        )

        right_confidence = (
            self._lane_confidence(
                right_lane
            )
            if right_lane
            else 0.0
        )

        # ---------------------------------------------------------------------
        # Confiança global da percepção.
        # ---------------------------------------------------------------------

        if confidences:

            mean_lane_confidence = float(
                np.mean(
                    confidences
                )
            )

            lane_count_score = _clip01(
                len(confidences)
                / 4.0
            )

            perception_confidence = (
                0.75
                * mean_lane_confidence
                + 0.25
                * lane_count_score
            )

        else:

            perception_confidence = 0.0

        return LaneDetectionResult(
            lanes=valid_lanes,
            lane_confidences=confidences,
            current_lane_index=None,
            left_lane=left_lane,
            right_lane=right_lane,
            additional_lanes=additional_lanes,
            left_confidence=left_confidence,
            right_confidence=right_confidence,
            valid=bool(valid_lanes),
            num_lanes_detected=len(
                valid_lanes
            ),
            input_width=int(
                frame_width
            ),
            input_height=int(
                frame_height
            ),
            model_output_shape=(
                self.last_output_shapes[-1]
                if self.last_output_shapes
                else tuple()
            ),
            error=None,
            objects=objects,
            drivable_area_mask=drivable_mask,
            metadata={
                "device": self.get_device_name(),
                "fp16": self.fp16_active,
                "lane_count": len(
                    valid_lanes
                ),
                "lane_confidence": (
                    perception_confidence
                ),
                "vehicle_count": len(
                    [
                        obj
                        for obj in objects
                        if obj.is_vehicle
                    ]
                ),
                "object_count": len(
                    objects
                ),
            },
        )

    # =========================================================================
    # DIAGNOSTICS
    # =========================================================================

    def _update_diagnostics(
        self,
        probability: np.ndarray,
        lane_mask: np.ndarray,
        drivable_mask: Optional[np.ndarray],
        rows: List[List[_RowSegment]],
        tracks: List[_LaneTrack],
        result: LaneDetectionResult,
        inference_ms: float,
    ) -> None:

        active_pixels = int(
            np.count_nonzero(
                lane_mask
            )
        )

        road_pixels = (
            int(
                np.count_nonzero(
                    drivable_mask
                )
            )
            if drivable_mask is not None
            else 0
        )

        self.last_diagnostics = {
            "device": self.get_device_name(),
            "fp16": self.fp16_active,
            "inference_ms": float(
                inference_ms
            ),
            "probability_shape": tuple(
                int(v)
                for v in probability.shape
            ),
            "lane_mask_shape": tuple(
                int(v)
                for v
                in lane_mask.shape
            ),
            "drivable_mask_shape": (
                tuple(
                    int(v)
                    for v
                    in drivable_mask.shape
                )
                if drivable_mask is not None
                else None
            ),
            "active_lane_pixels": active_pixels,
            "active_lane_pixel_ratio": (
                active_pixels
                / max(
                    1,
                    lane_mask.size,
                )
            ),
            "drivable_pixels": road_pixels,
            "row_count": len(
                rows
            ),
            "segment_count": sum(
                len(row)
                for row in rows
            ),
            "raw_tracks": len(
                tracks
            ),
            "valid_lanes": (
                result.num_lanes_detected
            ),
            "vehicle_count": (
                result.vehicle_count
            ),
            "object_count": (
                result.object_count
            ),
            "model_output_shapes": list(
                self.last_output_shapes
            ),
        }

    # =========================================================================
    # DETECT
    # =========================================================================

    def detect(
        self,
        frame: np.ndarray,
    ) -> LaneDetectionResult:
        """
        Pipeline completo de uma imagem/frame.

        Falha em objeto não derruba lanes.
        Falha em drivable area não derruba lanes.
        Falha geral do modelo gera resultado inválido.
        """

        self.last_error = None
        self.last_diagnostics = {}

        try:

            frame_width, frame_height = (
                self._validate_frame(
                    frame
                )
            )

            start = time.perf_counter()

            (
                prediction_output,
                drivable_output,
                lane_output,
            ) = self.infer(
                frame
            )

            inference_ms = (
                time.perf_counter()
                - start
            ) * 1000.0

            # -----------------------------------------------------------------
            # Lane segmentation
            # -----------------------------------------------------------------

            canonical_lane_probability = (
                self._decode_lane_output(
                    lane_output
                )
            )

            canonical_lane_mask = (
                canonical_lane_probability
                >= self.lane_threshold
            ).astype(
                np.uint8
            )

            canonical_lane_mask = (
                self._clean_mask(
                    canonical_lane_mask
                )
            )

            # -----------------------------------------------------------------
            # Drivable segmentation
            # -----------------------------------------------------------------

            try:

                canonical_drivable = (
                    self._decode_drivable_output(
                        drivable_output
                    )
                )

                drivable_frame = (
                    self._map_canonical_mask_to_frame(
                        canonical_drivable,
                        frame_width,
                        frame_height,
                    )
                )

            except Exception as exc:

                drivable_frame = None

                self.last_diagnostics[
                    "drivable_error"
                ] = (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

            # -----------------------------------------------------------------
            # Convert lane segmentation to actual frame coordinates.
            # -----------------------------------------------------------------

            lane_probability = (
                self._map_canonical_mask_to_frame(
                    canonical_lane_probability,
                    frame_width,
                    frame_height,
                )
                .astype(
                    np.float32,
                    copy=False,
                )
            )

            lane_mask = (
                self._map_canonical_mask_to_frame(
                    canonical_lane_mask,
                    frame_width,
                    frame_height,
                )
            )

            lane_mask = self._clean_mask(
                lane_mask
            )

            self.last_lane_probability = (
                lane_probability
            )

            self.last_lane_mask = (
                lane_mask
            )

            self.last_drivable_mask = (
                drivable_frame
            )

            # -----------------------------------------------------------------
            # Extract row segments.
            # -----------------------------------------------------------------

            rows = (
                self._extract_row_segments(
                    lane_probability,
                    lane_mask,
                    drivable_frame,
                )
            )

            tracks = (
                self._associate_segments(
                    rows
                )
            )

            lanes: List[
                List[LanePoint]
            ] = []

            for track in tracks:

                if not self._validate_track(
                    track
                ):
                    continue

                lane = (
                    self._track_to_lane(
                        track,
                        frame_width,
                        frame_height,
                    )
                )

                if (
                    len(lane)
                    >= self.min_points_per_lane
                ):
                    lanes.append(
                        lane
                    )

            # -----------------------------------------------------------------
            # Objects.
            # -----------------------------------------------------------------

            objects = (
                self._decode_objects(
                    prediction_output,
                    frame_width,
                    frame_height,
                )
            )

            # -----------------------------------------------------------------
            # Public result.
            # -----------------------------------------------------------------

            result = (
                self._build_result(
                    lanes=lanes,
                    objects=objects,
                    drivable_mask=drivable_frame,
                    frame_width=frame_width,
                    frame_height=frame_height,
                )
            )

            self._update_diagnostics(
                probability=lane_probability,
                lane_mask=lane_mask,
                drivable_mask=drivable_frame,
                rows=rows,
                tracks=tracks,
                result=result,
                inference_ms=inference_ms,
            )

            result.metadata.update(
                self.last_diagnostics
            )

            self.last_result = result

            return result

        except Exception as exc:

            self.last_error = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            logger.exception(
                "[YOLOPv2] Falha durante detecção."
            )

            width = (
                int(frame.shape[1])
                if isinstance(
                    frame,
                    np.ndarray,
                )
                and frame.ndim >= 2
                else self.input_width
            )

            height = (
                int(frame.shape[0])
                if isinstance(
                    frame,
                    np.ndarray,
                )
                and frame.ndim >= 2
                else self.input_height
            )

            result = LaneDetectionResult(
                lanes=[],
                lane_confidences=[],
                current_lane_index=None,
                left_lane=[],
                right_lane=[],
                additional_lanes=[],
                left_confidence=0.0,
                right_confidence=0.0,
                valid=False,
                num_lanes_detected=0,
                input_width=width,
                input_height=height,
                model_output_shape=(
                    self.last_output_shapes[-1]
                    if self.last_output_shapes
                    else tuple()
                ),
                error=self.last_error,
                objects=[],
                drivable_area_mask=None,
                metadata={
                    "device": self.get_device_name(),
                    "fatal_error": self.last_error,
                },
            )

            self.last_result = result

            return result

    # =========================================================================
    # IMAGE UTILITY
    # =========================================================================

    @staticmethod
    def load_image(
        path: str | Path,
    ) -> np.ndarray:

        path = Path(path)

        if not path.is_file():
            raise FileNotFoundError(
                f"Imagem não encontrada: {path}"
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
                f"Não foi possível carregar imagem: {path}"
            )

        return image

    @staticmethod
    def save_image(
        path: str | Path,
        image: np.ndarray,
    ) -> None:

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        suffix = (
            path.suffix
            if path.suffix
            else ".png"
        )

        ok, encoded = cv2.imencode(
            suffix,
            image,
        )

        if not ok:
            raise RuntimeError(
                f"Falha ao codificar imagem: {path}"
            )

        encoded.tofile(
            str(path)
        )


# =============================================================================
# PUBLIC COMPATIBILITY NAME
# =============================================================================

YOLOPDetector = YOLOPLaneDetector


# =============================================================================
# FACTORY
# =============================================================================


def create_default_detector(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    **kwargs: Any,
) -> YOLOPLaneDetector:

    return YOLOPLaneDetector(
        model_path=model_path,
        **kwargs,
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "ObjectDetection",
    "LanePoint",
    "LaneDetectionResult",
    "YOLOPDetector",
    "YOLOPLaneDetector",
    "create_default_detector",
]

