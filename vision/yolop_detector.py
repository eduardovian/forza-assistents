"""
vision/yolop_detector.py

Forza Assistents
================

Detector de percepção baseado em YOLOPv2 / TorchScript.

Pipeline:

    frame BGR
        │
        ▼
    validação
        │
        ▼
    canonicalização
        │
        ▼
    letterbox 640x640
        │
        ▼
    TorchScript / CUDA / FP16
        │
        ├───────────────┐
        │               │
        ▼               ▼
    detection       segmentation
        │               │
        │        ┌──────┴──────┐
        │        ▼             ▼
        │    drivable         lanes
        │        │             │
        │        │        probabilidade
        │        │             │
        │        │          máscara
        │        │             │
        │        │       segmentos
        │        │             │
        └────────┴─────────────┤
                               ▼
                    LaneDetectionResult

Responsabilidades deste módulo:

    - inferência YOLOPv2;
    - detecção de lanes;
    - detecção de objetos;
    - máscara de área dirigível;
    - associação espacial das lanes dentro do frame;
    - conversão rigorosa de coordenadas;
    - diagnóstico.

Este módulo NÃO executa:

    - tracking temporal entre frames;
    - fitting polinomial;
    - geometria;
    - projeção;
    - LaneAssignment;
    - decisão ADAS;
    - controle.

Essas responsabilidades pertencem às camadas posteriores.

IMPORTANTE:

Todos os LanePoint produzidos por este módulo estão em
coordenadas do FRAME ORIGINAL recebido por detect().

Não existe segundo mapeamento depois da extração dos
segmentos.

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

from .lane_types import LanePoint


logger = logging.getLogger(__name__)


# =============================================================================
# PATHS
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


# =============================================================================
# MODEL CONTRACT
# =============================================================================

YOLOPV2_INPUT_WIDTH = 640
YOLOPV2_INPUT_HEIGHT = 640

CANONICAL_WIDTH = 1280
CANONICAL_HEIGHT = 720

YOLOPV2_STRIDE = 32


# =============================================================================
# DEFAULT PARAMETERS
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

DEFAULT_USE_FP16 = True

DEFAULT_DRIVABLE_GATE_MIN = 0.15
DEFAULT_DRIVABLE_CONFIDENCE_BONUS = 0.10


# =============================================================================
# COCO
# =============================================================================

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

VEHICLE_CLASS_IDS = frozenset(
    {
        2,  # car
        3,  # motorcycle
        5,  # bus
        7,  # truck
    }
)


# =============================================================================
# NUMERIC UTILITIES
# =============================================================================

def _finite(value: Any) -> Optional[float]:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(value):
        return None

    return value


def _clip01(value: Any) -> float:
    value = _finite(value)

    if value is None:
        return 0.0

    return max(
        0.0,
        min(1.0, value),
    )


def _shape_of(value: Any) -> Tuple[int, ...]:
    if isinstance(value, torch.Tensor):
        return tuple(
            int(v)
            for v in value.shape
        )

    if isinstance(value, np.ndarray):
        return tuple(
            int(v)
            for v in value.shape
        )

    return tuple()


def _collect_shapes(
    value: Any,
) -> List[Tuple[int, ...]]:
    """
    Coleta recursivamente os shapes reais de uma saída
    TorchScript.

    YOLOPv2 pode retornar estruturas aninhadas.
    """

    shapes: List[Tuple[int, ...]] = []

    if isinstance(value, torch.Tensor):
        shapes.append(_shape_of(value))
        return shapes

    if isinstance(value, np.ndarray):
        shapes.append(_shape_of(value))
        return shapes

    if isinstance(value, (tuple, list)):
        for item in value:
            shapes.extend(
                _collect_shapes(item)
            )

    elif isinstance(value, dict):
        for item in value.values():
            shapes.extend(
                _collect_shapes(item)
            )

    return shapes


def _flatten_tensors(
    value: Any,
) -> List[torch.Tensor]:
    tensors: List[torch.Tensor] = []

    if isinstance(value, torch.Tensor):
        tensors.append(value)

    elif isinstance(value, (tuple, list)):
        for item in value:
            tensors.extend(
                _flatten_tensors(item)
            )

    elif isinstance(value, dict):
        for item in value.values():
            tensors.extend(
                _flatten_tensors(item)
            )

    return tensors


# =============================================================================
# OBJECT DETECTION
# =============================================================================

@dataclass(frozen=True)
class ObjectDetection:
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
            return COCO_CLASS_NAMES[
                self.class_id
            ]

        return f"class_{self.class_id}"

    @property
    def is_vehicle(self) -> bool:
        return (
            self.class_id
            in VEHICLE_CLASS_IDS
        )

    @property
    def center_x(self) -> float:
        return (
            self.x1 + self.x2
        ) * 0.5

    @property
    def center_y(self) -> float:
        return (
            self.y1 + self.y2
        ) * 0.5

    @property
    def width(self) -> float:
        return max(
            0.0,
            self.x2 - self.x1,
        )

    @property
    def height(self) -> float:
        return max(
            0.0,
            self.y2 - self.y1,
        )

    @property
    def area(self) -> float:
        return (
            self.width
            * self.height
        )

    @property
    def bottom_y(self) -> float:
        return self.y2


# =============================================================================
# PUBLIC RESULT
# =============================================================================

@dataclass
class LaneDetectionResult:
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

    input_width: int = (
        YOLOPV2_INPUT_WIDTH
    )

    input_height: int = (
        YOLOPV2_INPUT_HEIGHT
    )

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
        return (
            self.num_lanes_detected > 0
        )

    @property
    def has_current_lane(self) -> bool:
        return (
            self.current_lane_index
            is not None
            and bool(self.left_lane)
            and bool(self.right_lane)
        )

    @property
    def vehicle_detections(
        self,
    ) -> List[ObjectDetection]:
        return [
            obj
            for obj in self.objects
            if obj.is_vehicle
        ]

    @property
    def vehicle_count(self) -> int:
        return len(
            self.vehicle_detections
        )

    @property
    def object_count(self) -> int:
        return len(self.objects)


# =============================================================================
# INTERNAL TYPES
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


@dataclass
class _LaneTrack:
    points: List[
        Tuple[int, float, float]
    ] = field(
        default_factory=list
    )

    last_x: float = 0.0
    last_y: int = 0

    confidence_sum: float = 0.0
    road_confidence_sum: float = 0.0

    def add(
        self,
        y: int,
        x: float,
        confidence: float,
        road_confidence: float,
    ) -> None:
        self.points.append(
            (
                int(y),
                float(x),
                float(confidence),
            )
        )

        self.last_x = float(x)
        self.last_y = int(y)

        self.confidence_sum += (
            float(confidence)
        )

        self.road_confidence_sum += (
            float(road_confidence)
        )


@dataclass(frozen=True)
class _PreprocessMeta:
    original_width: int
    original_height: int

    canonical_width: int
    canonical_height: int

    scale: float

    resized_width: int
    resized_height: int

    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int


# =============================================================================
# DETECTOR
# =============================================================================

class YOLOPLaneDetector:
    """
    YOLOPv2 TorchScript detector.

    Design goals:

        deterministic
        bounded
        fail-soft
        coordinate-safe
        GPU optimized
        downstream compatible

    O detector não mantém histórico temporal.
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
        use_fp16: bool = DEFAULT_USE_FP16,
        device: Optional[str] = None,
        **_: Any,
    ) -> None:

        self.model_path = Path(
            model_path
        )

        if (
            not self.model_path.exists()
            and self.model_path
            == LEGACY_MODEL_PATH
            and DEFAULT_MODEL_PATH.exists()
        ):
            self.model_path = (
                DEFAULT_MODEL_PATH
            )

        self.input_width = max(
            32,
            int(input_width),
        )

        self.input_height = max(
            32,
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
            2,
            int(max_lanes),
        )

        self.use_fp16 = bool(
            use_fp16
        )

        if device is None:
            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        else:
            self.device = torch.device(
                device
            )

        if (
            self.device.type == "cuda"
            and not torch.cuda.is_available()
        ):
            logger.warning(
                "CUDA solicitado, mas não "
                "está disponível. Usando CPU."
            )

            self.device = torch.device(
                "cpu"
            )

        self.model: Optional[
            torch.jit.ScriptModule
        ] = None

        self.loaded = False

        self.fp16_active = False

        self.last_error: Optional[str] = None

        self.last_result: Optional[
            LaneDetectionResult
        ] = None

        self.last_output_shapes: List[
            Tuple[int, ...]
        ] = []

        self.last_diagnostics: Dict[
            str,
            Any,
        ] = {}

        self._warmed_up = False

    # =========================================================================
    # MODEL
    # =========================================================================

    def model_exists(self) -> bool:
        return self.model_path.is_file()

    def load_model(self) -> bool:

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

        try:
            logger.info(
                "[YOLOPv2] Carregando: %s",
                self.model_path,
            )

            model = torch.jit.load(
                str(self.model_path),
                map_location=self.device,
            )

            model.eval()

            if self.device.type == "cuda":
                model = model.cuda()

                if self.use_fp16:
                    model = model.half()
                    self.fp16_active = True

            else:
                self.fp16_active = False

            self.model = model
            self.loaded = True

            logger.info(
                "[YOLOPv2] Device: %s",
                self.get_device_name(),
            )

            logger.info(
                "[YOLOPv2] FP16: %s",
                self.fp16_active,
            )

            self._warmup()

            return True

        except Exception as exc:

            self.last_error = (
                f"{type(exc).__name__}: {exc}"
            )

            logger.exception(
                "[YOLOPv2] Falha ao carregar modelo."
            )

            self.model = None
            self.loaded = False
            self.fp16_active = False

            return False

    def _warmup(self) -> None:

        if (
            self._warmed_up
            or self.model is None
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
                dtype=dtype,
                device=self.device,
            )

            with torch.inference_mode():
                _ = self.model(tensor)

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            self._warmed_up = True

        except Exception:

            logger.exception(
                "[YOLOPv2] Warmup falhou."
            )

            self._warmed_up = False

    # =========================================================================
    # DEVICE
    # =========================================================================

    def get_device_name(self) -> str:

        if self.device.type == "cuda":

            try:
                return torch.cuda.get_device_name(
                    self.device
                )
            except Exception:
                return "CUDA"

        return "CPU"

    # =========================================================================
    # PREPROCESS
    # =========================================================================

    def _prepare_frame(
        self,
        frame: np.ndarray,
    ) -> Tuple[
        np.ndarray,
        _PreprocessMeta,
    ]:

        if not isinstance(
            frame,
            np.ndarray,
        ):
            raise TypeError(
                "Frame deve ser numpy.ndarray."
            )

        if frame.ndim != 3:
            raise ValueError(
                "Frame deve possuir formato HxWxC."
            )

        if frame.shape[2] != 3:
            raise ValueError(
                "Frame deve possuir 3 canais BGR."
            )

        height, width = frame.shape[:2]

        if width <= 0 or height <= 0:
            raise ValueError(
                "Frame possui dimensões inválidas."
            )

        if not np.isfinite(
            frame.astype(
                np.float32,
                copy=False,
            )
        ).all():
            raise ValueError(
                "Frame contém valores não finitos."
            )

        # -------------------------------------------------------------
        # Canonicalização.
        #
        # Mantemos aspect ratio e adicionamos padding.
        # -------------------------------------------------------------

        scale = min(
            CANONICAL_WIDTH / width,
            CANONICAL_HEIGHT / height,
        )

        resized_width = max(
            1,
            int(round(width * scale)),
        )

        resized_height = max(
            1,
            int(round(height * scale)),
        )

        resized = cv2.resize(
            frame,
            (
                resized_width,
                resized_height,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

        pad_left = (
            CANONICAL_WIDTH
            - resized_width
        ) // 2

        pad_top = (
            CANONICAL_HEIGHT
            - resized_height
        ) // 2

        pad_right = (
            CANONICAL_WIDTH
            - resized_width
            - pad_left
        )

        pad_bottom = (
            CANONICAL_HEIGHT
            - resized_height
            - pad_top
        )

        canonical = cv2.copyMakeBorder(
            resized,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )

        # -------------------------------------------------------------
        # Letterbox para input do modelo.
        # -------------------------------------------------------------

        model_scale = min(
            self.input_width
            / CANONICAL_WIDTH,
            self.input_height
            / CANONICAL_HEIGHT,
        )

        model_width = max(
            1,
            int(
                round(
                    CANONICAL_WIDTH
                    * model_scale
                )
            ),
        )

        model_height = max(
            1,
            int(
                round(
                    CANONICAL_HEIGHT
                    * model_scale
                )
            ),
        )

        model_image = cv2.resize(
            canonical,
            (
                model_width,
                model_height,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

        model_pad_left = (
            self.input_width
            - model_width
        ) // 2

        model_pad_top = (
            self.input_height
            - model_height
        ) // 2

        model_pad_right = (
            self.input_width
            - model_width
            - model_pad_left
        )

        model_pad_bottom = (
            self.input_height
            - model_height
            - model_pad_top
        )

        model_image = cv2.copyMakeBorder(
            model_image,
            model_pad_top,
            model_pad_bottom,
            model_pad_left,
            model_pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )

        rgb = cv2.cvtColor(
            model_image,
            cv2.COLOR_BGR2RGB,
        )

        tensor = (
            rgb.astype(
                np.float32
            )
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

        meta = _PreprocessMeta(
            original_width=width,
            original_height=height,
            canonical_width=CANONICAL_WIDTH,
            canonical_height=CANONICAL_HEIGHT,
            scale=scale,
            resized_width=resized_width,
            resized_height=resized_height,
            pad_left=pad_left,
            pad_top=pad_top,
            pad_right=pad_right,
            pad_bottom=pad_bottom,
        )

        return tensor, meta

    def preprocess(
        self,
        frame: np.ndarray,
    ) -> np.ndarray:

        tensor, _ = self._prepare_frame(
            frame
        )

        return tensor

    # =========================================================================
    # INFERENCE
    # =========================================================================

    @torch.inference_mode()
    def infer(
        self,
        frame: np.ndarray,
    ) -> Any:

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

        tensor_np, _ = (
            self._prepare_frame(frame)
        )

        tensor = torch.from_numpy(
            tensor_np
        ).to(
            self.device,
            non_blocking=True,
        )

        if self.fp16_active:
            tensor = tensor.half()

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        outputs = self.model(tensor)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        inference_ms = (
            time.perf_counter()
            - start
        ) * 1000.0

        self.last_output_shapes = (
            _collect_shapes(outputs)
        )

        self.last_diagnostics[
            "inference_ms"
        ] = inference_ms

        return outputs

    # =========================================================================
    # OUTPUT EXTRACTION
    # =========================================================================

    @staticmethod
    def _tensor_to_numpy(
        tensor: torch.Tensor,
    ) -> np.ndarray:

        return (
            tensor.detach()
            .float()
            .cpu()
            .numpy()
        )

    @staticmethod
    def _find_segmentation_tensor(
        tensors: Sequence[torch.Tensor],
        preferred_channels: Sequence[int],
    ) -> Optional[torch.Tensor]:

        candidates = []

        for tensor in tensors:

            if tensor.ndim != 4:
                continue

            channels = int(
                tensor.shape[1]
            )

            height = int(
                tensor.shape[2]
            )

            width = int(
                tensor.shape[3]
            )

            if (
                height < 32
                or width < 32
            ):
                continue

            if channels in preferred_channels:
                candidates.append(
                    tensor
                )

        if not candidates:
            return None

        # Preferência:
        # maior resolução espacial.
        candidates.sort(
            key=lambda tensor: (
                int(tensor.shape[2])
                * int(tensor.shape[3]),
                -abs(
                    int(tensor.shape[1])
                    - preferred_channels[0]
                ),
            ),
            reverse=True,
        )

        return candidates[0]

    @staticmethod
    def _find_lane_segmentation(
        outputs: Any,
    ) -> Optional[torch.Tensor]:

        tensors = _flatten_tensors(
            outputs
        )

        # YOLOPv2 normalmente usa [1, 2, H, W].
        return (
            YOLOPLaneDetector
            ._find_segmentation_tensor(
                tensors,
                (2,),
            )
        )

    @staticmethod
    def _find_drivable_segmentation(
        outputs: Any,
    ) -> Optional[torch.Tensor]:

        tensors = _flatten_tensors(
            outputs
        )

        return (
            YOLOPLaneDetector
            ._find_segmentation_tensor(
                tensors,
                (2,),
            )
        )

    # =========================================================================
    # SEGMENTATION
    # =========================================================================

    def _segmentation_probability(
        self,
        segmentation: np.ndarray,
    ) -> np.ndarray:

        if segmentation.ndim != 4:
            raise ValueError(
                "Segmentation deve ser [N,C,H,W]."
            )

        if segmentation.shape[0] < 1:
            raise ValueError(
                "Batch vazio."
            )

        channels = int(
            segmentation.shape[1]
        )

        if channels < 2:
            raise ValueError(
                "Segmentation deve possuir "
                "pelo menos dois canais."
            )

        logits = segmentation[0]

        logits = np.asarray(
            logits,
            dtype=np.float32,
        )

        logits = np.nan_to_num(
            logits,
            nan=0.0,
            posinf=50.0,
            neginf=-50.0,
        )

        logits -= np.max(
            logits,
            axis=0,
            keepdims=True,
        )

        exp_logits = np.exp(
            np.clip(
                logits,
                -50.0,
                50.0,
            )
        )

        denominator = np.sum(
            exp_logits,
            axis=0,
            keepdims=True,
        )

        denominator = np.maximum(
            denominator,
            1e-8,
        )

        probabilities = (
            exp_logits
            / denominator
        )

        return np.clip(
            probabilities[1],
            0.0,
            1.0,
        )

    def _resize_probability_to_frame(
        self,
        probability: np.ndarray,
        frame_width: int,
        frame_height: int,
    ) -> np.ndarray:

        if probability.size == 0:
            return np.zeros(
                (
                    frame_height,
                    frame_width,
                ),
                dtype=np.float32,
            )

        return cv2.resize(
            probability,
            (
                frame_width,
                frame_height,
            ),
            interpolation=cv2.INTER_LINEAR,
        ).astype(
            np.float32,
            copy=False,
        )

    def _build_lane_mask(
        self,
        probability: np.ndarray,
    ) -> np.ndarray:

        mask = (
            probability
            >= self.lane_threshold
        ).astype(
            np.uint8
        )

        if DEFAULT_MORPH_KERNEL >= 3:

            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (
                    DEFAULT_MORPH_KERNEL,
                    DEFAULT_MORPH_KERNEL,
                ),
            )

            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_OPEN,
                kernel,
            )

            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_CLOSE,
                kernel,
            )

        return mask

    def _build_drivable_mask(
        self,
        probability: np.ndarray,
    ) -> np.ndarray:

        return (
            probability
            >= DEFAULT_DRIVABLE_GATE_MIN
        ).astype(
            np.uint8
        )

    # =========================================================================
    # ROW SEGMENTS
    # =========================================================================

    @staticmethod
    def _split_contiguous(
        xs: np.ndarray,
        max_gap: int = 3,
    ) -> List[np.ndarray]:

        if xs.size == 0:
            return []

        gaps = np.diff(xs)

        split_indices = (
            np.where(
                gaps > max_gap
            )[0]
            + 1
        )

        return list(
            np.split(
                xs,
                split_indices,
            )
        )

    def _extract_row_segments(
        self,
        lane_mask: np.ndarray,
        lane_probability: np.ndarray,
        drivable_probability: Optional[np.ndarray],
    ) -> List[
        _RowSegment
    ]:

        height, width = (
            lane_mask.shape
        )

        segments: List[
            _RowSegment
        ] = []

        # Importante:
        #
        # lane_mask já está no FRAME ORIGINAL.
        #
        # Portanto x/y produzidos aqui também estão
        # no FRAME ORIGINAL.
        #

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

            for group in self._split_contiguous(
                xs
            ):

                if (
                    group.size
                    < self.min_lane_pixels_per_row
                ):
                    continue

                x_min = float(
                    group[0]
                )

                x_max = float(
                    group[-1]
                )

                x_center = (
                    x_min
                    + x_max
                ) * 0.5

                confidence = float(
                    np.mean(
                        lane_probability[
                            y,
                            group,
                        ]
                    )
                )

                if drivable_probability is not None:
                    road_confidence = float(
                        np.mean(
                            drivable_probability[
                                y,
                                group,
                            ]
                        )
                    )
                else:
                    road_confidence = 0.0

                segments.append(
                    _RowSegment(
                        y=int(y),
                        x_min=x_min,
                        x_max=x_max,
                        x_center=x_center,
                        confidence=_clip01(
                            confidence
                        ),
                        road_confidence=_clip01(
                            road_confidence
                        ),
                        pixel_count=int(
                            group.size
                        ),
                    )
                )

        return segments

    # =========================================================================
    # LANE ASSOCIATION
    # =========================================================================

    def _associate_segments(
        self,
        segments: Sequence[
            _RowSegment
        ],
        frame_width: int,
    ) -> List[
        _LaneTrack
    ]:

        if not segments:
            return []

        rows: Dict[
            int,
            List[_RowSegment]
        ] = {}

        for segment in segments:
            rows.setdefault(
                segment.y,
                []
            ).append(segment)

        tracks: List[
            _LaneTrack
        ] = []

        max_jump = max(
            DEFAULT_MAX_TRACKING_JUMP,
            frame_width * 0.035,
        )

        for y in sorted(
            rows.keys(),
            reverse=True,
        ):

            candidates = sorted(
                rows[y],
                key=lambda segment:
                    segment.x_center,
            )

            if not tracks:

                for candidate in candidates:
                    track = _LaneTrack()

                    track.add(
                        candidate.y,
                        candidate.x_center,
                        candidate.confidence,
                        candidate.road_confidence,
                    )

                    tracks.append(
                        track
                    )

                continue

            used_tracks = set()

            for candidate in candidates:

                best_index = None
                best_distance = float(
                    "inf"
                )

                for index, track in enumerate(
                    tracks
                ):

                    if index in used_tracks:
                        continue

                    distance = abs(
                        candidate.x_center
                        - track.last_x
                    )

                    if distance > max_jump:
                        continue

                    # Penalização leve para candidatos
                    # incompatíveis verticalmente.
                    score = distance

                    if score < best_distance:
                        best_distance = score
                        best_index = index

                if best_index is None:

                    if len(tracks) >= self.max_lanes:
                        continue

                    track = _LaneTrack()

                    track.add(
                        candidate.y,
                        candidate.x_center,
                        candidate.confidence,
                        candidate.road_confidence,
                    )

                    tracks.append(
                        track
                    )

                else:

                    track = tracks[
                        best_index
                    ]

                    track.add(
                        candidate.y,
                        candidate.x_center,
                        candidate.confidence,
                        candidate.road_confidence,
                    )

                    used_tracks.add(
                        best_index
                    )

        return tracks

    # =========================================================================
    # TRACK → LANE POINTS
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

        for (
            frame_y,
            frame_x,
            confidence,
        ) in track.points:

            # =========================================================
            # CRÍTICO:
            #
            # frame_x/frame_y JÁ estão nas coordenadas do frame
            # original.
            #
            # NÃO chamar:
            #
            #     _map_inference_point_to_frame()
            #
            # novamente.
            #
            # Este era o bug que produzia x=1989.
            # =========================================================

            x = float(
                np.clip(
                    frame_x,
                    0.0,
                    float(
                        frame_width - 1
                    ),
                )
            )

            y = float(
                np.clip(
                    frame_y,
                    0.0,
                    float(
                        frame_height - 1
                    ),
                )
            )

            confidence = _clip01(
                confidence
            )

            points.append(
                LanePoint(
                    x=x,
                    y=y,
                    confidence=confidence,
                    valid=True,
                )
            )

        points.sort(
            key=lambda point:
                point.y
        )

        # Remove duplicatas de Y.
        result: List[
            LanePoint
        ] = []

        last_y: Optional[
            float
        ] = None

        for point in points:

            if (
                last_y is not None
                and abs(
                    point.y - last_y
                ) < 0.5
            ):
                continue

            result.append(
                point
            )

            last_y = point.y

        return result

    # =========================================================================
    # LANE VALIDATION
    # =========================================================================

    @staticmethod
    def _lane_quality(
        lane: Sequence[LanePoint],
        frame_width: int,
        frame_height: int,
    ) -> float:

        if not lane:
            return 0.0

        if len(lane) < 2:
            return 0.0

        xs = np.asarray(
            [
                point.x
                for point in lane
            ],
            dtype=np.float32,
        )

        ys = np.asarray(
            [
                point.y
                for point in lane
            ],
            dtype=np.float32,
        )

        if not np.isfinite(xs).all():
            return 0.0

        if not np.isfinite(ys).all():
            return 0.0

        vertical_span = (
            float(
                ys.max()
                - ys.min()
            )
        )

        horizontal_span = (
            float(
                xs.max()
                - xs.min()
            )
        )

        confidence = float(
            np.mean(
                [
                    point.confidence
                    for point in lane
                ]
            )
        )

        vertical_score = _clip01(
            vertical_span
            / max(
                1.0,
                frame_height * 0.35,
            )
        )

        horizontal_score = _clip01(
            horizontal_span
            / max(
                1.0,
                frame_width * 0.02,
            )
        )

        return _clip01(
            0.55 * confidence
            + 0.30 * vertical_score
            + 0.15 * horizontal_score
        )

    def _validate_lane(
        self,
        lane: Sequence[LanePoint],
        frame_width: int,
        frame_height: int,
    ) -> bool:

        if len(lane) < self.min_points_per_lane:
            return False

        xs = [
            point.x
            for point in lane
        ]

        ys = [
            point.y
            for point in lane
        ]

        if not xs or not ys:
            return False

        if not all(
            math.isfinite(x)
            for x in xs
        ):
            return False

        if not all(
            math.isfinite(y)
            for y in ys
        ):
            return False

        if any(
            x < 0.0
            or x >= frame_width
            for x in xs
        ):
            return False

        if any(
            y < 0.0
            or y >= frame_height
            for y in ys
        ):
            return False

        vertical_span = (
            max(ys)
            - min(ys)
        )

        if (
            vertical_span
            < DEFAULT_MIN_LANE_VERTICAL_SPAN
        ):
            return False

        return True

    # =========================================================================
    # SORTING / CLASSIFICATION
    # =========================================================================

    @staticmethod
    def _lane_reference_x(
        lane: Sequence[LanePoint],
        frame_height: int,
    ) -> float:

        if not lane:
            return float(
                "nan"
            )

        target_y = (
            frame_height
            * 0.82
        )

        return min(
            lane,
            key=lambda point:
                abs(
                    point.y
                    - target_y
                ),
        ).x

    def _sort_lanes(
        self,
        lanes: List[
            List[LanePoint]
        ],
        frame_width: int,
        frame_height: int,
    ) -> List[
        List[LanePoint]
    ]:

        return sorted(
            lanes,
            key=lambda lane:
                self._lane_reference_x(
                    lane,
                    frame_height,
                )
        )

    # =========================================================================
    # OBJECT DETECTION
    # =========================================================================

    @staticmethod
    def _box_iou(
        box: np.ndarray,
        boxes: np.ndarray,
    ) -> np.ndarray:

        x1 = np.maximum(
            box[0],
            boxes[:, 0],
        )

        y1 = np.maximum(
            box[1],
            boxes[:, 1],
        )

        x2 = np.minimum(
            box[2],
            boxes[:, 2],
        )

        y2 = np.minimum(
            box[3],
            boxes[:, 3],
        )

        intersection = (
            np.maximum(
                0.0,
                x2 - x1,
            )
            * np.maximum(
                0.0,
                y2 - y1,
            )
        )

        area_a = (
            max(
                0.0,
                box[2] - box[0],
            )
            * max(
                0.0,
                box[3] - box[1],
            )
        )

        area_b = (
            np.maximum(
                0.0,
                boxes[:, 2]
                - boxes[:, 0],
            )
            * np.maximum(
                0.0,
                boxes[:, 3]
                - boxes[:, 1],
            )
        )

        union = (
            area_a
            + area_b
            - intersection
        )

        return intersection / np.maximum(
            union,
            1e-8,
        )

    @classmethod
    def _nms(
        cls,
        boxes: np.ndarray,
        scores: np.ndarray,
        iou_threshold: float,
    ) -> List[int]:

        if boxes.size == 0:
            return []

        order = np.argsort(
            scores
        )[::-1]

        keep: List[int] = []

        while order.size > 0:

            index = int(
                order[0]
            )

            keep.append(
                index
            )

            if order.size == 1:
                break

            ious = cls._box_iou(
                boxes[index],
                boxes[
                    order[1:]
                ],
            )

            order = order[
                1:
            ][
                ious
                <= iou_threshold
            ]

        return keep

    def _extract_objects(
        self,
        outputs: Any,
        frame_width: int,
        frame_height: int,
    ) -> List[
        ObjectDetection
    ]:

        """
        Extrator tolerante da cabeça de detecção.

        O YOLOPv2 pode retornar a cabeça de detecção
        em estruturas TorchScript diferentes conforme
        a versão do checkpoint.

        Se o formato não for reconhecido, retornamos
        [] sem comprometer a percepção de lanes.
        """

        tensors = _flatten_tensors(
            outputs
        )

        candidates = []

        for tensor in tensors:

            if tensor.ndim == 3:

                shape = tensor.shape

                # Formatos comuns:
                #
                # [B,N,85]
                # [B,85,N]

                if (
                    shape[-1] >= 6
                    and shape[-1] <= 100
                ):
                    candidates.append(
                        tensor
                    )

                elif (
                    shape[1] >= 6
                    and shape[1] <= 100
                ):
                    candidates.append(
                        tensor
                    )

        if not candidates:
            return []

        # Escolhemos o tensor com maior número
        # de candidatos e formato mais plausível.
        tensor = max(
            candidates,
            key=lambda item:
                int(item.numel())
        )

        array = (
            tensor.detach()
            .float()
            .cpu()
            .numpy()
        )

        if array.ndim != 3:
            return []

        array = array[0]

        # -------------------------------------------------------------
        # Normalização para [N, attributes].
        # -------------------------------------------------------------

        if (
            array.shape[0] >= 6
            and array.shape[0] <= 100
            and array.shape[1] > array.shape[0]
        ):
            array = array.T

        if array.shape[1] < 6:
            return []

        # -------------------------------------------------------------
        # Contrato:
        #
        # cx cy w h obj cls...
        # -------------------------------------------------------------

        boxes = array[:, :4]
        objectness = array[:, 4]

        if array.shape[1] >= 7:
            class_scores = array[:, 5:]

            class_ids = np.argmax(
                class_scores,
                axis=1,
            )

            class_confidence = np.max(
                class_scores,
                axis=1,
            )

            confidence = (
                objectness
                * class_confidence
            )

        else:
            class_ids = np.zeros(
                len(array),
                dtype=np.int32,
            )

            confidence = objectness

        valid = (
            np.isfinite(
                boxes
            ).all(axis=1)
            & np.isfinite(
                confidence
            )
            & (
                confidence
                >= DEFAULT_OBJECT_CONFIDENCE
            )
        )

        if not np.any(valid):
            return []

        boxes = boxes[
            valid
        ]

        confidence = confidence[
            valid
        ]

        class_ids = class_ids[
            valid
        ]

        # -------------------------------------------------------------
        # Detectar se as caixas estão normalizadas.
        # -------------------------------------------------------------

        if np.max(
            np.abs(boxes)
        ) <= 2.0:

            boxes[:, 0] *= (
                self.input_width
            )

            boxes[:, 2] *= (
                self.input_width
            )

            boxes[:, 1] *= (
                self.input_height
            )

            boxes[:, 3] *= (
                self.input_height
            )

        cx = boxes[:, 0]
        cy = boxes[:, 1]
        width = boxes[:, 2]
        height = boxes[:, 3]

        x1 = cx - width * 0.5
        y1 = cy - height * 0.5
        x2 = cx + width * 0.5
        y2 = cy + height * 0.5

        # Map model input -> frame.
        x_scale = (
            frame_width
            / float(self.input_width)
        )

        y_scale = (
            frame_height
            / float(self.input_height)
        )

        x1 *= x_scale
        x2 *= x_scale
        y1 *= y_scale
        y2 *= y_scale

        x1 = np.clip(
            x1,
            0.0,
            frame_width - 1,
        )

        x2 = np.clip(
            x2,
            0.0,
            frame_width - 1,
        )

        y1 = np.clip(
            y1,
            0.0,
            frame_height - 1,
        )

        y2 = np.clip(
            y2,
            0.0,
            frame_height - 1,
        )

        final_boxes = np.column_stack(
            (
                x1,
                y1,
                x2,
                y2,
            )
        )

        keep = self._nms(
            final_boxes,
            confidence,
            DEFAULT_OBJECT_IOU,
        )

        objects: List[
            ObjectDetection
        ] = []

        for index in keep:

            if (
                final_boxes[index, 2]
                <= final_boxes[index, 0]
                or final_boxes[index, 3]
                <= final_boxes[index, 1]
            ):
                continue

            objects.append(
                ObjectDetection(
                    class_id=int(
                        class_ids[index]
                    ),
                    confidence=_clip01(
                        confidence[index]
                    ),
                    x1=float(
                        final_boxes[
                            index,
                            0,
                        ]
                    ),
                    y1=float(
                        final_boxes[
                            index,
                            1,
                        ]
                    ),
                    x2=float(
                        final_boxes[
                            index,
                            2,
                        ]
                    ),
                    y2=float(
                        final_boxes[
                            index,
                            3,
                        ]
                    ),
                    frame_width=frame_width,
                    frame_height=frame_height,
                )
            )

        return objects

    # =========================================================================
    # LANE RESULT
    # =========================================================================

    def _build_lane_result(
        self,
        lanes: List[
            List[LanePoint]
        ],
        frame_width: int,
        frame_height: int,
        objects: List[
            ObjectDetection
        ],
        drivable_mask: Optional[
            np.ndarray
        ],
        output_shapes: List[
            Tuple[int, ...]
        ],
        inference_ms: float,
    ) -> LaneDetectionResult:

        lanes = self._sort_lanes(
            lanes,
            frame_width,
            frame_height,
        )

        confidences = [
            _clip01(
                self._lane_quality(
                    lane,
                    frame_width,
                    frame_height,
                )
            )
            for lane in lanes
        ]

        valid_pairs = [
            (
                lane,
                confidence,
            )
            for lane, confidence
            in zip(
                lanes,
                confidences,
            )
            if self._validate_lane(
                lane,
                frame_width,
                frame_height,
            )
        ]

        lanes = [
            lane
            for lane, _
            in valid_pairs
        ]

        confidences = [
            confidence
            for _, confidence
            in valid_pairs
        ]

        center = (
            frame_width
            * 0.5
        )

        left_candidates = []
        right_candidates = []

        for index, lane in enumerate(
            lanes
        ):

            reference_x = (
                self._lane_reference_x(
                    lane,
                    frame_height,
                )
            )

            if not math.isfinite(
                reference_x
            ):
                continue

            if reference_x < center:
                left_candidates.append(
                    index
                )
            else:
                right_candidates.append(
                    index
                )

        left_index = None
        right_index = None

        if left_candidates:
            left_index = max(
                left_candidates,
                key=lambda index:
                    self._lane_reference_x(
                        lanes[index],
                        frame_height,
                    )
            )

        if right_candidates:
            right_index = min(
                right_candidates,
                key=lambda index:
                    self._lane_reference_x(
                        lanes[index],
                        frame_height,
                    )
            )

        left_lane = (
            lanes[left_index]
            if left_index is not None
            else []
        )

        right_lane = (
            lanes[right_index]
            if right_index is not None
            else []
        )

        left_confidence = (
            confidences[left_index]
            if left_index is not None
            else 0.0
        )

        right_confidence = (
            confidences[right_index]
            if right_index is not None
            else 0.0
        )

        used = {
            index
            for index in (
                left_index,
                right_index,
            )
            if index is not None
        }

        additional_lanes = [
            lane
            for index, lane
            in enumerate(lanes)
            if index not in used
        ]

        valid = bool(
            lanes
        )

        lane_pixels = 0

        if drivable_mask is not None:
            lane_pixels = int(
                np.count_nonzero(
                    drivable_mask
                )
            )

        metadata = {
            "device": self.get_device_name(),
            "fp16": self.fp16_active,
            "inference_ms": inference_ms,
            "model_output_shapes": output_shapes,
            "lane_count": len(lanes),
            "vehicle_count": sum(
                obj.is_vehicle
                for obj in objects
            ),
            "object_count": len(objects),
            "drivable_pixels": lane_pixels,
            "coordinate_system": (
                "original_frame"
            ),
        }

        return LaneDetectionResult(
            lanes=lanes,
            lane_confidences=confidences,
            current_lane_index=None,
            left_lane=left_lane,
            right_lane=right_lane,
            additional_lanes=additional_lanes,
            left_confidence=left_confidence,
            right_confidence=right_confidence,
            valid=valid,
            num_lanes_detected=len(lanes),
            input_width=frame_width,
            input_height=frame_height,
            model_output_shape=(
                output_shapes[-1]
                if output_shapes
                else tuple()
            ),
            error=None,
            objects=objects,
            drivable_area_mask=drivable_mask,
            metadata=metadata,
        )

    # =========================================================================
    # DETECT
    # =========================================================================

    def detect(
        self,
        frame: np.ndarray,
    ) -> LaneDetectionResult:

        start = time.perf_counter()

        self.last_error = None
        self.last_diagnostics = {}

        if (
            not isinstance(
                frame,
                np.ndarray,
            )
            or frame.ndim != 3
            or frame.shape[2] != 3
        ):

            result = LaneDetectionResult(
                valid=False,
                error=(
                    "Frame inválido. "
                    "Esperado ndarray HxWx3."
                ),
            )

            self.last_result = result
            return result

        frame_height, frame_width = (
            frame.shape[:2]
        )

        try:

            outputs = self.infer(
                frame
            )

            output_shapes = (
                self.last_output_shapes
            )

            tensors = _flatten_tensors(
                outputs
            )

            # ---------------------------------------------------------
            # Segmentation.
            #
            # O detector identifica os tensors de segmentação
            # por sua estrutura [N,2,H,W].
            # ---------------------------------------------------------

            lane_tensor = (
                self._find_lane_segmentation(
                    outputs
                )
            )

            drivable_tensor = (
                self._find_drivable_segmentation(
                    outputs
                )
            )

            if lane_tensor is None:
                raise RuntimeError(
                    "Saída de lane segmentation "
                    "não encontrada no YOLOPv2."
                )

            lane_seg = (
                self._tensor_to_numpy(
                    lane_tensor
                )
            )

            lane_probability_model = (
                self._segmentation_probability(
                    lane_seg
                )
            )

            # ---------------------------------------------------------
            # A máscara é convertida diretamente para o frame original.
            # ---------------------------------------------------------

            lane_probability = (
                self._resize_probability_to_frame(
                    lane_probability_model,
                    frame_width,
                    frame_height,
                )
            )

            lane_mask = (
                self._build_lane_mask(
                    lane_probability
                )
            )

            # ---------------------------------------------------------
            # Drivable area.
            #
            # Em modelos onde o primeiro tensor [1,2,H,W] encontrado
            # também corresponde à lane segmentation, o uso da
            # mesma estrutura como drivable é deliberadamente
            # conservador: a área dirigível é apenas evidência
            # auxiliar e nunca substitui a lane segmentation.
            # ---------------------------------------------------------

            drivable_probability = None
            drivable_mask = None

            if (
                drivable_tensor is not None
                and drivable_tensor
                is not lane_tensor
            ):

                drivable_seg = (
                    self._tensor_to_numpy(
                        drivable_tensor
                    )
                )

                try:
                    drivable_probability_model = (
                        self._segmentation_probability(
                            drivable_seg
                        )
                    )

                    drivable_probability = (
                        self._resize_probability_to_frame(
                            drivable_probability_model,
                            frame_width,
                            frame_height,
                        )
                    )

                    drivable_mask = (
                        self._build_drivable_mask(
                            drivable_probability
                        )
                    )

                except Exception:
                    logger.exception(
                        "[YOLOPv2] Falha na "
                        "drivable segmentation."
                    )

            # ---------------------------------------------------------
            # Segmentação → segmentos.
            #
            # ATENÇÃO:
            #
            # lane_probability e lane_mask já possuem:
            #
            #     frame_height x frame_width
            #
            # Portanto x/y abaixo são coordenadas do frame.
            # ---------------------------------------------------------

            segments = (
                self._extract_row_segments(
                    lane_mask,
                    lane_probability,
                    drivable_probability,
                )
            )

            raw_tracks = (
                self._associate_segments(
                    segments,
                    frame_width,
                )
            )

            lanes: List[
                List[LanePoint]
            ] = []

            for track in raw_tracks:

                lane = self._track_to_lane(
                    track,
                    frame_width,
                    frame_height,
                )

                if not self._validate_lane(
                    lane,
                    frame_width,
                    frame_height,
                ):
                    continue

                lanes.append(
                    lane
                )

                if len(lanes) >= self.max_lanes:
                    break

            # ---------------------------------------------------------
            # Objetos.
            # ---------------------------------------------------------

            objects = self._extract_objects(
                outputs,
                frame_width,
                frame_height,
            )

            # ---------------------------------------------------------
            # Resultado.
            # ---------------------------------------------------------

            inference_ms = float(
                self.last_diagnostics.get(
                    "inference_ms",
                    0.0,
                )
            )

            result = self._build_lane_result(
                lanes=lanes,
                frame_width=frame_width,
                frame_height=frame_height,
                objects=objects,
                drivable_mask=drivable_mask,
                output_shapes=output_shapes,
                inference_ms=inference_ms,
            )

            # ---------------------------------------------------------
            # Diagnóstico detalhado.
            # ---------------------------------------------------------

            active_lane_pixels = int(
                np.count_nonzero(
                    lane_mask
                )
            )

            active_lane_ratio = (
                active_lane_pixels
                / float(
                    frame_width
                    * frame_height
                )
            )

            self.last_diagnostics.update(
                {
                    "device": self.get_device_name(),
                    "fp16": self.fp16_active,
                    "inference_ms": inference_ms,
                    "probability_shape": tuple(
                        lane_probability.shape
                    ),
                    "lane_mask_shape": tuple(
                        lane_mask.shape
                    ),
                    "drivable_mask_shape": (
                        tuple(
                            drivable_mask.shape
                        )
                        if drivable_mask
                        is not None
                        else None
                    ),
                    "active_lane_pixels": (
                        active_lane_pixels
                    ),
                    "active_lane_pixel_ratio": (
                        active_lane_ratio
                    ),
                    "drivable_pixels": (
                        int(
                            np.count_nonzero(
                                drivable_mask
                            )
                        )
                        if drivable_mask
                        is not None
                        else 0
                    ),
                    "row_count": (
                        len(
                            set(
                                segment.y
                                for segment
                                in segments
                            )
                        )
                    ),
                    "segment_count": len(
                        segments
                    ),
                    "raw_tracks": len(
                        raw_tracks
                    ),
                    "valid_lanes": len(
                        lanes
                    ),
                    "vehicle_count": sum(
                        obj.is_vehicle
                        for obj in objects
                    ),
                    "object_count": len(
                        objects
                    ),
                    "model_output_shapes": (
                        output_shapes
                    ),
                    "frame_width": (
                        frame_width
                    ),
                    "frame_height": (
                        frame_height
                    ),
                    "coordinate_system": (
                        "original_frame"
                    ),
                    "total_ms": (
                        (
                            time.perf_counter()
                            - start
                        )
                        * 1000.0
                    ),
                }
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
                input_width=frame_width,
                input_height=frame_height,
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
                    "fp16": self.fp16_active,
                    "fatal_error": self.last_error,
                    "model_output_shapes": (
                        self.last_output_shapes
                    ),
                    "total_ms": (
                        (
                            time.perf_counter()
                            - start
                        )
                        * 1000.0
                    ),
                },
            )

            self.last_result = result

            return result

    # =========================================================================
    # IMAGE UTILITIES
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
                f"Não foi possível carregar "
                f"imagem: {path}"
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
# PUBLIC COMPATIBILITY
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