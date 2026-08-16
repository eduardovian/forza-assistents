"""
vision/yolop_detector.py

Forza Assistents
================

Production-grade YOLOPv2 perception front-end.

Responsabilidades
-----------------
- carregar YOLOPv2 TorchScript de forma robusta no Windows;
- executar inferência CUDA/FP16;
- pré-processar frames deterministicamente;
- extrair lane segmentation;
- extrair drivable-area segmentation;
- extrair objetos quando disponíveis;
- converter coordenadas para o frame original;
- extrair segmentos espaciais;
- associar segmentos em lanes dentro do frame atual;
- produzir LaneDetectionResult;
- fornecer diagnósticos operacionais.

Este módulo NÃO executa:
- tracking temporal;
- fitting polinomial;
- LaneGeometry;
- LaneModel;
- LaneProjection;
- LaneAssignment;
- decisão ADAS;
- controle do veículo.

Contrato crítico
----------------
Todos os LanePoint retornados por detect() estão em:

    coordenadas do frame original recebido.

Nunca aplicar uma segunda transformação de coordenadas
a um LanePoint produzido por este detector.
"""

from __future__ import annotations

import logging
import math
import time

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from .lane_types import LanePoint


logger = logging.getLogger(__name__)


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODEL_PATH = PROJECT_ROOT / "weights" / "yolopv2.pt"

# Compatibilidade com o nome que está sendo usado no ambiente local.
LOCAL_MODEL_PATH = PROJECT_ROOT / "weights" / "yolopv2_local.pt"

LEGACY_MODEL_PATH = PROJECT_ROOT / "weights" / "yolop-640-640.onnx"


# =============================================================================
# MODEL CONTRACT
# =============================================================================

YOLOPV2_INPUT_WIDTH = 640
YOLOPV2_INPUT_HEIGHT = 640
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

DEFAULT_MORPH_KERNEL = 3

DEFAULT_USE_FP16 = True

DEFAULT_DRIVABLE_THRESHOLD = 0.15

DEFAULT_SEGMENT_MAX_GAP = 3

DEFAULT_MIN_LANE_CONFIDENCE = 0.20


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
        2,
        3,
        5,
        7,
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

    return max(0.0, min(1.0, value))


def _shape_of(value: Any) -> Tuple[int, ...]:
    if isinstance(value, torch.Tensor):
        return tuple(int(v) for v in value.shape)

    if isinstance(value, np.ndarray):
        return tuple(int(v) for v in value.shape)

    return tuple()


def _collect_shapes(value: Any) -> List[Tuple[int, ...]]:
    shapes: List[Tuple[int, ...]] = []

    if isinstance(value, torch.Tensor):
        shapes.append(_shape_of(value))
        return shapes

    if isinstance(value, np.ndarray):
        shapes.append(_shape_of(value))
        return shapes

    if isinstance(value, (tuple, list)):
        for item in value:
            shapes.extend(_collect_shapes(item))

    elif isinstance(value, dict):
        for item in value.values():
            shapes.extend(_collect_shapes(item))

    return shapes


def _flatten_tensors(value: Any) -> List[torch.Tensor]:
    tensors: List[torch.Tensor] = []

    if isinstance(value, torch.Tensor):
        tensors.append(value)

    elif isinstance(value, (tuple, list)):
        for item in value:
            tensors.extend(_flatten_tensors(item))

    elif isinstance(value, dict):
        for item in value.values():
            tensors.extend(_flatten_tensors(item))

    return tensors


# =============================================================================
# PUBLIC OBJECT TYPE
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


# =============================================================================
# PUBLIC RESULT
# =============================================================================

@dataclass
class LaneDetectionResult:
    lanes: List[List[LanePoint]] = field(default_factory=list)

    lane_confidences: List[float] = field(default_factory=list)

    current_lane_index: Optional[int] = None

    left_lane: List[LanePoint] = field(default_factory=list)

    right_lane: List[LanePoint] = field(default_factory=list)

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
    def vehicle_detections(self) -> List[ObjectDetection]:
        return [
            obj
            for obj in self.objects
            if obj.is_vehicle
        ]

    @property
    def vehicle_count(self) -> int:
        return len(self.vehicle_detections)

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
    points: List[Tuple[int, float, float]] = field(
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

        self.confidence_sum += float(confidence)
        self.road_confidence_sum += float(
            road_confidence
        )


@dataclass(frozen=True)
class _PreprocessMeta:
    original_width: int
    original_height: int

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
    YOLOPv2 production detector.

    Todas as coordenadas públicas pertencem ao frame original.
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

        self.model_path = Path(model_path)

        # ---------------------------------------------------------------------
        # Compatibilidade automática.
        #
        # Se o caminho configurado não existir, tenta o modelo local e depois
        # o modelo padrão.
        # ---------------------------------------------------------------------
        if not self.model_path.is_file():

            candidates = []

            if self.model_path.name == "yolopv2.pt":
                candidates.append(LOCAL_MODEL_PATH)

            if self.model_path.name == "yolop-640-640.onnx":
                candidates.append(DEFAULT_MODEL_PATH)
                candidates.append(LOCAL_MODEL_PATH)

            candidates.append(DEFAULT_MODEL_PATH)
            candidates.append(LOCAL_MODEL_PATH)

            for candidate in candidates:
                if candidate.is_file():
                    self.model_path = candidate
                    break

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

        self.use_fp16 = bool(use_fp16)

        if device is None:
            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        else:
            self.device = torch.device(device)

        if (
            self.device.type == "cuda"
            and not torch.cuda.is_available()
        ):
            logger.warning(
                "CUDA solicitado mas indisponível. "
                "Fallback para CPU."
            )

            self.device = torch.device("cpu")

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

        self._last_preprocess_meta: Optional[
            _PreprocessMeta
        ] = None

        self._morph_kernel = (
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (
                    DEFAULT_MORPH_KERNEL,
                    DEFAULT_MORPH_KERNEL,
                ),
            )
            if DEFAULT_MORPH_KERNEL >= 3
            else None
        )

    # =========================================================================
    # MODEL
    # =========================================================================

    def model_exists(self) -> bool:
        return self.model_path.is_file()

    def _describe_model_file(self) -> Dict[str, Any]:
        """
        Retorna informações físicas do checkpoint sem carregá-lo.
        """

        info: Dict[str, Any] = {
            "path": str(self.model_path),
            "exists": False,
            "is_file": False,
            "size_bytes": 0,
            "header": None,
        }

        try:
            info["exists"] = self.model_path.exists()
            info["is_file"] = self.model_path.is_file()

            if not info["is_file"]:
                return info

            info["size_bytes"] = self.model_path.stat().st_size

            with self.model_path.open("rb") as file:
                header = file.read(16)

            info["header"] = header.hex()

        except OSError as exc:
            info["error"] = (
                f"{type(exc).__name__}: {exc}"
            )

        return info

    def _resolve_model_path(self) -> bool:
        """
        Resolve novamente o checkpoint imediatamente antes do carregamento.

        Isso evita depender exclusivamente do caminho selecionado durante
        a construção do detector.
        """

        if self.model_path.is_file():
            return True

        candidates = [
            LOCAL_MODEL_PATH,
            DEFAULT_MODEL_PATH,
        ]

        if self.model_path.name == "yolop-640-640.onnx":
            candidates = [
                DEFAULT_MODEL_PATH,
                LOCAL_MODEL_PATH,
            ]

        for candidate in candidates:
            if candidate.is_file():
                logger.warning(
                    "[YOLOPv2] Modelo configurado não encontrado. "
                    "Usando fallback: %s",
                    candidate,
                )

                self.model_path = candidate
                return True

        return False

    def _load_torchscript_from_file(
        self,
    ) -> torch.jit.ScriptModule:
        """
        Carrega TorchScript através de um file object.

        Correção importante para Windows/OneDrive/Unicode.

        O caminho:
            C:\\Users\\...\\Área de Trabalho\\...

        pode funcionar com Python open() e falhar em determinadas
        chamadas internas do runtime C++ do TorchScript.

        Ao fornecer o arquivo já aberto, eliminamos esse ponto
        de falha.
        """

        with self.model_path.open("rb") as model_file:
            return torch.jit.load(
                model_file,
                map_location=self.device,
            )

    def load_model(self) -> bool:

        if (
            self.loaded
            and self.model is not None
        ):
            return True

        self.last_error = None

        if not self._resolve_model_path():

            info = self._describe_model_file()

            self.last_error = (
                "Modelo YOLOPv2 não encontrado. "
                f"caminho={self.model_path}; "
                f"exists={info.get('exists')}; "
                f"is_file={info.get('is_file')}"
            )

            logger.error(
                "[YOLOPv2] %s",
                self.last_error,
            )

            return False

        file_info = self._describe_model_file()

        logger.info(
            "[YOLOPv2] Abrindo checkpoint: %s",
            self.model_path,
        )

        logger.info(
            "[YOLOPv2] checkpoint_size=%d bytes",
            file_info.get("size_bytes", 0),
        )

        logger.debug(
            "[YOLOPv2] checkpoint_header=%s",
            file_info.get("header"),
        )

        try:

            # -----------------------------------------------------------------
            # CORREÇÃO PRINCIPAL:
            #
            # NÃO:
            #
            # torch.jit.load(str(self.model_path))
            #
            # SIM:
            #
            # torch.jit.load(file_object)
            #
            # Isso evita o problema de fopen() no caminho Unicode.
            # -----------------------------------------------------------------

            model = self._load_torchscript_from_file()

            model.eval()

            if self.device.type == "cuda":
                model = model.cuda()

                if self.use_fp16:
                    try:
                        model = model.half()
                        self.fp16_active = True

                    except Exception:
                        logger.warning(
                            "[YOLOPv2] FP16 indisponível para "
                            "este TorchScript. Usando FP32."
                        )

                        self.fp16_active = False

            else:
                self.fp16_active = False

            self.model = model
            self.loaded = True
            self._warmed_up = False

            # -----------------------------------------------------------------
            # Warmup.
            # -----------------------------------------------------------------

            self._warmup()

            logger.info(
                "[YOLOPv2] model=%s device=%s fp16=%s",
                self.model_path,
                self.get_device_name(),
                self.fp16_active,
            )

            return True

        except Exception as exc:

            self.last_error = (
                f"{type(exc).__name__}: {exc}"
            )

            self.model = None
            self.loaded = False
            self.fp16_active = False
            self._warmed_up = False

            logger.exception(
                "[YOLOPv2] Falha ao carregar modelo."
            )

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

            dummy = torch.zeros(
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
                _ = self.model(dummy)

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            self._warmed_up = True

            logger.info(
                "[YOLOPv2] Warmup concluído."
            )

        except Exception as exc:

            logger.warning(
                "[YOLOPv2] Warmup falhou: %s",
                exc,
            )

            # Warmup não invalida automaticamente o modelo.
            self._warmed_up = False

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
    # VALIDATION
    # =========================================================================

    @staticmethod
    def _validate_frame(
        frame: np.ndarray,
    ) -> Tuple[int, int]:

        if not isinstance(frame, np.ndarray):
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

        if frame.dtype != np.uint8:

            if not np.isfinite(
                frame.astype(
                    np.float32,
                    copy=False,
                )
            ).all():
                raise ValueError(
                    "Frame contém valores não finitos."
                )

        return int(width), int(height)

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

        width, height = self._validate_frame(frame)

        scale = min(
            self.input_width / width,
            self.input_height / height,
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
            self.input_width
            - resized_width
        ) // 2

        pad_top = (
            self.input_height
            - resized_height
        ) // 2

        pad_right = (
            self.input_width
            - resized_width
            - pad_left
        )

        pad_bottom = (
            self.input_height
            - resized_height
            - pad_top
        )

        letterboxed = cv2.copyMakeBorder(
            resized,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )

        rgb = cv2.cvtColor(
            letterboxed,
            cv2.COLOR_BGR2RGB,
        )

        normalized = (
            rgb.astype(
                np.float32,
                copy=False,
            )
            / 255.0
        )

        chw = np.transpose(
            normalized,
            (2, 0, 1),
        )

        tensor = np.ascontiguousarray(
            chw[None, ...],
            dtype=np.float32,
        )

        meta = _PreprocessMeta(
            original_width=width,
            original_height=height,
            scale=scale,
            resized_width=resized_width,
            resized_height=resized_height,
            pad_left=pad_left,
            pad_top=pad_top,
            pad_right=pad_right,
            pad_bottom=pad_bottom,
        )

        self._last_preprocess_meta = meta

        return tensor, meta

    def preprocess(
        self,
        frame: np.ndarray,
    ) -> np.ndarray:

        tensor, _ = self._prepare_frame(frame)
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

        tensor_np, _ = self._prepare_frame(frame)

        tensor = torch.from_numpy(
            tensor_np
        ).to(
            self.device,
            non_blocking=True,
        )

        if self.fp16_active:
            tensor = tensor.half()

        if self.device.type == "cuda":

            start_event = torch.cuda.Event(
                enable_timing=True
            )

            end_event = torch.cuda.Event(
                enable_timing=True
            )

            start_event.record()

            outputs = self.model(tensor)

            end_event.record()

            end_event.synchronize()

            inference_ms = float(
                start_event.elapsed_time(
                    end_event
                )
            )

        else:

            start = time.perf_counter()

            outputs = self.model(tensor)

            inference_ms = (
                time.perf_counter()
                - start
            ) * 1000.0

        self.last_output_shapes = _collect_shapes(
            outputs
        )

        self.last_diagnostics[
            "inference_ms"
        ] = inference_ms

        return outputs

    # =========================================================================
    # TENSOR UTILITIES
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

    # =========================================================================
    # SEGMENTATION DISCOVERY
    # =========================================================================

    @staticmethod
    def _find_lane_segmentation(
        outputs: Any,
    ) -> Optional[torch.Tensor]:

        tensors = _flatten_tensors(outputs)

        candidates = [
            tensor
            for tensor in tensors
            if (
                tensor.ndim == 4
                and int(tensor.shape[1]) == 2
                and int(tensor.shape[2]) >= 32
                and int(tensor.shape[3]) >= 32
            )
        ]

        if not candidates:
            return None

        return max(
            candidates,
            key=lambda tensor:
                int(tensor.shape[2])
                * int(tensor.shape[3]),
        )

    @staticmethod
    def _find_drivable_segmentation(
        outputs: Any,
    ) -> Optional[torch.Tensor]:

        tensors = _flatten_tensors(outputs)

        candidates = [
            tensor
            for tensor in tensors
            if (
                tensor.ndim == 4
                and int(tensor.shape[1]) == 1
                and int(tensor.shape[2]) >= 32
                and int(tensor.shape[3]) >= 32
            )
        ]

        if not candidates:
            return None

        return max(
            candidates,
            key=lambda tensor:
                int(tensor.shape[2])
                * int(tensor.shape[3]),
        )

    # =========================================================================
    # SEGMENTATION PROBABILITY
    # =========================================================================

    @staticmethod
    def _softmax_channels(
        logits: np.ndarray,
    ) -> np.ndarray:

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

        denominator = np.maximum(
            np.sum(
                exp_logits,
                axis=0,
                keepdims=True,
            ),
            1e-8,
        )

        return exp_logits / denominator

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

        logits = segmentation[0]

        if channels >= 2:

            probabilities = (
                self._softmax_channels(
                    logits
                )
            )

            return np.clip(
                probabilities[1],
                0.0,
                1.0,
            )

        logits = np.nan_to_num(
            logits[0],
            nan=0.0,
            posinf=50.0,
            neginf=-50.0,
        )

        probability = 1.0 / (
            1.0
            + np.exp(
                -np.clip(
                    logits,
                    -50.0,
                    50.0,
                )
            )
        )

        return np.clip(
            probability.astype(
                np.float32,
                copy=False,
            ),
            0.0,
            1.0,
        )

    # =========================================================================
    # COORDINATE MAPPING
    # =========================================================================

    @staticmethod
    def _model_mask_to_frame(
        probability: np.ndarray,
        meta: _PreprocessMeta,
    ) -> np.ndarray:

        if probability.ndim != 2:
            raise ValueError(
                "Probability map deve ser HxW."
            )

        y0 = meta.pad_top
        y1 = (
            meta.pad_top
            + meta.resized_height
        )

        x0 = meta.pad_left
        x1 = (
            meta.pad_left
            + meta.resized_width
        )

        model_height, model_width = (
            probability.shape
        )

        y0 = max(
            0,
            min(model_height, y0),
        )

        y1 = max(
            y0 + 1,
            min(model_height, y1),
        )

        x0 = max(
            0,
            min(model_width, x0),
        )

        x1 = max(
            x0 + 1,
            min(model_width, x1),
        )

        cropped = probability[
            y0:y1,
            x0:x1,
        ]

        return cv2.resize(
            cropped,
            (
                meta.original_width,
                meta.original_height,
            ),
            interpolation=cv2.INTER_LINEAR,
        ).astype(
            np.float32,
            copy=False,
        )

    def _resize_probability_to_frame(
        self,
        probability: np.ndarray,
        frame_width: int,
        frame_height: int,
    ) -> np.ndarray:

        if self._last_preprocess_meta is None:

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

        return self._model_mask_to_frame(
            probability,
            self._last_preprocess_meta,
        )

    # =========================================================================
    # MASKS
    # =========================================================================

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

        if self._morph_kernel is not None:

            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_OPEN,
                self._morph_kernel,
            )

            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_CLOSE,
                self._morph_kernel,
            )

        return mask

    @staticmethod
    def _build_drivable_mask(
        probability: np.ndarray,
    ) -> np.ndarray:

        return (
            probability
            >= DEFAULT_DRIVABLE_THRESHOLD
        ).astype(
            np.uint8
        )

    # =========================================================================
    # ROW SEGMENTS
    # =========================================================================

    @staticmethod
    def _split_contiguous(
        xs: np.ndarray,
        max_gap: int = DEFAULT_SEGMENT_MAX_GAP,
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
    ) -> List[_RowSegment]:

        height, _ = lane_mask.shape

        segments: List[_RowSegment] = []

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

            for group in self._split_contiguous(xs):

                if (
                    group.size
                    < self.min_lane_pixels_per_row
                ):
                    continue

                x_min = float(group[0])
                x_max = float(group[-1])

                x_center = (
                    x_min + x_max
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
    # SPATIAL ASSOCIATION
    # =========================================================================

    def _associate_segments(
        self,
        segments: Sequence[_RowSegment],
        frame_width: int,
    ) -> List[_LaneTrack]:

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

        tracks: List[_LaneTrack] = []

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
                key=lambda item:
                    item.x_center,
            )

            if not tracks:

                for candidate in candidates:

                    if len(tracks) >= self.max_lanes:
                        break

                    track = _LaneTrack()

                    track.add(
                        candidate.y,
                        candidate.x_center,
                        candidate.confidence,
                        candidate.road_confidence,
                    )

                    tracks.append(track)

                continue

            used_tracks = set()

            for candidate in candidates:

                best_index = None
                best_score = float("inf")

                for index, track in enumerate(tracks):

                    if index in used_tracks:
                        continue

                    dx = abs(
                        candidate.x_center
                        - track.last_x
                    )

                    if dx > max_jump:
                        continue

                    score = (
                        dx
                        - 8.0 * candidate.confidence
                        - 4.0 * candidate.road_confidence
                    )

                    if score < best_score:
                        best_score = score
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

                    tracks.append(track)

                else:

                    tracks[best_index].add(
                        candidate.y,
                        candidate.x_center,
                        candidate.confidence,
                        candidate.road_confidence,
                    )

                    used_tracks.add(best_index)

        return tracks

    # =========================================================================
    # TRACK → LANE
    # =========================================================================

    def _track_to_lane(
        self,
        track: _LaneTrack,
        frame_width: int,
        frame_height: int,
    ) -> List[LanePoint]:

        points: List[LanePoint] = []

        for (
            frame_y,
            frame_x,
            confidence,
        ) in track.points:

            x = float(
                np.clip(
                    frame_x,
                    0.0,
                    float(frame_width - 1),
                )
            )

            y = float(
                np.clip(
                    frame_y,
                    0.0,
                    float(frame_height - 1),
                )
            )

            point = LanePoint(
                x=x,
                y=y,
                confidence=_clip01(confidence),
                valid=True,
            )

            if point.is_valid():
                points.append(point)

        points.sort(
            key=lambda point:
                point.y
        )

        result: List[LanePoint] = []

        last_y: Optional[float] = None

        for point in points:

            if (
                last_y is not None
                and abs(point.y - last_y) < 0.5
            ):
                continue

            result.append(point)
            last_y = point.y

        return result

    # =========================================================================
    # LANE QUALITY
    # =========================================================================

    @staticmethod
    def _lane_quality(
        lane: Sequence[LanePoint],
        frame_width: int,
        frame_height: int,
    ) -> float:

        if len(lane) < 2:
            return 0.0

        xs = np.asarray(
            [point.x for point in lane],
            dtype=np.float32,
        )

        ys = np.asarray(
            [point.y for point in lane],
            dtype=np.float32,
        )

        confidences = np.asarray(
            [
                point.confidence
                for point in lane
            ],
            dtype=np.float32,
        )

        if (
            not np.isfinite(xs).all()
            or not np.isfinite(ys).all()
            or not np.isfinite(confidences).all()
        ):
            return 0.0

        vertical_span = float(
            ys.max() - ys.min()
        )

        horizontal_span = float(
            xs.max() - xs.min()
        )

        confidence = float(
            np.mean(confidences)
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

        for point in lane:

            if not point.is_valid():
                return False

            if (
                point.x < 0.0
                or point.x >= frame_width
                or point.y < 0.0
                or point.y >= frame_height
            ):
                return False

        ys = [
            point.y
            for point in lane
        ]

        vertical_span = (
            max(ys) - min(ys)
        )

        if (
            vertical_span
            < DEFAULT_MIN_LANE_VERTICAL_SPAN
        ):
            return False

        quality = self._lane_quality(
            lane,
            frame_width,
            frame_height,
        )

        return quality >= DEFAULT_MIN_LANE_CONFIDENCE

    # =========================================================================
    # LANE ORDERING
    # =========================================================================

    @staticmethod
    def _lane_reference_x(
        lane: Sequence[LanePoint],
        frame_height: int,
    ) -> float:

        if not lane:
            return float("nan")

        target_y = frame_height * 0.82

        point = min(
            lane,
            key=lambda item:
                abs(
                    item.y - target_y
                ),
        )

        return float(point.x)

    def _sort_lanes(
        self,
        lanes: List[List[LanePoint]],
        frame_width: int,
        frame_height: int,
    ) -> List[List[LanePoint]]:

        return sorted(
            lanes,
            key=lambda lane:
                self._lane_reference_x(
                    lane,
                    frame_height,
                ),
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
                boxes[:, 2] - boxes[:, 0],
            )
            * np.maximum(
                0.0,
                boxes[:, 3] - boxes[:, 1],
            )
        )

        union = (
            area_a
            + area_b
            - intersection
        )

        return (
            intersection
            / np.maximum(
                union,
                1e-8,
            )
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

        order = np.argsort(scores)[::-1]

        keep: List[int] = []

        while order.size > 0:

            index = int(order[0])

            keep.append(index)

            if order.size == 1:
                break

            ious = cls._box_iou(
                boxes[index],
                boxes[order[1:]],
            )

            order = order[1:][
                ious <= iou_threshold
            ]

        return keep

    def _map_box_to_frame(
        self,
        box: Sequence[float],
        frame_width: int,
        frame_height: int,
    ) -> Tuple[
        float,
        float,
        float,
        float,
    ]:

        meta = self._last_preprocess_meta

        if meta is None:

            x1, y1, x2, y2 = box

            return (
                float(x1),
                float(y1),
                float(x2),
                float(y2),
            )

        cx, cy, width, height = map(
            float,
            box,
        )

        x1 = cx - width * 0.5
        y1 = cy - height * 0.5
        x2 = cx + width * 0.5
        y2 = cy + height * 0.5

        # Normalized coordinates.
        if max(
            abs(x1),
            abs(y1),
            abs(x2),
            abs(y2),
        ) <= 2.0:

            x1 *= self.input_width
            x2 *= self.input_width
            y1 *= self.input_height
            y2 *= self.input_height

        x1 = (
            x1 - meta.pad_left
        ) / meta.scale

        x2 = (
            x2 - meta.pad_left
        ) / meta.scale

        y1 = (
            y1 - meta.pad_top
        ) / meta.scale

        y2 = (
            y2 - meta.pad_top
        ) / meta.scale

        x1 = float(
            np.clip(
                x1,
                0.0,
                frame_width - 1,
            )
        )

        y1 = float(
            np.clip(
                y1,
                0.0,
                frame_height - 1,
            )
        )

        x2 = float(
            np.clip(
                x2,
                0.0,
                frame_width - 1,
            )
        )

        y2 = float(
            np.clip(
                y2,
                0.0,
                frame_height - 1,
            )
        )

        return x1, y1, x2, y2

    def _extract_objects(
        self,
        outputs: Any,
        frame_width: int,
        frame_height: int,
    ) -> List[ObjectDetection]:

        tensors = _flatten_tensors(outputs)

        candidates: List[
            torch.Tensor
        ] = []

        for tensor in tensors:

            if tensor.ndim != 3:
                continue

            shape = tensor.shape

            if (
                6 <= int(shape[-1]) <= 100
                or 6 <= int(shape[1]) <= 100
            ):
                candidates.append(tensor)

        if not candidates:
            return []

        tensor = max(
            candidates,
            key=lambda item:
                int(item.numel()),
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

        if (
            6 <= array.shape[0] <= 100
            and array.shape[1] > array.shape[0]
        ):
            array = array.T

        if array.shape[1] < 6:
            return []

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
            np.isfinite(boxes).all(axis=1)
            & np.isfinite(confidence)
            & (
                confidence
                >= DEFAULT_OBJECT_CONFIDENCE
            )
        )

        if not np.any(valid):
            return []

        boxes = boxes[valid]
        confidence = confidence[valid]
        class_ids = class_ids[valid]

        frame_boxes = []

        for box in boxes:

            frame_boxes.append(
                self._map_box_to_frame(
                    box,
                    frame_width,
                    frame_height,
                )
            )

        final_boxes = np.asarray(
            frame_boxes,
            dtype=np.float32,
        )

        valid_boxes = (
            (final_boxes[:, 2]
             > final_boxes[:, 0])
            & (
                final_boxes[:, 3]
                > final_boxes[:, 1]
            )
        )

        if not np.any(valid_boxes):
            return []

        final_boxes = final_boxes[valid_boxes]
        confidence = confidence[valid_boxes]
        class_ids = class_ids[valid_boxes]

        keep: List[int] = []

        for class_id in np.unique(class_ids):

            indices = np.flatnonzero(
                class_ids == class_id
            )

            local_keep = self._nms(
                final_boxes[indices],
                confidence[indices],
                DEFAULT_OBJECT_IOU,
            )

            keep.extend(
                int(indices[index])
                for index in local_keep
            )

        keep.sort(
            key=lambda index:
                float(confidence[index]),
            reverse=True,
        )

        objects: List[
            ObjectDetection
        ] = []

        for index in keep:

            x1, y1, x2, y2 = (
                final_boxes[index]
            )

            objects.append(
                ObjectDetection(
                    class_id=int(
                        class_ids[index]
                    ),
                    confidence=_clip01(
                        confidence[index]
                    ),
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    frame_width=frame_width,
                    frame_height=frame_height,
                )
            )

        return objects

    # =========================================================================
    # RESULT
    # =========================================================================

    def _build_lane_result(
        self,
        lanes: List[List[LanePoint]],
        frame_width: int,
        frame_height: int,
        objects: List[ObjectDetection],
        drivable_mask: Optional[np.ndarray],
        output_shapes: List[Tuple[int, ...]],
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
            for lane, confidence in zip(
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

        center = frame_width * 0.5

        left_candidates = []
        right_candidates = []

        for index, lane in enumerate(lanes):

            reference_x = (
                self._lane_reference_x(
                    lane,
                    frame_height,
                )
            )

            if not math.isfinite(reference_x):
                continue

            if reference_x < center:
                left_candidates.append(index)
            else:
                right_candidates.append(index)

        left_index = None
        right_index = None

        if left_candidates:

            left_index = max(
                left_candidates,
                key=lambda index:
                    self._lane_reference_x(
                        lanes[index],
                        frame_height,
                    ),
            )

        if right_candidates:

            right_index = min(
                right_candidates,
                key=lambda index:
                    self._lane_reference_x(
                        lanes[index],
                        frame_height,
                    ),
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
            for index, lane in enumerate(lanes)
            if index not in used
        ]

        drivable_pixels = (
            int(
                np.count_nonzero(
                    drivable_mask
                )
            )
            if drivable_mask is not None
            else 0
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
            "drivable_pixels": drivable_pixels,
            "coordinate_system": "original_frame",
            "model_path": str(self.model_path),
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
            valid=bool(lanes),
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

        try:
            frame_width, frame_height = (
                self._validate_frame(frame)
            )

        except Exception as exc:

            error = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            self.last_error = error

            result = LaneDetectionResult(
                valid=False,
                error=error,
            )

            self.last_result = result

            return result

        try:

            outputs = self.infer(frame)

            output_shapes = list(
                self.last_output_shapes
            )

            # -----------------------------------------------------------------
            # Lane segmentation.
            # -----------------------------------------------------------------

            lane_tensor = (
                self._find_lane_segmentation(
                    outputs
                )
            )

            if lane_tensor is None:
                raise RuntimeError(
                    "Saída de lane segmentation "
                    "não encontrada no YOLOPv2."
                )

            lane_segmentation = (
                self._tensor_to_numpy(
                    lane_tensor
                )
            )

            lane_probability_model = (
                self._segmentation_probability(
                    lane_segmentation
                )
            )

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

            # -----------------------------------------------------------------
            # Drivable segmentation.
            # -----------------------------------------------------------------

            drivable_probability = None
            drivable_mask = None

            drivable_tensor = (
                self._find_drivable_segmentation(
                    outputs
                )
            )

            if drivable_tensor is not None:

                drivable_segmentation = (
                    self._tensor_to_numpy(
                        drivable_tensor
                    )
                )

                drivable_probability_model = (
                    self._segmentation_probability(
                        drivable_segmentation
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

            # -----------------------------------------------------------------
            # Spatial segmentation.
            # -----------------------------------------------------------------

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

                lanes.append(lane)

                if len(lanes) >= self.max_lanes:
                    break

            # -----------------------------------------------------------------
            # Objects.
            # -----------------------------------------------------------------

            objects = (
                self._extract_objects(
                    outputs,
                    frame_width,
                    frame_height,
                )
            )

            inference_ms = float(
                self.last_diagnostics.get(
                    "inference_ms",
                    0.0,
                )
            )

            result = (
                self._build_lane_result(
                    lanes=lanes,
                    frame_width=frame_width,
                    frame_height=frame_height,
                    objects=objects,
                    drivable_mask=drivable_mask,
                    output_shapes=output_shapes,
                    inference_ms=inference_ms,
                )
            )

            # -----------------------------------------------------------------
            # Diagnostics.
            # -----------------------------------------------------------------

            active_lane_pixels = int(
                np.count_nonzero(
                    lane_mask
                )
            )

            total_pixels = (
                frame_width
                * frame_height
            )

            active_lane_ratio = (
                active_lane_pixels
                / float(
                    max(
                        1,
                        total_pixels,
                    )
                )
            )

            total_ms = (
                time.perf_counter()
                - start
            ) * 1000.0

            diagnostics = {
                "device": self.get_device_name(),
                "fp16": self.fp16_active,

                "inference_ms": inference_ms,
                "total_ms": total_ms,

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
                    if drivable_mask is not None
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
                    if drivable_mask is not None
                    else 0
                ),

                "row_count": len(
                    {
                        segment.y
                        for segment in segments
                    }
                ),

                "segment_count": len(segments),

                "raw_tracks": len(raw_tracks),

                "valid_lanes": len(lanes),

                "vehicle_count": sum(
                    obj.is_vehicle
                    for obj in objects
                ),

                "object_count": len(objects),

                "model_output_shapes": output_shapes,

                "frame_width": frame_width,
                "frame_height": frame_height,

                "coordinate_system": "original_frame",

                "model_input_width": self.input_width,
                "model_input_height": self.input_height,

                "lane_threshold": self.lane_threshold,

                "drivable_threshold": (
                    DEFAULT_DRIVABLE_THRESHOLD
                ),

                "max_lanes": self.max_lanes,

                "model_path": str(self.model_path),

                "model_size_bytes": (
                    self.model_path.stat().st_size
                    if self.model_path.is_file()
                    else 0
                ),
            }

            self.last_diagnostics = diagnostics

            result.metadata.update(
                diagnostics
            )

            self.last_result = result

            return result

        except Exception as exc:

            self.last_error = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            logger.exception(
                "[YOLOPv2] Falha durante detect()."
            )

            total_ms = (
                time.perf_counter()
                - start
            ) * 1000.0

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
                    "coordinate_system": (
                        "original_frame"
                    ),
                    "model_path": str(
                        self.model_path
                    ),
                    "total_ms": total_ms,
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