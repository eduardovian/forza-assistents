

"""
vision/yolop_detector.py

YOLOPv2 perception front-end for Forza Assistents.

Contrato:
    - recebe o ROI/frame capturado em BGR;
    - executa YOLOPv2;
    - separa corretamente as cabeças DRIVABLE AREA e LANE LINE;
    - desfaz corretamente o letterbox das máscaras;
    - transforma a máscara de lane em linhas espaciais estáveis;
    - retorna todos os LanePoint em coordenadas do frame recebido.

Importante:
    YOLOPv2 possui duas cabeças de segmentação semântica de 2 canais:

        da_seg_out -> drivable area
        ll_seg_out -> lane lines

    Ambas são 2 canais. Não é correto identificar uma pelo número de
    canais. A ordem do output oficial é usada quando as duas cabeças têm
    o mesmo shape.
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

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "weights" / "yolopv2.pt"
LOCAL_MODEL_PATH = PROJECT_ROOT / "weights" / "yolopv2_local.pt"
LEGACY_MODEL_PATH = PROJECT_ROOT / "weights" / "yolop-640-640.onnx"

YOLOPV2_INPUT_WIDTH = 640
YOLOPV2_INPUT_HEIGHT = 640

DEFAULT_LANE_THRESHOLD = 0.42
DEFAULT_DRIVABLE_THRESHOLD = 0.50
DEFAULT_OBJECT_CONFIDENCE = 0.30
DEFAULT_OBJECT_IOU = 0.45

DEFAULT_MIN_POINTS_PER_LANE = 6
DEFAULT_ROW_STEP = 5
DEFAULT_MIN_LANE_PIXELS_PER_ROW = 1
DEFAULT_MAX_LANES = 8
DEFAULT_MAX_LANE_JUMP = 65.0
DEFAULT_MIN_VERTICAL_SPAN = 55.0
DEFAULT_MIN_LANE_CONFIDENCE = 0.30
DEFAULT_MAX_SEGMENT_GAP = 4
DEFAULT_MORPH_KERNEL = 3

COCO_CLASS_NAMES: Tuple[str, ...] = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
    "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
)

VEHICLE_CLASS_IDS = frozenset({2, 3, 5, 7})


def _finite(value: Any) -> Optional[float]:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _clip01(value: Any) -> float:
    value = _finite(value)
    if value is None:
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def _flatten_tensors(value: Any) -> List[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    result: List[torch.Tensor] = []
    if isinstance(value, (tuple, list)):
        for item in value:
            result.extend(_flatten_tensors(item))
    elif isinstance(value, dict):
        for item in value.values():
            result.extend(_flatten_tensors(item))
    return result


def _collect_shapes(value: Any) -> List[Tuple[int, ...]]:
    return [tuple(int(v) for v in t.shape) for t in _flatten_tensors(value)]


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
        return COCO_CLASS_NAMES[self.class_id] if 0 <= self.class_id < len(COCO_CLASS_NAMES) else f"class_{self.class_id}"

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


@dataclass
class LaneDetectionResult:
    lanes: List[List[LanePoint]] = field(default_factory=list)
    lane_confidences: List[float] = field(default_factory=list)
    current_lane_index: Optional[int] = None
    left_lane: List[LanePoint] = field(default_factory=list)
    right_lane: List[LanePoint] = field(default_factory=list)
    additional_lanes: List[List[LanePoint]] = field(default_factory=list)
    left_confidence: float = 0.0
    right_confidence: float = 0.0
    valid: bool = False
    num_lanes_detected: int = 0
    input_width: int = YOLOPV2_INPUT_WIDTH
    input_height: int = YOLOPV2_INPUT_HEIGHT
    model_output_shape: Tuple[int, ...] = field(default_factory=tuple)
    error: Optional[str] = None
    objects: List[ObjectDetection] = field(default_factory=list)
    drivable_area_mask: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def detected(self) -> bool:
        return self.num_lanes_detected > 0

    @property
    def has_current_lane(self) -> bool:
        return self.current_lane_index is not None and bool(self.left_lane) and bool(self.right_lane)

    @property
    def vehicle_detections(self) -> List[ObjectDetection]:
        return [obj for obj in self.objects if obj.is_vehicle]

    @property
    def vehicle_count(self) -> int:
        return len(self.vehicle_detections)

    @property
    def object_count(self) -> int:
        return len(self.objects)


@dataclass
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


@dataclass
class _RowSegment:
    y: int
    x_center: float
    confidence: float
    pixel_count: int


@dataclass
class _LaneTrack:
    points: List[Tuple[int, float, float]] = field(default_factory=list)
    last_x: float = 0.0
    last_y: int = 0

    def add(self, segment: _RowSegment) -> None:
        self.points.append((segment.y, segment.x_center, segment.confidence))
        self.last_x = segment.x_center
        self.last_y = segment.y


class YOLOPLaneDetector:
    """Detector YOLOPv2 sem estado temporal de tracking."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        input_width: int = YOLOPV2_INPUT_WIDTH,
        input_height: int = YOLOPV2_INPUT_HEIGHT,
        lane_threshold: float = DEFAULT_LANE_THRESHOLD,
        min_points_per_lane: int = DEFAULT_MIN_POINTS_PER_LANE,
        row_step: int = DEFAULT_ROW_STEP,
        min_lane_pixels_per_row: int = DEFAULT_MIN_LANE_PIXELS_PER_ROW,
        max_lanes: int = DEFAULT_MAX_LANES,
        use_fp16: bool = True,
        device: Optional[str] = None,
        **_: Any,
    ) -> None:
        self.model_path = Path(model_path)
        self.input_width = max(32, int(input_width))
        self.input_height = max(32, int(input_height))
        self.lane_threshold = _clip01(lane_threshold)
        self.min_points_per_lane = max(2, int(min_points_per_lane))
        self.row_step = max(1, int(row_step))
        self.min_lane_pixels_per_row = max(1, int(min_lane_pixels_per_row))
        self.max_lanes = max(2, int(max_lanes))
        self.use_fp16 = bool(use_fp16)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")

        self.model: Optional[torch.jit.ScriptModule] = None
        self.loaded = False
        self.fp16_active = False
        self.last_error: Optional[str] = None
        self.last_result: Optional[LaneDetectionResult] = None
        self.last_output_shapes: List[Tuple[int, ...]] = []
        self.last_diagnostics: Dict[str, Any] = {}
        self._last_preprocess_meta: Optional[_PreprocessMeta] = None
        self._warmed_up = False
        self._morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (DEFAULT_MORPH_KERNEL, DEFAULT_MORPH_KERNEL)
        )

    # ------------------------------------------------------------------ model
    def _resolve_model_path(self) -> bool:
        if self.model_path.is_file():
            return True
        candidates = [LOCAL_MODEL_PATH, DEFAULT_MODEL_PATH]
        if self.model_path.name == LEGACY_MODEL_PATH.name:
            candidates = [DEFAULT_MODEL_PATH, LOCAL_MODEL_PATH]
        for candidate in candidates:
            if candidate.is_file():
                self.model_path = candidate
                LOGGER.warning("YOLOPv2: usando fallback %s", candidate)
                return True
        return False

    def load_model(self) -> bool:
        if self.loaded and self.model is not None:
            return True
        self.last_error = None
        if not self._resolve_model_path():
            self.last_error = f"Modelo YOLOPv2 não encontrado: {self.model_path}"
            LOGGER.error(self.last_error)
            return False
        try:
            # File object: robusto para Windows + OneDrive + caracteres Unicode.
            with self.model_path.open("rb") as file:
                model = torch.jit.load(file, map_location=self.device)
            model.eval()
            if self.device.type == "cuda":
                model = model.cuda()
                if self.use_fp16:
                    try:
                        model = model.half()
                        self.fp16_active = True
                    except Exception:
                        self.fp16_active = False
            self.model = model
            self.loaded = True
            self._warmup()
            LOGGER.info(
                "YOLOPv2: model=%s | device=%s | FP16=%s",
                self.model_path,
                self.get_device_name(),
                self.fp16_active,
            )
            return True
        except Exception as exc:
            self.model = None
            self.loaded = False
            self.fp16_active = False
            self.last_error = f"{type(exc).__name__}: {exc}"
            LOGGER.exception("Falha ao carregar YOLOPv2")
            return False

    def _warmup(self) -> None:
        if self.model is None or self._warmed_up:
            return
        try:
            dtype = torch.float16 if self.fp16_active else torch.float32
            dummy = torch.zeros(
                (1, 3, self.input_height, self.input_width),
                dtype=dtype,
                device=self.device,
            )
            with torch.inference_mode():
                self.model(dummy)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            self._warmed_up = True
        except Exception as exc:
            LOGGER.warning("Warmup YOLOPv2 falhou: %s", exc)

    def get_device_name(self) -> str:
        if self.device.type == "cuda":
            try:
                return torch.cuda.get_device_name(self.device)
            except Exception:
                return "CUDA"
        return "CPU"

    # --------------------------------------------------------------- preprocess
    @staticmethod
    def _validate_frame(frame: np.ndarray) -> Tuple[int, int]:
        if not isinstance(frame, np.ndarray):
            raise TypeError("Frame deve ser numpy.ndarray")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Frame deve possuir formato HxWx3 BGR")
        height, width = frame.shape[:2]
        if width <= 0 or height <= 0:
            raise ValueError("Dimensões do frame inválidas")
        return int(width), int(height)

    def _prepare_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, _PreprocessMeta]:
        width, height = self._validate_frame(frame)
        scale = min(self.input_width / width, self.input_height / height)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        pad_left = (self.input_width - resized_width) // 2
        pad_top = (self.input_height - resized_height) // 2
        pad_right = self.input_width - resized_width - pad_left
        pad_bottom = self.input_height - resized_height - pad_top

        letterboxed = cv2.copyMakeBorder(
            resized,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
        tensor = np.ascontiguousarray(
            np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))[None],
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

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        tensor, _ = self._prepare_frame(frame)
        return tensor

    # ---------------------------------------------------------------- inference
    @torch.inference_mode()
    def infer(self, frame: np.ndarray) -> Any:
        if not self.loaded and not self.load_model():
            raise RuntimeError(self.last_error or "YOLOPv2 não carregado")
        if self.model is None:
            raise RuntimeError("Modelo YOLOPv2 inexistente")

        tensor_np, _ = self._prepare_frame(frame)
        tensor = torch.from_numpy(tensor_np).to(self.device, non_blocking=True)
        if self.fp16_active:
            tensor = tensor.half()

        if self.device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            outputs = self.model(tensor)
            end.record()
            end.synchronize()
            inference_ms = float(start.elapsed_time(end))
        else:
            t0 = time.perf_counter()
            outputs = self.model(tensor)
            inference_ms = (time.perf_counter() - t0) * 1000.0

        self.last_output_shapes = _collect_shapes(outputs)
        self.last_diagnostics = {"inference_ms": inference_ms}
        return outputs

    # ----------------------------------------------------------- output parsing
    @staticmethod
    def _segmentation_candidates(outputs: Any) -> List[torch.Tensor]:
        tensors = _flatten_tensors(outputs)
        return [
            t for t in tensors
            if t.ndim == 4
            and int(t.shape[0]) >= 1
            and int(t.shape[1]) == 2
            and int(t.shape[2]) >= 16
            and int(t.shape[3]) >= 16
        ]

    @classmethod
    def _find_segmentation_heads(cls, outputs: Any) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Retorna (drivable, lane).

        YOLOP oficial:
            output[0] = detection
            output[1] = da_seg_out
            output[2] = ll_seg_out

        As duas segmentações são 2-channel. Portanto o antigo critério
        2 canais = lane / 1 canal = drivable estava incorreto e podia fazer
        ambas as funções selecionarem o mesmo tensor.
        """
        candidates = cls._segmentation_candidates(outputs)
        if len(candidates) >= 2:
            # Mantém a ordem original dos outputs. Entre cabeças auxiliares,
            # escolhemos as duas de maior resolução, mas preservamos a ordem.
            max_h = max(int(t.shape[2]) for t in candidates)
            max_w = max(int(t.shape[3]) for t in candidates)
            primary = [
                t for t in candidates
                if int(t.shape[2]) * int(t.shape[3]) == max_h * max_w
            ]
            if len(primary) >= 2:
                return primary[0], primary[1]
            return candidates[0], candidates[1]

        # Compatibilidade com exports alterados que possam possuir apenas
        # uma cabeça de segmentação. Nunca reutilizamos o mesmo tensor para
        # duas funções semanticamente diferentes.
        one_channel = [
            t for t in _flatten_tensors(outputs)
            if t.ndim == 4 and int(t.shape[1]) == 1
            and int(t.shape[2]) >= 16 and int(t.shape[3]) >= 16
        ]
        if candidates and one_channel:
            return one_channel[0], candidates[0]
        return None, None

    @staticmethod
    def _tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().float().cpu().numpy()

    @staticmethod
    def _softmax_channels(logits: np.ndarray) -> np.ndarray:
        logits = np.asarray(logits, dtype=np.float32)
        logits = np.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
        logits -= np.max(logits, axis=0, keepdims=True)
        exp = np.exp(np.clip(logits, -50.0, 50.0))
        return exp / np.maximum(np.sum(exp, axis=0, keepdims=True), 1e-8)

    @classmethod
    def _segmentation_probability(cls, segmentation: np.ndarray) -> np.ndarray:
        if segmentation.ndim != 4 or segmentation.shape[0] < 1:
            raise ValueError("Segmentation deve possuir shape [N,C,H,W]")
        channels = int(segmentation.shape[1])
        logits = segmentation[0]
        if channels >= 2:
            return np.clip(cls._softmax_channels(logits)[1], 0.0, 1.0)
        logits = np.nan_to_num(logits[0], nan=0.0, posinf=50.0, neginf=-50.0)
        return (1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))).astype(np.float32)

    # --------------------------------------------------------- mask coordinates
    @staticmethod
    def _model_mask_to_frame(probability: np.ndarray, meta: _PreprocessMeta) -> np.ndarray:
        """
        Remove o padding no espaço da máscara, não no espaço 640x640.

        Este detalhe é crítico. Se a saída for 80x80, por exemplo, pad_left
        também precisa ser convertido para 80x80 antes do crop.
        """
        if probability.ndim != 2:
            raise ValueError("Probability map deve ser HxW")

        mask_h, mask_w = probability.shape
        sx = mask_w / float(meta.resized_width + meta.pad_left + meta.pad_right)
        sy = mask_h / float(meta.resized_height + meta.pad_top + meta.pad_bottom)

        x0 = int(round(meta.pad_left * sx))
        y0 = int(round(meta.pad_top * sy))
        x1 = int(round((meta.pad_left + meta.resized_width) * sx))
        y1 = int(round((meta.pad_top + meta.resized_height) * sy))

        x0 = int(np.clip(x0, 0, mask_w - 1))
        y0 = int(np.clip(y0, 0, mask_h - 1))
        x1 = int(np.clip(max(x0 + 1, x1), 1, mask_w))
        y1 = int(np.clip(max(y0 + 1, y1), 1, mask_h))

        cropped = probability[y0:y1, x0:x1]
        return cv2.resize(
            cropped,
            (meta.original_width, meta.original_height),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)

    def _resize_probability_to_frame(self, probability: np.ndarray, width: int, height: int) -> np.ndarray:
        if self._last_preprocess_meta is None:
            return cv2.resize(probability, (width, height), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        return self._model_mask_to_frame(probability, self._last_preprocess_meta)

    # --------------------------------------------------------------- lane mask
    def _build_lane_mask(self, probability: np.ndarray) -> np.ndarray:
        mask = (probability >= self.lane_threshold).astype(np.uint8)
        # Remove pequenos ruídos, mas não dilatar excessivamente a linha.
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._morph_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._morph_kernel)
        return mask

    @staticmethod
    def _build_drivable_mask(probability: np.ndarray) -> np.ndarray:
        return (probability >= DEFAULT_DRIVABLE_THRESHOLD).astype(np.uint8)

    # -------------------------------------------------------------- lane rows
    @staticmethod
    def _split_contiguous(xs: np.ndarray, max_gap: int = DEFAULT_MAX_SEGMENT_GAP) -> List[np.ndarray]:
        if xs.size == 0:
            return []
        return list(np.split(xs, np.where(np.diff(xs) > max_gap)[0] + 1))

    def _extract_row_segments(self, lane_mask: np.ndarray, lane_probability: np.ndarray) -> List[_RowSegment]:
        height, width = lane_mask.shape
        segments: List[_RowSegment] = []

        # Ignoramos os primeiros 12% superiores: em FH6 a linha útil está
        # concentrada na região inferior e o horizonte gera falsos positivos.
        min_y = int(height * 0.08)

        for y in range(height - 1, min_y, -self.row_step):
            xs = np.flatnonzero(lane_mask[y] > 0)
            if xs.size < self.min_lane_pixels_per_row:
                continue
            for group in self._split_contiguous(xs):
                if group.size < self.min_lane_pixels_per_row:
                    continue
                # Segmentos muito largos tendem a ser área/ruído, não uma
                # marca de faixa. A faixa pode ficar mais grossa em baixo,
                # portanto o limite cresce com a posição vertical.
                max_width = max(12.0, width * (0.010 + 0.035 * (y / max(1, height))))
                if float(group[-1] - group[0] + 1) > max_width:
                    continue
                conf = float(np.mean(lane_probability[y, group]))
                segments.append(_RowSegment(y, float(np.mean(group)), _clip01(conf), int(group.size)))
        return segments

    # ----------------------------------------------------------- lane tracking
    def _associate_segments(self, segments: Sequence[_RowSegment], frame_width: int) -> List[_LaneTrack]:
        if not segments:
            return []
        rows: Dict[int, List[_RowSegment]] = {}
        for segment in segments:
            rows.setdefault(segment.y, []).append(segment)

        tracks: List[_LaneTrack] = []
        max_jump = max(DEFAULT_MAX_LANE_JUMP, frame_width * 0.04)

        for y in sorted(rows.keys(), reverse=True):
            candidates = sorted(rows[y], key=lambda s: s.x_center)
            used: set[int] = set()

            for candidate in candidates:
                best_idx: Optional[int] = None
                best_score = float("inf")
                for idx, track in enumerate(tracks):
                    if idx in used:
                        continue
                    dy = max(1, abs(track.last_y - candidate.y))
                    # Quanto mais perto e mais verticalmente consistente,
                    # melhor. Pequenos movimentos laterais são permitidos.
                    dx = abs(candidate.x_center - track.last_x)
                    if dx > max_jump:
                        continue
                    slope = dx / dy
                    score = dx + 8.0 * slope - 18.0 * candidate.confidence
                    if score < best_score:
                        best_score = score
                        best_idx = idx

                if best_idx is None:
                    if len(tracks) >= self.max_lanes:
                        continue
                    track = _LaneTrack()
                    track.add(candidate)
                    tracks.append(track)
                else:
                    tracks[best_idx].add(candidate)
                    used.add(best_idx)

        return tracks

    def _track_to_lane(self, track: _LaneTrack, width: int, height: int) -> List[LanePoint]:
        points = [
            LanePoint(
                x=float(np.clip(x, 0, width - 1)),
                y=float(np.clip(y, 0, height - 1)),
                confidence=_clip01(conf),
                valid=True,
            )
            for y, x, conf in track.points
        ]
        points.sort(key=lambda p: p.y)
        return points

    @staticmethod
    def _lane_reference_x(lane: Sequence[LanePoint], height: int) -> float:
        if not lane:
            return float("nan")
        target = height * 0.82
        return float(min(lane, key=lambda p: abs(p.y - target)).x)

    def _lane_quality(self, lane: Sequence[LanePoint], width: int, height: int) -> float:
        if len(lane) < 2:
            return 0.0
        xs = np.asarray([p.x for p in lane], dtype=np.float32)
        ys = np.asarray([p.y for p in lane], dtype=np.float32)
        cs = np.asarray([p.confidence for p in lane], dtype=np.float32)
        if not (np.isfinite(xs).all() and np.isfinite(ys).all() and np.isfinite(cs).all()):
            return 0.0
        vertical = float(ys.max() - ys.min())
        vertical_score = _clip01(vertical / max(1.0, height * 0.45))
        confidence = float(np.mean(cs))
        return _clip01(0.65 * confidence + 0.35 * vertical_score)

    def _validate_lane(self, lane: Sequence[LanePoint], width: int, height: int) -> bool:
        if len(lane) < self.min_points_per_lane:
            return False
        if any(not p.is_valid() or p.x < 0 or p.x >= width or p.y < 0 or p.y >= height for p in lane):
            return False
        ys = [p.y for p in lane]
        if max(ys) - min(ys) < DEFAULT_MIN_VERTICAL_SPAN:
            return False
        return self._lane_quality(lane, width, height) >= DEFAULT_MIN_LANE_CONFIDENCE

    def _sort_lanes(self, lanes: List[List[LanePoint]], height: int) -> List[List[LanePoint]]:
        return sorted(lanes, key=lambda lane: self._lane_reference_x(lane, height))

    # --------------------------------------------------------------- objects
    @staticmethod
    def _box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])
        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area_a = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
        area_b = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
        return inter / np.maximum(area_a + area_b - inter, 1e-8)

    @classmethod
    def _nms(cls, boxes: np.ndarray, scores: np.ndarray, threshold: float) -> List[int]:
        if boxes.size == 0:
            return []
        order = np.argsort(scores)[::-1]
        keep: List[int] = []
        while order.size:
            idx = int(order[0])
            keep.append(idx)
            if order.size == 1:
                break
            iou = cls._box_iou(boxes[idx], boxes[order[1:]])
            order = order[1:][iou <= threshold]
        return keep

    def _map_box_to_frame(self, box: Sequence[float], width: int, height: int) -> Tuple[float, float, float, float]:
        meta = self._last_preprocess_meta
        cx, cy, bw, bh = map(float, box)
        x1, y1 = cx - bw * 0.5, cy - bh * 0.5
        x2, y2 = cx + bw * 0.5, cy + bh * 0.5
        if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 2.0:
            x1 *= self.input_width; x2 *= self.input_width
            y1 *= self.input_height; y2 *= self.input_height
        if meta is not None:
            x1 = (x1 - meta.pad_left) / meta.scale
            x2 = (x2 - meta.pad_left) / meta.scale
            y1 = (y1 - meta.pad_top) / meta.scale
            y2 = (y2 - meta.pad_top) / meta.scale
        return (
            float(np.clip(x1, 0, width - 1)),
            float(np.clip(y1, 0, height - 1)),
            float(np.clip(x2, 0, width - 1)),
            float(np.clip(y2, 0, height - 1)),
        )

    def _extract_objects(self, outputs: Any, width: int, height: int) -> List[ObjectDetection]:
        tensors = _flatten_tensors(outputs)
        candidates = [t for t in tensors if t.ndim == 3 and t.shape[-1] >= 6]
        if not candidates:
            return []
        tensor = max(candidates, key=lambda t: int(t.numel()))
        arr = tensor.detach().float().cpu().numpy()[0]
        if arr.ndim != 2:
            return []
        if arr.shape[0] < arr.shape[1] and arr.shape[0] <= 100:
            arr = arr.T
        if arr.shape[1] < 6:
            return []
        boxes = arr[:, :4]
        objectness = arr[:, 4]
        class_scores = arr[:, 5:]
        if class_scores.shape[1]:
            class_ids = np.argmax(class_scores, axis=1)
            scores = objectness * np.max(class_scores, axis=1)
        else:
            class_ids = np.zeros(len(arr), dtype=np.int32)
            scores = objectness
        valid = np.isfinite(boxes).all(axis=1) & np.isfinite(scores) & (scores >= DEFAULT_OBJECT_CONFIDENCE)
        if not np.any(valid):
            return []
        boxes, scores, class_ids = boxes[valid], scores[valid], class_ids[valid]
        frame_boxes = np.asarray([self._map_box_to_frame(b, width, height) for b in boxes], dtype=np.float32)
        good = (frame_boxes[:, 2] > frame_boxes[:, 0]) & (frame_boxes[:, 3] > frame_boxes[:, 1])
        if not np.any(good):
            return []
        frame_boxes, scores, class_ids = frame_boxes[good], scores[good], class_ids[good]
        keep: List[int] = []
        for cid in np.unique(class_ids):
            idx = np.flatnonzero(class_ids == cid)
            keep.extend(int(idx[i]) for i in self._nms(frame_boxes[idx], scores[idx], DEFAULT_OBJECT_IOU))
        keep.sort(key=lambda i: float(scores[i]), reverse=True)
        return [
            ObjectDetection(
                class_id=int(class_ids[i]),
                confidence=_clip01(scores[i]),
                x1=float(frame_boxes[i, 0]), y1=float(frame_boxes[i, 1]),
                x2=float(frame_boxes[i, 2]), y2=float(frame_boxes[i, 3]),
                frame_width=width, frame_height=height,
            )
            for i in keep
        ]

    # --------------------------------------------------------------- result
    def _build_lane_result(
        self,
        lanes: List[List[LanePoint]],
        width: int,
        height: int,
        objects: List[ObjectDetection],
        drivable_mask: Optional[np.ndarray],
        inference_ms: float,
        output_shapes: List[Tuple[int, ...]],
    ) -> LaneDetectionResult:
        lanes = self._sort_lanes(lanes, height)
        confidences = [self._lane_quality(l, width, height) for l in lanes]

        center_x = width * 0.5
        left = [i for i, lane in enumerate(lanes) if self._lane_reference_x(lane, height) < center_x]
        right = [i for i, lane in enumerate(lanes) if self._lane_reference_x(lane, height) >= center_x]
        left_idx = max(left, key=lambda i: self._lane_reference_x(lanes[i], height)) if left else None
        right_idx = min(right, key=lambda i: self._lane_reference_x(lanes[i], height)) if right else None

        used = {i for i in (left_idx, right_idx) if i is not None}
        additional = [lane for i, lane in enumerate(lanes) if i not in used]

        metadata = {
            "device": self.get_device_name(),
            "fp16": self.fp16_active,
            "inference_ms": inference_ms,
            "model_output_shapes": output_shapes,
            "coordinate_system": "original_frame",
            "lane_count": len(lanes),
            "vehicle_count": sum(obj.is_vehicle for obj in objects),
            "object_count": len(objects),
            "drivable_pixels": int(np.count_nonzero(drivable_mask)) if drivable_mask is not None else 0,
            "model_path": str(self.model_path),
        }
        return LaneDetectionResult(
            lanes=lanes,
            lane_confidences=confidences,
            current_lane_index=None,
            left_lane=lanes[left_idx] if left_idx is not None else [],
            right_lane=lanes[right_idx] if right_idx is not None else [],
            additional_lanes=additional,
            left_confidence=confidences[left_idx] if left_idx is not None else 0.0,
            right_confidence=confidences[right_idx] if right_idx is not None else 0.0,
            valid=bool(lanes),
            num_lanes_detected=len(lanes),
            input_width=width,
            input_height=height,
            model_output_shape=output_shapes[-1] if output_shapes else tuple(),
            objects=objects,
            drivable_area_mask=drivable_mask,
            metadata=metadata,
        )

    # ------------------------------------------------------------------ detect
    def detect(self, frame: np.ndarray) -> LaneDetectionResult:
        started = time.perf_counter()
        try:
            width, height = self._validate_frame(frame)
            outputs = self.infer(frame)
            shapes = list(self.last_output_shapes)

            drivable_tensor, lane_tensor = self._find_segmentation_heads(outputs)
            if lane_tensor is None:
                raise RuntimeError(
                    f"Cabeça de lane segmentation não encontrada. outputs={shapes}"
                )

            lane_prob_model = self._segmentation_probability(self._tensor_to_numpy(lane_tensor))
            lane_prob = self._resize_probability_to_frame(lane_prob_model, width, height)
            lane_mask = self._build_lane_mask(lane_prob)

            drivable_mask = None
            if drivable_tensor is not None:
                drivable_prob_model = self._segmentation_probability(self._tensor_to_numpy(drivable_tensor))
                drivable_prob = self._resize_probability_to_frame(drivable_prob_model, width, height)
                drivable_mask = self._build_drivable_mask(drivable_prob)

            segments = self._extract_row_segments(lane_mask, lane_prob)
            tracks = self._associate_segments(segments, width)
            lanes: List[List[LanePoint]] = []
            for track in tracks:
                lane = self._track_to_lane(track, width, height)
                if self._validate_lane(lane, width, height):
                    lanes.append(lane)
                if len(lanes) >= self.max_lanes:
                    break

            objects = self._extract_objects(outputs, width, height)
            inference_ms = float(self.last_diagnostics.get("inference_ms", 0.0))
            result = self._build_lane_result(
                lanes, width, height, objects, drivable_mask, inference_ms, shapes
            )

            total_ms = (time.perf_counter() - started) * 1000.0
            self.last_diagnostics.update({
                "total_ms": total_ms,
                "frame_width": width,
                "frame_height": height,
                "lane_threshold": self.lane_threshold,
                "drivable_threshold": DEFAULT_DRIVABLE_THRESHOLD,
                "lane_mask_pixels": int(np.count_nonzero(lane_mask)),
                "lane_mask_ratio": float(np.count_nonzero(lane_mask)) / max(1, width * height),
                "segment_count": len(segments),
                "raw_tracks": len(tracks),
                "valid_lanes": len(lanes),
                "drivable_pixels": int(np.count_nonzero(drivable_mask)) if drivable_mask is not None else 0,
                "coordinate_system": "original_frame",
                "segmentation_heads": {
                    "drivable": tuple(int(v) for v in drivable_tensor.shape) if drivable_tensor is not None else None,
                    "lane": tuple(int(v) for v in lane_tensor.shape) if lane_tensor is not None else None,
                },
            })
            result.metadata.update(self.last_diagnostics)
            self.last_result = result
            return result

        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            LOGGER.exception("YOLOPv2 detect() falhou")
            result = LaneDetectionResult(
                input_width=frame.shape[1] if isinstance(frame, np.ndarray) and frame.ndim >= 2 else 0,
                input_height=frame.shape[0] if isinstance(frame, np.ndarray) and frame.ndim >= 2 else 0,
                model_output_shape=self.last_output_shapes[-1] if self.last_output_shapes else tuple(),
                error=self.last_error,
                metadata={
                    "coordinate_system": "original_frame",
                    "fatal_error": self.last_error,
                    "model_output_shapes": self.last_output_shapes,
                },
            )
            self.last_result = result
            return result

    # ------------------------------------------------------------- image utils
    @staticmethod
    def load_image(path: str | Path) -> np.ndarray:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        data = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Não foi possível carregar: {path}")
        return image

    @staticmethod
    def save_image(path: str | Path, image: np.ndarray) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        suffix = path.suffix or ".png"
        ok, encoded = cv2.imencode(suffix, image)
        if not ok:
            raise RuntimeError(f"Falha ao codificar: {path}")
        encoded.tofile(str(path))


YOLOPDetector = YOLOPLaneDetector


def create_default_detector(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    **kwargs: Any,
) -> YOLOPLaneDetector:
    return YOLOPLaneDetector(model_path=model_path, **kwargs)


__all__ = [
    "ObjectDetection",
    "LanePoint",
    "LaneDetectionResult",
    "YOLOPDetector",
    "YOLOPLaneDetector",
    "create_default_detector",
]