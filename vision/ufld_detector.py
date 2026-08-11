"""
vision/ufld_detector.py

Detector UFLD para Forza Horizon ADAS/LKA.

Responsabilidades:
- Carregar o parsingNet do Ultra-Fast-Lane-Detection.
- Pré-processar frames BGR.
- Executar inferência em CUDA/CPU.
- Decodificar a saída [B, 201, 18, 4].
- Detectar até 4 faixas.
- Identificar a faixa atual do veículo.
- Separar faixa esquerda e direita.
- Converter coordenadas de 800x288 para a resolução
  original do frame.
"""

from __future__ import annotations

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch


logger = logging.getLogger(__name__)


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

UFLD_ROOT = PROJECT_ROOT / "Ultra-Fast-Lane-Detection"

if not UFLD_ROOT.exists():
    raise FileNotFoundError(
        f"Diretório UFLD não encontrado: {UFLD_ROOT}"
    )

if str(UFLD_ROOT) not in sys.path:
    sys.path.insert(0, str(UFLD_ROOT))


try:
    from model.model import parsingNet
except Exception as exc:
    raise ImportError(
        "Não foi possível importar "
        "'model.model.parsingNet'.\n"
        f"Verifique o diretório UFLD:\n{UFLD_ROOT}"
    ) from exc


# ============================================================================
# CONSTANTES UFLD / CULANE
# ============================================================================

UFLD_INPUT_WIDTH = 800
UFLD_INPUT_HEIGHT = 288

UFLD_GRIDING_NUM = 201
UFLD_VALID_GRID_NUM = 200

UFLD_NUM_ROW_ANCHORS = 18
UFLD_NUM_LANES = 4


# Row anchors oficiais do CULane para input 800x288.
CULANE_ROW_ANCHORS = np.asarray(
    [
        121,
        131,
        141,
        150,
        160,
        170,
        180,
        189,
        199,
        209,
        219,
        228,
        238,
        248,
        258,
        267,
        277,
        287,
    ],
    dtype=np.float32,
)


IMAGE_MEAN = np.asarray(
    [0.485, 0.456, 0.406],
    dtype=np.float32,
)

IMAGE_STD = np.asarray(
    [0.229, 0.224, 0.225],
    dtype=np.float32,
)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class LanePoint:
    """
    Um ponto pertencente a uma faixa.
    """

    x: float
    y: float
    confidence: float
    valid: bool = True


@dataclass
class LaneDetectionResult:
    """
    Resultado completo da detecção UFLD.
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

    input_width: int = UFLD_INPUT_WIDTH
    input_height: int = UFLD_INPUT_HEIGHT

    model_output_shape: Tuple[int, ...] = field(
        default_factory=tuple
    )

    error: Optional[str] = None

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


# ============================================================================
# DETECTOR
# ============================================================================

class UFLDLaneDetector:

    def __init__(
        self,
        model_path: str,
        backbone: str = "18",
        cls_dim: Tuple[int, int, int] = (
            UFLD_GRIDING_NUM,
            UFLD_NUM_ROW_ANCHORS,
            UFLD_NUM_LANES,
        ),
        use_aux: bool = False,
        input_width: int = UFLD_INPUT_WIDTH,
        input_height: int = UFLD_INPUT_HEIGHT,
        mean: Sequence[float] = IMAGE_MEAN,
        std: Sequence[float] = IMAGE_STD,
        confidence_threshold: float = 0.60,
        no_lane_threshold: float = 0.55,
        min_valid_points: int = 5,
        max_x_jump: float = 90.0,
        max_gap_rows: int = 5,
        min_lane_width: float = 35.0,
        max_lane_width: float = 500.0,
        device: Optional[str] = None,
    ) -> None:

        self.model_path = str(model_path)
        self.backbone = str(backbone)

        self.cls_dim = tuple(
            int(value)
            for value in cls_dim
        )

        self.use_aux = bool(use_aux)

        self.input_width = int(input_width)
        self.input_height = int(input_height)

        self.mean = np.asarray(
            mean,
            dtype=np.float32,
        ).reshape(1, 1, 3)

        self.std = np.asarray(
            std,
            dtype=np.float32,
        ).reshape(1, 1, 3)

        if np.any(self.std <= 0):
            raise ValueError(
                "std deve conter somente valores positivos."
            )

        self.confidence_threshold = float(
            np.clip(
                confidence_threshold,
                0.0,
                1.0,
            )
        )

        self.no_lane_threshold = float(
            np.clip(
                no_lane_threshold,
                0.0,
                1.0,
            )
        )

        self.min_valid_points = max(
            2,
            int(min_valid_points),
        )

        self.max_x_jump = max(
            1.0,
            float(max_x_jump),
        )

        self.max_gap_rows = max(
            0,
            int(max_gap_rows),
        )

        self.min_lane_width = max(
            1.0,
            float(min_lane_width),
        )

        self.max_lane_width = max(
            self.min_lane_width,
            float(max_lane_width),
        )

        self.device = self._resolve_device(device)

        self.griding_num = int(self.cls_dim[0])
        self.cls_num_per_lane = int(self.cls_dim[1])
        self.num_lanes = int(self.cls_dim[2])

        if self.cls_dim != (201, 18, 4):
            raise ValueError(
                "Este detector foi configurado para "
                "o modelo CULane padrão:\n"
                "cls_dim=(201, 18, 4)\n"
                f"Recebido: {self.cls_dim}"
            )

        if self.input_width != 800:
            raise ValueError(
                "Este detector CULane utiliza input_width=800."
            )

        if self.input_height != 288:
            raise ValueError(
                "Este detector CULane utiliza input_height=288."
            )

        self.row_anchors = CULANE_ROW_ANCHORS.copy()

        self.model: Optional[torch.nn.Module] = None

        self.loaded = False

        self.last_output_shape: Tuple[int, ...] = tuple()

        self.last_error: Optional[str] = None

        self.last_result: Optional[
            LaneDetectionResult
        ] = None

        # Resolução ORIGINAL do último frame.
        self.last_frame_width = self.input_width
        self.last_frame_height = self.input_height

        # Índices 0...199 usados no softmax.
        self._grid_indices = np.arange(
            UFLD_VALID_GRID_NUM,
            dtype=np.float64,
        )

    # ========================================================================
    # DEVICE
    # ========================================================================

    @staticmethod
    def _resolve_device(
        requested: Optional[str],
    ) -> str:

        if requested is None:
            return (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        requested = (
            str(requested)
            .strip()
            .lower()
        )

        if requested.startswith("cuda"):

            if torch.cuda.is_available():
                return requested

            logger.warning(
                "[UFLD] CUDA solicitado, "
                "mas CUDA não está disponível. "
                "Usando CPU."
            )

            return "cpu"

        return "cpu"

    def get_device_name(self) -> str:

        if self.device.startswith("cuda"):

            try:
                return torch.cuda.get_device_name(
                    torch.cuda.current_device()
                )
            except Exception:
                return "CUDA"

        return "CPU"

    # ========================================================================
    # MODEL
    # ========================================================================

    def model_exists(self) -> bool:
        return os.path.isfile(self.model_path)

    def load_model(self) -> bool:

        if self.loaded and self.model is not None:
            return True

        self.last_error = None

        if not self.model_exists():

            self.last_error = (
                "Checkpoint não encontrado: "
                f"{self.model_path}"
            )

            logger.error(
                "[UFLD] %s",
                self.last_error,
            )

            return False

        try:

            logger.info(
                "[UFLD] Construindo parsingNet..."
            )

            model = parsingNet(
                pretrained=False,
                backbone=self.backbone,
                cls_dim=self.cls_dim,
                use_aux=self.use_aux,
            )

            logger.info(
                "[UFLD] Carregando checkpoint: %s",
                self.model_path,
            )

            checkpoint = torch.load(
                self.model_path,
                map_location="cpu",
                weights_only=False,
            )

            state_dict = self._extract_state_dict(
                checkpoint
            )

            state_dict = self._clean_state_dict(
                state_dict
            )

            self._validate_checkpoint_shapes(
                model,
                state_dict,
            )

            missing, unexpected = (
                model.load_state_dict(
                    state_dict,
                    strict=False,
                )
            )

            if missing or unexpected:

                raise RuntimeError(
                    "Checkpoint não foi carregado "
                    "integralmente.\n"
                    f"Missing: {len(missing)}\n"
                    f"Unexpected: {len(unexpected)}\n"
                    f"Missing preview: {list(missing[:10])}\n"
                    f"Unexpected preview: {list(unexpected[:10])}"
                )

            model.to(self.device)
            model.eval()

            self.model = model
            self.loaded = True

            logger.info(
                "[UFLD] Modelo carregado com sucesso."
            )

            logger.info(
                "[UFLD] Device: %s",
                self.device,
            )

            logger.info(
                "[UFLD] Device name: %s",
                self.get_device_name(),
            )

            return True

        except Exception as exc:

            self.model = None
            self.loaded = False

            self.last_error = (
                f"{type(exc).__name__}: {exc}"
            )

            logger.exception(
                "[UFLD] Falha ao carregar modelo."
            )

            return False

    @staticmethod
    def _extract_state_dict(
        checkpoint: Any,
    ) -> Dict[str, torch.Tensor]:

        if not isinstance(
            checkpoint,
            dict,
        ):
            raise RuntimeError(
                "Checkpoint UFLD inválido."
            )

        # Alguns checkpoints utilizam "model".
        model_value = checkpoint.get("model")

        if isinstance(
            model_value,
            dict,
        ):
            return model_value

        # Outros utilizam estas chaves.
        for key in (
            "state_dict",
            "model_state_dict",
            "net",
            "weights",
        ):

            value = checkpoint.get(key)

            if isinstance(value, dict):
                return value

        # O próprio checkpoint pode ser o state_dict.
        if checkpoint:

            if all(
                isinstance(key, str)
                for key in checkpoint.keys()
            ):

                if all(
                    isinstance(value, torch.Tensor)
                    for value in checkpoint.values()
                ):
                    return checkpoint

        raise RuntimeError(
            "Formato de checkpoint UFLD não reconhecido."
        )

    @staticmethod
    def _clean_state_dict(
        state_dict: Dict[str, Any],
    ) -> Dict[str, Any]:

        cleaned: Dict[str, Any] = {}

        backbone_prefixes = (
            "conv1.",
            "bn1.",
            "layer1.",
            "layer2.",
            "layer3.",
            "layer4.",
        )

        for key, value in state_dict.items():

            new_key = str(key)

            # Remove prefixes adicionados por DataParallel.
            while new_key.startswith("module."):
                new_key = new_key[
                    len("module.") :
                ]

            # Remove prefixo net.
            while new_key.startswith("net."):
                new_key = new_key[
                    len("net.") :
                ]

            # Compatibilidade com checkpoints
            # que salvam o backbone sem "model.".
            if new_key.startswith(
                backbone_prefixes
            ):
                new_key = "model." + new_key

            cleaned[new_key] = value

        return cleaned

    @staticmethod
    def _validate_checkpoint_shapes(
        model: torch.nn.Module,
        state_dict: Dict[str, Any],
    ) -> None:

        model_state = model.state_dict()

        incompatible = []

        for key, value in state_dict.items():

            if key not in model_state:
                continue

            if not isinstance(
                value,
                torch.Tensor,
            ):
                continue

            expected = tuple(
                model_state[key].shape
            )

            received = tuple(
                value.shape
            )

            if expected != received:

                incompatible.append(
                    (
                        key,
                        expected,
                        received,
                    )
                )

        if incompatible:

            key, expected, received = (
                incompatible[0]
            )

            raise RuntimeError(
                "Checkpoint incompatível com "
                "a arquitetura UFLD.\n"
                f"Parâmetro: {key}\n"
                f"Esperado: {expected}\n"
                f"Recebido: {received}"
            )

    # ========================================================================
    # PREPROCESS
    # ========================================================================

    def preprocess(
        self,
        frame: np.ndarray,
    ) -> torch.Tensor:

        if frame is None:
            raise ValueError(
                "frame não pode ser None."
            )

        if not isinstance(
            frame,
            np.ndarray,
        ):
            raise TypeError(
                "frame deve ser numpy.ndarray."
            )

        if frame.size == 0:
            raise ValueError(
                "frame está vazio."
            )

        if (
            frame.ndim != 3
            or frame.shape[2] != 3
        ):
            raise ValueError(
                "frame deve possuir shape HxWx3."
            )

        # ================================================================
        # IMPORTANTE:
        # Guardamos a resolução ORIGINAL.
        # ================================================================

        self.last_frame_height = int(
            frame.shape[0]
        )

        self.last_frame_width = int(
            frame.shape[1]
        )

        # Resize para o tamanho esperado pelo UFLD.
        resized = cv2.resize(
            frame,
            (
                self.input_width,
                self.input_height,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

        # OpenCV = BGR
        # UFLD / ImageNet = RGB
        rgb = cv2.cvtColor(
            resized,
            cv2.COLOR_BGR2RGB,
        )

        image = (
            rgb.astype(
                np.float32
            )
            / 255.0
        )

        # Normalização ImageNet.
        image = (
            image - self.mean
        ) / self.std

        tensor = torch.from_numpy(
            image
        )

        # HWC -> CHW -> BATCH
        tensor = (
            tensor
            .permute(2, 0, 1)
            .contiguous()
            .unsqueeze(0)
        )

        tensor = tensor.to(
            device=self.device,
            dtype=torch.float32,
            non_blocking=self.device.startswith(
                "cuda"
            ),
        )

        return tensor

    # ========================================================================
    # INFERENCE
    # ========================================================================

    @torch.inference_mode()
    def infer(
        self,
        tensor: torch.Tensor,
    ) -> torch.Tensor:

        if (
            self.model is None
            or not self.loaded
        ):
            raise RuntimeError(
                "Modelo UFLD não está carregado."
            )

        if not isinstance(
            tensor,
            torch.Tensor,
        ):
            raise TypeError(
                "tensor deve ser torch.Tensor."
            )

        if tensor.ndim != 4:
            raise ValueError(
                "Tensor deve possuir shape [B,C,H,W]."
            )

        tensor = tensor.to(
            self.device,
            dtype=torch.float32,
        )

        output = self.model(
            tensor
        )

        return self._extract_output_tensor(
            output
        )

    @staticmethod
    def _extract_output_tensor(
        output: Any,
    ) -> torch.Tensor:

        if isinstance(
            output,
            torch.Tensor,
        ):
            return output

        if isinstance(
            output,
            (tuple, list),
        ):

            for item in output:

                if isinstance(
                    item,
                    torch.Tensor,
                ):
                    return item

        if isinstance(
            output,
            dict,
        ):

            for key in (
                "cls",
                "classification",
                "output",
                "out",
            ):

                value = output.get(key)

                if isinstance(
                    value,
                    torch.Tensor,
                ):
                    return value

            for value in output.values():

                if isinstance(
                    value,
                    torch.Tensor,
                ):
                    return value

        raise RuntimeError(
            "Não foi possível extrair o tensor "
            "principal da saída UFLD."
        )

    # ========================================================================
    # NUMERIC HELPERS
    # ========================================================================

    @staticmethod
    def _stable_softmax(
        values: np.ndarray,
        axis: int = 0,
    ) -> np.ndarray:

        values = np.asarray(
            values,
            dtype=np.float64,
        )

        if values.size == 0:
            return values

        maximum = np.max(
            values,
            axis=axis,
            keepdims=True,
        )

        shifted = values - maximum

        shifted = np.clip(
            shifted,
            -80.0,
            80.0,
        )

        exponential = np.exp(
            shifted
        )

        denominator = np.sum(
            exponential,
            axis=axis,
            keepdims=True,
        )

        denominator = np.maximum(
            denominator,
            1e-12,
        )

        return (
            exponential
            / denominator
        )

    @staticmethod
    def _logsumexp(
        values: np.ndarray,
    ) -> float:

        values = np.asarray(
            values,
            dtype=np.float64,
        )

        if values.size == 0:
            return 0.0

        maximum = float(
            np.max(values)
        )

        return (
            maximum
            + math.log(
                float(
                    np.sum(
                        np.exp(
                            np.clip(
                                values - maximum,
                                -80.0,
                                80.0,
                            )
                        )
                    )
                )
            )
        )

    # ========================================================================
    # DECODE
    # ========================================================================

    def decode(
        self,
        output: torch.Tensor,
    ) -> LaneDetectionResult:

        try:

            result = self._decode_internal(
                output
            )

            self.last_result = result
            self.last_error = None

            return result

        except Exception as exc:

            self.last_error = (
                f"{type(exc).__name__}: {exc}"
            )

            logger.exception(
                "[UFLD] Falha no decode."
            )

            result = LaneDetectionResult(
                input_width=self.input_width,
                input_height=self.input_height,
                model_output_shape=(
                    self.last_output_shape
                ),
                valid=False,
                error=self.last_error,
            )

            self.last_result = result

            return result

    def _decode_internal(
        self,
        output: torch.Tensor,
    ) -> LaneDetectionResult:

        if not isinstance(
            output,
            torch.Tensor,
        ):
            raise TypeError(
                "output deve ser torch.Tensor."
            )

        output_np = (
            output.detach()
            .to("cpu")
            .float()
            .numpy()
        )

        self.last_output_shape = tuple(
            output_np.shape
        )

        # UFLD CULane:
        #
        # [batch, griding_num, cls_num_per_lane, num_lanes]
        #
        # [1, 201, 18, 4]

        if output_np.ndim != 4:
            raise ValueError(
                "Saída UFLD inválida.\n"
                "Esperado: [B,G,R,L]\n"
                f"Recebido: {output_np.shape}"
            )

        batch, grid, rows, lanes = (
            output_np.shape
        )

        expected = (
            1,
            self.griding_num,
            self.cls_num_per_lane,
            self.num_lanes,
        )

        if (
            batch,
            grid,
            rows,
            lanes,
        ) != expected:

            raise ValueError(
                "Shape incompatível.\n"
                f"Recebido: {output_np.shape}\n"
                f"Esperado: {expected}"
            )

        logits = output_np[0]

        if not np.all(
            np.isfinite(logits)
        ):
            raise ValueError(
                "A saída UFLD contém NaN ou Inf."
            )

        # ================================================================
        # O UFLD CULane utiliza os row anchors na ordem inversa
        # em relação à saída da rede.
        # ================================================================

        logits = logits[
            :,
            ::-1,
            :,
        ]

        # 200 classes representam posições X.
        valid_logits = logits[
            :UFLD_VALID_GRID_NUM,
            :,
            :,
        ]

        # A classe 200 representa "no lane".
        no_lane_logits = logits[
            UFLD_VALID_GRID_NUM,
            :,
            :,
        ]

        decoded_lanes: List[
            List[LanePoint]
        ] = []

        lane_confidences: List[
            float
        ] = []

        # ================================================================
        # DECODIFICA CADA UMA DAS 4 LANES
        # ================================================================

        for lane_index in range(
            self.num_lanes
        ):

            lane = self._decode_single_lane(
                valid_logits=valid_logits,
                no_lane_logits=no_lane_logits,
                lane_index=lane_index,
            )

            # Remove saltos absurdos.
            lane = self._validate_lane_continuity(
                lane
            )

            valid_count = sum(
                1
                for point in lane
                if point.valid
            )

            confidence = (
                self._lane_confidence(
                    lane
                )
            )

            # Exige quantidade mínima de pontos.
            if valid_count < self.min_valid_points:

                lane = [
                    LanePoint(
                        x=0.0,
                        y=point.y,
                        confidence=0.0,
                        valid=False,
                    )
                    for point in lane
                ]

                confidence = 0.0

            decoded_lanes.append(
                lane
            )

            lane_confidences.append(
                confidence
            )

        # ================================================================
        # ORDENA AS LANES DA ESQUERDA PARA A DIREITA
        # ================================================================

        ordered_lanes, ordered_confidences = (
            self._order_lanes(
                decoded_lanes,
                lane_confidences,
            )
        )

        # ================================================================
        # ENCONTRA A FAIXA DO VEÍCULO
        # ================================================================

        current_lane_index = (
            self._find_current_lane(
                ordered_lanes
            )
        )

        left_lane: List[LanePoint] = []
        right_lane: List[LanePoint] = []

        left_confidence = 0.0
        right_confidence = 0.0

        if (
            current_lane_index is not None
            and current_lane_index + 1
            < len(ordered_lanes)
        ):

            left_lane = ordered_lanes[
                current_lane_index
            ]

            right_lane = ordered_lanes[
                current_lane_index + 1
            ]

            left_confidence = (
                ordered_confidences[
                    current_lane_index
                ]
            )

            right_confidence = (
                ordered_confidences[
                    current_lane_index + 1
                ]
            )

        # ================================================================
        # LANES ADICIONAIS
        # ================================================================

        additional_lanes: List[
            List[LanePoint]
        ] = []

        for index, lane in enumerate(
            ordered_lanes
        ):

            if (
                current_lane_index is not None
                and index in (
                    current_lane_index,
                    current_lane_index + 1,
                )
            ):
                continue

            additional_lanes.append(
                lane
            )

        # ================================================================
        # VALIDADE FINAL
        # ================================================================

        valid = (
            current_lane_index is not None
            and sum(
                point.valid
                for point in left_lane
            ) >= self.min_valid_points
            and sum(
                point.valid
                for point in right_lane
            ) >= self.min_valid_points
            and left_confidence
            >= self.confidence_threshold
            and right_confidence
            >= self.confidence_threshold
        )

        # ================================================================
        # CONVERTE 800x288 -> RESOLUÇÃO ORIGINAL
        #
        # O restante do sistema pode trabalhar diretamente
        # na resolução do screenshot/frame capturado.
        # ================================================================

        self._map_lanes_to_original_frame(
            ordered_lanes
        )

        return LaneDetectionResult(
            lanes=ordered_lanes,
            lane_confidences=ordered_confidences,
            current_lane_index=current_lane_index,
            left_lane=left_lane,
            right_lane=right_lane,
            additional_lanes=additional_lanes,
            left_confidence=left_confidence,
            right_confidence=right_confidence,
            valid=bool(valid),
            num_lanes_detected=len(
                ordered_lanes
            ),
            input_width=self.input_width,
            input_height=self.input_height,
            model_output_shape=self.last_output_shape,
        )

    # ========================================================================
    # COORDINATE MAPPING
    # ========================================================================

    def _map_lanes_to_original_frame(
        self,
        lanes: List[List[LanePoint]],
    ) -> None:

        source_width = float(
            self.last_frame_width
        )

        source_height = float(
            self.last_frame_height
        )

        if (
            source_width <= 0.0
            or source_height <= 0.0
        ):
            return

        scale_x = (
            source_width
            / float(self.input_width)
        )

        scale_y = (
            source_height
            / float(self.input_height)
        )

        for lane in lanes:

            for point in lane:

                if not point.valid:
                    continue

                point.x = float(
                    np.clip(
                        point.x * scale_x,
                        0.0,
                        source_width - 1.0,
                    )
                )

                point.y = float(
                    np.clip(
                        point.y * scale_y,
                        0.0,
                        source_height - 1.0,
                    )
                )

    # ========================================================================
    # SINGLE LANE DECODER
    # ========================================================================

    def _decode_single_lane(
        self,
        valid_logits: np.ndarray,
        no_lane_logits: np.ndarray,
        lane_index: int,
    ) -> List[LanePoint]:

        points: List[LanePoint] = []

        for row_index in range(
            self.cls_num_per_lane
        ):

            row_valid_logits = (
                valid_logits[
                    :,
                    row_index,
                    lane_index,
                ].astype(
                    np.float64
                )
            )

            no_lane_logit = float(
                no_lane_logits[
                    row_index,
                    lane_index,
                ]
            )

            y = float(
                self.row_anchors[
                    row_index
                ]
            )

            # ============================================================
            # Todas as 201 possibilidades:
            #
            # 0..199 = posição X
            # 200    = no lane
            # ============================================================

            all_logits = np.concatenate(
                [
                    row_valid_logits,
                    np.asarray(
                        [no_lane_logit],
                        dtype=np.float64,
                    ),
                ]
            )

            log_denominator = (
                self._logsumexp(
                    all_logits
                )
            )

            no_lane_probability = float(
                np.exp(
                    np.clip(
                        no_lane_logit
                        - log_denominator,
                        -80.0,
                        0.0,
                    )
                )
            )

            valid_probability_mass = float(
                1.0
                - no_lane_probability
            )

            # ============================================================
            # A rede diz que não existe lane neste row.
            # ============================================================

            if (
                no_lane_probability
                >= self.no_lane_threshold
            ):

                points.append(
                    LanePoint(
                        x=0.0,
                        y=y,
                        confidence=0.0,
                        valid=False,
                    )
                )

                continue

            # ============================================================
            # Softmax apenas nas 200 posições válidas.
            # ============================================================

            valid_probabilities = (
                self._stable_softmax(
                    row_valid_logits,
                    axis=0,
                )
            )

            if not np.all(
                np.isfinite(
                    valid_probabilities
                )
            ):

                points.append(
                    LanePoint(
                        x=0.0,
                        y=y,
                        confidence=0.0,
                        valid=False,
                    )
                )

                continue

            # ============================================================
            # Expected value / Integral Regression
            # ============================================================

            grid_position = float(
                np.sum(
                    valid_probabilities
                    * self._grid_indices
                )
            )

            # 0..199 -> 0..799
            x = (
                grid_position
                / float(
                    UFLD_VALID_GRID_NUM - 1
                )
                * float(
                    self.input_width - 1
                )
            )

            peak_probability = float(
                np.max(
                    valid_probabilities
                )
            )

            # Confiança combinando:
            #
            # - concentração da distribuição
            # - probabilidade total de existir uma lane
            confidence = (
                0.50 * peak_probability
                + 0.50 * valid_probability_mass
            )

            valid = (
                np.isfinite(x)
                and np.isfinite(confidence)
                and 0.0 <= x < self.input_width
                and confidence > 0.0
            )

            points.append(
                LanePoint(
                    x=(
                        float(x)
                        if valid
                        else 0.0
                    ),
                    y=y,
                    confidence=(
                        float(confidence)
                        if valid
                        else 0.0
                    ),
                    valid=bool(valid),
                )
            )

        return points

    # ========================================================================
    # CONTINUITY FILTER
    # ========================================================================

    def _validate_lane_continuity(
        self,
        points: List[LanePoint],
    ) -> List[LanePoint]:

        if not points:
            return points

        result = [
            LanePoint(
                x=point.x,
                y=point.y,
                confidence=point.confidence,
                valid=point.valid,
            )
            for point in points
        ]

        previous_index: Optional[int] = None
        previous_x: Optional[float] = None

        for index, point in enumerate(
            result
        ):

            if not point.valid:
                continue

            if previous_index is None:

                previous_index = index
                previous_x = point.x

                continue

            row_gap = (
                index
                - previous_index
            )

            dx = abs(
                point.x
                - float(previous_x)
            )

            allowed_jump = (
                self.max_x_jump
                * max(
                    1.0,
                    float(row_gap),
                )
            )

            # Gap muito grande + salto grande.
            if (
                row_gap > self.max_gap_rows
                and dx > self.max_x_jump
            ):

                point.valid = False
                point.x = 0.0
                point.confidence = 0.0

                continue

            # Salto incompatível.
            if dx > allowed_jump:

                point.valid = False
                point.x = 0.0
                point.confidence = 0.0

                continue

            previous_index = index
            previous_x = point.x

        return result

    # ========================================================================
    # CONFIDENCE
    # ========================================================================

    @staticmethod
    def _lane_confidence(
        lane: List[LanePoint],
    ) -> float:

        values = [
            point.confidence
            for point in lane
            if (
                point.valid
                and np.isfinite(
                    point.confidence
                )
            )
        ]

        if not values:
            return 0.0

        return float(
            np.mean(values)
        )

    # ========================================================================
    # ORDER LANES
    # ========================================================================

    def _order_lanes(
        self,
        lanes: List[List[LanePoint]],
        confidences: List[float],
    ) -> Tuple[
        List[List[LanePoint]],
        List[float],
    ]:

        candidates = []

        for index, lane in enumerate(
            lanes
        ):

            xs = [
                point.x
                for point in lane
                if (
                    point.valid
                    and np.isfinite(
                        point.x
                    )
                )
            ]

            if len(xs) < self.min_valid_points:
                continue

            center_x = float(
                np.median(xs)
            )

            candidates.append(
                (
                    center_x,
                    index,
                    lane,
                    confidences[index],
                )
            )

        # Esquerda -> direita.
        candidates.sort(
            key=lambda item: item[0]
        )

        return (
            [
                item[2]
                for item in candidates
            ],
            [
                item[3]
                for item in candidates
            ],
        )

    # ========================================================================
    # CURRENT LANE
    # ========================================================================

    def _find_current_lane(
        self,
        lanes: List[List[LanePoint]],
    ) -> Optional[int]:

        if len(lanes) < 2:
            return None

        # O veículo está no centro horizontal
        # do input 800x288.
        vehicle_x = (
            float(self.input_width)
            / 2.0
        )

        centers: List[
            Optional[float]
        ] = []

        for lane in lanes:

            xs = [
                point.x
                for point in lane
                if (
                    point.valid
                    and np.isfinite(
                        point.x
                    )
                )
            ]

            if len(xs) < self.min_valid_points:

                centers.append(None)

            else:

                centers.append(
                    float(
                        np.median(xs)
                    )
                )

        # Procura duas lanes consecutivas
        # que formem a faixa do veículo.
        for index in range(
            len(centers) - 1
        ):

            left = centers[index]
            right = centers[index + 1]

            if (
                left is None
                or right is None
            ):
                continue

            if left >= right:
                continue

            width = right - left

            if width < self.min_lane_width:
                continue

            if width > self.max_lane_width:
                continue

            if (
                left
                <= vehicle_x
                <= right
            ):
                return index

        return None

    # ========================================================================
    # DETECT
    # ========================================================================

    def detect(
        self,
        frame: np.ndarray,
    ) -> LaneDetectionResult:

        try:

            if not self.loaded:

                if not self.load_model():

                    result = LaneDetectionResult(
                        input_width=self.input_width,
                        input_height=self.input_height,
                        valid=False,
                        error=self.last_error,
                    )

                    self.last_result = result

                    return result

            tensor = self.preprocess(
                frame
            )

            output = self.infer(
                tensor
            )

            return self.decode(
                output
            )

        except Exception as exc:

            self.last_error = (
                f"{type(exc).__name__}: {exc}"
            )

            logger.exception(
                "[UFLD] Falha durante detect()."
            )

            result = LaneDetectionResult(
                input_width=self.input_width,
                input_height=self.input_height,
                model_output_shape=(
                    self.last_output_shape
                ),
                valid=False,
                error=self.last_error,
            )

            self.last_result = result

            return result

    # ========================================================================
    # WARMUP
    # ========================================================================

    @torch.inference_mode()
    def warmup(
        self,
        iterations: int = 3,
    ) -> None:

        if not self.loaded:

            if not self.load_model():

                raise RuntimeError(
                    self.last_error
                    or
                    "Falha ao carregar modelo."
                )

        iterations = max(
            1,
            int(iterations),
        )

        dummy = torch.zeros(
            (
                1,
                3,
                self.input_height,
                self.input_width,
            ),
            dtype=torch.float32,
            device=self.device,
        )

        for _ in range(iterations):

            _ = self.model(
                dummy
            )

        if self.device.startswith("cuda"):

            torch.cuda.synchronize()

        logger.info(
            "[UFLD] Warmup concluído."
        )

    # ========================================================================
    # INFO
    # ========================================================================

    def get_config(
        self,
    ) -> Dict[str, Any]:

        return {
            "model_path": self.model_path,
            "backbone": self.backbone,
            "cls_dim": self.cls_dim,
            "use_aux": self.use_aux,
            "input_width": self.input_width,
            "input_height": self.input_height,
            "device": self.device,
            "confidence_threshold": (
                self.confidence_threshold
            ),
            "no_lane_threshold": (
                self.no_lane_threshold
            ),
            "min_valid_points": (
                self.min_valid_points
            ),
            "max_x_jump": self.max_x_jump,
            "max_gap_rows": self.max_gap_rows,
            "min_lane_width": (
                self.min_lane_width
            ),
            "max_lane_width": (
                self.max_lane_width
            ),
        }

    def get_model_info(
        self,
    ) -> Dict[str, Any]:

        parameter_count = 0

        if self.model is not None:

            parameter_count = sum(
                parameter.numel()
                for parameter
                in self.model.parameters()
            )

        return {
            "model_path": self.model_path,
            "model_exists": self.model_exists(),
            "loaded": self.loaded,
            "device": self.device,
            "device_name": (
                self.get_device_name()
            ),
            "backbone": self.backbone,
            "cls_dim": self.cls_dim,
            "input_width": self.input_width,
            "input_height": self.input_height,
            "use_aux": self.use_aux,
            "griding_num": (
                self.griding_num
            ),
            "row_anchors": (
                self.cls_num_per_lane
            ),
            "num_lanes": self.num_lanes,
            "confidence_threshold": (
                self.confidence_threshold
            ),
            "no_lane_threshold": (
                self.no_lane_threshold
            ),
            "min_valid_points": (
                self.min_valid_points
            ),
            "parameter_count": (
                parameter_count
            ),
            "last_output_shape": (
                self.last_output_shape
            ),
            "last_error": self.last_error,
        }

    def get_last_output_shape(
        self,
    ) -> Tuple[int, ...]:

        return self.last_output_shape

    def get_last_error(
        self,
    ) -> Optional[str]:

        return self.last_error

    def is_ready(self) -> bool:

        return (
            self.loaded
            and self.model is not None
        )

    def release(self) -> None:

        self.model = None
        self.loaded = False
        self.last_result = None

        if (
            self.device.startswith("cuda")
            and torch.cuda.is_available()
        ):
            torch.cuda.empty_cache()


# ============================================================================
# FACTORY
# ============================================================================

def create_default_detector(
    model_path: Optional[str] = None,
    device: Optional[str] = None,
) -> UFLDLaneDetector:

    if model_path is None:

        candidates = [
            UFLD_ROOT / "culane_18.pth",
            UFLD_ROOT / "culane" / "18.pth",
            UFLD_ROOT / "culane" / "culane_18.pth",
            PROJECT_ROOT / "culane_18.pth",
            PROJECT_ROOT / "culane" / "18.pth",
        ]

        model_file = next(
            (
                path
                for path in candidates
                if path.is_file()
            ),
            None,
        )

        if model_file is None:

            searched = "\n".join(
                f"  - {path}"
                for path in candidates
            )

            raise FileNotFoundError(
                "Checkpoint UFLD não encontrado.\n\n"
                "Arquivos procurados:\n"
                f"{searched}"
            )

        model_path = str(
            model_file
        )

    return UFLDLaneDetector(
        model_path=model_path,
        backbone="18",
        cls_dim=(201, 18, 4),
        use_aux=False,
        input_width=800,
        input_height=288,
        confidence_threshold=0.60,
        no_lane_threshold=0.55,
        min_valid_points=5,
        max_x_jump=90.0,
        max_gap_rows=5,
        min_lane_width=35.0,
        max_lane_width=500.0,
        device=device,
    )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "LanePoint",
    "LaneDetectionResult",
    "UFLDLaneDetector",
    "CULANE_ROW_ANCHORS",
    "create_default_detector",
]