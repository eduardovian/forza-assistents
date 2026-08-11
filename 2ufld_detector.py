
"""
vision/ufld_detector.py

Detector UFLD para o projeto Forza Horizon 6 ADAS/LKA.

Compatível com:

Ultra-Fast-Lane-Detection/
    model/
        model.py
        backbone.py
    culane/
        _18.pth

Arquitetura:

    backbone = 18
    cls_dim = (201, 18, 4)

Saída:

    [B, 201, 18, 4]

    0..199 = posições da grade
    200    = NO-LANE

Princípios:

    - Não inventar lanes.
    - NO-LANE nunca vira X=0 válido.
    - Não interpolar pontos ausentes.
    - Softmax somente nas 200 posições válidas.
    - Expectativa probabilística da grade.
    - Classe 200 tratada separadamente como NO-LANE.
    - Continuidade espacial.
    - Validação geométrica.
    - Ordenação esquerda -> direita.
    - Quatro lanes independentes.
    - Fail-safe.
    - FP32.
    - Compatibilidade com left_lane/right_lane.
"""

from __future__ import annotations

import logging
import os
import sys

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch


# ============================================================================
# LOGGING
# ============================================================================

logger = logging.getLogger(__name__)


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UFLD_ROOT = PROJECT_ROOT / "Ultra-Fast-Lane-Detection"

if not UFLD_ROOT.exists():
    raise FileNotFoundError(
        f"Diretório UFLD não encontrado: {UFLD_ROOT}"
    )

if str(UFLD_ROOT) not in sys.path:
    sys.path.insert(0, str(UFLD_ROOT))


# ============================================================================
# UFLD IMPORT
# ============================================================================

try:
    from model.model import parsingNet
except Exception as exc:
    raise ImportError(
        "Não foi possível importar "
        "'model.model.parsingNet'. "
        f"Verifique o diretório UFLD: {UFLD_ROOT}"
    ) from exc


# ============================================================================
# CONSTANTES
# ============================================================================

UFLD_INPUT_WIDTH = 800
UFLD_INPUT_HEIGHT = 288

# 200 posições + 1 NO-LANE
UFLD_GRIDING_NUM = 201

UFLD_VALID_GRID_NUM = 200

UFLD_NUM_ROW_ANCHORS = 18
UFLD_NUM_LANES = 4

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
    Ponto individual de uma lane.

    x:
        Coordenada horizontal no espaço 800x288.

    y:
        Coordenada vertical no espaço 800x288.

    confidence:
        Confiança associada ao ponto.

    valid:
        Indica se o ponto é uma detecção válida.
    """

    x: float
    y: float
    confidence: float
    valid: bool = True


@dataclass
class LaneDetectionResult:
    """
    Resultado completo da detecção.
    """

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
    """
    Detector UFLD para o checkpoint CULane ResNet-18.
    """

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

        self.cls_dim = tuple(int(v) for v in cls_dim)

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
            np.clip(confidence_threshold, 0.0, 1.0)
        )

        self.no_lane_threshold = float(
            np.clip(no_lane_threshold, 0.0, 1.0)
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

        if self.griding_num != UFLD_GRIDING_NUM:
            raise ValueError(
                "Este detector exige griding_num=201. "
                f"Recebido: {self.griding_num}"
            )

        if self.cls_num_per_lane != UFLD_NUM_ROW_ANCHORS:
            raise ValueError(
                "Este detector exige 18 row anchors. "
                f"Recebido: {self.cls_num_per_lane}"
            )

        if self.num_lanes != UFLD_NUM_LANES:
            raise ValueError(
                "Este detector exige exatamente 4 lanes. "
                f"Recebido: {self.num_lanes}"
            )

        self.row_anchors = CULANE_ROW_ANCHORS.copy()

        self.model: Optional[torch.nn.Module] = None

        self.loaded = False

        self.last_output_shape: Tuple[int, ...] = tuple()

        self.last_error: Optional[str] = None

        self.last_result: Optional[LaneDetectionResult] = None

        # Somente as 200 posições válidas.
        self._grid_indices = np.arange(
            UFLD_VALID_GRID_NUM,
            dtype=np.float32,
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

        requested = str(requested).lower().strip()

        if requested.startswith("cuda"):
            if torch.cuda.is_available():
                return requested

            logger.warning(
                "[UFLD] CUDA solicitado, mas não está disponível. "
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
    # MODEL PATH
    # ========================================================================

    def model_exists(self) -> bool:
        return os.path.isfile(self.model_path)

    # ========================================================================
    # MODEL LOADING
    # ========================================================================

    def load_model(self) -> bool:
        """
        Carrega o checkpoint UFLD.

        O checkpoint possui pesos do backbone sem o prefixo
        "model.", enquanto parsingNet mantém o backbone dentro
        de model.
        """

        if self.loaded and self.model is not None:
            return True

        self.last_error = None

        if not self.model_exists():
            self.last_error = (
                f"Checkpoint não encontrado: {self.model_path}"
            )

            logger.error(
                "[UFLD] %s",
                self.last_error,
            )

            return False

        try:

            model = parsingNet(
                pretrained=False,
                backbone=self.backbone,
                cls_dim=self.cls_dim,
                use_aux=self.use_aux,
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

            missing, unexpected = model.load_state_dict(
                state_dict,
                strict=False,
            )

            if missing or unexpected:
                raise RuntimeError(
                    "Checkpoint não carregado integralmente. "
                    f"missing={len(missing)}, "
                    f"unexpected={len(unexpected)}. "
                    f"missing_preview={list(missing[:10])}, "
                    f"unexpected_preview={list(unexpected[:10])}"
                )

            model.to(self.device)
            model.eval()

            self.model = model
            self.loaded = True

            logger.info(
                "[UFLD] Modelo carregado integralmente."
            )

            logger.info(
                "[UFLD] Device: %s (%s)",
                self.device,
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

    # ========================================================================
    # CHECKPOINT
    # ========================================================================

    @staticmethod
    def _extract_state_dict(
        checkpoint: Any,
    ) -> Dict[str, torch.Tensor]:

        if not isinstance(checkpoint, dict):
            raise RuntimeError(
                "Checkpoint UFLD inválido."
            )

        if isinstance(
            checkpoint.get("model"),
            dict,
        ):
            return checkpoint["model"]

        for key in (
            "state_dict",
            "model_state_dict",
            "net",
            "weights",
        ):
            value = checkpoint.get(key)

            if isinstance(value, dict):
                return value

        if checkpoint and all(
            isinstance(k, str)
            for k in checkpoint.keys()
        ):
            if all(
                isinstance(v, torch.Tensor)
                for v in checkpoint.values()
            ):
                return checkpoint

        raise RuntimeError(
            "Formato de checkpoint não reconhecido."
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

            while new_key.startswith("module."):
                new_key = new_key[len("module."):]

            while new_key.startswith("net."):
                new_key = new_key[len("net."):]

            if new_key.startswith(backbone_prefixes):
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

            if not isinstance(value, torch.Tensor):
                continue

            expected = tuple(model_state[key].shape)
            received = tuple(value.shape)

            if expected != received:
                incompatible.append(
                    (
                        key,
                        expected,
                        received,
                    )
                )

        if incompatible:

            key, expected, received = incompatible[0]

            raise RuntimeError(
                "Checkpoint incompatível com a arquitetura. "
                f"Parâmetro='{key}', "
                f"esperado={expected}, "
                f"recebido={received}."
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

        if not isinstance(frame, np.ndarray):
            raise TypeError(
                "frame deve ser numpy.ndarray."
            )

        if frame.size == 0:
            raise ValueError(
                "frame está vazio."
            )

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                "frame deve possuir shape HxWx3."
            )

        resized = cv2.resize(
            frame,
            (
                self.input_width,
                self.input_height,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

        rgb = cv2.cvtColor(
            resized,
            cv2.COLOR_BGR2RGB,
        )

        image = (
            rgb.astype(np.float32) / 255.0
        )

        image = (
            image - self.mean
        ) / self.std

        tensor = torch.from_numpy(
            image
        )

        tensor = tensor.permute(
            2,
            0,
            1,
        ).contiguous()

        tensor = tensor.unsqueeze(0)

        tensor = tensor.to(
            device=self.device,
            dtype=torch.float32,
            non_blocking=self.device.startswith("cuda"),
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

        if self.model is None or not self.loaded:
            raise RuntimeError(
                "Modelo UFLD não está carregado."
            )

        if not isinstance(tensor, torch.Tensor):
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

        output = self.model(tensor)

        return self._extract_output_tensor(
            output
        )

    # ========================================================================
    # OUTPUT
    # ========================================================================

    @staticmethod
    def _extract_output_tensor(
        output: Any,
    ) -> torch.Tensor:

        if isinstance(output, torch.Tensor):
            return output

        if isinstance(output, (tuple, list)):

            for item in output:
                if isinstance(item, torch.Tensor):
                    return item

        if isinstance(output, dict):

            for key in (
                "cls",
                "classification",
                "output",
                "out",
            ):

                value = output.get(key)

                if isinstance(value, torch.Tensor):
                    return value

            for value in output.values():

                if isinstance(value, torch.Tensor):
                    return value

        raise RuntimeError(
            "Não foi possível extrair o tensor principal "
            "da saída UFLD."
        )

    # ========================================================================
    # SOFTMAX
    # ========================================================================

    @staticmethod
    def _softmax_grid(
        logits: np.ndarray,
    ) -> np.ndarray:
        """
        Softmax SOMENTE nas 200 posições válidas.

        A classe 200 é NO-LANE e não participa do softmax.

        Entrada:

            [200, 18, 4]

        Saída:

            [200, 18, 4]
        """

        if logits.ndim != 3:
            raise ValueError(
                "logits deve possuir 3 dimensões."
            )

        maximum = np.max(
            logits,
            axis=0,
            keepdims=True,
        )

        shifted = logits - maximum

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
            axis=0,
            keepdims=True,
        )

        denominator = np.maximum(
            denominator,
            1e-12,
        )

        return exponential / denominator

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
                model_output_shape=self.last_output_shape,
                valid=False,
                error=self.last_error,
            )

            self.last_result = result

            return result

    def _decode_internal(
        self,
        output: torch.Tensor,
    ) -> LaneDetectionResult:

        if not isinstance(output, torch.Tensor):
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

        if output_np.ndim != 4:
            raise ValueError(
                "Saída UFLD inválida. "
                f"Esperado [B,G,R,L], recebido {output_np.shape}."
            )

        batch, grid, rows, lanes = output_np.shape

        if batch != 1:
            raise ValueError(
                f"Batch deve ser 1. Recebido {batch}."
            )

        if grid != self.griding_num:
            raise ValueError(
                f"Grid incompatível: {grid}."
            )

        if rows != self.cls_num_per_lane:
            raise ValueError(
                f"Anchors incompatíveis: {rows}."
            )

        if lanes != self.num_lanes:
            raise ValueError(
                f"Lanes incompatíveis: {lanes}."
            )

        logits = output_np[0]

        if not np.all(np.isfinite(logits)):
            raise ValueError(
                "A saída contém NaN ou Inf."
            )

        # ------------------------------------------------------------------
        # IMPORTANTE:
        #
        # O decoder original do UFLD inverte a ordem dos row anchors.
        #
        # [G, R, L] -> [G, R invertido, L]
        #
        # Isso é necessário para manter os anchors alinhados com a saída.
        # ------------------------------------------------------------------

        logits = logits[:, ::-1, :]

        # ------------------------------------------------------------------
        # Softmax SOMENTE em 0..199.
        #
        # logits[200] permanece separado.
        # ------------------------------------------------------------------

        valid_logits = logits[
            :UFLD_VALID_GRID_NUM,
            :,
            :,
        ]

        probabilities = self._softmax_grid(
            valid_logits
        )

        # ------------------------------------------------------------------
        # NO-LANE:
        #
        # A classe 200 compete com as posições durante a classificação,
        # mas NÃO entra no softmax da expectativa.
        # ------------------------------------------------------------------

        no_lane_logits = logits[
            UFLD_VALID_GRID_NUM,
            :,
            :,
        ]

        decoded_lanes: List[List[LanePoint]] = []
        lane_confidences: List[float] = []

        for lane_index in range(
            self.num_lanes
        ):

            lane = self._decode_single_lane(
                probabilities,
                no_lane_logits[:, lane_index],
                lane_index,
            )

            lane = self._validate_lane_continuity(
                lane
            )

            valid_count = sum(
                1
                for point in lane
                if point.valid
            )

            confidence = self._lane_confidence(
                lane
            )

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

            decoded_lanes.append(lane)
            lane_confidences.append(confidence)

        ordered_lanes, ordered_confidences = (
            self._order_lanes(
                decoded_lanes,
                lane_confidences,
            )
        )

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

            left_confidence = ordered_confidences[
                current_lane_index
            ]

            right_confidence = ordered_confidences[
                current_lane_index + 1
            ]

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

            additional_lanes.append(lane)

        valid = (
            current_lane_index is not None
            and bool(left_lane)
            and bool(right_lane)
            and left_confidence >= self.confidence_threshold
            and right_confidence >= self.confidence_threshold
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
            num_lanes_detected=len(ordered_lanes),
            input_width=self.input_width,
            input_height=self.input_height,
            model_output_shape=self.last_output_shape,
        )

    # ========================================================================
    # SINGLE LANE DECODER
    # ========================================================================

    def _decode_single_lane(
        self,
        probabilities: np.ndarray,
        no_lane_logits: np.ndarray,
        lane_index: int,
    ) -> List[LanePoint]:
        """
        Decodifica uma lane.

        probabilities:

            [200, 18, 4]

        no_lane_logits:

            [18]

        A classe 200 NÃO participa da expectativa.

        Para decidir NO-LANE, comparamos o logit da classe 200
        contra o maior logit entre as 200 posições.
        """

        points: List[LanePoint] = []

        for row_index in range(
            self.cls_num_per_lane
        ):

            row_probabilities = probabilities[
                :,
                row_index,
                lane_index,
            ]

            valid_logits = np.log(
                np.maximum(
                    row_probabilities,
                    1e-12,
                )
            )

            no_lane_logit = float(
                no_lane_logits[row_index]
            )

            best_valid_index = int(
                np.argmax(valid_logits)
            )

            best_valid_logit = float(
                valid_logits[best_valid_index]
            )

            y = float(
                self.row_anchors[row_index]
            )

            # --------------------------------------------------------------
            # Decisão NO-LANE.
            #
            # Não usa simplesmente a probabilidade do softmax das 200
            # classes, porque NO-LANE está fora desse softmax.
            #
            # O logit NO-LANE precisa superar a melhor posição válida
            # por uma margem correspondente ao threshold configurado.
            # --------------------------------------------------------------

            no_lane_probability = float(
                1.0
                / (
                    1.0
                    + np.exp(
                        np.clip(
                            best_valid_logit
                            - no_lane_logit,
                            -80.0,
                            80.0,
                        )
                    )
                )
            )

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

            # --------------------------------------------------------------
            # Expectativa probabilística.
            # --------------------------------------------------------------

            probability_sum = float(
                np.sum(row_probabilities)
            )

            if probability_sum <= 1e-8:

                points.append(
                    LanePoint(
                        x=0.0,
                        y=y,
                        confidence=0.0,
                        valid=False,
                    )
                )

                continue

            normalized = (
                row_probabilities
                / probability_sum
            )

            grid_position = float(
                np.sum(
                    normalized
                    * self._grid_indices
                )
            )

            # 0..199 -> 0..799
            x = (
                grid_position
                / float(UFLD_VALID_GRID_NUM - 1)
                * float(self.input_width - 1)
            )

            # --------------------------------------------------------------
            # Confiança.
            #
            # Usamos:
            #
            #   peak = maior probabilidade da posição
            #
            #   no_lane_penalty = confiança de que não é NO-LANE
            #
            # Isso permite que probabilidades vizinhas contribuam
            # para uma detecção estável.
            # --------------------------------------------------------------

            peak_probability = float(
                np.max(row_probabilities)
            )

            valid_confidence = float(
                1.0 - no_lane_probability
            )

            confidence = (
                0.65 * peak_probability
                + 0.35 * valid_confidence
            )

            valid = (
                np.isfinite(x)
                and np.isfinite(confidence)
                and 0.0 <= x < self.input_width
                and confidence > 0.0
            )

            points.append(
                LanePoint(
                    x=float(x) if valid else 0.0,
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
    # CONTINUITY
    # ========================================================================

    def _validate_lane_continuity(
        self,
        points: List[LanePoint],
    ) -> List[LanePoint]:

        if not points:
            return points

        result = [
            LanePoint(
                x=p.x,
                y=p.y,
                confidence=p.confidence,
                valid=p.valid,
            )
            for p in points
        ]

        previous_index: Optional[int] = None
        previous_x: Optional[float] = None

        for index, point in enumerate(result):

            if not point.valid:
                continue

            if previous_index is None:

                previous_index = index
                previous_x = point.x

                continue

            row_gap = index - previous_index

            dx = abs(
                point.x - float(previous_x)
            )

            allowed_jump = (
                self.max_x_jump
                * max(
                    1.0,
                    float(row_gap),
                )
            )

            if (
                row_gap > self.max_gap_rows
                and dx > self.max_x_jump
            ):

                point.valid = False
                point.x = 0.0
                point.confidence = 0.0

                continue

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
            if point.valid
            and np.isfinite(point.confidence)
        ]

        if not values:
            return 0.0

        return float(
            np.mean(values)
        )

    # ========================================================================
    # ORDERING
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

        for index, lane in enumerate(lanes):

            xs = [
                point.x
                for point in lane
                if point.valid
                and np.isfinite(point.x)
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

        candidates.sort(
            key=lambda item: item[0]
        )

        return (
            [item[2] for item in candidates],
            [item[3] for item in candidates],
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

        vehicle_x = (
            float(self.input_width) / 2.0
        )

        centers: List[Optional[float]] = []

        for lane in lanes:

            xs = [
                point.x
                for point in lane
                if point.valid
            ]

            if len(xs) < self.min_valid_points:

                centers.append(None)

            else:

                centers.append(
                    float(
                        np.median(xs)
                    )
                )

        for index in range(
            len(centers) - 1
        ):

            left = centers[index]
            right = centers[index + 1]

            if left is None or right is None:
                continue

            if left >= right:
                continue

            width = right - left

            if width < self.min_lane_width:
                continue

            if width > self.max_lane_width:
                continue

            if left <= vehicle_x <= right:
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

            tensor = self.preprocess(frame)

            output = self.infer(tensor)

            return self.decode(output)

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
                model_output_shape=self.last_output_shape,
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
                    or "Falha ao carregar modelo."
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
            _ = self.model(dummy)

        if self.device.startswith("cuda"):
            torch.cuda.synchronize()

    # ========================================================================
    # CONFIGURATION
    # ========================================================================

    def get_config(self) -> Dict[str, Any]:

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
            "max_x_jump": (
                self.max_x_jump
            ),
            "max_gap_rows": (
                self.max_gap_rows
            ),
            "min_lane_width": (
                self.min_lane_width
            ),
            "max_lane_width": (
                self.max_lane_width
            ),
        }

    # ========================================================================
    # MODEL INFORMATION
    # ========================================================================

    def get_model_info(self) -> Dict[str, Any]:

        parameter_count = 0

        if self.model is not None:

            parameter_count = sum(
                parameter.numel()
                for parameter in self.model.parameters()
            )

        return {
            "model_path": self.model_path,
            "model_exists": self.model_exists(),
            "loaded": self.loaded,
            "device": self.device,
            "device_name": self.get_device_name(),
            "backbone": self.backbone,
            "cls_dim": self.cls_dim,
            "input_width": self.input_width,
            "input_height": self.input_height,
            "use_aux": self.use_aux,
            "griding_num": self.griding_num,
            "row_anchors": self.cls_num_per_lane,
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
            "parameter_count": parameter_count,
            "last_output_shape": self.last_output_shape,
            "last_error": self.last_error,
        }

    # ========================================================================
    # LAST OUTPUT
    # ========================================================================

    def get_last_output_shape(
        self,
    ) -> Tuple[int, ...]:

        return self.last_output_shape

    # ========================================================================
    # LAST ERROR
    # ========================================================================

    def get_last_error(
        self,
    ) -> Optional[str]:

        return self.last_error

    # ========================================================================
    # STATE
    # ========================================================================

    def is_ready(self) -> bool:

        return (
            self.loaded
            and self.model is not None
        )

    # ========================================================================
    # RELEASE
    # ========================================================================

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
    """
    Cria o detector UFLD padrão.

    Checkpoint:

        Ultra-Fast-Lane-Detection/culane/_18.pth

    Configuração:

        backbone = 18
        cls_dim = (201, 18, 4)
    """

    if model_path is None:

        candidates = [
            UFLD_ROOT / "culane_18.pth",
            UFLD_ROOT / "culane" / "_18.pth",
            UFLD_ROOT / "culane" / "18.pth",
            UFLD_ROOT / "culane" / "culane_18.pth",
            PROJECT_ROOT / "culane_18.pth",
            PROJECT_ROOT / "culane" / "_18.pth",
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
                "Checkpoint UFLD não encontrado.\n"
                "Arquivos procurados:\n"
                f"{searched}"
            )

        model_path = str(model_file)

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
# PUBLIC API
# ============================================================================

__all__ = [
    "LanePoint",
    "LaneDetectionResult",
    "UFLDLaneDetector",
    "CULANE_ROW_ANCHORS",
    "create_default_detector",
]

