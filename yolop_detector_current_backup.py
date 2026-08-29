"""
vision/yolop_detector.py

Detector de lanes baseado em YOLOP + ONNX Runtime.

Responsabilidade:

    frame
      ↓
    preprocessamento
      ↓
    YOLOP ONNX
      ↓
    lane_line_seg
      ↓
    probabilidade de lane
      ↓
    máscara
      ↓
    segmentos por linha
      ↓
    associação espacial
      ↓
    múltiplas lanes
      ↓
    LaneDetectionResult

Este módulo NÃO executa:

    - tracking temporal;
    - fitting polinomial;
    - geometria da faixa;
    - projeção;
    - identificação da faixa atual;
    - estado ADAS;
    - controle do veículo.

A saída é compatível com o LaneTracker atual:

    LaneDetectionResult.lanes
        -> List[List[LanePoint]]

O detector tenta preservar TODAS as linhas espacialmente
consistentes que o YOLOP conseguir segmentar.

Compatibilidade:
    vision.lane_types.LanePoint
    vision.lane_tracker.LaneTracker
    vision.lane_geometry.LaneGeometry
    vision.lane_model
    vision.lane_projection
    vision.lane_assignment
    vision.adas_state
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from .lane_types import LanePoint


logger = logging.getLogger(__name__)

try:
    ort.preload_dlls()
except Exception as exc:
    logger.debug(
        "[YOLOP] Não foi possível pré-carregar DLLs CUDA: %s",
        exc,
    )

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "weights"
    / "yolop-640-640.onnx"
)

YOLOP_INPUT_WIDTH = 640
YOLOP_INPUT_HEIGHT = 640

DEFAULT_LANE_THRESHOLD = 0.50
DEFAULT_MIN_POINTS_PER_LANE = 4

DEFAULT_ROW_STEP = 6
DEFAULT_MIN_PIXELS_PER_SEGMENT = 1

DEFAULT_MAX_LANES = 8

DEFAULT_MAX_TRACKING_JUMP = 55.0
DEFAULT_MAX_TRACKING_JUMP_PER_ROW = 0.14

DEFAULT_MIN_LANE_SPAN = 25.0
DEFAULT_MIN_LANE_VERTICAL_SPAN = 35.0

DEFAULT_MORPH_KERNEL = 3

DEFAULT_MIN_COMPONENT_AREA = 3


# =============================================================================
# RESULTADO
# =============================================================================

@dataclass
class LaneDetectionResult:
    """
    Resultado da percepção de lanes.

    lanes:
        Todas as lanes detectadas e validadas.

    left_lane / right_lane:
        Lanes mais próximas do centro da imagem que ficam,
        respectivamente, à esquerda e à direita.

        Essas propriedades existem por compatibilidade.
        Elas NÃO representam necessariamente a faixa atual.

    additional_lanes:
        Todas as outras lanes detectadas.

    current_lane_index:
        Sempre None neste estágio.

        A identificação da faixa atual pertence ao
        lane_assignment.py.
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

    input_width: int = YOLOP_INPUT_WIDTH

    input_height: int = YOLOP_INPUT_HEIGHT

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


# =============================================================================
# CANDIDATO INTERNO
# =============================================================================

@dataclass
class _RowSegment:
    """
    Segmento de pixels de uma determinada linha da máscara.
    """

    y: int
    x_min: float
    x_max: float
    x_center: float
    confidence: float
    pixel_count: int

    @property
    def width(self) -> float:
        return self.x_max - self.x_min


@dataclass
class _LaneTrack:
    """
    Track espacial temporário utilizado SOMENTE durante a
    extração de um único frame.

    Isto NÃO substitui LaneTracker.

    O objetivo aqui é reconstruir uma linha contínua
    dentro da própria máscara YOLOP.
    """

    points: List[Tuple[int, float, float]] = field(
        default_factory=list
    )

    last_x: float = 0.0

    last_y: int = 0

    previous_x: Optional[float] = None

    previous_y: Optional[int] = None

    missed_rows: int = 0

    confidence_sum: float = 0.0

    pixel_sum: int = 0

    def predicted_x(self) -> float:
        """
        Predição linear simples baseada nos dois últimos pontos.

        Isso serve somente para associar segmentos dentro do
        mesmo frame. O tracking temporal verdadeiro permanece
        no LaneTracker.
        """

        if (
            self.previous_x is None
            or self.previous_y is None
        ):
            return self.last_x

        dy = self.last_y - self.previous_y

        if dy == 0:
            return self.last_x

        dx = self.last_x - self.previous_x

        # O processamento ocorre de baixo para cima.
        target_dy = 1.0

        return (
            self.last_x
            + dx / float(dy) * target_dy
        )


# =============================================================================
# DETECTOR
# =============================================================================

class YOLOPLaneDetector:
    """
    Detector YOLOP industrializado para percepção de múltiplas lanes.

    Características:

        - ONNX Runtime;
        - CUDA quando disponível;
        - fallback automático para CPU;
        - seleção robusta da saída lane_line_seg;
        - suporte a logits ou probabilidades;
        - suporte a saída [1,2,H,W], [2,H,W] e [1,1,H,W];
        - extração de múltiplas linhas;
        - associação espacial dentro do frame;
        - remoção de ruído;
        - preservação de linhas adicionais;
        - coordenadas convertidas para o frame original;
        - fail-safe;
        - diagnóstico detalhado.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        input_width: int = YOLOP_INPUT_WIDTH,
        input_height: int = YOLOP_INPUT_HEIGHT,
        lane_threshold: float = DEFAULT_LANE_THRESHOLD,
        min_points_per_lane: int = DEFAULT_MIN_POINTS_PER_LANE,
        row_step: int = DEFAULT_ROW_STEP,
        min_lane_pixels_per_row: int = DEFAULT_MIN_PIXELS_PER_SEGMENT,
        max_lanes: int = DEFAULT_MAX_LANES,
        max_tracking_jump: Optional[float] = None,
        min_lane_span: float = DEFAULT_MIN_LANE_SPAN,
        min_lane_vertical_span: float = DEFAULT_MIN_LANE_VERTICAL_SPAN,
        morph_kernel: int = DEFAULT_MORPH_KERNEL,
        min_component_area: int = DEFAULT_MIN_COMPONENT_AREA,
        providers: Optional[List[str]] = None,
    ) -> None:

        self.model_path = Path(model_path)

        self.input_width = max(
            1,
            int(input_width),
        )

        self.input_height = max(
            1,
            int(input_height),
        )

        self.lane_threshold = float(
            np.clip(
                lane_threshold,
                0.0,
                1.0,
            )
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
                * DEFAULT_MAX_TRACKING_JUMP_PER_ROW,
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

        self.session: Optional[
            ort.InferenceSession
        ] = None

        self.input_name: Optional[str] = None

        self.input_shape: Optional[
            Tuple[Any, ...]
        ] = None

        self.output_names: List[str] = []

        self.output_shapes: List[
            Tuple[int, ...]
        ] = []

        self.loaded = False

        self.last_output_shape: Tuple[
            int, ...
        ] = tuple()

        self.last_error: Optional[str] = None

        self.last_result: Optional[
            LaneDetectionResult
        ] = None

        self.last_lane_probability: Optional[
            np.ndarray
        ] = None

        self.last_lane_mask: Optional[
            np.ndarray
        ] = None

        self.last_diagnostics: Dict[
            str,
            Any,
        ] = {}

    # =========================================================================
    # MODEL
    # =========================================================================

    def model_exists(self) -> bool:
        return self.model_path.is_file()

    def _provider_candidates(
        self,
    ) -> List[List[str]]:
        """
        Retorna as combinações de providers em ordem de preferência.

        CUDA é sempre preferido quando explicitamente solicitado
        ou quando disponível.

        CPU permanece como fallback final.
        """

        if self.providers is not None:
            return [
                list(self.providers)
            ]

        available = set(
            ort.get_available_providers()
        )

        candidates: List[
            List[str]
        ] = []

        if (
            "TensorrtExecutionProvider"
            in available
        ):
            candidates.append(
                [
                    "TensorrtExecutionProvider",
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider",
                ]
            )

        if (
            "CUDAExecutionProvider"
            in available
        ):
            candidates.append(
                [
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider",
                ]
            )

        if (
            "CPUExecutionProvider"
            in available
        ):
            candidates.append(
                [
                    "CPUExecutionProvider"
                ]
            )

        return candidates

    def load_model(self) -> bool:
        """
        Carrega o modelo ONNX.

        Estratégia:

            TensorRT/CUDA
                  ↓ falha
            CUDA
                  ↓ falha
            CPU
        """

        if (
            self.loaded
            and self.session is not None
        ):
            return True

        self.last_error = None

        if not self.model_exists():

            self.last_error = (
                "Modelo YOLOP não encontrado: "
                f"{self.model_path}"
            )

            logger.error(
                "[YOLOP] %s",
                self.last_error,
            )

            return False

        candidates = (
            self._provider_candidates()
        )

        if not candidates:

            self.last_error = (
                "Nenhum ExecutionProvider "
                "disponível no ONNX Runtime."
            )

            return False

        last_exception: Optional[
            Exception
        ] = None

        for providers in candidates:

            try:

                logger.info(
                    "[YOLOP] Tentando providers: %s",
                    providers,
                )

                session = (
                    ort.InferenceSession(
                        str(self.model_path),
                        providers=providers,
                    )
                )

                inputs = (
                    session.get_inputs()
                )

                if not inputs:
                    raise RuntimeError(
                        "O modelo YOLOP não possui entradas."
                    )

                self.session = session

                self.input_name = (
                    inputs[0].name
                )

                self.input_shape = tuple(
                    inputs[0].shape
                )

                self.output_names = [
                    output.name
                    for output
                    in session.get_outputs()
                ]

                self.output_shapes = [
                    tuple(
                        output.shape
                    )
                    for output
                    in session.get_outputs()
                ]

                self.loaded = True

                effective = (
                    session.get_providers()
                )

                logger.info(
                    "[YOLOP] Modelo carregado: %s",
                    self.model_path,
                )

                logger.info(
                    "[YOLOP] Providers efetivos: %s",
                    effective,
                )

                logger.info(
                    "[YOLOP] Input: %s %s",
                    self.input_name,
                    self.input_shape,
                )

                logger.info(
                    "[YOLOP] Outputs: %s",
                    list(
                        zip(
                            self.output_names,
                            self.output_shapes,
                        )
                    ),
                )

                return True

            except Exception as exc:

                last_exception = exc

                logger.warning(
                    "[YOLOP] Provider %s falhou: %s",
                    providers,
                    exc,
                )

                self.session = None
                self.input_name = None
                self.input_shape = None
                self.output_names = []
                self.output_shapes = []
                self.loaded = False

        self.last_error = (
            f"{type(last_exception).__name__}: "
            f"{last_exception}"
            if last_exception is not None
            else "Falha desconhecida ao carregar YOLOP."
        )

        logger.error(
            "[YOLOP] Modelo não pôde ser carregado: %s",
            self.last_error,
        )

        return False

    # =========================================================================
    # DEVICE
    # =========================================================================

    def get_device_name(self) -> str:
        if self.session is None:
            return "NOT_LOADED"

        providers = (
            self.session.get_providers()
        )

        if (
            "TensorrtExecutionProvider"
            in providers
        ):
            return "TENSORRT"

        if (
            "CUDAExecutionProvider"
            in providers
        ):
            return "CUDA"

        if (
            "CPUExecutionProvider"
            in providers
        ):
            return "CPU"

        if providers:
            return providers[0]

        return "UNKNOWN"

    # =========================================================================
    # IMAGEM
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
                "Frame está vazio."
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
                "Dimensões do frame são inválidas."
            )

        return (
            int(width),
            int(height),
        )

    @staticmethod
    def _load_image_unicode(
        path: Path,
    ) -> np.ndarray:

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
                "Não foi possível carregar imagem: "
                f"{path}"
            )

        return image

    @staticmethod
    def _save_image_unicode(
        path: Path,
        image: np.ndarray,
    ) -> None:

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        extension = (
            path.suffix
            if path.suffix
            else ".png"
        )

        success, encoded = cv2.imencode(
            extension,
            image,
        )

        if not success:
            raise RuntimeError(
                "Não foi possível codificar imagem: "
                f"{path}"
            )

        encoded.tofile(
            str(path)
        )

    # =========================================================================
    # PREPROCESSAMENTO
    # =========================================================================

    def preprocess(
        self,
        frame: np.ndarray,
    ) -> np.ndarray:
        """
        BGR HxWx3
            ↓
        RGB
            ↓
        [0,1]
            ↓
        NCHW float32
        """

        self._validate_frame(frame)

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

        return np.ascontiguousarray(
            tensor,
            dtype=np.float32,
        )

    # =========================================================================
    # SELEÇÃO DA SAÍDA YOLOP
    # =========================================================================

    @staticmethod
    def _shape_is_lane_candidate(
        shape: Sequence[Any],
    ) -> bool:
        """
        Identifica formatos compatíveis com lane segmentation.
        """

        if len(shape) == 4:

            channels = shape[1]

            return channels in (
                1,
                2,
                None,
            )

        if len(shape) == 3:

            channels = shape[0]

            return channels in (
                1,
                2,
                None,
            )

        return False

    def _select_lane_output(
        self,
        outputs: Sequence[np.ndarray],
    ) -> np.ndarray:
        """
        Seleciona lane_line_seg de forma robusta.

        Prioridade:

            1. nome contendo "lane";
            2. saída 4D com 2 canais;
            3. saída 3D com 2 canais;
            4. saída 4D/3D compatível;
            5. índice legado 2.

        Isso evita depender exclusivamente de outputs[2].
        """

        if not outputs:
            raise RuntimeError(
                "YOLOP não retornou nenhuma saída."
            )

        candidates: List[
            Tuple[int, int]
        ] = []

        # ---------------------------------------------------------------------
        # 1. Nome da saída
        # ---------------------------------------------------------------------

        for index, name in enumerate(
            self.output_names
        ):

            lowered = name.lower()

            if (
                "lane" in lowered
                or "line" in lowered
            ):

                if index < len(outputs):

                    score = 100

                    shape = np.asarray(
                        outputs[index]
                    ).shape

                    if (
                        len(shape) >= 3
                        and (
                            2 in shape[:2]
                        )
                    ):
                        score += 20

                    candidates.append(
                        (
                            score,
                            index,
                        )
                    )

        # ---------------------------------------------------------------------
        # 2. Formato conhecido
        # ---------------------------------------------------------------------

        for index, output in enumerate(
            outputs
        ):

            array = np.asarray(
                output
            )

            shape = array.shape

            score = 0

            if len(shape) == 4:

                if shape[0] == 1:
                    score += 10

                if shape[1] == 2:
                    score += 50

                if (
                    shape[-2] >= 32
                    and shape[-1] >= 32
                ):
                    score += 20

            elif len(shape) == 3:

                if shape[0] == 2:
                    score += 45

                if (
                    shape[-2] >= 32
                    and shape[-1] >= 32
                ):
                    score += 20

            if (
                self._shape_is_lane_candidate(
                    shape
                )
            ):
                candidates.append(
                    (
                        score,
                        index,
                    )
                )

        # ---------------------------------------------------------------------
        # 3. Legado YOLOP: outputs[2]
        # ---------------------------------------------------------------------

        if len(outputs) >= 3:

            legacy = np.asarray(
                outputs[2]
            )

            if (
                self._shape_is_lane_candidate(
                    legacy.shape
                )
            ):

                candidates.append(
                    (
                        60,
                        2,
                    )
                )

        if not candidates:

            raise RuntimeError(
                "Não foi possível identificar "
                "lane_line_seg entre as saídas YOLOP. "
                f"Shapes: "
                f"{[np.asarray(o).shape for o in outputs]}"
            )

        # Maior score; em empate, último índice.
        candidates.sort(
            key=lambda item: (
                item[0],
                item[1],
            )
        )

        selected_index = (
            candidates[-1][1]
        )

        selected = np.asarray(
            outputs[selected_index]
        )

        self.last_output_shape = tuple(
            int(value)
            for value in selected.shape
        )

        return selected

    # =========================================================================
    # INFERÊNCIA
    # =========================================================================

    def infer(
        self,
        frame: np.ndarray,
    ) -> np.ndarray:

        if not self.loaded:

            if not self.load_model():
                raise RuntimeError(
                    self.last_error
                    or "YOLOP não carregado."
                )

        if self.session is None:
            raise RuntimeError(
                "Sessão ONNX inexistente."
            )

        if self.input_name is None:
            raise RuntimeError(
                "Entrada ONNX não definida."
            )

        tensor = self.preprocess(
            frame
        )

        outputs = self.session.run(
            None,
            {
                self.input_name: tensor
            },
        )

        self.output_names = [
            output.name
            for output
            in self.session.get_outputs()
        ]

        self.output_shapes = [
            tuple(output.shape)
            for output
            in self.session.get_outputs()
        ]

        lane_line_seg = (
            self._select_lane_output(
                outputs
            )
        )

        self.last_diagnostics[
            "output_names"
        ] = list(
            self.output_names
        )

        self.last_diagnostics[
            "output_shapes"
        ] = list(
            self.output_shapes
        )

        return lane_line_seg

    # =========================================================================
    # PROBABILIDADE
    # =========================================================================

    @staticmethod
    def _stable_sigmoid(
        values: np.ndarray,
    ) -> np.ndarray:

        values = np.clip(
            values,
            -60.0,
            60.0,
        )

        return (
            1.0
            / (
                1.0
                + np.exp(-values)
            )
        )

    @staticmethod
    def _stable_softmax(
        values: np.ndarray,
        axis: int = 0,
    ) -> np.ndarray:

        maximum = np.max(
            values,
            axis=axis,
            keepdims=True,
        )

        exponentials = np.exp(
            values - maximum
        )

        denominator = np.sum(
            exponentials,
            axis=axis,
            keepdims=True,
        )

        denominator = np.maximum(
            denominator,
            np.finfo(
                np.float32
            ).eps,
        )

        return (
            exponentials
            / denominator
        )

    def _lane_probability(
        self,
        lane_line_seg: np.ndarray,
    ) -> np.ndarray:
        """
        Converte diferentes representações de segmentação
        para uma matriz HxW de probabilidade [0,1].

        Suporta:

            [1,2,H,W]
            [2,H,W]
            [1,1,H,W]
            [1,H,W]
            [H,W]
        """

        output = np.asarray(
            lane_line_seg,
            dtype=np.float32,
        )

        if output.size == 0:
            raise ValueError(
                "lane_line_seg está vazio."
            )

        # ---------------------------------------------------------------------
        # Remover batch
        # ---------------------------------------------------------------------

        if output.ndim == 4:

            if output.shape[0] != 1:
                raise ValueError(
                    "Somente batch=1 é suportado para "
                    f"lane_line_seg: {output.shape}"
                )

            output = output[0]

        # ---------------------------------------------------------------------
        # [C,H,W]
        # ---------------------------------------------------------------------

        if output.ndim == 3:

            channels = output.shape[0]

            if channels == 2:

                # Pode ser logits ou probabilidades.
                minimum = float(
                    np.nanmin(output)
                )

                maximum = float(
                    np.nanmax(output)
                )

                if (
                    minimum >= 0.0
                    and maximum <= 1.0
                ):

                    channel_sum = (
                        output[0]
                        + output[1]
                    )

                    # Caso os dois canais já sejam
                    # probabilidades.
                    if np.nanmean(
                        channel_sum
                    ) <= 1.05:

                        probability = output[1]

                    else:

                        probability = (
                            self._stable_softmax(
                                output,
                                axis=0,
                            )[1]
                        )

                else:

                    probability = (
                        self._stable_softmax(
                            output,
                            axis=0,
                        )[1]
                    )

                return np.asarray(
                    np.clip(
                        probability,
                        0.0,
                        1.0,
                    ),
                    dtype=np.float32,
                )

            if channels == 1:

                values = output[0]

                minimum = float(
                    np.nanmin(values)
                )

                maximum = float(
                    np.nanmax(values)
                )

                if (
                    minimum >= 0.0
                    and maximum <= 1.0
                ):

                    probability = values

                else:

                    probability = (
                        self._stable_sigmoid(
                            values
                        )
                    )

                return np.asarray(
                    np.clip(
                        probability,
                        0.0,
                        1.0,
                    ),
                    dtype=np.float32,
                )

            raise ValueError(
                "Formato de lane_line_seg "
                f"não suportado: {output.shape}"
            )

        # ---------------------------------------------------------------------
        # [H,W]
        # ---------------------------------------------------------------------

        if output.ndim == 2:

            minimum = float(
                np.nanmin(output)
            )

            maximum = float(
                np.nanmax(output)
            )

            if (
                minimum >= 0.0
                and maximum <= 1.0
            ):

                probability = output

            else:

                probability = (
                    self._stable_sigmoid(
                        output
                    )
                )

            return np.asarray(
                np.clip(
                    probability,
                    0.0,
                    1.0,
                ),
                dtype=np.float32,
            )

        raise ValueError(
            "Formato de lane_line_seg "
            f"não suportado: {output.shape}"
        )

    # =========================================================================
    # MÁSCARA
    # =========================================================================

    def create_lane_mask(
        self,
        lane_line_seg: np.ndarray,
    ) -> np.ndarray:

        probability = (
            self._lane_probability(
                lane_line_seg
            )
        )

        mask = (
            probability
            >= self.lane_threshold
        ).astype(
            np.uint8
        )

        # ---------------------------------------------------------------------
        # Limpeza mínima.
        #
        # Não utilizamos operações agressivas porque isso poderia
        # apagar linhas estreitas legítimas.
        # ---------------------------------------------------------------------

        if self.morph_kernel > 1:

            kernel = np.ones(
                (
                    self.morph_kernel,
                    self.morph_kernel,
                ),
                dtype=np.uint8,
            )

            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_OPEN,
                kernel,
                iterations=1,
            )

            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_CLOSE,
                kernel,
                iterations=1,
            )

        # ---------------------------------------------------------------------
        # Remover somente componentes muito pequenos.
        # ---------------------------------------------------------------------

        if self.min_component_area > 1:

            num_labels, labels, stats, _ = (
                cv2.connectedComponentsWithStats(
                    mask,
                    connectivity=8,
                )
            )

            cleaned = np.zeros_like(
                mask
            )

            for label in range(
                1,
                num_labels,
            ):

                area = int(
                    stats[
                        label,
                        cv2.CC_STAT_AREA,
                    ]
                )

                if (
                    area
                    >= self.min_component_area
                ):

                    cleaned[
                        labels == label
                    ] = 1

            mask = cleaned

        return mask.astype(
            np.uint8,
            copy=False,
        )

    # =========================================================================
    # SEGMENTOS POR LINHA
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

        split_indices = (
            np.where(
                gaps > 2
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
        mask: np.ndarray,
        probability: np.ndarray,
    ) -> List[
        List[_RowSegment]
    ]:
        """
        Extrai TODOS os segmentos ativos de cada linha.

        Não assume que existem somente duas lanes.
        """

        height, width = (
            mask.shape
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
                mask[y] > 0
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
                    np.mean(segment)
                )

                confidence = float(
                    np.mean(
                        probability[
                            y,
                            segment,
                        ]
                    )
                )

                if not math.isfinite(
                    confidence
                ):
                    continue

                row_segments.append(
                    _RowSegment(
                        y=int(y),
                        x_min=x_min,
                        x_max=x_max,
                        x_center=x_center,
                        confidence=float(
                            np.clip(
                                confidence,
                                0.0,
                                1.0,
                            )
                        ),
                        pixel_count=int(
                            segment.size
                        ),
                    )
                )

            if row_segments:

                row_segments.sort(
                    key=lambda segment:
                    segment.x_center
                )

                rows.append(
                    row_segments
                )

        return rows

    # =========================================================================
    # TRACKING ESPACIAL DO FRAME
    # =========================================================================

    def _segment_distance(
        self,
        track: _LaneTrack,
        segment: _RowSegment,
    ) -> float:
        """
        Distância espacial entre segmento e linha parcial.

        O objetivo é impedir que uma linha seja trocada por
        outra durante uma curva.
        """

        predicted = (
            track.predicted_x()
        )

        return abs(
            segment.x_center
            - predicted
        )

    def _associate_segments(
        self,
        row_segments: List[
            List[_RowSegment]
        ],
    ) -> List[_LaneTrack]:
        """
        Constrói múltiplas lanes dentro do frame.

        Processo:

            bottom → top

        Cada segmento pode pertencer a apenas uma lane.

        Novos tracks são criados quando existe evidência
        espacial suficiente.
        """

        tracks: List[
            _LaneTrack
        ] = []

        for segments in row_segments:

            if not segments:
                continue

            used_track_indices: set[
                int
            ] = set()

            # -----------------------------------------------------------------
            # Primeiro, associa os segmentos aos tracks existentes.
            #
            # Ordenamos por distância para priorizar as associações
            # mais confiáveis.
            # -----------------------------------------------------------------

            possible_matches: List[
                Tuple[
                    float,
                    int,
                    int,
                ]
            ] = []

            for segment_index, segment in enumerate(
                segments
            ):

                for track_index, track in enumerate(
                    tracks
                ):

                    if track_index in used_track_indices:
                        continue

                    distance = (
                        self._segment_distance(
                            track,
                            segment,
                        )
                    )

                    adaptive_jump = (
                        self.max_tracking_jump
                    )

                    # Linhas muito largas ou com forte
                    # evidência podem tolerar um pouco mais.
                    adaptive_jump += min(
                        20.0,
                        segment.width,
                    )

                    if (
                        distance
                        <= adaptive_jump
                    ):

                        possible_matches.append(
                            (
                                distance,
                                segment_index,
                                track_index,
                            )
                        )

            possible_matches.sort(
                key=lambda item:
                item[0]
            )

            used_segments: set[
                int
            ] = set()

            # -----------------------------------------------------------------
            # Aplicar associações.
            # -----------------------------------------------------------------

            for (
                _distance,
                segment_index,
                track_index,
            ) in possible_matches:

                if (
                    segment_index
                    in used_segments
                ):
                    continue

                if (
                    track_index
                    in used_track_indices
                ):
                    continue

                segment = (
                    segments[
                        segment_index
                    ]
                )

                track = (
                    tracks[
                        track_index
                    ]
                )

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

                track.pixel_sum += (
                    segment.pixel_count
                )

                track.missed_rows = 0

                used_segments.add(
                    segment_index
                )

                used_track_indices.add(
                    track_index
                )

            # -----------------------------------------------------------------
            # Criar tracks para segmentos não associados.
            # -----------------------------------------------------------------

            for segment_index, segment in enumerate(
                segments
            ):

                if (
                    segment_index
                    in used_segments
                ):
                    continue

                if len(tracks) >= (
                    self.max_lanes
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
                        missed_rows=0,
                        confidence_sum=(
                            segment.confidence
                        ),
                        pixel_sum=(
                            segment.pixel_count
                        ),
                    )
                )

        return tracks

    # =========================================================================
    # VALIDAÇÃO DE LANE
    # =========================================================================

    def _validate_track(
        self,
        track: _LaneTrack,
    ) -> bool:
        """
        Decide se uma lane possui evidência suficiente.

        Não exige duas lanes.

        Uma linha individual válida continua sendo uma
        observação válida e poderá ser usada pelo tracker.
        """

        if len(
            track.points
        ) < self.min_points_per_lane:

            return False

        ys = np.asarray(
            [
                point[0]
                for point in track.points
            ],
            dtype=np.float32,
        )

        xs = np.asarray(
            [
                point[1]
                for point in track.points
            ],
            dtype=np.float32,
        )

        if (
            not np.all(
                np.isfinite(ys)
            )
            or not np.all(
                np.isfinite(xs)
            )
        ):
            return False

        vertical_span = (
            float(
                np.max(ys)
                - np.min(ys)
            )
        )

        horizontal_span = (
            float(
                np.max(xs)
                - np.min(xs)
            )
        )

        # Uma linha reta pode ter pouca variação em X.
        # Portanto, o critério principal é o span vertical.
        if (
            vertical_span
            < self.min_lane_vertical_span
        ):
            return False

        if (
            horizontal_span
            < 1.0
            and vertical_span
            < self.min_lane_vertical_span
        ):
            return False

        return True

    def _track_to_lane(
        self,
        track: _LaneTrack,
        mask_width: int,
        mask_height: int,
        frame_width: int,
        frame_height: int,
    ) -> List[LanePoint]:

        points: List[
            LanePoint
        ] = []

        for (
            y,
            x,
            confidence,
        ) in track.points:

            point = self._make_point(
                x=x,
                y=y,
                confidence=confidence,
                mask_width=mask_width,
                mask_height=mask_height,
                frame_width=frame_width,
                frame_height=frame_height,
            )

            points.append(
                point
            )

        points.sort(
            key=lambda point:
            point.y
        )

        return points

    # =========================================================================
    # PONTO
    # =========================================================================

    @staticmethod
    def _make_point(
        x: float,
        y: float,
        confidence: float,
        mask_width: int,
        mask_height: int,
        frame_width: int,
        frame_height: int,
    ) -> LanePoint:

        if (
            mask_width <= 0
            or mask_height <= 0
        ):
            raise ValueError(
                "Dimensões da máscara inválidas."
            )

        px = (
            float(x)
            * float(frame_width)
            / float(mask_width)
        )

        py = (
            float(y)
            * float(frame_height)
            / float(mask_height)
        )

        return LanePoint(
            x=float(px),
            y=float(py),
            confidence=float(
                np.clip(
                    confidence,
                    0.0,
                    1.0,
                )
            ),
            valid=True,
        )

    # =========================================================================
    # CONFIANÇA
    # =========================================================================

    @staticmethod
    def _lane_confidence(
        lane: Sequence[LanePoint],
    ) -> float:

        values: List[
            float
        ] = []

        for point in lane:

            if not point.valid:
                continue

            try:
                confidence = float(
                    point.confidence
                )
            except (
                TypeError,
                ValueError,
            ):
                continue

            if not math.isfinite(
                confidence
            ):
                continue

            values.append(
                float(
                    np.clip(
                        confidence,
                        0.0,
                        1.0,
                    )
                )
            )

        if not values:
            return 0.0

        return float(
            np.clip(
                np.mean(values),
                0.0,
                1.0,
            )
        )

    # =========================================================================
    # ORDENAÇÃO
    # =========================================================================

    @staticmethod
    def _lane_reference_x(
        lane: Sequence[LanePoint],
    ) -> float:
        """
        X representativo da região inferior da lane.

        Essa é a região mais importante para o LaneTracker.
        """

        valid = [
            point
            for point in lane
            if point.valid
            and math.isfinite(
                float(point.x)
            )
            and math.isfinite(
                float(point.y)
            )
        ]

        if not valid:
            return float("inf")

        valid.sort(
            key=lambda point:
            point.y,
            reverse=True,
        )

        sample = valid[
            : min(
                8,
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
    # CLASSIFICAÇÃO ESQUERDA / DIREITA
    # =========================================================================

    def _classify_primary_lanes(
        self,
        lanes: List[
            List[LanePoint]
        ],
        frame_width: int,
    ) -> Tuple[
        List[LanePoint],
        List[LanePoint],
        List[List[LanePoint]],
    ]:
        """
        Classifica somente por posição espacial.

        Não identifica a faixa atual.

        A faixa atual continua sendo responsabilidade do
        lane_assignment.py.
        """

        if not lanes:
            return (
                [],
                [],
                [],
            )

        center_x = (
            float(frame_width)
            / 2.0
        )

        ordered = sorted(
            lanes,
            key=self._lane_reference_x,
        )

        left_candidates: List[
            List[LanePoint]
        ] = []

        right_candidates: List[
            List[LanePoint]
        ] = []

        for lane in ordered:

            reference_x = (
                self._lane_reference_x(
                    lane
                )
            )

            if (
                reference_x
                < center_x
            ):

                left_candidates.append(
                    lane
                )

            else:

                right_candidates.append(
                    lane
                )

        left_lane: List[
            LanePoint
        ] = []

        right_lane: List[
            LanePoint
        ] = []

        if left_candidates:

            # A lane esquerda primária é a mais próxima
            # do centro da imagem.
            left_lane = max(
                left_candidates,
                key=self._lane_reference_x,
            )

        if right_candidates:

            # A lane direita primária é a mais próxima
            # do centro da imagem.
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
    # RESULTADO
    # =========================================================================

    def _build_result(
        self,
        lanes: List[
            List[LanePoint]
        ],
        frame_width: int,
        frame_height: int,
    ) -> LaneDetectionResult:

        valid_lanes: List[
            List[LanePoint]
        ] = []

        for lane in lanes:

            valid_points = [
                point
                for point in lane
                if point.valid
                and math.isfinite(
                    float(point.x)
                )
                and math.isfinite(
                    float(point.y)
                )
            ]

            if (
                len(valid_points)
                >= self.min_points_per_lane
            ):

                valid_points.sort(
                    key=lambda point:
                    point.y
                )

                valid_lanes.append(
                    valid_points
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

        # Uma única lane já é uma detecção válida.
        #
        # Isso é proposital.
        #
        # Não devemos descartar uma marcação real somente
        # porque a outra borda da faixa não foi detectada.
        valid = bool(
            valid_lanes
        )

        return LaneDetectionResult(
            lanes=valid_lanes,

            lane_confidences=confidences,

            # Nunca definido neste estágio.
            current_lane_index=None,

            left_lane=left_lane,

            right_lane=right_lane,

            additional_lanes=additional_lanes,

            left_confidence=left_confidence,

            right_confidence=right_confidence,

            valid=valid,

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
                self.last_output_shape
            ),

            error=None,
        )

    # =========================================================================
    # DIAGNÓSTICO
    # =========================================================================

    def _update_diagnostics(
        self,
        probability: np.ndarray,
        mask: np.ndarray,
        row_segments: List[
            List[_RowSegment]
        ],
        tracks: List[_LaneTrack],
        result: LaneDetectionResult,
    ) -> None:

        active_segments = sum(
            len(row)
            for row
            in row_segments
        )

        active_pixels = int(
            np.count_nonzero(
                mask
            )
        )

        self.last_diagnostics = {
            "device": self.get_device_name(),
            "provider": (
                self.session.get_providers()
                if self.session is not None
                else []
            ),
            "probability_shape": tuple(
                int(value)
                for value
                in probability.shape
            ),
            "mask_shape": tuple(
                int(value)
                for value
                in mask.shape
            ),
            "active_pixels": active_pixels,
            "active_pixel_ratio": (
                float(
                    active_pixels
                )
                / float(
                    max(
                        1,
                        mask.size,
                    )
                )
            ),
            "rows_with_segments": len(
                row_segments
            ),
            "row_segments": active_segments,
            "raw_tracks": len(
                tracks
            ),
            "valid_lanes": (
                result.num_lanes_detected
            ),
            "lane_confidences": list(
                result.lane_confidences
            ),
            "output_shape": (
                self.last_output_shape
            ),
        }

    # =========================================================================
    # DETECÇÃO PRINCIPAL
    # =========================================================================

    def detect(
        self,
        frame: np.ndarray,
    ) -> LaneDetectionResult:
        """
        Pipeline completo.

        frame
          ↓
        YOLOP
          ↓
        lane_line_seg
          ↓
        probability
          ↓
        mask
          ↓
        row segments
          ↓
        spatial association
          ↓
        multiple lanes
          ↓
        LaneDetectionResult
        """

        self.last_error = None
        self.last_diagnostics = {}

        try:

            frame_width, frame_height = (
                self._validate_frame(
                    frame
                )
            )

            lane_line_seg = (
                self.infer(frame)
            )

            probability = (
                self._lane_probability(
                    lane_line_seg
                )
            )

            if (
                probability.ndim != 2
                or probability.size == 0
            ):
                raise RuntimeError(
                    "Probabilidade de lane inválida: "
                    f"{probability.shape}"
                )

            probability = np.nan_to_num(
                probability,
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            )

            probability = np.clip(
                probability,
                0.0,
                1.0,
            ).astype(
                np.float32,
                copy=False,
            )

            mask = (
                probability
                >= self.lane_threshold
            ).astype(
                np.uint8
            )

            # Limpeza da máscara.
            mask = self._clean_mask(
                mask
            )

            self.last_lane_probability = (
                probability
            )

            self.last_lane_mask = (
                mask
            )

            row_segments = (
                self._extract_row_segments(
                    mask,
                    probability,
                )
            )

            tracks = (
                self._associate_segments(
                    row_segments
                )
            )

            lanes: List[
                List[LanePoint]
            ] = []

            mask_height, mask_width = (
                mask.shape
            )

            for track in tracks:

                if not self._validate_track(
                    track
                ):
                    continue

                lane = (
                    self._track_to_lane(
                        track=track,
                        mask_width=mask_width,
                        mask_height=mask_height,
                        frame_width=frame_width,
                        frame_height=frame_height,
                    )
                )

                if (
                    len(lane)
                    >= self.min_points_per_lane
                ):

                    lanes.append(
                        lane
                    )

            result = (
                self._build_result(
                    lanes=lanes,
                    frame_width=frame_width,
                    frame_height=frame_height,
                )
            )

            self._update_diagnostics(
                probability=probability,
                mask=mask,
                row_segments=row_segments,
                tracks=tracks,
                result=result,
            )

            self.last_result = result

            return result

        except Exception as exc:

            self.last_error = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            logger.exception(
                "[YOLOP] Falha durante detecção."
            )

            if (
                isinstance(
                    frame,
                    np.ndarray,
                )
                and frame.ndim >= 2
            ):

                frame_height = int(
                    frame.shape[0]
                )

                frame_width = int(
                    frame.shape[1]
                )

            else:

                frame_width = (
                    self.input_width
                )

                frame_height = (
                    self.input_height
                )

            result = (
                LaneDetectionResult(
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
                        self.last_output_shape
                    ),

                    error=self.last_error,
                )
            )

            self.last_result = result

            return result

    # =========================================================================
    # LIMPEZA DA MÁSCARA
    # =========================================================================

    def _clean_mask(
        self,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Limpeza conservadora da máscara.

        Importante:

        Não usa erosão agressiva nem exige largura mínima
        artificial das linhas.

        O objetivo é preservar marcações finas.
        """

        mask = (
            np.asarray(mask)
            .astype(np.uint8)
        )

        if mask.ndim != 2:
            raise ValueError(
                "Máscara deve ser 2D."
            )

        if (
            self.morph_kernel <= 1
            and self.min_component_area <= 1
        ):
            return mask

        cleaned = mask.copy()

        if self.morph_kernel > 1:

            kernel = np.ones(
                (
                    self.morph_kernel,
                    self.morph_kernel,
                ),
                dtype=np.uint8,
            )

            # Somente fechamento.
            #
            # Evita quebrar linhas finas contínuas.
            cleaned = cv2.morphologyEx(
                cleaned,
                cv2.MORPH_CLOSE,
                kernel,
                iterations=1,
            )

        if self.min_component_area > 1:

            num_labels, labels, stats, _ = (
                cv2.connectedComponentsWithStats(
                    cleaned,
                    connectivity=8,
                )
            )

            filtered = np.zeros_like(
                cleaned
            )

            for label in range(
                1,
                num_labels,
            ):

                area = int(
                    stats[
                        label,
                        cv2.CC_STAT_AREA,
                    ]
                )

                if (
                    area
                    >= self.min_component_area
                ):

                    filtered[
                        labels == label
                    ] = 1

            cleaned = filtered

        return cleaned.astype(
            np.uint8,
            copy=False,
        )


# =============================================================================
# FACTORY
# =============================================================================

def create_default_detector(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    **kwargs: Any,
) -> YOLOPLaneDetector:
    """
    Cria o detector padrão do projeto.
    """

    return YOLOPLaneDetector(
        model_path=model_path,
        **kwargs,
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "LanePoint",
    "LaneDetectionResult",
    "YOLOPLaneDetector",
    "create_default_detector",
]