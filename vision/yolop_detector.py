"""
vision/yolop_detector.py

Detector de lanes baseado em YOLOP + ONNX Runtime.

Responsabilidade:

    YOLOP ONNX
        ↓
    lane_line_seg
        ↓
    probabilidade de lane
        ↓
    máscara binária
        ↓
    extração das linhas
        ↓
    LaneDetectionResult

Este módulo é responsável SOMENTE pela percepção inicial
das linhas de faixa.

Não calcula:

    - geometria da faixa;
    - modelo polinomial;
    - projeção;
    - associação semântica;
    - posição do veículo;
    - confiança temporal;
    - estado ADAS;
    - decisão ADAS.

Essas responsabilidades pertencem aos módulos posteriores.

Compatibilidade:

    LaneDetectionResult
        ↓
    LaneTracker / arquitetura legada

LanePoint é importado exclusivamente de lane_types.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from .lane_types import LanePoint


logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "weights"
    / "yolop-640-640.onnx"
)

YOLOP_INPUT_WIDTH = 640
YOLOP_INPUT_HEIGHT = 640


# ============================================================================
# RESULTADO
# ============================================================================

@dataclass
class LaneDetectionResult:
    """
    Resultado bruto da percepção de lanes pelo YOLOP.

    Observação:

    Este objeto ainda mantém a estrutura de compatibilidade
    usada pelo pipeline anterior.

    A transformação posterior para LaneLine / LaneModel
    deverá ocorrer nas camadas seguintes.

    O detector NÃO decide qual é a faixa ocupada pelo veículo.
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
        """
        Indica se alguma lane foi detectada.
        """

        return self.num_lanes_detected > 0

    @property
    def has_current_lane(self) -> bool:
        """
        Compatibilidade com o pipeline anterior.

        Importante:

        current_lane_index continua sendo None no detector,
        porque o detector não possui informação suficiente
        para determinar semanticamente a faixa ocupada.

        Essa decisão pertence ao lane_assignment.py.
        """

        return (
            self.current_lane_index is not None
            and bool(self.left_lane)
            and bool(self.right_lane)
        )


# ============================================================================
# DETECTOR
# ============================================================================

class YOLOPLaneDetector:
    """
    Detector YOLOP utilizando ONNX Runtime.

    O YOLOP fornece a segmentação das linhas da pista.

    O detector transforma essa segmentação em pontos no
    sistema de coordenadas do frame original.

    Não realiza fitting geométrico nem tracking temporal.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        input_width: int = YOLOP_INPUT_WIDTH,
        input_height: int = YOLOP_INPUT_HEIGHT,
        lane_threshold: float = 0.50,
        min_points_per_lane: int = 5,
        row_step: int = 8,
        min_lane_pixels_per_row: int = 2,
        providers: Optional[List[str]] = None,
    ) -> None:

        self.model_path = Path(
            model_path
        )

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

        self.providers = (
            list(providers)
            if providers is not None
            else None
        )

        self.session: Optional[
            ort.InferenceSession
        ] = None

        self.input_name: Optional[str] = None

        self.loaded = False

        self.last_output_shape: Tuple[
            int, ...
        ] = tuple()

        self.last_error: Optional[str] = None

        self.last_result: Optional[
            LaneDetectionResult
        ] = None

    # ========================================================================
    # MODEL
    # ========================================================================

    def model_exists(self) -> bool:
        """
        Verifica se o arquivo ONNX existe.
        """

        return self.model_path.is_file()

    def load_model(self) -> bool:
        """
        Carrega o modelo YOLOP.

        Se providers não forem especificados:

            CUDA → CPU

        quando disponíveis.
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

        try:

            if self.providers is None:

                available = (
                    ort.get_available_providers()
                )

                providers: List[str] = []

                if (
                    "CUDAExecutionProvider"
                    in available
                ):
                    providers.append(
                        "CUDAExecutionProvider"
                    )

                if (
                    "CPUExecutionProvider"
                    in available
                ):
                    providers.append(
                        "CPUExecutionProvider"
                    )

                if not providers:

                    raise RuntimeError(
                        "Nenhum ExecutionProvider "
                        "compatível foi encontrado."
                    )

            else:

                providers = list(
                    self.providers
                )

            self.session = (
                ort.InferenceSession(
                    str(self.model_path),
                    providers=providers,
                )
            )

            inputs = (
                self.session.get_inputs()
            )

            if not inputs:

                raise RuntimeError(
                    "O modelo YOLOP ONNX "
                    "não possui entradas."
                )

            self.input_name = (
                inputs[0].name
            )

            self.loaded = True

            logger.info(
                "[YOLOP] Modelo carregado: %s",
                self.model_path,
            )

            logger.info(
                "[YOLOP] Providers: %s",
                self.session.get_providers(),
            )

            logger.info(
                "[YOLOP] Device: %s",
                self.get_device_name(),
            )

            logger.info(
                "[YOLOP] Input: %s",
                self.input_name,
            )

            return True

        except Exception as exc:

            self.last_error = (
                f"{type(exc).__name__}: {exc}"
            )

            logger.exception(
                "[YOLOP] Falha ao carregar modelo."
            )

            self.session = None
            self.input_name = None
            self.loaded = False

            return False

    # ========================================================================
    # DEVICE
    # ========================================================================

    def get_device_name(self) -> str:
        """
        Retorna o provider efetivamente utilizado.
        """

        if self.session is None:
            return "NOT_LOADED"

        providers = (
            self.session.get_providers()
        )

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

    # ========================================================================
    # IMAGEM
    # ========================================================================

    @staticmethod
    def _load_image_unicode(
        path: Path,
    ) -> np.ndarray:
        """
        Carrega imagem suportando caminhos Unicode no Windows.
        """

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
                "Não foi possível carregar "
                f"a imagem:\n{path}"
            )

        return image

    @staticmethod
    def _save_image_unicode(
        path: Path,
        image: np.ndarray,
    ) -> None:
        """
        Salva imagem suportando caminhos Unicode no Windows.
        """

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        extension = (
            path.suffix
            if path.suffix
            else ".png"
        )

        success, encoded = (
            cv2.imencode(
                extension,
                image,
            )
        )

        if not success:

            raise RuntimeError(
                "Não foi possível salvar "
                f"a imagem:\n{path}"
            )

        encoded.tofile(
            str(path)
        )

    # ========================================================================
    # PREPROCESSAMENTO
    # ========================================================================

    def preprocess(
        self,
        frame: np.ndarray,
    ) -> np.ndarray:
        """
        Converte o frame BGR para:

            [1, 3, H, W]

        float32 normalizado em [0, 1].
        """

        if frame is None:

            raise ValueError(
                "Frame é None."
            )

        if not isinstance(
            frame,
            np.ndarray,
        ):

            raise TypeError(
                "Frame deve ser "
                "numpy.ndarray."
            )

        if frame.ndim != 3:

            raise ValueError(
                "Frame deve possuir "
                "formato HxWxC."
            )

        if frame.shape[2] != 3:

            raise ValueError(
                "Frame deve possuir "
                "3 canais BGR."
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

    # ========================================================================
    # INFERÊNCIA
    # ========================================================================

    def infer(
        self,
        frame: np.ndarray,
    ) -> np.ndarray:
        """
        Executa inferência YOLOP e retorna lane_line_seg.
        """

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
                "Nome da entrada ONNX "
                "não definido."
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

        if len(outputs) < 3:

            raise RuntimeError(
                "YOLOP retornou "
                f"{len(outputs)} saídas; "
                "eram esperadas pelo menos 3."
            )

        # YOLOP:
        #
        # outputs[0] -> det_out
        # outputs[1] -> drive_area_seg
        # outputs[2] -> lane_line_seg

        lane_line_seg = outputs[2]

        if not isinstance(
            lane_line_seg,
            np.ndarray,
        ):

            raise RuntimeError(
                "lane_line_seg não é "
                "numpy.ndarray."
            )

        self.last_output_shape = tuple(
            int(value)
            for value in lane_line_seg.shape
        )

        return lane_line_seg

    # ========================================================================
    # PROBABILIDADE
    # ========================================================================

    def _lane_probability(
        self,
        lane_line_seg: np.ndarray,
    ) -> np.ndarray:
        """
        Converte a saída de segmentação em probabilidade
        do pixel pertencer à classe lane.

        Esperado:

            [1, 2, H, W]

        ou:

            [2, H, W]
        """

        output = np.asarray(
            lane_line_seg
        )

        if output.ndim == 4:

            if output.shape[0] != 1:

                raise ValueError(
                    "Batch de lane_line_seg "
                    "deve possuir tamanho 1: "
                    f"{output.shape}"
                )

            output = output[0]

        if output.ndim != 3:

            raise ValueError(
                "lane_line_seg deve possuir "
                "formato [1,2,H,W] ou [2,H,W]. "
                f"Recebido: {output.shape}"
            )

        if output.shape[0] != 2:

            raise ValueError(
                "Esperados exatamente "
                "2 canais na lane_line_seg. "
                f"Recebido: {output.shape}"
            )

        logits = output.astype(
            np.float32,
            copy=False,
        )

        # Softmax numericamente estável.
        max_value = np.max(
            logits,
            axis=0,
            keepdims=True,
        )

        exponentials = np.exp(
            logits - max_value
        )

        denominator = np.sum(
            exponentials,
            axis=0,
            keepdims=True,
        )

        denominator = np.maximum(
            denominator,
            np.finfo(
                np.float32
            ).eps,
        )

        probabilities = (
            exponentials
            / denominator
        )

        return np.asarray(
            probabilities[1],
            dtype=np.float32,
        )

    # ========================================================================
    # MÁSCARA
    # ========================================================================

    def create_lane_mask(
        self,
        lane_line_seg: np.ndarray,
    ) -> np.ndarray:
        """
        Converte lane_line_seg em máscara binária.

        Retorno:

            uint8
            0 = background
            1 = lane
        """

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

        return mask

    # ========================================================================
    # SEGMENTOS
    # ========================================================================

    @staticmethod
    def _split_row_segments(
        xs: np.ndarray,
    ) -> List[np.ndarray]:
        """
        Divide os pixels ativos de uma linha em segmentos
        horizontalmente separados.
        """

        if xs.size == 0:
            return []

        gaps = np.diff(
            xs
        )

        split_indices = (
            np.where(
                gaps > 3
            )[0]
            + 1
        )

        return list(
            np.split(
                xs,
                split_indices,
            )
        )

    # ========================================================================
    # PONTO
    # ========================================================================

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
        """
        Converte coordenadas da máscara para o frame original.
        """

        if mask_width <= 0:
            raise ValueError(
                "mask_width inválido."
            )

        if mask_height <= 0:
            raise ValueError(
                "mask_height inválido."
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
            x=px,
            y=py,
            confidence=float(
                np.clip(
                    confidence,
                    0.0,
                    1.0,
                )
            ),
            valid=True,
        )

    # ========================================================================
    # EXTRAÇÃO
    # ========================================================================

    def _extract_lane_points(
        self,
        mask: np.ndarray,
        probability: np.ndarray,
        frame_width: int,
        frame_height: int,
    ) -> Tuple[
        List[LanePoint],
        List[LanePoint],
    ]:
        """
        Extrai as duas linhas principais da pista.

        A análise ocorre de baixo para cima.

        O algoritmo:

            1. encontra segmentos ativos;
            2. procura uma referência inicial;
            3. acompanha cada linha verticalmente;
            4. utiliza continuidade espacial;
            5. converte os pontos para o frame original.

        Este método NÃO faz fitting.
        """

        if mask.ndim != 2:

            raise ValueError(
                "Máscara deve ser 2D."
            )

        if probability.shape != mask.shape:

            raise ValueError(
                "Máscara e probabilidade "
                "possuem dimensões diferentes: "
                f"{mask.shape} vs "
                f"{probability.shape}"
            )

        mask_height, mask_width = (
            mask.shape
        )

        left_points: List[
            LanePoint
        ] = []

        right_points: List[
            LanePoint
        ] = []

        if (
            mask_height <= 0
            or mask_width <= 0
        ):
            return (
                left_points,
                right_points,
            )

        image_center = (
            mask_width / 2.0
        )

        max_tracking_jump = max(
            20.0,
            mask_width * 0.12,
        )

        # --------------------------------------------------------------------
        # Coleta das linhas
        # --------------------------------------------------------------------

        rows = range(
            mask_height - 1,
            -1,
            -self.row_step,
        )

        row_data = []

        for y in rows:

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

            candidates = []

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
                            segment
                        ]
                    )
                )

                candidates.append(
                    {
                        "min": x_min,
                        "max": x_max,
                        "center": x_center,
                        "confidence": confidence,
                        "size": int(
                            segment.size
                        ),
                    }
                )

            if candidates:

                row_data.append(
                    (
                        y,
                        candidates,
                    )
                )

        if not row_data:

            return (
                left_points,
                right_points,
            )

        # --------------------------------------------------------------------
        # Encontrar referência inicial
        # --------------------------------------------------------------------

        previous_left: Optional[
            float
        ] = None

        previous_right: Optional[
            float
        ] = None

        initial_found = False

        for _y, candidates in row_data:

            left_candidates = [
                candidate
                for candidate in candidates
                if candidate["center"]
                < image_center
            ]

            right_candidates = [
                candidate
                for candidate in candidates
                if candidate["center"]
                >= image_center
            ]

            if (
                left_candidates
                and right_candidates
            ):

                left_candidate = max(
                    left_candidates,
                    key=lambda candidate:
                    candidate["center"],
                )

                right_candidate = min(
                    right_candidates,
                    key=lambda candidate:
                    candidate["center"],
                )

                previous_left = (
                    left_candidate["center"]
                )

                previous_right = (
                    right_candidate["center"]
                )

                initial_found = True

                break

        # --------------------------------------------------------------------
        # Caso a máscara apresente apenas uma região:
        #
        # Não inventamos duas lanes.
        #
        # Utilizamos as bordas da região somente se ela possuir
        # largura suficiente para representar duas bordas.
        # --------------------------------------------------------------------

        if not initial_found:

            _y, candidates = (
                row_data[0]
            )

            if not candidates:

                return (
                    left_points,
                    right_points,
                )

            candidate = min(
                candidates,
                key=lambda item:
                abs(
                    item["center"]
                    - image_center
                ),
            )

            width = (
                candidate["max"]
                - candidate["min"]
            )

            minimum_double_edge_width = (
                mask_width * 0.08
            )

            if (
                width
                >= minimum_double_edge_width
            ):

                previous_left = (
                    candidate["min"]
                )

                previous_right = (
                    candidate["max"]
                )

            else:

                # Não há evidência suficiente
                # para produzir duas linhas.
                return (
                    left_points,
                    right_points,
                )

        # --------------------------------------------------------------------
        # Tracking vertical dentro do frame.
        # --------------------------------------------------------------------

        tracked_left = []

        tracked_right = []

        for y, candidates in row_data:

            if not candidates:
                continue

            # ================================================================
            # DUAS OU MAIS REGIÕES
            # ================================================================

            if len(candidates) >= 2:

                left_choice = min(
                    candidates,
                    key=lambda candidate:
                    abs(
                        candidate["center"]
                        - float(
                            previous_left
                        )
                    ),
                )

                right_choice = min(
                    candidates,
                    key=lambda candidate:
                    abs(
                        candidate["center"]
                        - float(
                            previous_right
                        )
                    ),
                )

                # Não utilizar a mesma região para os dois lados.
                if (
                    left_choice
                    is right_choice
                ):

                    ordered = sorted(
                        candidates,
                        key=lambda candidate:
                        candidate["center"],
                    )

                    if len(ordered) >= 2:

                        left_choice = min(
                            ordered,
                            key=lambda candidate:
                            abs(
                                candidate["center"]
                                - float(
                                    previous_left
                                )
                            ),
                        )

                        remaining = [
                            candidate
                            for candidate in ordered
                            if candidate
                            is not left_choice
                        ]

                        right_choice = min(
                            remaining,
                            key=lambda candidate:
                            abs(
                                candidate["center"]
                                - float(
                                    previous_right
                                )
                            ),
                        )

                left_distance = abs(
                    left_choice["center"]
                    - float(
                        previous_left
                    )
                )

                right_distance = abs(
                    right_choice["center"]
                    - float(
                        previous_right
                    )
                )

                if (
                    left_distance
                    <= max_tracking_jump
                ):

                    previous_left = (
                        left_choice["center"]
                    )

                    tracked_left.append(
                        (
                            y,
                            left_choice[
                                "center"
                            ],
                            left_choice[
                                "confidence"
                            ],
                        )
                    )

                if (
                    right_distance
                    <= max_tracking_jump
                ):

                    previous_right = (
                        right_choice["center"]
                    )

                    tracked_right.append(
                        (
                            y,
                            right_choice[
                                "center"
                            ],
                            right_choice[
                                "confidence"
                            ],
                        )
                    )

                continue

            # ================================================================
            # UMA REGIÃO
            # ================================================================

            candidate = candidates[0]

            segment_min = (
                candidate["min"]
            )

            segment_max = (
                candidate["max"]
            )

            segment_width = (
                segment_max
                - segment_min
            )

            # Se a região for muito estreita,
            # não podemos afirmar que ela representa
            # duas bordas da faixa.
            if (
                segment_width
                < minimum_double_edge_width
            ):

                continue

            left_x = segment_min

            right_x = segment_max

            left_distance = abs(
                left_x
                - float(
                    previous_left
                )
            )

            right_distance = abs(
                right_x
                - float(
                    previous_right
                )
            )

            if (
                left_distance
                <= max_tracking_jump
            ):

                previous_left = (
                    left_x
                )

                tracked_left.append(
                    (
                        y,
                        left_x,
                        candidate[
                            "confidence"
                        ],
                    )
                )

            if (
                right_distance
                <= max_tracking_jump
            ):

                previous_right = (
                    right_x
                )

                tracked_right.append(
                    (
                        y,
                        right_x,
                        candidate[
                            "confidence"
                        ],
                    )
                )

        # --------------------------------------------------------------------
        # Converter para LanePoint
        # --------------------------------------------------------------------

        for (
            y,
            x,
            confidence,
        ) in tracked_left:

            left_points.append(
                self._make_point(
                    x=x,
                    y=y,
                    confidence=confidence,
                    mask_width=mask_width,
                    mask_height=mask_height,
                    frame_width=frame_width,
                    frame_height=frame_height,
                )
            )

        for (
            y,
            x,
            confidence,
        ) in tracked_right:

            right_points.append(
                self._make_point(
                    x=x,
                    y=y,
                    confidence=confidence,
                    mask_width=mask_width,
                    mask_height=mask_height,
                    frame_width=frame_width,
                    frame_height=frame_height,
                )
            )

        # Coordenadas ordenadas verticalmente.
        left_points.sort(
            key=lambda point:
            point.y
        )

        right_points.sort(
            key=lambda point:
            point.y
        )

        return (
            left_points,
            right_points,
        )

    # ========================================================================
    # CONFIANÇA
    # ========================================================================

    @staticmethod
    def _lane_confidence(
        lane: List[LanePoint],
    ) -> float:
        """
        Calcula a confiança média dos pontos válidos.
        """

        valid_points = [
            point
            for point in lane
            if point.valid
            and np.isfinite(
                point.confidence
            )
        ]

        if not valid_points:
            return 0.0

        return float(
            np.clip(
                np.mean(
                    [
                        point.confidence
                        for point in valid_points
                    ]
                ),
                0.0,
                1.0,
            )
        )

    # ========================================================================
    # RESULTADO
    # ========================================================================

    def _build_result(
        self,
        left_lane: List[LanePoint],
        right_lane: List[LanePoint],
        frame_width: int,
        frame_height: int,
    ) -> LaneDetectionResult:
        """
        Monta o resultado compatível com o pipeline.

        O detector não determina current_lane_index.
        """

        if (
            len(left_lane)
            < self.min_points_per_lane
        ):

            left_lane = []

        if (
            len(right_lane)
            < self.min_points_per_lane
        ):

            right_lane = []

        lanes: List[
            List[LanePoint]
        ] = []

        if left_lane:
            lanes.append(
                left_lane
            )

        if right_lane:
            lanes.append(
                right_lane
            )

        left_confidence = (
            self._lane_confidence(
                left_lane
            )
        )

        right_confidence = (
            self._lane_confidence(
                right_lane
            )
        )

        valid = bool(
            left_lane
            and right_lane
        )

        return LaneDetectionResult(
            lanes=lanes,

            lane_confidences=[
                left_confidence,
                right_confidence,
            ],

            # IMPORTANTE:
            #
            # O detector não sabe qual faixa o
            # veículo está ocupando.
            #
            # lane_assignment.py será responsável
            # por essa decisão.
            current_lane_index=None,

            left_lane=left_lane,

            right_lane=right_lane,

            additional_lanes=[],

            left_confidence=(
                left_confidence
            ),

            right_confidence=(
                right_confidence
            ),

            valid=valid,

            num_lanes_detected=len(
                lanes
            ),

            input_width=frame_width,

            input_height=frame_height,

            model_output_shape=(
                self.last_output_shape
            ),

            error=None,
        )

    # ========================================================================
    # API PRINCIPAL
    # ========================================================================

    def detect(
        self,
        frame: np.ndarray,
    ) -> LaneDetectionResult:
        """
        Executa o pipeline completo:

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
            lane points
              ↓
            LaneDetectionResult
        """

        self.last_error = None

        try:

            if frame is None:

                raise ValueError(
                    "Frame é None."
                )

            if not isinstance(
                frame,
                np.ndarray,
            ):

                raise TypeError(
                    "Frame deve ser "
                    "numpy.ndarray."
                )

            if frame.ndim != 3:

                raise ValueError(
                    "Frame deve possuir "
                    "formato HxWxC."
                )

            if frame.shape[2] != 3:

                raise ValueError(
                    "Frame deve possuir "
                    "3 canais."
                )

            frame_height, frame_width = (
                frame.shape[:2]
            )

            lane_line_seg = (
                self.infer(frame)
            )

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

            left_lane, right_lane = (
                self._extract_lane_points(
                    mask=mask,
                    probability=probability,
                    frame_width=frame_width,
                    frame_height=frame_height,
                )
            )

            result = (
                self._build_result(
                    left_lane=left_lane,
                    right_lane=right_lane,
                    frame_width=frame_width,
                    frame_height=frame_height,
                )
            )

            self.last_result = result

            return result

        except Exception as exc:

            self.last_error = (
                f"{type(exc).__name__}: {exc}"
            )

            logger.exception(
                "[YOLOP] Falha durante detecção."
            )

            frame_width = (
                int(frame.shape[1])
                if (
                    isinstance(
                        frame,
                        np.ndarray,
                    )
                    and frame.ndim >= 2
                )
                else self.input_width
            )

            frame_height = (
                int(frame.shape[0])
                if (
                    isinstance(
                        frame,
                        np.ndarray,
                    )
                    and frame.ndim >= 2
                )
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

                input_width=frame_width,

                input_height=frame_height,

                model_output_shape=(
                    self.last_output_shape
                ),

                error=self.last_error,
            )

            self.last_result = result

            return result


# ============================================================================
# FACTORY
# ============================================================================

def create_default_detector(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    **kwargs,
) -> YOLOPLaneDetector:
    """
    Cria um detector YOLOP utilizando a configuração padrão.
    """

    return YOLOPLaneDetector(
        model_path=model_path,
        **kwargs,
    )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "LanePoint",
    "LaneDetectionResult",
    "YOLOPLaneDetector",
    "create_default_detector",
]