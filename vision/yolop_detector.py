
"""
vision/yolop_detector.py

Detector de lanes baseado em YOLOP + ONNX Runtime.

Responsabilidade:
YOLOP ONNX
↓
lane_line_seg
↓
máscara binária
↓
extração das linhas esquerda/direita
↓
LaneDetectionResult

Compatibilidade:
Mantém LanePoint e LaneDetectionResult compatíveis
com a arquitetura anteriormente utilizada pelo UFLD.

IMPORTANTE:
Este módulo não calcula:

- geometria da faixa
- erro lateral
- heading
- estado ADAS

Essas responsabilidades continuam nos módulos existentes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

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
# RESULTADOS
# ============================================================================

@dataclass
class LanePoint:
    """
    Ponto de uma linha de faixa.

    x:
        Coordenada X no frame original.

    y:
        Coordenada Y no frame original.

    confidence:
        Confiança estimada para o ponto.

    valid:
        Indica se o ponto é válido.
    """

    x: float
    y: float
    confidence: float
    valid: bool = True


@dataclass
class LaneDetectionResult:
    """
    Resultado da detecção de lanes.

    Mantém a estrutura usada pelo pipeline UFLD.
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


# ============================================================================
# DETECTOR
# ============================================================================

class YOLOPLaneDetector:
    """
    Detector YOLOP usando ONNX Runtime.

    A saída lane_line_seg do YOLOP é convertida em duas linhas:

        left_lane
        right_lane

    O restante do pipeline permanece independente do modelo.
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

        self.model_path = Path(model_path)

        self.input_width = int(input_width)
        self.input_height = int(input_height)

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

        self.providers = providers

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
        return self.model_path.is_file()

    def load_model(self) -> bool:
        """
        Carrega o modelo YOLOP ONNX.

        Prioridade:
            1. CUDAExecutionProvider
            2. CPUExecutionProvider

        Se CUDA não estiver disponível, utiliza CPU.
        """

        if self.loaded and self.session is not None:
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

                providers = []

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
                        "compatível encontrado."
                    )

            else:

                providers = list(
                    self.providers
                )

            self.session = ort.InferenceSession(
                str(self.model_path),
                providers=providers,
            )

            inputs = self.session.get_inputs()

            if not inputs:
                raise RuntimeError(
                    "YOLOP ONNX não possui entradas."
                )

            self.input_name = inputs[0].name

            self.loaded = True

            logger.info(
                "[YOLOP] Modelo carregado: %s",
                self.model_path,
            )

            logger.info(
                "[YOLOP] Providers da sessão: %s",
                self.session.get_providers(),
            )

            logger.info(
                "[YOLOP] Provider selecionado: %s",
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
            self.loaded = False

            return False

    # ========================================================================
    # DEVICE / PROVIDER
    # ========================================================================

    def get_device_name(self) -> str:

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

        return (
            providers[0]
            if providers
            else "UNKNOWN"
        )

    # ========================================================================
    # PREPROCESSAMENTO
    # ========================================================================

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
                "Não foi possível carregar "
                f"a imagem:\n{path}"
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
                "Não foi possível salvar "
                f"a imagem:\n{path}"
            )

        encoded.tofile(
            str(path)
        )

    def preprocess(
        self,
        frame: np.ndarray,
    ) -> np.ndarray:

        if frame is None:
            raise ValueError(
                "Frame é None."
            )

        if frame.ndim != 3:
            raise ValueError(
                "Frame deve possuir formato HxWxC."
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

    # ========================================================================
    # INFERÊNCIA
    # ========================================================================

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
                "Nome da entrada ONNX "
                "não definido."
            )

        tensor = self.preprocess(frame)

        outputs = self.session.run(
            None,
            {
                self.input_name: tensor
            },
        )

        if len(outputs) < 3:

            raise RuntimeError(
                "YOLOP retornou menos de "
                f"3 saídas: {len(outputs)}"
            )

        # Modelo YOLOP validado:
        #
        # outputs[0] = det_out
        # outputs[1] = drive_area_seg
        # outputs[2] = lane_line_seg

        lane_line_seg = outputs[2]

        self.last_output_shape = tuple(
            lane_line_seg.shape
        )

        return lane_line_seg

    # ========================================================================
    # MÁSCARA
    # ========================================================================

    def create_lane_mask(
        self,
        lane_line_seg: np.ndarray,
    ) -> np.ndarray:
        """
        Converte lane_line_seg [1, 2, H, W]
        em máscara binária.

        Canal 0 = background
        Canal 1 = lane.
        """

        if lane_line_seg.ndim != 4:

            raise ValueError(
                "lane_line_seg inválido: "
                f"{lane_line_seg.shape}"
            )

        output = lane_line_seg[0]

        if output.shape[0] != 2:

            raise ValueError(
                "Esperados 2 canais na "
                "lane_line_seg. Recebido: "
                f"{output.shape}"
            )

        # O modelo validado retorna os dois canais
        # diretamente para classificação por argmax.
        class_map = np.argmax(
            output,
            axis=0,
        )

        # Confiança relativa do canal lane.
        #
        # Como o modelo retorna dois scores por pixel,
        # convertemos a diferença entre os dois canais
        # em uma probabilidade aproximada via softmax.
        max_values = np.max(
            output,
            axis=0,
            keepdims=True,
        )

        exp_values = np.exp(
            output - max_values
        )

        probabilities = (
            exp_values
            / np.sum(
                exp_values,
                axis=0,
                keepdims=True,
            )
        )

        lane_probability = (
            probabilities[1]
        )

        mask = (
            (class_map == 1)
            & (
                lane_probability
                >= self.lane_threshold
            )
        ).astype(np.uint8)

        return mask

    # ========================================================================
    # EXTRAÇÃO DAS LANES
    # ========================================================================

    @staticmethod
    def _split_row_segments(
        xs: np.ndarray,
    ) -> List[np.ndarray]:

        if xs.size == 0:
            return []

        split_indices = (
            np.where(
                np.diff(xs) > 3
            )[0]
            + 1
        )

        return list(
            np.split(
                xs,
                split_indices,
            )
        )

    def _extract_lane_points(
        self,
        mask: np.ndarray,
        frame_width: int,
        frame_height: int,
    ) -> Tuple[
        List[LanePoint],
        List[LanePoint],
    ]:
        """
        Extrai as duas bordas da faixa acompanhando
        a continuidade vertical da máscara.

        Diferentemente da implementação anterior,
        não depende exclusivamente de:

            x < image_center
            x >= image_center

        Isso é importante nas regiões inferiores do
        frame, onde a máscara do YOLOP pode formar uma
        única região contínua ou deslocar-se em relação
        ao centro da imagem.

        A estratégia é:

        1. analisar a máscara de baixo para cima;
        2. encontrar segmentos válidos em cada linha;
        3. manter a posição anterior das duas lanes;
        4. procurar candidatos próximos dessas posições;
        5. quando houver apenas um segmento, utilizar suas
           extremidades como candidatos esquerdo/direito;
        6. rejeitar saltos muito grandes para evitar que
           ruído seja interpretado como faixa.

        Nenhuma geometria é calculada aqui.
        """

        mask_height, mask_width = mask.shape

        left_points: List[LanePoint] = []
        right_points: List[LanePoint] = []

        if mask_height <= 0 or mask_width <= 0:
            return left_points, right_points

        # --------------------------------------------------------------------
        # Estado das duas trajetórias.
        #
        # Começamos sem uma posição conhecida e deixamos
        # as linhas superiores/inferiores fornecerem a
        # primeira referência.
        # --------------------------------------------------------------------

        previous_left: Optional[float] = None
        previous_right: Optional[float] = None

        # Tolerância máxima de deslocamento horizontal entre
        # linhas consecutivas da máscara.
        max_tracking_jump = max(
            20.0,
            mask_width * 0.12,
        )

        image_center = mask_width / 2.0

        # --------------------------------------------------------------------
        # Primeiro levantamento: coletamos candidatos por linha.
        #
        # Isso permite encontrar uma referência inicial antes
        # de fazer o acompanhamento.
        # --------------------------------------------------------------------

        rows = list(
            range(
                mask_height - 1,
                -1,
                -self.row_step,
            )
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
                self._split_row_segments(xs)
            )

            valid_segments = []

            for segment in segments:

                if (
                    segment.size
                    < self.min_lane_pixels_per_row
                ):
                    continue

                valid_segments.append(
                    segment
                )

            if not valid_segments:
                continue

            candidates = []

            for segment in valid_segments:

                candidates.append(
                    {
                        "min": float(segment[0]),
                        "max": float(segment[-1]),
                        "center": float(
                            np.mean(segment)
                        ),
                        "size": int(
                            segment.size
                        ),
                    }
                )

            row_data.append(
                (
                    y,
                    candidates,
                )
            )

        if not row_data:
            return left_points, right_points

        # --------------------------------------------------------------------
        # Referências iniciais.
        #
        # Procuramos primeiro uma linha que forneça duas
        # regiões separadas. Isso é mais confiável do que
        # iniciar diretamente no último row, onde a máscara
        # pode conter somente uma região.
        # --------------------------------------------------------------------

        initial_found = False

        for y, candidates in row_data:

            left_candidates = [
                c for c in candidates
                if c["center"] < image_center
            ]

            right_candidates = [
                c for c in candidates
                if c["center"] >= image_center
            ]

            if (
                left_candidates
                and right_candidates
            ):

                left_candidate = max(
                    left_candidates,
                    key=lambda c: c["center"],
                )

                right_candidate = min(
                    right_candidates,
                    key=lambda c: c["center"],
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
        # Se não encontramos duas regiões separadas,
        # iniciamos pela região mais próxima do centro.
        # --------------------------------------------------------------------

        if not initial_found:

            _, candidates = row_data[0]

            closest = min(
                candidates,
                key=lambda c: abs(
                    c["center"] - image_center
                ),
            )

            center = closest["center"]

            half_width = mask_width * 0.13

            previous_left = max(
                0.0,
                center - half_width,
            )

            previous_right = min(
                float(mask_width - 1),
                center + half_width,
            )

        # --------------------------------------------------------------------
        # Acompanhamento vertical.
        #
        # row_data está de baixo para cima.
        # Depois revertemos os pontos para Y crescente.
        # --------------------------------------------------------------------

        tracked_left = []
        tracked_right = []

        for y, candidates in row_data:

            if not candidates:
                continue

            # ---------------------------------------------------------------
            # Caso 1:
            # existem várias regiões.
            #
            # Procuramos a candidata mais próxima de cada
            # trajetória conhecida.
            # ---------------------------------------------------------------

            if len(candidates) >= 2:

                left_choice = min(
                    candidates,
                    key=lambda c: abs(
                        c["center"]
                        - float(previous_left)
                    ),
                )

                right_choice = min(
                    candidates,
                    key=lambda c: abs(
                        c["center"]
                        - float(previous_right)
                    ),
                )

                # Não permitir que a mesma região seja usada
                # simultaneamente como esquerda e direita,
                # exceto quando não há alternativa.
                if (
                    left_choice is right_choice
                    and len(candidates) > 1
                ):

                    ordered = sorted(
                        candidates,
                        key=lambda c: c["center"],
                    )

                    left_choice = min(
                        ordered,
                        key=lambda c: abs(
                            c["center"]
                            - float(previous_left)
                        ),
                    )

                    remaining = [
                        c for c in ordered
                        if c is not left_choice
                    ]

                    right_choice = min(
                        remaining,
                        key=lambda c: abs(
                            c["center"]
                            - float(previous_right)
                        ),
                    )

                left_distance = abs(
                    left_choice["center"]
                    - float(previous_left)
                )

                right_distance = abs(
                    right_choice["center"]
                    - float(previous_right)
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
                            left_choice["center"],
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
                            right_choice["center"],
                        )
                    )

                continue

            # ---------------------------------------------------------------
            # Caso 2:
            # existe apenas uma região.
            #
            # Aqui está a principal correção.
            #
            # Em vez de descartar a linha porque ela não pode
            # ser classificada simplesmente pelo centro da imagem,
            # usamos as duas extremidades da região.
            #
            # A extremidade mais próxima da trajetória esquerda
            # alimenta a esquerda; a mais próxima da trajetória
            # direita alimenta a direita.
            # ---------------------------------------------------------------

            candidate = candidates[0]

            segment_min = candidate["min"]
            segment_max = candidate["max"]

            left_distance_to_min = abs(
                segment_min
                - float(previous_left)
            )

            left_distance_to_max = abs(
                segment_max
                - float(previous_left)
            )

            right_distance_to_min = abs(
                segment_min
                - float(previous_right)
            )

            right_distance_to_max = abs(
                segment_max
                - float(previous_right)
            )

            left_x = (
                segment_min
                if left_distance_to_min
                <= left_distance_to_max
                else segment_max
            )

            right_x = (
                segment_min
                if right_distance_to_min
                <= right_distance_to_max
                else segment_max
            )

            # Se as duas trajetórias apontarem para a mesma
            # extremidade, usamos a extremidade oposta para
            # preservar a separação esquerda/direita.
            if left_x == right_x:

                if (
                    previous_left
                    <= previous_right
                ):
                    left_x = segment_min
                    right_x = segment_max

                else:
                    left_x = segment_max
                    right_x = segment_min

            left_distance = abs(
                left_x
                - float(previous_left)
            )

            right_distance = abs(
                right_x
                - float(previous_right)
            )

            if (
                left_distance
                <= max_tracking_jump
            ):

                previous_left = left_x

                tracked_left.append(
                    (
                        y,
                        left_x,
                    )
                )

            if (
                right_distance
                <= max_tracking_jump
            ):

                previous_right = right_x

                tracked_right.append(
                    (
                        y,
                        right_x,
                    )
                )

        # --------------------------------------------------------------------
        # Conversão para LanePoint.
        #
        # Y é convertido para o frame original.
        # --------------------------------------------------------------------

        for y, x in tracked_left:

            left_points.append(
                self._make_point(
                    x,
                    y,
                    mask_width,
                    mask_height,
                    frame_width,
                    frame_height,
                )
            )

        for y, x in tracked_right:

            right_points.append(
                self._make_point(
                    x,
                    y,
                    mask_width,
                    mask_height,
                    frame_width,
                    frame_height,
                )
            )

        # A geometria posterior trabalha
        # com Y crescente.
        left_points.sort(
            key=lambda point: point.y
        )

        right_points.sort(
            key=lambda point: point.y
        )

        return (
            left_points,
            right_points,
        )

    @staticmethod
    def _make_point(
        x: float,
        y: float,
        mask_width: int,
        mask_height: int,
        frame_width: int,
        frame_height: int,
    ) -> LanePoint:

        px = (
            x
            * frame_width
            / float(mask_width)
        )

        py = (
            y
            * frame_height
            / float(mask_height)
        )

        return LanePoint(
            x=float(px),
            y=float(py),
            confidence=1.0,
            valid=True,
        )

    # ========================================================================
    # RESULTADO
    # ========================================================================

    @staticmethod
    def _lane_confidence(
        lane: List[LanePoint],
    ) -> float:

        valid = [
            point
            for point in lane
            if point.valid
            and np.isfinite(
                point.confidence
            )
        ]

        if not valid:
            return 0.0

        return float(
            np.mean(
                [
                    point.confidence
                    for point in valid
                ]
            )
        )

    def _build_result(
        self,
        left_lane: List[LanePoint],
        right_lane: List[LanePoint],
        frame_width: int,
        frame_height: int,
    ) -> LaneDetectionResult:

        lanes = []

        if (
            len(left_lane)
            >= self.min_points_per_lane
        ):
            lanes.append(left_lane)

        else:
            left_lane = []

        if (
            len(right_lane)
            >= self.min_points_per_lane
        ):
            lanes.append(right_lane)

        else:
            right_lane = []

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

        valid = (
            bool(left_lane)
            and bool(right_lane)
        )

        current_lane_index = (
            0
            if valid
            else None
        )

        return LaneDetectionResult(
            lanes=lanes,
            lane_confidences=[
                left_confidence,
                right_confidence,
            ],
            current_lane_index=(
                current_lane_index
            ),
            left_lane=left_lane,
            right_lane=right_lane,
            additional_lanes=[],
            left_confidence=left_confidence,
            right_confidence=right_confidence,
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

        self.last_error = None

        try:

            if frame is None:

                raise ValueError(
                    "Frame é None."
                )

            frame_height, frame_width = (
                frame.shape[:2]
            )

            lane_line_seg = self.infer(
                frame
            )

            mask = (
                self.create_lane_mask(
                    lane_line_seg
                )
            )

            left_lane, right_lane = (
                self._extract_lane_points(
                    mask,
                    frame_width,
                    frame_height,
                )
            )

            result = self._build_result(
                left_lane,
                right_lane,
                frame_width,
                frame_height,
            )

            self.last_result = result

            return result

        except Exception as exc:

            self.last_error = (
                f"{type(exc).__name__}: {exc}"
            )

            logger.exception(
                "[YOLOP] Falha na detecção."
            )

            result = LaneDetectionResult(
                input_width=(
                    frame.shape[1]
                    if frame is not None
                    else self.input_width
                ),
                input_height=(
                    frame.shape[0]
                    if frame is not None
                    else self.input_height
                ),
                model_output_shape=(
                    self.last_output_shape
                ),
                valid=False,
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

    return YOLOPLaneDetector(
        model_path=model_path,
        **kwargs,
    )


__all__ = [
    "LanePoint",
    "LaneDetectionResult",
    "YOLOPLaneDetector",
    "create_default_detector",
]

