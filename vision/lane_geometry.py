"""
vision/lane_geometry.py

Forza Assistents
================

Núcleo geométrico da percepção de faixas.

Responsabilidade
----------------

Transformar observações de lanes em uma representação geométrica
determinística, robusta e independente do detector.

Contrato:

    LaneDetectionResult
            │
            ▼
      LaneGeometry
            │
            ▼
    LaneGeometryResult
            │
            ├── LaneModel
            ├── LaneAssignment
            └── ADAS

Este módulo NÃO:

    - executa inferência;
    - conhece YOLOP;
    - conhece UFLD;
    - importa OpenCV;
    - importa ONNX Runtime;
    - captura tela;
    - mantém estado temporal;
    - faz tracking;
    - extrapola lanes;
    - prediz trajetória;
    - controla o volante;
    - decide o estado do ADAS.

Princípio
---------

LaneGeometry representa exclusivamente aquilo que pode ser inferido
da geometria OBSERVADA no frame atual.

Não existe memória temporal neste módulo.

A ausência de observação nunca é convertida em observação artificial.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from config import LANE_GEOMETRY, ROI, YOLOP
from vision.detection_types import LaneDetectionResult, LanePoint


# =============================================================================
# TIPOS
# =============================================================================

Point = Tuple[float, float]


# =============================================================================
# RESULTADO
# =============================================================================

@dataclass(frozen=True, slots=True)
class LaneGeometryResult:
    """
    Representação geométrica observada de um frame.

    Todas as coordenadas são expressas no sistema de coordenadas da
    imagem/tela definido pela chamada ao process().

    Nenhum campo numérico deve conter NaN ou infinito.

    Sign convention
    ---------------

    lateral_error:

        < 0  -> centro da faixa à esquerda do centro de referência
        > 0  -> centro da faixa à direita do centro de referência

    heading_error:

        < 0  -> orientação observada para a esquerda
        > 0  -> orientação observada para a direita

    curvature:

        < 0  -> curvatura observada para a esquerda
        > 0  -> curvatura observada para a direita
    """

    lane_center_x: float
    lane_center_y: float

    image_center_x: float
    image_center_y: float

    lateral_error: float
    heading_error: float

    lane_width: float
    curvature: float

    center_line: list[Point]

    valid: bool

    left_lane_screen: list[Point]
    right_lane_screen: list[Point]

    additional_lanes_screen: list[list[Point]]

    selected_left_index: int
    selected_right_index: int

    geometry_confidence: float = 0.0

    observed_y_min: float = 0.0
    observed_y_max: float = 0.0
    observed_span: float = 0.0

    enough_for_projection: bool = False


# =============================================================================
# CLASSE PRINCIPAL
# =============================================================================

class LaneGeometry:
    """
    Calculador puramente geométrico das lanes observadas.

    Propriedades:

        - stateless;
        - determinístico;
        - independente do detector;
        - independente de OpenCV;
        - independente de ONNX;
        - sem extrapolação;
        - rejeição robusta de outliers;
        - seleção de pares;
        - estimativa robusta de centro;
        - estimativa de largura;
        - heading observado;
        - curvatura observada;
        - confiança geométrica;
        - tratamento explícito de falhas.

    O objeto pode ser reutilizado indefinidamente porque não guarda
    qualquer informação entre frames.
    """

    # -------------------------------------------------------------------------
    # Construção
    # -------------------------------------------------------------------------

    def __init__(
        self,
        screen_width: Optional[float] = None,
        screen_height: Optional[float] = None,
        roi: Optional[Tuple[float, float, float, float]] = None,
        **_legacy_kwargs: object,
    ) -> None:
        """
        Cria o calculador geométrico.

        Parameters
        ----------
        screen_width:
            Largura da imagem/tela de referência.

        screen_height:
            Altura da imagem/tela de referência.

        roi:
            ROI opcional no formato:

                (left, top, right, bottom)

            Quando omitido, utiliza config.ROI.

        Observação
        ----------

        Argumentos antigos desconhecidos são aceitos e ignorados
        propositalmente para evitar que parâmetros históricos do detector
        contaminem a arquitetura geométrica.

        Em particular, parâmetros como:

            ufld_width
            ufld_height

        não possuem qualquer significado neste módulo.
        """

        self._screen_width = (
            self._positive_or_none(screen_width)
        )

        self._screen_height = (
            self._positive_or_none(screen_height)
        )

        self._custom_roi = self._normalize_roi(roi)

        self._validate_configuration()

    # =========================================================================
    # CONFIGURAÇÃO
    # =========================================================================

    @staticmethod
    def _positive_or_none(
        value: Optional[float],
    ) -> Optional[float]:
        if value is None:
            return None

        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                "Dimensão deve ser numérica."
            )

        if not math.isfinite(value) or value <= 0:
            raise ValueError(
                "Dimensão deve ser finita e maior que zero."
            )

        return value

    @classmethod
    def _normalize_roi(
        cls,
        roi: Optional[Tuple[float, float, float, float]],
    ) -> Optional[Tuple[float, float, float, float]]:
        if roi is None:
            return None

        if len(roi) != 4:
            raise ValueError(
                "ROI deve possuir quatro valores: "
                "(left, top, right, bottom)."
            )

        left, top, right, bottom = (
            cls._safe_float(v, math.nan)
            for v in roi
        )

        if not all(
            math.isfinite(v)
            for v in (
                left,
                top,
                right,
                bottom,
            )
        ):
            raise ValueError(
                "ROI contém valores inválidos."
            )

        if right <= left:
            raise ValueError(
                "ROI.right deve ser maior que ROI.left."
            )

        if bottom <= top:
            raise ValueError(
                "ROI.bottom deve ser maior que ROI.top."
            )

        return (
            left,
            top,
            right,
            bottom,
        )

    @staticmethod
    def _validate_configuration() -> None:
        """
        Valida apenas invariantes geométricos.

        Nenhum backend de inferência é carregado.
        """

        if hasattr(ROI, "validate"):
            ROI.validate()

        input_width = int(
            getattr(
                YOLOP,
                "input_width",
                640,
            )
        )

        input_height = int(
            getattr(
                YOLOP,
                "input_height",
                640,
            )
        )

        if input_width <= 0:
            raise ValueError(
                "YOLOP.input_width deve ser > 0."
            )

        if input_height <= 0:
            raise ValueError(
                "YOLOP.input_height deve ser > 0."
            )

        min_points = int(
            getattr(
                LANE_GEOMETRY,
                "min_points",
                4,
            )
        )

        if min_points < 2:
            raise ValueError(
                "LANE_GEOMETRY.min_points deve ser >= 2."
            )

        min_span = float(
            getattr(
                LANE_GEOMETRY,
                "min_observed_span",
                20.0,
            )
        )

        if not math.isfinite(min_span) or min_span <= 0:
            raise ValueError(
                "LANE_GEOMETRY.min_observed_span deve ser > 0."
            )

        min_width = float(
            getattr(
                LANE_GEOMETRY,
                "min_lane_width",
                40.0,
            )
        )

        max_width = float(
            getattr(
                LANE_GEOMETRY,
                "max_lane_width",
                1200.0,
            )
        )

        if min_width <= 0:
            raise ValueError(
                "LANE_GEOMETRY.min_lane_width deve ser > 0."
            )

        if max_width <= min_width:
            raise ValueError(
                "LANE_GEOMETRY.max_lane_width deve ser "
                "maior que min_lane_width."
            )

    # =========================================================================
    # UTILITÁRIOS NUMÉRICOS
    # =========================================================================

    @staticmethod
    def _safe_float(
        value: object,
        default: float = 0.0,
    ) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return default

        if not math.isfinite(result):
            return default

        return result

    @staticmethod
    def _finite(value: object) -> bool:
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _clip(
        value: float,
        lower: float,
        upper: float,
    ) -> float:
        if not math.isfinite(value):
            return lower

        return float(
            min(
                upper,
                max(
                    lower,
                    value,
                ),
            )
        )

    @staticmethod
    def _clip01(value: float) -> float:
        return LaneGeometry._clip(
            value,
            0.0,
            1.0,
        )

    # =========================================================================
    # DIMENSÕES
    # =========================================================================

    def _roi_bounds(
        self,
    ) -> Tuple[float, float, float, float]:
        """
        Retorna:

            left
            top
            right
            bottom
        """

        if self._custom_roi is not None:
            return self._custom_roi

        left = self._safe_float(
            getattr(ROI, "left", 0),
        )

        top = self._safe_float(
            getattr(ROI, "top", 0),
        )

        right = self._safe_float(
            getattr(ROI, "right", 1),
        )

        bottom = self._safe_float(
            getattr(ROI, "bottom", 1),
        )

        if right <= left:
            right = left + 1.0

        if bottom <= top:
            bottom = top + 1.0

        return (
            left,
            top,
            right,
            bottom,
        )

    def _roi_dimensions(
        self,
    ) -> Tuple[float, float, float, float]:
        left, top, right, bottom = self._roi_bounds()

        return (
            left,
            top,
            right - left,
            bottom - top,
        )

    @staticmethod
    def _detector_dimensions() -> Tuple[float, float]:
        return (
            float(
                getattr(
                    YOLOP,
                    "input_width",
                    640,
                )
            ),
            float(
                getattr(
                    YOLOP,
                    "input_height",
                    640,
                )
            ),
        )

    # =========================================================================
    # CONVERSÃO DE COORDENADAS
    # =========================================================================

    def detector_to_screen(
        self,
        x: float,
        y: float,
    ) -> Point:
        """
        Converte coordenadas do detector para coordenadas da imagem.

        O método representa somente uma transformação afim.

        Não há:
            - correção temporal;
            - perspectiva inventada;
            - extrapolação;
            - fitting.
        """

        detector_width, detector_height = (
            self._detector_dimensions()
        )

        roi_left, roi_top, roi_width, roi_height = (
            self._roi_dimensions()
        )

        x = self._safe_float(x)
        y = self._safe_float(y)

        screen_x = (
            roi_left
            + (
                x / detector_width
            )
            * roi_width
        )

        screen_y = (
            roi_top
            + (
                y / detector_height
            )
            * roi_height
        )

        return (
            float(screen_x),
            float(screen_y),
        )

    # =========================================================================
    # NORMALIZAÇÃO DE DETECÇÃO
    # =========================================================================

    @staticmethod
    def _extract_lanes(
        detection: object,
    ) -> Sequence[Sequence[LanePoint]]:
        """
        Extrai lanes de um LaneDetectionResult.

        Também aceita diretamente:

            Sequence[Sequence[LanePoint]]

        Isso mantém LaneGeometry independente da implementação do detector.
        """

        if isinstance(
            detection,
            LaneDetectionResult,
        ):
            if not detection.valid:
                return ()

            return detection.lanes

        lanes = getattr(
            detection,
            "lanes",
            None,
        )

        if lanes is not None:
            return lanes

        if isinstance(
            detection,
            Sequence,
        ):
            return detection

        return ()

    @staticmethod
    def _extract_detection_confidence(
        detection: object,
    ) -> float:
        return LaneGeometry._clip01(
            LaneGeometry._safe_float(
                getattr(
                    detection,
                    "confidence",
                    1.0,
                ),
                1.0,
            )
        )

    # =========================================================================
    # CONVERSÃO DE LANE
    # =========================================================================

    def convert_lane(
        self,
        lane: Sequence[LanePoint],
    ) -> list[Point]:
        """
        Converte uma lane para coordenadas da imagem.

        Pontos inválidos ou fora do domínio do detector são descartados.

        Nenhum ponto novo é criado.
        """

        if not lane:
            return []

        detector_width, detector_height = (
            self._detector_dimensions()
        )

        min_confidence = float(
            getattr(
                LANE_GEOMETRY,
                "min_lane_confidence",
                0.35,
            )
        )

        candidates: list[Point] = []

        for point in lane:

            if not isinstance(
                point,
                LanePoint,
            ):
                continue

            if not bool(
                getattr(
                    point,
                    "valid",
                    True,
                )
            ):
                continue

            confidence = self._safe_float(
                getattr(
                    point,
                    "confidence",
                    1.0,
                ),
                1.0,
            )

            if confidence < min_confidence:
                continue

            x = self._safe_float(
                getattr(point, "x", math.nan),
                math.nan,
            )

            y = self._safe_float(
                getattr(point, "y", math.nan),
                math.nan,
            )

            if not (
                math.isfinite(x)
                and math.isfinite(y)
            ):
                continue

            if not (
                0.0 <= x <= detector_width
                and 0.0 <= y <= detector_height
            ):
                continue

            screen_point = self.detector_to_screen(
                x,
                y,
            )

            if all(
                self._finite(value)
                for value in screen_point
            ):
                candidates.append(
                    screen_point
                )

        candidates.sort(
            key=lambda point: point[1]
        )

        return self._collapse_duplicate_y(
            candidates
        )

    @staticmethod
    def _collapse_duplicate_y(
        points: Sequence[Point],
        tolerance: float = 1e-4,
    ) -> list[Point]:
        """
        Remove amostras duplicadas no eixo Y.

        Quando existem vários X para o mesmo Y, utiliza a mediana.
        """

        if not points:
            return []

        groups: list[list[Point]] = []

        for point in sorted(
            points,
            key=lambda p: p[1],
        ):
            if not groups:
                groups.append([point])
                continue

            if abs(
                point[1]
                - groups[-1][-1][1]
            ) <= tolerance:
                groups[-1].append(point)
            else:
                groups.append([point])

        result: list[Point] = []

        for group in groups:
            y = float(
                np.median(
                    [point[1] for point in group]
                )
            )

            x = float(
                np.median(
                    [point[0] for point in group]
                )
            )

            result.append(
                (
                    x,
                    y,
                )
            )

        return result

    # =========================================================================
    # REJEIÇÃO ROBUSTA DE OUTLIERS
    # =========================================================================

    def remove_outliers(
        self,
        points: Sequence[Point],
    ) -> list[Point]:
        """
        Remove observações incompatíveis com a forma local da lane.

        Estratégia:

            1. ajuste quadrático inicial;
            2. resíduos;
            3. MAD robusto;
            4. rejeição;
            5. fallback seguro.

        Importante:

            Esta função NÃO extrapola a lane.
            Apenas remove observações ruins.
        """

        points = list(points)

        if not bool(
            getattr(
                LANE_GEOMETRY,
                "enable_outlier_rejection",
                True,
            )
        ):
            return points

        minimum = int(
            getattr(
                LANE_GEOMETRY,
                "min_points",
                4,
            )
        )

        if len(points) < max(
            minimum + 2,
            6,
        ):
            return points

        array = np.asarray(
            points,
            dtype=np.float64,
        )

        if (
            array.ndim != 2
            or array.shape[1] != 2
            or not np.all(
                np.isfinite(array)
            )
        ):
            return points

        x = array[:, 0]
        y = array[:, 1]

        y_center = float(
            np.mean(y)
        )

        y_scale = float(
            np.std(y)
        )

        if y_scale < 1e-9:
            return points

        yn = (
            y - y_center
        ) / y_scale

        try:
            coefficients = np.polyfit(
                yn,
                x,
                2,
            )

            predicted = np.polyval(
                coefficients,
                yn,
            )

        except (
            np.linalg.LinAlgError,
            ValueError,
            FloatingPointError,
        ):
            return points

        residuals = np.abs(
            x - predicted
        )

        median = float(
            np.median(residuals)
        )

        mad = float(
            np.median(
                np.abs(
                    residuals
                    - median
                )
            )
        )

        sigma = float(
            getattr(
                LANE_GEOMETRY,
                "outlier_sigma",
                2.5,
            )
        )

        if mad < 1e-9:
            threshold = max(
                3.0,
                median * 3.0,
            )
        else:
            robust_scale = (
                1.4826 * mad
            )

            threshold = (
                median
                + sigma * robust_scale
            )

        keep = residuals <= threshold

        filtered = [
            point
            for point, valid
            in zip(
                points,
                keep,
            )
            if bool(valid)
        ]

        if len(filtered) < minimum:
            return points

        return filtered

    # =========================================================================
    # MÉTRICAS BÁSICAS
    # =========================================================================

    @staticmethod
    def lane_span(
        lane: Sequence[Point],
    ) -> float:
        if len(lane) < 2:
            return 0.0

        ys = np.asarray(
            [point[1] for point in lane],
            dtype=np.float64,
        )

        return float(
            np.max(ys)
            - np.min(ys)
        )

    @staticmethod
    def lane_mean_x(
        lane: Sequence[Point],
    ) -> float:
        if not lane:
            return 0.0

        return float(
            np.mean(
                [
                    point[0]
                    for point in lane
                ]
            )
        )

    @staticmethod
    def lane_median_x(
        lane: Sequence[Point],
    ) -> float:
        if not lane:
            return 0.0

        return float(
            np.median(
                [
                    point[0]
                    for point in lane
                ]
            )
        )

    @staticmethod
    def _y_range(
        lane: Sequence[Point],
    ) -> Optional[Tuple[float, float]]:
        if len(lane) < 2:
            return None

        ys = [
            point[1]
            for point in lane
        ]

        return (
            float(min(ys)),
            float(max(ys)),
        )

    @classmethod
    def common_y_range(
        cls,
        left: Sequence[Point],
        right: Sequence[Point],
    ) -> Optional[Tuple[float, float]]:

        left_range = cls._y_range(left)
        right_range = cls._y_range(right)

        if (
            left_range is None
            or right_range is None
        ):
            return None

        lower = max(
            left_range[0],
            right_range[0],
        )

        upper = min(
            left_range[1],
            right_range[1],
        )

        if upper <= lower:
            return None

        return (
            lower,
            upper,
        )

    # =========================================================================
    # INTERPOLAÇÃO
    # =========================================================================

    @staticmethod
    def interpolate_x(
        lane: Sequence[Point],
        y: float,
    ) -> Optional[float]:
        """
        Interpola X para um Y já observado.

        Extrapolação é explicitamente proibida.
        """

        if len(lane) < 2:
            return None

        ordered = sorted(
            lane,
            key=lambda p: p[1],
        )

        ys = np.asarray(
            [
                point[1]
                for point in ordered
            ],
            dtype=np.float64,
        )

        xs = np.asarray(
            [
                point[0]
                for point in ordered
            ],
            dtype=np.float64,
        )

        if not (
            np.all(np.isfinite(ys))
            and np.all(np.isfinite(xs))
        ):
            return None

        if (
            y < ys[0]
            or y > ys[-1]
        ):
            return None

        return float(
            np.interp(
                y,
                ys,
                xs,
            )
        )

    # =========================================================================
    # AMOSTRAGEM
    # =========================================================================

    @staticmethod
    def _sample_y(
        lower: float,
        upper: float,
        count: int,
    ) -> np.ndarray:

        count = max(
            2,
            int(count),
        )

        return np.linspace(
            lower,
            upper,
            count,
            dtype=np.float64,
        )

    # =========================================================================
    # CENTRO DA FAIXA
    # =========================================================================

    def build_center_line(
        self,
        left: Sequence[Point],
        right: Sequence[Point],
    ) -> list[Point]:
        """
        Constrói o centro da faixa exclusivamente na região observada
        simultaneamente pelas duas bordas.
        """

        common = self.common_y_range(
            left,
            right,
        )

        if common is None:
            return []

        lower, upper = common

        sample_count = int(
            getattr(
                LANE_GEOMETRY,
                "center_samples",
                24,
            )
        )

        center: list[Point] = []

        for y in self._sample_y(
            lower,
            upper,
            sample_count,
        ):

            left_x = self.interpolate_x(
                left,
                float(y),
            )

            right_x = self.interpolate_x(
                right,
                float(y),
            )

            if (
                left_x is None
                or right_x is None
            ):
                continue

            if right_x <= left_x:
                continue

            center.append(
                (
                    float(
                        0.5
                        * (
                            left_x
                            + right_x
                        )
                    ),
                    float(y),
                )
            )

        return center

    # =========================================================================
    # LARGURA
    # =========================================================================

    def calculate_lane_width(
        self,
        left: Sequence[Point],
        right: Sequence[Point],
    ) -> float:
        """
        Calcula a largura robusta da faixa.

        A mediana é utilizada para reduzir influência de outliers.
        """

        common = self.common_y_range(
            left,
            right,
        )

        if common is None:
            return 0.0

        lower, upper = common

        widths: list[float] = []

        for y in self._sample_y(
            lower,
            upper,
            15,
        ):

            left_x = self.interpolate_x(
                left,
                float(y),
            )

            right_x = self.interpolate_x(
                right,
                float(y),
            )

            if (
                left_x is None
                or right_x is None
            ):
                continue

            width = (
                right_x
                - left_x
            )

            if width > 0:
                widths.append(
                    float(width)
                )

        if not widths:
            return 0.0

        return float(
            np.median(
                np.asarray(
                    widths,
                    dtype=np.float64,
                )
            )
        )

    # =========================================================================
    # ESTABILIDADE DE LARGURA
    # =========================================================================

    def _width_statistics(
        self,
        left: Sequence[Point],
        right: Sequence[Point],
    ) -> Tuple[float, float]:
        common = self.common_y_range(
            left,
            right,
        )

        if common is None:
            return (
                0.0,
                0.0,
            )

        lower, upper = common

        widths: list[float] = []

        for y in self._sample_y(
            lower,
            upper,
            11,
        ):

            lx = self.interpolate_x(
                left,
                float(y),
            )

            rx = self.interpolate_x(
                right,
                float(y),
            )

            if (
                lx is None
                or rx is None
            ):
                continue

            width = rx - lx

            if width > 0:
                widths.append(
                    width
                )

        if len(widths) < 2:
            return (
                0.0,
                0.0,
            )

        values = np.asarray(
            widths,
            dtype=np.float64,
        )

        median = float(
            np.median(values)
        )

        mad = float(
            np.median(
                np.abs(
                    values - median
                )
            )
        )

        if median <= 1e-9:
            return (
                0.0,
                0.0,
            )

        relative_mad = (
            1.4826 * mad
            / median
        )

        stability = 1.0 / (
            1.0
            + relative_mad
        )

        return (
            median,
            self._clip01(
                stability
            ),
        )

    # =========================================================================
    # SELEÇÃO DE PAR
    # =========================================================================

    def _pair_score(
        self,
        left: Sequence[Point],
        right: Sequence[Point],
    ) -> float:
        """
        Pontuação geométrica de um possível par.

        O score considera:

            - ordem esquerda/direita;
            - sobreposição vertical;
            - extensão observada;
            - largura plausível;
            - estabilidade da largura;
            - consistência da largura;
            - convergência/perspectiva.
        """

        if (
            len(left) < 2
            or len(right) < 2
        ):
            return -math.inf

        if (
            self.lane_mean_x(left)
            >= self.lane_mean_x(right)
        ):
            return -math.inf

        common = self.common_y_range(
            left,
            right,
        )

        if common is None:
            return -math.inf

        lower, upper = common

        span = upper - lower

        min_span = float(
            getattr(
                LANE_GEOMETRY,
                "min_observed_span",
                20.0,
            )
        )

        if span < min_span:
            return -math.inf

        width, width_stability = (
            self._width_statistics(
                left,
                right,
            )
        )

        if width <= 0:
            return -math.inf

        min_width = float(
            getattr(
                LANE_GEOMETRY,
                "min_lane_width",
                40.0,
            )
        )

        max_width = float(
            getattr(
                LANE_GEOMETRY,
                "max_lane_width",
                1200.0,
            )
        )

        if not (
            min_width
            <= width
            <= max_width
        ):
            return -math.inf

        span_score = self._clip01(
            span
            / max(
                min_span * 5.0,
                1.0,
            )
        )

        # Compatibilidade com largura esperada, quando disponível.
        expected_width = self._safe_float(
            getattr(
                LANE_GEOMETRY,
                "expected_lane_width",
                0.0,
            ),
            0.0,
        )

        if expected_width > 0:
            tolerance = self._safe_float(
                getattr(
                    LANE_GEOMETRY,
                    "lane_width_tolerance",
                    0.50,
                ),
                0.50,
            )

            relative_error = abs(
                width - expected_width
            ) / expected_width

            width_prior = self._clip01(
                1.0
                - relative_error
                / max(
                    tolerance,
                    1e-6,
                )
            )
        else:
            width_prior = 1.0

        # Uma faixa real pode apresentar perspectiva.
        # O que interessa é evitar largura absurdamente instável.
        perspective_score = (
            width_stability
        )

        return self._clip01(
            0.25 * span_score
            + 0.25 * width_stability
            + 0.25 * width_prior
            + 0.25 * perspective_score
        )

    def select_best_pair(
        self,
        lanes: Sequence[Sequence[Point]],
    ) -> Tuple[int, int, float]:
        """
        Seleciona o par geometricamente mais plausível.

        Retorno:

            (left_index, right_index, score)

        Sem par válido:

            (-1, -1, 0.0)
        """

        if len(lanes) < 2:
            return (
                -1,
                -1,
                0.0,
            )

        best_left = -1
        best_right = -1
        best_score = -math.inf

        for left_index in range(
            len(lanes)
        ):

            for right_index in range(
                len(lanes)
            ):

                if left_index == right_index:
                    continue

                left = lanes[left_index]
                right = lanes[right_index]

                score = self._pair_score(
                    left,
                    right,
                )

                if score > best_score:
                    best_score = score
                    best_left = left_index
                    best_right = right_index

        if (
            best_left < 0
            or best_right < 0
            or not math.isfinite(
                best_score
            )
        ):
            return (
                -1,
                -1,
                0.0,
            )

        return (
            best_left,
            best_right,
            self._clip01(
                best_score
            ),
        )

    # =========================================================================
    # CENTRO NA REGIÃO DE REFERÊNCIA
    # =========================================================================

    @staticmethod
    def _center_x_at_reference(
        center_line: Sequence[Point],
    ) -> float:
        """
        Retorna o centro observado próximo à região inferior.

        Isso é mais relevante para a posição lateral do veículo do que
        uma média global da linha central.
        """

        if not center_line:
            return 0.0

        ordered = sorted(
            center_line,
            key=lambda point: point[1],
        )

        count = max(
            2,
            len(ordered) // 4,
        )

        reference = ordered[-count:]

        return float(
            np.median(
                [
                    point[0]
                    for point in reference
                ]
            )
        )

    @staticmethod
    def _center_y_at_reference(
        center_line: Sequence[Point],
    ) -> float:
        if not center_line:
            return 0.0

        ordered = sorted(
            center_line,
            key=lambda point: point[1],
        )

        count = max(
            2,
            len(ordered) // 4,
        )

        return float(
            np.median(
                [
                    point[1]
                    for point in ordered[-count:]
                ]
            )
        )

    # =========================================================================
    # ERRO LATERAL
    # =========================================================================

    def calculate_lateral_error(
        self,
        lane_center_x: float,
        image_center_x: float,
        reference_width: Optional[float] = None,
    ) -> float:
        """
        Erro lateral normalizado em [-1, 1].

        O denominador é metade da largura da região de referência.

        Isso torna o valor independente da resolução absoluta.
        """

        if reference_width is None:

            if self._screen_width is not None:
                reference_width = (
                    self._screen_width
                )
            else:
                _, _, roi_width, _ = (
                    self._roi_dimensions()
                )
                reference_width = roi_width

        reference_width = self._safe_float(
            reference_width,
            1.0,
        )

        if reference_width <= 0:
            return 0.0

        error = (
            lane_center_x
            - image_center_x
        ) / (
            0.5 * reference_width
        )

        return self._clip(
            error,
            -1.0,
            1.0,
        )

    # =========================================================================
    # HEADING
    # =========================================================================

    @staticmethod
    def calculate_heading(
        center_line: Sequence[Point],
    ) -> float:
        """
        Estima a orientação observada da faixa.

        A estimativa utiliza regressão linear robusta em X(Y), evitando
        que dois pontos isolados dominem o resultado.

        Não há extrapolação.
        """

        if len(center_line) < 3:
            return 0.0

        array = np.asarray(
            center_line,
            dtype=np.float64,
        )

        if (
            array.ndim != 2
            or array.shape[1] != 2
            or not np.all(
                np.isfinite(array)
            )
        ):
            return 0.0

        x = array[:, 0]
        y = array[:, 1]

        y_center = float(
            np.mean(y)
        )

        y_scale = float(
            np.std(y)
        )

        if y_scale < 1e-9:
            return 0.0

        yn = (
            y - y_center
        ) / y_scale

        try:
            coefficients = np.polyfit(
                yn,
                x,
                1,
            )
        except (
            np.linalg.LinAlgError,
            ValueError,
            FloatingPointError,
        ):
            return 0.0

        slope = float(
            coefficients[0]
        )

        if not math.isfinite(slope):
            return 0.0

        # Normalização por escala geométrica.
        #
        # O resultado é deliberadamente limitado. O módulo downstream
        # não precisa lidar com valores arbitrariamente grandes.
        normalized = slope / max(
            y_scale,
            1.0,
        )

        # Em coordenadas de imagem, X/Y são pixels.
        # A escala abaixo produz um valor adimensional útil.
        normalized = math.atan(
            normalized
        ) / (
            math.pi / 2.0
        )

        return LaneGeometry._clip(
            normalized,
            -1.0,
            1.0,
        )

    # =========================================================================
    # CURVATURA
    # =========================================================================

    @staticmethod
    def calculate_curvature(
        center_line: Sequence[Point],
    ) -> float:
        """
        Estima curvatura observada através de X(Y).

        O ajuste é quadrático porque a curvatura local observada é uma
        propriedade geométrica de segunda ordem.

        A função:

            - não extrapola;
            - não mantém histórico;
            - não prediz trajetória.

        O resultado é adimensional e limitado.
        """

        if len(center_line) < 5:
            return 0.0

        array = np.asarray(
            center_line,
            dtype=np.float64,
        )

        if (
            array.ndim != 2
            or array.shape[1] != 2
            or not np.all(
                np.isfinite(array)
            )
        ):
            return 0.0

        x = array[:, 0]
        y = array[:, 1]

        y_center = float(
            np.mean(y)
        )

        y_scale = float(
            np.std(y)
        )

        if y_scale < 1e-9:
            return 0.0

        yn = (
            y - y_center
        ) / y_scale

        try:
            coefficients = np.polyfit(
                yn,
                x,
                2,
            )
        except (
            np.linalg.LinAlgError,
            ValueError,
            FloatingPointError,
        ):
            return 0.0

        second_derivative = float(
            2.0
            * coefficients[0]
        )

        if not math.isfinite(
            second_derivative
        ):
            return 0.0

        return LaneGeometry._clip(
            second_derivative / 100.0,
            -1.0,
            1.0,
        )

    # =========================================================================
    # CONFIANÇA
    # =========================================================================

    def calculate_confidence(
        self,
        left: Sequence[Point],
        right: Sequence[Point],
        center_line: Sequence[Point],
        pair_score: float,
        detection_confidence: float = 1.0,
    ) -> float:
        """
        Calcula confiança exclusivamente a partir da observação atual.

        Componentes:

            - confiança do detector;
            - densidade de pontos;
            - extensão vertical;
            - estabilidade da largura;
            - qualidade do par;
            - existência de centro consistente.
        """

        if (
            not left
            or not right
            or not center_line
        ):
            return 0.0

        minimum_points = float(
            getattr(
                LANE_GEOMETRY,
                "min_points",
                4,
            )
        )

        point_count = min(
            len(left),
            len(right),
        )

        point_score = self._clip01(
            point_count
            / max(
                minimum_points * 2.0,
                1.0,
            )
        )

        span = min(
            self.lane_span(left),
            self.lane_span(right),
        )

        min_span = float(
            getattr(
                LANE_GEOMETRY,
                "min_observed_span",
                20.0,
            )
        )

        span_score = self._clip01(
            span
            / max(
                min_span * 4.0,
                1.0,
            )
        )

        _, width_stability = (
            self._width_statistics(
                left,
                right,
            )
        )

        center_score = self._clip01(
            len(center_line)
            / 16.0
        )

        pair_score = self._clip01(
            pair_score
        )

        detection_confidence = (
            self._clip01(
                detection_confidence
            )
        )

        return self._clip01(
            0.20 * detection_confidence
            + 0.15 * point_score
            + 0.20 * span_score
            + 0.20 * width_stability
            + 0.15 * center_score
            + 0.10 * pair_score
        )

    # =========================================================================
    # PROCESSAMENTO
    # =========================================================================

    def process(
        self,
        detection: object,
        image_width: Optional[float] = None,
        image_height: Optional[float] = None,
    ) -> LaneGeometryResult:
        """
        Processa uma observação de lanes.

        Entrada aceita:

            LaneDetectionResult

        ou:

            Sequence[Sequence[LanePoint]]

        O resultado é totalmente novo a cada chamada.
        """

        # ---------------------------------------------------------------------
        # Dimensões da imagem
        # ---------------------------------------------------------------------

        if image_width is None:
            image_width = self._screen_width

        if image_height is None:
            image_height = self._screen_height

        if image_width is None:
            _, _, roi_width, _ = (
                self._roi_dimensions()
            )
            image_width = roi_width

        if image_height is None:
            _, _, _, roi_height = (
                self._roi_dimensions()
            )
            image_height = roi_height

        image_width = max(
            1.0,
            self._safe_float(
                image_width,
                1.0,
            ),
        )

        image_height = max(
            1.0,
            self._safe_float(
                image_height,
                1.0,
            ),
        )

        image_center_x = (
            image_width * 0.5
        )

        image_center_y = (
            image_height * 0.5
        )

        # ---------------------------------------------------------------------
        # Entrada
        # ---------------------------------------------------------------------

        lanes = self._extract_lanes(
            detection
        )

        detection_confidence = (
            self._extract_detection_confidence(
                detection
            )
        )

        if not lanes:
            return self._invalid_result(
                image_center_x,
                image_center_y,
            )

        # ---------------------------------------------------------------------
        # Conversão
        # ---------------------------------------------------------------------

        converted: list[list[Point]] = []

        minimum = int(
            getattr(
                LANE_GEOMETRY,
                "min_points",
                4,
            )
        )

        for lane in lanes:

            points = self.convert_lane(
                lane
            )

            if len(points) < minimum:
                continue

            points = self.remove_outliers(
                points
            )

            if len(points) < minimum:
                continue

            converted.append(
                points
            )

        if len(converted) < 2:
            return self._invalid_result(
                image_center_x,
                image_center_y,
                converted,
            )

        # ---------------------------------------------------------------------
        # Seleção
        # ---------------------------------------------------------------------

        (
            left_index,
            right_index,
            pair_score,
        ) = self.select_best_pair(
            converted
        )

        if (
            left_index < 0
            or right_index < 0
        ):
            return self._invalid_result(
                image_center_x,
                image_center_y,
                converted,
            )

        left = converted[
            left_index
        ]

        right = converted[
            right_index
        ]

        # ---------------------------------------------------------------------
        # Centro observado
        # ---------------------------------------------------------------------

        center_line = self.build_center_line(
            left,
            right,
        )

        if len(center_line) < 3:
            return self._invalid_result(
                image_center_x,
                image_center_y,
                converted,
                left_index,
                right_index,
            )

        # ---------------------------------------------------------------------
        # Métricas
        # ---------------------------------------------------------------------

        lane_center_x = (
            self._center_x_at_reference(
                center_line
            )
        )

        lane_center_y = (
            self._center_y_at_reference(
                center_line
            )
        )

        lateral_error = (
            self.calculate_lateral_error(
                lane_center_x,
                image_center_x,
                image_width,
            )
        )

        heading_error = (
            self.calculate_heading(
                center_line
            )
        )

        curvature = (
            self.calculate_curvature(
                center_line
            )
        )

        lane_width = (
            self.calculate_lane_width(
                left,
                right,
            )
        )

        geometry_confidence = (
            self.calculate_confidence(
                left=left,
                right=right,
                center_line=center_line,
                pair_score=pair_score,
                detection_confidence=detection_confidence,
            )
        )

        # ---------------------------------------------------------------------
        # Extensão observada
        # ---------------------------------------------------------------------

        observed_y_min = min(
            min(
                point[1]
                for point in left
            ),
            min(
                point[1]
                for point in right
            ),
        )

        observed_y_max = max(
            max(
                point[1]
                for point in left
            ),
            max(
                point[1]
                for point in right
            ),
        )

        observed_span = (
            observed_y_max
            - observed_y_min
        )

        min_span = float(
            getattr(
                LANE_GEOMETRY,
                "min_observed_span",
                20.0,
            )
        )

        enough_for_projection = (
            observed_span >= min_span
            and len(center_line) >= 3
        )

        # ---------------------------------------------------------------------
        # Validade
        # ---------------------------------------------------------------------

        min_width = float(
            getattr(
                LANE_GEOMETRY,
                "min_lane_width",
                40.0,
            )
        )

        max_width = float(
            getattr(
                LANE_GEOMETRY,
                "max_lane_width",
                1200.0,
            )
        )

        valid = bool(
            lane_width >= min_width
            and lane_width <= max_width
            and observed_span >= min_span
            and len(center_line) >= 3
            and geometry_confidence > 0.0
        )

        # ---------------------------------------------------------------------
        # Lanes adicionais
        # ---------------------------------------------------------------------

        additional = [
            list(lane)
            for index, lane
            in enumerate(converted)
            if index not in {
                left_index,
                right_index,
            }
        ]

        return LaneGeometryResult(
            lane_center_x=float(
                lane_center_x
            ),
            lane_center_y=float(
                lane_center_y
            ),
            image_center_x=float(
                image_center_x
            ),
            image_center_y=float(
                image_center_y
            ),
            lateral_error=float(
                lateral_error
            ),
            heading_error=float(
                heading_error
            ),
            lane_width=float(
                lane_width
            ),
            curvature=float(
                curvature
            ),
            center_line=list(
                center_line
            ),
            valid=valid,
            left_lane_screen=list(
                left
            ),
            right_lane_screen=list(
                right
            ),
            additional_lanes_screen=additional,
            selected_left_index=int(
                left_index
            ),
            selected_right_index=int(
                right_index
            ),
            geometry_confidence=float(
                geometry_confidence
            ),
            observed_y_min=float(
                observed_y_min
            ),
            observed_y_max=float(
                observed_y_max
            ),
            observed_span=float(
                observed_span
            ),
            enough_for_projection=bool(
                enough_for_projection
            ),
        )

    # =========================================================================
    # ALIASES
    # =========================================================================

    def compute(
        self,
        detection: object,
        image_width: Optional[float] = None,
        image_height: Optional[float] = None,
    ) -> LaneGeometryResult:
        """
        Alias compatível para o pipeline existente.
        """

        return self.process(
            detection=detection,
            image_width=image_width,
            image_height=image_height,
        )

    def calculate(
        self,
        detection: object,
        image_width: Optional[float] = None,
        image_height: Optional[float] = None,
    ) -> LaneGeometryResult:
        """
        Alias semântico para process().
        """

        return self.process(
            detection=detection,
            image_width=image_width,
            image_height=image_height,
        )

    # =========================================================================
    # FALHA SEGURA
    # =========================================================================

    @staticmethod
    def _invalid_result(
        image_center_x: float,
        image_center_y: float,
        lanes: Optional[
            Sequence[Sequence[Point]]
        ] = None,
        left_index: int = -1,
        right_index: int = -1,
    ) -> LaneGeometryResult:
        """
        Resultado seguro para observação insuficiente.

        Não são produzidos NaN, extrapolações ou estimativas artificiais.
        """

        lanes = (
            []
            if lanes is None
            else lanes
        )

        left_lane: list[Point] = []
        right_lane: list[Point] = []

        if (
            0 <= left_index < len(lanes)
        ):
            left_lane = list(
                lanes[left_index]
            )

        if (
            0 <= right_index < len(lanes)
        ):
            right_lane = list(
                lanes[right_index]
            )

        additional = [
            list(lane)
            for index, lane
            in enumerate(lanes)
            if index not in {
                left_index,
                right_index,
            }
        ]

        return LaneGeometryResult(
            lane_center_x=float(
                image_center_x
            ),
            lane_center_y=float(
                image_center_y
            ),
            image_center_x=float(
                image_center_x
            ),
            image_center_y=float(
                image_center_y
            ),
            lateral_error=0.0,
            heading_error=0.0,
            lane_width=0.0,
            curvature=0.0,
            center_line=[],
            valid=False,
            left_lane_screen=left_lane,
            right_lane_screen=right_lane,
            additional_lanes_screen=additional,
            selected_left_index=int(
                left_index
            ),
            selected_right_index=int(
                right_index
            ),
            geometry_confidence=0.0,
            observed_y_min=0.0,
            observed_y_max=0.0,
            observed_span=0.0,
            enough_for_projection=False,
        )


# =============================================================================
# API FUNCIONAL
# =============================================================================

def compute_lane_geometry(
    detection: object,
    image_width: Optional[float] = None,
    image_height: Optional[float] = None,
) -> LaneGeometryResult:
    """
    API funcional do módulo.

    Nenhum estado é mantido entre chamadas.
    """

    return LaneGeometry().process(
        detection=detection,
        image_width=image_width,
        image_height=image_height,
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "Point",
    "LaneGeometryResult",
    "LaneGeometry",
    "compute_lane_geometry",
]