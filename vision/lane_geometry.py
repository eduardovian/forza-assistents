"""
vision/lane_geometry.py

Forza Assistents
================

Geometria das faixas detectadas pelo YOLOP.

Responsabilidade deste módulo
-----------------------------

Receber lanes observadas pelo pipeline YOLOP e transformar essas
observações em uma representação geométrica robusta para as etapas
seguintes do ADAS.

Pipeline:

    YOLOP
      ↓
    LaneDetectionResult
      ↓
    LaneGeometry
      ↓
    LaneGeometryResult
      ↓
    LaneModel / LaneAssignment / ADAS

Este módulo NÃO:

    - executa inferência;
    - captura tela;
    - realiza tracking temporal;
    - extrapola lanes;
    - prevê trajetória;
    - decide estado ADAS;
    - controla volante.

Princípio fundamental
---------------------

Tudo produzido aqui deve representar apenas geometria observada.

Não há qualquer dependência de UFLD.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from config import LANE_GEOMETRY, ROI, YOLOP
from .lane_types import LanePoint
from vision.detection_types import LanePoint, LaneDetectionResult

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
    Resultado geométrico observado de um frame.

    Todas as coordenadas estão em coordenadas de tela.

    Valores numéricos permanecem finitos mesmo quando a geometria
    não é válida.
    """

    lane_center_x: float
    lane_center_y: float

    image_center_x: float
    image_center_y: float

    lateral_error: float
    heading_error: float

    lane_width: float
    curvature: float

    center_line: List[Point]

    valid: bool

    left_lane_screen: List[Point]
    right_lane_screen: List[Point]

    additional_lanes_screen: List[List[Point]]

    selected_left_index: int
    selected_right_index: int

    geometry_confidence: float = 0.0

    observed_y_min: float = 0.0
    observed_y_max: float = 0.0
    observed_span: float = 0.0

    enough_for_projection: bool = False


# =============================================================================
# GEOMETRIA
# =============================================================================

class LaneGeometry:
    """
    Calcula geometria observada das lanes YOLOP.

    Características:

        - sem estado temporal;
        - sem extrapolação;
        - sem UFLD;
        - suporte a múltiplas lanes;
        - seleção robusta de pares;
        - rejeição de outliers;
        - cálculo de centro;
        - cálculo de largura;
        - erro lateral normalizado;
        - heading observado;
        - curvatura observada;
        - confiança geométrica;
        - falha segura.
    """

    def __init__(
        self,
        ufld_width: Optional[int] = None,
        ufld_height: Optional[int] = None,
    ) -> None:
        """
        Mantém os argumentos antigos apenas por compatibilidade
        temporária com testes/integrações antigas.

        Eles NÃO são utilizados.

        O sistema oficial utiliza exclusivamente YOLOP.
        """

        del ufld_width
        del ufld_height

        self._validate_configuration()

    # =========================================================================
    # CONFIGURAÇÃO
    # =========================================================================

    @staticmethod
    def _validate_configuration() -> None:
        """
        Valida a configuração necessária.

        A configuração continua centralizada em config.py.
        """

        if hasattr(ROI, "validate"):
            ROI.validate()

        if not getattr(ROI, "enabled", True):
            raise RuntimeError(
                "ROI está desabilitado. "
                "A geometria YOLOP exige um ROI válido."
            )

        input_width = getattr(
            YOLOP,
            "input_width",
            640,
        )

        input_height = getattr(
            YOLOP,
            "input_height",
            384,
        )

        if input_width <= 0:
            raise ValueError(
                "YOLOP.input_width deve ser maior que zero."
            )

        if input_height <= 0:
            raise ValueError(
                "YOLOP.input_height deve ser maior que zero."
            )

        min_points = getattr(
            LANE_GEOMETRY,
            "min_points",
            4,
        )

        if min_points < 2:
            raise ValueError(
                "LANE_GEOMETRY.min_points deve ser >= 2."
            )

        min_span = getattr(
            LANE_GEOMETRY,
            "min_observed_span",
            20.0,
        )

        if min_span <= 0:
            raise ValueError(
                "LANE_GEOMETRY.min_observed_span deve ser > 0."
            )

        min_width = getattr(
            LANE_GEOMETRY,
            "min_lane_width",
            40.0,
        )

        max_width = getattr(
            LANE_GEOMETRY,
            "max_lane_width",
            1200.0,
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
    # NUMÉRICO
    # =========================================================================

    @staticmethod
    def _finite(value: object) -> bool:
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _clip01(value: float) -> float:
        if not math.isfinite(float(value)):
            return 0.0

        return float(
            np.clip(
                value,
                0.0,
                1.0,
            )
        )

    @staticmethod
    def _safe_float(
        value: object,
        default: float = 0.0,
    ) -> float:
        try:
            value = float(value)

            if math.isfinite(value):
                return value

        except (TypeError, ValueError):
            pass

        return default

    # =========================================================================
    # COORDENADAS
    # =========================================================================

    @staticmethod
    def _roi_value(
        name: str,
        default: float,
    ) -> float:
        return LaneGeometry._safe_float(
            getattr(ROI, name, default),
            default,
        )

    @classmethod
    def _roi_dimensions(cls) -> Tuple[float, float, float, float]:
        """
        Retorna:

            left
            top
            width
            height
        """

        left = cls._roi_value(
            "left",
            0.0,
        )

        top = cls._roi_value(
            "top",
            0.0,
        )

        width = cls._roi_value(
            "width",
            1.0,
        )

        height = cls._roi_value(
            "height",
            1.0,
        )

        if width <= 0:
            width = 1.0

        if height <= 0:
            height = 1.0

        return (
            left,
            top,
            width,
            height,
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
                    384,
                )
            ),
        )

    @classmethod
    def detector_to_screen(
        cls,
        x: float,
        y: float,
    ) -> Point:
        """
        Converte YOLOP → ROI → tela.

        YOLOP trabalha sobre o frame normalizado de entrada.
        """

        detector_width, detector_height = (
            cls._detector_dimensions()
        )

        roi_left, roi_top, roi_width, roi_height = (
            cls._roi_dimensions()
        )

        screen_x = (
            roi_left
            + (
                float(x)
                / detector_width
            )
            * roi_width
        )

        screen_y = (
            roi_top
            + (
                float(y)
                / detector_height
            )
            * roi_height
        )

        return (
            float(screen_x),
            float(screen_y),
        )

    # =========================================================================
    # CONVERSÃO DE LANES
    # =========================================================================

    def convert_lane(
        self,
        lane: Sequence[LanePoint],
    ) -> List[Point]:
        """
        Converte uma lane YOLOP para coordenadas de tela.
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
                0.20,
            )
        )

        points: List[Point] = []

        for point in lane:

            if not isinstance(
                point,
                LanePoint,
            ):
                continue

            try:
                valid = bool(
                    point.is_valid()
                )
            except AttributeError:
                valid = bool(
                    getattr(
                        point,
                        "valid",
                        True,
                    )
                )

            if not valid:
                continue

            confidence = self._safe_float(
                getattr(
                    point,
                    "confidence",
                    1.0,
                )
            )

            if confidence < min_confidence:
                continue

            x = self._safe_float(
                getattr(point, "x", None),
                math.nan,
            )

            y = self._safe_float(
                getattr(point, "y", None),
                math.nan,
            )

            if not (
                math.isfinite(x)
                and math.isfinite(y)
            ):
                continue

            if (
                x < 0.0
                or x > detector_width
                or y < 0.0
                or y > detector_height
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
                points.append(
                    screen_point
                )

        points.sort(
            key=lambda p: p[1]
        )

        return self._deduplicate_y(
            points
        )

    @staticmethod
    def _deduplicate_y(
        points: Sequence[Point],
        tolerance: float = 1e-5,
    ) -> List[Point]:
        """
        Remove pontos praticamente coincidentes em Y.
        """

        if not points:
            return []

        result: List[Point] = []

        for point in points:

            if not result:
                result.append(point)
                continue

            if abs(
                point[1]
                - result[-1][1]
            ) <= tolerance:
                continue

            result.append(point)

        return result

    # =========================================================================
    # OUTLIERS
    # =========================================================================

    def remove_outliers(
        self,
        points: Sequence[Point],
    ) -> List[Point]:
        """
        Remove outliers geométricos usando MAD.

        Nunca cria pontos artificiais.
        """

        points = list(points)

        enabled = bool(
            getattr(
                LANE_GEOMETRY,
                "enable_outlier_rejection",
                True,
            )
        )

        if not enabled:
            return points

        if len(points) < 6:
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

        y_mean = float(
            np.mean(y)
        )

        y_std = float(
            np.std(y)
        )

        if y_std < 1e-6:
            return points

        normalized_y = (
            y - y_mean
        ) / y_std

        try:
            coefficients = np.polyfit(
                normalized_y,
                x,
                2,
            )

            predicted = np.polyval(
                coefficients,
                normalized_y,
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
                    residuals - median
                )
            )
        )

        sigma = float(
            getattr(
                LANE_GEOMETRY,
                "outlier_sigma",
                3.5,
            )
        )

        if mad < 1e-8:
            threshold = max(
                5.0,
                median * 3.0,
            )
        else:
            robust_sigma = (
                1.4826 * mad
            )

            threshold = (
                median
                + sigma
                * robust_sigma
            )

        filtered = [
            point
            for point, residual
            in zip(
                points,
                residuals,
            )
            if residual <= threshold
        ]

        minimum = int(
            getattr(
                LANE_GEOMETRY,
                "min_points",
                4,
            )
        )

        if len(filtered) < minimum:
            return points

        return filtered

    # =========================================================================
    # MÉTRICAS
    # =========================================================================

    @staticmethod
    def lane_span(
        lane: Sequence[Point],
    ) -> float:
        if len(lane) < 2:
            return 0.0

        ys = [
            point[1]
            for point in lane
        ]

        return float(
            max(ys)
            - min(ys)
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
    def _lane_y_range(
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

        left_range = cls._lane_y_range(
            left
        )

        right_range = cls._lane_y_range(
            right
        )

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

    @staticmethod
    def interpolate_x(
        lane: Sequence[Point],
        y: float,
    ) -> Optional[float]:
        """
        Interpolação somente dentro da região observada.

        Extrapolação é proibida neste módulo.
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

        if (
            not np.all(
                np.isfinite(ys)
            )
            or not np.all(
                np.isfinite(xs)
            )
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
    # CENTRO DA LANE
    # =========================================================================

    def build_center_line(
        self,
        left: Sequence[Point],
        right: Sequence[Point],
    ) -> List[Point]:
        """
        Constrói o centro somente na região comum observada.
        """

        common_range = self.common_y_range(
            left,
            right,
        )

        if common_range is None:
            return []

        lower, upper = common_range

        samples = int(
            getattr(
                LANE_GEOMETRY,
                "center_samples",
                16,
            )
        )

        samples = max(
            2,
            samples,
        )

        ys = np.linspace(
            lower,
            upper,
            samples,
        )

        center: List[Point] = []

        for y in ys:

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

            center_x = (
                left_x
                + right_x
            ) * 0.5

            if not self._finite(
                center_x
            ):
                continue

            center.append(
                (
                    float(center_x),
                    float(y),
                )
            )

        return center

    # =========================================================================
    # SELEÇÃO DE PARES
    # =========================================================================

    def _pair_score(
        self,
        left: Sequence[Point],
        right: Sequence[Point],
    ) -> float:
        """
        Avalia a plausibilidade de duas lanes formarem uma faixa.

        Quanto maior, melhor.
        """

        common_range = self.common_y_range(
            left,
            right,
        )

        if common_range is None:
            return -math.inf

        lower, upper = common_range

        span = (
            upper
            - lower
        )

        min_span = float(
            getattr(
                LANE_GEOMETRY,
                "min_observed_span",
                20.0,
            )
        )

        if span < min_span:
            return -math.inf

        sample_count = 7

        ys = np.linspace(
            lower,
            upper,
            sample_count,
        )

        widths = []

        for y in ys:

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

            if width <= 0:
                return -math.inf

            widths.append(
                width
            )

        if len(widths) < 3:
            return -math.inf

        widths_array = np.asarray(
            widths,
            dtype=np.float64,
        )

        mean_width = float(
            np.mean(widths_array)
        )

        width_std = float(
            np.std(widths_array)
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

        if (
            mean_width < min_width
            or mean_width > max_width
        ):
            return -math.inf

        width_stability = (
            1.0
            / (
                1.0
                + width_std
                / max(
                    mean_width,
                    1.0,
                )
            )
        )

        span_score = self._clip01(
            span
            / max(
                min_span * 4.0,
                1.0,
            )
        )

        center_order_score = 1.0

        mean_left = self.lane_mean_x(
            left
        )

        mean_right = self.lane_mean_x(
            right
        )

        if mean_right <= mean_left:
            center_order_score = 0.0

        return float(
            0.45 * span_score
            + 0.35 * width_stability
            + 0.20 * center_order_score
        )

    def select_best_pair(
        self,
        lanes: Sequence[Sequence[Point]],
    ) -> Tuple[
        int,
        int,
        float,
    ]:
        """
        Seleciona o melhor par esquerdo/direito.

        Retorna:

            left_index
            right_index
            score

        Quando nenhum par é válido:

            -1
            -1
            0.0
        """

        best_left = -1
        best_right = -1
        best_score = -math.inf

        for i in range(
            len(lanes)
        ):

            for j in range(
                len(lanes)
            ):

                if i == j:
                    continue

                left = lanes[i]
                right = lanes[j]

                if (
                    self.lane_mean_x(left)
                    >= self.lane_mean_x(right)
                ):
                    continue

                score = self._pair_score(
                    left,
                    right,
                )

                if score > best_score:
                    best_score = score
                    best_left = i
                    best_right = j

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
    # LARGURA
    # =========================================================================

    def calculate_lane_width(
        self,
        left: Sequence[Point],
        right: Sequence[Point],
    ) -> float:
        """
        Calcula a largura média observada.
        """

        common_range = self.common_y_range(
            left,
            right,
        )

        if common_range is None:
            return 0.0

        lower, upper = common_range

        samples = np.linspace(
            lower,
            upper,
            9,
        )

        widths = []

        for y in samples:

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
                    width
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
    # ERRO LATERAL
    # =========================================================================

    def calculate_lateral_error(
        self,
        lane_center_x: float,
        image_center_x: float,
    ) -> float:
        """
        Erro lateral normalizado.

        Convenção:

            < 0 → veículo/lane center à esquerda
            > 0 → veículo/lane center à direita
        """

        width = self._roi_value(
            "width",
            1.0,
        )

        if width <= 0:
            return 0.0

        error = (
            lane_center_x
            - image_center_x
        ) / (
            width * 0.5
        )

        return float(
            np.clip(
                error,
                -1.0,
                1.0,
            )
        )

    # =========================================================================
    # HEADING
    # =========================================================================

    @staticmethod
    def calculate_heading(
        center_line: Sequence[Point],
    ) -> float:
        """
        Calcula heading observado em relação ao eixo vertical.

        Não extrapola a lane.
        """

        if len(center_line) < 2:
            return 0.0

        ordered = sorted(
            center_line,
            key=lambda p: p[1],
        )

        first = ordered[
            max(
                0,
                len(ordered) // 5,
            )
        ]

        last = ordered[
            min(
                len(ordered) - 1,
                len(ordered)
                - 1,
            )
        ]

        dx = (
            last[0]
            - first[0]
        )

        dy = (
            last[1]
            - first[1]
        )

        if abs(dy) < 1e-6:
            return 0.0

        angle = math.atan2(
            dx,
            abs(dy),
        )

        max_angle = math.pi / 2.0

        return float(
            np.clip(
                angle / max_angle,
                -1.0,
                1.0,
            )
        )

    # =========================================================================
    # CURVATURA
    # =========================================================================

    @staticmethod
    def calculate_curvature(
        center_line: Sequence[Point],
    ) -> float:
        """
        Estima curvatura observada normalizada.

        O valor é adimensional.

        Quanto maior o módulo:
            maior a curvatura observada.

        Nenhuma extrapolação é realizada.
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

        if y_scale < 1e-6:
            return 0.0

        normalized_y = (
            y - y_center
        ) / y_scale

        try:
            coefficients = np.polyfit(
                normalized_y,
                x,
                2,
            )

        except (
            np.linalg.LinAlgError,
            ValueError,
            FloatingPointError,
        ):
            return 0.0

        curvature = float(
            2.0
            * coefficients[0]
        )

        if not math.isfinite(
            curvature
        ):
            return 0.0

        return float(
            np.clip(
                curvature,
                -10.0,
                10.0,
            )
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
    ) -> float:
        """
        Confiança geométrica composta.

        Componentes:

            - quantidade de observações;
            - span vertical;
            - estabilidade da largura;
            - qualidade do centro;
            - qualidade do par.
        """

        if (
            not left
            or not right
            or not center_line
        ):
            return 0.0

        min_points = float(
            getattr(
                LANE_GEOMETRY,
                "min_points",
                4,
            )
        )

        point_score = self._clip01(
            min(
                len(left),
                len(right),
            )
            / max(
                min_points * 2.0,
                1.0,
            )
        )

        left_span = self.lane_span(
            left
        )

        right_span = self.lane_span(
            right
        )

        span = min(
            left_span,
            right_span,
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

        width = self.calculate_lane_width(
            left,
            right,
        )

        width_score = 0.0

        if width > 0:

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

            if (
                min_width
                <= width
                <= max_width
            ):
                middle = (
                    min_width
                    + max_width
                ) * 0.5

                half_range = (
                    max_width
                    - min_width
                ) * 0.5

                if half_range > 0:
                    width_score = self._clip01(
                        1.0
                        - abs(
                            width
                            - middle
                        )
                        / half_range
                    )

        center_score = self._clip01(
            len(center_line)
            / 16.0
        )

        return self._clip01(
            0.25 * point_score
            + 0.20 * span_score
            + 0.20 * width_score
            + 0.15 * center_score
            + 0.20 * pair_score
        )

    # =========================================================================
    # PROCESSAMENTO PRINCIPAL
    # =========================================================================

    def process(
        self,
        lanes: Sequence[Sequence[LanePoint]],
        image_width: Optional[float] = None,
        image_height: Optional[float] = None,
    ) -> LaneGeometryResult:
        """
        Processa lanes YOLOP observadas.

        A função é stateless.
        """

        converted: List[List[Point]] = []

        for lane in lanes:

            points = self.convert_lane(
                lane
            )

            points = self.remove_outliers(
                points
            )

            minimum = int(
                getattr(
                    LANE_GEOMETRY,
                    "min_points",
                    4,
                )
            )

            if len(points) < minimum:
                continue

            converted.append(
                points
            )

        if image_width is None:
            roi_left, _, roi_width, _ = (
                self._roi_dimensions()
            )

            image_center_x = (
                roi_left
                + roi_width * 0.5
            )

        else:
            image_center_x = (
                float(image_width)
                * 0.5
            )

        if image_height is None:
            _, roi_top, _, roi_height = (
                self._roi_dimensions()
            )

            image_center_y = (
                roi_top
                + roi_height * 0.5
            )

        else:
            image_center_y = (
                float(image_height)
                * 0.5
            )

        if len(converted) < 2:
            return self._invalid_result(
                image_center_x,
                image_center_y,
                converted,
            )

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

        center_line = self.build_center_line(
            left,
            right,
        )

        if not center_line:
            return self._invalid_result(
                image_center_x,
                image_center_y,
                converted,
                left_index,
                right_index,
            )

        lane_center_y = float(
            np.median(
                [
                    point[1]
                    for point in center_line
                ]
            )
        )

        center_x_at_reference = (
            self._center_x_at_reference(
                center_line
            )
        )

        lateral_error = (
            self.calculate_lateral_error(
                center_x_at_reference,
                image_center_x,
            )
        )

        heading_error = (
            self.calculate_heading(
                center_line
            )
        )

        lane_width = (
            self.calculate_lane_width(
                left,
                right,
            )
        )

        curvature = (
            self.calculate_curvature(
                center_line
            )
        )

        confidence = (
            self.calculate_confidence(
                left,
                right,
                center_line,
                pair_score,
            )
        )

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
            observed_span
            >= min_span
        )

        valid = (
            lane_width > 0.0
            and observed_span >= min_span
            and len(center_line) >= 3
            and confidence
            > 0.0
        )

        additional = [
            lane
            for index, lane
            in enumerate(converted)
            if index
            not in (
                left_index,
                right_index,
            )
        ]

        return LaneGeometryResult(
            lane_center_x=float(
                center_x_at_reference
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
            valid=bool(
                valid
            ),
            left_lane_screen=list(
                left
            ),
            right_lane_screen=list(
                right
            ),
            additional_lanes_screen=[
                list(lane)
                for lane in additional
            ],
            selected_left_index=int(
                left_index
            ),
            selected_right_index=int(
                right_index
            ),
            geometry_confidence=float(
                confidence
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
    # REFERÊNCIA DO CENTRO
    # =========================================================================

    @staticmethod
    def _center_x_at_reference(
        center_line: Sequence[Point],
    ) -> float:
        """
        Obtém o centro observado na região inferior
        da lane.

        Isso é preferível à média simples porque a posição
        relevante para LKA está próxima do veículo.
        """

        if not center_line:
            return 0.0

        ordered = sorted(
            center_line,
            key=lambda p: p[1],
        )

        count = max(
            1,
            len(ordered) // 4,
        )

        reference_points = (
            ordered[-count:]
        )

        return float(
            np.mean(
                [
                    point[0]
                    for point
                    in reference_points
                ]
            )
        )

    # =========================================================================
    # RESULTADO INVÁLIDO
    # =========================================================================

    @staticmethod
    def _invalid_result(
        image_center_x: float,
        image_center_y: float,
        lanes: Sequence[Sequence[Point]],
        left_index: int = -1,
        right_index: int = -1,
    ) -> LaneGeometryResult:
        """
        Falha segura.

        Nenhum NaN é propagado.
        """

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
            left_lane_screen=(
                list(
                    lanes[left_index]
                )
                if (
                    left_index >= 0
                    and left_index < len(lanes)
                )
                else []
            ),
            right_lane_screen=(
                list(
                    lanes[right_index]
                )
                if (
                    right_index >= 0
                    and right_index < len(lanes)
                )
                else []
            ),
            additional_lanes_screen=[
                list(lane)
                for index, lane
                in enumerate(lanes)
                if index
                not in (
                    left_index,
                    right_index,
                )
            ],
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

    # =========================================================================
    # ALIASES / COMPATIBILIDADE
    # =========================================================================

    def calculate(
        self,
        lanes: Sequence[Sequence[LanePoint]],
        image_width: Optional[float] = None,
        image_height: Optional[float] = None,
    ) -> LaneGeometryResult:
        """
        Alias semântico para process().
        """

        return self.process(
            lanes=lanes,
            image_width=image_width,
            image_height=image_height,
        )

    def compute(
        self,
        lanes: Sequence[Sequence[LanePoint]],
        image_width: Optional[float] = None,
        image_height: Optional[float] = None,
    ) -> LaneGeometryResult:
        """
        Alias para integrações que utilizem compute().
        """

        return self.process(
            lanes=lanes,
            image_width=image_width,
            image_height=image_height,
        )


# =============================================================================
# FUNÇÃO DE CONVENIÊNCIA
# =============================================================================

def compute_lane_geometry(
    lanes: Sequence[Sequence[LanePoint]],
    image_width: Optional[float] = None,
    image_height: Optional[float] = None,
) -> LaneGeometryResult:
    """
    API funcional para o pipeline.
    """

    geometry = LaneGeometry()

    return geometry.process(
        lanes=lanes,
        image_width=image_width,
        image_height=image_height,
    )


__all__ = [
    "Point",
    "LaneGeometryResult",
    "LaneGeometry",
    "compute_lane_geometry",
]