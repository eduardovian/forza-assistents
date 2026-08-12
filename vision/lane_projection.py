"""
vision/lane_projection.py

Projeção matemática das linhas de faixa.

Responsabilidade:

    LaneModel
        ↓
    LaneProjectionEngine
        ↓
    validação dos pontos
        ↓
    ajuste polinomial x(y)
        ↓
    extrapolação controlada
        ↓
    LaneProjectionResult

Este módulo NÃO:

    - executa YOLOP;
    - realiza tracking;
    - identifica a faixa atual;
    - calcula posição do veículo;
    - toma decisões ADAS.
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .lane_types import (
    LaneModel,
    LanePoint,
    LanePolynomial,
    LaneProjection,
    ProjectionQuality,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

DEFAULT_MIN_POINTS = 8
DEFAULT_MIN_VERTICAL_SPAN = 80.0
DEFAULT_MAX_FIT_ERROR = 25.0
DEFAULT_MAX_EXTRAPOLATION = 0.35
DEFAULT_SAMPLE_STEP = 8
DEFAULT_POLYNOMIAL_DEGREE = 3
DEFAULT_MIN_CONFIDENCE = 0.45
DEFAULT_MAX_PROJECTION_DISTANCE = None


# =============================================================================
# UTILITÁRIOS
# =============================================================================

def _clip01(value: float) -> float:
    """
    Limita um valor ao intervalo [0, 1].
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0

    if not np.isfinite(value):
        return 0.0

    return float(np.clip(value, 0.0, 1.0))


def _is_finite_scalar(value: object) -> bool:
    """
    Verifica se um valor pode ser convertido para float e é finito.
    """
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _finite_array(values: np.ndarray) -> bool:
    """
    Verifica se todos os elementos de um array são finitos.
    """
    if values is None:
        return False

    try:
        return bool(np.all(np.isfinite(values)))
    except (TypeError, ValueError):
        return False


# =============================================================================
# RESULTADO
# =============================================================================

class LaneProjectionResult:
    """
    Resultado detalhado da projeção de uma linha de faixa.

    Attributes
    ----------
    points:
        Pontos resultantes da projeção.

    coefficients:
        Coeficientes do polinômio normalizado utilizado no ajuste.

    degree:
        Grau efetivamente utilizado no ajuste.

    fitted:
        Indica se o ajuste polinomial foi realizado.

    extrapolated:
        Indica se houve extrapolação além dos dados observados.

    confidence:
        Confiança final da projeção.

    fit_error:
        Erro do ajuste em pixels.

    source_y_min/source_y_max:
        Limites verticais dos dados originais.

    projected_y_min/projected_y_max:
        Limites verticais efetivamente projetados.

    valid:
        Indica se a projeção é válida.

    error:
        Descrição do erro quando a projeção é rejeitada.
    """

    def __init__(
        self,
        points: Optional[List[LanePoint]] = None,
        coefficients: Tuple[float, ...] = (),
        degree: int = 0,
        fitted: bool = False,
        extrapolated: bool = False,
        confidence: float = 0.0,
        fit_error: float = float("inf"),
        source_y_min: float = 0.0,
        source_y_max: float = 0.0,
        projected_y_min: float = 0.0,
        projected_y_max: float = 0.0,
        valid: bool = False,
        error: Optional[str] = None,
    ) -> None:

        self.points = list(points) if points is not None else []

        self.coefficients = tuple(
            float(value)
            for value in coefficients
        )

        self.degree = int(degree)
        self.fitted = bool(fitted)
        self.extrapolated = bool(extrapolated)

        self.confidence = float(confidence)
        self.fit_error = float(fit_error)

        self.source_y_min = float(source_y_min)
        self.source_y_max = float(source_y_max)

        self.projected_y_min = float(projected_y_min)
        self.projected_y_max = float(projected_y_max)

        self.valid = bool(valid)
        self.error = error

    @property
    def quality(self) -> ProjectionQuality:
        """
        Classificação qualitativa da projeção.
        """

        if not self.valid:
            return ProjectionQuality.NONE

        if self.confidence >= 0.80:
            return ProjectionQuality.HIGH

        if self.confidence >= 0.60:
            return ProjectionQuality.MEDIUM

        if self.confidence >= 0.40:
            return ProjectionQuality.LOW

        return ProjectionQuality.NONE


# =============================================================================
# MOTOR
# =============================================================================

class LaneProjectionEngine:
    """
    Motor responsável por ajustar e projetar uma linha de faixa.

    Modelo:

        x(y) = a*y³ + b*y² + c*y + d

    O ajuste interno utiliza Y normalizado para melhorar
    a estabilidade numérica.

    A projeção final é limitada à imagem.
    """

    def __init__(
        self,
        min_points: int = DEFAULT_MIN_POINTS,
        min_vertical_span: float = DEFAULT_MIN_VERTICAL_SPAN,
        max_fit_error: float = DEFAULT_MAX_FIT_ERROR,
        max_extrapolation: float = DEFAULT_MAX_EXTRAPOLATION,
        sample_step: int = DEFAULT_SAMPLE_STEP,
        polynomial_degree: int = DEFAULT_POLYNOMIAL_DEGREE,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        degree: Optional[int] = None,
        max_projection_distance: Optional[float] = (
            DEFAULT_MAX_PROJECTION_DISTANCE
        ),
    ) -> None:

        if degree is not None:
            polynomial_degree = degree

        self.min_points = max(
            2,
            int(min_points),
        )

        self.min_vertical_span = max(
            0.0,
            float(min_vertical_span),
        )

        self.max_fit_error = max(
            0.1,
            float(max_fit_error),
        )

        self.degree = int(
            np.clip(
                int(polynomial_degree),
                1,
                3,
            )
        )

        # Compatibilidade.
        self.polynomial_degree = self.degree

        self.max_extrapolation = float(
            np.clip(
                float(max_extrapolation),
                0.0,
                2.0,
            )
        )

        self.sample_step = max(
            1,
            int(sample_step),
        )

        self.min_confidence = _clip01(
            min_confidence,
        )

        if max_projection_distance is None:
            self.max_projection_distance = None
        else:
            self.max_projection_distance = max(
                0.0,
                float(max_projection_distance),
            )

    # =========================================================================
    # VALIDAÇÃO DE PONTOS
    # =========================================================================

    @staticmethod
    def _valid_points(
        points: Iterable[LanePoint],
    ) -> List[LanePoint]:
        """
        Filtra pontos inválidos sem modificar os objetos originais.
        """

        if points is None:
            return []

        result: List[LanePoint] = []

        try:
            iterator = iter(points)
        except TypeError:
            return []

        for point in iterator:

            if point is None:
                continue

            try:
                if not bool(
                    getattr(
                        point,
                        "valid",
                        True,
                    )
                ):
                    continue

                x = float(point.x)
                y = float(point.y)
                confidence = float(point.confidence)

            except (
                AttributeError,
                TypeError,
                ValueError,
            ):
                continue

            if not (
                np.isfinite(x)
                and np.isfinite(y)
                and np.isfinite(confidence)
            ):
                continue

            result.append(point)

        return result

    # =========================================================================
    # PREPARAÇÃO
    # =========================================================================

    def _prepare_points(
        self,
        points: Sequence[LanePoint],
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """
        Valida, filtra, ordena e consolida os pontos.
        """

        valid = self._valid_points(points)

        valid = [
            point
            for point in valid
            if float(point.confidence)
            >= self.min_confidence
        ]

        if len(valid) < self.min_points:
            raise ValueError(
                "Pontos insuficientes para projeção."
            )

        valid.sort(
            key=lambda point: float(point.y)
        )

        groups = {}

        for point in valid:

            key = round(
                float(point.y),
                3,
            )

            groups.setdefault(
                key,
                [],
            ).append(
                point
            )

        unique_x: List[float] = []
        unique_y: List[float] = []
        unique_confidence: List[float] = []

        for group in groups.values():

            xs = np.asarray(
                [
                    float(point.x)
                    for point in group
                ],
                dtype=np.float64,
            )

            ys = np.asarray(
                [
                    float(point.y)
                    for point in group
                ],
                dtype=np.float64,
            )

            weights = np.asarray(
                [
                    max(
                        0.0,
                        float(point.confidence),
                    )
                    for point in group
                ],
                dtype=np.float64,
            )

            if not (
                _finite_array(xs)
                and _finite_array(ys)
                and _finite_array(weights)
            ):
                continue

            if np.sum(weights) <= 0.0:
                weights = np.ones_like(weights)

            unique_x.append(
                float(
                    np.average(
                        xs,
                        weights=weights,
                    )
                )
            )

            unique_y.append(
                float(
                    np.average(
                        ys,
                        weights=weights,
                    )
                )
            )

            unique_confidence.append(
                float(
                    np.mean(weights)
                )
            )

        if len(unique_x) < self.min_points:
            raise ValueError(
                "Pontos verticais insuficientes."
            )

        x = np.asarray(
            unique_x,
            dtype=np.float64,
        )

        y = np.asarray(
            unique_y,
            dtype=np.float64,
        )

        confidence = np.asarray(
            unique_confidence,
            dtype=np.float64,
        )

        if not (
            _finite_array(x)
            and _finite_array(y)
            and _finite_array(confidence)
        ):
            raise ValueError(
                "Pontos possuem valores não finitos."
            )

        order = np.argsort(y)

        x = x[order]
        y = y[order]
        confidence = confidence[order]

        vertical_span = float(
            y[-1] - y[0]
        )

        if vertical_span < self.min_vertical_span:
            raise ValueError(
                "Extensão vertical insuficiente."
            )

        return (
            x,
            y,
            confidence,
        )

    # =========================================================================
    # AJUSTE POLINOMIAL
    # =========================================================================

    def _fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[
        np.ndarray,
        float,
        float,
        float,
        int,
    ]:
        """
        Ajusta x(y) em coordenadas Y normalizadas.
        """

        if len(x) < 2:
            raise ValueError(
                "Pontos insuficientes para ajuste."
            )

        degree = min(
            self.degree,
            len(x) - 1,
        )

        if degree < 1:
            raise ValueError(
                "Grau polinomial inválido."
            )

        center = float(
            np.mean(y)
        )

        scale = float(
            np.max(
                np.abs(
                    y - center
                )
            )
        )

        if not np.isfinite(scale) or scale < 1e-9:
            raise ValueError(
                "Escala vertical inválida."
            )

        normalized_y = (
            y - center
        ) / scale

        if not _finite_array(normalized_y):
            raise ValueError(
                "Normalização vertical inválida."
            )

        try:
            coefficients = np.polyfit(
                normalized_y,
                x,
                degree,
            )
        except (
            TypeError,
            ValueError,
            np.linalg.LinAlgError,
        ) as exc:
            raise ValueError(
                f"Falha no ajuste polinomial: {exc}"
            ) from exc

        if not _finite_array(coefficients):
            raise ValueError(
                "Coeficientes polinomiais inválidos."
            )

        predicted = np.polyval(
            coefficients,
            normalized_y,
        )

        if not _finite_array(predicted):
            raise ValueError(
                "Predição polinomial inválida."
            )

        residuals = (
            x - predicted
        )

        if not _finite_array(residuals):
            raise ValueError(
                "Resíduos inválidos."
            )

        absolute_residuals = np.abs(
            residuals
        )

        rmse = float(
            np.sqrt(
                np.mean(
                    residuals ** 2
                )
            )
        )

        median_error = float(
            np.median(
                absolute_residuals
            )
        )

        fit_error = max(
            median_error,
            rmse * 0.75,
        )

        if not np.isfinite(fit_error):
            raise ValueError(
                "Erro do ajuste inválido."
            )

        return (
            coefficients,
            fit_error,
            center,
            scale,
            degree,
        )

    # =========================================================================
    # COEFICIENTES ABSOLUTOS
    # =========================================================================

    @staticmethod
    def _to_absolute_coefficients(
        coefficients: np.ndarray,
        center: float,
        scale: float,
    ) -> Tuple[
        float,
        float,
        float,
        float,
    ]:
        """
        Converte:

            p((y-center)/scale)

        para:

            a*y³ + b*y² + c*y + d
        """

        if not (
            _is_finite_scalar(center)
            and _is_finite_scalar(scale)
            and float(scale) > 0.0
        ):
            raise ValueError(
                "Centro ou escala inválidos."
            )

        coefficients = np.asarray(
            coefficients,
            dtype=np.float64,
        )

        if not _finite_array(coefficients):
            raise ValueError(
                "Coeficientes inválidos."
            )

        if len(coefficients) == 4:

            an, bn, cn, dn = coefficients

            a = (
                an
                / scale**3
            )

            b = (
                -3.0
                * an
                * center
                / scale**3
                + bn
                / scale**2
            )

            c = (
                3.0
                * an
                * center**2
                / scale**3
                - 2.0
                * bn
                * center
                / scale**2
                + cn
                / scale
            )

            d = (
                -an
                * center**3
                / scale**3
                + bn
                * center**2
                / scale**2
                - cn
                * center
                / scale
                + dn
            )

            result = (
                float(a),
                float(b),
                float(c),
                float(d),
            )

        elif len(coefficients) == 3:

            an, bn, cn = coefficients

            a = 0.0

            b = (
                an
                / scale**2
            )

            c = (
                -2.0
                * an
                * center
                / scale**2
                + bn
                / scale
            )

            d = (
                an
                * center**2
                / scale**2
                - bn
                * center
                / scale
                + cn
            )

            result = (
                float(a),
                float(b),
                float(c),
                float(d),
            )

        elif len(coefficients) == 2:

            an, bn = coefficients

            result = (
                0.0,
                0.0,
                float(an / scale),
                float(
                    bn
                    - an
                    * center
                    / scale
                ),
            )

        elif len(coefficients) == 1:

            result = (
                0.0,
                0.0,
                0.0,
                float(coefficients[0]),
            )

        else:
            raise ValueError(
                "Quantidade de coeficientes inválida."
            )

        if not all(
            np.isfinite(value)
            for value in result
        ):
            raise ValueError(
                "Coeficientes absolutos inválidos."
            )

        return result

    # =========================================================================
    # CONFIANÇA
    # =========================================================================

    def _confidence(
        self,
        point_count: int,
        vertical_span: float,
        fit_error: float,
        point_confidence: float,
    ) -> float:
        """
        Calcula confiança combinando:

        - quantidade de pontos;
        - extensão vertical;
        - erro do ajuste;
        - confiança média da detecção.
        """

        count_score = _clip01(
            point_count / 25.0
        )

        span_score = _clip01(
            vertical_span / 400.0
        )

        if not np.isfinite(fit_error):
            error_score = 0.0
        else:
            error_score = float(
                np.exp(
                    -fit_error / 18.0
                )
            )

        confidence = (
            0.30 * count_score
            + 0.25 * span_score
            + 0.25 * error_score
            + 0.20 * _clip01(
                point_confidence
            )
        )

        return _clip01(
            confidence
        )

    # =========================================================================
    # QUALIDADE
    # =========================================================================

    @staticmethod
    def _quality(
        confidence: float,
    ) -> ProjectionQuality:

        confidence = _clip01(
            confidence
        )

        if confidence >= 0.80:
            return ProjectionQuality.HIGH

        if confidence >= 0.60:
            return ProjectionQuality.MEDIUM

        if confidence >= 0.40:
            return ProjectionQuality.LOW

        return ProjectionQuality.NONE

    # =========================================================================
    # INTERVALO DE PROJEÇÃO
    # =========================================================================

    def _projection_interval(
        self,
        source_y_min: float,
        source_y_max: float,
        image_height: int,
    ) -> Tuple[
        float,
        float,
    ]:
        """
        Calcula o intervalo vertical da projeção.

        max_projection_distance, quando definido, limita a distância
        de qualquer ponto projetado em relação a source_y_max.
        """

        if (
            not np.isfinite(source_y_min)
            or not np.isfinite(source_y_max)
        ):
            raise ValueError(
                "Limites verticais inválidos."
            )

        if source_y_max <= source_y_min:
            raise ValueError(
                "Intervalo vertical inválido."
            )

        if self.max_projection_distance is not None:

            distance = (
                self.max_projection_distance
            )

            projected_y_min = max(
                0.0,
                source_y_max
                - distance,
            )

            projected_y_max = min(
                float(image_height - 1),
                source_y_max,
            )

        else:

            vertical_span = (
                source_y_max
                - source_y_min
            )

            extrapolation = (
                vertical_span
                * self.max_extrapolation
            )

            projected_y_min = max(
                0.0,
                source_y_min
                - extrapolation,
            )

            projected_y_max = min(
                float(image_height - 1),
                source_y_max
                + extrapolation,
            )

        if (
            projected_y_max
            <= projected_y_min
        ):
            raise ValueError(
                "Intervalo de projeção inválido."
            )

        return (
            float(projected_y_min),
            float(projected_y_max),
        )

    # =========================================================================
    # AMOSTRAGEM
    # =========================================================================

    def _sample_y(
        self,
        y_min: float,
        y_max: float,
    ) -> np.ndarray:
        """
        Gera amostras verticais incluindo obrigatoriamente o limite final.
        """

        if y_max <= y_min:
            return np.empty(
                0,
                dtype=np.float64,
            )

        ys = np.arange(
            y_min,
            y_max + self.sample_step,
            self.sample_step,
            dtype=np.float64,
        )

        if ys.size == 0:
            return np.asarray(
                [y_min, y_max],
                dtype=np.float64,
            )

        if ys[-1] < y_max:
            ys = np.append(
                ys,
                y_max,
            )

        else:
            ys[-1] = min(
                ys[-1],
                y_max,
            )

        return ys

    # =========================================================================
    # PROJEÇÃO PRINCIPAL
    # =========================================================================

    def project(
        self,
        points: Sequence[LanePoint],
        image_height: int = 480,
        image_width: int = 640,
    ) -> LaneProjectionResult:
        """
        Ajusta e projeta uma linha de faixa.
        """

        try:

            image_height = int(
                image_height
            )

            image_width = int(
                image_width
            )

            if image_height <= 0:
                raise ValueError(
                    "image_height inválido."
                )

            if image_width <= 0:
                raise ValueError(
                    "image_width inválido."
                )

            (
                x,
                y,
                confidence_values,
            ) = self._prepare_points(
                points
            )

            (
                coefficients,
                fit_error,
                y_center,
                y_scale,
                degree,
            ) = self._fit(
                x,
                y,
            )

            if (
                fit_error > self.max_fit_error
            ):
                raise ValueError(
                    "Erro do ajuste acima do limite permitido."
                )

            source_y_min = float(
                y[0]
            )

            source_y_max = float(
                y[-1]
            )

            (
                projected_y_min,
                projected_y_max,
            ) = self._projection_interval(
                source_y_min,
                source_y_max,
                image_height,
            )

            projected_y = self._sample_y(
                projected_y_min,
                projected_y_max,
            )

            if projected_y.size < 2:
                raise ValueError(
                    "Amostragem vertical insuficiente."
                )

            normalized_y = (
                projected_y
                - y_center
            ) / y_scale

            if not _finite_array(
                normalized_y
            ):
                raise ValueError(
                    "Coordenadas normalizadas inválidas."
                )

            projected_x = np.polyval(
                coefficients,
                normalized_y,
            )

            if not _finite_array(
                projected_x
            ):
                raise ValueError(
                    "Projeção produziu valores inválidos."
                )

            # -----------------------------------------------------------------
            # Mantém somente pontos dentro da imagem.
            # -----------------------------------------------------------------

            inside = (
                (projected_x >= 0.0)
                & (
                    projected_x
                    <= float(
                        image_width - 1
                    )
                )
                & (
                    projected_y >= 0.0
                )
                & (
                    projected_y
                    <= float(
                        image_height - 1
                    )
                )
            )

            projected_x = (
                projected_x[inside]
            )

            projected_y = (
                projected_y[inside]
            )

            if len(projected_x) < 2:
                raise ValueError(
                    "A projeção não possui pontos suficientes dentro da imagem."
                )

            mean_confidence = _clip01(
                float(
                    np.mean(
                        confidence_values
                    )
                )
            )

            vertical_span = (
                source_y_max
                - source_y_min
            )

            confidence = self._confidence(
                point_count=len(x),
                vertical_span=vertical_span,
                fit_error=fit_error,
                point_confidence=mean_confidence,
            )

            if confidence < self.min_confidence:
                raise ValueError(
                    "Confiança da projeção abaixo do limite."
                )

            quality = self._quality(
                confidence
            )

            if quality == ProjectionQuality.NONE:
                raise ValueError(
                    "Qualidade da projeção inválida."
                )

            absolute_coefficients = (
                self._to_absolute_coefficients(
                    coefficients,
                    y_center,
                    y_scale,
                )
            )

            polynomial = LanePolynomial(
                a=absolute_coefficients[0],
                b=absolute_coefficients[1],
                c=absolute_coefficients[2],
                d=absolute_coefficients[3],
                valid=True,
                fit_error=fit_error,
                sample_count=len(x),
                confidence=confidence,
                y_min=source_y_min,
                y_max=source_y_max,
            )

            result_points = [
                LanePoint(
                    x=float(px),
                    y=float(py),
                    confidence=confidence,
                    valid=True,
                )
                for px, py in zip(
                    projected_x,
                    projected_y,
                )
            ]

            result_points.sort(
                key=lambda point: point.y
            )

            actual_y_min = float(
                result_points[0].y
            )

            actual_y_max = float(
                result_points[-1].y
            )

            extrapolated = bool(
                actual_y_min < source_y_min
                or actual_y_max > source_y_max
            )

            projection = LaneProjection(
                polynomial=polynomial,
                points=result_points,
                quality=quality,
                extrapolated=extrapolated,
                valid=True,
                horizon_y=actual_y_min,
            )

            return LaneProjectionResult(
                points=result_points,
                coefficients=tuple(
                    float(value)
                    for value in coefficients
                ),
                degree=degree,
                fitted=True,
                extrapolated=projection.extrapolated,
                confidence=confidence,
                fit_error=fit_error,
                source_y_min=source_y_min,
                source_y_max=source_y_max,
                projected_y_min=actual_y_min,
                projected_y_max=actual_y_max,
                valid=True,
                error=None,
            )

        except Exception as exc:

            logger.debug(
                "[LANE PROJECTION] Projeção rejeitada: %s",
                exc,
            )

            return LaneProjectionResult(
                points=[],
                coefficients=(),
                degree=0,
                fitted=False,
                extrapolated=False,
                confidence=0.0,
                fit_error=float("inf"),
                source_y_min=0.0,
                source_y_max=0.0,
                projected_y_min=0.0,
                projected_y_max=0.0,
                valid=False,
                error=(
                    f"{type(exc).__name__}: {exc}"
                ),
            )

    # =========================================================================
    # LaneModel
    # =========================================================================

    def project_model(
        self,
        model: LaneModel,
        image_height: int = 480,
        image_width: int = 640,
    ) -> LaneProjectionResult:
        """
        Projeta uma linha contida em um LaneModel.
        """

        if model is None:
            return LaneProjectionResult(
                valid=False,
                error="LaneModel é None.",
            )

        try:
            line = model.line
        except AttributeError:
            return LaneProjectionResult(
                valid=False,
                error="LaneModel não possui atributo 'line'.",
            )

        if line is None:
            return LaneProjectionResult(
                valid=False,
                error="LaneModel não possui LaneLine.",
            )

        try:
            points = line.points
        except AttributeError:
            return LaneProjectionResult(
                valid=False,
                error="LaneLine não possui pontos.",
            )

        return self.project(
            points,
            image_height=image_height,
            image_width=image_width,
        )

    # =========================================================================
    # AVALIAÇÃO
    # =========================================================================

    @staticmethod
    def evaluate_polynomial(
        polynomial: LanePolynomial,
        y: float,
    ) -> Optional[float]:
        """
        Avalia um LanePolynomial em Y.
        """

        if polynomial is None:
            return None

        try:
            if not polynomial.valid:
                return None
        except AttributeError:
            return None

        if not _is_finite_scalar(y):
            return None

        try:
            value = polynomial.evaluate(
                float(y)
            )
        except (
            AttributeError,
            TypeError,
            ValueError,
            OverflowError,
        ):
            return None

        if not np.isfinite(value):
            return None

        return float(value)

    # =========================================================================
    # AMOSTRAGEM DE POLINÔMIO
    # =========================================================================

    @staticmethod
    def sample_polynomial(
        polynomial: LanePolynomial,
        y_min: float,
        y_max: float,
        step: float = DEFAULT_SAMPLE_STEP,
        confidence: Optional[float] = None,
    ) -> List[LanePoint]:
        """
        Amostra um LanePolynomial em um intervalo vertical.
        """

        if polynomial is None:
            return []

        try:
            if not polynomial.valid:
                return []
        except AttributeError:
            return []

        if not (
            _is_finite_scalar(y_min)
            and _is_finite_scalar(y_max)
        ):
            return []

        y_min = float(y_min)
        y_max = float(y_max)

        if y_max <= y_min:
            return []

        try:
            step = float(step)
        except (TypeError, ValueError):
            return []

        if not np.isfinite(step):
            return []

        step = max(
            1.0,
            step,
        )

        ys = np.arange(
            y_min,
            y_max + step,
            step,
            dtype=np.float64,
        )

        if ys.size == 0:
            ys = np.asarray(
                [y_min, y_max],
                dtype=np.float64,
            )

        elif ys[-1] < y_max:
            ys = np.append(
                ys,
                y_max,
            )

        else:
            ys[-1] = min(
                ys[-1],
                y_max,
            )

        if confidence is None:
            try:
                point_confidence = _clip01(
                    polynomial.confidence
                )
            except AttributeError:
                point_confidence = 0.0
        else:
            point_confidence = _clip01(
                confidence
            )

        result: List[LanePoint] = []

        for y in ys:

            try:
                x = polynomial.evaluate(
                    float(y)
                )
            except (
                AttributeError,
                TypeError,
                ValueError,
                OverflowError,
            ):
                continue

            if not np.isfinite(x):
                continue

            result.append(
                LanePoint(
                    x=float(x),
                    y=float(y),
                    confidence=point_confidence,
                    valid=True,
                )
            )

        return result


# =============================================================================
# FUNÇÕES DE CONVENIÊNCIA
# =============================================================================

def project_lane(
    points: Sequence[LanePoint],
    image_height: int = 480,
    image_width: int = 640,
    **kwargs,
) -> LaneProjectionResult:

    engine = LaneProjectionEngine(
        **kwargs
    )

    return engine.project(
        points,
        image_height=image_height,
        image_width=image_width,
    )


def project_lane_model(
    model: LaneModel,
    image_height: int = 480,
    image_width: int = 640,
    **kwargs,
) -> LaneProjectionResult:

    engine = LaneProjectionEngine(
        **kwargs
    )

    return engine.project_model(
        model,
        image_height=image_height,
        image_width=image_width,
    )


def create_default_projection(
    **kwargs,
) -> LaneProjectionEngine:

    return LaneProjectionEngine(
        **kwargs
    )


# =============================================================================
# COMPATIBILIDADE
# =============================================================================

LaneProjector = LaneProjectionEngine


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "LaneProjectionResult",
    "LaneProjectionEngine",
    "LaneProjector",
    "project_lane",
    "project_lane_model",
    "create_default_projection",
    "DEFAULT_MIN_POINTS",
    "DEFAULT_MIN_VERTICAL_SPAN",
    "DEFAULT_MAX_FIT_ERROR",
    "DEFAULT_MAX_EXTRAPOLATION",
    "DEFAULT_SAMPLE_STEP",
    "DEFAULT_POLYNOMIAL_DEGREE",
    "DEFAULT_MIN_CONFIDENCE",
    "DEFAULT_MAX_PROJECTION_DISTANCE",
]