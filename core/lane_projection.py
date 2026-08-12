"""
core/lane_projection.py

Projeção geométrica das linhas de faixa.

Responsabilidade
----------------
Receber pontos de lane já detectados/tratados pelo tracker e:

    pontos observados
            ↓
    ajuste polinomial cúbico
            ↓
    validação matemática
            ↓
    projeção da continuidade
            ↓
    LaneProjectionResult

Este módulo NÃO:

- identifica qual faixa o veículo ocupa;
- calcula erro lateral;
- decide intervenção;
- controla volante;
- decide se o ADAS deve atuar;
- realiza inferência YOLOP.

A responsabilidade deste módulo é exclusivamente geométrica.

Modelo
------
A faixa é representada por:

    x = f(y)

utilizando:

    x(y) = a*y³ + b*y² + c*y + d

O eixo Y representa profundidade vertical da imagem.

A representação x(y) é preferida aqui porque, em uma câmera frontal,
uma linha de faixa normalmente pode ser descrita como uma função de X
em relação à posição vertical Y da imagem.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .lane_types import LanePoint


logger = logging.getLogger(__name__)


# ============================================================================
# ENUMERAÇÕES
# ============================================================================


class ProjectionStatus(str, Enum):
    """
    Estado da projeção geométrica.
    """

    INVALID = "invalid"
    OBSERVED_ONLY = "observed_only"
    PROJECTED = "projected"


# ============================================================================
# CONFIGURAÇÃO
# ============================================================================


@dataclass(frozen=True)
class LaneProjectionConfig:
    """
    Configuração matemática da projeção.

    Os valores são deliberadamente conservadores.

    O sistema deve preferir NÃO projetar uma lane a projetar uma
    lane incorretamente.
    """

    # Grau máximo do polinômio.
    polynomial_degree: int = 3

    # Quantidade mínima absoluta de pontos.
    min_points: int = 8

    # Quantidade mínima de níveis Y diferentes.
    min_unique_y: int = 6

    # Número mínimo de pontos próximos da região inferior.
    min_near_points: int = 3

    # Fração mínima da imagem vertical que precisa possuir dados.
    min_vertical_coverage: float = 0.12

    # Erro RMS máximo permitido em pixels normalizados.
    max_fit_error_normalized: float = 0.035

    # Curvatura/variação máxima permitida entre regiões consecutivas.
    max_lateral_change_normalized: float = 0.35

    # Quantidade de amostras usadas na projeção.
    projection_samples: int = 40

    # Quantidade de pontos observados que serão preservados.
    observed_samples: int = 60

    # Quanto da imagem pode ser extrapolado além dos dados observados.
    max_extrapolation_ratio: float = 0.55

    # Evita polinômios numericamente instáveis.
    condition_limit: float = 1.0e8

    # Regularização pequena usada apenas para proteção numérica.
    ridge_lambda: float = 1.0e-8


# ============================================================================
# RESULTADOS
# ============================================================================


@dataclass
class PolynomialModel:
    """
    Modelo x(y).

    coefficients:

        [a, b, c, d]

    representando:

        x = a*y³ + b*y² + c*y + d

    O Y usado pelo modelo é normalizado para [-1, 1].
    """

    coefficients: np.ndarray

    degree: int

    y_min: float
    y_max: float

    x_min: float
    x_max: float

    fit_error: float

    condition_number: float

    valid: bool = True

    def evaluate(self, y: float | np.ndarray) -> float | np.ndarray:
        """
        Avalia x(y).
        """

        y_array = np.asarray(
            y,
            dtype=np.float64,
        )

        normalized_y = self._normalize_y(
            y_array
        )

        result = np.polyval(
            self.coefficients,
            normalized_y,
        )

        if np.isscalar(y):
            return float(result)

        return result

    def derivative(
        self,
        y: float | np.ndarray,
    ) -> float | np.ndarray:
        """
        Calcula dx/dy.
        """

        y_array = np.asarray(
            y,
            dtype=np.float64,
        )

        normalized_y = self._normalize_y(
            y_array
        )

        derivative_coefficients = np.polyder(
            self.coefficients
        )

        result = np.polyval(
            derivative_coefficients,
            normalized_y,
        )

        if np.isscalar(y):
            return float(result)

        return result

    def _normalize_y(
        self,
        y: np.ndarray,
    ) -> np.ndarray:
        """
        Converte Y de pixels para [-1, 1].
        """

        span = max(
            self.y_max - self.y_min,
            1.0,
        )

        return (
            2.0
            * (y - self.y_min)
            / span
            - 1.0
        )


@dataclass
class ProjectedLanePoint:
    """
    Ponto produzido pela projeção.

    observed:
        True  -> veio da detecção.

        False -> foi calculado pelo modelo.
    """

    x: float
    y: float

    confidence: float

    observed: bool

    valid: bool = True


@dataclass
class LaneProjectionResult:
    """
    Resultado completo da projeção de uma lane.
    """

    points: List[ProjectedLanePoint] = field(
        default_factory=list
    )

    observed_points: List[ProjectedLanePoint] = field(
        default_factory=list
    )

    projected_points: List[ProjectedLanePoint] = field(
        default_factory=list
    )

    model: Optional[PolynomialModel] = None

    status: ProjectionStatus = (
        ProjectionStatus.INVALID
    )

    confidence: float = 0.0

    fit_error: float = float("inf")

    vertical_coverage: float = 0.0

    projected: bool = False

    valid: bool = False

    reason: Optional[str] = None


# ============================================================================
# PROJETOR
# ============================================================================


class LaneProjection:
    """
    Projetor geométrico de lanes.

    Fluxo:

        LanePoint[]
             ↓
        filtragem
             ↓
        normalização
             ↓
        ajuste cúbico
             ↓
        validação
             ↓
        extrapolação controlada
    """

    def __init__(
        self,
        config: Optional[
            LaneProjectionConfig
        ] = None,
    ) -> None:

        self.config = (
            config
            if config is not None
            else LaneProjectionConfig()
        )

        self.last_result: Optional[
            LaneProjectionResult
        ] = None

    # ========================================================================
    # API PRINCIPAL
    # ========================================================================

    def project(
        self,
        lane_points: Sequence[LanePoint],
        image_height: int,
        image_width: int,
        target_y_min: Optional[float] = None,
        target_y_max: Optional[float] = None,
    ) -> LaneProjectionResult:
        """
        Ajusta e projeta uma lane.

        target_y_min / target_y_max definem a região final desejada.

        Se não forem fornecidos:

            0 -> image_height - 1

        será utilizado.
        """

        result = LaneProjectionResult()

        if image_height <= 0 or image_width <= 0:

            result.reason = (
                "Dimensões da imagem inválidas."
            )

            self.last_result = result

            return result

        if not lane_points:

            result.reason = (
                "Nenhum ponto recebido."
            )

            self.last_result = result

            return result

        target_y_min = (
            0.0
            if target_y_min is None
            else float(target_y_min)
        )

        target_y_max = (
            float(image_height - 1)
            if target_y_max is None
            else float(target_y_max)
        )

        if target_y_max <= target_y_min:

            result.reason = (
                "Intervalo Y inválido."
            )

            self.last_result = result

            return result

        # ---------------------------------------------------------------
        # Filtragem
        # ---------------------------------------------------------------

        points = self._sanitize_points(
            lane_points,
            image_width=image_width,
            image_height=image_height,
        )

        if len(points) < self.config.min_points:

            result.reason = (
                "Quantidade insuficiente de pontos."
            )

            self.last_result = result

            return result

        # ---------------------------------------------------------------
        # Cobertura vertical
        # ---------------------------------------------------------------

        y_values = np.asarray(
            [point.y for point in points],
            dtype=np.float64,
        )

        vertical_coverage = (
            self._vertical_coverage(
                y_values,
                image_height,
            )
        )

        result.vertical_coverage = (
            vertical_coverage
        )

        if vertical_coverage < (
            self.config.min_vertical_coverage
        ):

            result.reason = (
                "Cobertura vertical insuficiente."
            )

            self.last_result = result

            return result

        # ---------------------------------------------------------------
        # Ajuste
        # ---------------------------------------------------------------

        model = self._fit_polynomial(
            points
        )

        if model is None:

            result.reason = (
                "Não foi possível ajustar "
                "o modelo polinomial."
            )

            self.last_result = result

            return result

        result.model = model
        result.fit_error = model.fit_error

        # ---------------------------------------------------------------
        # Validação
        # ---------------------------------------------------------------

        valid, reason = self._validate_model(
            model=model,
            points=points,
            image_width=image_width,
            image_height=image_height,
        )

        if not valid:

            result.reason = reason

            self.last_result = result

            return result

        # ---------------------------------------------------------------
        # Pontos observados
        # ---------------------------------------------------------------

        observed_points = (
            self._build_observed_points(
                points
            )
        )

        result.observed_points = (
            observed_points
        )

        # ---------------------------------------------------------------
        # Definição da região de projeção
        # ---------------------------------------------------------------

        observed_y_min = float(
            np.min(y_values)
        )

        observed_y_max = float(
            np.max(y_values)
        )

        projection_min = max(
            target_y_min,
            observed_y_min,
        )

        projection_max = min(
            target_y_max,
            observed_y_max,
        )

        # ---------------------------------------------------------------
        # Extrapolação.
        #
        # Só projetamos além da região realmente observada
        # de forma limitada.
        # ---------------------------------------------------------------

        observed_span = max(
            observed_y_max
            - observed_y_min,
            1.0,
        )

        maximum_extra = (
            observed_span
            * self.config.max_extrapolation_ratio
        )

        lower_extra = min(
            maximum_extra,
            max(
                0.0,
                observed_y_min
                - target_y_min,
            ),
        )

        upper_extra = min(
            maximum_extra,
            max(
                0.0,
                target_y_max
                - observed_y_max,
            ),
        )

        projection_min = max(
            target_y_min,
            observed_y_min - lower_extra,
        )

        projection_max = min(
            target_y_max,
            observed_y_max + upper_extra,
        )

        # ---------------------------------------------------------------
        # Gerar projeção.
        # ---------------------------------------------------------------

        projected_points = (
            self._generate_projection(
                model=model,
                y_min=projection_min,
                y_max=projection_max,
                image_width=image_width,
                image_height=image_height,
                observed_y_min=observed_y_min,
                observed_y_max=observed_y_max,
            )
        )

        # ---------------------------------------------------------------
        # Resultado
        # ---------------------------------------------------------------

        result.projected_points = (
            projected_points
        )

        result.points = (
            observed_points
            + projected_points
        )

        result.points.sort(
            key=lambda point: point.y
        )

        result.projected = bool(
            projected_points
        )

        result.status = (
            ProjectionStatus.PROJECTED
            if projected_points
            else ProjectionStatus.OBSERVED_ONLY
        )

        result.valid = True

        result.confidence = (
            self._calculate_confidence(
                model=model,
                point_count=len(points),
                vertical_coverage=vertical_coverage,
            )
        )

        result.reason = None

        self.last_result = result

        return result

    # ========================================================================
    # FILTRAGEM
    # ========================================================================

    @staticmethod
    def _sanitize_points(
        points: Sequence[LanePoint],
        image_width: int,
        image_height: int,
    ) -> List[LanePoint]:

        sanitized = []

        for point in points:

            if point is None:
                continue

            if not getattr(
                point,
                "valid",
                True,
            ):
                continue

            try:
                x = float(point.x)
                y = float(point.y)
            except (
                TypeError,
                ValueError,
            ):
                continue

            if not (
                np.isfinite(x)
                and np.isfinite(y)
            ):
                continue

            if (
                x < 0.0
                or x > image_width - 1
            ):
                continue

            if (
                y < 0.0
                or y > image_height - 1
            ):
                continue

            sanitized.append(
                point
            )

        sanitized.sort(
            key=lambda point: point.y
        )

        return sanitized

    # ========================================================================
    # COBERTURA
    # ========================================================================

    @staticmethod
    def _vertical_coverage(
        y_values: np.ndarray,
        image_height: int,
    ) -> float:

        if y_values.size == 0:
            return 0.0

        span = (
            float(np.max(y_values))
            - float(np.min(y_values))
        )

        return float(
            np.clip(
                span
                / max(
                    image_height - 1,
                    1,
                ),
                0.0,
                1.0,
            )
        )

    # ========================================================================
    # AJUSTE POLINOMIAL
    # ========================================================================

    def _fit_polynomial(
        self,
        points: Sequence[LanePoint],
    ) -> Optional[PolynomialModel]:
        """
        Ajuste cúbico x(y).

        O Y é normalizado antes do ajuste para reduzir problemas
        numéricos.
        """

        if len(points) < self.config.min_points:
            return None

        x = np.asarray(
            [float(point.x) for point in points],
            dtype=np.float64,
        )

        y = np.asarray(
            [float(point.y) for point in points],
            dtype=np.float64,
        )

        y_min = float(np.min(y))
        y_max = float(np.max(y))

        x_min = float(np.min(x))
        x_max = float(np.max(x))

        unique_y = np.unique(
            np.round(y, decimals=3)
        )

        if (
            unique_y.size
            < self.config.min_unique_y
        ):
            return None

        y_span = max(
            y_max - y_min,
            1.0,
        )

        y_normalized = (
            2.0
            * (y - y_min)
            / y_span
            - 1.0
        )

        degree = min(
            self.config.polynomial_degree,
            len(points) - 1,
        )

        # ---------------------------------------------------------------
        # Matriz de Vandermonde.
        # ---------------------------------------------------------------

        A = np.vander(
            y_normalized,
            N=degree + 1,
            increasing=False,
        )

        try:

            condition_number = float(
                np.linalg.cond(A)
            )

        except np.linalg.LinAlgError:

            return None

        if (
            not np.isfinite(condition_number)
            or condition_number
            > self.config.condition_limit
        ):
            return None

        try:

            # Pequena regularização para maior estabilidade.
            ATA = A.T @ A

            regularization = (
                self.config.ridge_lambda
                * np.eye(
                    ATA.shape[0],
                    dtype=np.float64,
                )
            )

            ATy = A.T @ x

            coefficients = np.linalg.solve(
                ATA + regularization,
                ATy,
            )

        except np.linalg.LinAlgError:

            try:

                coefficients = np.linalg.lstsq(
                    A,
                    x,
                    rcond=None,
                )[0]

            except np.linalg.LinAlgError:

                return None

        predicted = np.polyval(
            coefficients,
            y_normalized,
        )

        residuals = (
            predicted - x
        )

        rms_error = float(
            np.sqrt(
                np.mean(
                    residuals ** 2
                )
            )
        )

        # Normalização do erro pela largura observada
        # da imagem.
        x_span = max(
            x_max - x_min,
            1.0,
        )

        normalized_error = (
            rms_error
            / x_span
        )

        return PolynomialModel(
            coefficients=np.asarray(
                coefficients,
                dtype=np.float64,
            ),
            degree=degree,
            y_min=y_min,
            y_max=y_max,
            x_min=x_min,
            x_max=x_max,
            fit_error=normalized_error,
            condition_number=condition_number,
            valid=True,
        )

    # ========================================================================
    # VALIDAÇÃO
    # ========================================================================

    def _validate_model(
        self,
        model: PolynomialModel,
        points: Sequence[LanePoint],
        image_width: int,
        image_height: int,
    ) -> Tuple[bool, Optional[str]]:
        """
        Impede extrapolações matematicamente absurdas.

        Um polinômio pode ajustar muito bem poucos pontos e ainda
        produzir uma curva completamente errada fora deles.
        """

        if not model.valid:
            return False, "Modelo inválido."

        if not np.isfinite(
            model.fit_error
        ):
            return False, "Erro do ajuste inválido."

        if (
            model.fit_error
            > self.config.max_fit_error_normalized
        ):
            return (
                False,
                "Erro do ajuste polinomial excessivo.",
            )

        # ---------------------------------------------------------------
        # Teste de sanidade dentro da região observada.
        # ---------------------------------------------------------------

        test_y = np.linspace(
            model.y_min,
            model.y_max,
            25,
        )

        test_x = model.evaluate(
            test_y
        )

        if not np.all(
            np.isfinite(test_x)
        ):
            return (
                False,
                "Modelo produz valores não finitos.",
            )

        # ---------------------------------------------------------------
        # Não permitir que a curva observada saia da imagem.
        # Pequena margem é permitida.
        # ---------------------------------------------------------------

        margin = (
            image_width
            * 0.15
        )

        if np.any(
            test_x
            < -margin
        ) or np.any(
            test_x
            > image_width + margin
        ):
            return (
                False,
                "Modelo extrapola lateralmente "
                "durante a região observada.",
            )

        # ---------------------------------------------------------------
        # Verificar deslocamento excessivo.
        # ---------------------------------------------------------------

        x_range = (
            float(np.max(test_x))
            - float(np.min(test_x))
        )

        if (
            x_range
            > image_width
            * (
                1.0
                + self.config.max_lateral_change_normalized
            )
        ):
            return (
                False,
                "Variação lateral incompatível.",
            )

        return True, None

    # ========================================================================
    # PONTOS OBSERVADOS
    # ========================================================================

    def _build_observed_points(
        self,
        points: Sequence[LanePoint],
    ) -> List[ProjectedLanePoint]:

        if not points:
            return []

        max_samples = max(
            1,
            self.config.observed_samples,
        )

        if len(points) <= max_samples:
            selected = list(points)

        else:

            indices = np.linspace(
                0,
                len(points) - 1,
                max_samples,
            ).astype(int)

            selected = [
                points[index]
                for index in indices
            ]

        result = []

        for point in selected:

            confidence = float(
                np.clip(
                    getattr(
                        point,
                        "confidence",
                        1.0,
                    ),
                    0.0,
                    1.0,
                )
            )

            result.append(
                ProjectedLanePoint(
                    x=float(point.x),
                    y=float(point.y),
                    confidence=confidence,
                    observed=True,
                    valid=True,
                )
            )

        return result

    # ========================================================================
    # PROJEÇÃO
    # ========================================================================

    def _generate_projection(
        self,
        model: PolynomialModel,
        y_min: float,
        y_max: float,
        image_width: int,
        image_height: int,
        observed_y_min: float,
        observed_y_max: float,
    ) -> List[ProjectedLanePoint]:
        """
        Gera pontos da curva.

        A confiança cai progressivamente conforme nos afastamos
        da região realmente observada.
        """

        if y_max <= y_min:
            return []

        samples = max(
            2,
            self.config.projection_samples,
        )

        y_values = np.linspace(
            y_min,
            y_max,
            samples,
        )

        x_values = model.evaluate(
            y_values
        )

        result = []

        observed_span = max(
            observed_y_max
            - observed_y_min,
            1.0,
        )

        for y, x in zip(
            y_values,
            x_values,
        ):

            y = float(y)
            x = float(x)

            if not (
                np.isfinite(x)
                and np.isfinite(y)
            ):
                continue

            # -----------------------------------------------------------
            # Não projetamos indefinidamente.
            # -----------------------------------------------------------

            if (
                x < 0.0
                or x > image_width - 1
            ):
                continue

            if (
                y < 0.0
                or y > image_height - 1
            ):
                continue

            # -----------------------------------------------------------
            # Distância normalizada em relação à região observada.
            # -----------------------------------------------------------

            if y < observed_y_min:

                distance = (
                    observed_y_min - y
                )

            elif y > observed_y_max:

                distance = (
                    y - observed_y_max
                )

            else:

                distance = 0.0

            extrapolation_ratio = min(
                1.0,
                distance
                / observed_span,
            )

            # Confiança da projeção.
            #
            # Dentro da região observada:
            # alta.
            #
            # Fora:
            # decai progressivamente.
            confidence = (
                1.0
                - 0.65
                * extrapolation_ratio
            )

            confidence *= (
                1.0
                - min(
                    model.fit_error,
                    1.0,
                )
            )

            confidence = float(
                np.clip(
                    confidence,
                    0.0,
                    1.0,
                )
            )

            result.append(
                ProjectedLanePoint(
                    x=x,
                    y=y,
                    confidence=confidence,
                    observed=False,
                    valid=True,
                )
            )

        return result

    # ========================================================================
    # CONFIANÇA
    # ========================================================================

    @staticmethod
    def _calculate_confidence(
        model: PolynomialModel,
        point_count: int,
        vertical_coverage: float,
    ) -> float:

        point_factor = float(
            np.clip(
                point_count / 30.0,
                0.0,
                1.0,
            )
        )

        coverage_factor = float(
            np.clip(
                vertical_coverage,
                0.0,
                1.0,
            )
        )

        fit_factor = float(
            np.clip(
                1.0
                - model.fit_error,
                0.0,
                1.0,
            )
        )

        confidence = (
            0.30 * point_factor
            + 0.35 * coverage_factor
            + 0.35 * fit_factor
        )

        return float(
            np.clip(
                confidence,
                0.0,
                1.0,
            )
        )


# ============================================================================
# FACTORY
# ============================================================================


def create_default_projection(
    **kwargs,
) -> LaneProjection:

    config = LaneProjectionConfig(
        **kwargs
    )

    return LaneProjection(
        config=config
    )


__all__ = [
    "ProjectionStatus",
    "LaneProjectionConfig",
    "PolynomialModel",
    "ProjectedLanePoint",
    "LaneProjectionResult",
    "LaneProjection",
    "create_default_projection",
]