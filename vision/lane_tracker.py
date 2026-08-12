"""
vision/lane_tracker.py

Lane Tracker para o sistema Forza Assistente.

Responsabilidade:

    LaneDetectionResult
            |
            v
    identificação das faixas
            |
            v
    associação temporal
            |
            v
    ajuste geométrico
            |
            v
    projeção das faixas
            |
            v
    LaneTrackingResult

O tracker NÃO é responsável por:

- controle do volante;
- decisão de intervenção;
- cálculo do estado ADAS;
- detecção de veículos;
- inferência do YOLOP;
- captura da tela.

Essas responsabilidades pertencem a outros módulos.

Modelo geométrico:

    x(y) = a*y³ + b*y² + c*y + d

O eixo principal utilizado pelo tracker é Y da imagem.

A utilização de um polinômio cúbico permite representar:

- retas;
- curvas suaves;
- mudanças progressivas de curvatura.

IMPORTANTE:

O tracker nunca deve transformar ausência de informação
em uma faixa artificial com confiança alta.

Uma faixa projetada possui:

    observed = False
    projected = True

e sua confiança é degradada progressivamente com o tempo.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .yolop_detector import LaneDetectionResult, LanePoint

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

MAX_LANES = 4

# Até 3 faixas de tráfego + acostamento.
MAX_TRAVEL_LANES = 3
MAX_SHOULDERS = 1

POLYNOMIAL_DEGREE = 3

MIN_FIT_POINTS = 6

# Número máximo de frames em que uma faixa pode sobreviver
# apenas através de projeção/histórico.
MAX_MISSED_FRAMES = 8

# Após este número de frames, a confiança cai fortemente.
CONFIDENCE_DECAY = 0.82

# Confiança mínima para considerar uma faixa rastreável.
MIN_TRACK_CONFIDENCE = 0.30

# Confiança mínima para permitir projeção.
MIN_PROJECTION_CONFIDENCE = 0.55

# Resolução vertical utilizada para produzir a curva.
DEFAULT_SAMPLE_COUNT = 32

# Resíduo máximo aceitável do ajuste normalizado.
MAX_NORMALIZED_FIT_ERROR = 0.08

# Distância horizontal máxima entre a previsão anterior
# e uma nova detecção para permitir associação.
MAX_ASSOCIATION_DISTANCE_RATIO = 0.16

# Penalidade por mudança brusca de posição.
POSITION_COST_WEIGHT = 1.0

# Penalidade por mudança brusca de inclinação.
SLOPE_COST_WEIGHT = 0.25

# Quantidade de histórico geométrico armazenado.
HISTORY_SIZE = 12


# ============================================================================
# TIPOS
# ============================================================================

@dataclass
class PolynomialModel:
    """
    Modelo x(y) de uma faixa.

    coefficients:

        [a, b, c, d]

    representando:

        x = a*y³ + b*y² + c*y + d

    Os coeficientes usam Y normalizado entre 0 e 1.
    """

    coefficients: np.ndarray

    degree: int = POLYNOMIAL_DEGREE

    fit_error: float = float("inf")

    valid: bool = False

    def evaluate(
        self,
        y_normalized: np.ndarray | float,
    ) -> np.ndarray | float:
        """Avalia o polinômio."""

        return np.polyval(
            self.coefficients,
            y_normalized,
        )

    def derivative(
        self,
        y_normalized: np.ndarray | float,
    ) -> np.ndarray | float:
        """Calcula dx/dy."""

        derivative = np.polyder(
            self.coefficients
        )

        return np.polyval(
            derivative,
            y_normalized,
        )


@dataclass
class TrackedLane:
    """
    Estado temporal de uma faixa.

    lane_id:
        Identidade persistente da faixa.

    points:
        Pontos atualmente disponíveis.

    polynomial:
        Modelo cúbico da faixa.

    observed:
        Pelo menos parte da faixa foi realmente observada.

    projected:
        Parte da faixa foi completada matematicamente.

    confidence:
        Confiança atual do tracking.

    missed_frames:
        Frames consecutivos sem observação suficiente.
    """

    lane_id: int

    points: List[LanePoint] = field(
        default_factory=list
    )

    polynomial: Optional[PolynomialModel] = None

    observed: bool = False

    projected: bool = False

    confidence: float = 0.0

    missed_frames: int = 0

    age: int = 0

    last_y_min: float = 0.0

    last_y_max: float = 0.0

    history: List[
        PolynomialModel
    ] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return (
            self.polynomial is not None
            and self.polynomial.valid
            and self.confidence
            >= MIN_TRACK_CONFIDENCE
        )


@dataclass
class LaneTrackingResult:
    """
    Resultado completo do tracking.

    lanes:
        Faixas rastreadas ordenadas da esquerda
        para a direita.

    active_lane_id:
        Faixa que representa a faixa atualmente
        ocupada pelo veículo, quando conhecida.

    active_lane_index:
        Índice da faixa ocupada na lista ordenada.

    confidence:
        Confiança global do tracking.

    can_project:
        Indica se existem informações suficientes
        para projeção confiável.

    safe_for_control:
        Indica que o tracking possui qualidade mínima
        para ser utilizado posteriormente pelo ADAS.

    observed_lane_count:
        Quantidade de faixas realmente observadas.

    projected_lane_count:
        Quantidade de faixas que possuem trechos
        projetados.
    """

    lanes: List[TrackedLane] = field(
        default_factory=list
    )

    active_lane_id: Optional[int] = None

    active_lane_index: Optional[int] = None

    confidence: float = 0.0

    can_project: bool = False

    safe_for_control: bool = False

    observed_lane_count: int = 0

    projected_lane_count: int = 0

    frame_width: int = 0

    frame_height: int = 0

    error: Optional[str] = None

    @property
    def valid(self) -> bool:
        return (
            self.safe_for_control
        )

    @property
    def lane_count(self) -> int:
        return len(self.lanes)


# ============================================================================
# TRACKER
# ============================================================================

class LaneTracker:
    """
    Rastreador temporal das faixas.

    O detector fornece observações.

    O tracker fornece continuidade.

    Fluxo:

        detector
            ↓
        observações
            ↓
        associação temporal
            ↓
        fitting cúbico
            ↓
        suavização
            ↓
        projeção
            ↓
        resultado temporal
    """

    def __init__(
        self,
        max_lanes: int = MAX_LANES,
        min_fit_points: int = MIN_FIT_POINTS,
        max_missed_frames: int = MAX_MISSED_FRAMES,
        confidence_decay: float = CONFIDENCE_DECAY,
        projection_confidence: float = (
            MIN_PROJECTION_CONFIDENCE
        ),
        sample_count: int = DEFAULT_SAMPLE_COUNT,
        history_size: int = HISTORY_SIZE,
    ) -> None:

        self.max_lanes = max(
            2,
            min(
                int(max_lanes),
                MAX_LANES,
            ),
        )

        self.min_fit_points = max(
            4,
            int(min_fit_points),
        )

        self.max_missed_frames = max(
            1,
            int(max_missed_frames),
        )

        self.confidence_decay = float(
            np.clip(
                confidence_decay,
                0.01,
                0.99,
            )
        )

        self.projection_confidence = float(
            np.clip(
                projection_confidence,
                0.0,
                1.0,
            )
        )

        self.sample_count = max(
            8,
            int(sample_count),
        )

        self.history_size = max(
            2,
            int(history_size),
        )

        self._tracks: Dict[
            int,
            TrackedLane,
        ] = {}

        self._next_lane_id = 0

        self._frame_index = 0

        self._last_frame_width = 0

        self._last_frame_height = 0

        self._last_result: Optional[
            LaneTrackingResult
        ] = None

    # ========================================================================
    # RESET
    # ========================================================================

    def reset(self) -> None:
        """Apaga completamente o histórico temporal."""

        self._tracks.clear()

        self._next_lane_id = 0

        self._frame_index = 0

        self._last_result = None

    # ========================================================================
    # UTILITÁRIOS
    # ========================================================================

    @staticmethod
    def _lane_center(
        lane: Sequence[LanePoint],
    ) -> Optional[float]:

        if not lane:
            return None

        xs = np.asarray(
            [
                point.x
                for point in lane
                if point.valid
                and np.isfinite(point.x)
                and np.isfinite(point.y)
            ],
            dtype=np.float64,
        )

        if xs.size == 0:
            return None

        return float(
            np.median(xs)
        )

    @staticmethod
    def _lane_y_range(
        lane: Sequence[LanePoint],
    ) -> Tuple[float, float]:

        ys = np.asarray(
            [
                point.y
                for point in lane
                if point.valid
                and np.isfinite(point.y)
            ],
            dtype=np.float64,
        )

        if ys.size == 0:
            return 0.0, 0.0

        return (
            float(np.min(ys)),
            float(np.max(ys)),
        )

    @staticmethod
    def _sanitize_lane(
        lane: Sequence[LanePoint],
    ) -> List[LanePoint]:

        result = []

        for point in lane:

            if not point.valid:
                continue

            if not np.isfinite(point.x):
                continue

            if not np.isfinite(point.y):
                continue

            result.append(point)

        result.sort(
            key=lambda point: point.y
        )

        return result

    # ========================================================================
    # POLINÔMIO
    # ========================================================================

    def _fit_polynomial(
        self,
        lane: Sequence[LanePoint],
        frame_height: int,
    ) -> Optional[PolynomialModel]:

        if len(lane) < self.min_fit_points:
            return None

        if frame_height <= 1:
            return None

        y = np.asarray(
            [
                point.y
                for point in lane
            ],
            dtype=np.float64,
        )

        x = np.asarray(
            [
                point.x
                for point in lane
            ],
            dtype=np.float64,
        )

        y_normalized = np.clip(
            y / float(frame_height - 1),
            0.0,
            1.0,
        )

        valid = (
            np.isfinite(x)
            & np.isfinite(y_normalized)
        )

        x = x[valid]
        y_normalized = (
            y_normalized[valid]
        )

        if x.size < self.min_fit_points:
            return None

        # Evita fitting cúbico em pontos praticamente
        # horizontais no eixo Y.
        if (
            np.ptp(y_normalized)
            < 0.05
        ):
            return None

        try:

            coefficients = np.polyfit(
                y_normalized,
                x,
                POLYNOMIAL_DEGREE,
            )

            predicted = np.polyval(
                coefficients,
                y_normalized,
            )

            residual = (
                np.sqrt(
                    np.mean(
                        (
                            predicted - x
                        ) ** 2
                    )
                )
            )

            frame_scale = max(
                1.0,
                float(
                    np.max(
                        x
                    )
                    - np.min(x)
                ),
            )

            normalized_error = (
                residual
                / frame_scale
            )

            return PolynomialModel(
                coefficients=np.asarray(
                    coefficients,
                    dtype=np.float64,
                ),
                degree=POLYNOMIAL_DEGREE,
                fit_error=float(
                    normalized_error
                ),
                valid=bool(
                    np.all(
                        np.isfinite(
                            coefficients
                        )
                    )
                ),
            )

        except (
            np.linalg.LinAlgError,
            ValueError,
            FloatingPointError,
        ):

            return None

    # ========================================================================
    # SUAVIZAÇÃO
    # ========================================================================

    def _smooth_model(
        self,
        track: TrackedLane,
        model: PolynomialModel,
    ) -> PolynomialModel:

        if (
            track.polynomial is None
            or not track.polynomial.valid
        ):
            return model

        previous = (
            track.polynomial.coefficients
        )

        current = (
            model.coefficients
        )

        # Suavização temporal.
        #
        # Não usamos uma média muito agressiva porque
        # isso atrasaria curvas.
        alpha = 0.55

        coefficients = (
            alpha * current
            + (1.0 - alpha) * previous
        )

        return PolynomialModel(
            coefficients=coefficients,
            degree=POLYNOMIAL_DEGREE,
            fit_error=model.fit_error,
            valid=True,
        )

    # ========================================================================
    # PROJEÇÃO
    # ========================================================================

    def _project_points(
        self,
        model: PolynomialModel,
        frame_width: int,
        frame_height: int,
        y_min: float,
        y_max: float,
    ) -> List[LanePoint]:

        if not model.valid:
            return []

        if frame_height <= 1:
            return []

        y_values = np.linspace(
            max(0.0, y_min),
            min(
                float(frame_height - 1),
                y_max,
            ),
            self.sample_count,
        )

        y_normalized = (
            y_values
            / float(frame_height - 1)
        )

        x_values = model.evaluate(
            y_normalized
        )

        points = []

        for x, y in zip(
            x_values,
            y_values,
        ):

            if not (
                np.isfinite(x)
                and np.isfinite(y)
            ):
                continue

            # Nunca permitir que a projeção
            # saia completamente da imagem.
            if (
                x < -0.15 * frame_width
                or x > 1.15 * frame_width
            ):
                continue

            points.append(
                LanePoint(
                    x=float(x),
                    y=float(y),
                    confidence=0.0,
                    valid=True,
                )
            )

        return points

    # ========================================================================
    # ASSOCIAÇÃO
    # ========================================================================

    def _predicted_x(
        self,
        track: TrackedLane,
        frame_height: int,
        y: float,
    ) -> Optional[float]:

        if (
            track.polynomial is None
            or not track.polynomial.valid
        ):
            return None

        if frame_height <= 1:
            return None

        yn = np.clip(
            y / float(frame_height - 1),
            0.0,
            1.0,
        )

        value = track.polynomial.evaluate(
            yn
        )

        if not np.isfinite(value):
            return None

        return float(value)

    def _association_cost(
        self,
        track: TrackedLane,
        lane: Sequence[LanePoint],
        frame_width: int,
        frame_height: int,
    ) -> float:

        center = self._lane_center(
            lane
        )

        if center is None:
            return float("inf")

        y_min, y_max = (
            self._lane_y_range(lane)
        )

        y_reference = (
            y_max
            if y_max > 0
            else frame_height * 0.75
        )

        predicted = self._predicted_x(
            track,
            frame_height,
            y_reference,
        )

        if predicted is None:

            # Faixa antiga sem modelo confiável.
            #
            # Utilizamos a última posição observada.
            if track.points:
                previous_center = (
                    self._lane_center(
                        track.points
                    )
                )

                if previous_center is not None:
                    predicted = previous_center

        if predicted is None:
            return float("inf")

        position_error = abs(
            center - predicted
        )

        normalized_position_error = (
            position_error
            / max(
                1.0,
                float(frame_width),
            )
        )

        # Inclinação aproximada.
        slope_cost = 0.0

        if (
            len(lane) >= 2
            and track.points
        ):

            new_first = lane[0]
            new_last = lane[-1]

            dy = (
                new_last.y
                - new_first.y
            )

            if abs(dy) > 1.0:

                new_slope = (
                    new_last.x
                    - new_first.x
                ) / dy

                old_first = track.points[0]
                old_last = track.points[-1]

                old_dy = (
                    old_last.y
                    - old_first.y
                )

                if abs(old_dy) > 1.0:

                    old_slope = (
                        old_last.x
                        - old_first.x
                    ) / old_dy

                    slope_cost = abs(
                        new_slope
                        - old_slope
                    )

        return (
            POSITION_COST_WEIGHT
            * normalized_position_error
            + SLOPE_COST_WEIGHT
            * slope_cost
        )

    # ========================================================================
    # CRIAÇÃO DE TRACK
    # ========================================================================

    def _create_track(
        self,
        lane: Sequence[LanePoint],
        frame_height: int,
    ) -> TrackedLane:

        lane = self._sanitize_lane(
            lane
        )

        model = self._fit_polynomial(
            lane,
            frame_height,
        )

        track = TrackedLane(
            lane_id=self._next_lane_id,
            points=list(lane),
            polynomial=model,
            observed=True,
            projected=False,
            confidence=(
                self._initial_confidence(
                    lane,
                    model,
                )
            ),
            missed_frames=0,
            age=1,
        )

        if model is not None:
            track.history.append(
                model
            )

        self._next_lane_id += 1

        return track

    @staticmethod
    def _initial_confidence(
        lane: Sequence[LanePoint],
        model: Optional[PolynomialModel],
    ) -> float:

        if not lane:
            return 0.0

        point_score = min(
            1.0,
            len(lane) / 20.0,
        )

        fit_score = 0.0

        if (
            model is not None
            and model.valid
        ):
            fit_score = max(
                0.0,
                1.0
                - model.fit_error
                / MAX_NORMALIZED_FIT_ERROR,
            )

        return float(
            np.clip(
                0.35 * point_score
                + 0.65 * fit_score,
                0.0,
                1.0,
            )
        )

    # ========================================================================
    # ATUALIZAÇÃO DE TRACK
    # ========================================================================

    def _update_track(
        self,
        track: TrackedLane,
        lane: Sequence[LanePoint],
        frame_width: int,
        frame_height: int,
    ) -> None:

        lane = self._sanitize_lane(
            lane
        )

        model = self._fit_polynomial(
            lane,
            frame_height,
        )

        if model is not None:

            model = self._smooth_model(
                track,
                model,
            )

            track.polynomial = model

            track.history.append(
                model
            )

            if len(track.history) > self.history_size:
                track.history = (
                    track.history[
                        -self.history_size:
                    ]
                )

        track.points = list(lane)

        track.observed = True

        track.projected = False

        track.missed_frames = 0

        track.age += 1

        observed_confidence = (
            self._initial_confidence(
                lane,
                model,
            )
        )

        # A confiança não sobe instantaneamente.
        # Isso evita que um frame isolado ruim
        # cause uma mudança brusca.
        track.confidence = float(
            np.clip(
                0.65 * track.confidence
                + 0.35
                * observed_confidence,
                0.0,
                1.0,
            )
        )

    # ========================================================================
    # PERDA TEMPORÁRIA
    # ========================================================================

    def _predict_track(
        self,
        track: TrackedLane,
        frame_width: int,
        frame_height: int,
    ) -> bool:

        if (
            track.polynomial is None
            or not track.polynomial.valid
        ):
            return False

        if (
            track.missed_frames
            >= self.max_missed_frames
        ):
            return False

        track.missed_frames += 1

        track.age += 1

        track.confidence *= (
            self.confidence_decay
        )

        # A projeção somente é válida enquanto
        # ainda houver confiança suficiente.
        if (
            track.confidence
            < self.projection_confidence
        ):
            track.projected = False
            return False

        y_min = (
            track.last_y_min
            if track.last_y_min > 0
            else 0.0
        )

        y_max = (
            track.last_y_max
            if track.last_y_max > 0
            else float(
                frame_height - 1
            )
        )

        projected = self._project_points(
            track.polynomial,
            frame_width,
            frame_height,
            y_min,
            y_max,
        )

        if len(projected) < self.min_fit_points:
            track.projected = False
            return False

        track.points = projected

        track.observed = False

        track.projected = True

        return True

    # ========================================================================
    # ORDENAÇÃO
    # ========================================================================

    def _sort_tracks(
        self,
        tracks: Sequence[TrackedLane],
    ) -> List[TrackedLane]:

        return sorted(
            tracks,
            key=lambda track: (
                self._lane_center(
                    track.points
                )
                if track.points
                else float("inf")
            ),
        )

    # ========================================================================
    # PROCESSAMENTO PRINCIPAL
    # ========================================================================

    def update(
        self,
        detection: LaneDetectionResult,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
    ) -> LaneTrackingResult:
        """
        Atualiza o tracker com uma nova detecção.

        Esta é a única função que o restante da arquitetura
        precisa chamar.
        """

        self._frame_index += 1

        if detection is None:
            return self._failure_result(
                "LaneDetectionResult é None."
            )

        width = int(
            frame_width
            or detection.input_width
            or 0
        )

        height = int(
            frame_height
            or detection.input_height
            or 0
        )

        self._last_frame_width = width
        self._last_frame_height = height

        try:

            observations = [
                self._sanitize_lane(
                    lane
                )
                for lane in detection.lanes
            ]

            observations = [
                lane
                for lane in observations
                if len(lane) >= 2
            ]

            # Nunca ultrapassar o número físico
            # máximo que definimos.
            observations = observations[
                :self.max_lanes
            ]

            existing_tracks = list(
                self._tracks.values()
            )

            matched_tracks = set()

            matched_observations = set()

            # --------------------------------------------------------------
            # Associação temporal.
            #
            # Primeiro encontramos os pares de menor custo.
            # --------------------------------------------------------------

            candidates = []

            for track in existing_tracks:

                for index, lane in enumerate(
                    observations
                ):

                    if (
                        track.lane_id
                        in matched_tracks
                    ):
                        continue

                    cost = (
                        self._association_cost(
                            track,
                            lane,
                            width,
                            height,
                        )
                    )

                    candidates.append(
                        (
                            cost,
                            track,
                            index,
                        )
                    )

            candidates.sort(
                key=lambda item: item[0]
            )

            max_cost = (
                MAX_ASSOCIATION_DISTANCE_RATIO
            )

            for (
                cost,
                track,
                observation_index,
            ) in candidates:

                if (
                    track.lane_id
                    in matched_tracks
                ):
                    continue

                if (
                    observation_index
                    in matched_observations
                ):
                    continue

                if cost > max_cost:
                    continue

                self._update_track(
                    track,
                    observations[
                        observation_index
                    ],
                    width,
                    height,
                )

                matched_tracks.add(
                    track.lane_id
                )

                matched_observations.add(
                    observation_index
                )

            # --------------------------------------------------------------
            # Novas faixas.
            # --------------------------------------------------------------

            for index, lane in enumerate(
                observations
            ):

                if index in matched_observations:
                    continue

                if len(self._tracks) >= self.max_lanes:
                    break

                track = self._create_track(
                    lane,
                    height,
                )

                track.last_y_min, track.last_y_max = (
                    self._lane_y_range(lane)
                )

                self._tracks[
                    track.lane_id
                ] = track

            # --------------------------------------------------------------
            # Faixas não observadas neste frame.
            # --------------------------------------------------------------

            for track in list(
                self._tracks.values()
            ):

                if (
                    track.lane_id
                    in matched_tracks
                ):
                    continue

                # Track recém-criada a partir de uma
                # observação já foi tratada.
                if (
                    track.age == self._frame_index
                    and track.missed_frames == 0
                    and track.observed
                ):
                    continue

                if not self._predict_track(
                    track,
                    width,
                    height,
                ):
                    # Ainda não apagamos imediatamente.
                    #
                    # Um track inválido é removido aqui
                    # somente quando excede o período máximo.
                    if (
                        track.missed_frames
                        >= self.max_missed_frames
                    ):
                        del self._tracks[
                            track.lane_id
                        ]

            # --------------------------------------------------------------
            # Atualiza ranges.
            # --------------------------------------------------------------

            for track in self._tracks.values():

                if track.points:

                    (
                        track.last_y_min,
                        track.last_y_max,
                    ) = self._lane_y_range(
                        track.points
                    )

            tracks = self._sort_tracks(
                list(
                    self._tracks.values()
                )
            )

            # --------------------------------------------------------------
            # Limpeza de tracks sem qualidade.
            # --------------------------------------------------------------

            tracks = [
                track
                for track in tracks
                if (
                    track.valid
                    or track.observed
                )
            ]

            self._tracks = {
                track.lane_id: track
                for track in tracks
            }

            result = self._build_result(
                tracks,
                width,
                height,
            )

            self._last_result = result

            return result

        except Exception as exc:

            logger.exception(
                "[LANE TRACKER] Falha ao atualizar."
            )

            return self._failure_result(
                f"{type(exc).__name__}: {exc}",
                width,
                height,
            )

    # ========================================================================
    # RESULTADO
    # ========================================================================

    def _build_result(
        self,
        tracks: Sequence[TrackedLane],
        frame_width: int,
        frame_height: int,
    ) -> LaneTrackingResult:

        tracks = self._sort_tracks(
            tracks
        )

        observed_count = sum(
            1
            for track in tracks
            if track.observed
        )

        projected_count = sum(
            1
            for track in tracks
            if track.projected
        )

        if tracks:

            confidence = float(
                np.mean(
                    [
                        track.confidence
                        for track in tracks
                    ]
                )
            )

        else:
            confidence = 0.0

        # Para o ADAS, não basta possuir uma única
        # faixa matemática.
        #
        # Precisamos de duas referências laterais
        # para definir uma faixa ocupada com segurança.
        can_project = (
            len(tracks) >= 2
            and any(
                track.observed
                and track.confidence
                >= self.projection_confidence
                for track in tracks
            )
        )

        # Estado seguro para controle:
        #
        # - pelo menos duas linhas;
        # - confiança global suficiente;
        # - pelo menos uma das linhas observada;
        # - não depender exclusivamente de projeções.
        safe_for_control = (
            len(tracks) >= 2
            and confidence >= 0.50
            and observed_count >= 1
            and can_project
        )

        return LaneTrackingResult(
            lanes=list(tracks),
            active_lane_id=None,
            active_lane_index=None,
            confidence=confidence,
            can_project=can_project,
            safe_for_control=safe_for_control,
            observed_lane_count=observed_count,
            projected_lane_count=projected_count,
            frame_width=frame_width,
            frame_height=frame_height,
            error=None,
        )

    def _failure_result(
        self,
        error: str,
        frame_width: int = 0,
        frame_height: int = 0,
    ) -> LaneTrackingResult:

        return LaneTrackingResult(
            lanes=[],
            active_lane_id=None,
            active_lane_index=None,
            confidence=0.0,
            can_project=False,
            safe_for_control=False,
            observed_lane_count=0,
            projected_lane_count=0,
            frame_width=frame_width,
            frame_height=frame_height,
            error=error,
        )

    # ========================================================================
    # CONSULTA
    # ========================================================================

    @property
    def last_result(
        self,
    ) -> Optional[LaneTrackingResult]:
        """Último resultado produzido."""

        return self._last_result

    @property
    def tracks(
        self,
    ) -> List[TrackedLane]:
        """Lista atual de faixas rastreadas."""

        return self._sort_tracks(
            list(
                self._tracks.values()
            )
        )


# ============================================================================
# FACTORY
# ============================================================================

def create_default_lane_tracker(
    **kwargs,
) -> LaneTracker:
    """
    Cria o tracker padrão do projeto.
    """

    return LaneTracker(
        **kwargs
    )


__all__ = [
    "PolynomialModel",
    "TrackedLane",
    "LaneTrackingResult",
    "LaneTracker",
    "create_default_lane_tracker",
]