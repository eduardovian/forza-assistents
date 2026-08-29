"""
vision/lane_tracker.py

Forza Assistents
================

Rastreamento temporal das linhas de faixa.

Pipeline:

    YOLOPv2
        ↓
    LaneLine
        ↓
    LaneTracker
        ↓
    TrackedLane
        ↓
    LaneGeometry / LaneModel / LaneProjection

Responsabilidades:

    - manter identidade temporal das lanes;
    - associar observações entre frames;
    - lidar com perda temporária;
    - preservar identidade durante recuperação;
    - estimar movimento temporal;
    - calcular estabilidade;
    - suavizar confiança;
    - manter histórico limitado.

Não é responsabilidade deste módulo:

    - inferência neural;
    - ajuste geométrico;
    - identificação da faixa atual;
    - decisão ADAS;
    - atuação física.

O contrato de dados oficial é exclusivamente:

    vision.lane_types
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import hypot, isfinite
from typing import Deque, Dict, Iterable, Optional, Sequence, Tuple

from .lane_types import LaneLine, LanePoint, LaneSource


# =============================================================================
# DEFAULTS
# =============================================================================

DEFAULT_MAX_LANES = 8
DEFAULT_HISTORY_SIZE = 8
DEFAULT_MIN_POINTS = 4

DEFAULT_MATCH_DISTANCE = 120.0
DEFAULT_MAX_MISSED_FRAMES = 12
DEFAULT_MIN_STABLE_FRAMES = 3

DEFAULT_CONFIDENCE_DECAY = 0.82
DEFAULT_CONFIDENCE_RECOVERY = 0.35
DEFAULT_STABILITY_ALPHA = 0.35

MIN_CONFIDENCE = 0.0
MAX_CONFIDENCE = 1.0


# =============================================================================
# UTILITIES
# =============================================================================

def _clamp01(value: float) -> float:
    """Limita um valor ao intervalo [0, 1]."""

    if not isfinite(value):
        return 0.0

    return max(
        MIN_CONFIDENCE,
        min(MAX_CONFIDENCE, float(value)),
    )


def _finite_float(
    value: object,
    default: float = 0.0,
) -> float:
    """Converte um valor para float finito."""

    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    return result if isfinite(result) else default


def _valid_points(
    points: Sequence[LanePoint],
) -> Tuple[LanePoint, ...]:
    """
    Retorna somente pontos válidos.

    LanePoint já garante coordenadas finitas. A verificação adicional
    mantém a fronteira robusta contra objetos externos incompatíveis.
    """

    valid = []

    for point in points:

        if not isinstance(point, LanePoint):
            continue

        if not (
            isfinite(point.x)
            and isfinite(point.y)
        ):
            continue

        valid.append(point)

    return tuple(valid)


def _bottom_center_x(
    points: Sequence[LanePoint],
) -> Optional[float]:
    """
    Obtém a posição horizontal representativa da lane.

    Prioriza os pontos mais próximos da parte inferior do frame,
    onde o erro lateral é mais relevante.
    """

    valid = _valid_points(points)

    if not valid:
        return None

    ordered = sorted(
        valid,
        key=lambda point: point.y,
        reverse=True,
    )

    sample = ordered[
        : min(5, len(ordered))
    ]

    return sum(
        point.x
        for point in sample
    ) / len(sample)


def _mean_y(
    points: Sequence[LanePoint],
) -> Optional[float]:

    valid = _valid_points(points)

    if not valid:
        return None

    return sum(
        point.y
        for point in valid
    ) / len(valid)


def _vertical_span(
    points: Sequence[LanePoint],
) -> float:
    """Calcula a extensão vertical observada da lane."""

    valid = _valid_points(points)

    if len(valid) < 2:
        return 0.0

    ys = [
        point.y
        for point in valid
    ]

    return max(ys) - min(ys)


# =============================================================================
# TRACKED LANE
# =============================================================================

@dataclass
class TrackedLane:
    """
    Estado temporal mutável de uma lane.

    A LaneLine continua sendo um objeto imutável.
    O tracker mantém o estado temporal separadamente.
    """

    track_id: int
    line: LaneLine

    confidence: float
    stability: float = 0.0

    age: int = 1
    missed_frames: int = 0
    stable_frames: int = 0

    detected_this_frame: bool = True

    last_timestamp: float = 0.0

    previous_center_x: Optional[float] = None
    current_center_x: Optional[float] = None

    velocity_x: float = 0.0
    velocity_y: float = 0.0

    history: Deque[LaneLine] = field(
        default_factory=deque
    )

    @property
    def points(self) -> Tuple[LanePoint, ...]:
        return self.line.points

    @property
    def valid(self) -> bool:
        return bool(
            self.line.points
            and self.confidence > 0.0
        )

    @property
    def stable(self) -> bool:
        return self.stable_frames >= 3

    @property
    def lost(self) -> bool:
        return self.missed_frames > 0

    def is_stable(
        self,
        min_stable_frames: int,
    ) -> bool:
        return (
            self.stable_frames
            >= max(1, int(min_stable_frames))
        )

    def is_lost(
        self,
        max_missed_frames: int,
    ) -> bool:
        return (
            self.missed_frames
            > max(0, int(max_missed_frames))
        )


# =============================================================================
# TRACKING RESULT
# =============================================================================

@dataclass(frozen=True, slots=True)
class LaneTrackingResult:
    """
    Resultado imutável de um ciclo do tracker.
    """

    lanes: Tuple[TrackedLane, ...]

    timestamp: float
    frame_index: int

    valid: bool

    stable_count: int
    detected_count: int
    lost_count: int

    @property
    def active_lanes(
        self,
    ) -> Tuple[TrackedLane, ...]:

        return tuple(
            lane
            for lane in self.lanes
            if lane.missed_frames == 0
        )

    @property
    def stable_lanes(
        self,
    ) -> Tuple[TrackedLane, ...]:

        return tuple(
            lane
            for lane in self.lanes
            if lane.stable
        )


# =============================================================================
# TRACKER
# =============================================================================

class LaneTracker:
    """
    Rastreador temporal determinístico de LaneLine.

    Associação baseada em:

        1. posição horizontal inferior;
        2. posição vertical média;
        3. extensão vertical;
        4. distância ao estado anterior.

    O algoritmo é deliberadamente conservador:

        - uma observação só pode ser associada a um track;
        - um track só pode receber uma observação;
        - tracks perdidos permanecem durante a janela configurada;
        - após expiração, a identidade é encerrada.
    """

    def __init__(
        self,
        *,
        max_lanes: int = DEFAULT_MAX_LANES,
        history_size: int = DEFAULT_HISTORY_SIZE,
        min_points: int = DEFAULT_MIN_POINTS,
        match_distance: float = DEFAULT_MATCH_DISTANCE,
        max_missed_frames: int = DEFAULT_MAX_MISSED_FRAMES,
        min_stable_frames: int = DEFAULT_MIN_STABLE_FRAMES,
        confidence_decay: float = DEFAULT_CONFIDENCE_DECAY,
        confidence_recovery: float = DEFAULT_CONFIDENCE_RECOVERY,
        stability_alpha: float = DEFAULT_STABILITY_ALPHA,
    ) -> None:

        if max_lanes < 1:
            raise ValueError(
                "max_lanes must be >= 1"
            )

        if history_size < 1:
            raise ValueError(
                "history_size must be >= 1"
            )

        if min_points < 1:
            raise ValueError(
                "min_points must be >= 1"
            )

        if match_distance <= 0.0:
            raise ValueError(
                "match_distance must be > 0"
            )

        if max_missed_frames < 0:
            raise ValueError(
                "max_missed_frames must be >= 0"
            )

        if min_stable_frames < 1:
            raise ValueError(
                "min_stable_frames must be >= 1"
            )

        if not 0.0 <= confidence_decay <= 1.0:
            raise ValueError(
                "confidence_decay must be in [0, 1]"
            )

        if not 0.0 <= confidence_recovery <= 1.0:
            raise ValueError(
                "confidence_recovery must be in [0, 1]"
            )

        if not 0.0 <= stability_alpha <= 1.0:
            raise ValueError(
                "stability_alpha must be in [0, 1]"
            )

        self.max_lanes = int(max_lanes)
        self.history_size = int(history_size)
        self.min_points = int(min_points)

        self.match_distance = float(
            match_distance
        )

        self.max_missed_frames = int(
            max_missed_frames
        )

        self.min_stable_frames = int(
            min_stable_frames
        )

        self.confidence_decay = float(
            confidence_decay
        )

        self.confidence_recovery = float(
            confidence_recovery
        )

        self.stability_alpha = float(
            stability_alpha
        )

        self._tracks: Dict[
            int,
            TrackedLane,
        ] = {}

        self._next_track_id = 0
        self._frame_index = 0
        self._last_timestamp: Optional[float] = None

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def tracks(
        self,
    ) -> Tuple[TrackedLane, ...]:

        return tuple(
            self._tracks.values()
        )

    @property
    def active_tracks(
        self,
    ) -> Tuple[TrackedLane, ...]:

        return tuple(
            track
            for track in self._tracks.values()
            if not track.is_lost(
                self.max_missed_frames
            )
        )

    @property
    def frame_index(self) -> int:
        return self._frame_index

    # =========================================================================
    # OBSERVATION VALIDATION
    # =========================================================================

    def _prepare_observations(
        self,
        detections: Iterable[LaneLine],
    ) -> Tuple[LaneLine, ...]:
        """
        Filtra observações incompatíveis com o contrato do tracker.
        """

        result = []

        for lane in detections:

            if not isinstance(
                lane,
                LaneLine,
            ):
                continue

            if len(lane.points) < self.min_points:
                continue

            points = _valid_points(
                lane.points
            )

            if len(points) < self.min_points:
                continue

            result.append(
                lane
            )

            if len(result) >= self.max_lanes:
                break

        return tuple(result)

    # =========================================================================
    # DISTANCE
    # =========================================================================

    def _association_distance(
        self,
        track: TrackedLane,
        detection: LaneLine,
    ) -> float:
        """
        Calcula custo de associação entre track e observação.

        A posição inferior possui maior peso porque representa melhor
        a relação lateral imediata do veículo com a lane.
        """

        track_x = track.current_center_x

        if track_x is None:
            track_x = _bottom_center_x(
                track.line.points
            )

        detection_x = _bottom_center_x(
            detection.points
        )

        if (
            track_x is None
            or detection_x is None
        ):
            return float("inf")

        dx = abs(
            track_x
            - detection_x
        )

        track_y = _mean_y(
            track.line.points
        )

        detection_y = _mean_y(
            detection.points
        )

        if (
            track_y is None
            or detection_y is None
        ):
            return dx

        dy = abs(
            track_y
            - detection_y
        )

        track_span = _vertical_span(
            track.line.points
        )

        detection_span = _vertical_span(
            detection.points
        )

        span_delta = abs(
            track_span
            - detection_span
        )

        # Pesos conservadores.
        cost = (
            dx
            + 0.15 * dy
            + 0.10 * span_delta
        )

        return float(cost)

    # =========================================================================
    # ASSOCIATION
    # =========================================================================

    def _associate(
        self,
        detections: Sequence[LaneLine],
    ) -> Tuple[
        Dict[int, int],
        Tuple[int, ...],
    ]:
        """
        Associação gulosa determinística.

        Retorna:

            matches:
                track_id -> detection_index

            unmatched_detections:
                índices sem track correspondente.
        """

        candidates = []

        for track_id, track in self._tracks.items():

            if track.is_lost(
                self.max_missed_frames
            ):
                continue

            for index, detection in enumerate(
                detections
            ):

                distance = (
                    self._association_distance(
                        track,
                        detection,
                    )
                )

                if distance <= self.match_distance:

                    candidates.append(
                        (
                            distance,
                            track_id,
                            index,
                        )
                    )

        # Determinismo:
        # custo → track_id → detection index.
        candidates.sort(
            key=lambda item: (
                item[0],
                item[1],
                item[2],
            )
        )

        matches: Dict[int, int] = {}

        used_tracks = set()
        used_detections = set()

        for (
            distance,
            track_id,
            detection_index,
        ) in candidates:

            del distance

            if track_id in used_tracks:
                continue

            if detection_index in used_detections:
                continue

            matches[
                track_id
            ] = detection_index

            used_tracks.add(
                track_id
            )

            used_detections.add(
                detection_index
            )

        unmatched = tuple(
            index
            for index in range(
                len(detections)
            )
            if index not in used_detections
        )

        return matches, unmatched

    # =========================================================================
    # TRACK CREATION
    # =========================================================================

    def _create_track(
        self,
        lane: LaneLine,
        timestamp: float,
    ) -> TrackedLane:

        track_id = self._next_track_id

        self._next_track_id += 1

        center_x = _bottom_center_x(
            lane.points
        )

        history: Deque[
            LaneLine
        ] = deque(
            maxlen=self.history_size
        )

        history.append(lane)

        track = TrackedLane(
            track_id=track_id,
            line=lane,
            confidence=lane.confidence,
            stability=0.0,
            age=1,
            missed_frames=0,
            stable_frames=1,
            detected_this_frame=True,
            last_timestamp=timestamp,
            previous_center_x=None,
            current_center_x=center_x,
            velocity_x=0.0,
            velocity_y=0.0,
            history=history,
        )

        return track

    # =========================================================================
    # TRACK UPDATE
    # =========================================================================

    def _update_track(
        self,
        track: TrackedLane,
        detection: LaneLine,
        timestamp: float,
    ) -> None:

        previous_center_x = (
            track.current_center_x
        )

        current_center_x = _bottom_center_x(
            detection.points
        )

        dt = (
            timestamp
            - track.last_timestamp
        )

        if dt <= 0.0:
            dt = 1.0

        if (
            previous_center_x is not None
            and current_center_x is not None
        ):

            raw_velocity = (
                current_center_x
                - previous_center_x
            ) / dt

            # Suavização temporal.
            track.velocity_x = (
                0.70 * track.velocity_x
                + 0.30 * raw_velocity
            )

        track.previous_center_x = (
            previous_center_x
        )

        track.current_center_x = (
            current_center_x
        )

        track.line = detection

        track.age += 1
        track.missed_frames = 0
        track.detected_this_frame = True

        track.confidence = _clamp01(
            (
                (1.0 - self.confidence_recovery)
                * track.confidence
            )
            + (
                self.confidence_recovery
                * detection.confidence
            )
        )

        # Estabilidade baseada na consistência da posição.
        if (
            previous_center_x is not None
            and current_center_x is not None
        ):

            displacement = abs(
                current_center_x
                - previous_center_x
            )

            consistency = max(
                0.0,
                1.0
                - (
                    displacement
                    / self.match_distance
                ),
            )

        else:
            consistency = 0.0

        track.stability = (
            (1.0 - self.stability_alpha)
            * track.stability
            + self.stability_alpha
            * consistency
        )

        if (
            track.confidence >= 0.50
            and track.stability >= 0.30
        ):
            track.stable_frames += 1
        else:
            track.stable_frames = max(
                0,
                track.stable_frames - 1,
            )

        track.history.append(
            detection
        )

        track.last_timestamp = timestamp

    # =========================================================================
    # TRACK LOSS
    # =========================================================================

    def _mark_track_lost(
        self,
        track: TrackedLane,
        timestamp: float,
    ) -> None:

        track.age += 1
        track.missed_frames += 1
        track.detected_this_frame = False

        track.confidence = _clamp01(
            track.confidence
            * self.confidence_decay
        )

        track.stability = _clamp01(
            track.stability
            * 0.90
        )

        track.last_timestamp = timestamp

    # =========================================================================
    # UPDATE
    # =========================================================================

    def update(
        self,
        detections: Iterable[LaneLine],
        *,
        timestamp: Optional[float] = None,
    ) -> LaneTrackingResult:
        """
        Processa um frame.

        Parameters
        ----------
        detections:
            Iterable de LaneLine produzido pelo detector.

        timestamp:
            Timestamp monotônico em segundos.

        Returns
        -------
        LaneTrackingResult
        """

        if timestamp is None:
            timestamp = (
                self._last_timestamp + 1.0
                if self._last_timestamp is not None
                else 0.0
            )

        timestamp = _finite_float(
            timestamp,
            default=0.0,
        )

        if (
            self._last_timestamp is not None
            and timestamp < self._last_timestamp
        ):
            raise ValueError(
                "timestamp must be monotonic"
            )

        self._frame_index += 1
        self._last_timestamp = timestamp

        prepared = self._prepare_observations(
            detections
        )

        matches, unmatched = self._associate(
            prepared
        )

        # ---------------------------------------------------------------------
        # Atualiza tracks existentes.
        # ---------------------------------------------------------------------

        matched_track_ids = set(
            matches.keys()
        )

        for track_id, track in tuple(
            self._tracks.items()
        ):

            if track_id in matched_track_ids:

                detection_index = matches[
                    track_id
                ]

                self._update_track(
                    track,
                    prepared[detection_index],
                    timestamp,
                )

            else:

                self._mark_track_lost(
                    track,
                    timestamp,
                )

        # ---------------------------------------------------------------------
        # Cria novas identidades.
        # ---------------------------------------------------------------------

        for detection_index in unmatched:

            if len(self._tracks) >= self.max_lanes:
                break

            lane = prepared[
                detection_index
            ]

            track = self._create_track(
                lane,
                timestamp,
            )

            self._tracks[
                track.track_id
            ] = track

        # ---------------------------------------------------------------------
        # Remove tracks expirados.
        # ---------------------------------------------------------------------

        expired_ids = [
            track_id
            for track_id, track
            in self._tracks.items()
            if track.is_lost(
                self.max_missed_frames
            )
        ]

        for track_id in expired_ids:
            del self._tracks[
                track_id
            ]

        # ---------------------------------------------------------------------
        # Estatísticas.
        # ---------------------------------------------------------------------

        lanes = tuple(
            sorted(
                self._tracks.values(),
                key=lambda track: (
                    track.current_center_x
                    if track.current_center_x
                    is not None
                    else float("inf"),
                    track.track_id,
                ),
            )
        )

        stable_count = sum(
            lane.is_stable(
                self.min_stable_frames
            )
            for lane in lanes
        )

        detected_count = sum(
            lane.detected_this_frame
            for lane in lanes
        )

        lost_count = sum(
            lane.missed_frames > 0
            for lane in lanes
        )

        valid = bool(
            detected_count > 0
        )

        return LaneTrackingResult(
            lanes=lanes,
            timestamp=timestamp,
            frame_index=self._frame_index,
            valid=valid,
            stable_count=stable_count,
            detected_count=detected_count,
            lost_count=lost_count,
        )

    # =========================================================================
    # RESET
    # =========================================================================

    def reset(self) -> None:
        """Remove todo o estado temporal."""

        self._tracks.clear()

        self._next_track_id = 0
        self._frame_index = 0
        self._last_timestamp = None


__all__ = [
    "TrackedLane",
    "LaneTrackingResult",
    "LaneTracker",
]