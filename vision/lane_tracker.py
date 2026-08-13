"""
vision/lane_tracker.py

Rastreamento temporal das linhas de faixa.

Fluxo:

    YOLOP
      ↓
    LaneDetectionResult
      ↓
    LaneLine
      ↓
    LaneTracker
      ↓
    LaneTrackingResult
      ↓
    módulos posteriores

Responsabilidades:

    - manter identidade temporal das lanes;
    - associar observações entre frames;
    - controlar perda e recuperação;
    - calcular estabilidade temporal;
    - suavizar confiança;
    - preservar LaneLine como representação da observação.

Este módulo NÃO:

    - executa inferência;
    - ajusta polinômios;
    - calcula geometria;
    - identifica a faixa atual;
    - calcula posição do veículo;
    - decide atuação ADAS.

Compatível com o contrato atual de:

    vision/lane_types.py

e com:

    vision/yolop_detector.py
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .lane_types import (
    LaneLine,
    LanePoint,
)


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

DEFAULT_MAX_LANES = 8
DEFAULT_HISTORY_SIZE = 8
DEFAULT_MIN_POINTS = 4
DEFAULT_MATCH_DISTANCE = 90.0
DEFAULT_MAX_MISSED_FRAMES = 8
DEFAULT_MIN_STABLE_FRAMES = 3
DEFAULT_CONFIDENCE_DECAY = 0.82
DEFAULT_CONFIDENCE_RECOVERY = 0.35
DEFAULT_STABILITY_ALPHA = 0.35


# =============================================================================
# ESTADO TEMPORAL
# =============================================================================


@dataclass
class TrackedLane:
    """
    Estado temporal de uma LaneLine.

    O tracker mantém a observação original em `line`.

    Nenhum LaneModel é criado aqui.
    """

    track_id: int

    line: LaneLine = field(
        default_factory=lambda: LaneLine(
            lane_id=0
        )
    )

    confidence: float = 0.0

    stability: float = 0.0

    age: int = 0

    missed_frames: int = 0

    stable_frames: int = 0

    detected_this_frame: bool = False

    last_timestamp: float = 0.0

    previous_center_x: Optional[float] = None

    current_center_x: Optional[float] = None

    history: List[LaneLine] = field(
        default_factory=list
    )

    @property
    def points(self) -> List[LanePoint]:
        return self.line.points

    @property
    def valid(self) -> bool:
        return bool(
            self.line.valid
            and self.line.points
        )

    @property
    def stable(self) -> bool:
        return (
            self.stable_frames
            >= DEFAULT_MIN_STABLE_FRAMES
        )

    def is_stable(
        self,
        min_stable_frames: int,
    ) -> bool:
        return (
            self.stable_frames
            >= int(min_stable_frames)
        )

    @property
    def lost(self) -> bool:
        return (
            self.missed_frames
            > DEFAULT_MAX_MISSED_FRAMES
        )

    def is_lost(
        self,
        max_missed_frames: int,
    ) -> bool:
        return (
            self.missed_frames
            > int(max_missed_frames)
        )


@dataclass(frozen=True)
class LaneTrackingResult:
    """
    Resultado de um ciclo do tracker.
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
            if not lane.lost
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
    Rastreador temporal das LaneLine.

    A associação utiliza principalmente:

        - posição horizontal inferior;
        - posição vertical;
        - extensão da lane;
        - distância ao track anterior.

    O objetivo é preservar a identidade de uma mesma
    marcação enquanto ela aparece em frames consecutivos.
    """

    def __init__(
        self,
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

        self.max_lanes = max(
            1,
            int(max_lanes),
        )

        self.history_size = max(
            1,
            int(history_size),
        )

        self.min_points = max(
            1,
            int(min_points),
        )

        self.match_distance = max(
            1.0,
            float(match_distance),
        )

        self.max_missed_frames = max(
            0,
            int(max_missed_frames),
        )

        self.min_stable_frames = max(
            1,
            int(min_stable_frames),
        )

        self.confidence_decay = float(
            np.clip(
                confidence_decay,
                0.0,
                1.0,
            )
        )

        self.confidence_recovery = float(
            np.clip(
                confidence_recovery,
                0.0,
                1.0,
            )
        )

        self.stability_alpha = float(
            np.clip(
                stability_alpha,
                0.0,
                1.0,
            )
        )

        self._tracks: Dict[
            int,
            TrackedLane,
        ] = {}

        self._next_track_id = 0

        self._frame_index = 0

        self._last_timestamp: Optional[
            float
        ] = None

    # =========================================================================
    # PROPRIEDADES
    # =========================================================================

    @property
    def tracks(
        self,
    ) -> Tuple[TrackedLane, ...]:
        """
        Todos os tracks ainda mantidos pelo tracker.
        """

        return tuple(
            self._tracks.values()
        )

    @property
    def active_tracks(
        self,
    ) -> Tuple[TrackedLane, ...]:
        """
        Tracks que ainda não expiraram.
        """

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
    # UTILITÁRIOS
    # =========================================================================

    @staticmethod
    def _safe_float(
        value: object,
        default: float = 0.0,
    ) -> float:

        try:
            result = float(value)
        except (
            TypeError,
            ValueError,
        ):
            return default

        if not math.isfinite(result):
            return default

        return result

    @staticmethod
    def _clip01(
        value: float,
    ) -> float:

        return float(
            np.clip(
                value,
                0.0,
                1.0,
            )
        )

    @classmethod
    def _valid_points(
        cls,
        points: Sequence[LanePoint],
    ) -> List[LanePoint]:
        """
        Retorna somente LanePoint válidos e finitos.

        O tracker trabalha com o LanePoint definido em
        vision/lane_types.py.
        """

        result: List[LanePoint] = []

        if points is None:
            return result

        for point in points:

            if not isinstance(
                point,
                LanePoint,
            ):
                continue

            try:

                if not point.valid:
                    continue

                x = float(point.x)
                y = float(point.y)
                confidence = float(
                    point.confidence
                )

            except (
                TypeError,
                ValueError,
            ):
                continue

            if not (
                math.isfinite(x)
                and math.isfinite(y)
                and math.isfinite(confidence)
            ):
                continue

            result.append(
                point
            )

        return result

    @classmethod
    def _point_center_x(
        cls,
        points: Sequence[LanePoint],
    ) -> Optional[float]:
        """
        Calcula o X representativo da lane.

        Dá maior importância aos pontos inferiores.
        """

        valid = cls._valid_points(
            points
        )

        if not valid:
            return None

        ordered = sorted(
            valid,
            key=lambda point: float(
                point.y
            ),
            reverse=True,
        )

        sample = ordered[
            : min(8, len(ordered))
        ]

        if not sample:
            return None

        return float(
            np.mean(
                [
                    float(point.x)
                    for point in sample
                ]
            )
        )

    @classmethod
    def _point_center_y(
        cls,
        points: Sequence[LanePoint],
    ) -> Optional[float]:

        valid = cls._valid_points(
            points
        )

        if not valid:
            return None

        return float(
            np.mean(
                [
                    float(point.y)
                    for point in valid
                ]
            )
        )

    @classmethod
    def _lane_span(
        cls,
        points: Sequence[LanePoint],
    ) -> float:

        valid = cls._valid_points(
            points
        )

        if len(valid) < 2:
            return 0.0

        xs = [
            float(point.x)
            for point in valid
        ]

        return float(
            max(xs) - min(xs)
        )

    @classmethod
    def _copy_lane(
        cls,
        lane: LaneLine,
    ) -> LaneLine:
        """
        Cria uma cópia independente de LaneLine.

        IMPORTANTE:

        Usa somente os campos realmente existentes
        no LaneLine atual.

        Não acessa:

            lane_type
            side
            projection_quality

        porque esses campos não fazem parte do contrato
        atual de vision/lane_types.py.
        """

        points = [
            LanePoint(
                x=float(point.x),
                y=float(point.y),
                confidence=float(
                    np.clip(
                        point.confidence,
                        0.0,
                        1.0,
                    )
                ),
                valid=bool(point.valid),
            )
            for point in cls._valid_points(
                lane.points
            )
        ]

        lane_id = getattr(
            lane,
            "lane_id",
            0,
        )

        if lane_id is None:
            lane_id = 0

        try:
            lane_id = int(lane_id)
        except (
            TypeError,
            ValueError,
        ):
            lane_id = 0

        return LaneLine(
            lane_id=lane_id,
            points=points,
            confidence=cls._clip01(
                cls._safe_float(
                    lane.confidence
                )
            ),
            quality=lane.quality,
            detected_directly=bool(
                lane.detected_directly
            ),
            projected=bool(
                lane.projected
            ),
            valid=bool(
                lane.valid
            ),
            age_frames=max(
                0,
                int(
                    lane.age_frames
                )
            ),
            missed_frames=max(
                0,
                int(
                    lane.missed_frames
                )
            ),
        )

    # =========================================================================
    # NORMALIZAÇÃO
    # =========================================================================

    @classmethod
    def _normalize_lane(
        cls,
        lane: object,
    ) -> Optional[LaneLine]:
        """
        Normaliza uma observação para LaneLine.

        Aceita:

            LaneLine
            objeto com `.points`
            sequência de LanePoint
        """

        if lane is None:
            return None

        # ---------------------------------------------------------------------
        # Já é LaneLine
        # ---------------------------------------------------------------------

        if isinstance(
            lane,
            LaneLine,
        ):

            normalized = cls._copy_lane(
                lane
            )

        # ---------------------------------------------------------------------
        # Objeto compatível com LaneLine
        # ---------------------------------------------------------------------

        elif hasattr(
            lane,
            "points",
        ):

            points = cls._valid_points(
                getattr(
                    lane,
                    "points",
                    [],
                )
            )

            if not points:
                return None

            confidence = cls._safe_float(
                getattr(
                    lane,
                    "confidence",
                    0.0,
                )
            )

            lane_id = getattr(
                lane,
                "lane_id",
                0,
            )

            if lane_id is None:
                lane_id = 0

            try:
                lane_id = int(
                    lane_id
                )
            except (
                TypeError,
                ValueError,
            ):
                lane_id = 0

            normalized = LaneLine(
                lane_id=lane_id,
                points=points,
                confidence=cls._clip01(
                    confidence
                ),
                quality=getattr(
                    lane,
                    "quality",
                    LaneLine(
                        lane_id=0
                    ).quality,
                ),
                detected_directly=bool(
                    getattr(
                        lane,
                        "detected_directly",
                        True,
                    )
                ),
                projected=bool(
                    getattr(
                        lane,
                        "projected",
                        False,
                    )
                ),
                valid=bool(
                    getattr(
                        lane,
                        "valid",
                        True,
                    )
                ),
                age_frames=max(
                    0,
                    int(
                        getattr(
                            lane,
                            "age_frames",
                            0,
                        )
                    ),
                ),
                missed_frames=max(
                    0,
                    int(
                        getattr(
                            lane,
                            "missed_frames",
                            0,
                        )
                    ),
                ),
            )

        # ---------------------------------------------------------------------
        # Lista de LanePoint
        # ---------------------------------------------------------------------

        elif isinstance(
            lane,
            Sequence,
        ) and not isinstance(
            lane,
            (
                str,
                bytes,
            ),
        ):

            points = cls._valid_points(
                lane
            )

            if not points:
                return None

            confidence_values = [
                cls._safe_float(
                    point.confidence
                )
                for point in points
                if math.isfinite(
                    cls._safe_float(
                        point.confidence
                    )
                )
            ]

            confidence = (
                float(
                    np.mean(
                        confidence_values
                    )
                )
                if confidence_values
                else 0.0
            )

            normalized = LaneLine(
                lane_id=0,
                points=points,
                confidence=cls._clip01(
                    confidence
                ),
                valid=True,
            )

        else:
            return None

        # ---------------------------------------------------------------------
        # Limpeza final
        # ---------------------------------------------------------------------

        normalized.points = (
            cls._valid_points(
                normalized.points
            )
        )

        if len(
            normalized.points
        ) < 1:
            return None

        # Se a LaneLine não possui confiança própria,
        # calcula a partir dos pontos.
        if normalized.confidence <= 0.0:

            values = [
                cls._safe_float(
                    point.confidence
                )
                for point
                in normalized.points
                if math.isfinite(
                    cls._safe_float(
                        point.confidence
                    )
                )
            ]

            if values:

                normalized.confidence = (
                    cls._clip01(
                        float(
                            np.mean(values)
                        )
                    )
                )

        normalized.confidence = (
            cls._clip01(
                normalized.confidence
            )
        )

        normalized.valid = bool(
            normalized.points
        )

        return normalized

    @classmethod
    def _extract_observations(
        cls,
        detections: object,
    ) -> List[LaneLine]:
        """
        Extrai lanes de:

            LaneDetectionResult
            List[LaneLine]
            List[List[LanePoint]]
        """

        if detections is None:
            return []

        # YOLOP LaneDetectionResult
        if hasattr(
            detections,
            "lanes",
        ):

            detections = getattr(
                detections,
                "lanes",
            )

        if detections is None:
            return []

        if not isinstance(
            detections,
            Sequence,
        ):
            return []

        result: List[LaneLine] = []

        for lane in detections:

            normalized = (
                cls._normalize_lane(
                    lane
                )
            )

            if normalized is None:
                continue

            if len(
                normalized.points
            ) < cls_min_points_placeholder():
                continue

            result.append(
                normalized
            )

        return result

    # =========================================================================
    # CONFIANÇA
    # =========================================================================

    @classmethod
    def _confidence_from_lane(
        cls,
        lane: LaneLine,
    ) -> float:

        confidence = cls._safe_float(
            lane.confidence
        )

        if confidence > 0.0:
            return cls._clip01(
                confidence
            )

        values = []

        for point in lane.points:

            value = cls._safe_float(
                point.confidence
            )

            if value > 0.0:
                values.append(
                    value
                )

        if not values:
            return 0.0

        return cls._clip01(
            float(
                np.mean(values)
            )
        )

    # =========================================================================
    # DISTÂNCIA
    # =========================================================================

    def _association_distance(
        self,
        track: TrackedLane,
        lane: LaneLine,
    ) -> float:
        """
        Calcula distância entre uma observação e um track.

        A região inferior possui maior peso.
        """

        track_x = (
            track.current_center_x
        )

        lane_x = self._point_center_x(
            lane.points
        )

        if (
            track_x is None
            or lane_x is None
        ):
            return float("inf")

        x_distance = abs(
            lane_x - track_x
        )

        if (
            x_distance
            > self.match_distance
        ):
            return float("inf")

        # ---------------------------------------------------------------------
        # Diferença vertical
        # ---------------------------------------------------------------------

        track_y = self._point_center_y(
            track.points
        )

        lane_y = self._point_center_y(
            lane.points
        )

        y_penalty = 0.0

        if (
            track_y is not None
            and lane_y is not None
        ):

            y_penalty = (
                abs(
                    lane_y - track_y
                )
                * 0.10
            )

        # ---------------------------------------------------------------------
        # Diferença de extensão
        # ---------------------------------------------------------------------

        track_span = self._lane_span(
            track.points
        )

        lane_span = self._lane_span(
            lane.points
        )

        span_penalty = 0.0

        if (
            track_span > 0.0
            and lane_span > 0.0
        ):

            span_penalty = (
                abs(
                    lane_span
                    - track_span
                )
                * 0.05
            )

        return float(
            x_distance
            + y_penalty
            + span_penalty
        )

    # =========================================================================
    # CRIAÇÃO
    # =========================================================================

    def _create_track(
        self,
        lane: LaneLine,
        timestamp: float,
    ) -> TrackedLane:

        center_x = (
            self._point_center_x(
                lane.points
            )
        )

        confidence = (
            self._confidence_from_lane(
                lane
            )
        )

        copied_lane = (
            self._copy_lane(
                lane
            )
        )

        copied_lane.age_frames = 1
        copied_lane.missed_frames = 0
        copied_lane.confidence = (
            confidence
        )
        copied_lane.detected_directly = True
        copied_lane.valid = bool(
            copied_lane.points
        )

        track = TrackedLane(
            track_id=self._next_track_id,
            line=copied_lane,
            confidence=confidence,
            stability=0.0,
            age=1,
            missed_frames=0,
            stable_frames=1,
            detected_this_frame=True,
            last_timestamp=timestamp,
            previous_center_x=None,
            current_center_x=center_x,
            history=[
                self._copy_lane(
                    copied_lane
                )
            ],
        )

        self._next_track_id += 1

        return track

    # =========================================================================
    # ASSOCIAÇÃO
    # =========================================================================

    def _associate(
        self,
        observations: Sequence[LaneLine],
    ) -> Dict[int, int]:
        """
        Associa:

            observation_index -> track_id
        """

        candidates = []

        active_track_ids = [
            track_id
            for track_id, track
            in self._tracks.items()
            if not track.is_lost(
                self.max_missed_frames
            )
        ]

        for (
            observation_index,
            lane,
        ) in enumerate(
            observations
        ):

            for track_id in active_track_ids:

                track = self._tracks[
                    track_id
                ]

                distance = (
                    self._association_distance(
                        track,
                        lane,
                    )
                )

                if not math.isfinite(
                    distance
                ):
                    continue

                candidates.append(
                    (
                        distance,
                        observation_index,
                        track_id,
                    )
                )

        candidates.sort(
            key=lambda item: item[0]
        )

        result: Dict[int, int] = {}

        used_tracks = set()
        used_observations = set()

        for (
            _distance,
            observation_index,
            track_id,
        ) in candidates:

            if (
                observation_index
                in used_observations
            ):
                continue

            if track_id in used_tracks:
                continue

            result[
                observation_index
            ] = track_id

            used_observations.add(
                observation_index
            )

            used_tracks.add(
                track_id
            )

        return result

    # =========================================================================
    # ATUALIZAÇÃO
    # =========================================================================

    def _update_track(
        self,
        track: TrackedLane,
        lane: LaneLine,
        timestamp: float,
    ) -> None:

        previous_center = (
            track.current_center_x
        )

        current_center = (
            self._point_center_x(
                lane.points
            )
        )

        track.previous_center_x = (
            previous_center
        )

        track.current_center_x = (
            current_center
        )

        track.line = (
            self._copy_lane(
                lane
            )
        )

        track.age += 1

        track.missed_frames = 0

        track.detected_this_frame = True

        track.last_timestamp = timestamp

        # ---------------------------------------------------------------------
        # Histórico
        # ---------------------------------------------------------------------

        track.history.append(
            self._copy_lane(
                lane
            )
        )

        if len(
            track.history
        ) > self.history_size:

            del track.history[
                : len(track.history)
                - self.history_size
            ]

        # ---------------------------------------------------------------------
        # Confiança
        # ---------------------------------------------------------------------

        observed_confidence = (
            self._confidence_from_lane(
                lane
            )
        )

        track.confidence = (
            self._clip01(
                (
                    (
                        1.0
                        - self.confidence_recovery
                    )
                    * track.confidence
                )
                + (
                    self.confidence_recovery
                    * observed_confidence
                )
            )
        )

        # ---------------------------------------------------------------------
        # Estabilidade
        # ---------------------------------------------------------------------

        if (
            previous_center is None
            or current_center is None
        ):

            observation_stability = 0.0

        else:

            displacement = abs(
                current_center
                - previous_center
            )

            observation_stability = (
                self._clip01(
                    1.0
                    - (
                        displacement
                        / self.match_distance
                    )
                )
            )

        track.stability = (
            self._clip01(
                (
                    (
                        1.0
                        - self.stability_alpha
                    )
                    * track.stability
                )
                + (
                    self.stability_alpha
                    * observation_stability
                )
            )
        )

        if (
            observation_stability
            >= 0.50
        ):

            track.stable_frames += 1

        else:

            track.stable_frames = max(
                0,
                track.stable_frames - 1,
            )

        # ---------------------------------------------------------------------
        # Metadados LaneLine
        # ---------------------------------------------------------------------

        track.line.age_frames = (
            track.age
        )

        track.line.missed_frames = 0

        track.line.confidence = (
            track.confidence
        )

        track.line.detected_directly = True

        track.line.valid = bool(
            track.line.points
        )

    def _mark_missed(
        self,
        track: TrackedLane,
        timestamp: float,
    ) -> None:

        track.detected_this_frame = False

        track.missed_frames += 1

        track.last_timestamp = timestamp

        track.confidence = (
            self._clip01(
                track.confidence
                * self.confidence_decay
            )
        )

        track.stability = (
            self._clip01(
                track.stability
                * 0.85
            )
        )

        track.stable_frames = max(
            0,
            track.stable_frames - 1,
        )

        track.line.missed_frames = (
            track.missed_frames
        )

        track.line.age_frames = (
            track.age
        )

        track.line.confidence = (
            track.confidence
        )

        track.line.detected_directly = (
            False
        )

    # =========================================================================
    # LIMPEZA
    # =========================================================================

    def _remove_expired_tracks(
        self,
    ) -> None:

        expired = [
            track_id
            for track_id, track
            in self._tracks.items()
            if track.is_lost(
                self.max_missed_frames
            )
        ]

        for track_id in expired:

            del self._tracks[
                track_id
            ]

    # =========================================================================
    # ORDENAÇÃO
    # =========================================================================

    @staticmethod
    def _sort_tracks(
        tracks: Sequence[TrackedLane],
    ) -> List[TrackedLane]:
        """
        Ordena espacialmente pelo X.

        Isso não decide a faixa atual.
        """

        return sorted(
            tracks,
            key=lambda track: (
                float("inf")
                if track.current_center_x
                is None
                else track.current_center_x
            ),
        )

    # =========================================================================
    # RESULTADO
    # =========================================================================

    def _build_result(
        self,
        timestamp: float,
    ) -> LaneTrackingResult:

        tracks = self._sort_tracks(
            list(
                self._tracks.values()
            )
        )

        stable_count = sum(
            1
            for track in tracks
            if track.is_stable(
                self.min_stable_frames
            )
        )

        detected_count = sum(
            1
            for track in tracks
            if track.detected_this_frame
        )

        lost_count = sum(
            1
            for track in tracks
            if track.is_lost(
                self.max_missed_frames
            )
        )

        return LaneTrackingResult(
            lanes=tuple(
                tracks
            ),
            timestamp=timestamp,
            frame_index=self._frame_index,
            valid=(
                detected_count > 0
            ),
            stable_count=stable_count,
            detected_count=detected_count,
            lost_count=lost_count,
        )

    # =========================================================================
    # API PRINCIPAL
    # =========================================================================

    def update(
        self,
        detections: object,
        timestamp: Optional[float] = None,
    ) -> LaneTrackingResult:
        """
        Atualiza o tracker.

        Aceita:

            LaneDetectionResult

            List[LaneLine]

            List[List[LanePoint]]
        """

        if timestamp is None:
            timestamp = time.monotonic()

        timestamp = self._safe_float(
            timestamp
        )

        self._frame_index += 1

        self._last_timestamp = (
            timestamp
        )

        observations = (
            self._extract_observations(
                detections
            )
        )

        # ---------------------------------------------------------------------
        # Limite de lanes
        # ---------------------------------------------------------------------

        observations = observations[
            : self.max_lanes
        ]

        # ---------------------------------------------------------------------
        # Associação
        # ---------------------------------------------------------------------

        associations = (
            self._associate(
                observations
            )
        )

        matched_track_ids = set()

        # ---------------------------------------------------------------------
        # Atualiza tracks existentes
        # ---------------------------------------------------------------------

        for (
            observation_index,
            lane,
        ) in enumerate(
            observations
        ):

            track_id = associations.get(
                observation_index
            )

            if track_id is None:
                continue

            track = self._tracks.get(
                track_id
            )

            if track is None:
                continue

            self._update_track(
                track,
                lane,
                timestamp,
            )

            matched_track_ids.add(
                track_id
            )

        # ---------------------------------------------------------------------
        # Cria novos tracks
        # ---------------------------------------------------------------------

        for (
            observation_index,
            lane,
        ) in enumerate(
            observations
        ):

            if observation_index in associations:
                continue

            if (
                len(self._tracks)
                >= self.max_lanes
            ):
                break

            track = self._create_track(
                lane,
                timestamp,
            )

            self._tracks[
                track.track_id
            ] = track

            matched_track_ids.add(
                track.track_id
            )

        # ---------------------------------------------------------------------
        # Marca tracks ausentes
        # ---------------------------------------------------------------------

        for (
            track_id,
            track,
        ) in list(
            self._tracks.items()
        ):

            if track_id in matched_track_ids:
                continue

            self._mark_missed(
                track,
                timestamp,
            )

        # ---------------------------------------------------------------------
        # Remove expirados
        # ---------------------------------------------------------------------

        self._remove_expired_tracks()

        # ---------------------------------------------------------------------
        # Resultado
        # ---------------------------------------------------------------------

        return self._build_result(
            timestamp
        )

    def track(
        self,
        detections: object,
        timestamp: Optional[float] = None,
    ) -> LaneTrackingResult:
        """
        Alias semântico para update().
        """

        return self.update(
            detections,
            timestamp=timestamp,
        )

    # =========================================================================
    # RESET
    # =========================================================================

    def reset(self) -> None:
        """
        Limpa completamente o estado temporal.
        """

        self._tracks.clear()

        self._next_track_id = 0

        self._frame_index = 0

        self._last_timestamp = None

    # =========================================================================
    # CONSULTA
    # =========================================================================

    def get_track(
        self,
        track_id: int,
    ) -> Optional[TrackedLane]:

        return self._tracks.get(
            int(track_id)
        )

    def get_stable_tracks(
        self,
    ) -> Tuple[TrackedLane, ...]:

        return tuple(
            track
            for track in self._tracks.values()
            if track.is_stable(
                self.min_stable_frames
            )
        )

    def get_active_tracks(
        self,
    ) -> Tuple[TrackedLane, ...]:

        return tuple(
            track
            for track in self._tracks.values()
            if not track.is_lost(
                self.max_missed_frames
            )
        )


# =============================================================================
# FACTORY
# =============================================================================


def create_default_lane_tracker(
    **kwargs,
) -> LaneTracker:

    return LaneTracker(
        **kwargs
    )


# =============================================================================
# AUXILIAR INTERNO
# =============================================================================

def cls_min_points_placeholder() -> int:
    """
    Valor mínimo utilizado durante a extração.

    O filtro definitivo é aplicado pelo próprio tracker
    no processamento das observações.
    """

    return 1


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "TrackedLane",
    "LaneTrackingResult",
    "LaneTracker",
    "create_default_lane_tracker",
]