"""
vision/lane_tracker.py

Rastreamento temporal das linhas de faixa.

Responsabilidades:
    YOLOP LaneDetectionResult
        ↓
    associação temporal
        ↓
    manutenção da identidade das lanes
        ↓
    estabilidade / perda / recuperação
        ↓
    TrackedLane / LaneTrackingResult

Este módulo NÃO:
    - executa inferência YOLOP;
    - cria um modelo polinomial definitivo;
    - determina a faixa atual do veículo;
    - calcula posição do veículo;
    - decide atuação ADAS;
    - controla o veículo.

O ajuste matemático das lanes é responsabilidade de lane_model.py.
A identificação da faixa atual é responsabilidade de lane_assignment.py.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .lane_types import (
    LaneLine,
    LaneModel,
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
# RESULTADOS
# =============================================================================


@dataclass
class TrackedLane:
    """
    Estado temporal de uma lane.

    O objeto mantém apenas informações relacionadas ao tracking.
    O modelo geométrico permanece em LaneModel.
    """

    track_id: int

    points: List[LanePoint] = field(default_factory=list)

    model: Optional[LaneModel] = None

    confidence: float = 0.0

    stability: float = 0.0

    age: int = 0

    missed_frames: int = 0

    stable_frames: int = 0

    detected_this_frame: bool = False

    last_timestamp: float = 0.0

    previous_center_x: Optional[float] = None

    current_center_x: Optional[float] = None

    @property
    def valid(self) -> bool:
        """
        Indica se a lane possui informação útil neste momento.
        """

        return bool(
            self.points
            or self.model is not None
        )

    @property
    def stable(self) -> bool:
        """
        Indica estabilidade temporal mínima.
        """

        return self.stable_frames >= DEFAULT_MIN_STABLE_FRAMES

    @property
    def lost(self) -> bool:
        """
        Indica se a lane ultrapassou o limite de frames perdidos.
        """

        return (
            self.missed_frames
            > DEFAULT_MAX_MISSED_FRAMES
        )


@dataclass(frozen=True)
class LaneTrackingResult:
    """
    Resultado de um ciclo de tracking.
    """

    lanes: Tuple[TrackedLane, ...]

    timestamp: float

    frame_index: int

    valid: bool

    stable_count: int

    detected_count: int

    lost_count: int


# =============================================================================
# TRACKER
# =============================================================================


class LaneTracker:
    """
    Mantém identidade temporal das lanes detectadas.

    O tracker não tenta descobrir qual lane é a esquerda/direita
    nem qual lane pertence ao veículo.

    Ele apenas responde:

        "Esta observação provavelmente corresponde à mesma
         lane observada anteriormente?"
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

    @staticmethod
    def _point_center_x(
        points: Sequence[LanePoint],
    ) -> Optional[float]:
        """
        Obtém o centro horizontal da observação.

        É usado somente para associação temporal.
        Não representa o centro da faixa do veículo.
        """

        valid = [
            point
            for point in points
            if getattr(point, "valid", True)
            and math.isfinite(
                float(point.x)
            )
            and math.isfinite(
                float(point.y)
            )
        ]

        if not valid:
            return None

        # Preferimos os pontos próximos à região inferior da imagem,
        # pois são normalmente mais úteis para associação temporal.
        valid = sorted(
            valid,
            key=lambda point: float(point.y),
            reverse=True,
        )

        sample = valid[
            : max(
                1,
                min(
                    5,
                    len(valid),
                ),
            )
        ]

        return float(
            np.mean(
                [
                    float(point.x)
                    for point in sample
                ]
            )
        )

    @staticmethod
    def _lane_points(
        lane: object,
    ) -> List[LanePoint]:
        """
        Extrai pontos de uma observação de forma compatível com
        LaneLine e estruturas simples contendo `.points`.
        """

        points = getattr(
            lane,
            "points",
            None,
        )

        if points is None:
            return []

        result: List[LanePoint] = []

        for point in points:

            if not isinstance(
                point,
                LanePoint,
            ):
                continue

            if not getattr(
                point,
                "valid",
                True,
            ):
                continue

            if not (
                math.isfinite(
                    float(point.x)
                )
                and math.isfinite(
                    float(point.y)
                )
            ):
                continue

            result.append(point)

        return result

    @staticmethod
    def _lane_confidence(
        lane: object,
        points: Sequence[LanePoint],
    ) -> float:
        """
        Obtém confiança da observação.

        Usa a confiança explícita da lane quando disponível.
        Caso contrário, calcula a média dos pontos.
        """

        confidence = getattr(
            lane,
            "confidence",
            None,
        )

        if confidence is not None:

            try:
                return float(
                    np.clip(
                        float(confidence),
                        0.0,
                        1.0,
                    )
                )
            except (
                TypeError,
                ValueError,
            ):
                pass

        if not points:
            return 0.0

        values = []

        for point in points:

            try:
                value = float(
                    point.confidence
                )
            except (
                TypeError,
                ValueError,
            ):
                continue

            if math.isfinite(value):
                values.append(
                    np.clip(
                        value,
                        0.0,
                        1.0,
                    )
                )

        if not values:
            return 0.0

        return float(
            np.mean(values)
        )

    def _new_track(
        self,
        points: Sequence[LanePoint],
        confidence: float,
        timestamp: float,
    ) -> TrackedLane:

        track = TrackedLane(
            track_id=self._next_track_id,
            points=list(points)[
                -self.history_size:
            ],
            confidence=self._clip01(
                confidence
            ),
            stability=0.0,
            age=1,
            missed_frames=0,
            stable_frames=1,
            detected_this_frame=True,
            last_timestamp=timestamp,
            current_center_x=self._point_center_x(
                points
            ),
        )

        self._next_track_id += 1

        return track

    # =========================================================================
    # DISTÂNCIA DE ASSOCIAÇÃO
    # =========================================================================

    def _association_distance(
        self,
        track: TrackedLane,
        points: Sequence[LanePoint],
    ) -> float:

        current_x = self._point_center_x(
            points
        )

        if current_x is None:
            return float("inf")

        if track.current_center_x is None:
            return float("inf")

        return abs(
            current_x
            - track.current_center_x
        )

    # =========================================================================
    # ASSOCIAÇÃO
    # =========================================================================

    def _associate(
        self,
        observations: Sequence[
            Tuple[
                Sequence[LanePoint],
                float,
            ]
        ],
    ) -> List[
        Tuple[
            Optional[int],
            Sequence[LanePoint],
            float,
        ]
    ]:

        available_tracks = set(
            self._tracks.keys()
        )

        matches = []

        candidates = []

        for observation_index, (
            points,
            confidence,
        ) in enumerate(
            observations
        ):

            for track_id in available_tracks:

                track = self._tracks[
                    track_id
                ]

                distance = (
                    self._association_distance(
                        track,
                        points,
                    )
                )

                if distance <= self.match_distance:

                    candidates.append(
                        (
                            distance,
                            observation_index,
                            track_id,
                        )
                    )

        # Melhor correspondência primeiro.
        candidates.sort(
            key=lambda item: item[0]
        )

        matched_observations = set()
        matched_tracks = set()

        for (
            _distance,
            observation_index,
            track_id,
        ) in candidates:

            if observation_index in matched_observations:
                continue

            if track_id in matched_tracks:
                continue

            points, confidence = observations[
                observation_index
            ]

            matches.append(
                (
                    track_id,
                    points,
                    confidence,
                )
            )

            matched_observations.add(
                observation_index
            )

            matched_tracks.add(
                track_id
            )

        # Observações que não conseguiram associação
        # recebem uma nova identidade.
        for index, (
            points,
            confidence,
        ) in enumerate(
            observations
        ):

            if index in matched_observations:
                continue

            matches.append(
                (
                    None,
                    points,
                    confidence,
                )
            )

        return matches

    # =========================================================================
    # ATUALIZAÇÃO
    # =========================================================================

    def _update_track(
        self,
        track: TrackedLane,
        points: Sequence[LanePoint],
        confidence: float,
        timestamp: float,
    ) -> None:

        new_center = self._point_center_x(
            points
        )

        previous_center = (
            track.current_center_x
        )

        track.previous_center_x = (
            previous_center
        )

        track.current_center_x = (
            new_center
        )

        track.points = list(points)[
            -self.history_size:
        ]

        track.age += 1

        track.missed_frames = 0

        track.detected_this_frame = True

        track.last_timestamp = timestamp

        # ---------------------------------------------------------------------
        # Confiança
        # ---------------------------------------------------------------------

        confidence = self._clip01(
            confidence
        )

        track.confidence = float(
            (
                (1.0 - self.confidence_recovery)
                * track.confidence
            )
            + (
                self.confidence_recovery
                * confidence
            )
        )

        # ---------------------------------------------------------------------
        # Estabilidade
        # ---------------------------------------------------------------------

        if previous_center is None or new_center is None:

            observation_stability = 0.0

        else:

            displacement = abs(
                new_center
                - previous_center
            )

            observation_stability = self._clip01(
                1.0
                - (
                    displacement
                    / self.match_distance
                )
            )

        track.stability = float(
            (
                (1.0 - self.stability_alpha)
                * track.stability
            )
            + (
                self.stability_alpha
                * observation_stability
            )
        )

        if observation_stability >= 0.50:

            track.stable_frames += 1

        else:

            track.stable_frames = max(
                0,
                track.stable_frames - 1,
            )

    def _mark_missed(
        self,
        track: TrackedLane,
        timestamp: float,
    ) -> None:

        track.detected_this_frame = False

        track.missed_frames += 1

        track.last_timestamp = timestamp

        track.confidence *= (
            self.confidence_decay
        )

        track.stability *= 0.85

        track.stable_frames = max(
            0,
            track.stable_frames - 1,
        )

    # =========================================================================
    # LIMPEZA
    # =========================================================================

    def _remove_expired_tracks(self) -> None:

        expired = [
            track_id
            for track_id, track
            in self._tracks.items()
            if track.missed_frames
            > self.max_missed_frames
        ]

        for track_id in expired:

            del self._tracks[
                track_id
            ]

    # =========================================================================
    # NORMALIZAÇÃO DA ENTRADA
    # =========================================================================

    @staticmethod
    def _extract_lanes(
        detection_result: object,
    ) -> Iterable[object]:
        """
        Extrai lanes do resultado do detector.

        O tracker não depende de uma implementação específica
        do detector além de `.lanes`.
        """

        lanes = getattr(
            detection_result,
            "lanes",
            None,
        )

        if lanes is None:
            return []

        return lanes

    def _prepare_observations(
        self,
        detection_result: object,
    ) -> List[
        Tuple[
            List[LanePoint],
            float,
        ]
    ]:

        observations = []

        for lane in self._extract_lanes(
            detection_result
        ):

            points = self._lane_points(
                lane
            )

            if len(points) < self.min_points:
                continue

            confidence = self._lane_confidence(
                lane,
                points,
            )

            observations.append(
                (
                    points,
                    confidence,
                )
            )

        return observations

    # =========================================================================
    # UPDATE PÚBLICO
    # =========================================================================

    def update(
        self,
        detection_result: object,
        timestamp: Optional[float] = None,
    ) -> LaneTrackingResult:
        """
        Atualiza o tracker com uma nova detecção.

        Parameters
        ----------
        detection_result:
            Resultado produzido pelo detector de lanes.

        timestamp:
            Timestamp monotônico opcional.

        Returns
        -------
        LaneTrackingResult
        """

        if timestamp is None:
            timestamp = time.monotonic()

        timestamp = self._safe_float(
            timestamp,
            time.monotonic(),
        )

        self._frame_index += 1

        self._last_timestamp = timestamp

        observations = self._prepare_observations(
            detection_result
        )

        observations = observations[
            : self.max_lanes
        ]

        matched = self._associate(
            observations
        )

        matched_track_ids = set()

        # ---------------------------------------------------------------------
        # Atualiza tracks existentes / cria novas
        # ---------------------------------------------------------------------

        for (
            track_id,
            points,
            confidence,
        ) in matched:

            if track_id is None:

                track = self._new_track(
                    points,
                    confidence,
                    timestamp,
                )

                self._tracks[
                    track.track_id
                ] = track

                matched_track_ids.add(
                    track.track_id
                )

                continue

            track = self._tracks.get(
                track_id
            )

            if track is None:
                continue

            self._update_track(
                track,
                points,
                confidence,
                timestamp,
            )

            matched_track_ids.add(
                track_id
            )

        # ---------------------------------------------------------------------
        # Tracks não observados neste frame
        # ---------------------------------------------------------------------

        for track_id, track in list(
            self._tracks.items()
        ):

            if track_id in matched_track_ids:
                continue

            self._mark_missed(
                track,
                timestamp,
            )

        # ---------------------------------------------------------------------
        # Remove tracks expirados
        # ---------------------------------------------------------------------

        self._remove_expired_tracks()

        # ---------------------------------------------------------------------
        # Resultado
        # ---------------------------------------------------------------------

        tracks = tuple(
            sorted(
                self._tracks.values(),
                key=lambda track: track.track_id,
            )
        )

        stable_count = sum(
            1
            for track in tracks
            if track.stable
        )

        detected_count = sum(
            1
            for track in tracks
            if track.detected_this_frame
        )

        lost_count = sum(
            1
            for track in tracks
            if track.missed_frames > 0
        )

        valid = bool(
            detected_count > 0
        )

        return LaneTrackingResult(
            lanes=tracks,
            timestamp=timestamp,
            frame_index=self._frame_index,
            valid=valid,
            stable_count=stable_count,
            detected_count=detected_count,
            lost_count=lost_count,
        )

    # =========================================================================
    # CONTROLE
    # =========================================================================

    def reset(self) -> None:
        """
        Remove todo o estado temporal.
        """

        self._tracks.clear()

        self._next_track_id = 0

        self._frame_index = 0

        self._last_timestamp = None

    # =========================================================================
    # PROPRIEDADES
    # =========================================================================

    @property
    def tracks(self) -> Tuple[TrackedLane, ...]:
        """
        Tracks atualmente mantidos.
        """

        return tuple(
            sorted(
                self._tracks.values(),
                key=lambda track: track.track_id,
            )
        )

    @property
    def active_count(self) -> int:
        """
        Número de tracks ativos.
        """

        return len(
            self._tracks
        )


# =============================================================================
# API PÚBLICA
# =============================================================================

__all__ = [
    "TrackedLane",
    "LaneTrackingResult",
    "LaneTracker",
]