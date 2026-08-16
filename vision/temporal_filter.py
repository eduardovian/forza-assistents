"""
vision/temporal_filter.py

Filtro temporal EMA para detecções e geometria de faixa.

Responsabilidades
-----------------
- Suavizar detecções de faixa entre frames.
- Reduzir jitter espacial.
- Tolerar perdas temporárias de detecção.
- Aplicar decay durante frames inválidos.
- Nunca transformar uma detecção inválida em uma detecção válida.
- Preservar a estrutura de LaneDetectionResult.
- Funcionar independentemente do detector utilizado (YOLOP, UFLD etc.).

O filtro não faz:
- classificação de faixas;
- modelagem geométrica;
- projeção;
- controle de direção.

Essas responsabilidades pertencem aos módulos seguintes do pipeline.
"""

from __future__ import annotations

from dataclasses import replace
from typing import List, Optional, Sequence

import math

from .detection_types import (
    LaneDetectionResult,
    LanePoint,
)


# ============================================================================
# CONFIGURAÇÃO
# ============================================================================


DEFAULT_ALPHA = 0.45
DEFAULT_INVALID_DECAY = 0.80
DEFAULT_MAX_MISSED_FRAMES = 8

DEFAULT_MIN_CONFIDENCE = 0.05
DEFAULT_MAX_POINT_DISTANCE = 100.0


# ============================================================================
# UTILITÁRIOS
# ============================================================================


def _finite(value: float) -> bool:
    """Retorna True somente para valores numéricos finitos."""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _valid_point(point: LanePoint) -> bool:
    """
    Valida um LanePoint sem alterar seu estado.

    Um ponto somente participa do filtro se:
    - x e y forem finitos;
    - confidence for finita;
    - valid estiver ativo;
    - confidence for positiva.
    """
    return (
        bool(getattr(point, "valid", True))
        and _finite(point.x)
        and _finite(point.y)
        and _finite(point.confidence)
        and float(point.confidence) > 0.0
    )


def _distance(a: LanePoint, b: LanePoint) -> float:
    return math.hypot(
        float(a.x) - float(b.x),
        float(a.y) - float(b.y),
    )


def _copy_point(
    point: LanePoint,
    *,
    x: Optional[float] = None,
    y: Optional[float] = None,
    confidence: Optional[float] = None,
    valid: Optional[bool] = None,
) -> LanePoint:
    """
    Cria uma cópia segura do LanePoint.

    Mantém compatibilidade com dataclasses que possam receber
    atributos adicionais através de replace().
    """
    values = {
        "x": float(point.x if x is None else x),
        "y": float(point.y if y is None else y),
        "confidence": float(
            point.confidence
            if confidence is None
            else confidence
        ),
        "valid": bool(
            point.valid
            if valid is None
            else valid
        ),
    }

    try:
        return replace(point, **values)
    except TypeError:
        return LanePoint(**values)


# ============================================================================
# EMA DE PONTOS
# ============================================================================


class EMATemporalFilter:
    """
    Filtro temporal EMA para LaneDetectionResult.

    Parâmetros
    ----------
    alpha:
        Peso da detecção atual.

        1.0 -> sem suavização.
        0.0 -> mantém completamente o estado anterior.

    invalid_decay:
        Fator aplicado à confiança quando a detecção atual está ausente
        ou inválida.

    max_missed_frames:
        Número máximo de frames que o estado anterior pode ser mantido
        durante uma perda temporária.

    max_point_distance:
        Distância máxima permitida para associar um novo ponto ao ponto
        filtrado anterior.
    """

    def __init__(
        self,
        alpha: float = DEFAULT_ALPHA,
        invalid_decay: float = DEFAULT_INVALID_DECAY,
        max_missed_frames: int = DEFAULT_MAX_MISSED_FRAMES,
        max_point_distance: float = DEFAULT_MAX_POINT_DISTANCE,
    ) -> None:

        if not _finite(alpha):
            raise ValueError("alpha deve ser finito.")

        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha deve estar entre 0 e 1.")

        if not _finite(invalid_decay):
            raise ValueError("invalid_decay deve ser finito.")

        if not 0.0 <= invalid_decay <= 1.0:
            raise ValueError(
                "invalid_decay deve estar entre 0 e 1."
            )

        if int(max_missed_frames) < 0:
            raise ValueError(
                "max_missed_frames não pode ser negativo."
            )

        if not _finite(max_point_distance):
            raise ValueError(
                "max_point_distance deve ser finito."
            )

        if float(max_point_distance) <= 0.0:
            raise ValueError(
                "max_point_distance deve ser positivo."
            )

        self.alpha = float(alpha)
        self.invalid_decay = float(invalid_decay)
        self.max_missed_frames = int(max_missed_frames)
        self.max_point_distance = float(max_point_distance)

        self._previous: Optional[LaneDetectionResult] = None
        self._missed_frames = 0

    # ------------------------------------------------------------------
    # PROPRIEDADES
    # ------------------------------------------------------------------

    @property
    def previous(self) -> Optional[LaneDetectionResult]:
        """Último estado filtrado."""
        return self._previous

    @property
    def missed_frames(self) -> int:
        """Quantidade de frames consecutivos sem detecção válida."""
        return self._missed_frames

    @property
    def initialized(self) -> bool:
        """Indica se o filtro já recebeu uma detecção."""
        return self._previous is not None

    # ------------------------------------------------------------------
    # RESET
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Limpa completamente o estado temporal."""
        self._previous = None
        self._missed_frames = 0

    # ------------------------------------------------------------------
    # PONTO
    # ------------------------------------------------------------------

    def _smooth_point(
        self,
        previous: LanePoint,
        current: LanePoint,
    ) -> LanePoint:
        """
        Aplica EMA espacial entre dois pontos associados.
        """

        alpha = self.alpha

        x = (
            alpha * float(current.x)
            + (1.0 - alpha) * float(previous.x)
        )

        y = (
            alpha * float(current.y)
            + (1.0 - alpha) * float(previous.y)
        )

        confidence = (
            alpha * float(current.confidence)
            + (1.0 - alpha) * float(previous.confidence)
        )

        confidence = _clamp(confidence, 0.0, 1.0)

        return _copy_point(
            current,
            x=x,
            y=y,
            confidence=confidence,
            valid=True,
        )

    # ------------------------------------------------------------------
    # ASSOCIAÇÃO
    # ------------------------------------------------------------------

    def _associate_points(
        self,
        previous_points: Sequence[LanePoint],
        current_points: Sequence[LanePoint],
    ) -> List[LanePoint]:
        """
        Associa pontos atuais aos pontos anteriores pelo vizinho mais próximo.

        Cada ponto anterior pode ser utilizado uma única vez.
        """

        previous_valid = [
            point
            for point in previous_points
            if _valid_point(point)
        ]

        current_valid = [
            point
            for point in current_points
            if _valid_point(point)
        ]

        if not previous_valid:
            return [
                _copy_point(point)
                for point in current_valid
            ]

        result: List[LanePoint] = []
        used_previous: set[int] = set()

        for current in current_valid:

            best_index: Optional[int] = None
            best_distance = float("inf")

            for index, previous in enumerate(previous_valid):

                if index in used_previous:
                    continue

                distance = _distance(current, previous)

                if distance < best_distance:
                    best_distance = distance
                    best_index = index

            if (
                best_index is not None
                and best_distance <= self.max_point_distance
            ):
                previous = previous_valid[best_index]
                used_previous.add(best_index)

                result.append(
                    self._smooth_point(
                        previous,
                        current,
                    )
                )
            else:
                # Novo ponto legítimo.
                result.append(
                    _copy_point(current)
                )

        return result

    # ------------------------------------------------------------------
    # LANES
    # ------------------------------------------------------------------

    def _filter_lanes(
        self,
        previous_lanes: Sequence[Sequence[LanePoint]],
        current_lanes: Sequence[Sequence[LanePoint]],
    ) -> List[List[LanePoint]]:
        """
        Filtra todas as lanes da detecção.

        A associação entre lanes é feita pela posição média horizontal
        para evitar que uma lane seja suavizada com outra.
        """

        previous = [
            list(lane)
            for lane in previous_lanes
        ]

        current = [
            list(lane)
            for lane in current_lanes
        ]

        if not current:
            return []

        if not previous:
            return [
                [
                    _copy_point(point)
                    for point in lane
                    if _valid_point(point)
                ]
                for lane in current
            ]

        def lane_center(
            lane: Sequence[LanePoint],
        ) -> Optional[float]:

            valid = [
                point
                for point in lane
                if _valid_point(point)
            ]

            if not valid:
                return None

            return sum(
                float(point.x)
                for point in valid
            ) / len(valid)

        previous_centers = [
            lane_center(lane)
            for lane in previous
        ]

        current_centers = [
            lane_center(lane)
            for lane in current
        ]

        result: List[List[LanePoint]] = []
        used_previous: set[int] = set()

        for current_lane, current_center in zip(
            current,
            current_centers,
        ):

            if current_center is None:
                continue

            best_index: Optional[int] = None
            best_distance = float("inf")

            for index, previous_center in enumerate(
                previous_centers
            ):

                if index in used_previous:
                    continue

                if previous_center is None:
                    continue

                distance = abs(
                    current_center - previous_center
                )

                if distance < best_distance:
                    best_distance = distance
                    best_index = index

            if best_index is not None:
                used_previous.add(best_index)

                filtered = self._associate_points(
                    previous[best_index],
                    current_lane,
                )
            else:
                filtered = [
                    _copy_point(point)
                    for point in current_lane
                    if _valid_point(point)
                ]

            if filtered:
                result.append(filtered)

        return result

    # ------------------------------------------------------------------
    # INVALID / DECAY
    # ------------------------------------------------------------------

    def _decay_previous(
        self,
    ) -> Optional[LaneDetectionResult]:
        """
        Mantém temporariamente o estado anterior com confiança reduzida.

        Depois de max_missed_frames, o estado é descartado.
        """

        if self._previous is None:
            return None

        self._missed_frames += 1

        if self._missed_frames > self.max_missed_frames:
            self.reset()
            return None

        lanes: List[List[LanePoint]] = []

        for lane in self._previous.lanes:
            filtered_lane: List[LanePoint] = []

            for point in lane:

                confidence = (
                    float(point.confidence)
                    * self.invalid_decay
                )

                confidence = _clamp(
                    confidence,
                    0.0,
                    1.0,
                )

                filtered_lane.append(
                    _copy_point(
                        point,
                        confidence=confidence,
                        valid=confidence > 0.0,
                    )
                )

            if filtered_lane:
                lanes.append(filtered_lane)

        previous = self._previous

        try:
            result = replace(
                previous,
                lanes=lanes,
            )
        except TypeError:
            result = LaneDetectionResult(
                lanes=lanes,
            )

        self._previous = result

        return result

    # ------------------------------------------------------------------
    # PROCESSAMENTO
    # ------------------------------------------------------------------

    def update(
        self,
        detection: Optional[LaneDetectionResult],
    ) -> Optional[LaneDetectionResult]:
        """
        Processa uma nova detecção.

        Regras:
        1. Primeira detecção válida -> inicializa.
        2. Detecção válida subsequente -> EMA.
        3. Detecção ausente/inválida -> decay.
        4. Após limite de perdas -> estado descartado.
        """

        if detection is None:
            return self._decay_previous()

        current_lanes = [
            lane
            for lane in detection.lanes
            if lane
        ]

        has_valid_points = any(
            _valid_point(point)
            for lane in current_lanes
            for point in lane
        )

        if not has_valid_points:
            return self._decay_previous()

        self._missed_frames = 0

        if self._previous is None:

            lanes = [
                [
                    _copy_point(point)
                    for point in lane
                    if _valid_point(point)
                ]
                for lane in current_lanes
            ]

        else:

            lanes = self._filter_lanes(
                self._previous.lanes,
                current_lanes,
            )

        try:
            result = replace(
                detection,
                lanes=lanes,
            )
        except TypeError:
            result = LaneDetectionResult(
                lanes=lanes,
            )

        self._previous = result

        return result

    # ------------------------------------------------------------------
    # ALIAS
    # ------------------------------------------------------------------

    def filter(
        self,
        detection: Optional[LaneDetectionResult],
    ) -> Optional[LaneDetectionResult]:
        """
        Alias semântico para update().
        """
        return self.update(detection)


# ============================================================================
# FUNÇÃO DE CONVENIÊNCIA
# ============================================================================


def create_default_filter() -> EMATemporalFilter:
    """
    Cria o filtro temporal padrão utilizado pelo pipeline.
    """
    return EMATemporalFilter(
        alpha=DEFAULT_ALPHA,
        invalid_decay=DEFAULT_INVALID_DECAY,
        max_missed_frames=DEFAULT_MAX_MISSED_FRAMES,
        max_point_distance=DEFAULT_MAX_POINT_DISTANCE,
    )


__all__ = [
    "EMATemporalFilter",
    "create_default_filter",
    "DEFAULT_ALPHA",
    "DEFAULT_INVALID_DECAY",
    "DEFAULT_MAX_MISSED_FRAMES",
    "DEFAULT_MIN_CONFIDENCE",
    "DEFAULT_MAX_POINT_DISTANCE",
]