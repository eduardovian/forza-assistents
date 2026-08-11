"""
Forza Horizon ADAS/LKA
Filtro temporal robusto para detecção e geometria de faixas.

Objetivos:
- Reduzir jitter entre frames.
- Preservar curvas reais.
- Rejeitar saltos impossíveis.
- Não inventar faixas quando a detecção desaparece.
- Manter compatibilidade com UFLDLaneDetector e LaneGeometry.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import List, Optional, Tuple

import numpy as np

from .ufld_detector import LaneDetectionResult, LanePoint
from .lane_geometry import LaneGeometryResult

logger = logging.getLogger(__name__)


class EMATemporalFilter:
    """
    Filtro temporal para lanes e geometria.

    O filtro trabalha separadamente nas duas bordas da faixa.

    Características:
    - EMA adaptativa.
    - Rejeição de saltos grandes.
    - Decaimento controlado quando uma lane desaparece.
    - Não transforma ausência de detecção em detecção válida.
    """

    def __init__(
        self,
        alpha: float = 0.35,
        min_alpha: float = 0.15,
        max_alpha: float = 0.65,
        max_x_jump: float = 90.0,
        max_y_jump: float = 20.0,
        max_missing_frames: int = 3,
        min_valid_points: int = 3,
    ):
        self.alpha = float(np.clip(alpha, 0.01, 1.0))
        self.min_alpha = float(np.clip(min_alpha, 0.01, 1.0))
        self.max_alpha = float(np.clip(max_alpha, self.min_alpha, 1.0))

        self.max_x_jump = float(max_x_jump)
        self.max_y_jump = float(max_y_jump)
        self.max_missing_frames = int(max(0, max_missing_frames))
        self.min_valid_points = int(max(1, min_valid_points))

        self._left: Optional[List[LanePoint]] = None
        self._right: Optional[List[LanePoint]] = None

        self._geometry: Optional[LaneGeometryResult] = None

        self._left_missing = 0
        self._right_missing = 0
        self._geometry_missing = 0

        self.reset()

    # ------------------------------------------------------------------
    # RESET
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Limpa completamente o estado temporal."""
        self._left = None
        self._right = None
        self._geometry = None

        self._left_missing = 0
        self._right_missing = 0
        self._geometry_missing = 0

    # ------------------------------------------------------------------
    # UTILITÁRIOS
    # ------------------------------------------------------------------

    @staticmethod
    def _valid_points(points: List[LanePoint]) -> List[LanePoint]:
        """Retorna somente pontos geometricamente válidos."""
        return [
            p
            for p in points
            if p.valid
            and np.isfinite(p.x)
            and np.isfinite(p.y)
            and np.isfinite(p.confidence)
            and p.confidence > 0.0
        ]

    def _lane_is_valid(self, lane: List[LanePoint]) -> bool:
        return len(self._valid_points(lane)) >= self.min_valid_points

    @staticmethod
    def _ema(old: float, new: float, alpha: float) -> float:
        return old + alpha * (new - old)

    def _adaptive_alpha(self, old_x: float, new_x: float) -> float:
        """
        Quanto maior o deslocamento, menor a confiança no novo frame.

        Isso evita que um único erro do detector faça a linha "pular".
        """
        distance = abs(new_x - old_x)

        if distance >= self.max_x_jump:
            return self.min_alpha

        ratio = distance / max(self.max_x_jump, 1e-6)

        return (
            self.max_alpha
            - ratio * (self.max_alpha - self.min_alpha)
        )

    # ------------------------------------------------------------------
    # LANE FILTER
    # ------------------------------------------------------------------

    def _filter_lane(
        self,
        current: List[LanePoint],
        previous: Optional[List[LanePoint]],
    ) -> List[LanePoint]:
        """
        Suaviza uma lane mantendo os mesmos row anchors.

        Pontos inválidos não são convertidos artificialmente em válidos.
        """

        if previous is None:
            return list(current)

        if len(current) != len(previous):
            return list(current)

        result: List[LanePoint] = []

        for current_point, previous_point in zip(current, previous):

            # ----------------------------------------------------------
            # Novo ponto inválido
            # ----------------------------------------------------------
            if not current_point.valid:

                # Mantemos o ponto anterior apenas como histórico,
                # mas não declaramos o novo frame como válido.
                result.append(
                    LanePoint(
                        x=previous_point.x,
                        y=previous_point.y,
                        confidence=max(
                            0.0,
                            previous_point.confidence * 0.85
                        ),
                        valid=False,
                    )
                )
                continue

            # ----------------------------------------------------------
            # Primeiro ponto válido
            # ----------------------------------------------------------
            if not previous_point.valid:
                result.append(current_point)
                continue

            # ----------------------------------------------------------
            # Verificações numéricas
            # ----------------------------------------------------------
            if not (
                np.isfinite(current_point.x)
                and np.isfinite(current_point.y)
                and np.isfinite(previous_point.x)
                and np.isfinite(previous_point.y)
            ):
                result.append(
                    LanePoint(
                        x=previous_point.x,
                        y=previous_point.y,
                        confidence=0.0,
                        valid=False,
                    )
                )
                continue

            dx = abs(current_point.x - previous_point.x)
            dy = abs(current_point.y - previous_point.y)

            # ----------------------------------------------------------
            # Salto impossível
            # ----------------------------------------------------------
            if dx > self.max_x_jump or dy > self.max_y_jump:
                alpha = self.min_alpha
            else:
                alpha = self._adaptive_alpha(
                    previous_point.x,
                    current_point.x,
                )

            filtered_x = self._ema(
                previous_point.x,
                current_point.x,
                alpha,
            )

            filtered_y = self._ema(
                previous_point.y,
                current_point.y,
                alpha,
            )

            filtered_confidence = self._ema(
                previous_point.confidence,
                current_point.confidence,
                alpha,
            )

            result.append(
                LanePoint(
                    x=float(filtered_x),
                    y=float(filtered_y),
                    confidence=float(
                        np.clip(filtered_confidence, 0.0, 1.0)
                    ),
                    valid=True,
                )
            )

        return result

    # ------------------------------------------------------------------
    # DETECTION
    # ------------------------------------------------------------------

    def filter_detection(
        self,
        detection: LaneDetectionResult,
    ) -> LaneDetectionResult:
        """
        Filtra uma detecção completa.

        Importante:
        o filtro nunca transforma uma detecção inválida em válida
        apenas porque existia uma detecção anterior.
        """

        if detection is None:
            return detection

        current_left = detection.left_lane
        current_right = detection.right_lane

        left_valid = self._lane_is_valid(current_left)
        right_valid = self._lane_is_valid(current_right)

        # --------------------------------------------------------------
        # LEFT
        # --------------------------------------------------------------

        if left_valid:
            filtered_left = self._filter_lane(
                current_left,
                self._left,
            )
            self._left = filtered_left
            self._left_missing = 0
        else:
            self._left_missing += 1

            if (
                self._left is not None
                and self._left_missing <= self.max_missing_frames
            ):
                filtered_left = self._decay_lane(self._left)
            else:
                filtered_left = list(current_left)

        # --------------------------------------------------------------
        # RIGHT
        # --------------------------------------------------------------

        if right_valid:
            filtered_right = self._filter_lane(
                current_right,
                self._right,
            )
            self._right = filtered_right
            self._right_missing = 0
        else:
            self._right_missing += 1

            if (
                self._right is not None
                and self._right_missing <= self.max_missing_frames
            ):
                filtered_right = self._decay_lane(self._right)
            else:
                filtered_right = list(current_right)

        # --------------------------------------------------------------
        # ADDITIONAL LANES
        # --------------------------------------------------------------

        additional_lanes = [
            list(lane)
            for lane in detection.additional_lanes
        ]

        # --------------------------------------------------------------
        # CONFIDENCE
        # --------------------------------------------------------------

        left_conf = self._lane_confidence(filtered_left)
        right_conf = self._lane_confidence(filtered_right)

        left_points = sum(
            1 for p in filtered_left if p.valid
        )

        right_points = sum(
            1 for p in filtered_right if p.valid
        )

        valid = (
            left_points >= self.min_valid_points
            and right_points >= self.min_valid_points
        )

        return LaneDetectionResult(
            left_lane=filtered_left,
            right_lane=filtered_right,
            additional_lanes=additional_lanes,
            left_confidence=float(left_conf),
            right_confidence=float(right_conf),
            valid=valid,
            num_lanes_detected=detection.num_lanes_detected,
        )

    def _decay_lane(
        self,
        lane: List[LanePoint],
    ) -> List[LanePoint]:
        """
        Reduz progressivamente a confiança de uma lane desaparecida.

        Não mantém a lane como válida indefinidamente.
        """

        result = []

        for point in lane:
            confidence = point.confidence * 0.70

            result.append(
                LanePoint(
                    x=point.x,
                    y=point.y,
                    confidence=float(confidence),
                    valid=False,
                )
            )

        return result

    @staticmethod
    def _lane_confidence(
        lane: List[LanePoint],
    ) -> float:
        values = [
            p.confidence
            for p in lane
            if p.valid and p.confidence > 0.0
        ]

        if not values:
            return 0.0

        return float(np.mean(values))

    # ------------------------------------------------------------------
    # GEOMETRY
    # ------------------------------------------------------------------

    def filter_geometry(
        self,
        geometry: LaneGeometryResult,
    ) -> LaneGeometryResult:
        """
        Suaviza a geometria calculada.

        A geometria só é suavizada quando válida.
        Uma geometria inválida não deve ser usada para fabricar
        uma trajetória aparentemente válida.
        """

        if geometry is None:
            return geometry

        if not geometry.valid:
            self._geometry_missing += 1

            # Depois de alguns frames sem geometria válida,
            # descartamos completamente o histórico.
            if self._geometry_missing > self.max_missing_frames:
                self._geometry = None

            return geometry

        self._geometry_missing = 0

        if self._geometry is None:
            self._geometry = geometry
            return geometry

        previous = self._geometry

        alpha = self._geometry_alpha(
            previous,
            geometry,
        )

        filtered_center_line = self._filter_points(
            previous.center_line,
            geometry.center_line,
            alpha,
        )

        filtered_left = self._filter_xy_points(
            previous.left_lane_screen,
            geometry.left_lane_screen,
            alpha,
        )

        filtered_right = self._filter_xy_points(
            previous.right_lane_screen,
            geometry.right_lane_screen,
            alpha,
        )

        result = replace(
            geometry,
            lane_center_x=float(
                self._ema(
                    previous.lane_center_x,
                    geometry.lane_center_x,
                    alpha,
                )
            ),
            lane_center_y=float(
                self._ema(
                    previous.lane_center_y,
                    geometry.lane_center_y,
                    alpha,
                )
            ),
            lateral_error=float(
                self._ema(
                    previous.lateral_error,
                    geometry.lateral_error,
                    alpha,
                )
            ),
            heading_error=float(
                self._ema(
                    previous.heading_error,
                    geometry.heading_error,
                    alpha,
                )
            ),
            lane_width=float(
                self._ema(
                    previous.lane_width,
                    geometry.lane_width,
                    alpha,
                )
            ),
            curvature=float(
                self._ema(
                    previous.curvature,
                    geometry.curvature,
                    alpha,
                )
            ),
            center_line=filtered_center_line,
            left_lane_screen=filtered_left,
            right_lane_screen=filtered_right,
            valid=True,
        )

        self._geometry = result

        return result

    def _geometry_alpha(
        self,
        previous: LaneGeometryResult,
        current: LaneGeometryResult,
    ) -> float:
        """
        Reduz alpha quando há uma mudança geométrica muito grande.
        """

        center_jump = abs(
            current.lane_center_x -
            previous.lane_center_x
        )

        heading_jump = abs(
            current.heading_error -
            previous.heading_error
        )

        if center_jump > 150.0:
            return self.min_alpha

        if heading_jump > 0.35:
            return self.min_alpha

        ratio = center_jump / 150.0

        return float(
            np.clip(
                self.max_alpha
                - ratio * (
                    self.max_alpha -
                    self.min_alpha
                ),
                self.min_alpha,
                self.max_alpha,
            )
        )

    @staticmethod
    def _filter_points(
        previous: List[Tuple[float, float]],
        current: List[Tuple[float, float]],
        alpha: float,
    ) -> List[Tuple[float, float]]:
        """
        Suaviza center_line.

        Quando as quantidades de pontos são diferentes,
        não tenta fazer correspondência por índice de forma cega.
        Nesse caso utiliza a linha atual.
        """

        if not previous:
            return list(current)

        if not current:
            return list(previous)

        if len(previous) != len(current):
            return list(current)

        result = []

        for (px, py), (cx, cy) in zip(previous, current):
            x = px + alpha * (cx - px)
            y = py + alpha * (cy - py)

            result.append(
                (
                    float(x),
                    float(y),
                )
            )

        return result

    @staticmethod
    def _filter_xy_points(
        previous: List[Tuple[float, float]],
        current: List[Tuple[float, float]],
        alpha: float,
    ) -> List[Tuple[float, float]]:
        """
        Suaviza pontos de uma lane na tela.
        """

        if not previous:
            return list(current)

        if not current:
            return list(previous)

        if len(previous) != len(current):
            return list(current)

        result = []

        for (px, py), (cx, cy) in zip(previous, current):

            x = px + alpha * (cx - px)
            y = py + alpha * (cy - py)

            result.append(
                (
                    float(x),
                    float(y),
                )
            )

        return result