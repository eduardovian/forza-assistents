"""
Forza Horizon 6 ADAS/LKA
Debug Overlay - visualização robusta das faixas

Responsabilidades:
- Desenhar ROI
- Desenhar todas as lanes detectadas
- Destacar a faixa atual
- Desenhar linhas suavizadas
- Desenhar centro da faixa
- Desenhar centro da imagem
- Mostrar heading/lateral error
- Mostrar métricas
- Nunca alterar os dados de detecção
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from vision.ufld_detector import LaneDetectionResult, LanePoint
from vision.lane_geometry import LaneGeometryResult

logger = logging.getLogger(__name__)

Point = Tuple[float, float]
Color = Tuple[int, int, int]


class DebugOverlay:
    """
    Overlay de debug para o sistema de visão.

    IMPORTANTE:
    Este módulo é somente visualização.
    Não modifica detecção, geometria ou controle.
    """

    def __init__(
        self,
        roi: Tuple[int, int, int, int] = (300, 700, 2200, 1600),
        font_scale: float = 0.6,
        font_thickness: int = 2,
        line_thickness: int = 4,
        point_radius: int = 3,
    ):
        self.roi = roi

        self.font_scale = float(font_scale)
        self.font_thickness = int(font_thickness)
        self.line_thickness = int(line_thickness)
        self.point_radius = int(point_radius)

        # ------------------------------------------------------------------
        # Cores BGR
        # ------------------------------------------------------------------

        self.color_roi: Color = (255, 255, 0)

        # Faixa atual
        self.color_current_left: Color = (0, 165, 255)
        self.color_current_right: Color = (255, 0, 255)

        # Faixas adjacentes
        self.color_adjacent: Color = (180, 180, 180)

        # Centro
        self.color_center: Color = (0, 255, 0)

        # Centro da imagem
        self.color_image_center: Color = (0, 0, 255)

        # Heading
        self.color_heading: Color = (255, 255, 255)

        # Pontos
        self.color_points: Color = (255, 255, 255)

    # ======================================================================
    # UTILITÁRIOS
    # ======================================================================

    @staticmethod
    def _sanitize_points(
        points: Optional[Sequence[Point]],
    ) -> List[Point]:
        """
        Remove pontos inválidos e garante ordenação vertical.

        A ordenação é feita por Y crescente.
        """

        if not points:
            return []

        clean: List[Point] = []

        for point in points:
            if point is None or len(point) != 2:
                continue

            try:
                x = float(point[0])
                y = float(point[1])
            except (TypeError, ValueError):
                continue

            if not np.isfinite(x) or not np.isfinite(y):
                continue

            if x <= 0.0 or y <= 0.0:
                continue

            clean.append((x, y))

        clean.sort(key=lambda p: p[1])

        return clean

    @staticmethod
    def _smooth_polyline(
        points: Sequence[Point],
        samples: int = 100,
    ) -> List[Point]:
        """
        Gera uma curva suave usando interpolação por polinômio de baixo grau.

        NÃO usa polinômio de grau alto:
        isso evita o comportamento de "cobra" causado por oscilações.

        O eixo de interpolação é Y.
        """

        clean = DebugOverlay._sanitize_points(points)

        if len(clean) < 2:
            return clean

        if len(clean) == 2:
            return clean

        arr = np.asarray(clean, dtype=np.float64)

        x = arr[:, 0]
        y = arr[:, 1]

        # ------------------------------------------------------------------
        # Remove Y duplicado.
        # ------------------------------------------------------------------

        unique_y, unique_indices = np.unique(y, return_index=True)

        if len(unique_y) < 2:
            return clean

        x = x[unique_indices]
        y = unique_y

        # ------------------------------------------------------------------
        # Grau limitado.
        #
        # Máximo 2 = curva quadrática.
        #
        # Isso é deliberado:
        # grau alto pode criar oscilações artificiais.
        # ------------------------------------------------------------------

        degree = min(2, len(x) - 1)

        try:
            coeffs = np.polyfit(y, x, degree)

            y_dense = np.linspace(
                float(y.min()),
                float(y.max()),
                max(samples, len(y) * 8),
            )

            x_dense = np.polyval(coeffs, y_dense)

        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            return clean

        # ------------------------------------------------------------------
        # Proteção contra extrapolação absurda.
        # ------------------------------------------------------------------

        x_min = float(np.min(x))
        x_max = float(np.max(x))

        margin = max(100.0, (x_max - x_min) * 0.5)

        x_dense = np.clip(
            x_dense,
            x_min - margin,
            x_max + margin,
        )

        return [
            (float(px), float(py))
            for px, py in zip(x_dense, y_dense)
        ]

    # ======================================================================
    # ROI
    # ======================================================================

    def draw_roi(
        self,
        frame: np.ndarray,
        color: Optional[Color] = None,
    ) -> None:
        """Desenha a região de interesse."""

        if color is None:
            color = self.color_roi

        left, top, right, bottom = self.roi

        cv2.rectangle(
            frame,
            (int(left), int(top)),
            (int(right), int(bottom)),
            color,
            2,
        )

        cv2.putText(
            frame,
            "VISION ROI",
            (int(left) + 8, max(20, int(top) - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.font_scale,
            color,
            self.font_thickness,
            cv2.LINE_AA,
        )

    # ======================================================================
    # LANE
    # ======================================================================

    def draw_lane(
        self,
        frame: np.ndarray,
        points: Sequence[Point],
        color: Color,
        label: str = "",
        thickness: Optional[int] = None,
        draw_points: bool = True,
    ) -> None:
        """
        Desenha uma lane como curva suavizada.

        Nunca conecta diretamente pontos consecutivos com segmentos
        extremamente agressivos.
        """

        clean = self._sanitize_points(points)

        if len(clean) < 2:
            return

        if thickness is None:
            thickness = self.line_thickness

        smooth = self._smooth_polyline(clean)

        if len(smooth) >= 2:
            pts = np.asarray(
                smooth,
                dtype=np.int32,
            ).reshape(-1, 1, 2)

            cv2.polylines(
                frame,
                [pts],
                False,
                color,
                int(thickness),
                cv2.LINE_AA,
            )

        # --------------------------------------------------------------
        # Pontos originais
        # --------------------------------------------------------------

        if draw_points:
            for x, y in clean:
                cv2.circle(
                    frame,
                    (int(round(x)), int(round(y))),
                    self.point_radius,
                    self.color_points,
                    -1,
                    cv2.LINE_AA,
                )

        # --------------------------------------------------------------
        # Label
        # --------------------------------------------------------------

        if label:
            x, y = clean[-1]

            cv2.putText(
                frame,
                label,
                (
                    int(x) + 10,
                    int(y),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                color,
                self.font_thickness,
                cv2.LINE_AA,
            )

    # ======================================================================
    # CENTER LINE
    # ======================================================================

    def draw_center_line(
        self,
        frame: np.ndarray,
        points: Sequence[Point],
        color: Optional[Color] = None,
    ) -> None:
        """Desenha o centro da faixa."""

        if color is None:
            color = self.color_center

        clean = self._sanitize_points(points)

        if len(clean) < 2:
            return

        smooth = self._smooth_polyline(clean, samples=120)

        pts = np.asarray(
            smooth,
            dtype=np.int32,
        ).reshape(-1, 1, 2)

        cv2.polylines(
            frame,
            [pts],
            False,
            color,
            max(2, self.line_thickness),
            cv2.LINE_AA,
        )

    # ======================================================================
    # IMAGE CENTER
    # ======================================================================

    def draw_image_center(
        self,
        frame: np.ndarray,
        center_x: float,
        color: Optional[Color] = None,
    ) -> None:
        """Desenha o centro horizontal da imagem."""

        if color is None:
            color = self.color_image_center

        h, _ = frame.shape[:2]

        x = int(round(center_x))

        cv2.line(
            frame,
            (x, 0),
            (x, h),
            color,
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            "IMAGE CENTER",
            (x + 8, h - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.font_scale,
            color,
            self.font_thickness,
            cv2.LINE_AA,
        )

    # ======================================================================
    # HEADING
    # ======================================================================

    def draw_heading(
        self,
        frame: np.ndarray,
        center_x: float,
        center_y: float,
        heading_error: float,
        color: Optional[Color] = None,
    ) -> None:
        """
        Desenha direção estimada da faixa.

        heading_error esperado em [-1, 1].
        """

        if color is None:
            color = self.color_heading

        if not np.isfinite(heading_error):
            return

        heading_error = float(
            np.clip(heading_error, -1.0, 1.0)
        )

        # Não desenha praticamente zero.
        if abs(heading_error) < 0.005:
            return

        arrow_length = 180.0

        angle = heading_error * (np.pi / 4.0)

        dx = arrow_length * np.sin(angle)
        dy = -arrow_length * np.cos(angle)

        start = (
            int(round(center_x)),
            int(round(center_y)),
        )

        end = (
            int(round(center_x + dx)),
            int(round(center_y + dy)),
        )

        cv2.arrowedLine(
            frame,
            start,
            end,
            color,
            4,
            cv2.LINE_AA,
            tipLength=0.20,
        )

    # ======================================================================
    # PONTOS UFLD
    # ======================================================================

    @staticmethod
    def _lane_points_to_screen(
        lane: Sequence[LanePoint],
    ) -> List[Point]:
        """
        Converte LanePoint em pontos de tela.

        A conversão para tela já deve ter sido realizada pela geometria.
        Este método existe apenas para casos de debug futuros.
        """

        result: List[Point] = []

        for point in lane:
            if not point.valid:
                continue

            if point.confidence <= 0:
                continue

            if point.x <= 0 or point.y <= 0:
                continue

            result.append(
                (
                    float(point.x),
                    float(point.y),
                )
            )

        return result

    # ======================================================================
    # MÉTRICAS
    # ======================================================================

    def draw_metrics(
        self,
        frame: np.ndarray,
        fps: float,
        capture_latency_ms: float,
        inference_latency_ms: float,
        total_latency_ms: float,
        left_conf: float,
        right_conf: float,
        lateral_error: float,
        heading_error: float,
        lane_width: float,
        mode: str = "VISION_ONLY",
        gpu: str = "CUDA",
        valid: bool = True,
    ) -> None:
        """Desenha painel de diagnóstico."""

        panel_width = 470
        panel_height = 410

        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (5, 5),
            (panel_width, panel_height),
            (0, 0, 0),
            -1,
        )

        cv2.addWeighted(
            overlay,
            0.68,
            frame,
            0.32,
            0,
            frame,
        )

        values = [
            (
                "FORZA ADAS / LKA",
                (0, 255, 255),
            ),
            (
                f"Mode: {mode}",
                (0, 255, 0),
            ),
            (
                f"GPU: {gpu}",
                (0, 255, 0),
            ),
            (
                "G29 CONTROL: DISABLED",
                (0, 0, 255),
            ),
            (
                "",
                (255, 255, 255),
            ),
            (
                f"FPS: {fps:.1f}",
                (0, 255, 0),
            ),
            (
                f"Capture: {capture_latency_ms:.1f} ms",
                (0, 255, 0),
            ),
            (
                f"Inference: {inference_latency_ms:.1f} ms",
                (0, 255, 0),
            ),
            (
                f"Total: {total_latency_ms:.1f} ms",
                (0, 255, 0),
            ),
            (
                "",
                (255, 255, 255),
            ),
            (
                f"Left confidence: {left_conf:.3f}",
                (0, 165, 255),
            ),
            (
                f"Right confidence: {right_conf:.3f}",
                (255, 0, 255),
            ),
            (
                f"Lateral error: {lateral_error:+.3f}",
                (0, 255, 0),
            ),
            (
                f"Heading error: {heading_error:+.3f}",
                (255, 255, 255),
            ),
            (
                f"Lane width: {lane_width:.1f} px",
                (0, 255, 0),
            ),
            (
                "",
                (255, 255, 255),
            ),
            (
                f"GEOMETRY: {'VALID' if valid else 'INVALID'}",
                (0, 255, 0) if valid else (0, 0, 255),
            ),
        ]

        x = 15
        y = 32
        line_height = 23

        for text, color in values:
            if not text:
                y += line_height
                continue

            cv2.putText(
                frame,
                text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                color,
                self.font_thickness,
                cv2.LINE_AA,
            )

            y += line_height

    # ======================================================================
    # RENDER PRINCIPAL
    # ======================================================================

    def render(
        self,
        frame: np.ndarray,
        detection: Optional[LaneDetectionResult],
        geometry: Optional[LaneGeometryResult],
        fps: float = 0.0,
        capture_latency_ms: float = 0.0,
        inference_latency_ms: float = 0.0,
        total_latency_ms: float = 0.0,
        mode: str = "VISION_ONLY",
        gpu: str = "CUDA",
    ) -> np.ndarray:
        """
        Renderiza o overlay completo.

        A ordem é:

            ROI
              ↓
            lanes adjacentes
              ↓
            lanes atuais
              ↓
            centro
              ↓
            heading
              ↓
            métricas
        """

        if frame is None:
            raise ValueError("DebugOverlay.render recebeu frame=None")

        if not isinstance(frame, np.ndarray):
            raise TypeError(
                "DebugOverlay.render espera numpy.ndarray"
            )

        result = frame.copy()

        # --------------------------------------------------------------
        # ROI
        # --------------------------------------------------------------

        self.draw_roi(result)

        # --------------------------------------------------------------
        # Geometria
        # --------------------------------------------------------------

        if geometry is not None:

            # ==========================================================
            # LANES ATUAIS
            # ==========================================================

            left_lane = self._sanitize_points(
                geometry.left_lane_screen
            )

            right_lane = self._sanitize_points(
                geometry.right_lane_screen
            )

            if left_lane:
                self.draw_lane(
                    result,
                    left_lane,
                    self.color_current_left,
                    label="CURRENT LEFT",
                    thickness=self.line_thickness,
                )

            if right_lane:
                self.draw_lane(
                    result,
                    right_lane,
                    self.color_current_right,
                    label="CURRENT RIGHT",
                    thickness=self.line_thickness,
                )

            # ==========================================================
            # CENTRO DA FAIXA
            # ==========================================================

            center_line = self._sanitize_points(
                geometry.center_line
            )

            if center_line:
                self.draw_center_line(
                    result,
                    center_line,
                    self.color_center,
                )

            # ==========================================================
            # CENTRO DA IMAGEM
            # ==========================================================

            self.draw_image_center(
                result,
                geometry.image_center_x,
            )

            # ==========================================================
            # HEADING
            # ==========================================================

            self.draw_heading(
                result,
                geometry.lane_center_x,
                geometry.lane_center_y,
                geometry.heading_error,
            )

        # --------------------------------------------------------------
        # MÉTRICAS
        # --------------------------------------------------------------

        left_conf = (
            float(detection.left_confidence)
            if detection is not None
            else 0.0
        )

        right_conf = (
            float(detection.right_confidence)
            if detection is not None
            else 0.0
        )

        lateral_error = (
            float(geometry.lateral_error)
            if geometry is not None
            else 0.0
        )

        heading_error = (
            float(geometry.heading_error)
            if geometry is not None
            else 0.0
        )

        lane_width = (
            float(geometry.lane_width)
            if geometry is not None
            else 0.0
        )

        valid = (
            bool(geometry.valid)
            if geometry is not None
            else False
        )

        self.draw_metrics(
            result,
            fps=fps,
            capture_latency_ms=capture_latency_ms,
            inference_latency_ms=inference_latency_ms,
            total_latency_ms=total_latency_ms,
            left_conf=left_conf,
            right_conf=right_conf,
            lateral_error=lateral_error,
            heading_error=heading_error,
            lane_width=lane_width,
            mode=mode,
            gpu=gpu,
            valid=valid,
        )

        return result
    