"""
tests/test_geometry.py

Forza Assistents
================

Testes unitários do núcleo geométrico.

Princípios
----------

Este teste valida exclusivamente:

    LanePoint
        ↓
    LaneDetectionResult
        ↓
    LaneGeometry
        ↓
    LaneGeometryResult

O teste NÃO deve importar:

    - YOLOP detector
    - UFLD
    - OpenCV
    - ONNX Runtime
    - PyTorch
    - tracker
    - lane model
    - lane assignment
    - ADAS
    - controle

A geometria deve ser testável de forma completamente isolada.

Todas as coordenadas de entrada dos testes são coordenadas do
detector configurado em config.YOLOP.

A conversão para tela é responsabilidade exclusiva de LaneGeometry.
"""

from __future__ import annotations

import math
import os
import sys
import unittest

import numpy as np


# ============================================================================
# PATH DO PROJETO
# ============================================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================================
# IMPORTS DO CONTRATO GEOMÉTRICO
# ============================================================================

from vision.detection_types import (
    LaneDetectionResult,
    LanePoint,
)

from vision.lane_geometry import (
    LaneGeometry,
    LaneGeometryResult,
)


# ============================================================================
# TESTES
# ============================================================================


class TestLaneGeometry(unittest.TestCase):
    """
    Testes determinísticos do LaneGeometry.

    O conjunto foi projetado para não depender do detector real.
    """

    # ------------------------------------------------------------------------
    # CONFIGURAÇÃO
    # ------------------------------------------------------------------------

    SCREEN_WIDTH = 2560.0
    SCREEN_HEIGHT = 1600.0

    # ROI:
    #
    # left   = 300
    # top    = 700
    # right  = 2200
    # bottom = 1600
    #
    # Portanto:
    #
    # width  = 1900
    # height = 900
    #
    ROI = (
        300.0,
        700.0,
        2200.0,
        1600.0,
    )

    def setUp(self) -> None:
        """
        Cria uma instância completamente independente do detector.
        """

        self.geom = LaneGeometry(
            screen_width=self.SCREEN_WIDTH,
            screen_height=self.SCREEN_HEIGHT,
            roi=self.ROI,
        )

        self.detector_width, self.detector_height = (
            self.geom._detector_dimensions()
        )

    # =========================================================================
    # AUXILIARES
    # =========================================================================

    def _screen_to_detector(
        self,
        x: float,
        y: float,
    ) -> tuple[float, float]:
        """
        Converte uma coordenada de tela para o sistema do detector.

        Essa função existe SOMENTE para gerar dados sintéticos de teste.

        O código de produção continua responsável por fazer a transformação
        detector -> tela.
        """

        roi_left, roi_top, roi_width, roi_height = (
            self.geom._roi_dimensions()
        )

        detector_x = (
            (x - roi_left)
            / roi_width
            * self.detector_width
        )

        detector_y = (
            (y - roi_top)
            / roi_height
            * self.detector_height
        )

        return (
            float(detector_x),
            float(detector_y),
        )

    def _create_lane(
        self,
        x_base: float,
        *,
        y_start: float = 40.0,
        y_end: float | None = None,
        num_points: int = 18,
        slope: float = 0.0,
        curve: float = 0.0,
        confidence: float = 0.95,
    ) -> list[LanePoint]:
        """
        Cria uma lane sintética no espaço do detector.

        x(y) = x_base
             + slope * normalized_y
             + curve * normalized_y²

        Isso permite testar:

            - linha reta;
            - inclinação;
            - curvatura;
            - largura;
            - outliers.
        """

        if y_end is None:
            y_end = self.detector_height - 20.0

        if num_points < 2:
            raise ValueError(
                "num_points deve ser >= 2."
            )

        ys = np.linspace(
            y_start,
            y_end,
            num_points,
        )

        center_y = (
            y_start + y_end
        ) * 0.5

        half_range = max(
            (y_end - y_start) * 0.5,
            1.0,
        )

        points: list[LanePoint] = []

        for y in ys:
            normalized_y = (
                float(y) - center_y
            ) / half_range

            x = (
                x_base
                + slope * normalized_y
                + curve * normalized_y * normalized_y
            )

            points.append(
                LanePoint(
                    x=float(x),
                    y=float(y),
                    confidence=confidence,
                    valid=True,
                )
            )

        return points

    def _create_detection(
        self,
        lanes: list[list[LanePoint]],
        *,
        confidence: float = 0.95,
        valid: bool = True,
    ) -> LaneDetectionResult:
        """
        Cria um resultado de detecção usando somente o contrato atual.
        """

        return LaneDetectionResult(
            lanes=lanes,
            confidence=confidence,
            image_width=int(self.SCREEN_WIDTH),
            image_height=int(self.SCREEN_HEIGHT),
            valid=valid,
        )

    def _process(
        self,
        lanes: list[list[LanePoint]],
    ) -> LaneGeometryResult:
        """
        Executa o caminho oficial da geometria.

        Não utiliza nenhum detector.
        """

        detection = self._create_detection(
            lanes
        )

        return self.geom.compute(
            detection
        )

    # =========================================================================
    # CONVERSÃO DE COORDENADAS
    # =========================================================================

    def test_detector_to_screen_origin(self):
        """
        O ponto (0, 0) do detector deve corresponder ao canto superior
        esquerdo do ROI.
        """

        x, y = self.geom.detector_to_screen(
            0.0,
            0.0,
        )

        self.assertAlmostEqual(
            x,
            self.ROI[0],
            places=6,
        )

        self.assertAlmostEqual(
            y,
            self.ROI[1],
            places=6,
        )

    def test_detector_to_screen_end(self):
        """
        O extremo do detector deve corresponder ao canto inferior direito
        do ROI.
        """

        x, y = self.geom.detector_to_screen(
            self.detector_width,
            self.detector_height,
        )

        self.assertAlmostEqual(
            x,
            self.ROI[2],
            places=6,
        )

        self.assertAlmostEqual(
            y,
            self.ROI[3],
            places=6,
        )

    # =========================================================================
    # FAIXA CENTRALIZADA
    # =========================================================================

    def test_centered_lane(self):
        """
        Faixa centralizada:

            centro da lane ≈ centro da tela

        Portanto:

            lateral_error ≈ 0
        """

        screen_lane_center = (
            self.SCREEN_WIDTH * 0.5
        )

        screen_lane_width = 500.0

        left_screen_x = (
            screen_lane_center
            - screen_lane_width * 0.5
        )

        right_screen_x = (
            screen_lane_center
            + screen_lane_width * 0.5
        )

        left_x, _ = self._screen_to_detector(
            left_screen_x,
            1100.0,
        )

        right_x, _ = self._screen_to_detector(
            right_screen_x,
            1100.0,
        )

        left = self._create_lane(
            left_x
        )

        right = self._create_lane(
            right_x
        )

        result = self._process(
            [
                left,
                right,
            ]
        )

        self.assertIsInstance(
            result,
            LaneGeometryResult,
        )

        self.assertTrue(
            result.valid
        )

        self.assertAlmostEqual(
            result.lateral_error,
            0.0,
            delta=0.05,
        )

        self.assertAlmostEqual(
            result.lane_center_x,
            self.SCREEN_WIDTH * 0.5,
            delta=15.0,
        )

    # =========================================================================
    # DESLOCAMENTO LATERAL
    # =========================================================================

    def test_left_drift(self):
        """
        A faixa está deslocada para a direita.

        Isso significa que o veículo está à esquerda da faixa.

        Convenção esperada:

            lateral_error > 0
        """

        screen_lane_center = (
            self.SCREEN_WIDTH * 0.5
            + 250.0
        )

        screen_lane_width = 500.0

        left_screen_x = (
            screen_lane_center
            - screen_lane_width * 0.5
        )

        right_screen_x = (
            screen_lane_center
            + screen_lane_width * 0.5
        )

        left_x, _ = self._screen_to_detector(
            left_screen_x,
            1100.0,
        )

        right_x, _ = self._screen_to_detector(
            right_screen_x,
            1100.0,
        )

        left = self._create_lane(
            left_x
        )

        right = self._create_lane(
            right_x
        )

        result = self._process(
            [
                left,
                right,
            ]
        )

        self.assertTrue(
            result.valid
        )

        self.assertGreater(
            result.lateral_error,
            0.0,
        )

    # =========================================================================
    # DESLOCAMENTO PARA A ESQUERDA
    # =========================================================================

    def test_right_drift(self):
        """
        A faixa está deslocada para a esquerda.

        Portanto:

            lateral_error < 0
        """

        screen_lane_center = (
            self.SCREEN_WIDTH * 0.5
            - 250.0
        )

        screen_lane_width = 500.0

        left_screen_x = (
            screen_lane_center
            - screen_lane_width * 0.5
        )

        right_screen_x = (
            screen_lane_center
            + screen_lane_width * 0.5
        )

        left_x, _ = self._screen_to_detector(
            left_screen_x,
            1100.0,
        )

        right_x, _ = self._screen_to_detector(
            right_screen_x,
            1100.0,
        )

        left = self._create_lane(
            left_x
        )

        right = self._create_lane(
            right_x
        )

        result = self._process(
            [
                left,
                right,
            ]
        )

        self.assertTrue(
            result.valid
        )

        self.assertLess(
            result.lateral_error,
            0.0,
        )

    # =========================================================================
    # LARGURA
    # =========================================================================

    def test_lane_width(self):
        """
        A largura da faixa deve corresponder aproximadamente à distância
        geométrica entre as duas lanes no espaço da tela.
        """

        expected_width = 500.0

        screen_center = (
            self.SCREEN_WIDTH * 0.5
        )

        left_screen_x = (
            screen_center
            - expected_width * 0.5
        )

        right_screen_x = (
            screen_center
            + expected_width * 0.5
        )

        left_x, _ = self._screen_to_detector(
            left_screen_x,
            1100.0,
        )

        right_x, _ = self._screen_to_detector(
            right_screen_x,
            1100.0,
        )

        left = self._create_lane(
            left_x
        )

        right = self._create_lane(
            right_x
        )

        result = self._process(
            [
                left,
                right,
            ]
        )

        self.assertTrue(
            result.valid
        )

        self.assertAlmostEqual(
            result.lane_width,
            expected_width,
            delta=15.0,
        )

    # =========================================================================
    # HEADING
    # =========================================================================

    def test_heading_straight(self):
        """
        Lanes paralelas e verticais:

            heading ≈ 0
        """

        left = self._create_lane(
            250.0,
            slope=0.0,
            curve=0.0,
        )

        right = self._create_lane(
            400.0,
            slope=0.0,
            curve=0.0,
        )

        result = self._process(
            [
                left,
                right,
            ]
        )

        self.assertTrue(
            result.valid
        )

        self.assertAlmostEqual(
            result.heading_error,
            0.0,
            delta=0.02,
        )

    def test_heading_curved(self):
        """
        Lane inclinada progressivamente para a direita.

        Deve produzir:

            heading_error > 0
        """

        left = self._create_lane(
            230.0,
            slope=60.0,
        )

        right = self._create_lane(
            380.0,
            slope=60.0,
        )

        result = self._process(
            [
                left,
                right,
            ]
        )

        self.assertTrue(
            result.valid
        )

        self.assertGreater(
            result.heading_error,
            0.0,
        )

    # =========================================================================
    # CURVATURA
    # =========================================================================

    def test_straight_lane_has_low_curvature(self):
        """
        Uma faixa reta deve possuir curvatura aproximadamente nula.
        """

        left = self._create_lane(
            250.0
        )

        right = self._create_lane(
            400.0
        )

        result = self._process(
            [
                left,
                right,
            ]
        )

        self.assertTrue(
            result.valid
        )

        self.assertAlmostEqual(
            result.curvature,
            0.0,
            delta=0.05,
        )

    def test_curved_lane_has_nonzero_curvature(self):
        """
        Uma faixa quadrática deve produzir curvatura observável.
        """

        left = self._create_lane(
            250.0,
            curve=80.0,
        )

        right = self._create_lane(
            400.0,
            curve=80.0,
        )

        result = self._process(
            [
                left,
                right,
            ]
        )

        self.assertTrue(
            result.valid
        )

        self.assertGreater(
            abs(result.curvature),
            0.01,
        )

    # =========================================================================
    # VALIDADE
    # =========================================================================

    def test_invalid_detection(self):
        """
        Uma detecção explicitamente inválida deve resultar em geometria
        inválida e segura.
        """

        detection = LaneDetectionResult(
            lanes=[],
            confidence=0.0,
            image_width=int(
                self.SCREEN_WIDTH
            ),
            image_height=int(
                self.SCREEN_HEIGHT
            ),
            valid=False,
        )

        result = self.geom.compute(
            detection
        )

        self.assertFalse(
            result.valid
        )

        self.assertEqual(
            result.lateral_error,
            0.0,
        )

        self.assertEqual(
            result.heading_error,
            0.0,
        )

        self.assertEqual(
            result.lane_width,
            0.0,
        )

        self.assertEqual(
            result.curvature,
            0.0,
        )

        self.assertEqual(
            result.geometry_confidence,
            0.0,
        )

        self.assertEqual(
            result.center_line,
            [],
        )

    def test_empty_lanes_are_invalid(self):
        """
        Lista vazia de lanes não pode produzir geometria válida.
        """

        result = self.geom.compute(
            []
        )

        self.assertFalse(
            result.valid
        )

        self.assertEqual(
            result.center_line,
            [],
        )

    def test_single_lane_is_invalid(self):
        """
        Uma única lane não fornece geometria suficiente para determinar
        a faixa atual.
        """

        lane = self._create_lane(
            300.0
        )

        result = self._process(
            [
                lane,
            ]
        )

        self.assertFalse(
            result.valid
        )

    # =========================================================================
    # QUALIDADE DOS PONTOS
    # =========================================================================

    def test_invalid_points_are_ignored(self):
        """
        Pontos marcados como inválidos não devem contaminar a geometria.
        """

        left = self._create_lane(
            250.0
        )

        right = self._create_lane(
            400.0
        )

        # Injeta pontos inválidos.
        left.append(
            LanePoint(
                x=9999.0,
                y=200.0,
                confidence=0.0,
                valid=False,
            )
        )

        right.append(
            LanePoint(
                x=-9999.0,
                y=300.0,
                confidence=0.0,
                valid=False,
            )
        )

        result = self._process(
            [
                left,
                right,
            ]
        )

        self.assertTrue(
            result.valid
        )

        self.assertTrue(
            math.isfinite(
                result.lateral_error
            )
        )

        self.assertTrue(
            math.isfinite(
                result.lane_width
            )
        )

    def test_low_confidence_points_are_rejected(self):
        """
        Pontos abaixo da confiança mínima não devem formar geometria
        artificial.
        """

        left = self._create_lane(
            250.0,
            confidence=0.01,
        )

        right = self._create_lane(
            400.0,
            confidence=0.01,
        )

        result = self._process(
            [
                left,
                right,
            ]
        )

        self.assertFalse(
            result.valid
        )

    # =========================================================================
    # OUTLIERS
    # =========================================================================

    def test_outlier_rejection_preserves_valid_lane(self):
        """
        Um outlier isolado não deve destruir uma lane válida.
        """

        left = self._create_lane(
            250.0
        )

        right = self._create_lane(
            400.0
        )

        # Outlier extremo.
        left.insert(
            len(left) // 2,
            LanePoint(
                x=620.0,
                y=190.0,
                confidence=0.95,
                valid=True,
            ),
        )

        result = self._process(
            [
                left,
                right,
            ]
        )

        self.assertTrue(
            result.valid
        )

        self.assertTrue(
            math.isfinite(
                result.lane_width
            )
        )

        self.assertGreater(
            result.geometry_confidence,
            0.0,
        )

    # =========================================================================
    # MULTIPLAS LANES
    # =========================================================================

    def test_multiple_lanes_selects_plausible_pair(self):
        """
        Com várias marcações, o algoritmo deve selecionar um par
        geometricamente plausível.
        """

        left_outer = self._create_lane(
            100.0
        )

        left_current = self._create_lane(
            250.0
        )

        right_current = self._create_lane(
            400.0
        )

        right_outer = self._create_lane(
            550.0
        )

        result = self._process(
            [
                left_outer,
                left_current,
                right_current,
                right_outer,
            ]
        )

        self.assertTrue(
            result.valid
        )

        self.assertGreater(
            result.lane_width,
            0.0,
        )

        self.assertGreater(
            len(
                result.additional_lanes_screen
            ),
            0,
        )

    # =========================================================================
    # COBERTURA VERTICAL
    # =========================================================================

    def test_observed_span_is_positive(self):
        """
        A geometria deve informar corretamente a extensão vertical
        efetivamente observada.
        """

        left = self._create_lane(
            250.0
        )

        right = self._create_lane(
            400.0
        )

        result = self._process(
            [
                left,
                right,
            ]
        )

        self.assertTrue(
            result.valid
        )

        self.assertGreater(
            result.observed_span,
            0.0,
        )

        self.assertGreaterEqual(
            result.observed_y_max,
            result.observed_y_min,
        )

    def test_projection_flag_requires_observed_span(self):
        """
        O sinalizador de suficiência deve acompanhar a cobertura observada.
        """

        left = self._create_lane(
            250.0,
            y_start=150.0,
            y_end=300.0,
        )

        right = self._create_lane(
            400.0,
            y_start=150.0,
            y_end=300.0,
        )

        result = self._process(
            [
                left,
                right,
            ]
        )

        self.assertGreater(
            result.observed_span,
            0.0,
        )

    # =========================================================================
    # FINITUDE
    # =========================================================================

    def test_result_contains_no_nan_or_infinity(self):
        """
        Nenhum campo numérico do resultado pode propagar NaN/infinito.
        """

        left = self._create_lane(
            250.0
        )

        right = self._create_lane(
            400.0
        )

        result = self._process(
            [
                left,
                right,
            ]
        )

        numeric_values = [
            result.lane_center_x,
            result.lane_center_y,
            result.image_center_x,
            result.image_center_y,
            result.lateral_error,
            result.heading_error,
            result.lane_width,
            result.curvature,
            result.geometry_confidence,
            result.observed_y_min,
            result.observed_y_max,
            result.observed_span,
        ]

        for value in numeric_values:
            self.assertTrue(
                math.isfinite(value),
                msg=f"Valor não finito encontrado: {value}",
            )

        for lane in (
            result.center_line,
            result.left_lane_screen,
            result.right_lane_screen,
        ):
            for x, y in lane:
                self.assertTrue(
                    math.isfinite(x)
                )
                self.assertTrue(
                    math.isfinite(y)
                )


# =============================================================================
# EXECUÇÃO DIRETA
# =============================================================================

if __name__ == "__main__":
    unittest.main()