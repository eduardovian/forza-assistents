"""
Testes para geometria da faixa, erro lateral e heading.

Cada lane tem 18 pontos (row anchors do CULane).
"""
import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.yolop_detector import LanePoint, LaneDetectionResult
from vision.lane_geometry import LaneGeometry, LaneGeometryResult


class TestLaneGeometry(unittest.TestCase):
    def setUp(self):
        self.geom = LaneGeometry(
            screen_width=2560,
            screen_height=1600,
            roi=(300, 700, 2200, 1600),
            ufld_width=800,
            ufld_height=288
        )

    def _create_lane_points(self, x_base, num_points=18, curve_amp=0):
        """Cria pontos de lane simulados com 18 row anchors."""
        from vision.ufld_detector import CULANE_ROW_ANCHORS
        points = []
        for i in range(num_points):
            y = CULANE_ROW_ANCHORS[i]
            x = x_base + np.sin(y / 50) * curve_amp
            points.append(LanePoint(x=x, y=y, confidence=0.9, valid=True))
        return points

    def test_centered_lane(self):
        """Veículo centrado na faixa -> erro lateral próximo de 0."""
        # Faixa centrada: left em ~300, right em ~500 (UFLD coords)
        # Centro = 400. ROI centro = 300 + 950 = 1250. Tela centro = 1280.
        left = self._create_lane_points(300)
        right = self._create_lane_points(500)

        detection = LaneDetectionResult(
            left_lane=left,
            right_lane=right,
            additional_lanes=[],
            left_confidence=0.9,
            right_confidence=0.9,
            valid=True,
            num_lanes_detected=2
        )

        result = self.geom.compute(detection)
        self.assertTrue(result.valid)
        # Erro lateral deve ser próximo de 0 (centro da faixa ~centro da tela)
        self.assertAlmostEqual(result.lateral_error, 0.0, delta=0.15)

    def test_left_drift(self):
        """Veículo desviando para a esquerda -> erro lateral positivo."""
        # Faixa deslocada para a direita (veículo à esquerda)
        left = self._create_lane_points(450)
        right = self._create_lane_points(650)

        detection = LaneDetectionResult(
            left_lane=left,
            right_lane=right,
            additional_lanes=[],
            left_confidence=0.9,
            right_confidence=0.9,
            valid=True,
            num_lanes_detected=2
        )

        result = self.geom.compute(detection)
        self.assertTrue(result.valid)
        # Erro lateral deve ser positivo (veículo à esquerda da faixa)
        self.assertGreater(result.lateral_error, 0.0)

    def test_invalid_detection(self):
        """Detecção inválida deve retornar geometria inválida."""
        detection = LaneDetectionResult(
            left_lane=[],
            right_lane=[],
            additional_lanes=[],
            left_confidence=0.0,
            right_confidence=0.0,
            valid=False,
            num_lanes_detected=0
        )

        result = self.geom.compute(detection)
        self.assertFalse(result.valid)
        self.assertEqual(result.lateral_error, 0.0)
        self.assertEqual(result.heading_error, 0.0)

    def test_lane_width(self):
        """Largura da faixa deve ser proporcional à distância entre lanes."""
        left = self._create_lane_points(300)
        right = self._create_lane_points(500)

        detection = LaneDetectionResult(
            left_lane=left,
            right_lane=right,
            additional_lanes=[],
            left_confidence=0.9,
            right_confidence=0.9,
            valid=True,
            num_lanes_detected=2
        )

        result = self.geom.compute(detection)
        # Distância no UFLD: 200 pixels
        # Escala: ROI_width/800 = 1900/800 = 2.375
        # Largura esperada na tela: ~200 * 2.375 = 475
        self.assertGreater(result.lane_width, 400)
        self.assertLess(result.lane_width, 600)

    def test_heading_straight(self):
        """Faixa reta -> heading próximo de 0."""
        left = self._create_lane_points(300, curve_amp=0)
        right = self._create_lane_points(500, curve_amp=0)

        detection = LaneDetectionResult(
            left_lane=left,
            right_lane=right,
            additional_lanes=[],
            left_confidence=0.9,
            right_confidence=0.9,
            valid=True,
            num_lanes_detected=2
        )

        result = self.geom.compute(detection)
        self.assertAlmostEqual(result.heading_error, 0.0, delta=0.1)

    def test_heading_curved(self):
        """Faixa curva -> heading diferente de 0."""
        # Cria lanes com inclinação (x aumenta com y)
        from vision.ufld_detector import CULANE_ROW_ANCHORS
        left_points = []
        right_points = []
        for i in range(18):
            y = CULANE_ROW_ANCHORS[i]
            # x aumenta com y (lane inclinada para direita)
            x_left = 200 + y * 0.3
            x_right = 400 + y * 0.3
            left_points.append(LanePoint(x=x_left, y=y, confidence=0.9, valid=True))
            right_points.append(LanePoint(x=x_right, y=y, confidence=0.9, valid=True))

        detection = LaneDetectionResult(
            left_lane=left_points,
            right_lane=right_points,
            additional_lanes=[],
            left_confidence=0.9,
            right_confidence=0.9,
            valid=True,
            num_lanes_detected=2
        )

        result = self.geom.compute(detection)
        # Heading deve ser positivo (inclinado para direita)
        self.assertGreater(result.heading_error, 0.0)


if __name__ == "__main__":
    unittest.main()