"""
Testes para o filtro temporal EMA.

Cada lane tem 18 pontos (row anchors do CULane).
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.ufld_detector import LanePoint, LaneDetectionResult
from vision.lane_geometry import LaneGeometryResult
from vision.temporal_filter import EMATemporalFilter


class TestEMATemporalFilter(unittest.TestCase):
    def setUp(self):
        self.filter = EMATemporalFilter(alpha=0.3)

    def _create_detection(self, x_offset, num_points=18):
        """Cria detecção simulada com 18 pontos por lane."""
        from vision.ufld_detector import CULANE_ROW_ANCHORS
        points = []
        for i in range(num_points):
            y = CULANE_ROW_ANCHORS[i]
            points.append(LanePoint(x=x_offset, y=y, confidence=0.9, valid=True))

        return LaneDetectionResult(
            left_lane=points,
            right_lane=points,
            additional_lanes=[],
            left_confidence=0.9,
            right_confidence=0.9,
            valid=True,
            num_lanes_detected=2
        )

    def test_ema_smoothing(self):
        """EMA deve suavizar transições bruscas."""
        det1 = self._create_detection(100.0)
        det2 = self._create_detection(200.0)

        self.filter.filter_detection(det1)
        result = self.filter.filter_detection(det2)

        # Resultado deve estar entre 100 e 200
        first_point = result.left_lane[0]
        self.assertGreater(first_point.x, 100.0)
        self.assertLess(first_point.x, 200.0)

    def test_ema_convergence(self):
        """EMA deve convergir após múltiplas amostras idênticas."""
        det = self._create_detection(150.0)

        for _ in range(20):
            result = self.filter.filter_detection(det)

        first_point = result.left_lane[0]
        self.assertAlmostEqual(first_point.x, 150.0, delta=1.0)

    def test_point_count_preserved(self):
        """O filtro deve preservar o número de pontos (18)."""
        det = self._create_detection(100.0)
        result = self.filter.filter_detection(det)
        self.assertEqual(len(result.left_lane), 18)
        self.assertEqual(len(result.right_lane), 18)

    def test_invalid_detection_decay(self):
        """Detecção inválida deve causar decaimento da geometria."""
        det_valid = self._create_detection(150.0)
        det_invalid = LaneDetectionResult(
            left_lane=[], right_lane=[], additional_lanes=[],
            left_confidence=0.0, right_confidence=0.0,
            valid=False, num_lanes_detected=0
        )

        self.filter.filter_detection(det_valid)

        geom_valid = LaneGeometryResult(
            additional_lanes_screen=[],
            selected_left_index=None,
            selected_right_index=None,
            lane_center_x=1280, lane_center_y=1150,
            image_center_x=1280, image_center_y=800,
            lateral_error=0.1, heading_error=0.05,
            lane_width=500, curvature=0.1,
            center_line=[(1280, 1150)], valid=True,
            left_lane_screen=[], right_lane_screen=[]
        )
        geom_invalid = LaneGeometryResult(
            additional_lanes_screen=[],
            selected_left_index=None,
            selected_right_index=None,
            lane_center_x=1280, lane_center_y=1150,
            image_center_x=1280, image_center_y=800,
            lateral_error=0.0, heading_error=0.0,
            lane_width=0, curvature=0.0,
            center_line=[], valid=False,
            left_lane_screen=[], right_lane_screen=[]
        )

        self.filter.filter_geometry(geom_valid)
        result = self.filter.filter_geometry(geom_invalid)

        self.assertFalse(result.valid)
        self.assertLess(abs(result.lateral_error), 0.1)

    def test_reset(self):
        """Reset deve limpar o estado do filtro."""
        det = self._create_detection(100.0)
        self.filter.filter_detection(det)
        self.filter.reset()

        det2 = self._create_detection(200.0)
        result = self.filter.filter_detection(det2)
        first_point = result.left_lane[0]
        self.assertEqual(first_point.x, 200.0)


if __name__ == "__main__":
    unittest.main()