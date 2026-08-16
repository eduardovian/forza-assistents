"""
Testes do filtro temporal EMA.

O teste utiliza exclusivamente o contrato atual do projeto:
    vision.detection_types
    vision.temporal_filter

Não depende de UFLD, CULane ou âncoras específicas do detector.
"""

from __future__ import annotations

import math

import pytest

from vision.detection_types import (
    LaneDetectionResult,
    LanePoint,
)
from vision.temporal_filter import (
    EMATemporalFilter,
)


# ============================================================================
# HELPERS
# ============================================================================


def make_lane(
    x_offset: float,
    *,
    count: int = 18,
    confidence: float = 0.9,
) -> list[LanePoint]:
    """
    Cria uma lane sintética vertical.

    Os pontos possuem espaçamento regular em Y e deslocamento horizontal
    controlado por x_offset.
    """

    return [
        LanePoint(
            x=float(x_offset + index * 2.0),
            y=float(100.0 + index * 20.0),
            confidence=float(confidence),
            valid=True,
        )
        for index in range(count)
    ]


def make_detection(
    x_offset: float,
    *,
    confidence: float = 0.9,
    count: int = 18,
) -> LaneDetectionResult:
    """
    Cria uma detecção com duas lanes.
    """

    return LaneDetectionResult(
        lanes=[
            make_lane(
                x_offset,
                count=count,
                confidence=confidence,
            ),
            make_lane(
                x_offset + 300.0,
                count=count,
                confidence=confidence,
            ),
        ],
    )


def make_invalid_detection() -> LaneDetectionResult:
    """
    Detecção estruturalmente presente, mas sem pontos válidos.
    """

    return LaneDetectionResult(
        lanes=[
            [
                LanePoint(
                    x=float("nan"),
                    y=float("nan"),
                    confidence=0.0,
                    valid=False,
                )
            ]
        ]
    )


# ============================================================================
# TEST CLASS
# ============================================================================


class TestEMATemporalFilter:
    """Testes do filtro temporal EMA."""

    # ------------------------------------------------------------------
    # CONSTRUÇÃO
    # ------------------------------------------------------------------

    def test_filter_initializes(self):
        filt = EMATemporalFilter()

        assert filt.initialized is False
        assert filt.previous is None
        assert filt.missed_frames == 0

    # ------------------------------------------------------------------
    # PRIMEIRA DETECÇÃO
    # ------------------------------------------------------------------

    def test_first_detection_is_preserved(self):
        filt = EMATemporalFilter(alpha=0.5)

        detection = make_detection(100.0)

        result = filt.update(detection)

        assert result is not None
        assert filt.initialized is True

        assert len(result.lanes) == 2
        assert len(result.lanes[0]) == 18
        assert len(result.lanes[1]) == 18

        assert result.lanes[0][0].x == pytest.approx(100.0)

    # ------------------------------------------------------------------
    # EMA
    # ------------------------------------------------------------------

    def test_ema_smoothing(self):
        """
        Uma mudança brusca deve ser suavizada.
        """

        filt = EMATemporalFilter(
            alpha=0.5,
            max_point_distance=1000.0,
        )

        first = make_detection(100.0)
        second = make_detection(200.0)

        result_1 = filt.update(first)
        result_2 = filt.update(second)

        assert result_1 is not None
        assert result_2 is not None

        x = result_2.lanes[0][0].x

        assert x == pytest.approx(
            150.0,
            abs=1e-6,
        )

    def test_ema_convergence(self):
        """
        Repetidas amostras idênticas devem convergir para a nova posição.
        """

        filt = EMATemporalFilter(
            alpha=0.5,
            max_point_distance=1000.0,
        )

        filt.update(make_detection(100.0))

        for _ in range(10):
            result = filt.update(
                make_detection(200.0)
            )

        assert result is not None

        x = result.lanes[0][0].x

        assert x > 199.0
        assert x <= 200.0

    # ------------------------------------------------------------------
    # CONFIANÇA
    # ------------------------------------------------------------------

    def test_confidence_is_smoothed(self):
        filt = EMATemporalFilter(
            alpha=0.5,
            max_point_distance=1000.0,
        )

        filt.update(
            make_detection(
                100.0,
                confidence=0.4,
            )
        )

        result = filt.update(
            make_detection(
                100.0,
                confidence=1.0,
            )
        )

        assert result is not None

        confidence = result.lanes[0][0].confidence

        assert confidence == pytest.approx(
            0.7,
            abs=1e-6,
        )

    # ------------------------------------------------------------------
    # CONTAGEM
    # ------------------------------------------------------------------

    def test_point_count_preserved(self):
        filt = EMATemporalFilter()

        detection = make_detection(
            100.0,
            count=18,
        )

        result = filt.update(detection)

        assert result is not None

        assert len(result.lanes) == 2

        for lane in result.lanes:
            assert len(lane) == 18

    # ------------------------------------------------------------------
    # DETECÇÃO INVÁLIDA
    # ------------------------------------------------------------------

    def test_invalid_detection_decay(self):
        """
        Uma detecção inválida deve preservar temporariamente o estado,
        reduzindo sua confiança.
        """

        filt = EMATemporalFilter(
            invalid_decay=0.8,
            max_missed_frames=5,
        )

        valid = make_detection(
            150.0,
            confidence=0.9,
        )

        result_valid = filt.update(valid)

        assert result_valid is not None

        initial_confidence = (
            result_valid.lanes[0][0].confidence
        )

        result_invalid = filt.update(
            make_invalid_detection()
        )

        assert result_invalid is not None

        assert filt.missed_frames == 1

        decayed_confidence = (
            result_invalid.lanes[0][0].confidence
        )

        assert decayed_confidence < initial_confidence

        assert decayed_confidence == pytest.approx(
            initial_confidence * 0.8,
            abs=1e-6,
        )

    # ------------------------------------------------------------------
    # PERDA TEMPORÁRIA
    # ------------------------------------------------------------------

    def test_temporary_loss_preserves_geometry(self):
        filt = EMATemporalFilter(
            invalid_decay=0.9,
            max_missed_frames=3,
        )

        detection = make_detection(120.0)

        result = filt.update(detection)

        assert result is not None

        original_x = result.lanes[0][0].x

        result = filt.update(None)

        assert result is not None

        assert result.lanes[0][0].x == pytest.approx(
            original_x
        )

    def test_state_expires_after_max_missed_frames(self):
        filt = EMATemporalFilter(
            max_missed_frames=2,
        )

        filt.update(make_detection(100.0))

        assert filt.update(None) is not None
        assert filt.update(None) is not None

        result = filt.update(None)

        assert result is None
        assert filt.previous is None
        assert filt.initialized is False

    # ------------------------------------------------------------------
    # RESET
    # ------------------------------------------------------------------

    def test_reset(self):
        filt = EMATemporalFilter()

        filt.update(make_detection(100.0))

        assert filt.initialized is True

        filt.reset()

        assert filt.initialized is False
        assert filt.previous is None
        assert filt.missed_frames == 0

    # ------------------------------------------------------------------
    # RECUPERAÇÃO
    # ------------------------------------------------------------------

    def test_filter_recovers_after_invalid_frame(self):
        filt = EMATemporalFilter(
            alpha=0.5,
            invalid_decay=0.8,
            max_missed_frames=3,
            max_point_distance=1000.0,
        )

        filt.update(make_detection(100.0))

        filt.update(None)

        result = filt.update(
            make_detection(200.0)
        )

        assert result is not None

        assert filt.missed_frames == 0

        # O filtro deve voltar a utilizar a nova detecção.
        assert result.lanes[0][0].x > 100.0

    # ------------------------------------------------------------------
    # DISTÂNCIA DE ASSOCIAÇÃO
    # ------------------------------------------------------------------

    def test_large_jump_is_not_forced_into_previous_point(self):
        """
        Um salto acima do limite de associação deve ser tratado como
        novo ponto, e não como uma correspondência artificial.
        """

        filt = EMATemporalFilter(
            alpha=0.5,
            max_point_distance=10.0,
        )

        filt.update(make_detection(100.0))

        result = filt.update(
            make_detection(500.0)
        )

        assert result is not None

        assert result.lanes[0][0].x == pytest.approx(
            500.0
        )

    # ------------------------------------------------------------------
    # VALIDADE
    # ------------------------------------------------------------------

    def test_invalid_points_are_removed(self):
        detection = LaneDetectionResult(
            lanes=[
                [
                    LanePoint(
                        x=100.0,
                        y=100.0,
                        confidence=0.9,
                        valid=True,
                    ),
                    LanePoint(
                        x=float("nan"),
                        y=200.0,
                        confidence=0.9,
                        valid=True,
                    ),
                    LanePoint(
                        x=120.0,
                        y=300.0,
                        confidence=0.0,
                        valid=False,
                    ),
                ]
            ]
        )

        filt = EMATemporalFilter()

        result = filt.update(detection)

        assert result is not None

        assert len(result.lanes) == 1
        assert len(result.lanes[0]) == 1

        point = result.lanes[0][0]

        assert math.isfinite(point.x)
        assert math.isfinite(point.y)
        assert point.valid is True

    # ------------------------------------------------------------------
    # ALIAS
    # ------------------------------------------------------------------

    def test_filter_alias_matches_update(self):
        detection = make_detection(100.0)

        filter_a = EMATemporalFilter()
        filter_b = EMATemporalFilter()

        result_a = filter_a.update(detection)
        result_b = filter_b.filter(detection)

        assert result_a is not None
        assert result_b is not None

        assert result_a.lanes[0][0].x == pytest.approx(
            result_b.lanes[0][0].x
        )