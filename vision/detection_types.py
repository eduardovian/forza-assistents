"""
vision/detection_types.py

Forza Assistents
================

Camada de compatibilidade para os tipos de detecção.

IMPORTANTE
----------
Os contratos canônicos do domínio estão definidos exclusivamente em:

    vision.lane_types

Este módulo NÃO deve criar versões alternativas de LanePoint,
LaneLine ou LaneDetectionResult.

Ele existe apenas para preservar imports durante a migração
da arquitetura antiga para o contrato canônico.

Arquitetura:

    detector
        ↓
    lane_types.LanePoint
    lane_types.LaneLine
    lane_types.LaneDetectionResult
        ↓
    tracker / geometry / model / projection / ADAS

Nenhum módulo novo deve adicionar novos tipos aqui.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from .lane_types import (
    CULANE_ROW_ANCHORS,
    LaneDetectionResult,
    LaneLine,
    LanePoint,
    LanePolynomial,
    LaneProjection,
    LaneSource,
    Point,
    points_from_xy,
)


# =============================================================================
# ALIASES DE COMPATIBILIDADE
# =============================================================================

# Nome antigo mantido temporariamente para evitar quebra imediata
# de módulos legados.
Point2D = Point


# =============================================================================
# HELPERS DE COMPATIBILIDADE
# =============================================================================

def make_lane_point(
    x: float,
    y: float,
    confidence: float = 1.0,
    valid: bool = True,
) -> LanePoint:
    """
    Cria um LanePoint usando o contrato canônico.

    Este helper existe somente para compatibilidade com código legado.
    """

    return LanePoint(
        x=x,
        y=y,
    )


def make_detection_result(
    lanes: Sequence[LaneLine],
    *,
    confidence: float = 0.0,
    image_width: int = 0,
    image_height: int = 0,
    valid: bool = True,
    frame_id: int | None = None,
    timestamp: float | None = None,
    metadata: dict | None = None,
) -> LaneDetectionResult:
    """
    Cria um LaneDetectionResult canônico.

    `valid`, `frame_id`, `timestamp` e `metadata` são aceitos para
    compatibilidade com consumidores antigos, mas o contrato principal
    permanece definido em vision.lane_types.
    """

    # O detector atual trabalha com LaneLine.
    #
    # Portanto não fazemos conversões implícitas ou reconstruções
    # perigosas aqui.
    normalized_lanes = tuple(lanes)

    return LaneDetectionResult(
        lanes=normalized_lanes,
        frame_width=image_width,
        frame_height=image_height,
        inference_ms=0.0,
        confidence=confidence,
    )


# =============================================================================
# API PÚBLICA
# =============================================================================

__all__ = [
    "Point",
    "Point2D",
    "CULANE_ROW_ANCHORS",
    "LaneSource",
    "LanePoint",
    "LaneLine",
    "LanePolynomial",
    "LaneProjection",
    "LaneDetectionResult",
    "points_from_xy",
    "make_lane_point",
    "make_detection_result",
]