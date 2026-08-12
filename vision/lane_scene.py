"""
vision/lane_scene.py

Construção do snapshot completo da cena de lanes.

Responsabilidade:

    LaneModel
        ↓
    LaneAssociation
        ↓
    LaneGeometry
        ↓
    VehiclePosition
        ↓
    LaneConfidence
        ↓
    LaneScene

Este módulo NÃO executa:
    - inferência YOLOP;
    - tracking;
    - ajuste polinomial;
    - projeção;
    - associação;
    - decisão ADAS;
    - atuação.

Ele apenas consolida as informações produzidas pelas camadas
anteriores em um único objeto de cena.

Compatibilidade:
    Utiliza os tipos já existentes em vision/lane_types.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from .lane_types import (
    CurrentLane,
    LaneAssociationResult,
    LaneGeometry,
    LaneModel,
    LaneWarning,
)
from .lane_confidence import (
    LaneConfidenceResult,
    SceneConfidenceResult,
)


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

DEFAULT_MIN_SCENE_CONFIDENCE = 0.55
DEFAULT_SAFE_SCENE_CONFIDENCE = 0.70


# =============================================================================
# RESULTADO
# =============================================================================

@dataclass
class LaneScene:
    """
    Estado consolidado da percepção de lanes em um frame.

    O objeto representa a informação necessária para as camadas
    superiores sem obrigá-las a conhecer a implementação interna
    de cada módulo de visão.
    """

    frame_index: int = 0

    timestamp: float = 0.0

    lanes: List[LaneModel] = field(
        default_factory=list
    )

    association: Optional[
        LaneAssociationResult
    ] = None

    current_lane: Optional[
        CurrentLane
    ] = None

    geometry: Optional[
        LaneGeometry
    ] = None

    vehicle_position: Optional[object] = None

    confidence: Optional[
        SceneConfidenceResult
    ] = None

    warning: Optional[
        LaneWarning
    ] = None

    valid: bool = False

    perception_valid: bool = False

    safe_for_adas: bool = False

    error: Optional[str] = None

    # -------------------------------------------------------------------------
    # Propriedades de acesso rápido
    # -------------------------------------------------------------------------

    @property
    def lane_count(self) -> int:
        return len(self.lanes)

    @property
    def current_lane_id(self) -> Optional[int]:

        if self.association is not None:
            return self.association.current_lane_id

        if self.current_lane is not None:

            left = self.current_lane.left_boundary

            right = self.current_lane.right_boundary

            if left is not None:
                return left.lane_id

            if right is not None:
                return right.lane_id

        return None

    @property
    def lateral_offset(self) -> Optional[float]:

        if self.current_lane is not None:

            if (
                self.current_lane.normalized_offset
                is not None
            ):
                return self.current_lane.normalized_offset

            return self.current_lane.lateral_offset

        if self.geometry is not None:

            if (
                self.geometry.normalized_offset
                is not None
            ):
                return self.geometry.normalized_offset

            return self.geometry.lateral_offset

        if self.vehicle_position is not None:

            value = getattr(
                self.vehicle_position,
                "normalized_error",
                None,
            )

            if value is not None:
                return value

        return None

    @property
    def lane_center_x(self) -> Optional[float]:

        if self.current_lane is not None:
            return self.current_lane.center_x

        if self.geometry is not None:
            return self.geometry.current_center_x

        value = getattr(
            self.vehicle_position,
            "lane_center_x",
            None,
        )

        return value

    @property
    def lane_width(self) -> Optional[float]:

        if self.current_lane is not None:
            return self.current_lane.lane_width

        if self.geometry is not None:
            return self.geometry.lane_width

        value = getattr(
            self.vehicle_position,
            "lane_width",
            None,
        )

        return value

    @property
    def heading_error(self) -> Optional[float]:

        if self.geometry is None:
            return None

        return self.geometry.heading_error

    @property
    def curvature(self) -> Optional[float]:

        if self.geometry is None:
            return None

        return self.geometry.curvature

    @property
    def confidence_value(self) -> float:

        if self.confidence is None:
            return 0.0

        return float(
            np.clip(
                self.confidence.confidence,
                0.0,
                1.0,
            )
        )

    @property
    def is_centered(self) -> bool:

        if self.current_lane is None:
            return False

        offset = (
            self.current_lane.normalized_offset
        )

        if offset is None:
            return False

        return abs(offset) <= 0.10


# =============================================================================
# CONSTRUTOR
# =============================================================================

class LaneSceneBuilder:
    """
    Constrói LaneScene a partir dos resultados das camadas anteriores.

    O builder não modifica os objetos recebidos.
    """

    def __init__(
        self,
        min_scene_confidence: float = (
            DEFAULT_MIN_SCENE_CONFIDENCE
        ),
        safe_scene_confidence: float = (
            DEFAULT_SAFE_SCENE_CONFIDENCE
        ),
    ) -> None:

        self.min_scene_confidence = float(
            np.clip(
                min_scene_confidence,
                0.0,
                1.0,
            )
        )

        self.safe_scene_confidence = float(
            np.clip(
                safe_scene_confidence,
                self.min_scene_confidence,
                1.0,
            )
        )

        self.last_scene: Optional[
            LaneScene
        ] = None

    # =========================================================================
    # UTILITÁRIOS
    # =========================================================================

    @staticmethod
    def _finite(value: object) -> bool:

        try:
            return bool(
                np.isfinite(
                    float(value)
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            return False

    @staticmethod
    def _clip01(value: float) -> float:

        if not np.isfinite(value):
            return 0.0

        return float(
            np.clip(
                value,
                0.0,
                1.0,
            )
        )

    @staticmethod
    def _copy_lanes(
        lanes: Optional[
            Sequence[LaneModel]
        ],
    ) -> List[LaneModel]:

        if lanes is None:
            return []

        return [
            lane
            for lane in lanes
            if lane is not None
        ]

    # =========================================================================
    # VALIDAÇÃO
    # =========================================================================

    def _validate_scene(
        self,
        lanes: Sequence[LaneModel],
        association: Optional[
            LaneAssociationResult
        ],
        geometry: Optional[
            LaneGeometry
        ],
        confidence: Optional[
            SceneConfidenceResult
        ],
        current_lane: Optional[
            CurrentLane
        ],
    ) -> tuple[
        bool,
        bool,
        bool,
        Optional[str],
    ]:

        if not lanes:

            return (
                False,
                False,
                False,
                "Nenhuma lane disponível.",
            )

        association_valid = (
            association is not None
            and association.valid
        )

        geometry_valid = (
            geometry is not None
            and geometry.valid
        )

        current_lane_valid = (
            current_lane is not None
            and current_lane.valid
        )

        confidence_valid = (
            confidence is not None
            and confidence.valid
            and confidence.confidence
            >= self.min_scene_confidence
        )

        perception_valid = (
            association_valid
            or current_lane_valid
            or geometry_valid
        )

        valid = (
            perception_valid
            and confidence_valid
        )

        safe = (
            valid
            and confidence is not None
            and confidence.safe_for_adas
            and confidence.confidence
            >= self.safe_scene_confidence
            and current_lane_valid
            and geometry_valid
        )

        if not perception_valid:
            error = (
                "Informação geométrica insuficiente."
            )

        elif not confidence_valid:
            error = (
                "Confiança da percepção insuficiente."
            )

        elif not safe:
            error = (
                "Percepção válida, mas não segura "
                "para ADAS."
            )

        else:
            error = None

        return (
            valid,
            perception_valid,
            safe,
            error,
        )

    # =========================================================================
    # BUILD
    # =========================================================================

    def build(
        self,
        lanes: Optional[
            Sequence[LaneModel]
        ] = None,
        association: Optional[
            LaneAssociationResult
        ] = None,
        current_lane: Optional[
            CurrentLane
        ] = None,
        geometry: Optional[
            LaneGeometry
        ] = None,
        vehicle_position: Optional[
            object
        ] = None,
        confidence: Optional[
            SceneConfidenceResult
        ] = None,
        warning: Optional[
            LaneWarning
        ] = None,
        frame_index: int = 0,
        timestamp: float = 0.0,
    ) -> LaneScene:

        lane_list = self._copy_lanes(
            lanes
        )

        # Se as lanes não foram fornecidas diretamente,
        # aproveitamos as lanes da associação.
        if (
            not lane_list
            and association is not None
        ):
            lane_list = self._copy_lanes(
                association.lanes
            )

        valid, perception_valid, safe, error = (
            self._validate_scene(
                lane_list,
                association,
                geometry,
                confidence,
                current_lane,
            )
        )

        scene = LaneScene(
            frame_index=int(
                frame_index
            ),
            timestamp=float(
                timestamp
            ),
            lanes=lane_list,
            association=association,
            current_lane=current_lane,
            geometry=geometry,
            vehicle_position=vehicle_position,
            confidence=confidence,
            warning=warning,
            valid=valid,
            perception_valid=perception_valid,
            safe_for_adas=safe,
            error=error,
        )

        self.last_scene = scene

        return scene

    # =========================================================================
    # BUILD A PARTIR DOS OBJETOS EXISTENTES
    # =========================================================================

    def from_association(
        self,
        association: LaneAssociationResult,
        geometry: Optional[
            LaneGeometry
        ] = None,
        vehicle_position: Optional[
            object
        ] = None,
        confidence: Optional[
            SceneConfidenceResult
        ] = None,
        warning: Optional[
            LaneWarning
        ] = None,
        frame_index: int = 0,
        timestamp: float = 0.0,
    ) -> LaneScene:

        current_lane = (
            association.current_lane
            if association is not None
            else None
        )

        lanes = (
            association.lanes
            if association is not None
            else []
        )

        return self.build(
            lanes=lanes,
            association=association,
            current_lane=current_lane,
            geometry=geometry,
            vehicle_position=vehicle_position,
            confidence=confidence,
            warning=warning,
            frame_index=frame_index,
            timestamp=timestamp,
        )

    # =========================================================================
    # ATUALIZAÇÃO
    # =========================================================================

    def update(
        self,
        lanes: Optional[
            Sequence[LaneModel]
        ] = None,
        association: Optional[
            LaneAssociationResult
        ] = None,
        current_lane: Optional[
            CurrentLane
        ] = None,
        geometry: Optional[
            LaneGeometry
        ] = None,
        vehicle_position: Optional[
            object
        ] = None,
        confidence: Optional[
            SceneConfidenceResult
        ] = None,
        warning: Optional[
            LaneWarning
        ] = None,
        frame_index: int = 0,
        timestamp: float = 0.0,
    ) -> LaneScene:

        return self.build(
            lanes=lanes,
            association=association,
            current_lane=current_lane,
            geometry=geometry,
            vehicle_position=vehicle_position,
            confidence=confidence,
            warning=warning,
            frame_index=frame_index,
            timestamp=timestamp,
        )


# =============================================================================
# FACTORY
# =============================================================================

def create_default_lane_scene_builder(
    **kwargs,
) -> LaneSceneBuilder:

    return LaneSceneBuilder(
        **kwargs
    )


__all__ = [
    "LaneScene",
    "LaneSceneBuilder",
    "create_default_lane_scene_builder",
]