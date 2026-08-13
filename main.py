"""
main.py

Forza Assistents
================

Pipeline principal em tempo real:

    Screen Capture
        ↓
    ROI
        ↓
    YOLOP Detector
        ↓
    Lane Tracker
        ↓
    Lane Geometry
        ↓
    Lane Models
        ↓
    Lane Projection
        ↓
    Lane Assignment
        ↓
    ADAS State
        ↓
    Monitor / Debug

IMPORTANTE:
- O sistema inicia em MONITOR.
- Nenhum comando é enviado ao G29.
- O main.py apenas orquestra os módulos existentes.
- Cada módulo continua responsável pela sua própria lógica.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from config import (
    CONFIG,
    RuntimeMode,
    SAFETY,
    CAPTURE,
    YOLOP,
    VISUALIZATION,
    PERFORMANCE,
    HOTKEYS,
    DEBUG,
    ensure_directories,
    validate_config,
)

from capture.screen_capture import ScreenCapture
from vision.yolop_detector import create_default_detector


# Os módulos abaixo podem ter pequenas diferenças de API entre versões.
# O main utiliza adaptadores para evitar espalhar essa compatibilidade
# pelo restante do projeto.

from vision.lane_tracker import LaneTracker
from vision.lane_geometry import LaneGeometry

from vision.lane_model import (
    build_lane_model,
)

from core.lane_projection import (
    LaneProjectionEngine,
)

from vision.lane_assignment import (
    LaneAssignmentEngine,
)

from vision.adas_state import (
    ADASStateEstimator,
)


# ============================================================================
# LOGGING
# ============================================================================

def setup_logging() -> logging.Logger:
    """
    Inicializa logging do sistema.
    """

    ensure_directories()

    logger = logging.getLogger("forza_assistents")

    if logger.handlers:
        return logger

    logger.setLevel(
        getattr(
            logging,
            CONFIG.logging.level.upper(),
            logging.INFO,
        )
    )

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    if CONFIG.logging.log_to_console:
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)

    if CONFIG.logging.log_to_file:
        file_path = (
            CONFIG.logging.directory
            / CONFIG.logging.filename
        )

        file_handler = logging.FileHandler(
            file_path,
            encoding="utf-8",
        )

        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


LOGGER = setup_logging()


# ============================================================================
# RUNTIME STATISTICS
# ============================================================================

@dataclass
class RuntimeStatistics:
    """
    Métricas do pipeline.
    """

    frame_index: int = 0

    total_frames: int = 0

    valid_detections: int = 0

    invalid_detections: int = 0

    valid_geometry: int = 0

    valid_assignment: int = 0

    lane_lost_frames: int = 0

    capture_time_ms: float = 0.0

    detection_time_ms: float = 0.0

    tracking_time_ms: float = 0.0

    geometry_time_ms: float = 0.0

    model_time_ms: float = 0.0

    projection_time_ms: float = 0.0

    assignment_time_ms: float = 0.0

    adas_time_ms: float = 0.0

    total_time_ms: float = 0.0

    fps: float = 0.0

    last_timestamp: float = 0.0

    def update_fps(self) -> None:
        now = time.perf_counter()

        if self.last_timestamp > 0.0:
            dt = now - self.last_timestamp

            if dt > 0.0:
                instant_fps = 1.0 / dt

                if self.fps <= 0.0:
                    self.fps = instant_fps
                else:
                    self.fps = (
                        self.fps * 0.90
                        + instant_fps * 0.10
                    )

        self.last_timestamp = now


# ============================================================================
# PIPELINE RESULT
# ============================================================================

@dataclass
class PipelineResult:
    """
    Resultado completo de um frame.
    """

    frame: Optional[np.ndarray]

    detection: Any = None

    tracking: Any = None

    geometry: Any = None

    models: list[Any] | None = None

    projections: list[Any] | None = None

    assignment: Any = None

    adas: Any = None

    valid: bool = False

    timestamp: float = 0.0

    frame_index: int = 0

    statistics: Optional[RuntimeStatistics] = None

    def __post_init__(self) -> None:
        if self.models is None:
            self.models = []

        if self.projections is None:
            self.projections = []


# ============================================================================
# MAIN APPLICATION
# ============================================================================

class ForzaAssistents:
    """
    Aplicação principal do Forza Assistents.
    """

    def __init__(self) -> None:

        validate_config()

        self.mode = CONFIG.runtime_mode

        self.running = False

        self.frame_index = 0

        self.statistics = RuntimeStatistics()

        self.capture: Optional[ScreenCapture] = None

        self.detector: Any = None

        self.tracker: Any = None

        self.geometry: Any = None

        self.projection: Any = None

        self.assignment: Any = None

        self.adas: Any = None

        self.last_result: Optional[PipelineResult] = None

        self._initialized = False

        self._warmup_remaining = (
            PERFORMANCE.warmup_frames
        )

    # ========================================================================
    # INITIALIZATION
    # ========================================================================

    def initialize(self) -> None:
        """
        Inicializa todos os componentes.
        """

        if self._initialized:
            return

        LOGGER.info(
            "Initializing Forza Assistents..."
        )

        LOGGER.info(
            "Runtime mode: %s",
            self.mode.value,
        )

        if not SAFETY.enable_control:
            LOGGER.info(
                "Physical control: DISABLED"
            )

        # --------------------------------------------------------------------
        # Capture
        # --------------------------------------------------------------------

        LOGGER.info(
            "Initializing screen capture..."
        )

        self.capture = self._create_capture()

        # --------------------------------------------------------------------
        # YOLOP
        # --------------------------------------------------------------------

        LOGGER.info(
            "Initializing YOLOP detector..."
        )

        self.detector = create_default_detector()

        loaded = self.detector.load_model()

        if not loaded:
            error = getattr(
                self.detector,
                "last_error",
                None,
            )

            raise RuntimeError(
                f"YOLOP model could not be loaded: {error}"
            )

        LOGGER.info(
            "YOLOP loaded | device=%s",
            self.detector.get_device_name(),
        )

        # --------------------------------------------------------------------
        # Tracker
        # --------------------------------------------------------------------

        LOGGER.info(
            "Initializing lane tracker..."
        )

        self.tracker = self._create_tracker()

        # --------------------------------------------------------------------
        # Geometry
        # --------------------------------------------------------------------

        LOGGER.info(
            "Initializing lane geometry..."
        )

        self.geometry = self._create_geometry()

        # --------------------------------------------------------------------
        # Projection
        # --------------------------------------------------------------------

        LOGGER.info(
            "Initializing lane projection..."
        )

        self.projection = self._create_projection()

        # --------------------------------------------------------------------
        # Assignment
        # --------------------------------------------------------------------

        LOGGER.info(
            "Initializing lane assignment..."
        )

        self.assignment = self._create_assignment()

        # --------------------------------------------------------------------
        # ADAS
        # --------------------------------------------------------------------

        LOGGER.info(
            "Initializing ADAS state estimator..."
        )

        self.adas = self._create_adas()

        self._initialized = True

        LOGGER.info(
            "Forza Assistents initialized successfully."
        )

    # ========================================================================
    # COMPONENT FACTORIES
    # ========================================================================

    def _create_capture(self) -> ScreenCapture:
        """
        Cria o capturador usando a API disponível no projeto.
        """

        try:
            return ScreenCapture(
                monitor_index=CAPTURE.monitor_index,
                target_fps=CAPTURE.target_fps,
                roi=CAPTURE.roi,
            )
        except TypeError:

            try:
                return ScreenCapture(
                    monitor_index=CAPTURE.monitor_index,
                    target_fps=CAPTURE.target_fps,
                )
            except TypeError:
                return ScreenCapture()

    def _create_tracker(self) -> LaneTracker:
        """
        Cria o LaneTracker.
        """

        try:
            return LaneTracker(
                max_lost_frames=CONFIG.lane_tracker.max_lost_frames,
                min_stable_frames=CONFIG.lane_tracker.min_stable_frames,
                history_size=CONFIG.lane_tracker.history_size,
            )
        except TypeError:
            return LaneTracker()

    def _create_geometry(self) -> LaneGeometry:
        """
        Cria o LaneGeometry com os parâmetros da configuração atual.
        """

        try:
            return LaneGeometry(
                image_width=LANE_GEOMETRY.image_width,
                image_height=LANE_GEOMETRY.image_height,
                min_points=LANE_GEOMETRY.min_points,
            )
        except TypeError:

            try:
                return LaneGeometry(
                    image_width=LANE_GEOMETRY.image_width,
                    image_height=LANE_GEOMETRY.image_height,
                )
            except TypeError:
                return LaneGeometry()

    def _create_projection(self) -> LaneProjectionEngine:
        """
        Cria o mecanismo de projeção.
        """

        try:
            return LaneProjectionEngine(
                max_projection_distance=(
                    CONFIG.lane_projection
                    .max_projection_distance
                ),
                minimum_confidence=(
                    CONFIG.lane_projection
                    .minimum_confidence
                ),
            )
        except TypeError:
            return LaneProjectionEngine()

    def _create_assignment(self) -> LaneAssignmentEngine:
        """
        Cria o LaneAssignment.
        """

        try:
            return LaneAssignmentEngine(
                expected_lane_width=(
                    CONFIG.lane_assignment
                    .expected_lane_width
                ),
            )
        except TypeError:
            return LaneAssignmentEngine()

    def _create_adas(self) -> ADASStateEstimator:
        """
        Cria o estimador de estado ADAS.
        """

        try:
            return ADASStateEstimator(
                minimum_confidence=(
                    CONFIG.adas.minimum_confidence
                ),
            )
        except TypeError:
            return ADASStateEstimator()

    # ========================================================================
    # CAPTURE
    # ========================================================================

    def _capture_frame(self) -> Optional[np.ndarray]:
        """
        Captura um frame.

        Aceita as APIs mais comuns do ScreenCapture atual.
        """

        if self.capture is None:
            return None

        start = time.perf_counter()

        frame = None

        if hasattr(self.capture, "capture"):
            frame = self.capture.capture()

        elif hasattr(self.capture, "grab"):
            frame = self.capture.grab()

        elif hasattr(self.capture, "get_frame"):
            frame = self.capture.get_frame()

        elif hasattr(self.capture, "read"):
            result = self.capture.read()

            if isinstance(result, tuple):
                if len(result) >= 2:
                    ok, frame = result[:2]

                    if not ok:
                        frame = None
            else:
                frame = result

        self.statistics.capture_time_ms = (
            time.perf_counter() - start
        ) * 1000.0

        return frame

    # ========================================================================
    # ROI
    # ========================================================================

    def _apply_roi(
        self,
        frame: np.ndarray,
    ) -> np.ndarray:
        """
        Aplica o ROI configurado.

        O detector recebe somente a região relevante.
        """

        if not CAPTURE.use_roi:
            return frame

        if frame is None:
            return frame

        height, width = frame.shape[:2]

        x1, y1, x2, y2 = CAPTURE.roi

        x1 = max(0, min(x1, width))
        x2 = max(0, min(x2, width))

        y1 = max(0, min(y1, height))
        y2 = max(0, min(y2, height))

        if x2 <= x1 or y2 <= y1:
            return frame

        return frame[y1:y2, x1:x2]

    # ========================================================================
    # DETECTION
    # ========================================================================

    def _detect(
        self,
        frame: np.ndarray,
    ) -> Any:
        """
        Executa YOLOP.
        """

        start = time.perf_counter()

        result = None

        if hasattr(self.detector, "detect"):
            result = self.detector.detect(frame)

        elif hasattr(self.detector, "infer"):
            result = self.detector.infer(frame)

        elif hasattr(self.detector, "process"):
            result = self.detector.process(frame)

        else:
            raise RuntimeError(
                "YOLOP detector does not expose a supported "
                "detection method."
            )

        self.statistics.detection_time_ms = (
            time.perf_counter() - start
        ) * 1000.0

        return result

    # ========================================================================
    # TRACKING
    # ========================================================================

    def _track(
        self,
        detection: Any,
    ) -> Any:
        """
        Atualiza o tracker.
        """

        start = time.perf_counter()

        result = None

        if hasattr(self.tracker, "update"):
            result = self.tracker.update(detection)

        elif hasattr(self.tracker, "track"):
            result = self.tracker.track(detection)

        else:
            raise RuntimeError(
                "LaneTracker does not expose update()/track()."
            )

        self.statistics.tracking_time_ms = (
            time.perf_counter() - start
        ) * 1000.0

        return result

    # ========================================================================
    # GEOMETRY
    # ========================================================================

    def _calculate_geometry(
        self,
        tracking: Any,
    ) -> Any:
        """
        Calcula geometria da faixa.
        """

        start = time.perf_counter()

        result = None

        if hasattr(self.geometry, "calculate"):
            result = self.geometry.calculate(tracking)

        elif hasattr(self.geometry, "estimate"):
            result = self.geometry.estimate(tracking)

        elif hasattr(self.geometry, "compute"):
            result = self.geometry.compute(tracking)

        elif hasattr(self.geometry, "update"):
            result = self.geometry.update(tracking)

        else:
            raise RuntimeError(
                "LaneGeometry does not expose a supported "
                "geometry method."
            )

        self.statistics.geometry_time_ms = (
            time.perf_counter() - start
        ) * 1000.0

        return result

    # ========================================================================
    # MODELS
    # ========================================================================

    def _extract_tracks(
        self,
        tracking: Any,
    ) -> list[Any]:
        """
        Extrai lanes/tracks do resultado do tracker.
        """

        if tracking is None:
            return []

        for name in (
            "tracks",
            "lanes",
            "active_tracks",
            "tracked_lanes",
        ):
            value = getattr(
                tracking,
                name,
                None,
            )

            if value is not None:
                return list(value)

        if isinstance(tracking, (list, tuple)):
            return list(tracking)

        return []

    def _build_models(
        self,
        tracking: Any,
    ) -> list[Any]:
        """
        Constrói modelos polinomiais para todas as lanes válidas.
        """

        start = time.perf_counter()

        models: list[Any] = []

        tracks = self._extract_tracks(tracking)

        for track in tracks:

            try:
                model = build_lane_model(track)

            except TypeError:

                try:
                    model = build_lane_model(
                        track,
                        degree=(
                            CONFIG.lane_model
                            .polynomial_degree
                        ),
                    )
                except Exception:
                    model = None

            except Exception as exc:

                LOGGER.debug(
                    "Lane model failed: %s",
                    exc,
                )

                model = None

            if model is not None:
                models.append(model)

        self.statistics.model_time_ms = (
            time.perf_counter() - start
        ) * 1000.0

        return models

    # ========================================================================
    # PROJECTION
    # ========================================================================

    def _project_models(
        self,
        models: list[Any],
    ) -> list[Any]:
        """
        Projeta todas as lanes modeladas.
        """

        start = time.perf_counter()

        projections: list[Any] = []

        for model in models:

            try:

                if hasattr(
                    self.projection,
                    "project",
                ):
                    result = self.projection.project(
                        model
                    )

                elif hasattr(
                    self.projection,
                    "project_lane",
                ):
                    result = self.projection.project_lane(
                        model
                    )

                else:
                    result = None

                if result is not None:
                    projections.append(result)

            except Exception as exc:

                LOGGER.debug(
                    "Lane projection failed: %s",
                    exc,
                )

        self.statistics.projection_time_ms = (
            time.perf_counter() - start
        ) * 1000.0

        return projections

    # ========================================================================
    # ASSIGNMENT
    # ========================================================================

    def _assign_lanes(
        self,
        projections: list[Any],
        geometry: Any,
    ) -> Any:
        """
        Determina faixa atual e lanes vizinhas.
        """

        start = time.perf_counter()

        result = None

        try:

            if hasattr(
                self.assignment,
                "assign",
            ):
                result = self.assignment.assign(
                    projections,
                    geometry,
                )

            elif hasattr(
                self.assignment,
                "update",
            ):
                result = self.assignment.update(
                    projections,
                    geometry,
                )

            elif hasattr(
                self.assignment,
                "process",
            ):
                result = self.assignment.process(
                    projections,
                    geometry,
                )

        except TypeError:

            try:
                result = self.assignment.assign(
                    projections
                )
            except Exception as exc:
                LOGGER.debug(
                    "Lane assignment failed: %s",
                    exc,
                )

        except Exception as exc:

            LOGGER.debug(
                "Lane assignment failed: %s",
                exc,
            )

        self.statistics.assignment_time_ms = (
            time.perf_counter() - start
        ) * 1000.0

        return result

    # ========================================================================
    # ADAS
    # ========================================================================

    def _calculate_adas(
        self,
        geometry: Any,
        assignment: Any,
    ) -> Any:
        """
        Calcula o estado ADAS.
        """

        start = time.perf_counter()

        result = None

        try:

            if hasattr(
                self.adas,
                "update",
            ):
                result = self.adas.update(
                    geometry,
                    assignment,
                )

            elif hasattr(
                self.adas,
                "estimate",
            ):
                result = self.adas.estimate(
                    geometry,
                    assignment,
                )

            elif hasattr(
                self.adas,
                "process",
            ):
                result = self.adas.process(
                    geometry,
                    assignment,
                )

        except TypeError:

            try:
                result = self.adas.update(
                    geometry
                )
            except Exception as exc:
                LOGGER.debug(
                    "ADAS estimation failed: %s",
                    exc,
                )

        except Exception as exc:

            LOGGER.debug(
                "ADAS estimation failed: %s",
                exc,
            )

        self.statistics.adas_time_ms = (
            time.perf_counter() - start
        ) * 1000.0

        return result

    # ========================================================================
    # VALIDITY
    # ========================================================================

    @staticmethod
    def _is_valid(
        obj: Any,
    ) -> bool:

        if obj is None:
            return False

        value = getattr(
            obj,
            "valid",
            None,
        )

        if value is None:
            return True

        return bool(value)

    # ========================================================================
    # PIPELINE
    # ========================================================================

    def process_frame(
        self,
        frame: np.ndarray,
    ) -> PipelineResult:
        """
        Executa o pipeline completo em um frame.
        """

        timestamp = time.perf_counter()

        self.frame_index += 1

        self.statistics.frame_index = (
            self.frame_index
        )

        self.statistics.total_frames += 1

        self.statistics.update_fps()

        # ------------------------------------------------------------
        # ROI
        # ------------------------------------------------------------

        roi = self._apply_roi(frame)

        # ------------------------------------------------------------
        # YOLOP
        # ------------------------------------------------------------

        detection = self._detect(roi)

        detection_valid = self._is_valid(
            detection
        )

        if detection_valid:
            self.statistics.valid_detections += 1
        else:
            self.statistics.invalid_detections += 1

        # ------------------------------------------------------------
        # TRACKER
        # ------------------------------------------------------------

        tracking = self._track(
            detection
        )

        # ------------------------------------------------------------
        # GEOMETRY
        # ------------------------------------------------------------

        geometry = self._calculate_geometry(
            tracking
        )

        geometry_valid = self._is_valid(
            geometry
        )

        if geometry_valid:
            self.statistics.valid_geometry += 1

        # ------------------------------------------------------------
        # MODELS
        # ------------------------------------------------------------

        models = self._build_models(
            tracking
        )

        # ------------------------------------------------------------
        # PROJECTION
        # ------------------------------------------------------------

        projections = self._project_models(
            models
        )

        # ------------------------------------------------------------
        # ASSIGNMENT
        # ------------------------------------------------------------

        assignment = self._assign_lanes(
            projections,
            geometry,
        )

        assignment_valid = self._is_valid(
            assignment
        )

        if assignment_valid:
            self.statistics.valid_assignment += 1

        # ------------------------------------------------------------
        # ADAS
        # ------------------------------------------------------------

        adas = self._calculate_adas(
            geometry,
            assignment,
        )

        adas_valid = self._is_valid(
            adas
        )

        if not adas_valid:
            self.statistics.lane_lost_frames += 1

        # ------------------------------------------------------------
        # FINAL VALIDITY
        # ------------------------------------------------------------

        valid = (
            detection_valid
            and geometry_valid
            and assignment_valid
            and adas_valid
        )

        self.statistics.total_time_ms = (
            time.perf_counter()
            - timestamp
        ) * 1000.0

        return PipelineResult(
            frame=frame,
            detection=detection,
            tracking=tracking,
            geometry=geometry,
            models=models,
            projections=projections,
            assignment=assignment,
            adas=adas,
            valid=valid,
            timestamp=timestamp,
            frame_index=self.frame_index,
            statistics=self.statistics,
        )

    # ========================================================================
    # VISUALIZATION
    # ========================================================================

    def _draw_overlay(
        self,
        result: PipelineResult,
    ) -> np.ndarray:
        """
        Desenha informações de debug sobre o frame.
        """

        frame = result.frame

        if frame is None:
            return np.zeros(
                (720, 1280, 3),
                dtype=np.uint8,
            )

        output = frame.copy()

        if not VISUALIZATION.enabled:
            return output

        # ------------------------------------------------------------
        # ROI
        # ------------------------------------------------------------

        if VISUALIZATION.show_roi:

            x1, y1, x2, y2 = CAPTURE.roi

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                2,
            )

        # ------------------------------------------------------------
        # FPS / timing
        # ------------------------------------------------------------

        if VISUALIZATION.show_fps:

            text = (
                f"FPS: "
                f"{self.statistics.fps:.1f}"
            )

            cv2.putText(
                output,
                text,
                (30, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        if VISUALIZATION.show_timing:

            text = (
                f"Pipeline: "
                f"{self.statistics.total_time_ms:.1f} ms"
            )

            cv2.putText(
                output,
                text,
                (30, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # ------------------------------------------------------------
        # Runtime mode
        # ------------------------------------------------------------

        cv2.putText(
            output,
            f"MODE: {self.mode.value.upper()}",
            (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # ------------------------------------------------------------
        # ADAS state
        # ------------------------------------------------------------

        if (
            VISUALIZATION.show_adas_state
            and result.adas is not None
        ):

            state = getattr(
                result.adas,
                "state",
                None,
            )

            confidence = getattr(
                result.adas,
                "confidence",
                0.0,
            )

            text = (
                f"ADAS: "
                f"{state} "
                f"conf={float(confidence):.2f}"
            )

            cv2.putText(
                output,
                text,
                (30, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # ------------------------------------------------------------
        # Geometry
        # ------------------------------------------------------------

        if (
            VISUALIZATION.show_geometry
            and result.geometry is not None
        ):

            lateral = getattr(
                result.geometry,
                "lateral_error",
                0.0,
            )

            heading = getattr(
                result.geometry,
                "heading_error",
                0.0,
            )

            curvature = getattr(
                result.geometry,
                "curvature",
                0.0,
            )

            text = (
                f"lat={float(lateral):+.3f} "
                f"head={float(heading):+.3f} "
                f"curve={float(curvature):+.3f}"
            )

            cv2.putText(
                output,
                text,
                (30, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # ------------------------------------------------------------
        # Lane count
        # ------------------------------------------------------------

        if VISUALIZATION.show_lane_ids:

            count = len(
                result.models or []
            )

            cv2.putText(
                output,
                f"LANES: {count}",
                (30, 205),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return output

    # ========================================================================
    # HOTKEYS
    # ========================================================================

    def _handle_key(
        self,
        key: int,
    ) -> bool:
        """
        Processa teclas de operação.

        Retorna False quando o programa deve encerrar.
        """

        if key < 0:
            return True

        # ESC
        if key == 27:
            return False

        # F8
        if key == 0x77:

            if self.mode == RuntimeMode.MONITOR:
                LOGGER.info(
                    "F8 pressed: monitor remains enabled."
                )

            elif self.mode == RuntimeMode.DISABLED:
                self.mode = RuntimeMode.MONITOR

            else:
                self.mode = RuntimeMode.MONITOR

            return True

        # F9
        if key == 0x78:

            LOGGER.warning(
                "EMERGENCY STOP requested."
            )

            self.mode = RuntimeMode.DISABLED

            return True

        # F10
        if key == 0x79:

            # A configuração é imutável; utilizamos
            # um atributo interno para o estado visual.
            VISUALIZATION.enabled = (
                not VISUALIZATION.enabled
            )

            return True

        return True

    # ========================================================================
    # RUN
    # ========================================================================

    def run(self) -> None:
        """
        Loop principal.
        """

        self.initialize()

        self.running = True

        LOGGER.info(
            "Starting runtime loop."
        )

        LOGGER.info(
            "Press ESC to exit."
        )

        LOGGER.info(
            "Press F9 for emergency stop."
        )

        try:

            while self.running:

                loop_start = time.perf_counter()

                frame = self._capture_frame()

                if frame is None:

                    LOGGER.warning(
                        "Frame capture failed."
                    )

                    time.sleep(0.005)

                    continue

                try:

                    result = self.process_frame(
                        frame
                    )

                    self.last_result = result

                except Exception as exc:

                    LOGGER.exception(
                        "Pipeline failure: %s",
                        exc,
                    )

                    # Falha de pipeline nunca deve
                    # habilitar controle físico.
                    self.mode = (
                        RuntimeMode.DISABLED
                    )

                    continue

                # ----------------------------------------------------
                # Debug
                # ----------------------------------------------------

                if DEBUG.print_pipeline_summary:

                    if (
                        self.frame_index == 1
                        or self.frame_index % 60 == 0
                    ):

                        self._print_summary(
                            result
                        )

                # ----------------------------------------------------
                # Overlay
                # ----------------------------------------------------

                display = self._draw_overlay(
                    result
                )

                if VISUALIZATION.enabled:

                    cv2.imshow(
                        VISUALIZATION.window_name,
                        display,
                    )

                # ----------------------------------------------------
                # Keyboard
                # ----------------------------------------------------

                key = cv2.waitKey(
                    VISUALIZATION.wait_key_ms
                )

                if not self._handle_key(key):

                    self.running = False

                # ----------------------------------------------------
                # Frame pacing
                # ----------------------------------------------------

                elapsed = (
                    time.perf_counter()
                    - loop_start
                )

                target_dt = (
                    1.0
                    / max(
                        1,
                        PERFORMANCE.target_pipeline_fps,
                    )
                )

                remaining = (
                    target_dt
                    - elapsed
                )

                if remaining > 0:
                    time.sleep(
                        min(
                            remaining,
                            0.01,
                        )
                    )

        except KeyboardInterrupt:

            LOGGER.info(
                "Keyboard interrupt."
            )

        finally:

            self.shutdown()

    # ========================================================================
    # SUMMARY
    # ========================================================================

    def _print_summary(
        self,
        result: PipelineResult,
    ) -> None:

        print()
        print("=" * 72)
        print("FORZA ASSISTENTS")
        print("=" * 72)

        print(
            f"frame={result.frame_index}"
        )

        print(
            f"fps={self.statistics.fps:.1f}"
        )

        print(
            f"pipeline="
            f"{self.statistics.total_time_ms:.2f} ms"
        )

        detection = result.detection

        detected_count = getattr(
            detection,
            "num_lanes_detected",
            None,
        )

        if detected_count is None:

            lanes = getattr(
                detection,
                "lanes",
                [],
            )

            detected_count = (
                len(lanes)
                if lanes is not None
                else 0
            )

        print(
            f"detected_lanes={detected_count}"
        )

        print(
            f"models={len(result.models or [])}"
        )

        print(
            f"projections="
            f"{len(result.projections or [])}"
        )

        print(
            f"geometry_valid="
            f"{self._is_valid(result.geometry)}"
        )

        print(
            f"assignment_valid="
            f"{self._is_valid(result.assignment)}"
        )

        adas_state = getattr(
            result.adas,
            "state",
            None,
        )

        print(
            f"adas={adas_state}"
        )

        print(
            f"pipeline_valid={result.valid}"
        )

        print("=" * 72)

    # ========================================================================
    # SHUTDOWN
    # ========================================================================

    def shutdown(self) -> None:
        """
        Encerramento seguro.
        """

        if not self.running and not self._initialized:
            return

        LOGGER.info(
            "Shutting down Forza Assistents..."
        )

        self.running = False

        # ------------------------------------------------------------
        # G29 / controle físico
        # ------------------------------------------------------------

        # Nunca enviamos comando de steering aqui.
        #
        # Quando o controle for implementado, o primeiro passo
        # deverá ser sempre colocar o volante em estado seguro.

        # ------------------------------------------------------------
        # Capture
        # ------------------------------------------------------------

        if self.capture is not None:

            for method_name in (
                "stop",
                "release",
                "close",
            ):

                method = getattr(
                    self.capture,
                    method_name,
                    None,
                )

                if callable(method):

                    try:
                        method()
                    except Exception:
                        LOGGER.debug(
                            "Capture cleanup failed.",
                            exc_info=True,
                        )

                    break

        # ------------------------------------------------------------
        # Detector
        # ------------------------------------------------------------

        if self.detector is not None:

            for method_name in (
                "close",
                "release",
                "shutdown",
            ):

                method = getattr(
                    self.detector,
                    method_name,
                    None,
                )

                if callable(method):

                    try:
                        method()
                    except Exception:
                        LOGGER.debug(
                            "Detector cleanup failed.",
                            exc_info=True,
                        )

                    break

        cv2.destroyAllWindows()

        LOGGER.info(
            "Forza Assistents stopped."
        )


# ============================================================================
# ENTRY POINT
# ============================================================================

def main() -> None:
    """
    Entry point.
    """

    application = ForzaAssistents()

    application.run()


if __name__ == "__main__":
    main()