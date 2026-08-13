"""
main.py

Forza Assistents
================

Orquestrador principal do pipeline de percepção.

Pipeline:

    ScreenCapture
        ↓
    YOLOP
        ↓
    LaneTracker
        ↓
    LaneGeometry
        ↓
    LaneModel
        ↓
    LaneProjection
        ↓
    LaneAssignment
        ↓
    ADAS State
        ↓
    Monitor / Debug

Princípios:

- nenhuma resolução fixa é assumida pelo pipeline;
- o tamanho real do frame define a geometria;
- ROI é aplicado exclusivamente pelo ScreenCapture;
- LaneGeometry trabalha no sistema de coordenadas do frame recebido;
- todas as lanes detectadas pelo YOLOP são preservadas;
- MONITOR é o modo padrão;
- nenhum comando físico é enviado ao G29;
- componentes permanecem desacoplados;
- falhas de percepção não derrubam o processo;
- preparado para execução em tempo real.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np

from config import (
    CONFIG,
    CAPTURE,
    DEBUG,
    HOTKEYS,
    PERFORMANCE,
    SAFETY,
    VISUALIZATION,
    RuntimeMode,
    ensure_directories,
    validate_config,
)

from capture.screen_capture import ScreenCapture
from vision.yolop_detector import create_default_detector
from vision.lane_tracker import LaneTracker
from vision.lane_geometry import LaneGeometry
from vision.lane_model import build_lane_model

from vision.lane_projection import LaneProjectionEngine
from vision.lane_assignment import LaneAssignmentEngine
from vision.adas_state import ADASStateEstimator


# ============================================================================
# LOGGING
# ============================================================================

def setup_logging() -> logging.Logger:
    ensure_directories()

    logger = logging.getLogger("forza_assistents")

    if logger.handlers:
        return logger

    level = getattr(
        logging,
        CONFIG.logging.level.upper(),
        logging.INFO,
    )

    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    if CONFIG.logging.log_to_console:
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)

    if CONFIG.logging.log_to_file:
        path = (
            CONFIG.logging.directory
            / CONFIG.logging.filename
        )

        file_handler = logging.FileHandler(
            path,
            encoding="utf-8",
        )

        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


LOGGER = setup_logging()


# ============================================================================
# ESTATÍSTICAS
# ============================================================================

@dataclass
class RuntimeStatistics:

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

    _last_timestamp: float = 0.0

    def update_fps(self) -> None:

        now = time.perf_counter()

        if self._last_timestamp > 0.0:

            dt = now - self._last_timestamp

            if dt > 0.0:

                instant = 1.0 / dt

                if self.fps <= 0.0:
                    self.fps = instant

                else:
                    self.fps = (
                        self.fps * 0.90
                        + instant * 0.10
                    )

        self._last_timestamp = now


# ============================================================================
# RESULTADO
# ============================================================================

@dataclass
class PipelineResult:

    frame: Optional[np.ndarray]

    detection: Any = None

    tracking: Any = None

    geometry: Any = None

    models: list[Any] | None = None

    projections: list[Any] | None = None

    assignment: Any = None

    adas: Any = None

    valid: bool = False

    frame_index: int = 0

    timestamp: float = 0.0

    statistics: Optional[RuntimeStatistics] = None

    def __post_init__(self) -> None:

        if self.models is None:
            self.models = []

        if self.projections is None:
            self.projections = []


# ============================================================================
# APLICAÇÃO
# ============================================================================

class ForzaAssistents:

    def __init__(self) -> None:

        validate_config()

        self.mode = CONFIG.runtime_mode

        self.running = False

        self.frame_index = 0

        self.statistics = RuntimeStatistics()

        self.capture: Optional[ScreenCapture] = None

        self.detector: Any = None

        self.tracker: Optional[LaneTracker] = None

        self.geometry: Optional[LaneGeometry] = None

        self.projection: Any = None

        self.assignment: Any = None

        self.adas: Any = None

        self.last_result: Optional[PipelineResult] = None

        self._initialized = False

        self._geometry_shape: Optional[tuple[int, int]] = None

        self._display_enabled = VISUALIZATION.enabled


    # ========================================================================
    # INITIALIZAÇÃO
    # ========================================================================

    def initialize(self) -> None:

        if self._initialized:
            return

        LOGGER.info(
            "Initializing Forza Assistents..."
        )

        LOGGER.info(
            "Runtime mode: %s",
            self.mode.value,
        )

        LOGGER.info(
            "Physical control: %s",
            "ENABLED"
            if SAFETY.enable_control
            else "DISABLED",
        )

        # --------------------------------------------------------------------
        # CAPTURE
        # --------------------------------------------------------------------

        self.capture = ScreenCapture(
            region=CAPTURE.roi if CAPTURE.use_roi else None,
            target_fps=CAPTURE.target_fps,
            backend=CAPTURE.backend,
            output_color=CAPTURE.output_color_format,
            max_buffer_size=PERFORMANCE.max_frame_queue,
        )

        if not self.capture.initialize():

            raise RuntimeError(
                "Não foi possível inicializar "
                "o sistema de captura."
            )

        self.capture.start()

        LOGGER.info(
            "Screen capture initialized."
        )

        # --------------------------------------------------------------------
        # YOLOP
        # --------------------------------------------------------------------

        self.detector = create_default_detector()

        if not self.detector.load_model():

            error = getattr(
                self.detector,
                "last_error",
                None,
            )

            raise RuntimeError(
                "YOLOP não pôde ser carregado: "
                f"{error}"
            )

        LOGGER.info(
            "YOLOP ready | device=%s | providers=%s",
            self.detector.get_device_name(),
            (
                self.detector.session.get_providers()
                if self.detector.session is not None
                else None
            ),
        )

        # --------------------------------------------------------------------
        # TRACKER
        # --------------------------------------------------------------------

        self.tracker = LaneTracker(
            max_lanes=CONFIG.lane_tracker.max_lanes,
            history_size=CONFIG.lane_tracker.history_size,
            min_points=CONFIG.lane_tracker.min_points,
            match_distance=CONFIG.lane_tracker.match_distance,
            max_missed_frames=(
                CONFIG.lane_tracker.max_lost_frames
            ),
            min_stable_frames=(
                CONFIG.lane_tracker.min_stable_frames
            ),
        )

        # --------------------------------------------------------------------
        # PROJECTION
        # --------------------------------------------------------------------

        try:

            self.projection = LaneProjectionEngine(
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

            self.projection = LaneProjectionEngine()

        # --------------------------------------------------------------------
        # ASSIGNMENT
        # --------------------------------------------------------------------

        try:

            self.assignment = LaneAssignmentEngine(
                expected_lane_width=(
                    CONFIG.lane_assignment
                    .expected_lane_width
                ),
            )

        except TypeError:

            self.assignment = LaneAssignmentEngine()

        # --------------------------------------------------------------------
        # ADAS
        # --------------------------------------------------------------------

        try:

            self.adas = ADASStateEstimator(
                minimum_confidence=(
                    CONFIG.adas.minimum_confidence
                ),
            )

        except TypeError:

            self.adas = ADASStateEstimator()

        # --------------------------------------------------------------------
        # GEOMETRY
        #
        # NÃO criamos aqui.
        #
        # O motivo é fundamental:
        #
        # ScreenCapture pode entregar:
        #
        #     1920x1080
        #     2560x1600
        #     ROI 1900x900
        #     etc.
        #
        # LaneGeometry precisa trabalhar no sistema de coordenadas
        # REAL do frame recebido.
        # --------------------------------------------------------------------

        self.geometry = None
        self._geometry_shape = None

        self._initialized = True

        LOGGER.info(
            "Forza Assistents initialized."
        )


    # ========================================================================
    # GEOMETRY DINÂMICA
    # ========================================================================

    def _ensure_geometry(
        self,
        frame: np.ndarray,
    ) -> LaneGeometry:

        if frame is None:

            raise ValueError(
                "Frame inválido."
            )

        height, width = frame.shape[:2]

        shape = (
            int(width),
            int(height),
        )

        # ------------------------------------------------------------
        # Reutiliza o objeto se a resolução não mudou.
        # ------------------------------------------------------------

        if (
            self.geometry is not None
            and self._geometry_shape == shape
        ):

            return self.geometry

        LOGGER.info(
            "Creating LaneGeometry for frame %dx%d",
            width,
            height,
        )

        # ------------------------------------------------------------
        # IMPORTANTE:
        #
        # O frame recebido pelo main já é o ROI quando
        # CAPTURE.use_roi=True.
        #
        # Portanto o sistema de coordenadas da geometria começa
        # em (0, 0).
        # ------------------------------------------------------------

        self.geometry = LaneGeometry(
            screen_width=width,
            screen_height=height,
            roi=(
                0,
                0,
                width,
                height,
            ),
            detector_width= self.detector.input_width,
            detector_height= self.detector.input_height,
            min_points=CONFIG.lane_geometry.min_points,
        )

        self._geometry_shape = shape

        LOGGER.info(
            "LaneGeometry ready | frame=%dx%d | roi=0,0,%d,%d",
            width,
            height,
            width,
            height,
        )

        return self.geometry


    # ========================================================================
    # CAPTURE
    # ========================================================================

    def _get_frame(
        self,
    ) -> Optional[np.ndarray]:

        if self.capture is None:
            return None

        start = time.perf_counter()

        frame = self.capture.get_latest_frame()

        self.statistics.capture_time_ms = (
            time.perf_counter() - start
        ) * 1000.0

        if frame is None:
            return None

        if not isinstance(frame, np.ndarray):
            return None

        if frame.ndim != 3:
            return None

        if frame.shape[2] < 3:
            return None

        return frame


    # ========================================================================
    # DETECTOR
    # ========================================================================

    def _detect(
        self,
        frame: np.ndarray,
    ) -> Any:

        if self.detector is None:
            raise RuntimeError(
                "Detector não inicializado."
            )

        start = time.perf_counter()

        result = self.detector.detect(
            frame
        )

        self.statistics.detection_time_ms = (
            time.perf_counter() - start
        ) * 1000.0

        return result


    # ========================================================================
    # TRACKER
    # ========================================================================

    def _track(
        self,
        detection: Any,
    ) -> Any:

        if self.tracker is None:
            raise RuntimeError(
                "Tracker não inicializado."
            )

        start = time.perf_counter()

        result = self.tracker.update(
            detection,
            timestamp=time.monotonic(),
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
        detection: Any,
        frame: np.ndarray,
    ) -> Any:

        geometry = self._ensure_geometry(
            frame
        )

        start = time.perf_counter()

        result = geometry.compute(
            detection
        )

        self.statistics.geometry_time_ms = (
            time.perf_counter() - start
        ) * 1000.0

        return result


    # ========================================================================
    # MODELS
    # ========================================================================

    @staticmethod
    def _extract_tracks(
        tracking: Any,
    ) -> list[Any]:

        if tracking is None:
            return []

        tracks = getattr(
            tracking,
            "lanes",
            None,
        )

        if tracks is not None:
            return list(tracks)

        tracks = getattr(
            tracking,
            "tracks",
            None,
        )

        if tracks is not None:
            return list(tracks)

        if isinstance(
            tracking,
            (list, tuple),
        ):
            return list(tracking)

        return []


    def _build_models(
        self,
        tracking: Any,
    ) -> list[Any]:

        start = time.perf_counter()

        models: list[Any] = []

        for track in self._extract_tracks(
            tracking
        ):

            try:

                model = build_lane_model(
                    track
                )

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

        start = time.perf_counter()

        projections: list[Any] = []

        for model in models:

            try:

                if hasattr(
                    self.projection,
                    "project",
                ):

                    projection = (
                        self.projection.project(
                            model
                        )
                    )

                elif hasattr(
                    self.projection,
                    "project_lane",
                ):

                    projection = (
                        self.projection.project_lane(
                            model
                        )
                    )

                else:

                    projection = None

                if projection is not None:
                    projections.append(
                        projection
                    )

            except Exception as exc:

                LOGGER.debug(
                    "Projection failed: %s",
                    exc,
                )

        self.statistics.projection_time_ms = (
            time.perf_counter() - start
        ) * 1000.0

        return projections


    # ========================================================================
    # ASSIGNMENT
    # ========================================================================

    def _assign(
        self,
        projections: list[Any],
        geometry: Any,
    ) -> Any:

        if self.assignment is None:
            return None

        start = time.perf_counter()

        result = None

        try:

            if hasattr(
                self.assignment,
                "assign",
            ):

                try:

                    result = (
                        self.assignment.assign(
                            projections,
                            geometry,
                        )
                    )

                except TypeError:

                    result = (
                        self.assignment.assign(
                            projections
                        )
                    )

            elif hasattr(
                self.assignment,
                "update",
            ):

                result = (
                    self.assignment.update(
                        projections,
                        geometry,
                    )
                )

            elif hasattr(
                self.assignment,
                "process",
            ):

                result = (
                    self.assignment.process(
                        projections,
                        geometry,
                    )
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

    def _adas_update(
        self,
        geometry: Any,
        assignment: Any,
    ) -> Any:

        if self.adas is None:
            return None

        start = time.perf_counter()

        result = None

        try:

            if hasattr(
                self.adas,
                "update",
            ):

                try:

                    result = self.adas.update(
                        geometry,
                        assignment,
                    )

                except TypeError:

                    result = self.adas.update(
                        geometry
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

        except Exception as exc:

            LOGGER.debug(
                "ADAS update failed: %s",
                exc,
            )

        self.statistics.adas_time_ms = (
            time.perf_counter() - start
        ) * 1000.0

        return result


    # ========================================================================
    # VALIDADE
    # ========================================================================

    @staticmethod
    def _valid(
        value: Any,
    ) -> bool:

        if value is None:
            return False

        valid = getattr(
            value,
            "valid",
            None,
        )

        if valid is None:
            return True

        return bool(valid)


    # ========================================================================
    # PIPELINE
    # ========================================================================

    def process_frame(
        self,
        frame: np.ndarray,
    ) -> PipelineResult:

        start = time.perf_counter()

        self.frame_index += 1

        self.statistics.frame_index = (
            self.frame_index
        )

        self.statistics.total_frames += 1

        self.statistics.update_fps()

        # --------------------------------------------------------------------
        # GEOMETRY
        #
        # Primeiro garante que o sistema conhece a resolução REAL.
        # --------------------------------------------------------------------

        self._ensure_geometry(
            frame
        )

        # --------------------------------------------------------------------
        # YOLOP
        # --------------------------------------------------------------------

        detection = self._detect(
            frame
        )

        detection_valid = self._valid(
            detection
        )

        if detection_valid:
            self.statistics.valid_detections += 1
        else:
            self.statistics.invalid_detections += 1

        # --------------------------------------------------------------------
        # TRACKER
        # --------------------------------------------------------------------

        tracking = self._track(
            detection
        )

        # --------------------------------------------------------------------
        # GEOMETRIA
        #
        # Usa a detecção original do YOLOP.
        #
        # Isso é intencional:
        # LaneGeometry trabalha com lanes observadas no frame atual.
        # Tracker mantém identidade temporal separadamente.
        # --------------------------------------------------------------------

        geometry = self._calculate_geometry(
            detection,
            frame,
        )

        geometry_valid = self._valid(
            geometry
        )

        if geometry_valid:
            self.statistics.valid_geometry += 1

        # --------------------------------------------------------------------
        # MODELS
        # --------------------------------------------------------------------

        models = self._build_models(
            tracking
        )

        # --------------------------------------------------------------------
        # PROJECTION
        # --------------------------------------------------------------------

        projections = self._project_models(
            models
        )

        # --------------------------------------------------------------------
        # ASSIGNMENT
        # --------------------------------------------------------------------

        assignment = self._assign(
            projections,
            geometry,
        )

        assignment_valid = self._valid(
            assignment
        )

        if assignment_valid:
            self.statistics.valid_assignment += 1

        # --------------------------------------------------------------------
        # ADAS
        # --------------------------------------------------------------------

        adas = self._adas_update(
            geometry,
            assignment,
        )

        adas_valid = self._valid(
            adas
        )

        if not adas_valid:
            self.statistics.lane_lost_frames += 1

        # --------------------------------------------------------------------
        # VALIDIDADE FINAL
        # --------------------------------------------------------------------

        valid = bool(
            detection_valid
            and geometry_valid
            and assignment_valid
            and adas_valid
        )

        self.statistics.total_time_ms = (
            time.perf_counter() - start
        ) * 1000.0

        result = PipelineResult(
            frame=frame,
            detection=detection,
            tracking=tracking,
            geometry=geometry,
            models=models,
            projections=projections,
            assignment=assignment,
            adas=adas,
            valid=valid,
            frame_index=self.frame_index,
            timestamp=time.monotonic(),
            statistics=self.statistics,
        )

        self.last_result = result

        return result


    # ========================================================================
    # OVERLAY
    # ========================================================================

    def _draw_overlay(
        self,
        result: PipelineResult,
    ) -> np.ndarray:

        frame = result.frame

        if frame is None:

            return np.zeros(
                (720, 1280, 3),
                dtype=np.uint8,
            )

        output = frame.copy()

        if not self._display_enabled:
            return output

        # --------------------------------------------------------------------
        # FPS
        # --------------------------------------------------------------------

        cv2.putText(
            output,
            f"FPS: {self.statistics.fps:.1f}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # --------------------------------------------------------------------
        # PIPELINE LATENCY
        # --------------------------------------------------------------------

        cv2.putText(
            output,
            f"Pipeline: "
            f"{self.statistics.total_time_ms:.1f} ms",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # --------------------------------------------------------------------
        # DEVICE
        # --------------------------------------------------------------------

        device = "UNKNOWN"

        if self.detector is not None:

            try:
                device = (
                    self.detector.get_device_name()
                )

            except Exception:
                pass

        cv2.putText(
            output,
            f"YOLOP: {device}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # --------------------------------------------------------------------
        # MODE
        # --------------------------------------------------------------------

        cv2.putText(
            output,
            f"MODE: {self.mode.value.upper()}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # --------------------------------------------------------------------
        # DETECTION
        # --------------------------------------------------------------------

        detection = result.detection

        lane_count = 0

        if detection is not None:

            lanes = getattr(
                detection,
                "lanes",
                None,
            )

            if lanes is not None:
                lane_count = len(lanes)

        cv2.putText(
            output,
            f"YOLOP lanes: {lane_count}",
            (20, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # --------------------------------------------------------------------
        # GEOMETRY
        # --------------------------------------------------------------------

        geometry = result.geometry

        if geometry is not None:

            cv2.putText(
                output,
                (
                    f"Geometry: "
                    f"{'VALID' if geometry.valid else 'INVALID'}"
                ),
                (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                output,
                (
                    f"Lat: "
                    f"{geometry.lateral_error:+.3f} | "
                    f"Head: "
                    f"{geometry.heading_error:+.3f}"
                ),
                (20, 210),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                output,
                (
                    f"Width: "
                    f"{geometry.lane_width:.1f} | "
                    f"Conf: "
                    f"{geometry.geometry_confidence:.2f}"
                ),
                (20, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # --------------------------------------------------------------------
        # ASSIGNMENT
        # --------------------------------------------------------------------

        assignment = result.assignment

        if assignment is not None:

            lane_id = getattr(
                assignment,
                "current_lane_id",
                None,
            )

            offset = getattr(
                assignment,
                "normalized_offset",
                0.0,
            )

            cv2.putText(
                output,
                (
                    f"Lane: {lane_id} | "
                    f"Offset: {float(offset):+.3f}"
                ),
                (20, 270),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # --------------------------------------------------------------------
        # ADAS
        # --------------------------------------------------------------------

        adas = result.adas

        if adas is not None:

            state = getattr(
                adas,
                "state",
                "UNKNOWN",
            )

            confidence = getattr(
                adas,
                "confidence",
                0.0,
            )

            cv2.putText(
                output,
                (
                    f"ADAS: {state} | "
                    f"Conf: {float(confidence):.2f}"
                ),
                (20, 300),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return output


    # ========================================================================
    # LOOP
    # ========================================================================

    def run(
        self,
        max_frames: Optional[int] = None,
    ) -> None:

        self.initialize()

        self.running = True

        processed = 0

        LOGGER.info(
            "Runtime started."
        )

        try:

            while self.running:

                frame = self._get_frame()

                if frame is None:

                    time.sleep(
                        0.001
                    )

                    continue

                try:

                    result = self.process_frame(
                        frame
                    )

                except Exception as exc:

                    LOGGER.exception(
                        "Pipeline error: %s",
                        exc,
                    )

                    continue

                processed += 1

                # ------------------------------------------------------------
                # DEBUG
                # ------------------------------------------------------------

                if (
                    DEBUG.enabled
                    and DEBUG.print_pipeline_summary
                    and (
                        processed == 1
                        or processed % DEBUG.debug_frame_interval == 0
                    )
                ):

                    self._print_summary(
                        result
                    )

                # ------------------------------------------------------------
                # DISPLAY
                # ------------------------------------------------------------

                if self._display_enabled:

                    output = self._draw_overlay(
                        result
                    )

                    cv2.imshow(
                        VISUALIZATION.window_name,
                        output,
                    )

                    key = (
                        cv2.waitKey(
                            VISUALIZATION.wait_key_ms
                        )
                        & 0xFF
                    )

                    if key == 27:
                        break

                # ------------------------------------------------------------
                # LIMITADOR DE TESTE
                # ------------------------------------------------------------

                if (
                    max_frames is not None
                    and processed >= max_frames
                ):

                    break

        finally:

            self.shutdown()


    # ========================================================================
    # SUMMARY
    # ========================================================================

    def _print_summary(
        self,
        result: PipelineResult,
    ) -> None:

        detection = result.detection

        lane_count = 0

        if detection is not None:

            lanes = getattr(
                detection,
                "lanes",
                None,
            )

            if lanes is not None:
                lane_count = len(lanes)

        tracking_count = 0

        if result.tracking is not None:

            lanes = getattr(
                result.tracking,
                "lanes",
                None,
            )

            if lanes is not None:
                tracking_count = len(lanes)

        geometry = result.geometry

        LOGGER.info(
            (
                "FRAME %d | "
                "YOLOP lanes=%d | "
                "tracks=%d | "
                "geometry=%s | "
                "models=%d | "
                "projections=%d | "
                "assignment=%s | "
                "ADAS=%s | "
                "%.1fms"
            ),
            result.frame_index,
            lane_count,
            tracking_count,
            (
                "VALID"
                if geometry is not None
                and geometry.valid
                else "INVALID"
            ),
            len(result.models),
            len(result.projections),
            (
                "VALID"
                if self._valid(result.assignment)
                else "INVALID"
            ),
            getattr(
                result.adas,
                "state",
                "lane_lost",
            ),
            self.statistics.total_time_ms,
        )


    # ========================================================================
    # SHUTDOWN
    # ========================================================================

    def shutdown(self) -> None:

        if not self.running:
            return

        self.running = False

        LOGGER.info(
            "Shutting down Forza Assistents..."
        )

        if self.capture is not None:

            try:
                self.capture.stop()

            except Exception as exc:

                LOGGER.debug(
                    "Capture shutdown error: %s",
                    exc,
                )

        try:
            cv2.destroyAllWindows()

        except Exception:
            pass

        LOGGER.info(
            "Forza Assistents stopped."
        )


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Forza Assistents"
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Número máximo de frames a processar.",
    )

    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Executa sem janela OpenCV.",
    )

    return parser.parse_args()


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    args = parse_args()

    application = ForzaAssistents()

    if args.no_display:
        application._display_enabled = False

    try:

        application.run(
            max_frames=args.frames
        )

    except KeyboardInterrupt:

        LOGGER.info(
            "Interrupted by user."
        )

        application.shutdown()

    except Exception:

        LOGGER.exception(
            "Fatal application error."
        )

        application.shutdown()

        raise


if __name__ == "__main__":
    main()