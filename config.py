"""
Forza Assistents - Central Configuration
========================================

Configuração central e imutável do sistema.

Pipeline:

    Screen Capture
        ↓
    ROI
        ↓
    YOLOP
        ↓
    Lane Tracker
        ↓
    Lane Geometry
        ↓
    Lane Model
        ↓
    Lane Projection
        ↓
    Lane Assignment
        ↓
    ADAS State
        ↓
    Visualization
        ↓
    Future LKA / G29

Princípios:

- uma única fonte de configuração;
- nenhuma lógica de visão neste módulo;
- parâmetros explícitos;
- configuração imutável;
- validação centralizada;
- ROI definido na coordenada da tela;
- dimensões internas derivadas do ROI;
- MONITOR como padrão;
- controle físico desligado por padrão;
- compatibilidade com os módulos atuais;
- preparado para execução em tempo real.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Tuple


# ============================================================================
# PROJECT
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

WEIGHTS_DIR = PROJECT_ROOT / "weights"
CALIBRATION_DIR = PROJECT_ROOT / "calibration"
LOG_DIR = PROJECT_ROOT / "logs"
SCREENSHOT_DIR = PROJECT_ROOT / "screenshots"

YOLOP_MODEL_PATH = WEIGHTS_DIR / "yolop-640-640.onnx"
CALIBRATION_FILE = CALIBRATION_DIR / "camera_calibration.json"


# ============================================================================
# DISPLAY / SCREEN
# ============================================================================

@dataclass(frozen=True)
class DisplayConfig:
    """
    Configuração física do monitor utilizado pelo jogo.

    Forza Horizon:
        2560 x 1600
    """

    width: int = 2560
    height: int = 1600

    monitor_index: int = 0

    # O jogo ocupa a tela inteira.
    fullscreen: bool = True


DISPLAY = DisplayConfig()


# ============================================================================
# RUNTIME MODE
# ============================================================================

class RuntimeMode(str, Enum):
    DISABLED = "disabled"
    MONITOR = "monitor"
    ASSIST = "assist"


DEFAULT_RUNTIME_MODE = RuntimeMode.MONITOR


# ============================================================================
# SAFETY
# ============================================================================

@dataclass(frozen=True)
class SafetyConfig:
    """
    Camada de segurança do sistema.

    Nenhum comando físico é permitido enquanto o sistema
    estiver em MONITOR.
    """

    enable_control: bool = False

    emergency_stop_enabled: bool = True

    require_valid_lane: bool = True

    require_geometry_valid: bool = True

    require_assignment_valid: bool = True

    minimum_adas_confidence: float = 0.55

    maximum_allowed_detection_age_frames: int = 5

    maximum_lost_frames_before_disable: int = 15

    zero_command_on_invalid_state: bool = True


SAFETY = SafetyConfig()


# ============================================================================
# SCREEN CAPTURE
# ============================================================================

@dataclass(frozen=True)
class CaptureConfig:
    """
    Captura da tela.

    A coordenada do ROI é definida na resolução ORIGINAL da tela.

    Tela:
        2560 x 1600

    ROI:
        x1 = 300
        y1 = 700
        x2 = 2200
        y2 = 1600

    Resultado:
        1900 x 900

    Esse é o ROI utilizado antes da integração do ADAS Display.
    """

    backend: str = "dxcam"

    monitor_index: int = DISPLAY.monitor_index

    target_fps: int = 60

    max_fps: int = 120

    capture_full_screen: bool = False

    use_roi: bool = True

    # ------------------------------------------------------------------------
    # ROI - SCREEN COORDINATES
    # ------------------------------------------------------------------------

    roi: Tuple[int, int, int, int] = (
        300,
        700,
        2200,
        1600,
    )

    copy_frame: bool = False

    output_color_format: str = "BGR"

    # Permite que a captura se recupere de mudanças do Desktop Duplication.
    enable_recovery: bool = True

    recovery_delay_ms: int = 100


CAPTURE = CaptureConfig()


# ============================================================================
# YOLOP
# ============================================================================

@dataclass(frozen=True)
class YOLOPConfig:

    model_path: Path = YOLOP_MODEL_PATH

    input_width: int = 640

    input_height: int = 640

    confidence_threshold: float = 0.50

    lane_confidence_threshold: float = 0.50

    segmentation_threshold: float = 0.50

    use_cuda: bool = True

    use_tensorrt: bool = False

    use_cpu_fallback: bool = True

    enable_dynamic_provider_fallback: bool = True

    max_lanes: int = 16

    minimum_lane_points: int = 4

    preserve_low_confidence_lanes: bool = True

    input_scale: float = 1.0 / 255.0

    mean: Tuple[float, float, float] = (
        0.0,
        0.0,
        0.0,
    )

    std: Tuple[float, float, float] = (
        1.0,
        1.0,
        1.0,
    )


YOLOP = YOLOPConfig()


# ============================================================================
# LANE TRACKER
# ============================================================================

@dataclass(frozen=True)
class LaneTrackerConfig:

    max_lost_frames: int = 12

    min_stable_frames: int = 3

    history_size: int = 12

    max_tracks: int = 16

    min_confidence: float = 0.35

    association_distance: float = 120.0

    max_lane_width_change_ratio: float = 0.35

    confidence_decay: float = 0.92

    velocity_smoothing: float = 0.70

    enable_prediction: bool = True

    enable_identity_preservation: bool = True

    enable_lane_swap_protection: bool = True


LANE_TRACKER = LaneTrackerConfig()


# ============================================================================
# LANE GEOMETRY
# ============================================================================

@dataclass(frozen=True)
class LaneGeometryConfig:
    """
    Geometria sempre trabalha com as dimensões do FRAME APÓS ROI.

    ROI:
        1900 x 900
    """

    image_width: int = 1900

    image_height: int = 900

    roi_height: int = 900

    min_lane_confidence: float = 0.35

    min_points: int = 4

    min_observed_span: float = 20.0

    min_lane_width: float = 120.0

    max_lane_width: float = 1000.0

    expected_lane_width: float = 312.0

    lane_width_tolerance: float = 0.50

    heading_lookahead_ratio: float = 0.75

    curvature_lookahead_ratio: float = 0.70

    max_heading_error: float = 1.0

    max_curvature_score: float = 1.0

    enable_polynomial_fit: bool = True

    polynomial_degree: int = 2

    enable_outlier_rejection: bool = True

    outlier_sigma: float = 2.5

    confidence_weight_detection: float = 0.35

    confidence_weight_span: float = 0.20

    confidence_weight_width: float = 0.20

    confidence_weight_geometry: float = 0.25


LANE_GEOMETRY = LaneGeometryConfig()


# ============================================================================
# LANE MODEL
# ============================================================================

@dataclass(frozen=True)
class LaneModelConfig:

    polynomial_degree: int = 2

    minimum_points: int = 6

    minimum_y_span: float = 20.0

    max_outlier_iterations: int = 3

    outlier_threshold: float = 2.5

    minimum_confidence: float = 0.35

    projection_samples: int = 32

    enable_polynomial_smoothing: bool = True

    enable_outlier_rejection: bool = True


LANE_MODEL = LaneModelConfig()


# ============================================================================
# LANE PROJECTION
# ============================================================================

@dataclass(frozen=True)
class LaneProjectionConfig:

    enabled: bool = True

    max_projection_distance: float = 900.0

    minimum_points: int = 4

    minimum_confidence: float = 0.40

    samples: int = 32

    lookahead_distance: float = 500.0

    near_distance: float = 100.0

    far_distance: float = 700.0

    enable_extrapolation: bool = True

    extrapolation_limit: float = 300.0


LANE_PROJECTION = LaneProjectionConfig()


# ============================================================================
# LANE ASSIGNMENT
# ============================================================================

@dataclass(frozen=True)
class LaneAssignmentConfig:

    enabled: bool = True

    minimum_confidence: float = 0.40

    expected_lane_width: float = 312.0

    lane_width_tolerance: float = 0.45

    maximum_lateral_offset_ratio: float = 1.25

    center_reference_ratio: float = 0.50

    minimum_lane_separation: float = 80.0

    maximum_lane_separation: float = 900.0

    enable_multi_lane_assignment: bool = True

    max_left_lanes: int = 8

    max_right_lanes: int = 8


LANE_ASSIGNMENT = LaneAssignmentConfig()


# ============================================================================
# ADAS
# ============================================================================

@dataclass(frozen=True)
class ADASConfig:

    enabled: bool = True

    minimum_confidence: float = 0.50

    warning_threshold: float = 0.35

    critical_threshold: float = 0.75

    heading_warning_threshold: float = 0.35

    heading_critical_threshold: float = 0.65

    curvature_warning_threshold: float = 0.50

    enable_lane_lost_state: bool = True

    enable_left_warning: bool = True

    enable_right_warning: bool = True

    enable_centered_state: bool = True

    enable_slight_left_state: bool = True

    enable_slight_right_state: bool = True

    enable_critical_left_state: bool = True

    enable_critical_right_state: bool = True


ADAS = ADASConfig()


# ============================================================================
# TEMPORAL FILTER
# ============================================================================

@dataclass(frozen=True)
class TemporalFilterConfig:

    enabled: bool = True

    alpha: float = 0.35

    invalid_decay: float = 0.90

    maximum_history: int = 12

    reset_after_invalid_frames: int = 15

    filter_lateral_error: bool = True

    filter_heading_error: bool = True

    filter_curvature: bool = True

    filter_confidence: bool = True


TEMPORAL_FILTER = TemporalFilterConfig()


# ============================================================================
# LKA
# ============================================================================

@dataclass(frozen=True)
class LKAConfig:

    enabled: bool = False

    kp: float = 0.80

    kd: float = 0.15

    heading_gain: float = 0.50

    curvature_gain: float = 0.20

    max_steering: float = 0.70

    max_steering_rate: float = 0.08

    deadband: float = 0.02

    minimum_confidence: float = 0.65

    disable_on_lane_loss: bool = True

    disable_on_geometry_invalid: bool = True

    disable_on_assignment_invalid: bool = True


LKA = LKAConfig()


# ============================================================================
# G29
# ============================================================================

@dataclass(frozen=True)
class G29Config:

    enabled: bool = False

    device_name: str = "Logitech G29"

    steering_min: float = -1.0

    steering_max: float = 1.0

    center: float = 0.0

    deadzone: float = 0.02

    maximum_output: float = 0.70

    smoothing: float = 0.20

    fail_safe_enabled: bool = True


G29 = G29Config()


# ============================================================================
# VISUALIZATION
# ============================================================================

@dataclass(frozen=True)
class VisualizationConfig:

    enabled: bool = True

    show_roi: bool = True

    show_raw_lanes: bool = True

    show_tracked_lanes: bool = True

    show_lane_models: bool = True

    show_projected_lanes: bool = True

    show_current_lane: bool = True

    show_geometry: bool = True

    show_adas_state: bool = True

    show_confidence: bool = True

    show_fps: bool = True

    show_timing: bool = True

    show_debug_text: bool = True

    show_lane_ids: bool = True

    window_name: str = "Forza Assistents ADAS"

    wait_key_ms: int = 1


VISUALIZATION = VisualizationConfig()


# ============================================================================
# PERFORMANCE
# ============================================================================

@dataclass(frozen=True)
class PerformanceConfig:

    target_capture_fps: int = 60

    target_inference_fps: int = 30

    target_pipeline_fps: int = 30

    target_control_hz: int = 120

    enable_timing_metrics: bool = True

    enable_fps_metrics: bool = True

    warmup_frames: int = 10

    max_pipeline_latency_ms: float = 50.0

    max_inference_latency_ms: float = 35.0

    enable_frame_skip: bool = False

    max_frame_queue: int = 2


PERFORMANCE = PerformanceConfig()


# ============================================================================
# LOGGING
# ============================================================================

@dataclass(frozen=True)
class LoggingConfig:

    enabled: bool = True

    level: str = "INFO"

    log_to_file: bool = True

    log_to_console: bool = True

    directory: Path = LOG_DIR

    filename: str = "forza_assistents.log"

    max_file_size_mb: int = 20

    backup_count: int = 5

    log_detection_failures: bool = True

    log_provider_changes: bool = True

    log_performance_warnings: bool = True

    log_safety_events: bool = True


LOGGING = LoggingConfig()


# ============================================================================
# HOTKEYS
# ============================================================================

@dataclass(frozen=True)
class HotkeyConfig:

    toggle_monitor: str = "m"

    emergency_stop: str = "F9"

    toggle_overlay: str = "F10"

    exit: str = "ESC"


HOTKEYS = HotkeyConfig()


# ============================================================================
# DEBUG
# ============================================================================

@dataclass(frozen=True)
class DebugConfig:

    enabled: bool = True

    print_pipeline_summary: bool = True

    print_detector_summary: bool = True

    print_tracker_summary: bool = True

    print_geometry_summary: bool = True

    print_assignment_summary: bool = True

    print_adas_summary: bool = True

    save_debug_frames: bool = False

    save_invalid_frames: bool = False

    save_detection_failures: bool = False

    debug_frame_interval: int = 30


DEBUG = DebugConfig()


# ============================================================================
# LEGACY UFLD COMPATIBILITY
# ============================================================================

UFLD_INPUT_WIDTH = 800
UFLD_INPUT_HEIGHT = 288
UFLD_GRIDING_NUM = 200
UFLD_ROW_ANCHORS = 18


# ============================================================================
# SYSTEM CONFIG
# ============================================================================

@dataclass(frozen=True)
class SystemConfig:

    project_root: Path = PROJECT_ROOT

    display: DisplayConfig = field(default=DISPLAY)

    runtime_mode: RuntimeMode = DEFAULT_RUNTIME_MODE

    safety: SafetyConfig = field(default=SAFETY)

    capture: CaptureConfig = field(default=CAPTURE)

    yolop: YOLOPConfig = field(default=YOLOP)

    lane_tracker: LaneTrackerConfig = field(
        default=LANE_TRACKER
    )

    lane_geometry: LaneGeometryConfig = field(
        default=LANE_GEOMETRY
    )

    lane_model: LaneModelConfig = field(
        default=LANE_MODEL
    )

    lane_projection: LaneProjectionConfig = field(
        default=LANE_PROJECTION
    )

    lane_assignment: LaneAssignmentConfig = field(
        default=LANE_ASSIGNMENT
    )

    adas: ADASConfig = field(default=ADAS)

    temporal_filter: TemporalFilterConfig = field(
        default=TEMPORAL_FILTER
    )

    lka: LKAConfig = field(default=LKA)

    g29: G29Config = field(default=G29)

    visualization: VisualizationConfig = field(
        default=VISUALIZATION
    )

    performance: PerformanceConfig = field(
        default=PERFORMANCE
    )

    logging: LoggingConfig = field(
        default=LOGGING
    )

    hotkeys: HotkeyConfig = field(
        default=HOTKEYS
    )

    debug: DebugConfig = field(
        default=DEBUG
    )


CONFIG = SystemConfig()


# ============================================================================
# ROI HELPERS
# ============================================================================

def get_roi() -> Tuple[int, int, int, int]:
    """
    Retorna o ROI na coordenada original da tela.
    """

    return CAPTURE.roi


def get_roi_size() -> Tuple[int, int]:
    """
    Retorna:

        width, height

    do frame após aplicação do ROI.
    """

    x1, y1, x2, y2 = CAPTURE.roi

    return (
        x2 - x1,
        y2 - y1,
    )


def get_roi_width() -> int:
    return CAPTURE.roi[2] - CAPTURE.roi[0]


def get_roi_height() -> int:
    return CAPTURE.roi[3] - CAPTURE.roi[1]


# ============================================================================
# PATH HELPERS
# ============================================================================

def get_project_root() -> Path:
    return PROJECT_ROOT


def get_yolop_model_path() -> Path:
    return YOLOP_MODEL_PATH


def get_calibration_path() -> Path:
    return CALIBRATION_FILE


def get_log_directory() -> Path:
    return LOG_DIR


# ============================================================================
# DIRECTORY MANAGEMENT
# ============================================================================

def ensure_directories() -> None:
    """
    Cria os diretórios necessários para execução.
    """

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    SCREENSHOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CALIBRATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================================
# VALIDATION
# ============================================================================

def validate_config() -> None:
    """
    Validação central da configuração.

    Qualquer configuração impossível deve falhar antes
    da inicialização do pipeline.
    """

    # ------------------------------------------------------------------------
    # DISPLAY
    # ------------------------------------------------------------------------

    if DISPLAY.width <= 0:
        raise ValueError(
            "DISPLAY.width must be > 0"
        )

    if DISPLAY.height <= 0:
        raise ValueError(
            "DISPLAY.height must be > 0"
        )

    # ------------------------------------------------------------------------
    # CAPTURE
    # ------------------------------------------------------------------------

    if CAPTURE.target_fps <= 0:
        raise ValueError(
            "CAPTURE.target_fps must be > 0"
        )

    if CAPTURE.max_fps < CAPTURE.target_fps:
        raise ValueError(
            "CAPTURE.max_fps must be >= CAPTURE.target_fps"
        )

    if CAPTURE.backend.lower() != "dxcam":
        raise ValueError(
            f"Unsupported capture backend: {CAPTURE.backend}"
        )

    x1, y1, x2, y2 = CAPTURE.roi

    if x1 < 0 or y1 < 0:
        raise ValueError(
            f"ROI coordinates cannot be negative: {CAPTURE.roi}"
        )

    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"Invalid ROI coordinates: {CAPTURE.roi}"
        )

    if x2 > DISPLAY.width:
        raise ValueError(
            f"ROI exceeds screen width: "
            f"{CAPTURE.roi} > {DISPLAY.width}px"
        )

    if y2 > DISPLAY.height:
        raise ValueError(
            f"ROI exceeds screen height: "
            f"{CAPTURE.roi} > {DISPLAY.height}px"
        )

    roi_width = x2 - x1
    roi_height = y2 - y1

    if roi_width < 100:
        raise ValueError(
            f"ROI width is too small: {roi_width}px"
        )

    if roi_height < 100:
        raise ValueError(
            f"ROI height is too small: {roi_height}px"
        )

    # ------------------------------------------------------------------------
    # YOLOP
    # ------------------------------------------------------------------------

    if YOLOP.input_width <= 0:
        raise ValueError(
            "YOLOP.input_width must be > 0"
        )

    if YOLOP.input_height <= 0:
        raise ValueError(
            "YOLOP.input_height must be > 0"
        )

    for name, value in (
        (
            "YOLOP.confidence_threshold",
            YOLOP.confidence_threshold,
        ),
        (
            "YOLOP.lane_confidence_threshold",
            YOLOP.lane_confidence_threshold,
        ),
        (
            "YOLOP.segmentation_threshold",
            YOLOP.segmentation_threshold,
        ),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{name} must be between 0 and 1"
            )

    if YOLOP.max_lanes <= 0:
        raise ValueError(
            "YOLOP.max_lanes must be > 0"
        )

    if YOLOP.minimum_lane_points < 2:
        raise ValueError(
            "YOLOP.minimum_lane_points must be >= 2"
        )

    # ------------------------------------------------------------------------
    # TRACKER
    # ------------------------------------------------------------------------

    if LANE_TRACKER.max_lost_frames < 0:
        raise ValueError(
            "LANE_TRACKER.max_lost_frames must be >= 0"
        )

    if LANE_TRACKER.history_size <= 0:
        raise ValueError(
            "LANE_TRACKER.history_size must be > 0"
        )

    if LANE_TRACKER.max_tracks <= 0:
        raise ValueError(
            "LANE_TRACKER.max_tracks must be > 0"
        )

    if not 0.0 <= LANE_TRACKER.confidence_decay <= 1.0:
        raise ValueError(
            "LANE_TRACKER.confidence_decay must be between 0 and 1"
        )

    # ------------------------------------------------------------------------
    # GEOMETRY
    # ------------------------------------------------------------------------

    if LANE_GEOMETRY.image_width != roi_width:
        raise ValueError(
            "LANE_GEOMETRY.image_width does not match ROI width: "
            f"{LANE_GEOMETRY.image_width} != {roi_width}"
        )

    if LANE_GEOMETRY.image_height != roi_height:
        raise ValueError(
            "LANE_GEOMETRY.image_height does not match ROI height: "
            f"{LANE_GEOMETRY.image_height} != {roi_height}"
        )

    if LANE_GEOMETRY.min_points < 2:
        raise ValueError(
            "LANE_GEOMETRY.min_points must be >= 2"
        )

    if LANE_GEOMETRY.min_lane_width <= 0:
        raise ValueError(
            "LANE_GEOMETRY.min_lane_width must be > 0"
        )

    if (
        LANE_GEOMETRY.max_lane_width
        <= LANE_GEOMETRY.min_lane_width
    ):
        raise ValueError(
            "LANE_GEOMETRY.max_lane_width must be greater "
            "than min_lane_width"
        )

    # ------------------------------------------------------------------------
    # LANE MODEL
    # ------------------------------------------------------------------------

    if LANE_MODEL.polynomial_degree < 1:
        raise ValueError(
            "LANE_MODEL.polynomial_degree must be >= 1"
        )

    if LANE_MODEL.minimum_points < 2:
        raise ValueError(
            "LANE_MODEL.minimum_points must be >= 2"
        )

    # ------------------------------------------------------------------------
    # PROJECTION
    # ------------------------------------------------------------------------

    if LANE_PROJECTION.samples <= 0:
        raise ValueError(
            "LANE_PROJECTION.samples must be > 0"
        )

    if LANE_PROJECTION.minimum_points < 2:
        raise ValueError(
            "LANE_PROJECTION.minimum_points must be >= 2"
        )

    # ------------------------------------------------------------------------
    # ADAS
    # ------------------------------------------------------------------------

    if not 0.0 <= ADAS.minimum_confidence <= 1.0:
        raise ValueError(
            "ADAS.minimum_confidence must be between 0 and 1"
        )

    if not 0.0 <= SAFETY.minimum_adas_confidence <= 1.0:
        raise ValueError(
            "SAFETY.minimum_adas_confidence must be between 0 and 1"
        )

    # ------------------------------------------------------------------------
    # TEMPORAL FILTER
    # ------------------------------------------------------------------------

    if not 0.0 < TEMPORAL_FILTER.alpha <= 1.0:
        raise ValueError(
            "TEMPORAL_FILTER.alpha must be > 0 and <= 1"
        )

    if not 0.0 <= TEMPORAL_FILTER.invalid_decay <= 1.0:
        raise ValueError(
            "TEMPORAL_FILTER.invalid_decay must be between 0 and 1"
        )

    # ------------------------------------------------------------------------
    # LKA
    # ------------------------------------------------------------------------

    if not 0.0 <= LKA.max_steering <= 1.0:
        raise ValueError(
            "LKA.max_steering must be between 0 and 1"
        )

    if LKA.max_steering_rate <= 0:
        raise ValueError(
            "LKA.max_steering_rate must be > 0"
        )

    # ------------------------------------------------------------------------
    # G29
    # ------------------------------------------------------------------------

    if not (
        G29.steering_min
        < G29.center
        < G29.steering_max
    ):
        raise ValueError(
            "Invalid G29 steering range"
        )

    # ------------------------------------------------------------------------
    # SAFETY
    # ------------------------------------------------------------------------

    if (
        DEFAULT_RUNTIME_MODE != RuntimeMode.ASSIST
        and G29.enabled
    ):
        raise ValueError(
            "G29 cannot be enabled outside ASSIST mode"
        )

    if (
        DEFAULT_RUNTIME_MODE != RuntimeMode.ASSIST
        and LKA.enabled
    ):
        raise ValueError(
            "LKA cannot be enabled outside ASSIST mode"
        )


# ============================================================================
# STARTUP VALIDATION
# ============================================================================

validate_config()