"""
config.py

Configuração central do Forza Assistents.

Arquitetura:

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
    Lane Models
        ↓
    Lane Projection
        ↓
    Lane Assignment
        ↓
    ADAS State
        ↓
    Main / Monitor
        ↓
    Futuramente: LKA / G29

Princípios:
- configuração centralizada;
- nenhuma lógica de visão neste arquivo;
- compatibilidade com os módulos atuais;
- MONITOR como modo padrão;
- controle físico desabilitado por segurança;
- parâmetros explícitos e facilmente ajustáveis;
- preparado para execução em tempo real.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Tuple


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

WEIGHTS_DIR = PROJECT_ROOT / "weights"

YOLOP_MODEL_PATH = WEIGHTS_DIR / "yolop-640-640.onnx"

CALIBRATION_DIR = PROJECT_ROOT / "calibration"
CALIBRATION_FILE = CALIBRATION_DIR / "camera_calibration.json"

LOG_DIR = PROJECT_ROOT / "logs"

SCREENSHOT_DIR = PROJECT_ROOT / "screenshots"


# ============================================================================
# RUNTIME MODE
# ============================================================================

class RuntimeMode(str, Enum):
    """
    Estado operacional do sistema.
    """

    DISABLED = "disabled"
    MONITOR = "monitor"
    ASSIST = "assist"


DEFAULT_RUNTIME_MODE = RuntimeMode.MONITOR


# ============================================================================
# GLOBAL SAFETY
# ============================================================================

@dataclass(frozen=True)
class SafetyConfig:
    """
    Configurações de segurança do sistema.

    O projeto deve permanecer em MONITOR até que toda a cadeia
    de percepção e decisão seja validada em condições reais.
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
    Configuração da captura da tela.

    A captura deve permanecer independente da inferência.
    """

    backend: str = "dxcam"

    monitor_index: int = 0

    target_fps: int = 60

    max_fps: int = 120

    capture_full_screen: bool = True

    # ROI na tela original.
    #
    # x1, y1, x2, y2
    roi: Tuple[int, int, int, int] = (
        300,
        700,
        2200,
        1600,
    )

    # Quando True, o ROI acima é usado diretamente.
    # Quando False, o sistema pode utilizar a imagem inteira.
    use_roi: bool = True

    # Evita cópias desnecessárias quando possível.
    copy_frame: bool = False

    # BGR é utilizado pelo OpenCV.
    output_color_format: str = "BGR"


CAPTURE = CaptureConfig()


# ============================================================================
# YOLOP
# ============================================================================

@dataclass(frozen=True)
class YOLOPConfig:
    """
    Configuração do detector YOLOP.
    """

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

    # Número máximo de lanes que o pipeline aceita.
    #
    # Importante:
    # o detector deve preservar todas as linhas detectadas.
    max_lanes: int = 16

    # Número mínimo de pontos para uma lane ser considerada.
    minimum_lane_points: int = 4

    # Permite manter detecções de baixa confiança para posterior
    # filtragem pelo tracker/modelo.
    preserve_low_confidence_lanes: bool = True

    # Normalização esperada pelo modelo.
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
    """
    Rastreamento temporal das lanes.

    O detector trabalha frame a frame.
    O tracker fornece continuidade temporal.
    """

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
    Estima a geometria da faixa atual.

    Trabalha com:
    - posição lateral;
    - heading;
    - largura;
    - curvatura;
    - extensão observada;
    - confiança.
    """

    image_width: int = 1987

    image_height: int = 698

    roi_height: int = 400

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
    """
    Configuração da modelagem matemática das lanes.
    """

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
    """
    Projeção das lanes para pontos úteis ao ADAS.
    """

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
    """
    Determina qual lane representa a faixa atual e quais são
    as lanes à esquerda e à direita.
    """

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
# ADAS STATE
# ============================================================================

@dataclass(frozen=True)
class ADASConfig:
    """
    Máquina de estados do ADAS.
    """

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
    """
    Suavização temporal das métricas.

    Reduz jitter sem introduzir atraso excessivo.
    """

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
# CONTROL / LKA
# ============================================================================

@dataclass(frozen=True)
class LKAConfig:
    """
    Configuração futura do Lane Keeping Assist.

    Por segurança, o controle permanece desabilitado.
    """

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
    """
    Interface do Logitech G29.

    Permanece desabilitada durante a fase de validação.
    """

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
# VISUALIZATION / DEBUG
# ============================================================================

@dataclass(frozen=True)
class VisualizationConfig:
    """
    Debug visual do sistema.

    Extremamente importante durante a fase de validação no jogo.
    """

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
    """
    Metas de execução em tempo real.
    """

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
    """
    Logging operacional.
    """

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
    """
    Teclas de operação durante os testes.
    """

    toggle_monitor: str = "m"

    emergency_stop: str = "F9"

    toggle_overlay: str = "F10"

    exit: str = "ESC"


HOTKEYS = HotkeyConfig()


# ============================================================================
# DEBUG / DEVELOPMENT
# ============================================================================

@dataclass(frozen=True)
class DebugConfig:
    """
    Opções úteis durante desenvolvimento.
    """

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
# COMPATIBILITY / LEGACY
# ============================================================================

# Compatibilidade com código antigo que ainda possa importar
# constantes relacionadas ao UFLD.

UFLD_INPUT_WIDTH = 800
UFLD_INPUT_HEIGHT = 288
UFLD_GRIDING_NUM = 200
UFLD_ROW_ANCHORS = 18


# ============================================================================
# GLOBAL SYSTEM CONFIG
# ============================================================================

@dataclass(frozen=True)
class SystemConfig:
    """
    Configuração global do sistema.
    """

    project_root: Path = PROJECT_ROOT

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
# VALIDATION
# ============================================================================

def validate_config() -> None:
    """
    Valida configurações críticas antes da inicialização.
    """

    if CAPTURE.target_fps <= 0:
        raise ValueError("CAPTURE.target_fps must be > 0")

    if YOLOP.input_width <= 0 or YOLOP.input_height <= 0:
        raise ValueError("YOLOP input dimensions must be > 0")

    if not 0.0 <= YOLOP.confidence_threshold <= 1.0:
        raise ValueError(
            "YOLOP.confidence_threshold must be between 0 and 1"
        )

    if not 0.0 <= YOLOP.lane_confidence_threshold <= 1.0:
        raise ValueError(
            "YOLOP.lane_confidence_threshold must be between 0 and 1"
        )

    if LANE_TRACKER.max_lost_frames < 0:
        raise ValueError(
            "LANE_TRACKER.max_lost_frames must be >= 0"
        )

    if LANE_TRACKER.history_size <= 0:
        raise ValueError(
            "LANE_TRACKER.history_size must be > 0"
        )

    if LANE_GEOMETRY.min_points < 2:
        raise ValueError(
            "LANE_GEOMETRY.min_points must be >= 2"
        )

    if LANE_MODEL.polynomial_degree < 1:
        raise ValueError(
            "LANE_MODEL.polynomial_degree must be >= 1"
        )

    if LANE_PROJECTION.samples <= 0:
        raise ValueError(
            "LANE_PROJECTION.samples must be > 0"
        )

    if not 0.0 <= SAFETY.minimum_adas_confidence <= 1.0:
        raise ValueError(
            "SAFETY.minimum_adas_confidence must be between 0 and 1"
        )

    if not 0.0 <= LKA.max_steering <= 1.0:
        raise ValueError(
            "LKA.max_steering must be between 0 and 1"
        )

    # Controle físico nunca deve ser habilitado acidentalmente.
    if DEFAULT_RUNTIME_MODE != RuntimeMode.ASSIST:
        if G29.enabled:
            raise ValueError(
                "G29 cannot be enabled outside ASSIST mode"
            )

        if LKA.enabled:
            raise ValueError(
                "LKA cannot be enabled outside ASSIST mode"
            )


# ============================================================================
# HELPERS
# ============================================================================

def get_project_root() -> Path:
    return PROJECT_ROOT


def get_yolop_model_path() -> Path:
    return YOLOP_MODEL_PATH


def get_calibration_path() -> Path:
    return CALIBRATION_FILE


def get_log_directory() -> Path:
    return LOG_DIR


def ensure_directories() -> None:
    """
    Cria somente diretórios necessários.
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
# INITIAL VALIDATION
# ============================================================================

validate_config()