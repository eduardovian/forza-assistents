"""
config.py

Forza Assistents
================

Configuração central e única do sistema.

PRINCÍPIOS
----------

1. Este módulo é a única fonte de verdade para configuração.
2. ROI é produzido pelo calibrador e persistido em:
       calibration/camera_calibration.json
3. Todos os módulos consomem o mesmo ROI através de:
       config.ROI
4. Nenhum módulo downstream deve declarar ROI próprio.
5. Configurações são imutáveis em runtime.
6. Valores inválidos são rejeitados cedo.
7. O pipeline trabalha em coordenadas do frame recebido.
8. Segurança possui prioridade sobre disponibilidade.
9. Monitoramento é o modo padrão.
10. Controle físico permanece desabilitado por padrão.
11. O modelo oficial de lane é cúbico.
12. Nenhuma camada deve duplicar configuração pertencente
    a este módulo.

IMPORTANTE
----------

Este projeto não é um sistema automotivo certificado.

As estruturas abaixo buscam aplicar princípios de engenharia
robusta, determinística, fail-safe e testável.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Final


# =============================================================================
# PROJECT
# =============================================================================

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent

CALIBRATION_DIR: Final[Path] = (
    PROJECT_ROOT / "calibration"
)

CALIBRATION_FILE: Final[Path] = (
    CALIBRATION_DIR / "camera_calibration.json"
)

WEIGHTS_DIR: Final[Path] = (
    PROJECT_ROOT / "weights"
)

YOLOP_MODEL_PATH: Final[Path] = (
    
    WEIGHTS_DIR / "yolopv2.pt"
    
)

LOG_DIR: Final[Path] = (
    PROJECT_ROOT / "logs"
)

SCREENSHOT_DIR: Final[Path] = (
    PROJECT_ROOT / "screenshots"
)


# =============================================================================
# RUNTIME
# =============================================================================


class RuntimeMode(str, Enum):
    """
    Estado operacional global do sistema.
    """

    DISABLED = "disabled"
    MONITOR = "monitor"
    ASSIST = "assist"


DEFAULT_RUNTIME_MODE: Final[RuntimeMode] = (
    RuntimeMode.MONITOR
)


# =============================================================================
# ROI
# =============================================================================


@dataclass(frozen=True, slots=True)
class ROIConfig:
    """
    Região de interesse produzida pelo calibrador.

    Coordenadas em relação à tela inteira:

        left
        top
        right
        bottom

    Depois do recorte, o frame resultante possui seu próprio
    sistema de coordenadas:

        x = 0 ... width
        y = 0 ... height

    Os módulos de visão trabalham nesse sistema local.

    Nenhum módulo downstream deve redefinir ROI.
    """

    enabled: bool

    left: int
    top: int
    right: int
    bottom: int

    @property
    def rectangle(self) -> tuple[int, int, int, int]:
        return (
            self.left,
            self.top,
            self.right,
            self.bottom,
        )

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def size(self) -> tuple[int, int]:
        return (
            self.width,
            self.height,
        )

    def validate(
        self,
        *,
        screen_width: int | None = None,
        screen_height: int | None = None,
    ) -> None:
        """
        Valida a geometria do ROI.
        """

        if not self.enabled:
            return

        if self.left < 0:
            raise ValueError(
                "ROI.left não pode ser negativo."
            )

        if self.top < 0:
            raise ValueError(
                "ROI.top não pode ser negativo."
            )

        if self.right <= self.left:
            raise ValueError(
                "ROI.right deve ser maior que ROI.left."
            )

        if self.bottom <= self.top:
            raise ValueError(
                "ROI.bottom deve ser maior que ROI.top."
            )

        if screen_width is not None:
            if self.right > screen_width:
                raise ValueError(
                    "ROI excede a largura da tela."
                )

        if screen_height is not None:
            if self.bottom > screen_height:
                raise ValueError(
                    "ROI excede a altura da tela."
                )


def _load_roi() -> ROIConfig:
    """
    Carrega exclusivamente o ROI produzido pelo calibrador.
    """

    if not CALIBRATION_FILE.exists():
        return ROIConfig(
            enabled=False,
            left=0,
            top=0,
            right=0,
            bottom=0,
        )

    try:
        with CALIBRATION_FILE.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

    except (
        OSError,
        json.JSONDecodeError,
    ) as exc:

        raise RuntimeError(
            "Não foi possível carregar "
            f"a calibração: {CALIBRATION_FILE}"
        ) from exc

    roi_data = data.get("roi")

    if not isinstance(
        roi_data,
        dict,
    ):
        raise RuntimeError(
            "Arquivo de calibração não contém "
            "um objeto 'roi' válido."
        )

    required = (
        "left",
        "top",
        "right",
        "bottom",
    )

    missing = [
        key
        for key in required
        if key not in roi_data
    ]

    if missing:
        raise RuntimeError(
            "ROI incompleto no arquivo de calibração. "
            f"Campos ausentes: {missing}"
        )

    try:
        roi = ROIConfig(
            enabled=True,
            left=int(roi_data["left"]),
            top=int(roi_data["top"]),
            right=int(roi_data["right"]),
            bottom=int(roi_data["bottom"]),
        )

    except (
        TypeError,
        ValueError,
    ) as exc:

        raise RuntimeError(
            "Valores inválidos no ROI calibrado."
        ) from exc

    roi.validate()

    return roi


# ÚNICO ROI DO SISTEMA.
ROI: Final[ROIConfig] = _load_roi()


# =============================================================================
# SCREEN CAPTURE
# =============================================================================


@dataclass(frozen=True, slots=True)
class CaptureConfig:
    """
    Configuração da captura.

    ROI não pertence a esta configuração.

    O capturador deve consumir:

        from config import ROI
    """

    backend: str = "dxgi"

    monitor_index: int = 0

    target_fps: int = 60

    max_fps: int = 120

    capture_full_screen: bool = True

    copy_frame: bool = False

    output_color_format: str = "BGR"

    max_buffer_size: int = 2

    def validate(self) -> None:

        if self.target_fps <= 0:
            raise ValueError(
                "target_fps deve ser > 0."
            )

        if self.max_fps < self.target_fps:
            raise ValueError(
                "max_fps deve ser >= target_fps."
            )

        if self.monitor_index < 0:
            raise ValueError(
                "monitor_index deve ser >= 0."
            )

        if self.output_color_format not in {
            "BGR",
            "RGB",
        }:
            raise ValueError(
                "Formato de cor inválido."
            )

        if self.max_buffer_size < 1:
            raise ValueError(
                "max_buffer_size deve ser >= 1."
            )


CAPTURE: Final[CaptureConfig] = CaptureConfig()


# =============================================================================
# YOLOP
# =============================================================================


@dataclass(frozen=True, slots=True)
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

    mean: tuple[float, float, float] = (
        0.0,
        0.0,
        0.0,
    )

    std: tuple[float, float, float] = (
        1.0,
        1.0,
        1.0,
    )

    def validate(self) -> None:

        if self.input_width <= 0:
            raise ValueError(
                "YOLOP input_width inválido."
            )

        if self.input_height <= 0:
            raise ValueError(
                "YOLOP input_height inválido."
            )

        for name, value in (
            (
                "confidence_threshold",
                self.confidence_threshold,
            ),
            (
                "lane_confidence_threshold",
                self.lane_confidence_threshold,
            ),
            (
                "segmentation_threshold",
                self.segmentation_threshold,
            ),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} deve estar entre 0 e 1."
                )

        if self.max_lanes <= 0:
            raise ValueError(
                "max_lanes deve ser > 0."
            )

        if self.minimum_lane_points < 2:
            raise ValueError(
                "minimum_lane_points deve ser >= 2."
            )

        if self.input_scale <= 0:
            raise ValueError(
                "input_scale deve ser > 0."
            )


YOLOP: Final[YOLOPConfig] = YOLOPConfig()


# =============================================================================
# LANE SELECTOR
# =============================================================================


@dataclass(frozen=True, slots=True)
class LaneSelectorConfig:

    enabled: bool = True

    minimum_confidence: float = 0.35

    maximum_lanes: int = 16

    center_reference_ratio: float = 0.50

    minimum_lane_separation: float = 80.0

    enable_multi_lane_selection: bool = True

    def validate(self) -> None:

        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ValueError(
                "LANE_SELECTOR.minimum_confidence inválido."
            )

        if self.maximum_lanes <= 0:
            raise ValueError(
                "LANE_SELECTOR.maximum_lanes deve ser > 0."
            )

        if not 0.0 <= self.center_reference_ratio <= 1.0:
            raise ValueError(
                "LANE_SELECTOR.center_reference_ratio inválido."
            )

        if self.minimum_lane_separation < 0:
            raise ValueError(
                "LANE_SELECTOR.minimum_lane_separation inválido."
            )


LANE_SELECTOR: Final[LaneSelectorConfig] = (
    LaneSelectorConfig()
)


# =============================================================================
# LANE GEOMETRY
# =============================================================================


@dataclass(frozen=True, slots=True)
class LaneGeometryConfig:
    """
    Configuração da geometria das lanes.

    IMPORTANTE:
    O modelo matemático oficial é cúbico.
    """

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

    # CONTRATO OFICIAL: LanePolynomial é cúbico.
    polynomial_degree: int = 3

    enable_outlier_rejection: bool = True

    outlier_sigma: float = 2.5

    confidence_weight_detection: float = 0.35

    confidence_weight_span: float = 0.20

    confidence_weight_width: float = 0.20

    confidence_weight_geometry: float = 0.25

    def validate(self) -> None:

        if not 0.0 <= self.min_lane_confidence <= 1.0:
            raise ValueError(
                "min_lane_confidence inválido."
            )

        if self.min_points < 2:
            raise ValueError(
                "min_points deve ser >= 2."
            )

        if self.min_observed_span <= 0:
            raise ValueError(
                "min_observed_span deve ser > 0."
            )

        if self.min_lane_width <= 0:
            raise ValueError(
                "min_lane_width deve ser > 0."
            )

        if self.max_lane_width <= self.min_lane_width:
            raise ValueError(
                "max_lane_width deve ser > min_lane_width."
            )

        if self.polynomial_degree != 3:
            raise ValueError(
                "O modelo LanePolynomial oficial é cúbico. "
                "polynomial_degree deve ser 3."
            )


LANE_GEOMETRY: Final[LaneGeometryConfig] = (
    LaneGeometryConfig()
)


# =============================================================================
# LANE MODEL
# =============================================================================


@dataclass(frozen=True, slots=True)
class LaneModelConfig:
    """
    Configuração do ajuste matemático das lanes.

    O fitting deve consumir esta configuração.
    Não devem existir DEFAULT_* paralelos em lane_model.py.
    """

    # CONTRATO OFICIAL: LanePolynomial é cúbico.
    polynomial_degree: int = 3

    minimum_points: int = 6

    minimum_y_span: float = 20.0

    max_outlier_iterations: int = 3

    outlier_threshold: float = 2.5

    minimum_confidence: float = 0.35

    projection_samples: int = 32

    enable_polynomial_smoothing: bool = True

    enable_outlier_rejection: bool = True

    def validate(self) -> None:

        if self.polynomial_degree != 3:
            raise ValueError(
                "LaneModelConfig deve utilizar "
                "polynomial_degree=3."
            )

        if self.minimum_points < 4:
            raise ValueError(
                "minimum_points deve ser >= 4."
            )

        if self.minimum_y_span <= 0:
            raise ValueError(
                "minimum_y_span deve ser > 0."
            )

        if self.max_outlier_iterations < 0:
            raise ValueError(
                "max_outlier_iterations inválido."
            )

        if self.outlier_threshold <= 0:
            raise ValueError(
                "outlier_threshold deve ser > 0."
            )

        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ValueError(
                "minimum_confidence inválido."
            )

        if self.projection_samples < 2:
            raise ValueError(
                "projection_samples deve ser >= 2."
            )


LANE_MODEL: Final[LaneModelConfig] = (
    LaneModelConfig()
)


# =============================================================================
# LANE TRACKER
# =============================================================================


@dataclass(frozen=True, slots=True)
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

    def validate(self) -> None:

        if self.max_lost_frames < 0:
            raise ValueError(
                "max_lost_frames inválido."
            )

        if self.min_stable_frames < 1:
            raise ValueError(
                "min_stable_frames deve ser >= 1."
            )

        if self.history_size < 1:
            raise ValueError(
                "history_size deve ser >= 1."
            )

        if self.max_tracks < 1:
            raise ValueError(
                "max_tracks deve ser >= 1."
            )

        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError(
                "min_confidence inválido."
            )

        if self.association_distance <= 0:
            raise ValueError(
                "association_distance deve ser > 0."
            )

        if not 0.0 <= self.confidence_decay <= 1.0:
            raise ValueError(
                "confidence_decay inválido."
            )

        if not 0.0 <= self.velocity_smoothing <= 1.0:
            raise ValueError(
                "velocity_smoothing inválido."
            )


LANE_TRACKER: Final[LaneTrackerConfig] = (
    LaneTrackerConfig()
)


# =============================================================================
# LANE PROJECTION
# =============================================================================


@dataclass(frozen=True, slots=True)
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

    confidence_decay_distance: float = 400.0

    maximum_curvature: float = 0.02

    reject_non_finite_points: bool = True

    def validate(self) -> None:

        if self.max_projection_distance <= 0:
            raise ValueError(
                "max_projection_distance deve ser > 0."
            )

        if self.minimum_points < 2:
            raise ValueError(
                "minimum_points deve ser >= 2."
            )

        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ValueError(
                "minimum_confidence inválido."
            )

        if self.samples < 2:
            raise ValueError(
                "samples deve ser >= 2."
            )

        if self.lookahead_distance <= 0:
            raise ValueError(
                "lookahead_distance deve ser > 0."
            )

        if self.near_distance < 0:
            raise ValueError(
                "near_distance inválido."
            )

        if self.far_distance <= self.near_distance:
            raise ValueError(
                "far_distance deve ser > near_distance."
            )

        if self.extrapolation_limit < 0:
            raise ValueError(
                "extrapolation_limit não pode ser negativo."
            )

        if self.confidence_decay_distance <= 0:
            raise ValueError(
                "confidence_decay_distance deve ser > 0."
            )

        if self.maximum_curvature <= 0:
            raise ValueError(
                "maximum_curvature deve ser > 0."
            )


LANE_PROJECTION: Final[
    LaneProjectionConfig
] = LaneProjectionConfig()


# =============================================================================
# LANE ASSIGNMENT
# =============================================================================


@dataclass(frozen=True, slots=True)
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


LANE_ASSIGNMENT: Final[
    LaneAssignmentConfig
] = LaneAssignmentConfig()


# =============================================================================
# ADAS STATE
# =============================================================================


@dataclass(frozen=True, slots=True)
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


ADAS: Final[ADASConfig] = ADASConfig()


# =============================================================================
# TEMPORAL FILTER
# =============================================================================


@dataclass(frozen=True, slots=True)
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


TEMPORAL_FILTER: Final[
    TemporalFilterConfig
] = TemporalFilterConfig()


# =============================================================================
# SAFETY
# =============================================================================


@dataclass(frozen=True, slots=True)
class SafetyConfig:
    """
    Safety gate central.

    Nenhum controlador deve produzir comando físico se as
    condições mínimas não forem satisfeitas.
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


SAFETY: Final[SafetyConfig] = SafetyConfig()


# =============================================================================
# LKA
# =============================================================================


@dataclass(frozen=True, slots=True)
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


LKA: Final[LKAConfig] = LKAConfig()


# =============================================================================
# G29
# =============================================================================


@dataclass(frozen=True, slots=True)
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


G29: Final[G29Config] = G29Config()


# =============================================================================
# VISUALIZATION
# =============================================================================


@dataclass(frozen=True, slots=True)
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

    window_name: str = (
        "Forza Assistents ADAS"
    )

    wait_key_ms: int = 1


VISUALIZATION: Final[
    VisualizationConfig
] = VisualizationConfig()


# =============================================================================
# PERFORMANCE
# =============================================================================


@dataclass(frozen=True, slots=True)
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


PERFORMANCE: Final[
    PerformanceConfig
] = PerformanceConfig()


# =============================================================================
# LOGGING
# =============================================================================


@dataclass(frozen=True, slots=True)
class LoggingConfig:

    enabled: bool = True

    level: str = "INFO"

    log_to_file: bool = True

    log_to_console: bool = True

    directory: Path = LOG_DIR

    filename: str = (
        "forza_assistents.log"
    )

    max_file_size_mb: int = 20

    backup_count: int = 5

    log_detection_failures: bool = True

    log_provider_changes: bool = True

    log_performance_warnings: bool = True

    log_safety_events: bool = True


LOGGING: Final[LoggingConfig] = LoggingConfig()


# =============================================================================
# HOTKEYS
# =============================================================================


@dataclass(frozen=True, slots=True)
class HotkeyConfig:

    toggle_monitor: str = "m"

    emergency_stop: str = "F9"

    toggle_overlay: str = "F10"

    exit: str = "ESC"


HOTKEYS: Final[HotkeyConfig] = HotkeyConfig()


# =============================================================================
# DEBUG
# =============================================================================


@dataclass(frozen=True, slots=True)
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


DEBUG: Final[DebugConfig] = DebugConfig()


# =============================================================================
# GLOBAL VALIDATION
# =============================================================================


def validate_config() -> None:
    """
    Valida toda a configuração estática.

    Deve ser chamado no início do main.py.
    """

    CAPTURE.validate()
    YOLOP.validate()
    LANE_SELECTOR.validate()
    LANE_GEOMETRY.validate()
    LANE_MODEL.validate()
    LANE_TRACKER.validate()
    LANE_PROJECTION.validate()

    if ROI.enabled:
        ROI.validate()

    # -------------------------------------------------------------------------
    # Contrato matemático global.
    # -------------------------------------------------------------------------

    if LANE_GEOMETRY.polynomial_degree != 3:
        raise ValueError(
            "LANE_GEOMETRY deve utilizar "
            "polynomial_degree=3."
        )

    if LANE_MODEL.polynomial_degree != 3:
        raise ValueError(
            "LANE_MODEL deve utilizar "
            "polynomial_degree=3."
        )

    if (
        LANE_GEOMETRY.polynomial_degree
        != LANE_MODEL.polynomial_degree
    ):
        raise ValueError(
            "Grau polinomial inconsistente entre "
            "LANE_GEOMETRY e LANE_MODEL."
        )

    # -------------------------------------------------------------------------
    # Projection.
    # -------------------------------------------------------------------------

    if (
        LANE_PROJECTION.extrapolation_limit
        > LANE_PROJECTION.max_projection_distance
    ):
        raise ValueError(
            "extrapolation_limit não pode exceder "
            "max_projection_distance."
        )

    # -------------------------------------------------------------------------
    # Safety.
    # -------------------------------------------------------------------------

    if not 0.0 <= SAFETY.minimum_adas_confidence <= 1.0:
        raise ValueError(
            "SAFETY.minimum_adas_confidence inválido."
        )

    if (
        SAFETY.maximum_allowed_detection_age_frames
        < 0
    ):
        raise ValueError(
            "SAFETY.maximum_allowed_detection_age_frames "
            "inválido."
        )

    if (
        SAFETY.maximum_lost_frames_before_disable
        < 0
    ):
        raise ValueError(
            "SAFETY.maximum_lost_frames_before_disable "
            "inválido."
        )

    # -------------------------------------------------------------------------
    # Performance.
    # -------------------------------------------------------------------------

    if PERFORMANCE.target_capture_fps <= 0:
        raise ValueError(
            "target_capture_fps deve ser > 0."
        )

    if PERFORMANCE.target_inference_fps <= 0:
        raise ValueError(
            "target_inference_fps deve ser > 0."
        )

    if PERFORMANCE.target_pipeline_fps <= 0:
        raise ValueError(
            "target_pipeline_fps deve ser > 0."
        )

    if PERFORMANCE.target_control_hz <= 0:
        raise ValueError(
            "target_control_hz deve ser > 0."
        )

    if PERFORMANCE.max_pipeline_latency_ms <= 0:
        raise ValueError(
            "max_pipeline_latency_ms deve ser > 0."
        )

    if PERFORMANCE.max_inference_latency_ms <= 0:
        raise ValueError(
            "max_inference_latency_ms deve ser > 0."
        )


# =============================================================================
# PUBLIC API
# =============================================================================


__all__ = [
    "PROJECT_ROOT",
    "CALIBRATION_DIR",
    "CALIBRATION_FILE",
    "WEIGHTS_DIR",
    "YOLOP_MODEL_PATH",
    "LOG_DIR",
    "SCREENSHOT_DIR",

    "RuntimeMode",
    "DEFAULT_RUNTIME_MODE",

    "ROIConfig",
    "ROI",

    "CaptureConfig",
    "CAPTURE",

    "YOLOPConfig",
    "YOLOP",

    "LaneSelectorConfig",
    "LANE_SELECTOR",

    "LaneGeometryConfig",
    "LANE_GEOMETRY",

    "LaneModelConfig",
    "LANE_MODEL",

    "LaneTrackerConfig",
    "LANE_TRACKER",

    "LaneProjectionConfig",
    "LANE_PROJECTION",

    "LaneAssignmentConfig",
    "LANE_ASSIGNMENT",

    "ADASConfig",
    "ADAS",

    "TemporalFilterConfig",
    "TEMPORAL_FILTER",

    "SafetyConfig",
    "SAFETY",

    "LKAConfig",
    "LKA",

    "G29Config",
    "G29",

    "VisualizationConfig",
    "VISUALIZATION",

    "PerformanceConfig",
    "PERFORMANCE",

    "LoggingConfig",
    "LOGGING",

    "HotkeyConfig",
    "HOTKEYS",

    "DebugConfig",
    "DEBUG",

    "validate_config",
]