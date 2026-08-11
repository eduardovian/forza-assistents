"""
Forza Horizon 6 ADAS/LKA - Configuracao Centralizada
"""
import os
import json

# =============================================================================
# HARDWARE / DISPLAY
# =============================================================================
SCREEN_WIDTH = 2560
SCREEN_HEIGHT = 1600

# =============================================================================
# ROI (Region of Interest)
# =============================================================================
ROI_LEFT = 300
ROI_TOP = 700
ROI_RIGHT = 2200
ROI_BOTTOM = 1600

# =============================================================================
# UFLD - Ultra Fast Lane Detection
# =============================================================================
UFLD_BACKBONE = "18"
UFLD_CLS_DIM = (201, 18, 4)
UFLD_USE_AUX = False
UFLD_INPUT_WIDTH = 800
UFLD_INPUT_HEIGHT = 288

# Caminho para o checkpoint do modelo (relativo ao projeto)
UFLD_MODEL_PATH = os.path.join("Ultra-Fast-Lane-Detection", "culane_18.pth")

# Normalizacao ImageNet (ja validada)
UFLD_MEAN = (0.485, 0.456, 0.406)
UFLD_STD = (0.229, 0.224, 0.225)

# Threshold de confianca para considerar um ponto de lane valido
# 0.6 e o valor padrao do UFLD oficial
UFLD_CONFIDENCE_THRESHOLD = 0.45

# =============================================================================
# FILTRO TEMPORAL (EMA)
# =============================================================================
FILTER_ALPHA = 0.3

# =============================================================================
# GEOMETRIA DA FAIXA
# =============================================================================
LANE_GEOMETRY_NEAR_WEIGHT = 0.7
LANE_GEOMETRY_FAR_WEIGHT = 0.3
MIN_POINTS_PER_LANE = 5

# =============================================================================
# CAPTURA DE TELA
# =============================================================================
CAPTURE_BACKEND = "bettercam"
CAPTURE_OUTPUT_COLOR = "BGR"
CAPTURE_TARGET_FPS = 60

# =============================================================================
# OVERLAY / VISUALIZACAO
# =============================================================================
OVERLAY_FONT_SCALE = 0.6
OVERLAY_FONT_THICKNESS = 2
OVERLAY_LINE_THICKNESS = 3
OVERLAY_POINT_RADIUS = 4

# Cores BGR
COLOR_LEFT_LANE = (0, 165, 255)
COLOR_RIGHT_LANE = (255, 0, 255)
COLOR_LANE_CENTER = (0, 255, 0)
COLOR_IMAGE_CENTER = (0, 0, 255)
COLOR_ROI = (255, 255, 0)
COLOR_HEADING = (255, 255, 255)
COLOR_TEXT = (0, 255, 0)
COLOR_WARNING = (0, 0, 255)

# =============================================================================
# PERFORMANCE / THREADING
# =============================================================================
INFERENCE_TARGET_FPS = 30
VISUALIZATION_FPS = 30
MAX_FRAME_BUFFER_SIZE = 2

# =============================================================================
# MODO DE OPERACAO
# =============================================================================
MODE = "VISION_ONLY"
G29_CONTROL_ENABLED = False

# =============================================================================
# HOTKEYS
# =============================================================================
HOTKEY_TOGGLE = 0x77  # F8
HOTKEY_EXIT = 0x1B     # ESC

# =============================================================================
# LOGGING
# =============================================================================
LOG_LEVEL = "INFO"

# =============================================================================
# ARQUIVO DE CALIBRACAO
# =============================================================================
CALIBRATION_FILE = "camera_calibration.json"


def load_calibration(filepath: str = CALIBRATION_FILE) -> dict:
    """Carrega calibracao da ROI de arquivo JSON."""
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
        return data
    return {
        "left": ROI_LEFT,
        "top": ROI_TOP,
        "right": ROI_RIGHT,
        "bottom": ROI_BOTTOM
    }


def save_calibration(data: dict, filepath: str = CALIBRATION_FILE):
    """Salva calibracao da ROI em arquivo JSON."""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)