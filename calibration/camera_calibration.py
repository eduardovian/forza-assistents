"""
calibration/camera_calibration.py

Forza Assistents
================

Ferramenta de calibração da câmera/tela.

Responsabilidades
-----------------

Este módulo é responsável por:

    1. Detectar a resolução da tela.
    2. Permitir a seleção visual do ROI.
    3. Validar o ROI selecionado.
    4. Persistir a calibração em:
           calibration/camera_calibration.json
    5. Permitir que config.py carregue essa calibração.

Arquitetura
-----------

    Camera / Screen
          │
          ▼
    CameraCalibration
          │
          ▼
    camera_calibration.json
          │
          ▼
       config.py
          │
          ▼
        ROI

PRINCÍPIO FUNDAMENTAL
---------------------

Este módulo é o DONO DA CALIBRAÇÃO.

Ele não deve:

    - definir ROI para outros módulos;
    - importar ROI para uso operacional;
    - alterar config.py;
    - executar YOLOP;
    - executar LaneGeometry;
    - executar tracking;
    - executar ADAS.

Depois da calibração, todos os outros módulos utilizam:

    from config import ROI

O arquivo JSON é a fonte persistente da calibração.
O config.py é a fonte de verdade em runtime.

Segurança
---------

Uma calibração inválida nunca deve ser salva.

A gravação é atômica para evitar corrupção do arquivo.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Final

import cv2
import numpy as np


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT: Final[Path] = (
    Path(__file__).resolve().parents[1]
)

CALIBRATION_DIR: Final[Path] = (
    PROJECT_ROOT / "calibration"
)

CALIBRATION_FILE: Final[Path] = (
    CALIBRATION_DIR / "camera_calibration.json"
)


# =============================================================================
# TYPES
# =============================================================================


@dataclass(frozen=True, slots=True)
class ScreenGeometry:
    """
    Geometria da tela utilizada durante a calibração.
    """

    width: int
    height: int

    def validate(self) -> None:

        if self.width <= 0:
            raise ValueError(
                "Largura da tela deve ser > 0."
            )

        if self.height <= 0:
            raise ValueError(
                "Altura da tela deve ser > 0."
            )


@dataclass(frozen=True, slots=True)
class CalibrationROI:
    """
    ROI produzido pelo processo de calibração.

    Coordenadas absolutas da tela:

        left
        top
        right
        bottom
    """

    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def rectangle(
        self,
    ) -> tuple[int, int, int, int]:
        return (
            self.left,
            self.top,
            self.right,
            self.bottom,
        )

    def validate(
        self,
        screen: ScreenGeometry,
    ) -> None:
        """
        Valida o ROI contra a resolução da tela.
        """

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

        if self.right > screen.width:
            raise ValueError(
                "ROI excede a largura da tela."
            )

        if self.bottom > screen.height:
            raise ValueError(
                "ROI excede a altura da tela."
            )

        if self.width < 32:
            raise ValueError(
                "ROI muito estreito."
            )

        if self.height < 32:
            raise ValueError(
                "ROI muito baixo."
            )


# =============================================================================
# CALIBRATION RESULT
# =============================================================================


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """
    Resultado completo de uma calibração.
    """

    screen: ScreenGeometry

    roi: CalibrationROI

    timestamp_utc: float

    version: int = 1

    coordinate_system: str = "screen_absolute"

    source: str = "manual_gui"

    def validate(self) -> None:

        self.screen.validate()

        self.roi.validate(
            self.screen
        )

        if self.version <= 0:
            raise ValueError(
                "Versão de calibração inválida."
            )

        if not self.coordinate_system:
            raise ValueError(
                "Sistema de coordenadas ausente."
            )


# =============================================================================
# CALIBRATION STORAGE
# =============================================================================


class CalibrationStorage:
    """
    Persistência da calibração.

    A escrita é atômica:

        temporary file
              ↓
        os.replace()
              ↓
        calibration.json
    """

    def __init__(
        self,
        path: Path = CALIBRATION_FILE,
    ) -> None:

        self.path = Path(path)

    def save(
        self,
        result: CalibrationResult,
    ) -> None:

        result.validate()

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "version": result.version,
            "timestamp_utc": result.timestamp_utc,
            "coordinate_system": (
                result.coordinate_system
            ),
            "source": result.source,
            "screen": {
                "width": result.screen.width,
                "height": result.screen.height,
            },
            "roi": {
                "left": result.roi.left,
                "top": result.roi.top,
                "right": result.roi.right,
                "bottom": result.roi.bottom,
            },
        }

        temporary_path = self.path.with_suffix(
            ".tmp"
        )

        with temporary_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                payload,
                file,
                indent=4,
            )

            file.write("\n")

            file.flush()

        temporary_path.replace(
            self.path
        )

    def load(
        self,
    ) -> CalibrationResult:

        if not self.path.exists():
            raise FileNotFoundError(
                "Calibração não encontrada: "
                f"{self.path}"
            )

        try:

            with self.path.open(
                "r",
                encoding="utf-8",
            ) as file:

                data = json.load(file)

        except json.JSONDecodeError as exc:

            raise RuntimeError(
                "Arquivo de calibração corrompido."
            ) from exc

        try:

            screen_data = data["screen"]
            roi_data = data["roi"]

            result = CalibrationResult(
                version=int(
                    data.get(
                        "version",
                        1,
                    )
                ),
                timestamp_utc=float(
                    data.get(
                        "timestamp_utc",
                        0.0,
                    )
                ),
                coordinate_system=str(
                    data.get(
                        "coordinate_system",
                        "screen_absolute",
                    )
                ),
                source=str(
                    data.get(
                        "source",
                        "unknown",
                    )
                ),
                screen=ScreenGeometry(
                    width=int(
                        screen_data["width"]
                    ),
                    height=int(
                        screen_data["height"]
                    ),
                ),
                roi=CalibrationROI(
                    left=int(
                        roi_data["left"]
                    ),
                    top=int(
                        roi_data["top"]
                    ),
                    right=int(
                        roi_data["right"]
                    ),
                    bottom=int(
                        roi_data["bottom"]
                    ),
                ),
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ) as exc:

            raise RuntimeError(
                "Estrutura de calibração inválida."
            ) from exc

        result.validate()

        return result


# =============================================================================
# CAMERA CALIBRATION
# =============================================================================


class CameraCalibration:
    """
    Interface de calibração visual.

    Permite selecionar o ROI diretamente sobre um frame
    da tela.
    """

    WINDOW_NAME: Final[str] = (
        "Forza Assistents - Calibration"
    )

    def __init__(
        self,
        storage: CalibrationStorage | None = None,
    ) -> None:

        self.storage = (
            storage
            if storage is not None
            else CalibrationStorage()
        )

    # =========================================================================
    # SCREEN CAPTURE
    # =========================================================================

    @staticmethod
    def capture_screen() -> np.ndarray:
        """
        Captura a tela atual.

        Usa DXCam quando disponível, com fallback para
        ImageGrab.

        Esta captura é utilizada SOMENTE durante a
        calibração.

        O runtime do sistema utiliza screen_capture.py.
        """

        try:

            import dxcam

            camera = dxcam.create(
                output_color="BGR"
            )

            frame = camera.grab()

            camera.stop()

            if frame is not None:

                return frame

        except Exception:
            pass

        try:

            from PIL import ImageGrab

            image = ImageGrab.grab()

            frame = np.asarray(
                image
            )

            return cv2.cvtColor(
                frame,
                cv2.COLOR_RGB2BGR,
            )

        except Exception as exc:

            raise RuntimeError(
                "Não foi possível capturar a tela "
                "para calibração."
            ) from exc

    # =========================================================================
    # SCREEN GEOMETRY
    # =========================================================================

    @staticmethod
    def detect_screen_geometry(
        frame: np.ndarray,
    ) -> ScreenGeometry:

        if frame is None:
            raise ValueError(
                "Frame de calibração é None."
            )

        if frame.ndim < 2:
            raise ValueError(
                "Frame de calibração inválido."
            )

        height, width = frame.shape[:2]

        geometry = ScreenGeometry(
            width=int(width),
            height=int(height),
        )

        geometry.validate()

        return geometry

    # =========================================================================
    # ROI SELECTION
    # =========================================================================

    def select_roi(
        self,
        frame: np.ndarray,
    ) -> CalibrationROI:
        """
        Abre uma interface OpenCV para seleção manual do ROI.

        O usuário deve selecionar:

            canto superior esquerdo
            →
            canto inferior direito
        """

        if frame is None:
            raise ValueError(
                "Frame de calibração é None."
            )

        screen = self.detect_screen_geometry(
            frame
        )

        display = frame.copy()

        cv2.namedWindow(
            self.WINDOW_NAME,
            cv2.WINDOW_NORMAL,
        )

        cv2.resizeWindow(
            self.WINDOW_NAME,
            min(screen.width, 1600),
            min(screen.height, 900),
        )

        cv2.imshow(
            self.WINDOW_NAME,
            display,
        )

        cv2.waitKey(1)

        x, y, width, height = cv2.selectROI(
            self.WINDOW_NAME,
            display,
            showCrosshair=True,
            fromCenter=False,
        )

        cv2.destroyWindow(
            self.WINDOW_NAME
        )

        if width <= 0 or height <= 0:
            raise RuntimeError(
                "Nenhum ROI válido foi selecionado."
            )

        roi = CalibrationROI(
            left=int(x),
            top=int(y),
            right=int(
                x + width
            ),
            bottom=int(
                y + height
            ),
        )

        roi.validate(
            screen
        )

        return roi

    # =========================================================================
    # SAVE
    # =========================================================================

    def save(
        self,
        roi: CalibrationROI,
        screen: ScreenGeometry,
    ) -> CalibrationResult:

        roi.validate(
            screen
        )

        result = CalibrationResult(
            screen=screen,
            roi=roi,
            timestamp_utc=time.time(),
            version=1,
            coordinate_system=(
                "screen_absolute"
            ),
            source="manual_gui",
        )

        self.storage.save(
            result
        )

        return result

    # =========================================================================
    # RUN
    # =========================================================================

    def run(
        self,
        frame: np.ndarray | None = None,
    ) -> CalibrationResult:
        """
        Executa uma sessão completa de calibração.
        """

        if frame is None:
            frame = self.capture_screen()

        screen = self.detect_screen_geometry(
            frame
        )

        roi = self.select_roi(
            frame
        )

        result = self.save(
            roi,
            screen,
        )

        return result


# =============================================================================
# VALIDATION
# =============================================================================


def validate_calibration(
    path: Path = CALIBRATION_FILE,
) -> CalibrationResult:
    """
    Valida uma calibração existente.
    """

    storage = CalibrationStorage(
        path
    )

    return storage.load()


# =============================================================================
# CLI
# =============================================================================


def main() -> int:
    """
    Executa o calibrador pela linha de comando.
    """

    print(
        "Forza Assistents"
    )

    print(
        "Camera Calibration"
    )

    print()

    try:

        calibration = CameraCalibration()

        result = calibration.run()

    except KeyboardInterrupt:

        print(
            "\nCalibração cancelada."
        )

        return 130

    except Exception as exc:

        print(
            f"\nERRO: {exc}"
        )

        return 1

    print()

    print(
        "Calibração concluída."
    )

    print(
        f"Tela: "
        f"{result.screen.width}x"
        f"{result.screen.height}"
    )

    print(
        "ROI:"
        f" ({result.roi.left}, "
        f"{result.roi.top}, "
        f"{result.roi.right}, "
        f"{result.roi.bottom})"
    )

    print(
        f"Tamanho do ROI: "
        f"{result.roi.width}x"
        f"{result.roi.height}"
    )

    print(
        f"Arquivo: "
        f"{CALIBRATION_FILE}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )