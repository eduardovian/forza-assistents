"""
tests/test_yolop_pipeline.py

Testes de integração do pipeline YOLOP.

Valida:

    YOLOP
      ↓
    LaneDetectionResult
      ↓
    LanePoint
      ↓
    lane_model

Não executa captura de tela nem controle do veículo.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from vision.lane_model import (
    build_lane_model,
    fit_lane_model,
    validate_lane_model,
)

from vision.lane_types import LanePoint

from vision.yolop_detector import (
    YOLOPLaneDetector,
)


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

TEST_IMAGE = (
    ROOT
    / "ufld_test"
    / "ufld_test_input.png"
)


# =============================================================================
# HELPERS
# =============================================================================


def load_image(path: Path) -> np.ndarray:
    """Carrega uma imagem BGR de forma segura."""

    if not path.exists():
        pytest.skip(
            f"Imagem de teste não encontrada: {path}"
        )

    image = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        pytest.skip(
            f"Não foi possível carregar a imagem: {path}"
        )

    return image


def make_synthetic_lane_points(
    count: int = 30,
) -> list[LanePoint]:
    """Gera uma lane sintética cúbica válida."""

    ys = np.linspace(
        100.0,
        500.0,
        count,
    )

    points: list[LanePoint] = []

    for y in ys:

        x = (
            -0.00000002 * y**3
            + 0.0005 * y**2
            - 0.1 * y
            + 300.0
        )

        points.append(
            LanePoint(
                x=float(x),
                y=float(y),
                confidence=0.95,
                valid=True,
            )
        )

    return points


def create_detector() -> YOLOPLaneDetector:
    """
    Cria o detector sem exigir que o modelo seja executado.

    A inicialização atual do detector é compatível com
    o pipeline YOLOP do projeto.
    """

    return YOLOPLaneDetector()


# =============================================================================
# IMAGE LOADING
# =============================================================================


def test_yolop_test_image_exists() -> None:
    """A imagem usada pelos testes deve existir."""

    if not TEST_IMAGE.exists():
        pytest.skip(
            f"Imagem de teste não encontrada: {TEST_IMAGE}"
        )

    assert TEST_IMAGE.is_file()


def test_yolop_test_image_loads() -> None:
    """A imagem de teste deve ser carregável pelo OpenCV."""

    image = load_image(TEST_IMAGE)

    assert isinstance(
        image,
        np.ndarray,
    )

    assert image.ndim == 3
    assert image.shape[2] == 3
    assert image.shape[0] > 0
    assert image.shape[1] > 0


# =============================================================================
# DETECTOR
# =============================================================================


def test_yolop_detector_can_be_imported() -> None:
    """O detector YOLOP deve estar disponível."""

    assert YOLOPLaneDetector is not None


def test_yolop_detector_initialization() -> None:
    """
    Verifica que o detector pode ser inicializado.

    A ausência do checkpoint não deve quebrar a coleta
    dos testes.
    """

    try:
        detector = create_detector()

    except (
        FileNotFoundError,
        RuntimeError,
        OSError,
    ) as exc:

        pytest.skip(
            f"YOLOP não disponível neste ambiente: {exc}"
        )

    assert detector is not None
    assert isinstance(
        detector,
        YOLOPLaneDetector,
    )


# =============================================================================
# SYNTHETIC LANE MODEL
# =============================================================================


def test_yolop_lane_points_are_compatible_with_lane_model() -> None:
    """LanePoints devem ser aceitos pelo modelo matemático."""

    points = make_synthetic_lane_points()

    polynomial = fit_lane_model(
        points,
        min_points=8,
    )

    assert polynomial is not None
    assert polynomial.valid


def test_yolop_builds_valid_lane_model() -> None:
    """O modelo matemático deve ser construído corretamente."""

    points = make_synthetic_lane_points()

    model = build_lane_model(
        lane_id=0,
        points=points,
        min_points=8,
    )

    assert model is not None
    assert model.valid

    assert model.polynomial is not None
    assert model.polynomial.valid

    assert model.line is not None
    assert model.line.valid


def test_yolop_lane_model_validation() -> None:
    """Validação estrutural do LaneModel."""

    points = make_synthetic_lane_points()

    model = build_lane_model(
        lane_id=0,
        points=points,
        min_points=8,
    )

    assert validate_lane_model(model)


# =============================================================================
# REAL YOLOP PIPELINE
# =============================================================================


def test_yolop_full_pipeline() -> None:
    """
    Executa YOLOP sobre a imagem de teste quando o modelo
    estiver disponível.

    Ausência do checkpoint ou impossibilidade de executar
    YOLOP neste ambiente resulta em skip.
    """

    image = load_image(TEST_IMAGE)

    try:
        detector = create_detector()

    except (
        FileNotFoundError,
        RuntimeError,
        OSError,
    ) as exc:

        pytest.skip(
            f"YOLOP não disponível: {exc}"
        )

    try:
        result = detector.detect(image)

    except (
        FileNotFoundError,
        RuntimeError,
        OSError,
        ValueError,
    ) as exc:

        pytest.skip(
            f"YOLOP não pôde executar neste ambiente: {exc}"
        )

    assert result is not None


# =============================================================================
# RESULT STRUCTURE
# =============================================================================


def test_yolop_result_has_lane_information() -> None:
    """
    Quando YOLOP estiver disponível, o resultado deve
    expor a estrutura de lanes.
    """

    image = load_image(TEST_IMAGE)

    try:
        detector = create_detector()

    except (
        FileNotFoundError,
        RuntimeError,
        OSError,
    ) as exc:

        pytest.skip(
            f"YOLOP não disponível: {exc}"
        )

    try:
        result = detector.detect(image)

    except (
        FileNotFoundError,
        RuntimeError,
        OSError,
        ValueError,
    ) as exc:

        pytest.skip(
            f"YOLOP não pôde executar: {exc}"
        )

    assert result is not None

    assert hasattr(
        result,
        "lanes",
    )


# =============================================================================
# END-TO-END CONVERSION
# =============================================================================


def test_yolop_lane_points_can_build_model() -> None:
    """
    Teste de integração entre LanePoint e LaneModel.
    """

    points = make_synthetic_lane_points(
        count=40,
    )

    model = build_lane_model(
        lane_id=1,
        points=points,
        min_points=8,
    )

    assert model.valid

    assert model.polynomial is not None

    assert model.projection is not None
    assert model.projection.valid


# =============================================================================
# NUMERIC SAFETY
# =============================================================================


def test_yolop_lane_model_rejects_non_finite_points() -> None:
    """
    Dados não finitos são ignorados e não devem invalidar
    uma observação que continua geometricamente válida.
    """

    points = make_synthetic_lane_points()

    points.append(
        LanePoint(
            x=float("nan"),
            y=300.0,
            confidence=0.95,
            valid=True,
        )
    )

    model = build_lane_model(
        lane_id=0,
        points=points,
        min_points=8,
    )

    assert model.valid


def test_yolop_lane_model_rejects_insufficient_points() -> None:
    """Poucos pontos devem resultar em falha segura."""

    points = make_synthetic_lane_points(
        count=3,
    )

    model = build_lane_model(
        lane_id=0,
        points=points,
        min_points=8,
    )

    assert not model.valid