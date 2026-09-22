from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from inspeccion.adapters.video_source import CameraControls, OpenCvVideoSource, VideoSourceError

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("cv2") is None,
    reason="requiere el extra 'media' (opencv)",
)


@pytest.fixture
def video(tmp_path: Path) -> Path:
    """Genera un AVI/MJPG corto. Es el formato con soporte mas estable en las
    ruedas de opencv-python-headless en los tres sistemas operativos."""
    import cv2

    destino = tmp_path / "pieza.avi"
    writer = cv2.VideoWriter(str(destino), cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (64, 48))
    if not writer.isOpened():
        pytest.skip("este build de OpenCV no puede escribir AVI/MJPG")

    for i in range(10):
        writer.write(np.full((48, 64, 3), 20 + i * 20, dtype=np.uint8))
    writer.release()

    if not destino.is_file() or destino.stat().st_size == 0:
        pytest.skip("este build de OpenCV no produjo un video legible")
    return destino


class TestFuenteDeArchivo:
    def test_lee_todos_los_frames(self, video):
        with OpenCvVideoSource.from_file(video) as source:
            frames = list(source.frames())

        assert len(frames) == 10

    def test_los_indices_son_consecutivos_desde_cero(self, video):
        with OpenCvVideoSource.from_file(video) as source:
            indices = [index for index, _, _ in source.frames()]

        assert indices == list(range(10))

    def test_los_timestamps_crecen(self, video):
        with OpenCvVideoSource.from_file(video) as source:
            tiempos = [ts for _, ts, _ in source.frames()]

        assert tiempos[0] == 0.0
        assert tiempos == sorted(tiempos)

    def test_reporta_los_fps_del_contenedor(self, video):
        with OpenCvVideoSource.from_file(video) as source:
            assert source.fps == pytest.approx(25.0)
            assert source.fps_is_measured is True

    def test_entrega_imagenes_bgr_uint8(self, video):
        with OpenCvVideoSource.from_file(video) as source:
            _, _, imagen = next(iter(source.frames()))

        assert imagen.dtype == np.uint8
        assert imagen.shape == (48, 64, 3)

    def test_no_aplica_controles_de_camara_a_un_archivo(self, video):
        with OpenCvVideoSource.from_file(video) as source:
            assert source.applied_controls == {}

    def test_el_gestor_de_contexto_libera_el_dispositivo(self, video):
        source = OpenCvVideoSource.from_file(video)

        with source:
            pass

        source.close()  # idempotente: no debe fallar al cerrar dos veces

    def test_falla_si_el_archivo_no_existe(self, tmp_path):
        with pytest.raises(VideoSourceError, match="no existe"):
            OpenCvVideoSource.from_file(tmp_path / "inexistente.avi")


class TestValidaciones:
    def test_rtsp_exige_el_esquema_correcto(self):
        with pytest.raises(ValueError, match="rtsp://"):
            OpenCvVideoSource.from_rtsp("http://camara.local/stream")

    def test_fallback_fps_debe_ser_positivo(self, video):
        with pytest.raises(ValueError, match="fallback_fps"):
            OpenCvVideoSource(str(video), fallback_fps=0)

    def test_fuente_inabrible(self, tmp_path):
        roto = tmp_path / "roto.avi"
        roto.write_bytes(b"esto no es un video")

        with pytest.raises(VideoSourceError, match="no se pudo abrir"):
            OpenCvVideoSource(str(roto))


class TestControlesDeCamara:
    def test_los_valores_por_defecto_desactivan_los_tres_automatismos(self):
        # El protocolo de captura prohibe autofoco, autoexposicion y balance de
        # blancos automatico: una pieza normal con exposicion variable produce
        # frames que el modelo no puede distinguir de un defecto.
        controls = CameraControls()

        assert controls.autofocus is False
        assert controls.auto_exposure is False
        assert controls.auto_white_balance is False

    def test_reporta_que_controles_acepto_el_driver(self, video):
        # `VideoCapture.set()` falla en silencio segun backend y sistema
        # operativo. Sobre un archivo no se aplica ninguno, pero la clave debe
        # existir para que la captura quede documentada.
        source = OpenCvVideoSource(str(video), controls=CameraControls())

        try:
            assert set(source.applied_controls) >= {
                "autofocus",
                "auto_exposure",
                "auto_white_balance",
            }
        finally:
            source.close()
