from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from inspeccion.adapters.storage import (
    FilesystemArtifactStore,
    FilesystemFrameWriter,
    StorageKeyError,
)


class TestFilesystemArtifactStore:
    def test_ida_y_vuelta(self, tmp_path):
        store = FilesystemArtifactStore(root=tmp_path)

        store.put("modelos/patchcore/v1.bin", b"contenido")

        assert store.get("modelos/patchcore/v1.bin") == b"contenido"
        assert store.exists("modelos/patchcore/v1.bin")

    def test_crea_los_directorios_intermedios(self, tmp_path):
        store = FilesystemArtifactStore(root=tmp_path)

        ruta = store.put("a/b/c/d.bin", b"x")

        assert (tmp_path / "a" / "b" / "c" / "d.bin").is_file()
        assert ruta.endswith("d.bin")

    def test_exists_es_falso_para_lo_no_escrito(self, tmp_path):
        assert FilesystemArtifactStore(root=tmp_path).exists("no/existe.bin") is False

    @pytest.mark.parametrize(
        "clave",
        ["../fuera.bin", "a/../../fuera.bin", "/etc/passwd", "..\\fuera.bin"],
    )
    def test_rechaza_claves_que_escapan_de_la_raiz(self, tmp_path, clave):
        # El nombre de la pieza y del proyecto vienen de la interfaz web.
        store = FilesystemArtifactStore(root=tmp_path)

        with pytest.raises(StorageKeyError, match="no permitida"):
            store.put(clave, b"x")

    def test_acepta_separadores_de_windows(self, tmp_path):
        store = FilesystemArtifactStore(root=tmp_path)

        store.put("frames\\ok-001\\00000.bin", b"x")

        assert store.exists("frames/ok-001/00000.bin")


@pytest.mark.skipif(
    importlib.util.find_spec("cv2") is None,
    reason="requiere el extra 'media' (opencv)",
)
class TestFilesystemFrameWriter:
    def test_escribe_png_sin_perdida(self, tmp_path):
        # Un artefacto de compresion con perdida alrededor de un borde es
        # indistinguible de una rayadura: el dataset quedaria contaminado.
        import cv2

        rng = np.random.default_rng(0)
        imagen = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        writer = FilesystemFrameWriter(root=tmp_path)

        ruta = writer.write(imagen, "frames/ok-001/00000.png")

        assert np.array_equal(cv2.imread(ruta, cv2.IMREAD_COLOR), imagen)

    def test_crea_los_directorios_intermedios(self, tmp_path):
        writer = FilesystemFrameWriter(root=tmp_path)

        writer.write(np.zeros((16, 16, 3), dtype=np.uint8), "frames/ok-001/00000.png")

        assert (tmp_path / "frames" / "ok-001" / "00000.png").is_file()

    def test_rechaza_claves_que_escapan_de_la_raiz(self, tmp_path):
        writer = FilesystemFrameWriter(root=tmp_path)

        with pytest.raises(StorageKeyError, match="no permitida"):
            writer.write(np.zeros((16, 16, 3), dtype=np.uint8), "../fuera.png")
