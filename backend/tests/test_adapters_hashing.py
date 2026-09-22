from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from inspeccion.adapters.hashing import HashAlgorithm, ImageHashAdapter
from inspeccion.application.curation.dedup import hamming

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("imagehash") is None,
    reason="requiere el extra 'media' (imagehash, pillow)",
)


def textura(seed: int = 0, size: int = 128) -> np.ndarray:
    """Imagen con estructura real: el hash perceptual de un lienzo plano es
    degenerado y no distinguiria nada."""
    rng = np.random.default_rng(seed)
    base = np.linspace(0, 255, size, dtype=np.float64)
    campo = np.outer(base, np.ones(size)) * 0.5 + np.outer(np.ones(size), base) * 0.5
    campo += rng.normal(0, 12, (size, size))
    grey = np.clip(campo, 0, 255).astype(np.uint8)
    return np.dstack([grey, grey, grey])


class TestImageHashAdapter:
    def test_imagenes_identicas_dan_el_mismo_hash(self):
        hasher = ImageHashAdapter()
        imagen = textura()

        assert hasher.hash(imagen) == hasher.hash(imagen.copy())

    def test_una_perturbacion_leve_da_distancia_pequena(self):
        # Es el caso de dos frames consecutivos de un video.
        hasher = ImageHashAdapter()
        original = textura(seed=1)
        desplazada = np.roll(original, shift=1, axis=0)

        assert hamming(hasher.hash(original), hasher.hash(desplazada)) <= 8

    def test_imagenes_distintas_dan_distancia_grande(self):
        hasher = ImageHashAdapter()

        distancia = hamming(hasher.hash(textura(seed=1)), hasher.hash(textura(seed=99)[::-1]))

        assert distancia > 8

    def test_el_hash_cabe_en_n_bits(self):
        hasher = ImageHashAdapter(hash_size=8)

        assert hasher.n_bits == 64
        assert 0 <= hasher.hash(textura()) < (1 << 64)

    def test_hash_size_mayor_produce_mas_bits(self):
        hasher = ImageHashAdapter(hash_size=16)

        assert hasher.n_bits == 256
        assert 0 <= hasher.hash(textura()) < (1 << 256)

    def test_acepta_escala_de_grises(self):
        hasher = ImageHashAdapter()

        assert isinstance(hasher.hash(textura()[:, :, 0]), int)

    def test_convierte_de_bgr_a_rgb(self):
        # Sin la conversion, el hash no coincidiria con el de la misma imagen
        # leida por cualquier otra herramienta, y los resultados no serian
        # reproducibles fuera de este proceso.
        import imagehash
        from PIL import Image as PILImage

        rgb = textura(seed=7)
        rgb[:, :, 0] = 255  # canal R saturado: la diferencia BGR/RGB importa
        bgr = rgb[:, :, ::-1].copy()

        esperado = int.from_bytes(
            np.packbits(
                np.asarray(imagehash.phash(PILImage.fromarray(rgb)).hash).flatten()
            ).tobytes(),
            "big",
        )

        assert ImageHashAdapter().hash(bgr) == esperado

    def test_rechaza_forma_invalida(self):
        with pytest.raises(ValueError, match="se esperaba una imagen"):
            ImageHashAdapter().hash(np.zeros((4, 4, 4), dtype=np.uint8))


class TestEstrategias:
    @pytest.mark.parametrize("algoritmo", list(HashAlgorithm))
    def test_todas_las_estrategias_producen_n_bits_comparables(self, algoritmo):
        # Comparabilidad entre estrategias: sin ella, el umbral de Hamming del
        # experimento E4 significaria cosas distintas segun el algoritmo.
        hasher = ImageHashAdapter(algorithm=algoritmo, hash_size=8)

        assert hasher.n_bits == 64
        assert 0 <= hasher.hash(textura()) < (1 << 64)

    def test_estrategias_distintas_dan_hashes_distintos(self):
        imagen = textura(seed=3)

        phash = ImageHashAdapter(algorithm=HashAlgorithm.PHASH).hash(imagen)
        dhash = ImageHashAdapter(algorithm=HashAlgorithm.DHASH).hash(imagen)

        assert phash != dhash

    def test_whash_exige_hash_size_potencia_de_dos(self):
        with pytest.raises(ValueError, match="potencia de dos"):
            ImageHashAdapter(algorithm=HashAlgorithm.WHASH, hash_size=6)

    def test_rechaza_hash_size_degenerado(self):
        with pytest.raises(ValueError, match="hash_size"):
            ImageHashAdapter(hash_size=1)
