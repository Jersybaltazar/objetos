from __future__ import annotations

import numpy as np
import pytest

from inspeccion.application.curation.quality import (
    assess_clip,
    assess_frame,
    compare_to_reference,
)


def escena(brillo: int = 128, *, ruido: float = 20.0, seed: int = 0, size: int = 96) -> np.ndarray:
    """Escena con textura: el laplaciano de un lienzo plano es cero."""
    rng = np.random.default_rng(seed)
    base = np.clip(brillo + rng.normal(0, ruido, (size, size)), 0, 255).astype(np.uint8)
    return np.dstack([base, base, base])


def desenfocada(imagen: np.ndarray) -> np.ndarray:
    """Promedio de 5x5 aproximado por desplazamientos: pierde alta frecuencia."""
    acumulado = np.zeros_like(imagen, dtype=np.float64)
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            acumulado += np.roll(np.roll(imagen, dy, axis=0), dx, axis=1)
    return (acumulado / 25).astype(np.uint8)


class TestFrame:
    def test_brillo_normalizado(self):
        assert assess_frame(np.full((10, 10, 3), 255, np.uint8)).brightness == pytest.approx(1.0)
        assert assess_frame(np.zeros((10, 10, 3), np.uint8)).brightness == pytest.approx(0.0)

    def test_detecta_recorte_en_blanco(self):
        imagen = escena(128)
        imagen[:20, :, :] = 255

        assert assess_frame(imagen).clipped_bright > 0.15

    def test_detecta_recorte_en_negro(self):
        imagen = escena(128)
        imagen[:20, :, :] = 0

        assert assess_frame(imagen).clipped_dark > 0.15

    def test_la_imagen_desenfocada_es_menos_nitida(self):
        nitida = escena(128, ruido=40)

        assert assess_frame(desenfocada(nitida)).sharpness < assess_frame(nitida).sharpness / 3

    def test_un_lienzo_plano_no_tiene_nitidez(self):
        assert assess_frame(np.full((20, 20), 90, np.uint8)).sharpness == 0.0

    def test_acepta_escala_de_grises(self):
        assert assess_frame(escena(100)[:, :, 0]).brightness == pytest.approx(100 / 255, abs=0.02)

    def test_rechaza_forma_invalida(self):
        with pytest.raises(ValueError, match="se esperaba una imagen"):
            assess_frame(np.zeros((2, 2, 2, 2), np.uint8))


class TestClip:
    def test_clip_estable_no_avisa(self):
        frames = [assess_frame(escena(128, seed=s)) for s in range(10)]

        assert assess_clip(frames).is_ok

    def test_detecta_la_autoexposicion_activa(self):
        # El driver puede aceptar la orden de fijar la exposicion sin aplicarla.
        # La oscilacion del brillo entre frames es la evidencia que queda.
        frames = [assess_frame(escena(100 + 15 * (s % 2), seed=s)) for s in range(10)]

        clip = assess_clip(frames)

        assert not clip.is_ok
        assert any("autoexposicion" in aviso for aviso in clip.warnings)

    def test_detecta_la_sobreexposicion(self):
        imagen = escena(128)
        imagen[:30, :, :] = 255

        clip = assess_clip([assess_frame(imagen)])

        assert any("sobreexpuesto" in aviso for aviso in clip.warnings)

    def test_detecta_la_subexposicion(self):
        imagen = escena(128)
        imagen[:30, :, :] = 0

        clip = assess_clip([assess_frame(imagen)])

        assert any("subexpuesto" in aviso for aviso in clip.warnings)

    def test_un_solo_frame_no_puede_mostrar_parpadeo(self):
        assert assess_clip([assess_frame(escena(128))]).brightness_cv == 0.0

    def test_rechaza_un_clip_vacio(self):
        with pytest.raises(ValueError, match="no hay frames"):
            assess_clip([])

    def test_resumen_serializable(self):
        resumen = assess_clip([assess_frame(escena(128))]).summary()

        assert set(resumen) == {
            "n_frames",
            "brightness_mean",
            "brightness_cv",
            "clipped_max",
            "sharpness_median",
            "warnings",
        }


class TestDeriva:
    def test_misma_iluminacion_no_avisa(self):
        referencia = assess_clip([assess_frame(escena(128, seed=0))])
        actual = assess_clip([assess_frame(escena(128, seed=1))])

        assert compare_to_reference(actual, referencia) == ()

    def test_detecta_un_cambio_de_iluminacion(self):
        # El protocolo exige la misma luz en todos los clips: si cambia, el
        # modelo aprenderia la iluminacion como parte de lo "normal".
        referencia = assess_clip([assess_frame(escena(128))])
        actual = assess_clip([assess_frame(escena(90))])

        avisos = compare_to_reference(actual, referencia)

        assert any("iluminacion" in aviso for aviso in avisos)

    def test_detecta_una_perdida_de_enfoque(self):
        nitida = escena(128, ruido=40)
        referencia = assess_clip([assess_frame(nitida)])
        actual = assess_clip([assess_frame(desenfocada(nitida))])

        avisos = compare_to_reference(actual, referencia)

        assert any("enfoque" in aviso for aviso in avisos)
