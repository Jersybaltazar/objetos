from __future__ import annotations

import numpy as np
import pytest

from inspeccion.application.curation.sampling import PerceptualDifferenceSampler


def lienzo(valor: int, size: int = 128) -> np.ndarray:
    return np.full((size, size, 3), valor, dtype=np.uint8)


class TestPerceptualDifferenceSampler:
    def test_siempre_conserva_el_primer_frame(self):
        sampler = PerceptualDifferenceSampler()

        assert sampler.should_keep(lienzo(120)) is True

    def test_escena_estatica_solo_conserva_por_cobertura_temporal(self):
        sampler = PerceptualDifferenceSampler(min_difference=0.02, max_interval=50)

        conservados = sum(sampler.should_keep(lienzo(120)) for _ in range(150))

        # 1 inicial + 1 cada 50 frames, no 150.
        assert conservados <= 4

    def test_sin_cobertura_temporal_la_escena_estatica_da_un_solo_frame(self):
        sampler = PerceptualDifferenceSampler(min_difference=0.02, max_interval=None)

        conservados = sum(sampler.should_keep(lienzo(120)) for _ in range(150))

        assert conservados == 1

    def test_escena_cambiante_conserva_mucho(self):
        sampler = PerceptualDifferenceSampler(min_difference=0.02, max_interval=None)

        conservados = sum(sampler.should_keep(lienzo(10 + i * 20)) for i in range(10))

        assert conservados == 10

    def test_min_interval_evita_rafagas(self):
        sampler = PerceptualDifferenceSampler(min_difference=0.0, min_interval=5, max_interval=None)

        conservados = sum(sampler.should_keep(lienzo(10 + i * 7)) for i in range(20))

        assert conservados <= 5

    def test_reset_olvida_la_referencia(self):
        # Obligatorio al cambiar de pieza: el estado no debe cruzar grupos.
        sampler = PerceptualDifferenceSampler(min_difference=0.5, max_interval=None)
        sampler.should_keep(lienzo(120))
        assert sampler.should_keep(lienzo(121)) is False

        sampler.reset()

        assert sampler.should_keep(lienzo(121)) is True

    def test_estadisticas(self):
        sampler = PerceptualDifferenceSampler(min_difference=0.02, max_interval=None)
        for _ in range(10):
            sampler.should_keep(lienzo(120))

        assert sampler.stats() == {"seen": 10, "kept": 1, "retained_fraction": 0.1}

    def test_acepta_escala_de_grises(self):
        sampler = PerceptualDifferenceSampler()

        assert sampler.should_keep(np.full((128, 128), 120, dtype=np.uint8)) is True

    def test_rechaza_imagen_menor_que_el_thumbnail(self):
        sampler = PerceptualDifferenceSampler(thumb_size=64)

        with pytest.raises(ValueError, match="menor que el thumbnail"):
            sampler.should_keep(lienzo(120, size=32))

    @pytest.mark.parametrize(
        ("kwargs", "mensaje"),
        [
            ({"min_difference": 1.5}, "min_difference"),
            ({"min_interval": 0}, "min_interval"),
            ({"min_interval": 10, "max_interval": 5}, "max_interval"),
            ({"thumb_size": 4}, "thumb_size"),
        ],
    )
    def test_valida_su_configuracion(self, kwargs, mensaje):
        with pytest.raises(ValueError, match=mensaje):
            PerceptualDifferenceSampler(**kwargs)
