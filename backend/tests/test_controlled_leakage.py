from __future__ import annotations

from pathlib import Path

import pytest

from inspeccion.application.curation import (
    InsufficientDuplicatesError,
    audit_leakage,
    build_leaked_arms,
)
from inspeccion.domain.models import Label, Split


@pytest.fixture
def originales() -> dict[str, list[Path]]:
    """20 piezas con 3 casi-duplicados cada una."""
    return {f"p{i:03d}": [Path(f"p{i:03d}__{j:03d}.png") for j in range(4)] for i in range(20)}


@pytest.fixture
def defectos() -> list[Path]:
    return [Path(f"broken/{i:03d}.png") for i in range(10)]


class TestEmparejamiento:
    """Las invariantes que hacen interpretable el delta de E2.

    El diseno anterior, con dos particiones independientes, media dos cosas a la
    vez: la fuga y la diferencia de informacion entre los dos conjuntos de
    entrenamiento. Estos tests fijan que ahora solo varia la dosis.
    """

    @pytest.mark.parametrize("dosis", [0.0, 0.05, 0.1, 0.2])
    def test_el_conjunto_de_test_es_identico_en_ambos_brazos(self, originales, defectos, dosis):
        brazos = build_leaked_arms(originales, defectos, seed=0, leak_fraction=dosis)

        limpio = [f.path for f in brazos.clean.frames(Split.TEST)]
        filtrado = [f.path for f in brazos.leaked.frames(Split.TEST)]
        assert limpio == filtrado

    def test_el_conjunto_de_test_no_cambia_con_la_dosis(self, originales, defectos):
        # Es lo que elimina la mayor fuente de varianza: los AUROC de distintas
        # dosis se miden sobre exactamente las mismas imagenes.
        referencia = None
        for dosis in (0.0, 0.05, 0.1, 0.2):
            brazos = build_leaked_arms(originales, defectos, seed=0, leak_fraction=dosis)
            actual = [f.path for f in brazos.clean.frames(Split.TEST)]
            if referencia is None:
                referencia = actual
            assert actual == referencia

    @pytest.mark.parametrize("dosis", [0.0, 0.05, 0.1, 0.2])
    def test_el_entrenamiento_tiene_el_mismo_tamano(self, originales, defectos, dosis):
        brazos = build_leaked_arms(originales, defectos, seed=0, leak_fraction=dosis)

        assert len(brazos.clean.frames(Split.TRAIN)) == len(brazos.leaked.frames(Split.TRAIN))

    def test_el_brazo_limpio_no_depende_de_la_dosis(self, originales, defectos):
        # Permite entrenarlo una vez por semilla en vez de una por (semilla, dosis).
        base = build_leaked_arms(originales, defectos, seed=3, leak_fraction=0.0).clean
        otra = build_leaked_arms(originales, defectos, seed=3, leak_fraction=0.2).clean

        assert [f.path for f in base.frames(Split.TRAIN)] == [
            f.path for f in otra.frames(Split.TRAIN)
        ]


class TestControlNegativo:
    def test_a_dosis_cero_los_dos_brazos_son_el_mismo(self, originales, defectos):
        # El diseno anterior daba +3.62 pp aqui, por varianza entre particiones.
        # Ahora el delta es cero por construccion, no por suerte.
        brazos = build_leaked_arms(originales, defectos, seed=0, leak_fraction=0.0)

        assert brazos.n_leaked_frames == 0
        assert [f.path for f in brazos.clean.frames(Split.TRAIN)] == [
            f.path for f in brazos.leaked.frames(Split.TRAIN)
        ]


class TestFuga:
    def test_el_brazo_limpio_nunca_tiene_fuga(self, originales, defectos):
        for dosis in (0.0, 0.05, 0.1, 0.2):
            brazos = build_leaked_arms(originales, defectos, seed=0, leak_fraction=dosis)
            assert audit_leakage(brazos.clean).is_clean

    def test_el_brazo_contaminado_si_la_tiene(self, originales, defectos):
        brazos = build_leaked_arms(originales, defectos, seed=0, leak_fraction=0.1)

        assert audit_leakage(brazos.leaked).shared_groups_train_test > 0

    def test_la_dosis_se_traduce_en_frames(self, originales, defectos):
        brazos = build_leaked_arms(originales, defectos, seed=0, leak_fraction=0.25)

        esperados = round(brazos.n_train * 0.25)
        assert brazos.n_leaked_frames == esperados

    def test_nunca_filtra_el_ejemplar_intacto_de_test(self, originales, defectos):
        # Colar la imagen exacta de test seria duplicacion perfecta, un fenomeno
        # mas fuerte que el que produce un video.
        brazos = build_leaked_arms(originales, defectos, seed=0, leak_fraction=0.2)

        en_test = {f.path for f in brazos.leaked.frames(Split.TEST)}
        en_train = {f.path for f in brazos.leaked.frames(Split.TRAIN)}
        assert not en_test & en_train


class TestDefectos:
    def test_van_enteros_a_test(self, originales, defectos):
        brazos = build_leaked_arms(originales, defectos, seed=0, leak_fraction=0.1)

        for manifest in (brazos.clean, brazos.leaked):
            en_test = [f for f in manifest.frames(Split.TEST) if f.label is Label.DEFECT]
            assert len(en_test) == len(defectos)

    def test_nunca_entran_al_entrenamiento(self, originales, defectos):
        brazos = build_leaked_arms(originales, defectos, seed=0, leak_fraction=0.1)

        for manifest in (brazos.clean, brazos.leaked):
            assert {f.label for f in manifest.frames(Split.TRAIN)} == {Label.NORMAL}


class TestValidaciones:
    def test_avisa_si_no_hay_duplicados_suficientes(self, originales, defectos):
        with pytest.raises(InsufficientDuplicatesError, match="casi-duplicados"):
            build_leaked_arms(originales, defectos, seed=0, leak_fraction=0.9)

    @pytest.mark.parametrize("dosis", [-0.1, 1.5])
    def test_rechaza_dosis_fuera_de_rango(self, originales, defectos, dosis):
        with pytest.raises(ValueError, match="leak_fraction"):
            build_leaked_arms(originales, defectos, seed=0, leak_fraction=dosis)

    def test_rechaza_pocas_piezas(self, defectos):
        with pytest.raises(ValueError, match="piezas suficientes"):
            build_leaked_arms({"p0": [Path("a.png")]}, defectos, seed=0, leak_fraction=0.0)

    def test_es_determinista(self, originales, defectos):
        a = build_leaked_arms(originales, defectos, seed=7, leak_fraction=0.1)
        b = build_leaked_arms(originales, defectos, seed=7, leak_fraction=0.1)

        assert [f.path for f in a.leaked.frames(Split.TRAIN)] == [
            f.path for f in b.leaked.frames(Split.TRAIN)
        ]
