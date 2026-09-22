from __future__ import annotations

import random

import pytest

from inspeccion.application.curation.dedup import HashedFrame
from inspeccion.application.curation.pipeline import CurationPipeline
from inspeccion.domain.models import FrameRef, GroupId, Label, Split, SplitStrategy


def hashed_dataset(
    *,
    n_normal: int = 30,
    n_defect: int = 15,
    orientaciones: int = 4,
    repeticiones: int = 10,
) -> list[HashedFrame]:
    """Piezas con `orientaciones` vistas distintas, cada una repetida.

    Reproduce la estructura real de un video: rafagas de frames casi identicos
    separadas por transiciones. Solo `orientaciones` frames por pieza aportan
    informacion; el resto es redundancia.
    """
    items: list[HashedFrame] = []
    for label, count, prefix in ((Label.NORMAL, n_normal, "ok"), (Label.DEFECT, n_defect, "def")):
        for pieza in range(count):
            group_id = GroupId(f"{prefix}-{pieza:03d}")
            index = 0
            for orientacion in range(orientaciones):
                # Hashes muy separados entre orientaciones, identicos dentro.
                # `random.Random(str)` siembra por SHA-512, no por el hash de
                # Python, que esta aleatorizado por proceso: usarlo aqui haria
                # que el test pasara o fallara segun PYTHONHASHSEED.
                phash = random.Random(f"{prefix}-{pieza}-{orientacion}").getrandbits(64)
                for _ in range(repeticiones):
                    items.append(
                        HashedFrame(
                            frame=FrameRef(
                                path=f"frames/{group_id}/{index:05d}.png",
                                group_id=group_id,
                                label=label,
                                source_video=f"{group_id}.mp4",
                                frame_index=index,
                                timestamp_s=index / 30.0,
                            ),
                            phash=phash,
                        )
                    )
                    index += 1
    return items


@pytest.fixture
def dataset() -> list[HashedFrame]:
    return hashed_dataset()


class TestOrdenDeOperaciones:
    def test_deduplica_despues_de_particionar_no_antes(self, dataset):
        # Si la deduplicacion ocurriera antes del split, los frames de test que
        # duplican a los de train habrian desaparecido y la fuga se veria baja
        # sin haberse corregido. Con el orden correcto, la particion ingenua
        # sigue mostrando contaminacion total pese a la deduplicacion.
        pipeline = CurationPipeline(dedup_max_distance=4)

        reporte = pipeline.run(dataset, strategy=SplitStrategy.NAIVE_BY_FRAME, seed=0)

        assert reporte.leakage.contaminated_test_fraction == pytest.approx(1.0)

    def test_la_particion_por_pieza_queda_limpia(self, dataset):
        reporte = CurationPipeline().run(dataset, strategy=SplitStrategy.BY_GROUP, seed=0)

        assert reporte.leakage.is_clean
        assert reporte.leakage.contaminated_test_fraction == 0.0

    def test_la_deduplicacion_no_mezcla_splits(self, dataset):
        # Cada split se deduplica por separado: un frame de test nunca puede ser
        # eliminado por parecerse a uno de train.
        reporte = CurationPipeline(dedup_max_distance=4).run(dataset, seed=0)

        for split in Split:
            grupos = reporte.manifest.groups(split)
            assert grupos, f"{split} quedo vacio tras deduplicar"


class TestDeduplicacion:
    def test_colapsa_la_redundancia_intra_pieza(self, dataset):
        # 4 orientaciones x 10 repeticiones -> deben quedar 4 por pieza.
        reporte = CurationPipeline(dedup_max_distance=4).run(dataset, seed=0)

        antes = sum(reporte.frames_before_dedup.values())
        despues = sum(reporte.frames_after_dedup.values())
        assert despues == pytest.approx(antes / 10, rel=0.05)

    def test_umbral_cero_conserva_las_orientaciones_distintas(self, dataset):
        reporte = CurationPipeline(dedup_max_distance=0).run(dataset, seed=0)

        por_pieza = len(reporte.manifest.frames(Split.TRAIN)) / len(
            reporte.manifest.groups(Split.TRAIN)
        )
        assert por_pieza == pytest.approx(4.0)

    def test_puede_restringirse_a_train(self, dataset):
        # Aislar el efecto de la deduplicacion sobre el entrenamiento dejando el
        # conjunto de evaluacion intacto.
        pipeline = CurationPipeline(dedup_max_distance=4, dedup_splits=frozenset({Split.TRAIN}))

        reporte = pipeline.run(dataset, seed=0)

        assert reporte.dedup[Split.TRAIN]["removed"] > 0
        assert reporte.dedup[Split.TEST]["removed"] == 0
        assert reporte.frames_after_dedup[Split.TEST] == reporte.frames_before_dedup[Split.TEST]

    def test_registra_el_umbral_en_el_manifiesto(self, dataset):
        reporte = CurationPipeline(dedup_max_distance=6).run(dataset, seed=0)

        assert reporte.manifest.metadata["dedup_max_distance"] == "6"
        assert "frames_before_dedup" in reporte.manifest.metadata


class TestIntercambioDeEstrategia:
    def test_el_mismo_pipeline_ejecuta_ambas_estrategias(self, dataset):
        # Criterio de exito del prototipo: intercambiar una implementacion tras
        # el puerto sin tocar el caso de uso. Mismo objeto, misma semilla, mismos
        # datos; solo cambia la estrategia inyectada.
        pipeline = CurationPipeline(dedup_max_distance=4)

        ingenuo = pipeline.run(dataset, strategy=SplitStrategy.NAIVE_BY_FRAME, seed=0)
        por_pieza = pipeline.run(dataset, strategy=SplitStrategy.BY_GROUP, seed=0)

        assert ingenuo.leakage.contaminated_test_fraction > 0.9
        assert por_pieza.leakage.contaminated_test_fraction == 0.0
        assert (
            ingenuo.manifest.group_counts()[Split.TRAIN]
            > (por_pieza.manifest.group_counts()[Split.TRAIN])
        )


class TestReporte:
    def test_el_resumen_lleva_lo_que_el_experimento_necesita(self, dataset):
        resumen = CurationPipeline().run(dataset, seed=3).summary()

        assert set(resumen) == {
            "strategy",
            "seed",
            "frames_before_dedup",
            "frames_after_dedup",
            "group_counts",
            "contaminated_test_fraction",
            "dedup",
        }
        assert resumen["seed"] == 3

    def test_rechaza_una_entrada_vacia(self):
        with pytest.raises(ValueError, match="no hay frames"):
            CurationPipeline().run([])
