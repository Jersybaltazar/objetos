from __future__ import annotations

import pytest

from fakes import (
    EndlessVideoSource,
    FakeVideoSource,
    InMemoryFrameWriter,
    StubHasher,
    canvas,
)
from inspeccion.application.curation.extraction import FrameExtractor
from inspeccion.application.curation.sampling import PerceptualDifferenceSampler
from inspeccion.domain.models import GroupId, Label


@pytest.fixture
def writer() -> InMemoryFrameWriter:
    return InMemoryFrameWriter()


def build_extractor(writer: InMemoryFrameWriter, **kwargs) -> FrameExtractor:
    return FrameExtractor(
        hasher=StubHasher(),
        writer=writer,
        sampler_factory=lambda: PerceptualDifferenceSampler(min_difference=0.02, max_interval=None),
        **kwargs,
    )


class TestFrameExtractor:
    def test_se_ejecuta_sin_camara_sin_opencv_y_sin_disco(self, writer):
        # Es la propiedad que justifica la arquitectura hexagonal.
        source = FakeVideoSource([canvas(10 + i * 30) for i in range(5)])

        resultado = build_extractor(writer).extract(
            source, group_id=GroupId("ok-001"), label=Label.NORMAL
        )

        assert len(resultado.frames) == 5

    def test_solo_persiste_los_frames_conservados(self, writer):
        # 20 frames identicos: el muestreador conserva uno solo.
        source = FakeVideoSource([canvas(120) for _ in range(20)])

        resultado = build_extractor(writer).extract(
            source, group_id=GroupId("ok-001"), label=Label.NORMAL
        )

        assert len(resultado.frames) == 1
        assert len(writer.written) == 1
        assert resultado.seen == 20

    def test_propaga_la_pieza_y_la_etiqueta_a_todos_los_frames(self, writer):
        source = FakeVideoSource([canvas(10 + i * 30) for i in range(4)])

        resultado = build_extractor(writer).extract(
            source, group_id=GroupId("def-007"), label=Label.DEFECT
        )

        assert {f.frame.group_id for f in resultado.frames} == {GroupId("def-007")}
        assert {f.frame.label for f in resultado.frames} == {Label.DEFECT}

    def test_hashea_con_la_imagen_ya_decodificada(self, writer):
        # Frames identicos -> mismo hash; distintos -> hashes distintos.
        source = FakeVideoSource([canvas(10), canvas(200), canvas(10)])
        extractor = build_extractor(writer, key_template="f/{group_id}/{index}.png")

        hashes = [
            f.phash
            for f in extractor.extract(
                source, group_id=GroupId("ok-001"), label=Label.NORMAL
            ).frames
        ]

        assert hashes[0] == hashes[2]
        assert hashes[0] != hashes[1]

    def test_timestamps_desde_la_fuente(self, writer):
        source = FakeVideoSource([canvas(10 + i * 30) for i in range(3)], fps=25.0)

        resultado = build_extractor(writer).extract(
            source, group_id=GroupId("ok-001"), label=Label.NORMAL
        )

        assert resultado.fps == 25.0
        assert resultado.frames[1].frame.timestamp_s == pytest.approx(1 / 25.0)

    def test_max_frames_acota_una_fuente_en_vivo(self, writer):
        # Sin este tope, una webcam colgaria el pipeline indefinidamente.
        extractor = build_extractor(writer, max_frames=10)

        resultado = extractor.extract(
            EndlessVideoSource(canvas(120)), group_id=GroupId("ok-001"), label=Label.NORMAL
        )

        assert resultado.seen == 10

    def test_el_estado_del_muestreador_no_cruza_piezas(self, writer):
        # Si el muestreador se reutilizara, el primer frame de la segunda pieza
        # se compararia con el ultimo de la primera y podria descartarse.
        extractor = build_extractor(writer)

        primera = extractor.extract(
            FakeVideoSource([canvas(120)]), group_id=GroupId("ok-001"), label=Label.NORMAL
        )
        segunda = extractor.extract(
            FakeVideoSource([canvas(120)]), group_id=GroupId("ok-002"), label=Label.NORMAL
        )

        assert len(primera.frames) == 1
        assert len(segunda.frames) == 1

    def test_estadisticas_de_reduccion(self, writer):
        source = FakeVideoSource([canvas(120) for _ in range(50)])

        resumen = (
            build_extractor(writer)
            .extract(source, group_id=GroupId("ok-001"), label=Label.NORMAL)
            .summary()
        )

        assert resumen["seen"] == 50
        assert resumen["kept"] == 1
        assert resumen["retained_fraction"] == pytest.approx(0.02)
