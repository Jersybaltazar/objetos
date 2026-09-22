from __future__ import annotations

import pytest

from inspeccion.application.curation.dedup import (
    HashedFrame,
    deduplicate,
    hamming,
    redundancy_profile,
)
from inspeccion.domain.models import FrameRef, GroupId, Label


def hashed(group: str, index: int, phash: int) -> HashedFrame:
    return HashedFrame(
        frame=FrameRef(
            path=f"{group}/{index}.png",
            group_id=GroupId(group),
            label=Label.NORMAL,
            source_video=f"{group}.mp4",
            frame_index=index,
            timestamp_s=index / 30.0,
        ),
        phash=phash,
    )


def con_n_bits_distintos(base: int, n: int) -> int:
    """Hash a distancia de Hamming exactamente `n` de `base`."""
    mask = (1 << n) - 1
    return base ^ mask


class TestHamming:
    def test_hash_identico_da_cero(self):
        assert hamming(0xDEADBEEF, 0xDEADBEEF) == 0

    def test_cuenta_los_bits_distintos(self):
        assert hamming(0b1010, 0b0101) == 4

    def test_coincide_con_la_construccion_de_prueba(self):
        assert hamming(0xFF00FF00, con_n_bits_distintos(0xFF00FF00, 5)) == 5


class TestDeduplicate:
    def test_colapsa_hashes_identicos(self):
        items = [hashed("p1", i, 0xABCD) for i in range(50)]

        resultado = deduplicate(items, max_distance=0)

        assert len(resultado.kept) == 1
        assert len(resultado.removed) == 49

    def test_conserva_frames_por_encima_del_umbral(self):
        base = 0x0F0F0F0F0F0F0F0F
        items = [
            hashed("p1", 0, base),
            hashed("p1", 1, con_n_bits_distintos(base, 10)),
        ]

        assert len(deduplicate(items, max_distance=4).kept) == 2

    def test_descarta_frames_por_debajo_del_umbral(self):
        base = 0x0F0F0F0F0F0F0F0F
        items = [
            hashed("p1", 0, base),
            hashed("p1", 1, con_n_bits_distintos(base, 3)),
        ]

        assert len(deduplicate(items, max_distance=4).kept) == 1

    def test_no_deduplica_entre_piezas_distintas(self):
        # Dos piezas fisicas distintas pueden verse casi iguales: eliminarlas
        # destruiria la variabilidad inter-pieza que el modelo debe aprender.
        items = [hashed("p1", 0, 0xABCD), hashed("p2", 0, 0xABCD)]

        assert len(deduplicate(items, max_distance=0).kept) == 2

    def test_within_group_false_si_compara_entre_piezas(self):
        items = [hashed("p1", 0, 0xABCD), hashed("p2", 0, 0xABCD)]

        assert len(deduplicate(items, max_distance=0, within_group=False).kept) == 1

    def test_detecta_duplicados_no_consecutivos(self):
        # Una pieza en rotacion vuelve a una orientacion ya vista: el duplicado
        # no es del frame anterior sino de uno antiguo.
        base = 0x0F0F0F0F0F0F0F0F
        items = [
            hashed("p1", 0, base),
            hashed("p1", 1, con_n_bits_distintos(base, 30)),
            hashed("p1", 2, base),  # vuelve al inicio
        ]

        resultado = deduplicate(items, max_distance=2)

        assert len(resultado.kept) == 2
        assert resultado.removed[0].frame_index == 2

    def test_fracciones_consistentes(self):
        items = [hashed("p1", i, 0xABCD) for i in range(10)]

        resultado = deduplicate(items, max_distance=0)

        assert resultado.total == 10
        assert resultado.retained_fraction == pytest.approx(0.1)
        assert resultado.removed_fraction == pytest.approx(0.9)

    def test_lista_vacia(self):
        assert deduplicate([], max_distance=4).kept == ()

    def test_rechaza_umbral_negativo(self):
        with pytest.raises(ValueError, match="no puede ser negativo"):
            deduplicate([], max_distance=-1)

    def test_ruta_vectorizada_coincide_con_la_escalar(self):
        # Por encima de 64 hashes conservados se activa la ruta numpy; ambas
        # implementaciones deben dar el mismo resultado.
        items = [hashed("p1", i, i << 20) for i in range(300)]

        assert len(deduplicate(items, max_distance=0).kept) == 300


class TestRedundancyProfile:
    def test_umbrales_mayores_conservan_menos(self):
        base = 0x0F0F0F0F0F0F0F0F
        items = [hashed("p1", i, con_n_bits_distintos(base, i)) for i in range(20)]

        perfil = redundancy_profile(items, distances=(0, 4, 8, 12))
        conservados = [fila["kept"] for fila in perfil]

        assert conservados == sorted(conservados, reverse=True)

    def test_produce_la_tabla_del_experimento_e4(self):
        items = [hashed("p1", i, 0xABCD) for i in range(10)]

        fila = redundancy_profile(items, distances=(0,))[0]

        assert set(fila) == {"max_distance", "total", "kept", "removed", "retained_fraction"}
