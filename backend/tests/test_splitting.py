from __future__ import annotations

import json

import pytest

from inspeccion.application.curation.splitting import (
    InsufficientGroupsError,
    audit_leakage,
    read_manifest,
    split_by_group,
    split_naive_by_frame,
    write_manifest,
)
from inspeccion.domain.models import Label, Split, SplitStrategy


class TestSplitByGroup:
    def test_ninguna_pieza_cruza_el_limite_del_split(self, frames):
        manifest = split_by_group(frames, seed=0)

        train, val, test = (manifest.groups(s) for s in (Split.TRAIN, Split.VAL, Split.TEST))
        assert not train & val
        assert not train & test
        assert not val & test

    def test_no_hay_fuga(self, frames):
        assert audit_leakage(split_by_group(frames, seed=0)).is_clean

    def test_entrenamiento_no_supervisado_no_ve_defectos(self, frames):
        manifest = split_by_group(frames, seed=0)

        assert {f.label for f in manifest.frames(Split.TRAIN)} == {Label.NORMAL}

    def test_validacion_recibe_defectos_para_calibrar_el_umbral(self, frames):
        # Sin positivos en VAL el umbral solo podria calibrarse mirando TEST,
        # que es exactamente lo que la seccion 7 del protocolo prohibe.
        manifest = split_by_group(frames, seed=0)

        assert any(f.label is Label.DEFECT for f in manifest.frames(Split.VAL))

    def test_todos_los_frames_se_asignan_exactamente_una_vez(self, frames):
        manifest = split_by_group(frames, seed=0)

        assigned = [f for s in Split for f in manifest.frames(s)]
        assert len(assigned) == len(frames)
        assert len({f.path for f in assigned}) == len(frames)

    def test_es_determinista_con_la_misma_semilla(self, frames):
        a = split_by_group(frames, seed=7)
        b = split_by_group(frames, seed=7)

        assert a.groups(Split.TRAIN) == b.groups(Split.TRAIN)

    def test_semillas_distintas_dan_particiones_distintas(self, frames):
        # Es lo que hace posible las 10 replicas de la seccion 9 del protocolo.
        particiones = {
            frozenset(split_by_group(frames, seed=s).groups(Split.TRAIN)) for s in range(10)
        }

        assert len(particiones) >= 8

    def test_proporciones_de_frames_cercanas_al_objetivo(self, frames):
        manifest = split_by_group(frames, seed=0)
        normales = sum(1 for s in Split for f in manifest.frames(s) if f.label is Label.NORMAL)
        en_train = sum(1 for f in manifest.frames(Split.TRAIN) if f.label is Label.NORMAL)

        assert 0.5 <= en_train / normales <= 0.7

    def test_falla_ruidosamente_con_pocas_piezas(self, build_frames):
        # El protocolo de captura v1 ("minimo 3 videos") producia este caso.
        pocas = build_frames(n_normal_groups=2, n_defect_groups=1, frames_per_group=10)

        with pytest.raises(InsufficientGroupsError, match="al menos 3 piezas"):
            split_by_group(pocas)


class TestSplitNaiveByFrame:
    def test_produce_fuga_masiva(self, frames):
        # Este test NO documenta un defecto: documenta el fenomeno bajo estudio.
        # Es la evidencia de que el baseline del experimento E3 esta contaminado.
        report = audit_leakage(split_naive_by_frame(frames, seed=0))

        assert report.contaminated_test_fraction > 0.9
        assert report.shared_groups_train_test > 0

    def test_tampoco_mete_defectos_en_entrenamiento(self, frames):
        # La unica diferencia con split_by_group debe ser la unidad de reparto,
        # para que el contraste del experimento quede aislado.
        manifest = split_naive_by_frame(frames, seed=0)

        assert {f.label for f in manifest.frames(Split.TRAIN)} == {Label.NORMAL}

    def test_asigna_todos_los_frames(self, frames):
        manifest = split_naive_by_frame(frames, seed=0)

        assert sum(manifest.frame_counts().values()) == len(frames)


class TestContraste:
    def test_el_split_por_grupo_usa_menos_grupos_en_train(self, frames):
        # Dato que el reporte debe explicitar: la estrategia ingenua ve casi todas
        # las piezas en entrenamiento, no solo mas frames. Es la objeción de
        # tamaño de muestra de la seccion 12 del protocolo, hecha auditable.
        ingenuo = split_naive_by_frame(frames, seed=0)
        por_grupo = split_by_group(frames, seed=0)

        assert ingenuo.group_counts()[Split.TRAIN] > por_grupo.group_counts()[Split.TRAIN]


class TestManifiesto:
    def test_ida_y_vuelta_preserva_la_particion(self, frames, tmp_path):
        original = split_by_group(frames, seed=3)

        recuperado = read_manifest(write_manifest(original, tmp_path / "split.json"))

        assert recuperado.strategy is SplitStrategy.BY_GROUP
        assert recuperado.seed == 3
        assert recuperado.frame_counts() == original.frame_counts()
        assert recuperado.groups(Split.TRAIN) == original.groups(Split.TRAIN)

    def test_archiva_el_diagnostico_de_fuga(self, frames, tmp_path):
        destino = write_manifest(split_naive_by_frame(frames, seed=0), tmp_path / "s.json")
        payload = json.loads(destino.read_text(encoding="utf-8"))

        assert payload["leakage"]["contaminated_test_fraction"] > 0.9


class TestDefectosIntegrosATest:
    """Configuracion que exige el experimento E2.

    E2 no calibra ningun umbral —solo reporta AUROC, que es independiente del
    umbral— asi que VAL no necesita positivos y los defectos van enteros a TEST.
    """

    def test_por_grupo_manda_todos_los_defectos_a_test(self, frames):
        manifest = split_by_group(frames, seed=0, defect_val_fraction=0.0)

        assert not any(f.label is Label.DEFECT for f in manifest.frames(Split.VAL))
        defectos_en_test = sum(1 for f in manifest.frames(Split.TEST) if f.label is Label.DEFECT)
        assert defectos_en_test == sum(1 for f in frames if f.label is Label.DEFECT)

    def test_ingenuo_manda_todos_los_defectos_a_test(self, frames):
        manifest = split_naive_by_frame(frames, seed=0, defect_val_fraction=0.0)

        assert not any(f.label is Label.DEFECT for f in manifest.frames(Split.VAL))

    def test_validacion_sigue_teniendo_normales(self, frames):
        # VAL no puede quedar vacio: el trainer necesita un dataloader valido.
        manifest = split_by_group(frames, seed=0, defect_val_fraction=0.0)

        assert manifest.frames(Split.VAL)
        assert {f.label for f in manifest.frames(Split.VAL)} == {Label.NORMAL}

    def test_sigue_sin_haber_fuga(self, frames):
        assert audit_leakage(split_by_group(frames, seed=0, defect_val_fraction=0.0)).is_clean

    @pytest.mark.parametrize("fraccion", [-0.1, 1.5])
    def test_rechaza_fracciones_invalidas(self, frames, fraccion):
        with pytest.raises(ValueError, match="defect_val_fraction"):
            split_naive_by_frame(frames, defect_val_fraction=fraccion)
