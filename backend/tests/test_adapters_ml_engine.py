from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from inspeccion.adapters.ml_engine.anomalib_engine import (
    AnomalibMLEngine,
    AnomalyModel,
    TrainingError,
)
from inspeccion.application.curation.splitting import split_by_group
from inspeccion.domain.models import FrameRef, GroupId, Label, Split

requiere_anomalib = pytest.mark.skipif(
    importlib.util.find_spec("anomalib") is None,
    reason="requiere el extra 'ml' (anomalib); solo disponible en Python 3.12",
)


@dataclass
class FakePredictionBatch:
    """Lote de predicciones con la forma que devuelve anomalib."""

    image_path: list[str]
    pred_score: np.ndarray


def frame_at(path: str) -> FrameRef:
    return FrameRef(
        path=path,
        group_id=GroupId("ok-000"),
        label=Label.NORMAL,
        source_video="ok-000.avi",
        frame_index=0,
        timestamp_s=0.0,
    )


class TestAlineacionDeScores:
    """La proteccion contra el fallo mas silencioso de toda la integracion.

    El setter de `samples` de anomalib hace `sort_values(by="image_path")`, asi
    que el dataset se recorre en orden alfabetico de ruta y no en el orden en que
    se le paso la lista. Emparejar los scores por posicion asignaria el valor de
    un frame a otro: el AUROC seguiria saliendo, con un numero plausible y
    equivocado. Estos tests no necesitan anomalib.
    """

    def test_reordena_segun_la_lista_de_entrada(self):
        # El modelo devuelve las rutas ordenadas alfabeticamente; la lista de
        # entrada va al reves.
        frames = [frame_at("z.png"), frame_at("a.png"), frame_at("m.png")]
        predicciones = [FakePredictionBatch(["a.png", "m.png", "z.png"], np.array([0.1, 0.5, 0.9]))]

        scores = AnomalibMLEngine._align(predicciones, frames)

        assert list(scores) == [0.9, 0.1, 0.5]

    def test_emparejar_por_posicion_daria_otro_resultado(self):
        # Deja constancia de que el bug es real y no teorico.
        frames = [frame_at("z.png"), frame_at("a.png")]
        predicciones = [FakePredictionBatch(["a.png", "z.png"], np.array([0.1, 0.9]))]

        correcto = AnomalibMLEngine._align(predicciones, frames)
        por_posicion = np.array([0.1, 0.9])

        assert not np.array_equal(correcto, por_posicion)

    def test_recompone_varios_lotes(self):
        frames = [frame_at(f"{i}.png") for i in range(5)]
        predicciones = [
            FakePredictionBatch(["0.png", "1.png"], np.array([0.0, 0.1])),
            FakePredictionBatch(["2.png", "3.png"], np.array([0.2, 0.3])),
            FakePredictionBatch(["4.png"], np.array([0.4])),
        ]

        assert list(AnomalibMLEngine._align(predicciones, frames)) == [0.0, 0.1, 0.2, 0.3, 0.4]

    def test_falla_si_falta_alguna_prediccion(self):
        frames = [frame_at("a.png"), frame_at("b.png")]
        predicciones = [FakePredictionBatch(["a.png"], np.array([0.1]))]

        with pytest.raises(TrainingError, match="faltan predicciones"):
            AnomalibMLEngine._align(predicciones, frames)

    def test_falla_si_el_lote_viene_descuadrado(self):
        predicciones = [FakePredictionBatch(["a.png", "b.png"], np.array([0.1]))]

        with pytest.raises(TrainingError, match="no se puede emparejar"):
            AnomalibMLEngine._align(predicciones, [frame_at("a.png")])


class TestValidaciones:
    def test_scores_rechaza_lista_vacia(self, tmp_path):
        with pytest.raises(ValueError, match="no hay frames"):
            AnomalibMLEngine().scores(tmp_path / "model.ckpt", [])

    def test_train_rechaza_un_manifiesto_sin_entrenamiento(self, tmp_path):
        from inspeccion.domain.models import SplitManifest, SplitStrategy

        vacio = SplitManifest(
            strategy=SplitStrategy.BY_GROUP, seed=0, assignments={Split.TRAIN: ()}
        )

        with pytest.raises(TrainingError, match=r"entrenamiento .* esta vacio"):
            AnomalibMLEngine().train(vacio, tmp_path)


@pytest.fixture
def dataset_en_disco(tmp_path: Path) -> list[FrameRef]:
    """Escribe PNG reales: anomalib valida que cada `image_path` exista."""
    import cv2

    rng = np.random.default_rng(0)
    frames: list[FrameRef] = []
    for label, count, prefix in ((Label.NORMAL, 12, "ok"), (Label.DEFECT, 6, "def")):
        for pieza in range(count):
            group_id = GroupId(f"{prefix}-{pieza:03d}")
            carpeta = tmp_path / group_id
            carpeta.mkdir(parents=True, exist_ok=True)
            for index in range(4):
                ruta = carpeta / f"{index:05d}.png"
                cv2.imwrite(str(ruta), rng.integers(0, 256, (64, 64, 3), dtype=np.uint8))
                frames.append(
                    FrameRef(
                        path=str(ruta),
                        group_id=group_id,
                        label=label,
                        source_video=f"{group_id}.avi",
                        frame_index=index,
                        timestamp_s=index / 30.0,
                    )
                )
    return frames


@requiere_anomalib
class TestSamplesFrame:
    def test_traduce_las_etiquetas_del_dominio(self, dataset_en_disco):
        from anomalib.data.utils import LabelName

        from inspeccion.adapters.ml_engine.manifest_dataset import samples_frame

        tabla = samples_frame(dataset_en_disco, Split.TEST)

        normales = tabla[tabla.image_path.str.contains("ok-")]
        defectos = tabla[tabla.image_path.str.contains("def-")]
        assert set(normales.label_index) == {LabelName.NORMAL}
        assert set(defectos.label_index) == {LabelName.ABNORMAL}

    def test_no_declara_mascaras(self, dataset_en_disco):
        # El dataset propio no tiene mascaras anotadas: la tarea es de
        # clasificacion, no de segmentacion (seccion 6.3 del protocolo).
        from inspeccion.adapters.ml_engine.manifest_dataset import samples_frame

        tabla = samples_frame(dataset_en_disco, Split.TRAIN)

        assert set(tabla.mask_path) == {""}

    def test_marca_el_split(self, dataset_en_disco):
        from inspeccion.adapters.ml_engine.manifest_dataset import samples_frame

        assert set(samples_frame(dataset_en_disco, Split.VAL)["split"]) == {"val"}


@requiere_anomalib
class TestNoReparticiona:
    """La invariante mas importante de la integracion con anomalib.

    Si alguno de estos tests falla tras actualizar la libreria, las metricas de
    la tesis dejan de ser validas: significa que anomalib volvio a repartir los
    datos por su cuenta y la particion por pieza se perdio.
    """

    @pytest.fixture
    def datamodule(self, dataset_en_disco):
        from inspeccion.adapters.ml_engine.manifest_dataset import build_datamodule

        manifest = split_by_group(dataset_en_disco, seed=0)
        dm = build_datamodule(manifest, train_batch_size=4, eval_batch_size=4)
        dm.setup()
        return manifest, dm

    def test_entrega_exactamente_los_ficheros_del_manifiesto(self, datamodule):
        manifest, dm = datamodule

        for split, data in (
            (Split.TRAIN, dm.train_data),
            (Split.VAL, dm.val_data),
            (Split.TEST, dm.test_data),
        ):
            esperados = {f.path for f in manifest.frames(split)}
            entregados = set(data.samples.image_path)
            assert entregados == esperados, f"anomalib altero el split {split}"

    def test_conserva_las_piezas_buenas_en_test(self, datamodule):
        # Con `TestSplitMode.NONE` anomalib descarta los normales de test y el
        # AUROC queda indefinido por tener una sola clase. Este test lo detecta.
        from anomalib.data.utils import LabelName

        _, dm = datamodule

        etiquetas = set(dm.test_data.samples.label_index)
        assert LabelName.NORMAL in etiquetas
        assert LabelName.ABNORMAL in etiquetas

    def test_el_entrenamiento_no_ve_ningun_defecto(self, datamodule):
        from anomalib.data.utils import LabelName

        _, dm = datamodule

        assert set(dm.train_data.samples.label_index) == {LabelName.NORMAL}

    def test_validacion_tiene_ambas_clases_para_calibrar_el_umbral(self, datamodule):
        from anomalib.data.utils import LabelName

        _, dm = datamodule

        assert set(dm.val_data.samples.label_index) == {LabelName.NORMAL, LabelName.ABNORMAL}

    def test_ninguna_pieza_cruza_los_splits_tras_pasar_por_anomalib(self, datamodule):
        manifest, dm = datamodule
        por_ruta = {f.path: f.group_id for f in (f for s in Split for f in manifest.frames(s))}

        grupos = {
            split: {por_ruta[p] for p in data.samples.image_path}
            for split, data in (
                (Split.TRAIN, dm.train_data),
                (Split.VAL, dm.val_data),
                (Split.TEST, dm.test_data),
            )
        }

        assert not grupos[Split.TRAIN] & grupos[Split.TEST]
        assert not grupos[Split.TRAIN] & grupos[Split.VAL]
        assert not grupos[Split.VAL] & grupos[Split.TEST]

    def test_no_se_pierde_ni_se_duplica_ningun_frame(self, datamodule):
        manifest, dm = datamodule

        total_manifiesto = sum(manifest.frame_counts().values())
        total_entregado = len(dm.train_data) + len(dm.val_data) + len(dm.test_data)

        assert total_entregado == total_manifiesto


@requiere_anomalib
@pytest.mark.slow
class TestEntrenamientoReal:
    """Recorrido completo del protocolo con un modelo de verdad.

    Los defectos sinteticos son un cuadrado oscuro, trivialmente separables: el
    objetivo no es medir la calidad del modelo sino comprobar que el flujo
    entrena, puntua, calibra y exporta sin romperse.
    """

    @pytest.fixture(scope="class")
    def entrenado(self, tmp_path_factory, request):
        from inspeccion.adapters.ml_engine import AnomalibMLEngine

        tmp = tmp_path_factory.mktemp("entrenamiento")
        frames = _dataset_con_defecto_visible(tmp)
        manifest = split_by_group(frames, seed=0)
        engine = AnomalibMLEngine(coreset_sampling_ratio=0.5, batch_size=8, seed=0)
        return engine, manifest, engine.train(manifest, tmp / "run"), tmp

    def test_produce_un_checkpoint(self, entrenado):
        _, _, checkpoint, _ = entrenado

        assert checkpoint.is_file()
        assert checkpoint.stat().st_size > 0

    def test_separa_normales_de_defectos(self, entrenado):
        engine, manifest, checkpoint, _ = entrenado
        test = manifest.frames(Split.TEST)

        scores = engine.scores(checkpoint, test)
        defectuoso = np.array([f.label is Label.DEFECT for f in test])

        assert scores[defectuoso].mean() > scores[~defectuoso].mean()

    def test_devuelve_un_score_por_frame_en_el_orden_pedido(self, entrenado):
        engine, manifest, checkpoint, _ = entrenado
        # Orden invertido a proposito: si el adaptador emparejara por posicion,
        # los scores no coincidirian con los de la lista sin invertir.
        test = list(manifest.frames(Split.TEST))
        invertido = test[::-1]

        directo = engine.scores(checkpoint, test)
        inverso = engine.scores(checkpoint, invertido)

        assert directo.size == len(test)
        assert list(inverso) == list(directo[::-1])

    def test_el_umbral_se_calibra_en_validacion_y_se_aplica_a_test(self, entrenado):
        from inspeccion.application.evaluation import calibrate_f1

        engine, manifest, checkpoint, _ = entrenado
        val, test = manifest.frames(Split.VAL), manifest.frames(Split.TEST)

        umbral = calibrate_f1(
            engine.scores(checkpoint, val),
            np.array([f.label is Label.DEFECT for f in val]),
            split=Split.VAL,
        )
        matriz = umbral.evaluate(
            engine.scores(checkpoint, test),
            np.array([f.label is Label.DEFECT for f in test]),
        )

        assert matriz.recall > 0.5

    def test_evaluate_reporta_auroc(self, entrenado):
        engine, manifest, checkpoint, _ = entrenado

        metricas = engine.evaluate(checkpoint, manifest.frames(Split.TEST))

        assert 0.0 <= metricas["image_auroc"] <= 1.0
        assert metricas["image_auroc"] > 0.5

    def test_exporta_onnx_valido(self, entrenado):
        import onnx

        engine, _, checkpoint, tmp = entrenado

        destino = engine.export_onnx(checkpoint, tmp / "model.onnx", image_size=(256, 256))
        modelo = onnx.load(str(destino))
        onnx.checker.check_model(modelo)

        assert destino.is_file()
        assert "pred_score" in {salida.name for salida in modelo.graph.output}


def _dataset_con_defecto_visible(root: Path) -> list[FrameRef]:
    import cv2

    rng = np.random.default_rng(0)
    frames: list[FrameRef] = []
    for label, count, prefix in ((Label.NORMAL, 12, "ok"), (Label.DEFECT, 6, "def")):
        for pieza in range(count):
            group_id = GroupId(f"{prefix}-{pieza:03d}")
            carpeta = root / group_id
            carpeta.mkdir(parents=True, exist_ok=True)
            for index in range(4):
                imagen = np.clip(
                    np.full((64, 64, 3), 120, np.int16) + rng.integers(-10, 10, (64, 64, 3)),
                    0,
                    255,
                ).astype(np.uint8)
                if label is Label.DEFECT:
                    imagen[24:40, 24:40] = 60
                ruta = carpeta / f"{index:05d}.png"
                cv2.imwrite(str(ruta), imagen)
                frames.append(
                    FrameRef(
                        path=str(ruta),
                        group_id=group_id,
                        label=label,
                        source_video=f"{group_id}.avi",
                        frame_index=index,
                        timestamp_s=index / 30.0,
                    )
                )
    return frames


class TestConfiguracionDelModelo:
    """Reproducir una cifra publicada exige replicar la configuracion que la produjo.

    PaDiM es muy sensible al backbone: con el `resnet18` por defecto de anomalib
    da 0.83 de AUROC en `screw`, frente a los 0.975 que el paper reporta con
    `wide_resnet50_2`. Sin poder fijarlo, E1 no seria una reproduccion.
    """

    def test_sin_configurar_no_se_pasa_nada(self):
        motor = AnomalibMLEngine()

        assert motor.backbone is None
        assert motor.layers is None
        assert motor.n_features is None

    def test_guarda_la_configuracion_declarada(self):
        motor = AnomalibMLEngine(
            model=AnomalyModel.PADIM,
            backbone="wide_resnet50_2",
            layers=("layer1", "layer2", "layer3"),
            n_features=550,
        )

        assert motor.backbone == "wide_resnet50_2"
        assert motor.layers == ("layer1", "layer2", "layer3")
        assert motor.n_features == 550


@requiere_anomalib
class TestConstruccionDelModelo:
    def test_patchcore_respeta_el_backbone(self):
        modelo = AnomalibMLEngine(backbone="resnet18")._build_model()

        assert modelo.model.backbone == "resnet18"

    def test_padim_usa_el_default_de_anomalib_si_no_se_declara(self):
        modelo = AnomalibMLEngine(model=AnomalyModel.PADIM)._build_model()

        assert modelo.model.backbone == "resnet18"

    def test_padim_replica_la_configuracion_del_paper(self):
        modelo = AnomalibMLEngine(
            model=AnomalyModel.PADIM,
            backbone="wide_resnet50_2",
            layers=("layer1", "layer2", "layer3"),
            n_features=550,
        )._build_model()

        assert modelo.model.backbone == "wide_resnet50_2"

    def test_n_features_no_se_pasa_a_patchcore(self):
        # PatchCore no acepta ese argumento; pasarlo reventaria la construccion.
        modelo = AnomalibMLEngine(model=AnomalyModel.PATCHCORE, n_features=550)._build_model()

        assert modelo is not None
