from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from inspeccion.adapters.hashing import HashAlgorithm
from inspeccion.config import Container, Settings
from inspeccion.domain.models import SplitStrategy
from inspeccion.domain.ports import ArtifactStore, FrameWriter, PerceptualHasher

requiere_media = pytest.mark.skipif(
    importlib.util.find_spec("imagehash") is None,
    reason="requiere el extra 'media'",
)


class TestSettings:
    def test_los_defaults_son_los_del_protocolo(self):
        # Por defecto se particiona por pieza. La particion ingenua hay que
        # pedirla explicitamente: es el baseline del experimento, no una opcion.
        settings = Settings()

        assert settings.split_strategy is SplitStrategy.BY_GROUP
        assert settings.hash_algorithm is HashAlgorithm.PHASH
        assert settings.hash_size == 8
        assert settings.dedup_max_distance == 4

    def test_lee_del_entorno(self):
        settings = Settings.from_env(
            {
                "INSPECCION_STORAGE_ROOT": "/datos/artefactos",
                "INSPECCION_HASH_ALGORITHM": "dhash",
                "INSPECCION_DEDUP_MAX_DISTANCE": "8",
                "INSPECCION_SPLIT_STRATEGY": "naive_by_frame",
                "INSPECCION_SEED": "42",
            }
        )

        assert settings.storage_root == Path("/datos/artefactos")
        assert settings.hash_algorithm is HashAlgorithm.DHASH
        assert settings.dedup_max_distance == 8
        assert settings.split_strategy is SplitStrategy.NAIVE_BY_FRAME
        assert settings.seed == 42

    def test_entorno_vacio_deja_los_defaults(self):
        assert Settings.from_env({}) == Settings()

    def test_sampler_max_interval_admite_desactivarse(self):
        settings = Settings.from_env({"INSPECCION_SAMPLER_MAX_INTERVAL": "none"})

        assert settings.sampler_max_interval is None

    def test_rechaza_un_umbral_que_borraria_todo(self):
        # Una distancia mayor que el numero de bits del hash colapsa el dataset
        # entero a un frame por pieza sin que nadie lo note.
        with pytest.raises(ValueError, match="excede los 64 bits"):
            Settings(hash_size=8, dedup_max_distance=65)

    @pytest.mark.parametrize(
        ("kwargs", "mensaje"),
        [
            ({"hash_size": 1}, "hash_size"),
            ({"dedup_max_distance": -1}, "dedup_max_distance"),
            ({"live_max_frames": 0}, "live_max_frames"),
        ],
    )
    def test_valida_su_configuracion(self, kwargs, mensaje):
        with pytest.raises(ValueError, match=mensaje):
            Settings(**kwargs)

    def test_error_claro_ante_un_entero_mal_escrito(self):
        with pytest.raises(ValueError, match="INSPECCION_SEED no es un entero"):
            Settings.from_env({"INSPECCION_SEED": "cuarenta"})


class TestContainer:
    def test_puede_importarse_y_construirse_sin_el_extra_media(self, tmp_path):
        # Los adaptadores importan cv2/imagehash dentro de sus funciones, no en
        # el encabezado: el contenedor se construye siempre y solo falla al pedir
        # el adaptador que de verdad necesita la libreria.
        container = Container(Settings(storage_root=tmp_path))

        assert container.settings.storage_root == tmp_path

    def test_el_almacen_satisface_su_puerto(self, tmp_path):
        container = Container(Settings(storage_root=tmp_path))

        assert isinstance(container.artifact_store, ArtifactStore)
        assert isinstance(container.frame_writer, FrameWriter)

    def test_el_muestreador_es_nuevo_en_cada_llamada(self, tmp_path):
        # Si se reutilizara, el estado cruzaria de una pieza a la siguiente.
        container = Container(Settings(storage_root=tmp_path))

        assert container.sampler_factory() is not container.sampler_factory()

    def test_propaga_la_configuracion_al_muestreador(self, tmp_path):
        container = Container(Settings(storage_root=tmp_path, sampler_min_difference=0.15))

        assert container.sampler_factory().min_difference == 0.15

    def test_propaga_el_umbral_al_pipeline(self, tmp_path):
        container = Container(Settings(storage_root=tmp_path, dedup_max_distance=10))

        assert container.curation_pipeline.dedup_max_distance == 10

    @requiere_media
    def test_el_hasher_satisface_su_puerto(self, tmp_path):
        container = Container(Settings(storage_root=tmp_path))

        assert isinstance(container.hasher, PerceptualHasher)
        assert container.hasher.n_bits == 64

    @requiere_media
    def test_cambiar_de_algoritmo_solo_toca_la_configuracion(self, tmp_path):
        # Criterio de exito del prototipo: intercambiar un adaptador sin tocar
        # ningun caso de uso.
        por_defecto = Container(Settings(storage_root=tmp_path)).hasher
        alterno = Container(
            Settings(storage_root=tmp_path, hash_algorithm=HashAlgorithm.DHASH)
        ).hasher

        assert type(por_defecto) is type(alterno)
        assert por_defecto != alterno

    @requiere_media
    def test_arma_el_extractor_completo(self, tmp_path):
        container = Container(Settings(storage_root=tmp_path, live_max_frames=120))

        extractor = container.frame_extractor

        assert extractor.max_frames == 120
        assert extractor.hasher is container.hasher
        assert extractor.writer is container.frame_writer


class TestMotorML:
    def test_defaults_del_motor(self):
        from inspeccion.adapters.ml_engine import AnomalyModel

        settings = Settings()

        assert settings.anomaly_model is AnomalyModel.PATCHCORE
        assert settings.coreset_sampling_ratio == 0.1

    def test_el_motor_satisface_su_puerto(self, tmp_path):
        from inspeccion.domain.ports import MLEngine

        assert isinstance(Container(Settings(storage_root=tmp_path)).ml_engine, MLEngine)

    def test_intercambiar_patchcore_por_padim_es_solo_configuracion(self, tmp_path):
        # Criterio de exito del prototipo: cambiar el adaptador sin tocar ningun
        # caso de uso. Solo cambia una linea de Settings.
        from inspeccion.adapters.ml_engine import AnomalyModel

        patchcore = Container(Settings(storage_root=tmp_path)).ml_engine
        padim = Container(
            Settings(storage_root=tmp_path, anomaly_model=AnomalyModel.PADIM)
        ).ml_engine

        assert type(patchcore) is type(padim)
        assert patchcore.model is AnomalyModel.PATCHCORE
        assert padim.model is AnomalyModel.PADIM

    def test_lee_el_motor_del_entorno(self):
        from inspeccion.adapters.ml_engine import AnomalyModel

        settings = Settings.from_env(
            {"INSPECCION_ANOMALY_MODEL": "padim", "INSPECCION_CORESET_SAMPLING_RATIO": "0.05"}
        )

        assert settings.anomaly_model is AnomalyModel.PADIM
        assert settings.coreset_sampling_ratio == 0.05

    @pytest.mark.parametrize(
        ("kwargs", "mensaje"),
        [
            ({"coreset_sampling_ratio": 0.0}, "coreset_sampling_ratio"),
            ({"coreset_sampling_ratio": 1.5}, "coreset_sampling_ratio"),
            ({"batch_size": 0}, "batch_size"),
        ],
    )
    def test_valida_la_configuracion_del_motor(self, kwargs, mensaje):
        with pytest.raises(ValueError, match=mensaje):
            Settings(**kwargs)
