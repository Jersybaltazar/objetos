"""Raiz de composicion: el unico modulo que conoce los adaptadores concretos.

Patron Factory + inyeccion de dependencias. El dominio y la capa de aplicacion
importan solo puertos; aqui se decide que implementacion los satisface. Cambiar
disco local por MinIO, o pHash por dHash, se hace en este archivo y en ningun
otro — es lo que hace verificable el criterio de exito de intercambiar un
adaptador sin tocar el nucleo.

Los adaptadores importan `cv2` e `imagehash` dentro de sus funciones, no en el
encabezado del modulo. Por eso este contenedor puede importarse en un entorno sin
el extra `media` instalado: solo falla al construir el adaptador que lo necesita.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cached_property
from pathlib import Path

from inspeccion.adapters.hashing import ImageHashAdapter
from inspeccion.adapters.ml_engine import AnomalibMLEngine
from inspeccion.adapters.storage import FilesystemArtifactStore, FilesystemFrameWriter
from inspeccion.adapters.video_source import CameraControls, OpenCvVideoSource
from inspeccion.application.curation.extraction import FrameExtractor
from inspeccion.application.curation.pipeline import CurationPipeline
from inspeccion.application.curation.sampling import PerceptualDifferenceSampler
from inspeccion.config.settings import Settings
from inspeccion.domain.ports import (
    ArtifactStore,
    FrameWriter,
    MLEngine,
    PerceptualHasher,
    VideoSource,
)


class Container:
    """Ensambla los casos de uso a partir de `Settings`."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()

    # --- adaptadores ------------------------------------------------------

    @cached_property
    def hasher(self) -> PerceptualHasher:
        return ImageHashAdapter(
            algorithm=self.settings.hash_algorithm,
            hash_size=self.settings.hash_size,
        )

    @cached_property
    def artifact_store(self) -> ArtifactStore:
        return FilesystemArtifactStore(root=self.settings.storage_root)

    @cached_property
    def frame_writer(self) -> FrameWriter:
        return FilesystemFrameWriter(root=self.settings.storage_root)

    @cached_property
    def ml_engine(self) -> MLEngine:
        """PatchCore o PaDiM segun configuracion. Es el intercambio de adaptador
        que el criterio de exito del prototipo pide poder demostrar."""
        return AnomalibMLEngine(
            model=self.settings.anomaly_model,
            coreset_sampling_ratio=self.settings.coreset_sampling_ratio,
            batch_size=self.settings.batch_size,
            num_workers=0,
            accelerator=self.settings.accelerator,
            seed=self.settings.seed,
        )

    def video_source_from_file(self, path: Path | str) -> VideoSource:
        """Fabrica, no propiedad: una fuente de video es de un solo uso."""
        return OpenCvVideoSource.from_file(path)

    def video_source_from_webcam(
        self, index: int = 0, *, controls: CameraControls | None = None
    ) -> VideoSource:
        return OpenCvVideoSource.from_webcam(index, controls=controls)

    def video_source_from_rtsp(
        self, url: str, *, controls: CameraControls | None = None
    ) -> VideoSource:
        return OpenCvVideoSource.from_rtsp(url, controls=controls)

    # --- casos de uso -----------------------------------------------------

    @cached_property
    def sampler_factory(self) -> Callable[[], PerceptualDifferenceSampler]:
        """Devuelve un muestreador nuevo por pieza: su estado no cruza grupos."""
        settings = self.settings

        def build() -> PerceptualDifferenceSampler:
            return PerceptualDifferenceSampler(
                min_difference=settings.sampler_min_difference,
                min_interval=settings.sampler_min_interval,
                max_interval=settings.sampler_max_interval,
            )

        return build

    @cached_property
    def frame_extractor(self) -> FrameExtractor:
        return FrameExtractor(
            hasher=self.hasher,
            writer=self.frame_writer,
            sampler_factory=self.sampler_factory,
            max_frames=self.settings.live_max_frames,
        )

    @cached_property
    def curation_pipeline(self) -> CurationPipeline:
        return CurationPipeline(dedup_max_distance=self.settings.dedup_max_distance)
