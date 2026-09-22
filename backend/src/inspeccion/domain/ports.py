"""Puertos del dominio (hexagonal selectivo).

Se definen puertos SOLO en las cuatro fronteras que realmente van a variar:
motor ML, almacenamiento, fuente de video y runtime de inferencia. El resto del
sistema es codigo directo; poner puertos en todas partes seria sobreingenieria.

Se usan `Protocol` en lugar de clases base abstractas para que los adaptadores no
tengan que heredar de nada del dominio: la dependencia apunta hacia adentro.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from inspeccion.domain.models import FrameRef, SplitManifest

Image = npt.NDArray[np.uint8]
"""Frame decodificado, layout HxWxC en BGR (convencion de OpenCV)."""


@runtime_checkable
class VideoSource(Protocol):
    """Fuente de frames. Adaptadores: archivo, webcam USB, RTSP."""

    def frames(self) -> Iterator[tuple[int, float, Image]]:
        """Produce (indice_de_frame, timestamp_en_segundos, imagen)."""
        ...

    @property
    def fps(self) -> float: ...

    def close(self) -> None: ...


@runtime_checkable
class PerceptualHasher(Protocol):
    """Hashing perceptual para deduplicacion.

    Se aisla como puerto para poder testear el algoritmo de deduplicacion con
    hashes sinteticos, sin depender de la libreria de imagenes.
    """

    def hash(self, image: Image) -> int:
        """Devuelve un hash de `n_bits` bits empaquetado en un entero."""
        ...

    @property
    def n_bits(self) -> int: ...


@runtime_checkable
class MLEngine(Protocol):
    """Motor de entrenamiento y evaluacion. Adaptadores: PatchCore, PaDiM."""

    def train(self, manifest: SplitManifest, work_dir: Path) -> Path:
        """Entrena con el split TRAIN del manifiesto. Devuelve la ruta del modelo."""
        ...

    def evaluate(self, model_path: Path, frames: Sequence[FrameRef]) -> dict[str, float]:
        """Devuelve metricas. Nunca debe invocarse sobre TEST para calibrar."""
        ...

    def scores(self, model_path: Path, frames: Sequence[FrameRef]) -> npt.NDArray[np.float64]:
        """Score de anomalia por frame, sin umbralizar."""
        ...

    def export_onnx(self, model_path: Path, destination: Path) -> Path: ...


@runtime_checkable
class InferenceRuntime(Protocol):
    """Runtime de inferencia. Adaptadores: ONNX Runtime, OpenVINO."""

    def load(self, model_path: Path) -> None: ...

    def score(self, image: Image) -> float:
        """Score de anomalia de un unico frame."""
        ...


@runtime_checkable
class ArtifactStore(Protocol):
    """Almacenamiento de videos, frames y modelos. Adaptadores: disco, MinIO."""

    def put(self, key: str, data: bytes) -> str: ...

    def get(self, key: str) -> bytes: ...

    def exists(self, key: str) -> bool: ...


@runtime_checkable
class FrameWriter(Protocol):
    """Persistencia de un frame decodificado. Pertenece a la frontera de
    almacenamiento, junto a `ArtifactStore`.

    Se declara aparte porque codificar una imagen (PNG, JPEG, calidad, submuestreo
    de croma) es una decision de infraestructura que no debe filtrarse al caso de
    uso de extraccion: este solo necesita saber que el frame quedo guardado y bajo
    que clave.
    """

    def write(self, image: Image, key: str) -> str:
        """Persiste el frame y devuelve la ruta o URI resultante."""
        ...
