"""Dobles en memoria de los puertos del dominio.

Existen para demostrar la propiedad que justifica la arquitectura hexagonal: los
casos de uso se ejecutan completos sin camara, sin OpenCV, sin imagehash y sin
tocar el disco. Si algun dia un caso de uso deja de poder testearse asi, es que se
filtro una dependencia de infraestructura hacia adentro.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

Image = npt.NDArray[np.uint8]


@dataclass
class FakeVideoSource:
    """Satisface `VideoSource` a partir de una lista de imagenes en memoria."""

    images: Sequence[Image]
    fps: float = 30.0
    closed: bool = field(default=False, init=False)

    def frames(self) -> Iterator[tuple[int, float, Image]]:
        for index, image in enumerate(self.images):
            yield index, index / self.fps, image

    def close(self) -> None:
        self.closed = True


@dataclass
class EndlessVideoSource:
    """Fuente que nunca termina, como una webcam o un stream RTSP."""

    image: Image
    fps: float = 30.0

    def frames(self) -> Iterator[tuple[int, float, Image]]:
        index = 0
        while True:
            yield index, index / self.fps, self.image
            index += 1

    def close(self) -> None:  # pragma: no cover - simetria con el puerto
        pass


@dataclass
class StubHasher:
    """Satisface `PerceptualHasher` con un hash derivado del contenido.

    Imagenes identicas producen el mismo hash e imagenes distintas producen
    hashes distintos, que es lo unico que la deduplicacion necesita para ser
    testeada. No pretende tener propiedades perceptuales.
    """

    n_bits: int = 64

    def hash(self, image: Image) -> int:
        digest = int(np.asarray(image, dtype=np.uint64).sum())
        return digest % (1 << self.n_bits)


@dataclass
class InMemoryFrameWriter:
    """Satisface `FrameWriter`. Registra lo escrito para poder inspeccionarlo."""

    written: dict[str, Image] = field(default_factory=dict)

    def write(self, image: Image, key: str) -> str:
        self.written[key] = image
        return key


@dataclass
class InMemoryArtifactStore:
    """Satisface `ArtifactStore`."""

    objects: dict[str, bytes] = field(default_factory=dict)

    def put(self, key: str, data: bytes) -> str:
        self.objects[key] = data
        return key

    def get(self, key: str) -> bytes:
        return self.objects[key]

    def exists(self, key: str) -> bool:
        return key in self.objects


def canvas(value: int, size: int = 128) -> Image:
    return np.full((size, size, 3), value, dtype=np.uint8)
