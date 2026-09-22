"""Adaptador de hashing perceptual sobre la libreria `imagehash` (BSD-2).

Implementa el puerto `PerceptualHasher` mediante el patron **Strategy**: el
algoritmo concreto (aHash, pHash, dHash, wHash) es una funcion inyectada y
seleccionable en tiempo de configuracion, de modo que el nucleo de deduplicacion
no conoce ninguno de ellos.

Los cuatro algoritmos soportados producen `hash_size**2` bits, lo que mantiene la
distancia de Hamming comparable entre estrategias. `colorhash` queda fuera
deliberadamente: su longitud sigue otra formula y romperia esa comparabilidad, que
es justamente lo que el barrido del experimento E4 necesita.
"""

from __future__ import annotations

import enum
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from inspeccion.domain.ports import Image

if TYPE_CHECKING:  # pragma: no cover
    import imagehash


class HashAlgorithm(enum.StrEnum):
    """Estrategias de hashing disponibles.

    pHash es el valor por defecto del proyecto: opera sobre la DCT y es el mas
    robusto de los cuatro frente a cambios de brillo y escala, que es exactamente
    la variacion que introduce una pieza girando bajo iluminacion fija.
    """

    AHASH = "ahash"
    PHASH = "phash"
    DHASH = "dhash"
    WHASH = "whash"


@dataclass(frozen=True, slots=True)
class ImageHashAdapter:
    """Adaptador que satisface `PerceptualHasher`.

    Args:
        algorithm: estrategia de hashing.
        hash_size: lado de la rejilla de bits. 8 produce hashes de 64 bits, que
            es la escala sobre la que estan calibrados los umbrales del
            experimento E4 (0, 2, 4, 6, 8, 10, 12).
    """

    algorithm: HashAlgorithm = HashAlgorithm.PHASH
    hash_size: int = 8
    _strategy: Callable[..., imagehash.ImageHash] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.hash_size < 2:
            raise ValueError(f"hash_size debe ser >= 2, llego {self.hash_size}")
        if self.algorithm is HashAlgorithm.WHASH and self.hash_size & (self.hash_size - 1):
            raise ValueError(f"wHash exige hash_size potencia de dos, llego {self.hash_size}")
        object.__setattr__(self, "_strategy", _resolve_strategy(self.algorithm))

    @property
    def n_bits(self) -> int:
        return self.hash_size * self.hash_size

    def hash(self, image: Image) -> int:
        """Devuelve el hash perceptual empaquetado como entero sin signo.

        Se empaqueta a entero y no se devuelve el objeto de la libreria para que
        el dominio no dependa de `imagehash`: la distancia de Hamming se calcula
        con `int.bit_count()`, sin conocer de donde salio el hash.
        """
        bits = np.asarray(self._strategy(_to_pil(image), hash_size=self.hash_size).hash).flatten()
        return int.from_bytes(np.packbits(bits).tobytes(), "big")


def _resolve_strategy(algorithm: HashAlgorithm) -> Callable[..., imagehash.ImageHash]:
    import imagehash

    strategies: dict[HashAlgorithm, Callable[..., imagehash.ImageHash]] = {
        HashAlgorithm.AHASH: imagehash.average_hash,
        HashAlgorithm.PHASH: imagehash.phash,
        HashAlgorithm.DHASH: imagehash.dhash,
        HashAlgorithm.WHASH: imagehash.whash,
    }
    return strategies[algorithm]


def _to_pil(image: Image) -> object:
    """Convierte de la convencion BGR de OpenCV al RGB que espera Pillow.

    Omitir esta conversion no rompe nada visible —el hash sigue siendo estable—
    pero produciria hashes distintos a los de cualquier herramienta que lea la
    misma imagen desde disco, y por tanto resultados no reproducibles fuera de
    este proceso.
    """
    from PIL import Image as PILImage

    if image.ndim == 2:
        return PILImage.fromarray(image, mode="L")
    if image.ndim == 3 and image.shape[2] == 3:
        return PILImage.fromarray(image[:, :, ::-1], mode="RGB")
    raise ValueError(f"se esperaba una imagen HxW o HxWx3, llego shape={image.shape}")
