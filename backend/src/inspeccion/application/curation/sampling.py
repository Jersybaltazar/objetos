"""Muestreo de frames por diferencia perceptual (RF-03).

Primera barrera contra la redundancia: decide en el momento de la decodificacion
si un frame merece guardarse, comparandolo con el ultimo conservado. Es mas
barato que extraer todo y deduplicar despues, y reduce el volumen en disco.

No sustituye a la deduplicacion por hashing (`dedup.py`): este filtro es local
(solo mira el frame anterior conservado) y por tanto no detecta que una pieza en
rotacion ha vuelto a una orientacion ya capturada. Los dos mecanismos son
complementarios y el experimento E4 mide el efecto conjunto.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

Image = npt.NDArray[np.uint8]


@dataclass(slots=True)
class PerceptualDifferenceSampler:
    """Conserva un frame solo si difiere lo bastante del ultimo conservado.

    Args:
        min_difference: diferencia absoluta media minima, normalizada a [0,1],
            para considerar el frame novedoso. 0.02 equivale a ~5 niveles de gris
            de diferencia promedio sobre 255.
        min_interval: numero minimo de frames entre dos conservados. Evita
            conservar rafagas durante un movimiento brusco.
        max_interval: si se supera sin conservar nada, se fuerza la conservacion.
            Garantiza cobertura temporal aunque la escena este quieta. `None`
            desactiva la garantia.
        thumb_size: lado del thumbnail sobre el que se mide la diferencia.
    """

    min_difference: float = 0.02
    min_interval: int = 1
    max_interval: int | None = 90
    thumb_size: int = 64

    _reference: npt.NDArray[np.float32] | None = field(default=None, init=False, repr=False)
    _since_kept: int = field(default=0, init=False, repr=False)
    seen: int = field(default=0, init=False)
    kept: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_difference <= 1.0:
            raise ValueError(f"min_difference fuera de [0,1]: {self.min_difference}")
        if self.min_interval < 1:
            raise ValueError("min_interval debe ser >= 1")
        if self.max_interval is not None and self.max_interval < self.min_interval:
            raise ValueError("max_interval debe ser >= min_interval")
        if self.thumb_size < 8:
            raise ValueError("thumb_size debe ser >= 8")

    def should_keep(self, image: Image) -> bool:
        """Decide si conservar el frame y actualiza el estado interno."""
        self.seen += 1
        thumb = _thumbnail(image, self.thumb_size)

        if self._reference is None:
            return self._accept(thumb)

        self._since_kept += 1
        if self._since_kept < self.min_interval:
            return False
        if self.max_interval is not None and self._since_kept >= self.max_interval:
            return self._accept(thumb)

        difference = float(np.abs(thumb - self._reference).mean()) / 255.0
        if difference >= self.min_difference:
            return self._accept(thumb)
        return False

    def reset(self) -> None:
        """Se invoca al cambiar de video o de pieza: el estado no debe cruzar grupos."""
        self._reference = None
        self._since_kept = 0

    def stats(self) -> dict[str, float | int]:
        return {
            "seen": self.seen,
            "kept": self.kept,
            "retained_fraction": round(self.kept / self.seen, 6) if self.seen else 1.0,
        }

    def _accept(self, thumb: npt.NDArray[np.float32]) -> bool:
        self._reference = thumb
        self._since_kept = 0
        self.kept += 1
        return True


def _thumbnail(image: Image, size: int) -> npt.NDArray[np.float32]:
    """Reduce a escala de grises y `size`x`size` por submuestreo con paso.

    Se usa submuestreo y no interpolacion para no depender de OpenCV aqui: el
    modulo debe poder testearse sin el stack de imagen completo. El sesgo de
    aliasing es irrelevante para una medida de diferencia agregada.
    """
    if image.ndim == 3:
        grey = image.mean(axis=2)
    elif image.ndim == 2:
        grey = image.astype(np.float64)
    else:
        raise ValueError(f"se esperaba una imagen HxW o HxWxC, llego shape={image.shape}")

    height, width = grey.shape
    if height < size or width < size:
        raise ValueError(f"imagen {height}x{width} menor que el thumbnail {size}x{size}")

    rows = np.linspace(0, height - 1, size).astype(np.intp)
    cols = np.linspace(0, width - 1, size).astype(np.intp)
    return grey[np.ix_(rows, cols)].astype(np.float32)
