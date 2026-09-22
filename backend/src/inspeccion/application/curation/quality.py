"""Control de calidad de la captura (protocolo de captura v2, seccion 8).

El protocolo exige iluminacion constante y difusa, y enfoque, exposicion y balance
de blancos fijos. `CameraControls` intenta fijarlos y `applied_controls` reporta
si el driver *acepto* la orden, pero eso no prueba nada: `VideoCapture.set()`
puede devolver True sin efecto real segun backend y sistema operativo. La unica
forma de saber si la exposicion quedo fija es **medirlo en los frames**.

Este modulo mide sobre los propios frames:

- **Exposicion**: brillo medio y fraccion de pixeles recortados en negro o blanco.
  Un recorte alto borra textura, justo donde vive un defecto de superficie.
- **Parpadeo / autoexposicion**: variacion del brillo a lo largo del clip. Con la
  exposicion realmente fija y la luz constante, el brillo de una pieza girando
  apenas cambia; si oscila, la autoexposicion sigue activa o la luz parpadea.
- **Enfoque**: varianza del laplaciano, la medida de nitidez estandar.
- **Deriva entre clips**: brillo y nitidez de cada clip frente al primero de la
  sesion. El protocolo pide la misma iluminacion en todos los videos; si cambia,
  el modelo aprenderia la iluminacion como parte de lo "normal".

Todos los umbrales son heuristicos y estan declarados como tales. Su funcion es
avisar durante la captura, cuando repetir un clip cuesta segundos, no validar
nada a posteriori.

Implementacion en numpy puro, sin OpenCV, para poder testearse sin el stack de
imagen.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

Image = npt.NDArray[np.uint8]

UMBRAL_RECORTE = 0.02
"""Fraccion de pixeles recortados a partir de la que se avisa (2%)."""

UMBRAL_PARPADEO = 0.03
"""Coeficiente de variacion del brillo entre frames a partir del que se avisa (3%)."""

UMBRAL_DERIVA_BRILLO = 0.10
"""Cambio relativo de brillo frente al clip de referencia (10%)."""

UMBRAL_PERDIDA_NITIDEZ = 0.40
"""Caida relativa de nitidez frente al clip de referencia (40%). La nitidez depende
mucho del contenido de la escena, asi que solo se compara contra la propia sesion y
con un margen amplio."""


@dataclass(frozen=True, slots=True)
class FrameQuality:
    brightness: float
    """Brillo medio en [0, 1]."""
    clipped_dark: float
    """Fraccion de pixeles en negro saturado."""
    clipped_bright: float
    """Fraccion de pixeles en blanco saturado."""
    sharpness: float
    """Varianza del laplaciano sobre la imagen en grises."""


@dataclass(frozen=True, slots=True)
class ClipQuality:
    """Resumen de calidad de un clip, con los avisos que dispara."""

    n_frames: int
    brightness_mean: float
    brightness_cv: float
    """Coeficiente de variacion del brillo entre frames: desviacion / media."""
    clipped_max: float
    sharpness_median: float
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_ok(self) -> bool:
        return not self.warnings

    def summary(self) -> dict[str, float | int | list[str]]:
        return {
            "n_frames": self.n_frames,
            "brightness_mean": round(self.brightness_mean, 4),
            "brightness_cv": round(self.brightness_cv, 4),
            "clipped_max": round(self.clipped_max, 4),
            "sharpness_median": round(self.sharpness_median, 2),
            "warnings": list(self.warnings),
        }


def assess_frame(image: Image) -> FrameQuality:
    """Mide exposicion y nitidez de un frame (BGR o grises, uint8)."""
    grey = _grey(image)
    total = grey.size
    return FrameQuality(
        brightness=float(grey.mean()) / 255.0,
        clipped_dark=float(np.count_nonzero(grey <= 5)) / total,
        clipped_bright=float(np.count_nonzero(grey >= 250)) / total,
        sharpness=_laplacian_variance(grey),
    )


def assess_clip(frames: Sequence[FrameQuality]) -> ClipQuality:
    """Agrega la calidad de los frames de un clip y emite avisos."""
    if not frames:
        raise ValueError("no hay frames que evaluar")

    brillos = np.array([f.brightness for f in frames], dtype=np.float64)
    media = float(brillos.mean())
    cv = float(brillos.std() / media) if media > 0 else 0.0
    recorte = max(max(f.clipped_dark, f.clipped_bright) for f in frames)
    nitidez = float(np.median([f.sharpness for f in frames]))

    avisos: list[str] = []
    if recorte > UMBRAL_RECORTE:
        oscuro = max(f.clipped_dark for f in frames) >= max(f.clipped_bright for f in frames)
        avisos.append(
            f"{recorte:.1%} de pixeles recortados en "
            f"{'negro (subexpuesto)' if oscuro else 'blanco (sobreexpuesto)'}: "
            f"se pierde la textura donde aparecen los defectos"
        )
    if len(frames) > 1 and cv > UMBRAL_PARPADEO:
        avisos.append(
            f"el brillo varia un {cv:.1%} a lo largo del clip: la autoexposicion "
            f"sigue activa o la luz parpadea, aunque el driver diga lo contrario"
        )
    return ClipQuality(
        n_frames=len(frames),
        brightness_mean=media,
        brightness_cv=cv,
        clipped_max=recorte,
        sharpness_median=nitidez,
        warnings=tuple(avisos),
    )


def compare_to_reference(clip: ClipQuality, reference: ClipQuality) -> tuple[str, ...]:
    """Avisos de deriva frente al primer clip de la sesion."""
    avisos: list[str] = []
    if reference.brightness_mean > 0:
        deriva = abs(clip.brightness_mean - reference.brightness_mean) / reference.brightness_mean
        if deriva > UMBRAL_DERIVA_BRILLO:
            avisos.append(
                f"el brillo cambio un {deriva:.0%} respecto del primer clip: la "
                f"iluminacion no es la misma y el protocolo exige que lo sea"
            )
    if reference.sharpness_median > 0:
        caida = 1.0 - clip.sharpness_median / reference.sharpness_median
        if caida > UMBRAL_PERDIDA_NITIDEZ:
            avisos.append(
                f"la nitidez cayo un {caida:.0%} respecto del primer clip: "
                f"revisar enfoque o distancia camara-pieza"
            )
    return tuple(avisos)


def _grey(image: Image) -> npt.NDArray[np.float64]:
    if image.ndim == 3:
        return np.asarray(image, dtype=np.float64).mean(axis=2)
    if image.ndim == 2:
        return np.asarray(image, dtype=np.float64)
    raise ValueError(f"se esperaba una imagen HxW o HxWxC, llego shape={image.shape}")


def _laplacian_variance(grey: npt.NDArray[np.float64]) -> float:
    """Laplaciano de 4 vecinos, calculado por desplazamiento de matrices."""
    if grey.shape[0] < 3 or grey.shape[1] < 3:
        return 0.0
    centro = grey[1:-1, 1:-1]
    lap = grey[:-2, 1:-1] + grey[2:, 1:-1] + grey[1:-1, :-2] + grey[1:-1, 2:] - 4.0 * centro
    return float(lap.var())
