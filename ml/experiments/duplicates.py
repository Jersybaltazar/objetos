"""Generacion de casi-duplicados sinteticos para el experimento E2.

Emula la redundancia que produce un video: frames consecutivos que difieren en un
temblor de camara, un cambio minimo de iluminacion y ruido del sensor, pero que
representan la misma vista de la misma pieza.

Lo que se aplica y lo que no, y por que:

- **Si**: traslacion <=2% del lado, rotacion <=2 grados, escala +-2%, brillo y
  contraste +-3%, ruido gaussiano de baja varianza.
- **No**: volteos, rotaciones amplias, recortes agresivos, cambios de color. Eso
  seria *data augmentation*, que genera vistas nuevas. Aqui se necesita lo
  contrario: vistas que no aportan informacion. Si las transformaciones fueran
  fuertes, los duplicados dejarian de ser duplicados y el experimento medi­ria
  otra cosa.

La generacion es determinista: la semilla sale del nombre del fichero y del
indice del duplicado, nunca del reloj. Asi las replicas del experimento varian
solo en como se particiona, no en los datos, que es lo que aisla el efecto bajo
estudio.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

Image = npt.NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class JitterConfig:
    """Magnitud de las perturbaciones. Los valores por defecto son los del protocolo."""

    max_translation_fraction: float = 0.02
    max_rotation_degrees: float = 2.0
    max_scale_fraction: float = 0.02
    max_brightness_fraction: float = 0.03
    noise_sigma: float = 2.0

    def __post_init__(self) -> None:
        if self.max_rotation_degrees > 15.0:
            raise ValueError(
                f"rotacion de {self.max_rotation_degrees} grados: eso es augmentation, "
                f"no duplicacion. El experimento dejaria de medir redundancia."
            )
        for nombre in (
            "max_translation_fraction",
            "max_scale_fraction",
            "max_brightness_fraction",
        ):
            valor = getattr(self, nombre)
            if not 0.0 <= valor <= 0.25:
                raise ValueError(f"{nombre} fuera de [0, 0.25]: {valor}")
        if self.noise_sigma < 0:
            raise ValueError(f"noise_sigma no puede ser negativo: {self.noise_sigma}")


DEFAULT_JITTER = JitterConfig()
"""Singleton inmutable con los valores del protocolo."""


def jitter(image: Image, *, seed: int, config: JitterConfig = DEFAULT_JITTER) -> Image:
    """Devuelve un casi-duplicado determinista de `image`."""
    rng = np.random.default_rng(seed)
    height, width = image.shape[:2]

    angulo = rng.uniform(-config.max_rotation_degrees, config.max_rotation_degrees)
    escala = 1.0 + rng.uniform(-config.max_scale_fraction, config.max_scale_fraction)
    dx = rng.uniform(-1, 1) * config.max_translation_fraction * width
    dy = rng.uniform(-1, 1) * config.max_translation_fraction * height

    matriz = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angulo, escala)
    matriz[0, 2] += dx
    matriz[1, 2] += dy
    # BORDER_REFLECT_101 y no un borde constante: un borde negro seria una señal
    # artificial que el modelo podria aprender, y contaminaria el experimento.
    movida = cv2.warpAffine(
        image, matriz, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )

    ganancia = 1.0 + rng.uniform(-config.max_brightness_fraction, config.max_brightness_fraction)
    desplazamiento = rng.uniform(-1, 1) * config.max_brightness_fraction * 255.0
    ruido = rng.normal(0.0, config.noise_sigma, movida.shape) if config.noise_sigma > 0 else 0.0

    return np.clip(movida.astype(np.float64) * ganancia + desplazamiento + ruido, 0, 255).astype(
        np.uint8
    )


def materialise(
    originals: Mapping[str, Path],
    destination: Path,
    *,
    duplicates: int,
    config: JitterConfig = DEFAULT_JITTER,
    overwrite: bool = False,
) -> dict[str, list[Path]]:
    """Escribe en disco cada original mas `duplicates` casi-duplicados suyos.

    Args:
        originals: {id_de_grupo: ruta_original}. El identificador lo decide quien
            llama, y no se deriva aqui del nombre del fichero a proposito: MVTec AD
            numera igual las imagenes de `train/good` y de `test/good`
            (`000.png` en ambas), asi que usar el `stem` colapsaria pares de
            imagenes distintas en un mismo grupo y las perderia sin aviso.

    Devuelve {id_de_grupo: [rutas]}. Todos los duplicados de un original comparten
    identificador, que es lo que permite reutilizar sin cambios las dos
    estrategias de particion del nucleo.

    Es idempotente: si el destino ya tiene los ficheros, no los regenera. Los
    duplicados se generan una vez por valor de `duplicates` y se reutilizan en
    las diez replicas, de modo que entre replicas solo cambia la particion.
    """
    if duplicates < 0:
        raise ValueError(f"duplicates no puede ser negativo: {duplicates}")

    destination.mkdir(parents=True, exist_ok=True)
    grupos: dict[str, list[Path]] = {}

    for group_id, original in originals.items():
        carpeta = destination / group_id
        rutas = [carpeta / f"{group_id}__000.png"] + [
            carpeta / f"{group_id}__{i + 1:03d}.png" for i in range(duplicates)
        ]

        if overwrite or not all(r.is_file() for r in rutas):
            carpeta.mkdir(parents=True, exist_ok=True)
            imagen = cv2.imread(str(original), cv2.IMREAD_COLOR)
            if imagen is None:
                raise OSError(f"no se pudo leer la imagen original: {original}")

            cv2.imwrite(str(rutas[0]), imagen)
            for indice in range(duplicates):
                cv2.imwrite(
                    str(rutas[indice + 1]),
                    jitter(imagen, seed=_seed(group_id, indice), config=config),
                )

        grupos[group_id] = rutas

    return grupos


def _seed(group_id: str, index: int) -> int:
    """Semilla estable derivada del identificador de grupo.

    SHA-256 y no `hash()`: el hash de Python esta aleatorizado por proceso, asi
    que los duplicados no serian reproducibles entre corridas.
    """
    digest = hashlib.sha256(f"{group_id}:{index}".encode()).digest()
    return int.from_bytes(digest[:4], "big")
