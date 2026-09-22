"""Construccion emparejada de brazos con fuga controlada (experimento E2).

Vive junto a `splitting.py` porque hace lo mismo que el: producir
`SplitManifest`. La diferencia es que aqui la contaminacion se inyecta a
proposito y en una dosis medida, para poder cuantificar su efecto. Igual que
`split_naive_by_frame`, es codigo de soporte experimental y NO debe usarse
para producir ningun modelo de produccion.

El piloto de E2 con particiones independientes mostro dos problemas que hacian el
delta no interpretable:

1. **Varianza entre particiones.** A k=0, donde no hay ningun duplicado, los dos
   brazos daban un delta de +3.62 pp. No era fuga: eran dos particiones distintas
   del mismo conjunto, ambas validas. PaDiM sobre `bottle` oscila entre 0.86 y
   0.98 segun que imagenes toquen en entrenamiento, y esa varianza tapaba el
   efecto buscado.

2. **Confundido de tamano de muestra.** Con particion aleatoria por frame, el
   brazo ingenuo entrenaba con imagenes procedentes de casi todos los originales,
   mientras que el brazo por grupo solo veia los originales de su 60%. El ingenuo
   disponia de mas informacion, y eso subia su AUROC al margen de cualquier fuga.
   Es la objecion de la seccion 12 del protocolo, hecha concreta.

Este modulo construye los dos brazos de forma **emparejada**:

- **El conjunto de test es identico** en ambos brazos y para todos los niveles de
  dosis: un unico ejemplar (el original intacto) de cada pieza de test, mas todos
  los defectos. Al no moverse el test, desaparece la mayor fuente de varianza y
  los AUROC pasan a ser directamente comparables.
- **El tamano del entrenamiento es identico** en ambos brazos.
- La **unica** diferencia es que en el brazo contaminado una fraccion `lambda` del
  entrenamiento son casi-duplicados de piezas que estan en test.

Asi `lambda` es la dosis, y es la variable que de verdad describe el mecanismo:
que proporcion del entrenamiento es material que reaparece en la evaluacion. El
numero de duplicados `k` del protocolo original era un proxy indirecto de esto.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from inspeccion.domain.models import (
    FrameRef,
    GroupId,
    Label,
    Split,
    SplitManifest,
    SplitRatios,
    SplitStrategy,
)

DEFAULT_RATIOS = SplitRatios()
"""Singleton inmutable: 60/20/20 sobre las piezas."""


class InsufficientDuplicatesError(ValueError):
    """No hay casi-duplicados suficientes para alcanzar la dosis pedida."""


@dataclass(frozen=True, slots=True)
class PairedArms:
    """Los dos brazos de una replica, ya emparejados."""

    clean: SplitManifest
    leaked: SplitManifest
    leak_fraction: float
    n_leaked_frames: int
    n_train: int
    n_test: int

    def summary(self) -> dict[str, float | int]:
        return {
            "leak_fraction": round(self.leak_fraction, 6),
            "n_leaked_frames": self.n_leaked_frames,
            "n_train": self.n_train,
            "n_test": self.n_test,
        }


def build_leaked_arms(
    originals: Mapping[str, Sequence[Path]],
    defects: Sequence[Path],
    *,
    seed: int,
    leak_fraction: float,
    ratios: SplitRatios = DEFAULT_RATIOS,
) -> PairedArms:
    """Construye los dos brazos de una replica.

    Args:
        originals: {id_de_pieza: [original, duplicado_1, ...]}. El elemento 0 es
            siempre la imagen intacta.
        defects: imagenes defectuosas, sin duplicar. Van enteras a test en ambos
            brazos: no participan del entrenamiento y por tanto no son via de fuga.
        leak_fraction: proporcion del entrenamiento que, en el brazo contaminado,
            son casi-duplicados de piezas de test. 0.0 hace los dos brazos
            identicos, lo que sirve de control negativo real.
    """
    if not 0.0 <= leak_fraction <= 1.0:
        raise ValueError(f"leak_fraction fuera de [0,1]: {leak_fraction}")

    rng = random.Random(seed)
    piezas = sorted(originals)
    rng.shuffle(piezas)

    n_train = round(len(piezas) * ratios.train)
    n_val = round(len(piezas) * ratios.val)
    train_ids = piezas[:n_train]
    val_ids = piezas[n_train : n_train + n_val]
    test_ids = piezas[n_train + n_val :]
    if not (train_ids and val_ids and test_ids):
        raise ValueError(f"no hay piezas suficientes para particionar: {len(piezas)}")

    # Test: un solo ejemplar intacto por pieza. Fijo en ambos brazos y en todas
    # las dosis, que es lo que elimina la varianza del conjunto de evaluacion.
    test_frames = [_normal(originals[p][0], p, 0) for p in test_ids]
    test_frames += [_defect(ruta) for ruta in defects]
    val_frames = [_normal(originals[p][0], p, 0) for p in val_ids]

    train_limpio = [
        _normal(ruta, pieza, indice)
        for pieza in train_ids
        for indice, ruta in enumerate(originals[pieza])
    ]

    n_fuga = round(len(train_limpio) * leak_fraction)
    disponibles = [
        (pieza, indice, ruta)
        for pieza in test_ids
        for indice, ruta in enumerate(originals[pieza])
        if indice > 0  # nunca el original intacto: seria duplicacion exacta
    ]
    if n_fuga > len(disponibles):
        raise InsufficientDuplicatesError(
            f"la dosis {leak_fraction:.0%} exige {n_fuga} casi-duplicados de piezas de "
            f"test y solo hay {len(disponibles)}. Genera mas duplicados por original."
        )

    rng.shuffle(disponibles)
    train_contaminado = train_limpio[n_fuga:] + [
        _normal(ruta, pieza, indice) for pieza, indice, ruta in disponibles[:n_fuga]
    ]

    return PairedArms(
        clean=_manifest(SplitStrategy.BY_GROUP, seed, train_limpio, val_frames, test_frames),
        leaked=_manifest(
            SplitStrategy.NAIVE_BY_FRAME, seed, train_contaminado, val_frames, test_frames
        ),
        leak_fraction=leak_fraction,
        n_leaked_frames=n_fuga,
        n_train=len(train_limpio),
        n_test=len(test_frames),
    )


def _normal(ruta: Path, pieza: str, indice: int) -> FrameRef:
    return FrameRef(
        path=str(ruta),
        group_id=GroupId(pieza),
        label=Label.NORMAL,
        source_video=f"normal/{pieza}",
        frame_index=indice,
        timestamp_s=indice / 30.0,
    )


def _defect(ruta: Path) -> FrameRef:
    return FrameRef(
        path=str(ruta),
        group_id=GroupId(f"defecto/{ruta.parent.name}/{ruta.stem}"),
        label=Label.DEFECT,
        source_video=f"defecto/{ruta.parent.name}",
        frame_index=0,
        timestamp_s=0.0,
    )


def _manifest(
    strategy: SplitStrategy,
    seed: int,
    train: list[FrameRef],
    val: list[FrameRef],
    test: list[FrameRef],
) -> SplitManifest:
    return SplitManifest(
        strategy=strategy,
        seed=seed,
        assignments={
            Split.TRAIN: tuple(train),
            Split.VAL: tuple(val),
            Split.TEST: tuple(test),
        },
        metadata={"diseno": "brazos emparejados, conjunto de test fijo"},
    )
