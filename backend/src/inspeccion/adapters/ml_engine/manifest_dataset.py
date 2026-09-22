"""Puente entre el manifiesto de particion propio y los datasets de anomalib.

**Este modulo existe para impedir que anomalib vuelva a particionar los datos.**

`anomalib.data.Folder` y los demas datamodules de la libreria hacen su propio
reparto train/val/test (`test_split_mode`, `val_split_mode`, `val_split_ratio`,
`normal_split_ratio`). Alimentarlos con directorios y dejarlos repartir anularia
por completo la contribucion de esta tesis: la particion por pieza se perderia y
las metricas volverian a estar contaminadas, sin ningun aviso.

Aqui el manifiesto se inyecta directamente como el `DataFrame` de muestras que
anomalib consume, y los modos de reparto se configuran para que la base no
reasigne nada:

- `val_split_mode=NONE`: ninguna rama de `_create_val_split` coincide, asi que el
  conjunto de validacion queda tal como lo dejo el manifiesto.
- `test_split_mode=FROM_DIR`: **no** `NONE`. Con `NONE`, `_create_test_split`
  separa igualmente los normales de los anomalos y **no los vuelve a unir**: el
  conjunto de test perderia todas sus piezas buenas y el AUROC quedaria indefinido
  por tener una sola clase. Con `FROM_DIR` los separa y los reagrega, de modo que
  la operacion es la identidad.

El test de integracion `test_no_reparticiona` comprueba esa invariante fichero a
fichero. Debe seguir ejecutandose en cada actualizacion de anomalib: es la unica
forma de detectar que un cambio de la libreria volvio a mover los datos.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from inspeccion.domain.models import FrameRef, Label, Split, SplitManifest

if TYPE_CHECKING:  # pragma: no cover
    from pandas import DataFrame


def samples_frame(frames: Sequence[FrameRef], split: Split) -> DataFrame:
    """Construye el `DataFrame` de muestras que espera `AnomalibDataset`.

    Columnas obligatorias: `image_path` y `split`. `__getitem__` lee ademas
    `label_index` y `mask_path`. `mask_path` va vacio en todas las filas: el
    dataset propio no tiene mascaras anotadas y la tarea es de clasificacion, no
    de segmentacion (seccion 6.3 del protocolo de evaluacion).
    """
    from anomalib.data.utils import LabelName
    from pandas import DataFrame

    frame = DataFrame(
        {
            "image_path": [frame.path for frame in frames],
            "split": [split.value] * len(frames),
            "label_index": [
                LabelName.ABNORMAL if frame.label is Label.DEFECT else LabelName.NORMAL
                for frame in frames
            ],
            "mask_path": [""] * len(frames),
        }
    )
    # anomalib lee la tarea de `samples.attrs`, no de una columna. Sin esto,
    # `__getitem__` revienta con KeyError: 'task'.
    frame.attrs["task"] = "classification"
    return frame


def build_dataset(frames: Sequence[FrameRef], split: Split, augmentations: Any = None) -> Any:
    """Dataset de anomalib respaldado por una lista explicita de frames."""
    from anomalib.data.datasets.base.image import AnomalibDataset

    dataset = AnomalibDataset(augmentations=augmentations)
    dataset.samples = samples_frame(frames, split)
    return dataset


def build_datamodule(
    manifest: SplitManifest,
    *,
    train_batch_size: int = 32,
    eval_batch_size: int = 32,
    num_workers: int = 0,
    augmentations: Any = None,
) -> Any:
    """Datamodule que entrega exactamente los splits del manifiesto.

    `num_workers=0` por defecto: en Windows cada worker relanza el proceso, y
    para los volumenes de esta tesis el coste de arranque supera la ganancia.
    """
    from anomalib.data.datamodules.base.image import AnomalibDataModule
    from anomalib.data.utils.split import TestSplitMode, ValSplitMode

    class _ManifestDataModule(AnomalibDataModule):  # type: ignore[misc]
        def _setup(self, _stage: str | None = None) -> None:
            self.train_data = build_dataset(
                manifest.frames(Split.TRAIN), Split.TRAIN, augmentations
            )
            self.val_data = build_dataset(manifest.frames(Split.VAL), Split.VAL, augmentations)
            self.test_data = build_dataset(manifest.frames(Split.TEST), Split.TEST, augmentations)

    return _ManifestDataModule(
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
        augmentations=augmentations,
        val_split_mode=ValSplitMode.NONE,
        val_split_ratio=0.0,
        test_split_mode=TestSplitMode.FROM_DIR,
        test_split_ratio=0.0,
        seed=None,
    )
