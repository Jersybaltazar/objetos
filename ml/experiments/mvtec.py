"""Carga de MVTec AD desde una copia local, respetando la particion oficial.

No se usa el descargador de anomalib: el enlace que lleva codificado
(`mydrive.ch`) devuelve 404 desde que MVTec movio la distribucion a descarga con
registro en su propio sitio. El dataset se obtiene a mano de la fuente oficial y
este modulo solo lo lee.

La particion **es la de los autores y no se toca**. E1 compara contra cifras
publicadas, y esas cifras se calcularon sobre esa particion; aplicar aqui el
metodo de particion de la tesis haria la comparacion invalida. El metodo propio
entra en E2 y E3.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from inspeccion.domain.models import FrameRef, GroupId, Label, Split, SplitManifest, SplitStrategy

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

CATEGORIAS_OBJETO = (
    "bottle",
    "cable",
    "capsule",
    "hazelnut",
    "metal_nut",
    "pill",
    "screw",
    "toothbrush",
    "transistor",
    "zipper",
)
"""Categorias de objeto de MVTec AD. E1 usa un subconjunto de estas y no las de
textura, por analogia directa con las piezas del dataset propio (tapas, tuercas)."""


class MVTecLayoutError(FileNotFoundError):
    """La copia local no tiene la estructura que MVTec AD publica."""


@dataclass(frozen=True, slots=True)
class CategoryReport:
    """Diagnostico de una categoria en disco."""

    category: str
    train_normal: int
    test_normal: int
    test_defect: int
    defect_types: tuple[str, ...]

    @property
    def is_usable(self) -> bool:
        return self.train_normal > 0 and self.test_normal > 0 and self.test_defect > 0

    def describe(self) -> str:
        estado = "ok" if self.is_usable else "INCOMPLETA"
        return (
            f"{self.category:<12} {estado:<10} "
            f"train/good={self.train_normal:<5} test/good={self.test_normal:<5} "
            f"test/defecto={self.test_defect:<5} tipos={len(self.defect_types)}"
        )


def inspect_category(root: Path, category: str) -> CategoryReport:
    """Cuenta lo que hay en disco sin cargar ninguna imagen."""
    base = root / category
    train_good = _images(base / "train" / "good")
    test_dir = base / "test"

    test_normal: list[Path] = []
    test_defect: list[Path] = []
    defect_types: list[str] = []

    if test_dir.is_dir():
        for sub in sorted(p for p in test_dir.iterdir() if p.is_dir()):
            if sub.name == "good":
                test_normal = _images(sub)
            else:
                encontrados = _images(sub)
                if encontrados:
                    defect_types.append(sub.name)
                    test_defect.extend(encontrados)

    return CategoryReport(
        category=category,
        train_normal=len(train_good),
        test_normal=len(test_normal),
        test_defect=len(test_defect),
        defect_types=tuple(defect_types),
    )


def official_manifest(root: Path, category: str) -> SplitManifest:
    """Manifiesto con la particion oficial: train/good -> TRAIN, test/* -> TEST.

    VAL queda vacio a proposito. MVTec AD no define validacion, E1 no calibra
    ningun umbral (el AUROC es independiente del umbral) y el memory bank de
    PatchCore se construye igual en `on_train_epoch_end`.
    """
    report = inspect_category(root, category)
    if not report.is_usable:
        raise MVTecLayoutError(
            f"la categoria {category!r} esta incompleta en {root}: "
            f"train/good={report.train_normal}, test/good={report.test_normal}, "
            f"test/defecto={report.test_defect}. "
            f"Se espera la estructura oficial <root>/<categoria>/{{train,test}}/..."
        )

    base = root / category
    train = [
        _frame(path, Label.NORMAL, category, "train_good")
        for path in _images(base / "train" / "good")
    ]

    test: list[FrameRef] = []
    for sub in sorted(p for p in (base / "test").iterdir() if p.is_dir()):
        etiqueta = Label.NORMAL if sub.name == "good" else Label.DEFECT
        test.extend(_frame(path, etiqueta, category, sub.name) for path in _images(sub))

    return SplitManifest(
        strategy=SplitStrategy.BY_GROUP,
        seed=0,
        assignments={Split.TRAIN: tuple(train), Split.VAL: (), Split.TEST: tuple(test)},
        metadata={
            "dataset": "MVTec AD",
            "category": category,
            "split": "oficial de los autores, sin modificar",
            "licencia": "CC BY-NC-SA 4.0 (solo investigacion no comercial)",
            "defect_types": ",".join(report.defect_types),
        },
    )


def _frame(path: Path, label: Label, category: str, subset: str) -> FrameRef:
    """En MVTec AD cada imagen es independiente: no hay piezas que agrupar.

    El `group_id` se construye unico por imagen para que ninguna agrupacion
    accidental cambie nada. E1 no reparte nada de todos modos.
    """
    return FrameRef(
        path=str(path),
        group_id=GroupId(f"{category}/{subset}/{path.stem}"),
        label=label,
        source_video=f"{category}/{subset}",
        frame_index=0,
        timestamp_s=0.0,
    )


def _images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
