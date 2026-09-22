"""Entidades y value objects del dominio.

El concepto central es el *grupo*: la unidad dentro de la cual los datos estan
correlacionados y que por tanto no puede repartirse entre splits. En este dominio
el grupo es la pieza fisica individual, no el video ni el frame.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import NewType

GroupId = NewType("GroupId", str)
"""Identificador de la pieza fisica. Acompana al frame desde la captura."""


class Split(enum.StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class Label(enum.StrEnum):
    """Etiqueta a nivel de pieza. Se propaga a todos sus frames."""

    NORMAL = "normal"
    DEFECT = "defect"


@dataclass(frozen=True, slots=True)
class FrameRef:
    """Referencia a un frame ya extraido y persistido.

    Es un value object: inmutable e identificado por su contenido. `group_id` es
    obligatorio precisamente para que sea imposible construir un frame sin saber
    a que pieza pertenece.
    """

    path: str
    group_id: GroupId
    label: Label
    source_video: str
    frame_index: int
    timestamp_s: float

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError(f"frame_index no puede ser negativo: {self.frame_index}")
        if self.timestamp_s < 0:
            raise ValueError(f"timestamp_s no puede ser negativo: {self.timestamp_s}")
        if not self.group_id:
            raise ValueError("group_id es obligatorio: sin el no se puede particionar por grupo")


@dataclass(frozen=True, slots=True)
class SplitRatios:
    train: float = 0.6
    val: float = 0.2
    test: float = 0.2

    def __post_init__(self) -> None:
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"las proporciones deben sumar 1.0, suman {total}")
        if min(self.train, self.val, self.test) <= 0:
            raise ValueError("toda proporcion debe ser estrictamente positiva")


class SplitStrategy(enum.StrEnum):
    """Las dos estrategias que el experimento E3 contrasta."""

    NAIVE_BY_FRAME = "naive_by_frame"
    """Asignacion aleatoria a nivel de frame. Produce fuga. Solo como baseline."""

    BY_GROUP = "by_group"
    """Asignacion aleatoria a nivel de pieza fisica. Es el metodo propuesto."""


@dataclass(frozen=True, slots=True)
class LeakageReport:
    """Diagnostico cuantitativo de fuga sobre un manifiesto ya construido.

    `contaminated_test_fraction` es la metrica que hace observable el fenomeno:
    que proporcion del test evaluable proviene de piezas que el modelo ya vio en
    entrenamiento.

    El denominador son los frames NORMALES de test, no todos. En el modo no
    supervisado el entrenamiento solo contiene piezas normales, asi que una pieza
    defectuosa nunca puede filtrarse por construccion: incluirla en el denominador
    solo diluiria la medida y haria que la cifra dependiera de la proporcion de
    defectos del dataset en vez de la fuga real. La fuga que infla el AUROC es
    precisamente la del lado normal: el modelo memorizo los frames sobre los que
    luego se mide su tasa de falsos positivos.
    """

    shared_groups_train_test: int
    shared_groups_train_val: int
    contaminated_test_frames: int
    normal_test_frames: int
    total_test_frames: int

    @property
    def contaminated_test_fraction(self) -> float:
        if self.normal_test_frames == 0:
            return 0.0
        return self.contaminated_test_frames / self.normal_test_frames

    @property
    def is_clean(self) -> bool:
        return self.shared_groups_train_test == 0 and self.shared_groups_train_val == 0


@dataclass(frozen=True, slots=True)
class SplitManifest:
    """Particion serializable.

    Se archiva como lista explicita de rutas, no como una semilla: reproducir un
    resultado no debe depender de que la implementacion del generador aleatorio
    no cambie nunca.
    """

    strategy: SplitStrategy
    seed: int
    assignments: dict[Split, tuple[FrameRef, ...]]
    metadata: dict[str, str] = field(default_factory=dict)

    def frames(self, split: Split) -> tuple[FrameRef, ...]:
        return self.assignments.get(split, ())

    def groups(self, split: Split) -> frozenset[GroupId]:
        return frozenset(f.group_id for f in self.frames(split))

    def frame_counts(self) -> dict[Split, int]:
        return {s: len(self.frames(s)) for s in Split}

    def group_counts(self) -> dict[Split, int]:
        """Se reporta junto a `frame_counts` para que la comparacion entre
        estrategias sea auditable: la estrategia ingenua tendra el mismo numero
        de frames pero muchos mas grupos compartidos."""
        return {s: len(self.groups(s)) for s in Split}
