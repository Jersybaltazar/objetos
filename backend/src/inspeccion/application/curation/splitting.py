"""Particion de datasets derivados de video.

Este modulo es el nucleo de la contribucion de la tesis. Implementa las dos
estrategias que el experimento E3 contrasta:

- `split_naive_by_frame`: reparte frames individuales al azar. Los frames vecinos
  de una misma pieza caen a ambos lados y el modelo evalua sobre material que ya
  memorizo. Se incluye deliberadamente como baseline: es el objeto de estudio,
  no un descuido.
- `split_by_group`: reparte piezas fisicas completas. Ninguna pieza cruza el
  limite del split.

`audit_leakage` hace observable la diferencia sin necesidad de entrenar nada.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

from inspeccion.domain.models import (
    FrameRef,
    GroupId,
    Label,
    LeakageReport,
    Split,
    SplitManifest,
    SplitRatios,
    SplitStrategy,
)

DEFAULT_RATIOS = SplitRatios()
"""Singleton inmutable: 60/20/20 sobre las piezas normales."""


class InsufficientGroupsError(ValueError):
    """Se lanza cuando no hay piezas suficientes para construir la particion.

    Es el modo de fallo que el protocolo v1 de captura provocaba: con 3 videos
    solo hay 3 grupos, y ninguna replica de particion es posible. Fallar aqui,
    ruidosamente, es preferible a producir un experimento sin potencia.
    """


def split_by_group(
    frames: Sequence[FrameRef],
    *,
    ratios: SplitRatios = DEFAULT_RATIOS,
    seed: int = 0,
    defect_val_fraction: float = 0.3,
) -> SplitManifest:
    """Particiona por pieza fisica.

    Las piezas normales se reparten entre TRAIN/VAL/TEST segun `ratios`. Las
    piezas con defecto se reparten solo entre VAL y TEST: el entrenamiento es no
    supervisado y nunca ve un defecto. Las piezas defectuosas entran a VAL porque
    el umbral de decision debe calibrarse sobre positivos que no sean los de TEST.

    El reparto es aleatorio en el orden de las piezas y avaro en el destino: cada
    pieza va al split con mayor deficit de frames respecto de su objetivo. Asi el
    numero de frames queda equilibrado sin romper ningun grupo.
    """
    if not 0.0 <= defect_val_fraction <= 1.0:
        raise ValueError(f"defect_val_fraction fuera de [0,1]: {defect_val_fraction}")

    rng = random.Random(seed)
    by_group = _group_frames(frames)

    normal_groups = {g: fs for g, fs in by_group.items() if _group_label(g, fs) is Label.NORMAL}
    defect_groups = {g: fs for g, fs in by_group.items() if _group_label(g, fs) is Label.DEFECT}

    if len(normal_groups) < 3:
        raise InsufficientGroupsError(
            f"se necesitan al menos 3 piezas normales para poblar train/val/test, "
            f"hay {len(normal_groups)}. Ver seccion 8 del protocolo de evaluacion: "
            f"el objetivo es >= 30 piezas normales por clase."
        )

    assignment: dict[GroupId, Split] = {}
    assignment.update(
        _assign_deficit_greedy(
            normal_groups,
            targets={Split.TRAIN: ratios.train, Split.VAL: ratios.val, Split.TEST: ratios.test},
            rng=rng,
        )
    )
    assignment.update(
        _assign_deficit_greedy(
            defect_groups,
            targets={Split.VAL: defect_val_fraction, Split.TEST: 1.0 - defect_val_fraction},
            rng=rng,
        )
    )

    manifest = _materialise(
        assignment, by_group, strategy=SplitStrategy.BY_GROUP, seed=seed, ratios=ratios
    )
    assert_no_group_leakage(manifest)
    return manifest


def split_naive_by_frame(
    frames: Sequence[FrameRef],
    *,
    ratios: SplitRatios = DEFAULT_RATIOS,
    seed: int = 0,
    defect_val_fraction: float = 0.3,
) -> SplitManifest:
    """Particiona frames individuales al azar, ignorando la pieza de origen.

    BASELINE DEL EXPERIMENTO E3. No usar para producir metricas reportadas como
    validas: por construccion filtra informacion de train a test.

    Los frames de piezas defectuosas se excluyen de TRAIN igualmente (el modo es
    no supervisado), de modo que la unica diferencia respecto de `split_by_group`
    sea la unidad de reparto y el contraste quede aislado.

    `defect_val_fraction=0.0` manda todos los defectos a TEST. Es lo que pide el
    experimento E2, donde no se calibra ningun umbral y VAL no necesita positivos.
    """
    if not 0.0 <= defect_val_fraction <= 1.0:
        raise ValueError(f"defect_val_fraction fuera de [0,1]: {defect_val_fraction}")

    rng = random.Random(seed)

    normals = [f for f in frames if f.label is Label.NORMAL]
    defects = [f for f in frames if f.label is Label.DEFECT]
    rng.shuffle(normals)
    rng.shuffle(defects)

    n = len(normals)
    n_train = round(n * ratios.train)
    n_val = round(n * ratios.val)
    buckets: dict[Split, list[FrameRef]] = {
        Split.TRAIN: normals[:n_train],
        Split.VAL: normals[n_train : n_train + n_val],
        Split.TEST: normals[n_train + n_val :],
    }

    cut = round(len(defects) * defect_val_fraction)
    buckets[Split.VAL].extend(defects[:cut])
    buckets[Split.TEST].extend(defects[cut:])

    return SplitManifest(
        strategy=SplitStrategy.NAIVE_BY_FRAME,
        seed=seed,
        assignments={s: tuple(v) for s, v in buckets.items()},
        metadata={
            "ratios": f"{ratios.train}/{ratios.val}/{ratios.test}",
            "warning": "particion ingenua: produce fuga por construccion, solo baseline",
        },
    )


def audit_leakage(manifest: SplitManifest) -> LeakageReport:
    """Cuantifica la fuga de un manifiesto sin entrenar ningun modelo.

    `contaminated_test_fraction` es la cifra que hace visible el fenomeno: la
    proporcion del test normal que proviene de piezas ya vistas en TRAIN. Ver la
    justificacion del denominador en `LeakageReport`.
    """
    train_groups = manifest.groups(Split.TRAIN)
    test_frames = manifest.frames(Split.TEST)

    return LeakageReport(
        shared_groups_train_test=len(train_groups & manifest.groups(Split.TEST)),
        shared_groups_train_val=len(train_groups & manifest.groups(Split.VAL)),
        contaminated_test_frames=sum(1 for f in test_frames if f.group_id in train_groups),
        normal_test_frames=sum(1 for f in test_frames if f.label is Label.NORMAL),
        total_test_frames=len(test_frames),
    )


def assert_no_group_leakage(manifest: SplitManifest) -> None:
    """Control obligatorio de la seccion 10 del protocolo de evaluacion."""
    report = audit_leakage(manifest)
    if not report.is_clean:
        raise AssertionError(
            f"fuga detectada en un manifiesto que deberia estar limpio: "
            f"{report.shared_groups_train_test} grupos compartidos train/test, "
            f"{report.shared_groups_train_val} train/val"
        )


def write_manifest(manifest: SplitManifest, destination: Path) -> Path:
    """Serializa la particion como lista explicita de rutas.

    Deliberadamente NO se archiva solo la semilla: reproducir un resultado dentro
    de dos anos no debe depender de que la implementacion del generador aleatorio
    de la libreria estandar no haya cambiado.
    """
    payload = {
        "strategy": str(manifest.strategy),
        "seed": manifest.seed,
        "metadata": manifest.metadata,
        "frame_counts": {str(k): v for k, v in manifest.frame_counts().items()},
        "group_counts": {str(k): v for k, v in manifest.group_counts().items()},
        "leakage": _leakage_payload(manifest),
        "assignments": {
            str(split): [
                {
                    "path": f.path,
                    "group_id": str(f.group_id),
                    "label": str(f.label),
                    "source_video": f.source_video,
                    "frame_index": f.frame_index,
                    "timestamp_s": f.timestamp_s,
                }
                for f in manifest.frames(split)
            ]
            for split in Split
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return destination


def read_manifest(source: Path) -> SplitManifest:
    payload = json.loads(source.read_text(encoding="utf-8"))
    return SplitManifest(
        strategy=SplitStrategy(payload["strategy"]),
        seed=int(payload["seed"]),
        assignments={
            Split(split): tuple(
                FrameRef(
                    path=item["path"],
                    group_id=GroupId(item["group_id"]),
                    label=Label(item["label"]),
                    source_video=item["source_video"],
                    frame_index=int(item["frame_index"]),
                    timestamp_s=float(item["timestamp_s"]),
                )
                for item in items
            )
            for split, items in payload["assignments"].items()
        },
        metadata=dict(payload.get("metadata", {})),
    )


# --- internos -------------------------------------------------------------


def _group_frames(frames: Iterable[FrameRef]) -> dict[GroupId, list[FrameRef]]:
    grouped: dict[GroupId, list[FrameRef]] = defaultdict(list)
    for frame in frames:
        grouped[frame.group_id].append(frame)
    return dict(grouped)


def _group_label(group_id: GroupId, frames: Sequence[FrameRef]) -> Label:
    """Una pieza defectuosa lo es en todos sus frames; se valida la consistencia."""
    labels = {f.label for f in frames}
    if len(labels) > 1:
        raise ValueError(
            f"la pieza {group_id} tiene frames con etiquetas distintas ({labels}). "
            f"La etiqueta es una propiedad de la pieza, no del frame."
        )
    return labels.pop()


def _assign_deficit_greedy(
    groups: dict[GroupId, list[FrameRef]],
    *,
    targets: dict[Split, float],
    rng: random.Random,
) -> dict[GroupId, Split]:
    """Asigna cada grupo al split con mayor deficit de frames respecto del objetivo.

    El orden de recorrido es aleatorio (de ahi la variabilidad entre replicas) y
    el destino es determinista dado el orden, lo que mantiene el equilibrio de
    frames sin romper ningun grupo.
    """
    if not groups:
        return {}

    order = list(groups)
    rng.shuffle(order)

    total_frames = sum(len(v) for v in groups.values())
    current: dict[Split, int] = dict.fromkeys(targets, 0)
    assignment: dict[GroupId, Split] = {}

    for group_id in order:
        size = len(groups[group_id])
        chosen = max(targets, key=lambda s: targets[s] * total_frames - current[s])
        assignment[group_id] = chosen
        current[chosen] += size

    # Un split con objetivo 0 debe quedar vacio: es lo que pide E2 al mandar
    # todos los defectos a TEST. Solo es un error que quede vacio un split que
    # si pedia muestras.
    empty = [s for s, n in current.items() if n == 0 and targets[s] > 0]
    if empty:
        raise InsufficientGroupsError(
            f"los splits {[str(s) for s in empty]} quedaron vacios con "
            f"{len(groups)} piezas. Se necesitan mas piezas fisicas; ver seccion 8 "
            f"del protocolo de evaluacion."
        )
    return assignment


def _materialise(
    assignment: dict[GroupId, Split],
    by_group: dict[GroupId, list[FrameRef]],
    *,
    strategy: SplitStrategy,
    seed: int,
    ratios: SplitRatios,
) -> SplitManifest:
    buckets: dict[Split, list[FrameRef]] = {s: [] for s in Split}
    for group_id, split in assignment.items():
        buckets[split].extend(by_group[group_id])
    return SplitManifest(
        strategy=strategy,
        seed=seed,
        assignments={s: tuple(v) for s, v in buckets.items()},
        metadata={"ratios": f"{ratios.train}/{ratios.val}/{ratios.test}"},
    )


def _leakage_payload(manifest: SplitManifest) -> dict[str, float | int]:
    report = audit_leakage(manifest)
    return {
        "shared_groups_train_test": report.shared_groups_train_test,
        "shared_groups_train_val": report.shared_groups_train_val,
        "contaminated_test_frames": report.contaminated_test_frames,
        "normal_test_frames": report.normal_test_frames,
        "total_test_frames": report.total_test_frames,
        "contaminated_test_fraction": round(report.contaminated_test_fraction, 6),
    }
