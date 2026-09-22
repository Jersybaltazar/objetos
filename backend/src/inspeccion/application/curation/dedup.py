"""Deduplicacion de frames por hashing perceptual.

Opera sobre hashes ya calculados, no sobre imagenes, para que el algoritmo sea
testeable sin dependencias de decodificacion y para que la eleccion del hash
(pHash, dHash, wHash, embeddings binarizados) sea intercambiable.

Dos decisiones de diseno que conviene justificar:

1. **La deduplicacion es por defecto INTRA-GRUPO.** Comparar frames de piezas
   distintas y eliminar los parecidos destruiria la variabilidad inter-pieza, que
   es justamente lo que el modelo de anomalia necesita aprender como "normal".
   Ademas, deduplicar globalmente antes de particionar eliminaria de test los
   frames que duplican a los de train, ocultando la fuga en vez de corregirla
   (seccion 10 del protocolo de evaluacion).

2. **Se conserva el primer frame de cada cluster, no el "mejor".** Cualquier
   criterio de calidad introduciria un sesgo de seleccion dificil de justificar.
   El orden de recorrido es temporal y determinista.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from inspeccion.domain.models import FrameRef, GroupId

_POPCOUNT: npt.NDArray[np.uint8] = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


@dataclass(frozen=True, slots=True)
class HashedFrame:
    frame: FrameRef
    phash: int

    def __post_init__(self) -> None:
        if self.phash < 0:
            raise ValueError("el hash debe ser un entero sin signo")


@dataclass(frozen=True, slots=True)
class DedupResult:
    kept: tuple[FrameRef, ...]
    removed: tuple[FrameRef, ...]
    max_distance: int

    @property
    def total(self) -> int:
        return len(self.kept) + len(self.removed)

    @property
    def retained_fraction(self) -> float:
        return len(self.kept) / self.total if self.total else 1.0

    @property
    def removed_fraction(self) -> float:
        return 1.0 - self.retained_fraction

    def summary(self) -> dict[str, float | int]:
        """Fila de la tabla del experimento E4."""
        return {
            "max_distance": self.max_distance,
            "total": self.total,
            "kept": len(self.kept),
            "removed": len(self.removed),
            "retained_fraction": round(self.retained_fraction, 6),
        }


def deduplicate(
    items: Sequence[HashedFrame],
    *,
    max_distance: int,
    within_group: bool = True,
) -> DedupResult:
    """Elimina frames cuya distancia de Hamming a un frame ya conservado sea <= `max_distance`.

    `max_distance=0` conserva todo salvo duplicados exactos de hash. Valores
    tipicos del barrido de E4 sobre pHash de 64 bits: 0, 2, 4, 6, 8, 10, 12.
    """
    if max_distance < 0:
        raise ValueError(f"max_distance no puede ser negativo: {max_distance}")

    if not within_group:
        kept, removed = _dedup_one_pool(items, max_distance)
        return DedupResult(tuple(kept), tuple(removed), max_distance)

    by_group: dict[GroupId, list[HashedFrame]] = defaultdict(list)
    for item in items:
        by_group[item.frame.group_id].append(item)

    kept_all: list[FrameRef] = []
    removed_all: list[FrameRef] = []
    for group_items in by_group.values():
        kept, removed = _dedup_one_pool(group_items, max_distance)
        kept_all.extend(kept)
        removed_all.extend(removed)

    return DedupResult(tuple(kept_all), tuple(removed_all), max_distance)


def redundancy_profile(
    items: Sequence[HashedFrame],
    *,
    distances: Sequence[int] = (0, 2, 4, 6, 8, 10, 12),
) -> list[dict[str, float | int]]:
    """Barrido completo del umbral: produce directamente la tabla de E4."""
    return [deduplicate(items, max_distance=d).summary() for d in distances]


def hamming(a: int, b: int) -> int:
    """Distancia de Hamming entre dos hashes empaquetados como enteros."""
    return (a ^ b).bit_count()


# --- internos -------------------------------------------------------------


def _dedup_one_pool(
    items: Sequence[HashedFrame], max_distance: int
) -> tuple[list[FrameRef], list[FrameRef]]:
    """Avaro sobre el orden temporal, comparando contra TODOS los conservados.

    Comparar solo contra el ultimo conservado seria O(n) pero fallaria con una
    pieza en rotacion: al volver a una orientacion ya vista, el frame es un
    duplicado de uno antiguo, no del inmediatamente anterior.
    """
    ordered = sorted(items, key=lambda i: (i.frame.source_video, i.frame.frame_index))
    if not ordered:
        return [], []

    kept: list[FrameRef] = [ordered[0].frame]
    kept_hashes: list[int] = [ordered[0].phash]
    removed: list[FrameRef] = []

    for item in ordered[1:]:
        if _min_distance(item.phash, kept_hashes) <= max_distance:
            removed.append(item.frame)
        else:
            kept.append(item.frame)
            kept_hashes.append(item.phash)

    return kept, removed


def _min_distance(value: int, pool: Sequence[int]) -> int:
    if len(pool) < 64:  # por debajo de este tamano el bucle puro es mas rapido
        return min((value ^ other).bit_count() for other in pool)

    arr = np.array(pool, dtype=np.uint64)
    xor = np.bitwise_xor(arr, np.uint64(value))
    counts = _POPCOUNT[xor.view(np.uint8).reshape(-1, 8)].sum(axis=1)
    return int(counts.min())
