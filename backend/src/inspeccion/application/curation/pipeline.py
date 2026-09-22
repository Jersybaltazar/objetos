"""Orquestacion de la curacion: el orden de operaciones, hecho codigo.

    extraer -> agrupar por pieza -> PARTICIONAR por grupo -> deduplicar dentro del split

El orden no es configurable a proposito. Deduplicar antes de particionar
eliminaria del conjunto de test los frames que duplican a los de entrenamiento:
la fuga desapareceria de la medicion sin desaparecer del experimento, que es el
peor resultado posible para esta tesis (seccion 10 del protocolo de evaluacion).
`run()` es el unico camino, y ejecuta las etapas en ese orden.

Patron Pipeline con seleccion de estrategia de particion inyectada, de modo que
el experimento E3 contrasta ambas ejecutando el mismo objeto con distinta
estrategia y la misma semilla.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from inspeccion.application.curation.dedup import DedupResult, HashedFrame, deduplicate
from inspeccion.application.curation.splitting import (
    DEFAULT_RATIOS,
    audit_leakage,
    split_by_group,
    split_naive_by_frame,
)
from inspeccion.domain.models import (
    FrameRef,
    LeakageReport,
    Split,
    SplitManifest,
    SplitRatios,
    SplitStrategy,
)

SplitFunction = Callable[..., SplitManifest]

_STRATEGIES: dict[SplitStrategy, SplitFunction] = {
    SplitStrategy.BY_GROUP: split_by_group,
    SplitStrategy.NAIVE_BY_FRAME: split_naive_by_frame,
}


@dataclass(frozen=True, slots=True)
class CurationReport:
    """Todo lo que el experimento necesita registrar de una corrida de curacion."""

    manifest: SplitManifest
    leakage: LeakageReport
    frames_before_dedup: dict[Split, int]
    dedup: dict[Split, dict[str, float | int]]

    @property
    def frames_after_dedup(self) -> dict[Split, int]:
        return self.manifest.frame_counts()

    def summary(self) -> dict[str, object]:
        """Fila de resultados lista para MLflow o para la tabla de la tesis."""
        return {
            "strategy": str(self.manifest.strategy),
            "seed": self.manifest.seed,
            "frames_before_dedup": {str(k): v for k, v in self.frames_before_dedup.items()},
            "frames_after_dedup": {str(k): v for k, v in self.frames_after_dedup.items()},
            "group_counts": {str(k): v for k, v in self.manifest.group_counts().items()},
            "contaminated_test_fraction": round(self.leakage.contaminated_test_fraction, 6),
            "dedup": {str(k): v for k, v in self.dedup.items()},
        }


@dataclass(frozen=True, slots=True)
class CurationPipeline:
    """Caso de uso: de frames hasheados a una particion curada y auditada.

    Args:
        dedup_max_distance: umbral de Hamming. Es la variable que barre E4.
        dedup_splits: sobre que splits se deduplica. Por defecto los tres, segun
            la seccion 10 del protocolo. Restringirlo a TRAIN es legitimo cuando
            se quiere aislar el efecto de la deduplicacion sobre el entrenamiento
            dejando el conjunto de evaluacion intacto; la eleccion debe declararse
            junto a los resultados porque cambia lo que significa la metrica.
        ratios: proporciones de la particion sobre las piezas normales.
    """

    dedup_max_distance: int = 4
    dedup_splits: frozenset[Split] = field(default_factory=lambda: frozenset(Split))
    ratios: SplitRatios = DEFAULT_RATIOS

    def run(
        self,
        frames: Sequence[HashedFrame],
        *,
        strategy: SplitStrategy = SplitStrategy.BY_GROUP,
        seed: int = 0,
    ) -> CurationReport:
        """Particiona y despues deduplica. En ese orden, siempre."""
        if not frames:
            raise ValueError("no hay frames que curar")

        split_fn = _STRATEGIES[strategy]
        manifest = split_fn([item.frame for item in frames], ratios=self.ratios, seed=seed)

        by_path = {item.frame.path: item for item in frames}
        before: dict[Split, int] = {}
        after: dict[Split, tuple[FrameRef, ...]] = {}
        summaries: dict[Split, dict[str, float | int]] = {}

        for split in Split:
            items = [by_path[frame.path] for frame in manifest.frames(split)]
            before[split] = len(items)
            result = self._dedup(split, items)
            after[split] = result.kept
            summaries[split] = result.summary()

        curated = SplitManifest(
            strategy=manifest.strategy,
            seed=manifest.seed,
            assignments=after,
            metadata={
                **manifest.metadata,
                "dedup_max_distance": str(self.dedup_max_distance),
                "dedup_splits": ",".join(sorted(str(s) for s in self.dedup_splits)),
                "frames_before_dedup": ",".join(f"{s}={before[s]}" for s in Split),
            },
        )

        return CurationReport(
            manifest=curated,
            leakage=audit_leakage(curated),
            frames_before_dedup=before,
            dedup=summaries,
        )

    def _dedup(self, split: Split, items: Sequence[HashedFrame]) -> DedupResult:
        """La deduplicacion es intra-split e intra-pieza: nunca compara entre
        splits, porque hacerlo volveria a mezclar la informacion que la particion
        acaba de separar."""
        distance = self.dedup_max_distance if split in self.dedup_splits else -1
        if distance < 0:
            return DedupResult(kept=tuple(i.frame for i in items), removed=(), max_distance=0)
        return deduplicate(items, max_distance=distance, within_group=True)
