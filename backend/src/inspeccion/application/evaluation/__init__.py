"""Evaluacion: metricas, calibracion de umbral y pruebas estadisticas.

Independiente de la libreria de entrenamiento a proposito. Las cifras que van a
la tesis se calculan aqui, con codigo propio y testeado, no con los defaults de
anomalib. Eso hace que cambiar de modelo no cambie como se mide, y —mas
importante— pone bajo control propio la regla del protocolo que prohibe calibrar
el umbral sobre el conjunto de test.
"""

from inspeccion.application.evaluation.metrics import (
    Confusion,
    DegenerateLabelsError,
    auroc,
    average_ranks,
    confusion_at,
)
from inspeccion.application.evaluation.statistics import (
    EquivalenceResult,
    PairedComparison,
    cliffs_delta,
    equivalence,
    holm_bonferroni,
    paired_comparison,
    rank_biserial,
)
from inspeccion.application.evaluation.thresholding import (
    CalibratedThreshold,
    ThresholdLeakageError,
    calibrate_f1,
)

__all__ = [
    "CalibratedThreshold",
    "Confusion",
    "DegenerateLabelsError",
    "EquivalenceResult",
    "PairedComparison",
    "ThresholdLeakageError",
    "auroc",
    "average_ranks",
    "calibrate_f1",
    "cliffs_delta",
    "confusion_at",
    "equivalence",
    "holm_bonferroni",
    "paired_comparison",
    "rank_biserial",
]
