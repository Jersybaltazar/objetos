"""Metricas de deteccion.

Operan sobre vectores de score y etiqueta, no sobre modelos. Esa separacion es
deliberada: las cifras que se reportan en la tesis se calculan con codigo propio
y testeado, no con los defaults de la libreria de entrenamiento. Cambiar
PatchCore por PaDiM, o anomalib por cualquier otra cosa, no cambia como se mide.

La metrica principal del protocolo es el **AUROC a nivel de imagen**: es
independiente del umbral, es la metrica estandar de la literatura de deteccion de
anomalias industrial —lo que hace los resultados comparables— y solo exige una
etiqueta binaria por muestra, sin anotar mascaras.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

Scores = npt.NDArray[np.float64]
Labels = npt.NDArray[np.bool_]


class DegenerateLabelsError(ValueError):
    """El conjunto no tiene ambas clases y la metrica no esta definida."""


@dataclass(frozen=True, slots=True)
class Confusion:
    """Matriz de confusion en un umbral concreto. Positivo = defecto."""

    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int

    @property
    def recall(self) -> float:
        """Proporcion de defectos detectados.

        Es la cifra que le importa al usuario industrial: un defecto no detectado
        llega al cliente, un falso positivo solo cuesta una revision.
        """
        actual = self.true_positives + self.false_negatives
        return self.true_positives / actual if actual else 0.0

    @property
    def precision(self) -> float:
        flagged = self.true_positives + self.false_positives
        return self.true_positives / flagged if flagged else 0.0

    @property
    def f1(self) -> float:
        denominator = self.precision + self.recall
        return 2 * self.precision * self.recall / denominator if denominator else 0.0

    @property
    def false_alarm_rate(self) -> float:
        """Proporcion de piezas buenas marcadas como defectuosas."""
        actual = self.false_positives + self.true_negatives
        return self.false_positives / actual if actual else 0.0

    def summary(self) -> dict[str, float | int]:
        return {
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
            "recall": round(self.recall, 6),
            "precision": round(self.precision, 6),
            "f1": round(self.f1, 6),
            "false_alarm_rate": round(self.false_alarm_rate, 6),
        }


def auroc(scores: Scores, labels: Labels) -> float:
    """Area bajo la curva ROC, por la identidad con el estadistico de Mann-Whitney.

    Se calcula por rangos y no integrando la curva porque el metodo de rangos
    trata los empates de forma exacta: dos frames con identico score de anomalia,
    uno bueno y uno defectuoso, contribuyen 0.5 en lugar de 0 o 1 segun el orden
    accidental del vector. Con scores cuantizados los empates son frecuentes y el
    sesgo seria sistematico.
    """
    scores, labels = _validated(scores, labels)

    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        raise DegenerateLabelsError(
            f"el AUROC exige ambas clases; hay {positives} defectos y {negatives} normales"
        )

    ranks = average_ranks(scores)
    rank_sum = float(ranks[labels].sum())
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def confusion_at(scores: Scores, labels: Labels, threshold: float) -> Confusion:
    """Matriz de confusion marcando como defecto todo score `>= threshold`."""
    scores, labels = _validated(scores, labels)
    flagged = scores >= threshold

    return Confusion(
        true_positives=int(np.count_nonzero(flagged & labels)),
        false_positives=int(np.count_nonzero(flagged & ~labels)),
        true_negatives=int(np.count_nonzero(~flagged & ~labels)),
        false_negatives=int(np.count_nonzero(~flagged & labels)),
    )


def average_ranks(values: Scores) -> npt.NDArray[np.float64]:
    """Rangos 1-based con promedio en los empates.

    Publica porque el modulo de estadistica la reutiliza para la correlacion
    biserial de rangos: una sola implementacion, un solo sitio donde equivocarse.
    """
    order = np.argsort(values, kind="mergesort")
    _, inverse, counts = np.unique(values[order], return_inverse=True, return_counts=True)

    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    averaged = starts[inverse] + (counts[inverse] + 1) / 2.0

    ranks = np.empty_like(averaged, dtype=np.float64)
    ranks[order] = averaged
    return ranks


def _validated(scores: Scores, labels: Labels) -> tuple[Scores, Labels]:
    scores_array = np.asarray(scores, dtype=np.float64).ravel()
    labels_array = np.asarray(labels, dtype=np.bool_).ravel()

    if scores_array.size != labels_array.size:
        raise ValueError(
            f"scores y labels deben tener el mismo tamano: "
            f"{scores_array.size} vs {labels_array.size}"
        )
    if scores_array.size == 0:
        raise ValueError("no hay muestras que evaluar")
    if not np.isfinite(scores_array).all():
        raise ValueError("hay scores no finitos (NaN o infinito)")
    return scores_array, labels_array
