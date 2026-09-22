"""Calibracion del umbral de decision (seccion 7 del protocolo de evaluacion).

Elegir el umbral que maximiza F1 **mirando el conjunto de test** es ajustar un
parametro sobre el test. Es una fuga de informacion por la puerta de atras, y en
una tesis cuyo objeto de estudio es precisamente la fuga de informacion seria un
defecto fatal.

Varias librerias de deteccion de anomalias ofrecen un umbral "adaptativo" que,
segun como se configure el flujo, puede acabar calculandose sobre el conjunto de
test. Por eso el umbral se calibra aqui y no se delega: `calibrate_f1` **exige**
declarar sobre que split se esta calibrando y rechaza cualquiera que no sea VAL.
La regla deja de depender de la disciplina de quien ejecuta el experimento.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from inspeccion.application.evaluation.metrics import (
    Confusion,
    Labels,
    Scores,
    confusion_at,
)
from inspeccion.domain.models import Split


class ThresholdLeakageError(RuntimeError):
    """Se intento calibrar el umbral sobre un split que no es de validacion."""


@dataclass(frozen=True, slots=True)
class CalibratedThreshold:
    """Umbral congelado. Una vez construido, se aplica a test sin reajuste."""

    value: float
    f1_on_val: float
    candidates_evaluated: int

    def apply(self, scores: Scores) -> np.ndarray:
        """Marca como defecto todo score `>= value`."""
        return np.asarray(scores, dtype=np.float64) >= self.value

    def evaluate(self, scores: Scores, labels: Labels) -> Confusion:
        return confusion_at(scores, labels, self.value)

    def summary(self) -> dict[str, float | int]:
        return {
            "threshold": self.value,
            "f1_on_val": round(self.f1_on_val, 6),
            "candidates_evaluated": self.candidates_evaluated,
        }


def calibrate_f1(scores: Scores, labels: Labels, *, split: Split) -> CalibratedThreshold:
    """Umbral que maximiza F1 sobre el conjunto de **validacion**.

    Args:
        split: sobre que split se esta calibrando. No tiene valor por defecto a
            proposito: obliga a quien llama a declararlo, y cualquier valor que no
            sea VAL levanta `ThresholdLeakageError`.

    Los candidatos son los propios scores observados: el F1 solo cambia cuando el
    umbral cruza un score, asi que evaluar valores intermedios no encontraria
    nada nuevo y solo daria una falsa sensacion de busqueda fina.
    """
    if split is not Split.VAL:
        raise ThresholdLeakageError(
            f"el umbral solo puede calibrarse sobre {Split.VAL}, se intento sobre {split}. "
            f"Calibrar sobre test seria ajustar un parametro mirando el test "
            f"(seccion 7 del protocolo de evaluacion)."
        )

    candidates = np.unique(np.asarray(scores, dtype=np.float64))
    if candidates.size == 0:
        raise ValueError("no hay scores sobre los que calibrar")

    best_value = float(candidates[0])
    best_f1 = -1.0
    for candidate in candidates:
        f1 = confusion_at(scores, labels, float(candidate)).f1
        if f1 > best_f1:
            best_f1, best_value = f1, float(candidate)

    return CalibratedThreshold(
        value=best_value,
        f1_on_val=best_f1,
        candidates_evaluated=int(candidates.size),
    )
