"""Pruebas estadisticas del protocolo de evaluacion (seccion 9).

Contrastes soportados:

- **He1** (la particion ingenua infla la metrica): Wilcoxon de rangos con signo,
  pareado por semilla de particion, con correccion de Holm-Bonferroni sobre las
  multiples categorias y tamano del efecto obligatorio junto al p-valor.
- **He2** (la deduplicacion no degrada): prueba de **equivalencia**, no de
  diferencia. Afirmar "no degrada" a partir de un p-valor no significativo es un
  error: ausencia de evidencia no es evidencia de ausencia. Se comprueba que el
  intervalo de confianza de la diferencia queda contenido en el margen declarado.

Se reporta siempre el tamano del efecto ademas del p-valor. Con suficientes
replicas, una diferencia de 0.3 pp de AUROC sale significativa y es irrelevante
en la practica; el umbral de relevancia preregistrado del protocolo es 2 pp.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from inspeccion.application.evaluation.metrics import average_ranks

Samples = npt.NDArray[np.float64]

NEGLIGIBLE, SMALL, MEDIUM = 0.147, 0.33, 0.474
"""Umbrales de magnitud del tamano del efecto (Romano et al.)."""


@dataclass(frozen=True, slots=True)
class PairedComparison:
    """Contraste pareado entre dos condiciones medidas sobre las mismas replicas."""

    n: int
    mean_difference: float
    median_difference: float
    ci_low: float
    ci_high: float
    p_value: float
    rank_biserial: float
    cliffs_delta: float

    @property
    def effect_magnitude(self) -> str:
        magnitude = abs(self.rank_biserial)
        if magnitude < NEGLIGIBLE:
            return "despreciable"
        if magnitude < SMALL:
            return "pequeno"
        if magnitude < MEDIUM:
            return "mediano"
        return "grande"

    def is_significant(self, *, alpha: float = 0.05, min_effect: float = 0.0) -> bool:
        """Significativo **y** practicamente relevante.

        `min_effect` es el umbral de relevancia preregistrado (0.02 = 2 pp de
        AUROC). Sin el, la significancia estadistica por si sola no autoriza a
        concluir nada sobre el sistema real.
        """
        return self.p_value < alpha and abs(self.mean_difference) >= min_effect

    def summary(self) -> dict[str, float | int | str]:
        return {
            "n": self.n,
            "mean_difference": round(self.mean_difference, 6),
            "median_difference": round(self.median_difference, 6),
            "ci95": f"[{self.ci_low:.6f}, {self.ci_high:.6f}]",
            "p_value": self.p_value,
            "rank_biserial": round(self.rank_biserial, 6),
            "cliffs_delta": round(self.cliffs_delta, 6),
            "effect_magnitude": self.effect_magnitude,
        }


@dataclass(frozen=True, slots=True)
class EquivalenceResult:
    """Resultado de una prueba de equivalencia con margen declarado."""

    n: int
    mean_difference: float
    ci_low: float
    ci_high: float
    margin: float

    @property
    def is_equivalent(self) -> bool:
        """El IC95 completo cae dentro de [-margen, +margen]."""
        return -self.margin <= self.ci_low and self.ci_high <= self.margin

    @property
    def is_inconclusive(self) -> bool:
        """El IC95 se sale del margen pero tambien contiene el cero.

        Ni equivalencia ni diferencia: hacen falta mas replicas. Distinguirlo de
        una degradacion real evita concluir de mas.
        """
        return not self.is_equivalent and self.ci_low <= 0.0 <= self.ci_high

    def summary(self) -> dict[str, float | int | str | bool]:
        return {
            "n": self.n,
            "mean_difference": round(self.mean_difference, 6),
            "ci95": f"[{self.ci_low:.6f}, {self.ci_high:.6f}]",
            "margin": self.margin,
            "is_equivalent": self.is_equivalent,
            "is_inconclusive": self.is_inconclusive,
        }


def paired_comparison(
    treatment: Samples,
    control: Samples,
    *,
    n_bootstrap: int = 10_000,
    seed: int = 0,
) -> PairedComparison:
    """Contrasta `treatment` contra `control` medidos sobre las mismas replicas.

    Para He1: `treatment` son los AUROC del split ingenuo y `control` los del
    split por pieza, emparejados por semilla. Una diferencia media positiva
    significa que la particion ingenua reporta mas de lo que el modelo vale.
    """
    a, b = _paired(treatment, control)
    differences = a - b

    return PairedComparison(
        n=int(differences.size),
        mean_difference=float(differences.mean()),
        median_difference=float(np.median(differences)),
        **_bootstrap_ci(differences, n_bootstrap=n_bootstrap, seed=seed),
        p_value=_wilcoxon_p(differences),
        rank_biserial=rank_biserial(a, b),
        cliffs_delta=cliffs_delta(a, b),
    )


def equivalence(
    treatment: Samples,
    control: Samples,
    *,
    margin: float,
    n_bootstrap: int = 10_000,
    seed: int = 0,
) -> EquivalenceResult:
    """Prueba de equivalencia para He2.

    Args:
        margin: margen de equivalencia en las unidades de la metrica. El protocolo
            declara 0.02 (2 pp de AUROC).
    """
    if margin <= 0:
        raise ValueError(f"el margen de equivalencia debe ser positivo, llego {margin}")

    a, b = _paired(treatment, control)
    differences = a - b
    bounds = _bootstrap_ci(differences, n_bootstrap=n_bootstrap, seed=seed)

    return EquivalenceResult(
        n=int(differences.size),
        mean_difference=float(differences.mean()),
        margin=margin,
        **bounds,
    )


def holm_bonferroni(p_values: Samples) -> npt.NDArray[np.float64]:
    """Correccion de Holm-Bonferroni (descendente) para comparaciones multiples.

    El protocolo contrasta varias categorias y varios valores de duplicacion. Sin
    correccion, la probabilidad de al menos un falso positivo se acumula con cada
    contraste adicional.
    """
    values = np.asarray(p_values, dtype=np.float64).ravel()
    if values.size == 0:
        return values
    if np.any((values < 0) | (values > 1)):
        raise ValueError("los p-valores deben estar en [0, 1]")

    order = np.argsort(values, kind="mergesort")
    ascending = values[order]
    factors = values.size - np.arange(values.size)

    # Paso descendente: cada p-valor se escala por el numero de contrastes que
    # quedan, y se acumula el maximo para preservar la monotonia.
    corrected_sorted = np.minimum(np.maximum.accumulate(ascending * factors), 1.0)

    corrected = np.empty_like(corrected_sorted)
    corrected[order] = corrected_sorted
    return corrected


def cliffs_delta(treatment: Samples, control: Samples) -> float:
    """Delta de Cliff: tamano del efecto para muestras **independientes**.

    Se incluye porque es la medida que nombra el protocolo, pero para un diseno
    pareado la companera correcta es `rank_biserial`: la delta de Cliff ignora el
    emparejamiento y por tanto subestima el efecto cuando la varianza entre
    replicas es alta. Se reportan las dos.
    """
    a = np.asarray(treatment, dtype=np.float64).ravel()
    b = np.asarray(control, dtype=np.float64).ravel()
    if a.size == 0 or b.size == 0:
        raise ValueError("ambas muestras deben tener elementos")

    comparisons = np.sign(a[:, None] - b[None, :])
    return float(comparisons.sum() / (a.size * b.size))


def rank_biserial(treatment: Samples, control: Samples) -> float:
    """Correlacion biserial de rangos para pares emparejados.

    Es el tamano del efecto que acompana al Wilcoxon de rangos con signo:
    proporcion de la suma de rangos que favorece al tratamiento, en [-1, 1].
    """
    a, b = _paired(treatment, control)
    differences = a - b
    nonzero = differences[differences != 0]
    if nonzero.size == 0:
        return 0.0

    ranks = average_ranks(np.abs(nonzero))
    total = ranks.sum()
    positive = ranks[nonzero > 0].sum()
    negative = ranks[nonzero < 0].sum()
    return float((positive - negative) / total)


# --- internos -------------------------------------------------------------


def _paired(treatment: Samples, control: Samples) -> tuple[Samples, Samples]:
    a = np.asarray(treatment, dtype=np.float64).ravel()
    b = np.asarray(control, dtype=np.float64).ravel()

    if a.size != b.size:
        raise ValueError(
            f"un contraste pareado exige el mismo numero de replicas: {a.size} vs {b.size}"
        )
    if a.size < 2:
        raise ValueError(f"hacen falta al menos 2 replicas, hay {a.size}")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("hay valores no finitos entre las replicas")
    return a, b


def _wilcoxon_p(differences: Samples) -> float:
    """Wilcoxon de rangos con signo. No parametrico: no asume normalidad, que es
    lo apropiado para las 10 replicas del protocolo."""
    from scipy.stats import wilcoxon

    if np.all(differences == 0):
        return 1.0
    result = wilcoxon(differences, alternative="two-sided", zero_method="wilcox")
    return float(result.pvalue)


def _bootstrap_ci(differences: Samples, *, n_bootstrap: int, seed: int) -> dict[str, float]:
    """IC95 percentil de la diferencia media por bootstrap.

    Se usa bootstrap y no el intervalo normal porque con 10 replicas el supuesto
    de normalidad no es verificable, y el percentil no lo necesita.
    """
    if n_bootstrap < 100:
        raise ValueError(f"n_bootstrap demasiado pequeno: {n_bootstrap}")

    rng = np.random.default_rng(seed)
    draws = rng.choice(differences, size=(n_bootstrap, differences.size), replace=True)
    means = draws.mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {"ci_low": float(low), "ci_high": float(high)}
