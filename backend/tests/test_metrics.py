from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from inspeccion.application.evaluation.metrics import (
    DegenerateLabelsError,
    auroc,
    average_ranks,
    confusion_at,
)


class TestAuroc:
    def test_separacion_perfecta(self):
        assert auroc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1], dtype=bool)) == 1.0

    def test_separacion_invertida(self):
        assert auroc(np.array([0.9, 0.8, 0.2, 0.1]), np.array([0, 0, 1, 1], dtype=bool)) == 0.0

    def test_valor_calculado_a_mano(self):
        # 4 pares normal-defecto, 3 ordenados correctamente -> 0.75
        scores = np.array([0.1, 0.4, 0.35, 0.8])
        labels = np.array([0, 0, 1, 1], dtype=bool)

        assert auroc(scores, labels) == pytest.approx(0.75)

    def test_todos_los_scores_empatados_dan_media_moneda(self):
        assert auroc(np.full(6, 0.5), np.array([0, 0, 0, 1, 1, 1], dtype=bool)) == 0.5

    def test_un_empate_aporta_medio_punto(self):
        # Sin tratamiento exacto de empates, este caso daria 0.75 o 1.0 segun el
        # orden accidental del vector.
        scores = np.array([1.0, 2.0, 2.0, 3.0])
        labels = np.array([0, 0, 1, 1], dtype=bool)

        assert auroc(scores, labels) == pytest.approx(0.875)

    def test_es_invariante_a_transformaciones_monotonas(self):
        scores = np.array([0.1, 0.4, 0.35, 0.8])
        labels = np.array([0, 0, 1, 1], dtype=bool)

        assert auroc(scores, labels) == pytest.approx(auroc(scores * 100 + 7, labels))

    def test_exige_ambas_clases(self):
        with pytest.raises(DegenerateLabelsError, match="ambas clases"):
            auroc(np.array([0.1, 0.2]), np.array([0, 0], dtype=bool))

    def test_rechaza_tamanos_distintos(self):
        with pytest.raises(ValueError, match="mismo tamano"):
            auroc(np.array([0.1, 0.2, 0.3]), np.array([0, 1], dtype=bool))

    def test_rechaza_scores_no_finitos(self):
        # Un NaN silencioso de la inferencia falsearia toda la tabla.
        with pytest.raises(ValueError, match="no finitos"):
            auroc(np.array([0.1, np.nan]), np.array([0, 1], dtype=bool))

    def test_rechaza_entrada_vacia(self):
        with pytest.raises(ValueError, match="no hay muestras"):
            auroc(np.array([]), np.array([], dtype=bool))


@pytest.mark.skipif(
    importlib.util.find_spec("sklearn") is None,
    reason="scikit-learn es solo la referencia de validacion",
)
class TestContraReferenciaIndependiente:
    """Valida el AUROC propio contra `sklearn.metrics.roc_auc_score`.

    El AUROC de la tesis se calcula con codigo propio para no depender de los
    defaults de la libreria de entrenamiento. Esa decision obliga a demostrar que
    la implementacion es correcta, y la forma de demostrarlo es contrastarla con
    una referencia que nadie discute.
    """

    @pytest.mark.parametrize("seed", range(10))
    def test_coincide_en_datos_continuos(self, seed):
        from sklearn.metrics import roc_auc_score

        rng = np.random.default_rng(seed)
        labels = rng.integers(0, 2, 200).astype(bool)
        scores = rng.normal(labels * 0.8, 1.0, 200)

        assert auroc(scores, labels) == pytest.approx(roc_auc_score(labels, scores))

    @pytest.mark.parametrize("seed", range(10))
    def test_coincide_con_muchos_empates(self, seed):
        # Los scores cuantizados producen empates masivos: es el caso en el que
        # una implementacion ingenua se desvia de la referencia.
        from sklearn.metrics import roc_auc_score

        rng = np.random.default_rng(seed)
        labels = rng.integers(0, 2, 200).astype(bool)
        scores = np.round(rng.normal(labels * 0.5, 1.0, 200) * 2) / 2

        assert auroc(scores, labels) == pytest.approx(roc_auc_score(labels, scores))

    def test_coincide_con_clases_muy_desbalanceadas(self):
        # En inspeccion real los defectos son raros: 3 defectos entre 200 piezas.
        from sklearn.metrics import roc_auc_score

        rng = np.random.default_rng(7)
        labels = np.zeros(200, dtype=bool)
        labels[:3] = True
        scores = rng.normal(labels * 1.5, 1.0, 200)

        assert auroc(scores, labels) == pytest.approx(roc_auc_score(labels, scores))


class TestAverageRanks:
    def test_sin_empates(self):
        assert list(average_ranks(np.array([30.0, 10.0, 20.0]))) == [3.0, 1.0, 2.0]

    def test_promedia_los_empates(self):
        assert list(average_ranks(np.array([10.0, 10.0, 20.0]))) == [1.5, 1.5, 3.0]


class TestConfusion:
    def test_conteos(self):
        scores = np.array([0.1, 0.6, 0.7, 0.2])
        labels = np.array([0, 0, 1, 1], dtype=bool)

        matriz = confusion_at(scores, labels, threshold=0.5)

        assert matriz.true_positives == 1  # 0.7
        assert matriz.false_positives == 1  # 0.6
        assert matriz.true_negatives == 1  # 0.1
        assert matriz.false_negatives == 1  # 0.2

    def test_el_umbral_es_inclusivo(self):
        matriz = confusion_at(np.array([0.5, 0.4]), np.array([1, 1], dtype=bool), threshold=0.5)

        assert matriz.true_positives == 1

    def test_metricas_derivadas(self):
        scores = np.array([0.9, 0.8, 0.7, 0.1, 0.2])
        labels = np.array([1, 1, 0, 0, 0], dtype=bool)

        matriz = confusion_at(scores, labels, threshold=0.75)

        assert matriz.recall == pytest.approx(1.0)
        assert matriz.precision == pytest.approx(1.0)
        assert matriz.f1 == pytest.approx(1.0)
        assert matriz.false_alarm_rate == pytest.approx(0.0)

    def test_falsa_alarma_sobre_las_piezas_buenas(self):
        scores = np.array([0.9, 0.9, 0.1])
        labels = np.array([0, 0, 0], dtype=bool)

        assert confusion_at(scores, labels, 0.5).false_alarm_rate == pytest.approx(2 / 3)

    def test_recall_es_cero_si_no_detecta_nada(self):
        scores = np.array([0.1, 0.1])
        labels = np.array([1, 1], dtype=bool)

        matriz = confusion_at(scores, labels, 0.5)

        assert matriz.recall == 0.0
        assert matriz.f1 == 0.0

    def test_resumen_completo(self):
        resumen = confusion_at(np.array([0.9, 0.1]), np.array([1, 0], dtype=bool), 0.5).summary()

        assert set(resumen) == {
            "true_positives",
            "false_positives",
            "true_negatives",
            "false_negatives",
            "recall",
            "precision",
            "f1",
            "false_alarm_rate",
        }
