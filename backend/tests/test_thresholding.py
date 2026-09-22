from __future__ import annotations

import numpy as np
import pytest

from inspeccion.application.evaluation.metrics import confusion_at
from inspeccion.application.evaluation.thresholding import (
    ThresholdLeakageError,
    calibrate_f1,
)
from inspeccion.domain.models import Split


@pytest.fixture
def val_scores() -> tuple[np.ndarray, np.ndarray]:
    scores = np.array([0.05, 0.10, 0.15, 0.60, 0.70, 0.80])
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=bool)
    return scores, labels


class TestProteccionContraFuga:
    @pytest.mark.parametrize("split", [Split.TEST, Split.TRAIN])
    def test_rechaza_calibrar_fuera_de_validacion(self, val_scores, split):
        # La regla de la seccion 7 del protocolo deja de depender de la
        # disciplina de quien ejecuta el experimento: el codigo la impone.
        scores, labels = val_scores

        with pytest.raises(ThresholdLeakageError, match="solo puede calibrarse"):
            calibrate_f1(scores, labels, split=split)

    def test_el_mensaje_explica_por_que(self, val_scores):
        scores, labels = val_scores

        with pytest.raises(ThresholdLeakageError, match="ajustar un parametro mirando el test"):
            calibrate_f1(scores, labels, split=Split.TEST)

    def test_el_split_no_tiene_valor_por_defecto(self, val_scores):
        # Obliga a declararlo en cada llamada; no se puede omitir por descuido.
        scores, labels = val_scores

        with pytest.raises(TypeError):
            calibrate_f1(scores, labels)  # type: ignore[call-arg]


class TestCalibracion:
    def test_encuentra_el_umbral_que_separa(self, val_scores):
        scores, labels = val_scores

        umbral = calibrate_f1(scores, labels, split=Split.VAL)

        assert umbral.value == pytest.approx(0.60)
        assert umbral.f1_on_val == pytest.approx(1.0)

    def test_maximiza_f1_entre_todos_los_candidatos(self, val_scores):
        scores, labels = val_scores

        umbral = calibrate_f1(scores, labels, split=Split.VAL)

        for candidato in np.unique(scores):
            assert confusion_at(scores, labels, float(candidato)).f1 <= umbral.f1_on_val

    def test_los_candidatos_son_los_scores_observados(self, val_scores):
        scores, labels = val_scores

        umbral = calibrate_f1(scores, labels, split=Split.VAL)

        assert umbral.candidates_evaluated == len(np.unique(scores))

    def test_con_solapamiento_elige_el_mejor_compromiso(self):
        scores = np.array([0.1, 0.5, 0.4, 0.9])
        labels = np.array([0, 0, 1, 1], dtype=bool)

        umbral = calibrate_f1(scores, labels, split=Split.VAL)

        assert 0.0 < umbral.f1_on_val <= 1.0
        assert umbral.value in set(scores)

    def test_rechaza_entrada_vacia(self):
        with pytest.raises(ValueError, match="no hay scores"):
            calibrate_f1(np.array([]), np.array([], dtype=bool), split=Split.VAL)


class TestAplicacion:
    def test_el_umbral_congelado_se_aplica_a_test_sin_reajuste(self, val_scores):
        scores, labels = val_scores
        umbral = calibrate_f1(scores, labels, split=Split.VAL)

        test_scores = np.array([0.02, 0.65, 0.90])
        test_labels = np.array([0, 1, 1], dtype=bool)

        matriz = umbral.evaluate(test_scores, test_labels)

        assert matriz.recall == pytest.approx(1.0)
        assert matriz.false_alarm_rate == pytest.approx(0.0)

    def test_apply_devuelve_predicciones_booleanas(self, val_scores):
        scores, labels = val_scores
        umbral = calibrate_f1(scores, labels, split=Split.VAL)

        predicciones = umbral.apply(np.array([0.1, 0.9]))

        assert predicciones.dtype == np.bool_
        assert list(predicciones) == [False, True]

    def test_el_umbral_es_inmutable(self, val_scores):
        scores, labels = val_scores
        umbral = calibrate_f1(scores, labels, split=Split.VAL)

        with pytest.raises(AttributeError):
            umbral.value = 0.99  # type: ignore[misc]

    def test_resumen_para_el_registro(self, val_scores):
        scores, labels = val_scores

        resumen = calibrate_f1(scores, labels, split=Split.VAL).summary()

        assert set(resumen) == {"threshold", "f1_on_val", "candidates_evaluated"}
