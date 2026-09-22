from __future__ import annotations

import numpy as np
import pytest

from inspeccion.application.evaluation.statistics import (
    cliffs_delta,
    equivalence,
    holm_bonferroni,
    paired_comparison,
    rank_biserial,
)

RNG = np.random.default_rng(20260919)


def auroc_replicas(media: float, ruido: float = 0.01, n: int = 10) -> np.ndarray:
    """Simula n replicas de AUROC alrededor de una media."""
    return np.clip(RNG.normal(media, ruido, n), 0.0, 1.0)


class TestPairedComparison:
    def test_detecta_la_inflacion_del_split_ingenuo(self):
        # El escenario de He1: el split ingenuo reporta ~8 pp mas.
        ingenuo = np.array([0.97, 0.98, 0.96, 0.97, 0.99, 0.97, 0.98, 0.96, 0.97, 0.98])
        por_pieza = np.array([0.89, 0.90, 0.88, 0.91, 0.90, 0.89, 0.90, 0.88, 0.91, 0.89])

        resultado = paired_comparison(ingenuo, por_pieza)

        assert resultado.mean_difference == pytest.approx(0.081, abs=0.005)
        assert resultado.p_value < 0.05
        assert resultado.is_significant(min_effect=0.02)
        assert resultado.effect_magnitude == "grande"

    def test_muestras_identicas_no_dan_diferencia(self):
        valores = auroc_replicas(0.93)

        resultado = paired_comparison(valores, valores.copy())

        assert resultado.mean_difference == 0.0
        assert resultado.p_value == 1.0
        assert resultado.rank_biserial == 0.0

    def test_significativo_pero_irrelevante(self):
        # La trampa que la seccion 9 del protocolo previene: una diferencia
        # consistente de 0.3 pp sale significativa y no significa nada.
        control = auroc_replicas(0.90)
        tratamiento = control + 0.003

        resultado = paired_comparison(tratamiento, control)

        assert resultado.p_value < 0.05
        assert resultado.is_significant(min_effect=0.0) is True
        assert resultado.is_significant(min_effect=0.02) is False

    def test_el_intervalo_de_confianza_rodea_la_diferencia(self):
        tratamiento = auroc_replicas(0.97)
        control = auroc_replicas(0.89)

        resultado = paired_comparison(tratamiento, control)

        assert resultado.ci_low <= resultado.mean_difference <= resultado.ci_high

    def test_exige_replicas_emparejadas(self):
        with pytest.raises(ValueError, match="mismo numero de replicas"):
            paired_comparison(np.array([0.9, 0.8, 0.7]), np.array([0.9, 0.8]))

    def test_exige_al_menos_dos_replicas(self):
        with pytest.raises(ValueError, match="al menos 2 replicas"):
            paired_comparison(np.array([0.9]), np.array([0.8]))

    def test_rechaza_valores_no_finitos(self):
        with pytest.raises(ValueError, match="no finitos"):
            paired_comparison(np.array([0.9, np.nan]), np.array([0.8, 0.7]))

    def test_resumen_lleva_efecto_ademas_de_p_valor(self):
        resumen = paired_comparison(auroc_replicas(0.97), auroc_replicas(0.89)).summary()

        assert "p_value" in resumen
        assert "rank_biserial" in resumen
        assert "cliffs_delta" in resumen
        assert "effect_magnitude" in resumen


class TestEquivalence:
    def test_la_deduplicacion_no_degrada(self):
        # Escenario de He2: diferencias muy por debajo del margen de 2 pp.
        con_dedup = np.array([0.901, 0.899, 0.902, 0.898, 0.900, 0.901, 0.899, 0.900, 0.902, 0.898])
        sin_dedup = np.array([0.900, 0.900, 0.901, 0.899, 0.901, 0.900, 0.900, 0.899, 0.901, 0.900])

        resultado = equivalence(con_dedup, sin_dedup, margin=0.02)

        assert resultado.is_equivalent
        assert resultado.is_inconclusive is False

    def test_una_degradacion_real_no_es_equivalente(self):
        con_dedup = auroc_replicas(0.82, ruido=0.005)
        sin_dedup = auroc_replicas(0.90, ruido=0.005)

        resultado = equivalence(con_dedup, sin_dedup, margin=0.02)

        assert resultado.is_equivalent is False
        assert resultado.is_inconclusive is False

    def test_datos_ruidosos_quedan_no_concluyentes(self):
        # Ni equivalencia ni diferencia: hacen falta mas replicas. Distinguirlo
        # evita reportar "no degrada" cuando lo que falta es potencia.
        con_dedup = auroc_replicas(0.90, ruido=0.08)
        sin_dedup = auroc_replicas(0.90, ruido=0.08)

        resultado = equivalence(con_dedup, sin_dedup, margin=0.001)

        assert resultado.is_equivalent is False
        assert resultado.is_inconclusive is True

    def test_rechaza_margen_no_positivo(self):
        with pytest.raises(ValueError, match="margen de equivalencia"):
            equivalence(auroc_replicas(0.9), auroc_replicas(0.9), margin=0.0)

    def test_resumen_declara_el_margen(self):
        resumen = equivalence(auroc_replicas(0.9), auroc_replicas(0.9), margin=0.02).summary()

        assert resumen["margin"] == 0.02
        assert "is_equivalent" in resumen


class TestHolmBonferroni:
    def test_valores_calculados_a_mano(self):
        # p=[0.01,0.02,0.03], n=3 -> 0.01*3=0.03; 0.02*2=0.04; 0.03*1=0.03
        # el maximo acumulado preserva la monotonia -> [0.03, 0.04, 0.04]
        corregidos = holm_bonferroni(np.array([0.01, 0.02, 0.03]))

        assert corregidos == pytest.approx([0.03, 0.04, 0.04])

    def test_es_monotono_creciente_en_el_orden_de_los_p_valores(self):
        originales = np.array([0.001, 0.04, 0.02, 0.5])

        corregidos = holm_bonferroni(originales)
        orden = np.argsort(originales)

        assert list(corregidos[orden]) == sorted(corregidos[orden])

    def test_nunca_reduce_un_p_valor(self):
        originales = np.array([0.001, 0.04, 0.02, 0.5])

        assert np.all(holm_bonferroni(originales) >= originales)

    def test_se_topa_en_uno(self):
        assert np.all(holm_bonferroni(np.array([0.4, 0.5, 0.6])) <= 1.0)

    def test_preserva_el_orden_de_entrada(self):
        corregidos = holm_bonferroni(np.array([0.03, 0.01, 0.02]))

        assert corregidos[1] < corregidos[0]

    def test_lista_vacia(self):
        assert holm_bonferroni(np.array([])).size == 0

    def test_rechaza_valores_fuera_de_rango(self):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            holm_bonferroni(np.array([0.5, 1.5]))


class TestTamanosDeEfecto:
    def test_cliffs_delta_con_dominancia_total(self):
        assert cliffs_delta(np.array([5.0, 6.0]), np.array([1.0, 2.0])) == 1.0

    def test_cliffs_delta_invertido(self):
        assert cliffs_delta(np.array([1.0, 2.0]), np.array([5.0, 6.0])) == -1.0

    def test_cliffs_delta_con_muestras_identicas(self):
        assert cliffs_delta(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == 0.0

    def test_rank_biserial_con_todas_las_diferencias_positivas(self):
        assert rank_biserial(np.array([2.0, 3.0, 4.0]), np.array([1.0, 1.0, 1.0])) == 1.0

    def test_rank_biserial_con_todas_las_diferencias_negativas(self):
        assert rank_biserial(np.array([1.0, 1.0, 1.0]), np.array([2.0, 3.0, 4.0])) == -1.0

    def test_rank_biserial_sin_diferencias(self):
        assert rank_biserial(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == 0.0

    def test_el_emparejamiento_importa(self):
        # Diferencias pequenas y consistentes con varianza grande entre replicas:
        # el efecto pareado es total, el no pareado casi no lo ve.
        control = np.array([0.50, 0.70, 0.90])
        tratamiento = control + 0.01

        assert rank_biserial(tratamiento, control) == pytest.approx(1.0)
        assert abs(cliffs_delta(tratamiento, control)) < 0.5
