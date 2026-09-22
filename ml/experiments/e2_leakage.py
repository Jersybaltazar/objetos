"""E2 - Curva dosis-respuesta de fuga, con brazos emparejados.

Primera evidencia cientifica de la tesis, y la unica que no depende de la camara
ni de las piezas fisicas.

Mide cuanto se infla el AUROC en funcion de `lambda`: la proporcion del conjunto
de entrenamiento que son casi-duplicados de piezas que estan en test. Ese es el
mecanismo exacto de la fuga por frames de video: el modelo memoriza material que
luego reaparece en la evaluacion, produce menos falsos positivos de los que
produciria sobre material no visto, y el AUROC sube sin que el modelo sea mejor.

**Revision respecto del protocolo original.** La seccion 5/E2 planteaba barrer el
numero de duplicados `k` con dos particiones independientes. El piloto mostro que
ese diseno no era interpretable: a k=0 daba un delta de +3.62 pp que no era fuga
sino varianza entre particiones, y a k>0 el brazo ingenuo entrenaba con mas
originales distintos, lo que subia su AUROC al margen de la fuga (el confundido de
la seccion 12). El diseno emparejado de `controlled_leakage` fija el conjunto de test y
el tamano del entrenamiento, dejando `lambda` como unica diferencia. `k` pasa a
ser solo la reserva de casi-duplicados de la que se toma el material filtrado.

Uso:
    python e2_leakage.py --estimate                      # coste previsto
    python e2_leakage.py --leak 0 0.1 --seeds 3          # piloto corto
    python e2_leakage.py                                 # corrida completa

Requiere el entorno de Python 3.12 con el extra 'ml' (.venv312).
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from duplicates import JitterConfig, materialise
from mvtec import inspect_category

from inspeccion.adapters.ml_engine import AnomalibMLEngine, AnomalyModel
from inspeccion.application.curation import build_leaked_arms
from inspeccion.application.evaluation import auroc, holm_bonferroni, paired_comparison
from inspeccion.domain.models import Label, Split, SplitManifest

RAIZ = Path(__file__).resolve().parent
DATOS = RAIZ.parent / "data" / "MVTecAD"
CACHE = RAIZ.parent / "data" / "e2_cache"
RESULTADOS = RAIZ / "results"

DOSIS_POR_DEFECTO = (0.0, 0.05, 0.10, 0.20)
"""Fraccion del entrenamiento que es casi-duplicado de material de test.

El techo teorico es `k/(3(k+1))` ~ 33%: solo hay `k` duplicados por cada pieza de
test y el entrenamiento es tres veces mayor que el test. Una particion aleatoria
por frame sobre un video real cae en el entorno del 20%.
"""


@dataclass
class PuntoDeCurva:
    dosis: float
    n_train: int
    n_test: int
    n_frames_filtrados: int
    auroc_limpio: list[float] = field(default_factory=list)
    auroc_filtrado: list[float] = field(default_factory=list)
    delta_medio: float = 0.0
    ci_low: float = 0.0
    ci_high: float = 0.0
    p_value: float = 1.0
    p_value_corregido: float = 1.0
    rank_biserial: float = 0.0
    magnitud: str = ""
    relevante: bool = False


def main() -> int:
    args = _parse_args()

    reporte = inspect_category(args.root, args.category)
    if not reporte.is_usable:
        print(f"Falta la categoria {args.category} en {args.root}", file=sys.stderr)
        return 2
    print(f"\nCategoria: {reporte.describe()}")

    if args.estimate:
        _estimar(args, reporte.train_normal + reporte.test_normal)
        return 0

    originales, defectos = _materializar(args)
    engine = AnomalibMLEngine(
        model=AnomalyModel(args.model),
        backbone=args.backbone,
        coreset_sampling_ratio=args.coreset,
        batch_size=args.batch_size,
        num_workers=0,
        accelerator="auto",
        seed=None,
    )

    inicio = time.perf_counter()
    puntos = _recorrer(originales, defectos, args, engine)
    _corregir_multiples(puntos, args.alpha)

    _imprimir_curva(puntos)
    destino = _archivar(puntos, args, time.perf_counter() - inicio)
    print(f"\nResultados archivados en {destino}")
    return _veredicto(puntos, args)


def _materializar(args: argparse.Namespace) -> tuple[dict[str, list[Path]], list[Path]]:
    """Genera la reserva de casi-duplicados y recoge los defectos."""
    base = args.root / args.category
    # El identificador lleva el subconjunto de origen: MVTec numera igual las
    # imagenes de train/good y de test/good (000.png en ambas), asi que usar solo
    # el nombre del fichero fusionaria piezas distintas y perderia normales.
    fuentes = {
        f"{subconjunto}_{ruta.stem}": ruta
        for subconjunto in ("train", "test")
        for ruta in sorted((base / subconjunto / "good").glob("*.png"))
    }
    destino = args.cache / args.category / f"k{args.duplicates:02d}"
    print(f"Materializando {args.duplicates} duplicados por pieza en {destino} ...", flush=True)

    originales = materialise(
        fuentes,
        destino,
        duplicates=args.duplicates,
        config=JitterConfig(
            max_translation_fraction=args.translation,
            max_rotation_degrees=args.rotation,
            max_scale_fraction=args.scale,
            max_brightness_fraction=args.brightness,
            noise_sigma=args.noise,
        ),
    )
    defectos = [
        ruta
        for sub in sorted(p for p in (base / "test").iterdir() if p.is_dir() and p.name != "good")
        for ruta in sorted(sub.glob("*.png"))
    ]
    return originales, defectos


def _recorrer(
    originales: dict[str, list[Path]],
    defectos: list[Path],
    args: argparse.Namespace,
    engine: AnomalibMLEngine,
) -> list[PuntoDeCurva]:
    """Ejecuta la curva reutilizando el brazo limpio.

    El brazo limpio de una semilla **no depende de la dosis**: es siempre el mismo
    entrenamiento sobre las mismas piezas. Entrenarlo una vez por semilla en vez
    de una vez por (semilla, dosis) reduce casi a la mitad el coste total sin
    cambiar nada del contraste.
    """
    limpio_por_semilla: dict[int, float] = {}
    puntos: list[PuntoDeCurva] = []

    for dosis in args.leak:
        punto: PuntoDeCurva | None = None

        for semilla in range(args.seeds):
            brazos = build_leaked_arms(originales, defectos, seed=semilla, leak_fraction=dosis)
            if punto is None:
                punto = PuntoDeCurva(
                    dosis=dosis,
                    n_train=brazos.n_train,
                    n_test=brazos.n_test,
                    n_frames_filtrados=brazos.n_leaked_frames,
                )

            if semilla not in limpio_por_semilla:
                limpio_por_semilla[semilla] = _evaluar(
                    brazos.clean, args, engine, f"limpio_s{semilla}"
                )
            valor_limpio = limpio_por_semilla[semilla]

            valor_filtrado = (
                valor_limpio
                if brazos.n_leaked_frames == 0
                else _evaluar(brazos.leaked, args, engine, f"fuga{dosis:.2f}_s{semilla}")
            )

            punto.auroc_limpio.append(valor_limpio)
            punto.auroc_filtrado.append(valor_filtrado)
            print(
                f"  lambda={dosis:>5.0%} semilla={semilla:<3} "
                f"limpio={valor_limpio:.4f}  filtrado={valor_filtrado:.4f}  "
                f"delta={(valor_filtrado - valor_limpio) * 100:+.2f} pp",
                flush=True,
            )

        assert punto is not None
        _contrastar(punto, args)
        puntos.append(punto)
        _imprimir_curva(puntos)

    return puntos


def _evaluar(
    manifest: SplitManifest, args: argparse.Namespace, engine: AnomalibMLEngine, nombre: str
) -> float:
    """Entrena, puntua y devuelve el AUROC, con cache reanudable.

    Cada resultado se anade a `cache.jsonl` en cuanto se obtiene. Una corrida de
    120 entrenamientos en un mismo proceso termino con un segmentation fault en
    codigo nativo tras ~24 entrenamientos; con el cache, relanzar el script
    retoma donde se quedo y un fallo pierde como mucho un entrenamiento.

    El cache se invalida si cambia la configuracion: la clave incluye todo lo que
    afecta al resultado.
    """
    clave = f"{nombre}|{args.category}|{args.model}|{args.backbone}|k{args.duplicates}"
    cache = _leer_cache(args.out)
    if clave in cache:
        return cache[clave]

    directorio = args.out / "runs" / nombre
    checkpoint = engine.train(manifest, directorio)
    test = manifest.frames(Split.TEST)
    scores = engine.scores(checkpoint, test)
    valor = auroc(scores, np.array([f.label is Label.DEFECT for f in test], dtype=np.bool_))

    _escribir_cache(args.out, clave, valor)
    if not args.keep_checkpoints:
        # ~177 MB por checkpoint: 120 entrenamientos serian ~21 GB de modelos
        # que no se vuelven a usar. El resultado es el AUROC, y ya esta guardado.
        shutil.rmtree(directorio, ignore_errors=True)
    gc.collect()
    return valor


def _leer_cache(out: Path) -> dict[str, float]:
    ruta = out / "cache.jsonl"
    if not ruta.is_file():
        return {}
    registros = (json.loads(linea) for linea in ruta.read_text(encoding="utf-8").splitlines())
    return {r["clave"]: float(r["auroc"]) for r in registros if r}


def _escribir_cache(out: Path, clave: str, valor: float) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / "cache.jsonl").open("a", encoding="utf-8") as fichero:
        fichero.write(json.dumps({"clave": clave, "auroc": valor}) + "\n")


def _contrastar(punto: PuntoDeCurva, args: argparse.Namespace) -> None:
    if punto.n_frames_filtrados == 0:
        # A dosis 0 los dos brazos son literalmente el mismo entrenamiento: el
        # delta es exactamente 0 por construccion y no hay nada que contrastar.
        return
    contraste = paired_comparison(np.array(punto.auroc_filtrado), np.array(punto.auroc_limpio))
    punto.delta_medio = contraste.mean_difference
    punto.ci_low, punto.ci_high = contraste.ci_low, contraste.ci_high
    punto.p_value = contraste.p_value
    punto.rank_biserial = contraste.rank_biserial
    punto.magnitud = contraste.effect_magnitude
    punto.relevante = contraste.is_significant(alpha=args.alpha, min_effect=args.min_effect)


def _corregir_multiples(puntos: list[PuntoDeCurva], alpha: float) -> None:
    contrastados = [p for p in puntos if p.n_frames_filtrados > 0]
    if not contrastados:
        return
    corregidos = holm_bonferroni(np.array([p.p_value for p in contrastados]))
    for punto, corregido in zip(contrastados, corregidos, strict=True):
        punto.p_value_corregido = float(corregido)
        punto.relevante = punto.relevante and punto.p_value_corregido < alpha


def _monotonia(puntos: list[PuntoDeCurva]) -> tuple[float, float] | None:
    """He1b: la inflacion crece con la dosis. Spearman entre lambda y delta."""
    if len(puntos) < 3:
        return None
    from scipy.stats import spearmanr

    resultado = spearmanr([p.dosis for p in puntos], [p.delta_medio for p in puntos])
    return float(resultado.statistic), float(resultado.pvalue)


def _imprimir_curva(puntos: list[PuntoDeCurva]) -> None:
    print("\n" + "=" * 100)
    print(
        f"{'lambda':>8}{'filtrados':>11}{'AUROC limpio':>14}{'AUROC filtr.':>14}"
        f"{'delta pp':>10}{'IC95 pp':>20}{'p corr.':>10}  efecto"
    )
    print("-" * 100)
    for p in puntos:
        limpio = np.mean(p.auroc_limpio) if p.auroc_limpio else float("nan")
        filtrado = np.mean(p.auroc_filtrado) if p.auroc_filtrado else float("nan")
        marca = " *" if p.relevante else ""
        ic = f"[{p.ci_low * 100:+.2f}, {p.ci_high * 100:+.2f}]"
        print(
            f"{p.dosis:>8.0%}{p.n_frames_filtrados:>11}{limpio:>14.4f}{filtrado:>14.4f}"
            f"{p.delta_medio * 100:>10.2f}{ic:>20}{p.p_value_corregido:>10.4f}  "
            f"{p.magnitud}{marca}"
        )
    print("=" * 100)
    print("  * = significativo tras Holm-Bonferroni y por encima del umbral de relevancia")


def _veredicto(puntos: list[PuntoDeCurva], args: argparse.Namespace) -> int:
    control = next((p for p in puntos if p.n_frames_filtrados == 0), None)
    if control is not None and control.delta_medio != 0.0:
        print(
            f"\nATENCION: el control (lambda=0) da delta {control.delta_medio * 100:+.2f} pp.\n"
            f"A dosis cero los dos brazos son el mismo entrenamiento, asi que el delta debe\n"
            f"ser exactamente 0. Cualquier otra cosa es un defecto del montaje.",
            file=sys.stderr,
        )
        return 1

    relevantes = [p for p in puntos if p.relevante]
    monotonia = _monotonia(puntos)
    print()
    if monotonia is not None:
        rho, p_rho = monotonia
        print(f"He1b (monotonia): Spearman rho={rho:+.3f}, p={p_rho:.4f}")

    if relevantes:
        print(
            f"He1: SE RECHAZA H0 en {len(relevantes)} de "
            f"{len([p for p in puntos if p.n_frames_filtrados > 0])} niveles de dosis.\n"
            f"Umbral de relevancia preregistrado: {args.min_effect * 100:.0f} pp."
        )
    else:
        print(
            "He1: NO se rechaza H0 en ningun nivel. El resultado es igualmente\n"
            "publicable y debe reportarse como tal: la condicion de refutacion\n"
            "estaba declarada de antemano en el protocolo."
        )
    return 0


def _estimar(args: argparse.Namespace, normales_base: int) -> None:
    con_fuga = len([d for d in args.leak if d > 0])
    entrenamientos = args.seeds + con_fuga * args.seeds
    train = int(normales_base * 0.6) * (args.duplicates + 1)
    # Segundos de entrenamiento + inferencia por imagen de entrenamiento, medidos
    # en la maquina de desarrollo (CPU, 6 nucleos). PaDiM/resnet18 sale de la
    # corrida primaria de E2 (40 entrenamientos de 548 imagenes en 36,3 min);
    # PatchCore de E1. El resto, aproximado.
    segundos_por_imagen = {
        ("padim", "resnet18"): 0.099,
        ("padim", "wide_resnet50_2"): 2.57,
        ("patchcore", "wide_resnet50_2"): 0.88,
    }.get((args.model, args.backbone), 0.9)
    horas = entrenamientos * train * segundos_por_imagen / 3600
    techo = args.duplicates / (3 * (args.duplicates + 1))
    print(
        f"\nPlan: {args.seeds} brazos limpios (uno por semilla, reutilizados) +\n"
        f"      {con_fuga} dosis x {args.seeds} semillas = {entrenamientos} entrenamientos\n"
        f"  {train:,} imagenes de entrenamiento por corrida (fijo en todas las dosis)\n"
        f"  ETA aproximada: {horas:.1f} h\n\n"
        f"Techo de dosis con k={args.duplicates}: lambda <= {techo:.0%}. "
        f"Sube --duplicates para llegar mas alto."
    )


def _archivar(puntos: list[PuntoDeCurva], args: argparse.Namespace, segundos: float) -> Path:
    args.out.mkdir(parents=True, exist_ok=True)
    destino = args.out / f"e2_{args.category}_{args.model}.json"
    monotonia = _monotonia(puntos)
    destino.write_text(
        json.dumps(
            {
                "experimento": "E2 - curva dosis-respuesta de fuga (brazos emparejados)",
                "fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
                "diseno": (
                    "Conjunto de test identico en ambos brazos y en todas las dosis; "
                    "mismo tamano de entrenamiento; la unica diferencia es la fraccion "
                    "lambda del entrenamiento que son casi-duplicados de piezas de test."
                ),
                "entorno": {"python": platform.python_version(), "plataforma": platform.platform()},
                "configuracion": {
                    "categoria": args.category,
                    "modelo": args.model,
                    "backbone": args.backbone,
                    "coreset_sampling_ratio": args.coreset,
                    "duplicados_disponibles": args.duplicates,
                    "dosis": list(args.leak),
                    "semillas": args.seeds,
                    "alpha": args.alpha,
                    "umbral_relevancia_pp": args.min_effect * 100,
                    "jitter": {
                        "translation": args.translation,
                        "rotation_deg": args.rotation,
                        "scale": args.scale,
                        "brightness": args.brightness,
                        "noise_sigma": args.noise,
                    },
                },
                "monotonia_spearman": (
                    None if monotonia is None else {"rho": monotonia[0], "p": monotonia[1]}
                ),
                "segundos_totales": round(segundos, 1),
                "curva": [asdict(p) for p in puntos],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return destino


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=DATOS)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--out", type=Path, default=RESULTADOS / "e2")
    parser.add_argument("--category", default="bottle")
    parser.add_argument(
        "--model",
        default="padim",
        choices=[m.value for m in AnomalyModel],
        help=(
            "PaDiM por defecto: el fenomeno que E2 mide es de los DATOS, no del "
            "modelo, y PaDiM entrena ~4x mas rapido, lo que permite las 10 "
            "replicas que el protocolo exige. Conviene confirmar con PatchCore."
        ),
    )
    parser.add_argument(
        "--backbone",
        default="resnet18",
        help=(
            "resnet18 por defecto, y no el wide_resnet50_2 del paper. E1 exige "
            "replicar la configuracion publicada porque compara contra una cifra "
            "externa; E2 no: contrasta dos brazos entrenados con el MISMO modelo y "
            "la MISMA configuracion, asi que el valor absoluto del AUROC es "
            "irrelevante y solo importa la diferencia. PaDiM-WR50-Rd550 consume "
            "~10 GB por categoria y agota la memoria de la maquina de desarrollo."
        ),
    )
    parser.add_argument("--leak", nargs="+", type=float, default=list(DOSIS_POR_DEFECTO))
    parser.add_argument("--duplicates", type=int, default=3, help="reserva de casi-duplicados")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--coreset", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--min-effect", type=float, default=0.02)
    parser.add_argument("--translation", type=float, default=0.02)
    parser.add_argument("--rotation", type=float, default=2.0)
    parser.add_argument("--scale", type=float, default=0.02)
    parser.add_argument("--brightness", type=float, default=0.03)
    parser.add_argument("--noise", type=float, default=2.0)
    parser.add_argument(
        "--keep-checkpoints",
        action="store_true",
        help="conservar los checkpoints (~177 MB cada uno); por defecto se borran tras puntuar",
    )
    parser.add_argument("--estimate", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
