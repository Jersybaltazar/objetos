"""E1 - Validacion de la integracion contra MVTec AD.

E1 no prueba ninguna hipotesis de la tesis. Es el control de calidad de la
instrumentacion: comprueba que el pipeline reproduce cifras conocidas sobre un
dataset conocido, usando la particion oficial de los autores sin modificarla.

Sin este control, un defecto de implementacion seria indistinguible de un
hallazgo. Si E1 falla, se detiene el avance experimental hasta corregirlo
(seccion 5/E1 del protocolo de evaluacion).

Uso:
    python e1_baseline.py --check                      # solo verifica el dataset
    python e1_baseline.py --models padim --categories bottle
    python e1_baseline.py                              # las 6 celdas completas

Requiere el entorno de Python 3.12 con el extra 'ml' (.venv312).
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mvtec import CategoryReport, inspect_category, official_manifest

from inspeccion.adapters.ml_engine import AnomalibMLEngine, AnomalyModel
from inspeccion.application.evaluation import auroc
from inspeccion.domain.models import Label, Split

RAIZ = Path(__file__).resolve().parent
DATOS_POR_DEFECTO = RAIZ.parent / "data" / "MVTecAD"
RESULTADOS_POR_DEFECTO = RAIZ / "results"
REFERENCIAS = RAIZ / "e1_references.json"

CATEGORIAS_POR_DEFECTO = ("bottle", "screw", "metal_nut")
"""Categorias de objeto, por analogia con las piezas del dataset propio."""

CONFIG_POR_MODELO: dict[str, dict[str, object]] = {
    "patchcore": {"backbone": "wide_resnet50_2", "layers": ("layer2", "layer3")},
    "padim": {
        "backbone": "wide_resnet50_2",
        "layers": ("layer1", "layer2", "layer3"),
        "n_features": 550,
    },
}
"""Configuracion de cada modelo, elegida para replicar la del paper de referencia.

Para PaDiM esto NO es el default de anomalib, que usa `resnet18`. El paper reporta
PaDiM-WR50-Rd550 y PaDiM es muy sensible al backbone: con `resnet18` la primera
corrida de E1 dio 0.8268 en `screw` frente a los 0.975 publicados. Reproducir una
cifra exige replicar la configuracion que la produjo.
"""


@dataclass
class Celda:
    """Un (modelo, categoria) ya ejecutado."""

    modelo: str
    categoria: str
    auroc: float
    n_train: int
    n_test: int
    n_test_defecto: int
    segundos_entrenamiento: float
    segundos_inferencia: float
    ram_pico_mb: float | None
    referencia_anomalib: float | None
    referencia_paper: float | None
    suelo: float
    veredicto: str
    motivo: str


def main() -> int:
    args = _parse_args()
    referencias = json.loads(REFERENCIAS.read_text(encoding="utf-8"))

    reportes = [inspect_category(args.root, c) for c in args.categories]
    _imprimir_dataset(args.root, reportes)
    faltantes = [r for r in reportes if not r.is_usable]
    if faltantes:
        print(_ayuda_descarga(args.root), file=sys.stderr)
        return 2
    if args.check:
        print("\nDataset correcto. Quita --check para ejecutar el experimento.")
        return 0

    _avisar_referencias_sin_verificar(referencias)

    celdas: list[Celda] = []
    for modelo in args.models:
        for categoria in args.categories:
            celdas.append(_ejecutar(modelo, categoria, args, referencias))
            _imprimir_tabla(celdas)

    destino = _archivar(celdas, args, referencias)
    print(f"\nResultados archivados en {destino}")

    fallidas = [c for c in celdas if c.veredicto == "FALLA"]
    if fallidas:
        print(
            f"\nE1 NO PASA: {len(fallidas)} de {len(celdas)} celdas fuera de tolerancia.\n"
            f"El protocolo exige detener el avance experimental y depurar antes de E2.",
            file=sys.stderr,
        )
        return 1

    print("\nE1 PASA. La instrumentacion esta validada; se puede avanzar a E2.")
    return 0


def _ejecutar(modelo: str, categoria: str, args: argparse.Namespace, referencias: dict) -> Celda:
    print(f"\n>>> {modelo} / {categoria} ...", flush=True)
    manifest = official_manifest(args.root, categoria)
    train = manifest.frames(Split.TRAIN)
    test = manifest.frames(Split.TEST)

    engine = AnomalibMLEngine(
        model=AnomalyModel(modelo),
        coreset_sampling_ratio=args.coreset,
        batch_size=args.batch_size,
        num_workers=0,
        accelerator="auto",
        seed=args.seed,
        **CONFIG_POR_MODELO[modelo],  # type: ignore[arg-type]
    )

    work_dir = args.out / "runs" / f"{modelo}_{categoria}"
    pico = _RamPico()
    inicio = time.perf_counter()
    with pico:
        checkpoint = engine.train(manifest, work_dir)
    entrenamiento = time.perf_counter() - inicio

    inicio = time.perf_counter()
    scores = engine.scores(checkpoint, test)
    inferencia = time.perf_counter() - inicio

    etiquetas = np.array([f.label is Label.DEFECT for f in test], dtype=np.bool_)
    valor = auroc(scores, etiquetas)

    return _evaluar(
        modelo=modelo,
        categoria=categoria,
        valor=valor,
        n_train=len(train),
        n_test=len(test),
        n_defecto=int(etiquetas.sum()),
        entrenamiento=entrenamiento,
        inferencia=inferencia,
        ram=pico.mb,
        referencias=referencias,
    )


def _evaluar(
    *,
    modelo: str,
    categoria: str,
    valor: float,
    n_train: int,
    n_test: int,
    n_defecto: int,
    entrenamiento: float,
    inferencia: float,
    ram: float | None,
    referencias: dict,
) -> Celda:
    """Emite el veredicto contra la primera referencia verificada disponible.

    El orden es paper -> anomalib -> suelo conservador. Comparar contra una cifra
    publicada solo vale si la corrida replica su configuracion, y por eso
    `CONFIG_POR_MODELO` fija backbone, capas y dimensiones a las del paper. El
    suelo es la red de seguridad para cuando no hay referencia; detecta codigo
    roto, no diferencias de calidad.
    """
    tolerancia = float(referencias["tolerancia_pp"]) / 100.0
    suelo = float(referencias["suelo_conservador"][modelo])
    bloque = referencias["referencias"][modelo]
    anomalib = bloque["fuente_anomalib"]
    paper = bloque["fuente_paper"]

    ref_anomalib = anomalib.get(categoria) if anomalib.get("verificado") else None
    ref_paper = paper.get(categoria) if paper.get("verificado") else None

    referencia = ref_paper if ref_paper is not None else ref_anomalib
    origen = "paper" if ref_paper is not None else "anomalib"

    if referencia is not None:
        delta = valor - float(referencia)
        pasa = abs(delta) <= tolerancia
        motivo = f"delta {delta * 100:+.2f} pp vs {origen} (tolerancia +-{tolerancia * 100:.1f})"
    else:
        pasa = valor >= suelo
        motivo = f"sin referencia verificada; contra suelo {suelo:.3f}"

    return Celda(
        modelo=modelo,
        categoria=categoria,
        auroc=valor,
        n_train=n_train,
        n_test=n_test,
        n_test_defecto=n_defecto,
        segundos_entrenamiento=round(entrenamiento, 2),
        segundos_inferencia=round(inferencia, 2),
        ram_pico_mb=None if ram is None else round(ram, 1),
        referencia_anomalib=ref_anomalib,
        referencia_paper=ref_paper,
        suelo=suelo,
        veredicto="PASA" if pasa else "FALLA",
        motivo=motivo,
    )


class _RamPico:
    """Pico de memoria residente durante el bloque, por muestreo en segundo plano.

    Se muestrea en un hilo en vez de leer la RSS al entrar y al salir: el memory
    bank de PatchCore se construye y se poda dentro del bloque, asi que la RSS
    final puede ser muy inferior al maximo alcanzado. El pico es el dato que
    importa para RNF-02 y para justificar la deduplicacion en E4.

    `mb` queda en `None` si psutil no esta instalado.
    """

    INTERVALO_S = 0.2

    def __init__(self) -> None:
        self.mb: float | None = None
        try:
            import psutil

            self._proceso = psutil.Process()
        except ImportError:
            self._proceso = None
        self._parar = threading.Event()
        self._hilo: threading.Thread | None = None

    def __enter__(self) -> _RamPico:
        if self._proceso is None:
            return self
        self.mb = self._proceso.memory_info().rss / (1024 * 1024)
        self._hilo = threading.Thread(target=self._muestrear, daemon=True)
        self._hilo.start()
        return self

    def __exit__(self, *_: object) -> None:
        if self._hilo is None:
            return
        self._parar.set()
        self._hilo.join(timeout=2.0)

    def _muestrear(self) -> None:
        assert self._proceso is not None
        while not self._parar.wait(self.INTERVALO_S):
            actual = self._proceso.memory_info().rss / (1024 * 1024)
            if self.mb is None or actual > self.mb:
                self.mb = actual


def _imprimir_dataset(root: Path, reportes: list[CategoryReport]) -> None:
    print(f"\nMVTec AD en {root}\n" + "-" * 78)
    for reporte in reportes:
        print("  " + reporte.describe())


def _imprimir_tabla(celdas: list[Celda]) -> None:
    print("\n" + "=" * 96)
    print(
        f"{'modelo':<11}{'categoria':<12}{'AUROC':>8}{'ref':>8}{'suelo':>8}"
        f"{'train_s':>10}{'infer_s':>9}{'RAM_MB':>9}  veredicto"
    )
    print("-" * 96)
    for c in celdas:
        usada = c.referencia_paper if c.referencia_paper is not None else c.referencia_anomalib
        ref = f"{usada:.3f}" if usada is not None else "-"
        ram = f"{c.ram_pico_mb:.0f}" if c.ram_pico_mb is not None else "-"
        print(
            f"{c.modelo:<11}{c.categoria:<12}{c.auroc:>8.4f}{ref:>8}{c.suelo:>8.3f}"
            f"{c.segundos_entrenamiento:>10.1f}{c.segundos_inferencia:>9.1f}{ram:>9}  "
            f"{c.veredicto}  ({c.motivo})"
        )
    print("=" * 96)


def _avisar_referencias_sin_verificar(referencias: dict) -> None:
    pendientes = [
        modelo
        for modelo, bloque in referencias["referencias"].items()
        if not any(bloque[f].get("verificado") for f in ("fuente_paper", "fuente_anomalib"))
    ]
    if pendientes:
        print(
            "\nAVISO: referencias sin verificar -> " + ", ".join(pendientes) + "\n"
            "  El veredicto se emitira contra el suelo conservador, que solo detecta\n"
            "  codigo roto, no diferencias de calidad. Antes de citar E1 en la tesis,\n"
            f"  completa {REFERENCIAS.name} desde la fuente y pon verificado: true."
        )


def _archivar(celdas: list[Celda], args: argparse.Namespace, referencias: dict) -> Path:
    args.out.mkdir(parents=True, exist_ok=True)
    destino = args.out / "e1_resultados.json"
    destino.write_text(
        json.dumps(
            {
                "experimento": "E1 - validacion de la integracion",
                "fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
                "commit": _commit(),
                "entorno": {
                    "python": platform.python_version(),
                    "plataforma": platform.platform(),
                    "torch": _version("torch"),
                    "anomalib": _version("anomalib"),
                    "cuda": _cuda(),
                },
                "configuracion": {
                    "root": str(args.root),
                    "categorias": list(args.categories),
                    "modelos": list(args.models),
                    "coreset_sampling_ratio": args.coreset,
                    "batch_size": args.batch_size,
                    "seed": args.seed,
                    "split": "oficial de MVTec AD, sin modificar",
                },
                "tolerancia_pp": referencias["tolerancia_pp"],
                "celdas": [asdict(c) for c in celdas],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return destino


def _version(modulo: str) -> str:
    try:
        return __import__(modulo).__version__
    except Exception:
        return "desconocida"


def _cuda() -> str:
    try:
        import torch

        return f"disponible={torch.cuda.is_available()}"
    except Exception:
        return "desconocida"


def _commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=RAIZ, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "sin commit"


def _ayuda_descarga(root: Path) -> str:
    return (
        f"\nFaltan datos en {root}.\n\n"
        "El descargador de anomalib NO sirve: el enlace que lleva codificado\n"
        "(mydrive.ch) devuelve 404 desde que MVTec movio la distribucion a\n"
        "descarga con registro. Hay que bajarlo a mano:\n\n"
        "  1. Registrarse en https://www.mvtec.com/company/research/datasets/mvtec-ad\n"
        "     (gratuito, licencia CC BY-NC-SA 4.0: valida para la tesis, no para uso\n"
        "     comercial)\n"
        "  2. Descargar el archivo completo (~5 GB)\n"
        "  3. Extraerlo de modo que quede esta estructura:\n\n"
        f"       {root}\\bottle\\train\\good\\*.png\n"
        f"       {root}\\bottle\\test\\good\\*.png\n"
        f"       {root}\\bottle\\test\\broken_large\\*.png\n"
        f"       {root}\\screw\\...\n"
        f"       {root}\\metal_nut\\...\n\n"
        "  4. Volver a ejecutar con --check para confirmar\n"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=DATOS_POR_DEFECTO, help="raiz de MVTec AD")
    parser.add_argument(
        "--out", type=Path, default=RESULTADOS_POR_DEFECTO, help="carpeta de salida"
    )
    parser.add_argument("--categories", nargs="+", default=list(CATEGORIAS_POR_DEFECTO))
    parser.add_argument(
        "--models",
        nargs="+",
        default=[m.value for m in AnomalyModel],
        choices=[m.value for m in AnomalyModel],
    )
    parser.add_argument(
        "--coreset",
        type=float,
        default=0.01,
        help=(
            "ratio de coreset de PatchCore. 0.01 es la configuracion de los "
            "resultados principales del paper (PatchCore-1%%) y la unica viable en "
            "CPU: la seleccion greedy es O(n*k), asi que 0.1 multiplica por diez "
            "el coste del paso mas caro. El valor usado queda registrado en el JSON "
            "de resultados y debe coincidir con el de la referencia que se cite."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check", action="store_true", help="solo verificar el dataset y salir")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
