"""E3 - Cadena completa sobre el dataset propio, sin frontend.

Toma los clips grabados con `ml/capture/capturar.py` y recorre todo el pipeline:

    clips -> extraccion -> hashing -> particion por pieza -> deduplicacion
          -> entrenamiento -> umbral calibrado en validacion -> AUROC en test

Es el mismo recorrido que hara la plataforma cuando tenga interfaz; aqui se
ejecuta con un comando para poder validarlo antes de escribir una sola linea de
frontend.

**Modo ensayo.** Por debajo de los objetivos del protocolo (30 piezas normales y
15 con defecto por clase) el script funciona igual, pero avisa en cada salida de
que el resultado **no es evidencia**: con pocas piezas no hay potencia
estadistica para nada. Sirve para comprobar que el montaje fisico y la cadena
funcionan, que es justo lo que conviene saber antes de grabar 45 piezas.

Uso:
    python e3_propio.py --clase tapas
    python e3_propio.py --clase tapas --dedup 6 --semilla 1

Requiere el entorno de Python 3.12 con el extra 'ml' (.venv312).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from inspeccion.adapters.hashing import ImageHashAdapter
from inspeccion.adapters.ml_engine import AnomalibMLEngine, AnomalyModel
from inspeccion.adapters.storage import FilesystemFrameWriter
from inspeccion.adapters.video_source import OpenCvVideoSource
from inspeccion.application.curation import (
    CurationPipeline,
    FrameExtractor,
    HashedFrame,
    PerceptualDifferenceSampler,
    write_manifest,
)
from inspeccion.application.evaluation import auroc, calibrate_f1
from inspeccion.domain.models import GroupId, Label, Split

RAIZ = Path(__file__).resolve().parent
DATOS = RAIZ.parent / "data" / "propio"
RESULTADOS = RAIZ / "results" / "e3"

OBJETIVO_PROTOCOLO = {"normal": 30, "defecto": 15}


def main() -> int:
    args = _parse_args()
    clips = _leer_capturas(args.datos / args.clase / "capturas.jsonl")
    if not clips:
        print(
            f"No hay clips en {args.datos / args.clase}.\n"
            f"Grabalos primero:  python ml/capture/capturar.py --clase {args.clase} "
            f"--etiqueta normal",
            file=sys.stderr,
        )
        return 2

    ensayo = _avisar_si_es_ensayo(clips)

    frames, tiempo_extraccion = _extraer(clips, args)
    reporte, tiempo_curacion = _curar(frames, args)
    resultado, tiempo_ml = _entrenar_y_evaluar(reporte, args)

    manifiesto = write_manifest(reporte.manifest, args.out / args.clase / "split.json")
    destino = _archivar(
        args, clips, reporte, resultado, ensayo, (tiempo_extraccion, tiempo_curacion, tiempo_ml)
    )

    _informar(clips, reporte, resultado, ensayo, (tiempo_extraccion, tiempo_curacion, tiempo_ml))
    print(f"\nManifiesto de particion: {manifiesto}")
    print(f"Resultado:               {destino}")
    return 0


def _extraer(clips: list[dict], args: argparse.Namespace) -> tuple[list[HashedFrame], float]:
    """Decodifica cada clip una sola vez: muestrea, hashea y persiste."""
    extractor = FrameExtractor(
        hasher=ImageHashAdapter(),
        writer=FilesystemFrameWriter(root=args.out / args.clase / "frames"),
        sampler_factory=lambda: PerceptualDifferenceSampler(
            min_difference=args.min_diferencia, max_interval=args.max_intervalo
        ),
    )

    print(f"\nExtrayendo frames de {len(clips)} clips ...")
    inicio = time.perf_counter()
    frames: list[HashedFrame] = []
    for clip in clips:
        etiqueta = Label.NORMAL if clip["etiqueta"] == "normal" else Label.DEFECT
        with OpenCvVideoSource.from_file(clip["ruta"]) as fuente:
            resultado = extractor.extract(
                fuente,
                group_id=GroupId(str(clip["pieza"])),
                label=etiqueta,
                source_video=str(clip["ruta"]),
            )
        frames.extend(resultado.frames)
        print(
            f"  {clip['pieza']:<10} {resultado.seen:>5} frames -> {resultado.kept:>4} "
            f"({resultado.retained_fraction:.1%})"
        )
    return frames, time.perf_counter() - inicio


def _curar(frames: list[HashedFrame], args: argparse.Namespace):  # noqa: ANN202
    """Particiona por pieza y despues deduplica dentro de cada split."""
    print("\nCurando (particion por pieza -> deduplicacion) ...")
    inicio = time.perf_counter()
    pipeline = CurationPipeline(dedup_max_distance=args.dedup)
    return pipeline.run(frames, seed=args.semilla), time.perf_counter() - inicio


def _entrenar_y_evaluar(reporte, args: argparse.Namespace) -> tuple[dict[str, object], float]:  # noqa: ANN001
    """Entrena, calibra el umbral en validacion y evalua en test.

    El umbral se calibra sobre VAL y se congela. Mirar el test para elegirlo
    seria ajustar un parametro sobre el test, que es lo que toda la tesis
    denuncia.
    """
    engine = AnomalibMLEngine(
        model=AnomalyModel(args.modelo),
        coreset_sampling_ratio=args.coreset,
        batch_size=args.batch,
        num_workers=0,
        seed=args.semilla,
    )
    manifest = reporte.manifest
    print(f"\nEntrenando {args.modelo} con {len(manifest.frames(Split.TRAIN))} frames ...")

    inicio = time.perf_counter()
    checkpoint = engine.train(manifest, args.out / args.clase / "modelo")

    val, test = manifest.frames(Split.VAL), manifest.frames(Split.TEST)
    etiquetas_val = np.array([f.label is Label.DEFECT for f in val], dtype=np.bool_)
    etiquetas_test = np.array([f.label is Label.DEFECT for f in test], dtype=np.bool_)

    scores_test = engine.scores(checkpoint, test)
    resultado: dict[str, object] = {"auroc": None, "umbral": None, "confusion": None}

    if etiquetas_test.any() and not etiquetas_test.all():
        resultado["auroc"] = auroc(scores_test, etiquetas_test)

    if etiquetas_val.any() and not etiquetas_val.all():
        umbral = calibrate_f1(engine.scores(checkpoint, val), etiquetas_val, split=Split.VAL)
        resultado["umbral"] = umbral.summary()
        resultado["confusion"] = umbral.evaluate(scores_test, etiquetas_test).summary()

    return resultado, time.perf_counter() - inicio


def _leer_capturas(manifiesto: Path) -> list[dict]:
    if not manifiesto.is_file():
        return []
    return [
        json.loads(linea) for linea in manifiesto.read_text(encoding="utf-8").splitlines() if linea
    ]


def _avisar_si_es_ensayo(clips: list[dict]) -> bool:
    """True si el dataset no alcanza los objetivos del protocolo."""
    conteo = {e: sum(1 for c in clips if c["etiqueta"] == e) for e in OBJETIVO_PROTOCOLO}
    falta = {e: n for e, n in conteo.items() if n < OBJETIVO_PROTOCOLO[e]}
    if not falta:
        return False
    print("\n" + "!" * 78)
    print("ENSAYO: el dataset no alcanza los objetivos del protocolo de captura v2.")
    for etiqueta, n in conteo.items():
        print(f"  {etiqueta:<8} {n:>3} piezas (objetivo {OBJETIVO_PROTOCOLO[etiqueta]})")
    print(
        "Con estas cantidades NO hay potencia estadistica: el resultado sirve para\n"
        "comprobar que el montaje y la cadena funcionan, NO como evidencia de la tesis."
    )
    print("!" * 78)
    return True


def _informar(clips, reporte, resultado, ensayo, tiempos) -> None:  # noqa: ANN001
    extraccion, curacion, ml = tiempos
    manifest = reporte.manifest
    print("\n" + "=" * 78)
    print("RESULTADO")
    print("=" * 78)

    piezas = {s: len(manifest.groups(s)) for s in Split}
    print(
        f"  Piezas          train {piezas[Split.TRAIN]}  "
        f"val {piezas[Split.VAL]}  test {piezas[Split.TEST]}"
    )
    antes = sum(reporte.frames_before_dedup.values())
    despues = sum(reporte.frames_after_dedup.values())
    print(
        f"  Frames          {antes} -> {despues} tras deduplicar ({despues / antes:.0%} retenidos)"
    )
    print(f"  Fuga en test    {reporte.leakage.contaminated_test_fraction:.1%} (debe ser 0.0%)")

    if resultado["auroc"] is not None:
        print(f"\n  AUROC (test)    {resultado['auroc']:.4f}")
    else:
        print("\n  AUROC           no calculable: el test necesita piezas de ambas clases")

    if resultado["confusion"] is not None:
        c = resultado["confusion"]
        print(
            f"  Con el umbral calibrado en validacion:\n"
            f"    detecta          {c['recall']:.0%} de los frames defectuosos\n"
            f"    falsas alarmas   {c['false_alarm_rate']:.0%} de los frames buenos\n"
            f"    F1               {c['f1']:.3f}"
        )
    else:
        print("  Umbral          no calibrable: validacion sin defectos")

    print(
        f"\n  Tiempos         extraccion {extraccion:.0f}s | curacion {curacion:.1f}s | "
        f"entrenamiento+inferencia {ml:.0f}s"
    )
    if ensayo:
        print("\n  Recuerda: es un ENSAYO. No reportes estas cifras como resultado de la tesis.")
    print("=" * 78)


def _archivar(args, clips, reporte, resultado, ensayo, tiempos) -> Path:  # noqa: ANN001
    destino = args.out / args.clase / "e3_resultado.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    extraccion, curacion, ml = tiempos
    destino.write_text(
        json.dumps(
            {
                "experimento": "E3 - cadena completa sobre dataset propio",
                "es_ensayo": ensayo,
                "fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
                "clase": args.clase,
                "clips": len(clips),
                "piezas_por_etiqueta": {
                    e: sum(1 for c in clips if c["etiqueta"] == e) for e in OBJETIVO_PROTOCOLO
                },
                "configuracion": {
                    "modelo": args.modelo,
                    "dedup_max_distance": args.dedup,
                    "min_diferencia": args.min_diferencia,
                    "semilla": args.semilla,
                },
                "curacion": reporte.summary(),
                "resultado": resultado,
                "segundos": {
                    "extraccion": round(extraccion, 1),
                    "curacion": round(curacion, 1),
                    "entrenamiento_inferencia": round(ml, 1),
                },
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )
    return destino


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--clase", required=True, help="la misma que usaste al capturar")
    p.add_argument("--datos", type=Path, default=DATOS)
    p.add_argument("--out", type=Path, default=RESULTADOS)
    p.add_argument("--modelo", default="patchcore", choices=[m.value for m in AnomalyModel])
    p.add_argument("--dedup", type=int, default=4, help="umbral de Hamming sobre pHash de 64 bits")
    p.add_argument("--min-diferencia", type=float, default=0.02, dest="min_diferencia")
    p.add_argument("--max-intervalo", type=int, default=90, dest="max_intervalo")
    p.add_argument(
        "--coreset",
        type=float,
        default=0.01,
        help=(
            "ratio de coreset de PatchCore. 0.01 es la configuracion de los "
            "resultados principales del paper y la unica viable en CPU: la "
            "seleccion greedy es O(n*k), asi que con unos pocos miles de frames "
            "un ratio alto tarda horas. Subirlo solo con GPU."
        ),
    )
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--semilla", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
