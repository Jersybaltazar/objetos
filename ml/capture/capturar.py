"""Captura del dataset propio segun el protocolo de captura v2 (seccion 8).

Graba un clip corto por pieza fisica, con los automatismos de la camara fijados,
y registra cada clip en un manifiesto con su identificador de pieza, su etiqueta
y la calidad medida sobre los propios frames.

Por que cada decision:

- **Un clip por pieza, con su ID en el nombre.** La pieza fisica es la unidad de
  agrupamiento de toda la tesis. La trazabilidad del grupo tiene que existir desde
  la captura; reconstruirla despues es propenso a errores.
- **FFV1, sin perdida.** MJPG meteria artefactos de compresion en cada frame, y un
  artefacto alrededor de un borde es indistinguible de una rayadura.
- **Control de calidad al terminar cada clip.** Repetir un clip durante la sesion
  cuesta quince segundos; descubrir una semana despues que la exposicion oscilaba
  cuesta la sesion entera. El control mide sobre los frames si la exposicion
  quedo realmente fija, porque el driver puede aceptar la orden sin aplicarla.
- **Deriva frente al primer clip.** El protocolo exige la misma iluminacion en
  todos los clips.

Uso:
    python capturar.py --clase tapas --preview              # encuadre y controles
    python capturar.py --clase tapas --etiqueta normal      # graba la siguiente pieza buena
    python capturar.py --clase tapas --etiqueta defecto
    python capturar.py --clase tapas --resumen              # cuantas piezas hay

Funciona en cualquiera de los dos entornos (.venv o .venv312).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2

from inspeccion.adapters.video_source import CameraControls, OpenCvVideoSource
from inspeccion.application.curation.quality import (
    ClipQuality,
    assess_clip,
    assess_frame,
    compare_to_reference,
)
from inspeccion.domain.ports import VideoSource

RAIZ = Path(__file__).resolve().parents[1]
DESTINO_POR_DEFECTO = RAIZ / "data" / "propio"

PREFIJO = {"normal": "ok", "defecto": "def"}
OBJETIVO = {"normal": 30, "defecto": 15}
"""Minimo por clase del protocolo v2: habilita 10 replicas de particion por pieza."""

CODEC = "FFV1"
EXTENSION = ".mkv"
MUESTREO_QC = 15
"""Se evalua 1 de cada N frames: suficiente para ver deriva, sin frenar la grabacion."""


@dataclass
class Registro:
    """Una linea del manifiesto de captura."""

    pieza: str
    clase: str
    etiqueta: str
    ruta: str
    segundos_pedidos: float
    frames: int
    fps_medido: float
    fps_declarado: float
    resolucion: tuple[int, int]
    controles_aplicados: dict[str, bool]
    calidad: dict[str, object]
    avisos_deriva: list[str]
    fecha: str


def main() -> int:
    args = _parse_args()
    carpeta = args.out / args.clase
    manifiesto = carpeta / "capturas.jsonl"

    if args.resumen:
        _resumen(manifiesto)
        return 0

    fuente = _abrir(args)
    try:
        _informar_controles(fuente)
        if args.preview:
            return _preview(fuente, carpeta)

        if args.etiqueta is None:
            print("Falta --etiqueta normal|defecto (o usa --preview / --resumen).", file=sys.stderr)
            return 2

        pieza = args.pieza or _siguiente_id(manifiesto, args.etiqueta)
        if _ya_existe(manifiesto, pieza):
            print(f"La pieza {pieza} ya esta capturada. Usa otro --pieza.", file=sys.stderr)
            return 2

        destino = carpeta / args.etiqueta / f"{pieza}{EXTENSION}"
        print(f"\nPieza {pieza} ({args.etiqueta}). Coloca la pieza y gira despacio.")
        _cuenta_atras(args.espera)

        registro = grabar(
            fuente,
            destino,
            pieza=pieza,
            clase=args.clase,
            etiqueta=args.etiqueta,
            segundos=args.segundos,
            referencia=_referencia(manifiesto),
        )
    finally:
        fuente.close()

    _anadir(manifiesto, registro)
    _informar(registro, manifiesto)
    return 0 if not registro.calidad["warnings"] and not registro.avisos_deriva else 3


def grabar(
    fuente: VideoSource,
    destino: Path,
    *,
    pieza: str,
    clase: str,
    etiqueta: str,
    segundos: float,
    referencia: ClipQuality | None,
) -> Registro:
    """Graba `segundos` de la fuente en FFV1 y evalua la calidad del clip.

    Separada de la CLI para poder testearse con una fuente sintetica.
    """
    destino.parent.mkdir(parents=True, exist_ok=True)
    escritor: cv2.VideoWriter | None = None
    calidades = []
    frames = 0
    resolucion = (0, 0)
    inicio = time.perf_counter()
    limite = round(segundos * fuente.fps)

    try:
        for indice, _, imagen in fuente.frames():
            if indice >= limite:
                break
            if escritor is None:
                alto, ancho = imagen.shape[:2]
                resolucion = (ancho, alto)
                escritor = cv2.VideoWriter(
                    str(destino), cv2.VideoWriter_fourcc(*CODEC), fuente.fps, resolucion
                )
                if not escritor.isOpened():
                    raise OSError(f"OpenCV no puede escribir {CODEC} en {destino}")
            escritor.write(imagen)
            if indice % MUESTREO_QC == 0:
                calidades.append(assess_frame(imagen))
            frames += 1
    finally:
        if escritor is not None:
            escritor.release()

    if frames == 0:
        raise OSError("la camara no entrego ningun frame")

    transcurrido = time.perf_counter() - inicio
    calidad = assess_clip(calidades)
    return Registro(
        pieza=pieza,
        clase=clase,
        etiqueta=etiqueta,
        ruta=str(destino),
        segundos_pedidos=segundos,
        frames=frames,
        fps_medido=round(frames / transcurrido, 2) if transcurrido > 0 else 0.0,
        fps_declarado=fuente.fps,
        resolucion=resolucion,
        controles_aplicados=getattr(fuente, "applied_controls", {}),
        calidad=calidad.summary(),
        avisos_deriva=list(compare_to_reference(calidad, referencia)) if referencia else [],
        fecha=time.strftime("%Y-%m-%d %H:%M:%S"),
    )


def _abrir(args: argparse.Namespace) -> OpenCvVideoSource:
    controles = CameraControls(frame_width=args.ancho, frame_height=args.alto)
    if args.rtsp:
        return OpenCvVideoSource.from_rtsp(args.rtsp, controls=controles)
    return OpenCvVideoSource.from_webcam(args.camara, controls=controles)


def _informar_controles(fuente: OpenCvVideoSource) -> None:
    print("\nControles de camara (lo que el driver ACEPTO, no lo que aplico):")
    for nombre, ok in fuente.applied_controls.items():
        print(f"  {'si' if ok else 'NO':>3}  {nombre}")
    if not fuente.fps_is_measured:
        print(f"  El driver no reporta fps; se asume {fuente.fps}.")
    print("La exposicion real se comprueba midiendo los frames al terminar cada clip.")


def _preview(fuente: OpenCvVideoSource, carpeta: Path) -> int:
    """Guarda un frame para revisar encuadre e iluminacion antes de grabar.

    La build de OpenCV del proyecto es headless (sin ventanas), asi que el
    encuadre se revisa sobre una imagen en disco en vez de en una ventana en vivo.
    """
    muestras = []
    ultimo = None
    for indice, _, imagen in fuente.frames():
        muestras.append(assess_frame(imagen))
        ultimo = imagen
        if indice >= 30:  # ~1 s: da tiempo a que la camara estabilice
            break
    if ultimo is None:
        print("La camara no entrego ningun frame.", file=sys.stderr)
        return 1

    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / "preview.png"
    cv2.imwrite(str(ruta), ultimo)
    calidad = assess_clip(muestras[5:] or muestras)

    alto, ancho = ultimo.shape[:2]
    print(f"\nPreview guardado en {ruta}  ({ancho}x{alto})")
    print(f"  brillo medio  {calidad.brightness_mean:.2f}  (0=negro, 1=blanco)")
    print(f"  recorte max   {calidad.clipped_max:.1%}")
    print(f"  nitidez       {calidad.sharpness_median:.1f}")
    for aviso in calidad.warnings:
        print(f"  AVISO: {aviso}")
    if calidad.is_ok:
        print("  Sin avisos. Abre la imagen y revisa el encuadre antes de grabar.")
    return 0 if calidad.is_ok else 3


def _registros(manifiesto: Path) -> list[dict[str, object]]:
    if not manifiesto.is_file():
        return []
    return [
        json.loads(linea) for linea in manifiesto.read_text(encoding="utf-8").splitlines() if linea
    ]


def _siguiente_id(manifiesto: Path, etiqueta: str) -> str:
    prefijo = PREFIJO[etiqueta]
    usados = [
        int(str(r["pieza"]).split("-")[1])
        for r in _registros(manifiesto)
        if str(r["pieza"]).startswith(f"{prefijo}-")
    ]
    return f"{prefijo}-{(max(usados) + 1) if usados else 0:03d}"


def _ya_existe(manifiesto: Path, pieza: str) -> bool:
    return any(r["pieza"] == pieza for r in _registros(manifiesto))


def _referencia(manifiesto: Path) -> ClipQuality | None:
    """Calidad del primer clip de la clase: la referencia de la sesion."""
    registros = _registros(manifiesto)
    if not registros:
        return None
    c = registros[0]["calidad"]
    assert isinstance(c, dict)
    return ClipQuality(
        n_frames=int(c["n_frames"]),
        brightness_mean=float(c["brightness_mean"]),
        brightness_cv=float(c["brightness_cv"]),
        clipped_max=float(c["clipped_max"]),
        sharpness_median=float(c["sharpness_median"]),
    )


def _anadir(manifiesto: Path, registro: Registro) -> None:
    manifiesto.parent.mkdir(parents=True, exist_ok=True)
    with manifiesto.open("a", encoding="utf-8") as fichero:
        fichero.write(json.dumps(registro.__dict__, ensure_ascii=False) + "\n")


def _informar(registro: Registro, manifiesto: Path) -> None:
    c = registro.calidad
    print(
        f"\nGuardado {registro.ruta}\n"
        f"  {registro.frames} frames, {registro.fps_medido} fps medidos, "
        f"{registro.resolucion[0]}x{registro.resolucion[1]}\n"
        f"  brillo {c['brightness_mean']}  variacion {float(c['brightness_cv']):.1%}  "  # type: ignore[arg-type]
        f"recorte {float(c['clipped_max']):.1%}  nitidez {c['sharpness_median']}"  # type: ignore[arg-type]
    )
    avisos = [*c["warnings"], *registro.avisos_deriva]  # type: ignore[misc]
    for aviso in avisos:
        print(f"  AVISO: {aviso}")
    print(
        "  Clip correcto." if not avisos else "  Revisa los avisos: probablemente conviene repetir."
    )
    _resumen(manifiesto)


def _resumen(manifiesto: Path) -> None:
    registros = _registros(manifiesto)
    print(f"\nProgreso ({manifiesto.parent.name}):")
    for etiqueta, objetivo in OBJETIVO.items():
        n = sum(1 for r in registros if r["etiqueta"] == etiqueta)
        barra = "#" * min(n, objetivo) + "." * max(objetivo - n, 0)
        print(f"  {etiqueta:<8} {n:>3}/{objetivo}  [{barra}]")
    con_avisos = [r["pieza"] for r in registros if r["calidad"]["warnings"] or r["avisos_deriva"]]  # type: ignore[index]
    if con_avisos:
        print(f"  Con avisos de calidad: {', '.join(map(str, con_avisos))}")


def _cuenta_atras(segundos: int) -> None:
    for restante in range(segundos, 0, -1):
        print(f"  grabando en {restante}...", flush=True)
        time.sleep(1)
    print("  GRABANDO", flush=True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--clase", required=True, help="tipo de pieza: tapas, tuercas...")
    p.add_argument("--etiqueta", choices=list(PREFIJO))
    p.add_argument(
        "--pieza", help="ID explicito; por defecto el siguiente libre (ok-000, def-000...)"
    )
    p.add_argument("--segundos", type=float, default=15.0, help="10-20 s segun el protocolo v2")
    p.add_argument("--espera", type=int, default=3, help="cuenta atras antes de grabar")
    p.add_argument("--camara", type=int, default=0)
    p.add_argument("--rtsp", help="URL rtsp:// en lugar de webcam USB")
    p.add_argument("--ancho", type=int, default=1920)
    p.add_argument("--alto", type=int, default=1080)
    p.add_argument("--out", type=Path, default=DESTINO_POR_DEFECTO)
    p.add_argument("--preview", action="store_true")
    p.add_argument("--resumen", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
