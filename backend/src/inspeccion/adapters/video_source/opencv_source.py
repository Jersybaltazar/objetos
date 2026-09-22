"""Adaptador de fuente de video sobre OpenCV.

Implementa el puerto `VideoSource` para las tres fuentes del alcance: archivo,
webcam USB y cámara IP por RTSP. Es el mismo adaptador para las tres porque
`cv2.VideoCapture` las unifica; lo que cambia es el URI y si procede fijar los
controles de cámara.

Los controles de cámara no son un detalle opcional. El protocolo de captura
prohibe explicitamente autofoco, autoexposicion y balance de blancos automatico:
una pieza normal grabada con exposicion variable produce frames que el modelo de
anomalia no puede distinguir de un defecto real. `CameraControls` los fija, y
`applied_controls` reporta cuales acepto el driver — porque `VideoCapture.set()`
falla en silencio segun el backend y el sistema operativo, y dar por hecho que se
aplicaron invalidaria la captura sin que nadie se entere.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

import numpy as np

from inspeccion.domain.ports import Image


class VideoSourceError(RuntimeError):
    """La fuente de video no pudo abrirse o quedo inutilizable."""


@dataclass(frozen=True, slots=True)
class CameraControls:
    """Fijacion de los automatismos de la camara (solo fuentes en vivo).

    `None` significa "no tocar". Los valores por defecto desactivan los tres
    automatismos que el protocolo de captura prohibe.
    """

    autofocus: bool = False
    auto_exposure: bool = False
    auto_white_balance: bool = False
    focus: float | None = None
    exposure: float | None = None
    white_balance_temperature: float | None = None
    frame_width: int | None = None
    frame_height: int | None = None


class OpenCvVideoSource:
    """Adaptador que satisface `VideoSource`.

    El flujo se consume una sola vez: `frames()` no es re-iterable, igual que la
    fuente real. Usar como gestor de contexto para garantizar la liberacion del
    dispositivo.
    """

    def __init__(
        self,
        uri: str | int,
        *,
        controls: CameraControls | None = None,
        fallback_fps: float = 30.0,
    ) -> None:
        import cv2

        if fallback_fps <= 0:
            raise ValueError(f"fallback_fps debe ser positivo, llego {fallback_fps}")

        self._cv2 = cv2
        self._uri = uri
        self._fallback_fps = fallback_fps
        self._capture = cv2.VideoCapture(uri)
        if not self._capture.isOpened():
            raise VideoSourceError(f"no se pudo abrir la fuente de video: {uri!r}")

        self._applied: dict[str, bool] = {}
        if controls is not None:
            self._applied = self._apply_controls(controls)

        reported = float(self._capture.get(cv2.CAP_PROP_FPS))
        self._fps = reported if reported > 0 else fallback_fps
        self._fps_is_measured = reported > 0

    @classmethod
    def from_file(cls, path: Path | str, *, fallback_fps: float = 30.0) -> OpenCvVideoSource:
        """Fuente de archivo. No se tocan controles: el video ya esta grabado."""
        resolved = Path(path)
        if not resolved.is_file():
            raise VideoSourceError(f"el archivo de video no existe: {resolved}")
        return cls(str(resolved), fallback_fps=fallback_fps)

    @classmethod
    def from_webcam(
        cls,
        index: int = 0,
        *,
        controls: CameraControls | None = None,
        fallback_fps: float = 30.0,
    ) -> OpenCvVideoSource:
        """Webcam USB. Fija los automatismos salvo que se pidan otros controles."""
        return cls(index, controls=controls or CameraControls(), fallback_fps=fallback_fps)

    @classmethod
    def from_rtsp(
        cls,
        url: str,
        *,
        controls: CameraControls | None = None,
        fallback_fps: float = 25.0,
    ) -> OpenCvVideoSource:
        """Camara IP. `fallback_fps` 25 por ser el valor habitual en RTSP."""
        if not url.lower().startswith(("rtsp://", "rtsps://")):
            raise ValueError(f"se esperaba una URL rtsp://, llego {url!r}")
        return cls(url, controls=controls, fallback_fps=fallback_fps)

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def fps_is_measured(self) -> bool:
        """False si el driver no reporto fps y se esta usando el valor de respaldo.

        Debe quedar registrado junto a la captura: un timestamp derivado de un fps
        supuesto no es una medida.
        """
        return self._fps_is_measured

    @property
    def applied_controls(self) -> dict[str, bool]:
        """Que controles de camara acepto el driver. Vacio para fuentes de archivo."""
        return dict(self._applied)

    def frames(self) -> Iterator[tuple[int, float, Image]]:
        """Produce (indice, timestamp_en_segundos, imagen BGR)."""
        index = 0
        while True:
            ok, frame = self._capture.read()
            if not ok:
                break
            yield index, self._timestamp(index), np.asarray(frame, dtype=np.uint8)
            index += 1

    def close(self) -> None:
        if self._capture.isOpened():
            self._capture.release()

    def __enter__(self) -> OpenCvVideoSource:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def _timestamp(self, index: int) -> float:
        """Prefiere el timestamp del contenedor; cae al indice solo si no hay.

        Los streams en vivo reportan 0 en `CAP_PROP_POS_MSEC`, y el primer frame
        de un archivo tambien: por eso el respaldo se aplica a cualquier valor no
        positivo salvo en el frame 0, donde 0.0 es el valor correcto.
        """
        position = float(self._capture.get(self._cv2.CAP_PROP_POS_MSEC))
        if position > 0:
            return position / 1000.0
        return index / self._fps

    def _apply_controls(self, controls: CameraControls) -> dict[str, bool]:
        """Aplica los controles y reporta cuales acepto el driver.

        `VideoCapture.set()` devuelve False —o incluso True sin efecto— segun
        backend y sistema operativo. No se lanza excepcion: una webcam que no
        permite fijar el foco sigue siendo utilizable, pero el hecho debe quedar
        registrado en la documentacion de la captura.
        """
        cv2 = self._cv2
        # CAP_PROP_AUTO_EXPOSURE usa 0.25 = manual y 0.75 = automatico en el
        # backend V4L2, y 0/1 en DirectShow. Se usa la convencion V4L2 por ser la
        # del entorno de despliegue (Linux/edge); en Windows el driver lo
        # rechazara y quedara reportado como no aplicado.
        exposure_mode = 0.75 if controls.auto_exposure else 0.25
        wb_temperature = controls.white_balance_temperature
        requested: list[tuple[str, int, float]] = [
            ("autofocus", cv2.CAP_PROP_AUTOFOCUS, float(controls.autofocus)),
            ("auto_exposure", cv2.CAP_PROP_AUTO_EXPOSURE, exposure_mode),
            ("auto_white_balance", cv2.CAP_PROP_AUTO_WB, float(controls.auto_white_balance)),
        ]
        optional: list[tuple[str, int, float | None]] = [
            ("focus", cv2.CAP_PROP_FOCUS, controls.focus),
            ("exposure", cv2.CAP_PROP_EXPOSURE, controls.exposure),
            ("white_balance_temperature", cv2.CAP_PROP_WB_TEMPERATURE, wb_temperature),
            ("frame_width", cv2.CAP_PROP_FRAME_WIDTH, controls.frame_width),
            ("frame_height", cv2.CAP_PROP_FRAME_HEIGHT, controls.frame_height),
        ]
        requested.extend(
            (name, prop, float(value)) for name, prop, value in optional if value is not None
        )

        return {name: bool(self._capture.set(prop, value)) for name, prop, value in requested}
