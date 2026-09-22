"""Adaptador de almacenamiento sobre el sistema de archivos local.

Cubre la frontera de almacenamiento para el MVP y para el escenario offline
(RNF-04). El adaptador de MinIO/S3 implementa los mismos puertos y se inyecta en
su lugar sin tocar ningun caso de uso.

Los frames se guardan en **PNG, sin perdida**, a proposito: un artefacto de
compresion JPEG alrededor de un borde es indistinguible de una rayadura o una
mancha, que es justamente lo que el modelo debe detectar. Un dataset comprimido
con perdida contamina el experimento de forma irreversible.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from inspeccion.domain.ports import Image


class StorageKeyError(ValueError):
    """La clave solicitada escapa de la raiz del almacen."""


@dataclass(frozen=True, slots=True)
class FilesystemArtifactStore:
    """Adaptador que satisface `ArtifactStore`."""

    root: Path

    def put(self, key: str, data: bytes) -> str:
        destination = self._resolve(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        return str(destination)

    def get(self, key: str) -> bytes:
        return self._resolve(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._resolve(key).is_file()

    def _resolve(self, key: str) -> Path:
        return _resolve_within(self.root, key)


@dataclass(frozen=True, slots=True)
class FilesystemFrameWriter:
    """Adaptador que satisface `FrameWriter`. Codifica en PNG sin perdida."""

    root: Path

    def write(self, image: Image, key: str) -> str:
        import cv2

        destination = _resolve_within(self.root, key)
        destination.parent.mkdir(parents=True, exist_ok=True)

        ok, buffer = cv2.imencode(".png", image)
        if not ok:
            raise OSError(f"OpenCV no pudo codificar el frame como PNG: {key!r}")
        destination.write_bytes(buffer.tobytes())
        return str(destination)


def _resolve_within(root: Path, key: str) -> Path:
    """Resuelve `key` bajo `root` rechazando cualquier escape del arbol.

    La clave la construye el sistema, pero el nombre de la pieza y del proyecto
    vienen de la interfaz web: sin esta comprobacion, un identificador con `..`
    permitiria escribir fuera del almacen.
    """
    candidate = PurePosixPath(key.replace("\\", "/"))
    if candidate.is_absolute() or ".." in candidate.parts:
        raise StorageKeyError(f"clave de almacenamiento no permitida: {key!r}")

    base = root.resolve()
    resolved = (base / candidate).resolve()
    if not resolved.is_relative_to(base):
        raise StorageKeyError(f"clave de almacenamiento no permitida: {key!r}")
    return resolved
