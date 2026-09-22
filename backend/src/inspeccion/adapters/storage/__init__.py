"""Adaptadores de almacenamiento (disco local; MinIO/S3 en fase posterior)."""

from inspeccion.adapters.storage.filesystem import (
    FilesystemArtifactStore,
    FilesystemFrameWriter,
    StorageKeyError,
)

__all__ = ["FilesystemArtifactStore", "FilesystemFrameWriter", "StorageKeyError"]
