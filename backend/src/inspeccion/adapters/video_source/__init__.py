"""Adaptadores de fuente de video (archivo, webcam USB, RTSP)."""

from inspeccion.adapters.video_source.opencv_source import (
    CameraControls,
    OpenCvVideoSource,
    VideoSourceError,
)

__all__ = ["CameraControls", "OpenCvVideoSource", "VideoSourceError"]
