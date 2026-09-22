"""Extraccion de frames desde una fuente de video (RF-03).

Primera etapa del pipeline de curacion. Depende exclusivamente de puertos
(`VideoSource`, `PerceptualHasher`, `FrameWriter`), de modo que se testea con
dobles en memoria, sin camara, sin OpenCV y sin tocar el disco.

Decision de diseno: el hash perceptual se calcula **aqui**, con el frame ya
decodificado en memoria, no despues leyendo de disco. Un video de 15 s a 30 fps
son 450 decodificaciones; repetirlas para hashear duplicaria el costo de la etapa
mas cara del pipeline sin ninguna ganancia.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from inspeccion.application.curation.dedup import HashedFrame
from inspeccion.application.curation.sampling import PerceptualDifferenceSampler
from inspeccion.domain.models import FrameRef, GroupId, Label
from inspeccion.domain.ports import FrameWriter, PerceptualHasher, VideoSource


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    """Resultado de extraer una pieza fisica."""

    frames: tuple[HashedFrame, ...]
    seen: int
    kept: int
    fps: float

    @property
    def retained_fraction(self) -> float:
        return self.kept / self.seen if self.seen else 1.0

    def summary(self) -> dict[str, float | int]:
        return {
            "seen": self.seen,
            "kept": self.kept,
            "retained_fraction": round(self.retained_fraction, 6),
            "fps": self.fps,
        }


@dataclass(frozen=True, slots=True)
class FrameExtractor:
    """Caso de uso: convertir un video de una pieza en frames hasheados.

    Args:
        hasher: puerto de hashing perceptual.
        writer: puerto de persistencia de frames.
        sampler_factory: construye un muestreador **nuevo por cada pieza**. El
            muestreador guarda el ultimo frame conservado como referencia, y ese
            estado no puede cruzar de una pieza a la siguiente: el primer frame de
            una pieza se compararia con el ultimo de la anterior.
        key_template: plantilla de la clave de almacenamiento.
        max_frames: tope de frames leidos. Obligatorio para fuentes en vivo, que
            no terminan por si solas.
    """

    hasher: PerceptualHasher
    writer: FrameWriter
    sampler_factory: Callable[[], PerceptualDifferenceSampler]
    key_template: str = "frames/{group_id}/{index:05d}.png"
    max_frames: int | None = None

    def extract(
        self,
        source: VideoSource,
        *,
        group_id: GroupId,
        label: Label,
        source_video: str | None = None,
    ) -> ExtractionResult:
        """Recorre la fuente una vez y persiste solo los frames conservados."""
        sampler = self.sampler_factory()
        video_name = source_video or f"{group_id}.mp4"
        kept: list[HashedFrame] = []

        for index, timestamp, image in source.frames():
            if self.max_frames is not None and index >= self.max_frames:
                break
            if not sampler.should_keep(image):
                continue

            key = self.key_template.format(group_id=group_id, index=index)
            kept.append(
                HashedFrame(
                    frame=FrameRef(
                        path=self.writer.write(image, key),
                        group_id=group_id,
                        label=label,
                        source_video=video_name,
                        frame_index=index,
                        timestamp_s=timestamp,
                    ),
                    phash=self.hasher.hash(image),
                )
            )

        return ExtractionResult(
            frames=tuple(kept),
            seen=sampler.seen,
            kept=sampler.kept,
            fps=source.fps,
        )
