"""Curacion de datasets derivados de video.

Es el modulo donde vive la contribucion cientifica de la tesis: muestreo por
diferencia perceptual, deduplicacion por hashing y particion anti-fuga.

ORDEN DE OPERACIONES OBLIGATORIO (seccion 10 del protocolo de evaluacion):

    extraer -> agrupar por pieza -> PARTICIONAR por grupo -> deduplicar dentro del split

Deduplicar antes de particionar eliminaria de test los frames que duplican a los
de train, ocultando la fuga en lugar de corregirla.
"""

from inspeccion.application.curation.controlled_leakage import (
    InsufficientDuplicatesError,
    PairedArms,
    build_leaked_arms,
)
from inspeccion.application.curation.dedup import (
    DedupResult,
    HashedFrame,
    deduplicate,
    hamming,
    redundancy_profile,
)
from inspeccion.application.curation.extraction import ExtractionResult, FrameExtractor
from inspeccion.application.curation.pipeline import CurationPipeline, CurationReport
from inspeccion.application.curation.sampling import PerceptualDifferenceSampler
from inspeccion.application.curation.splitting import (
    DEFAULT_RATIOS,
    InsufficientGroupsError,
    assert_no_group_leakage,
    audit_leakage,
    read_manifest,
    split_by_group,
    split_naive_by_frame,
    write_manifest,
)

__all__ = [
    "DEFAULT_RATIOS",
    "CurationPipeline",
    "CurationReport",
    "DedupResult",
    "ExtractionResult",
    "FrameExtractor",
    "HashedFrame",
    "InsufficientDuplicatesError",
    "InsufficientGroupsError",
    "PairedArms",
    "PerceptualDifferenceSampler",
    "assert_no_group_leakage",
    "audit_leakage",
    "build_leaked_arms",
    "deduplicate",
    "hamming",
    "read_manifest",
    "redundancy_profile",
    "split_by_group",
    "split_naive_by_frame",
    "write_manifest",
]
