"""Configuracion del sistema.

Valores por defecto alineados con el protocolo de evaluacion: pHash de 64 bits,
umbral de deduplicacion 4 (punto medio del barrido de E4) y particion por pieza.
Los defaults son los del experimento, no los mas permisivos: quien quiera la
particion ingenua tiene que pedirla explicitamente.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from inspeccion.adapters.hashing import HashAlgorithm
from inspeccion.adapters.ml_engine import AnomalyModel
from inspeccion.domain.models import SplitStrategy

ENV_PREFIX = "INSPECCION_"


@dataclass(frozen=True, slots=True)
class Settings:
    storage_root: Path = Path("ml/data/artifacts")

    hash_algorithm: HashAlgorithm = HashAlgorithm.PHASH
    hash_size: int = 8

    dedup_max_distance: int = 4

    sampler_min_difference: float = 0.02
    sampler_min_interval: int = 1
    sampler_max_interval: int | None = 90

    split_strategy: SplitStrategy = SplitStrategy.BY_GROUP
    seed: int = 0

    anomaly_model: AnomalyModel = AnomalyModel.PATCHCORE
    coreset_sampling_ratio: float = 0.1
    batch_size: int = 32
    accelerator: str = "auto"

    live_max_frames: int = 3600
    """Tope de frames para fuentes en vivo, que no terminan solas. 3600 = 2 min a 30 fps."""

    def __post_init__(self) -> None:
        if self.hash_size < 2:
            raise ValueError(f"hash_size debe ser >= 2, llego {self.hash_size}")
        if self.dedup_max_distance < 0:
            raise ValueError(f"dedup_max_distance no puede ser negativo: {self.dedup_max_distance}")
        if self.dedup_max_distance > self.hash_size**2:
            raise ValueError(
                f"dedup_max_distance ({self.dedup_max_distance}) excede los "
                f"{self.hash_size**2} bits del hash: eliminaria todos los frames"
            )
        if self.live_max_frames <= 0:
            raise ValueError(f"live_max_frames debe ser positivo: {self.live_max_frames}")
        if not 0.0 < self.coreset_sampling_ratio <= 1.0:
            raise ValueError(
                f"coreset_sampling_ratio debe estar en (0, 1]: {self.coreset_sampling_ratio}"
            )
        if self.batch_size <= 0:
            raise ValueError(f"batch_size debe ser positivo: {self.batch_size}")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        """Construye desde variables de entorno con prefijo `INSPECCION_`."""
        source = os.environ if env is None else env
        defaults = cls()

        def read(name: str) -> str | None:
            return source.get(f"{ENV_PREFIX}{name.upper()}")

        raw_max_interval = read("sampler_max_interval")
        if raw_max_interval is None:
            max_interval = defaults.sampler_max_interval
        elif raw_max_interval.strip().lower() in {"", "none"}:
            max_interval = None
        else:
            max_interval = _as_int("sampler_max_interval", raw_max_interval, 0)

        return cls(
            storage_root=Path(read("storage_root") or defaults.storage_root),
            hash_algorithm=HashAlgorithm(read("hash_algorithm") or defaults.hash_algorithm),
            hash_size=_as_int("hash_size", read("hash_size"), defaults.hash_size),
            dedup_max_distance=_as_int(
                "dedup_max_distance", read("dedup_max_distance"), defaults.dedup_max_distance
            ),
            sampler_min_difference=_as_float(
                "sampler_min_difference",
                read("sampler_min_difference"),
                defaults.sampler_min_difference,
            ),
            sampler_min_interval=_as_int(
                "sampler_min_interval", read("sampler_min_interval"), defaults.sampler_min_interval
            ),
            sampler_max_interval=max_interval,
            split_strategy=SplitStrategy(read("split_strategy") or defaults.split_strategy),
            seed=_as_int("seed", read("seed"), defaults.seed),
            anomaly_model=AnomalyModel(read("anomaly_model") or defaults.anomaly_model),
            coreset_sampling_ratio=_as_float(
                "coreset_sampling_ratio",
                read("coreset_sampling_ratio"),
                defaults.coreset_sampling_ratio,
            ),
            batch_size=_as_int("batch_size", read("batch_size"), defaults.batch_size),
            accelerator=read("accelerator") or defaults.accelerator,
            live_max_frames=_as_int(
                "live_max_frames", read("live_max_frames"), defaults.live_max_frames
            ),
        )


def _as_int(name: str, raw: str | None, default: int) -> int:
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as error:
        raise ValueError(f"{ENV_PREFIX}{name.upper()} no es un entero: {raw!r}") from error


def _as_float(name: str, raw: str | None, default: float) -> float:
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as error:
        raise ValueError(f"{ENV_PREFIX}{name.upper()} no es un numero: {raw!r}") from error
