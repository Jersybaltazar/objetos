from __future__ import annotations

from collections.abc import Callable

import pytest

from inspeccion.domain.models import FrameRef, GroupId, Label


def make_frames(
    *,
    n_normal_groups: int = 30,
    n_defect_groups: int = 15,
    frames_per_group: int = 40,
) -> list[FrameRef]:
    """Dataset sintetico con la forma que exige la seccion 8 del protocolo.

    Cada "pieza" es un grupo con sus frames; el nombre del video codifica la
    pieza, igual que en el protocolo de captura v2.
    """
    frames: list[FrameRef] = []
    for label, count, prefix in (
        (Label.NORMAL, n_normal_groups, "ok"),
        (Label.DEFECT, n_defect_groups, "def"),
    ):
        for piece in range(count):
            group_id = GroupId(f"{prefix}-{piece:03d}")
            video = f"{group_id}.mp4"
            for index in range(frames_per_group):
                frames.append(
                    FrameRef(
                        path=f"frames/{group_id}/{index:05d}.png",
                        group_id=group_id,
                        label=label,
                        source_video=video,
                        frame_index=index,
                        timestamp_s=index / 30.0,
                    )
                )
    return frames


@pytest.fixture
def frames() -> list[FrameRef]:
    return make_frames()


@pytest.fixture
def build_frames() -> Callable[..., list[FrameRef]]:
    """Permite construir datasets de otras dimensiones sin importar conftest."""
    return make_frames
