from __future__ import annotations

from pathlib import Path
from typing import Iterable

import imageio.v2 as imageio
import numpy as np


SUPPORTED_FORMATS = {".mp4", ".webm", ".gif"}


def _as_rgb_uint8(frame: np.ndarray) -> np.ndarray:
    """Validate and normalize a rendered RGB frame."""
    array = np.asarray(frame)

    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(
            f"Expected an RGB frame with shape (height, width, 3), "
            f"got {array.shape}."
        )

    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and array.max() <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(array)


def save_video(
    frames: Iterable[np.ndarray],
    output_path: str | Path,
    fps: int = 20,
) -> Path:
    """
    Save the same RGB frame sequence as MP4, WebM, or GIF.

    The renderer produces RGB frames, so no RGB/BGR conversion is performed.
    MP4 uses H.264, WebM uses VP9, and GIF uses imageio's animated GIF writer.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported output format {output_path.suffix!r}. "
            f"Choose one of: {', '.join(sorted(SUPPORTED_FORMATS))}."
        )

    prepared_frames = [_as_rgb_uint8(frame) for frame in frames]
    if not prepared_frames:
        raise ValueError("Cannot write a video with zero frames.")

    first_shape = prepared_frames[0].shape
    if any(frame.shape != first_shape for frame in prepared_frames):
        raise ValueError("All video frames must have the same dimensions.")

    suffix = output_path.suffix.lower()

    writer_kwargs: dict[str, object] = {"fps": fps}

    if suffix == ".mp4":
        writer_kwargs.update(
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=None,
        )
    elif suffix == ".webm":
        writer_kwargs.update(
            codec="libvpx-vp9",
            macro_block_size=None,
        )
    elif suffix == ".gif":
        # GIF is kept mainly as a convenient preview format.
        writer_kwargs.pop("fps", None)
        writer_kwargs["duration"] = 1.0 / fps
        writer_kwargs["loop"] = 0

    with imageio.get_writer(output_path, **writer_kwargs) as writer:
        for frame in prepared_frames:
            writer.append_data(frame)

    return output_path
