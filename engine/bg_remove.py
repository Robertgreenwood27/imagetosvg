"""
Border-connected white (or near-white) background removal.

Algorithm
---------
1. Convert the image to RGBA if it isn't already.
2. Build a boolean mask of pixels that are "white-ish" — luminance above
   `luma_threshold` (0-255) AND each channel within `channel_tolerance` of
   255.  This catches pure white and common scan/photo backgrounds that are
   slightly off-white or yellow-tinted.
3. Flood-fill from every border pixel that satisfies the white-ish test
   (8-connected, iterative to avoid Python recursion limits).
4. Set the alpha channel of all flood-filled pixels to 0.
5. Return the resulting RGBA image.

The result feeds naturally into the quantize step: transparent pixels are
already excluded from k-means by the existing `alpha_threshold` logic, so
the background simply disappears from the palette.

Parameters
----------
luma_threshold : int  (0-255, default 230)
    Pixels whose perceptual luminance is AT OR ABOVE this value are
    candidates for removal. Lower = more aggressive (removes cream/beige).
    Raise to 245+ to only strip pure or near-pure white.

channel_tolerance : int  (0-255, default 30)
    Each RGB channel must be within this distance of 255 to qualify.
    At 30, a pixel like (225, 240, 235) passes; at 10 only near-pure
    white does.

feather_px : int  (0-4, default 1)
    If > 0, erode the kept region by this many pixels so the hard
    mask edge gets softened. Avoids a harsh white halo on the result.
    Uses scipy binary erosion; set to 0 to disable (and drop scipy dep).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class BgRemoveConfig:
    luma_threshold: int = 230       # pixels at or above this luma are candidates
    channel_tolerance: int = 30     # max distance from 255 per channel
    feather_px: int = 1             # erosion radius on kept mask edge (0 = off)


def remove_white_background(
    image: Image.Image,
    config: BgRemoveConfig | None = None,
) -> Image.Image:
    """
    Remove the border-connected white/near-white background from *image*.

    Returns an RGBA image with the background pixels set to alpha=0.
    The original image is not modified.
    """
    cfg = config or BgRemoveConfig()
    rgba = image.convert("RGBA")
    arr = np.asarray(rgba, dtype=np.uint8).copy()

    h, w = arr.shape[:2]
    r, g, b, a = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]

    # --- Step 1: build white-ish candidate mask ----------------------------
    # Perceptual luminance (BT.601): Y = 0.299R + 0.587G + 0.114B
    luma = (0.299 * r.astype(np.float32)
            + 0.587 * g.astype(np.float32)
            + 0.114 * b.astype(np.float32))

    candidate = (
        (luma >= cfg.luma_threshold)
        & (r.astype(np.int16) >= 255 - cfg.channel_tolerance)
        & (g.astype(np.int16) >= 255 - cfg.channel_tolerance)
        & (b.astype(np.int16) >= 255 - cfg.channel_tolerance)
        & (a >= 128)   # don't re-process already-transparent pixels
    )

    # --- Step 2: flood-fill from border ------------------------------------
    visited = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    # Seed from every border pixel that is a candidate
    def _enqueue_if_candidate(row: int, col: int) -> None:
        if candidate[row, col] and not visited[row, col]:
            visited[row, col] = True
            queue.append((row, col))

    for col in range(w):
        _enqueue_if_candidate(0, col)
        _enqueue_if_candidate(h - 1, col)
    for row in range(1, h - 1):
        _enqueue_if_candidate(row, 0)
        _enqueue_if_candidate(row, w - 1)

    # 8-connected BFS
    neighbours = ((-1, -1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1),  (1, 0),  (1, 1))

    while queue:
        row, col = queue.popleft()
        for dr, dc in neighbours:
            nr, nc = row + dr, col + dc
            if 0 <= nr < h and 0 <= nc < w:
                if candidate[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    queue.append((nr, nc))

    # `visited` is now True for every pixel to be removed.

    # --- Step 3: optional feathering (erode the *kept* region) ------------
    if cfg.feather_px > 0:
        try:
            from scipy import ndimage

            kept = ~visited
            # Erode the kept region so the edge pixels (likely partially-white)
            # are removed too, eliminating the white halo artifact.
            struct = ndimage.generate_binary_structure(2, 2)  # 8-connected
            for _ in range(cfg.feather_px):
                kept = ndimage.binary_erosion(kept, structure=struct)
            visited = ~kept
        except ImportError:
            pass  # scipy not available — skip feathering silently

    # --- Step 4: apply mask -----------------------------------------------
    arr[visited, 3] = 0

    return Image.fromarray(arr, mode="RGBA")