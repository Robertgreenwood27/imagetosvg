"""
Color quantization in OKLab space.

Why OKLab: RGB distance lies about perceived color similarity. OKLab is a modern
perceptually-uniform space (Ottosson 2020). Clustering here produces palettes
that look right to human eyes, not to pixel math.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans


# -- sRGB <-> OKLab ----------------------------------------------------------
# Reference: https://bottosson.github.io/posts/oklab/

def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """sRGB in [0,1] -> linear RGB."""
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    """Linear RGB -> sRGB in [0,1]."""
    return np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * (rgb ** (1 / 2.4)) - 0.055)


_M1 = np.array([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
])

_M2 = np.array([
    [0.2104542553,  0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050,  0.4505937099],
    [0.0259040371,  0.7827717662, -0.8086757660],
])


def rgb_to_oklab(rgb_uint8: np.ndarray) -> np.ndarray:
    """(..., 3) uint8 sRGB -> (..., 3) float OKLab."""
    rgb = rgb_uint8.astype(np.float64) / 255.0
    lin = _srgb_to_linear(rgb)
    lms = lin @ _M1.T
    lms_cbrt = np.cbrt(lms)
    return lms_cbrt @ _M2.T


def oklab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """(..., 3) OKLab -> (..., 3) uint8 sRGB."""
    inv_M2 = np.linalg.inv(_M2)
    inv_M1 = np.linalg.inv(_M1)
    lms_cbrt = lab @ inv_M2.T
    lms = lms_cbrt ** 3
    lin = lms @ inv_M1.T
    rgb = _linear_to_srgb(np.clip(lin, 0, 1))
    return np.clip(np.round(rgb * 255), 0, 255).astype(np.uint8)


# -- Quantization ------------------------------------------------------------

def quantize(
    image: Image.Image,
    k: int,
    *,
    sample_size: int = 20000,
    merge_threshold: float = 0.0,
    seed: int = 0,
) -> tuple[Image.Image, np.ndarray]:
    """
    Reduce an image to k perceptually-distinct colors.

    Args:
        image: PIL Image (any mode; will be converted to RGB).
        k: target color count (2-32 reasonable).
        sample_size: pixels to sample for k-means fit (speed). Full image is
            then mapped to the fitted palette.
        merge_threshold: if > 0, merge clusters whose OKLab distance (ΔE_OK)
            is below this value. A good range is 0.01-0.05. Set 0 to disable.
        seed: RNG seed for reproducibility.

    Returns:
        (quantized PIL Image in RGB mode, palette as (n, 3) uint8 RGB array)
    """
    rgb_img = image.convert("RGB")
    arr = np.asarray(rgb_img)
    h, w = arr.shape[:2]
    pixels = arr.reshape(-1, 3)

    # Sample for k-means fit (full clustering on multi-MP images is slow and
    # unnecessary — a representative sample finds the same centers).
    rng = np.random.default_rng(seed)
    if len(pixels) > sample_size:
        idx = rng.choice(len(pixels), size=sample_size, replace=False)
        sample = pixels[idx]
    else:
        sample = pixels

    sample_lab = rgb_to_oklab(sample)

    km = KMeans(n_clusters=k, n_init=4, random_state=seed)
    km.fit(sample_lab)
    centers_lab = km.cluster_centers_

    # Optional: merge near-duplicate centers (perceptually identical).
    if merge_threshold > 0:
        centers_lab = _merge_close(centers_lab, merge_threshold)

    # Map every full-res pixel to nearest center (in OKLab).
    all_lab = rgb_to_oklab(pixels)
    # Squared distance is fine for nearest-neighbor.
    d2 = ((all_lab[:, None, :] - centers_lab[None, :, :]) ** 2).sum(-1)
    labels = d2.argmin(axis=1)

    palette_rgb = oklab_to_rgb(centers_lab)
    out = palette_rgb[labels].reshape(h, w, 3)
    return Image.fromarray(out, mode="RGB"), palette_rgb


def _merge_close(centers_lab: np.ndarray, threshold: float) -> np.ndarray:
    """Iteratively merge cluster centers within `threshold` OKLab distance."""
    centers = [c for c in centers_lab]
    changed = True
    while changed and len(centers) > 1:
        changed = False
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                d = np.linalg.norm(centers[i] - centers[j])
                if d < threshold:
                    merged = (centers[i] + centers[j]) / 2
                    centers = [c for k, c in enumerate(centers) if k not in (i, j)]
                    centers.append(merged)
                    changed = True
                    break
            if changed:
                break
    return np.array(centers)


def cleanup_specks(
    image: Image.Image,
    min_region_px: int = 8,
    progress: "callable | None" = None,
) -> Image.Image:
    """
    Replace tiny color regions with their dominant neighbor's color.

    Print-on-demand benefits from this even though it isn't strictly required:
    fewer micro-paths in the trace = smaller, cleaner SVG and crisper print.

    Performance: uses ndimage.find_objects to operate within each region's
    bounding box rather than scanning the whole image per region. On a 4 MP
    photo with thousands of small regions, this is 10000x+ faster than the
    naive approach.
    """
    from scipy import ndimage  # local import; only used here
    say = progress or (lambda _m: None)

    arr = np.asarray(image.convert("RGB"))
    h, w = arr.shape[:2]

    # Encode each (r,g,b) as a single int for fast labeling.
    encoded = (arr[..., 0].astype(np.int64) << 16
               | arr[..., 1].astype(np.int64) << 8
               | arr[..., 2].astype(np.int64))

    total_replaced = 0
    unique_colors = np.unique(encoded)
    say(f"  scanning {len(unique_colors)} color regions…")

    for ci, color in enumerate(unique_colors):
        mask = encoded == color
        labeled, n_regions = ndimage.label(mask)
        if n_regions == 0:
            continue
        sizes = ndimage.sum(mask, labeled, range(1, n_regions + 1))
        small_ids = np.where(sizes < min_region_px)[0] + 1
        if len(small_ids) == 0:
            continue

        # Bounding boxes for each region — keyed by region_id - 1.
        objects = ndimage.find_objects(labeled)

        for rid in small_ids:
            bbox = objects[rid - 1]
            if bbox is None:
                continue
            # Pad bbox by 1 px to capture border pixels via dilation.
            y0 = max(0, bbox[0].start - 1)
            y1 = min(h, bbox[0].stop + 1)
            x0 = max(0, bbox[1].start - 1)
            x1 = min(w, bbox[1].stop + 1)

            local_labeled = labeled[y0:y1, x0:x1]
            local_encoded = encoded[y0:y1, x0:x1]
            local_mask = local_labeled == rid

            dilated = ndimage.binary_dilation(local_mask) & ~local_mask
            if not dilated.any():
                continue

            border_colors = local_encoded[dilated]
            vals, counts = np.unique(border_colors, return_counts=True)
            replacement = vals[counts.argmax()]

            # Write back in-place to the global array's bbox slice.
            local_encoded_view = encoded[y0:y1, x0:x1]
            local_encoded_view[local_mask] = replacement
            total_replaced += 1

        say(f"  color {ci + 1}/{len(unique_colors)}: cleaned {len(small_ids)} regions")

    # Decode back to RGB
    output = np.stack([
        ((encoded >> 16) & 0xFF).astype(np.uint8),
        ((encoded >> 8) & 0xFF).astype(np.uint8),
        (encoded & 0xFF).astype(np.uint8),
    ], axis=-1)
    say(f"  total: replaced {total_replaced} speck regions")

    return Image.fromarray(output, mode="RGB")
