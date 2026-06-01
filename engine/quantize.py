"""
Color quantization in OKLab space.

Alpha-aware version:
- Transparent pixels are ignored during k-means.
- Transparent pixels stay transparent in the returned image.
- Speck cleanup ignores transparent pixels so they do not turn into black regions.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans


# -- sRGB <-> OKLab ----------------------------------------------------------
# Reference: https://bottosson.github.io/posts/oklab/

def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """sRGB in [0,1] -> linear RGB."""
    return np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )


def _linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    """Linear RGB -> sRGB in [0,1]."""
    return np.where(
        rgb <= 0.0031308,
        rgb * 12.92,
        1.055 * (rgb ** (1 / 2.4)) - 0.055,
    )


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
    alpha_threshold: int = 8,
) -> tuple[Image.Image, np.ndarray]:
    """
    Reduce an image to k perceptually-distinct colors.

    Args:
        image: PIL Image. RGBA transparency is preserved.
        k: target visible color count.
        sample_size: pixels to sample for k-means fit.
        merge_threshold: if > 0, merge clusters whose OKLab distance is below this value.
        seed: RNG seed for reproducibility.
        alpha_threshold: alpha below this value is treated as transparent.

    Returns:
        (quantized PIL Image in RGBA mode, palette as (n, 3) uint8 RGB array)
    """
    rgba_img = image.convert("RGBA")
    arr_rgba = np.asarray(rgba_img)
    h, w = arr_rgba.shape[:2]

    rgb = arr_rgba[..., :3]
    alpha = arr_rgba[..., 3]
    opaque_mask = alpha >= alpha_threshold

    if not opaque_mask.any():
        empty = np.zeros((h, w, 4), dtype=np.uint8)
        return Image.fromarray(empty, mode="RGBA"), np.empty((0, 3), dtype=np.uint8)

    opaque_pixels = rgb[opaque_mask].reshape(-1, 3)
    actual_k = max(1, min(k, len(np.unique(opaque_pixels, axis=0))))

    rng = np.random.default_rng(seed)
    if len(opaque_pixels) > sample_size:
        idx = rng.choice(len(opaque_pixels), size=sample_size, replace=False)
        sample = opaque_pixels[idx]
    else:
        sample = opaque_pixels

    sample_lab = rgb_to_oklab(sample)

    km = KMeans(n_clusters=actual_k, n_init=4, random_state=seed)
    km.fit(sample_lab)
    centers_lab = km.cluster_centers_

    if merge_threshold > 0:
        centers_lab = _merge_close(centers_lab, merge_threshold)

    all_lab = rgb_to_oklab(opaque_pixels)
    d2 = ((all_lab[:, None, :] - centers_lab[None, :, :]) ** 2).sum(-1)
    labels = d2.argmin(axis=1)

    palette_rgb = oklab_to_rgb(centers_lab)

    out_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    out_rgba[..., 3] = 0
    out_rgba[opaque_mask, :3] = palette_rgb[labels]
    out_rgba[opaque_mask, 3] = 255

    return Image.fromarray(out_rgba, mode="RGBA"), palette_rgb


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
                    centers = [c for idx, c in enumerate(centers) if idx not in (i, j)]
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
    alpha_threshold: int = 8,
) -> Image.Image:
    """
    Replace tiny opaque color regions with their dominant opaque neighbor color.

    Transparent pixels stay transparent and are ignored during cleanup.
    """
    from scipy import ndimage

    say = progress or (lambda _m: None)

    rgba_img = image.convert("RGBA")
    arr = np.asarray(rgba_img).copy()
    h, w = arr.shape[:2]

    rgb = arr[..., :3]
    alpha = arr[..., 3]
    opaque_mask = alpha >= alpha_threshold

    if not opaque_mask.any():
        return rgba_img

    encoded = (
        (rgb[..., 0].astype(np.int64) << 16)
        | (rgb[..., 1].astype(np.int64) << 8)
        | rgb[..., 2].astype(np.int64)
    )

    transparent_sentinel = -1
    encoded[~opaque_mask] = transparent_sentinel

    total_replaced = 0
    unique_colors = np.unique(encoded[opaque_mask])
    say(f"  scanning {len(unique_colors)} opaque color regions…")

    for ci, color in enumerate(unique_colors):
        mask = encoded == color
        labeled, n_regions = ndimage.label(mask)
        if n_regions == 0:
            continue

        sizes = ndimage.sum(mask, labeled, range(1, n_regions + 1))
        small_ids = np.where(sizes < min_region_px)[0] + 1
        if len(small_ids) == 0:
            continue

        objects = ndimage.find_objects(labeled)

        for rid in small_ids:
            bbox = objects[rid - 1]
            if bbox is None:
                continue

            y0 = max(0, bbox[0].start - 1)
            y1 = min(h, bbox[0].stop + 1)
            x0 = max(0, bbox[1].start - 1)
            x1 = min(w, bbox[1].stop + 1)

            local_labeled = labeled[y0:y1, x0:x1]
            local_encoded = encoded[y0:y1, x0:x1]
            local_mask = local_labeled == rid

            dilated = ndimage.binary_dilation(local_mask) & ~local_mask
            neighbor_mask = dilated & (local_encoded != transparent_sentinel)
            if not neighbor_mask.any():
                continue

            border_colors = local_encoded[neighbor_mask]
            vals, counts = np.unique(border_colors, return_counts=True)
            replacement = vals[counts.argmax()]

            local_encoded_view = encoded[y0:y1, x0:x1]
            local_encoded_view[local_mask] = replacement
            total_replaced += 1

        say(f"  color {ci + 1}/{len(unique_colors)}: cleaned {len(small_ids)} regions")

    output_rgb = np.stack([
        ((encoded >> 16) & 0xFF).astype(np.uint8),
        ((encoded >> 8) & 0xFF).astype(np.uint8),
        (encoded & 0xFF).astype(np.uint8),
    ], axis=-1)

    out = np.zeros((h, w, 4), dtype=np.uint8)
    out[..., :3] = output_rgb
    out[..., 3] = 0
    out[opaque_mask, 3] = 255
    out[~opaque_mask, :3] = 0
    out[~opaque_mask, 3] = 0

    say(f"  total: replaced {total_replaced} speck regions")
    return Image.fromarray(out, mode="RGBA")
