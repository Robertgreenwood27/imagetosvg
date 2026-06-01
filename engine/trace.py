"""
Raster -> SVG tracing via VTracer (Rust-backed, fast, color-aware).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile

from PIL import Image
import vtracer


@dataclass
class TraceConfig:
    """
    Knobs for VTracer. Defaults are tuned for already-quantized images:
    aggressive merging (since colors are already discrete) and modest detail.

    color_precision: 1-8. Bits per color channel for vtracer's internal
        quantization. Since we pre-quantized, set high (8) — we don't want
        vtracer doing another color reduction on top of ours.
    layer_difference: 0-256. How aggressively to merge similar adjacent
        layers. Higher = fewer layers in output.
    corner_threshold: 0-180 degrees. Sharper angles than this become corners.
    length_threshold: min path segment length in pixels.
    splice_threshold: 0-180 degrees. Curve smoothness.
    filter_speckle: ignore connected regions smaller than this many pixels.
        (Belt-and-braces with cleanup_specks; either or both is fine.)
    path_precision: decimal places in output coordinates.
    """
    color_precision: int = 8
    layer_difference: int = 0
    corner_threshold: int = 60
    length_threshold: float = 4.0
    splice_threshold: int = 45
    filter_speckle: int = 4
    path_precision: int = 2
    mode: str = "spline"  # "spline" | "polygon" | "none"


def trace(image: Image.Image, config: TraceConfig | None = None) -> str:
    """
    Trace a (preferably already-quantized) PIL image into SVG.

    Returns SVG content as a string.
    """
    cfg = config or TraceConfig()

    # VTracer's Python API works on file paths, so we round-trip through temp
    # files. The image write is cheap compared to the trace itself.
    with tempfile.TemporaryDirectory() as td:
        in_path = Path(td) / "in.png"
        out_path = Path(td) / "out.svg"
        image.convert("RGBA").save(in_path)

        vtracer.convert_image_to_svg_py(
            str(in_path),
            str(out_path),
            colormode="color",
            hierarchical="stacked",
            mode=cfg.mode,
            filter_speckle=cfg.filter_speckle,
            color_precision=cfg.color_precision,
            layer_difference=cfg.layer_difference,
            corner_threshold=cfg.corner_threshold,
            length_threshold=cfg.length_threshold,
            splice_threshold=cfg.splice_threshold,
            path_precision=cfg.path_precision,
        )

        return out_path.read_text(encoding="utf-8")
