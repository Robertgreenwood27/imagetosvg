"""
Top-level pipeline: photo -> quantized image -> SVG -> N smoothing passes.

This is the engine API. Callers (CLI today, GUI tomorrow, web later) use only
this and don't reach into the submodules unless they have to.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from PIL import Image

from .quantize import quantize, cleanup_specks
from .trace import trace, TraceConfig
from .smooth import smooth_pass, SmoothConfig


@dataclass
class PipelineConfig:
    k_colors: int = 6
    merge_threshold: float = 0.02          # OKLab ΔE; 0 to disable
    speck_min_px: int = 8                  # 0 to disable
    max_dim_px: int = 0                    # 0 = no downscaling; else max(w,h) cap
    trace: TraceConfig = field(default_factory=TraceConfig)
    smooth: SmoothConfig = field(default_factory=SmoothConfig)
    smoothing_passes: int = 3


@dataclass
class PipelineResult:
    quantized: Image.Image
    palette_rgb: list
    raw_svg: str
    smoothed_svgs: list[str]
    timings: dict


def run(
    image: Image.Image,
    config: PipelineConfig | None = None,
    progress: Callable[[str], None] | None = None,
) -> PipelineResult:
    """
    Run the full pipeline. `progress` is an optional callback for status text.
    Each stage prints "Stage..." then "  done in Xs" so the caller sees forward
    motion even on long-running stages.
    """
    cfg = config or PipelineConfig()
    say = progress or (lambda _msg: None)
    timings: dict[str, float] = {}

    # Image info upfront so the user knows what we're working with.
    w, h = image.size
    mp = (w * h) / 1_000_000
    say(f"Input: {w}x{h} pixels ({mp:.1f} MP, mode={image.mode})")

    # Optional downscale. Huge photos make every stage slow without giving
    # better SVG output because the trace can't represent that detail anyway.
    if cfg.max_dim_px > 0 and max(w, h) > cfg.max_dim_px:
        scale = cfg.max_dim_px / max(w, h)
        new_size = (int(w * scale), int(h * scale))
        say(f"Downscaling to {new_size[0]}x{new_size[1]} (max_dim={cfg.max_dim_px})")
        image = image.resize(new_size, Image.LANCZOS)

    # Quantize
    t0 = time.perf_counter()
    say(f"Quantizing to {cfg.k_colors} colors (OKLab k-means)...")
    quantized, palette = quantize(
        image,
        k=cfg.k_colors,
        merge_threshold=cfg.merge_threshold,
    )
    timings["quantize"] = time.perf_counter() - t0
    say(f"  done in {timings['quantize']:.2f}s ({len(palette)} colors after merge)")

    # Speck cleanup
    if cfg.speck_min_px > 0:
        t0 = time.perf_counter()
        say(f"Removing regions smaller than {cfg.speck_min_px} px...")
        quantized = cleanup_specks(
            quantized,
            min_region_px=cfg.speck_min_px,
            progress=say,
        )
        timings["cleanup"] = time.perf_counter() - t0
        say(f"  done in {timings['cleanup']:.2f}s")

    # Trace
    t0 = time.perf_counter()
    say("Tracing to SVG (vtracer)...")
    raw_svg = trace(quantized, cfg.trace)
    timings["trace"] = time.perf_counter() - t0
    say(f"  done in {timings['trace']:.2f}s ({len(raw_svg) // 1024} KB)")

    # Smoothing passes
    smoothed = []
    current = raw_svg
    for i in range(cfg.smoothing_passes):
        t0 = time.perf_counter()
        say(f"Smoothing pass {i + 1}/{cfg.smoothing_passes}...")
        current = smooth_pass(current, cfg.smooth)
        dt = time.perf_counter() - t0
        timings[f"smooth_pass_{i + 1}"] = dt
        say(f"  done in {dt:.2f}s ({len(current) // 1024} KB)")
        smoothed.append(current)

    return PipelineResult(
        quantized=quantized,
        palette_rgb=[tuple(int(c) for c in row) for row in palette],
        raw_svg=raw_svg,
        smoothed_svgs=smoothed,
        timings=timings,
    )


def run_from_file(
    image_path: str | Path,
    config: PipelineConfig | None = None,
    progress: Callable[[str], None] | None = None,
) -> PipelineResult:
    """Convenience wrapper for file input."""
    return run(Image.open(image_path), config=config, progress=progress)
