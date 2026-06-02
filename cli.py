"""
CLI entry point. Writes:
  out/quantized.png       - the posterized preview
  out/raw.svg             - vtracer output, no smoothing
  out/pass1.svg ...       - smoothing passes 1..N
  out/final.svg           - last pass after primitive dispatch simplification
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

from PIL import Image

from engine.pipeline import run_from_file, PipelineConfig
from engine.trace import TraceConfig
from engine.smooth import SmoothConfig
from engine.simplify import SimplifyConfig


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Photo -> quantized -> SVG -> smoothing -> primitive dispatch"
    )
    ap.add_argument("image", type=Path, help="Input photo (jpg/png/...)")
    ap.add_argument("-o", "--out", type=Path, default=Path("out"), help="Output directory")
    ap.add_argument("-k", "--colors", type=int, default=6, help="Palette size (2-32)")
    ap.add_argument("--merge", type=float, default=0.02,
                    help="OKLab dE threshold for merging near-duplicate colors (0 disables)")
    ap.add_argument("--speck", type=int, default=8,
                    help="Min region size in pixels (0 disables)")
    ap.add_argument("--max-dim", type=int, default=1500,
                    help="Downscale so max(w,h) <= this (0 disables)")
    ap.add_argument("--passes", type=int, default=5, help="Number of smoothing passes")
    ap.add_argument("--rdp", type=float, default=0.5, help="RDP tolerance per pass (px)")
    ap.add_argument("--corner-deg", type=float, default=60.0,
                    help="Corner angle threshold (degrees)")
    ap.add_argument("--fit-error", type=float, default=0.8,
                    help="Max Bezier fit error before subdivision (px)")

    # Simplify flags
    ap.add_argument("--no-simplify", action="store_true",
                    help="Skip primitive dispatch simplification")
    ap.add_argument("--simplify-error", type=float, default=1.0,
                    help="Max deviation for primitive substitution (px). "
                         "Higher = more aggressive. Try 0.5-2.0.")
    ap.add_argument("--no-circles", action="store_true", help="Don't try circle fit")
    ap.add_argument("--no-ellipses", action="store_true", help="Don't try ellipse fit")
    ap.add_argument("--no-rects", action="store_true", help="Don't try rect fit")
    ap.add_argument("--no-polygons", action="store_true", help="Don't try polygon fit")

    args = ap.parse_args()

    if not args.image.exists():
        print(f"Image not found: {args.image}", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)

    cfg = PipelineConfig(
        k_colors=args.colors,
        merge_threshold=args.merge,
        speck_min_px=args.speck,
        max_dim_px=args.max_dim,
        smoothing_passes=args.passes,
        run_simplify=not args.no_simplify,
        trace=TraceConfig(),
        smooth=SmoothConfig(
            rdp_tolerance_px=args.rdp,
            corner_angle_deg=args.corner_deg,
            fit_error_px=args.fit_error,
        ),
        simplify=SimplifyConfig(
            error_px=args.simplify_error,
            try_circle=not args.no_circles,
            try_ellipse=not args.no_ellipses,
            try_rect=not args.no_rects,
            try_polygon=not args.no_polygons,
        ),
    )

    t_total = time.perf_counter()
    result = run_from_file(args.image, cfg, progress=lambda m: print(m, flush=True))
    total_dt = time.perf_counter() - t_total

    qpath = args.out / "quantized.png"
    result.quantized.save(qpath)
    print(f"Wrote {qpath}")

    raw_path = args.out / "raw.svg"
    raw_path.write_text(result.raw_svg, encoding="utf-8")
    print(f"Wrote {raw_path}  ({len(result.raw_svg) // 1024} KB)")

    for i, svg in enumerate(result.smoothed_svgs, start=1):
        p = args.out / f"pass{i}.svg"
        p.write_text(svg, encoding="utf-8")
        print(f"Wrote {p}  ({len(svg) // 1024} KB)")

    if result.final_svg is not result.smoothed_svgs[-1]:
        final_path = args.out / "final.svg"
        final_path.write_text(result.final_svg, encoding="utf-8")
        print(f"Wrote {final_path}  ({len(result.final_svg) // 1024} KB)  <- submit this one")

    print("\nPalette:")
    for r, g, b in result.palette_rgb:
        print(f"  #{r:02x}{g:02x}{b:02x}  rgb({r}, {g}, {b})")

    print(f"\nTotal time: {total_dt:.2f}s")
    print("Stage breakdown:")
    for name, dt in result.timings.items():
        print(f"  {name:18s} {dt:6.2f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())