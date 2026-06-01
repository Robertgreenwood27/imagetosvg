"""
CLI entry point. Writes:
  out/quantized.png       - the posterized preview
  out/raw.svg             - vtracer output, no smoothing
  out/pass1.svg ...       - smoothing passes 1..N
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


def main() -> int:
    ap = argparse.ArgumentParser(description="Photo -> quantized -> SVG -> smoothing passes")
    ap.add_argument("image", type=Path, help="Input photo (jpg/png/...)")
    ap.add_argument("-o", "--out", type=Path, default=Path("out"), help="Output directory")
    ap.add_argument("-k", "--colors", type=int, default=6, help="Palette size (2-32)")
    ap.add_argument("--merge", type=float, default=0.02,
                    help="OKLab dE threshold for merging near-duplicate colors (0 disables)")
    ap.add_argument("--speck", type=int, default=8,
                    help="Min region size in pixels (0 disables; set 0 if cleanup is slow)")
    ap.add_argument("--max-dim", type=int, default=1500,
                    help="Downscale so max(w,h) <= this many pixels (0 disables). "
                         "Big photos give no extra SVG fidelity but cost a lot of time.")
    ap.add_argument("--passes", type=int, default=5, help="Number of smoothing passes to emit")
    ap.add_argument("--rdp", type=float, default=0.5, help="RDP tolerance per pass (px)")
    ap.add_argument("--corner-deg", type=float, default=60.0,
                    help="Corner angle threshold (degrees)")
    ap.add_argument("--fit-error", type=float, default=0.8,
                    help="Max Bezier fit error before subdivision (px)")
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
        trace=TraceConfig(),
        smooth=SmoothConfig(
            rdp_tolerance_px=args.rdp,
            corner_angle_deg=args.corner_deg,
            fit_error_px=args.fit_error,
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
