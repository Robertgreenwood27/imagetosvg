# Photo → SVG engine

Local pipeline that converts photos to stylized, multi-color SVGs suitable for
print-on-demand (Printify, etc.). Built around an **engine / UI separation** so
the same core runs behind a CLI today, a PyQt window tomorrow, and a web API
later — without rewriting anything.

## Pipeline

```
photo.jpg
  → quantize  (OKLab k-means, perceptually correct color reduction)
  → cleanup   (drop micro-regions that print muddy)
  → trace     (VTracer, Rust-backed, color-aware)
  → smooth × N (corner-preserving RDP + Schneider Bézier refit, idempotent-ish)
  → final.svg
```

## Project layout

```
engine/
  quantize.py   # OKLab k-means + speck cleanup
  trace.py      # vtracer wrapper
  smooth.py     # corner-preserving smoother (one pass)
  pipeline.py   # orchestrator
ui/
  desktop.py    # PyQt6 desktop UI
cli.py          # command-line entry point
```

`engine/` has no GUI awareness. The desktop UI and a future FastAPI web shell
both import from it without modification.

## Install

```bash
pip install -r requirements.txt
```

For the desktop UI, also:

```bash
pip install PyQt6
```

## Desktop UI

```bash
python -m ui.desktop
# or
python ui\desktop.py
```

**Flow:**
1. **Open Image** (Ctrl+O) — loads the file. Original shows on the left.
2. **Drag the Colors slider** — the middle pane updates live with a quantized
   preview (on a downscaled copy for speed). Find a color count that captures
   what you want.
3. **Click Trace & Smooth** — runs the full pipeline on a background thread.
   Status bar shows progress. SVG appears in the right pane when done.
4. **Click pass buttons** (Raw / 1 / 2 / 3 / 4 / 5) or use **← / →** to flip
   between smoothing levels. Pick whichever looks best.
5. **+ Smooth More** — adds another smoothing pass on top of the latest one,
   if even pass 5 isn't smooth enough.
6. **Save Current SVG** (Ctrl+S) — writes the currently-displayed pass to disk.

## CLI

```bash
python cli.py photo.jpg -k 6 --passes 5
# Output:
#   out/quantized.png   - posterized preview
#   out/raw.svg         - vtracer output, no smoothing
#   out/pass1.svg .. pass5.svg
```

### Flags worth knowing

| Flag | What it does | Default |
|---|---|---|
| `-k N` | palette size (2–32). For Printify, 5–10 is the sweet spot. | 6 |
| `--passes N` | how many smoothing passes to emit | 5 |
| `--rdp F` | RDP tolerance per pass, pixels. Higher = more aggressive. | 0.5 |
| `--corner-deg F` | angle threshold for corner preservation | 60 |
| `--merge F` | merge OKLab ΔE near-duplicate palette colors | 0.02 |
| `--speck N` | drop regions smaller than N pixels before tracing | 8 |
| `--fit-error F` | max Bézier deviation before subdividing | 0.8 |

### Iteration pattern

The smoother is **gentle and convergent**: running it 5 times with rdp=0.5
typically produces a different result than once with rdp=2.5. Each pass:

1. Resamples paths into polylines
2. Detects corners (turn > `--corner-deg`) — protected from simplification
3. Runs RDP between corners with `--rdp` tolerance
4. Refits each smooth segment with cubic Béziers

Open multiple `passN.svg` files side by side, pick whichever looks right.
On the synthetic test image, complexity drops from 135 cubics → 56 cubics in
pass 1 and then stabilizes at 54–56 through pass 5 (corner preservation keeps
real structure intact while noise is gone).

## Programmatic use

```python
from engine.pipeline import run_from_file, PipelineConfig
from engine.smooth import SmoothConfig

result = run_from_file(
    "photo.jpg",
    PipelineConfig(
        k_colors=8,
        smoothing_passes=5,
        smooth=SmoothConfig(rdp_tolerance_px=0.6, corner_angle_deg=55),
    ),
)
result.quantized.save("preview.png")
for i, svg in enumerate(result.smoothed_svgs, 1):
    open(f"pass{i}.svg", "w").write(svg)
```

## What's next

1. **PyQt GUI** wrapping the same engine — slider for k, live posterize
   preview, "smooth again" button with undo, QSvgWidget for native render.
2. **AI judge loop** (Anthropic API): send a grid of smoothed passes,
   ask "which pass is best for a printed garment?", auto-select the winner.
   Optional enhancement — engine works fully offline without it.
3. **Web/SaaS shell**: FastAPI in front of the same `engine/` package.

## Known limitations

- Photo-to-SVG is inherently lossy. The output is **stylized** — paint-by-
  numbers / poster look. Photo realism in vector form is mathematically not
  a thing; if you want a faithful photo, stay raster.
- Very fine detail (eyes, hair strands) will be smoothed away. That's the
  cost of clean cuttable/printable regions.
- VTracer occasionally over-segments smooth color regions. The smoothing
  passes mostly heal this, but very noisy source photos benefit from
  pre-blurring before the pipeline (`Image.filter(GaussianBlur(radius=1))`).
