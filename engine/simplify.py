"""
Primitive dispatch: replace <path> elements with cheaper SVG primitives
where the geometry fits within a perceptual error budget.

Dispatch order (cheapest first):
  circle       -> 3 numbers  (cx, cy, r)
  ellipse      -> 4-5 numbers (cx, cy, rx, ry [+ rotate transform])
  rect         -> 4 numbers  (x, y, w, h)
  rotated rect -> 4 numbers  + transform="rotate(...)"
  line-path    -> M/L/Z only — same <path> tag, cheaper grammar
  polygon      -> N*2 numbers, no curve math
  path         -> unchanged

Four correctness gates before any substitution is accepted:
  1. Topology check   — skip if path has holes or multiple subpaths
  2. Boundary error   — max(min_dist(original_pt, candidate)) <= error_px
  3. Area error       — |area_original - area_candidate| / area_original <= area_tol
  4. Byte gate        — serialised candidate must be strictly shorter than original d=

Usage:
    from engine.simplify import simplify_pass, SimplifyConfig
    svg_out = simplify_pass(svg_in)
    svg_out = simplify_pass(svg_in, SimplifyConfig(error_px=1.5))
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np
from svgpathtools import parse_path

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)


@dataclass
class SimplifyConfig:
    """
    error_px:     Max boundary deviation for a substitution to be accepted.
                  1.0 is safe; 2.0 is aggressive.
    area_tol:     Max fractional area difference (0.05 = 5%). Catches fits
                  that match the outline but cover the wrong filled area.
    min_area_px2: Skip regions smaller than this (already cheap).
    min_byte_saving: Only accept a substitution if it saves at least this
                  many bytes in the serialised attribute string.
    try_circle:         attempt circle fit
    try_ellipse:        attempt ellipse fit (axis-aligned + rotated)
    try_rect:           attempt axis-aligned rect fit
    try_rotated_rect:   attempt rotated rect fit
    try_line_path:      downgrade cubics to M/L/Z where curve is negligible
    try_polygon:        attempt polygon fit
    """
    error_px: float = 1.0
    area_tol: float = 0.05
    min_area_px2: float = 20.0
    min_byte_saving: int = 8
    try_circle: bool = True
    try_ellipse: bool = True
    try_rect: bool = True
    try_rotated_rect: bool = True
    try_line_path: bool = True
    try_polygon: bool = True


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def simplify_pass(svg_text: str, config: SimplifyConfig | None = None) -> str:
    cfg = config or SimplifyConfig()
    root = ET.fromstring(svg_text)
    path_tag = f"{{{SVG_NS}}}path"

    stats = {
        "circle": 0, "ellipse": 0, "rect": 0, "rotated_rect": 0,
        "line_path": 0, "polygon": 0, "path": 0, "skipped_topology": 0,
    }

    for elem in root.iter(path_tag):
        d = elem.get("d", "")
        fill = elem.get("fill", "none")
        if not d or fill == "none":
            continue

        # --- Topology gate: skip compound paths (holes, multiple subpaths) ---
        if _has_multiple_subpaths(d):
            stats["skipped_topology"] += 1
            continue

        pts = _sample_path_dense(d)
        if pts is None or len(pts) < 6:
            continue

        area_orig = _polygon_area(pts)
        if area_orig < cfg.min_area_px2:
            continue

        original_d_bytes = len(d.encode())

        def _accept(candidate_tag: str, candidate_attribs: dict, candidate_pts: np.ndarray) -> bool:
            """All four gates in one place."""
            # Gate 1: boundary deviation
            dev = _max_deviation(pts, candidate_pts)
            if dev > cfg.error_px:
                return False
            # Gate 2: area
            area_cand = _polygon_area(candidate_pts)
            if area_orig > 0:
                frac = abs(area_orig - area_cand) / area_orig
                if frac > cfg.area_tol:
                    return False
            # Gate 3: byte saving
            serialised = _serialise_attribs(candidate_tag, candidate_attribs)
            if original_d_bytes - len(serialised.encode()) < cfg.min_byte_saving:
                return False
            return True

        replacement = None

        if cfg.try_circle:
            r = _try_circle(pts, cfg.error_px, area_orig, cfg.area_tol,
                            original_d_bytes, cfg.min_byte_saving)
            if r:
                replacement = r

        if replacement is None and cfg.try_ellipse:
            r = _try_ellipse(pts, cfg.error_px, area_orig, cfg.area_tol,
                             original_d_bytes, cfg.min_byte_saving)
            if r:
                replacement = r

        if replacement is None and cfg.try_rect:
            r = _try_rect(pts, cfg.error_px, area_orig, cfg.area_tol,
                          original_d_bytes, cfg.min_byte_saving, rotated=False)
            if r:
                replacement = r

        if replacement is None and cfg.try_rotated_rect:
            r = _try_rect(pts, cfg.error_px, area_orig, cfg.area_tol,
                          original_d_bytes, cfg.min_byte_saving, rotated=True)
            if r:
                replacement = r

        if replacement is None and cfg.try_line_path:
            r = _try_line_path(pts, d, cfg.error_px, area_orig, cfg.area_tol,
                               original_d_bytes, cfg.min_byte_saving)
            if r:
                replacement = r

        if replacement is None and cfg.try_polygon:
            r = _try_polygon(pts, cfg.error_px, area_orig, cfg.area_tol,
                             original_d_bytes, cfg.min_byte_saving)
            if r:
                replacement = r

        if replacement is not None:
            kind, attribs = replacement
            _replace_elem(elem, kind, attribs)
            stats[kind.replace("-", "_")] += 1
        else:
            stats["path"] += 1

    comment_text = (
        f" simplify_pass: "
        + " ".join(f"{k}={v}" for k, v in stats.items())
        + " "
    )
    root.insert(0, ET.Comment(comment_text))
    return ET.tostring(root, encoding="unicode")


# ---------------------------------------------------------------------------
# Topology check
# ---------------------------------------------------------------------------

def _has_multiple_subpaths(d: str) -> bool:
    """
    Return True if the path has more than one subpath (M command appears
    more than once after stripping leading whitespace).
    Multiple subpaths = holes, disconnected islands — cannot be represented
    by simple primitives.
    """
    # Count 'M' or 'm' tokens that start a new subpath.
    # A Z/z followed by M starts a new subpath; M at the start is the first.
    import re
    moves = re.findall(r'(?<![eE])[Mm]', d)
    return len(moves) > 1


# ---------------------------------------------------------------------------
# Dense path sampling
# ---------------------------------------------------------------------------

def _sample_path_dense(d: str, step: float = 0.5) -> np.ndarray | None:
    try:
        path = parse_path(d)
    except Exception:
        return None

    pts: list[tuple[float, float]] = []
    for seg in path:
        try:
            seg_len = seg.length(error=1e-3)
        except Exception:
            continue
        n = max(2, int(math.ceil(seg_len / step)))
        for t in np.linspace(0.0, 1.0, n):
            p = seg.point(t)
            pts.append((p.real, p.imag))

    if not pts:
        return None

    arr = np.array(pts, dtype=np.float64)
    mask = np.ones(len(arr), dtype=bool)
    mask[1:] = np.any(np.diff(arr, axis=0) != 0, axis=1)
    arr = arr[mask]
    return arr if len(arr) >= 4 else None


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _polygon_area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _max_deviation(pts: np.ndarray, pred_pts: np.ndarray) -> float:
    if len(pred_pts) == 0:
        return 1e9
    diff = pts[:, None, :] - pred_pts[None, :, :]
    d2 = (diff ** 2).sum(axis=2)
    return float(np.sqrt(d2.min(axis=1).max()))


def _area_ok(area_orig: float, candidate_pts: np.ndarray, area_tol: float) -> bool:
    if area_orig <= 0:
        return True
    area_cand = _polygon_area(candidate_pts)
    return abs(area_orig - area_cand) / area_orig <= area_tol


def _bytes_saved(original_d_bytes: int, tag: str, attribs: dict,
                 min_saving: int) -> bool:
    serialised = _serialise_attribs(tag, attribs)
    return (original_d_bytes - len(serialised.encode())) >= min_saving


def _serialise_attribs(tag: str, attribs: dict) -> str:
    """Rough serialisation of geometry attributes for byte-count comparison."""
    parts = [f'{k}="{_fmt(v)}"' for k, v in attribs.items()]
    return f"<{tag} " + " ".join(parts) + "/>"


def _sample_circle(cx, cy, r, n=180) -> np.ndarray:
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t)])


def _sample_ellipse(cx, cy, rx, ry, angle_deg=0.0, n=180) -> np.ndarray:
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    x = rx * np.cos(t)
    y = ry * np.sin(t)
    if angle_deg != 0.0:
        a = math.radians(angle_deg)
        ca, sa = math.cos(a), math.sin(a)
        x, y = ca * x - sa * y, sa * x + ca * y
    return np.column_stack([cx + x, cy + y])


def _sample_rect_pts(cx, cy, w, h, angle_deg=0.0, n=200) -> np.ndarray:
    """Sample perimeter of a (possibly rotated) rectangle, centred at cx,cy."""
    hw, hh = w / 2, h / 2
    corners = np.array([
        [-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh], [-hw, -hh]
    ])
    # Distribute n points proportionally along edges
    edge_lens = [w, h, w, h]
    total = 2 * (w + h)
    pts = []
    for i in range(4):
        a, b = corners[i], corners[i + 1]
        k = max(2, int(round(n * edge_lens[i] / total)))
        for t in np.linspace(0, 1, k, endpoint=False):
            pts.append(a + t * (b - a))
    pts = np.array(pts, dtype=np.float64)
    if angle_deg != 0.0:
        a = math.radians(angle_deg)
        ca, sa = math.cos(a), math.sin(a)
        rot = np.array([[ca, -sa], [sa, ca]])
        pts = pts @ rot.T
    pts[:, 0] += cx
    pts[:, 1] += cy
    return pts


# ---------------------------------------------------------------------------
# Primitive fitters  (all return (tag, attribs) or None)
# ---------------------------------------------------------------------------

def _try_circle(pts, error_px, area_orig, area_tol, orig_bytes, min_saving):
    A = np.column_stack([2 * pts[:, 0], 2 * pts[:, 1], np.ones(len(pts))])
    b = pts[:, 0] ** 2 + pts[:, 1] ** 2
    try:
        result, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy = result[0], result[1]
    r2 = result[2] + cx ** 2 + cy ** 2
    if r2 <= 0:
        return None
    r = math.sqrt(r2)

    sample = _sample_circle(cx, cy, r)
    if _max_deviation(pts, sample) > error_px:
        return None
    if not _area_ok(area_orig, sample, area_tol):
        return None

    attribs = {"cx": cx, "cy": cy, "r": r}
    if not _bytes_saved(orig_bytes, "circle", attribs, min_saving):
        return None
    return ("circle", attribs)


def _try_ellipse(pts, error_px, area_orig, area_tol, orig_bytes, min_saving):
    cx, cy = pts.mean(axis=0)
    x = pts[:, 0] - cx
    y = pts[:, 1] - cy

    mu20 = float(np.mean(x ** 2))
    mu02 = float(np.mean(y ** 2))
    mu11 = float(np.mean(x * y))

    trace_val = mu20 + mu02
    det = max(0.0, mu20 * mu02 - mu11 ** 2)
    disc = math.sqrt(max(0, (trace_val / 2) ** 2 - det))
    lam1 = trace_val / 2 + disc
    lam2 = trace_val / 2 - disc

    if lam1 <= 0 or lam2 <= 0:
        return None

    rx = math.sqrt(2 * lam1)
    ry = math.sqrt(2 * lam2)

    # Skip near-circles (circle is cheaper)
    if abs(rx - ry) < error_px * 0.5:
        return None

    angle_deg = (0.0 if (abs(mu11) < 1e-9 and abs(mu20 - mu02) < 1e-9)
                 else math.degrees(0.5 * math.atan2(2 * mu11, mu20 - mu02)))

    best_dev = 1e9
    best_angle = 0.0
    for angle in [0.0, angle_deg]:
        sample = _sample_ellipse(cx, cy, rx, ry, angle)
        dev = _max_deviation(pts, sample)
        if dev < best_dev:
            best_dev = dev
            best_angle = angle

    if best_dev > error_px:
        return None

    sample_best = _sample_ellipse(cx, cy, rx, ry, best_angle)
    if not _area_ok(area_orig, sample_best, area_tol):
        return None

    attribs: dict = {"cx": cx, "cy": cy, "rx": rx, "ry": ry}
    if abs(best_angle) > 0.5:
        attribs["transform"] = f"rotate({best_angle:.2f},{cx:.2f},{cy:.2f})"

    if not _bytes_saved(orig_bytes, "ellipse", attribs, min_saving):
        return None
    return ("ellipse", attribs)


def _try_rect(pts, error_px, area_orig, area_tol, orig_bytes, min_saving,
              rotated: bool = False):
    """
    Axis-aligned rect (rotated=False) or minimum-area rotated rect (rotated=True).
    For the rotated case we find the principal axis via PCA and try both the
    PCA angle and 90° from it.
    """
    if not rotated:
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)
        w = x_max - x_min
        h = y_max - y_min
        if w < 1 or h < 1:
            return None
        cx = x_min + w / 2
        cy = y_min + h / 2
        sample = _sample_rect_pts(cx, cy, w, h, 0.0)
        if _max_deviation(pts, sample) > error_px:
            return None
        if not _area_ok(area_orig, sample, area_tol):
            return None
        attribs = {"x": x_min, "y": y_min, "width": w, "height": h}
        if not _bytes_saved(orig_bytes, "rect", attribs, min_saving):
            return None
        return ("rect", attribs)

    # Rotated rect: PCA to find orientation
    cx, cy = pts.mean(axis=0)
    centered = pts - np.array([cx, cy])
    cov = np.cov(centered.T)
    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return None

    # Principal axis angle
    angle_rad = math.atan2(float(eigvecs[1, -1]), float(eigvecs[0, -1]))
    angle_deg = math.degrees(angle_rad)

    # Skip near-zero rotation (already handled by axis-aligned rect)
    if abs(angle_deg % 90) < 2.0:
        return None

    # Rotate points to axis-aligned frame, get bounding box, rotate back
    ca, sa = math.cos(-angle_rad), math.sin(-angle_rad)
    rot = np.array([[ca, -sa], [sa, ca]])
    rotated_pts = centered @ rot.T
    x_min, y_min = rotated_pts.min(axis=0)
    x_max, y_max = rotated_pts.max(axis=0)
    w = x_max - x_min
    h = y_max - y_min

    sample = _sample_rect_pts(cx, cy, w, h, angle_deg)
    if _max_deviation(pts, sample) > error_px:
        return None
    if not _area_ok(area_orig, sample, area_tol):
        return None

    attribs = {
        "x": cx - w / 2,
        "y": cy - h / 2,
        "width": w,
        "height": h,
        "transform": f"rotate({angle_deg:.2f},{cx:.2f},{cy:.2f})",
    }
    if not _bytes_saved(orig_bytes, "rect", attribs, min_saving):
        return None
    return ("rect", attribs)


def _try_line_path(pts, original_d, error_px, area_orig, area_tol,
                   orig_bytes, min_saving):
    """
    Downgrade a cubic Bézier path to a straight-line path (M/L/Z) using RDP.
    Keeps the <path> tag but replaces the d attribute with cheaper grammar.
    Returns ("path", {"d": new_d}) or None.
    """
    simplified = _rdp_simple(pts, error_px)
    if len(simplified) < 3:
        return None

    # Build M/L/Z path string
    new_d = _pts_to_line_path(simplified, closed=True)

    if len(new_d.encode()) >= orig_bytes - min_saving:
        return None

    # Area check on the simplified polygon
    if not _area_ok(area_orig, simplified, area_tol):
        return None

    return ("path", {"d": new_d})


def _try_polygon(pts, error_px, area_orig, area_tol, orig_bytes, min_saving):
    simplified = _rdp_simple(pts, error_px)
    if len(simplified) < 3:
        return None

    pts_str = " ".join(f"{_fmt(p[0])},{_fmt(p[1])}" for p in simplified)
    attribs = {"points": pts_str}

    if not _area_ok(area_orig, simplified, area_tol):
        return None
    if not _bytes_saved(orig_bytes, "polygon", attribs, min_saving):
        return None
    return ("polygon", attribs)


# ---------------------------------------------------------------------------
# RDP
# ---------------------------------------------------------------------------

def _rdp_simple(pts: np.ndarray, tol: float) -> np.ndarray:
    if len(pts) < 3:
        return pts
    keep = [True] * len(pts)
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = pts[i], pts[j]
        ab = b - a
        ab_len = np.linalg.norm(ab)
        if ab_len < 1e-12:
            dists = np.linalg.norm(pts[i + 1:j] - a, axis=1)
        else:
            rel = pts[i + 1:j] - a
            cross = rel[:, 0] * ab[1] - rel[:, 1] * ab[0]
            dists = np.abs(cross) / ab_len
        k = int(np.argmax(dists))
        if dists[k] > tol:
            stack.append((i, i + 1 + k))
            stack.append((i + 1 + k, j))
        else:
            for idx in range(i + 1, j):
                keep[idx] = False
    return pts[keep]


# ---------------------------------------------------------------------------
# Path / attribute helpers
# ---------------------------------------------------------------------------

def _pts_to_line_path(pts: np.ndarray, closed: bool = True) -> str:
    parts = [f"M{_fmt(pts[0,0])},{_fmt(pts[0,1])}"]
    for p in pts[1:]:
        parts.append(f"L{_fmt(p[0])},{_fmt(p[1])}")
    if closed:
        parts.append("Z")
    return "".join(parts)


def _fmt(v) -> str:
    if isinstance(v, str):
        return v
    s = f"{float(v):.2f}".rstrip("0").rstrip(".")
    return s or "0"


def _replace_elem(elem: ET.Element, kind: str, attribs: dict) -> None:
    """
    Mutate elem in-place: update tag (unless kind=="path") and geometry attrs.
    Preserves all presentation attributes from the original element.
    """
    presentation_keys = {
        "fill", "stroke", "stroke-width", "opacity", "fill-opacity",
        "stroke-opacity", "fill-rule", "clip-path", "mask", "filter",
        "transform", "style",
    }
    preserved = {k: v for k, v in elem.attrib.items() if k in presentation_keys}

    elem.attrib.clear()

    if kind == "path":
        # Stay as <path>, just update d=
        elem.tag = f"{{{SVG_NS}}}path"
    else:
        elem.tag = f"{{{SVG_NS}}}{kind}"

    for k, v in attribs.items():
        if k == "transform" and "transform" in preserved:
            elem.set("transform", f"{preserved.pop('transform')} {v}")
        else:
            elem.set(k, _fmt(v) if not isinstance(v, str) else v)

    for k, v in preserved.items():
        elem.set(k, v)


# ---------------------------------------------------------------------------
# Public alias
# ---------------------------------------------------------------------------

def simplify_svg(svg_text: str, config: SimplifyConfig | None = None) -> str:
    return simplify_pass(svg_text, config)