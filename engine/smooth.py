"""
Iterative, corner-preserving SVG path smoothing.

Each pass:
  1. Sample every path into a dense polyline.
  2. Detect corners (turn angles above a threshold) and mark them protected.
  3. Run Ramer-Douglas-Peucker between corners with a small tolerance.
  4. Refit the resulting polyline as cubic Béziers (Schneider 1990,
     "An Algorithm for Automatically Fitting Digitized Curves").
  5. Reassemble path data.

The pass is intentionally gentle so multiple invocations compose well.
Corners survive every pass; only "noise" between corners gets shaved.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
import xml.etree.ElementTree as ET

import numpy as np
from svgpathtools import parse_path, Path, Line, CubicBezier, QuadraticBezier, Arc


SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)


@dataclass
class SmoothConfig:
    """
    sample_step_px: how densely to sample paths into polylines. Smaller =
        higher fidelity but slower. 1.0 is a good balance.
    corner_angle_deg: turn angle above which a point is treated as a corner
        and protected from smoothing. 60° is conservative; lower = more
        aggressive corner preservation.
    rdp_tolerance_px: per-pass RDP tolerance. Keep small (0.3-0.7) so each
        pass is gentle and iteration is meaningful.
    fit_error_px: max allowed deviation when fitting a Bézier to a segment.
        If exceeded, the segment is split and refit recursively.
    max_subdivisions: cap on recursion depth in the fitter.
    """
    sample_step_px: float = 1.0
    corner_angle_deg: float = 60.0
    rdp_tolerance_px: float = 0.5
    fit_error_px: float = 0.8
    max_subdivisions: int = 8


# -- SVG-level driver --------------------------------------------------------

def smooth_pass(svg_text: str, config: SmoothConfig | None = None) -> str:
    """
    Apply one smoothing pass to every <path> in the SVG.
    """
    cfg = config or SmoothConfig()

    # Parse SVG. Namespace handling: vtracer's output uses the default SVG ns.
    root = ET.fromstring(svg_text)
    path_tag = f"{{{SVG_NS}}}path"

    for elem in root.iter(path_tag):
        d = elem.get("d")
        if not d:
            continue
        new_d = _smooth_path_d(d, cfg)
        if new_d:
            elem.set("d", new_d)

    return ET.tostring(root, encoding="unicode")


def _smooth_path_d(d_attr: str, cfg: SmoothConfig) -> str:
    """Smooth one path's d-attribute, preserving subpath structure."""
    try:
        path = parse_path(d_attr)
    except Exception:
        return d_attr  # leave malformed paths alone

    # svgpathtools groups segments into continuous subpaths for us.
    out_parts: list[str] = []
    for sub in path.continuous_subpaths():
        polyline, closed = _sample_subpath(sub, cfg.sample_step_px)
        if len(polyline) < 3:
            # Trivial; just emit as a line.
            out_parts.append(_polyline_to_d(polyline, closed))
            continue

        corners = _detect_corners(polyline, cfg.corner_angle_deg, closed)
        smoothed = _smooth_polyline(polyline, corners, cfg.rdp_tolerance_px, closed)
        beziers = _fit_polyline_to_beziers(smoothed, corners, cfg, closed)
        out_parts.append(_beziers_to_d(beziers, closed))

    return " ".join(out_parts)


# -- Polyline sampling -------------------------------------------------------

def _sample_subpath(sub: Path, step_px: float) -> tuple[np.ndarray, bool]:
    """
    Convert a subpath into an (N,2) polyline by sampling each segment.
    Returns (polyline, is_closed).
    """
    pts: list[complex] = []
    for seg in sub:
        seg_len = seg.length(error=1e-3)
        n = max(2, int(math.ceil(seg_len / step_px)))
        ts = np.linspace(0.0, 1.0, n)
        for t in ts:
            p = seg.point(t)
            if not pts or abs(pts[-1] - p) > 1e-9:
                pts.append(p)

    arr = np.array([[p.real, p.imag] for p in pts], dtype=np.float64)
    # Closed if first and last coincide.
    closed = len(arr) >= 2 and np.allclose(arr[0], arr[-1])
    if closed and len(arr) > 1:
        arr = arr[:-1]  # drop duplicate closing point; we'll add closure back
    return arr, closed


# -- Corner detection --------------------------------------------------------

def _detect_corners(polyline: np.ndarray, angle_thresh_deg: float, closed: bool) -> np.ndarray:
    """
    Return a boolean mask of length N marking corners (True = protected).
    Endpoints of an open path are always corners.
    """
    n = len(polyline)
    mask = np.zeros(n, dtype=bool)
    if n < 3:
        mask[:] = True
        return mask

    # Turn angle = arccos(dot(v1_unit, v2_unit)). Higher turn = lower dot.
    # We mark as corner if turn_angle > angle_thresh, i.e. dot < cos(angle_thresh).
    cos_thresh = math.cos(math.radians(angle_thresh_deg))

    for i in range(n):
        if i == 0 or i == n - 1:
            if not closed:
                mask[i] = True
                continue
            prev_i = (i - 1) % n
            next_i = (i + 1) % n
        else:
            prev_i = i - 1
            next_i = i + 1

        v1 = polyline[i] - polyline[prev_i]
        v2 = polyline[next_i] - polyline[i]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-9 or n2 < 1e-9:
            continue
        cos_a = float(np.dot(v1, v2) / (n1 * n2))
        if cos_a < cos_thresh:
            mask[i] = True

    return mask


# -- RDP between corners -----------------------------------------------------

def _smooth_polyline(
    polyline: np.ndarray,
    corners: np.ndarray,
    tol: float,
    closed: bool,
) -> np.ndarray:
    """
    Apply RDP to each run of non-corner points, keeping all corners fixed.
    Returns a reduced polyline + an updated corner mask (handled by caller
    via re-detection if needed; we return just the polyline and assume the
    fitter re-derives corners).
    """
    n = len(polyline)
    corner_idx = list(np.where(corners)[0])
    if not corner_idx:
        # No corners — treat entire polyline as one segment between virtual
        # endpoints.
        if closed:
            # Just RDP the whole thing as a loop by pinning point 0.
            return _rdp(polyline, tol)
        return _rdp(polyline, tol)

    keep: list[np.ndarray] = []
    for k in range(len(corner_idx)):
        a = corner_idx[k]
        b = corner_idx[(k + 1) % len(corner_idx)]
        if a < b:
            segment = polyline[a:b + 1]
            reduced = _rdp(segment, tol)
            # Drop trailing corner; the next iteration adds it as its own a.
            keep.append(reduced[:-1])
        else:
            # Wrap-around segment (only happens in closed paths).
            segment = np.concatenate([polyline[a:], polyline[:b + 1]])
            reduced = _rdp(segment, tol)
            keep.append(reduced[:-1])

    if not closed:
        # Re-append the very last corner that the loop dropped.
        last = corner_idx[-1]
        keep.append(polyline[last:last + 1])

    return np.concatenate(keep, axis=0)


def _rdp(points: np.ndarray, tol: float) -> np.ndarray:
    """Ramer-Douglas-Peucker polyline simplification."""
    if len(points) < 3:
        return points
    keep = np.zeros(len(points), dtype=bool)
    keep[0] = keep[-1] = True
    _rdp_recurse(points, 0, len(points) - 1, tol, keep)
    return points[keep]


def _rdp_recurse(pts: np.ndarray, i: int, j: int, tol: float, keep: np.ndarray) -> None:
    if j <= i + 1:
        return
    a, b = pts[i], pts[j]
    ab = b - a
    ab_len = np.linalg.norm(ab)
    if ab_len < 1e-12:
        # Degenerate; measure to point a
        dists = np.linalg.norm(pts[i + 1:j] - a, axis=1)
    else:
        # Perpendicular distance from each interior point to segment ab
        rel = pts[i + 1:j] - a
        cross = rel[:, 0] * ab[1] - rel[:, 1] * ab[0]
        dists = np.abs(cross) / ab_len
    if len(dists) == 0:
        return
    k = int(np.argmax(dists))
    if dists[k] > tol:
        keep[i + 1 + k] = True
        _rdp_recurse(pts, i, i + 1 + k, tol, keep)
        _rdp_recurse(pts, i + 1 + k, j, tol, keep)


# -- Bézier fitting (Schneider 1990, simplified) -----------------------------

def _fit_polyline_to_beziers(
    polyline: np.ndarray,
    _corners_unused: np.ndarray,
    cfg: SmoothConfig,
    closed: bool,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Fit cubic Béziers to the polyline. Re-detect corners on the simplified
    polyline (rather than reusing the dense-corner mask, which is now
    invalidated by the index shift).
    """
    if len(polyline) < 2:
        return []

    # Re-detect corners on the simplified polyline.
    fresh_corners = _detect_corners(polyline, cfg.corner_angle_deg, closed)
    corner_idx = list(np.where(fresh_corners)[0])

    beziers: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    if not corner_idx:
        # All-smooth loop. Pin at index 0.
        if closed:
            corner_idx = [0]
        else:
            corner_idx = [0, len(polyline) - 1]

    for k in range(len(corner_idx)):
        a = corner_idx[k]
        if k + 1 < len(corner_idx):
            b = corner_idx[k + 1]
            seg = polyline[a:b + 1]
        elif closed:
            seg = np.concatenate([polyline[a:], polyline[:corner_idx[0] + 1]])
        else:
            break  # last corner — no segment after it

        if len(seg) < 2:
            continue
        if len(seg) == 2:
            # Line — represent as a degenerate cubic
            p0, p3 = seg[0], seg[1]
            p1 = p0 + (p3 - p0) / 3.0
            p2 = p0 + 2 * (p3 - p0) / 3.0
            beziers.append((p0, p1, p2, p3))
            continue
        _fit_segment(seg, cfg, beziers, depth=0)

    return beziers


def _fit_segment(seg, cfg: SmoothConfig, out: list, depth: int) -> None:
    """Fit a single smooth polyline segment with one or more cubic Béziers."""
    p0, p3 = seg[0], seg[-1]
    t_hat1 = _unit(seg[1] - seg[0])
    t_hat2 = _unit(seg[-2] - seg[-1])

    # Chord-length parameterization
    diffs = np.diff(seg, axis=0)
    chord = np.linalg.norm(diffs, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(chord)])
    total = cumulative[-1]
    if total < 1e-9:
        return
    u = cumulative / total

    # Schneider's least-squares for alpha1, alpha2:
    # P1 = P0 + alpha1 * t_hat1
    # P2 = P3 + alpha2 * t_hat2
    A1 = 3 * (1 - u) ** 2 * u  # coefficient on P1
    A2 = 3 * (1 - u) * u ** 2  # coefficient on P2

    # Target = points minus the parts contributed by P0, P3
    target = seg - ((1 - u)[:, None] ** 3) * p0 - (u[:, None] ** 3) * p3

    # Each target point t_i = A1_i * (P0 + a1*t_hat1) + A2_i * (P3 + a2*t_hat2)
    # Rearrange:
    rhs = target - A1[:, None] * p0 - A2[:, None] * p3
    # Stack equations (one per point, 2D):
    M = np.column_stack([A1 * t_hat1[0], A2 * t_hat2[0],
                         A1 * t_hat1[1], A2 * t_hat2[1]])
    # Actually we want a 2-column matrix where col0 is alpha1, col1 is alpha2.
    # The system is overdetermined; build it explicitly:
    a_rows = []
    b_rows = []
    for i in range(len(seg)):
        a_rows.append([A1[i] * t_hat1[0], A2[i] * t_hat2[0]])
        b_rows.append(rhs[i, 0])
        a_rows.append([A1[i] * t_hat1[1], A2[i] * t_hat2[1]])
        b_rows.append(rhs[i, 1])
    A = np.array(a_rows)
    b = np.array(b_rows)
    try:
        alphas, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        alphas = np.array([total / 3, total / 3])

    a1, a2 = alphas
    # Guard against negative or absurd magnitudes
    if a1 < 1e-6 or a2 < 1e-6 or a1 > total * 1.5 or a2 > total * 1.5:
        a1 = a2 = total / 3.0

    p1 = p0 + a1 * t_hat1
    p2 = p3 + a2 * t_hat2

    # Measure max error
    err, split = _max_bezier_error(seg, u, p0, p1, p2, p3)
    if err <= cfg.fit_error_px or depth >= cfg.max_subdivisions or len(seg) < 5:
        out.append((p0, p1, p2, p3))
        return

    # Split at the worst-fit point and recurse
    _fit_segment(seg[:split + 1], cfg, out, depth + 1)
    _fit_segment(seg[split:], cfg, out, depth + 1)


def _max_bezier_error(seg, u, p0, p1, p2, p3) -> tuple[float, int]:
    """Return (max_distance, index_of_max) between seg points and the Bézier."""
    one_minus_u = (1 - u)[:, None]
    uu = u[:, None]
    curve = (one_minus_u ** 3) * p0 + 3 * (one_minus_u ** 2) * uu * p1 \
        + 3 * one_minus_u * (uu ** 2) * p2 + (uu ** 3) * p3
    d = np.linalg.norm(seg - curve, axis=1)
    idx = int(np.argmax(d))
    return float(d[idx]), idx


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.array([1.0, 0.0])


# -- d-attribute emission ----------------------------------------------------

def _fmt(x: float) -> str:
    """Compact float formatting for SVG d-attributes."""
    s = f"{x:.2f}"
    # Trim trailing zeros: 12.30 -> 12.3, 12.00 -> 12
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s or "0"


def _polyline_to_d(pts: np.ndarray, closed: bool) -> str:
    if len(pts) == 0:
        return ""
    parts = [f"M{_fmt(pts[0,0])},{_fmt(pts[0,1])}"]
    for p in pts[1:]:
        parts.append(f"L{_fmt(p[0])},{_fmt(p[1])}")
    if closed:
        parts.append("Z")
    return "".join(parts)


def _beziers_to_d(beziers, closed: bool) -> str:
    if not beziers:
        return ""
    p0 = beziers[0][0]
    parts = [f"M{_fmt(p0[0])},{_fmt(p0[1])}"]
    for (_, p1, p2, p3) in beziers:
        parts.append(
            f"C{_fmt(p1[0])},{_fmt(p1[1])} "
            f"{_fmt(p2[0])},{_fmt(p2[1])} "
            f"{_fmt(p3[0])},{_fmt(p3[1])}"
        )
    if closed:
        parts.append("Z")
    return "".join(parts)
