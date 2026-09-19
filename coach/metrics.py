"""CV metrics."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

MAX_EDGE = 1024
TILE_GRID = 4


def prepare(img_bgr: np.ndarray, max_edge: int = MAX_EDGE) -> np.ndarray:
    """Resize so the longest edge is max_edge. Never upscales."""
    h, w = img_bgr.shape[:2]
    longest = max(h, w)
    if longest <= max_edge or longest == 0:
        return img_bgr
    scale = max_edge / float(longest)
    return cv2.resize(
        img_bgr,
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def to_gray(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr.ndim == 2:
        return img_bgr
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def _norm_bbox(x: int, y: int, w: int, h: int, shape: Tuple[int, int]) -> List[float]:
    H, W = shape[:2]
    return [
        round(x / float(W), 4),
        round(y / float(H), 4),
        round(w / float(W), 4),
        round(h / float(H), 4),
    ]


def _region_name(row: int, col: int, grid: int) -> str:
    """Human name for a tile position, e.g. 'top-left', 'centre'."""
    mid = (grid - 1) / 2.0
    vert = "top" if row < mid - 0.25 else ("bottom" if row > mid + 0.25 else "middle")
    horz = "left" if col < mid - 0.25 else ("right" if col > mid + 0.25 else "centre")
    if vert == "middle" and horz == "centre":
        return "centre"
    if vert == "middle":
        return f"{horz} side"
    if horz == "centre":
        return f"{vert} centre"
    return f"{vert}-{horz}"


def edge_sharpness(gray: np.ndarray) -> float:
    """Focus, measured as edge narrowness rather than edge strength."""
    if min(gray.shape[:2]) >= 200:
        gray = cv2.resize(gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    g = gray.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    lap = np.abs(cv2.Laplacian(g, cv2.CV_32F))
    if grad.size < 200:
        return 0.0
    thresh = float(np.percentile(grad, 90))
    mask = grad > max(thresh, 1e-3)
    if int(mask.sum()) < 50:
        return 0.0
    return float(lap[mask].sum() / max(float(grad[mask].sum()), 1e-6))


def blur(img_bgr: np.ndarray, grid: int = TILE_GRID,
         bbox: Optional[List[float]] = None) -> Dict[str, Any]:
    """Laplacian variance, globally and over a tile map."""
    gray = to_gray(img_bgr)
    H, W = gray.shape[:2]
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    global_score = float(lap.var())
    global_sharpness = edge_sharpness(gray)

    tile_scores: List[List[float]] = []
    tile_texture: List[List[float]] = []
    th, tw = H // grid, W // grid
    for r in range(grid):
        row_scores, row_tex = [], []
        for c in range(grid):
            y0, y1 = r * th, (r + 1) * th if r < grid - 1 else H
            x0, x1 = c * tw, (c + 1) * tw if c < grid - 1 else W
            patch = gray[y0:y1, x0:x1]
            if patch.size == 0:
                row_scores.append(0.0)
                row_tex.append(0.0)
                continue
            row_scores.append(round(edge_sharpness(patch), 4))
            row_tex.append(float(patch.std()))
        tile_scores.append([round(v, 2) for v in row_scores])
        tile_texture.append([round(v, 2) for v in row_tex])

    texture_floor = 12.0
    candidates = [
        (tile_scores[r][c], r, c)
        for r in range(grid) for c in range(grid)
        if tile_texture[r][c] >= texture_floor
    ]
    median_tile = float(np.median([c[0] for c in candidates])) if candidates else 0.0
    if candidates:
        worst_score, wr, wc = min(candidates)
        y0, x0 = wr * th, wc * tw
        h = (H - y0) if wr == grid - 1 else th
        w = (W - x0) if wc == grid - 1 else tw
        worst_bbox = _norm_bbox(x0, y0, w, h, gray.shape)
        worst_region = _region_name(wr, wc, grid)
        textured_tiles = len(candidates)
    else:
        worst_score, worst_bbox, worst_region, textured_tiles = global_score, None, None, 0

    subject: Dict[str, Any] = {"status": "unavailable"}
    if bbox:
        x, y, bw, bh = [int(round(v * sc)) for v, sc in zip(bbox, (W, H, W, H))]
        x, y = max(0, min(x, W - 1)), max(0, min(y, H - 1))
        bw, bh = max(1, min(bw, W - x)), max(1, min(bh, H - y))
        if bw * bh >= 400 and (W * H - bw * bh) >= 400:
            inside = lap[y:y + bh, x:x + bw]
            mask = np.ones(gray.shape, bool)
            mask[y:y + bh, x:x + bw] = False
            outside = lap[mask]
            s_score = float(inside.var())
            s_sharp = edge_sharpness(gray[y:y + bh, x:x + bw])
            o_sharp = edge_sharpness(np.delete(gray, slice(y, y + bh), axis=0)) \
                if bh < H else None
            o_score = float(outside.var()) if outside.size >= 400 else None
            subject = {
                "status": "ok",
                "score": round(s_score, 2),
                "sharpness": round(s_sharp, 4),
                "surround_sharpness": (round(o_sharp, 4) if o_sharp else None),
                "texture": round(float(gray[y:y + bh, x:x + bw].std()), 2),
                "surround_score": round(o_score, 2) if o_score is not None else None,
                "ratio_to_surround": (round(s_score / o_score, 3)
                                      if o_score and o_score > 0 else None),
                "area_frac": round((bw * bh) / float(W * H), 4),
            }

    return {
        "status": "ok",
        "global_score": round(global_score, 2),
        "global_sharpness": round(global_sharpness, 4),
        "subject": subject,
        "tile_scores": tile_scores,
        "tile_texture": tile_texture,
        "worst_tile_score": round(float(worst_score), 2),
        "median_tile_score": round(median_tile, 2),
        "worst_tile_bbox": worst_bbox,
        "worst_tile_region": worst_region,
        "textured_tiles": textured_tiles,
        "grid": grid,
    }


def exposure(img_bgr: np.ndarray) -> Dict[str, Any]:
    """Brightness, clipping at both ends, contrast."""
    gray = to_gray(img_bgr)
    g = gray.astype(np.float32)
    return {
        "status": "ok",
        "mean_luma": round(float(g.mean()), 2),
        "clipped_black": round(float((gray <= 5).mean()), 5),
        "clipped_white": round(float((gray >= 250).mean()), 5),
        "contrast": round(float(g.std()), 2),
    }


def glare(img_bgr: np.ndarray) -> Dict[str, Any]:
    """Largest compact blown-out blob."""
    gray = to_gray(img_bgr)
    H, W = gray.shape[:2]
    frame_area = float(H * W)

    _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    n, _labels, stats, _cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return {
            "status": "ok",
            "largest_blob_area_frac": 0.0,
            "largest_blob_extent": 0.0,
            "blob_count": 0,
            "bbox": None,
            "total_blown_frac": 0.0,
        }

    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = int(np.argmax(areas)) + 1
    area = float(stats[idx, cv2.CC_STAT_AREA])
    x = int(stats[idx, cv2.CC_STAT_LEFT])
    y = int(stats[idx, cv2.CC_STAT_TOP])
    w = int(stats[idx, cv2.CC_STAT_WIDTH])
    h = int(stats[idx, cv2.CC_STAT_HEIGHT])
    bbox_area = float(max(1, w * h))

    return {
        "status": "ok",
        "largest_blob_area_frac": round(area / frame_area, 5),
        "largest_blob_extent": round(area / bbox_area, 4),
        "blob_count": int(n - 1),
        "bbox": _norm_bbox(x, y, w, h, gray.shape),
        "total_blown_frac": round(float(areas.sum()) / frame_area, 5),
    }


def occlusion(img_bgr: np.ndarray, bbox: Optional[List[float]] = None) -> Dict[str, Any]:
    """How much of the unit is covered by a hand."""
    H, W = img_bgr.shape[:2]
    ycc = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    cr, cb = ycc[:, :, 1], ycc[:, :, 2]
    m1 = (cr >= 133) & (cr <= 178) & (cb >= 77) & (cb <= 130)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    m2 = ((h <= 25) | (h >= 172)) & (sat >= 30) & (sat <= 180) & (val >= 55)

    mask = (m1 & m2).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    frame_frac = float((mask > 0).mean())
    gray = to_gray(img_bgr)
    region_texture = float(gray[mask > 0].std()) if int((mask > 0).sum()) > 200 else 0.0

    rectangularity, vertices = 0.0, 0
    cnts, _ = cv2.findContours((mask > 0).astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        big = max(cnts, key=cv2.contourArea)
        area = float(cv2.contourArea(big))
        bx, by, bw_, bh_ = cv2.boundingRect(big)
        if bw_ * bh_ > 0:
            rectangularity = area / float(bw_ * bh_)
        peri = cv2.arcLength(big, True)
        vertices = len(cv2.approxPolyDP(big, 0.02 * peri, True))

    too_big = frame_frac >= 0.60
    boxy = rectangularity >= 0.88 and vertices <= 6
    too_flat = region_texture < 4.0
    if too_big or boxy or too_flat:
        return {
            "status": "ok", "frame_frac": round(frame_frac, 4),
            "device_frac": 0.0, "texture": round(region_texture, 2),
            "rectangularity": round(rectangularity, 3), "vertices": int(vertices),
            "rejected": ("covers most of the frame - reads as a tan surface, not a hand"
                         if too_big else
                         f"rectangular ({rectangularity:.2f} of its box, {vertices} corners)"
                         " - reads as tan equipment, not a hand" if boxy else
                         "no texture at all - reads as flat tan plastic"),
            "bbox": None,
        }

    device_frac, obox = 0.0, None
    if bbox:
        x, y, bw, bh = [int(round(v * sc)) for v, sc in zip(bbox, (W, H, W, H))]
        x, y = max(0, min(x, W - 1)), max(0, min(y, H - 1))
        bw, bh = max(1, min(bw, W - x)), max(1, min(bh, H - y))
        sub = mask[y:y + bh, x:x + bw]
        device_frac = float((sub > 0).mean())
        n, _l, st, _c = cv2.connectedComponentsWithStats((sub > 0).astype(np.uint8), 8)
        if n > 1:
            i = int(np.argmax(st[1:, cv2.CC_STAT_AREA])) + 1
            obox = _norm_bbox(x + int(st[i, cv2.CC_STAT_LEFT]), y + int(st[i, cv2.CC_STAT_TOP]),
                              int(st[i, cv2.CC_STAT_WIDTH]), int(st[i, cv2.CC_STAT_HEIGHT]),
                              gray.shape)

    return {
        "status": "ok",
        "frame_frac": round(frame_frac, 4),
        "device_frac": round(device_frac, 4),
        "texture": round(region_texture, 2),
        "rectangularity": round(rectangularity, 3),
        "vertices": int(vertices),
        "bbox": obox,
    }


def framing(img_bgr: np.ndarray) -> Dict[str, Any]:
    """Locate the device and measure how it sits in the frame."""
    gray = to_gray(img_bgr)
    H, W = gray.shape[:2]

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=2)

    if edges.max() == 0:
        return {"status": "unavailable", "reason": "no edges detected (flat or out-of-focus image)"}

    gw = gh = 48
    density = cv2.resize(edges.astype(np.float32) / 255.0, (gw, gh), interpolation=cv2.INTER_AREA)

    cutoff = max(0.02, min(0.25, 2.5 * float(np.median(density))))
    content = density >= cutoff
    n_content = int(content.sum())
    if n_content < 4:
        return {"status": "unavailable", "reason": "no region dense enough to be the device"}

    def _span(mass: np.ndarray) -> Tuple[int, int]:
        total = float(mass.sum())
        cum = np.cumsum(mass) / total
        lo = int(np.searchsorted(cum, 0.02))
        hi = int(np.searchsorted(cum, 0.98))
        return lo, max(hi, lo)

    closed = cv2.morphologyEx(content.astype(np.uint8), cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    n_lab, _lbl, blob_stats, _cent = cv2.connectedComponentsWithStats(closed, connectivity=8)
    candidates: List[Dict[str, Any]] = []
    for i in range(1, n_lab):
        cx0 = int(blob_stats[i, cv2.CC_STAT_LEFT]); cy0 = int(blob_stats[i, cv2.CC_STAT_TOP])
        cw = int(blob_stats[i, cv2.CC_STAT_WIDTH]); ch = int(blob_stats[i, cv2.CC_STAT_HEIGHT])
        cells = int(blob_stats[i, cv2.CC_STAT_AREA])
        if cells < 3:
            continue
        px = int(round(cx0 / gw * W)); py = int(round(cy0 / gh * H))
        pw = max(1, int(round(cw / gw * W))); phh = max(1, int(round(ch / gh * H)))
        candidates.append({
            "bbox": _norm_bbox(px, py, pw, phh, gray.shape),
            "fill_ratio": round((pw * phh) / float(H * W), 4),
            "content_cells": cells,
            "occupancy": round(cells / float(max(1, cw * ch)), 3),
        })
    candidates.sort(key=lambda c: -c["content_cells"])
    candidates = candidates[:4]

    gx, gx1 = _span(content.sum(axis=0).astype(np.float64))
    gy, gy1 = _span(content.sum(axis=1).astype(np.float64))
    gwid, ghei = gx1 - gx + 1, gy1 - gy + 1

    x = int(round(gx / gw * W))
    y = int(round(gy / gh * H))
    w = max(1, int(round(gwid / gw * W)))
    h = max(1, int(round(ghei / gh * H)))
    fill_ratio = (w * h) / float(H * W)

    box_occupancy = float(content[gy:gy1 + 1, gx:gx1 + 1].mean())

    ring = np.zeros((gh, gw), bool)
    ring[0, :] = ring[-1, :] = True
    ring[:, 0] = ring[:, -1] = True
    border_content_frac = float(content[ring].mean())

    edge_content = {
        "top": round(float(content[0, :].mean()), 4),
        "bottom": round(float(content[-1, :].mean()), 4),
        "left": round(float(content[:, 0].mean()), 4),
        "right": round(float(content[:, -1].mean()), 4),
    }
    max_edge_content = max(edge_content.values())

    if fill_ratio < 0.02:
        return {
            "status": "unavailable",
            "reason": "no region large enough to be the device",
            "fill_ratio": round(float(fill_ratio), 4),
        }
    if fill_ratio > 0.97 and border_content_frac < 0.25:
        return {
            "status": "unavailable",
            "reason": "content region spans the frame without reaching the border (busy background)",
            "fill_ratio": round(float(fill_ratio), 4),
            "border_content_frac": round(border_content_frac, 4),
        }

    margin_x, margin_y = 0.02 * W, 0.02 * H
    edges_touched = []
    if x <= margin_x:
        edges_touched.append("left")
    if y <= margin_y:
        edges_touched.append("top")
    if (x + w) >= (W - margin_x):
        edges_touched.append("right")
    if (y + h) >= (H - margin_y):
        edges_touched.append("bottom")

    cx, cy = x + w / 2.0, y + h / 2.0
    dx = (cx - W / 2.0) / float(W)
    dy = (cy - H / 2.0) / float(H)
    center_offset = float(np.hypot(dx, dy))

    return {
        "status": "ok",
        "detector": "edge-density-grid",
        "fill_ratio": round(float(fill_ratio), 4),
        "touches_edge": bool(edges_touched),
        "edges_touched": edges_touched,
        "center_offset": round(center_offset, 4),
        "border_content_frac": round(border_content_frac, 4),
        "box_occupancy": round(box_occupancy, 4),
        "candidates": candidates,
        "ambiguous": bool(
            len(candidates) >= 2
            and candidates[1]["content_cells"] >= 0.55 * candidates[0]["content_cells"]),
        "edge_content": edge_content,
        "max_edge_content": round(max_edge_content, 4),
        "content_cutoff": round(cutoff, 4),
        "bbox": _norm_bbox(x, y, w, h, gray.shape),
    }


def _safe(fn, img, name: str) -> Dict[str, Any]:
    try:
        return fn(img)
    except Exception as exc:
        return {"status": "unavailable", "reason": f"{name} failed: {exc}"}


def all_metrics(img_bgr: np.ndarray, cheap_only: bool = False) -> Dict[str, Any]:
    """Every metric family for one already-decoded BGR image."""
    img = prepare(img_bgr)
    H, W = img.shape[:2]
    out: Dict[str, Any] = {
        "width": int(W),
        "height": int(H),
        "exposure": _safe(exposure, img, "exposure"),
        "glare": _safe(glare, img, "glare"),
        "framing": _safe(framing, img, "framing"),
    }
    fr = out["framing"]
    device_bbox = fr.get("bbox") if fr.get("status") == "ok" else None
    try:
        out["blur"] = blur(img, bbox=device_bbox)
    except Exception as exc:
        out["blur"] = {"status": "unavailable", "reason": f"blur failed: {exc}"}
    try:
        out["occlusion"] = occlusion(img, bbox=device_bbox)
    except Exception as exc:
        out["occlusion"] = {"status": "unavailable", "reason": f"occlusion failed: {exc}"}
    return out


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print("usage: python -m analysis.metrics <image> [<image> ...]")
        raise SystemExit(2)
    for path in sys.argv[1:]:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"{path}: UNREADABLE")
            continue
        m = all_metrics(img)
        b, e, g, f = m["blur"], m["exposure"], m["glare"], m["framing"]
        print(f"\n=== {path}  ({m['width']}x{m['height']}) ===")
        print(f"  blur      global={b.get('global_score')}  worst_tile={b.get('worst_tile_score')} "
              f"@{b.get('worst_tile_region')}")
        print(f"  exposure  luma={e.get('mean_luma')}  contrast={e.get('contrast')}  "
              f"clipW={e.get('clipped_white')}  clipB={e.get('clipped_black')}")
        print(f"  glare     blob={g.get('largest_blob_area_frac')}  extent={g.get('largest_blob_extent')}")
        print(f"  framing   [{f.get('status')}] fill={f.get('fill_ratio')}  "
              f"edges={f.get('edges_touched')}  offset={f.get('center_offset')}")
