"""One photo -> a verdict."""
from __future__ import annotations
import base64
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from . import metrics as M
from . import vlm as V
from .rules import Finding, run_measured_checks
from .config import REQUIRED_VIEWS, VIEWS

THUMB_EDGE = 420


def _decode(image_bytes: bytes) -> Optional[np.ndarray]:
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def _thumb(img_bgr: np.ndarray) -> str:
    h, w = img_bgr.shape[:2]
    s = THUMB_EDGE / float(max(h, w))
    if s < 1:
        img_bgr = cv2.resize(img_bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _semantic_findings(sem: Dict[str, Any], view: str,
                       too_dark: bool = False) -> List[Finding]:
    """The model's answers as findings."""
    out: List[Finding] = []
    if sem.get("status") != "ok" or too_dark:
        return out

    if view == "label":
        if sem.get("label_obstructed"):
            out.append(Finding(
                "label_obstructed", "retake",
                "Something is covering part of the DEMO label.",
                "Take off whatever is over the label \u2014 tape, a finger, a cable \u2014 and retake "
                "so every character is visible.",
                {"model_said": sem.get("note", ""), "confidence": sem.get("confidence")},
                source="model",
            ))
        if sem.get("label_cut_off"):
            out.append(Finding(
                "framing", "retake",
                "Part of the DEMO label is outside the photo.",
                "Back off slightly and re-centre so the whole label, including its borders, "
                "sits inside the frame.",
                {"model_said": sem.get("note", ""), "confidence": sem.get("confidence")},
                source="model",
            ))
        if not sem.get("demo_label_visible"):
            out.append(Finding(
                "label_obstructed", "review",
                "No DEMO label could be found in this photo.",
                "Check the right photo was selected, and that the temporary DEMO label is "
                "attached and facing the camera.",
                {"model_said": sem.get("note", "")},
                source="model",
            ))
    return out


def _view_mismatch(sem: Dict[str, Any], view: str) -> Optional[Dict[str, Any]]:
    """A supplied view name is the intended view, not proof of content."""
    if sem.get("status") != "ok":
        return None
    seen = sem.get("observed_view")
    if seen in (None, "uncertain", view):
        return None
    if view == "label" or seen == "label":
        return None
    if float(sem.get("confidence") or 0) < 0.7:
        return None
    return {
        "intended": view,
        "observed": seen,
        "confidence": sem.get("confidence"),
        "message": (f"Filed as {VIEWS[view]['label']}, but this looks like a "
                    f"{VIEWS.get(seen, {}).get('label', seen)} shot."),
    }


def analyze_image(image_bytes: bytes, view: str, image_id: str = "",
                  filename: str = "", use_vlm: bool = True,
                  use_cache: bool = True) -> Dict[str, Any]:
    img = _decode(image_bytes)
    if img is None:
        return {"image_id": image_id, "filename": filename, "intended_view": view,
                "status": "needs_review", "issues": [], "findings": [],
                "error": "This file could not be read as an image."}

    small = M.prepare(img)
    fr = M.framing(small)
    m = {
        "framing": fr,
        "blur": M.blur(small, bbox=fr.get("bbox")),
        "exposure": M.exposure(small),
        "glare": M.glare(small),
    }

    findings = run_measured_checks(m, view)
    sem: Dict[str, Any] = {"status": "skipped"}
    if use_vlm:
        sem = V.inspect(img, image_bytes, use_cache=use_cache)
        findings += _semantic_findings(
            sem, view, too_dark=any(
                f.code == "underexposed" and f.severity == "retake"
                for f in findings))

    mismatch = _view_mismatch(sem, view)

    if any(f.severity == "retake" for f in findings):
        status = "retake"
    elif any(f.severity == "review" for f in findings) or mismatch:
        status = "needs_review"
    else:
        status = "usable"

    order = {c: i for i, c in enumerate(
        ["blur", "underexposed", "glare_or_overexposed", "framing", "label_obstructed"])}
    issues = sorted({f.code for f in findings}, key=lambda c: order.get(c, 99))

    return {
        "image_id": image_id,
        "filename": filename,
        "intended_view": view,
        "status": status,
        "issues": issues,
        "findings": [{
            "code": f.code, "severity": f.severity, "detail": f.detail,
            "guidance": f.guidance, "evidence": f.evidence, "source": f.source,
        } for f in findings],
        "view_mismatch": mismatch,
        "semantic": {
            "status": sem.get("status"),
            "observed_view": sem.get("observed_view"),
            "confidence": sem.get("confidence"),
            "note": sem.get("note"),
            "reason": sem.get("reason"),
            "cached": sem.get("cached"),
        },
        "metrics": {
            "sharpness": (m["blur"] or {}).get("global_sharpness"),
            "mean_luma": (m["exposure"] or {}).get("mean_luma"),
            "contrast": (m["exposure"] or {}).get("contrast"),
            "fill_ratio": fr.get("fill_ratio"),
            "glare_blob_frac": (m["glare"] or {}).get("largest_blob_area_frac"),
        },
        "thumb": _thumb(img),
    }


def summarize_set(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Set-level checklist."""
    by_view: Dict[str, List[Dict[str, Any]]] = {v: [] for v in REQUIRED_VIEWS}
    for r in results:
        by_view.setdefault(r["intended_view"], []).append(r)

    checklist, missing, needs_work = [], [], []
    for v in REQUIRED_VIEWS:
        got = by_view.get(v, [])
        if not got:
            state = "missing"
            missing.append(v)
        elif any(g["status"] == "usable" for g in got):
            state = "satisfied"
        elif any(g["status"] == "needs_review" for g in got):
            state = "unconfirmed"
            needs_work.append(v)
        else:
            state = "all_need_retake"
            needs_work.append(v)
        checklist.append({
            "view": v, "label": VIEWS[v]["label"], "needs": VIEWS[v]["needs"],
            "state": state, "count": len(got),
            "usable": sum(1 for g in got if g["status"] == "usable"),
            "retake": sum(1 for g in got if g["status"] == "retake"),
            "review": sum(1 for g in got if g["status"] == "needs_review"),
        })

    extra = [v for v in by_view if v not in REQUIRED_VIEWS and by_view[v]]
    complete = not missing and not needs_work

    if missing:
        names = ", ".join(VIEWS[v]["label"] for v in missing)
        nxt = f"Shoot the missing view{'s' if len(missing) > 1 else ''}: {names}."
    elif needs_work:
        names = ", ".join(VIEWS[v]["label"] for v in needs_work)
        nxt = f"Every required view has a photo, but {names} has no usable one yet \u2014 retake it."
    else:
        nxt = "All three required views have a usable photo. This set is complete."

    return {
        "checklist": checklist,
        "missing_views": missing,
        "views_needing_retake": needs_work,
        "extra_views": extra,
        "complete": complete,
        "next_action": nxt,
        "counts": {
            "total": len(results),
            "usable": sum(1 for r in results if r["status"] == "usable"),
            "retake": sum(1 for r in results if r["status"] == "retake"),
            "needs_review": sum(1 for r in results if r["status"] == "needs_review"),
        },
    }
