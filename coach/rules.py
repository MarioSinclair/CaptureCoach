"""The measured checks."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .config import THRESHOLDS as T, VIEWS


@dataclass
class Finding:
    code: str
    severity: str
    detail: str
    guidance: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    source: str = "measured"


def _subject_word(view: str) -> str:
    return "DEMO label" if view == "label" else "device"


def check_exposure(m: Dict[str, Any], view: str) -> List[Finding]:
    ex = m.get("exposure") or {}
    if ex.get("status") != "ok":
        return []
    luma, contrast = ex["mean_luma"], ex["contrast"]
    subj = _subject_word(view)

    if luma < T["luma_min"]:
        return [Finding(
            "underexposed", "retake",
            f"Too dark to make out the {subj}.",
            "Add light or move somewhere brighter, then retake. Avoid shooting into your own shadow.",
            {"mean_luma": luma, "threshold": T["luma_min"]},
        )]

    if luma > T["luma_max"]:
        washed = contrast < T["washed_contrast_max"]
        if washed:
            return [Finding(
                "glare_or_overexposed", "retake",
                f"So bright that the detail is washed off the {subj}.",
                "Move the light source or change your angle so it is not bouncing straight back "
                "into the lens, then retake.",
                {"mean_luma": luma, "contrast": contrast, "threshold": T["luma_max"]},
            )]
        return [Finding(
            "glare_or_overexposed", "review",
            f"Very bright, but the {subj} may still be readable.",
            "A reviewer should confirm the required detail is legible before this is accepted.",
            {"mean_luma": luma, "contrast": contrast, "threshold": T["luma_max"]},
        )]
    return []


def exposure_failed(findings: List[Finding]) -> bool:
    """Exposure bad enough that other measurements can't be trusted."""
    return any(f.code in ("underexposed", "glare_or_overexposed") and f.severity == "retake"
               for f in findings)


def exposure_doubtful(findings: List[Finding]) -> bool:
    """Any exposure doubt, including the ambiguous bright band."""
    return any(f.code in ("underexposed", "glare_or_overexposed") for f in findings)


def check_blur(m: Dict[str, Any], view: str, exposure_bad: bool) -> List[Finding]:
    """Focus, measured as edge narrowness."""
    if exposure_bad:
        return []
    bl = m.get("blur") or {}
    if bl.get("status") != "ok":
        return []
    s = bl.get("global_sharpness")
    if s is None:
        return []
    subj = _subject_word(view)

    if s < T["sharpness_retake"]:
        return [Finding(
            "blur", "retake",
            f"Out of focus, or the camera moved. The {subj} is not sharp enough to read.",
            f"Brace your elbows or rest the camera on something, tap the {subj} on screen to "
            "focus, wait for it to lock, then take the shot.",
            {"sharpness": s, "threshold": T["sharpness_retake"]},
        )]
    if s < T["sharpness_review"]:
        return [Finding(
            "blur", "review",
            f"The {subj} looks slightly soft. This one is borderline.",
            f"If the {subj} is not crisply readable, refocus and retake; otherwise a reviewer "
            "can accept it.",
            {"sharpness": s, "threshold": T["sharpness_review"]},
        )]
    return []


def check_framing(m: Dict[str, Any], view: str, exposure_bad: bool, blur_bad: bool) -> List[Finding]:
    """Does the whole subject fit in the frame?"""
    if VIEWS.get(view, {}).get("whole_device") is not True:
        return []
    if exposure_bad or blur_bad:
        return []
    fr = m.get("framing") or {}
    if fr.get("status") != "ok":
        return []
    fill = fr.get("fill_ratio")
    if fill is None:
        return []
    limit = T["fill_max_front"] if view == "front" else T["fill_max_rear"]
    if fill <= limit:
        return []

    edges = fr.get("edges_touched") or []
    if edges:
        named = " and ".join([", ".join(edges[:-1]), edges[-1]] if len(edges) > 2 else edges)
        detail = (f"The device runs off the {named} "
                  f"edge{'s' if len(edges) > 1 else ''} of the frame.")
    else:
        detail = "The device fills the frame, so its outer edges are not all visible."
    return [Finding(
        "framing", "retake", detail,
        "Step back about an arm's length, or turn the phone to landscape, so the whole device "
        "sits inside the frame with a margin around it.",
        {"fill_ratio": fill, "threshold": limit, "edges_touched": edges},
    )]


def check_glare(m: Dict[str, Any], view: str, already_flagged: bool) -> List[Finding]:
    """A compact hotspot on an otherwise well-exposed frame."""
    if already_flagged:
        return []
    gl = m.get("glare") or {}
    if gl.get("status") != "ok":
        return []
    frac = gl.get("largest_blob_area_frac", 0.0)
    extent = gl.get("largest_blob_extent", 0.0)
    if frac < T["glare_blob_frac"] or extent < T["glare_extent_min"]:
        return []
    return [Finding(
        "glare_or_overexposed", "retake",
        "A bright reflection is hiding part of the photo.",
        "Tilt the device or your camera about 20 degrees so the light is not reflecting "
        f"straight back at the lens, then retake.",
        {"blob_frac": frac, "extent": extent, "threshold": T["glare_blob_frac"]},
    )]


def run_measured_checks(m: Dict[str, Any], view: str) -> List[Finding]:
    """All measured checks, in dependency order."""
    out: List[Finding] = []
    exp = check_exposure(m, view)
    out += exp
    exp_bad = exposure_failed(exp)

    bl = check_blur(m, view, exp_bad)
    out += bl
    blur_bad = any(f.severity == "retake" for f in bl)

    out += check_glare(m, view, already_flagged=any(f.code == "glare_or_overexposed" for f in out))
    out += check_framing(m, view, exposure_doubtful(out), blur_bad)
    return out
