"""The two questions a Laplacian can't answer, asked of Claude."""
from __future__ import annotations
import base64, hashlib, json, os, re
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

from .config import MODEL, ROOT

PROMPT_VERSION = "v4"
CACHE = ROOT / "cache"
CACHE.mkdir(exist_ok=True)

VLM_MAX_EDGE = 2048
_disabled_reason: Optional[str] = None
_last_error: Optional[str] = None
_FATAL = ("credit balance is too low", "authentication_error", "invalid x-api-key",
          "permission_error", "not_found_error")

SYSTEM = (
    "You inspect photographs of scrap electronic equipment for a recycling intake workflow. "
    "You answer only what you can see. You never guess at text you cannot read, and you never "
    "invent a serial number, model or asset tag.\n\n"
    "Some photographs have solid BLACK RECTANGLES painted over them. Those are deliberate "
    "privacy masks hiding real serial numbers or names before the photo was shared. They are "
    "NOT a defect, NOT an obstruction, and NOT the label you are asked about. Ignore them "
    "entirely.\n\n"
    "The label you care about is a TEMPORARY DEMO LABEL stuck on the equipment for this "
    "exercise -- a small paper or sticker label, typically orange or bright, reading DEMO. "
    "That label, and only that label, is what 'the label' means below."
)

QUESTION = """Answer these questions about this photograph.

1. observed_view -- which ONE of these does the photograph actually show?
   - "front": the front face of the device
   - "rear_ports": the back or connection side, where the ports and connectors are
   - "label": a close-up whose subject is the temporary DEMO label
   - "uncertain": you genuinely cannot tell
   Judge what is in the picture, not what it ought to be.

2. demo_label_visible -- is a temporary DEMO label present anywhere in this photograph?
   true / false.

3. label_obstructed -- is any part of the DEMO label's PRINTED FACE covered up, so that
   some of what is written on it cannot be read?
   Count as obstructed: a strip of tape laid ACROSS the label, a second piece of paper
   or label overlapping it, a finger, a cable, a box, any object resting on it.
   Look carefully at tape. Clear or matte tape running over the middle of the label,
   over its text, still counts as covering it even though you can partly see through it.
   Do NOT count: tape that only touches the outer border or corners while leaving all
   the writing exposed; the label merely being dark, blurred, angled, small or glary;
   a black privacy mask.
   Ask yourself: is any CHARACTER on this label hidden or obscured by something on top
   of it? If yes, true. false if no DEMO label is visible at all.

4. label_cut_off -- does any part of the DEMO label fall outside the edges of the
   photograph, so you cannot see the whole label? false if no DEMO label is visible.

5. confidence -- 0.0 to 1.0, how sure you are about observed_view.

6. note -- one short sentence of visual description. Do not state a verdict.

Reply with ONLY a JSON object:
{"observed_view": "...", "demo_label_visible": true, "label_obstructed": false,
 "label_cut_off": false, "confidence": 0.0, "note": "..."}"""


def api_configured() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY")) and _disabled_reason is None


def disabled_reason() -> Optional[str]:
    return _disabled_reason


def last_error() -> Optional[str]:
    """Most recent call failure, disabling or not."""
    return _last_error


def cache_key(image_bytes: bytes) -> str:
    return f"{hashlib.sha1(image_bytes).hexdigest()[:16]}_{PROMPT_VERSION}_{VLM_MAX_EDGE}"


def cache_get(key: str) -> Optional[Dict[str, Any]]:
    p = CACHE / f"{key}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def cache_put(key: str, payload: Dict[str, Any]) -> None:
    try:
        (CACHE / f"{key}.json").write_text(json.dumps(payload, indent=1))
    except Exception:
        pass


def _encode(img_bgr: np.ndarray) -> str:
    h, w = img_bgr.shape[:2]
    longest = max(h, w)
    if longest > VLM_MAX_EDGE:
        s = VLM_MAX_EDGE / float(longest)
        img_bgr = cv2.resize(img_bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return base64.b64encode(buf.tobytes()).decode("ascii")


_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def _parse(text: str) -> Optional[Dict[str, Any]]:
    t = _FENCE.sub("", text.strip())
    try:
        return json.loads(t)
    except Exception:
        m = re.search(r"\{.*\}", t, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None


def _normalize(raw: Dict[str, Any]) -> Dict[str, Any]:
    view = str(raw.get("observed_view", "uncertain")).strip().lower()
    if view not in ("front", "rear_ports", "label", "uncertain"):
        view = "uncertain"
    try:
        conf = max(0.0, min(1.0, float(raw.get("confidence", 0.0))))
    except Exception:
        conf = 0.0
    return {
        "status": "ok",
        "observed_view": view,
        "demo_label_visible": bool(raw.get("demo_label_visible", False)),
        "label_obstructed": bool(raw.get("label_obstructed", False)),
        "label_cut_off": bool(raw.get("label_cut_off", False)),
        "confidence": round(conf, 2),
        "note": str(raw.get("note", ""))[:300],
        "model": MODEL,
    }


def _call(b64: str) -> Optional[Dict[str, Any]]:
    global _disabled_reason
    try:
        import anthropic
    except ImportError:
        _disabled_reason = "the anthropic package is not installed"
        return None
    try:
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        resp = client.messages.create(
            model=MODEL, max_tokens=400, system=SYSTEM,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": QUESTION},
            ]}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        raw = _parse(text)
        if raw is None:
            return None
        out = _normalize(raw)
        u = getattr(resp, "usage", None)
        if u:
            out["usage"] = {"input_tokens": u.input_tokens, "output_tokens": u.output_tokens}
        return out
    except Exception as exc:
        msg = str(exc).lower()
        if any(f in msg for f in _FATAL):
            _disabled_reason = str(exc)[:200]
        elif isinstance(exc, TypeError):
            _disabled_reason = f"client call is malformed: {exc}"
        _last_error = str(exc)[:200]
        globals()["_last_error"] = _last_error
        return None


def inspect(img_bgr: np.ndarray, image_bytes: bytes, use_cache: bool = True) -> Dict[str, Any]:
    """Read one photo. Always returns a dict, never raises."""
    key = cache_key(image_bytes)
    if use_cache:
        hit = cache_get(key)
        if hit:
            hit["cached"] = True
            return hit
    if not api_configured():
        return {"status": "unavailable",
                "reason": _disabled_reason or "no ANTHROPIC_API_KEY set",
                "observed_view": "uncertain", "demo_label_visible": False,
                "label_obstructed": False, "label_cut_off": False, "confidence": 0.0, "note": ""}
    out = _call(_encode(img_bgr))
    if out is None:
        return {"status": "unavailable",
                "reason": _disabled_reason or "the model call failed or returned unparseable output",
                "observed_view": "uncertain", "demo_label_visible": False,
                "label_obstructed": False, "label_cut_off": False, "confidence": 0.0, "note": ""}
    out["cached"] = False
    cache_put(key, out)
    return out
