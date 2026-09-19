"""Required views and the thresholds the measured checks fire on."""
from __future__ import annotations
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_env() -> None:
    """Read .env without a dependency. Real env vars win."""
    p = ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env()

MODEL = os.environ.get("CAPTURE_COACH_MODEL", "claude-sonnet-5")

VIEWS = {
    "front": {
        "label": "Front",
        "needs": "The front of the device, including its outer edges",
        "whole_device": True,
    },
    "rear_ports": {
        "label": "Rear / ports",
        "needs": "The rear or connection side, including the port area",
        "whole_device": True,
    },
    "label": {
        "label": "DEMO label",
        "needs": "The complete temporary DEMO label, readable and unobstructed",
        "whole_device": False,
    },
}
REQUIRED_VIEWS = ["front", "rear_ports", "label"]

ISSUE_CODES = ["blur", "underexposed", "glare_or_overexposed", "framing", "label_obstructed"]

THRESHOLDS = {
    "luma_min": 70.0,
    "_luma_min": "33.1 vs 123.0 -- wide gap",

    "luma_max": 210.0,
    "_luma_max": "199.8 clean vs 213.8 ambiguous -- narrow, 3 examples",

    "washed_contrast_max": 50.0,
    "_washed_contrast_max": "one example each side -- weakly supported",

    "sharpness_retake": 0.155,
    "sharpness_review": 0.172,
    "_sharpness": "3 clear at <=0.094; 4th at 0.168 vs 0.175 clean -- tight",

    "fill_max_front": 0.58,
    "fill_max_rear": 0.80,
    "_fill": "front 0.534 clean vs 0.608 cropped; rear has no positive example",

    "glare_blob_frac": 0.12,
    "glare_extent_min": 0.35,
    "_glare_blob": "no positive example; above the 0.083 false positive",
}
