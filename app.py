"""HTTP API and static host."""
from __future__ import annotations
import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from coach import pipeline, vlm
from coach.config import REQUIRED_VIEWS, VIEWS, ROOT, MODEL

app = FastAPI(title="Circular Capture Coach")
MAX_FILES = 24
MAX_BYTES = 25 * 1024 * 1024


def _practice_sets():
    sets = {}
    if not (ROOT / "photo_sets.csv").exists():
        return sets
    man = {r["image_id"]: r for r in csv.DictReader(open(ROOT / "manifest.csv"))}
    for r in csv.DictReader(open(ROOT / "photo_sets.csv")):
        sets.setdefault(r["set_id"], {"set_id": r["set_id"], "device_id": r["device_id"], "images": []})
        sets[r["set_id"]]["images"].append({
            "image_id": r["image_id"], "view": r["intended_view"],
            "path": man.get(r["image_id"], {}).get("relative_path", ""),
        })
    return sets


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "model": MODEL,
        "semantic_checks": "available" if vlm.api_configured() else "unavailable",
        "semantic_reason": vlm.disabled_reason(),
        "cached_responses": len(list((ROOT / "cache").glob("*.json"))),
        "required_views": REQUIRED_VIEWS,
        "views": VIEWS,
        "practice_sets": sorted(_practice_sets().keys()),
    }


def _run(payloads, use_vlm: bool, use_cache: bool = True):
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(
            lambda p: pipeline.analyze_image(p["bytes"], p["view"], image_id=p["id"],
                                             filename=p["name"], use_vlm=use_vlm,
                                             use_cache=use_cache),
            payloads))
    return {"images": results, "summary": pipeline.summarize_set(results)}


@app.post("/api/analyze")
async def analyze(files: List[UploadFile] = File(...), views: List[str] = Form(...),
                  use_vlm: str = Form("1"), fresh: str = Form("0")):
    if not files:
        raise HTTPException(400, "No files were uploaded.")
    if len(files) > MAX_FILES:
        raise HTTPException(400, f"At most {MAX_FILES} photos per set.")
    if len(views) != len(files):
        raise HTTPException(400, "Every photo needs an intended view.")
    payloads = []
    for f, v in zip(files, views):
        if v not in VIEWS:
            raise HTTPException(400, f"Unknown view {v!r}.")
        data = await f.read()
        if len(data) > MAX_BYTES:
            raise HTTPException(400, f"{f.filename} is larger than 25 MB.")
        payloads.append({"bytes": data, "view": v, "id": f.filename, "name": f.filename})
    return JSONResponse(_run(payloads, use_vlm != "0", use_cache=fresh == "0"))


@app.get("/api/practice")
def practice():
    return {"sets": list(_practice_sets().values())}


@app.post("/api/practice/{set_id}")
def practice_set(set_id: str, use_vlm: str = "1", fresh: str = "0"):
    s = _practice_sets().get(set_id)
    if not s:
        raise HTTPException(404, f"No practice set {set_id}.")
    payloads = []
    for im in s["images"]:
        p = ROOT / im["path"]
        if not p.exists():
            continue
        payloads.append({"bytes": p.read_bytes(), "view": im["view"],
                         "id": im["image_id"], "name": p.name})
    out = _run(payloads, use_vlm != "0", use_cache=fresh == "0")
    out["set_id"] = set_id
    out["device_id"] = s["device_id"]
    return JSONResponse(out)


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
