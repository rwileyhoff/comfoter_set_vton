"""
main.py -- FABRIC render API.

    uvicorn main:app --port 8010

Endpoints
    GET  /health   liveness + renderer version
    GET  /rooms    rooms, their views, sizes, pieces, and which are gridded
    POST /render   multipart form; renders one or more views through the warp

POST /render form fields
    pattern              file   Fabric A repeat (PNG / JPG / WEBP). Required.
    file_inches          float  true width of that FILE in inches. Required.
    views                str    comma list, e.g. "quarter,headon". Default: all four.
    size                 int    2048 (templates_2k) or 1024 (templates). Default 2048.
    seed                 int    repeatable noise. Default: drawn and returned.
    options              str    JSON of any RenderOptions field, e.g.
                                {"fuzz":0.12,"threads":140,"ao":0.45}
    shams_pattern        file   optional: a different fabric on the shams
    shams_file_inches    float  required with shams_pattern
    comforter_pattern    file   optional: a different fabric on the comforter
    comforter_file_inches float required with comforter_pattern
    format               str    "json" (default: base64 PNGs) or "zip"

The response's "settings" block is everything needed to redo the render.
This layer holds no image code: render.py calls render_v2.py.
"""

import base64
import io
import json
import os
import zipfile
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

import render as R

MAX_UPLOAD_MB = float(os.environ.get("FABRIC_MAX_UPLOAD_MB", "40"))
MAX_PATTERN_PX = int(os.environ.get("FABRIC_MAX_PATTERN_PX", "10000"))

app = FastAPI(title="FABRIC render API", version=R.RENDERER_VERSION)

# Comma list of allowed browser origins, e.g. "http://localhost:5173".
# Unset = no cross-origin browser access.
_origins = [o.strip() for o in os.environ.get("FABRIC_ORIGINS", "").split(",") if o.strip()]
if _origins:
    app.add_middleware(CORSMiddleware, allow_origins=_origins,
                       allow_methods=["GET", "POST"], allow_headers=["*"])


def _read_image(upload: UploadFile, label: str):
    data = upload.file.read(int(MAX_UPLOAD_MB * 1024 * 1024) + 1)
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"{label} is over {MAX_UPLOAD_MB:g} MB")
    try:
        img = R.decode_image(data)
    except ValueError as exc:
        raise HTTPException(400, f"{label}: {exc}")
    if max(img.shape[:2]) > MAX_PATTERN_PX:
        raise HTTPException(400, f"{label} is {img.shape[1]}x{img.shape[0]}px; "
                                 f"max side is {MAX_PATTERN_PX}px")
    return img


def _parse_options(raw: Optional[str]) -> R.RenderOptions:
    if not raw:
        return R.RenderOptions()
    try:
        given = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(400, f"options is not valid JSON: {exc}")
    if not isinstance(given, dict):
        raise HTTPException(400, "options must be a JSON object")
    known = set(R.RenderOptions.__dataclass_fields__)
    unknown = sorted(set(given) - known)
    if unknown:
        raise HTTPException(400, f"unknown option(s) {unknown}; known: {sorted(known)}")
    try:
        return R.RenderOptions(**{k: float(v) for k, v in given.items()})
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, f"options: {exc}")


@app.get("/health")
def health():
    return {"ok": True, "renderer_version": R.RENDERER_VERSION}


@app.get("/rooms")
def rooms():
    return R.list_rooms()


@app.post("/render")
def render_endpoint(
    pattern: UploadFile = File(...),
    file_inches: float = Form(...),
    views: Optional[str] = Form(None),
    size: int = Form(2048),
    seed: Optional[int] = Form(None),
    options: Optional[str] = Form(None),
    shams_pattern: Optional[UploadFile] = File(None),
    shams_file_inches: Optional[float] = Form(None),
    comforter_pattern: Optional[UploadFile] = File(None),
    comforter_file_inches: Optional[float] = Form(None),
    format: str = Form("json"),
):
    # plain def: FastAPI runs it in a worker thread, so a CPU-heavy render
    # does not block /health or /rooms
    if file_inches <= 0:
        raise HTTPException(400, "file_inches must be > 0")
    if format not in ("json", "zip"):
        raise HTTPException(400, 'format must be "json" or "zip"')

    fabric_a = _read_image(pattern, "pattern")

    pieces = {}
    for piece, up, inches in (("shams", shams_pattern, shams_file_inches),
                              ("comforter", comforter_pattern, comforter_file_inches)):
        if up is not None and up.filename:
            if not inches or inches <= 0:
                raise HTTPException(400, f"{piece}_file_inches is required with {piece}_pattern")
            pieces[piece] = (_read_image(up, f"{piece}_pattern"), inches)

    view_list = [v.strip() for v in views.split(",") if v.strip()] if views else None

    try:
        result = R.render(fabric_a, file_inches, views=view_list,
                          options=_parse_options(options), pieces=pieces or None,
                          size=size, seed=seed)
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    if not result.views:
        raise HTTPException(422, {"error": "no view rendered", "failed": result.failed})

    if format == "zip":
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
            for name, vr in result.views.items():
                z.writestr(f"render_{name}.png", R.encode_png(vr.image))
            z.writestr("settings.json", json.dumps(
                {"settings": result.settings, "failed": result.failed}, indent=2))
        return Response(buf.getvalue(), media_type="application/zip",
                        headers={"Content-Disposition": 'attachment; filename="fabric_render.zip"'})

    return JSONResponse({
        "settings": result.settings,
        "failed": result.failed,
        "views": {
            name: {
                "mode": vr.mode,
                "seconds": vr.seconds,
                "warnings": vr.warnings,
                "width": int(vr.image.shape[1]),
                "height": int(vr.image.shape[0]),
                "png_base64": base64.b64encode(R.encode_png(vr.image)).decode("ascii"),
            }
            for name, vr in result.views.items()
        },
    })
