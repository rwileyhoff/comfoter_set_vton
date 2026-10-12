"""
render.py -- FABRIC's render service layer.

This file owns NO image code. Every pixel decision lives in render_v2.py, the
same engine the pickers, app.py and the command line already use. This layer
gives that engine a clean call for a web backend:

    render(pattern, file_inches, views=..., options=..., pieces=..., seed=...)
        -> RenderResult with one RGB array per view, plus the exact settings
           and renderer version used, so any render can be redone.

    list_rooms() -> what views exist, at what sizes, and which are gridded.

Defaults are Anthony's working settings from SESSION_STATE_FABRIC_006, NOT the
older command-line defaults in render_v2.py (fuzz 0.35, threads 90, ...).
"""

import hashlib
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import render_v2 as engine

RENDERER_VERSION = "render_v2/2026-09-22+service-1"

HERE = os.path.dirname(os.path.abspath(__file__))

# output size -> template folder. 2048 is the working path; 1024 is the
# authoring size, and what fits Render's free plan for CADENCE.
TEMPLATE_DIRS = {2048: "templates_2k", 1024: "templates"}

VIEWS = list(engine.VIEWS)      # headon, quarter, side, overhead
PIECES = list(engine.PIECES)    # comforter, shams


@dataclass
class RenderOptions:
    """Dials for the warp engine. Defaults = the working settings."""
    fuzz: float = 0.12
    threads: float = 140.0
    weave_amount: float = 0.12
    ao: float = 0.45
    feather: float = 1.0
    specular: float = 0.10
    disp_max: float = 6.0
    grain: float = 0.02
    seam_blend: float = 1.5     # only matters for a view with quads and no grid

    def clean(self) -> "RenderOptions":
        """Clamp to sane ranges so a bad request cannot wreck a render."""
        def c(v, lo, hi):
            return float(min(max(float(v), lo), hi))
        return RenderOptions(
            fuzz=c(self.fuzz, 0, 1), threads=c(self.threads, 1, 600),
            weave_amount=c(self.weave_amount, 0, 1), ao=c(self.ao, 0, 1),
            feather=c(self.feather, 0, 20), specular=c(self.specular, 0, 1),
            disp_max=c(self.disp_max, 0, 50), grain=c(self.grain, 0, 0.2),
            seam_blend=c(self.seam_blend, 0, 20))


@dataclass
class ViewRender:
    view: str
    image: np.ndarray           # RGB uint8
    mode: str                   # grid | perspective | uniform
    seconds: float
    warnings: List[str] = field(default_factory=list)


@dataclass
class RenderResult:
    views: Dict[str, ViewRender]
    failed: Dict[str, str]
    settings: dict              # everything needed to redo this render


def templates_root(size: int) -> str:
    if size not in TEMPLATE_DIRS:
        raise ValueError(f"size must be one of {sorted(TEMPLATE_DIRS)}")
    return os.path.join(HERE, TEMPLATE_DIRS[size])


def view_mode(view_dir: str) -> str:
    if any(os.path.exists(os.path.join(view_dir, f"grid_{p}.json")) for p in PIECES):
        return "grid"
    if os.path.exists(os.path.join(view_dir, "quads.json")):
        return "perspective"
    return "uniform"


def list_rooms() -> dict:
    """
    V1 has one room: the base renders in templates/ and templates_2k/.
    Shaped as a list so a second room drops in without changing callers.
    """
    views = []
    for v in VIEWS:
        sizes = [s for s in sorted(TEMPLATE_DIRS)
                 if os.path.isdir(os.path.join(templates_root(s), v))]
        if not sizes:
            continue
        vdir = os.path.join(templates_root(max(sizes)), v)
        pieces = [p for p in PIECES
                  if engine.load_asset(vdir, f"mask_{p}", grayscale=True) is not None]
        cfg = engine.load_json(os.path.join(vdir, "config.json")) or {}
        views.append({
            "view": v,
            "sizes": sizes,
            "mode": view_mode(vdir),
            "pieces": pieces,
            "gridded": [p for p in PIECES
                        if os.path.exists(os.path.join(vdir, f"grid_{p}.json"))],
            "comforter_ppi": cfg.get("comforter_ppi"),
            "sham_ppi": cfg.get("sham_ppi"),
        })
    return {"rooms": [{"id": "base-room-1", "name": "Base room 1", "views": views}],
            "renderer_version": RENDERER_VERSION}


def decode_image(data: bytes) -> np.ndarray:
    """Bytes of a PNG/JPG/WEBP -> RGB uint8. Raises ValueError if unreadable."""
    arr = np.frombuffer(data, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("could not read the image (PNG, JPG or WEBP expected)")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def encode_png(rgb: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("PNG encode failed")
    return buf.tobytes()


def _sha(rgb: np.ndarray) -> str:
    return hashlib.sha256(rgb.tobytes()).hexdigest()[:16]


class _Capture:
    """render_v2 prints grid-fold warnings; keep them for the response."""
    def __init__(self):
        self.lines = []

    def write(self, s):
        if s.strip():
            self.lines.append(s.strip())

    def flush(self):
        pass


def render(pattern: np.ndarray,
           file_inches: float,
           views: Optional[List[str]] = None,
           options: Optional[RenderOptions] = None,
           pieces: Optional[Dict[str, Tuple[np.ndarray, float]]] = None,
           size: int = 2048,
           seed: Optional[int] = None) -> RenderResult:
    """
    pattern / file_inches: Fabric A, the default for every piece.
    pieces: optional per-piece override, {"shams": (pattern_rgb, file_inches)}.
    seed: makes the noise terms (grain, slub, edge fibre) repeatable. None
          draws a fresh seed and records it, so the render can still be redone.
    """
    import contextlib

    if file_inches is None or float(file_inches) <= 0:
        raise ValueError("file_inches must be > 0")
    opts = (options or RenderOptions()).clean()
    views = list(views or VIEWS)
    bad = [v for v in views if v not in VIEWS]
    if bad:
        raise ValueError(f"unknown view(s) {bad}; expected some of {VIEWS}")
    bad = [p for p in (pieces or {}) if p not in PIECES]
    if bad:
        raise ValueError(f"unknown piece(s) {bad}; expected some of {PIECES}")
    if seed is None:
        seed = int(np.random.SeedSequence().entropy % (2 ** 31))
    root = templates_root(size)

    # render_v2.render_view reads options by attribute
    engine_opts = type("Opts", (), asdict(opts))()
    by_piece = {p: (img, float(inch)) for p, (img, inch) in (pieces or {}).items()}

    out, failed = {}, {}
    for v in views:
        vdir = os.path.join(root, v)
        if not os.path.isdir(vdir):
            failed[v] = f"no {v} folder at {size}px"
            continue
        # one RandomState per view, derived from the seed: same seed, same
        # pixels, whichever subset of views is asked for and in any order
        rng = np.random.RandomState((seed + VIEWS.index(v) * 7919) % (2 ** 32))
        cap = _Capture()
        t0 = time.perf_counter()
        try:
            with contextlib.redirect_stdout(cap):
                img = engine.render_view(vdir, pattern, float(file_inches),
                                         engine_opts, patterns_by_piece=by_piece,
                                         rng=rng)
        except Exception as exc:          # one bad view must not sink the rest
            failed[v] = str(exc)
            continue
        out[v] = ViewRender(view=v, image=img, mode=view_mode(vdir),
                            seconds=round(time.perf_counter() - t0, 2),
                            warnings=cap.lines)

    settings = {
        "renderer_version": RENDERER_VERSION,
        "engine": "warp",
        "size": size,
        "views": views,
        "seed": seed,
        "options": asdict(opts),
        "fabric_a": {"file_inches": float(file_inches),
                     "px": [int(pattern.shape[1]), int(pattern.shape[0])],
                     "sha": _sha(pattern)},
        "pieces": {p: {"file_inches": inch,
                       "px": [int(img.shape[1]), int(img.shape[0])],
                       "sha": _sha(img)}
                   for p, (img, inch) in by_piece.items()},
    }
    return RenderResult(views=out, failed=failed, settings=settings)
