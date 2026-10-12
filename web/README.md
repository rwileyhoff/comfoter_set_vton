# FABRIC web shell

React + Vite front end for the warp render API (`main.py`).

Run (two windows, `fabric` conda env for the API):

    uvicorn main:app --port 8010        # repo root
    cd web; npm install; npm run dev     # then open http://localhost:5180

Or double-click `start_fabric.bat` in the repo root (after `npm install` once).

The dev server forwards `/api/*` to the API on port 8010, so no CORS setup is
needed. Set `FABRIC_API` to point it somewhere else.

Only WARP renders today. LIBRARY and MODEL (M1 Base, M2 HMC) are switches
with no GPU worker behind them yet.

FOLIO (the render bin) is saved in the browser's IndexedDB (database `fabric-bin`):
every render image, reel, tag, note and compare A survives a reload. It lives
in this browser profile on this machine only; "Clear bin" deletes it all.
Fabric uploads are not saved, so re-add Fabric A after a reload to render.

Right-click a render in FOLIO, on the main canvas, on a view tile
or on a Results / FOLIO palette thumbnail (or press the menu key / Shift+F10 on a
focused frame) for: Open in viewer, Send to render, Export (PNG / JPEG / WEBP
at Full, 1024 or Web 1200), Copy image, Export reel as ZIP (with
settings.json), Colour tag, Rename / note, Move or Duplicate to reel,
Compare A/B (wipe or side by side, with a settings diff), Info, Copy settings
and Delete. Right-click a reel's header for the reel ZIP. All of it runs in
the browser; the API is not involved. Export settings are remembered.

FOLIO desktop (Flame-style):
- Lock reels (Lock button or L): locked reels step together by position, and
  the viewer shows each locked reel's frame side by side.
- Proxy -> final: right-click a frame or reel header -> Re-render at 2048.
  Same seed and settings, swapped in place. Load the same fabric first.
- Hotkeys over a frame or reel: C collapse, L lock, G go to R#, 1 2 3 tag,
  0 untag, Del delete, Space play, arrows step.
