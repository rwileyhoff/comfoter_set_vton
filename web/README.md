# FABRIC web shell

React + Vite front end for the warp render API (`main.py`).

Run (two windows, `fabric` conda env for the API):

    uvicorn main:app --port 8010        # repo root
    cd web; npm install; npm run dev     # then open http://localhost:5180

Or double-click `start_fabric.bat` in the repo root (after `npm install` once).

The dev server forwards `/api/*` to the API on port 8010, so no CORS setup is
needed. Set `FABRIC_API` to point it somewhere else.

Only WARP renders today. LIBRARY and MODEL (M1 Base, M2 HMC) are switches
with no GPU worker behind them yet. Render Bin frames live in memory and are
lost on reload; workspaces and the viewer setting are kept in the browser.

Right-click a frame in the Render Bin (or press the menu key / Shift+F10 on a
focused frame) for: Open in viewer, Send to render, Export (PNG / JPEG / WEBP
at Full, 1024 or Web 1200), Copy image, Export reel as ZIP (with
settings.json), Colour tag, Rename / note, Move or Duplicate to reel,
Compare A/B (wipe or side by side, with a settings diff), Info, Copy settings
and Delete. Right-click a reel's header for the reel ZIP. All of it runs in
the browser; the API is not involved. Export settings are remembered.
