"""
Comforter Set Quad Views -- Streamlit front end.

This is a THIN SHELL over render_v2.py. It owns no image code.

Everything the renderer does -- grid/Bezier surfaces, perspective quads,
uniform tiling, authored_width resolution scaling, displacement, ambient
occlusion, weave, feather, fuzz -- lives in render_v2.py and is called
through render_v2.render_view(). If a render looks wrong, fix render_v2.py,
not this file.

The old private copies of tile_pattern / generate_angled_texture /
displacement_warp / blend_texture were deleted. They were the pre-grid POC
path and produced the flat two-quad look with no AO and no weave.
"""

import io
import json
import os
from contextlib import redirect_stdout
from types import SimpleNamespace

import cv2
import numpy as np
import streamlit as st
from PIL import Image

import render_v2

VIEWS = render_v2.VIEWS          # headon, quarter, side, overhead
PIECES = render_v2.PIECES        # comforter, shams

# Render settings, per view.
#
# DEFAULT_OPTS is the quarter set measured good on 2026-09-20 against
# ELANARI S1-01: the Bezier surface renders within 1.5% of the approved
# comforter_ppi of 8.41 with these numbers.
#
#   ao 0.45          back to the renderer default. The 0.30 used earlier was
#                    a workaround for grey stains on the point grid; the
#                    Bezier surface carries full shading. If stains come
#                    back on a view, try 0.40 for that view.
#   threads 140      thread count. Below ~4px per thread the weave fades out
#                    on its own, so this is safe at 2K.
#   weave_amount .12
#   fuzz 0.12        mask edge noise.
#   feather 1        mask edge softness. Anything higher ghosts the print
#                    over the white sheet rim at 2K.
#   seam_blend       ignored for any piece that has a grid.
#
# Per-view entries override individual keys. Views not listed inherit
# DEFAULT_OPTS untouched.
DEFAULT_OPTS = dict(
    ao=0.45,
    weave_amount=0.12,
    threads=140.0,
    feather=1.0,
    seam_blend=1.5,
    fuzz=0.12,
    grain=0.02,
    specular=0.10,
    disp_max=6.0,
)

VIEW_OPTS = {
    # quarter: verified 2026-09-20, Bezier surface, --down 0 34 51 68 --width 70
    "quarter": {},
    # headon: still the 5x5 POINT grid, front face only. These numbers are
    # inherited from quarter and have NOT been verified on headon.
    "headon": {},
    "side": {},
    # overhead: raised 2026-09-22. At the shared 0.45 the foot corners and the
    # fold under the cuff had almost no shading holding them apart -- looking
    # straight down, there is no silhouette doing that work, so AO carries it
    # alone. Overhead's depth.png is a soft blurred greyscale rather than a
    # real depth pass, so past ~0.85 this deepens blobs instead of folds; if
    # it goes smudgy rather than sharper, regenerate that depth map.
    "overhead": {"ao": 0.70},
}


def opts_for(view):
    o = dict(DEFAULT_OPTS)
    o.update(VIEW_OPTS.get(view, {}))
    return SimpleNamespace(**o)


def view_mode(view_dir):
    """What the renderer will actually do for this view, and with what."""
    gridded = [p for p in PIECES
               if os.path.exists(os.path.join(view_dir, f"grid_{p}.json"))]
    if gridded:
        kinds = []
        for p in gridded:
            doc = render_v2.load_json(os.path.join(view_dir, f"grid_{p}.json"))
            kinds.append(f"{p}: {'bezier' if doc and 'bezier' in doc else 'point grid'}")
        return "grid", "; ".join(kinds)
    if os.path.exists(os.path.join(view_dir, "quads.json")):
        return "perspective", "quads.json -- no grid, flat quads"
    return "uniform", "no grid, no quads -- flat tiling at config ppi"


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------

st.set_page_config(page_title="Bedding V-TON", layout="wide")
st.title("Comforter Set Quad Views")

with st.sidebar:
    st.header("1. Pattern")
    uploaded_file = st.file_uploader("Swatch image", type=["jpg", "jpeg", "png"])

    st.header("2. Scale")
    file_inches = st.number_input(
        "Swatch real width (inches)",
        min_value=1.0, value=35.8, step=0.1,
        help="The true width of the WHOLE swatch file, not one motif. "
             "ELANARI S1-01 is 35.8.",
    )

    st.header("3. Templates")
    templates_root = st.selectbox(
        "Template folder", ["templates_2k", "templates"], index=0,
        help="templates_2k is the 2048px render set built by upscale_views.py. "
             "templates is the 1024px AUTHORING set -- rendering from it gives "
             "a 1024 output.",
    )
    if templates_root == "templates":
        st.warning("Authoring folder selected. Output will be 1024px.")

    st.caption(
        "After any picker save, run `python upscale_views.py --views <view>` "
        "before rendering, or the render uses the old grid."
    )

if uploaded_file is None:
    st.info("Upload a swatch in the sidebar to begin.")
    st.stop()

file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
pattern_bgr = cv2.imdecode(file_bytes, 1)
if pattern_bgr is None:
    st.sidebar.error("Could not decode that image.")
    st.stop()

pattern_rgb = cv2.cvtColor(pattern_bgr, cv2.COLOR_BGR2RGB)
st.sidebar.image(pattern_rgb, caption="Swatch", use_container_width=True)
st.sidebar.caption(
    f"{pattern_rgb.shape[1]}x{pattern_rgb.shape[0]}px = {file_inches}in "
    f"-> {pattern_rgb.shape[1] / file_inches:.2f} ppi"
)

# Streamlit reruns this whole script on EVERY widget interaction, including a
# download click. A plain button is only True on the run that follows the
# click, so rendering inline meant a download reset the page and threw the
# renders away. Results live in session_state instead: the click renders and
# stores, every later rerun just redraws from the store.
if st.sidebar.button("Run Simulation"):
    available = [v for v in VIEWS
                 if os.path.isdir(os.path.join(templates_root, v))]
    if not available:
        st.error(f"No view folders found under {templates_root}/")
        st.stop()

    done = {}
    bar = st.progress(0.0, text="Rendering...")
    for i, view in enumerate(available):
        view_dir = os.path.join(templates_root, view)
        bar.progress(i / len(available), text=f"Rendering {view}...")
        mode, detail = view_mode(view_dir)
        entry = {"mode": mode, "detail": detail, "warnings": [],
                 "png": None, "size": None, "error": None}
        try:
            log = io.StringIO()
            with redirect_stdout(log):
                result = render_v2.render_view(
                    view_dir, pattern_rgb, file_inches, opts_for(view)
                )
            # render_v2 prints grid-fold warnings to stdout; keep them so they
            # can be shown instead of vanishing into the Streamlit console.
            entry["warnings"] = [ln.strip() for ln in log.getvalue().splitlines()
                                 if ln.strip()]
            buf = io.BytesIO()
            Image.fromarray(result).save(buf, format="PNG")
            entry["png"] = buf.getvalue()
            entry["size"] = (result.shape[1], result.shape[0])
        except Exception as exc:
            entry["error"] = str(exc)
        done[view] = entry
    bar.empty()

    st.session_state["renders"] = done
    st.session_state["rendered_with"] = {
        "templates": templates_root,
        "file_inches": file_inches,
        "swatch": uploaded_file.name,
    }

renders = st.session_state.get("renders")
if not renders:
    st.info("Press Run Simulation.")
    st.stop()

# The sidebar can be changed without re-running; say so rather than letting
# stale renders pass for current ones.
was = st.session_state.get("rendered_with", {})
now = {"templates": templates_root, "file_inches": file_inches,
       "swatch": uploaded_file.name}
if was != now:
    st.warning("Settings changed since these were rendered. "
               "Press Run Simulation to update.")

cols = st.columns(2)

for idx, (view, entry) in enumerate(renders.items()):
    with cols[idx % 2]:
        st.subheader(f"{view.title()} View")
        st.caption(f"{entry['mode']} -- {entry['detail']}")

        if entry["mode"] != "grid":
            st.warning(
                f"{view} has no grid. It will render flat -- the print will "
                f"not follow the cloth. Author one with pick_bezier_view.py."
            )
        for line in entry["warnings"]:
            st.warning(line)

        if entry["error"]:
            st.error(f"{view} failed: {entry['error']}")
            continue

        st.image(entry["png"], use_container_width=True)
        w, h = entry["size"]
        st.download_button(
            label=f"Download {view.title()} ({w}x{h})",
            data=entry["png"],
            file_name=f"render_{view}.png",
            mime="image/png",
            key=f"dl_{view}",
        )
