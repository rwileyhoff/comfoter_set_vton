"""
upscale_views.py -- build a higher-resolution copy of the view templates.

    python upscale_views.py                     # templates -> templates_2k, 2x
    python upscale_views.py --scale 2 --views headon quarter side

For each view it writes <dst>/<view>/ with:
  base.png          from base_upscaled.png if you made one (ComfyUI), else Lanczos
  depth.png         cubic resize + light blur (depth is soft anyway)
  mask_*.png        resized, then re-thresholded to a crisp ~2 px edge
  quads.json        copied UNCHANGED
  grid_*.json       copied UNCHANGED
  config.json       copied, plus "authored_width" = the original width

render_v2.py reads authored_width and scales the quad corners, the ppi
values and displacement by itself, so nothing is re-authored by hand.

The original templates folder is never modified.
"""

import argparse
import json
import os
import shutil

import cv2

VIEWS = ["headon", "quarter", "side", "overhead"]


def find(folder, name):
    for ext in (".png", ".jpg", ".jpeg"):
        p = os.path.join(folder, name + ext)
        if os.path.exists(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="templates")
    ap.add_argument("--dst", default="templates_2k")
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--views", nargs="*", default=VIEWS)
    args = ap.parse_args()

    if os.path.abspath(args.src) == os.path.abspath(args.dst):
        raise SystemExit("--dst must be a different folder from --src")

    for view in args.views:
        src = os.path.join(args.src, view)
        dst = os.path.join(args.dst, view)
        base_path = find(src, "base")
        if base_path is None:
            print(f"  {view}: no base image, skipped")
            continue
        os.makedirs(dst, exist_ok=True)

        base = cv2.imread(base_path, 1)
        h, w = base.shape[:2]
        W, H = int(round(w * args.scale)), int(round(h * args.scale))

        up_path = find(src, "base_upscaled")
        if up_path:
            up = cv2.imread(up_path, 1)
            if up.shape[:2] != (H, W):
                up = cv2.resize(up, (W, H), interpolation=cv2.INTER_AREA
                                if up.shape[1] > W else cv2.INTER_LANCZOS4)
            src_note = f"from {os.path.basename(up_path)}"
        else:
            up = cv2.resize(base, (W, H), interpolation=cv2.INTER_LANCZOS4)
            src_note = "Lanczos (no base_upscaled found)"
        cv2.imwrite(os.path.join(dst, "base.png"), up)

        depth_path = find(src, "depth")
        if depth_path:
            d = cv2.imread(depth_path, 0)
            d = cv2.resize(d, (W, H), interpolation=cv2.INTER_CUBIC)
            d = cv2.GaussianBlur(d, (0, 0), args.scale * 0.6)
            cv2.imwrite(os.path.join(dst, "depth.png"), d)

        for piece in ("comforter", "shams"):
            mp = find(src, f"mask_{piece}")
            if mp:
                m = cv2.imread(mp, 0)
                # The December masks already have a ~11 px soft ramp at 1024.
                # A plain resize doubles it to ~22 px, and the renderer's
                # feather widens it again -- a 25-30 px band where the print
                # is half-transparent over whatever white is underneath (the
                # fold's rim, the sheet). Re-threshold so the edge stays at the
                # same place but only ~2 px wide.
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)
                _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
                m = cv2.GaussianBlur(m, (0, 0), 0.6 * args.scale)
                cv2.imwrite(os.path.join(dst, f"mask_{piece}.png"), m)

        # quads.json and grid_*.json are copied unchanged; render_v2 scales
        # their pixel coordinates using authored_width
        for fn in os.listdir(src):
            if fn == "quads.json" or (fn.startswith("grid_") and fn.endswith(".json")):
                shutil.copyfile(os.path.join(src, fn), os.path.join(dst, fn))

        cp = os.path.join(src, "config.json")
        if os.path.exists(cp):
            with open(cp) as f:
                cfg = json.load(f)
            cfg.setdefault("authored_width", w)
            with open(os.path.join(dst, "config.json"), "w") as f:
                json.dump(cfg, f, indent=2)

        print(f"  {view}: {w}x{h} -> {W}x{H}, base {src_note}")


if __name__ == "__main__":
    main()
