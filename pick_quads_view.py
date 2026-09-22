"""
pick_quads_view.py -- FABRIC surface geometry authoring tool.

Marks the flat-ish surfaces of a bedding piece in a base render and records
the REAL-WORLD SIZE of each surface. render_v2.py uses this to tile the
pattern in fabric space and project it through perspective, so motifs
converge with distance instead of tiling uniformly in image space.

Run once per view. The result is stored beside the view's other assets and
never needs redoing for that view.

USAGE
    python pick_quads_view.py headon
    python pick_quads_view.py headon --templates templates

CONTROLS
    left click   place a corner
    u            undo last corner
    n            finish this quad, prompt for its size, start the next
    s            save and quit
    q            quit without saving

CORNER ORDER
    Click TL, TR, BR, BL as they appear ON THE FABRIC, not on screen.
    For the front drop of a comforter, "top" is the mattress break.

SUGGESTED QUADS -- headon comforter
    quad 1  top surface   pillows down to the mattress break
    quad 2  front drop    mattress break down to the hem
    They should OVERLAP by 10-20px or a seam line shows through.

SIZES
    After each quad you are asked for its width and height in inches --
    the real size of that piece of cloth, not the pixel size. For a 90x90
    comforter the top surface is about 90 wide by the mattress depth plus
    the top run; the front drop is about 90 wide by the drop length.
    Estimates are fine. They set the scale, and you can re-run to adjust.
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

MAX_DISPLAY = 1100
COLORS = [(0, 255, 255), (255, 128, 0), (0, 255, 0), (255, 0, 255)]


def load_base(folder):
    for ext in (".png", ".jpg", ".jpeg"):
        path = os.path.join(folder, "base" + ext)
        if os.path.exists(path):
            img = cv2.imread(path, 1)
            if img is not None:
                return img, path
    return None, None


def ask_float(prompt, default):
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if raw == "":
            return float(default)
        try:
            val = float(raw)
            if val > 0:
                return val
            print("  must be greater than zero")
        except ValueError:
            print("  numbers only")


def ask_piece(default="comforter"):
    raw = input(f"which piece is this quad part of? [{default}]: ").strip()
    return raw if raw else default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("view", help="view folder name, e.g. headon")
    ap.add_argument("--templates", default="templates")
    args = ap.parse_args()

    folder = os.path.join(args.templates, args.view)
    if not os.path.isdir(folder):
        print(f"no such folder: {folder}")
        sys.exit(1)

    img, base_path = load_base(folder)
    if img is None:
        print(f"no base image found in {folder}")
        sys.exit(1)

    print(f"base: {base_path}")
    h, w = img.shape[:2]
    scale = min(1.0, MAX_DISPLAY / max(h, w))
    disp_base = cv2.resize(img, None, fx=scale, fy=scale) if scale < 1.0 else img.copy()

    quads = []
    current = []

    def redraw():
        canvas = disp_base.copy()
        for i, q in enumerate(quads):
            pts = (np.array(q["corners"]) * scale).astype(np.int32)
            col = COLORS[i % len(COLORS)]
            cv2.polylines(canvas, [pts], True, col, 2)
            label = f'{q["piece"]} {q["width_inches"]:.0f}x{q["height_inches"]:.0f}"'
            cv2.putText(canvas, label, tuple(pts[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

        col = COLORS[len(quads) % len(COLORS)]
        for j, p in enumerate(current):
            cv2.circle(canvas, p, 5, col, -1)
            cv2.putText(canvas, str(j + 1), (p[0] + 8, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        if len(current) > 1:
            cv2.polylines(canvas, [np.array(current, np.int32)], False, col, 1)

        msg = (f"{args.view}  quad {len(quads) + 1}  corner {len(current) + 1}/4"
               f"   [u]ndo [n]ext [s]ave [q]uit")
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(canvas, msg, (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("pick quads", canvas)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(current) < 4:
            current.append((x, y))
            redraw()

    def commit_current():
        corners = [[int(px / scale), int(py / scale)] for px, py in current]
        current.clear()
        cv2.destroyWindow("pick quads")
        cv2.waitKey(1)
        print(f"\nquad {len(quads) + 1} corners captured")
        piece = ask_piece()
        width_in = ask_float("  width in inches", 90)
        height_in = ask_float("  height in inches", 30)
        quads.append({
            "piece": piece,
            "corners": corners,
            "width_inches": width_in,
            "height_inches": height_in,
        })
        cv2.namedWindow("pick quads", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("pick quads", on_mouse)
        redraw()

    cv2.namedWindow("pick quads", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("pick quads", on_mouse)
    redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key == ord('u') and current:
            current.pop()
            redraw()

        elif key == ord('n'):
            if len(current) != 4:
                print("need exactly 4 corners before pressing n")
            else:
                commit_current()

        elif key == ord('s'):
            if len(current) == 4:
                commit_current()
            if not quads:
                print("no quads to save")
                continue
            out_path = os.path.join(folder, "quads.json")
            with open(out_path, "w") as f:
                json.dump({
                    "view": args.view,
                    "base": os.path.basename(base_path),
                    "size": [w, h],
                    "quads": quads,
                }, f, indent=2)
            print(f"\nsaved {len(quads)} quad(s) to {out_path}")
            break

        elif key == ord('q'):
            print("quit without saving")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
