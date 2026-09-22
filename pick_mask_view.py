"""
pick_mask_view.py -- redraw a piece's mask by clicking around it.

    python pick_mask_view.py --view quarter --piece shams

Click around the FIRST pillow, press n, click around the SECOND pillow,
press s. Each outline is filled as one shape.

KEYS
    left click      add a point to the current outline
    right click, u  undo last point
    n               close this outline, start the next one
    c               toggle showing the OLD mask (cyan) for comparison
    s               save
    q, Esc          quit without saving

On save the old file is kept as mask_<piece>_old.png, so nothing is lost.
Author on the ORIGINAL templates folder, then run upscale_views.py.

Trace the pillow's own edge. Where a pillow sits behind the comforter, follow
where you can SEE the pillow stop -- the comforter mask handles the rest.
"""

import argparse
import os
import shutil

import cv2
import numpy as np

import render_v2 as R

WIN = "mask picker"


def build_mask(shape, polys):
    m = np.zeros(shape[:2], np.uint8)
    for p in polys:
        if len(p) >= 3:
            cv2.fillPoly(m, [np.round(np.array(p)).astype(np.int32)], 255, cv2.LINE_AA)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates", default="templates")
    ap.add_argument("--view", required=True)
    ap.add_argument("--piece", default="shams")
    args = ap.parse_args()

    view_dir = os.path.join(args.templates, args.view)
    base = R.load_asset(view_dir, "base")
    if base is None:
        raise SystemExit(f"no base image in {view_dir}")
    old = R.load_asset(view_dir, f"mask_{args.piece}", grayscale=True)
    h, w = base.shape[:2]

    polys = [[]]
    state = {"old": True}

    def draw():
        img = base.copy()
        if state["old"] and old is not None:
            tint = img.copy()
            tint[old > 128] = (255, 255, 0)
            img = cv2.addWeighted(img, 0.65, tint, 0.35, 0)
        m = build_mask(base.shape, polys)
        red = img.copy()
        red[m > 0] = (0, 0, 255)
        img = cv2.addWeighted(img, 0.6, red, 0.4, 0)
        for p in polys:
            for i, q in enumerate(p):
                cv2.circle(img, (int(q[0]), int(q[1])), 3, (255, 60, 0), -1)
                if i > 0:
                    cv2.line(img, tuple(map(int, p[i - 1])), tuple(map(int, q)), (0, 200, 0), 1)
        msg = (f"outline {len(polys)}: {len(polys[-1])} points   "
               f"n next outline, s save, c old mask")
        cv2.rectangle(img, (0, 0), (w, 24), (0, 0, 0), -1)
        cv2.putText(img, msg, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow(WIN, img)

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            polys[-1].append([float(x), float(y)])
            draw()
        elif event == cv2.EVENT_RBUTTONDOWN and polys[-1]:
            polys[-1].pop()
            draw()

    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN, on_mouse)
    draw()

    while True:
        key = cv2.waitKey(30) & 0xFF
        if key in (27, ord("q")):
            print("quit without saving")
            break
        if key == ord("u") and polys[-1]:
            polys[-1].pop()
        elif key == ord("n") and len(polys[-1]) >= 3:
            polys.append([])
        elif key == ord("c"):
            state["old"] = not state["old"]
        elif key == ord("s"):
            done = [p for p in polys if len(p) >= 3]
            if not done:
                print("nothing to save -- click at least 3 points")
                continue
            m = build_mask(base.shape, done)
            out = os.path.join(view_dir, f"mask_{args.piece}.png")
            if os.path.exists(out):
                backup = os.path.join(view_dir, f"mask_{args.piece}_old.png")
                if not os.path.exists(backup):
                    shutil.copyfile(out, backup)
                    print(f"old mask kept as {backup}")
            cv2.imwrite(out, m)
            print(f"saved {out} ({len(done)} outlines)")
            break
        draw()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
