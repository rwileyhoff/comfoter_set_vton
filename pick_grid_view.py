"""
pick_grid_view.py -- author ONE continuous grid over a piece, per view.

Replaces the two-quad setup. The print follows a smooth surface through every
point you click, so there is no hinge and no seam.

    python pick_grid_view.py --view quarter --piece comforter --down 0 12 24 36 40 52 64 76
    python pick_grid_view.py --view headon  --piece comforter --down 0 12 24 36 40 48 60

--down    inches DOWN the cloth for each row, head of the bed first.
          One row per number. Put two rows close together where the cloth
          rolls over the mattress edge (36 and 40 above) -- that is where the
          curve comes from.
--width   inches ACROSS the cloth (default 90).
--cols    points per row (default 5, evenly spaced across --width).

CLICK ORDER
    Row by row, head of the bed to the hem.
    Within a row, the cloth's LEFT edge to its RIGHT edge (as the cloth runs,
    not necessarily as the screen runs).

KEYS
    left click      place next point / drag an existing point
    right click, u  undo last point
    e               EVEN OUT: slide points 2..n-1 of every row so they are
                    evenly spaced along that row's own line. Ends (points 1
                    and last) never move, and the row keeps its curve. Press
                    it after placing the edges; it removes the column wobble
                    that bends the squares.
    p               toggle 6-inch checker preview (grid must be complete)
    m               toggle the piece's mask outline
    s               save grid_<piece>.json
    r               reset all points
    q, Esc          quit without saving

THE PREVIEW IS THE CHECK. Every square is 6x6 inches of cloth. They should
look like squares lying on the bed: smaller toward the back, squashed where
the cloth rolls over the edge, even down the drop. A square that is
stretched tall means that row is too far from the one above; squashed means
too close.

Author on the ORIGINAL templates folder (the 1024 images). upscale_views.py
copies the grid to templates_2k and render_v2 scales it.
"""

import argparse
import json
import os

import cv2
import numpy as np

import render_v2 as R

WIN = "grid picker"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates", default="templates")
    ap.add_argument("--view", required=True)
    ap.add_argument("--piece", default="comforter")
    ap.add_argument("--down", type=float, nargs="+", required=True,
                    help="inches down the cloth for each row")
    ap.add_argument("--width", type=float, default=90.0)
    ap.add_argument("--cols", type=int, default=5)
    args = ap.parse_args()

    view_dir = os.path.join(args.templates, args.view)
    base = R.load_asset(view_dir, "base")
    if base is None:
        raise SystemExit(f"no base image in {view_dir}")
    h, w = base.shape[:2]

    rows_in = list(args.down)
    if any(b <= a for a, b in zip(rows_in, rows_in[1:])):
        raise SystemExit("--down values must increase, head of the bed first")
    cols_in = list(np.linspace(0.0, args.width, args.cols))
    n_rows, n_cols = len(rows_in), len(cols_in)
    need = n_rows * n_cols

    out_path = os.path.join(view_dir, f"grid_{args.piece}.json")
    pts = []
    if os.path.exists(out_path):
        with open(out_path) as f:
            old = json.load(f)
        old_pts = np.array(old["points"], dtype=float)
        if old_pts.shape[:2] == (n_rows, n_cols):
            pts = [list(p) for p in old_pts.reshape(-1, 2)]
            print(f"loaded existing {out_path} -- drag points to adjust")
        else:
            print(f"existing grid is {old_pts.shape[0]}x{old_pts.shape[1]}, "
                  f"asked for {n_rows}x{n_cols}; starting fresh (old file kept "
                  f"until you press s)")

    mask = R.load_asset(view_dir, f"mask_{args.piece}", grayscale=True)
    contours = []
    if mask is not None:
        _, mb = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    state = {"drag": None, "preview": False, "mask": True,
             "cache": None, "dirty": True}

    def as_grid():
        return {"piece": args.piece, "authored_width": w,
                "rows_inches": rows_in, "cols_inches": cols_in,
                "points": np.array(pts, dtype=float).reshape(n_rows, n_cols, 2).tolist()}

    def preview_overlay():
        if state["cache"] is not None and not state["dirty"]:
            return state["cache"]
        u, v, cover, folds = R.build_grid_uv(as_grid(), base.shape)
        valid = cover > 0
        chk = (np.floor(np.nan_to_num(u) / 6.0) + np.floor(np.nan_to_num(v) / 6.0)) % 2
        over = base.copy()
        tint = np.zeros_like(base)
        tint[chk == 0] = (40, 40, 200)
        tint[chk == 1] = (230, 230, 230)
        over[valid] = cv2.addWeighted(base, 0.45, tint, 0.55, 0)[valid]
        state["cache"] = (over, folds)
        state["dirty"] = False
        return state["cache"]

    def draw():
        img = base.copy()
        folds = 0
        if state["preview"] and len(pts) == need:
            img, folds = preview_overlay()
            img = img.copy()
        if state["mask"] and contours:
            cv2.drawContours(img, contours, -1, (0, 220, 255), 1)

        P = pts
        for k, p in enumerate(P):
            r, c = divmod(k, n_cols)
            if c > 0:
                cv2.line(img, tuple(map(int, P[k - 1])), tuple(map(int, p)), (0, 200, 0), 1)
            if r > 0:
                cv2.line(img, tuple(map(int, P[k - n_cols])), tuple(map(int, p)), (0, 200, 0), 1)
        for k, p in enumerate(P):
            col = (0, 0, 255) if k == state["drag"] else (255, 60, 0)
            cv2.circle(img, (int(p[0]), int(p[1])), 4, col, -1)

        if len(pts) < need:
            r, c = divmod(len(pts), n_cols)
            msg = (f"row {r + 1}/{n_rows} ({rows_in[r]:g}in down)   "
                   f"point {c + 1}/{n_cols} ({cols_in[c]:g}in across)")
        else:
            msg = "complete -- p preview, drag to adjust, s save"
            if folds:
                msg += f"   FOLDS IN ~{folds} CELLS: points out of order"
        cv2.rectangle(img, (0, 0), (w, 24), (0, 0, 0), -1)
        cv2.putText(img, msg, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow(WIN, img)

    def even_out():
        if len(pts) != need or n_cols < 3:
            return
        P = np.array(pts, dtype=float).reshape(n_rows, n_cols, 2)
        for r in range(n_rows):
            row = P[r]
            for _ in range(3):                       # settle: the path moves as points move
                seg = np.linalg.norm(np.diff(row, axis=0), axis=1)
                cum = np.concatenate([[0.0], np.cumsum(seg)])
                if cum[-1] <= 1e-6:
                    break
                targets = np.linspace(0.0, cum[-1], n_cols)
                new = row.copy()
                for c in range(1, n_cols - 1):
                    k = min(np.searchsorted(cum, targets[c]) - 1, n_cols - 2)
                    k = max(k, 0)
                    f = (targets[c] - cum[k]) / max(seg[k], 1e-6)
                    new[c] = row[k] + f * (row[k + 1] - row[k])
                row = new
            P[r] = row
        pts[:] = [list(p) for p in P.reshape(-1, 2)]
        state["dirty"] = True

    def nearest(x, y, radius=10):
        if not pts:
            return None
        d = np.hypot(np.array(pts)[:, 0] - x, np.array(pts)[:, 1] - y)
        i = int(np.argmin(d))
        return i if d[i] <= radius else None

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            i = nearest(x, y)
            if i is not None:
                state["drag"] = i
            elif len(pts) < need:
                pts.append([float(x), float(y)])
                state["dirty"] = True
            draw()
        elif event == cv2.EVENT_MOUSEMOVE and state["drag"] is not None:
            pts[state["drag"]] = [float(x), float(y)]
            state["dirty"] = True
            draw()
        elif event == cv2.EVENT_LBUTTONUP and state["drag"] is not None:
            state["drag"] = None
            draw()
        elif event == cv2.EVENT_RBUTTONDOWN and pts:
            pts.pop()
            state["dirty"] = True
            draw()

    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN, on_mouse)
    draw()

    while True:
        key = cv2.waitKey(30) & 0xFF
        if key in (27, ord("q")):
            print("quit without saving")
            break
        if key == ord("u") and pts:
            pts.pop()
            state["dirty"] = True
            draw()
        elif key == ord("r"):
            pts.clear()
            state["dirty"] = True
            draw()
        elif key == ord("e"):
            even_out()
            draw()
        elif key == ord("m"):
            state["mask"] = not state["mask"]
            draw()
        elif key == ord("p"):
            if len(pts) == need:
                state["preview"] = not state["preview"]
            draw()
        elif key == ord("s"):
            if len(pts) != need:
                print(f"not complete: {len(pts)}/{need} points")
                continue
            with open(out_path, "w") as f:
                json.dump(as_grid(), f, indent=2)
            print(f"saved {out_path}")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
