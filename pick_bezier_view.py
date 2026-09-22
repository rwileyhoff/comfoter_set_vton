"""
pick_bezier_view.py -- author a piece's surface with BEZIER CURVES.

    python pick_bezier_view.py --view side --piece comforter --down 0 34 43 68 --width 70

Instead of clicking every grid point, you click only the two ENDS of each
row. Each row becomes a curve with two handles; each side (left and right)
becomes a curve through the row ends with its own handles. Between two rows
the surface is filled smoothly from its four curved edges (a Coons patch),
so it cannot zigzag.

--down    inches DOWN the cloth for each row, head of the bed first. Minimum
          two (head and hem). Add a close pair where the cloth rolls over an
          edge (34 and 43 above) -- that is where the curve comes from.
--width   inches ACROSS the cloth, left end to right end of every row.
--across  clicked points per row. Default 2 (the original tool: one curve per
          row). 4 makes each row a chain of three curves and runs a vertical
          curve down every anchor column, so the middle of a wide row can be
          pinned instead of riding one smooth arc end to end. The anchor
          columns split --width EVENLY: with 4, each span is width/3.

CLICK ORDER (placing)
    For each row, head to hem: click its points LEFT TO RIGHT as the cloth
    runs -- two of them by default, --across of them otherwise. Right click
    / u undoes the last point.

EDITING (after all ends are placed)
    drag a round dot      moves a row end (its handles come with it)
    drag a square         bends that curve
    a                     reset all handles to automatic (straight-ish)
    p                     6-inch checker preview -- THE check, same as render
    g                     show / hide the fill lines
    m                     show / hide the mask outline
    s                     save grid_<piece>.json
    r                     start over
    q, Esc                quit without saving

Saving writes the same grid_<piece>.json render_v2.py already reads, plus the
curves themselves so this tool can reopen and edit them. If the existing
file was a point grid (pick_grid_view.py), it is kept as
grid_<piece>_points.json first.

Author on the ORIGINAL templates folder, then run upscale_views.py.
"""

import argparse
import json
import os
import shutil

import cv2
import numpy as np

import render_v2 as R

WIN = "bezier picker"
SUB_ROWS = 4          # fill rows per band written to the grid file
FILL_COLS = 13        # fill columns per row segment when --across 2
SEG_COLS = 7          # fill columns per row segment when --across > 2
RENDER_DENSITY = 4    # spline steps per cell the renderer uses on this grid


# ----------------------------------------------------------------------------
# curve maths
# ----------------------------------------------------------------------------

def bez(P, t):
    """Cubic Bezier through control points P (4, 2) at parameters t (n,)."""
    t = np.asarray(t, np.float64)[:, None]
    a = (1 - t) ** 3
    b = 3 * (1 - t) ** 2 * t
    c = 3 * (1 - t) * t ** 2
    d = t ** 3
    return a * P[0] + b * P[1] + c * P[2] + d * P[3]


def bez_even(P, s):
    """Bezier sampled at EVEN SPACING along its length (s in 0..1), so the
    print does not bunch up where a handle happens to be dragged far out."""
    t = np.linspace(0, 1, 400)
    pts = bez(P, t)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] < 1e-9:
        return np.repeat(P[:1], len(s), axis=0)
    tt = np.interp(np.asarray(s) * cum[-1], cum, t)
    return bez(P, tt)


def coons(top, bottom, left, right, S, T):
    """Coons patch from four boundary curves given as callables of 0..1.
    top/bottom run left->right, left/right run top->bottom."""
    Ts, Bs = top(S), bottom(S)            # (nS, 2)
    Lt, Rt = left(T), right(T)            # (nT, 2)
    c00, c10 = top(np.array([0.0]))[0], top(np.array([1.0]))[0]
    c01, c11 = bottom(np.array([0.0]))[0], bottom(np.array([1.0]))[0]
    s = S[None, :, None]
    t = T[:, None, None]
    ruled_st = (1 - t) * Ts[None] + t * Bs[None]
    ruled_lr = (1 - s) * Lt[:, None] + s * Rt[:, None]
    bilin = ((1 - s) * (1 - t) * c00 + s * (1 - t) * c10
             + (1 - s) * t * c01 + s * t * c11)
    return ruled_st + ruled_lr - bilin     # (nT, nS, 2)


# ----------------------------------------------------------------------------
# the model: rows (curves across) and sides (curves down)
# ----------------------------------------------------------------------------

class Model:
    """
    A grid of ANCHORS: n_rows across-curves x n_cols clicked points per row.

    n_cols == 2 is the original tool: one cubic per row, side curves only at
    the far left and right. With n_cols > 2 each row becomes a CHAIN of
    cubics (one per gap between anchors) and a vertical curve runs down every
    anchor column, so the interior of a wide row can be pinned instead of
    being forced onto one smooth arc end to end.

    The anchor columns divide --width EVENLY in inches: with --across 4 the
    three spans are width/3 each, whatever the pixel distances look like.
    """

    def __init__(self, n_rows, n_cols=2):
        self.n = n_rows
        self.c = n_cols
        self.ends = []          # click order, row-major: row0 left..right, row1, ...
        self.row_h = None       # (n, c-1, 2, 2): two handles per row segment
        self.col_h = None       # (n-1, c, 2, 2): two handles per column segment

    @property
    def n_ends(self):
        return self.n * self.c

    @property
    def complete(self):
        return len(self.ends) == self.n_ends and self.row_h is not None

    def anchors(self):
        return np.array(self.ends, np.float64).reshape(self.n, self.c, 2)

    def auto_handles(self):
        A = self.anchors()

        def chain(P):
            """Handles for a chain of points P (k, 2) -> (k-1, 2, 2)."""
            k = len(P)
            m = np.zeros_like(P)
            if k >= 3:
                m[1:-1] = (P[2:] - P[:-2]) / 2.0
            m[0] = P[1] - P[0]
            m[-1] = P[-1] - P[-2]
            return np.stack([P[:-1] + m[:-1] / 3.0, P[1:] - m[1:] / 3.0], axis=1)

        self.row_h = np.stack([chain(A[i]) for i in range(self.n)], axis=0)
        self.col_h = np.stack([chain(A[:, j]) for j in range(self.c)], axis=1)

    # control polygons ------------------------------------------------------
    def row_ctrl(self, i, j=0):
        """Row i, segment j (between anchor j and j+1)."""
        A = self.anchors()
        return np.array([A[i, j], self.row_h[i, j, 0], self.row_h[i, j, 1], A[i, j + 1]])

    def col_ctrl(self, i, j):
        """Column j, segment i (between row i and i+1)."""
        A = self.anchors()
        return np.array([A[i, j], self.col_h[i, j, 0], self.col_h[i, j, 1], A[i + 1, j]])

    # everything draggable, as (kind, index...) -> position ------------------
    def handles(self):
        out = []
        for k in range(len(self.ends)):
            out.append((("end", k), np.array(self.ends[k], float)))
        if self.row_h is not None:
            for i in range(self.n):
                for j in range(self.c - 1):
                    for e in range(2):
                        out.append((("row", i, j, e), self.row_h[i, j, e]))
            for i in range(self.n - 1):
                for j in range(self.c):
                    for e in range(2):
                        out.append((("col", i, j, e), self.col_h[i, j, e]))
        return out

    def move(self, key, pos):
        pos = np.array(pos, float)
        kind = key[0]
        if kind == "end":
            k = key[1]
            old = np.array(self.ends[k], float)
            d = pos - old
            self.ends[k] = list(pos)
            if self.row_h is None:
                return
            i, j = divmod(k, self.c)
            # the two row segments meeting at this anchor
            if j > 0:
                self.row_h[i, j - 1, 1] += d
            if j < self.c - 1:
                self.row_h[i, j, 0] += d
            # the two column segments meeting at this anchor
            if i > 0:
                self.col_h[i - 1, j, 1] += d
            if i < self.n - 1:
                self.col_h[i, j, 0] += d
        elif kind == "row":
            self.row_h[key[1], key[2], key[3]] = pos
        elif kind == "col":
            self.col_h[key[1], key[2], key[3]] = pos

    # sampling --------------------------------------------------------------
    def seg_cols(self):
        """Fill columns sampled per row segment. Keeps the written grid a
        similar size whether there are one or several segments."""
        return FILL_COLS if self.c == 2 else SEG_COLS

    def lattice(self, rows_in, width):
        """Sampled points for the renderer: a regular lattice of
        (n-1)*SUB_ROWS+1 rows x ((c-1)*(seg_cols-1)+1) columns."""
        nS = self.seg_cols()
        S = np.linspace(0, 1, nS)
        span = width / (self.c - 1)

        rows_out, pts_out = [], []
        for i in range(self.n - 1):
            T = np.linspace(0, 1, SUB_ROWS + 1)
            if i > 0:
                T = T[1:]                      # shared row with the band above
            band = None
            for j in range(self.c - 1):
                top = lambda s, i=i, j=j: bez_even(self.row_ctrl(i, j), s)
                bot = lambda s, i=i, j=j: bez_even(self.row_ctrl(i + 1, j), s)
                lef = lambda t, i=i, j=j: bez_even(self.col_ctrl(i, j), t)
                rig = lambda t, i=i, j=j: bez_even(self.col_ctrl(i, j + 1), t)
                patch = coons(top, bot, lef, rig, S, T)
                if band is None:
                    band = patch
                else:
                    band = np.concatenate([band, patch[:, 1:]], axis=1)
            v = rows_in[i] + T * (rows_in[i + 1] - rows_in[i])
            rows_out.extend(v.tolist())
            pts_out.extend(band.tolist())

        cols_out = []
        for j in range(self.c - 1):
            u = j * span + S * span
            cols_out.extend(u.tolist() if j == 0 else u[1:].tolist())
        return rows_out, cols_out, pts_out

    # persistence -----------------------------------------------------------
    def to_json(self):
        return {"n_cols": self.c, "ends": self.ends,
                "row_h": self.row_h.tolist(), "col_h": self.col_h.tolist()}

    def from_json(self, d):
        self.ends = [list(map(float, e)) for e in d["ends"]]
        if "col_h" in d:
            self.c = int(d.get("n_cols", 2))
            self.row_h = np.array(d["row_h"], float)
            self.col_h = np.array(d["col_h"], float)
        else:
            # original format: 2 columns, row_h (n,2,2), left_h/right_h (n-1,2,2)
            self.c = 2
            self.row_h = np.array(d["row_h"], float)[:, None, :, :]
            left = np.array(d["left_h"], float)
            right = np.array(d["right_h"], float)
            self.col_h = np.stack([left, right], axis=1)

# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates", default="templates")
    ap.add_argument("--view", required=True)
    ap.add_argument("--piece", default="comforter")
    ap.add_argument("--down", type=float, nargs="+", required=True)
    ap.add_argument("--width", type=float, required=True)
    ap.add_argument("--across", type=int, default=2,
                    help="clicked points per row (default 2). 4 gives three "
                         "curve segments per row and a vertical curve down "
                         "every anchor column.")
    args = ap.parse_args()

    rows_in = list(args.down)
    if len(rows_in) < 2 or any(b <= a for a, b in zip(rows_in, rows_in[1:])):
        raise SystemExit("--down needs at least two increasing values, head first")
    if args.across < 2:
        raise SystemExit("--across needs at least 2")

    view_dir = os.path.join(args.templates, args.view)
    base = R.load_asset(view_dir, "base")
    if base is None:
        raise SystemExit(f"no base image in {view_dir}")
    h, w = base.shape[:2]
    out_path = os.path.join(view_dir, f"grid_{args.piece}.json")

    model = Model(len(rows_in), args.across)
    if os.path.exists(out_path):
        with open(out_path) as f:
            old = json.load(f)
        bz = old.get("bezier")
        saved_across = int(bz.get("curves", {}).get("n_cols", 2)) if bz else 2
        if bz and len(bz.get("down", [])) == len(rows_in) and saved_across == args.across:
            model.from_json(bz["curves"])
            print(f"loaded curves from {out_path} -- drag to adjust")
        elif bz:
            print(f"existing curves have {len(bz['down'])} rows x {saved_across} "
                  f"across, asked for {len(rows_in)} x {args.across}; starting fresh")
        else:
            print(f"{out_path} is a point grid; it will be kept as "
                  f"grid_{args.piece}_points.json when you save")

    mask = R.load_asset(view_dir, f"mask_{args.piece}", grayscale=True)
    contours = []
    if mask is not None:
        _, mb = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    st = {"drag": None, "preview": False, "fill": True, "mask": True,
          "cache": None, "dirty": True}

    def grid_doc():
        r_in, c_in, pts = model.lattice(rows_in, args.width)
        return {"piece": args.piece, "authored_width": w, "density": RENDER_DENSITY,
                "rows_inches": r_in, "cols_inches": c_in, "points": pts,
                "bezier": {"down": rows_in, "width": args.width,
                           "curves": model.to_json()}}

    def preview():
        if st["cache"] is not None and not st["dirty"]:
            return st["cache"]
        g = grid_doc()
        u, v, cover, folds = R.build_grid_uv(g, base.shape, density=RENDER_DENSITY)
        valid = cover > 0
        chk = (np.floor(np.nan_to_num(u) / 6.0) + np.floor(np.nan_to_num(v) / 6.0)) % 2
        over = base.copy()
        tint = np.zeros_like(base)
        tint[chk == 0] = (40, 40, 200)
        tint[chk == 1] = (230, 230, 230)
        over[valid] = cv2.addWeighted(base, 0.45, tint, 0.55, 0)[valid]
        st["cache"] = (over, folds)
        st["dirty"] = False
        return st["cache"]

    def poly(img, pts, col, th=1):
        cv2.polylines(img, [np.round(pts).astype(np.int32)], False, col, th, cv2.LINE_AA)

    def draw():
        folds = 0
        if st["preview"] and model.complete:
            img, folds = preview()
            img = img.copy()
        else:
            img = base.copy()
        if st["mask"] and contours:
            cv2.drawContours(img, contours, -1, (0, 220, 255), 1)

        if model.complete:
            if st["fill"]:
                _, _, pts = model.lattice(rows_in, args.width)
                P = np.array(pts)
                for r in range(P.shape[0]):
                    poly(img, P[r], (120, 200, 120))
                for c in range(P.shape[1]):
                    poly(img, P[:, c], (120, 200, 120))
            ss = np.linspace(0, 1, 60)
            for i in range(model.n):
                for j in range(model.c - 1):
                    poly(img, bez(model.row_ctrl(i, j), ss), (0, 170, 0), 2)
            for i in range(model.n - 1):
                for j in range(model.c):
                    poly(img, bez(model.col_ctrl(i, j), ss), (0, 170, 0), 2)
            A = model.anchors()
            for i in range(model.n):
                for j in range(model.c - 1):
                    cv2.line(img, tuple(map(int, A[i, j])),
                             tuple(map(int, model.row_h[i, j, 0])),
                             (160, 160, 160), 1, cv2.LINE_AA)
                    cv2.line(img, tuple(map(int, A[i, j + 1])),
                             tuple(map(int, model.row_h[i, j, 1])),
                             (160, 160, 160), 1, cv2.LINE_AA)
            for i in range(model.n - 1):
                for j in range(model.c):
                    cv2.line(img, tuple(map(int, A[i, j])),
                             tuple(map(int, model.col_h[i, j, 0])),
                             (160, 160, 160), 1, cv2.LINE_AA)
                    cv2.line(img, tuple(map(int, A[i + 1, j])),
                             tuple(map(int, model.col_h[i, j, 1])),
                             (160, 160, 160), 1, cv2.LINE_AA)

        for key, p in model.handles():
            hot = key == st["drag"]
            x, y = int(p[0]), int(p[1])
            if key[0] == "end":
                cv2.circle(img, (x, y), 5, (0, 0, 255) if hot else (255, 60, 0), -1)
            else:
                cv2.rectangle(img, (x - 3, y - 3), (x + 3, y + 3),
                              (0, 0, 255) if hot else (0, 140, 255), -1)

        if not model.complete:
            r, col = divmod(len(model.ends), model.c)
            if model.c == 2:
                where = "LEFT" if col == 0 else "RIGHT"
            else:
                where = f"point {col + 1}/{model.c}, left to right"
            msg = (f"row {r + 1}/{model.n} ({rows_in[r]:g}in down): click "
                   f"{where}")
        else:
            msg = "drag dots / squares, p preview, a auto handles, s save"
            if folds:
                msg += f"   FOLDS IN ~{folds} CELLS: a curve crosses itself"
        cv2.rectangle(img, (0, 0), (w, 24), (0, 0, 0), -1)
        cv2.putText(img, msg, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow(WIN, img)

    def nearest(x, y, radius=9):
        best, bd = None, radius
        for key, p in model.handles():
            d = np.hypot(p[0] - x, p[1] - y)
            if d <= bd:
                best, bd = key, d
        return best

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            k = nearest(x, y)
            if k is not None:
                st["drag"] = k
            elif not model.complete:
                model.ends.append([float(x), float(y)])
                if len(model.ends) == model.n_ends:
                    model.auto_handles()
                st["dirty"] = True
            draw()
        elif event == cv2.EVENT_MOUSEMOVE and st["drag"] is not None:
            model.move(st["drag"], (x, y))
            st["dirty"] = True
            draw()
        elif event == cv2.EVENT_LBUTTONUP and st["drag"] is not None:
            st["drag"] = None
            draw()
        elif event == cv2.EVENT_RBUTTONDOWN and not model.complete and model.ends:
            model.ends.pop()
            draw()

    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN, on_mouse)
    draw()

    while True:
        key = cv2.waitKey(30) & 0xFF
        if key in (27, ord("q")):
            print("quit without saving")
            break
        if key == ord("u") and not model.complete and model.ends:
            model.ends.pop()
        elif key == ord("r"):
            model = Model(len(rows_in), args.across)
            st["dirty"] = True
        elif key == ord("a") and len(model.ends) == model.n_ends:
            model.auto_handles()
            st["dirty"] = True
        elif key == ord("p") and model.complete:
            st["preview"] = not st["preview"]
        elif key == ord("g"):
            st["fill"] = not st["fill"]
        elif key == ord("m"):
            st["mask"] = not st["mask"]
        elif key == ord("s"):
            if not model.complete:
                print(f"not complete: place all {model.c} points on every row first")
                continue
            if os.path.exists(out_path):
                with open(out_path) as f:
                    old = json.load(f)
                if "bezier" not in old:
                    keep = os.path.join(view_dir, f"grid_{args.piece}_points.json")
                    if not os.path.exists(keep):
                        shutil.copyfile(out_path, keep)
                        print(f"point grid kept as {keep}")
            with open(out_path, "w") as f:
                json.dump(grid_doc(), f, indent=1)
            print(f"saved {out_path}")
            break
        if key != 255:
            draw()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
