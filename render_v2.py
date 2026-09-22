"""
render_v2.py -- FABRIC renderer.

Lifts the working pipeline out of the December POC and adds the four things
that were making the output read as printed wallpaper rather than cloth:

  1. PERSPECTIVE CONVERGENCE
     The pattern is tiled in FABRIC SPACE at a known pixels-per-inch, then
     projected onto each surface quad. Motifs shrink with distance because
     the projection does it, not because anything guesses. Needs quads.json
     per view (see pick_quads_view.py). Falls back to the old uniform tiling
     when no quads exist, so it still runs on an un-authored view.

  2. AMBIENT OCCLUSION
     Creases get darker than a straight luminance multiply makes them. The
     valleys are found from the depth map and the base luminance, and
     multiplied in separately from the broad shading.

  3. FEATHERED MASK EDGES
     The old hard silhouette produced a cut-out edge. The mask is feathered,
     and a little noise is added along the boundary so the silhouette reads
     as fibre rather than as a vector path.

  4. WEAVE
     A procedural warp/weft structure multiplied in at low opacity, scaled to
     a real thread count. This is the cheapest single thing that stops the
     surface looking like a flat print.

USAGE
    python render_v2.py --pattern ELANARI.png --file-inches 35.8
    python render_v2.py --pattern ELANARI.png --file-inches 35.8 --views headon quarter
    python render_v2.py --pattern ELANARI.png --file-inches 35.8 --no-weave --ao 0.0

TUNING
    --file-inches   how many inches wide the pattern FILE is at true scale.
                    This is the calibration number, one per fabric. Store it
                    on the fabric record rather than retyping it.
    --ao            0 off, 1 heavy. Default 0.45.
    --weave-amount  0 off, 1 heavy. Default 0.18.
    --feather       mask edge softness in pixels. Default 2.5.
    --grain         film grain. Default 0.02, as the POC had it.
    --specular      sheen on bright folds. Default 0.10 (was 0.32, washed out
                    the print at the mattress break).
"""

import argparse
import json
import os

import cv2
import numpy as np

VIEWS = ["headon", "quarter", "side", "overhead"]
PIECES = ["comforter", "shams"]


# ----------------------------------------------------------------------------
# asset loading
# ----------------------------------------------------------------------------

def load_asset(folder, name, grayscale=False):
    for ext in (".png", ".jpg", ".jpeg"):
        path = os.path.join(folder, name + ext)
        if os.path.exists(path):
            return cv2.imread(path, 0 if grayscale else 1)
    return None


def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ----------------------------------------------------------------------------
# 1. tiling -- fabric space and image space
# ----------------------------------------------------------------------------

def tile_uniform(pattern, shape, pattern_ppi, target_ppi, angle=0):
    """The POC's original behaviour. Used when a view has no quads."""
    tgt_h, tgt_w = shape[:2]
    pattern_ppi = max(1e-6, pattern_ppi)

    if angle in (90, -90, 270, 180):
        rot = {90: cv2.ROTATE_90_CLOCKWISE,
               -90: cv2.ROTATE_90_COUNTERCLOCKWISE,
               270: cv2.ROTATE_90_COUNTERCLOCKWISE,
               180: cv2.ROTATE_180}[angle]
        pattern = cv2.rotate(pattern, rot)
        angle = 0

    if angle == 0:
        return _tile_to(pattern, tgt_w, tgt_h, pattern_ppi, target_ppi)

    diagonal = int(np.hypot(tgt_h, tgt_w))
    pad = int(diagonal * 0.2)
    big = _tile_to(pattern, diagonal + pad, diagonal + pad, pattern_ppi, target_ppi)
    center = ((diagonal + pad) // 2, (diagonal + pad) // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(big, M, (diagonal + pad, diagonal + pad),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    x0 = max(0, center[0] - tgt_w // 2)
    y0 = max(0, center[1] - tgt_h // 2)
    crop = rotated[y0:y0 + tgt_h, x0:x0 + tgt_w]
    if crop.shape[:2] != (tgt_h, tgt_w):
        crop = cv2.resize(crop, (tgt_w, tgt_h))
    return crop


def _tile_to(pattern, width, height, pattern_ppi, target_ppi):
    scale = target_ppi / max(1e-6, pattern_ppi)
    new_w = max(1, int(pattern.shape[1] * scale))
    new_h = max(1, int(pattern.shape[0] * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(pattern, (new_w, new_h), interpolation=interp)
    nx = width // new_w + 2
    ny = height // new_h + 2
    tiled = np.tile(resized, (ny, nx, 1))
    return tiled[:height, :width]


def order_quad(pts):
    """
    Corners as clicked -- TL, TR, BR, BL on the CLOTH, not on screen.

    An earlier version re-sorted by x+y and y-x. That works for a quad that
    is roughly axis-aligned and fails badly for a rotated one: it can pick
    the same corner twice, the homography collapses, and the pattern smears
    into stripes. The picker already asks for a fixed click order, so trust
    it. Only guard against a self-intersecting quad, which means two corners
    were swapped.
    """
    pts = np.array(pts, dtype=np.float32)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    signs = [np.sign(cross(pts[i], pts[(i + 1) % 4], pts[(i + 2) % 4]))
             for i in range(4)]
    if len(set(s for s in signs if s != 0)) > 1:
        # bow-tie: sort by angle around the centroid, keeping the first
        # clicked corner as the starting point
        c = pts.mean(axis=0)
        ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
        order = np.argsort(ang)
        start = int(np.where(order == 0)[0][0])
        order = np.roll(order, -start)
        pts = pts[order]

    return pts


def tile_perspective(pattern, quads, out_shape, pattern_ppi, angle=0,
                     fabric_ppi=None, seam_blend=1.5):
    """
    THE REALISM FIX.

    For each quad: build the cloth at its true size in fabric space, then
    perspective-project it onto the quad. Because the projection is a real
    homography, the far edge compresses and the motifs converge.

    The quads of one piece are treated as ONE CONTINUOUS LENGTH of cloth.
    Quad 2 starts where quad 1 ended, so the print runs over the mattress
    break the way real fabric does instead of restarting. Override per quad
    with "offset_inches" if a piece is genuinely cut there.

    Overlapping quads are cross-faded rather than overwritten, so the join
    does not show as a hard line.

    Returns (layer, coverage).
    """
    out_h, out_w = out_shape[:2]
    layer = np.zeros((out_h, out_w, 3), np.float32)
    weight = np.zeros((out_h, out_w), np.float32)
    coverage = np.zeros((out_h, out_w), np.uint8)

    running_offset = 0.0

    for q in quads:
        dst = order_quad(q["corners"])
        w_in = float(q.get("width_inches", 90))
        h_in = float(q.get("height_inches", 30))
        off_in = float(q.get("offset_inches", running_offset))
        running_offset = off_in + h_in

        # Build the flat cloth at roughly the size the quad occupies ON
        # SCREEN. warpPerspective does not area-average, so a flat cloth much
        # larger than its destination is minified without filtering and the
        # print turns into moire stripes. A small headroom factor keeps the
        # near edge sharp without inviting that.
        px_w = max(np.linalg.norm(dst[1] - dst[0]), np.linalg.norm(dst[2] - dst[3]))
        px_h = max(np.linalg.norm(dst[3] - dst[0]), np.linalg.norm(dst[2] - dst[1]))

        headroom = 1.4
        flat_w = max(8, int(round(px_w * headroom)))

        max_side = 4000
        if flat_w > max_side:
            flat_w = max_side

        # pixels per inch of cloth in that flat buffer
        fabric_ppi_eff = flat_w / max(1e-6, w_in)

        # the buffer is w_in x h_in of cloth, so its height must follow the
        # same scale or the pattern stretches along one axis
        flat_h = max(8, int(round(h_in * fabric_ppi_eff)))
        off_px = int(round(off_in * fabric_ppi_eff))

        if flat_h + off_px > max_side * 2:
            off_px = off_px % max(1, flat_h)

        # tile enough cloth to reach this quad's place along the run, then
        # take the slice that belongs to it
        tall = tile_uniform(pattern, (flat_h + off_px, flat_w), pattern_ppi,
                            fabric_ppi_eff, angle)
        flat = tall[off_px:off_px + flat_h]
        if flat.shape[0] < flat_h:
            flat = cv2.resize(flat, (flat_w, flat_h))

        src = np.array([[0, 0], [flat_w, 0], [flat_w, flat_h], [0, flat_h]],
                       dtype=np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(flat, M, (out_w, out_h),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REFLECT)

        cover = np.zeros((out_h, out_w), np.uint8)
        cv2.fillConvexPoly(cover, dst.astype(np.int32), 255)

        # OVERLAP: the LATER quad wins. The old code summed every quad and
        # divided by the count, so wherever two quads overlapped -- and the
        # picker encourages overlap so no gap shows -- the output was a 50/50
        # average of two DIFFERENT warps of the print. Two prints half-
        # transparent over each other is the ghosted, smeared band. That
        # happened even at --seam-blend 0, because the averaging did not
        # depend on the blend width. Now the new quad is laid OVER what is
        # already there; seam_blend only softens its edge by a pixel or two.
        a = cover.astype(np.float32) / 255.0
        if seam_blend > 0:
            a = cv2.GaussianBlur(a, (0, 0), seam_blend)
            a[cover == 0] = 0.0
        a = np.where(weight > 0, a, (cover > 0).astype(np.float32))[:, :, None]

        layer = layer * (1.0 - a) + warped.astype(np.float32) * a
        weight = np.maximum(weight, (cover > 0).astype(np.float32))
        coverage = np.maximum(coverage, cover)

    layer = np.clip(layer, 0, 255).astype(np.uint8)

    return layer, coverage



# ----------------------------------------------------------------------------
# 1b. GRID MAP -- one continuous surface per piece (replaces quads)
# ----------------------------------------------------------------------------
#
# grid_<piece>.json (written by pick_grid_view.py) holds a rows x cols lattice
# of clicked screen points, plus how far DOWN the cloth each row is and how
# far ACROSS each column is, in inches. A smooth surface is fitted through the
# points and turned into a per-pixel map: screen pixel -> fabric inches. The
# print is then looked up at those inches. There is no hinge, so no seam; the
# print compresses where the rows crowd together (the roll over the mattress
# edge) and spreads where they open up, which is what reads as a curve.
#
# This map is the same thing a Blender UV pass is. If the 3D route happens
# later, a rendered UV pass replaces the clicked grid and nothing else moves.

def _catmull_rom(P, n):
    """Resample points along axis 0 with a Catmull-Rom spline, n steps per
    segment. P is (N, ..., 2). Returns (dense, t) with t in [0, N-1]."""
    N = P.shape[0]
    if N < 2:
        return P.copy(), np.zeros(1)
    ext = np.concatenate([2 * P[:1] - P[1:2], P, 2 * P[-1:] - P[-2:-1]], axis=0)
    t = np.linspace(0.0, N - 1.0, (N - 1) * n + 1)
    i = np.clip(np.floor(t).astype(int), 0, N - 2)
    s = (t - i).reshape((-1,) + (1,) * (P.ndim - 1))
    p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
    s2, s3 = s * s, s * s * s
    dense = 0.5 * ((2 * p1) + (-p0 + p2) * s
                   + (2 * p0 - 5 * p1 + 4 * p2 - p3) * s2
                   + (-p0 + 3 * p1 - 3 * p2 + p3) * s3)
    return dense, t


def build_grid_uv(grid, out_shape, px=1.0, density=12):
    """
    Returns (u_map, v_map, coverage, folds).
    u = inches ACROSS the cloth, v = inches DOWN the cloth, NaN where the grid
    does not reach. folds = number of cells whose triangles are flipped,
    which means points were clicked out of order.
    """
    out_h, out_w = out_shape[:2]
    pts = np.array(grid["points"], dtype=np.float64) * px     # (R, C, 2)
    rows_in = np.array(grid["rows_inches"], dtype=np.float64)
    cols_in = np.array(grid["cols_inches"], dtype=np.float64)
    R, C = pts.shape[:2]

    # smooth along each row (across), then along each dense column (down)
    across, tc = _catmull_rom(np.transpose(pts, (1, 0, 2)), density)   # (Cd, R, 2)
    across = np.transpose(across, (1, 0, 2))                           # (R, Cd, 2)
    dense, tr = _catmull_rom(across, density)                          # (Rd, Cd, 2)
    U = np.interp(tc, np.arange(C), cols_in)
    V = np.interp(tr, np.arange(R), rows_in)

    u_map = np.full((out_h, out_w), np.nan, np.float32)
    v_map = np.full((out_h, out_w), np.nan, np.float32)

    Rd, Cd = dense.shape[:2]
    signs = []
    tris = []
    for i in range(Rd - 1):
        for j in range(Cd - 1):
            a, b, c, d = (i, j), (i, j + 1), (i + 1, j + 1), (i + 1, j)
            tris.append((a, b, c))
            tris.append((a, c, d))

    for (p, q, r) in tris:
        P = np.array([dense[p], dense[q], dense[r]])
        UV = np.array([[U[p[1]], V[p[0]]], [U[q[1]], V[q[0]]], [U[r[1]], V[r[0]]]])
        x0, y0 = np.floor(P.min(axis=0)).astype(int)
        x1, y1 = np.ceil(P.max(axis=0)).astype(int)
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, out_w - 1), min(y1, out_h - 1)
        det = (P[1, 1] - P[2, 1]) * (P[0, 0] - P[2, 0]) + (P[2, 0] - P[1, 0]) * (P[0, 1] - P[2, 1])
        if abs(det) < 1e-9:
            continue
        signs.append(np.sign(det))
        if x1 < x0 or y1 < y0:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        l0 = ((P[1, 1] - P[2, 1]) * (xs - P[2, 0]) + (P[2, 0] - P[1, 0]) * (ys - P[2, 1])) / det
        l1 = ((P[2, 1] - P[0, 1]) * (xs - P[2, 0]) + (P[0, 0] - P[2, 0]) * (ys - P[2, 1])) / det
        l2 = 1.0 - l0 - l1
        inside = (l0 >= -1e-6) & (l1 >= -1e-6) & (l2 >= -1e-6)
        if not inside.any():
            continue
        uu = l0 * UV[0, 0] + l1 * UV[1, 0] + l2 * UV[2, 0]
        vv = l0 * UV[0, 1] + l1 * UV[1, 1] + l2 * UV[2, 1]
        yy, xx = ys[inside], xs[inside]
        u_map[yy, xx] = uu[inside]
        v_map[yy, xx] = vv[inside]

    signs = np.array(signs)
    majority = np.sign(signs.sum()) if signs.size else 1
    folds = int(np.sum(signs == -majority) // 2)

    coverage = np.where(np.isnan(u_map), 0, 255).astype(np.uint8)
    return u_map, v_map, coverage, folds


def sample_pattern_uv(pattern, u_map, v_map, pattern_ppi, taps=4,
                      steps_per_octave=4, bias=-1.0, max_octaves=5):
    """
    Look the print up at (u, v) inches, tiling it.

    Filtering matters because the print is shrunk 5-15x on screen. Tuned
    2026-09-20 against a 24x-supersampled reference on three surfaces (a
    receding top, a 3x-squashed roll, a flat drop). The first version kept
    ~40% of the reference detail -- the mushy, smeared look. This one lands
    within ~5% of the reference detail and cut the error roughly in half:
      - quarter-octave pyramid (not powers of two) so the chosen level is
        close to the real scale and bilinear adds little extra blur
      - level taken one octave sharper than the raw footprint (the pyramid
        and bilinear already blur about that much)
      - 4 samples along the most-squashed direction, so the roll and the
        receding top stay sharp across the bed while not aliasing down it
    """
    H, W = pattern.shape[:2]
    valid = ~np.isnan(u_map)
    u = np.where(valid, u_map, 0.0).astype(np.float32)
    v = np.where(valid, v_map, 0.0).astype(np.float32)
    X = u * pattern_ppi
    Y = v * pattern_ppi

    dXdy, dXdx = np.gradient(X)
    dYdy, dYdx = np.gradient(Y)
    ax_len = np.hypot(dXdx, dYdx)
    ay_len = np.hypot(dXdy, dYdy)
    edge = (~valid | ~np.roll(valid, 1, 0) | ~np.roll(valid, 1, 1)
            | ~np.roll(valid, -1, 0) | ~np.roll(valid, -1, 1))
    inner = valid & ~edge
    if inner.any():
        for arr in (dXdx, dYdx, dXdy, dYdy, ax_len, ay_len):
            arr[edge] = np.median(arr[inner])

    use_x = ax_len >= ay_len
    major = np.where(use_x, ax_len, ay_len)
    minor = np.where(use_x, ay_len, ax_len)
    mvx = np.where(use_x, dXdx, dXdy)
    mvy = np.where(use_x, dYdx, dYdy)

    fp = np.maximum(minor, major / taps)
    L = np.clip(np.log2(np.maximum(fp, 1.0)) + bias, 0, max_octaves) * steps_per_octave
    lo = np.floor(L).astype(int)
    fr = (L - lo).astype(np.float32)

    out = np.zeros(u.shape + (3,), np.float32)
    total = np.zeros(u.shape, np.float32)
    offsets = (np.arange(taps) + 0.5) / taps - 0.5
    for k in range(int(max_octaves * steps_per_octave) + 1):
        w_k = np.where(lo == k, 1.0 - fr, 0.0) + np.where(lo + 1 == k, fr, 0.0)
        if not (w_k > 1e-4).any():
            continue
        sc = 2.0 ** (-k / steps_per_octave)
        img = pattern if k == 0 else cv2.resize(
            pattern, (max(1, round(W * sc)), max(1, round(H * sc))),
            interpolation=cv2.INTER_AREA)
        lh, lw = img.shape[:2]
        sx, sy = lw / W, lh / H
        acc = np.zeros(u.shape + (3,), np.float32)
        for t in offsets:
            mx = np.mod((X + t * mvx) * sx, lw).astype(np.float32)
            my = np.mod((Y + t * mvy) * sy, lh).astype(np.float32)
            acc += cv2.remap(img, mx, my, interpolation=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_WRAP).astype(np.float32)
        out += (acc / taps) * w_k[:, :, None].astype(np.float32)
        total += w_k.astype(np.float32)

    out = out / np.maximum(total, 1e-6)[:, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)




def grid_screen_ppi(grid, px=1.0):
    """Median on-screen pixels per inch across the grid -- used as the
    fallback scale for any mask area the grid does not cover."""
    pts = np.array(grid["points"], dtype=np.float64) * px
    cols_in = np.array(grid["cols_inches"], dtype=np.float64)
    seg = np.linalg.norm(np.diff(pts, axis=1), axis=2)
    span = np.abs(np.diff(cols_in))[None, :]
    return float(np.median(seg / np.maximum(span, 1e-6)))

# ----------------------------------------------------------------------------
# displacement -- unchanged from the POC, it works
# ----------------------------------------------------------------------------

def displacement_warp(layer, depth_map, base_gray, strength=30, max_px=6.0, px=1.0):
    """
    The POC's displacement, with two guards added 2026-09-20.

    The raw push is Sobel(gradient)/255 * strength. On a hard luminance edge
    in the base render (the mattress break, the lip of a turned-back fold) a
    3x3 Sobel reaches ~1020, so at strength 15 the print was being dragged
    ~35 px across the edge. Neighbouring rows then sampled the same source
    pixels, which is the smeared, ghosted streak along the break.

    1. The gradient field is blurred a little, so a one-pixel edge becomes a
       gentle ramp instead of a spike.
    2. The push is soft-clipped with tanh to max_px. Gentle folds (small
       pushes) are untouched because tanh is linear near zero; only the
       spikes are capped.
    """
    if depth_map is None:
        return layer
    rows, cols = layer.shape[:2]

    depth = cv2.resize(depth_map, (cols, rows)).astype(np.float32)
    if base_gray is None:
        detail = np.zeros((rows, cols), np.float32)
    else:
        detail = 255.0 - cv2.resize(base_gray, (cols, rows)).astype(np.float32)

    gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3) * 0.4 \
        + cv2.Sobel(detail, cv2.CV_32F, 1, 0, ksize=3) * 0.6
    gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3) * 0.4 \
        + cv2.Sobel(detail, cv2.CV_32F, 0, 1, ksize=3) * 0.6

    sigma = 1.5 * px
    gx = cv2.GaussianBlur(gx, (0, 0), sigma)
    gy = cv2.GaussianBlur(gy, (0, 0), sigma)

    dx = (gx / 255.0) * strength
    dy = (gy / 255.0) * strength
    cap = max(0.5, max_px * px)
    mag = np.sqrt(dx * dx + dy * dy)
    k = np.where(mag > 1e-6, cap * np.tanh(mag / cap) / np.maximum(mag, 1e-6), 1.0)
    dx *= k
    dy *= k

    xg, yg = np.meshgrid(np.arange(cols), np.arange(rows))
    map_x = (xg + dx).astype(np.float32)
    map_y = (yg + dy).astype(np.float32)

    return cv2.remap(layer, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)


# ----------------------------------------------------------------------------
# 2. ambient occlusion
# ----------------------------------------------------------------------------

def occlusion_map(base_gray, depth_map, amount=0.45, px=1.0):
    """
    Find the valleys and darken them.

    A crease is a place darker than its surroundings, so blur minus original
    is positive in a valley and negative on a ridge. Both the base luminance
    and the depth map vote. Two blur radii so a deep fold and a small
    wrinkle both register.
    """
    if amount <= 0:
        return None

    g = base_gray.astype(np.float32) / 255.0
    valleys = np.zeros_like(g)

    for radius in (9, 31):
        blurred = cv2.GaussianBlur(g, (0, 0), radius * px)
        valleys += np.clip(blurred - g, 0, None)

    if depth_map is not None:
        d = cv2.resize(depth_map, (g.shape[1], g.shape[0])).astype(np.float32) / 255.0
        for radius in (9, 31):
            blurred = cv2.GaussianBlur(d, (0, 0), radius * px)
            valleys += np.clip(blurred - d, 0, None) * 0.75

    hi = np.percentile(valleys, 99)
    if hi <= 1e-6:
        return None
    valleys = np.clip(valleys / hi, 0, 1)
    valleys = np.power(valleys, 0.7)

    return np.clip(1.0 - valleys * amount, 0.25, 1.0)


# ----------------------------------------------------------------------------
# 4. weave
# ----------------------------------------------------------------------------

def weave_map(shape, ppi, threads_per_inch=90.0, amount=0.18, px=1.0):
    """
    Procedural warp and weft, BAND-LIMITED.

    The thread period on screen is ppi / threads * 2 pixels. At full-bed
    framing that is well under 1 px (8.4 ppi at 140 threads = 0.12 px), so
    real threads cannot be drawn. The old version clamped the period to
    2 px, which put the sinusoids exactly at Nyquist: sin(pi * x) is zero
    at every integer x, so what survived was float32 rounding error, which
    grows with x. Normalising by its max blew that error up into a
    column/row grid that got stronger toward the bottom-right -- the
    graph-paper look.

    Now the warp/weft term fades to zero as the period drops below ~4 px and
    only the slub (yarn irregularity) remains. On a close-up crop where the
    threads are genuinely resolvable, the weave comes back on its own.
    """
    if amount <= 0:
        return None

    h, w = shape[:2]
    period = ppi / max(1.0, threads_per_inch) * 2.0

    # 0 at <= 2 px (unresolvable), 1 at >= 4 px (clean)
    resolvable = float(np.clip((period - 2.0) / 2.0, 0.0, 1.0))

    fabric = np.zeros((h, w), np.float32)
    if resolvable > 0:
        x = np.arange(w, dtype=np.float64)
        y = np.arange(h, dtype=np.float64)
        warp = np.sin(2 * np.pi * x / period)[None, :]
        weft = np.sin(2 * np.pi * y / period)[:, None]
        # over-under: where warp is up the weft dips behind it
        f = (warp * 0.5 + weft * 0.5) + (warp * weft) * 0.25
        fabric = (f / (np.abs(f).max() + 1e-6)).astype(np.float32) * resolvable

    slub = cv2.GaussianBlur(
        np.random.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 1.2 * px)
    slub = slub / (np.abs(slub).max() + 1e-6)

    texture = fabric * 0.8 + slub * 0.2
    return np.clip(1.0 + texture * amount, 0.5, 1.5)


# ----------------------------------------------------------------------------
# 3. feathered compositing
# ----------------------------------------------------------------------------

def blend(base_rgb, mask, texture_rgb, ao=None, weave=None,
          brightness=1.0, feather=2.5, fuzz=0.35, grain=0.02,
          specular=0.10, px=1.0):
    if mask is None:
        return base_rgb

    h, w = base_rgb.shape[:2]
    if texture_rgb.shape[:2] != (h, w):
        texture_rgb = cv2.resize(texture_rgb, (w, h))
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    gray = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    # broad shading, normalised inside the mask as the POC did
    inside = gray[mask > 128]
    white_point = np.percentile(inside, 80) if inside.size else 1.0
    if white_point < 0.1:
        white_point = 0.5

    shading = np.clip(gray / white_point, 0, 1.0)
    shading = np.power(shading, 0.8)
    shading = shading * shading * (3 - 2 * shading)
    shading = np.clip(shading * brightness, 0, 1.0)

    if ao is not None:
        shading = np.clip(shading * ao, 0, 1.0)

    # specular, softened so it reads as sheen not plastic. This is ADDED as
    # white light on top of the print, so where the base render has a bright
    # fold (the mattress break) it washes the motifs out into a pale band.
    # Was hard-coded 0.32; now --specular, default 0.10.
    highlights = np.clip(gray - 0.72, 0, 1.0)
    highlights = np.power(highlights, 2.0) * specular
    highlights = cv2.GaussianBlur(highlights, (0, 0), 1.5 * px)

    tex = texture_rgb.astype(np.float32) / 255.0
    if weave is not None:
        tex = np.clip(tex * weave[:, :, None], 0, 1.0)

    composite = tex * shading[:, :, None] + highlights[:, :, None]
    composite += np.random.normal(0, grain, composite.shape).astype(np.float32)
    composite = np.clip(composite, 0, 1.0)

    # feathered edge with a little fibre noise along the boundary
    alpha = mask.astype(np.float32) / 255.0
    if feather > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), feather * px)
    if fuzz > 0:
        edge = cv2.GaussianBlur(np.abs(cv2.Laplacian(alpha, cv2.CV_32F)), (0, 0), 1.0 * px)
        edge = edge / (edge.max() + 1e-6)
        noise = cv2.GaussianBlur(
            np.random.normal(0, 1, alpha.shape).astype(np.float32), (0, 0), 0.8 * px)
        alpha = np.clip(alpha + edge * noise * fuzz, 0, 1)
    alpha = alpha[:, :, None]

    base = base_rgb.astype(np.float32) / 255.0
    out = composite * alpha + base * (1 - alpha)
    return np.clip(out * 255, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------
# view rendering
# ----------------------------------------------------------------------------

def render_view(view_dir, pattern_rgb, file_inches, opts, patterns_by_piece=None):
    base_bgr = load_asset(view_dir, "base")
    if base_bgr is None:
        raise FileNotFoundError(f"no base image in {view_dir}")

    base_rgb = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2RGB)
    base_gray = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2GRAY)
    depth = load_asset(view_dir, "depth", grayscale=True)

    config = load_json(os.path.join(view_dir, "config.json"))
    if config is None:
        raise FileNotFoundError(
            f"no config.json in {view_dir} -- scale would be guessed, refusing")

    # RESOLUTION SCALING. quads.json corners, the *_ppi values and
    # displacement_strength are all in PIXELS of the image they were authored
    # on. config.json "authored_width" records that width. When the base is
    # bigger (an upscaled templates_2k folder), everything pixel-based is
    # scaled by px so the same quads, config and CLI settings still hold.
    authored_w = float(config.get("authored_width", base_rgb.shape[1]))
    px = base_rgb.shape[1] / max(1.0, authored_w)
    if abs(px - 1.0) > 1e-6:
        config = dict(config)
        for k in ("comforter_ppi", "sham_ppi"):
            if k in config:
                config[k] = config[k] * px
        config["displacement_strength"] = config.get("displacement_strength", 30) * px

    quads_doc = load_json(os.path.join(view_dir, "quads.json"))
    quads_by_piece = {}
    if quads_doc:
        for q in quads_doc["quads"]:
            if abs(px - 1.0) > 1e-6:
                q = dict(q)
                q["corners"] = [[c[0] * px, c[1] * px] for c in q["corners"]]
            quads_by_piece.setdefault(q.get("piece", "comforter"), []).append(q)

    # pattern_ppi: pixels per inch of the source file at true scale
    pattern_ppi = pattern_rgb.shape[1] / max(1e-6, file_inches)

    ao = occlusion_map(base_gray, depth, opts.ao, px=px)
    result = base_rgb.copy()

    for piece in PIECES:
        mask = load_asset(view_dir, f"mask_{piece}", grayscale=True)
        if mask is None:
            continue

        pat = (patterns_by_piece or {}).get(piece, pattern_rgb)
        angle = config.get("rotation" if piece == "comforter" else "sham_rotation", 0)

        grid = load_json(os.path.join(view_dir, f"grid_{piece}.json"))

        if grid is not None:
            gpx = base_rgb.shape[1] / float(grid.get("authored_width", base_rgb.shape[1]))
            # a Bezier grid (pick_bezier_view.py) is already densely sampled
            # from its curves, so it asks for fewer spline steps per cell
            u_map, v_map, coverage, folds = build_grid_uv(
                grid, base_rgb.shape, px=gpx, density=int(grid.get("density", 12)))
            if folds:
                print(f"    WARNING {piece}: grid folds over in ~{folds} cells -- "
                      f"points clicked out of order; re-check in pick_grid_view.py")
            # A grid supplies u = inches across the cloth, v = inches down it.
            # config.json's rotation key only ever fed the flat tiler, so a
            # gridded piece ignored it -- the side view's print ran 90 degrees
            # off from quarter. Swapping the two maps turns the print on the
            # cloth without touching the clicked grid.
            if angle in (90, 270, -90, -270):
                layer = sample_pattern_uv(pat, v_map, u_map, pattern_ppi)
            else:
                layer = sample_pattern_uv(pat, u_map, v_map, pattern_ppi)

            gaps = cv2.bitwise_and(mask, cv2.bitwise_not(coverage))
            if np.any(gaps > 128):
                key = "comforter_ppi" if piece == "comforter" else "sham_ppi"
                fallback_ppi = config.get(key) or grid_screen_ppi(grid, gpx)
                filler = tile_uniform(pat, base_rgb.shape, pattern_ppi,
                                      fallback_ppi, angle)
                layer[gaps > 128] = filler[gaps > 128]

            target_ppi = None
        elif piece in quads_by_piece:
            layer, coverage = tile_perspective(
                pat, quads_by_piece[piece], base_rgb.shape, pattern_ppi, angle,
                seam_blend=getattr(opts, "seam_blend", 1.5) * px)

            # A quad only approximates the cloth, so parts of the mask fall
            # outside it -- the rolled edge, the corner that tucks under.
            # Fill those from the uniform tiler rather than leaving them
            # bare, using the scale the quads themselves imply.
            gaps = cv2.bitwise_and(mask, cv2.bitwise_not(coverage))
            if np.any(gaps > 128):
                key = "comforter_ppi" if piece == "comforter" else "sham_ppi"
                fallback_ppi = config.get(key)
                if fallback_ppi is None:
                    q0 = quads_by_piece[piece][0]
                    dst = order_quad(q0["corners"])
                    px_w = max(np.linalg.norm(dst[1] - dst[0]),
                               np.linalg.norm(dst[2] - dst[3]))
                    fallback_ppi = px_w / max(1e-6, float(q0.get("width_inches", 90)))
                filler = tile_uniform(pat, base_rgb.shape, pattern_ppi,
                                      fallback_ppi, angle)
                layer[gaps > 128] = filler[gaps > 128]

            target_ppi = None
        else:
            key = "comforter_ppi" if piece == "comforter" else "sham_ppi"
            if key not in config:
                raise KeyError(f"{view_dir}: config.json has no {key}")
            target_ppi = config[key]
            layer = tile_uniform(pat, base_rgb.shape, pattern_ppi, target_ppi, angle)

        layer = displacement_warp(layer, depth, base_gray,
                                  config.get("displacement_strength", 30),
                                  max_px=opts.disp_max, px=px)

        ppi_for_weave = target_ppi if target_ppi else config.get("comforter_ppi", 20)
        weave = weave_map(base_rgb.shape, ppi_for_weave,
                          opts.threads, opts.weave_amount, px=px)

        result = blend(result, mask, layer, ao=ao, weave=weave,
                       brightness=config.get("brightness", 1.0),
                       feather=opts.feather, fuzz=opts.fuzz, grain=opts.grain,
                       specular=opts.specular, px=px)

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", required=True)
    ap.add_argument("--file-inches", type=float, required=True,
                    help="true width of the pattern FILE in inches")
    ap.add_argument("--templates", default="templates")
    ap.add_argument("--views", nargs="*", default=VIEWS)
    ap.add_argument("--out", default="out")
    ap.add_argument("--ao", type=float, default=0.45)
    ap.add_argument("--weave-amount", type=float, default=0.18, dest="weave_amount")
    ap.add_argument("--threads", type=float, default=90.0,
                    help="threads per inch for the weave")
    ap.add_argument("--feather", type=float, default=2.5)
    ap.add_argument("--seam-blend", type=float, default=1.5, dest="seam_blend",
                    help="cross-fade width where two quads overlap; "
                         "0 for a hard join")
    ap.add_argument("--fuzz", type=float, default=0.35)
    ap.add_argument("--grain", type=float, default=0.02)
    ap.add_argument("--specular", type=float, default=0.10,
                    help="white sheen added on bright folds; was 0.32")
    ap.add_argument("--disp-max", type=float, default=6.0, dest="disp_max",
                    help="cap on how far displacement can push the print, in "
                         "pixels at the authored size; stops smear at hard edges")
    ap.add_argument("--no-weave", action="store_true")
    args = ap.parse_args()

    if args.no_weave:
        args.weave_amount = 0.0

    pattern_bgr = cv2.imread(args.pattern, 1)
    if pattern_bgr is None:
        raise SystemExit(f"could not read {args.pattern}")
    pattern_rgb = cv2.cvtColor(pattern_bgr, cv2.COLOR_BGR2RGB)

    os.makedirs(args.out, exist_ok=True)
    print(f"pattern {pattern_rgb.shape[1]}x{pattern_rgb.shape[0]}px "
          f"= {args.file_inches}in  ->  {pattern_rgb.shape[1] / args.file_inches:.2f} ppi")

    for view in args.views:
        view_dir = os.path.join(args.templates, view)
        if not os.path.isdir(view_dir):
            print(f"  {view}: no folder, skipped")
            continue
        try:
            result = render_view(view_dir, pattern_rgb, args.file_inches, args)
            path = os.path.join(args.out, f"render_{view}.png")
            cv2.imwrite(path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            if any(os.path.exists(os.path.join(view_dir, f"grid_{p}.json")) for p in PIECES):
                mode = "grid"
            elif os.path.exists(os.path.join(view_dir, "quads.json")):
                mode = "perspective"
            else:
                mode = "uniform"
            print(f"  {view}: {mode} -> {path}")
        except Exception as exc:
            print(f"  {view}: FAILED -- {exc}")


if __name__ == "__main__":
    main()
