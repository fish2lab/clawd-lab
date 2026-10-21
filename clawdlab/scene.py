"""The lab set: wall, floor, pipes, lamps, cryo pods and the Fable reactor.

Everything lives in one world coordinate system (y-up, floor at y=0 against
the back wall), so the Fable porthole has exactly one size no matter how the
camera frames it.
"""
import math

import numpy as np
import skia

from . import palette as P
from .gfx import (rect_path, rrect_path, circle_path, ellipse_path, poly_path, smooth_path,
                  col, mix, clamp, lerp, hash01, ease_out, smooth)
from .clawd import draw_clawd

# ------------------------------------------------------------------ layout

PODS = {
    "opus": dict(x=0.0, k=1.0, label="OPUS 5.5"),
    "sonnet": dict(x=2.95, k=0.78, label="SONNET 5.5"),
    "haiku": dict(x=5.30, k=0.56, label="HAIKU 5.5"),
}
XR = 12.8                 # reactor centre
PORT_C = (XR, 3.35)       # porthole centre
PORT_R = 2.0              # glass radius
RING_R = 2.55             # flange outer radius
FABLE_W = 4.8
MANIFOLD_Y = 3.38
CEIL = 15.0
COLUMNS = (-3.6, 8.05, 17.6)


def pod_door(k, X):
    return dict(x0=X - 0.56 * k, x1=X + 0.56 * k, y0=0.46 * k, y1=1.88 * k)


# ------------------------------------------------------------------ wall & floor

def draw_wall(g, st):
    x0 = g.cam.cx - g.view_w / 2 - 1
    x1 = g.cam.cx + g.view_w / 2 + 1
    y1 = g.cam.cy + g.cam.h / 2 + 1
    g.fill(rect_path(x0, 0, x1, y1), P.WALL)
    # big panels with alternating tone
    pw = 2.4
    i0 = int(math.floor(x0 / pw)) - 1
    i1 = int(math.ceil(x1 / pw)) + 1
    for i in range(i0, i1):
        px = i * pw
        for j, (py0, py1) in enumerate(((0.0, 2.7), (2.7, 6.2), (6.2, 10.0), (10.0, CEIL))):
            if not g.visible(px, py0, px + pw, py1, 0.1):
                continue
            if (i + j) % 2 == 0:
                g.fill(rect_path(px, py0, px + pw, py1), P.WALL_SHADE, 0.35)
            g.hatch(rect_path(px, py0, px + pw, py1), P.INK, 0.035, "vert", 1.2)
            g.stroke(rect_path(px, py0, px + pw, py1), 0.45, P.WALL_LINE)
            # corner screws
            for sx, sy in ((px + 0.1, py0 + 0.1), (px + pw - 0.1, py0 + 0.1), (px + 0.1, py1 - 0.1), (px + pw - 0.1, py1 - 0.1)):
                g.fill(circle_path(sx, sy, 0.018), P.WALL_LINE)
    # baseboard
    g.fs(rect_path(x0, -0.02, x1, 0.16), P.NAVY, 0.9)
    g.fill(rect_path(x0, 0.12, x1, 0.16), P.NAVY_LIGHT, 0.6)
    # ceiling beam
    if y1 > CEIL - 0.6:
        g.fs(rect_path(x0, CEIL - 0.45, x1, CEIL + 3), P.NAVY_DARK, 1.0)
        for i in range(int(x0 // 1.2) - 1, int(x1 // 1.2) + 2):
            xx = i * 1.2
            g.stroke(poly_path([(xx, CEIL - 0.45), (xx + 0.6, CEIL + 0.3)], close=False), 1.2, P.NAVY_LIGHT)
    # columns
    for cxx in COLUMNS:
        if g.visible(cxx - 0.4, 0, cxx + 0.4, CEIL):
            g.fs(rect_path(cxx - 0.32, 0.0, cxx + 0.32, CEIL), P.REACTOR, 1.0)
            g.fill_shader(rect_path(cxx - 0.32, 0.0, cxx + 0.32, CEIL),
                          g.lin_grad(cxx - 0.32, 0, cxx + 0.32, 0, ["#FFFFFF", P.REACTOR_SHADE], None, 0.5))
            g.hatch(rect_path(cxx + 0.05, 0.0, cxx + 0.32, CEIL), P.INK, 0.18, scale=0.7)
            g.fs(rect_path(cxx - 0.42, 0.0, cxx + 0.42, 0.35), P.NAVY, 1.0)
            for yy in np.arange(0.9, CEIL, 1.3):
                g.fill(circle_path(cxx - 0.2, yy, 0.03), P.INK_SOFT)
                g.fill(circle_path(cxx + 0.2, yy, 0.03), P.INK_SOFT)
            g.stroke(rect_path(cxx - 0.32, 0.0, cxx + 0.32, CEIL), 1.0)
    # stencils
    if g.visible(-2, 5.2, 7, 6.4):
        g.text("CRYO  BAY", 2.65, 5.55, 0.62, P.WALL_LINE, "slab", a=0.9)
        g.fill(rect_path(-1.2, 5.37, 6.5, 5.42), P.WALL_LINE, 0.9)
    if g.visible(XR - 4, 11.0, XR + 4, 12.4):
        g.text("CONTAINMENT", XR, 11.15, 0.95, P.WALL_LINE, "slab", a=0.9)
        for s in (-1, 1):
            g.fill(rect_path(XR + s * 4.0, 11.4, XR + s * 3.55 + s * 0.0, 11.5), P.WALL_LINE)


def floor_xy(X, d, cx):
    """Floor point at world-x X and depth d (toward camera) -> drawing coords."""
    return X + (X - cx) * d * 0.16, -d * (1.0 + 0.10 * d)


def draw_floor(g, st):
    cx = g.cam.cx
    x0 = cx - g.view_w / 2 - 2
    x1 = cx + g.view_w / 2 + 2
    ybot = g.cam.cy - g.cam.h / 2 - 1
    if ybot > 0:
        return
    g.fill(rect_path(x0, ybot, x1, 0.0), P.FLOOR)
    # receding tone
    g.fill_shader(rect_path(x0, ybot, x1, 0.0), g.lin_grad(0, 0, 0, -3.0, [P.FLOOR_DARK, P.FLOOR], None, 0.7))
    dmax = 6.0
    # depth lines
    d = 0.0
    k = 0
    while d < dmax:
        y = floor_xy(0, d, cx)[1]
        if y < ybot:
            break
        g.line(x0, y, x1, y, 0.5, P.FLOOR_LINE)
        k += 1
        d = 0.42 * k * (1 + 0.05 * k)
    # receding lines
    tile = 0.9
    i0 = int(math.floor((cx - g.view_w) / tile)) - 2
    i1 = int(math.ceil((cx + g.view_w) / tile)) + 2
    for i in range(i0, i1):
        X = i * tile
        a = floor_xy(X, 0.0, cx)
        b = floor_xy(X, dmax, cx)
        g.line(a[0], a[1], b[0], b[1], 0.5, P.FLOOR_LINE)
    # floor wear hatch
    g.hatch(rect_path(x0, ybot, x1, 0.0), P.INK, 0.04, "diag", 1.4)
    # safety lines
    for d0, d1 in ((0.10, 0.15), (1.55, 1.63)):
        ya, yb = floor_xy(0, d0, cx)[1], floor_xy(0, d1, cx)[1]
        g.fill(rect_path(x0, yb, x1, ya), P.SAFETY, 0.95)
    # wall contact shadow
    g.soft(rect_path(x0, -0.06, x1, 0.0), P.INK, 0.05, 0.25)


# ------------------------------------------------------------------ pipes / lamps

def pipe_h(g, x0, x1, y, r, color=P.POD, flange_every=1.6):
    p = rect_path(x0, y - r, x1, y + r)
    g.fill(p, color)
    g.fill_shader(p, g.lin_grad(0, y + r, 0, y - r, ["#FFFFFF", color, P.POD_DEEP], [0, 0.35, 1], 0.8))
    g.stroke(poly_path([(x0, y + r), (x1, y + r)], close=False), 1.0)
    g.stroke(poly_path([(x0, y - r), (x1, y - r)], close=False), 1.0)
    if flange_every:
        a = math.ceil(x0 / flange_every)
        b = math.floor(x1 / flange_every)
        for i in range(a, b + 1):
            fx = i * flange_every
            g.fs(rect_path(fx - 0.04, y - r * 1.4, fx + 0.04, y + r * 1.4), P.NAVY, 0.8)


def pipe_v(g, x, y0, y1, r, color=P.POD):
    p = rect_path(x - r, y0, x + r, y1)
    g.fill(p, color)
    g.fill_shader(p, g.lin_grad(x - r, 0, x + r, 0, ["#FFFFFF", color, P.POD_DEEP], [0, 0.35, 1], 0.8))
    g.stroke(poly_path([(x - r, y0), (x - r, y1)], close=False), 1.0)
    g.stroke(poly_path([(x + r, y0), (x + r, y1)], close=False), 1.0)


def draw_pipes(g, st):
    if g.visible(-8, 2.9, XR, 4.2):
        pipe_h(g, -8.0, XR - 3.2, MANIFOLD_Y, 0.10)
        pipe_h(g, -8.0, XR - 3.2, MANIFOLD_Y + 0.42, 0.065, P.REACTOR, 2.3)
    # risers to the ceiling near the reactor
    for xx in (XR - 1.6, XR + 1.6):
        if g.visible(xx - 0.3, 9, xx + 0.3, CEIL):
            pipe_v(g, xx, 9.0, CEIL, 0.16, P.REACTOR)
            for yy in (10.5, 12.5, 14.2):
                g.fs(rect_path(xx - 0.22, yy - 0.05, xx + 0.22, yy + 0.05), P.NAVY, 0.8)


LAMPS = [
    dict(id="opus", x=0.0, y=4.65, k=1.0),
    dict(id="sonnet", x=2.95, y=4.65, k=0.9),
    dict(id="haiku", x=5.3, y=4.65, k=0.8),
    dict(id="fable_l", x=XR - 2.6, y=12.2, k=1.6),
    dict(id="fable_r", x=XR + 2.6, y=12.2, k=1.6),
]


def draw_lamp(g, L, on, sway=0.0):
    x, y, k = L["x"], L["y"], L["k"]
    top = CEIL
    ang = sway
    cv = g.c
    cv.save()
    cv.translate(x, top)
    cv.rotate(-ang)
    ly = y - top
    g.line(0, 0, 0, ly + 0.25 * k, 1.0, P.INK)
    # shade
    sh = skia.Path()
    sh.moveTo(-0.08 * k, ly + 0.28 * k)
    sh.lineTo(0.08 * k, ly + 0.28 * k)
    sh.lineTo(0.12 * k, ly + 0.18 * k)
    sh.cubicTo(0.30 * k, ly + 0.10 * k, 0.36 * k, ly + 0.02 * k, 0.38 * k, ly)
    sh.lineTo(-0.38 * k, ly)
    sh.cubicTo(-0.36 * k, ly + 0.02 * k, -0.30 * k, ly + 0.10 * k, -0.12 * k, ly + 0.18 * k)
    sh.close()
    bulb = ellipse_path(0, ly - 0.02 * k, 0.13 * k, 0.07 * k)
    g.fs(bulb, mix("#FFF7DE", "#9A9384", 1 - on), 0.9)
    g.fs(sh, P.NAVY, 1.0)
    g.fill_shader(sh, g.lin_grad(-0.38 * k, 0, 0.38 * k, 0, ["#FFFFFF", P.NAVY, P.NAVY_DARK], [0, 0.4, 1], 0.35))
    cv.restore()


def lamp_pos(L, sway=0.0):
    r = CEIL - L["y"]
    a = math.radians(sway)
    return L["x"] + math.sin(a) * r, CEIL - math.cos(a) * r


def draw_light_cones(g, st):
    for L in LAMPS:
        on = st.lamps.get(L["id"], 0.0)
        if on <= 0.01:
            continue
        x, y = lamp_pos(L, st.lamp_sway)
        k = L["k"]
        floor_y = -0.6
        spread = 1.15 * k + (y - floor_y) * 0.32
        p = poly_path([(x - 0.33 * k, y), (x + 0.33 * k, y), (x + spread, floor_y), (x - spread, floor_y)])
        g.fill_shader(p, skia.GradientShader.MakeLinear([skia.Point(0, y), skia.Point(0, floor_y)],
                                                        [col("#FFF4D6", 0.22 * on), col("#FFF4D6", 0.0)], None))


# ------------------------------------------------------------------ pods

def draw_pod_body(g, X, k, label, gauge=0.2, jiggle=(0.0, 0.0)):
    cv = g.c
    cv.save()
    cv.translate(X + jiggle[0], 0)
    cv.rotate(-jiggle[1])
    cv.translate(-X, 0)
    xl, xr = X - 0.82 * k, X + 0.82 * k
    # riser pipe to manifold
    pipe_v(g, X, 2.6 * k, MANIFOLD_Y, 0.065)
    g.fs(rect_path(X - 0.11, MANIFOLD_Y - 0.17, X + 0.11, MANIFOLD_Y + 0.17), P.NAVY, 0.8)
    g.fs(rect_path(X - 0.12 * k, 2.56 * k, X + 0.12 * k, 2.74 * k), P.NAVY, 1.0)
    # dome
    dome = skia.Path()
    dome.moveTo(xl, 2.05 * k)
    dome.cubicTo(xl, 2.05 * k + 0.75 * k, xr, 2.05 * k + 0.75 * k, xr, 2.05 * k)
    dome.close()
    grad = ["#FFFFFF", P.POD, P.POD, P.POD_SHADE, P.POD_DEEP]
    gpos = [0.0, 0.22, 0.5, 0.82, 1.0]
    g.fill(dome, P.POD)
    g.fill_shader(dome, g.lin_grad(xl, 0, xr, 0, grad, gpos), 0.9)
    g.hatch(path_clip_right(dome, X + 0.25 * k), P.INK, 0.15, scale=0.5)
    g.stroke(dome, 1.2)
    # body
    body = rect_path(xl, 0.30 * k, xr, 2.05 * k)
    g.fill(body, P.POD)
    g.fill_shader(body, g.lin_grad(xl, 0, xr, 0, grad, gpos), 0.9)
    g.hatch(rect_path(X + 0.35 * k, 0.30 * k, xr, 2.05 * k), P.INK, 0.18, scale=0.5)
    g.stroke(body, 1.2)
    # bands
    for (b0, b1) in ((0.30 * k, 0.42 * k), (1.94 * k, 2.06 * k)):
        bp = rect_path(xl - 0.02 * k, b0, xr + 0.02 * k, b1)
        g.fs(bp, P.NAVY, 1.0)
        g.fill_shader(bp, g.lin_grad(xl, 0, xr, 0, [P.NAVY_LIGHT, P.NAVY, P.NAVY_DARK], [0, 0.4, 1]), 0.8)
        n = int(1.64 * k / 0.13)
        for i in range(n + 1):
            rx = xl + 0.06 * k + i * (1.52 * k) / n
            g.fill(circle_path(rx, (b0 + b1) / 2, 0.016 * k), "#C9CCE6")
    # gauge
    gx, gy, gr = X + 0.52 * k, 2.28 * k, 0.10 * k
    g.fs(circle_path(gx, gy, gr), P.BRASS, 1.0)
    g.fs(circle_path(gx, gy, gr * 0.75), "#FFFFFF", 0.6)
    for i in range(7):
        a = math.radians(210 - i * 40)
        g.line(gx + math.cos(a) * gr * 0.55, gy + math.sin(a) * gr * 0.55, gx + math.cos(a) * gr * 0.7,
               gy + math.sin(a) * gr * 0.7, 0.4)
    a = math.radians(210 - 240 * clamp(gauge))
    g.line(gx, gy, gx + math.cos(a) * gr * 0.62, gy + math.sin(a) * gr * 0.62, 1.0, P.TIE_RED)
    # pedestal + floor shadow
    g.soft(ellipse_path(X, -0.02, 1.05 * k, 0.09), P.INK, 0.06, 0.28)
    ped = rrect_path(X - 0.98 * k, 0.0, X + 0.98 * k, 0.32 * k, 0.05 * k)
    g.fs(ped, P.POD, 1.2)
    g.fill_shader(ped, g.lin_grad(X - k, 0, X + k, 0, grad, gpos), 0.7)
    g.fs(rect_path(X - 0.98 * k, 0.27 * k, X + 0.98 * k, 0.32 * k), P.NAVY, 0.8)
    plate = rect_path(X - 0.66 * k, 0.05 * k, X + 0.66 * k, 0.23 * k)
    g.fs(plate, "#FAF7EE", 1.0)
    g.stroke(rect_path(X - 0.63 * k, 0.07 * k, X + 0.63 * k, 0.21 * k), 0.5)
    g.text(label, X, 0.093 * k, 0.118 * k, P.INK, "slab")
    cv.restore()


def path_clip_right(p, x):
    b = p.getBounds()
    r = rect_path(x, b.top() - 1, b.right() + 1, b.bottom() + 1)
    out = skia.Op(p, r, skia.PathOp.kIntersect_PathOp)
    return out if out is not None else p


def door_matrix(g, k, X, phi):
    """Perspective matrix mapping the closed door (world rect) to its swung pose.
    Hinge on the right edge; door swings toward the camera."""
    d = pod_door(k, X)
    w = d["x1"] - d["x0"]
    xh = d["x1"]
    cx, cy = g.cam.cx, g.cam.cy
    F = max(4.5, 1.7 * g.cam.h)          # camera distance for a fixed field of view
    xf = xh - w * math.cos(phi)
    z = w * math.sin(phi)
    f = F / (F - z)

    def proj(x, y, f):
        return skia.Point(cx + (x - cx) * f, cy + (y - cy) * f)

    src = [skia.Point(d["x0"], d["y0"]), skia.Point(d["x1"], d["y0"]), skia.Point(d["x1"], d["y1"]), skia.Point(d["x0"], d["y1"])]
    dst = [proj(xf, d["y0"], f), proj(xh, d["y0"], 1.0), proj(xh, d["y1"], 1.0), proj(xf, d["y1"], f)]
    m = skia.Matrix()
    m.setPolyToPoly(src, dst)
    return m


def draw_pod_interior(g, X, k, occupant, t, light=1.0):
    d = pod_door(k, X)
    cav = rrect_path(d["x0"] + 0.02 * k, d["y0"], d["x1"] - 0.02 * k, d["y1"], 0.16 * k)
    g.fill(cav, P.POD_INSIDE)
    g.fill_shader(cav, g.rad_grad(X, d["y0"] + 0.1 * k, 1.3 * k, [P.POD_INSIDE_GLOW, P.POD_INSIDE], [0, 1], [0.55 * light, 0.0]))
    # back wall ribs
    for i in range(1, 4):
        yy = d["y0"] + i * (d["y1"] - d["y0"]) / 4
        g.line(d["x0"] + 0.08 * k, yy, d["x1"] - 0.08 * k, yy, 0.6, P.NAVY_LIGHT, 0.6)
    g.c.save()
    g.c.clipPath(cav, skia.ClipOp.kIntersect, True)
    if occupant is not None:
        draw_clawd(g, occupant, t)
    # cold mist on the floor
    for i in range(7):
        mx = d["x0"] + (i + 0.5) * (d["x1"] - d["x0"]) / 7 + 0.04 * k * math.sin(t * 1.3 + i)
        g.fill(circle_path(mx, d["y0"] + 0.02 * k, (0.12 + 0.05 * hash01(i, X)) * k), "#BFD9F0", 0.55)
    g.c.restore()
    g.stroke(cav, 1.0)


def draw_door(g, X, k, phi, wheel_ang, frost, wipe, t):
    """Door in its own (closed-door) coordinates; caller sets matrix."""
    d = pod_door(k, X)
    outer = rrect_path(d["x0"], d["y0"], d["x1"], d["y1"], 0.16 * k)
    win = rrect_path(X - 0.45 * k, 0.95 * k, X + 0.45 * k, 1.78 * k, 0.12 * k)
    panel = rrect_path(X - 0.45 * k, 0.56 * k, X + 0.45 * k, 0.88 * k, 0.06 * k)
    inside = math.cos(phi) < 0
    frame = path_diff_safe(outer, win)
    if inside:
        g.fs(outer, P.NAVY_DARK, 1.2)
        g.fs(rrect_path(d["x0"] + 0.08 * k, d["y0"] + 0.08 * k, d["x1"] - 0.08 * k, d["y1"] - 0.08 * k, 0.12 * k), P.NAVY, 0.8)
        g.fs(circle_path(X, 0.72 * k, 0.06 * k), P.NAVY_LIGHT, 0.8)
        g.fill(win, P.GLASS, 0.85)
        g.fill(win, P.FROST, 0.6)
        return
    g.fs(frame, P.NAVY, 1.2)
    g.fill_shader(frame, g.lin_grad(d["x0"], 0, d["x1"], 0, [P.NAVY_LIGHT, P.NAVY, P.NAVY_DARK], [0, 0.45, 1]), 0.8)
    g.fs(panel, P.POD, 1.0)
    g.hatch(panel, P.INK, 0.15, scale=0.4)
    # rivets on frame
    for i in range(6):
        yy = 0.62 * k + i * 0.22 * k
        g.fill(circle_path(d["x0"] + 0.05 * k, yy, 0.014 * k), "#C9CCE6")
        g.fill(circle_path(d["x1"] - 0.05 * k, yy, 0.014 * k), "#C9CCE6")
    # hinges (right)
    for yy in (0.62 * k, 1.68 * k):
        g.fs(rrect_path(d["x1"] - 0.02 * k, yy - 0.09 * k, d["x1"] + 0.07 * k, yy + 0.09 * k, 0.02 * k), P.NAVY_DARK, 0.8)
    # latch (left)
    g.fs(rrect_path(d["x0"] - 0.05 * k, 1.15 * k, d["x0"] + 0.05 * k, 1.42 * k, 0.03 * k), P.BRASS, 0.9)
    # glass
    if phi > 0.05:
        g.fill(win, P.GLASS, 0.9)
        g.fill(win, P.FROST, 0.55)
    else:
        if frost > 0:
            fr = win
            if wipe is not None:
                fr = path_diff_safe(win, wipe)
            g.fill(fr, P.FROST, 0.78 * frost)
            g.fill(fr, P.GLASS, 0.18 * frost)
            if wipe is not None:
                g.fill(path_isect_safe(win, wipe), P.GLASS, 0.12)
            # condensation beads
            for (bu, bv, br) in BEADS:
                bx, by, r = X + bu * k, bv * k, br * k
                g.fill(circle_path(bx, by, r), "#FFFFFF", 0.6 * frost)
                g.stroke(circle_path(bx, by, r), 0.35, P.GLASS_DEEP, 0.5 * frost)
    # reflections
    g.c.save()
    g.c.clipPath(win, skia.ClipOp.kIntersect, True)
    for off, wd in ((-0.18, 0.06), (-0.02, 0.025)):
        x = X + off * k
        g.fill(poly_path([(x - 0.3 * k, 0.95 * k), (x - 0.3 * k + wd * k, 0.95 * k), (x + 0.25 * k + wd * k, 1.78 * k),
                          (x + 0.25 * k, 1.78 * k)]), "#FFFFFF", 0.35)
    g.c.restore()
    g.stroke(win, 1.2)
    # wheel
    draw_wheel(g, X, 0.72 * k, 0.15 * k, wheel_ang)


def draw_wheel(g, x, y, r, ang):
    g.fs(circle_path(x, y, r), P.WHEEL_RED, 1.0)
    g.fill(circle_path(x, y, r * 0.74), P.POD)
    g.stroke(circle_path(x, y, r * 0.74), 0.9)
    for i in range(6):
        a = ang + i * math.pi / 3
        g.c.save()
        g.c.translate(x, y)
        g.c.rotate(-math.degrees(a))
        g.fs(rect_path(0, -r * 0.08, r * 0.8, r * 0.08), P.WHEEL_RED, 0.7)
        g.fs(circle_path(r * 0.92, 0, r * 0.11), P.WHEEL_RED_DARK, 0.6)
        g.c.restore()
    g.fs(circle_path(x, y, r * 0.22), P.WHEEL_RED_DARK, 0.8)
    g.fill(circle_path(x - r * 0.05, y + r * 0.06, r * 0.07), "#FFFFFF", 0.6)


def path_diff_safe(a, b):
    r = skia.Op(a, b, skia.PathOp.kDifference_PathOp)
    return r if r is not None else a


def path_isect_safe(a, b):
    r = skia.Op(a, b, skia.PathOp.kIntersect_PathOp)
    return r if r is not None else skia.Path()


def draw_pod(g, name, ps, t):
    """ps: dict(phi, wheel, frost, wipe(path|None), occupant(CS|None), jiggle, gauge, light)"""
    cfg = PODS[name]
    X, k = cfg["x"], cfg["k"]
    if not g.visible(X - 1.1 * k, 0, X + 1.1 * k + 1.2 * k, 3.0 * k):
        return
    jig = ps.get("jiggle", (0.0, 0.0))
    draw_pod_body(g, X, k, cfg["label"], ps.get("gauge", 0.2), jig)
    cv = g.c
    cv.save()
    cv.translate(X + jig[0], 0)
    cv.rotate(-jig[1])
    cv.translate(-X, 0)
    phi = ps.get("phi", 0.0)
    occ = ps.get("occupant")
    if phi > 0.01:
        draw_pod_interior(g, X, k, occ, t, ps.get("light", 1.0))
    else:
        # see the occupant through the closed window
        d = pod_door(k, X)
        win = rrect_path(X - 0.45 * k, 0.95 * k, X + 0.45 * k, 1.78 * k, 0.12 * k)
        cv.save()
        cv.clipPath(win, skia.ClipOp.kIntersect, True)
        draw_pod_interior(g, X, k, occ, t, ps.get("light", 1.0))
        if ps.get("screen", 0) > 0:
            s = ps["screen"]
            g.fill_shader(win, g.rad_grad(X, 1.05 * k, 0.6 * k, [P.SCREEN, P.SCREEN], [0, 1], [0.55 * s, 0.0]))
        cv.restore()
    m = door_matrix(g, k, X, phi)
    cv.save()
    cv.concat(m)
    draw_door(g, X, k, phi, ps.get("wheel", 0.0), ps.get("frost", 1.0), ps.get("wipe"), t)
    cv.restore()
    # door seam steam jets while pressurising
    jets = ps.get("jets", 0.0)
    if jets > 0 and phi < 0.01:
        d = pod_door(k, X)
        for i in range(5):
            ph = (t * 3.1 + hash01("jet", i)) % 1.0
            jx = d["x0"] - 0.02 * k - ph * 0.35 * k
            jy = d["y0"] + (0.2 + 0.15 * i) * (d["y1"] - d["y0"]) + ph * 0.12 * k
            g.fill(circle_path(jx, jy, (0.03 + 0.08 * ph) * k), "#FFFFFF", jets * (1 - ph) * 0.8)
    cv.restore()


# ------------------------------------------------------------------ reactor

def reactor_paths():
    vx0, vx1 = XR - 3.3, XR + 3.3
    shell = rect_path(vx0, 0.5, vx1, 7.8)
    dome = skia.Path()
    dome.moveTo(vx0 - 0.1, 8.0)
    dome.cubicTo(vx0 + 0.2, 9.75, vx1 - 0.2, 9.75, vx1 + 0.1, 8.0)
    dome.close()
    return shell, dome


def draw_reactor(g, rs, t):
    if not g.visible(XR - 4.6, -0.2, XR + 4.6, CEIL):
        return
    vx0, vx1 = XR - 3.3, XR + 3.3
    shell, dome = reactor_paths()
    grad = ["#EFE9DC", P.REACTOR, P.REACTOR, P.REACTOR_SHADE, P.REACTOR_DEEP]
    gpos = [0.0, 0.2, 0.55, 0.85, 1.0]
    # dome
    g.fill(dome, P.REACTOR)
    g.fill_shader(dome, g.lin_grad(vx0, 0, vx1, 0, grad, gpos))
    g.hatch(dome, P.INK, 0.12, scale=0.9)
    g.stroke(dome, 1.3)
    # drive on top
    gx0, gx1 = XR - 0.8, XR + 0.8
    g.fs(rect_path(XR - 0.25, 9.0, XR + 0.25, 9.6), P.NAVY_DARK, 1.0)
    gb = rrect_path(gx0, 9.45, gx1, 10.35, 0.08)
    g.fs(gb, P.NAVY, 1.2)
    g.fill_shader(gb, g.lin_grad(gx0, 0, gx1, 0, [P.NAVY_LIGHT, P.NAVY, P.NAVY_DARK], [0, 0.4, 1]), 0.8)
    mot = rrect_path(XR - 0.55, 10.35, XR + 0.55, 11.5, 0.12)
    g.fs(mot, P.REACTOR, 1.2)
    g.fill_shader(mot, g.lin_grad(XR - 0.55, 0, XR + 0.55, 0, grad, gpos))
    for i in range(7):
        yy = 10.5 + i * 0.14
        g.line(XR - 0.5, yy, XR + 0.5, yy, 0.6, P.REACTOR_DEEP)
    g.stroke(mot, 1.2)
    # beacon
    bx, by = XR + 0.55, 10.35
    on = rs.get("alarm", 0.0)
    g.fs(rect_path(bx - 0.1, by - 0.03, bx + 0.1, by + 0.06), P.NAVY_DARK, 0.8)
    bc = mix("#7A3A3A", P.ALARM, on)
    g.fs(rrect_path(bx - 0.09, by + 0.06, bx + 0.09, by + 0.32, 0.08), bc, 1.0)
    # top flange
    fl = rect_path(vx0 - 0.2, 7.75, vx1 + 0.2, 8.1)
    g.fs(fl, P.NAVY, 1.2)
    g.fill_shader(fl, g.lin_grad(vx0, 0, vx1, 0, [P.NAVY_LIGHT, P.NAVY, P.NAVY_DARK], [0, 0.4, 1]), 0.8)
    for i in range(23):
        bxp = vx0 - 0.05 + i * (6.7 / 22)
        g.fill(circle_path(bxp, 7.925, 0.045), "#C9CCE6")
    # shell
    g.fill(shell, P.REACTOR)
    g.fill_shader(shell, g.lin_grad(vx0, 0, vx1, 0, grad, gpos))
    g.hatch(rect_path(XR + 1.6, 0.5, vx1, 7.8), P.INK, 0.2, scale=0.9)
    g.hatch(rect_path(vx0, 0.5, vx0 + 0.5, 7.8), P.INK, 0.08, scale=0.9)
    g.stroke(shell, 1.3)
    # seams & bands
    for y0, y1 in ((0.5, 0.9), (6.15, 6.42)):
        b = rect_path(vx0 - 0.06, y0, vx1 + 0.06, y1)
        g.fs(b, P.NAVY, 1.1)
        g.fill_shader(b, g.lin_grad(vx0, 0, vx1, 0, [P.NAVY_LIGHT, P.NAVY, P.NAVY_DARK], [0, 0.4, 1]), 0.8)
        for i in range(31):
            g.fill(circle_path(vx0 + 0.1 + i * 0.2133, (y0 + y1) / 2, 0.028), "#C9CCE6")
    for yy in (2.0, 4.6):
        for side in (-1, 1):
            x_in = XR + side * (RING_R + 0.05)
            x_out = vx0 if side < 0 else vx1
            g.line(min(x_in, x_out), yy, max(x_in, x_out), yy, 0.6, P.REACTOR_DEEP)
    # nameplate & gauges
    np0, np1 = XR - 1.45, XR + 1.45
    g.fs(rect_path(np0, 6.62, np1, 7.45), "#FAF7EE", 1.3)
    g.stroke(rect_path(np0 + 0.07, 6.69, np1 - 0.07, 7.38), 0.7)
    for sx in (np0 + 0.14, np1 - 0.14):
        for sy in (6.76, 7.31):
            g.fill(circle_path(sx, sy, 0.03), P.INK_SOFT)
    g.text("FABLE 5.5", XR, 6.84, 0.48, P.INK, "slab")
    for i, gx in enumerate((XR - 2.3, XR + 2.3)):
        draw_gauge(g, gx, 7.0, 0.36, rs.get("gauges", (0.2, 0.25))[i], rs.get("alarm", 0))
    # side pipe into manifold
    g.fs(rect_path(vx0 - 0.28, MANIFOLD_Y - 0.2, vx0 + 0.02, MANIFOLD_Y + 0.2), P.NAVY, 1.0)
    # ladder
    lx0, lx1 = vx1 + 0.22, vx1 + 0.62
    g.line(lx0, 0.5, lx0, 8.4, 1.6, P.NAVY)
    g.line(lx1, 0.5, lx1, 8.4, 1.6, P.NAVY)
    for i in range(27):
        yy = 0.8 + i * 0.28
        g.line(lx0, yy, lx1, yy, 1.1, P.NAVY)
    # plinth with hazard stripes
    g.soft(rect_path(XR - 4.3, -0.16, XR + 4.3, 0.0), P.INK, 0.08, 0.3)
    pl = rect_path(XR - 4.15, 0.0, XR + 4.15, 0.52)
    g.fs(pl, P.SAFETY, 1.2)
    g.c.save()
    g.c.clipPath(pl, skia.ClipOp.kIntersect, True)
    for i in range(-2, 40):
        x = XR - 4.4 + i * 0.42
        g.fill(poly_path([(x, 0.0), (x + 0.21, 0.0), (x + 0.21 + 0.52, 0.52), (x + 0.52, 0.52)]), P.HAZARD_DARK)
    g.c.restore()
    g.stroke(pl, 1.2)
    g.fs(rect_path(XR - 4.15, 0.48, XR + 4.15, 0.56), P.NAVY_DARK, 0.8)

    draw_porthole(g, rs, t)
    # warning sign
    sx, sy = XR + 2.88, 2.7
    sg = rect_path(sx - 0.33, sy - 0.62, sx + 0.33, sy + 0.62)
    g.fs(sg, P.SAFETY, 1.0)
    g.stroke(rect_path(sx - 0.28, sy - 0.57, sx + 0.28, sy + 0.57), 0.6)
    tri = poly_path([(sx, sy + 0.48), (sx - 0.16, sy + 0.2), (sx + 0.16, sy + 0.2)])
    g.fs(tri, P.SAFETY, 0.9)
    g.text("!", sx, sy + 0.235, 0.2, P.INK, "slab")
    for i, line in enumerate(("PLEASE", "DO NOT", "TAP", "THE", "GLASS")):
        g.text(line, sx, sy + 0.06 - i * 0.135, 0.105, P.INK, "slab")


def draw_gauge(g, x, y, r, v, alarm=0.0):
    g.fs(circle_path(x, y, r), P.BRASS, 1.2)
    g.fs(circle_path(x, y, r * 0.8), "#FFFFFF", 0.7)
    arc = skia.Path()
    arc.addArc(skia.Rect(x - r * 0.62, y - r * 0.62, x + r * 0.62, y + r * 0.62), -60, 60)
    g.stroke(arc, 2.2, P.ALARM)
    for i in range(9):
        a = math.radians(225 - i * 33.75)
        g.line(x + math.cos(a) * r * 0.55, y + math.sin(a) * r * 0.55, x + math.cos(a) * r * 0.72,
               y + math.sin(a) * r * 0.72, 0.6)
    a = math.radians(225 - 270 * clamp(v, 0, 1.02))
    g.line(x, y, x + math.cos(a) * r * 0.64, y + math.sin(a) * r * 0.64, 1.6, P.TIE_RED)
    g.fs(circle_path(x, y, r * 0.09), P.INK, 0.5)
    g.fill(circle_path(x - r * 0.35, y + r * 0.35, r * 0.18), "#FFFFFF", 0.35)


# ------------------------------------------------------------------ porthole + Fable

_rng = np.random.default_rng(21)
CURLS = []
for _i in range(18):
    _a = _rng.uniform(0, 2 * math.pi)
    _r = PORT_R * math.sqrt(_rng.uniform(0.02, 1.0)) * 0.95
    CURLS.append(dict(
        x=math.cos(_a) * _r, y=math.sin(_a) * _r,
        r=0.16 + 0.32 * _rng.uniform(), a0=_rng.uniform(0, 6.28),
        sp=(0.15 + 0.25 * _rng.uniform()) * (1 if _i % 2 else -1),
        drift=0.05 + 0.08 * _rng.uniform()))

DROPS = []
for _i in range(80):
    _a = _rng.uniform(0, 2 * math.pi)
    _rr = PORT_R * math.sqrt(_rng.uniform()) * 0.97
    DROPS.append((math.cos(_a) * _rr, math.sin(_a) * _rr, 0.012 + 0.03 * _rng.uniform() ** 2))

BEADS = [(_rng.uniform(-0.42, 0.42), 0.98 + _rng.uniform() ** 1.5 * 0.76, 0.008 + 0.014 * _rng.uniform()) for _ in range(28)]


def draw_curl(g, cx, cy, r, a0, color, a, w):
    pts = []
    n = 26
    for i in range(n):
        u = i / (n - 1)
        ang = a0 + u * 4.2
        rad = r * (1 - 0.75 * u)
        pts.append((cx + math.cos(ang) * rad, cy + math.sin(ang) * rad * 0.75))
    # tail
    tail = [(cx + math.cos(a0) * r + (r * 0.8) * j, cy + math.sin(a0) * r * 0.75 - 0.05 * j) for j in (0.3, 0.7)]
    g.stroke(smooth_path(list(reversed(tail)) + pts, close=False), w, color, a)


def fable_face(g, rs, t):
    """The giant inside: only ever seen through the porthole."""
    cx, cy = PORT_C
    glow = rs.get("eyes", 0.0)
    lunge = rs.get("lunge", 0.0)
    W = FABLE_W * (1.0 + 0.32 * lunge)
    ox, oy = rs.get("face_off", (0.0, 0.0))
    fx, fy = cx + ox, cy - 0.28 + oy          # eye line sits a little below centre
    H = 0.667 * W
    top = fy + 0.25 * W                       # eyes are 0.25 W below the top
    body = rrect_path(fx - W / 2, top - H, fx + W / 2, top, 0.06 * W)
    vis = rs.get("face_vis", 0.25 + 0.75 * glow)
    g.fill(body, mix(P.MURK, P.ORANGE_DEEP, 0.55 * vis))
    # eye-lit face
    eye_y = fy
    exs = (fx - 0.2917 * W, fx + 0.2917 * W)
    for ex in exs:
        g.fill_shader(body, g.rad_grad(ex, eye_y, 1.6 * W / FABLE_W,
                                       [P.ORANGE, P.ORANGE_DEEP], [0, 1], [0.9 * glow * vis, 0.0]))
    g.hatch(body, P.INK, 0.12 * vis, scale=1.0)
    g.stroke(body, 1.4, P.INK, 0.5 * vis)
    # eyes
    ew, eh = 0.0833 * W, 0.1667 * W
    op = rs.get("eye_open", 1.0)
    angry = rs.get("angry", 0.0)
    for i, ex in enumerate(exs):
        hh = eh * max(op, 0.05) / 2
        hw = ew / 2
        inner = 1 if i == 0 else -1  # +x is toward the centre for the left eye
        tl = (ex - hw, eye_y + hh - (angry * eh * 0.3 if inner < 0 else 0))
        tr = (ex + hw, eye_y + hh - (angry * eh * 0.3 if inner > 0 else 0))
        pts = [(ex - hw, eye_y - hh), (ex + hw, eye_y - hh), tr, tl]
        ep = poly_path(pts)
        if glow > 0.02:
            k = W / FABLE_W
            g.glow(ep, P.GLOW, 0.55 * k, 0.55 * glow)
            g.glow(ep, "#FF7A2E", 0.22 * k, 0.6 * glow)
            g.fill(ep, mix(P.INK, "#F59A3A", glow))
            g.c.save()
            g.c.clipPath(ep, skia.ClipOp.kIntersect, True)
            g.fill_shader(ep, g.lin_grad(0, eye_y + hh, 0, eye_y - hh, [P.GLOW_CORE, P.GLOW, "#F28A33"], [0.0, 0.55, 1.0]), glow)
            g.soft(rect_path(ex - hw * 0.35, eye_y - hh * 0.6, ex + hw * 0.35, eye_y + hh * 0.5), "#FFFFFF", 0.05 * k, 0.75 * glow)
            g.fill(rect_path(ex - hw * 0.45, tl[1] - hh * 0.22 if inner < 0 else tr[1] - hh * 0.22, ex - hw * 0.1,
                             (tl[1] if inner < 0 else tr[1]) - hh * 0.08), "#FFFFFF", 0.8 * glow)
            g.c.restore()
            g.stroke(ep, 1.0, "#7A3418", 0.5 * glow)
        else:
            g.fill(ep, P.INK, 0.6)


def draw_porthole(g, rs, t):
    cx, cy = PORT_C
    if not g.visible(cx - RING_R, cy - RING_R, cx + RING_R, cy + RING_R):
        return
    glassp = circle_path(cx, cy, PORT_R)
    cv = g.c
    cv.save()
    cv.clipPath(glassp, skia.ClipOp.kIntersect, True)
    g.fill(glassp, P.MURK)
    g.fill_shader(glassp, g.rad_grad(cx, cy - 0.3, PORT_R * 1.1, [P.MURK_LIGHT, P.MURK], [0, 1]))
    fable_face(g, rs, t)
    # swirling vapour
    glow = rs.get("eyes", 0.0)
    for i, c in enumerate(CURLS):
        a0 = c["a0"] + t * c["sp"]
        x = cx + c["x"] + math.sin(t * 0.4 + i) * c["drift"] * 2
        y = cy + c["y"] + math.cos(t * 0.33 + i * 1.7) * c["drift"]
        colr = mix("#B9AFCB", P.GLOW_CORE, glow * 0.6)
        draw_curl(g, x, y, c["r"], a0, colr, 0.32 + 0.2 * glow, 0.8)
    # haze
    g.fill_shader(glassp, g.rad_grad(cx, cy - 0.6, PORT_R, ["#8A7C95", "#8A7C95"], [0, 1], [0.0, 0.35]))
    flash = rs.get("flash", 0.0)
    if flash > 0:
        g.fill(glassp, "#FFE8C0", 0.55 * flash)
    # condensation
    for (dx, dy, r) in DROPS:
        g.fill(circle_path(cx + dx, cy + dy, r), "#E8E4F0", 0.25)
        g.fill(circle_path(cx + dx - r * 0.3, cy + dy + r * 0.3, r * 0.35), "#FFFFFF", 0.4)
    # cracks
    draw_cracks(g, rs.get("cracks", 0.0), t)
    # glass highlights
    hl = skia.Path()
    hl.addArc(skia.Rect(cx - PORT_R * 0.9, cy - PORT_R * 0.9, cx + PORT_R * 0.9, cy + PORT_R * 0.9), 110, 70)
    g.stroke(hl, 4.0, "#FFFFFF", 0.22)
    g.stroke(poly_path([(cx - 0.95, cy + 0.15), (cx - 0.35, cy + 1.05)], close=False), 5.0, "#FFFFFF", 0.35)
    g.stroke(poly_path([(cx - 0.72, cy - 0.05), (cx - 0.55, cy + 0.22)], close=False), 3.0, "#FFFFFF", 0.25)
    cv.restore()
    # ring
    ring = path_diff_safe(circle_path(cx, cy, RING_R), circle_path(cx, cy, PORT_R))
    g.fill(ring, P.NAVY)
    g.fill_shader(ring, g.lin_grad(cx - RING_R, cy + RING_R, cx + RING_R, cy - RING_R,
                                   [P.NAVY_LIGHT, P.NAVY, P.NAVY_DARK], [0, 0.45, 1]))
    g.hatch(path_isect_safe(ring, rect_path(cx, cy - RING_R, cx + RING_R, cy + RING_R)), P.INK, 0.25, scale=0.8)
    g.stroke(circle_path(cx, cy, PORT_R + 0.07), 2.0, P.NAVY_DARK)
    g.stroke(circle_path(cx, cy, PORT_R + 0.03), 1.0, P.NAVY_LIGHT, 0.7)
    hl2 = skia.Path()
    hl2.addArc(skia.Rect(cx - RING_R + 0.12, cy - RING_R + 0.12, cx + RING_R - 0.12, cy + RING_R - 0.12), 120, 60)
    g.stroke(hl2, 3.0, "#FFFFFF", 0.35)
    popped = rs.get("popped", ())
    for i in range(16):
        a = math.radians(90 + i * 22.5)
        bx, by = cx + math.cos(a) * (PORT_R + 0.27), cy + math.sin(a) * (PORT_R + 0.27)
        if i in popped:
            g.fill(circle_path(bx, by, 0.06), P.INK)
            continue
        g.fs(circle_path(bx, by, 0.09), P.NAVY_DARK, 0.9)
        g.fill(circle_path(bx - 0.025, by + 0.025, 0.03), "#C9CCE6", 0.8)
    g.stroke(ring, 1.4)


# ------------------------------------------------------------------ cracks

def _build_cracks():
    rng = np.random.default_rng(55)
    c0 = (0.18, 0.28)
    segs = []
    # radial spokes
    n = 11
    for i in range(n):
        ang = i / n * 2 * math.pi + rng.uniform(-0.2, 0.2)
        x, y = c0
        length = 0.0
        pts = [(x, y)]
        while True:
            step = rng.uniform(0.12, 0.26)
            ang += rng.uniform(-0.35, 0.35)
            x += math.cos(ang) * step
            y += math.sin(ang) * step
            length += step
            pts.append((x, y))
            if math.hypot(x, y) > PORT_R * 1.05 or length > 2.8:
                break
        acc = 0.0
        for j in range(len(pts) - 1):
            a, b = pts[j], pts[j + 1]
            d0 = acc
            acc += math.dist(a, b)
            lvl = 1 if d0 < 0.45 else (2 if d0 < 1.15 else 3)
            if i % 3 == 2 and lvl == 1:
                lvl = 2
            segs.append(dict(a=a, b=b, lvl=lvl, d=d0, w=max(0.5, 1.9 - acc * 0.55)))
    # concentric links
    for ring, lvl in ((0.35, 1), (0.7, 2), (1.2, 3), (1.55, 3)):
        prev = None
        m = 18
        for i in range(m + 1):
            if rng.random() < 0.35 and i:
                prev = None
                continue
            ang = i / m * 2 * math.pi
            r = ring * rng.uniform(0.85, 1.15)
            p = (c0[0] + math.cos(ang) * r, c0[1] + math.sin(ang) * r * 0.95)
            if prev is not None:
                segs.append(dict(a=prev, b=p, lvl=lvl, d=ring, w=0.8))
            prev = p
    # tiny chips around impact
    chips = []
    for i in range(18):
        a = rng.uniform(0, 2 * math.pi)
        r = rng.uniform(0.02, 0.2)
        chips.append((c0[0] + math.cos(a) * r, c0[1] + math.sin(a) * r, rng.uniform(0.015, 0.04), rng.uniform(0, 6)))
    return c0, segs, chips


CRACK_C0, CRACKS, CHIPS = _build_cracks()


def draw_cracks(g, level, t):
    if level <= 0:
        return
    cx, cy = PORT_C
    for s in CRACKS:
        lv = s["lvl"]
        p = clamp(level - (lv - 1))  # 0..1 progress of this level
        if p <= 0:
            continue
        # grow outward within the level
        span = {1: 0.45, 2: 1.15, 3: 2.8}[lv]
        start = {1: 0.0, 2: 0.45, 3: 1.15}[lv]
        frac = clamp((p * 1.25 * (span - start) + start - s["d"]) / max(math.dist(s["a"], s["b"]), 1e-3))
        if frac <= 0:
            continue
        a = s["a"]
        b = (lerp(s["a"][0], s["b"][0], frac), lerp(s["a"][1], s["b"][1], frac))
        ax, ay, bx, by = cx + a[0], cy + a[1], cx + b[0], cy + b[1]
        g.line(ax + 0.012, ay - 0.012, bx + 0.012, by - 0.012, s["w"] * 1.1, "#1A1424", 0.55)
        g.line(ax, ay, bx, by, s["w"], "#F4F1FA", 0.9)
    # impact star
    k = clamp(level)
    g.fill(circle_path(cx + CRACK_C0[0], cy + CRACK_C0[1], 0.07 + 0.08 * clamp(level - 1, 0, 2)), "#FFFFFF", 0.75 * k)
    for (x, y, r, a) in CHIPS:
        g.c.save()
        g.c.translate(cx + x, cy + y)
        g.c.rotate(math.degrees(a))
        g.fill(poly_path([(-r, -r * 0.4), (r, -r * 0.6), (r * 0.3, r)]), "#FFFFFF", 0.6 * k)
        g.c.restore()


# ------------------------------------------------------------------ steam & dust

class Burst:
    """A cartoon steam/dust cloud made of puffs, spawned at t0."""

    def __init__(self, t0, x, y, n, spread, vel, size, life, seed, color="#FFFFFF", up=0.35, shade_col="#C9D4E4",
                 front=False):
        rng = np.random.default_rng(seed)
        self.front = front
        self.t0 = t0
        self.color = color
        self.shade_col = shade_col
        self.puffs = []
        for i in range(n):
            ang = rng.uniform(0, 2 * math.pi)
            sp = rng.uniform(0.3, 1.0)
            vx = vel[0] + math.cos(ang) * spread * sp
            vy = vel[1] + math.sin(ang) * spread * sp * 0.7
            self.puffs.append(dict(
                x=x + rng.uniform(-0.1, 0.1) * size * 2, y=y + rng.uniform(-0.1, 0.1) * size * 2,
                vx=vx, vy=vy, r=size * rng.uniform(0.55, 1.15), life=life * rng.uniform(0.7, 1.15),
                delay=rng.uniform(0, 0.12), up=up * rng.uniform(0.6, 1.3)))

    def circles(self, t):
        out = []
        for p in self.puffs:
            age = t - self.t0 - p["delay"]
            if age <= 0 or age >= p["life"]:
                continue
            u = age / p["life"]
            mv = 1 - (1 - min(1, age / 0.6)) ** 3
            x = p["x"] + p["vx"] * 0.6 * mv + p["vx"] * 0.08 * age
            y = p["y"] + p["vy"] * 0.6 * mv + p["up"] * age * age * 0.6
            grow = ease_out(min(1, age / 0.35))
            shrink = 1 - smooth(clamp((u - 0.55) / 0.45))
            r = p["r"] * (0.35 + 0.65 * grow) * (0.25 + 0.75 * shrink) * (1 + 0.4 * u)
            if r > 0.004:
                out.append((x, y, r))
        return out

    def draw(self, g, t):
        cs = self.circles(t)
        if not cs:
            return
        lw = g.lw(2.4)
        sp = skia.Paint(AntiAlias=True, Color=col(P.INK), Style=skia.Paint.kStroke_Style, StrokeWidth=lw)
        for (x, y, r) in cs:
            g.c.drawCircle(x, y, r, sp)
        fp = skia.Paint(AntiAlias=True, Color=col(self.color))
        for (x, y, r) in cs:
            g.c.drawCircle(x, y, r, fp)
        for (x, y, r) in cs:
            g.fill(circle_path(x + r * 0.22, y - r * 0.25, r * 0.62), self.shade_col, 0.55)
            g.fill(circle_path(x - r * 0.3, y + r * 0.32, r * 0.25), "#FFFFFF", 0.8)
        # re-cover inner shading edges so the silhouette stays clean
        for (x, y, r) in cs:
            g.c.drawCircle(x - r * 0.1, y + r * 0.1, r * 0.55, fp)
