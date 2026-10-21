"""Clawd, drawn as a little box that can turn in 3-D.

Proportions come straight from the terminal glyph:

     ▐▛███▜▌
    ▝▜█████▛▘
      ▘▘ ▝▝

body 1.0 x 0.667, eye slits 0.083 x 0.167, side stubs 0.167, four legs.
Each box face (front / sides / back) is drawn as a flat decal and mapped
onto the projected face, so outfits stay attached when a Clawd turns.
"""
import math
from dataclasses import dataclass

import skia

from . import palette as P
from .gfx import rect_path, rrect_path, circle_path, poly_path, shade, clamp, lerp


@dataclass
class CS:
    """Clawd state for one frame."""
    W: float = 1.0
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0          # 0 front, +pi/2 faces screen-right, pi back
    lift: float = 0.0
    sq: float = 1.0             # squash (>1 wide & short)
    tilt: float = 0.0           # degrees, + leans clockwise
    walk: float = 0.0           # gait phase
    walk_amp: float = 0.0
    eye_open: float = 1.0
    look: tuple = (0.0, 0.0)
    eye_wide: float = 0.0
    eye_happy: float = 0.0
    arm_l: float = 0.0          # raise of the screen-left arm (front view)
    arm_r: float = 0.0
    outfit: str = None          # 'labcoat' | 'hoodie' | 'uniform'
    goggles_down: float = 0.0
    glint: float = -1.0         # glasses glint sweep 0..1
    prop: str = None            # 'clipboard' | 'laptop'
    laptop_open: float = 0.0
    typing: float = 0.0
    checks: tuple = (0.0, 0.0, 0.0, 0.0)
    circle4: float = 0.0
    alpha: float = 1.0
    shadow: bool = True
    rim: float = 0.0            # warm rim light from Fable's eyes
    screen_glow: float = 0.0
    pencil: float = -1.0        # pencil position along the checklist (insert shots)
    depth: float = None         # draw order; defaults to -y (nearer = lower on the floor)
    name: str = ""


LEGS = (-0.375, -0.2083, 0.2083, 0.375)
SIDE = 0.82                    # brightness of side faces


class Face:
    """Maps face-local (u, v) into Clawd-local space, v=0 at body bottom."""

    def __init__(self, g, sx, tx, y0, width, kind, bright=1.0):
        self.g = g
        self.m = skia.Matrix.MakeAll(sx, 0, tx, 0, 1, y0, 0, 0, 1)
        self.sx = sx
        self.width = width
        self.kind = kind
        self.bright = bright

    def tp(self, p):
        q = skia.Path()
        p.transform(self.m, q)
        return q

    def col(self, c):
        return shade(c, self.bright) if self.bright != 1.0 else c

    def fs(self, p, c, w=1.0, a=1.0):
        self.g.fs(self.tp(p), self.col(c), w, a)

    def fill(self, p, c, a=1.0):
        self.g.fill(self.tp(p), self.col(c), a)

    def stroke(self, p, w=1.0, c=P.INK, a=1.0):
        self.g.stroke(self.tp(p), w, c, a)

    def line(self, u0, v0, u1, v1, w=1.0, c=P.INK, a=1.0):
        self.stroke(poly_path([(u0, v0), (u1, v1)], close=False), w, c, a)

    def hatch(self, p, a=0.2, kind="diag", c=P.INK, scale=0.6):
        self.g.hatch(self.tp(p), c, a, kind, scale)

    def text(self, s, u, v, size, color=P.INK, font="slab", a=1.0):
        c = self.g.c
        c.save()
        c.concat(self.m)
        self.g.text(s, u, v, size, color, font, a=a)
        c.restore()

    def begin(self):
        self.g.c.save()
        self.g.c.concat(self.m)

    def end(self):
        self.g.c.restore()


# ------------------------------------------------------------------ eyes

def draw_eyes(F, st, W, H):
    look_x, look_y = st.look
    ew, eh = 0.0833 * W, 0.1667 * W
    vc = 0.4167 * W
    op = clamp(st.eye_open, 0.0, 1.2)
    wide = 1.0 + 0.32 * st.eye_wide
    for i, uc in enumerate((0.2083 * W, 0.7917 * W)):
        u = uc + 0.035 * W * look_x
        v = vc + 0.03 * W * look_y
        if st.eye_happy > 0.5:
            pts = [(u - ew * 0.75, v - eh * 0.12), (u, v + eh * 0.32), (u + ew * 0.75, v - eh * 0.12)]
            F.stroke(poly_path(pts, close=False), 2.2)
            continue
        hh = eh * wide * max(op, 0.09) / 2
        hw = ew * wide / 2
        p = rect_path(u - hw, v - hh, u + hw, v + hh)
        F.fill(p, P.INK)
        if op > 0.5 and W > 0.3:
            s = 0.022 * W * wide
            F.fill(rect_path(u + hw * 0.1, v + hh - s * 1.6, u + hw * 0.1 + s, v + hh - s * 0.6), "#FFFFFF", 0.85)


# ------------------------------------------------------------------ outfits

def _hem(W):
    return -0.075 * W


def outfit_front(F, st, W, H):
    o = st.outfit
    hem = _hem(W)
    if o == "labcoat":
        top = 0.29 * W
        body = poly_path([(-0.03 * W, hem), (W + 0.03 * W, hem), (W + 0.03 * W, top),
                          (0.69 * W, top), (0.5 * W, 0.075 * W), (0.31 * W, top), (-0.03 * W, top)])
        F.fs(body, P.COAT)
        F.hatch(rect_path(0.72 * W, hem, W + 0.03 * W, top), 0.10, scale=0.5)
        # lapels
        for s in (-1, 1):
            cx = 0.5 * W
            a = (cx + s * 0.19 * W, top)
            b = (cx + s * 0.005 * W, 0.085 * W)
            c = (cx + s * 0.26 * W, 0.17 * W)
            d = (cx + s * 0.235 * W, 0.235 * W)
            F.fs(poly_path([a, b, c, d]), P.COAT_SHADE, 0.9)
        # buttons
        for v in (0.03 * W, -0.035 * W):
            F.fs(circle_path(0.5 * W, v, 0.017 * W), "#E9E4DA", 0.7)
        # breast pocket + pens
        F.fs(rect_path(0.75 * W, 0.17 * W, 0.765 * W + 0.0, 0.245 * W), "#3A62C4", 0.6)
        F.fs(rect_path(0.80 * W, 0.16 * W, 0.815 * W, 0.235 * W), P.TIE_RED, 0.6)
        F.fs(rect_path(0.70 * W, 0.04 * W, 0.87 * W, 0.175 * W), P.COAT, 0.8)
        F.line(0.70 * W, 0.155 * W, 0.87 * W, 0.155 * W, 0.6)
        # name tag
        F.fs(rect_path(0.10 * W, 0.115 * W, 0.27 * W, 0.17 * W), "#FFFFFF", 0.6)
        F.fill(rect_path(0.10 * W, 0.115 * W, 0.12 * W, 0.17 * W), P.ORANGE)
        F.text("OPUS", 0.197 * W, 0.128 * W, 0.034 * W, P.INK, "slab")
        # lower pocket
        F.line(0.08 * W, -0.005 * W, 0.27 * W, -0.005 * W, 0.8)
    elif o == "hoodie":
        top = 0.285 * W
        p = skia.Path()
        p.moveTo(-0.025 * W, hem)
        p.lineTo(W + 0.025 * W, hem)
        p.lineTo(W + 0.025 * W, top)
        p.lineTo(0.66 * W, top)
        p.quadTo(0.5 * W, 0.20 * W, 0.34 * W, top)
        p.lineTo(-0.025 * W, top)
        p.close()
        F.fs(p, P.HOODIE)
        F.hatch(p, 0.10, kind="cross", scale=0.45)
        # neck rib
        q = skia.Path()
        q.moveTo(0.33 * W, top)
        q.quadTo(0.5 * W, 0.19 * W, 0.67 * W, top)
        F.stroke(q, 3.0, P.HOODIE_DARK)
        F.stroke(q, 0.8)
        # rib hem
        F.fs(rect_path(-0.025 * W, hem, W + 0.025 * W, hem + 0.035 * W), P.HOODIE_DARK, 0.8)
        # kangaroo pocket
        k = poly_path([(0.22 * W, hem + 0.035 * W), (0.78 * W, hem + 0.035 * W), (0.71 * W, 0.105 * W), (0.29 * W, 0.105 * W)])
        F.fs(k, P.HOODIE_LIGHT, 0.8)
        F.line(0.29 * W, 0.105 * W, 0.24 * W, hem + 0.04 * W, 0.6, P.HOODIE_DARK)
        # lanyard + badge
        F.stroke(poly_path([(0.30 * W, top + 0.005 * W), (0.5 * W, 0.075 * W), (0.70 * W, top + 0.005 * W)], close=False), 2.0, P.LANYARD)
        F.fs(rrect_path(0.43 * W, -0.02 * W, 0.57 * W, 0.085 * W, 0.01 * W), "#FFFFFF", 0.7)
        F.fill(rect_path(0.45 * W, 0.035 * W, 0.49 * W, 0.07 * W), P.ORANGE)
        F.line(0.45 * W, 0.015 * W, 0.55 * W, 0.015 * W, 0.5, P.INK_SOFT)
        F.line(0.45 * W, 0.0, 0.53 * W, 0.0, 0.5, P.INK_SOFT)
        # drawstrings
        for u0, u1 in ((0.43 * W, 0.40 * W), (0.57 * W, 0.61 * W)):
            F.stroke(poly_path([(u0, 0.235 * W), (u1, 0.13 * W)], close=False), 1.3, "#F3F1EC")
            F.fs(rect_path(u1 - 0.008 * W, 0.11 * W, u1 + 0.008 * W, 0.135 * W), P.INK_SOFT, 0.4)
    elif o == "uniform":
        top = 0.30 * W
        # shirt V
        F.fs(poly_path([(0.31 * W, top), (0.69 * W, top), (0.5 * W, 0.03 * W)]), P.SHIRT, 0.6)
        # tie
        tie = poly_path([(0.47 * W, 0.235 * W), (0.53 * W, 0.235 * W), (0.555 * W, 0.06 * W), (0.5 * W, 0.0),
                         (0.445 * W, 0.06 * W)])
        F.fs(tie, P.TIE_RED, 0.8)
        F.begin()
        F.g.c.clipPath(tie, skia.ClipOp.kIntersect, True)
        for i in range(-3, 6):
            v = 0.05 * W * i
            F.g.stroke(poly_path([(0.42 * W, v), (0.6 * W, v + 0.07 * W)], close=False), 2.4, P.BLAZER_DARK)
        F.end()
        F.fs(poly_path([(0.465 * W, 0.275 * W), (0.535 * W, 0.275 * W), (0.525 * W, 0.235 * W), (0.475 * W, 0.235 * W)]), P.TIE_RED, 0.8)
        # collar points
        for s in (-1, 1):
            F.fs(poly_path([(0.5 * W + s * 0.17 * W, top + 0.012 * W), (0.5 * W + s * 0.03 * W, 0.275 * W),
                            (0.5 * W + s * 0.11 * W, 0.215 * W)]), P.SHIRT, 0.7)
        # blazer
        bl = poly_path([(-0.03 * W, hem), (W + 0.03 * W, hem), (W + 0.03 * W, top), (0.69 * W, top),
                        (0.5 * W, 0.03 * W), (0.31 * W, top), (-0.03 * W, top)])
        F.fs(bl, P.BLAZER)
        F.hatch(rect_path(0.7 * W, hem, W + 0.03 * W, top), 0.12, scale=0.45)
        for s in (-1, 1):
            cx = 0.5 * W
            F.fs(poly_path([(cx + s * 0.19 * W, top), (cx + s * 0.005 * W, 0.04 * W), (cx + s * 0.25 * W, 0.15 * W),
                            (cx + s * 0.225 * W, 0.23 * W)]), P.BLAZER_DARK, 0.8)
        for v in (-0.005 * W, -0.05 * W):
            F.fs(circle_path(0.5 * W, v, 0.016 * W), P.GOLD, 0.6)
        # crest
        cr = poly_path([(0.73 * W, 0.17 * W), (0.85 * W, 0.17 * W), (0.85 * W, 0.11 * W), (0.79 * W, 0.065 * W), (0.73 * W, 0.11 * W)])
        F.fs(cr, P.TIE_RED, 0.6)
        F.stroke(cr, 1.0, P.GOLD)
        F.text("H", 0.79 * W, 0.098 * W, 0.05 * W, P.GOLD, "slab")
        # backpack straps
        for u in (0.085 * W, 0.915 * W):
            s = rect_path(u - 0.03 * W, hem, u + 0.03 * W, top + 0.012 * W)
            F.fs(s, P.BACKPACK, 0.8)
            F.fs(rect_path(u - 0.036 * W, 0.08 * W, u + 0.036 * W, 0.11 * W), "#C9CBD1", 0.6)


def outfit_back(F, st, W, H):
    o = st.outfit
    hem = _hem(W)
    if o == "labcoat":
        b = rect_path(-0.03 * W, hem, W + 0.03 * W, 0.31 * W)
        F.fs(b, P.COAT)
        F.hatch(b, 0.08, scale=0.5)
        F.fs(rect_path(-0.03 * W, 0.255 * W, W + 0.03 * W, 0.33 * W), P.COAT_SHADE, 0.9)
        F.line(0.5 * W, hem, 0.5 * W, 0.04 * W, 0.8)
        F.fs(rect_path(0.24 * W, 0.045 * W, 0.76 * W, 0.09 * W), P.COAT, 0.8)
        for u in (0.29 * W, 0.71 * W):
            F.fs(circle_path(u, 0.0675 * W, 0.013 * W), "#E9E4DA", 0.6)
    elif o == "hoodie":
        b = rect_path(-0.025 * W, hem, W + 0.025 * W, 0.295 * W)
        F.fs(b, P.HOODIE)
        F.hatch(b, 0.10, kind="cross", scale=0.45)
        F.fs(rect_path(-0.025 * W, hem, W + 0.025 * W, hem + 0.035 * W), P.HOODIE_DARK, 0.8)
        # the hood, lying flat on the back with its lining showing at the top
        hood = skia.Path()
        hood.moveTo(0.16 * W, 0.335 * W)
        hood.cubicTo(0.15 * W, 0.20 * W, 0.30 * W, 0.075 * W, 0.5 * W, 0.045 * W)
        hood.cubicTo(0.70 * W, 0.075 * W, 0.85 * W, 0.20 * W, 0.84 * W, 0.335 * W)
        hood.cubicTo(0.70 * W, 0.37 * W, 0.30 * W, 0.37 * W, 0.16 * W, 0.335 * W)
        hood.close()
        F.fs(hood, P.HOODIE_LIGHT)
        F.hatch(hood, 0.10, kind="diag", scale=0.4)
        lining = skia.Path()
        lining.moveTo(0.22 * W, 0.33 * W)
        lining.cubicTo(0.30 * W, 0.255 * W, 0.70 * W, 0.255 * W, 0.78 * W, 0.33 * W)
        lining.cubicTo(0.66 * W, 0.355 * W, 0.34 * W, 0.355 * W, 0.22 * W, 0.33 * W)
        lining.close()
        F.fs(lining, P.HOODIE_DARK, 0.8)
        F.line(0.5 * W, 0.06 * W, 0.5 * W, 0.26 * W, 0.6, P.HOODIE_DARK)
    elif o == "uniform":
        b = rect_path(-0.03 * W, hem, W + 0.03 * W, 0.31 * W)
        F.fs(b, P.BLAZER)
        F.fs(rect_path(-0.03 * W, 0.265 * W, W + 0.03 * W, 0.33 * W), P.BLAZER_DARK, 0.8)
        # backpack
        bp = rrect_path(0.13 * W, -0.05 * W, 0.87 * W, 0.42 * W, 0.08 * W)
        F.fs(bp, P.BACKPACK, 1.1)
        F.hatch(rect_path(0.6 * W, -0.05 * W, 0.87 * W, 0.42 * W), 0.12, scale=0.4)
        handle = skia.Path()
        handle.moveTo(0.42 * W, 0.415 * W)
        handle.cubicTo(0.43 * W, 0.48 * W, 0.57 * W, 0.48 * W, 0.58 * W, 0.415 * W)
        F.stroke(handle, 2.2, P.BACKPACK_DARK)
        F.stroke(handle, 0.6)
        pk = rrect_path(0.23 * W, -0.02 * W, 0.77 * W, 0.17 * W, 0.05 * W)
        F.fs(pk, P.BACKPACK_DARK, 0.9)
        zp = skia.Path()
        zp.moveTo(0.18 * W, 0.30 * W)
        zp.cubicTo(0.22 * W, 0.38 * W, 0.78 * W, 0.38 * W, 0.82 * W, 0.30 * W)
        F.stroke(zp, 0.9)
        # tiny clawd keychain
        F.line(0.74 * W, 0.335 * W, 0.76 * W, 0.27 * W, 0.6)
        F.fs(rect_path(0.73 * W, 0.215 * W, 0.80 * W, 0.265 * W), P.ORANGE, 0.6)
        F.fill(rect_path(0.745 * W, 0.24 * W, 0.752 * W, 0.255 * W), P.INK)
        F.fill(rect_path(0.778 * W, 0.24 * W, 0.785 * W, 0.255 * W), P.INK)


def outfit_side(F, st, W, H, D, back_at_zero):
    o = st.outfit
    hem = _hem(W)

    def ub(u):  # distance from the back edge -> local u
        return u if back_at_zero else D - u

    if o == "labcoat":
        b = rect_path(ub(-0.02 * W), hem, ub(D + 0.02 * W), 0.30 * W)
        F.fs(b, P.COAT)
        F.hatch(b, 0.14, scale=0.5)
        F.line(ub(D * 0.5), hem, ub(D * 0.5), 0.29 * W, 0.6, P.INK_SOFT)
    elif o == "hoodie":
        b = rect_path(ub(-0.02 * W), hem, ub(D + 0.02 * W), 0.29 * W)
        F.fs(b, P.HOODIE)
        F.hatch(b, 0.14, kind="cross", scale=0.45)
        hood = rrect_path(ub(-0.075 * W), 0.08 * W, ub(0.07 * W), 0.39 * W, 0.06 * W)
        F.fs(hood, P.HOODIE_LIGHT)
    elif o == "uniform":
        b = rect_path(ub(-0.02 * W), hem, ub(D + 0.02 * W), 0.30 * W)
        F.fs(b, P.BLAZER)
        bp = rrect_path(ub(-0.16 * W), -0.05 * W, ub(0.02 * W), 0.42 * W, 0.05 * W)
        F.fs(bp, P.BACKPACK, 1.1)
        F.fs(rect_path(ub(0.02 * W), hem, ub(0.08 * W), 0.30 * W), P.BACKPACK, 0.8)


def goggles(F, st, W, H, kind):
    """Opus's brass goggles: strap all round, lenses on the front face."""
    gd = clamp(st.goggles_down)
    vs = lerp(H - 0.06 * W, 0.415 * W, gd)
    if kind in ("front", "back", "side"):
        F.fs(rect_path(-0.01 * F.width, vs - 0.022 * W, F.width * 1.01, vs + 0.022 * W), "#4B3A36", 0.8)
    if kind == "back":
        F.fs(rrect_path(F.width * 0.44, vs - 0.035 * W, F.width * 0.56, vs + 0.035 * W, 0.01 * W), P.BRASS, 0.7)
    if kind == "front":
        vl = lerp(H - 0.025 * W, 0.415 * W, gd)
        for u in (0.29 * W, 0.71 * W):
            u2 = lerp(u, 0.2083 * W if u < 0.5 * W else 0.7917 * W, gd)
            F.fs(circle_path(u2, vl, 0.098 * W), P.BRASS, 1.0)
            F.fs(circle_path(u2, vl, 0.068 * W), P.GLASS, 0.8)
            F.fill(circle_path(u2, vl, 0.068 * W), P.GLASS_DEEP, 0.35 + 0.4 * gd)
            F.stroke(poly_path([(u2 - 0.035 * W, vl + 0.012 * W), (u2 - 0.005 * W, vl + 0.045 * W)], close=False), 1.4, "#FFFFFF")
            F.fs(circle_path(u2 + 0.075 * W, vl + 0.045 * W, 0.009 * W), P.BRASS_LIGHT, 0.4)
        F.stroke(poly_path([(0.29 * W + 0.098 * W, vl), (0.71 * W - 0.098 * W, vl)], close=False), 2.2, P.BRASS)


def glasses(F, st, W, H):
    ew, eh = 0.0833 * W, 0.1667 * W
    vc = 0.4167 * W
    lx = 0.035 * W * st.look[0]
    pad_w, pad_h = 0.045 * W, 0.032 * W
    frames = []
    for uc in (0.2083 * W, 0.7917 * W):
        u = uc + lx * 0.5
        r = rrect_path(u - ew / 2 - pad_w, vc - eh / 2 - pad_h, u + ew / 2 + pad_w, vc + eh / 2 + pad_h, 0.03 * W)
        frames.append((u, r))
        F.fill(r, "#DDEBFF", 0.18)
    for u, r in frames:
        F.stroke(r, 2.6)
    F.line(frames[0][0] + ew / 2 + pad_w, vc + 0.03 * W, frames[1][0] - ew / 2 - pad_w, vc + 0.03 * W, 2.2)
    F.line(0.0, vc + 0.04 * W, frames[0][0] - ew / 2 - pad_w, vc + 0.03 * W, 1.6)
    F.line(W, vc + 0.04 * W, frames[1][0] + ew / 2 + pad_w, vc + 0.03 * W, 1.6)
    if 0.0 <= st.glint <= 1.0:
        for u, r in frames:
            F.begin()
            F.g.c.clipPath(r, skia.ClipOp.kIntersect, True)
            x = lerp(u - 0.15 * W, u + 0.15 * W, st.glint)
            F.g.fill(poly_path([(x - 0.02 * W, vc - 0.15 * W), (x + 0.015 * W, vc - 0.15 * W),
                                (x + 0.045 * W, vc + 0.15 * W), (x + 0.01 * W, vc + 0.15 * W)]), "#FFFFFF", 0.9)
            F.end()


# ------------------------------------------------------------------ props

def draw_clipboard(g, st, x, y, W, sx=1.0):
    """Clipboard centred at clawd-local (x, y)."""
    cw, ch = 0.36 * W, 0.46 * W
    c = g.c
    c.save()
    c.translate(x, y)
    c.scale(sx, 1.0)
    c.rotate(4)
    x0, y0 = -cw / 2, -ch / 2
    g.fs(rrect_path(x0, y0, x0 + cw, y0 + ch, 0.03 * W), "#A8734A", 1.1)
    g.hatch(rrect_path(x0, y0, x0 + cw, y0 + ch, 0.03 * W), P.INK, 0.12, scale=0.3)
    px0, py0, px1, py1 = x0 + 0.03 * W, y0 + 0.03 * W, x0 + cw - 0.03 * W, y0 + ch - 0.06 * W
    g.fs(rect_path(px0, py0, px1, py1), P.PAPER, 0.7)
    if sx > 0:
        g.text("TO-DO", 0.0, py1 - 0.055 * W, 0.05 * W, P.INK, "hand")
        items = ("Exit pod", "Wake Sonnet 5.5", "Wake Haiku 5.5", "Check on Fable 5.5")
        lh = 0.068 * W
        for i, it in enumerate(items):
            yy = py1 - 0.11 * W - i * lh
            bx = px0 + 0.02 * W
            g.stroke(rect_path(bx, yy - 0.004 * W, bx + 0.028 * W, yy + 0.024 * W), 0.7)
            g.text(it, bx + 0.042 * W, yy, 0.035 * W, P.INK, "hand", align="left")
            p = clamp(st.checks[i]) if i < len(st.checks) else 0.0
            if p > 0:
                a = (bx + 0.003 * W, yy + 0.012 * W)
                b = (bx + 0.013 * W, yy - 0.002 * W)
                cc = (bx + 0.042 * W, yy + 0.04 * W)
                if p < 0.4:
                    t = p / 0.4
                    pts = [a, (lerp(a[0], b[0], t), lerp(a[1], b[1], t))]
                else:
                    t = (p - 0.4) / 0.6
                    pts = [a, b, (lerp(b[0], cc[0], t), lerp(b[1], cc[1], t))]
                g.stroke(poly_path(pts, close=False), 1.6, "#C8402F")
        if st.circle4 > 0:
            yy = py1 - 0.11 * W - 3 * lh + 0.01 * W
            rx, ry = 0.15 * W, 0.028 * W
            cxx = px0 + 0.045 * W + rx * 0.95
            p = skia.Path()
            n = 40
            sweep = clamp(st.circle4) * 1.1
            pts = []
            for k in range(n + 1):
                a = math.pi * 0.9 + 2 * math.pi * sweep * k / n
                pts.append((cxx + rx * math.cos(a), yy + ry * math.sin(a) + 0.004 * W * k / n))
            g.stroke(poly_path(pts, close=False), 1.4, "#C8402F")
    if st.pencil >= 0 and sx > 0:
        lh = 0.068 * W
        i = st.pencil
        yy = py1 - 0.11 * W - i * lh
        tipx = px0 + 0.02 * W + 0.03 * W + 0.01 * W * math.sin(i * 40)
        if st.circle4 > 0:
            ang = math.pi * 0.9 + 2 * math.pi * clamp(st.circle4) * 1.1
            tipx = px0 + 0.045 * W + 0.15 * W * 0.95 + 0.15 * W * math.cos(ang)
            yy = yy + 0.01 * W + 0.028 * W * math.sin(ang)
        c.save()
        c.translate(tipx, yy + 0.01 * W)
        c.rotate(-35)
        L, R = 0.30 * W, 0.016 * W
        g.fs(poly_path([(0, 0), (0.035 * W, R), (0.035 * W, -R)]), "#F2D3A6", 0.6)
        g.fill(poly_path([(0, 0), (0.012 * W, R * 0.35), (0.012 * W, -R * 0.35)]), "#C8402F")
        g.fs(rect_path(0.035 * W, -R, L, R), "#C8402F", 0.7)
        g.fs(rect_path(L, -R, L + 0.03 * W, R), "#E8A0A8", 0.6)
        c.restore()
    # clip
    g.fs(rrect_path(-0.08 * W, y0 + ch - 0.075 * W, 0.08 * W, y0 + ch - 0.005 * W, 0.015 * W), "#C9CDD3", 0.9)
    g.fs(circle_path(0.0, y0 + ch - 0.03 * W, 0.012 * W), "#8E949E", 0.5)
    c.restore()


CODE = ("{ }", "</>", "=>", "( )", ";", "[ ]", "fn", "&&", "++", "#!")


def draw_laptop(g, st, x, y, W, t):
    """Open laptop, lid facing the camera (Sonnet holds it in front)."""
    lw, lh = 0.50 * W, 0.33 * W
    o = clamp(st.laptop_open)
    jig = 0.004 * W * math.sin(t * 60) * st.typing
    c = g.c
    c.save()
    c.translate(x, y + jig)
    # base
    g.fs(rrect_path(-lw / 2 - 0.02 * W, -0.035 * W, lw / 2 + 0.02 * W, 0.0, 0.01 * W), "#9AA0AA", 0.9)
    hgt = lh * (0.12 + 0.88 * o)
    lid = rrect_path(-lw / 2, 0.0, lw / 2, hgt, 0.02 * W)
    g.fill_shader(lid, g.lin_grad(-lw / 2, 0, lw / 2, hgt, ["#E2E5EA", "#B9BEC6"]))
    g.stroke(lid, 1.0)
    if o > 0.6:
        k = (o - 0.6) / 0.4
        # stickers
        g.fs(rrect_path(-0.17 * W, hgt * 0.52, -0.05 * W, hgt * 0.80, 0.02 * W), P.ORANGE, 0.6, a=k)
        g.fill(rect_path(-0.14 * W, hgt * 0.63, -0.128 * W, hgt * 0.71), P.INK, a=k)
        g.fill(rect_path(-0.093 * W, hgt * 0.63, -0.081 * W, hgt * 0.71), P.INK, a=k)
        g.fs(rrect_path(0.04 * W, hgt * 0.22, 0.19 * W, hgt * 0.42, 0.015 * W), P.INK, 0.4, a=k)
        g.text("LGTM", 0.115 * W, hgt * 0.27, 0.042 * W, P.CODE_GREEN, "mono", a=k)
        g.fs(circle_path(0.12 * W, hgt * 0.72, 0.045 * W), "#3E6FD8", 0.5, a=k)
        g.text("</>", 0.12 * W, hgt * 0.72 - 0.014 * W, 0.034 * W, "#FFFFFF", "mono", a=k)
        g.fs(rrect_path(-0.2 * W, hgt * 0.12, -0.08 * W, hgt * 0.36, 0.01 * W), "#F2EFE6", 0.4, a=k)
        g.text("✓", -0.14 * W, hgt * 0.15, 0.06 * W, P.CODE_GREEN, "sans", a=k)
    c.restore()


# ------------------------------------------------------------------ main

def draw_clawd(g, st, t=0.0):
    if st.alpha <= 0.01:
        return
    W = st.W
    H = 0.6667 * W
    D = 0.62 * W
    LEG = 0.1667 * W
    th = st.theta
    c, s = math.cos(th), math.sin(th)

    # gait
    bob = abs(math.sin(st.walk)) * 0.035 * W * st.walk_amp

    if st.shadow:
        sw = (W * abs(c) + D * abs(s)) * 0.62
        k = 1.0 / (1.0 + 2.2 * max(st.lift, 0) / W)
        g.shadow(st.x, st.y, sw * k, 0.07 * W * k, 0.26 * k)

    cv = g.c
    if st.alpha < 1:
        cv.saveLayerAlpha(None, int(255 * st.alpha))
    cv.save()
    cv.translate(st.x, st.y + st.lift + bob)
    cy = LEG + H / 2
    if st.tilt:
        cv.translate(0, cy)
        cv.rotate(-st.tilt)
        cv.translate(0, -cy)
    if st.sq != 1.0:
        cv.scale(st.sq, 1.0 / st.sq)

    items = []

    # legs
    for i, lx in enumerate(LEGS):
        xc = lx * W * c
        dep = -lx * W * s - 0.001
        ph = st.walk + (0 if i in (0, 2) else math.pi)
        lift = max(0.0, math.sin(ph)) * 0.08 * W * st.walk_amp
        swing = math.cos(ph) * 0.035 * W * st.walk_amp * (1 if s >= 0 else -1) * min(1, abs(s) + 0.3)
        lwid = 0.0833 * W * (abs(c) + 0.9 * abs(s))

        def leg(xc=xc, lift=lift, swing=swing, lwid=lwid):
            p = rrect_path(xc + swing - lwid / 2, lift - bob * 0.5, xc + swing + lwid / 2, LEG + 0.04 * W, 0.012 * W)
            g.fs(p, P.ORANGE_DARK, 1.0)
        items.append((dep - 10, leg))

    # body faces
    faces = []
    if c > 0.02:
        faces.append(("front", c * D / 2, Face(g, c, -W / 2 * c + D / 2 * s, LEG, W, "front")))
    if c < -0.02:
        faces.append(("back", -c * D / 2, Face(g, -c, W / 2 * c - D / 2 * s, LEG, W, "back")))
    if s > 0.02:
        faces.append(("side_r", s * W / 2, Face(g, s, -W / 2 * c - D / 2 * s, LEG, D, "side", SIDE)))
    if s < -0.02:
        faces.append(("side_l", -s * W / 2, Face(g, -s, W / 2 * c + D / 2 * s, LEG, D, "side", SIDE)))

    for kind, dep, F in faces:
        def draw_face(kind=kind, F=F):
            body = rrect_path(0, 0, F.width, H, 0.035 * W)
            base = P.ORANGE
            F.fill(body, base)
            # soft top-light gradient
            sh = g.lin_grad(0, H, 0, 0, [shade(base, 1.10 * F.bright), shade(base, 0.93 * F.bright)])
            g.fill_shader(F.tp(body), sh, 0.55)
            if kind.startswith("side"):
                F.hatch(body, 0.13, scale=0.5)
            else:
                F.hatch(rect_path(0, 0, F.width, H * 0.45), 0.07, scale=0.5)
            if st.screen_glow > 0 and kind == "front":
                g.fill(F.tp(rect_path(0.05 * W, H * 0.25, W * 0.95, H * 0.95)), P.SCREEN, 0.10 * st.screen_glow)
            if kind == "front":
                draw_eyes(F, st, W, H)
                outfit_front(F, st, W, H)
                if st.outfit == "hoodie":
                    glasses(F, st, W, H)
            elif kind == "back":
                outfit_back(F, st, W, H)
            else:
                outfit_side(F, st, W, H, D, back_at_zero=(kind == "side_r"))
            if st.outfit == "labcoat":
                goggles(F, st, W, H, "front" if kind == "front" else ("back" if kind == "back" else "side"))
            F.stroke(body, 1.25)
            if st.rim > 0:
                rimp = rect_path(0, H * 0.55, F.width, H)
                g.soft(F.tp(rimp), P.GLOW, 0.02 * W, 0.18 * st.rim, skia.BlendMode.kPlus)
                g.stroke(F.tp(poly_path([(0, H), (F.width, H)], close=False)), 2.2, P.GLOW, 0.8 * st.rim)
        items.append((dep, draw_face))

    # arms
    a = 0.1667 * W
    for side, raise_ in ((-1, st.arm_l), (1, st.arm_r)):
        xa = side * (W / 2 + a / 2 - 0.01 * W)
        r = clamp(raise_, -0.5, 1.6)
        xc = xa * c
        dep = -xa * s
        yc = LEG + 0.25 * W + r * 0.24 * W
        ang = side * r * 38 * (1 if c >= 0 else -1)
        aw = a * (abs(c) + 0.85 * abs(s))

        def arm(side=side, xc=xc, yc=yc, ang=ang, aw=aw, dep=dep):
            cv.save()
            cv.translate(xc - side * a * 0.45 * abs(c) * (1 if c >= 0 else -1), yc)
            cv.rotate(-ang)
            dirx = side * (1 if c >= 0 else -1)
            x0 = -aw / 2 + side * a * 0.45 * abs(c) * (1 if c >= 0 else -1)
            box = rrect_path(x0, -a / 2, x0 + aw, a / 2, 0.02 * W)
            g.fs(box, P.ORANGE if dep >= 0 else P.ORANGE_DARK, 1.0)
            sleeve = {"labcoat": P.COAT, "hoodie": P.HOODIE, "uniform": P.BLAZER}.get(st.outfit)
            if sleeve and abs(c) > 0.15:
                sl = aw * 0.58
                if dirx > 0:
                    sp = rrect_path(x0 - 0.02 * W, -a / 2 - 0.012 * W, x0 + sl, a / 2 + 0.012 * W, 0.02 * W)
                else:
                    sp = rrect_path(x0 + aw - sl, -a / 2 - 0.012 * W, x0 + aw + 0.02 * W, a / 2 + 0.012 * W, 0.02 * W)
                g.fs(sp, sleeve, 1.0)
                cuff = {"hoodie": P.HOODIE_DARK, "uniform": P.SHIRT}.get(st.outfit)
                if cuff:
                    cx0 = x0 + sl - 0.03 * W if dirx > 0 else x0 + aw - sl
                    g.fs(rect_path(cx0, -a / 2 - 0.012 * W, cx0 + 0.03 * W, a / 2 + 0.012 * W), cuff, 0.7)
            cv.restore()
        items.append((dep, arm))

    # props
    if st.prop == "clipboard":
        cx_, cy_ = clipboard_offset(st)
        dep = (W / 2) * s + 0.2 * W * c

        def clip(cx_=cx_, cy_=cy_):
            if c > 0:
                draw_clipboard(g, st, cx_, cy_, W, sx=max(0.25, c))
        items.append((dep + 0.5, clip))
    elif st.prop == "laptop":
        dep = 0.5 * D * c + 0.1

        def lap():
            if st.laptop_open > 0 and c > 0.2:
                draw_laptop(g, st, 0.0 + 0.0 * s, LEG + 0.02 * W, W, t)
            elif c > -0.5:
                # tucked under the screen-left arm
                xa = -(W / 2 + 0.02 * W) * c
                cv.save()
                cv.translate(xa, LEG + 0.16 * W)
                cv.rotate(-8)
                g.fs(rrect_path(-0.20 * W * max(abs(c), 0.4), -0.025 * W, 0.20 * W * max(abs(c), 0.4), 0.025 * W, 0.01 * W),
                     "#B9BEC6", 1.0)
                cv.restore()
        items.append((dep, lap))

    items.sort(key=lambda it: it[0])
    for _, fn in items:
        fn()

    cv.restore()
    if st.alpha < 1:
        cv.restore()


def clawd_height(W):
    return 0.8333 * W


def clipboard_offset(st):
    """Clipboard centre in Clawd-local coords (held by the screen-left stub)."""
    W = st.W
    a = 0.1667 * W
    c = math.cos(st.theta)
    return -(W / 2 + a * 0.55) * c - 0.05 * W * c, 0.1667 * W + 0.27 * W + st.arm_l * 0.24 * W


def clipboard_world(st):
    ox, oy = clipboard_offset(st)
    bob = abs(math.sin(st.walk)) * 0.035 * st.W * st.walk_amp
    return st.x + ox * st.sq, st.y + st.lift + bob + oy / st.sq
