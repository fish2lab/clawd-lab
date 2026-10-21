"""Thin drawing layer over skia: world-space camera, colours, paths,
outlined fills, hatching and text.

World units are y-up.  One unit is roughly the body width of Opus.
"""
import math
import os
from functools import lru_cache

import numpy as np
import skia

from . import palette as P

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT_DIR = os.path.join(ROOT, "assets", "fonts")


# ---------------------------------------------------------------- colour

def rgb(c):
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def col(c, a=1.0):
    """'#rrggbb' or (r,g,b) floats -> skia ARGB int."""
    if isinstance(c, str):
        r, g, b = rgb(c)
    else:
        r, g, b = c[:3]
    a = max(0.0, min(1.0, a))
    return skia.ColorSetARGB(int(a * 255 + 0.5), int(r * 255 + 0.5), int(g * 255 + 0.5), int(b * 255 + 0.5))


def mix(c1, c2, t):
    a = rgb(c1) if isinstance(c1, str) else c1
    b = rgb(c2) if isinstance(c2, str) else c2
    t = max(0.0, min(1.0, t))
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))


def shade(c, f):
    a = rgb(c) if isinstance(c, str) else c
    if f < 1:
        return tuple(v * f for v in a)
    return tuple(v + (1 - v) * (f - 1) for v in a)


# ---------------------------------------------------------------- fonts

@lru_cache(None)
def typeface(name):
    files = {
        "slab": "ZillaSlab-Bold.ttf",
        "hand": "Caveat.ttf",
        "zh": "ZCOOLQingKeHuangYou-Regular.ttf",
    }
    if name in files:
        return skia.Typeface.MakeFromFile(os.path.join(FONT_DIR, files[name]))
    if name == "mono":
        return skia.Typeface("JetBrains Mono", skia.FontStyle.Bold())
    if name == "sans":
        return skia.Typeface("Inter", skia.FontStyle.Bold())
    return skia.Typeface(name)


# ---------------------------------------------------------------- paths

def rect_path(x0, y0, x1, y1):
    p = skia.Path()
    p.addRect(skia.Rect(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)))
    return p


def rrect_path(x0, y0, x1, y1, r):
    p = skia.Path()
    r = max(0.0, min(r, abs(x1 - x0) / 2, abs(y1 - y0) / 2))
    p.addRRect(skia.RRect.MakeRectXY(skia.Rect(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)), r, r))
    return p


def circle_path(x, y, r):
    p = skia.Path()
    p.addCircle(x, y, max(r, 1e-5))
    return p


def ellipse_path(x, y, rx, ry):
    p = skia.Path()
    p.addOval(skia.Rect(x - rx, y - ry, x + rx, y + ry))
    return p


def poly_path(pts, close=True):
    p = skia.Path()
    p.moveTo(*pts[0])
    for q in pts[1:]:
        p.lineTo(*q)
    if close:
        p.close()
    return p


def smooth_path(pts, close=True, tension=0.5):
    """Catmull-Rom through points -> cubic path."""
    n = len(pts)
    p = skia.Path()
    p.moveTo(*pts[0])
    rng = range(n) if close else range(n - 1)
    for i in rng:
        p0 = pts[(i - 1) % n] if (close or i > 0) else pts[i]
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        p3 = pts[(i + 2) % n] if (close or i + 2 < n) else p2
        k = tension / 3.0 * 2
        c1 = (p1[0] + (p2[0] - p0[0]) * k / 2, p1[1] + (p2[1] - p0[1]) * k / 2)
        c2 = (p2[0] - (p3[0] - p1[0]) * k / 2, p2[1] - (p3[1] - p1[1]) * k / 2)
        p.cubicTo(*c1, *c2, *p2)
    if close:
        p.close()
    return p


def path_union(paths):
    if not paths:
        return skia.Path()
    out = paths[0]
    for q in paths[1:]:
        r = skia.Op(out, q, skia.PathOp.kUnion_PathOp)
        out = r if r is not None else out
    return out


def path_diff(a, b):
    r = skia.Op(a, b, skia.PathOp.kDifference_PathOp)
    return r if r is not None else a


def path_isect(a, b):
    r = skia.Op(a, b, skia.PathOp.kIntersect_PathOp)
    return r if r is not None else skia.Path()


# ---------------------------------------------------------------- hatch tile

@lru_cache(None)
def hatch_image(kind="diag"):
    n = 96
    rng = np.random.default_rng(7 if kind == "diag" else 11)
    a = np.zeros((n, n), np.float32)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    if kind == "diag":
        d = (xx + yy) % 12.0
        w = 1.25 + 0.5 * np.sin(xx * 0.21 + rng.random() * 6)
        a = np.clip(1.4 - np.abs(d - 6.0) / w, 0, 1)
    elif kind == "cross":
        d1 = (xx + yy) % 12.0
        d2 = (xx - yy) % 12.0
        a = np.maximum(np.clip(1.3 - np.abs(d1 - 6.0) / 1.2, 0, 1), 0.75 * np.clip(1.3 - np.abs(d2 - 6.0) / 1.1, 0, 1))
    elif kind == "vert":
        d = xx % 9.0
        a = np.clip(1.3 - np.abs(d - 4.5) / 1.0, 0, 1) * (0.7 + 0.3 * np.sin(yy * 0.13 + xx))
    # pencil breakup
    a *= 0.65 + 0.35 * rng.random((n, n)).astype(np.float32)
    img = np.zeros((n, n, 4), np.uint8)
    img[..., 3] = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    return skia.Image.fromarray(img, colorType=skia.ColorType.kRGBA_8888_ColorType,
                                alphaType=skia.AlphaType.kUnpremul_AlphaType)


# ---------------------------------------------------------------- camera / context

class Cam:
    def __init__(self, cx, cy, h, rot=0.0):
        self.cx, self.cy, self.h, self.rot = cx, cy, h, rot


class G:
    """Per-frame drawing context."""

    def __init__(self, canvas, wpx, hpx, cam, t=0.0):
        self.c = canvas
        self.wpx, self.hpx = wpx, hpx
        self.cam = cam
        self.t = t
        self.s = hpx / cam.h               # px per world unit
        self.zoomk = (cam.h / 4.0) ** 0.38  # world line width factor
        self.view_w = cam.h * wpx / hpx

    # world transform --------------------------------------------------
    def begin_world(self):
        c = self.c
        c.save()
        c.translate(self.wpx / 2, self.hpx / 2)
        if self.cam.rot:
            c.rotate(self.cam.rot)
        c.scale(self.s, -self.s)
        c.translate(-self.cam.cx, -self.cam.cy)

    def end(self):
        self.c.restore()

    def visible(self, x0, y0, x1, y1, pad=0.5):
        cx, cy = self.cam.cx, self.cam.cy
        hw, hh = self.view_w / 2 + pad, self.cam.h / 2 + pad
        return not (x1 < cx - hw or x0 > cx + hw or y1 < cy - hh or y0 > cy + hh)

    def lw(self, w=1.0):
        """Outline width in world units (scales sub-linearly with zoom)."""
        return 0.016 * w * self.zoomk

    # paints -----------------------------------------------------------
    @staticmethod
    def paint_fill(color, a=1.0):
        return skia.Paint(AntiAlias=True, Color=col(color, a))

    def paint_stroke(self, color=P.INK, w=1.0, a=1.0, cap=True):
        return skia.Paint(AntiAlias=True, Color=col(color, a), Style=skia.Paint.kStroke_Style,
                          StrokeWidth=self.lw(w), StrokeJoin=skia.Paint.kRound_Join,
                          StrokeCap=skia.Paint.kRound_Cap if cap else skia.Paint.kButt_Cap)

    # primitives -------------------------------------------------------
    def fill(self, path, color, a=1.0):
        self.c.drawPath(path, self.paint_fill(color, a))

    def fill_shader(self, path, shader, a=1.0):
        p = skia.Paint(AntiAlias=True)
        p.setShader(shader)
        if a < 1:
            p.setAlphaf(a)
        self.c.drawPath(path, p)

    def stroke(self, path, w=1.0, color=P.INK, a=1.0, cap=True):
        self.c.drawPath(path, self.paint_stroke(color, w, a, cap))

    def fs(self, path, color, w=1.0, a=1.0, ink=P.INK):
        self.fill(path, color, a)
        if w > 0:
            self.stroke(path, w, ink, a)

    def line(self, x0, y0, x1, y1, w=1.0, color=P.INK, a=1.0):
        self.c.drawLine(x0, y0, x1, y1, self.paint_stroke(color, w, a))

    def lin_grad(self, x0, y0, x1, y1, colors, pos=None, a=1.0):
        cs = [col(cc, a) for cc in colors]
        return skia.GradientShader.MakeLinear([skia.Point(x0, y0), skia.Point(x1, y1)], cs, pos)

    def rad_grad(self, x, y, r, colors, pos=None, alphas=None):
        if alphas is None:
            alphas = [1.0] * len(colors)
        cs = [col(cc, aa) for cc, aa in zip(colors, alphas)]
        return skia.GradientShader.MakeRadial(skia.Point(x, y), max(r, 1e-4), cs, pos)

    def hatch(self, path, color=P.INK, a=0.25, kind="diag", scale=1.0, rot=0.0):
        """Fill path with a pencil-hatch pattern anchored in world space."""
        img = hatch_image(kind)
        tile = 0.30 * scale * self.zoomk ** 0.6  # world size of one tile
        m = skia.Matrix()
        m.setScale(tile / img.width(), -tile / img.height())
        if rot:
            m.postRotate(rot)
        sh = img.makeShader(skia.TileMode.kRepeat, skia.TileMode.kRepeat,
                            skia.SamplingOptions(skia.FilterMode.kLinear), m)
        p = skia.Paint(AntiAlias=True)
        p.setShader(skia.Shaders.Blend(skia.BlendMode.kSrcIn, sh, skia.Shaders.Color(col(color))))
        p.setAlphaf(max(0.0, min(1.0, a)))
        self.c.drawPath(path, p)

    def glow(self, path, color, sigma, a=1.0, plus=True):
        p = skia.Paint(AntiAlias=True, Color=col(color, a))
        p.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, max(sigma, 1e-4)))
        if plus:
            p.setBlendMode(skia.BlendMode.kPlus)
        self.c.drawPath(path, p)

    def soft(self, path, color, sigma, a=1.0, blend=None):
        p = skia.Paint(AntiAlias=True, Color=col(color, a))
        p.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, max(sigma, 1e-4)))
        if blend is not None:
            p.setBlendMode(blend)
        self.c.drawPath(path, p)

    def text(self, s, x, y, size, color=P.INK, font="slab", align="center", a=1.0, rot=0.0,
             outline=None, ow=0.0, scale_x=1.0):
        """Draw text with baseline at world (x, y)."""
        f = skia.Font(typeface(font), 100.0)
        f.setSubpixel(True)
        f.setEdging(skia.Font.Edging.kAntiAlias)
        k = size / 100.0
        w = f.measureText(s)
        dx = {"center": -w / 2, "left": 0.0, "right": -w}[align]
        c = self.c
        c.save()
        c.translate(x, y)
        if rot:
            c.rotate(rot)
        c.scale(k * scale_x, -k)
        if outline is not None:
            sp = skia.Paint(AntiAlias=True, Color=col(outline, a), Style=skia.Paint.kStroke_Style,
                            StrokeWidth=ow / k, StrokeJoin=skia.Paint.kRound_Join)
            c.drawString(s, dx, 0, f, sp)
        c.drawString(s, dx, 0, f, skia.Paint(AntiAlias=True, Color=col(color, a)))
        c.restore()
        return w * k * scale_x

    def text_width(self, s, size, font="slab"):
        f = skia.Font(typeface(font), 100.0)
        return f.measureText(s) * size / 100.0

    def shadow(self, x, y, rx, ry, a=0.22):
        p = ellipse_path(x, y, rx, ry)
        self.soft(p, P.INK, ry * 0.45, a)


# ---------------------------------------------------------------- misc math

def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def lerp(a, b, t):
    return a + (b - a) * t


def smooth(t):
    t = clamp(t)
    return t * t * (3 - 2 * t)


def ease_io(t):
    t = clamp(t)
    return 4 * t * t * t if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def ease_out(t, p=3):
    t = clamp(t)
    return 1 - (1 - t) ** p


def ease_in(t, p=2):
    t = clamp(t)
    return t ** p


def back_out(t, s=1.7):
    t = clamp(t)
    t -= 1
    return t * t * ((s + 1) * t + s) + 1


def elastic_out(t):
    t = clamp(t)
    if t in (0, 1):
        return t
    return 2 ** (-10 * t) * math.sin((t * 10 - 0.75) * (2 * math.pi) / 3) + 1


def seg(t, t0, t1):
    if t1 <= t0:
        return 1.0 if t >= t1 else 0.0
    return clamp((t - t0) / (t1 - t0))


def hash01(*args):
    """Deterministic hash -> [0, 1) (FNV-1a + murmur3 finaliser)."""
    h = 2166136261
    for a in args:
        for ch in repr(a).encode():
            h = ((h ^ ch) * 16777619) & 0xFFFFFFFF
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & 0xFFFFFFFF
    h ^= h >> 16
    return (h & 0xFFFFFF) / float(0x1000000)


def vnoise(x, seed=0):
    """1-D smooth value noise in [-1, 1]."""
    i = math.floor(x)
    f = x - i
    a = hash01(seed, i) * 2 - 1
    b = hash01(seed, i + 1) * 2 - 1
    u = f * f * (3 - 2 * f)
    return a + (b - a) * u
