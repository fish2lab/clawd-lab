#!/usr/bin/env python3
"""Cover art (16:9 and 4:3) staged in the same world as the film.

  python make_covers.py            # -> out/covers/*.png + *.jpg
"""
import os
import sys

import skia
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from clawdlab import story, palette as P  # noqa: E402
from clawdlab import scene as S  # noqa: E402
from clawdlab.gfx import Cam, col, typeface  # noqa: E402
from clawdlab.render import render  # noqa: E402

XR = S.XR
T_KEY = 28.95          # eyes have just lit up; Haiku is still tapping


def key_state(cam, cracks=0.0):
    st = story.build(T_KEY)
    st.cam = cam
    st.shake, st.cam_rot = (0.0, 0.0), 0.0
    st.alarm, st.flash, st.night = 0.0, 0.0, 0.62
    st.lamps.update(fable_l=0.25, fable_r=0.25)
    rs = st.reactor
    rs.update(eyes=1.0, eye_open=1.0, angry=0.45, lunge=0.06, cracks=cracks, flash=0.0, vapour=0.45)
    h = next(c for c in st.chars if c.name == "haiku")
    h.arm_r = 1.35
    st.fx_front = [lambda g: story.fx_alert(g, T_KEY, h.x + 0.36, h.y + h.lift + 0.78, 0.0, T_KEY + 1, 0.46)]
    return st


# ------------------------------------------------------------------ type

def zh_font(px):
    f = skia.Font(typeface("zh"), px)
    f.setEdging(skia.Font.Edging.kAntiAlias)
    return f


def draw_text(cv, s, x, y, px, fill, stroke=None, sw=0.0, align="left"):
    f = zh_font(px)
    w = f.measureText(s)
    dx = {"left": 0, "center": -w / 2, "right": -w}[align]
    if stroke:
        sp = skia.Paint(AntiAlias=True, Color=col(stroke), Style=skia.Paint.kStroke_Style, StrokeWidth=sw,
                        StrokeJoin=skia.Paint.kRound_Join)
        cv.drawString(s, x + dx, y, f, sp)
    cv.drawString(s, x + dx, y, f, skia.Paint(AntiAlias=True, Color=col(fill)))
    return w


def spaced(cv, s, x, y, px, fill, font="slab", track=0.18):
    """Letter-spaced Latin caps (poster credit-line style)."""
    f = skia.Font(typeface(font), px)
    paint = skia.Paint(AntiAlias=True, Color=col(fill))
    for ch in s:
        cv.drawString(ch, x, y, f, paint)
        x += f.measureText(ch) + px * track
    return x


def poster_text(cv, s, x, y, px, font="zh", k=1.0):
    """Gold title lettering with an ink outline and a warm glow."""
    f = skia.Font(typeface(font), px)
    f.setEdging(skia.Font.Edging.kAntiAlias)
    glow = skia.Paint(AntiAlias=True, Color=col(P.GLOW, 0.55))
    glow.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, px * 0.12))
    cv.drawString(s, x, y, f, glow)
    ink = skia.Paint(AntiAlias=True, Color=col(P.INK), Style=skia.Paint.kStroke_Style, StrokeWidth=px * 0.11,
                     StrokeJoin=skia.Paint.kRound_Join)
    cv.drawString(s, x, y, f, ink)
    gold = skia.Paint(AntiAlias=True)
    gold.setShader(skia.GradientShader.MakeLinear(
        [skia.Point(0, y - px * 0.85), skia.Point(0, y + px * 0.05)],
        [col("#FFF1C4"), col("#F6C766"), col("#D98A35")], [0.0, 0.5, 1.0]))
    cv.drawString(s, x, y, f, gold)
    return f.measureText(s)


def tagline(cv, lines, x, y, px):
    f = skia.Font(typeface("slab"), px)
    sh = skia.Paint(AntiAlias=True, Color=col(P.INK, 0.8))
    sh.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, px * 0.08))
    for i, s in enumerate(lines):
        yy = y + i * px * 1.18
        cv.drawString(s, x + px * 0.04, yy + px * 0.06, f, sh)
        cv.drawString(s, x, yy, f, skia.Paint(AntiAlias=True, Color=col("#F4ECDA")))


def badge(cv, s, x, y, k):
    f = skia.Font(typeface("slab"), 58 * k)
    w = f.measureText(s) + 60 * k
    r = skia.Rect(x, y, x + w, y + 88 * k)
    cv.drawRoundRect(r, 44 * k, 44 * k, skia.Paint(AntiAlias=True, Color=col(P.NAVY_DARK, 0.85)))
    cv.drawRoundRect(r, 44 * k, 44 * k, skia.Paint(AntiAlias=True, Color=col("#F6C766"), Style=skia.Paint.kStroke_Style,
                                                   StrokeWidth=4 * k))
    cv.drawString(s, x + 30 * k, y + 64 * k, f, skia.Paint(AntiAlias=True, Color=col("#F4ECDA")))


def scrim(cv, w, h, x1):
    """Darken the text side so the lettering reads at thumbnail size."""
    p = skia.Paint()
    p.setShader(skia.GradientShader.MakeLinear([skia.Point(0, 0), skia.Point(x1, 0)],
                                               [col(P.NIGHT, 0.62), col(P.NIGHT, 0.0)], None))
    cv.drawRect(skia.Rect(0, 0, x1, h), p)


TAG = ("Is it finally time", "for Fable 5.5?")


def compose(img, w, h, layout):
    surf = skia.Surface(w, h)
    cv = surf.getCanvas()
    cv.drawImage(img, 0, 0)
    k = h / 2160.0
    if layout == "zh_16x9":
        scrim(cv, w, h, 2100 * k)
        badge(cv, "4K · 60FPS", 140 * k, 120 * k, k)
        spaced(cv, "NIGHT AT THE LAB", 150 * k, 470 * k, 84 * k, "#F6C766")
        poster_text(cv, "实验室", 130 * k, 860 * k, 360 * k)
        poster_text(cv, "奇妙夜", 130 * k, 1250 * k, 360 * k)
        tagline(cv, TAG, 150 * k, 1490 * k, 112 * k)
    elif layout == "zh_4x3":
        scrim(cv, w, h, 1700 * k)
        badge(cv, "4K · 60FPS", 110 * k, 110 * k, k)
        spaced(cv, "NIGHT AT THE LAB", 120 * k, 440 * k, 70 * k, "#F6C766")
        poster_text(cv, "实验室", 100 * k, 790 * k, 300 * k)
        poster_text(cv, "奇妙夜", 100 * k, 1120 * k, 300 * k)
        tagline(cv, TAG, 120 * k, 1340 * k, 96 * k)
    elif layout == "en_16x9":
        scrim(cv, w, h, 2100 * k)
        badge(cv, "4K · 60FPS", 140 * k, 120 * k, k)
        poster_text(cv, "NIGHT AT", 130 * k, 700 * k, 300 * k, "slab")
        poster_text(cv, "THE LAB", 130 * k, 1010 * k, 300 * k, "slab")
        tagline(cv, TAG, 150 * k, 1300 * k, 120 * k)
    elif layout == "en_4x3":
        scrim(cv, w, h, 1700 * k)
        badge(cv, "4K · 60FPS", 110 * k, 110 * k, k)
        poster_text(cv, "NIGHT AT", 100 * k, 640 * k, 245 * k, "slab")
        poster_text(cv, "THE LAB", 100 * k, 900 * k, 245 * k, "slab")
        tagline(cv, TAG, 120 * k, 1150 * k, 100 * k)
    return surf.makeImageSnapshot()


COVERS = [
    # name, size, camera, text layout
    ("cover_16x9_zh", (3840, 2160), Cam(XR - 1.95, 2.5, 5.3), "zh_16x9"),
    ("cover_4x3_zh", (2880, 2160), Cam(XR - 1.45, 2.6, 6.1), "zh_4x3"),
    ("cover_16x9_en", (3840, 2160), Cam(XR - 1.95, 2.5, 5.3), "en_16x9"),
    ("cover_4x3_en", (2880, 2160), Cam(XR - 1.45, 2.6, 6.1), "en_4x3"),
    ("cover_16x9_clean", (3840, 2160), Cam(XR, 2.55, 5.5), None),
    ("cover_4x3_clean", (2880, 2160), Cam(XR, 2.75, 6.5), None),
]


def main():
    out = os.path.join(ROOT, "out", "covers")
    os.makedirs(out, exist_ok=True)
    for name, (w, h), cam, layout in COVERS:
        img = render(key_state(cam), w, h)
        if layout:
            img = compose(img, w, h, layout)
        png = os.path.join(out, name + ".png")
        img.save(png, skia.kPNG)
        im = Image.open(png).convert("RGB")
        small = im.resize((1920, 1080) if w > h * 1.5 else (1440, 1080), Image.LANCZOS)
        small.save(os.path.join(out, name + ".jpg"), quality=92, optimize=True)
        print(png)


if __name__ == "__main__":
    main()
