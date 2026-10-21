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
    st.alarm, st.flash, st.night = 0.18, 0.0, 0.28
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


def warning_panel(cv, x, y, w, k, rot=-3.0):
    """Lab-sign style panel: '实验室守则 第一条 / 请勿敲击玻璃'."""
    h = 430 * k
    cv.save()
    cv.translate(x + w / 2, y + h / 2)
    cv.rotate(rot)
    cv.translate(-w / 2, -h / 2)
    # drop shadow
    sh = skia.Paint(AntiAlias=True, Color=col(P.INK, 0.45))
    sh.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, 22 * k))
    cv.drawRoundRect(skia.Rect(14 * k, 22 * k, w + 14 * k, h + 22 * k), 26 * k, 26 * k, sh)
    cv.drawRoundRect(skia.Rect(0, 0, w, h), 26 * k, 26 * k, skia.Paint(AntiAlias=True, Color=col(P.HAZARD_DARK)))
    cv.drawRoundRect(skia.Rect(14 * k, 14 * k, w - 14 * k, h - 14 * k), 16 * k, 16 * k,
                     skia.Paint(AntiAlias=True, Color=col(P.SAFETY)))
    inner = skia.Paint(AntiAlias=True, Color=col(P.HAZARD_DARK), Style=skia.Paint.kStroke_Style, StrokeWidth=5 * k)
    cv.drawRoundRect(skia.Rect(34 * k, 34 * k, w - 34 * k, h - 34 * k), 10 * k, 10 * k, inner)
    # warning triangle
    tx, ty, ts = 80 * k, 70 * k, 92 * k
    tri = skia.Path()
    tri.moveTo(tx + ts / 2, ty)
    tri.lineTo(tx + ts, ty + ts * 0.88)
    tri.lineTo(tx, ty + ts * 0.88)
    tri.close()
    cv.drawPath(tri, skia.Paint(AntiAlias=True, Color=col(P.HAZARD_DARK), Style=skia.Paint.kStroke_Style,
                                StrokeWidth=11 * k, StrokeJoin=skia.Paint.kRound_Join))
    f = skia.Font(typeface("slab"), 62 * k)
    cv.drawString("!", tx + ts / 2 - 9 * k, ty + ts * 0.8, f, skia.Paint(AntiAlias=True, Color=col(P.HAZARD_DARK)))
    draw_text(cv, "实验室守则 第一条", tx + ts + 30 * k, ty + 76 * k, 78 * k, P.HAZARD_DARK)
    draw_text(cv, "请勿敲击玻璃", w / 2, h - 82 * k, 196 * k, P.HAZARD_DARK, align="center")
    cv.restore()
    return h


def caption(cv, lines, x, y, k, rot=2.0):
    cv.save()
    cv.translate(x, y)
    cv.rotate(rot)
    yy = 0.0
    for s, px, fill in lines:
        draw_text(cv, s, 0, yy, px * k, fill, P.INK, 22 * k)
        yy += px * k * 1.12
    cv.restore()


def badge(cv, s, x, y, k):
    f = skia.Font(typeface("slab"), 70 * k)
    w = f.measureText(s) + 70 * k
    cv.drawRoundRect(skia.Rect(x, y, x + w, y + 104 * k), 52 * k, 52 * k, skia.Paint(AntiAlias=True, Color=col(P.NAVY_DARK)))
    cv.drawRoundRect(skia.Rect(x, y, x + w, y + 104 * k), 52 * k, 52 * k,
                     skia.Paint(AntiAlias=True, Color=col(P.SAFETY), Style=skia.Paint.kStroke_Style, StrokeWidth=5 * k))
    cv.drawString(s, x + 35 * k, y + 76 * k, f, skia.Paint(AntiAlias=True, Color=col("#F1E9D8")))


def compose(img, w, h, layout):
    surf = skia.Surface(w, h)
    cv = surf.getCanvas()
    cv.drawImage(img, 0, 0)
    k = h / 2160.0
    if layout == "zh_16x9":
        badge(cv, "4K · 60FPS", 120 * k, 110 * k, k)
        ph = warning_panel(cv, 110 * k, 300 * k, 1480 * k, k)
        caption(cv, [("Haiku：", 170, "#F7F1E3"), ("收到（咚咚咚）", 170, P.ORANGE_LIGHT)], 150 * k, 300 * k + ph + 260 * k, k)
    elif layout == "zh_4x3":
        badge(cv, "4K · 60FPS", 110 * k, 100 * k, k)
        ph = warning_panel(cv, 90 * k, 270 * k, 1480 * k, k)
        caption(cv, [("Haiku：", 165, "#F7F1E3"), ("收到（咚咚咚）", 165, P.ORANGE_LIGHT)], 120 * k, 270 * k + ph + 250 * k, k)
    return surf.makeImageSnapshot()


COVERS = [
    # name, size, camera, text layout
    ("cover_16x9_zh", (3840, 2160), Cam(XR - 1.95, 2.5, 5.3), "zh_16x9"),
    ("cover_4x3_zh", (2880, 2160), Cam(XR - 1.45, 2.6, 6.1), "zh_4x3"),
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
