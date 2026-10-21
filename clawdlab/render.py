"""Frame compositor: world pass, lighting, emissive pass and film finish."""
import math
from functools import lru_cache

import numpy as np
import skia

from . import palette as P
from .gfx import G, Cam, col, rect_path, circle_path, ellipse_path
from . import scene as S
from .clawd import draw_clawd


@lru_cache(None)
def paper_image(w, h):
    rng = np.random.default_rng(3)
    # fine grain
    g = rng.normal(0, 1, (h // 2 + 1, w // 2 + 1)).astype(np.float32)
    g = np.repeat(np.repeat(g, 2, 0), 2, 1)[:h, :w]
    # soft blotches
    lo = rng.normal(0, 1, (h // 64 + 2, w // 64 + 2)).astype(np.float32)
    lo = np.kron(lo, np.ones((64, 64), np.float32))[:h, :w]
    k = np.ones(31, np.float32) / 31
    lo = np.apply_along_axis(lambda r: np.convolve(r, k, "same"), 1, lo)
    lo = np.apply_along_axis(lambda r: np.convolve(r, k, "same"), 0, lo)
    v = 1.0 - 0.022 * np.abs(g) - 0.018 * (lo - lo.min()) / (np.ptp(lo) + 1e-6)
    v = np.clip(v, 0, 1)
    img = np.zeros((h, w, 4), np.uint8)
    img[..., 0] = (v * 255).astype(np.uint8)
    img[..., 1] = (v * 252).astype(np.uint8)
    img[..., 2] = (v * 246).astype(np.uint8)
    img[..., 3] = 255
    return skia.Image.fromarray(img, colorType=skia.ColorType.kRGBA_8888_ColorType)


def draw_world(g, st):
    t = st.t
    S.draw_wall(g, st)
    S.draw_floor(g, st)
    S.draw_light_cones(g, st)
    S.draw_pipes(g, st)
    S.draw_reactor(g, st.reactor, t)
    for name in ("opus", "sonnet", "haiku"):
        S.draw_pod(g, name, st.pods.get(name, {}), t)
    for L in S.LAMPS:
        if g.visible(L["x"] - 1, L["y"] - 0.5, L["x"] + 1, S.CEIL):
            S.draw_lamp(g, L, st.lamps.get(L["id"], 0.0), st.lamp_sway)
    for fn in st.fx_back:
        fn(g)
    for b in st.bursts:
        if not b.front:
            b.draw(g, t)
    chars = sorted(st.chars, key=lambda c: c.depth if c.depth is not None else -c.y)
    for c in chars:
        draw_clawd(g, c, t)
    for b in st.bursts:
        if b.front:
            b.draw(g, t)
    for fn in st.fx_front:
        fn(g)


def draw_lighting(g, st):
    """Darkness with lamp pools cut out, then emissive glows on top."""
    night = st.night
    if night > 0.005:
        cv = g.c
        cv.saveLayer(None, None)
        x0 = g.cam.cx - g.view_w
        x1 = g.cam.cx + g.view_w
        g.fill(rect_path(x0, -20, x1, 30), P.NIGHT, 0.86 * night)
        cut = skia.Paint(AntiAlias=True)
        cut.setBlendMode(skia.BlendMode.kDstOut)
        for L in S.LAMPS:
            on = st.lamps.get(L["id"], 0.0)
            if on <= 0.01:
                continue
            lx, ly = S.lamp_pos(L, st.lamp_sway)
            k = L["k"]
            r = 2.3 * k + 0.25 * (ly - 0)
            cut.setShader(skia.GradientShader.MakeRadial(
                skia.Point(lx, ly * 0.45), r, [col("#000000", 0.95 * on), col("#000000", 0.6 * on), col("#000000", 0.0)],
                [0.0, 0.55, 1.0]))
            cv.drawRect(skia.Rect(lx - r, ly * 0.45 - r, lx + r, ly * 0.45 + r), cut)
        cv.restore()
    # emissive
    for L in S.LAMPS:
        on = st.lamps.get(L["id"], 0.0)
        if on > 0.01:
            lx, ly = S.lamp_pos(L, st.lamp_sway)
            k = L["k"]
            g.glow(ellipse_path(lx, ly - 0.02 * k, 0.16 * k, 0.09 * k), "#FFF2C8", 0.12 * k, on)
    if st.night > 0.05:
        for name, cfg in S.PODS.items():
            ps = st.pods.get(name, {})
            k, X = cfg["k"], cfg["x"]
            a = st.night * (0.6 + 0.4 * ps.get("light", 1.0))
            g.glow(S.rrect_path(X - 0.45 * k, 0.95 * k, X + 0.45 * k, 1.78 * k, 0.12 * k), P.POD_INSIDE_GLOW, 0.12 * k, 0.45 * a)
    rs = st.reactor
    e = rs.get("eyes", 0.0)
    if e > 0.01:
        cx, cy = S.PORT_C
        g.glow(circle_path(cx, cy, S.PORT_R * 0.9), P.GLOW, 0.9, 0.12 * e)


def finish(g, st, wpx, hpx):
    cv = g.c
    # alarm wash
    if st.alarm > 0.005:
        p = skia.Paint()
        p.setShader(skia.GradientShader.MakeLinear(
            [skia.Point(0, 0), skia.Point(0, hpx)], [col(P.ALARM, 0.35 * st.alarm), col(P.ALARM, 0.05 * st.alarm)], None))
        cv.drawRect(skia.Rect(0, 0, wpx, hpx), p)
    # vignette
    vp = skia.Paint()
    vp.setShader(skia.GradientShader.MakeRadial(
        skia.Point(wpx / 2, hpx / 2), math.hypot(wpx, hpx) * 0.56,
        [col("#0B0A14", 0.0), col("#0B0A14", 0.06), col("#0B0A14", 0.34)], [0.0, 0.55, 1.0]))
    cv.drawRect(skia.Rect(0, 0, wpx, hpx), vp)
    # paper grain
    pp = skia.Paint()
    pp.setBlendMode(skia.BlendMode.kMultiply)
    cv.drawImage(paper_image(wpx, hpx), 0, 0, skia.SamplingOptions(), pp)
    if st.flash > 0:
        cv.drawRect(skia.Rect(0, 0, wpx, hpx), skia.Paint(Color=col("#FFF6E6", 0.6 * st.flash)))
    if st.fade > 0:
        cv.drawRect(skia.Rect(0, 0, wpx, hpx), skia.Paint(Color=col("#0B0A14", st.fade)))
    for fn in st.overlay:
        fn(cv, wpx, hpx)


def render(st, wpx, hpx):
    surf = skia.Surface(wpx, hpx)
    cv = surf.getCanvas()
    cv.clear(col(P.WALL))
    cam = st.cam
    if st.shake != (0.0, 0.0) or st.cam_rot:
        cam = Cam(cam.cx + st.shake[0], cam.cy + st.shake[1], cam.h, st.cam_rot)
    g = G(cv, wpx, hpx, cam, st.t)
    if not st.black:
        g.begin_world()
        draw_world(g, st)
        draw_lighting(g, st)
        g.end()
    else:
        cv.clear(col("#0B0A14"))
    finish(g, st, wpx, hpx)
    img = surf.makeImageSnapshot()
    return img
