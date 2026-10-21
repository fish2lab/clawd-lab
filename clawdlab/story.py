"""The storyboard.  Every value on screen is a function of time t (seconds).

  0.0  lights clunk on over a dark lab: three cryo pods and the Fable reactor
  4.3  Opus 5.5 wakes, wipes the frost, spins its own wheel and jumps out
  8.3  insert: Opus ticks "Exit pod" on its to-do list
  9.6  Opus walks to the smaller Sonnet pod and opens it; Sonnet was already coding
 14.3  both walk to the smallest pod; Haiku bursts out and laps them at speed
 17.5  insert: "Wake Haiku" ticked, "Check on Fable" circled... the lab rumbles
 19.6  the trio walks to the reactor; the camera pulls back to show the scale
 24.3  they stack up to reach the porthole; camera pushes in (same world, same glass)
 27.4  Haiku taps the glass.  Eyes light up.  Three hits.  Glass cracks.
 32.4  black; the eyes blink once more; title.
"""
import math
from functools import lru_cache

import numpy as np
import skia

from . import palette as P
from .gfx import (Cam, clamp, lerp, seg, smooth, ease_io, ease_out, ease_in, back_out,
                  vnoise, hash01, rect_path, rrect_path, circle_path, col, typeface)
from .clawd import CS, clipboard_world
from .state import State
from . import scene as S

DURATION = 35.6
PI = math.pi

# ------------------------------------------------------------------ key times

T_LAMP = dict(opus=0.55, sonnet=0.95, haiku=1.35, fable_l=1.95, fable_r=2.1)
T_AMBIENT = (2.35, 2.9)

T_OPUS_WAKE = 4.45
T_WIPE = (5.05, 5.75)
T_WHEEL_O = (5.85, 6.65)
T_DOOR_O = 6.68
T_JUMP_O = (6.92, 7.62)
T_INSERT_A = (8.3, 9.6)
T_CHECK1 = (8.7, 9.05)
T_WALK1 = (9.6, 11.0)
T_WHEEL_S = (11.05, 11.85)
T_DOOR_S = 11.88
T_JUMP_S = (12.12, 12.6)
T_TYPE = (12.95, 13.85)
T_PASS = 13.82
T_CHECK2 = (14.0, 14.3)
T_WALK2 = (14.35, 15.6)
T_WHEEL_H = (15.62, 16.02)
T_DOOR_H = 16.05
T_DASH_OUT = (16.07, 16.30)
T_DASH_IN = (16.55, 17.0)
T_INSERT_B = (17.5, 19.0)
T_CHECK3 = (17.7, 18.05)
T_CIRCLE4 = (18.2, 18.75)
T_RUMBLE = 18.35
T_WALK3 = (19.65, 23.7)
T_TURN = (23.75, 24.15)
T_HOP_S = (24.3, 24.75)
T_HOP_H = (24.85, 25.3)
T_PUSH = (25.2, 27.4)
T_TAP = (27.45, 28.1)
T_EYES = 28.55
T_IMPACTS = (29.85, 30.5, 31.05)
T_BLACK = 32.4
T_END_EYES = 32.95
T_TITLE = 33.35
T_THUD = 33.95
T_FADE = (35.0, DURATION)

# ------------------------------------------------------------------ geometry

OX, SX, HX = S.PODS["opus"]["x"], S.PODS["sonnet"]["x"], S.PODS["haiku"]["x"]
KO, KS, KH = 1.0, 0.78, 0.56
WO, WS, WH = 1.0, 0.78, 0.56
LANE_O, LANE_S, LANE_H = -0.34, -0.42, -0.40
XR = S.XR

O_STAND_S = SX - 0.72          # Opus beside Sonnet's wheel
O_STAND_H = HX - 0.86          # Opus beside Haiku's wheel
O_FINAL = XR - 0.05
S_FINAL = XR - 1.05
H_FINAL = XR + 0.95
O_TOP = LANE_O + 0.8333 * WO   # top of Opus's head at the reactor
S_TOP = O_TOP + 0.8333 * WS


def lin(t, keys, ease=ease_io):
    """Piecewise interpolation; keys = [(t, v), ...]; v float or tuple."""
    if t <= keys[0][0]:
        return keys[0][1]
    for i in range(1, len(keys)):
        t0, v0 = keys[i - 1][0], keys[i - 1][1]
        t1, v1 = keys[i][0], keys[i][1]
        if t <= t1:
            e = keys[i][2] if len(keys[i]) > 2 else ease
            u = e((t - t0) / (t1 - t0)) if t1 > t0 else 1.0
            if isinstance(v0, tuple):
                return tuple(a + (b - a) * u for a, b in zip(v0, v1))
            return v0 + (v1 - v0) * u
    return keys[-1][1]


def arc(t, t0, t1, p0, p1, h):
    """Jump: returns (x, y_base, lift, u)."""
    u = seg(t, t0, t1)
    x = lerp(p0[0], p1[0], u)
    y = lerp(p0[1], p1[1], u)
    return x, y, 4 * h * u * (1 - u), u


def impulse(t, t0, decay=6.0, freq=0.0):
    if t < t0:
        return 0.0
    d = t - t0
    e = math.exp(-decay * d)
    return e * (math.cos(freq * d * 2 * PI) if freq else 1.0)


# ------------------------------------------------------------------ positions

def opus_pos(t):
    """(x, y, lift, inside_pod)"""
    floor_in = 0.46 * KO + 0.02
    if t < T_JUMP_O[0]:
        return OX, floor_in, 0.0, True
    if t < T_JUMP_O[1]:
        x, y, l, u = arc(t, *T_JUMP_O, (OX, floor_in), (OX - 0.34, LANE_O), 0.75)
        return x, y, l, False
    x0 = OX - 0.34
    if t < T_WALK1[0]:
        return x0, LANE_O, 0.0, False
    if t < T_WHEEL_S[1] + 0.05:
        x = lin(t, [(T_WALK1[0], x0), (T_WALK1[1], O_STAND_S)])
        return x, LANE_O, 0.0, False
    if t < T_WALK2[0]:
        # recoil when the door bursts
        x = lin(t, [(T_DOOR_S, O_STAND_S), (T_DOOR_S + 0.3, O_STAND_S - 0.18, ease_out)])
        l = 0.12 * math.sin(PI * seg(t, T_DOOR_S, T_DOOR_S + 0.28))
        return x, LANE_O, l, False
    if t < T_WALK3[0]:
        x = lin(t, [(T_WALK2[0], O_STAND_S - 0.18), (T_WALK2[1], O_STAND_H)])
        l = 0.08 * math.sin(PI * seg(t, T_DOOR_H, T_DOOR_H + 0.22))
        l += 0.06 * math.sin(PI * seg(t, 19.05, 19.3))
        return x, LANE_O, l, False
    if t < T_IMPACTS[2]:
        x = lin(t, [(T_WALK3[0], O_STAND_H), (T_WALK3[1], O_FINAL)], lambda u: smooth(u) * 0.3 + u * 0.7)
        return x, LANE_O, 0.0, False
    # final hit: knocked back a step
    x = lin(t, [(T_IMPACTS[2], O_FINAL), (T_IMPACTS[2] + 0.35, O_FINAL - 0.12, ease_out)])
    l = 0.18 * math.sin(PI * seg(t, T_IMPACTS[2] + 0.05, T_IMPACTS[2] + 0.4))
    return x, LANE_O, l, False


def sonnet_pos(t):
    floor_in = 0.46 * KS + 0.02
    if t < T_JUMP_S[0]:
        return SX, floor_in, 0.0, True, False
    if t < T_JUMP_S[1]:
        x, y, l, u = arc(t, *T_JUMP_S, (SX, floor_in), (SX + 0.1, LANE_S), 0.55)
        return x, y, l, False, False
    x0 = SX + 0.1
    if t < T_WALK2[0]:
        return x0, LANE_S, 0.0, False, False
    if t < T_WALK3[0]:
        x = x0
        l = 0.1 * math.sin(PI * seg(t, T_DOOR_H, T_DOOR_H + 0.22))
        l += 0.08 * math.sin(PI * seg(t, 19.1, 19.35))
        return x, LANE_S, l, False, False
    if t < T_HOP_S[0]:
        x = lin(t, [(T_WALK3[0] + 0.1, SX + 0.1), (T_WALK3[1] + 0.05, S_FINAL)], lambda u: smooth(u) * 0.3 + u * 0.7)
        return x, LANE_S, 0.0, False, False
    if t < T_HOP_S[1]:
        x, y, l, u = arc(t, *T_HOP_S, (S_FINAL, LANE_S), (O_FINAL, O_TOP), 0.55)
        return x, y, l, False, u > 0.5
    if t < T_IMPACTS[2]:
        wob = 0.0
        return O_FINAL + wob, O_TOP, 0.0, False, True
    # tumbles off to the left
    x, y, l, u = arc(t, T_IMPACTS[2] + 0.05, T_IMPACTS[2] + 0.55, (O_FINAL, O_TOP), (S_FINAL - 0.2, LANE_S), 0.45)
    return x, y, l, False, u < 0.5


def haiku_pos(t):
    floor_in = 0.46 * KH + 0.02
    if t < T_DASH_OUT[0]:
        return HX, floor_in, 0.0, True, False
    if t < T_DASH_OUT[1]:
        u = ease_in(seg(t, *T_DASH_OUT), 1.6)
        x = lerp(HX, HX + 3.6, u)
        y = lerp(floor_in, LANE_H + 0.05, ease_out(seg(t, T_DASH_OUT[0], T_DASH_OUT[0] + 0.1)))
        return x, y, 0.0, False, False
    if t < T_DASH_IN[0]:
        return HX + 8, LANE_H, 0.0, False, False
    if t < T_DASH_IN[1]:
        u = seg(t, *T_DASH_IN)
        e = 1 - (1 - u) ** 2.2
        x = lerp(1.2, HX + 0.62, e)
        y = LANE_H - 0.22 * math.sin(PI * min(1, u * 1.15))
        return x, y, 0.0, False, False
    xs = HX + 0.62
    if t < T_WALK3[0]:
        l = 0.12 * math.sin(PI * seg(t, 17.02, 17.3)) + 0.16 * math.sin(PI * seg(t, 19.0, 19.32))
        return xs, LANE_H, l, False, False
    if t < T_HOP_H[0]:
        x = lin(t, [(T_WALK3[0] + 0.15, xs), (T_WALK3[0] + 0.75, 8.9, ease_io), (21.4, 8.9),
                    (21.95, 11.6, ease_io), (22.75, 11.6), (23.2, H_FINAL, ease_io)])
        return x, LANE_H, 0.0, False, False
    if t < T_HOP_H[1]:
        x, y, l, u = arc(t, *T_HOP_H, (H_FINAL, LANE_H), (O_FINAL, S_TOP), 0.5)
        return x, y, l, False, u > 0.5
    if t < T_IMPACTS[2]:
        return O_FINAL, S_TOP, 0.0, False, True
    # launched up and over to the right
    x, y, l, u = arc(t, T_IMPACTS[2], T_IMPACTS[2] + 0.62, (O_FINAL, S_TOP), (H_FINAL + 0.25, LANE_H), 0.9)
    return x, y, l, False, u < 0.4


@lru_cache(None)
def gait_tables():
    """Cumulative ground distance per character -> walk phase."""
    ts = np.arange(0, DURATION + 0.01, 1 / 240)
    out = {}
    for name, fn, W in (("opus", opus_pos, WO), ("sonnet", sonnet_pos, WS), ("haiku", haiku_pos, WH)):
        xs = np.array([fn(t)[0] for t in ts])
        ls = np.array([fn(t)[2] for t in ts])
        dx = np.abs(np.diff(xs, prepend=xs[0]))
        dx[ls > 0.02] = 0.0          # no stepping while airborne
        dx[dx > 0.2] = 0.0           # teleports
        dist = np.cumsum(dx)
        spd = dx * 240
        k = np.ones(25) / 25
        spd = np.convolve(spd, k, "same")
        out[name] = (ts, dist, spd)
    return out


def gait(name, t, W):
    ts, dist, spd = gait_tables()[name]
    d = float(np.interp(t, ts, dist))
    v = float(np.interp(t, ts, spd))
    stride = 0.55 * W
    return d / stride * PI, clamp(v / (0.6 * W)), v


# ------------------------------------------------------------------ camera

def camera(t):
    if t < T_INSERT_A[0] or T_INSERT_A[1] <= t < T_INSERT_B[0] or t >= T_INSERT_B[1]:
        keys = [
            (0.0, (6.6, 5.6, 13.4)),
            (2.95, (6.1, 5.3, 12.6), lambda u: u),
            (4.25, (0.05, 1.32, 3.4), ease_io),
            (6.6, (0.0, 1.30, 3.45), ease_io),
            (7.9, (-0.12, 1.12, 3.55), ease_io),
            (T_INSERT_A[0], (-0.12, 1.12, 3.5)),
            (T_INSERT_A[1], (0.35, 1.08, 3.35)),
            (T_WALK1[1] + 0.15, (2.55, 1.05, 3.1), ease_io),
            (T_JUMP_S[1] + 0.4, (2.75, 1.0, 3.1), ease_io),
            (T_CHECK2[1], (2.8, 0.98, 3.05), ease_io),
            (T_WALK2[1] + 0.1, (4.55, 0.86, 3.0), ease_io),
            (T_INSERT_B[0], (4.6, 0.86, 3.0)),
            (T_INSERT_B[1], (4.75, 0.95, 3.2)),
            (T_WALK3[0], (4.8, 0.98, 3.25), ease_io),
            (21.2, (7.6, 1.9, 5.4), ease_in),
            (T_WALK3[1] - 0.2, (XR - 1.1, 5.7, 13.2), ease_out),
            (T_WALK3[1] + 0.2, (XR - 1.0, 5.65, 13.1)),
            (T_HOP_S[1] + 0.2, (XR, 2.75, 7.2), ease_io),
            (T_PUSH[1], (XR, 2.62, 6.2), ease_io),
            (T_IMPACTS[0], (XR, 2.7, 5.8), ease_io),
            (T_IMPACTS[2], (XR, 2.75, 5.45), ease_io),
            (T_IMPACTS[2] + 0.7, (XR, 2.05, 5.6), ease_io),
            (T_BLACK, (XR, 2.0, 5.45), ease_out),
        ]
        cx, cy, h = lin(t, keys)
        return Cam(cx, cy, h)
    # insert shots: frame Opus's clipboard
    o, _ = opus_state(t)
    x, y = clipboard_world(o)
    if t < T_INSERT_A[1]:
        u = seg(t, *T_INSERT_A)
        return Cam(x + 0.01, y + 0.005, lerp(0.6, 0.56, u))
    u = seg(t, *T_INSERT_B)
    return Cam(x + 0.01, y + 0.005, lerp(0.6, 0.55, u))


def shake(t):
    a = 0.0
    a += 0.010 * clamp((t - T_RUMBLE) / 0.3) * (1 - seg(t, 19.4, 20.4))
    a += 0.03 * impulse(t, T_DOOR_O, 9)
    a += 0.012 * impulse(t, T_DOOR_S, 10)
    a += 0.012 * impulse(t, T_DOOR_H, 10)
    for i, ti in enumerate(T_IMPACTS):
        a += (0.05 + 0.03 * i) * impulse(t, ti, 5.5)
    a += 0.004 * (seg(t, T_EYES, T_EYES + 0.4)) * (1 - seg(t, T_BLACK - 0.01, T_BLACK))
    a += 0.03 * impulse(t, T_THUD, 6)
    sx = vnoise(t * 38, 1) * a
    sy = vnoise(t * 41, 2) * a
    rot = vnoise(t * 23, 3) * a * 18
    return sx, sy, rot


# ------------------------------------------------------------------ characters

def opus_state(t):
    x, y, lift, inside = opus_pos(t)
    ph, amp, v = gait("opus", t, WO)
    st = CS(W=WO, x=x, y=y, lift=lift, outfit="labcoat", prop="clipboard", walk=ph, walk_amp=amp, name="opus")
    # facing
    th = 0.0
    if t >= T_WALK1[0]:
        th = lin(t, [(T_WALK1[0], 0.0), (T_WALK1[0] + 0.3, 0.55), (T_WALK1[1] - 0.2, 0.55), (T_WALK1[1] + 0.1, 0.22),
                     (T_DOOR_S + 0.15, 0.22), (T_DOOR_S + 0.45, 0.35),
                     (T_WALK2[0], 0.35), (T_WALK2[0] + 0.3, 0.55), (T_WALK2[1] - 0.2, 0.55), (T_WALK2[1] + 0.1, 0.22),
                     (16.4, 0.22), (16.7, -0.25), (17.0, 0.3), (17.3, 0.05),
                     (19.05, 0.05), (19.35, 0.35), (T_WALK3[0] + 0.2, 0.55), (T_WALK3[1] - 0.3, 0.55),
                     (T_TURN[0], 0.3), (T_TURN[1], PI, ease_io),
                     (T_IMPACTS[2] + 0.1, PI), (T_IMPACTS[2] + 0.45, 2 * PI, ease_out)])
    st.theta = th
    # eyes
    if t < T_OPUS_WAKE:
        st.eye_open = 0.08
    else:
        bl = 1.0 - 0.92 * (math.exp(-((t - 4.92) / 0.05) ** 2) + math.exp(-((t - 10.4) / 0.05) ** 2)
                           + math.exp(-((t - 17.2) / 0.05) ** 2) + math.exp(-((t - 21.9) / 0.05) ** 2))
        st.eye_open = min(ease_out(seg(t, T_OPUS_WAKE, T_OPUS_WAKE + 0.15)), bl)
    look = lin(t, [(4.6, (0.0, 0.0)), (4.75, (-0.7, 0.0)), (4.95, (0.7, 0.0)), (5.15, (0.0, 0.0)),
                   (5.8, (0.0, -0.3)), (6.6, (0.0, -0.5)), (7.9, (0.0, 0.0)), (8.1, (-0.6, -0.4)),
                   (T_WALK1[0], (-0.6, -0.4)), (T_WALK1[0] + 0.2, (0.6, 0.0)),
                   (T_WHEEL_S[0], (0.8, 0.5)), (T_JUMP_S[1], (0.9, -0.2)), (T_PASS + 0.1, (0.9, 0.1)),
                   (T_CHECK2[0], (-0.6, -0.4)), (T_CHECK2[1], (-0.6, -0.4)), (T_WALK2[0] + 0.1, (0.6, 0.0)),
                   (T_WHEEL_H[0], (0.8, 0.2)), (16.15, (0.9, 0.0)), (16.3, (0.9, 0.0)), (16.45, (-0.9, 0.0)),
                   (16.75, (-0.6, -0.3)), (16.95, (0.9, -0.3)), (17.3, (0.0, 0.0)), (17.4, (-0.6, -0.4)),
                   (19.05, (-0.6, -0.4)), (19.2, (0.9, 0.3)), (T_WALK3[1], (0.6, 0.6)), (T_TURN[0], (0.0, 0.8))],
               ease=smooth)
    st.look = look
    st.eye_wide = max(0.0, 1 - seg(t, T_DOOR_S, T_DOOR_S + 0.6)) * (t >= T_DOOR_S) + \
        (t >= T_IMPACTS[2] + 0.2) * 1.0 + (19.0 <= t < 19.8) * 1.0
    st.eye_happy = float(T_PASS + 0.1 <= t < T_PASS + 0.55)
    # arms: left holds the clipboard, right works the wheels
    al = 0.0
    al = lin(t, [(7.95, 0.0), (8.25, 1.0, back_out), (T_INSERT_A[1], 1.0), (T_INSERT_A[1] + 0.3, 0.2),
                 (T_CHECK2[0] - 0.2, 0.2), (T_CHECK2[0], 0.9), (T_CHECK2[1] + 0.1, 0.9), (T_CHECK2[1] + 0.35, 0.2),
                 (T_INSERT_B[0] - 0.25, 0.2), (T_INSERT_B[0], 1.0), (T_INSERT_B[1], 1.0), (T_INSERT_B[1] + 0.3, 0.2)])
    ar = 0.0
    if T_WIPE[0] <= t < T_WIPE[1]:
        u = seg(t, *T_WIPE)
        ar = 0.9 + 0.35 * math.sin(u * PI * 6)
    if T_WHEEL_S[0] - 0.2 <= t < T_WHEEL_S[1] + 0.1:
        u = seg(t, T_WHEEL_S[0] - 0.2, T_WHEEL_S[0] + 0.05)
        ar = lerp(0.0, 1.45, ease_out(u)) + 0.12 * math.sin((t - T_WHEEL_S[0]) * 18) * (t > T_WHEEL_S[0])
        st.lift += 0.06 * ease_out(u)
    if T_WHEEL_H[0] - 0.2 <= t < T_WHEEL_H[1] + 0.1:
        u = seg(t, T_WHEEL_H[0] - 0.2, T_WHEEL_H[0] + 0.05)
        ar = lerp(0.0, 0.85, ease_out(u)) + 0.12 * math.sin((t - T_WHEEL_H[0]) * 22) * (t > T_WHEEL_H[0])
    if T_TURN[1] <= t < T_IMPACTS[2]:
        ar = 0.15
    st.arm_l, st.arm_r = al, ar
    # squash & stretch
    sq = 1.0
    if T_JUMP_O[0] - 0.12 <= t < T_JUMP_O[0]:
        sq = 1.0 + 0.16 * math.sin(PI * seg(t, T_JUMP_O[0] - 0.12, T_JUMP_O[0]))
    elif T_JUMP_O[0] <= t < T_JUMP_O[1]:
        sq = 1.0 - 0.1 * math.sin(PI * seg(t, *T_JUMP_O))
    elif T_JUMP_O[1] <= t < T_JUMP_O[1] + 0.45:
        u = seg(t, T_JUMP_O[1], T_JUMP_O[1] + 0.45)
        sq = 1.0 + 0.22 * math.exp(-5 * u) * math.cos(u * 3 * PI)
    if T_HOP_S[1] - 0.02 <= t < T_HOP_S[1] + 0.35:
        sq *= 1.0 + 0.10 * math.exp(-6 * seg(t, T_HOP_S[1] - 0.02, T_HOP_S[1] + 0.35)) * \
            math.cos(seg(t, T_HOP_S[1], T_HOP_S[1] + 0.35) * 2.5 * PI)
    if T_HOP_H[1] - 0.02 <= t < T_HOP_H[1] + 0.35:
        sq *= 1.0 + 0.05 * math.exp(-6 * seg(t, T_HOP_H[1], T_HOP_H[1] + 0.35))
    for ti in T_IMPACTS:
        sq *= 1.0 + 0.06 * impulse(t, ti, 7, 3)
    st.sq = sq
    # coat shake after landing
    if T_JUMP_O[1] + 0.3 <= t < T_JUMP_O[1] + 0.85:
        u = seg(t, T_JUMP_O[1] + 0.3, T_JUMP_O[1] + 0.85)
        st.tilt = 5 * math.sin(u * 5 * PI) * (1 - u)
    for i, ti in enumerate(T_IMPACTS[:2]):
        st.tilt += 3.0 * impulse(t, ti, 5, 2.2)
    # checklist
    st.checks = (seg(t, *T_CHECK1), seg(t, *T_CHECK2), seg(t, *T_CHECK3), 0.0)
    st.circle4 = seg(t, *T_CIRCLE4)
    if T_CHECK1[0] - 0.25 <= t < T_CHECK1[1] + 0.3:
        st.pencil = 0
    elif T_CHECK3[0] - 0.25 <= t < T_CHECK3[1] + 0.12:
        st.pencil = 2
    elif T_CHECK3[1] + 0.12 <= t < T_CIRCLE4[1] + 0.3:
        st.pencil = 3
    st.goggles_down = ease_out(seg(t, T_IMPACTS[2] + 0.1, T_IMPACTS[2] + 0.35), 2)
    st.rim = rim(t)
    st.depth = 0.34
    return st, inside


def sonnet_state(t):
    x, y, lift, inside, stacked = sonnet_pos(t)
    ph, amp, v = gait("sonnet", t, WS)
    st = CS(W=WS, x=x, y=y, lift=lift, outfit="hoodie", prop="laptop", walk=ph, walk_amp=amp, name="sonnet")
    # inside the pod it is already coding
    st.laptop_open = 1.0 if t < T_WALK2[0] - 0.1 else 0.0
    if T_JUMP_S[0] <= t < T_TYPE[0]:
        st.laptop_open = 0.0 if t < T_JUMP_S[1] + 0.1 else ease_out(seg(t, T_JUMP_S[1] + 0.1, T_TYPE[0]))
    st.typing = 1.0 if (t < T_JUMP_S[0] - 0.4 or T_TYPE[0] <= t < T_TYPE[1]) else 0.0
    st.screen_glow = st.laptop_open
    th = lin(t, [(T_JUMP_S[1], 0.0), (T_JUMP_S[1] + 0.4, -0.15), (T_TYPE[1], -0.15), (T_TYPE[1] + 0.2, 0.0),
                 (T_WALK2[0], 0.0), (T_WALK2[0] + 0.3, 0.55), (T_WALK2[1] - 0.1, 0.55), (T_WALK2[1] + 0.3, 0.2),
                 (16.45, 0.2), (16.75, -0.3), (17.0, 0.3), (17.3, 0.1),
                 (19.1, 0.1), (19.4, 0.4), (T_WALK3[0] + 0.3, 0.55), (T_WALK3[1] - 0.2, 0.55),
                 (T_TURN[0] + 0.05, 0.3), (T_TURN[1] + 0.05, PI, ease_io)])
    if t >= T_IMPACTS[2]:
        th = lerp(PI, 2 * PI + 0.2, ease_out(seg(t, T_IMPACTS[2] + 0.05, T_IMPACTS[2] + 0.55)))
    st.theta = th
    if t < T_DOOR_S:
        st.look = (0.0, -0.8)
    else:
        st.look = lin(t, [(T_DOOR_S, (0.0, -0.8)), (T_JUMP_S[1] + 0.15, (-0.6, 0.0)), (T_TYPE[0], (0.0, -0.8)),
                          (T_TYPE[1], (0.0, -0.8)), (T_PASS + 0.15, (-0.5, 0.2)), (T_CHECK2[1], (-0.5, 0.2)),
                          (T_WALK2[0] + 0.1, (0.6, 0.0)), (16.15, (0.9, 0.0)), (16.4, (-0.9, 0.0)),
                          (16.8, (-0.3, -0.4)), (16.98, (0.9, -0.2)), (17.4, (0.6, 0.0)),
                          (19.1, (0.6, 0.0)), (19.25, (0.9, 0.3)), (T_WALK3[1], (0.6, 0.6)), (T_TURN[0], (0.0, 0.8))],
                      ease=smooth)
    st.eye_open = 1.0 - 0.92 * (math.exp(-((t - 17.25) / 0.05) ** 2) + math.exp(-((t - 22.4) / 0.05) ** 2))
    st.eye_happy = float(T_PASS + 0.05 <= t < T_PASS + 0.6)
    st.eye_wide = (19.0 <= t < 19.8) * 1.0 + (t >= T_IMPACTS[2] + 0.3) * 1.0
    st.glint = seg(t, 17.3, 17.6) if 17.3 <= t < 17.6 else (seg(t, 12.75, 13.0) if 12.75 <= t < 13.0 else -1)
    # squash
    sq = 1.0
    if T_JUMP_S[1] <= t < T_JUMP_S[1] + 0.4:
        u = seg(t, T_JUMP_S[1], T_JUMP_S[1] + 0.4)
        sq = 1.0 + 0.2 * math.exp(-5 * u) * math.cos(u * 3 * PI)
    if T_PASS <= t < T_PASS + 0.4:
        st.lift += 0.09 * math.sin(PI * seg(t, T_PASS, T_PASS + 0.3))
    if T_HOP_S[1] <= t < T_HOP_S[1] + 0.35:
        u = seg(t, T_HOP_S[1], T_HOP_S[1] + 0.35)
        sq = 1.0 + 0.15 * math.exp(-5 * u) * math.cos(u * 3 * PI)
    st.sq = sq
    if stacked:
        st.shadow = False
        st.depth = 0.5
        st.tilt = 2.5 * math.sin(t * 5.3) * (1 - seg(t, T_HOP_S[1], T_HOP_S[1] + 1.5)) + 3.0 * impulse(t, T_IMPACTS[0], 4, 2) \
            + 3.5 * impulse(t, T_IMPACTS[1], 4, 2.3)
        st.arm_l = 0.3 + 0.2 * math.sin(t * 4)
        st.arm_r = 0.3 + 0.2 * math.sin(t * 4 + 1)
    else:
        st.depth = 0.42
    if t >= T_IMPACTS[2]:
        u = seg(t, T_IMPACTS[2] + 0.05, T_IMPACTS[2] + 0.55)
        st.tilt = -360 * ease_out(u) if u < 1 else 0.0
        st.arm_l = st.arm_r = 1.2 * (1 - u) + 0.8 * u
    st.rim = rim(t)
    return st, inside


def haiku_state(t):
    x, y, lift, inside, stacked = haiku_pos(t)
    ph, amp, v = gait("haiku", t, WH)
    st = CS(W=WH, x=x, y=y, lift=lift, outfit="uniform", walk=ph, walk_amp=min(1.0, amp), name="haiku")
    # facing: dash direction
    th = 0.0
    if T_DASH_OUT[0] <= t < T_DASH_IN[1] + 0.05:
        th = 1.2
    elif t >= T_DASH_IN[1] + 0.05:
        th = lin(t, [(T_DASH_IN[1] + 0.05, 1.2), (T_DASH_IN[1] + 0.3, 0.0, back_out),
                     (19.0, 0.0), (19.2, 0.4), (T_WALK3[0] + 0.2, 0.6), (T_WALK3[1] - 0.6, 0.6),
                     (23.4, 0.1), (T_TURN[0] + 0.1, 0.3), (T_TURN[1] + 0.1, PI, ease_io)])
    if t >= T_IMPACTS[2]:
        th = lerp(PI, 3 * PI, ease_out(seg(t, T_IMPACTS[2], T_IMPACTS[2] + 0.62)))
    st.theta = th
    st.look = (0.0, 0.0)
    if inside:
        st.look = (math.sin(t * 9) * 0.9, 0.2)
        st.lift = max(0.0, math.sin(t * 15)) * 0.08 * seg(t, 14.6, 14.9)
    elif t < T_WALK3[0]:
        st.look = lin(t, [(17.0, (0.0, 0.0)), (17.3, (-0.5, 0.0)), (19.0, (-0.5, 0.0)), (19.2, (0.9, 0.3))])
    else:
        st.look = lin(t, [(T_WALK3[0], (0.8, 0.2)), (T_WALK3[1], (0.6, 0.6)), (T_TURN[0], (0.0, 0.8))])
    st.eye_happy = float(17.0 <= t < 17.55)
    st.eye_wide = (19.0 <= t < 19.9) * 1.0 + (t >= T_IMPACTS[2] + 0.35) * 1.0
    # salute, then tap the glass
    ar = 0.0
    if 17.05 <= t < 17.6:
        ar = 1.5 * back_out(seg(t, 17.05, 17.2))
    if T_TAP[0] <= t < T_TAP[1] + 0.2:
        u = seg(t, *T_TAP)
        ar = 1.3 * ease_out(seg(t, T_TAP[0], T_TAP[0] + 0.15)) + 0.25 * max(0, math.sin(u * 4 * PI)) \
            - 1.3 * seg(t, T_TAP[1], T_TAP[1] + 0.2)
    if t >= T_IMPACTS[2]:
        ar = 1.5
        st.arm_l = 1.5
    st.arm_r = ar
    # motion stretch when dashing
    if v > 3.0:
        st.sq = 1.0 + min(0.25, (v - 3.0) * 0.03)
        st.tilt = 5 if T_DASH_OUT[0] <= t < T_DASH_OUT[1] else 6
    if T_DASH_IN[1] <= t < T_DASH_IN[1] + 0.3:
        u = seg(t, T_DASH_IN[1], T_DASH_IN[1] + 0.3)
        st.tilt = -14 * (1 - u) * math.cos(u * 2 * PI)
    if stacked:
        st.shadow = False
        st.depth = 0.6
        st.tilt = 3.0 * math.sin(t * 6.1 + 1) * (1 - seg(t, T_HOP_H[1], T_HOP_H[1] + 1.8)) \
            + 5 * impulse(t, T_IMPACTS[0], 4, 2) + 6 * impulse(t, T_IMPACTS[1], 4, 2.3)
        st.lift += 0.05 * impulse(t, T_IMPACTS[0], 8) + 0.07 * impulse(t, T_IMPACTS[1], 8)
    else:
        st.depth = 0.40 if t < T_DASH_IN[0] else (0.9 if t < T_DASH_IN[1] + 0.1 else 0.40)
    if t >= T_IMPACTS[2]:
        u = seg(t, T_IMPACTS[2], T_IMPACTS[2] + 0.62)
        st.tilt = 540 * ease_out(u) if u < 1 else 0.0
        st.depth = 0.7
    st.rim = rim(t)
    return st, inside


def rim(t):
    return eyes(t) * (1 - seg(t, T_IMPACTS[2], T_IMPACTS[2] + 0.4)) * 0.9


# ------------------------------------------------------------------ props

def eyes(t):
    if t < T_EYES:
        return 0.0
    d = t - T_EYES
    if d < 0.35:
        # fluorescent-tube flicker
        return [0.0, 0.65, 0.1, 0.0, 0.85, 0.3, 1.0][min(6, int(d / 0.05))]
    return 1.0


def reactor_state(t):
    rs = {}
    rs["eyes"] = eyes(t)
    rs["eye_open"] = 1.0 if t < 31.8 else 1.0 - 0.95 * math.exp(-((t - 32.0) / 0.05) ** 2)
    rs["angry"] = ease_io(seg(t, 29.3, 29.6))
    # lunges
    l = 0.0
    l = lin(t, [(29.3, 0.0), (29.65, -0.22), (T_IMPACTS[0], 1.0, lambda u: u ** 2.2), (30.15, 0.05, ease_out),
                (30.32, -0.15), (T_IMPACTS[1], 1.0, lambda u: u ** 2.2), (30.75, 0.0, ease_out),
                (30.88, -0.2), (T_IMPACTS[2], 1.05, lambda u: u ** 2.2), (31.5, 0.3, ease_out), (T_BLACK, 0.22)])
    rs["lunge"] = l
    rs["face_vis"] = 0.22 + 0.78 * rs["eyes"]
    # cracks
    c = 0.0
    for i, ti in enumerate(T_IMPACTS):
        c = max(c, i + ease_out(seg(t, ti, ti + 0.2)) if t >= ti else c)
    c += 0.08 * seg(t, 31.6, T_BLACK)
    rs["cracks"] = c
    fl = 0.0
    for ti in T_IMPACTS:
        fl += 0.9 * impulse(t, ti, 10)
    fl += 0.5 * (T_EYES + 0.2 <= t < T_EYES + 0.3)
    rs["flash"] = clamp(fl)
    g0 = lin(t, [(0, 0.12), (T_WALK3[0], 0.15), (T_WALK3[1], 0.42), (T_EYES, 0.5), (T_EYES + 0.3, 0.92, back_out)])
    g1 = lin(t, [(0, 0.2), (T_WALK3[0], 0.22), (T_WALK3[1], 0.5), (T_EYES, 0.55), (T_EYES + 0.4, 0.98, back_out)])
    for ti in T_IMPACTS:
        g0 += 0.06 * impulse(t, ti, 6, 4)
        g1 += 0.05 * impulse(t, ti, 6, 5)
    rs["gauges"] = (g0, g1)
    rs["alarm"] = alarm(t)
    rs["popped"] = (5,) if t >= T_IMPACTS[1] else ()
    if t >= T_IMPACTS[2]:
        rs["popped"] = (5, 11)
    return rs


def alarm(t):
    if t < T_EYES + 0.35:
        return 0.0
    return 0.5 + 0.5 * math.sin((t - T_EYES) * 2 * PI * 1.4 - PI / 2)


def pod_states(t, occ):
    ps = {}
    # Opus pod
    phi = 0.0
    if t >= T_DOOR_O:
        u = seg(t, T_DOOR_O, T_DOOR_O + 0.32)
        phi = 2.05 * back_out(u, 2.2) - 0.06 * math.sin(seg(t, T_DOOR_O + 0.32, T_DOOR_O + 0.9) * PI)
    wheel = -2 * 2 * PI * ease_io(seg(t, *T_WHEEL_O))
    wipe = None
    if t >= T_WIPE[0]:
        u = seg(t, *T_WIPE)
        wipe = wipe_path(OX, 1.18, u, KO)
    ps["opus"] = dict(phi=phi, wheel=wheel, frost=0.78, wipe=wipe, occupant=occ.get("opus"),
                      gauge=lin(t, [(0, 0.15), (T_WHEEL_O[0], 0.2), (T_WHEEL_O[1], 0.95, ease_in), (T_DOOR_O + 0.3, 0.1)]),
                      jets=seg(t, 6.15, 6.35) * (t < T_DOOR_O), light=1.0)
    # Sonnet pod
    phi = 0.0
    if t >= T_DOOR_S:
        phi = 2.0 * back_out(seg(t, T_DOOR_S, T_DOOR_S + 0.3), 2.0)
    wheel = -3 * PI * ease_io(seg(t, *T_WHEEL_S))
    flick = 0.65 + 0.35 * math.sin(t * 23) * math.sin(t * 7.3)
    ps["sonnet"] = dict(phi=phi, wheel=wheel, frost=0.9, occupant=occ.get("sonnet"),
                        screen=flick if t < T_DOOR_S else 0.0,
                        gauge=lin(t, [(0, 0.3), (T_WHEEL_S[0], 0.3), (T_WHEEL_S[1], 0.9, ease_in), (T_DOOR_S + 0.3, 0.15)]),
                        jets=seg(t, 11.5, 11.65) * (t < T_DOOR_S))
    # Haiku pod (rattles: Haiku can't wait)
    phi = 0.0
    if t >= T_DOOR_H:
        phi = 2.2 * back_out(seg(t, T_DOOR_H, T_DOOR_H + 0.18), 2.5)
    wheel = -2 * PI * ease_io(seg(t, *T_WHEEL_H))
    amp = seg(t, 14.6, 14.9) * (t < T_DOOR_H) * (0.6 + 0.4 * max(0, math.sin(t * 5.0)))
    jig = (0.012 * math.sin(t * 57) * amp, 1.6 * math.sin(t * 43) * amp)
    ps["haiku"] = dict(phi=phi, wheel=wheel, frost=0.85, occupant=occ.get("haiku"), jiggle=jig,
                       gauge=0.5 + 0.4 * amp * math.sin(t * 30))
    return ps


def wipe_path(X, y, u, k):
    """Clear patch Opus rubs into the frost: a growing blobby ellipse."""
    if u <= 0:
        return None
    n = 28
    rx = 0.36 * k * ease_out(u)
    ry = 0.24 * k * ease_out(u)
    pts = []
    for i in range(n):
        a = i / n * 2 * PI
        wob = 1.0 + 0.12 * math.sin(a * 5 + 1.3) + 0.06 * math.sin(a * 9)
        pts.append((X + math.cos(a) * rx * wob, y + math.sin(a) * ry * wob))
    from .gfx import smooth_path
    return smooth_path(pts)


# ------------------------------------------------------------------ bursts

@lru_cache(None)
def all_bursts():
    B = S.Burst
    b = []
    b.append(B(T_DOOR_O - 0.02, OX, 1.25, 18, 1.5, (0.0, 0.45), 0.30, 1.6, 1, up=0.9))
    b.append(B(T_DOOR_O + 0.05, OX - 0.45, 0.6, 10, 1.1, (-0.8, 0.1), 0.22, 1.3, 2))
    b.append(B(T_DOOR_O + 0.08, OX + 0.5, 1.7, 8, 0.9, (0.5, 0.6), 0.2, 1.3, 3, up=0.8))
    b.append(B(T_JUMP_O[1] - 0.02, OX - 0.34, LANE_O + 0.04, 10, 0.7, (0, 0.05), 0.11, 0.8, 4,
               color="#EFE7D6", up=0.1, shade_col="#D6CBB4", front=True))
    b.append(B(T_DOOR_S - 0.02, SX, 0.95, 13, 1.0, (0.15, 0.4), 0.22, 1.3, 5, up=0.8))
    b.append(B(T_JUMP_S[1] - 0.02, SX + 0.1, LANE_S + 0.03, 8, 0.5, (0, 0.05), 0.08, 0.7, 6,
               color="#EFE7D6", up=0.1, shade_col="#D6CBB4", front=True))
    b.append(B(T_DOOR_H - 0.01, HX + 0.15, 0.6, 9, 0.8, (0.8, 0.2), 0.14, 1.0, 7, up=0.6))
    b.append(B(T_DASH_IN[1] - 0.06, HX + 0.66, LANE_H + 0.03, 12, 0.5, (0.6, 0.06), 0.1, 0.9, 8,
               color="#EFE7D6", up=0.12, shade_col="#D6CBB4", front=True))
    b.append(B(T_DASH_OUT[0] + 0.05, HX + 0.4, LANE_H + 0.15, 6, 0.4, (-0.4, 0.05), 0.08, 0.6, 9,
               color="#EFE7D6", up=0.1, shade_col="#D6CBB4", front=True))
    # dust ringing from the porthole flange on impacts
    for i, ti in enumerate(T_IMPACTS):
        cx, cy = S.PORT_C
        for j in range(5):
            a = (j / 5) * 2 * PI + i * 0.9 + 0.3
            r = S.RING_R + 0.02
            b.append(B(ti, cx + math.cos(a) * r, cy + math.sin(a) * r, 4, 0.4, (math.cos(a) * 0.45, math.sin(a) * 0.45),
                       0.08 + 0.025 * i, 0.75, 20 + i * 10 + j, color="#EFE7D6", up=-0.4, shade_col="#D6CBB4"))
    b.append(B(T_IMPACTS[2] + 0.58, H_FINAL + 0.25, LANE_H + 0.03, 8, 0.5, (0, 0.05), 0.1, 0.8, 60,
               color="#EFE7D6", up=0.1, shade_col="#D6CBB4", front=True))
    b.append(B(T_IMPACTS[2] + 0.52, S_FINAL - 0.2, LANE_S + 0.03, 8, 0.5, (0, 0.05), 0.11, 0.8, 61,
               color="#EFE7D6", up=0.1, shade_col="#D6CBB4", front=True))
    return b


# ------------------------------------------------------------------ fx

def fx_code(g, t, st):
    """Glyphs floating up from Sonnet's laptop."""
    if not (T_TYPE[0] <= t < T_TYPE[1] + 1.0):
        return
    from .clawd import CODE
    n = int((T_TYPE[1] - T_TYPE[0]) / 0.07)
    for i in range(n):
        t0 = T_TYPE[0] + i * 0.07
        age = t - t0
        if age < 0 or age > 0.9:
            continue
        u = age / 0.9
        x = st.x + (hash01("gx", i) - 0.5) * 0.45 + math.sin(age * 4 + i) * 0.03
        y = st.y + 0.74 + age * 0.7
        a = (1 - u) * min(1, age * 8)
        c = (P.CODE_GREEN, "#FFFFFF", P.SCREEN, P.ORANGE_LIGHT)[i % 4]
        g.text(CODE[i % len(CODE)], x, y, 0.10 + 0.03 * hash01("gs", i), c, "mono", a=a, outline=P.INK, ow=0.016)


def fx_pass(g, t, st):
    if not (T_PASS <= t < T_PASS + 1.0):
        return
    age = t - T_PASS
    s = back_out(min(1, age / 0.25), 2.5)
    a = 1 - seg(t, T_PASS + 0.7, T_PASS + 1.0)
    x, y = st.x, st.y + 0.95 + 0.15 * age
    w, h = 0.62 * s, 0.17 * s
    if s <= 0.01:
        return
    p = rrect_path(x - w / 2, y - h / 2, x + w / 2, y + h / 2, h / 2)
    g.fill(p, P.CODE_GREEN, a)
    g.stroke(p, 1.0, P.INK, a)
    g.text("✓ TESTS PASS", x, y - 0.045 * s, 0.10 * s, "#FFFFFF", "sans", a=a)


def fx_speed(g, t, st):
    _, _, v = gait("haiku", t, WH)
    if v < 2.5 or not (T_DASH_OUT[0] <= t < T_DASH_IN[1] + 0.1):
        return
    a = clamp((v - 2.5) / 4)
    for i in range(7):
        yy = st.y + 0.06 + i * 0.065 + st.lift
        L = 0.3 + 0.5 * hash01("sl", i, int(t * 30))
        x0 = st.x - 0.33 - 0.05 * i % 3
        g.line(x0 - L, yy, x0, yy, 1.1, P.INK, 0.55 * a)


def fx_ghosts(g, t, st_fn):
    from .clawd import draw_clawd
    _, _, v = gait("haiku", t, WH)
    if v < 3.0 or not (T_DASH_OUT[0] <= t < T_DASH_IN[1]):
        return
    for k, dt in enumerate((0.05, 0.03, 0.015)):
        s2, _ = st_fn(t - dt)
        s2.alpha = 0.12 + 0.1 * k
        s2.shadow = False
        draw_clawd(g, s2, t - dt)


def fx_tap(g, t):
    for tt in (T_TAP[0] + 0.2, T_TAP[0] + 0.38, T_TAP[0] + 0.56):
        age = t - tt
        if 0 <= age < 0.35:
            u = age / 0.35
            x, y = O_FINAL + 0.32, S_TOP + 0.62
            g.stroke(circle_path(x, y, 0.04 + 0.22 * u), 1.0, "#FFFFFF", 0.7 * (1 - u))
            g.stroke(circle_path(x, y, 0.02 + 0.1 * u), 0.7, "#FFFFFF", 0.7 * (1 - u))


def fx_alert(g, t, x, y, t0, t1, s=0.32):
    if not (t0 <= t < t1):
        return
    k = back_out(seg(t, t0, t0 + 0.18), 3)
    a = 1 - seg(t, t1 - 0.15, t1)
    if k <= 0.01:
        return
    sz = s * k
    g.text("!", x, y, sz, P.ALARM, "slab", a=a, outline=P.INK, ow=0.025 * k)


def fx_bolt(g, t):
    t0 = T_IMPACTS[1]
    if not (t0 <= t < t0 + 0.8):
        return
    u = (t - t0) / 0.8
    a0 = math.radians(90 + 5 * 22.5)
    cx, cy = S.PORT_C
    bx, by = cx + math.cos(a0) * (S.PORT_R + 0.27), cy + math.sin(a0) * (S.PORT_R + 0.27)
    x = bx - 2.4 * u
    y = by + 1.4 * u - 3.8 * u * u
    r = 0.09 * (1 + 2.2 * u)
    g.c.save()
    g.c.translate(x, y)
    g.c.rotate(u * 900)
    g.fs(rrect_path(-r, -r * 0.6, r, r * 0.6, r * 0.2), P.NAVY_DARK, 1.0)
    g.fs(rect_path(-r * 0.3, -r * 1.8, r * 0.3, -r * 0.6), "#9A9FB3", 0.8)
    g.c.restore()


def fx_dust(g, t):
    """Specks falling from the ceiling during the rumble and the hits."""
    srcs = [(T_RUMBLE + 0.4, 4.0, 7.2, 40)] + [(ti, S.XR - 4.0, S.XR + 4.0, 30) for ti in T_IMPACTS]
    for (t0, x0, x1, n) in srcs:
        for i in range(n):
            st = t0 + hash01("dt", t0, i) * 0.8
            age = t - st
            if age < 0 or age > 1.6:
                continue
            x = lerp(x0, x1, hash01("dx", t0, i)) + math.sin(age * 3 + i) * 0.05
            ytop = g.cam.cy + g.cam.h / 2
            y = ytop - 0.5 * 9.8 * 0.18 * age * age * (0.7 + 0.6 * hash01("dv", i)) * 3
            r = 0.008 + 0.012 * hash01("dr", t0, i)
            g.fill(circle_path(x, y, r * g.cam.h / 3), P.INK_SOFT, 0.6 * (1 - age / 1.6))


def title_overlay(t):
    def draw(cv, w, h):
        if t < T_END_EYES:
            return
        k = h / 2160.0
        sx, sy, _ = shake(t)
        ox, oy = sx * 900 * k, sy * 900 * k
        cx, cy = w / 2 + ox, h * 0.42 + oy
        # the eyes, once more, in the dark
        op = ease_out(seg(t, T_END_EYES, T_END_EYES + 0.12))
        op *= 1.0 - 0.95 * math.exp(-((t - 34.45) / 0.045) ** 2)
        a_all = 1.0 - seg(t, *T_FADE)
        if op > 0.02:
            for ex in (-1, 1):
                ew, eh = 34 * k, 78 * k * op
                r = skia.Rect(cx + ex * 210 * k - ew, cy - eh, cx + ex * 210 * k + ew, cy + eh)
                p = skia.Paint(AntiAlias=True, Color=col(P.GLOW, 0.8 * a_all))
                p.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, 40 * k))
                cv.drawRect(r.makeOutset(20 * k, 20 * k), p)
                cv.drawRoundRect(r, 8 * k, 8 * k, skia.Paint(AntiAlias=True, Color=col(P.GLOW, a_all)))
                cv.drawRoundRect(r.makeInset(12 * k, 18 * k * op), 6 * k, 6 * k,
                                 skia.Paint(AntiAlias=True, Color=col(P.GLOW_CORE, a_all)))
        ta = ease_out(seg(t, T_TITLE, T_TITLE + 0.5)) * a_all
        if ta > 0.01:
            f = skia.Font(typeface("slab"), 280 * k)
            s = "FABLE 5.5"
            wtxt = f.measureText(s)
            cv.drawString(s, w / 2 - wtxt / 2 + ox * 0.6, h * 0.74 + oy * 0.6, f,
                          skia.Paint(AntiAlias=True, Color=col("#F1E9D8", ta)))
            lw = 680 * k * ease_out(seg(t, T_TITLE + 0.2, T_TITLE + 0.8))
            cv.drawRect(skia.Rect(w / 2 - lw, h * 0.785, w / 2 + lw, h * 0.785 + 8 * k),
                        skia.Paint(Color=col(P.ORANGE, ta)))
    return draw


# ------------------------------------------------------------------ assemble

def build(t):
    st = State(t=t)
    st.cam = camera(t)
    sx, sy, rot = shake(t)
    st.shake = (sx * st.cam.h, sy * st.cam.h)
    st.cam_rot = rot
    # lights
    lamps = {}
    for k, t0 in T_LAMP.items():
        on = 1.0 if t >= t0 else 0.0
        if t0 <= t < t0 + 0.2:
            d = t - t0
            on = [1, 0.2, 1, 0.6, 1][min(4, int(d / 0.04))]
        if T_EYES <= t < T_EYES + 0.35:
            on *= [1, 0.4, 0.9, 0.2, 1, 0.5, 1][min(6, int((t - T_EYES) / 0.05))]
        lamps[k] = on
    st.lamps = lamps
    st.night = lerp(1.0, 0.0, smooth(seg(t, *T_AMBIENT))) + 0.35 * smooth(seg(t, T_EYES, T_EYES + 0.3))
    sway = 0.0
    sway += 2.5 * impulse(t, 19.0, 0.9, 0.6) * (t >= T_RUMBLE)
    for ti in T_IMPACTS:
        sway += 2.0 * impulse(t, ti, 1.0, 0.7)
    st.lamp_sway = sway
    # characters
    o, o_in = opus_state(t)
    s, s_in = sonnet_state(t)
    h, h_in = haiku_state(t)
    occ = {}
    if o_in:
        o.shadow = False
        occ["opus"] = o
    if s_in:
        s.shadow = False
        occ["sonnet"] = s
    if h_in:
        h.shadow = False
        occ["haiku"] = h
    st.pods = pod_states(t, occ)
    st.reactor = reactor_state(t)
    st.chars = [c for c, inside in ((o, o_in), (s, s_in), (h, h_in)) if not inside]
    st.bursts = all_bursts()
    st.fx_back = [lambda g: fx_tap(g, t)]
    st.fx_front = [
        lambda g: fx_ghosts(g, t, haiku_state),
        lambda g: fx_code(g, t, s),
        lambda g: fx_pass(g, t, s),
        lambda g: fx_speed(g, t, h),
        lambda g: fx_alert(g, t, h.x + 0.35, h.y + h.lift + 0.75, T_EYES + 0.1, T_IMPACTS[0] - 0.05),
        lambda g: fx_alert(g, t, s.x, s.y + s.lift + 0.95, 19.0, 19.75, 0.26),
        lambda g: fx_alert(g, t, o.x + 0.1, o.y + o.lift + 1.15, 19.08, 19.8, 0.3),
        lambda g: fx_alert(g, t, h.x, h.y + h.lift + 0.75, 19.0, 19.7, 0.22),
        lambda g: fx_alert(g, t, o.x + 0.1, o.y + o.lift + 1.15, T_IMPACTS[2] + 0.3, T_BLACK, 0.3),
        lambda g: fx_alert(g, t, s.x, s.y + s.lift + 0.95, T_IMPACTS[2] + 0.4, T_BLACK, 0.26),
        lambda g: fx_alert(g, t, h.x, h.y + h.lift + 0.75, T_IMPACTS[2] + 0.5, T_BLACK, 0.22),
        lambda g: fx_bolt(g, t),
        lambda g: fx_dust(g, t),
    ]
    st.alarm = 0.55 * alarm(t) * (t < T_BLACK)
    st.flash = 0.35 * sum(impulse(t, ti, 14) for ti in T_IMPACTS) + 0.25 * impulse(t, T_DOOR_O, 12)
    if t >= T_BLACK:
        st.black = True
        st.alarm = 0.0
        st.flash = 0.0
        st.overlay = [title_overlay(t)]
    st.fade = 1.0 - seg(t, 0.0, 0.45)
    return st


def cue_list():
    """Sound cues (time, name) shared with the audio generator."""
    c = []
    for k, t0 in T_LAMP.items():
        c.append((t0, "lamp_big" if k.startswith("fable") else "lamp"))
    c.append((T_OPUS_WAKE, "blink"))
    for i in range(6):
        c.append((T_WIPE[0] + i * 0.11, "squeak"))
    c.append((T_WHEEL_O[0], "wheel_long"))
    c.append((6.15, "hiss_short"))
    c.append((T_DOOR_O, "door_big"))
    c.append((T_JUMP_O[0], "jump"))
    c.append((T_JUMP_O[1], "land_big"))
    c.append((T_CHECK1[0], "scribble"))
    for t0, t1, W, lane in ((T_WALK1[0], T_WALK1[1], WO, "o"), (T_WALK2[0], T_WALK2[1], WO, "o"),
                            (T_WALK3[0], T_WALK3[1], WO, "o")):
        c.append((t0, f"steps:{t1:.2f}:{W}"))
    c.append((T_WHEEL_S[0], "wheel_mid"))
    c.append((11.5, "hiss_short"))
    c.append((T_DOOR_S, "door_mid"))
    c.append((T_JUMP_S[1], "land_mid"))
    c.append((T_TYPE[0], f"typing:{T_TYPE[1]:.2f}"))
    c.append((T_PASS, "ding"))
    c.append((T_CHECK2[0], "tick"))
    c.append((14.6, "rattle:16.0"))
    c.append((T_WHEEL_H[0], "wheel_small"))
    c.append((T_DOOR_H, "door_small"))
    c.append((T_DASH_OUT[0], "whoosh"))
    c.append((T_DASH_IN[0] + 0.05, "whoosh2"))
    c.append((T_DASH_IN[1] - 0.12, "skid"))
    c.append((17.08, "salute"))
    c.append((T_CHECK3[0], "scribble"))
    c.append((T_CIRCLE4[0], "scribble_long"))
    c.append((T_RUMBLE, "rumble"))
    c.append((T_HOP_S[0], "hop"))
    c.append((T_HOP_S[1], "land_small"))
    c.append((T_HOP_H[0], "hop"))
    c.append((T_HOP_H[1], "land_small"))
    for i in range(3):
        c.append((T_TAP[0] + 0.2 + i * 0.18, "tap"))
    c.append((T_EYES, "eyes_on"))
    c.append((T_EYES + 0.35, "alarm"))
    c.append((29.3, "growl"))
    for i, ti in enumerate(T_IMPACTS):
        c.append((ti, f"impact:{i}"))
    c.append((T_IMPACTS[1] + 0.02, "bolt"))
    c.append((T_IMPACTS[2] + 0.5, "tumble"))
    c.append((T_BLACK, "cut"))
    c.append((T_END_EYES, "eyes_small"))
    c.append((T_THUD, "thud"))
    return sorted(c)
