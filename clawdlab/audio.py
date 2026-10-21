"""Procedural soundtrack: every sound is synthesised with numpy and placed on
the cue times exported by story.cue_list(), so picture and sound share one
clock.  Footsteps are derived from each Clawd's gait phase."""
import math

import numpy as np
from scipy import signal
from scipy.io import wavfile

from . import story

SR = 48000
RNG = np.random.default_rng(1234)


# ------------------------------------------------------------------ helpers

def tt(d):
    return np.arange(int(d * SR)) / SR


def noise(d):
    return RNG.standard_normal(int(d * SR))


def env(d, a=0.005, decay=0.2, hold=0.0):
    t = tt(d)
    e = np.minimum(1.0, t / max(a, 1e-4))
    e *= np.where(t < a + hold, 1.0, np.exp(-(t - a - hold) / max(decay, 1e-4)))
    return e


def bp(x, lo, hi, order=2):
    b, a = signal.butter(order, [lo / (SR / 2), min(hi / (SR / 2), 0.99)], "band")
    return signal.lfilter(b, a, x)


def lp(x, f, order=2):
    b, a = signal.butter(order, min(f / (SR / 2), 0.99), "low")
    return signal.lfilter(b, a, x)


def hp(x, f, order=2):
    b, a = signal.butter(order, f / (SR / 2), "high")
    return signal.lfilter(b, a, x)


def sine(f, d, phase=0.0):
    t = tt(d)
    if callable(f):
        ph = 2 * np.pi * np.cumsum(f(t)) / SR
    else:
        ph = 2 * np.pi * f * t
    return np.sin(ph + phase)


def partials(freqs, amps, decays, d):
    t = tt(d)
    out = np.zeros_like(t)
    for f, a, k in zip(freqs, amps, decays):
        out += a * np.sin(2 * np.pi * f * t + RNG.uniform(0, 6)) * np.exp(-t / k)
    return out


def place(out, i, x):
    i = max(0, i)
    n = min(len(x), len(out) - i)
    if n > 0:
        out[i:i + n] += x[:n]


def norm(x, peak=1.0):
    m = np.max(np.abs(x)) + 1e-9
    return x / m * peak


class Mix:
    def __init__(self, dur):
        self.n = int(dur * SR) + SR
        self.dry = np.zeros((self.n, 2))
        self.wet = np.zeros((self.n, 2))

    def add(self, t, x, gain=1.0, pan=0.0, rev=0.25):
        i = int(t * SR)
        if i >= self.n or len(x) == 0:
            return
        x = x[: self.n - i] * gain
        l = math.cos((pan + 1) * math.pi / 4)
        r = math.sin((pan + 1) * math.pi / 4)
        self.dry[i:i + len(x), 0] += x * l
        self.dry[i:i + len(x), 1] += x * r
        if rev > 0:
            self.wet[i:i + len(x), 0] += x * l * rev
            self.wet[i:i + len(x), 1] += x * r * rev

    def render(self):
        # lab reverb: decaying noise impulse, slightly different per channel
        d = 1.4
        t = tt(d)
        ir = []
        for c in range(2):
            n = lp(RNG.standard_normal(len(t)), 6000) * np.exp(-t / 0.32)
            n[: int(0.012 * SR)] = 0
            ir.append(n / np.sqrt(np.sum(n ** 2)))
        out = self.dry.copy()
        for c in range(2):
            out[:, c] += signal.fftconvolve(self.wet[:, c], ir[c])[: self.n] * 0.9
        return out


# ------------------------------------------------------------------ sounds

def s_clunk(big=False):
    d = 1.2 if big else 0.6
    th = sine(lambda t: 95 - 35 * np.minimum(1, t / 0.08), d) * env(d, 0.002, 0.09)
    click = hp(noise(d), 2500) * env(d, 0.0005, 0.006)
    t = tt(d)
    buzz = np.sign(np.sin(2 * np.pi * 120 * t)) * 0.5 + np.sin(2 * np.pi * 240 * t)
    buzz = lp(buzz, 900) * np.minimum(1, t / 0.02) * np.exp(-t / (0.5 if big else 0.25)) * 0.12
    metal = partials([310, 742, 1290], [0.3, 0.2, 0.1], [0.12, 0.08, 0.05], d)
    return th * (1.2 if big else 0.8) + click * 0.5 + buzz + metal * 0.4


def s_blip():
    d = 0.08
    return sine(1300, d) * env(d, 0.002, 0.02) * 0.3


def s_squeak():
    d = 0.11
    f = lambda t: 1900 + 500 * t / d + 60 * np.sin(2 * np.pi * 45 * t)
    x = sine(f, d) + 0.4 * sine(lambda t: 2 * f(t), d)
    return bp(x, 1500, 5000) * env(d, 0.01, 0.04, 0.05) * 0.25


def s_ratchet(d, rate=13.0, pitch=1.0):
    out = np.zeros(int(d * SR) + SR // 4)
    k = 0
    while k / rate < d:
        tclick = k / rate + RNG.uniform(-0.006, 0.006)
        c = partials([2100 * pitch, 3150 * pitch, 4400 * pitch], [0.5, 0.35, 0.2], [0.025, 0.018, 0.012], 0.12)
        c += hp(noise(0.12), 1500) * env(0.12, 0.0005, 0.004) * 0.5
        place(out, int(tclick * SR), c)
        k += 1
    t = np.arange(len(out)) / SR
    creak = signal.sawtooth(2 * np.pi * (68 * pitch + 6 * np.sin(2 * np.pi * 3.1 * t)) * t)
    creak = lp(creak, 700) * np.clip(np.minimum(t / 0.05, (d - t) / 0.1), 0, 1) * 0.12
    return out * 0.5 + creak


def s_hiss(d, f=2500, a=0.02, decay=0.5):
    x = hp(noise(d), f) + 0.3 * bp(noise(d), 600, 1800)
    return x * env(d, a, decay) * 0.35


def s_clang(base, d=1.2):
    fr = np.array([1.0, 2.76, 5.40, 8.93, 13.3]) * base
    return partials(fr, [0.6, 0.45, 0.3, 0.18, 0.1], [0.5, 0.35, 0.25, 0.15, 0.1], d)


def s_door(scale):
    d = 2.4 * scale + 0.4
    x = np.zeros(int(d * SR))
    cl = s_clang(170 / scale, 1.4)
    x[: len(cl)] += cl * 0.9
    th = sine(lambda t: 80 / scale - 30 * np.minimum(1, t / 0.1), 0.6) * env(0.6, 0.002, 0.12)
    x[: len(th)] += th * 1.2
    hs = s_hiss(d, 1800, 0.015, 0.55 * scale + 0.15)
    x += hs * (1.6 * scale)
    return x


def s_whoosh(d=0.4, f0=300, f1=3200):
    n = noise(d)
    t = tt(d)
    out = np.zeros_like(n)
    seg = 512
    for i in range(0, len(n), seg):
        u = i / len(n)
        fc = f0 * (f1 / f0) ** u
        out[i:i + seg] = bp(n[max(0, i - 2048):i + seg], fc * 0.7, fc * 1.4)[-len(n[i:i + seg]):]
    e = np.sin(np.pi * np.clip(t / d, 0, 1)) ** 1.5
    return out * e * 0.8


def s_thud(f=70, d=0.35, g=1.0):
    th = sine(lambda t: f * (1 + 0.6 * np.exp(-t / 0.02)), d) * env(d, 0.001, d / 3.5)
    nz = lp(noise(d), 400) * env(d, 0.001, 0.04) * 0.6
    return (th + nz) * g


def s_step(W):
    d = 0.12
    f = 160 / W
    x = sine(f, d) * env(d, 0.001, 0.025) * 0.6
    x += bp(noise(d), 800 / W, 3000 / W) * env(d, 0.0005, 0.01) * 0.5
    return x


def s_scribble(d, strokes):
    t = tt(d)
    n = bp(noise(d), 1800, 7000)
    am = 0.5 + 0.5 * np.sin(2 * np.pi * strokes / d * t - np.pi / 2)
    grain = 0.7 + 0.3 * np.sin(2 * np.pi * 90 * t) * (RNG.random(len(t)) > 0.3)
    return n * am * grain * np.clip(np.minimum(t / 0.02, (d - t) / 0.04), 0, 1) * 0.35


def s_key():
    d = 0.06
    x = bp(noise(d), 1800, 6000) * env(d, 0.0005, 0.006)
    x += sine(RNG.uniform(380, 520), d) * env(d, 0.0005, 0.008) * 0.4
    return x * RNG.uniform(0.5, 1.0)


def s_bell(f, d=1.4):
    return partials([f, f * 2.0, f * 3.01, f * 4.17], [0.6, 0.25, 0.12, 0.06], [0.6, 0.35, 0.2, 0.12], d)


def s_tink():
    d = 0.4
    return partials([3150, 4730, 6880], [0.5, 0.3, 0.15], [0.09, 0.06, 0.04], d) + \
        hp(noise(d), 4000) * env(d, 0.0003, 0.003) * 0.3


def s_rumble(d=2.4):
    t = tt(d)
    br = lp(np.cumsum(noise(d)) / 300, 90)
    br = br / (np.max(np.abs(br)) + 1e-9)
    sub = np.sin(2 * np.pi * 38 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 7 * t))
    groan = lp(signal.sawtooth(2 * np.pi * (115 - 25 * t / d) * t), 500) * 0.25
    e = np.clip(np.minimum(t / 0.5, (d - t) / 1.0), 0, 1)
    return (br * 0.8 + sub * 0.7 + groan) * e


def s_crackle(d=0.5, density=900, g=1.0):
    out = np.zeros(int(d * SR))
    n = int(density * d)
    for _ in range(n):
        tc = RNG.uniform(0, d) ** 1.8 / d ** 0.8
        i = int(tc * SR)
        L = int(RNG.uniform(0.0004, 0.002) * SR)
        if i + L >= len(out):
            continue
        out[i:i + L] += RNG.standard_normal(L) * RNG.uniform(0.2, 1.0) * math.exp(-tc / (d * 0.4))
    return hp(out, 2500) * g


def s_impact(i):
    d = 2.6
    boom = sine(lambda t: 42 + 70 * np.exp(-t / 0.05), d) * env(d, 0.001, 0.45 + 0.15 * i) * (1.4 + 0.3 * i)
    body = lp(noise(d), 260) * env(d, 0.001, 0.18) * 1.6
    cl = s_clang(110 - 12 * i, d) * (0.5 + 0.15 * i)
    cr = s_crackle(0.6 + 0.2 * i, 700 + 500 * i, 0.9 + 0.3 * i)
    x = boom + body + cl
    place(x, 0, cr)
    return x


def s_ping():
    d = 0.9
    return sine(lambda t: 2650 - 300 * t, d) * env(d, 0.001, 0.22) * 0.5 + \
        sine(lambda t: 3980 - 450 * t, d) * env(d, 0.001, 0.12) * 0.25


def s_power_up(d=1.6):
    t = tt(d)
    f = 45 + 20 * np.minimum(1, t / 0.9)
    saw = signal.sawtooth(2 * np.pi * np.cumsum(f) / SR)
    x = np.zeros_like(saw)
    seg = 1024
    for k in range(0, len(saw), seg):
        u = min(1, k / SR / 1.0)
        x[k:k + seg] = lp(saw[max(0, k - 4096):k + seg], 120 + 1600 * u)[-len(saw[k:k + seg]):]
    e = np.clip(t / 0.5, 0, 1) * np.clip((d - t) / 0.4, 0, 1)
    return x * e * 0.8


def s_drone(d):
    t = tt(d)
    x = signal.sawtooth(2 * np.pi * 55 * t) + 0.6 * signal.sawtooth(2 * np.pi * 55.4 * t) + \
        0.4 * np.sin(2 * np.pi * 110 * t)
    x = lp(x, 340)
    return x * (0.85 + 0.15 * np.sin(2 * np.pi * 0.7 * t)) * 0.35


def s_klaxon(d, rate=1.4):
    t = tt(d)
    ph = (t * rate) % 1.0
    f = 520 + 380 * np.minimum(1, ph / 0.55)
    x = np.sign(np.sin(2 * np.pi * np.cumsum(f) / SR)) * 0.5 + np.sin(2 * np.pi * np.cumsum(f) / SR)
    x = bp(x, 400, 2400)
    gate = np.clip(np.minimum(ph / 0.03, (0.8 - ph) / 0.08), 0, 1)
    return x * gate * 0.22


def s_growl(d=0.7):
    t = tt(d)
    n = lp(noise(d), 220) * (0.6 + 0.4 * np.sin(2 * np.pi * 28 * t))
    sub = np.sin(2 * np.pi * (48 - 8 * t) * t)
    return (n * 1.4 + sub * 0.6) * np.clip(np.minimum(t / 0.15, (d - t) / 0.25), 0, 1)


def s_marimba(f, d=0.6):
    return partials([f, f * 3.93, f * 9.2], [0.7, 0.18, 0.05], [0.28, 0.06, 0.02], d) * env(d, 0.002, 1.0)


def s_pizz(f, d=0.7):
    t = tt(d)
    x = signal.sawtooth(2 * np.pi * f * t) * np.exp(-t / 0.18)
    return lp(x, 900) * env(d, 0.004, 1.0) * 0.6


def s_pad(freqs, d, attack=1.5, release=1.2):
    t = tt(d)
    x = np.zeros_like(t)
    for f in freqs:
        for det in (-0.3, 0.0, 0.35):
            x += signal.sawtooth(2 * np.pi * (f + det) * t + RNG.uniform(0, 6))
    x = lp(x, 900) / (3 * len(freqs))
    e = np.clip(np.minimum(t / attack, (d - t) / release), 0, 1)
    return x * e


# ------------------------------------------------------------------ score

NOTE = {n: 440 * 2 ** ((i - 9) / 12) for i, n in enumerate(["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])}


def hz(name):
    n, o = name[:-1], int(name[-1])
    return NOTE[n] * 2 ** (o - 4)


def score(m):
    # curious marimba ostinato while the pods are opened
    bar = 2.0
    mel = [["C5", "E5", "G5", "E5", "A5", "G5", "E5", "D5"],
           ["C5", "E5", "G5", "C6", "B5", "G5", "E5", "G5"],
           ["A4", "C5", "E5", "C5", "F5", "E5", "C5", "B4"],
           ["G4", "B4", "D5", "G5", "F5", "D5", "B4", "D5"]]
    bass = [("C3", "G2"), ("C3", "E3"), ("A2", "E3"), ("G2", "D3")]
    t0, t1 = 4.4, story.T_RUMBLE
    k = 0
    t = t0
    while t < t1 - 0.01:
        b = k % 4
        for j, nn in enumerate(mel[b]):
            tn = t + j * bar / 8
            if tn >= t1:
                break
            if story.T_INSERT_A[0] <= tn < story.T_INSERT_A[1] or story.T_INSERT_B[0] <= tn < story.T_INSERT_B[1]:
                g = 0.09
            else:
                g = 0.12
            m.add(tn, s_marimba(hz(nn)), g * (1.0 if j % 2 == 0 else 0.75), pan=-0.25 + 0.5 * (j % 2), rev=0.35)
        for j, nn in enumerate(bass[b]):
            tn = t + j * bar / 2
            if tn < t1:
                m.add(tn, s_pizz(hz(nn)), 0.2, pan=0.1, rev=0.3)
        t += bar
        k += 1
    # tension pad on the long walk, cut just before the tap
    ta, tb = story.T_WALK3[0] - 0.2, story.T_TAP[0] - 0.05
    pad = s_pad([hz("C3"), hz("D#3"), hz("G3"), hz("B3")], tb - ta, attack=3.5, release=0.25)
    ramp = np.linspace(0.4, 1.0, len(pad))
    m.add(ta, pad * ramp, 0.22, rev=0.5)
    # heartbeat pulses accelerating toward the reactor
    tb_ = story.T_WALK3[0] + 0.5
    period = 1.1
    while tb_ < story.T_TAP[0] - 0.4:
        m.add(tb_, s_thud(52, 0.3, 0.55), 1.0, rev=0.2)
        m.add(tb_ + 0.22, s_thud(48, 0.3, 0.35), 1.0, rev=0.2)
        tb_ += period
        period = max(0.62, period * 0.95)
    # end sting
    m.add(story.T_THUD, s_pad([hz("C2"), hz("G2"), hz("C3"), hz("D#3")], story.DURATION - story.T_THUD + 0.2,
                              attack=0.05, release=1.4), 0.32, rev=0.6)


# ------------------------------------------------------------------ footsteps

def footsteps(m):
    for name, W, pan in (("opus", story.WO, -0.1), ("sonnet", story.WS, -0.3), ("haiku", story.WH, 0.3)):
        ts = np.arange(0, story.DURATION, 1 / 240)
        last = None
        for t in ts:
            ph, amp, v = story.gait(name, t, W)
            k = math.floor(ph / math.pi)
            if last is not None and k != last and amp > 0.35 and v < 4.0 and t < story.T_BLACK:
                m.add(t, s_step(W), 0.22 * W + 0.05, pan=pan, rev=0.15)
            last = k


# ------------------------------------------------------------------ build

def build():
    m = Mix(story.DURATION)
    D = story.DURATION
    # room tone + fluorescent hum once the lights are on
    t = tt(D)
    room = lp(noise(D), 300) * 0.02
    hum = (np.sin(2 * np.pi * 60 * t) * 0.5 + np.sin(2 * np.pi * 120 * t) * 0.3) * 0.012
    hum *= np.clip((t - 2.0) / 0.5, 0, 1)
    bed = room + hum
    bed[t >= story.T_BLACK] = 0
    m.add(0, bed, 1.0, rev=0.0)

    for tc, name in story.cue_list():
        n, _, arg = name.partition(":")
        if n == "lamp":
            m.add(tc, s_clunk(), 0.35, pan=-0.4 + 0.2 * (tc - 0.5), rev=0.4)
        elif n == "lamp_big":
            m.add(tc, s_clunk(True), 0.5, pan=0.4, rev=0.6)
        elif n == "blink":
            m.add(tc, s_blip(), 0.6)
            m.add(tc + 0.47, s_blip(), 0.5)
        elif n == "squeak":
            m.add(tc, s_squeak(), 0.7, pan=-0.05)
        elif n.startswith("wheel"):
            d = {"wheel_long": 0.8, "wheel_mid": 0.8, "wheel_small": 0.4}[n]
            p = {"wheel_long": 0.9, "wheel_mid": 1.1, "wheel_small": 1.35}[n]
            m.add(tc, s_ratchet(d, 13 if p < 1.2 else 18, p), 0.55, rev=0.25)
        elif n == "hiss_short":
            m.add(tc, s_hiss(0.6, 3000, 0.05, 0.2), 0.5)
        elif n == "door_big":
            m.add(tc, s_door(1.0), 0.7, rev=0.45)
        elif n == "door_mid":
            m.add(tc, s_door(0.75), 0.7, pan=0.1, rev=0.4)
        elif n == "door_small":
            m.add(tc, s_door(0.5), 0.6, pan=0.2, rev=0.35)
        elif n == "jump":
            m.add(tc, s_whoosh(0.45, 250, 1400), 0.35)
        elif n == "land_big":
            m.add(tc, s_thud(62, 0.4, 1.0), 0.8, rev=0.3)
        elif n == "land_mid":
            m.add(tc, s_thud(80, 0.3, 0.8), 0.6, rev=0.3)
        elif n == "land_small":
            m.add(tc, s_thud(110, 0.25, 0.6), 0.5, rev=0.25)
        elif n == "hop":
            m.add(tc, sine(lambda x: 320 + 900 * x, 0.12) * env(0.12, 0.003, 0.05) * 0.4, 0.5)
        elif n == "scribble":
            m.add(tc + 0.08, s_scribble(0.32, 2), 0.8)
        elif n == "scribble_long":
            m.add(tc, s_scribble(0.6, 3), 0.7)
        elif n == "tick":
            m.add(tc, s_scribble(0.2, 2), 0.45)
        elif n == "typing":
            t_end = float(arg)
            tk = tc
            while tk < t_end:
                m.add(tk, s_key(), 0.5, pan=0.15 + RNG.uniform(-0.1, 0.1), rev=0.1)
                tk += RNG.uniform(0.035, 0.085)
            # muffled typing inside the closed pod
            tk = 9.7
            while tk < story.T_DOOR_S - 0.4:
                m.add(tk, lp(s_key(), 1800), 0.18, pan=0.1, rev=0.05)
                tk += RNG.uniform(0.04, 0.1)
        elif n == "ding":
            m.add(tc, s_bell(1318.5), 0.35, pan=0.15, rev=0.35)
            m.add(tc + 0.12, s_bell(1760), 0.35, pan=0.15, rev=0.35)
        elif n == "rattle":
            t_end = float(arg)
            tk = tc
            while tk < t_end:
                burst = partials([1600, 2450, 3900], [0.4, 0.3, 0.2], [0.02, 0.015, 0.01], 0.06)
                amp = 0.6 + 0.4 * max(0, math.sin(tk * 5.0))
                m.add(tk, burst, 0.35 * amp, pan=0.35, rev=0.15)
                tk += RNG.uniform(0.025, 0.05)
        elif n == "whoosh":
            m.add(tc, s_whoosh(0.32, 400, 4000), 0.9, pan=0.6)
        elif n == "whoosh2":
            x = s_whoosh(0.45, 300, 3500)
            for k in range(4):  # pan sweep left -> right
                seg = x[k * len(x) // 4:(k + 1) * len(x) // 4]
                m.add(tc + k * 0.45 / 4, seg, 0.9, pan=-0.8 + 0.5 * k)
        elif n == "skid":
            d = 0.38
            sq = sine(lambda x: 2300 + 120 * np.sin(2 * np.pi * 30 * x), d) * env(d, 0.005, 0.12, 0.12)
            m.add(tc, bp(sq, 1500, 4000) * 0.3 + bp(noise(d), 600, 3000) * env(d, 0.003, 0.15) * 0.4, 0.6, pan=0.4)
        elif n == "salute":
            m.add(tc, s_bell(2093, 0.5), 0.12, pan=0.4)
        elif n == "rumble":
            m.add(tc, s_rumble(2.6), 0.42, rev=0.5)
            m.add(tc, s_thud(40, 0.8, 0.9), 0.5, rev=0.4)
        elif n == "tap":
            m.add(tc, s_tink(), 0.5, pan=0.05, rev=0.5)
        elif n == "eyes_on":
            for k, dt in enumerate((0.05, 0.15, 0.2, 0.3)):
                m.add(tc + dt, hp(noise(0.03), 2000) * env(0.03, 0.0005, 0.005), 0.4)
            m.add(tc, s_power_up(1.6), 0.9, rev=0.4)
            d = story.T_BLACK - tc - 0.6
            m.add(tc + 0.6, s_drone(d), 0.35, rev=0.3)
        elif n == "alarm":
            m.add(tc, s_klaxon(story.T_BLACK - tc), 0.3, pan=0.0, rev=0.7)
        elif n == "growl":
            m.add(tc, s_growl(0.6), 0.9, rev=0.4)
        elif n == "impact":
            i = int(arg)
            m.add(tc, s_impact(i), 0.9 + 0.1 * i, rev=0.55)
        elif n == "bolt":
            m.add(tc, s_ping(), 0.45, pan=-0.5, rev=0.4)
            m.add(tc + 0.62, partials([3100, 4400], [0.3, 0.2], [0.05, 0.03], 0.2), 0.3, pan=-0.8)
        elif n == "tumble":
            for k, dt in enumerate((0.0, 0.08, 0.15)):
                m.add(tc + dt, s_thud(90 + 30 * k, 0.25, 0.6), 0.5, pan=-0.3 + 0.3 * k, rev=0.2)
        elif n == "eyes_small":
            m.add(tc, s_power_up(0.9), 0.4, rev=0.6)
        elif n == "thud":
            m.add(tc, s_impact(2), 0.8, rev=0.7)
    footsteps(m)
    score(m)
    out = m.render()
    # hard cut to black: only the end-card sounds survive
    i0 = int(story.T_BLACK * SR)
    i1 = int((story.T_END_EYES - 0.02) * SR)
    fade = int(0.01 * SR)
    out[i0 - fade:i0] *= np.linspace(1, 0, fade)[:, None]
    out[i0:i1] = 0
    # tail fade
    j0 = int(story.T_FADE[0] * SR)
    j1 = int(story.DURATION * SR)
    out[j0:j1] *= np.linspace(1, 0, j1 - j0)[:, None]
    out = out[:j1]
    # gentle bus limiter
    out = np.tanh(out * 1.1) / np.tanh(1.1)
    out = norm(out, 0.89)
    return out


def write(path):
    x = build()
    wavfile.write(path, SR, (x * 32767).astype(np.int16))
    return path
