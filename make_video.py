#!/usr/bin/env python3
"""Render "Clawd Lab" to video (or stills).

  python make_video.py                         # 4K / 30 fps, full film + audio
  python make_video.py --res 1080 --fps 30     # quick 1080p preview
  python make_video.py --stills 5.2,12.4 --res 1080
"""
import argparse
import os
import subprocess
import sys
import time
from multiprocessing import Process

import skia

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from clawdlab import story  # noqa: E402
from clawdlab.render import render  # noqa: E402

RES = {"4k": (3840, 2160), "1440": (2560, 1440), "1080": (1920, 1080), "720": (1280, 720), "540": (960, 540)}


def frame_rgba(t, w, h):
    img = render(story.build(t), w, h)
    return img.toarray(colorType=skia.ColorType.kRGBA_8888_ColorType).tobytes()


def worker(idx, frames, fps, w, h, path, crf, preset):
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}", "-r", str(fps),
           "-i", "-", "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-tune", "animation",
           "-pix_fmt", "yuv420p", "-x264-params", "keyint=60:min-keyint=1", "-threads", "2", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    for k, f in enumerate(frames):
        p.stdin.write(frame_rgba(f / fps, w, h))
        if idx == 0 and k % 20 == 0:
            el = time.time() - t0
            print(f"  worker0 {k}/{len(frames)}  {el / max(k, 1):.2f}s/frame", flush=True)
    p.stdin.close()
    p.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="4k", choices=RES)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=story.DURATION)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument("--crf", type=int, default=16)
    ap.add_argument("--preset", default="slow")
    ap.add_argument("--out", default=None)
    ap.add_argument("--stills", default=None, help="comma separated times -> PNGs")
    ap.add_argument("--no-audio", action="store_true")
    a = ap.parse_args()
    w, h = RES[a.res]
    os.makedirs(os.path.join(ROOT, "out"), exist_ok=True)

    if a.stills:
        for s in a.stills.split(","):
            t = float(s)
            img = render(story.build(t), w, h)
            p = os.path.join(ROOT, "out", f"still_{t:06.2f}.png")
            img.save(p, skia.kPNG)
            print(p)
        return

    out = a.out or os.path.join(ROOT, "out", f"clawd_lab_{a.res}.mp4")
    f0 = int(round(a.start * a.fps))
    f1 = int(round(a.end * a.fps))
    frames = list(range(f0, f1))
    n = a.workers
    chunk = (len(frames) + n - 1) // n
    tmp = os.path.join(ROOT, "out", "parts")
    os.makedirs(tmp, exist_ok=True)
    parts = []
    procs = []
    t0 = time.time()
    for i in range(n):
        fr = frames[i * chunk:(i + 1) * chunk]
        if not fr:
            continue
        path = os.path.join(tmp, f"part_{i:02d}.mp4")
        parts.append(path)
        pr = Process(target=worker, args=(i, fr, a.fps, w, h, path, a.crf, a.preset))
        pr.start()
        procs.append(pr)
    for pr in procs:
        pr.join()
        if pr.exitcode != 0:
            sys.exit(f"worker failed: {pr.exitcode}")
    print(f"rendered {len(frames)} frames in {time.time() - t0:.0f}s")

    lst = os.path.join(tmp, "list.txt")
    with open(lst, "w") as fh:
        for p in parts:
            fh.write(f"file '{p}'\n")
    video_only = os.path.join(tmp, "video.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", video_only],
                   check=True)
    if a.no_audio:
        os.replace(video_only, out)
    else:
        from clawdlab import audio
        wav = os.path.join(ROOT, "out", "soundtrack.wav")
        audio.write(wav)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", video_only, "-ss", f"{a.start:.3f}", "-i", wav,
                        "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-shortest",
                        "-movflags", "+faststart", out], check=True)
    print(out)


if __name__ == "__main__":
    main()
