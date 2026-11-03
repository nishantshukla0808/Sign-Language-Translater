"""
Few-shot phrase matching: compare a new sign against your recorded clips with DTW.
No training; works with a handful of clips per phrase.

    .venv/bin/python fewshot.py loo  --person nishu            # how consistent are my clips?
    .venv/bin/python fewshot.py live --person nishu --phrase hello [--stream-port 8083]

`loo` (leave-one-out) scores every clip against all the OTHER clips of the data set.
`live` runs a guided test: alternating "sign it" and "do NOT sign it" trials, and reports
how often a real sign matched and how often a non-sign was wrongly accepted.
"""

from __future__ import annotations

import argparse
import os
import re
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from landmarks import LHAND_OFF, POSE_OFF, RHAND_OFF

T_LEN = 32
POSE_IDX = [0, 11, 12, 13, 14, 15, 16]         # nose, shoulders, elbows, wrists
LS, RS = 11, 12


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def features(win: np.ndarray) -> np.ndarray:
    """(T,543,3) landmark window -> (T_LEN, 98): shoulder-normalised xy of 7 pose + 2x21 hand points.
    Missing points become 0 (after normalisation)."""
    w = win.astype(np.float64)
    pose = w[:, POSE_OFF:POSE_OFF + 33, :2]
    lh = w[:, LHAND_OFF:LHAND_OFF + 21, :2]
    rh = w[:, RHAND_OFF:RHAND_OFF + 21, :2]
    ls, rs = pose[:, LS], pose[:, RS]
    ok = np.isfinite(ls).all(1) & np.isfinite(rs).all(1)
    if not ok.any():
        return np.zeros((T_LEN, 98))
    centre = np.nanmean((ls[ok] + rs[ok]) / 2, axis=0)
    scale = max(float(np.nanmean(np.linalg.norm(ls[ok] - rs[ok], axis=1))), 1e-3)
    pts = np.concatenate([pose[:, POSE_IDX], lh, rh], axis=1)          # (T, 49, 2)
    pts = (pts - centre) / scale
    pts = np.nan_to_num(pts, nan=0.0)
    idx = np.linspace(0, len(pts) - 1, T_LEN)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, len(pts) - 1)
    f = idx - lo
    out = pts[lo] * (1 - f)[:, None, None] + pts[hi] * f[:, None, None]
    return out.reshape(T_LEN, -1)


def dtw(a: np.ndarray, b: np.ndarray, band: int = 8) -> float:
    """Length-normalised DTW distance with a Sakoe-Chiba band."""
    n, m = len(a), len(b)
    d = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).mean(-1))
    c = np.full((n + 1, m + 1), np.inf)
    c[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(max(1, i - band), min(m, i + band) + 1):
            c[i, j] = d[i - 1, j - 1] + min(c[i - 1, j], c[i, j - 1], c[i - 1, j - 1])
    return float(c[n, m] / (n + m))


def load_templates(data: Path, person: str) -> dict[str, list[tuple[str, np.ndarray]]]:
    out: dict[str, list] = {}
    base = data / slug(person)
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        clips = sorted(d.glob("*.npy"))
        if clips:
            out[d.name] = [(c.name, features(np.load(c))) for c in clips]
    return out


def score(seq: np.ndarray, templates, skip: str | None = None) -> dict[str, float]:
    """Per phrase: mean of the 3 smallest DTW distances to its clips (lower = closer)."""
    res = {}
    for ph, items in templates.items():
        ds = sorted(dtw(seq, f) for name, f in items if name != skip)
        if ds:
            res[ph] = float(np.mean(ds[:3]))
    return res


def cmd_loo(a):
    t = load_templates(Path(a.data), a.person)
    print("clips per phrase:", {k: len(v) for k, v in t.items()})
    for ph, items in t.items():
        own = []
        for name, f in items:
            s = score(f, t, skip=name)
            own.append(s.get(ph, np.nan))
        own = np.array(own)
        print(f"\n{ph}: distance of each clip to its own phrase's other clips (lower = more consistent)")
        print("  min %.3f  median %.3f  max %.3f   -> suggested accept threshold %.3f"
              % (np.nanmin(own), np.nanmedian(own), np.nanmax(own), np.nanmax(own) * 1.25))
        worst = np.argsort(own)[::-1][:3]
        print("  most unusual clips:", ", ".join(f"{items[i][0]} ({own[i]:.3f})" for i in worst))
        others = [p for p in t if p != ph]
        if others:
            conf = sum(1 for name, f in items if min(score(f, t, skip=name), key=score(f, t, skip=name).get) != ph)
            print(f"  nearest phrase is wrong for {conf}/{len(items)} clips")


def cmd_live(a):
    import random

    import mpdetect
    from landmarks import hands_present
    t = load_templates(Path(a.data), a.person)
    want = [slug(x) for x in a.phrase.split(",") if x.strip()]
    for ph in want:
        if ph not in t:
            raise SystemExit(f"no clips for '{ph}' under {a.data}/{slug(a.person)}  (have: {', '.join(t)})")
    thr = {}
    for ph in want:
        loo = np.array([score(f, t, skip=n)[ph] for n, f in t[ph]])
        thr[ph] = float(np.median(loo) * a.margin)
    print("[fewshot] classes in the data: " + ", ".join(f"{k} ({len(v)} clips)" for k, v in t.items()), flush=True)
    print("[fewshot] accept thresholds: " + ", ".join(f"{k} < {v:.3f}" for k, v in thr.items()), flush=True)

    holder: dict = {}

    def cam():
        cap = cv2.VideoCapture(a.camera)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(3, 1280)
        cap.set(4, 720)
        while True:
            ok, f = cap.read()
            if ok:
                holder["f"] = cv2.resize(f, (640, 360), interpolation=cv2.INTER_AREA)

    threading.Thread(target=cam, daemon=True).start()
    streamer = None
    if a.stream_port:
        from streamserver import StreamServer
        streamer = StreamServer(a.stream_port, button="START TEST")
        print(f"[stream] watch at http://<pi-ip>:{a.stream_port}/", flush=True)
    while "f" not in holder:
        time.sleep(0.05)
    det = mpdetect.build_detectors(Path("models"), False)
    tsf = mpdetect.MonotonicTs()
    t0 = time.monotonic()

    trials = [(p, p) for p in want for _ in range(a.trials)] + [("NOT", "") for _ in range(a.trials)]
    random.Random(1).shuffle(trials)
    label = lambda k: "do NOT sign (rest, or any other movement)" if k == "NOT" else f"sign: {k}"
    st = {"i": 0, "phase": "rest", "until": time.monotonic() + 5.0, "buf": [], "res": [], "last": ""}
    if streamer:
        print("[test] ready. Open the page and press START TEST.", flush=True)
        from overlay import draw_overlay
        while not streamer.started.is_set():
            bgr = holder["f"].copy()
            lm = mpdetect.detect_frame(mpdetect.to_mp_image(bgr), tsf((time.monotonic() - t0) * 1000), det)
            draw_overlay(bgr, lm, "idle", 0.0, 0.0, None, 0.0)
            cv2.rectangle(bgr, (0, 0), (640, 44), (0, 0, 0), -1)
            cv2.putText(bgr, "press START TEST above", (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 200, 255), 2, cv2.LINE_AA)
            streamer.publish(bgr)
        st["until"] = time.monotonic() + 5.0
    print(f"[test] {len(trials)} trials in random order. First in 5 s. Rest your hands.", flush=True)
    print(f"[test] next: {label(trials[0][0])}", flush=True)
    while st["i"] < len(trials):
        bgr = holder["f"].copy()
        now = time.monotonic()
        lm = mpdetect.detect_frame(mpdetect.to_mp_image(bgr), tsf((now - t0) * 1000), det)
        kind = trials[st["i"]][0]
        if st["phase"] == "rest" and now >= st["until"]:
            st.update(phase="rec", until=now + a.clip_secs, buf=[])
            print(f"\n   >>> [{st['i'] + 1}/{len(trials)}] NOW: {label(kind).upper()}", flush=True)
        elif st["phase"] == "rec":
            st["buf"].append(lm)
            if now >= st["until"]:
                win = np.stack(st["buf"])
                hands = float(np.mean([hands_present(f) > 0 for f in st["buf"]]))
                sc = score(features(win), t)
                near = min(sc, key=sc.get)
                accepted = near in thr and sc[near] < thr[near]
                said = near if accepted else "none"
                good = (said == kind) if kind != "NOT" else (said == "none")
                st["res"].append((kind, said, sc[near], good))
                ranked = "  ".join(f"{k}:{v:.3f}" for k, v in sorted(sc.items(), key=lambda kv: kv[1]))
                st["last"] = f"{'sign ' + kind if kind != 'NOT' else 'no sign'} -> {said} ({'right' if good else 'WRONG'})"
                print(f"   you did: {kind if kind != 'NOT' else 'nothing'}  ->  matched: {said}   "
                      f"[{'correct' if good else 'WRONG'}]   distances {ranked}   hands seen {hands:.0%}", flush=True)
                st["i"] += 1
                st.update(phase="rest", until=now + a.rest_secs)
                if st["i"] < len(trials):
                    print(f"[test] next: {label(trials[st['i']][0])}", flush=True)
        if streamer:
            from overlay import draw_overlay
            draw_overlay(bgr, lm, "idle", 0.0, 0.0, None, 0.0)
            if st["i"] < len(trials):
                k = trials[st["i"]][0]
                left = max(0.0, st["until"] - now)
                what = "DO NOT SIGN" if k == "NOT" else k.upper()
                txt = f"SIGN NOW: {what}  ({left:.0f}s)" if st["phase"] == "rec" and k != "NOT" else (
                    f"NOW: {what}  ({left:.0f}s)" if st["phase"] == "rec" else f"get ready: {what}  ({left:.0f}s)")
                col = (60, 220, 60) if st["phase"] == "rec" else (0, 200, 255)
                cv2.rectangle(bgr, (0, 0), (640, 44), (0, 0, 0), -1)
                cv2.putText(bgr, txt, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2, cv2.LINE_AA)
            if st["last"]:
                cv2.rectangle(bgr, (0, 332), (640, 360), (0, 0, 0), -1)
                cv2.putText(bgr, "last: " + st["last"], (10, 352), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (255, 255, 255), 1, cv2.LINE_AA)
            streamer.publish(bgr)
    res = st["res"]
    print(f"\n[test] DONE. correct {sum(r[3] for r in res)}/{len(res)}")
    for k in want + ["NOT"]:
        rr = [r for r in res if r[0] == k]
        print(f"   {k if k != 'NOT' else 'no sign':<10} {sum(r[3] for r in rr)}/{len(rr)} right   matched as: "
              + " ".join(r[1] for r in rr))
    print("   (distance of each trial's best match: " + " ".join(f"{r[2]:.2f}" for r in res) + ")", flush=True)
    os._exit(0)             # skip MediaPipe's noisy teardown


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("loo", "live"))
    ap.add_argument("--person", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--phrase", default="hello", help="live: phrase or comma list, e.g. hello,hii")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--trials", type=int, default=5, help="live: trials per phrase (and as many do-nothing trials)")
    ap.add_argument("--clip-secs", type=float, default=3.0)
    ap.add_argument("--rest-secs", type=float, default=3.0)
    ap.add_argument("--margin", type=float, default=4.0, help="live: threshold = leave-one-out median x margin")
    ap.add_argument("--stream-port", type=int, default=0)
    a = ap.parse_args()
    (cmd_loo if a.cmd == "loo" else cmd_live)(a)
