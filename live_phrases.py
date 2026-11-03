"""
Live test of the TRAINED phrase network (models/phrases.onnx), with the DTW matcher run on the
same window for comparison. Guided: random "sign X" / "do NOT sign" trials, 3 s each.

    .venv/bin/python live_phrases.py --person nishu --stream-port 8083 [--trials 3] [--thr 0.80]

Open http://<pi-ip>:8083/ and press START TEST. Stop the recorder first (one camera user only).
A prediction is accepted only if its probability >= --thr; otherwise the answer is "none".
"""

from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

import fewshot
import mpdetect
from fewshot import T_LEN, features, score
from landmarks import LHAND_OFF, RHAND_OFF, hands_present

ap = argparse.ArgumentParser()
ap.add_argument("--person", required=True)
ap.add_argument("--data", default="data")
ap.add_argument("--camera", type=int, default=0)
ap.add_argument("--trials", type=int, default=3, help="trials per phrase (do-nothing trials: 2x this)")
ap.add_argument("--phrase", default="", help="comma list; default = all phrases the model knows")
ap.add_argument("--thr", type=float, default=0.80, help="accept only if network probability >= this")
ap.add_argument("--clip-secs", type=float, default=3.0)
ap.add_argument("--rest-secs", type=float, default=3.0)
ap.add_argument("--stream-port", type=int, default=0)
a = ap.parse_args()

labels = json.loads(Path("models/phrases_labels.json").read_text())
sess = ort.InferenceSession("models/phrases.onnx", providers=["CPUExecutionProvider"])
want = [fewshot.slug(x) for x in a.phrase.split(",") if x.strip()] or labels
tpl = fewshot.load_templates(Path(a.data), a.person)       # for the DTW comparison
print(f"[net] phrases: {', '.join(labels)}   accept if prob >= {a.thr}", flush=True)


def net_predict(win: np.ndarray) -> np.ndarray:
    """Same input as train_phrases.build_input: positions, velocities x4, hand flags."""
    base = features(win).reshape(T_LEN, 98).astype(np.float32)
    idx = np.round(np.linspace(0, len(win) - 1, T_LEN)).astype(int)
    m = np.stack([np.isfinite(win[idx, LHAND_OFF, 0]), np.isfinite(win[idx, RHAND_OFF, 0])], 1).astype(np.float32)
    vel = np.concatenate([np.zeros((1, 98), np.float32), base[1:] - base[:-1]], 0) * 4.0
    x = np.concatenate([base, vel, m], -1)[None]
    z = sess.run(None, {"x": x})[0][0]
    e = np.exp(z - z.max())
    return e / e.sum()


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
from overlay import draw_overlay

# DTW accept thresholds (median leave-one-out x 4), as in fewshot.py
thr_dtw = {ph: float(np.median([score(f, tpl, skip=n)[ph] for n, f in tpl[ph]]) * 4.0) for ph in want if ph in tpl}

trials = [(p, p) for p in want for _ in range(a.trials)] + [("NOT", "") for _ in range(a.trials * 2)]
random.Random(2).shuffle(trials)
label = lambda k: "do NOT sign (rest or any movement)" if k == "NOT" else f"sign: {k}"


def banner(bgr, txt, col):
    cv2.rectangle(bgr, (0, 0), (640, 44), (0, 0, 0), -1)
    cv2.putText(bgr, txt, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2, cv2.LINE_AA)


if streamer:
    print("[test] ready. Open the page and press START TEST.", flush=True)
    while not streamer.started.is_set():
        bgr = holder["f"].copy()
        lm = mpdetect.detect_frame(mpdetect.to_mp_image(bgr), tsf((time.monotonic() - t0) * 1000), det)
        draw_overlay(bgr, lm, "idle", 0.0, 0.0, None, 0.0)
        banner(bgr, "press START TEST above", (0, 200, 255))
        streamer.publish(bgr)
st = {"i": 0, "phase": "rest", "until": time.monotonic() + 5.0, "buf": [], "res": [], "last": ""}
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
            pr = net_predict(win)
            top = int(pr.argmax())
            said = labels[top] if pr[top] >= a.thr else "none"
            sc = score(features(win), tpl)
            near = min(sc, key=sc.get)
            dtw_said = near if near in thr_dtw and sc[near] < thr_dtw[near] else "none"
            ok = (said == kind) if kind != "NOT" else (said == "none")
            ok_d = (dtw_said == kind) if kind != "NOT" else (dtw_said == "none")
            st["res"].append((kind, said, float(pr[top]), ok, dtw_said, ok_d))
            two = np.argsort(pr)[::-1][:2]
            print(f"   you did: {kind if kind != 'NOT' else 'nothing'}  ->  network: {said} "
                  f"({labels[top]} {pr[top]:.2f}, then {labels[two[1]]} {pr[two[1]]:.2f}) [{'correct' if ok else 'WRONG'}]"
                  f"   | DTW: {dtw_said} [{'correct' if ok_d else 'WRONG'}]   hands seen {hands:.0%}", flush=True)
            st["last"] = f"{'sign ' + kind if kind != 'NOT' else 'no sign'} -> {said} ({'right' if ok else 'WRONG'})"
            st["i"] += 1
            st.update(phase="rest", until=now + a.rest_secs)
            if st["i"] < len(trials):
                print(f"[test] next: {label(trials[st['i']][0])}", flush=True)
    if streamer:
        draw_overlay(bgr, lm, "idle", 0.0, 0.0, None, 0.0)
        if st["i"] < len(trials):
            k = trials[st["i"]][0]
            left = max(0.0, st["until"] - now)
            what = "DO NOT SIGN" if k == "NOT" else k.upper()
            if st["phase"] == "rec":
                banner(bgr, f"NOW: {what}  ({left:.0f}s)", (60, 220, 60))
            else:
                banner(bgr, f"get ready: {what}  ({left:.0f}s)", (0, 200, 255))
        if st["last"]:
            cv2.rectangle(bgr, (0, 332), (640, 360), (0, 0, 0), -1)
            cv2.putText(bgr, "last: " + st["last"], (10, 352), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)
        streamer.publish(bgr)

res = st["res"]
n = len(res)
print(f"\n[test] DONE.  network correct {sum(r[3] for r in res)}/{n}   |   DTW correct {sum(r[5] for r in res)}/{n}")
for k in want + ["NOT"]:
    rr = [r for r in res if r[0] == k]
    print(f"   {k if k != 'NOT' else 'no sign':<16} network {sum(r[3] for r in rr)}/{len(rr)}  said: "
          + " ".join(r[1] for r in rr) + f"   | DTW {sum(r[5] for r in rr)}/{len(rr)}")
sg = [r for r in res if r[0] != "NOT"]
wrong_say = sum(1 for r in sg if r[1] not in (r[0], "none"))
missed = sum(1 for r in sg if r[1] == "none")
false_pos = sum(1 for r in res if r[0] == "NOT" and r[1] != "none")
print(f"   of {len(sg)} real signs: {wrong_say} spoken as the WRONG phrase, {missed} not recognised (below {a.thr});"
      f"  {false_pos} of {n - len(sg)} non-sign trials wrongly spoken as a phrase", flush=True)
os._exit(0)
