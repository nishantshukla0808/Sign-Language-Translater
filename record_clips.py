"""
Phrase-clip recorder with a browser UI (for a Pi with no display).

    .venv/bin/python record_clips.py --person nishu
    open http://<pi-ip>:8082/

Pick a phrase, press Record. You get a 3-2-1 countdown, then SIGN NOW for --clip-secs
seconds; the clip is saved as landmarks (T,543,3) .npy + a small .mp4 (so landmarks can
be re-extracted later if tracking settings change) under

    data/<person>/<phrase>/<session>_<n>.npy / .mp4      and   data/<person>/meta.csv

"Record x N" repeats clip after clip hands-free. Phrases come from data/phrases.json
(created on first run; edit it to add or remove phrases).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

import mpdetect
from landmarks import LHAND_OFF, RHAND_OFF

DEFAULT_PHRASES = [
    "hello", "thank you", "who are you", "help me", "what is your name", "my name is Nishu",
    "what is the time", "which place is this", "how are you", "I am fine", "yes", "no", "please",
    "sorry", "I need water", "I am hungry", "where is the toilet", "call a doctor",
    "I don't understand", "please repeat", "goodbye", "IDLE",
]


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


p = argparse.ArgumentParser()
p.add_argument("--person", required=True, help="who is signing, e.g. nishu")
p.add_argument("--data", default="data")
p.add_argument("--camera", type=int, default=0)
p.add_argument("--cap-width", type=int, default=1280)
p.add_argument("--cap-height", type=int, default=720)
p.add_argument("--width", type=int, default=640, help="processing width (frames are shrunk to this)")
p.add_argument("--height", type=int, default=360)
p.add_argument("--port", type=int, default=8082)
p.add_argument("--clip-secs", type=float, default=3.0)
p.add_argument("--countdown", type=int, default=3)
p.add_argument("--pause-secs", type=float, default=2.0, help="gap between clips in an auto-repeat run")
p.add_argument("--session", default=None, help="session label (default: date_time)")
args = p.parse_args()

DATA = Path(args.data)
PERSON_DIR = DATA / slug(args.person)
SESSION = args.session or datetime.now().strftime("%Y%m%d_%H%M")
PHR_FILE = DATA / "phrases.json"
DATA.mkdir(exist_ok=True)
if not PHR_FILE.exists():
    PHR_FILE.write_text(json.dumps(DEFAULT_PHRASES, indent=1))
PHRASES: list[str] = json.loads(PHR_FILE.read_text())
PERSON_DIR.mkdir(parents=True, exist_ok=True)
META = PERSON_DIR / "meta.csv"
if not META.exists():
    META.write_text("file,phrase,person,session,date,frames,fps,left_hand,right_hand,either_hand\n")


def counts() -> dict[str, int]:
    return {ph: len(list((PERSON_DIR / slug(ph)).glob("*.npy"))) for ph in PHRASES}


# ---- shared state ---------------------------------------------------------------------
S = {"mode": "idle", "phrase": PHRASES[0], "left": 0, "msg": "Pick a phrase and press Record.",
     "until": 0.0, "last": None, "hands": 0.0, "jpg": None, "fps": 0.0, "live_hands": 0.0}
CMD = {"start": None, "stop": False, "delete": False}
lock = threading.Lock()


def camera_thread(holder):
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cap_height)
    while True:
        ok, f = cap.read()
        if ok:
            holder["f"] = cv2.resize(f, (args.width, args.height), interpolation=cv2.INTER_AREA)
        else:
            time.sleep(0.05)


def draw(bgr, lm):
    from overlay import draw_overlay
    draw_overlay(bgr, lm, "idle", 0.0, 0.0, None, S["fps"])
    h, w = bgr.shape[:2]
    mode = S["mode"]
    if mode in ("countdown", "recording", "pause"):
        t = max(0.0, S["until"] - time.monotonic())
        if mode == "countdown":
            txt, col = f"GET READY: {S['phrase']}   {int(t) + 1}", (0, 200, 255)
        elif mode == "recording":
            txt, col = f"SIGN NOW: {S['phrase']}   {t:.0f}s", (60, 220, 60)
        else:
            txt, col = f"rest... next in {int(t) + 1}   ({S['left']} left)", (200, 200, 200)
        cv2.rectangle(bgr, (0, 0), (w, 46), (0, 0, 0), -1)
        cv2.putText(bgr, txt, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2, cv2.LINE_AA)
        if mode == "recording":
            cv2.circle(bgr, (w - 24, 24), 10, (0, 0, 255), -1)
    cv2.rectangle(bgr, (0, h - 28), (w, h), (0, 0, 0), -1)
    cv2.putText(bgr, f"hands now: {S['live_hands']:.0%}   fps {S['fps']:.0f}", (10, h - 9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def save_clip(phrase, frames, lms, ts):
    d = PERSON_DIR / slug(phrase)
    d.mkdir(parents=True, exist_ok=True)
    n = len(list(d.glob(f"{SESSION}_*.npy"))) + 1
    base = d / f"{SESSION}_{n:03d}"
    arr = np.stack(lms).astype(np.float32)
    np.save(base.with_suffix(".npy"), arr)
    dur = max(1e-3, ts[-1] - ts[0])
    fps = len(frames) / dur
    vw = cv2.VideoWriter(str(base.with_suffix(".mp4")), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                         (frames[0].shape[1], frames[0].shape[0]))
    for f in frames:
        vw.write(f)
    vw.release()
    lh = float(np.mean(np.isfinite(arr[:, LHAND_OFF, 0])))
    rh = float(np.mean(np.isfinite(arr[:, RHAND_OFF, 0])))
    eh = float(np.mean(np.isfinite(arr[:, LHAND_OFF, 0]) | np.isfinite(arr[:, RHAND_OFF, 0])))
    with META.open("a", newline="") as fh:
        csv.writer(fh).writerow([str(base.relative_to(DATA)) + ".npy", phrase, args.person, SESSION,
                                 datetime.now().isoformat(timespec="seconds"), len(frames),
                                 round(fps, 1), round(lh, 2), round(rh, 2), round(eh, 2)])
    return base, eh, len(frames), fps


def main_loop():
    holder: dict = {}
    threading.Thread(target=camera_thread, args=(holder,), daemon=True).start()
    det = mpdetect.build_detectors(Path("models"), False)
    tsf = mpdetect.MonotonicTs()
    t0 = time.monotonic()
    while "f" not in holder:
        time.sleep(0.05)
    buf_f, buf_l, buf_t = [], [], []
    run_left, last_files, n_frames = 0, [], 0
    tfps = time.monotonic()
    while True:
        bgr = holder["f"].copy()
        now = time.monotonic()
        lm = mpdetect.detect_frame(mpdetect.to_mp_image(bgr), tsf((now - t0) * 1000), det)
        n_frames += 1
        if now - tfps > 2:
            S["fps"], n_frames, tfps = n_frames / (now - tfps), 0, now
        S["live_hands"] = float(np.isfinite(lm[LHAND_OFF, 0]) or np.isfinite(lm[RHAND_OFF, 0]))

        with lock:
            cmd_start, CMD["start"] = CMD["start"], None
            stop, CMD["stop"] = CMD["stop"], False
            delete, CMD["delete"] = CMD["delete"], False
        if stop:
            S["mode"], run_left, S["msg"] = "idle", 0, "Stopped."
        if delete and last_files:
            for ext in (".npy", ".mp4"):
                last_files[-1].with_suffix(ext).unlink(missing_ok=True)
            S["msg"] = f"Deleted {last_files.pop().name} (meta.csv row kept; clip files removed)."
        if cmd_start and S["mode"] == "idle":
            S["phrase"], run_left = cmd_start[0], cmd_start[1]
            S["mode"], S["until"] = "countdown", now + args.countdown
            S["left"] = run_left

        if S["mode"] == "countdown" and now >= S["until"]:
            S["mode"], S["until"] = "recording", now + args.clip_secs
            buf_f, buf_l, buf_t = [], [], []
        elif S["mode"] == "recording":
            buf_f.append(bgr.copy())
            buf_l.append(lm)
            buf_t.append(now)
            if now >= S["until"]:
                base, eh, nf, fps = save_clip(S["phrase"], buf_f, buf_l, buf_t)
                last_files.append(base)
                S["hands"], S["last"] = eh, base.name
                run_left -= 1
                S["left"] = run_left
                warn = "   LOW hand detection - consider Delete last and redo" if eh < 0.6 else ""
                S["msg"] = (f"Saved {S['phrase']} / {base.name}: {nf} frames @ {fps:.0f} fps, "
                            f"hands seen {eh:.0%}{warn}")
                if run_left > 0:
                    S["mode"], S["until"] = "pause", now + args.pause_secs
                else:
                    S["mode"] = "idle"
        elif S["mode"] == "pause" and now >= S["until"]:
            S["mode"], S["until"] = "countdown", now + 1.0   # short re-countdown inside a run

        draw(bgr, lm)
        S["jpg"] = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()


PAGE = """<!doctype html><meta name=viewport content='width=device-width,initial-scale=1'>
<title>Clip recorder</title>
<style>body{font-family:sans-serif;background:#111;color:#eee;margin:0;padding:10px}
img{width:100%;max-width:900px;display:block}button,select{font-size:18px;padding:10px 14px;margin:4px;border-radius:8px;border:0}
.go{background:#2e8b57;color:#fff}.stop{background:#b22;color:#fff}.o{background:#445;color:#fff}
#msg{margin:8px 0;min-height:24px;color:#9fd}table{border-collapse:collapse}td{padding:2px 10px}
.ok{color:#7e7}.mid{color:#fc6}.lo{color:#f88}</style>
<img src=/stream>
<div id=msg></div>
<div><select id=ph></select>
<button class=go onclick="rec(1)">Record 1</button>
<button class=go onclick="rec(10)">Record x 10</button>
<button class=stop onclick="fetch('/stop',{method:'POST'})">Stop</button>
<button class=o onclick="fetch('/delete',{method:'POST'})">Delete last</button></div>
<div><input id=newph placeholder="new phrase, e.g. good morning" style="font-size:18px;padding:10px;border-radius:8px;border:0;width:260px">
<button class=o onclick="addph()">Add phrase</button></div>
<h3>Clips for <span id=who></span> (target 30-60 per phrase, session <span id=ses></span>)</h3>
<table id=tb></table>
<script>
let phrases=[];
async function addph(){const i=document.getElementById('newph');const v=i.value.trim();if(!v)return;
 await fetch('/addphrase?name='+encodeURIComponent(v),{method:'POST'});i.value='';phrases=[];}
function rec(n){fetch('/start?phrase='+encodeURIComponent(document.getElementById('ph').value)+'&n='+n,{method:'POST'})}
async function tick(){
 const s=await (await fetch('/state')).json();
 if(!phrases.length){phrases=s.phrases;const sel=document.getElementById('ph');
  const keep=sel.value;sel.innerHTML='';
  phrases.forEach(p=>{const o=document.createElement('option');o.textContent=p;sel.appendChild(o)});
  if(keep&&phrases.includes(keep))sel.value=keep;else if(keep)sel.value=phrases[phrases.length-1];
  document.getElementById('who').textContent=s.person;document.getElementById('ses').textContent=s.session}
 document.getElementById('msg').textContent=s.msg;
 document.getElementById('tb').innerHTML=phrases.map(p=>{const c=s.counts[p];
  return `<tr><td>${p}</td><td class=${c>=30?'ok':c>=10?'mid':'lo'}>${c}</td></tr>`}).join('');
}
setInterval(tick,700);tick();
</script>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
            self.end_headers()
            try:
                while True:
                    j = S["jpg"]
                    if j:
                        self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(j))
                        self.wfile.write(j + b"\r\n")
                    time.sleep(0.08)
            except (BrokenPipeError, ConnectionResetError):
                return
        elif u.path == "/state":
            self._json({"phrases": PHRASES, "counts": counts(), "msg": S["msg"], "person": args.person,
                        "session": SESSION, "mode": S["mode"]})
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(PAGE.encode())

    def do_POST(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        with lock:
            if u.path == "/start":
                ph = q.get("phrase", [PHRASES[0]])[0]
                if ph in PHRASES:
                    CMD["start"] = (ph, int(q.get("n", ["1"])[0]))
            elif u.path == "/addphrase":
                name = " ".join(q.get("name", [""])[0].split())[:60]
                if name and slug(name) and slug(name) not in {slug(x) for x in PHRASES}:
                    PHRASES.append(name)
                    PHR_FILE.write_text(json.dumps(PHRASES, indent=1, ensure_ascii=False))
                    S["msg"] = f"Added phrase: {name}"
                else:
                    S["msg"] = f"Not added (empty or already in the list): {name!r}"
            elif u.path == "/stop":
                CMD["stop"] = True
            elif u.path == "/delete":
                CMD["delete"] = True
        self._json({"ok": True})


if __name__ == "__main__":
    subprocess.run(["v4l2-ctl", "-d", f"/dev/video{args.camera}", "--set-ctrl=exposure_dynamic_framerate=0"],
                   capture_output=True)
    threading.Thread(target=main_loop, daemon=True).start()
    print(f"recorder for '{args.person}', session {SESSION}  ->  http://<pi-ip>:{args.port}/", flush=True)
    ThreadingHTTPServer(("0.0.0.0", args.port), H).serve_forever()
