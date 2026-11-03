"""
Checks the running web server with real recorded clips (no browser, no camera):

    .venv/bin/python webapp/test_server.py [--url https://127.0.0.1:8443] [--person nishu] [--per-phrase 3]

Each stored clip is sent exactly like the phone page sends it (7 pose + 2x21 hand points, x/y only,
rounded to 4 decimals) and must come back as its own phrase, and with the same probabilities as the
direct model call. Also checks static files, gzip, caching and bad input.
"""
import argparse
import glob
import json
import ssl
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fewshot import POSE_IDX                                  # noqa: E402
from landmarks import LHAND_OFF, POSE_OFF, RHAND_OFF         # noqa: E402
from phrase_model import PhraseModel                          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="https://127.0.0.1:8443")
ap.add_argument("--person", default="nishu")
ap.add_argument("--per-phrase", type=int, default=3)
a = ap.parse_args()
ctx = ssl._create_unverified_context()
fails = []


def req(path, body=None, headers=None, method=None):
    r = urllib.request.Request(a.url + path, data=None if body is None else json.dumps(body).encode(),
                               headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(r, context=ctx, timeout=30) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def check(name, ok, extra=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {extra}")
    if not ok:
        fails.append(name)


def to_payload(win):
    frames = []
    for f in win:
        def part(rows):
            xy = f[rows, :2]
            return None if not np.isfinite(xy).all() else [round(float(v), 4) for v in xy.reshape(-1)]
        frames.append([part([POSE_OFF + i for i in POSE_IDX]), part(slice(LHAND_OFF, LHAND_OFF + 21)),
                       part(slice(RHAND_OFF, RHAND_OFF + 21))])
    return frames


print("static + api")
s, h, b = req("/")
check("index served", s == 200 and b"SignTalk" in b)
s, h, b = req("/app.js", headers={"Accept-Encoding": "gzip"})
check("app.js gzip", s == 200 and h.get("Content-Encoding") == "gzip")
etag = h.get("ETag")
s, h, b = req("/app.js", headers={"If-None-Match": etag})
check("ETag 304", s == 304)
s, h, b = req("/mp/vision_bundle.mjs")
check("mediapipe bundle + js mime", s == 200 and "javascript" in h.get("Content-Type", ""))
s, h, b = req("/mp/wasm/vision_wasm_internal.wasm", headers={"Accept-Encoding": "gzip"})
check("wasm served as application/wasm", s == 200 and h.get("Content-Type", "").startswith("application/wasm"),
      f"({len(b) / 1e6:.1f} MB on the wire, {h.get('Content-Encoding')})")
s, h, b = req("/../server.py")
check("path traversal blocked", s == 404)
s, h, b = req("/%2e%2e/server.py")
check("encoded traversal blocked", s == 404)
cfg = json.loads(req("/api/config")[2])
check("config lists phrases", len(cfg["labels"]) >= 2, f"({', '.join(l['id'] for l in cfg['labels'])})")
check("ping", json.loads(req("/api/ping")[2])["ok"])
s, _, b = req("/api/predict", {"frames": "x", "aspect": 1.7})
check("bad payload -> 400", s == 400, b.decode()[:60])
s, _, b = req("/api/predict", {"frames": [[None, None, None]] * 40, "aspect": 1.78})
j = json.loads(b)
check("empty frames -> no_body", j.get("reason") == "no_body", str(j.get("reason")))

print("real clips through the server")
model = PhraseModel()
ok = tot = 0
worst = 0.0
for ph in model.labels:
    for f in sorted(glob.glob(f"data/{a.person}/{ph}/*.npy"))[:a.per_phrase]:
        win = np.load(f)
        s, _, b = req("/api/predict", {"aspect": 16 / 9, "frames": to_payload(win)})
        j = json.loads(b)
        probs, _ = model.predict(win)
        tot += 1
        good = j.get("best", {}).get("id") == ph
        ok += good
        d = abs(j["best"]["p"] - float(probs.max())) if "best" in j else 1
        worst = max(worst, d)
        print(f"    {ph:<15} -> {j.get('best', {}).get('id')} {j.get('best', {}).get('p')} {'accepted' if j.get('accepted') else j.get('reason')} {j.get('ms')} ms")
check(f"clips recognised ({ok}/{tot}; training clips, wiring check)", ok == tot)
check("server probabilities match the direct model call (rounding only)", worst < 0.02, f"(max diff {worst:.4f})")

print("aspect handling")
win = np.load(sorted(glob.glob(f"data/{a.person}/hello/*.npy"))[0])
fr = to_payload(win)
fr_p = [[None if p is None else [v / 1.0 if i % 2 else v / (9 / 16 * 16 / 9 * (1.0)) for i, v in enumerate(p)] for p in f] for f in fr]
j = json.loads(req("/api/predict", {"aspect": 16 / 9, "frames": fr})[2])
check("16:9 phone image = training geometry", j.get("best", {}).get("id") == "hello")

print("\nFAILED: " + ", ".join(fails) if fails else "\nALL PASSED")
sys.exit(1 if fails else 0)
