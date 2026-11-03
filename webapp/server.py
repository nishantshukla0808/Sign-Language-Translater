"""
SignTalk web server for the Raspberry Pi.

    .venv/bin/python webapp/server.py            # https://<pi-ip>:8443/  (and http :8080 -> redirect)

Phones open the page, the browser finds hand/pose landmarks on the phone, and only those numbers
(~40 KB per sign) come here. The Pi runs the trained phrase network and replies; the phone speaks.
Standard library HTTP + onnxruntime only, so it stays light on the Pi.

API:  GET /api/config  /api/ping  /api/retrain_status      POST /api/predict  /api/teach
      /api/teach_undo  /api/retrain  /api/log
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import gzip
import json
import mimetypes
import os
import re
import shutil
import socket
import ssl
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from fewshot import slug                                                        # noqa: E402
from phrase_model import PhraseModel, quality, window_from_payload              # noqa: E402
from mqtt_bus import Bus                                                        # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8443)
ap.add_argument("--http-port", type=int, default=8080, help="plain-http redirect to https (0 = off)")
ap.add_argument("--thr", type=float, default=0.55, help="accept a phrase only at this probability or higher")
ap.add_argument("--data", default="data")
ap.add_argument("--teach-person", default="phone", help="clips taught from phones are saved under data/<this>/")
ap.add_argument("--train-python", default="/home/vlmpi/venv-export/bin/python", help="python with PyTorch, for retraining")
ap.add_argument("--threads", type=int, default=2, help="onnxruntime threads")
ap.add_argument("--mqtt-host", default=os.environ.get("MQTT_HOST", "127.0.0.1"), help="MQTT broker (empty = MQTT off)")
ap.add_argument("--mqtt-port", type=int, default=int(os.environ.get("MQTT_PORT", "1883")))
ap.add_argument("--mqtt-user", default=os.environ.get("MQTT_USER", "signtalk"))
ap.add_argument("--mqtt-pass", default=os.environ.get("MQTT_PASS", "12345678"))
ap.add_argument("--no-tls", action="store_true", help="plain http only (the phone camera will NOT work, testing only)")
args = ap.parse_args()

STATIC = HERE / "static"
CACHE = HERE / ".gzcache"
DATA = Path(args.data)
MODELS = ROOT / "models"
MAX_BODY = 4 * 1024 * 1024
REC_MIN_HANDS, REC_MIN_BODY, MIN_FRAMES = 0.15, 0.5, 12
model = PhraseModel(MODELS, threads=args.threads)
display_over = json.loads((HERE / "phrases_display.json").read_text()) if (HERE / "phrases_display.json").exists() else {}
state_lock = threading.Lock()
last_saved: dict = {}


def pretty(s: str) -> dict:
    d = display_over.get(s, {})
    txt = s.replace("_", " ").strip().capitalize()
    return {"id": s, "label": d.get("label", txt), "speak": d.get("speak", d.get("label", txt))}


def now_str() -> str:
    return dt.datetime.now().strftime("%H:%M:%S")


# ---- data on disk ---------------------------------------------------------------------------
def people_dirs() -> list[str]:
    return sorted(d.name for d in DATA.iterdir() if d.is_dir() and any(d.glob("*/*.npy"))) if DATA.exists() else []


def clip_counts() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for person in people_dirs():
        for d in (DATA / person).iterdir():
            if d.is_dir():
                n = len(list(d.glob("*.npy")))
                if n:
                    e = out.setdefault(d.name, {"phone": 0, "total": 0})
                    e["total"] += n
                    if person == args.teach_person:
                        e["phone"] += n
    return out


def save_clip(phrase: str, win: np.ndarray) -> tuple[Path, int]:
    s = slug(phrase)
    if not s or len(s) > 40:
        raise ValueError("phrase name must be 1-40 letters/numbers")
    d = DATA / args.teach_person / s
    d.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now()
    f = d / f"{stamp:%Y%m%d_%H%M%S}.npy"
    k = 1
    while f.exists():
        f = d / f"{stamp:%Y%m%d_%H%M%S}_{k}.npy"
        k += 1
    np.save(f, win)
    hand, body = quality(win)
    meta = DATA / args.teach_person / "meta.csv"
    new = not meta.exists()
    with meta.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow("file,phrase,person,session,date,frames,fps,left_hand,right_hand,either_hand".split(","))
        lh = float(np.isfinite(win[:, 468, 0]).mean())
        rh = float(np.isfinite(win[:, 522, 0]).mean())
        w.writerow([f"{args.teach_person}/{s}/{f.name}", s, args.teach_person, f"{stamp:%Y%m%d_%H%M}",
                    stamp.isoformat(timespec="seconds"), len(win), round(len(win) / 3.0, 1),
                    round(lh, 2), round(rh, 2), round(hand, 2)])
    pj = DATA / "phrases.json"
    try:
        known = json.loads(pj.read_text()) if pj.exists() else []
        if s not in [slug(x) for x in known]:
            known.append(phrase.strip())
            pj.write_text(json.dumps(known, indent=1))
    except (OSError, ValueError):
        pass
    return f, len(list(d.glob("*.npy")))


def undo_clip() -> str | None:
    with state_lock:
        f = last_saved.pop("file", None)
    if not f or not Path(f).exists():
        return None
    Path(f).unlink()
    meta = DATA / args.teach_person / "meta.csv"
    if meta.exists():
        rel = f"{args.teach_person}/{Path(f).parent.name}/{Path(f).name}"
        rows = list(csv.reader(meta.open()))
        with meta.open("w", newline="") as fh:
            csv.writer(fh).writerows([r for r in rows if not r or r[0] != rel])
    return Path(f).parent.name


# ---- retraining (background, low priority) -----------------------------------------------------
class Retrain:
    def __init__(self):
        self.state, self.pct, self.log, self.started, self.msg = "idle", 0, collections.deque(maxlen=12), 0.0, ""
        self.lock = threading.Lock()

    def start(self) -> str | None:
        short = {k: v["total"] for k, v in clip_counts().items() if v["total"] < 5}
        if short:
            return "need at least 5 clips for: " + ", ".join(f"{k} ({n})" for k, n in short.items())
        if len(clip_counts()) < 2:
            return "need at least 2 phrases"
        if not Path(args.train_python).exists():
            return f"training python not found: {args.train_python}"
        with self.lock:
            if self.state == "running":
                return "training is already running"
            self.state, self.pct, self.log, self.started, self.msg = "running", 0, collections.deque(maxlen=12), time.time(), ""
        threading.Thread(target=self._run, daemon=True, name="retrain").start()
        return None

    def _run(self):
        tmp = MODELS / "_new"
        shutil.rmtree(tmp, ignore_errors=True)
        cmd = [args.train_python, "train_phrases.py", "--person", ",".join(people_dirs()), "--no-cv",
               "--epochs", "80", "--out", str(tmp)]
        try:
            p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                 preexec_fn=lambda: os.nice(10))
            for line in p.stdout:
                line = line.rstrip()
                self.log.append(line)
                m = re.search(r"epoch (\d+)/(\d+)", line)
                if m:
                    self.pct = int(100 * int(m.group(1)) / int(m.group(2)))
            rc = p.wait()
            if rc != 0 or not (tmp / "phrases.onnx").exists():
                raise RuntimeError(f"training failed (exit {rc}); last line: {self.log[-1] if self.log else ''}")
            bk = MODELS / "backup"
            bk.mkdir(exist_ok=True)
            tag = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            for n in ("phrases.onnx", "phrases_labels.json"):
                if (MODELS / n).exists():
                    shutil.copy2(MODELS / n, bk / f"{tag}_{n}")
            os.replace(tmp / "phrases_labels.json", MODELS / "phrases_labels.json")
            os.replace(tmp / "phrases.onnx", MODELS / "phrases.onnx")
            shutil.rmtree(tmp, ignore_errors=True)
            model.maybe_reload(force=True)
            self.pct, self.state = 100, "done"
            self.msg = f"new model live: {len(model.labels)} phrases"
        except Exception as e:                                       # noqa: BLE001
            self.state, self.msg = "error", str(e)
        print(f"[retrain] {self.state}: {self.msg}", flush=True)

    def status(self) -> dict:
        return {"state": self.state, "pct": self.pct, "msg": self.msg, "log": list(self.log)[-4:],
                "elapsed": int(time.time() - self.started) if self.started and self.state == "running" else 0}


retrain = Retrain()


# ---- predict ---------------------------------------------------------------------------------
def do_predict(body: dict) -> dict:
    t0 = time.perf_counter()
    win = window_from_payload(body.get("frames"), body.get("aspect", 16 / 9))
    hands, bodyf = quality(win)
    try:
        thr = min(0.95, max(0.2, float(body.get("thr", args.thr))))     # the phone may ask for a stricter/looser cut-off
    except (TypeError, ValueError):
        thr = args.thr
    base = {"frames": len(win), "hands_seen": round(hands, 2), "body_seen": round(bodyf, 2), "threshold": thr}
    if len(win) < MIN_FRAMES:
        return {**base, "accepted": False, "reason": "too_short"}
    if bodyf < REC_MIN_BODY:
        return {**base, "accepted": False, "reason": "no_body"}
    if hands < REC_MIN_HANDS:
        return {**base, "accepted": False, "reason": "no_hands"}
    probs, labels = model.predict(win)
    order = np.argsort(probs)[::-1][:3]
    top = [{**pretty(labels[i]), "p": round(float(probs[i]), 3)} for i in order]
    ok = float(probs[order[0]]) >= thr
    res = {**base, "accepted": ok, "reason": "" if ok else "unsure", "top": top, "best": top[0],
           "ms": round((time.perf_counter() - t0) * 1000, 1)}
    bus.prediction(res)
    print(f"[{now_str()}] predict: {top[0]['id']} {top[0]['p']:.2f} ({'ok' if ok else 'unsure'}) "
          f"| {top[1]['id']} {top[1]['p']:.2f} | frames {len(win)} hands {hands:.0%}  {res['ms']} ms", flush=True)
    return res


def pi_temp() -> float | None:
    try:
        return round(int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000, 1)
    except (OSError, ValueError):
        return None


bus = Bus(args.mqtt_host, args.mqtt_port, args.mqtt_user, args.mqtt_pass,
          telemetry=lambda: {"temp": pi_temp(), "load": round(os.getloadavg()[0], 2), "phrases": len(model.labels)})


# ---- static files (gzip, ETag, long cache for the big MediaPipe assets) ------------------------------
GZIP_EXT = {".html", ".js", ".mjs", ".css", ".json", ".svg", ".wasm"}
TYPES = {".mjs": "text/javascript", ".js": "text/javascript", ".wasm": "application/wasm",
         ".task": "application/octet-stream", ".svg": "image/svg+xml", ".webmanifest": "application/manifest+json"}
gz_lock = threading.Lock()


def gz_path(f: Path, rel: str) -> Path:
    g = CACHE / (rel.replace("/", "__") + ".gz")
    with gz_lock:
        if not g.exists() or g.stat().st_mtime < f.stat().st_mtime:
            CACHE.mkdir(exist_ok=True)
            with f.open("rb") as src, gzip.open(g, "wb", compresslevel=9 if f.suffix == ".wasm" else 6) as dst:
                shutil.copyfileobj(src, dst)
    return g


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = 30
    server_version = "SignTalk"

    def log_message(self, *a):
        pass

    # --- helpers
    def send_json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def body_json(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            raise ValueError("request too large")
        return json.loads(self.rfile.read(n) or b"{}")

    # --- routes
    def do_GET(self):
        u = urlsplit(self.path)
        if u.path == "/api/config":
            counts = clip_counts()
            ids = list(model.labels)
            self.send_json({
                "labels": [pretty(s) for s in ids], "threshold": args.thr, "teach_person": args.teach_person,
                "clips": counts, "new_phrases": [pretty(s) for s in counts if s not in ids],
                "model_loaded": int(model.loaded_at), "model_error": model.error})
        elif u.path == "/api/ping":
            la = os.getloadavg()[0]
            self.send_json({"ok": True, "temp": pi_temp(), "load": round(la, 2), "classes": len(model.labels),
                            "retrain": retrain.state})
        elif u.path == "/api/retrain_status":
            self.send_json(retrain.status())
        elif u.path.startswith("/api/"):
            self.send_json({"error": "not found"}, 404)
        else:
            self.serve_static(u.path)

    def do_POST(self):
        path = urlsplit(self.path).path
        try:
            if path == "/api/predict":
                self.send_json(do_predict(self.body_json()))
            elif path == "/api/teach":
                b = self.body_json()
                win = window_from_payload(b.get("frames"), b.get("aspect", 16 / 9))
                hands, bodyf = quality(win)
                if len(win) < MIN_FRAMES or bodyf < REC_MIN_BODY or hands < 0.3:
                    self.send_json({"saved": False, "error": "hands or shoulders were not visible enough - try again",
                                    "hands_seen": round(hands, 2), "body_seen": round(bodyf, 2)})
                    return
                f, n = save_clip(str(b.get("phrase", "")), win)
                with state_lock:
                    last_saved["file"] = str(f)
                print(f"[{now_str()}] teach: saved {f.parent.name} #{n} (hands {hands:.0%})", flush=True)
                self.send_json({"saved": True, "phrase": f.parent.name, "count": n, "hands_seen": round(hands, 2)})
            elif path == "/api/teach_undo":
                p = undo_clip()
                self.send_json({"undone": bool(p), "phrase": p})
            elif path == "/api/retrain":
                err = retrain.start()
                self.send_json({"started": err is None, "error": err}, 200 if err is None else 409)
            elif path == "/api/log":
                b = self.body_json()
                print(f"[{now_str()}] client {self.client_address[0]}: {str(b.get('msg'))[:500]}", flush=True)
                self.send_json({"ok": True})
            else:
                self.send_json({"error": "not found"}, 404)
        except (ValueError, json.JSONDecodeError) as e:
            self.send_json({"error": str(e)}, 400)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:                                                  # noqa: BLE001
            print(f"[server] error on {path}: {e!r}", flush=True)
            self.send_json({"error": "server error"}, 500)

    def serve_static(self, urlpath: str):
        rel = urlpath.lstrip("/") or "index.html"
        f = (STATIC / rel).resolve()
        if STATIC.resolve() not in f.parents or not f.is_file():
            self.send_json({"error": "not found"}, 404)
            return
        st = f.stat()
        etag = f'"{st.st_mtime_ns:x}-{st.st_size:x}"'
        big = rel.startswith("mp/")
        if self.headers.get("If-None-Match") == etag:
            self.send_response(304)
            self.send_header("ETag", etag)
            self.send_header("Cache-Control", "public, max-age=604800" if big else "no-cache")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        ext = f.suffix.lower()
        ctype = TYPES.get(ext) or mimetypes.guess_type(str(f))[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/json", "image/svg+xml"):
            ctype += "; charset=utf-8"
        src, size, enc = f, st.st_size, None
        if ext in GZIP_EXT and st.st_size > 1024 and "gzip" in self.headers.get("Accept-Encoding", ""):
            src = gz_path(f, rel)
            size, enc = src.stat().st_size, "gzip"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(size))
        self.send_header("ETag", etag)
        self.send_header("Cache-Control", "public, max-age=604800" if big else "no-cache")
        if enc:
            self.send_header("Content-Encoding", enc)
            self.send_header("Vary", "Accept-Encoding")
        self.end_headers()
        with src.open("rb") as fh:
            try:
                shutil.copyfileobj(fh, self.wfile, 64 * 1024)
            except (BrokenPipeError, ConnectionResetError, ssl.SSLError):
                pass


class TLSServer(ThreadingHTTPServer):
    """Does the TLS handshake in the worker thread so a half-open phone connection cannot block accept()."""
    daemon_threads = True
    request_queue_size = 32
    ctx: ssl.SSLContext | None = None

    def process_request_thread(self, request, client_address):
        try:
            request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            if self.ctx is not None:
                request.settimeout(10)
                request = self.ctx.wrap_socket(request, server_side=True)
        except (ssl.SSLError, OSError):
            self.shutdown_request(request)
            return
        super().process_request_thread(request, client_address)


def local_ips() -> list[str]:
    try:
        out = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=5).stdout.split()
    except (OSError, subprocess.SubprocessError):
        out = []
    return [i for i in out if re.fullmatch(r"\d+\.\d+\.\d+\.\d+", i)]


def ensure_cert() -> tuple[Path, Path]:
    d = HERE / "certs"
    d.mkdir(exist_ok=True)
    cert, key, ipf = d / "cert.pem", d / "key.pem", d / "ips.txt"
    have = set(ipf.read_text().split()) if ipf.exists() else set()
    want = have | set(local_ips()) | {"10.42.0.1", "127.0.0.1"}
    if cert.exists() and key.exists() and want == have:
        return cert, key
    san = "DNS:signtalk.local,DNS:localhost," + ",".join(f"IP:{i}" for i in sorted(want))
    subprocess.run(["openssl", "req", "-x509", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:prime256v1", "-nodes",
                    "-keyout", str(key), "-out", str(cert), "-days", "3650", "-subj", "/CN=SignTalk",
                    "-addext", f"subjectAltName={san}"], check=True, capture_output=True)
    ipf.write_text("\n".join(sorted(want)))
    print(f"[tls] new self-signed certificate for {', '.join(sorted(want))}", flush=True)
    return cert, key


class Redirect(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        host = (self.headers.get("Host") or "signtalk").split(":")[0]
        self.send_response(301)
        self.send_header("Location", f"https://{host}:{args.port}/")
        self.send_header("Content-Length", "0")
        self.end_headers()


def main():
    srv = TLSServer(("0.0.0.0", args.port), Handler)
    if not args.no_tls:
        cert, key = ensure_cert()
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        ctx.load_cert_chain(cert, key)
        srv.ctx = ctx
    if args.http_port and not args.no_tls:
        r = ThreadingHTTPServer(("0.0.0.0", args.http_port), Redirect)
        r.daemon_threads = True
        threading.Thread(target=r.serve_forever, daemon=True, name="redirect").start()
    scheme = "http" if args.no_tls else "https"
    print(f"[server] {len(model.labels)} phrases: {', '.join(model.labels)}   accept at p >= {args.thr}", flush=True)
    for ip in (local_ips() or ["<pi-ip>"]):
        print(f"[server] open on the phone:  {scheme}://{ip}:{args.port}/", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        bus.close()


if __name__ == "__main__":
    main()
