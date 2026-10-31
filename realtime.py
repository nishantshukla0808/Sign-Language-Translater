"""
Live loop: camera -> MediaPipe Tasks -> motion gate -> model -> phrase buffer -> speech.

Three threads, bounded queues, capture never blocks:

    capture --frame_q--> inference --audio_q--> audio

Laptop:
    python realtime.py --show
    python realtime.py --model aslcitizen --show

First time on a camera (writes calibration.json, auto-loaded afterwards):
    python realtime.py --calibrate

Raspberry Pi 5:
    python realtime.py --width 640 --height 480 --fps 15 --threads 3 --no-face

Needs the three MediaPipe .task bundles in models/ (see README). Face is optional;
--no-face is a large speedup and loses mouth-shape cues.

CLI change vs v1: --model is now the model FAMILY {popsign,aslcitizen};
the file path moved to --model-path.
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path
from queue import Empty, Queue

import cv2
import numpy as np

import mpdetect
from landmarks import hands_present
from calibrate import DEFAULT_PATH as CALIB_PATH, load_calibration, run_calibration
from classifier import TokenAdmitter
from engine import DEFAULT_REJECT, Engine
from gate import MotionGate
from models import CHOICES, load_model
from phrase import PhraseBuffer
from pipeline import DropStats, Speaker, put_drop_oldest
from session_log import SessionLogger

# Back-compat re-exports (v1 imported these from realtime).
build_detectors = mpdetect.build_detectors
detect_frame = mpdetect.detect_frame


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=CHOICES, default="popsign")
    p.add_argument("--model-path", default=None, help="override the .onnx path")
    p.add_argument("--labels", default=None,
                   help="popsign: sign->index JSON; aslcitizen: {index: gloss} JSON")
    p.add_argument("--aslcitizen-norm", choices=("train", "valid"), default="train",
                   help="shoulder-normalisation statistics: 'train' mirrors the training "
                        "code (incl. zero padding); 'valid' = real frames only (experiment)")
    p.add_argument("--tasks", default="models", help="dir holding *.task bundles")
    p.add_argument("--wavs", default="wavs", help="pre-rendered WAVs, one per word")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--video", default=None, help="read this file instead of a camera (testing)")
    p.add_argument("--url", default=None,
                   help="network camera stream (e.g. http://PHONE_IP:8080/video from IP Webcam, or rtsp://...); treated as a live camera; omit it to use the USB camera (--camera)")
    p.add_argument("--drill", default=None, metavar="WORDS",
                   help="guided test WITHOUT the motion gate, e.g. HELLO,YES,NO,YOU: prompts you word by word, "
                        "records a fixed window after each prompt, classifies it and prints a score summary")
    p.add_argument("--rounds", type=int, default=3, help="--drill: repeats of the whole word list")
    p.add_argument("--sign-secs", type=float, default=3.0, help="--drill: length of the recorded window")
    p.add_argument("--rest-secs", type=float, default=3.0, help="--drill: pause before each prompt")
    p.add_argument("--stream-port", type=int, default=0,
                   help="serve the annotated camera feed (landmarks + prompts) at http://<pi-ip>:PORT/ (0 = off)")
    p.add_argument("--vocab", default=None, metavar="WORDS",
                   help="limit the model to these words, e.g. HELLO,YES,NO,YOU,HOW (HOW also allows HOW1/HOW2); "
                        "'drill' = the --drill words. Scores are renormalised over this set")
    p.add_argument("--rotate", type=int, choices=(0, 90, 180, 270), default=0,
                   help="rotate --url frames clockwise by this many degrees (phone mounted sideways/upside down)")
    p.add_argument("--cap-width", type=int, default=0,
                   help="camera capture width (0 = same as --width); use 1280 with --cap-height 720 for a wider "
                        "view on webcams whose 640x480 mode is cropped; frames are shrunk to --width/--height")
    p.add_argument("--cap-height", type=int, default=0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--no-face", action="store_true", help="skip FaceLandmarker (faster)")
    p.add_argument("--no-audio", action="store_true")
    p.add_argument("--show", action="store_true", help="preview window + debug overlay")
    # phrase / speech
    p.add_argument("--speak-immediately", action="store_true",
                   help="speak each word as it is committed (words stack up) instead of "
                        "buffering a phrase")
    p.add_argument("--phrase-gap", type=float, default=1.5,
                   help="seconds of stillness before the buffered phrase is spoken")
    # admission
    p.add_argument("--top1-threshold", type=float, default=0.30,
                   help="confident-commit probability (and tick-mode 'stable')")
    p.add_argument("--top2-threshold", type=float, default=0.10,
                   help="band admission floor for top-1 and the runner-up")
    p.add_argument("--player", default="aplay", help="command used to play a WAV (-q <file> appended)")
    p.add_argument("--reject-threshold", type=float, default=DEFAULT_REJECT,
                   help="top-1 probability floor; below it nothing is emitted")
    # gate / calibration
    p.add_argument("--calibrate", action="store_true",
                   help="sample 5 s of stillness, write calibration.json, exit")
    p.add_argument("--config", default=CALIB_PATH, help="calibration file (auto-loaded if present)")
    p.add_argument("--start-velocity", type=float, default=None)
    p.add_argument("--stop-velocity", type=float, default=None)
    # logging
    p.add_argument("--session-dir", default="sessions")
    p.add_argument("--no-log", action="store_true")
    return p.parse_args(argv)


# --- threads ----------------------------------------------------------------

class Capture(threading.Thread):
    """Reads frames; never blocks on downstream. Camera: drop-oldest. File: back-pressure."""

    def __init__(self, args, frame_q: Queue, stats: DropStats, stop: threading.Event):
        super().__init__(daemon=True, name="capture")
        self.args, self.q, self.stats, self.stop = args, frame_q, stats, stop
        self.is_file = args.video is not None
        self.cap = cv2.VideoCapture(args.video if self.is_file else (args.url or args.camera))
        if not self.is_file and not args.url:
            if args.cap_width:
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cap_width or args.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cap_height or args.height)
            self.cap.set(cv2.CAP_PROP_FPS, args.fps)
        self.opened = self.cap.isOpened()
        self.src_fps = self.cap.get(cv2.CAP_PROP_FPS) or float(args.fps)
        self.finished = threading.Event()

    def run(self):
        args = self.args
        t0, i = time.monotonic(), 0
        while not self.stop.is_set():
            ok, bgr = self.cap.read()
            if not ok:
                break
            if args.url and args.rotate:
                bgr = cv2.rotate(bgr, {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                                       270: cv2.ROTATE_90_COUNTERCLOCKWISE}[args.rotate])
            if not self.is_file:               # shrink big phone streams, keeping aspect (portrait stays portrait)
                h, w = bgr.shape[:2]
                sc = min(args.width / w, args.height / h, 1.0)
                if sc < 1.0:
                    bgr = cv2.resize(bgr, (round(w * sc), round(h * sc)), interpolation=cv2.INTER_AREA)
            ts_ms = (i / self.src_fps * 1000.0) if self.is_file else (time.monotonic() - t0) * 1000.0
            i += 1
            if self.is_file:                      # offline: don't drop, wait for inference
                while not self.stop.is_set():
                    try:
                        self.q.put((bgr, ts_ms), timeout=0.1)
                        break
                    except Exception:
                        pass
            else:
                put_drop_oldest(self.q, (bgr, ts_ms), self.stats)
        self.cap.release()
        self.finished.set()


class Inference(threading.Thread):
    def __init__(self, args, frame_q, show_q, engine, det, phrase, speaker, logger, stop, cap):
        super().__init__(daemon=True, name="inference")
        self.args, self.frame_q, self.show_q = args, frame_q, show_q
        self.streamer = None
        if args.stream_port:
            from streamserver import StreamServer
            self.streamer = StreamServer(args.stream_port)
            print(f"[stream] watch the feed at http://<pi-ip>:{args.stream_port}/", flush=True)
        self.engine, self.det, self.phrase = engine, det, phrase
        self.speaker, self.logger, self.stop, self.cap = speaker, logger, stop, cap
        self.ts = mpdetect.MonotonicTs()
        self.frames = 0
        self.t_start = time.monotonic()
        self.last_top3 = None
        self.infer_ms: list[float] = []
        self.track_ms: list[float] = []
        self.transcript: list[str] = []

    def _emit_words(self, words):
        if not words:
            return
        print(f"   >> SAY: {' '.join(words)}")
        self.transcript.extend(words)
        if self.speaker:
            self.speaker.say_phrase(words)

    def run(self):
        a, eng = self.args, self.engine
        last_report = time.monotonic()
        while not self.stop.is_set():
            try:
                bgr, ts_ms = self.frame_q.get(timeout=0.1)
            except Empty:
                if self.cap.finished.is_set() and self.frame_q.empty():
                    break
                self._poll_phrase()
                continue
            t0 = time.perf_counter()
            frame = mpdetect.detect_frame(mpdetect.to_mp_image(bgr), self.ts(ts_ms), self.det)
            self.track_ms.append((time.perf_counter() - t0) * 1000)
            self.frames += 1
            seg = self._drill_step(frame) if a.drill else eng.push(frame)
            if seg is not None:
                self.infer_ms.append(seg.infer_ms)
                self.last_top3 = seg.top[:3]
                tok = seg.token
                head = (f"[{seg.window.shape[0]:3d}f] "
                        + "  ".join(f"{l}:{s:.2f}" for l, s in seg.top[:3]))
                if tok:
                    print(f"{head}   -> {tok.label.upper()} ({tok.via})  [{seg.infer_ms:.0f} ms]")
                    if a.speak_immediately:
                        self._emit_words([tok.label])
                    else:
                        self.phrase.add(tok.label)
                else:
                    print(f"{head}   ({'rejected < floor' if seg.rejected else 'held'})")
                if self.logger:
                    self.logger.log_segment(
                        seg.window, seg.top, seg.velocity_trace,
                        emitted=tok.label if tok else None, via=tok.via if tok else None,
                        extra={"model": a.model, "infer_ms": round(seg.infer_ms, 2),
                               "rejected": seg.rejected})
            self._poll_phrase()

            if self.streamer:
                self._draw_stream(bgr, frame, self.frames / max(1e-6, time.monotonic() - self.t_start))
            if a.show:
                from overlay import draw_overlay
                fps = self.frames / max(1e-6, time.monotonic() - self.t_start)
                draw_overlay(bgr, frame, eng.gate.state, eng.gate.last_velocity,
                             eng.gate.start_velocity, self.last_top3, fps)
                put_drop_oldest(self.show_q, bgr)

            now = time.monotonic()
            if now - last_report >= 5.0:
                print(f"       ... {self.frames / (now - self.t_start):.1f} fps  "
                      f"(capture dropped {self.cap_stats.dropped})")
                last_report = now
        self._poll_phrase(force=True)
        self.stop.set()

    def _drill_step(self, frame):
        """Guided test: fixed recording windows instead of the motion gate. Prints its own results."""
        d = getattr(self, "_drill", None)
        now = time.monotonic()
        if d is None:
            words = [w.strip().upper() for w in self.args.drill.split(",") if w.strip()]
            d = self._drill = {"trials": words * self.args.rounds, "i": 0, "phase": "rest",
                               "until": now + 5.0, "buf": [], "res": []}
            print(f"\n[drill] {len(d['trials'])} trials. Rest your hands; first prompt in 5 s.", flush=True)
            print(f"[drill] trial 1/{len(d['trials'])}: get ready to sign  {d['trials'][0]}", flush=True)
        if d["phase"] == "rest" and now >= d["until"]:
            d["phase"], d["until"], d["buf"] = "sign", now + self.args.sign_secs, []
            print(f"\n   >>> [{d['i'] + 1}/{len(d['trials'])}] SIGN NOW:  {d['trials'][d['i']]}", flush=True)
        elif d["phase"] == "sign":
            d["buf"].append(frame)
            if now >= d["until"]:
                want = d["trials"][d["i"]]
                win = np.stack(d["buf"]).astype(np.float32)
                hands = float(np.mean([hands_present(f) > 0 for f in d["buf"]]))
                seg = self.engine.classify(win)
                labs = [l for l, _ in seg.top]
                ok = lambda l: l == want or (l.startswith(want) and l[len(want):].isdigit())
                r1, r3 = ok(labs[0]), any(ok(l) for l in labs[:3])
                d["res"].append((want, labs[0], r1, r3, hands))
                top = "  ".join(f"{l}:{p:.2f}" for l, p in seg.top[:3])
                verdict = "OK" if r1 else ("in top-3" if r3 else "MISS")
                d["last"] = f"{want}: model said {labs[0]} ({seg.top[0][1]:.2f}) -> {verdict}"
                print(f"   you signed {want}  ->  model said {labs[0]} ({seg.top[0][1]:.2f})   {verdict}\n"
                      f"      top-3: {top}   [{win.shape[0]} frames, hands seen {hands:.0%}]", flush=True)
                d["i"] += 1
                if d["i"] >= len(d["trials"]):
                    res = d["res"]
                    n = len(res)
                    print(f"\n[drill] DONE: top-1 {sum(r[2] for r in res)}/{n}   top-3 {sum(r[3] for r in res)}/{n}   "
                          f"mean hands seen {np.mean([r[4] for r in res]):.0%}", flush=True)
                    for w in dict.fromkeys(d["trials"]):
                        rr = [r for r in res if r[0] == w]
                        print(f"   {w:<10} top-1 {sum(r[2] for r in rr)}/{len(rr)}  said: {' '.join(r[1] for r in rr)}")
                    self.stop.set()
                else:
                    d["phase"], d["until"] = "rest", now + self.args.rest_secs
                    print(f"[drill] trial {d['i'] + 1}/{len(d['trials'])}: get ready to sign  {d['trials'][d['i']]}", flush=True)
        return None

    def _draw_stream(self, bgr, frame, fps):
        from overlay import draw_overlay
        eng = self.engine
        draw_overlay(bgr, frame, eng.gate.state, eng.gate.last_velocity,
                     eng.gate.start_velocity, self.last_top3, fps)
        d = getattr(self, "_drill", None)
        if d and d["i"] < len(d["trials"]):
            left = max(0.0, d["until"] - time.monotonic())
            word = d["trials"][d["i"]]
            sign = d["phase"] == "sign"
            txt = f"SIGN NOW: {word}  ({left:.0f}s)" if sign else f"get ready: {word}  ({left:.0f}s)"
            col = (60, 220, 60) if sign else (0, 200, 255)
            cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 44), (0, 0, 0), -1)
            cv2.putText(bgr, txt, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2, cv2.LINE_AA)
            cv2.putText(bgr, f"trial {d['i'] + 1}/{len(d['trials'])}", (bgr.shape[1] - 150, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            if d.get("last"):
                cv2.rectangle(bgr, (0, bgr.shape[0] - 30), (bgr.shape[1], bgr.shape[0]), (0, 0, 0), -1)
                cv2.putText(bgr, "last: " + d["last"], (10, bgr.shape[0] - 9),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        self.streamer.publish(bgr)

    def _poll_phrase(self, force=False):
        if self.args.speak_immediately:
            return
        busy = self.engine.gate.state == "recording"
        words = self.phrase.flush() if force else self.phrase.poll(busy=busy)
        self._emit_words(words)


# --- main -------------------------------------------------------------------

def main(argv=None) -> None:
    args = parse_args(argv)

    cfg = load_calibration(args.config, args.fps) if not args.calibrate else None
    gate_kw = {}
    if cfg:
        gate_kw = {"start_velocity": cfg["start_velocity"], "stop_velocity": cfg["stop_velocity"]}
        print(f"[config] gate from {args.config}: start={cfg['start_velocity']} stop={cfg['stop_velocity']}")
    if args.start_velocity is not None:
        gate_kw["start_velocity"] = args.start_velocity
    if args.stop_velocity is not None:
        gate_kw["stop_velocity"] = args.stop_velocity

    stop = threading.Event()
    frame_q: Queue = Queue(maxsize=4)
    show_q: Queue = Queue(maxsize=1)
    cap_stats = DropStats()
    cap = Capture(args, frame_q, cap_stats, stop)
    if not cap.opened:
        raise SystemExit(f"cannot open {'video ' + args.video if args.video else 'camera ' + str(args.camera)}")

    det = mpdetect.build_detectors(Path(args.tasks), not args.no_face)

    if args.calibrate:
        cap.start()
        ts = mpdetect.MonotonicTs()

        def read_frame():
            try:
                bgr, ts_ms = frame_q.get(timeout=1.0)
            except Empty:
                return None
            return mpdetect.detect_frame(mpdetect.to_mp_image(bgr), ts(ts_ms), det)

        try:
            run_calibration(read_frame, out_path=args.config,
                            camera_info={"width": args.width, "height": args.height,
                                         "target_fps": args.fps, "face": not args.no_face})
        finally:
            stop.set()
            det.close()
        return

    model = load_model(args.model, args.model_path, args.labels, args.threads, args.aslcitizen_norm)
    if args.vocab:
        from models import RestrictedModel
        src = args.drill if args.vocab.lower() == "drill" and args.drill else args.vocab
        model = RestrictedModel(model, [w for w in src.split(",") if w.strip()])
        print(f"[vocab] limited to {int(model.mask.sum())} classes: "
              + ", ".join(f"{w}->{'/'.join(v)}" for w, v in model.matched.items())
              + (f"   NOT IN MODEL: {', '.join(model.unmatched)}" if model.unmatched else ""), flush=True)
    warm = model.warmup()
    print(f"[warmup] one dummy inference: {warm * 1000:.0f} ms ({model.name})")

    gate = MotionGate(**gate_kw)
    engine = Engine(model, gate, TokenAdmitter(args.top1_threshold, args.top2_threshold),
                    args.reject_threshold)
    speaker = None if args.no_audio else Speaker(Path(args.wavs), player=args.player)
    phrase = PhraseBuffer(flush_after_s=args.phrase_gap)
    logger = None if args.no_log else SessionLogger(args.session_dir, meta={
        "argv": vars(args), "model": model.name, "labels_are_real": getattr(model, "labels_are_real", True),
        "gate": {"start_velocity": gate.start_velocity, "stop_velocity": gate.stop_velocity,
                 "start_frames": gate.start_frames, "stop_frames": gate.stop_frames,
                 "min_frames": gate.min_frames, "max_frames": gate.max_frames}})

    inf = Inference(args, frame_q, show_q, engine, det, phrase, speaker, logger, stop, cap)
    inf.cap_stats = cap_stats

    print(f"running  {args.width}x{args.height}  model={model.name}  face={'off' if args.no_face else 'on'}"
          f"  vocab={model.vocab_size}  speech={'immediate' if args.speak_immediately else f'phrase({args.phrase_gap}s)'}"
          f"  reject<{args.reject_threshold}    ctrl-c to stop\n")
    if logger:
        print(f"[log] {logger.dir}")

    cap.start()
    inf.start()
    try:
        while not stop.is_set():
            if args.show:
                try:
                    cv2.imshow("signtalk", show_q.get(timeout=0.05))
                except Empty:
                    pass
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        inf.join(timeout=3.0)
        if args.show:
            cv2.destroyAllWindows()
        if speaker:
            speaker.close(drain_s=3.0)
        if logger:
            logger.close()
        det.close()
        def _pct(xs, q):
            return f"{np.percentile(xs, q):.1f}" if xs else "n/a"
        print(f"\nframes={inf.frames}  tracking ms p50/p95={_pct(inf.track_ms,50)}/{_pct(inf.track_ms,95)}"
              f"  classify ms p50/p95={_pct(inf.infer_ms,50)}/{_pct(inf.infer_ms,95)}"
              f"  capture-dropped={cap_stats.dropped}"
              + (f"  audio-dropped={speaker.stats.dropped}" if speaker else ""))
        print(f"transcript: {' '.join(inf.transcript) or '(nothing)'}")


if __name__ == "__main__":
    main()
