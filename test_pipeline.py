"""Synthetic end-to-end test: no camera, no MediaPipe. Verifies wiring."""
import numpy as np

from classifier import SignClassifier, TokenAdmitter
from gate import MotionGate
from landmarks import (
    LHAND_OFF, N_TOTAL, POSE_L_SHOULDER, POSE_L_WRIST, POSE_OFF,
    POSE_R_SHOULDER, POSE_R_WRIST, RHAND_OFF, FrameRing, assemble_frame,
    detected_points, hands_present, new_frame,
)
import json, tempfile, threading, time
from pathlib import Path
from queue import Queue

import os
MODEL = os.environ.get("SIGN_MODEL", "models/asl-signs.onnx")
VOCAB = os.environ.get("SIGN_VOCAB", "models/sign_to_prediction_index_map.json")


from test_helpers import LM, synth_frame


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    assert cond, name


print("1. landmark assembly")
f = synth_frame(0.5)
check(f"shape {f.shape} == (543, 3)", f.shape == (N_TOTAL, 3))
check("face block is NaN (not supplied)", bool(np.isnan(f[0, 0])))
check("LHand block finite", bool(np.isfinite(f[LHAND_OFF, 0])))
check("RHand block finite", bool(np.isfinite(f[RHAND_OFF, 0])))
check("Pose block finite", bool(np.isfinite(f[POSE_OFF, 0])))
check("hands_present == 2", hands_present(f) == 2)
check(f"detected_points == 75 (21+33+21), got {detected_points(f)}", detected_points(f) == 75)
empty = new_frame()
check("empty frame is all-NaN", bool(np.isnan(empty).all()))

print("\n2. ring buffer")
ring = FrameRing(capacity=4)
for i in range(6):
    ring.push(synth_frame(0.5 + i * 0.01))
check(f"len capped at 4, got {len(ring)}", len(ring) == 4)
check(f"stack shape {ring.stack().shape}", ring.stack().shape == (4, N_TOTAL, 3))

print("\n3. motion gate state machine")
gate = MotionGate()
emitted = []
# 10 still frames -> must stay idle
for _ in range(10):
    assert gate.push(synth_frame(0.30)) is None
check("stays idle while still", gate.state == "idle")
# 25 moving frames (0.01/frame = 0.05 shoulder-widths -> above start threshold)
for i in range(25):
    out = gate.push(synth_frame(0.30 + i * 0.010))
    if out is not None:
        emitted.append(out)
check("entered recording on motion", gate.state == "recording")
# 15 still frames -> should close the segment
for _ in range(15):
    out = gate.push(synth_frame(0.55))
    if out is not None:
        emitted.append(out)
check(f"emitted exactly 1 window, got {len(emitted)}", len(emitted) == 1)
check(f"window shape {emitted[0].shape}", emitted[0].shape[1:] == (N_TOTAL, 3))
check(f"window length {emitted[0].shape[0]} in [10, 90]", 10 <= emitted[0].shape[0] <= 90)
check("returned to idle", gate.state == "idle")

print("\n4. classifier")
clf = SignClassifier(MODEL, VOCAB)
check(f"vocab size {clf.vocab_size} == 250", clf.vocab_size == 250)
probs = clf.predict(emitted[0])
check(f"probs shape {probs.shape}", probs.shape == (250,))
check(f"probs sum to 1 ({probs.sum():.6f})", abs(float(probs.sum()) - 1.0) < 1e-4)
check("all probs in [0,1]", bool(((probs >= 0) & (probs <= 1)).all()))
top = clf.top_k(probs, 3)
print(f"       top-3 on synthetic noise: {[(l, round(s,3)) for l,s in top]}")

print("\n5. token admitter")
adm = TokenAdmitter(refractory_s=0.0)
check("single tick does not commit", adm.offer([("apple", 0.9), ("bird", 0.05)]) is None)
tok = adm.offer([("apple", 0.9), ("bird", 0.05)])
check("second identical tick commits via 'stable'", tok is not None and tok.via == "stable")
adm2 = TokenAdmitter(refractory_s=0.0)
adm2.offer([("x", 0.01), ("y", 0.005)])
tok2 = adm2.offer([("cat", 0.2), ("dog", 0.15)])
check("mid-confidence with credible runner-up commits via 'band'",
      tok2 is not None and tok2.via == "band")
adm3 = TokenAdmitter(refractory_s=5.0)
adm3.offer([("apple", 0.9), ("b", 0.01)])
adm3.offer([("apple", 0.9), ("b", 0.01)])
check("refractory blocks immediate repeat",
      adm3.offer([("apple", 0.9), ("b", 0.01)]) is None)

print("\n6. latency on variable window lengths")
import time
for T in (16, 32, 48, 64):
    w = emitted[0][:1].repeat(T, axis=0)
    t0 = time.perf_counter()
    for _ in range(5):
        clf.predict(w)
    print(f"       T={T:3d}  {(time.perf_counter()-t0)/5*1000:6.2f} ms/inference")



# =============================================================================
# v2 components. Every test below uses synthetic data: PLUMBING ONLY, says
# nothing about recognition accuracy.
# =============================================================================
import aslcitizen as ac
from calibrate import compute_thresholds, velocities_from_frames
from engine import Engine
from phrase import PhraseBuffer
from pipeline import DropStats, Speaker, put_drop_newest, put_drop_oldest
from session_log import SessionLogger


def marker_window(T=128):
    """Left hand block x=0.20, right hand block x=0.80, distinct y per landmark index."""
    w = np.full((T, N_TOTAL, 3), np.nan, dtype=np.float32)
    for i in range(21):
        w[:, LHAND_OFF + i] = (0.20, 0.10 + 0.01 * i, 0.0)
        w[:, RHAND_OFF + i] = (0.80, 0.50 + 0.01 * i, 0.0)
    for i in range(33):
        w[:, POSE_OFF + i] = (0.50, 0.30, 0.0)
    w[:, POSE_OFF + 11] = (0.40, 0.40, 0.0)   # user's left shoulder (image left of centre)
    w[:, POSE_OFF + 12] = (0.60, 0.40, 0.0)
    return w


print("\n7. ASL Citizen adapter: hand order (fails if left/right swapped)")
mw = marker_window()
raw = ac.window_to_raw75(mw)
check("raw layout is (T, 75, 2)", raw.shape == (128, 75, 2))
check("raw 33-53 (RIGHT hand) holds our RHand block (x=0.80)", bool(np.allclose(raw[:, 33:54, 0], 0.80)))
check("raw 54-74 (LEFT hand) holds our LHand block (x=0.20)", bool(np.allclose(raw[:, 54:75, 0], 0.20)))
x = ac.preprocess(mw, norm="valid")
check(f"model input shape {x.shape} == (2, 128, 27)", x.shape == (2, 128, 27))
check("nodes 7-16 are the LEFT hand (x smaller than right hand)",
      bool(x[0, :, 7:17].mean() < x[0, :, 17:27].mean()))
check("node 7 (left wrist) carries LHand landmark 0 (y=0.10 -> lowest y of the left hand)",
      bool(np.argmin(x[1, 0, 7:17]) == 0))
check("node 17 (right wrist) carries RHand landmark 0 (lowest y of the right hand)",
      bool(np.argmin(x[1, 0, 17:27]) == 0))
check("node 3 (pose 11, left shoulder) left of node 4 (pose 12)", bool(x[0, 0, 3] < x[0, 0, 4]))
swapped = marker_window()
swapped[:, [LHAND_OFF + i for i in range(21)]], swapped[:, [RHAND_OFF + i for i in range(21)]] = \
    marker_window()[:, [RHAND_OFF + i for i in range(21)]], marker_window()[:, [LHAND_OFF + i for i in range(21)]]
xs = ac.preprocess(swapped, norm="valid")
check("a window with hands swapped DOES change which nodes are leftmost (test has teeth)",
      bool(xs[0, :, 7:17].mean() > xs[0, :, 17:27].mean()))

print("\n8. ASL Citizen adapter: missing landmarks, length handling, exact preprocessing")
nanw = np.full((40, N_TOTAL, 3), np.nan, dtype=np.float32)
xn = ac.preprocess(nanw)
check("all-NaN window -> finite model input (NaN -> 0 as in training)", bool(np.isfinite(xn).all()))
hw = marker_window(40); hw[:, LHAND_OFF:LHAND_OFF + 21] = np.nan
check("missing left hand becomes raw zeros",
      bool((ac.window_to_raw75(hw)[:, 54:75] == 0).all()))
check("300-frame clip -> 128 frames", ac.preprocess(np.random.rand(300, N_TOTAL, 3).astype("f")).shape == (2, 128, 27))
check("20-frame clip is zero-padded to 128", ac.preprocess(np.random.rand(20, N_TOTAL, 3).astype("f")).shape == (2, 128, 27))
check("downsample keeps exactly 128 of 300", ac.downsample(np.arange(300.0).reshape(300, 1, 1), 128).shape[0] == 128)


def reference_getitem(raw_window_75x2, max_frames=128):
    """Port of ASLCitizen.__getitem__ (vendor/.../asl_citizen_dataset_pose.py) on a (T,75,2) array."""
    data0 = raw_window_75x2.copy()
    length = data0.shape[0]
    if length > max_frames:
        data0 = ac.downsample(data0, max_frames)
    if length < max_frames:
        data0 = np.pad(data0, ((0, max_frames - length), (0, 0), (0, 0)))
    sl, sr = data0[:, 11, :], data0[:, 12, :]
    center = np.zeros(2)
    for i in range(len(sl)):
        center = center + (sr[i] + sl[i]) / 2
    center = center / sl.shape[0]
    md = np.mean(np.sqrt(((sl - sr) ** 2).sum(-1)))
    if md != 0:
        data0 = (data0 - center) * (1.0 / md)
    kp = [0, 2, 5, 11, 12, 13, 14, 33, 37, 38, 41, 42, 45, 46, 49, 50, 53, 54, 58, 59, 62, 63, 66, 67, 70, 71, 74]
    d = np.concatenate([data0[:, 0:33], data0[:, 54:], data0[:, 33:54]], axis=1)[:, kp, :]
    return np.transpose(d, (2, 0, 1))


rng = np.random.default_rng(1)
for T in (20, 90, 128, 250):
    w = rng.random((T, N_TOTAL, 3)).astype(np.float32)
    w[3:6, LHAND_OFF:LHAND_OFF + 21] = np.nan
    ref = reference_getitem(ac.window_to_raw75(w))
    check(f"preprocess == reference port of the dataset code (T={T}), max diff "
          f"{np.abs(ref - ac.preprocess(w)).max():.1e}", bool(np.allclose(ref, ac.preprocess(w), atol=1e-5)))
try:
    import torch  # noqa: F401
    TORCH = True
except Exception:
    TORCH = False
if TORCH:
    import sys as _sys
    _sys.path.insert(0, str(Path("vendor/ASL-citizen-code/ST-GCN").resolve()))
    from asl_citizen_dataset_pose import ASLCitizen as _DS
    for T in (20, 90, 250):
        w = rng.random((T, N_TOTAL, 3)).astype(np.float32)
        d = tempfile.mkdtemp()
        np.save(f"{d}/p.npy", ac.window_to_raw75(w))
        ds = _DS.__new__(_DS)
        ds.max_frames, ds.transforms = 128, None
        ds.pose_paths, ds.video_info, ds.labels = [f"{d}/p.npy"], [["u", "f", "g"]], [0]
        ds.gloss_dict = {"g": 0}
        up = ds[0][0].numpy()
        check(f"preprocess == the ACTUAL vendor ASLCitizen.__getitem__ (T={T}), max diff "
              f"{np.abs(up - ac.preprocess(w)).max():.1e}", bool(np.allclose(up, ac.preprocess(w), atol=1e-5)))
else:
    print("  SKIP  vendor-class comparison (torch not installed here)")

print("\n9. phrase buffer (fake clock)")
pb = PhraseBuffer(flush_after_s=1.5)
pb.add("apple", now=0.0); pb.add("eat", now=0.5)
check("not flushed at t=1.9 (1.4 s after last word)", pb.poll(now=1.9) is None)
check("flushed as ONE utterance at t=2.0 (1.5 s after last word)", pb.poll(now=2.0) == ["apple", "eat"])
check("flush happens once", pb.poll(now=9.0) is None)
pb.add("go", now=10.0)
check("busy gate (mid-sign) defers the flush", pb.poll(now=12.0, busy=True) is None)
check("...and counts as activity (still waiting at t=13.0)", pb.poll(now=13.0) is None)
check("flushes 1.5 s after the sign ended", pb.poll(now=13.6) == ["go"])
pb2 = PhraseBuffer(flush_after_s=1.0, reorder=lambda w: list(reversed(w)))
pb2.add("a", now=0); pb2.add("b", now=0.1)
check("reorder seam is applied between buffer and speech", pb2.poll(now=2.0) == ["b", "a"])
check("default reorder is identity", PhraseBuffer().reorder(["x", "y"]) == ["x", "y"])

print("\n10. reject threshold + segment admission")
a = TokenAdmitter(refractory_s=0.0)
check("below floor: nothing emitted even with a credible runner-up",
      a.offer_segment([("cat", 0.08), ("dog", 0.07)], reject_threshold=0.10) is None)
t = a.offer_segment([("cat", 0.62), ("dog", 0.07)], reject_threshold=0.10)
check("confident single segment commits (via 'confident')", t is not None and t.via == "confident")
a = TokenAdmitter(refractory_s=0.0)
t = a.offer_segment([("cat", 0.20), ("dog", 0.15)], reject_threshold=0.10)
check("mid-confidence + credible runner-up commits via 'band'", t is not None and t.via == "band")
check("same label again in the band does not repeat",
      a.offer_segment([("cat", 0.20), ("dog", 0.15)], reject_threshold=0.10) is None)
a = TokenAdmitter(refractory_s=0.0)
check("tick-mode offer() also honours the floor",
      a.offer([("cat", 0.05), ("dog", 0.04)], reject_threshold=0.10) is None)


class FakeModel:
    name, labels_are_real, vocab_size = "fake", True, 3

    def __init__(self, p): self.p = np.asarray(p)
    def predict(self, w): return self.p
    def top_k(self, probs, k=2):
        names = ["alpha", "beta", "gamma"]
        idx = np.argsort(probs)[::-1][:k]
        return [(names[i], float(probs[i])) for i in idx]


def drive(model_probs, reject):
    eng = Engine(FakeModel(model_probs), MotionGate(), TokenAdmitter(refractory_s=0.0), reject)
    segs = [eng.push(synth_frame(0.30)) for _ in range(10)]
    segs += [eng.push(synth_frame(0.30 + i * 0.010)) for i in range(25)]
    segs += [eng.push(synth_frame(0.55)) for _ in range(15)]
    return [s for s in segs if s is not None]


segs = drive([0.7, 0.2, 0.1], 0.10)
check("engine: one segment from still/move/still", len(segs) == 1)
check("engine: segment carries top-3 with scores", len(segs[0].top) == 3 and segs[0].top[0][0] == "alpha")
check("engine: segment carries a velocity trace of window length",
      len(segs[0].velocity_trace) == segs[0].window.shape[0])
check("engine: confident segment emitted", segs[0].token is not None and segs[0].token.label == "alpha")
segs = drive([0.06, 0.05, 0.04], 0.10)
check("engine: low-confidence segment is rejected and emits nothing",
      segs[0].rejected and segs[0].token is None)

print("\n11. gate: flush + velocity trace")
g = MotionGate()
for i in range(25):
    g.push(synth_frame(0.30 + i * 0.010))
check("mid-sign gate is recording", g.state == "recording")
w = g.flush()
check("flush() emits the in-progress sign", w is not None and w.shape[0] >= 10)
check("flush() returns to idle", g.state == "idle" and g.flush() is None)

print("\n12. threading: capture never blocks, queues stay bounded")
q = Queue(maxsize=4); st = DropStats(); got = []
stop = threading.Event()


def slow_consumer():
    while not stop.is_set():
        try:
            got.append(q.get(timeout=0.05)); time.sleep(0.01)
        except Exception:
            pass


th = threading.Thread(target=slow_consumer, daemon=True); th.start()
t0 = time.perf_counter(); maxq = 0
for i in range(300):                         # producer ~ 1 ms/frame, consumer ~ 10 ms/frame
    put_drop_oldest(q, i, st); maxq = max(maxq, q.qsize()); time.sleep(0.001)
prod_s = time.perf_counter() - t0
time.sleep(0.2); stop.set(); th.join(1.0)
check(f"producer ran 300 frames in {prod_s:.2f}s without stalling on the slow consumer", prod_s < 1.5)
check(f"queue never exceeded its bound (max {maxq} <= 4)", maxq <= 4)
check(f"frames were dropped, not blocked ({st.dropped} dropped)", st.dropped > 100)
check("freshest frame survived (drop-OLDEST)", got[-1] >= 290)
q2 = Queue(maxsize=2); s2 = DropStats()
r = [put_drop_newest(q2, i, s2) for i in range(5)]
check("put_drop_newest keeps the first items and reports drops", r == [False, False, True, True, True] and s2.dropped == 3)

tmp = Path(tempfile.mkdtemp())
for wname in ("hello", "world"):
    (tmp / f"{wname}.wav").write_bytes(b"RIFF")
played = []
spk = Speaker(tmp, maxsize=2, play_fn=lambda p: (time.sleep(0.3), played.append(p.stem)))
t0 = time.perf_counter()
res = [spk.say_phrase(["hello", "world"]) for _ in range(6)]      # playback is slow
call_s = time.perf_counter() - t0
check(f"say_phrase x6 returned in {call_s * 1000:.0f} ms (playback is 0.3 s/word): never blocks", call_s < 0.2)
check(f"bounded audio queue dropped overflow ({spk.stats.dropped} dropped)", spk.stats.dropped >= 2)
spk.close(drain_s=3.0)
check("audio still plays what it accepted, in order", played[:2] == ["hello", "world"])
spk2 = Speaker(tmp / "does_not_exist", play_fn=lambda p: None)
spk2.say_phrase(["nope"]); time.sleep(0.3); spk2.close()
check("missing WAV is skipped with a warning, no crash", spk2.spoken == [])

print("\n13. session logging")
sl = SessionLogger(tempfile.mkdtemp(), meta={"model": "fake"})
sl.log_segment(emitted_window := np.random.rand(17, N_TOTAL, 3).astype("f"),
               [(f"w{i}", 0.5 - 0.1 * i) for i in range(5)], np.linspace(0, 0.1, 17), "w0", "confident")
sl.log_segment(emitted_window, [("a", 0.05)], np.zeros(17), None)
sl.close()
lines = [json.loads(l) for l in (sl.dir / "index.jsonl").read_text().splitlines()]
check("two JSONL lines written", len(lines) == 2)
check("record has top-5, frames, trace, timestamp",
      len(lines[0]["top5"]) == 5 and lines[0]["frames"] == 17 and len(lines[0]["velocity_trace"]) == 17
      and "ts_iso" in lines[0])
check("raw landmark window round-trips from .npy",
      bool(np.array_equal(np.load(sl.dir / lines[0]["window_file"]), emitted_window)))
check("meta.json written", json.loads((sl.dir / "meta.json").read_text())["model"] == "fake")

print("\n14. calibration thresholds (synthetic jitter)")
rngc = np.random.default_rng(3)
jit = [synth_frame(0.30 + float(rngc.normal(0, 0.0015))) for _ in range(150)]
vel = velocities_from_frames(jit)
th_cfg = compute_thresholds(vel)
check(f"start_velocity {th_cfg['start_velocity']} above resting p99 {th_cfg['rest_velocity']['p99']}",
      th_cfg["start_velocity"] > th_cfg["rest_velocity"]["p99"])
check("stop_velocity < start_velocity (hysteresis)", th_cfg["stop_velocity"] < th_cfg["start_velocity"])
gj = MotionGate(start_velocity=th_cfg["start_velocity"], stop_velocity=th_cfg["stop_velocity"])
outs = [gj.push(f) for f in jit]
check("calibrated gate does not fire on the jitter it was calibrated on", gj.state == "idle" and not any(o is not None for o in outs))
try:
    compute_thresholds(np.zeros(3)); ok = False
except ValueError:
    ok = True
check("too few samples is a clear error", ok)

print("\n15. model warmup")
check(f"PopSign warmup runs one dummy inference ({clf.warmup() * 1000:.0f} ms)", True)

print("\n16. ASL Citizen ONNX end-to-end (skipped if models/aslcitizen_stgcn.onnx is absent)")
ACM = os.environ.get("ASLC_MODEL", "models/aslcitizen_stgcn.onnx")
if os.path.exists(ACM):
    acm = ac.AslCitizenClassifier(ACM, "models/aslcitizen_gloss.json")
    ww = emitted[0]
    pr = acm.predict(ww)
    check(f"{acm.vocab_size} classes", acm.vocab_size == 2731 and pr.shape == (2731,))
    check("probs sum to 1", abs(float(pr.sum()) - 1.0) < 1e-6)
    check("top-3 labels carry real gloss names from the gloss file",
          not any(l.startswith("sign_") for l, _ in acm.top_k(pr, 3)))
    check("labels_are_real is True with a gloss file", acm.labels_are_real is True)
    if TORCH:
        import copy
        import export_stgcn as _E
        _net, _, _ = _E.build(Path("models/ASL_citizen_stgcn_weights.pt"))
        with torch.no_grad():
            _ref = torch.softmax(_net(torch.from_numpy(ac.preprocess(ww)[None])), -1).numpy().ravel()
        check(f"adapter+ONNX == adapter+torch checkpoint forward (max prob diff {np.abs(_ref - pr).max():.1e})",
              bool(np.abs(_ref - pr).max() < 1e-4) and int(_ref.argmax()) == int(pr.argmax()))
    for T in (16, 32, 64, 90, 128, 200):
        wT = np.random.default_rng(T).random((T, N_TOTAL, 3)).astype(np.float32)
        acm.predict(wT)
        t0 = time.perf_counter()
        for _ in range(5):
            acm.predict(wT)
        print(f"       ASL Citizen T={T:3d}  {(time.perf_counter() - t0) / 5 * 1000:6.2f} ms/inference (preprocess + ONNX)")
    print(f"       ASL Citizen warmup: {acm.warmup() * 1000:.0f} ms")
else:
    print("  SKIP  models/aslcitizen_stgcn.onnx not present (run export_stgcn.py)")

print("\nALL CHECKS PASSED")
