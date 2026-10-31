"""
Offline accuracy evaluation from recorded clips (no live camera).

    eval_clips/
        apple_01.mp4
        ...
        labels.csv            filename,label

    python eval_clips.py                          # PopSign-250, ./eval_clips
    python eval_clips.py --model aslcitizen       # ASL Citizen ST-GCN
    python eval_clips.py --selftest               # wiring check on synthetic data

Each clip goes through the SAME MediaPipe -> assemble_frame path as realtime.py
(mpdetect.detect_frame), then is scored two ways:

  whole-clip : the entire clip fed to the model as one window.
  gate       : the live MotionGate replayed over the frames (flushed at clip end);
               prediction = the longest window the gate produced. Also recorded:
               how many windows it produced (0 / 1 / several).

If the two accuracies differ a lot the gate (segmentation) is the problem, not
the model. Labels for --model popsign are PopSign-250 glosses. For --model
aslcitizen they must be the model's label strings ("sign_<index>" until a real
gloss file is supplied).

Record clips UNMIRRORED (no selfie-flip), shoulders and both hands in frame; use
--flip if your phone saved them mirrored.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from calibrate import DEFAULT_PATH as CALIB_PATH, load_calibration
from gate import MotionGate
from landmarks import LHAND_OFF, POSE_OFF, RHAND_OFF
from models import CHOICES, load_model

VIDEO_EXT = {".mp4", ".mov", ".webm", ".m4v", ".avi", ".mkv"}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


# --- extraction ---------------------------------------------------------------

def extract_clip(path: Path, task_dir: Path, use_face: bool, flip: bool = False):
    """Video file -> ((T,543,3) frames, native fps, frame count). Fresh detectors per clip."""
    import cv2
    import mpdetect

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps <= 0 or fps > 240:
        fps = 30.0
    det = mpdetect.build_detectors(task_dir, use_face)
    ts = mpdetect.MonotonicTs()
    frames, i = [], 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if flip:
                bgr = cv2.flip(bgr, 1)
            frames.append(mpdetect.detect_frame(mpdetect.to_mp_image(bgr), ts(i / fps * 1000.0), det))
            i += 1
    finally:
        cap.release()
        det.close()
    if not frames:
        raise IOError(f"no frames decoded from {path}")
    return np.stack(frames).astype(np.float32), float(fps), len(frames)


def presence(frames: np.ndarray) -> dict:
    l = np.isfinite(frames[:, LHAND_OFF, 0])
    r = np.isfinite(frames[:, RHAND_OFF, 0])
    p = np.isfinite(frames[:, POSE_OFF + 11, 0]) & np.isfinite(frames[:, POSE_OFF + 12, 0])
    return {"left_hand": float(l.mean()), "right_hand": float(r.mean()),
            "any_hand": float((l | r).mean()), "shoulders": float(p.mean())}


# --- scoring ------------------------------------------------------------------

def gate_windows(frames: np.ndarray, gate_kw: dict) -> list[np.ndarray]:
    g = MotionGate(**gate_kw)
    wins = []
    for f in frames:
        w = g.push(f)
        if w is not None:
            wins.append(w)
    w = g.flush()
    if w is not None:
        wins.append(w)
    return wins


def fmt_top(top) -> str:
    return "|".join(f"{l}:{s:.3f}" for l, s in top)


def score_clip(frames: np.ndarray, model, gate_kw: dict) -> dict:
    t0 = time.perf_counter()
    whole = model.top_k(model.predict(frames), 3)
    whole_ms = (time.perf_counter() - t0) * 1000
    wins = gate_windows(frames, gate_kw)
    gate_tops = [model.top_k(model.predict(w), 3) for w in wins]
    longest = int(np.argmax([w.shape[0] for w in wins])) if wins else None
    return {
        "whole_top3": whole, "whole_ms": whole_ms,
        "gate_n": len(wins), "gate_frames": [int(w.shape[0]) for w in wins],
        "gate_tops": gate_tops,
        "gate_top3": gate_tops[longest] if longest is not None else None,
    }


def correct(top, label, k) -> bool:
    return top is not None and label in [l for l, _ in top[:k]]


# --- reporting ----------------------------------------------------------------

def pct(k, n):
    return f"{100 * k / n:5.1f}%" if n else "  n/a "


def ci(k, n):
    lo, hi = wilson(k, n)
    return f"[{100 * lo:.0f}-{100 * hi:.0f}%]"


def summarize(rows: list[dict], skipped: list[tuple[str, str]], model_name: str,
              labels_real: bool, out=print, plumbing_only: bool = False) -> None:
    n = len(rows)
    out("\n" + "=" * 72)
    out(f"EVAL SUMMARY  model={model_name}  clips scored={n}  skipped={len(skipped)}")
    if not labels_real:
        out("  NOTE: label names are 'sign_<index>' placeholders (real glosses unavailable)")
    for f, why in skipped:
        out(f"  skipped {f}: {why}")
    if not n:
        return
    if n < 30:
        out(f"  WARNING: only {n} clips -- intervals below are wide; do not over-read these numbers.")
    out("=" * 72)

    def line(name, k, tot):
        if plumbing_only:
            out(f"  {name:<34} (suppressed: plumbing only, says nothing about accuracy)")
            return
        out(f"  {name:<34} {k:>4}/{tot:<4} {pct(k, tot)}  95% CI {ci(k, tot)}")

    w1 = sum(r["whole_top1_ok"] for r in rows)
    w3 = sum(r["whole_top3_ok"] for r in rows)
    out("\nWHOLE-CLIP (entire clip = one window)")
    line("top-1", w1, n)
    line("top-3", w3, n)
    g1 = sum(r["gate_top1_ok"] for r in rows)
    g3 = sum(r["gate_top3_ok"] for r in rows)
    ga = sum(r["gate_any_top1_ok"] for r in rows)
    out("\nGATE-SEGMENTED (live MotionGate replay; longest window; zero windows = wrong)")
    line("top-1", g1, n)
    line("top-3", g3, n)
    line("top-1 on ANY window", ga, n)
    wc = Counter(min(r["gate_windows"], 2) for r in rows)
    out(f"  windows per clip:  zero={wc[0]}  one={wc[1]}  several={wc[2]}")
    if not plumbing_only and abs(w1 - g1) / n > 0.15:
        out(f"  >> whole-clip and gate top-1 differ by {100 * abs(w1 - g1) / n:.0f} points: "
            f"{'the GATE is losing signs' if g1 < w1 else 'whole-clip padding/idle frames are hurting'}"
            " -- tune/calibrate the gate before blaming the model.")

    out("\nLANDMARK PRESENCE (mean over all frames of all clips)")
    tot_f = sum(r["frames"] for r in rows)
    for key, name in (("left_hand", "left hand"), ("right_hand", "right hand"),
                      ("any_hand", "either hand"), ("shoulders", "both shoulders (gate needs these)")):
        v = sum(r[f"p_{key}"] * r["frames"] for r in rows) / tot_f
        out(f"  {name:<36} {100 * v:5.1f}% of frames")
    low = [r for r in rows if r["p_any_hand"] < 0.5]
    if low:
        out(f"  {len(low)} clip(s) had a hand in <50% of frames (tracking failure, not classifier): "
            + ", ".join(r["file"] for r in low[:8]) + (" ..." if len(low) > 8 else ""))

    if plumbing_only:
        out("\n(per-label table and confusions suppressed: plumbing only)")
        return
    out("\nPER-LABEL (whole-clip top-1 / gate top-1)")
    by = defaultdict(list)
    for r in rows:
        by[r["label"]].append(r)
    for lab in sorted(by):
        rs = by[lab]
        out(f"  {lab:<22} n={len(rs):<3} whole {sum(x['whole_top1_ok'] for x in rs)}/{len(rs)}"
            f"   gate {sum(x['gate_top1_ok'] for x in rs)}/{len(rs)}")

    conf = Counter((r["label"], r["whole_pred1"]) for r in rows if not r["whole_top1_ok"])
    out("\nCONFUSIONS (whole-clip, wrong top-1 only; true -> predicted)")
    if conf:
        for (t, p), c in conf.most_common(15):
            out(f"  {t:<22} -> {p:<22} x{c}")
    else:
        out("  none")


CSV_FIELDS = ["file", "label", "frames", "fps", "p_left_hand", "p_right_hand", "p_any_hand",
              "p_shoulders", "whole_top3", "whole_top1_ok", "whole_top3_ok", "whole_pred1",
              "gate_windows", "gate_window_frames", "gate_top3", "gate_top1_ok",
              "gate_top3_ok", "gate_any_top1_ok", "whole_ms"]


def write_csv(rows, path):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def make_row(name, label, frames, fps, sc) -> dict:
    pr = presence(frames)
    wt = sc["whole_top3"]
    return {
        "file": name, "label": label, "frames": int(frames.shape[0]), "fps": round(fps, 2),
        **{f"p_{k}": round(v, 4) for k, v in pr.items()},
        "whole_top3": fmt_top(wt), "whole_pred1": wt[0][0],
        "whole_top1_ok": int(correct(wt, label, 1)), "whole_top3_ok": int(correct(wt, label, 3)),
        "gate_windows": sc["gate_n"], "gate_window_frames": " ".join(map(str, sc["gate_frames"])),
        "gate_top3": fmt_top(sc["gate_top3"]) if sc["gate_top3"] else "",
        "gate_top1_ok": int(correct(sc["gate_top3"], label, 1)),
        "gate_top3_ok": int(correct(sc["gate_top3"], label, 3)),
        "gate_any_top1_ok": int(any(correct(t, label, 1) for t in sc["gate_tops"])),
        "whole_ms": round(sc["whole_ms"], 2),
    }


# --- selftest -----------------------------------------------------------------

def selftest(args) -> int:
    """Plumbing only. Says NOTHING about accuracy."""
    import tempfile
    print("SELFTEST -- PLUMBING ONLY, says nothing about accuracy.\n")
    model = load_model(args.model, args.model_path, args.labels, args.threads, args.aslcitizen_norm)
    sys.path.insert(0, str(Path(__file__).parent))
    from test_helpers import synth_frame

    # 1. synthetic landmark frames: still / moving / still -> gate must yield one window
    frames = np.stack([synth_frame(0.30)] * 12 + [synth_frame(0.30 + i * 0.01) for i in range(25)]
                      + [synth_frame(0.55)] * 15)
    sc = score_clip(frames, model, {})
    assert sc["gate_n"] == 1, f"expected 1 gate window, got {sc['gate_n']}"
    row = make_row("synthetic_landmarks", model.top_k(model.predict(frames), 1)[0][0], frames, 30.0, sc)
    print(f"  synthetic landmarks : gate windows={sc['gate_n']} frames={sc['gate_frames']}  PASS")

    # 2. real decode + MediaPipe path on a synthetic video (no person -> nothing detected)
    import cv2
    tmp = Path(tempfile.mkdtemp(prefix="evalselftest_"))
    vp = tmp / "synthetic.mp4"
    vw = cv2.VideoWriter(str(vp), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (320, 240))
    rng = np.random.default_rng(0)
    for i in range(30):
        img = (rng.random((240, 320, 3)) * 255).astype(np.uint8)
        cv2.rectangle(img, (10 + 5 * i, 50), (60 + 5 * i, 120), (0, 255, 0), -1)
        vw.write(img)
    vw.release()
    fr, fps, n = extract_clip(vp, Path(args.tasks), False)
    assert fr.shape == (30, 543, 3) and abs(fps - 15.0) < 0.5, (fr.shape, fps)
    sc2 = score_clip(fr, model, {})
    print(f"  video decode+MediaPipe: frames={n} fps={fps:.1f} hands-in-frames="
          f"{presence(fr)['any_hand']:.2f} (expected 0: noise has no person)  PASS")
    row2 = make_row(vp.name, "n/a", fr, fps, sc2)

    # 3. csv + summary
    summarize([row, row2], [], model.name, getattr(model, "labels_are_real", True),
              plumbing_only=True)
    out = tmp / "eval_results.csv"
    write_csv([row, row2], out)
    print(f"\n  wrote {out} ({len(CSV_FIELDS)} columns)")
    print("\nSELFTEST OK -- plumbing only, says nothing about accuracy.")
    return 0


# --- main ---------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", default="eval_clips")
    ap.add_argument("--labels-csv", default=None, help="default: <clips>/labels.csv")
    ap.add_argument("--out", default="eval_results.csv")
    ap.add_argument("--model", choices=CHOICES, default="popsign")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--labels", default=None, help="vocab / gloss JSON for the model")
    ap.add_argument("--aslcitizen-norm", choices=("train", "valid"), default="train")
    ap.add_argument("--tasks", default="models")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--no-face", action="store_true",
                    help="skip FaceLandmarker (PopSign model was trained WITH face; ST-GCN ignores face)")
    ap.add_argument("--flip", action="store_true", help="mirror frames horizontally before detection")
    ap.add_argument("--config", default=CALIB_PATH, help="calibration.json for gate thresholds")
    ap.add_argument("--start-velocity", type=float, default=None)
    ap.add_argument("--stop-velocity", type=float, default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest(args)

    clips = Path(args.clips)
    lab_csv = Path(args.labels_csv) if args.labels_csv else clips / "labels.csv"
    files = sorted(p for p in clips.glob("*") if p.suffix.lower() in VIDEO_EXT) if clips.is_dir() else []
    if not files or not lab_csv.exists():
        print(f"""eval_clips.py: nothing to evaluate.

  clips dir   : {clips}/   ({'missing' if not clips.is_dir() else f'{len(files)} video files'})
  labels file : {lab_csv}   ({'found' if lab_csv.exists() else 'missing'})

Usage:
  1. Put video files ({', '.join(sorted(VIDEO_EXT))}) in {clips}/
  2. Create {lab_csv} with a header and one row per clip:
         filename,label
         apple_01.mp4,apple
     (PopSign-250 glosses for --model popsign; see models/sign_to_prediction_index_map.json)
  3. python eval_clips.py [--model popsign|aslcitizen]

To check the harness wiring without clips:  python eval_clips.py --selftest""")
        return 2

    with open(lab_csv, newline="") as fh:
        rd = csv.DictReader(fh)
        if not rd.fieldnames or not {"filename", "label"} <= set(rd.fieldnames):
            print(f"{lab_csv}: header must be exactly `filename,label`")
            return 2
        labels = {r["filename"].strip(): r["label"].strip() for r in rd}

    model = load_model(args.model, args.model_path, args.labels, args.threads, args.aslcitizen_norm)
    vocab = set(model.idx_to_sign.values())
    gate_kw = {}
    cfg = load_calibration(args.config)
    if cfg:
        gate_kw = {"start_velocity": cfg["start_velocity"], "stop_velocity": cfg["stop_velocity"]}
        print(f"[config] gate thresholds from {args.config}: {gate_kw}")
    if args.start_velocity is not None:
        gate_kw["start_velocity"] = args.start_velocity
    if args.stop_velocity is not None:
        gate_kw["stop_velocity"] = args.stop_velocity
    if not gate_kw:
        print("[config] gate thresholds: library defaults (no calibration.json; clip fps may differ)")

    rows, skipped = [], []
    for fp in files:
        label = labels.get(fp.name)
        if label is None:
            skipped.append((fp.name, "not in labels.csv"))
            continue
        if label not in vocab:
            skipped.append((fp.name, f"label '{label}' is not in the {args.model} vocabulary"))
            continue
        try:
            frames, fps, _ = extract_clip(fp, Path(args.tasks), not args.no_face, args.flip)
            sc = score_clip(frames, model, gate_kw)
        except Exception as e:                      # one bad clip must not stop the run
            skipped.append((fp.name, f"error: {e}"))
            continue
        row = make_row(fp.name, label, frames, fps, sc)
        rows.append(row)
        print(f"  {fp.name:<28} {label:<14} whole={row['whole_pred1']:<14} "
              f"{'OK ' if row['whole_top1_ok'] else 'ERR'} gate_windows={row['gate_windows']} "
              f"hands={row['p_any_hand']:.0%}")
    missing = [n for n in labels if not (clips / n).exists()]
    skipped += [(n, "listed in labels.csv but file missing") for n in missing]

    write_csv(rows, args.out)
    summarize(rows, skipped, model.name, getattr(model, "labels_are_real", True))
    print(f"\nper-clip results: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
