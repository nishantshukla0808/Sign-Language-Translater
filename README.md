# signtalk — ASL sign **lookup** → spoken word

Word-level, isolated ASL signs only (not fingerspelling, not continuous sentences).
Pretrained weights only; nothing here trains anything.

**Read this framing first.** This is a *lookup with ranked candidates*, not a
translator. A model that is right ~6 times in 10 on unseen signers (ASL Citizen
ST-GCN: 59.52 % top-1 / 82.68 % top-5, published) will say a wrong word with
full confidence about 2 times in 5. The UI shows the top-3 with scores for that
reason, and anything under the reject floor is not spoken at all.

```
camera ──capture thread──▶ frame_q ──inference thread──▶ audio_q ──audio thread──▶ speaker
 (never blocks;            (bounded,   MediaPipe → gate →  (bounded,   (plays WAVs;
  drops oldest)             4)         model → admission →   8)         drops newest)
                                       phrase buffer
```

## Files

| File | Role |
|---|---|
| `landmarks.py` | MediaPipe → `(543,3)` assembly, NaN padding, `FrameRing` |
| `mpdetect.py` | **Shared** detector build + `detect_frame()` — used by live *and* offline eval |
| `gate.py` | `MotionGate`: sign start/end by wrist velocity; `flush()`, per-window velocity trace |
| `classifier.py` | PopSign-250 ONNX wrapper, `TokenAdmitter` (stable / band / segment-confident, reject floor) |
| `aslcitizen.py` | 543-layout → ST-GCN input adapter + ONNX wrapper (no PyTorch at runtime) |
| `export_stgcn.py` | One-time `.pt` → `.onnx` export with parity check (needs PyTorch once) |
| `models.py` | `--model {popsign,aslcitizen}` loader |
| `engine.py` | frame → (maybe) classified segment; no I/O, shared by live/eval/tests |
| `phrase.py` | phrase buffer with the reorder seam |
| `pipeline.py` | bounded queues, drop policies, `Speaker` |
| `session_log.py` | per-session directory + JSONL index + raw `.npy` windows |
| `calibrate.py` | `--calibrate` threshold fitting |
| `overlay.py` | `--show` debug overlay |
| `realtime.py` | the three-thread live app |
| `eval_clips.py` | **offline accuracy from recorded clips** |
| `test_pipeline.py` | wiring tests (synthetic — plumbing only, says nothing about accuracy) |
| `scripts/` | `fetch_models.sh`, `render_wavs.sh`, `pi_gate_check.sh`, `pi_setup.sh`, `signtalk.service` |

## Setup (laptop)

```bash
pip install -r requirements.txt            # mediapipe onnxruntime numpy opencv
bash scripts/fetch_models.sh               # PopSign onnx + ASL Citizen .pt (raw GitHub / release assets)
# MediaPipe Tasks bundles (storage.googleapis.com)
B=https://storage.googleapis.com/mediapipe-models
curl -L -o models/hand_landmarker.task      $B/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
curl -L -o models/pose_landmarker_lite.task $B/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
curl -L -o models/face_landmarker.task      $B/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
bash scripts/render_wavs.sh                # speech assets (Piper if PIPER_VOICE set, else espeak-ng)
python test_pipeline.py                    # must end with ALL CHECKS PASSED
```

On Linux the mediapipe wheel needs `libegl1` (`sudo apt install libegl1 libgles2`).

### ASL Citizen model (optional second model)

```bash
pip install -r requirements-export.txt     # torch + onnx, export machine only
python export_stgcn.py                     # -> models/aslcitizen_stgcn.onnx, prints the parity check
python realtime.py --model aslcitizen --show
```

**Labels:** the model has 2,731 outputs. The index→gloss list is the alphabetically
sorted gloss column of the ASL Citizen CSVs, which are not in this repo. Until you
supply `models/aslcitizen_gloss.json` (`{"0": "...", ...}` or a JSON list) the
labels print as `sign_<index>` and the program says so at startup. Do not use
`models/aslcitizen_targeted.json` for this — it is a 244-sign shortlist from another project.

## Measuring accuracy from clips (the main workflow)

```
eval_clips/
    apple_01.mp4  …
    labels.csv          # header: filename,label   (PopSign-250 glosses for --model popsign)
python eval_clips.py                       # prints summary, writes eval_results.csv
python eval_clips.py --model aslcitizen
python eval_clips.py --selftest            # wiring only; accuracy lines are suppressed
```

Record **unmirrored**, both shoulders and both hands in frame (`--flip` if your
phone saved them mirrored). The same `mpdetect.detect_frame` path as the live app
is used, then each clip is scored **two ways** — whole clip as one window, and
through a replay of the live `MotionGate`. If they disagree a lot, the gate is the
problem, not the model. The summary also reports how often each hand was
detected, because "the model was wrong" and "the tracker lost the hand" look the
same from the output side. Fewer than ~30 clips: read the printed confidence
intervals, not the point estimate.

## Live

```bash
python realtime.py --calibrate --no-face      # once per camera / frame rate; writes calibration.json
python realtime.py --show                     # laptop, with debug overlay
python realtime.py --width 640 --height 480 --fps 15 --threads 3 --no-face   # Pi
```

`calibration.json` is auto-loaded when present. `--start-velocity/--stop-velocity` override it.
A dummy inference runs at startup so the first real sign is not eaten by warm-up.

`--show` draws both 21-point hand skeletons and the upper-body pose on the frame
(hands that are not detected simply do not appear — that is the signal), the gate
state with velocity vs its start threshold, which hands are present, measured fps,
and the **top-3 candidates with scores from the last segment**. If the skeleton is
wrong the problem is tracking; if it is right and the candidates are wrong it is
the classifier. It costs nothing without `--show`.

### Speech: phrase buffer vs speak-immediately — you choose

| | default: **phrase buffer** | `--speak-immediately` |
|---|---|---|
| Behaviour | words accumulate; after `--phrase-gap` seconds (1.5 default) of stillness (and the gate not mid-sign) they are spoken as one utterance | each word is spoken the moment it is committed; words stack up in the audio queue |
| Pro | one clean utterance; a seam for gloss→English reordering (`phrase.py: reorder`, identity today) | lowest latency; the original request |
| Con | adds up to the gap of delay; a wrong word is spoken along with the right ones | a wrong word is spoken *immediately*, with no chance to notice; overlapping words if the signer is quick |

With a ~60 %-accurate model, delaying until the end of a phrase gives the signer
(and `--show`) a moment to see the candidates before anything is said. Reordering
is not implemented: ASL gloss order is not English order.

### Admission and the reject floor

The model always returns one of N classes. `--reject-threshold` (default 0.10) is
a top-1 probability floor below which nothing is emitted. Above it, the original
stable/band logic applies, with one change: v1 required the same label on two
consecutive ticks ("stable"), which with one prediction per gate segment meant a
confident word only committed if the signer *repeated* it. Segment mode commits a
confident (≥ 0.30) segment directly (`via=confident`). The tick-based `offer()` is
unchanged. **The defaults were tuned for the 250-way softmax; the 2,731-way
ST-GCN's probability scale has not been calibrated here** — set it from your own
data (the logs and `eval_results.csv` have the scores).

### Session logs

Every emitted segment goes to `sessions/<timestamp>/`: `seg_NNNN.npy` (raw
`(T,543,3)` window, NaN = missing), one line in `index.jsonl` (top-5 with scores,
frame count, gate velocity trace, timestamp, whether a word was emitted, model),
and `meta.json` (all arguments, gate config). Replay a `.npy` through any model to
separate tracking failures from model failures. `--no-log` disables it.

## Raspberry Pi 5 (nothing here was run on a Pi)

1. **GATE 2 — run first:** `bash scripts/pi_gate_check.sh` on 64-bit Raspberry Pi OS.
   It reports whether `mediapipe` installs and imports on aarch64, whether the
   Hand/Pose landmarkers actually run (with a rough per-frame time on a blank
   frame), libEGL, audio devices, cameras, throttling.
   PyPI currently lists `manylinux_2_28_aarch64` wheels for mediapipe ≥ 1.0
   (checked via the PyPI JSON API); **whether they work on a Pi 5 is exactly what
   the script establishes.** 32-bit OS has no wheels.
2. `bash scripts/pi_setup.sh [--service]` — native venv + systemd unit. Not Docker:
   if you containerise later, **build on the Pi** (an x86_64 image will not run on ARM64).
3. Pi 5 has **no 3.5 mm jack**: USB audio adapter, I2S DAC, or Bluetooth speaker. Active cooling is needed.
4. Start at 640×480 @ 15 fps with `--no-face`. Gate thresholds are per-frame, so
   **re-run `--calibrate` on the Pi**; `calibration.json` warns if its fps differs
   from the run's by > 25 %.

### If GATE 2 fails

The architecture changes; do not build further on the Pi. Options, in order:
(a) a community aarch64 mediapipe build for your Python/OS; (b) build from source
(Google documents a Bazel build; slow on a Pi); (c) keep capture on the Pi and run
MediaPipe + inference on a laptop/server over the network; (d) a different
landmarker (e.g. a hand-only model) — which would mean re-validating the model input spec.

## Tuning

`gate.py` thresholds are shoulder-widths per **frame** (invariant to distance from
camera, not to frame rate): `start 0.020, stop 0.008, start_frames 3, stop_frames 8,
min_frames 10, max_frames 90` — or whatever `--calibrate` wrote. Note `max_frames=90` is
3 s at 30 fps but 6 s at 15 fps.

## Vocabulary

PopSign-250 was built to teach parents signs for toddlers (`potty`, `owie`, `kitty`,
`nap`…; no `work`, `help`, `name`, `hospital`, `phone`). ASL Citizen has 2,731 signs
(a far broader lexicon) but lower per-sign accuracy and, as shipped here, no names.

## The two models' input specs (they differ — easy to get wrong)

| | PopSign-250 (`asl-signs.onnx`) | ASL Citizen ST-GCN |
|---|---|---|
| Input | `(T, 543, 3)` `[Face, LHand, Pose, RHand]`, variable T | `(1, 2, T, 27)`, x,y only; adapter always produces T = 128 |
| Missing landmark | **NaN** (model normalises finite values itself) | **0.0** (that is what training used), *before* shoulder-normalisation |
| Hand order | L then pose then R | pose, **L (nodes 7–16), R (17–26)** — a test fails if swapped |
| Normalisation | inside the model | in `aslcitizen.preprocess`, mirrors the dataset code incl. its use of the zero-padded array for centre/scale (`--aslcitizen-norm valid` = real frames only, experimental) |
| Face | used (468 pts) | ignored |
| Trained on | Kaggle ASL Signs / PopSign | ASL Citizen (MediaPipe *Holistic*, legacy API); we feed Tasks-API Pose-lite + Hands |

## Known limits

- ST-GCN: 59.52 % top-1, 82.68 % top-5 on unseen signers (published); per-user 43–75 %.
  A WLASL-2000 model scored 8.49 % on overlapping ASL Citizen signs — expect a drop on a new camera and signer.
- Subtle handshape differences collide; overlapping hands drop or swap a hand; `--no-face` loses non-manual markers.
- Training landmarks came from legacy MediaPipe Holistic; the Tasks-API models used here
  are different networks, a distribution shift nobody has measured for this pipeline.

## Provenance

PopSign model and admission logic: [`nlevites/signchat`](https://github.com/nlevites/signchat) (MIT).
ASL Citizen ST-GCN: [`microsoft/ASL-citizen-code`](https://github.com/microsoft/ASL-citizen-code) (MIT), vendored in `vendor/`.
