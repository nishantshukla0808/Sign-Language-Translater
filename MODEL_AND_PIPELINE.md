# signtalk — models and pipeline

Word-level, isolated ASL sign **lookup**: a camera sees you sign one word, the program prints (or speaks) the most likely word(s). It is not a translator, not a fingerspelling (A–Z) recogniser, and not a continuous-sentence recogniser. No training or fine-tuning is done anywhere here; both models are pretrained.

## 1. The two classifiers

Both read **landmarks** (points on your hands and body), never raw pixels. MediaPipe turns each camera frame into landmarks; the classifier sees a short sequence of them.

| | PopSign-250 | ASL Citizen ST-GCN |
|---|---|---|
| File | `models/asl-signs.onnx` (22 MB) | `models/aslcitizen_stgcn.onnx` (15 MB) |
| Flag | `--model popsign` (default) | `--model aslcitizen` |
| Vocabulary | **250** words, built for toddlers and parents (apple, mom, nap, hello, please, thankyou, ...). No `how`, `you`, `help`, `name`. | **2,731** signs from ASL-LEX (HELLO, HOW1/HOW2, YOU, YES, NO, PLEASE, THANKYOU, WATER, ...). Labels in `models/aslcitizen_gloss.json`. |
| Input | `(T, 543, 3)`: x, y, z for 543 points `[Face 468, Left hand 21, Pose 33, Right hand 21]`, any length T | `(1, 2, T, 27)`: x, y only, 27 graph nodes (7 pose + 10 per hand), the adapter always produces T = 128 |
| Missing points | **NaN** (the model normalises finite values itself) | **0.0**, set *before* shoulder-normalisation (matches how it was trained) |
| Normalisation | inside the model | in `aslcitizen.preprocess`, a line-for-line mirror of the dataset code (shoulder centre and scale taken over the zero-padded 128-frame array, as in training) |
| Face used | yes | no |
| Output | 250 probabilities | 2,731 probabilities |
| Published accuracy | not measured here | 59.52 % top-1 / 82.68 % top-5 on unseen signers (Microsoft, published); per-user 43–75 % |
| Speed on this Pi 5 | 8–19 ms per inference (T = 16–64) | ~170–190 ms per sign (measured live) |
| Origin | `nlevites/signchat` (MIT) | `microsoft/ASL-citizen-code` (MIT), vendored in `vendor/`, exported to ONNX by `export_stgcn.py` |

**Why the ASL Citizen model is the one for phrases like "how are you":** PopSign has no `how` or `you`. ASL Citizen has `HOW1`/`HOW2` and `YOU`. There is no sign for "are" in ASL; "how are you" is signed HOW then YOU.

### Setup notes for the ST-GCN
- The `.onnx` was exported once on this Pi in a separate environment (`~/venv-export`, with PyTorch). PyTorch is **not** needed at runtime (`.venv` has only onnxruntime).
- Export check: the float64 graph matches the original PyTorch model to ~1e-9. The deployed float32 file differs by up to ~1e-4 on random logits of size ~200 (float32 rounding) and always gives the same top answer. It does not meet a strict 1e-4 criterion.
- The 2,731 labels were rebuilt from the ASL Citizen CSVs (`splits/train|val|test.csv`, column `Gloss`): the sorted unique glosses, which count exactly 2,731 and match the model's outputs. The CSVs were read out of Microsoft's official zip with HTTP range requests (the 46 GB file was not downloaded).
- Training landmarks came from the legacy MediaPipe *Holistic*; this app feeds landmarks from the newer MediaPipe *Tasks* Hand and Pose models. That is a distribution shift nobody has measured for this pipeline.

## 2. Pipeline

```
IP-Webcam phone / USB cam / video file
        |
 capture thread ──▶ frame queue (4) ──▶ inference thread ──▶ audio queue (8) ──▶ audio thread
 never blocks        drop-oldest        MediaPipe → gate →       drop-newest        plays WAVs
                                        model → admission →                         (--no-audio: off)
                                        phrase buffer → console
```

1. **Capture** (`realtime.py`, `Capture`). Reads frames. For a network stream (`--url`, default is the IP Webcam phone at `192.168.29.233:8080/video`) it can rotate the image (`--rotate 0/90/180/270`) and shrinks large frames while keeping the aspect ratio. A live source drops the *oldest* frame when the queue is full, so latency stays low; a video file waits for inference instead.
2. **Tracking** (`mpdetect.py`, `landmarks.py`). MediaPipe HandLandmarker (up to two hands), PoseLandmarker-lite, and optionally FaceLandmarker (`--no-face` skips it). The results are assembled into one `(543, 3)` frame, NaN where a part was not found.
3. **Motion gate** (`gate.py`). Decides when a sign starts and ends. It measures wrist speed in shoulder-widths per frame, which is independent of distance from the camera but not of frame rate. A state machine goes idle → recording when speed stays above a start threshold for a few frames, and back when it stays below a stop threshold. Defaults: start 0.020, stop 0.008, 3 frames to start, 8 to stop, minimum 10 frames, maximum 90 frames per sign. `--calibrate` replaces these with thresholds measured from your resting jitter, saved in `calibration.json`. The gate needs both shoulders in view.
4. **Classifier** (`classifier.py`, `aslcitizen.py`, `engine.py`). The recorded window goes to the chosen model and returns ranked probabilities. A dummy inference at startup avoids a slow first sign.
5. **Admission** (`classifier.py`, `TokenAdmitter`). The model always returns *some* word, so it is filtered: below the **reject floor** (`--reject-threshold`, default 0.10) nothing is emitted; a top-1 score of at least 0.30 is committed directly (`confident`); a score between 0.10 and 0.30 with a credible runner-up commits as `band`. The defaults were tuned for the 250-way model; the 2,731-way model's probabilities are spread more thinly and the floor has not been calibrated for it.
6. **Phrase buffer** (`phrase.py`). Committed words collect and are output together after 1.5 s of stillness (`--phrase-gap`), or at once with `--speak-immediately`. Reordering ASL gloss order into English order is a placeholder that does nothing yet.
7. **Output.** Each segment prints its top-3 with scores, e.g. `[ 90f] HELLO:0.62 ...  -> HELLO (confident)  [172 ms]`, then `>> SAY: <words>`. With audio on, pre-rendered WAVs (`wavs/`, made by `scripts/render_wavs.sh`) are played; audio is currently off (`--no-audio`) because the Pi has only HDMI audio and no `espeak-ng`.
8. **Logging** (`session_log.py`). Each segment is saved under `sessions/<timestamp>/` as `seg_NNNN.npy` (the raw landmark window), a line in `index.jsonl` (top-5, frames, velocity trace, model) and `meta.json`. `--no-log` disables it. A saved window can be replayed through any model to separate tracking failures from model failures.

### Run it
```bash
cd ~/nishu_hand_skeleton/signtalk
.venv/bin/python realtime.py --model aslcitizen --rotate 270 --no-face --no-audio --threads 3 --fps 15
```
`--rotate 270` is what the phone stream currently needs; change it if the phone orientation changes.

### Offline accuracy check
```bash
.venv/bin/python eval_clips.py --model aslcitizen
```
Reads `eval_clips/*.mp4` and `eval_clips/labels.csv`, scores each clip twice (whole clip as one window, and through the live gate) and reports tracker hand-detection rates.

## 3. What has and has not been measured on this Pi 5

Measured:
- MediaPipe runs on aarch64; tracking about 62–90 ms per frame at 640x480 with `--no-face`; peak memory under 400 MB; no throttling.
- Pipeline tests pass (`test_pipeline.py`, including the ASL Citizen ONNX path).
- On 16 real ASL Citizen **training** clips (HELLO, YES, NO, YOU, THANKYOU, PLEASE, WATER, HOW1; two signers each) the model scored 15/16 on whole clips and 12/16 through the live gate. These are training clips, so this proves the wiring, not real-world accuracy. Hands were detected in only about 43 % of frames, even there.

Not working yet (live phone test):
- Live signing was not recognised. Every live window hit the 90-frame maximum, so the gate never saw a sign end. Causes found: the stream was portrait/upside-down at first (now fixed with `--rotate`), hands were out of frame, and wrist estimates from the pose model jump when hands are not detected, which fools the gate. Calibration produced unusable thresholds (resting jitter 0.58 against a normal ~0.02).

Suggested next steps:
1. Add a push-to-record mode (press Enter, sign, press Enter) to test the model on your signing without the gate.
2. Make the gate follow detected hands rather than pose-wrist estimates, then calibrate.
3. Frame the camera so head, both shoulders and both hands are visible at rest.

## 4. Known limits
- A ~60 %-accurate model will confidently say a wrong word about 2 times in 5; always read the top-3.
- Subtle handshape differences collide; overlapping hands drop or swap a hand.
- `--no-face` loses non-manual markers (eyebrows, mouth) that some signs use.
- Sample clips in `sign_refs/` are from the ASL Citizen dataset (research use; contact Microsoft for commercial use).
