# signtalk v2 — overnight report

## The one thing to know

**No recognition accuracy has been measured, for either model.** There were no real sign clips in the session, and I did not substitute anything. Everything that runs end-to-end was run on a synthetic or artificially-shifted still photo, which proves wiring only. The offline harness (`eval_clips.py`) is finished and tested; the moment you give it clips plus a `labels.csv`, it produces the numbers. That is step 1 of "What you must do next" (section e).

Second most important: **MediaPipe works in this sandbox (GATE 1 passed)**, so the whole pipeline ran with real landmark detection. The ASL Citizen model exported to ONNX, but the strict float32 parity criterion (< 1e-4) was **not met** (2.1e-4 worst). Details in (c).

---

**Follow-up search for real clips (after the run above):** reachable hosts are only PyPI, npm, `raw.githubusercontent.com`, `storage.googleapis.com` (MediaPipe bundles/assets) and plain GitHub release assets. Unreachable (HTTP 000/403): Hugging Face, Kaggle, Zenodo, OSF, Mendeley, DagsHub, YouTube, Wikimedia, SignASL, HandSpeak, the GitHub API. npm/PyPI searches found only downloader packages (e.g. `sign-language-datasets`) that point at those blocked hosts, and the signchat repo's README lists no sample clips. So no labelled sign videos could be obtained legitimately, and accuracy remains NOT MEASURED. I did not use synthetic or self-labelled clips as a substitute.

## (a) GATE 1 — can MediaPipe run? **YES**

- This session's network differed from the earlier one: `storage.googleapis.com` was reachable, so the official bundles downloaded directly (`hand_landmarker.task` 7.8 MB, `pose_landmarker_lite.task` 5.8 MB, `face_landmarker.task` 3.8 MB). No assembling needed.
- `pip install mediapipe` gave **1.2.0** (not the 0.10.32 from the earlier session). It needed `libegl1` (`apt-get install libegl1 libgles2`) or the shared library fails to load.
- Verified by loading with `create_from_options` and running on real photos (from `storage.googleapis.com/mediapipe-assets`), `logs/gate1_mediapipe.txt`:
  `woman_hands`: 2 hands × 21 landmarks, pose 33, face 478. `male_full_height_hands`: 2×21, pose 33. `thumb_up`, `pointing_up`: 1×21.
- Handedness check (`logs/handedness_check.txt`, `scripts/handedness_check.py`): on 2 unmirrored photos the "Left"/"Right" hand labels matched the anatomical pose wrists (15/16) in all 4 hands. This supports v1's "no flip needed" claim, but n = 2 images only.
- Only tested on mediapipe 1.2.0. 0.10.32 was not tested.

## (b) What was built, and what was not

Built and tested (all under `signtalk/`):

| # | Item | Status |
|---|---|---|
| 0 | Fetch models + `test_pipeline.py` | Done. Pristine tarball code printed `ALL CHECKS PASSED` (`logs/step0_pristine_test.txt`). |
| 1 | `eval_clips.py` (whole-clip + gate-replay scoring, CSV, summary, Wilson CIs, `--selftest`, clear usage on empty) | Done. Shared code factored into `mpdetect.py`. Run on a one-clip scratch folder to prove the non-selftest path and CSV parsing; **that result is circular (I labelled the clip with the model's own output) and is not reported as accuracy.** |
| 2 | `--show` overlay (hands, pose, gate state+velocity, top-3, fps, which hands present) | Drawing verified on a real photo (rendered PNG viewed). **The live `cv2.imshow` loop was not run** (no display here). |
| 3 | ST-GCN → ONNX | Done (`models/aslcitizen_stgcn.onnx`, 15.2 MB, dynamic batch+time). PyTorch 2.15 installed fine from default PyPI. Parity: see (c). |
| 4 | `aslcitizen.py` adapter, `--model {popsign,aslcitizen}` on both CLIs | Done. Preprocessing matches the **actual vendor `ASLCitizen.__getitem__`** (run in torch) to ≤ 5e-7. Hand-order test fails if left/right swap. Labels are placeholders `sign_<index>` (see d). |
| 5 | Phrase buffer + `--speak-immediately` + reorder seam | Done, unit-tested with a fake clock. README explains the tradeoff. |
| 6 | Reject floor + top-3 | Done (`--reject-threshold`, default 0.10). |
| 7 | Session logging | Done (`sessions/<ts>/` with `.npy`, `index.jsonl` incl. top-5 / velocity trace / timestamp, `meta.json`). |
| 8 | `--calibrate`, startup warmup | Done. Calibration ran through the real code path on the plumbing video; **never run with a real still person** — the numbers it produced there are meaningless. |
| 9 | Three threads, bounded queues | Done; queue/drop behaviour tested with synthetic input; full capture→inference→audio run on a video file. |
| 10 | Pi 5 prep (`pi_gate_check.sh`, `pi_setup.sh`, `signtalk.service`, README) | Written. `pi_gate_check.sh` was run **on this x86 sandbox only** (correctly fails the aarch64 check, passes the rest). **Nothing was run on a Pi; no Pi claims are made.** `pi_setup.sh` and the systemd unit are untested. |
| 11 | TTS | Piper installed (`piper-tts 1.8.0`) but **no voice model is reachable** (Hugging Face blocked; no PyPI/npm package bundles one). `scripts/render_wavs.sh` uses Piper when `PIPER_VOICE` is set, else **espeak-ng** (robotic). I rendered 250 espeak-ng WAVs into `wavs/` locally as a stand-in; they are not in the tarball. |

Not built (by instruction or impossibility): no reordering; no training/fine-tuning; no real gloss names; no Pi measurements; no real-clip accuracy; `signtalk_flow.svg` / `make_flowgraph.py` were not updated and are now stale.

## (c) Every measured number, with its command

All run in this sandbox (x86_64, 4 vCPU, Python 3.13, mediapipe 1.2.0, onnxruntime 1.29). Logs are in `logs/`. Timings are single-run, noisy, and **laptop-class numbers that say nothing about a Pi**.

**Tests**
- `python test_pipeline.py` → 90 PASS, 0 FAIL, `ALL CHECKS PASSED` (`logs/test_pipeline_v2.txt`). All synthetic: **plumbing only, says nothing about accuracy.**
- `python eval_clips.py --selftest` and `--selftest --model aslcitizen` → `SELFTEST OK` (accuracy lines deliberately suppressed). `python eval_clips.py` with no clips → usage message, exit code 2 (`logs/eval_no_clips_usage.txt`).

**ONNX parity, ASL Citizen ST-GCN** (`python export_stgcn.py`, `logs/export_stgcn.txt`)
- Checkpoint: 172 tensors, 2731 classes, 3.79 M params, stored dtype float32, `A`=(3,27,27), `data_bn`=(54,) — matches your spec.
- **float32 deployed ONNX vs the original torch forward: worst max abs diff 2.136e-04 → does NOT meet the < 1e-4 criterion.** Per-case 1.37e-4 to 2.14e-4. Logits on random N(0,1) input reach |~250|, so this is ~6–9e-7 *relative*; argmax matched in every case.
- Evidence it is rounding, not a wrong graph: (i) the same graph exported in float64 and evaluated with onnx's `ReferenceEvaluator` (ORT's CPU provider has no float64 Conv) gives worst diff **1.67e-09**; (ii) in a separate ad-hoc check torch-float32 vs torch-float64 itself differed by 2.4e-5–7.1e-5, and disabling ORT graph optimisation did not change the ONNX error. (That ad-hoc check is not saved as a log; the output is in the session transcript only.) I did not change the threshold or the test inputs to make it pass.
- One deliberate graph change: the encoder's `F.avg_pool2d(x, x.size()[2:])` was replaced by `x.mean((2,3))` (identical maths; the original bakes T into the graph). Parity is measured against the *unmodified* original forward.
- On `adapter-preprocessed` windows (from the ad-hoc check) the float32 diffs were 8.4e-5 – 9.9e-5; end-to-end adapter+ONNX vs adapter+torch softmax max diff 2.7e-6 (`test_pipeline.py` §16).

**Latency** (`logs/test_pipeline_v2.txt`; threads=2; one run each, ±30 % run-to-run)
- PopSign ONNX: 4.7 / 6.2 / 7.7 / 9.6 ms at T = 16 / 32 / 48 / 64. Warm-up 8 ms.
- ASL Citizen (adapter + ONNX): 78–124 ms across T = 16…200 (always padded/downsampled to 128 frames, so roughly flat). Warm-up ~109 ms. Ad-hoc split: preprocessing 0.48 ms; ONNX-only 101 / 92 / 85 ms at 1 / 2 / 4 threads.
- MediaPipe tracking in the live app on this CPU (`logs/e2e_plumbing_*.txt`, 75 frames of a 306×480 video): hands+pose only ≈ 38–46 ms/frame p50, with face ≈ 43 ms (one run each). The Pi will differ; **NOT MEASURED on a Pi.**

**Observations from plumbing runs (single synthetic input; not accuracy)**
- On the same shifted still photo, PopSign's top-1 was 0.13 and ASL Citizen's was **0.91** (`logs/e2e_plumbing_*.txt`). So a probability floor alone will not reject non-signs, especially on the 2,731-way model. One input only; but treat the default 0.10 floor as uncalibrated and decide it from real clips.

**Accuracy (either model), real-clip gate-vs-whole comparison, hands-detected ratios on real signing, Pi performance: NOT MEASURED** — no clips, no Pi.

## (d) Decisions I made without asking

1. **ASL Citizen labels:** the real 2,731-gloss index list was not obtainable (searched; Microsoft download is blocked and no raw GitHub source found; the paper says glosses come from ASL-LEX, a different list). Used `sign_<index>` placeholders via a swappable `models/aslcitizen_gloss.json`, and the program prints the limitation at startup. `aslcitizen_targeted.json` was not used.
2. **Missing landmarks for ASL Citizen = 0.0, not NaN.** The training extractor (`ST-GCN/pose.py`) pre-fills `np.zeros`, so absent hands/pose were zeros, and normalisation was applied after. I mirrored that. PopSign still gets NaN.
3. **Normalisation quirk mirrored exactly:** the dataset code averages shoulder centre/distance over the *zero-padded* 128-frame array, so a short clip gets its centre shrunk and scale inflated by 128/T (your spec said "over the clip"). Default `--aslcitizen-norm train` reproduces training; `valid` (real frames only) is available as an experiment. I could not tell which works better live without clips.
4. **Admission bug fix:** v1's "stable" rule needed the same label on two consecutive predictions, but the live app makes one prediction per gate segment — so a confident word would only commit if the signer repeated it. Added `offer_segment()` (confident ≥ 0.30 commits directly as `via=confident`); the tick-based `offer()` is unchanged and still tested. Added `--top1-threshold/--top2-threshold`.
5. **CLI change:** `--model` is now the model family; the path moved to `--model-path`.
6. **Gate evaluation:** `MotionGate.flush()` added so a clip that ends mid-sign still yields a window; the gate-segmented prediction is the longest window, zero windows counts wrong, plus an "any window" metric. Fresh detectors are built per clip (MediaPipe VIDEO mode keeps tracking state).
7. **Phrase flush** counts the gate being mid-sign as activity, so a long sign does not split a phrase.
8. **Calibration:** `start = max(0.006, 1.25 × p99 resting velocity)`, `stop = 0.4 × start`, warning above 0.05; fps stored and compared on load. These constants are my choices, unvalidated.
9. **Parity:** reported the float32 miss honestly and added the float64 structural check rather than loosening the criterion (see c).
10. **Pi:** a PyPI JSON query shows `manylinux_2_28_aarch64` wheels for mediapipe ≥ 1.0 — this contradicts the brief's "not a clean pip install on aarch64". I kept the gate script as the arbiter since I cannot test on ARM. Native venv + systemd, not Docker; if Docker later, build on the Pi.
11. Pinned nothing; `requirements.txt` notes the versions this was run with. Added `.gitignore` (models, sessions, wavs, clips) at repo root.
12. Resulting changes to one v1 default: eval defaults to face **on** (PopSign used face); `--no-face` is available.

## (e) What you must do next

1. **Attach N real clips + `labels.csv`** (`filename,label`, PopSign-250 glosses) in `signtalk/eval_clips/`. Record unmirrored, shoulders + both hands in frame, one sign per clip (≥ 5 per word and ≥ 30 clips total, or the confidence intervals will be useless). Then: `python eval_clips.py` and `python eval_clips.py --model aslcitizen` (the latter needs a gloss JSON, or labels as `sign_<index>`). Read the whole-clip vs gate gap and the hands-detected ratios first.
2. **Run `bash scripts/pi_gate_check.sh` on the Pi 5** (64-bit OS) before anything else. If FAIL, see README "If GATE 2 fails".
3. Run `python realtime.py --calibrate --no-face` on each camera (and again on the Pi), then try `--show` on your laptop — that code path (window display) is the one thing I could not execute.
4. Decide `--speak-immediately` vs phrase buffer (README table), and set `--reject-threshold` from real logs.
5. If you want a better voice: `PIPER_VOICE=<en_US-lessac-medium.onnx> bash scripts/render_wavs.sh` on a machine that can fetch the voice from Hugging Face (delete `wavs/` first).
6. If you have the ASL Citizen CSVs: sorted unique gloss column → `models/aslcitizen_gloss.json` to get real names.
