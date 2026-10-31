# Prompt for Claude Code on the Raspberry Pi 5 (4 GB) over SSH

Paste everything below the line into Claude Code running on the Pi.

---

You are testing the `signtalk` project (live ASL word-sign lookup -> spoken word) **locally on this Raspberry Pi 5 (4 GB RAM)**, over SSH. Nobody may be watching the screen: do not ask questions; where something is ambiguous take the most reasonable reading and write the decision down. The only hard stop is GATE 2 below.

## Ground rules
- **Never invent a result.** Every number you report must come from a command you ran; record the command and its output. If you could not measure something, write `NOT MEASURED` and why.
- Synthetic input proves wiring only. Label any such run "plumbing only, says nothing about accuracy".
- No training or fine-tuning. Do not install PyTorch on this Pi (4 GB; not needed at runtime).
- Scope: word-level isolated ASL signs.
- If a step fails: record the exact error, try the next-best alternative, continue. One failure must not stop the run.
- Do not push to GitHub or change files outside the repo folder and its `.venv`. Write results to `PI_RESULTS.md` in the repo (it is yours to create).

## 0. Get the code
```bash
git clone --branch ccr-07d9bf3e-o72joh https://github.com/Vedant28082005/nishu_hand_skeleton.git
cd nishu_hand_skeleton/signtalk
cat README.md REPORT.md | head -200      # read first; REPORT.md lists what is and is not already measured
```
(If the clone needs credentials, ask the owner's pre-provided access or fall back to `scp` of `signtalk_v2.tar.gz`; note which you used.)

## 1. GATE 2 — does MediaPipe install and run on this Pi?
```bash
bash scripts/pi_gate_check.sh 2>&1 | tee logs_pi_gate.txt
```
Record the full verdict. **If it prints `GATE 2 FAIL` on "mediapipe import" or "HandLandmarker+Pose run": STOP the project steps, write the exact errors and the options listed in README "If GATE 2 fails" into `PI_RESULTS.md`, and finish there.** A failure only on "aarch64"/audio/camera hints is informational: read it and decide. Also record: OS (`/etc/os-release`), `uname -m`, python version, `free -m`, `vcgencmd measure_temp`, `vcgencmd get_throttled`, `nproc`, whether cooling is active.

## 2. Install
```bash
bash scripts/pi_setup.sh            # native venv at .venv; NO Docker. Do not use --service yet.
.venv/bin/python test_pipeline.py 2>&1 | tee logs_pi_tests.txt
```
Expected: ends with `ALL CHECKS PASSED`. Note that section 16 (ASL Citizen) prints SKIP unless `models/aslcitizen_stgcn.onnx` is present. That file is **not** in the repo (it is built on a PyTorch machine with `python export_stgcn.py`). If the owner placed it in `models/` (check `ls -la models/`), the tests will exercise it; otherwise record "ASL Citizen model not present: not tested on Pi" and test PopSign only. Report any test that fails with its output.

## 3. Hardware checks (record what you see)
- Camera: `ls /dev/video*`; `v4l2-ctl --list-devices` if installed. A USB webcam appears as `/dev/video0`; a Pi camera module needs libcamera and will NOT work as cv2 index 0 (say so and stop the live steps if that is all there is).
- Audio: `aplay -l`. Pi 5 has **no 3.5 mm jack**; a USB audio adapter/Bluetooth/I2S DAC is needed. Test `aplay wavs/hello.wav` if such a WAV exists (`ls wavs | head`; the setup script renders them with espeak-ng if Piper is unavailable). If no output device exists, run everything with `--no-audio` and say so.
- Display: if the SSH session has no display, do not use `--show`.

## 4. Performance measurements (the main value of this run)
Use a real person/hand image or video if one exists on the Pi; otherwise use the repo's plumbing video generator ONLY if you can obtain a real photo, and label results "plumbing". Measure, each with its command and raw output:
1. Tracking cost per frame at **640x480, 15 fps target, `--no-face`**, 3 threads: run `.venv/bin/python realtime.py --video <clip.mp4> --no-face --no-audio --no-log --threads 3 --width 640 --height 480 --fps 15` and copy the final line (`tracking ms p50/p95`, `classify ms p50/p95`, `capture-dropped`). If no clip is available: record NOT MEASURED for per-frame tracking and rely on the blank-frame number from `pi_gate_check.sh` (a lower bound).
2. Same with the face model on (omit `--no-face`).
3. Live camera, if a USB webcam is present: `.venv/bin/python realtime.py --width 640 --height 480 --fps 15 --threads 3 --no-face --no-audio` for 60 s; copy the printed `... N fps` lines and the final summary. State whether the achieved fps is >= 15 and how many frames capture dropped.
4. Peak memory: `/usr/bin/time -v .venv/bin/python realtime.py ... 2>&1 | grep -E "Maximum resident|Elapsed"` for the PopSign model (and ASL Citizen if the .onnx is present). 4 GB is tight: report RSS in MB and whether anything was OOM-killed (`dmesg | tail`).
5. Thermals: `vcgencmd measure_temp` and `vcgencmd get_throttled` before and immediately after the 60 s run. `0x0` = never throttled.
6. Model-only latency: the T=16/32/48/64 PopSign lines from `logs_pi_tests.txt`, and the `ASL Citizen T=...` lines if present.

## 5. Calibration (needs a person; skip gracefully if nobody is in front of the camera)
`.venv/bin/python realtime.py --calibrate --width 640 --height 480 --fps 15 --no-face` then show `calibration.json`. If the script warns the jitter is high, say so. If no one is present or no camera exists: NOT MEASURED. Do not calibrate on a video file and present it as a camera calibration.

## 6. Accuracy (only if the owner left clips)
If `eval_clips/` contains videos and `labels.csv`:
```bash
.venv/bin/python eval_clips.py 2>&1 | tee logs_pi_eval.txt
```
Report the summary exactly as printed (top-1/top-3 whole-clip and gate, hands-detected ratios, per-label, confusions) and `eval_results.csv`. Do not interpret results from fewer than ~30 clips as accuracy; repeat the printed confidence intervals. If there are no clips: `python eval_clips.py --selftest` (plumbing only) and write "accuracy NOT MEASURED: no clips".

## 7. Service (optional, last)
Only if everything above works: `bash scripts/pi_setup.sh --service`, `sudo systemctl start signtalk`, check `journalctl -u signtalk -n 50 --no-pager`, then `sudo systemctl stop signtalk` and report whether it started and stayed up for 60 s. Disable it again (`sudo systemctl disable signtalk`) unless the owner wants it kept.

## Deliverable: `PI_RESULTS.md`
In this order: (a) GATE 2 outcome (verdict at the very top, with the single most important finding); (b) environment (board, OS, python, RAM, cooling, camera, audio); (c) every measured number with its exact command and output, or NOT MEASURED with the reason; (d) decisions you made without asking; (e) what the owner must do next (e.g. "attach clips", "plug in a USB audio adapter", "copy aslcitizen_stgcn.onnx"). Lead with what failed, if anything. Finally print the file's contents.
