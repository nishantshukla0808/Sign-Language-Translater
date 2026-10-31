# PI_RESULTS

## (a) Verdict
**GATE 2 PASS** — MediaPipe 1.2.0 installs, imports and runs HandLandmarker + Pose on this Pi 5 (aarch64, Python 3.13.5).
**Most important finding:** the software stack works, but **no real-sign test was possible**: this Pi has no USB webcam, no Pi camera module, no usable audio output, no real hand video and no labelled clips. Accuracy, calibration, live fps and audio are all NOT MEASURED. Tracking cost on a non-hand clip (plumbing only) is ~86 ms p50 at 640x480, which is below the 15 fps target (achieved ~11.6 fps, see below).

What failed / was blocked
- `scripts/pi_setup.sh` aborted at `sudo apt-get` (no passwordless sudo, no TTY). I ran its non-sudo steps manually (venv, requirements-pi.txt, fetch_models.sh). `espeak-ng` is NOT installed, so `wavs/` was not rendered.
- Face landmarker model was not fetched by setup; I downloaded it manually for the face-on run.
- Section 7 (systemd service) NOT RUN: needs sudo.

## (b) Environment
Raw: `logs_env.txt`, `logs_pi_gate.txt`
- Board: Raspberry Pi 5 Model B Rev 1.0, 4 GB (4049 MB total, 2047 MB swap), 4 cores
- OS: Debian GNU/Linux 13 (trixie), kernel 6.18.50+rpt-rpi-2712, aarch64, glibc 2.41
- Python 3.13.5; mediapipe 1.2.0; onnxruntime imports OK
- Cooling: pwm-fan present (`/sys/class/hwmon/hwmon2/fan1_input`)
- Camera: `/dev/video19..35` are only `rpi-hevc-dec` and `pispbe` (ISP) nodes, no capture device. `rpicam-hello --list-cameras` -> "No cameras available!"
- Audio: only `vc4-hdmi-0` and `vc4-hdmi-1` (HDMI). No USB audio / DAC. Not tested with `aplay`.
- Display: `DISPLAY` empty (SSH); `--show` not used.

## (c) Measurements
Test clip for every tracking number: `/home/vlmpi/pi_bench/llama.cpp/tools/mtmd/test-3.mp4` (720x358, 24 fps, 10 s, **animated cartoon faces, no person signing**). **Plumbing only, says nothing about accuracy**, and tracking cost with real hands in frame will likely be higher than reported here.

1. Gate check, blank 640x480 frame, hands+pose: median **57.3 ms** (`logs_pi_gate.txt`)
2. Tracking, `--no-face`, 3 threads, 640x480, 15 fps:
   `.venv/bin/python realtime.py --video <clip> --no-face --no-audio --no-log --threads 3 --width 640 --height 480 --fps 15`
   `frames=240  tracking ms p50/p95=86.4/113.5  classify ms p50/p95=13.1/14.8  capture-dropped=0`; throughput line `11.6 fps` (below the 15 fps target)
3. Same with face model on (omit `--no-face`):
   `frames=240  tracking ms p50/p95=92.2/132.3  classify ms p50/p95=14.3/14.6  capture-dropped=0`; `10.3 fps`
4. Peak RSS (python `resource` wrapper; `/usr/bin/time` is not installed): no-face **370 MB**, face **396 MB**. No OOM lines in `dmesg`. ASL Citizen model not measured (see below).
5. Thermals: before 49.4 C, `throttled=0x0`; after both runs (~50 s) 56.0 C, `throttled=0x0`. A 60 s live-camera run was not possible; these are video runs only.
6. Model-only latency, PopSign (`logs_pi_tests.txt`): T=16 8.26 ms, T=32 11.87 ms, T=48 15.23 ms, T=64 18.49 ms per inference. Warmup 12 ms.
7. Tests: `ALL CHECKS PASSED` (`logs_pi_tests.txt`). SKIP: ASL Citizen ONNX (`models/aslcitizen_stgcn.onnx` not present: **not tested on Pi**, PopSign only), vendor-class comparison (torch not installed, intentional).
8. Eval selftest (`logs_pi_eval.txt`): `SELFTEST OK -- plumbing only, says nothing about accuracy`.

NOT MEASURED, with reasons
- Live camera fps / 60 s run / dropped frames: no camera.
- Calibration (`calibration.json`): no camera and nobody in front of it; not done on a video file on purpose.
- Accuracy (top-1/top-3, hands-detected ratio on real signing): `eval_clips/` absent. **Accuracy NOT MEASURED: no clips.**
- Audio playback: no output device; `wavs/` not rendered.
- ASL Citizen RSS/latency: model file absent.
- systemd service: needs sudo.

## (d) Decisions made without asking
- Ran the non-sudo parts of `pi_setup.sh` by hand instead of stopping.
- Used a cartoon video found on disk for tracking cost, labelled plumbing, rather than generating synthetic hands.
- Replaced `/usr/bin/time -v` with a Python `resource.getrusage` wrapper for peak RSS.
- Skipped live/calibration/service steps as the prompt instructs when hardware or privilege is missing.

## (e) Owner must do next
1. Attach a USB webcam (a Pi camera module will not work as cv2 index 0), then re-run live fps, 60 s thermals and `--calibrate`.
2. Plug in a USB audio adapter / Bluetooth speaker, and run `sudo apt-get install -y espeak-ng` (or set `PIPER_VOICE`) then `bash scripts/render_wavs.sh`; otherwise keep `--no-audio`.
3. Provide `eval_clips/` with `labels.csv` (>= ~30 real signing clips) and run `eval_clips.py`.
4. Copy `models/aslcitizen_stgcn.onnx` (export on a PyTorch machine) if the second model is wanted.
5. Provide sudo access (or run `bash scripts/pi_setup.sh --service` yourself) if the service is wanted.
6. Frame rate: tracking p50 ~86 ms suggests ~11 fps ceiling on this clip at 640x480 with 3 threads. Consider a lower resolution or the `--fps 10` target after measuring on real input; `calibrate` must be done at the final fps.
