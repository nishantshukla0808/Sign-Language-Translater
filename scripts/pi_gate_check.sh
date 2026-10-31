#!/usr/bin/env bash
# GATE 2 -- run this FIRST on the Raspberry Pi 5 (64-bit Raspberry Pi OS).
#
#   bash scripts/pi_gate_check.sh            # creates ./.venv-gate, installs, probes
#   bash scripts/pi_gate_check.sh --keep     # reuse an existing .venv-gate
#
# Question it answers: does MediaPipe install AND run on this ARM64 Pi?
# It prints a PASS/FAIL line per check and a verdict. It changes nothing outside
# ./.venv-gate and ./models (downloads the .task bundles if missing and reachable).
#
# If the verdict is FAIL on "mediapipe import" or "HandLandmarker runs": the
# architecture changes (see README "If GATE 2 fails"). Do not build further on the Pi
# until this passes.
set -uo pipefail
cd "$(dirname "$0")/.."
VENV=.venv-gate
RES=()
rec() { RES+=("$1|$2|$3"); printf '  %-4s %-34s %s\n' "$1" "$2" "$3"; }

echo "== environment =="
ARCH=$(uname -m); echo "  uname -m: $ARCH   kernel: $(uname -r)"
[ -r /etc/os-release ] && . /etc/os-release && echo "  OS: ${PRETTY_NAME:-?}"
[ -r /proc/device-tree/model ] && echo "  board: $(tr -d '\0' </proc/device-tree/model)"
echo "  python: $(python3 --version 2>&1)   glibc: $(ldd --version 2>&1 | head -1 | awk '{print $NF}')"
echo "  RAM: $(awk '/MemTotal/{printf "%.1f GB", $2/1048576}' /proc/meminfo)"
echo
echo "== checks =="
[ "$ARCH" = aarch64 ] && rec PASS "64-bit ARM (aarch64)" "ok" \
  || rec FAIL "64-bit ARM (aarch64)" "got '$ARCH' -- need 64-bit Raspberry Pi OS (32-bit has no mediapipe wheels)"

# glibc >= 2.28 is what the manylinux_2_28 wheel needs
GL=$(ldd --version 2>&1 | head -1 | awk '{print $NF}')
if python3 -c "import sys; a=tuple(map(int,'$GL'.split('.')[:2])); sys.exit(0 if a>=(2,28) else 1)" 2>/dev/null; then
  rec PASS "glibc >= 2.28" "$GL"; else rec FAIL "glibc >= 2.28" "$GL (too old for manylinux_2_28 wheels)"; fi

# libEGL/GLES: the mediapipe 1.x shared library links them (headless Pi OS Lite may lack them)
if ldconfig -p 2>/dev/null | grep -q 'libEGL.so.1'; then rec PASS "libEGL.so.1 present" "ok"
else rec FAIL "libEGL.so.1 present" "missing -> sudo apt install -y libegl1 libgles2"; fi

if [ "${1:-}" != "--keep" ] || [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV" 2>/dev/null && rec PASS "python3 -m venv" "ok" \
    || { rec FAIL "python3 -m venv" "sudo apt install -y python3-venv"; }
fi
PY="$VENV/bin/python"
if [ -x "$PY" ]; then
  "$PY" -m pip install -q --upgrade pip >/dev/null 2>&1
  LOG=$(mktemp)
  if "$PY" -m pip install -q mediapipe onnxruntime numpy opencv-python-headless >"$LOG" 2>&1; then
    rec PASS "pip install mediapipe+deps" "$("$PY" -m pip show mediapipe 2>/dev/null | awk '/^Version/{print $2}')"
  else
    rec FAIL "pip install mediapipe+deps" "$(tail -2 "$LOG" | tr '\n' ' ')"
    echo "      -> no wheel for this platform. Options: community aarch64 wheels, or build from"
    echo "         source (Bazel; Google documents it; hours on a Pi). See README 'If GATE 2 fails'."
  fi

  OUT=$("$PY" - <<'PY' 2>&1
import sys
try:
    import mediapipe as mp
    print("OK", mp.__version__)
except Exception as e:
    print("ERR", type(e).__name__, e); sys.exit(1)
PY
)
  RC=$?; OUT=$(echo "$OUT" | grep -E '^(OK|ERR)' | tail -1)
  [ $RC -eq 0 ] && rec PASS "mediapipe import" "$OUT" || rec FAIL "mediapipe import" "${OUT:-import failed}"
  "$PY" -c "import onnxruntime as o; print(o.__version__)" >/dev/null 2>&1 \
    && rec PASS "onnxruntime import" "ok" || rec FAIL "onnxruntime import" "failed"

  mkdir -p models
  B=https://storage.googleapis.com/mediapipe-models
  fetch() { [ -f "models/$2" ] || curl -fsL -m 120 -o "models/$2" "$B/$1"; }
  fetch hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task hand_landmarker.task
  fetch pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task pose_landmarker_lite.task
  if [ -s models/hand_landmarker.task ] && [ -s models/pose_landmarker_lite.task ]; then
    rec PASS ".task bundles present" "models/"
    RUN=$("$PY" - <<'PY' 2>&1
import time, sys, numpy as np
import mediapipe as mp
from mediapipe.tasks import python as P
from mediapipe.tasks.python import vision as V
try:
    h = V.HandLandmarker.create_from_options(V.HandLandmarkerOptions(
        base_options=P.BaseOptions(model_asset_path="models/hand_landmarker.task"),
        running_mode=V.RunningMode.VIDEO, num_hands=2))
    p = V.PoseLandmarker.create_from_options(V.PoseLandmarkerOptions(
        base_options=P.BaseOptions(model_asset_path="models/pose_landmarker_lite.task"),
        running_mode=V.RunningMode.VIDEO))
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.full((480, 640, 3), 127, np.uint8))
    for i in range(3):                       # warm-up
        h.detect_for_video(img, i * 66); p.detect_for_video(img, i * 66)
    t = []
    for i in range(3, 33):
        t0 = time.perf_counter()
        h.detect_for_video(img, i * 66); p.detect_for_video(img, i * 66)
        t.append((time.perf_counter() - t0) * 1000)
    t.sort()
    print(f"OK hands+pose on a BLANK 640x480 frame: median {t[len(t)//2]:.1f} ms (blank frame: "
          f"a lower bound, real frames with hands cost more)")
    h.close(); p.close()
except Exception as e:
    print("ERR", type(e).__name__, e); sys.exit(1)
PY
)
    RC=$?; RUN=$(echo "$RUN" | grep -E '^(OK|ERR)' | tail -1)
    [ $RC -eq 0 ] && rec PASS "HandLandmarker+Pose run" "$RUN" || rec FAIL "HandLandmarker+Pose run" "${RUN:-did not run}"
  else
    rec FAIL ".task bundles present" "download failed (network?) -- copy them into models/ manually"
  fi
fi

echo
echo "== hardware notes (informational) =="
command -v vcgencmd >/dev/null && echo "  temp: $(vcgencmd measure_temp)  throttled: $(vcgencmd get_throttled)  (0x0 = never throttled)"
echo "  audio out devices (Pi 5 has NO 3.5 mm jack):"; (aplay -l 2>/dev/null | grep -E '^card' || echo "    none found -> plug a USB audio adapter / pair Bluetooth / fit an I2S DAC") | sed 's/^/    /'
ls /dev/video* >/dev/null 2>&1 && echo "  cameras: $(ls /dev/video* | tr '\n' ' ')" || echo "  cameras: no /dev/video* (USB webcam not seen; Pi camera modules need libcamera, not cv2 index 0)"

echo
FAILS=0; for r in "${RES[@]}"; do [ "${r%%|*}" = FAIL ] && FAILS=$((FAILS+1)); done
if [ "$FAILS" -eq 0 ]; then
  echo "VERDICT: GATE 2 PASS -- MediaPipe runs on this Pi. Next: re-run --calibrate here, then 'realtime.py --no-face'."
else
  echo "VERDICT: GATE 2 FAIL ($FAILS failing check(s) above). Do not proceed; see README 'If GATE 2 fails'."
fi
exit "$FAILS"
