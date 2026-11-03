#!/usr/bin/env bash
# Downloads the MediaPipe browser files into webapp/static/mp (not stored in git: ~45 MB).
# The version matches the Python mediapipe package the model was trained with (1.2.0) - older
# browser builds (0.10.x) track hands differently and gave wrong answers.
set -euo pipefail
cd "$(dirname "$0")/static/mp"
mkdir -p wasm
B=https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.0
curl -fL $B/vision_bundle.mjs -o vision_bundle.mjs
for f in vision_wasm_internal.js vision_wasm_internal.wasm vision_wasm_module_internal.js vision_wasm_module_internal.wasm \
         vision_wasm_nosimd_internal.js vision_wasm_nosimd_internal.wasm; do curl -fL $B/wasm/$f -o wasm/$f; done
cp ../../../models/hand_landmarker.task ../../../models/pose_landmarker_lite.task .
echo done
