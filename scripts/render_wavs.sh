#!/usr/bin/env bash
# Pre-render one WAV per vocabulary word into wavs/ so runtime TTS latency is zero.
#
#   PIPER_VOICE=/path/en_US-lessac-medium.onnx bash scripts/render_wavs.sh      # preferred
#   bash scripts/render_wavs.sh                                                 # espeak-ng fallback (robotic)
#
# Usage: render_wavs.sh [vocab.json] [out_dir]
#   vocab.json: {"word": index, ...} (PopSign map, default) or a JSON list of words.
#   For the ASL Citizen model pass models/aslcitizen_gloss.json once real glosses exist.
#
# Piper voices are hosted on Hugging Face (rhasspy/piper-voices); download one
# .onnx + .onnx.json on a machine that can reach it. `pip install piper-tts`.
set -euo pipefail
cd "$(dirname "$0")/.."
VOCAB="${1:-models/sign_to_prediction_index_map.json}"
OUT="${2:-wavs}"
mkdir -p "$OUT"

words() {
  python3 - "$VOCAB" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
for w in (d if isinstance(d, list) else (d.values() if all(isinstance(k, str) and k.isdigit() for k in d) else d.keys())):
    print(w)
PY
}

if [ -n "${PIPER_VOICE:-}" ] && python3 -c "import piper" 2>/dev/null; then
  ENGINE=piper
elif command -v espeak-ng >/dev/null; then
  ENGINE=espeak-ng
else
  echo "no TTS engine: set PIPER_VOICE (+ pip install piper-tts) or install espeak-ng" >&2; exit 1
fi
echo "engine: $ENGINE  vocab: $VOCAB  out: $OUT"

n=0
while IFS= read -r w; do
  [ -f "$OUT/$w.wav" ] && continue
  if [ "$ENGINE" = piper ]; then
    echo "$w" | python3 -m piper -m "$PIPER_VOICE" -f "$OUT/$w.wav" >/dev/null 2>&1
  else
    espeak-ng -v en-us -s 150 -w "$OUT/$w.wav" "$w"
  fi
  n=$((n+1))
done < <(words)
echo "rendered $n new files; $(ls "$OUT"/*.wav | wc -l) total in $OUT/"
