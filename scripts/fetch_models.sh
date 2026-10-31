#!/usr/bin/env bash
# Downloads the large model files (kept out of the tarball to stay under upload limits).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models
R=https://raw.githubusercontent.com/nlevites/signchat/main/apps/web/public/models/asl-signs
[ -f models/asl-signs.onnx ] || curl -fL -o models/asl-signs.onnx $R/asl-signs.onnx
[ -f models/sign_to_prediction_index_map.json ] || curl -fL -o models/sign_to_prediction_index_map.json $R/sign_to_prediction_index_map.json
if [ ! -f models/ASL_citizen_stgcn_weights.pt ]; then
  curl -fL -o /tmp/stgcn.zip https://github.com/microsoft/ASL-citizen-code/releases/download/checkpoints_v1/ASL_citizen_stgcn_weights.zip
  unzip -o /tmp/stgcn.zip -d models/ && find models -name "*.pt" ! -path "models/ASL_citizen_stgcn_weights.pt" -exec mv {} models/ASL_citizen_stgcn_weights.pt \;
fi
ls -la models
