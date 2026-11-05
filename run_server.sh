#!/usr/bin/env bash
# Full resumable pipeline. PUBLISH_RESULTS=1 pushes the portable draft bundle at completion.
set -euo pipefail
cd "$(dirname "$0")"
source scripts/env.sh
: "${RUN_NAME:?Set RUN_NAME, e.g. pilot32b_v1}"
./preflight.sh
./prepare_data.sh
./smoke_test.sh
.venv/bin/python -m cat_labeler.label --config "$CONFIG"
.venv/bin/python -m cat_labeler.export --name "$RUN_NAME"
if [[ "${PUBLISH_RESULTS:-1}" == 1 ]]; then
  ./publish_results.sh
fi
