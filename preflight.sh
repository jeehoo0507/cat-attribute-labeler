#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source scripts/env.sh
exec .venv/bin/python -m cat_labeler.preflight --config "$CONFIG"
