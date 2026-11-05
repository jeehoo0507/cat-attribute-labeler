#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv sync --locked --extra gpu --extra dev
.venv/bin/python -m pytest -q tests
printf '\nSetup complete. Run ./preflight.sh before downloading model weights.\n'
