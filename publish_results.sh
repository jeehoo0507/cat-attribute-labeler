#!/usr/bin/env bash
# Commit only the requested result bundle. Uses the clone's existing git authentication.
set -euo pipefail
cd "$(dirname "$0")"
: "${RUN_NAME:?Set RUN_NAME}"
[[ "$RUN_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$ ]] || { echo 'Invalid RUN_NAME' >&2; exit 1; }
result_dir="results/$RUN_NAME"
[[ -f "$result_dir/summary.json" ]] || { echo "Missing $result_dir/summary.json; export first" >&2; exit 1; }
[[ -z "$(git diff --cached --name-only)" ]] || { echo 'Index has staged changes; commit or unstage those first' >&2; exit 1; }
branch="$(git symbolic-ref --quiet --short HEAD)" || { echo 'Detached HEAD; check out a branch' >&2; exit 1; }
.venv/bin/python - "$result_dir" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
allowed = {'.json', '.jsonl'}
for path in root.rglob('*'):
    if path.is_symlink():
        raise SystemExit(f'Refusing symlink {path}')
    if path.is_file() and (path.suffix not in allowed or path.stat().st_size > 40 * 1024**2):
        raise SystemExit(f'Unsupported or oversized Git result file: {path}')
summary = json.loads((root/'summary.json').read_text())
print(f"Publishing {summary['labeled']} annotation drafts; complete={summary['complete']}")
PY
# Export locks prevent a concurrent export replacing this directory while Git reads it.
exec .venv/bin/python - "$result_dir" "$branch" <<'PY'
import subprocess
import sys
from cat_labeler.common import exclusive_lock
result_dir, branch = sys.argv[1:]
with exclusive_lock('results/.export.lock'):
    subprocess.run(['git', 'add', '--', result_dir], check=True)
    changed = subprocess.run(['git', 'diff', '--cached', '--quiet']).returncode
    if changed == 1:
        subprocess.run(['git', 'commit', '-m', f'data: add {result_dir.split("/")[-1]} label drafts'], check=True)
    elif changed != 0:
        raise SystemExit('git diff failed')
    # No force push or automatic merge on a shared branch. Retry after resolving a remote conflict.
    subprocess.run(['git', 'push', 'origin', f'HEAD:refs/heads/{branch}'], check=True)
PY
