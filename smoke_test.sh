#!/usr/bin/env bash
# A real two-image GPU smoke test. Results remain available to the full run.
set -euo pipefail
cd "$(dirname "$0")"
source scripts/env.sh
.venv/bin/python -m cat_labeler.label --config "$CONFIG" --limit 2
.venv/bin/python - <<'PY'
from cat_labeler.common import atomic_json, paths, read_json
_, output = paths()
manifest = read_json(output / 'manifest.json')
assert len(manifest['images']) >= 2, 'Smoke requires at least two downloaded images'
results = [read_json(output / 'items' / row['id'] / 'result.json') for row in manifest['images'][:2]]
assert all(row['review_status'] == 'unreviewed' for row in results)
atomic_json(output / 'smoke.json', {'passed': True, 'images': [row['image_id'] for row in results],
                                  'note': 'GPU/model/schema plumbing passed; annotation accuracy requires human review'})
print('GPU smoke passed. Review the two outputs before large-scale annotation.')
PY
