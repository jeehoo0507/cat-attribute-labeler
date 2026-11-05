"""Export portable annotation drafts and provenance, without images or weights."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import shutil
import tempfile

from .common import atomic_json, exclusive_lock, fingerprint, paths, read_json
from .schema import FaceLabels, FullLabels


def write_shards(directory, prefix, records):
    for start in range(0, len(records), 200):
        path = directory / f"{prefix}-{start // 200:05d}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in records[start : start + 200]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def export_bundle(name, allow_partial=False):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name):
        raise ValueError(
            "Run name must be 1..64 ASCII letters, digits, underscore or dash"
        )
    root, output = paths()
    manifest = read_json(output / "manifest.json")
    run = read_json(output / "run.json")
    labels, raw, missing, counts = [], [], [], Counter()
    destination = Path("results") / name
    with (
        exclusive_lock(output / ".label.lock"),
        exclusive_lock(Path("results") / ".export.lock"),
    ):
        for image in manifest["images"]:
            item = output / "items" / image["id"]
            if not (item / "result.json").exists():
                missing.append(image["id"])
                continue
            result = read_json(item / "result.json")
            if (
                result["image_sha256"] != image["sha256"]
                or result["run_signature"] != run["signature"]
            ):
                raise ValueError(f"Stale result: {image['id']}")
            full = FullLabels.model_validate(read_json(item / "full.json")).model_dump()
            face = FaceLabels.model_validate(read_json(item / "face.json")).model_dump()
            labels.append(result)
            raw.append(
                {
                    "image_id": image["id"],
                    "full": full,
                    "face": face,
                    "responses": {
                        path.name: path.read_text()
                        for path in sorted(item.glob("*-attempt-*.txt"))
                    },
                }
            )
            counts[result["review_status"]] += 1
        if not labels:
            raise RuntimeError("No completed annotations to export")
        if (missing or manifest["download_errors"]) and not allow_partial:
            raise RuntimeError(
                "Run is incomplete. Retry first or explicitly export with --allow-partial."
            )
        summary = {
            "images": len(manifest["images"]),
            "labeled": len(labels),
            "missing": missing,
            "review_status_counts": dict(counts),
            "complete": not missing and not manifest["download_errors"],
            "training_ready": False,
            "note": "VLM annotation drafts. Human verification and duplicate/group split required before training.",
        }
        manifest_hash = fingerprint(manifest)
        if destination.exists():
            old_run = read_json(destination / "run.json")
            old_summary = read_json(destination / "summary.json")
            if (
                old_run["signature"] != run["signature"]
                or old_summary.get("manifest_hash") != manifest_hash
            ):
                raise RuntimeError(
                    "Export name already belongs to a different run. Use a different RUN_NAME."
                )
        summary["manifest_hash"] = manifest_hash
        destination.parent.mkdir(exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".export-", dir=destination.parent))
        try:
            write_shards(temporary, "labels", labels)
            write_shards(temporary, "observations", raw)
            write_shards(temporary, "images", manifest["images"])
            atomic_json(temporary / "summary.json", summary)
            atomic_json(temporary / "run.json", run)
            atomic_json(
                temporary / "dataset.json",
                {key: value for key, value in manifest.items() if key != "images"},
            )
            for filename in ["preflight.json", "model-pin.json", "smoke.json"]:
                if (output / filename).exists():
                    value = read_json(output / filename)
                    # Paths differ on the workstation; retain relative source image paths above.
                    if filename == "preflight.json":
                        value.pop("data_root", None)
                        value.pop("output_root", None)
                    atomic_json(temporary / filename, value)
            # Install under lock. Rename the old version to allow recovery on interruption.
            backup = destination.with_name(destination.name + ".previous")
            if backup.exists():
                raise RuntimeError(
                    f"Interrupted prior export: inspect {backup} before retrying"
                )
            if destination.exists():
                destination.rename(backup)
            try:
                temporary.rename(destination)
            except BaseException:
                if backup.exists():
                    backup.rename(destination)
                raise
            if backup.exists():
                shutil.rmtree(backup)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Portable bundle: {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    export_bundle(args.name, args.allow_partial)
