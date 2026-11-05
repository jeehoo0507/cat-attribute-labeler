from pathlib import Path

import json
import tarfile
import io

from PIL import Image
import pytest

from cat_labeler.common import atomic_json, exclusive_lock
from cat_labeler.export import export_bundle
from cat_labeler.label import bounded_image, face_crop, generate_stage
from cat_labeler.prepare import inspect_image, oxford_catalog
from cat_labeler.schema import Box, FaceLabels, FullLabels, merge_labels, parse_response


def observation(value="unknown"):
    return {"value": value, "evidence": "Not enough detail"}


def full_labels():
    fields = [
        "target_visible",
        "tucked_front_paws",
        "curled_body",
        "forelegs_extended",
        "paw_pads",
        "glasses",
        "collar",
        "scarf",
        "headwear",
        "leash_visible",
        "posture",
    ]
    return {
        **{key: observation() for key in fields},
        "face_box": None,
        "clothes": {
            "presence": observation(),
            "type_raw": None,
            "colors": [],
            "patterns": [],
        },
        "scene": [],
        "relations": [],
        "interactions": [],
    }


def face_labels():
    return {
        key: observation()
        for key in [
            "target_visible",
            "left_eye",
            "right_eye",
            "mouth",
            "tongue",
            "teeth",
            "glasses",
            "collar",
        ]
    }


def test_schema_rejects_guess_values_and_invalid_coordinates():
    parse_response(json.dumps(full_labels()), FullLabels)
    bad = face_labels()
    bad["tongue"]["value"] = "probably_out"
    with pytest.raises(ValueError):
        parse_response(json.dumps(bad), FaceLabels)
    with pytest.raises(ValueError):
        Box(xyxy=[900, 0, 100, 10], evidence="bad")


def test_conflicting_passes_require_review():
    full, face = full_labels(), face_labels()
    full["glasses"] = observation("present")
    face["glasses"] = observation("absent")
    full["collar"] = observation("present")
    result, conflicts = merge_labels(full, face)
    assert result["glasses"]["value"] == "unknown"
    assert conflicts == ["glasses"]
    assert result["collar"]["value"] == "present"


def test_crop_preserves_source_information_and_geometry():
    image = Image.new("RGB", (200, 100))
    full = full_labels()
    full["face_box"] = {"xyxy": [0, 0, 500, 500], "evidence": "test"}
    crop, geometry = face_crop(image, {"head_box_xyxy": None}, full)
    assert crop.width <= image.width and crop.height <= image.height
    assert geometry["origin"] == "vlm_face_box_unverified"
    assert bounded_image(crop, 100000).size == crop.size
    assert face_crop(image, {}, full_labels())[0] is None


def test_invalid_json_retry_and_stage_resume(tmp_path):
    class Backend:
        def __init__(self):
            self.calls = 0

        def generate(self, images, instruction):
            self.calls += 1
            return "invalid" if self.calls == 1 else json.dumps(face_labels())

    backend = Backend()
    args = (
        backend,
        "face",
        [],
        {"target_box": [0, 0, 1000, 1000]},
        tmp_path,
        {"json_attempts": 2},
    )
    generate_stage(*args)
    generate_stage(*args)
    assert backend.calls == 2
    assert (tmp_path / "face-attempt-1.txt").read_text() == "invalid"


def test_exact_bytes_dedup_key_and_bbox(tmp_path):
    image = tmp_path / "images/test.jpg"
    image.parent.mkdir()
    Image.new("RGB", (200, 100), "orange").save(image)
    row = {
        "id": "test",
        "source": "oxford_pets",
        "source_id": "test",
        "boxes_xywh": [[20, 10, 100, 50]],
        "head_box_xyxy": None,
    }
    inspected = inspect_image(tmp_path, row)
    assert inspected["target_box"] == [100, 100, 600, 600]
    assert len(inspected["sha256"]) == 64
    assert inspected["exif_orientation"] == 1


def test_export_drafts_and_integrity_guards(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "outputs"
    monkeypatch.setenv("OUTPUT_ROOT", str(output))
    manifest = {
        "images": [{"id": "test", "sha256": "abc", "path": "images/test.jpg"}],
        "download_errors": [],
        "unique": 1,
    }
    atomic_json(output / "manifest.json", manifest)
    atomic_json(output / "run.json", {"signature": "sig", "backend": "qwen_cuda"})
    with pytest.raises(RuntimeError, match="No completed"):
        export_bundle("test")
    item = output / "items/test"
    atomic_json(item / "full.json", full_labels())
    atomic_json(item / "face.json", face_labels())
    atomic_json(
        item / "result.json",
        {"image_sha256": "abc", "run_signature": "sig", "review_status": "unreviewed"},
    )
    export_bundle("test")
    summary = json.loads((tmp_path / "results/test/summary.json").read_text())
    assert summary["complete"] and not summary["training_ready"]
    assert summary["review_status_counts"] == {"unreviewed": 1}
    export_bundle("test")
    atomic_json(output / "run.json", {"signature": "changed"})
    with pytest.raises(ValueError, match="Stale"):
        export_bundle("test")


def test_lock_prevents_concurrent_writes(tmp_path):
    with exclusive_lock(tmp_path / "lock"):
        with pytest.raises(RuntimeError, match="Another process"):
            with exclusive_lock(tmp_path / "lock"):
                pass


def test_oxford_safe_selected_members(tmp_path, monkeypatch):
    from cat_labeler import prepare

    archives = {}
    for name in ["annotations", "images"]:
        archive = tmp_path / f"{name}.tar.gz"
        with tarfile.open(archive, "w:gz") as handle:
            data = (
                {"annotations/list.txt": b"Cat_1 1 1 1\ndog_1 2 2 1\n"}
                if name == "annotations"
                else {
                    "images/Cat_1.jpg": b"image",
                    "images/dog_1.jpg": b"dog",
                    "../../escape": b"unsafe",
                }
            )
            for path, content in data.items():
                entry = tarfile.TarInfo(path)
                entry.size = len(content)
                handle.addfile(entry, io.BytesIO(content))
        archives[name] = archive
    monkeypatch.setattr(
        prepare,
        "download",
        lambda url, target, **kwargs: archives[
            "annotations" if "annotations" in url else "images"
        ],
    )
    records = oxford_catalog(tmp_path / "data", 10, 42)
    assert len(records) == 1
    assert (tmp_path / "data/images/oxford_Cat_1.jpg").exists()
    assert not (tmp_path / "escape").exists()


def test_full_pipeline_resumes_face_pass_after_failure(tmp_path, monkeypatch):
    from cat_labeler import label
    from cat_labeler.common import sha256

    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "output"))
    image_path = tmp_path / "data/images/test.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (200, 100), "orange").save(image_path)
    record = {
        "id": "test",
        "sha256": sha256(image_path),
        "path": "images/test.jpg",
        "target_box": [0, 0, 1000, 1000],
        "multiple_cats": False,
    }
    atomic_json(tmp_path / "output/manifest.json", {"images": [record]})
    config = {
        "model": "fake-test-only",
        "revision": "test",
        "full_max_pixels": 50000,
        "crop_max_pixels": 10000,
        "json_attempts": 2,
    }
    atomic_json(tmp_path / "config.json", config)
    monkeypatch.setattr(
        label, "pin_model", lambda config, output: {"resolved_revision": "test"}
    )
    monkeypatch.setattr(label.importlib.metadata, "version", lambda name: "test")
    calls = []

    class Backend:
        fail = True

        def __init__(self, config, revision):
            pass

        def generate(self, images, instruction):
            stage = "full" if "face_box" in instruction else "face"
            calls.append(stage)
            if stage == "face" and Backend.fail:
                raise ValueError("Injected interruption")
            return json.dumps(full_labels() if stage == "full" else face_labels())

    monkeypatch.setattr(label, "QwenBackend", Backend)
    with pytest.raises(RuntimeError, match="failed images"):
        label.run(tmp_path / "config.json")
    assert (tmp_path / "output/items/test/full.json").exists()
    Backend.fail = False
    label.run(tmp_path / "config.json")
    label.run(tmp_path / "config.json")
    assert calls == ["full", "face", "face"]
    result = json.loads((tmp_path / "output/items/test/result.json").read_text())
    assert result["review_status"] == "unreviewed"
    assert not (tmp_path / "output/items/test/error.json").exists()


def test_publish_pushes_only_result_bundle(tmp_path):
    import os
    import shutil
    import subprocess
    import sys

    repository = Path(__file__).resolve().parents[1]
    remote, checkout = tmp_path / "remote.git", tmp_path / "checkout"

    def git(*args, cwd=None):
        return subprocess.check_output(
            ["git", *args], cwd=cwd, stderr=subprocess.STDOUT, text=True
        )

    git("init", "--bare", str(remote))
    git("init", "-b", "main", str(checkout))
    git("config", "user.name", "Test User", cwd=checkout)
    git("config", "user.email", "test@example.invalid", cwd=checkout)
    git("remote", "add", "origin", str(remote), cwd=checkout)
    shutil.copy(repository / "publish_results.sh", checkout)
    shutil.copytree(
        repository / "cat_labeler",
        checkout / "cat_labeler",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copy(repository / ".gitignore", checkout)
    git("add", ".", cwd=checkout)
    git("commit", "-m", "initial fixture", cwd=checkout)
    python = checkout / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    bundle = checkout / "results/pilot"
    atomic_json(bundle / "summary.json", {"labeled": 2, "complete": True})
    atomic_json(bundle / "run.json", {"signature": "test"})
    (checkout / "private-file.txt").write_text("do not stage")
    env = {**os.environ, "RUN_NAME": "pilot"}
    subprocess.run(
        ["bash", "./publish_results.sh"],
        cwd=checkout,
        env=env,
        check=True,
        capture_output=True,
    )
    files = git(
        "--git-dir", str(remote), "ls-tree", "-r", "--name-only", "main"
    ).splitlines()
    assert "results/pilot/summary.json" in files
    assert "private-file.txt" not in files
    # Running again handles no-change exports and re-pushes safely.
    subprocess.run(
        ["bash", "./publish_results.sh"],
        cwd=checkout,
        env=env,
        check=True,
        capture_output=True,
    )


@pytest.mark.parametrize("honor_range", [True, False])
def test_download_resume_and_range_ignored(tmp_path, monkeypatch, honor_range):
    import hashlib
    from cat_labeler import prepare

    content = b"known-archive-bytes" * 100
    target = tmp_path / "archive.tar.gz"
    target.with_suffix(".gz.part").write_bytes(content[:57])
    expected = hashlib.md5(content).hexdigest()
    requests_seen = []

    class Response:
        def __init__(self, headers):
            offset = int(headers.get("Range", "bytes=0-").split("=")[1].rstrip("-"))
            self.status_code = 206 if honor_range else 200
            self.body = content[offset:] if honor_range else content
            self.headers = {"Content-Length": str(len(self.body))}
            if honor_range:
                self.headers["Content-Range"] = (
                    f"bytes {offset}-{len(content) - 1}/{len(content)}"
                )

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            pass

        def iter_content(self, size):
            yield self.body

    def get(url, **kwargs):
        requests_seen.append(kwargs["headers"])
        return Response(kwargs["headers"])

    monkeypatch.setattr(prepare.requests, "get", get)
    prepare.download("https://example.invalid/archive", target, expected_md5=expected)
    assert requests_seen[0]["Range"] == "bytes=57-"
    assert target.read_bytes() == content
    assert not target.with_suffix(".gz.part").exists()


def test_download_checksum_failure_uses_next_source(tmp_path, monkeypatch):
    import hashlib
    from cat_labeler import prepare

    content = b"valid archive"
    urls = []

    class Response:
        status_code = 200

        def __init__(self, body):
            self.body = body
            self.headers = {"Content-Length": str(len(body))}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            pass

        def iter_content(self, size):
            yield self.body

    def get(url, **kwargs):
        urls.append(url)
        return Response(b"wrong bytes" if len(urls) == 1 else content)

    monkeypatch.setattr(prepare.requests, "get", get)
    monkeypatch.setattr(prepare.time, "sleep", lambda duration: None)
    target = tmp_path / "archive.tar.gz"
    prepare.download(
        "primary",
        target,
        mirrors=("mirror",),
        expected_md5=hashlib.md5(content).hexdigest(),
    )
    assert urls == ["primary", "mirror"]
    assert target.read_bytes() == content


def test_local_lock_root_supports_drive_outputs(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCK_ROOT", str(tmp_path / "local-locks"))
    target = tmp_path / "drive/output/.label.lock"
    with exclusive_lock(target):
        with pytest.raises(RuntimeError, match="Another process"):
            with exclusive_lock(target):
                pass
    assert not target.exists()
    assert len(list((tmp_path / "local-locks").glob("*.lock"))) == 1


def test_colab_notebook_cells_compile_and_preserve_32b():
    repository = Path(__file__).resolve().parents[1]
    notebook = json.loads(
        (repository / "notebooks/cat_attribute_labeler_32b.ipynb").read_text()
    )
    assert notebook["nbformat"] == 4
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            compile("".join(cell["source"]), cell["id"], "exec")
    text = json.dumps(notebook)
    assert "PUBLISH_RESULTS" in text and "LOCK_ROOT" in text
    assert (
        json.loads((repository / "configs/pilot.json").read_text())["model"]
        == "Qwen/Qwen3-VL-32B-Instruct"
    )
