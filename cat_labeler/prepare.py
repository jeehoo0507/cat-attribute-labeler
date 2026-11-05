"""Download only external cat images. No local pet-photo importer is provided."""

import argparse
import csv
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import io
import hashlib
import json
from pathlib import Path
import random
import tarfile
import time
import xml.etree.ElementTree as ET
import zipfile

from PIL import Image
import requests

from .common import atomic_json, exclusive_lock, fingerprint, paths, read_json, sha256

COCO_BUCKET = "https://s3.amazonaws.com/images.cocodataset.org/"
COCO_ARCHIVE = COCO_BUCKET + "annotations/annotations_trainval2017.zip"
OXFORD_ROOT = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/"
OXFORD_IMAGES_MIRROR = (
    "https://huggingface.co/datasets/Shiym/ViT-FineTune/resolve/"
    "1cc46adc2014ae0f8d67cc55a84d2022886b0364/oxford-iiit-pet/images.tar.gz"
)
OXFORD_IMAGES_MD5 = "5c4f3ee8e5d25df40f4fd59a7f44e54c"
OXFORD_ANNOTATIONS_MD5 = "95a8c909bbe2e81eed6a22bccdf3f68f"
OI_BOXES = {
    "train": "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv",
    "validation": "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
}
OI_METADATA = {
    "train": "https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv",
    "validation": "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv",
}
CAT_CLASS = "/m/01yrx"


def md5_matches(path, expected):
    if expected is None:
        return True
    digest = hashlib.md5(usedforsecurity=False)
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest() == expected


def download(url, target, *, mirrors=(), expected_md5=None):
    """Resume known-checksum archives, bound slow transfers, and log progress."""
    target = Path(target)
    if target.exists():
        if not md5_matches(target, expected_md5):
            raise IOError(
                f"Cached file failed checksum: {target}. Remove this corrupt file and retry."
            )
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    sources = [url, *mirrors]
    for attempt in range(max(3, len(sources))):
        source = sources[attempt % len(sources)]
        # Cross-source resumption requires a known archive checksum. Ordinary
        # images restart instead of appending bytes from an unverified object.
        offset = temporary.stat().st_size if expected_md5 and temporary.exists() else 0
        if offset and md5_matches(temporary, expected_md5):
            temporary.replace(target)
            return target
        headers = {"Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        started = last_report = time.monotonic()
        transferred = 0
        print(
            f"Downloading {target.name} (attempt {attempt + 1}, resume {offset / 1024**2:.1f} MiB)",
            flush=True,
        )
        try:
            with requests.get(
                source, headers=headers, stream=True, timeout=(20, 45)
            ) as response:
                if response.status_code == 416:
                    temporary.unlink(missing_ok=True)
                    raise IOError("Server rejected the partial range; restarting")
                response.raise_for_status()
                if response.status_code == 206:
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {offset}-"):
                        raise IOError(f"Unexpected range response: {content_range}")
                else:
                    # Some origins ignore Range. Replace instead of corrupting
                    # the archive by appending a full response to the prefix.
                    offset = 0
                expected = response.headers.get("Content-Length")
                expected = int(expected) if expected else None
                with temporary.open("ab" if offset else "wb") as handle:
                    for block in response.iter_content(8192):
                        if not block:
                            continue
                        handle.write(block)
                        transferred += len(block)
                        now = time.monotonic()
                        elapsed = now - started
                        if now - last_report >= 10:
                            handle.flush()
                            total = (
                                f"/{(offset + expected) / 1024**2:.1f}"
                                if expected
                                else ""
                            )
                            print(
                                f"  {target.name}: {(offset + transferred) / 1024**2:.1f}{total} MiB, "
                                f"{transferred / max(elapsed, 0.001) / 1024:.1f} KiB/s",
                                flush=True,
                            )
                            last_report = now
                        if elapsed > 60 and transferred / elapsed < 64 * 1024:
                            raise IOError(
                                "Transfer below 64 KiB/s for over 60 seconds; switching source"
                            )
                        if elapsed > 1800:
                            raise IOError(
                                "Transfer exceeded 30 minutes; retrying from partial archive"
                            )
                if expected is not None and transferred != expected:
                    raise IOError(
                        f"Incomplete transfer: expected {expected} bytes, got {transferred}"
                    )
            if not md5_matches(temporary, expected_md5):
                temporary.unlink(missing_ok=True)
                raise IOError("Archive checksum mismatch; discarding partial bytes")
            temporary.replace(target)
            atomic_json(
                target.with_name(target.name + ".download.json"),
                {
                    "url": source,
                    "bytes": target.stat().st_size,
                    "expected_md5": expected_md5,
                },
            )
            print(
                f"Downloaded {target.name}: {target.stat().st_size / 1024**2:.1f} MiB",
                flush=True,
            )
            return target
        except (requests.RequestException, IOError) as error:
            print(f"Download retry for {target.name}: {error}", flush=True)
            if not expected_md5:
                temporary.unlink(missing_ok=True)
            if attempt + 1 == max(3, len(sources)):
                raise
            time.sleep(attempt + 1)


def choose(records, count, seed):
    records = sorted(records, key=lambda row: row["id"])
    random.Random(seed).shuffle(records)
    return records[:count]


def coco_catalog(root, count, seed):
    archive = download(COCO_ARCHIVE, root / "downloads/annotations_trainval2017.zip")
    with zipfile.ZipFile(archive) as handle:
        instances = json.loads(handle.read("annotations/instances_train2017.json"))
        captions = json.loads(handle.read("annotations/captions_train2017.json"))
    boxes = defaultdict(list)
    for annotation in instances["annotations"]:
        if annotation["category_id"] == 17 and not annotation["iscrowd"]:
            boxes[annotation["image_id"]].append(annotation["bbox"])
    texts = defaultdict(list)
    for annotation in captions["annotations"]:
        texts[annotation["image_id"]].append(annotation["caption"])
    licenses = {item["id"]: item for item in instances["licenses"]}
    records = []
    for image in instances["images"]:
        if image["id"] not in boxes:
            continue
        records.append(
            {
                "id": f"coco_{image['id']:012d}",
                "source": "coco2017",
                "source_split": "train2017",
                "source_id": str(image["id"]),
                "download_url": COCO_BUCKET + "train2017/" + image["file_name"],
                "source_url": image.get("flickr_url"),
                "license": licenses[image["license"]],
                "original_captions": texts[image["id"]],
                "boxes_xywh": boxes[image["id"]],
                "head_box_xyxy": None,
            }
        )
    return choose(records, count, seed)


def oxford_catalog(root, count, seed):
    annotations = download(
        OXFORD_ROOT + "annotations.tar.gz",
        root / "downloads/oxford_annotations.tar.gz",
        mirrors=("https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz",),
        expected_md5=OXFORD_ANNOTATIONS_MD5,
    )
    records = []
    with tarfile.open(annotations) as handle:
        lines = handle.extractfile("annotations/list.txt").read().decode().splitlines()
        members = set(handle.getnames())
        for line in lines:
            if not line or line.startswith("#"):
                continue
            name, class_id, species, breed_id = line.split()
            if species != "1":
                continue
            head = None
            xml_name = f"annotations/xmls/{name}.xml"
            if xml_name in members:
                tree = ET.fromstring(handle.extractfile(xml_name).read())
                box = tree.find("object/bndbox")
                if box is not None:
                    head = [
                        int(box.findtext(key))
                        for key in ["xmin", "ymin", "xmax", "ymax"]
                    ]
            records.append(
                {
                    "id": f"oxford_{name}",
                    "source": "oxford_pets",
                    "source_split": "source_pool",
                    "source_id": name,
                    "download_url": OXFORD_ROOT + "images.tar.gz",
                    "source_url": "https://www.robots.ox.ac.uk/~vgg/data/pets/",
                    "license": {
                        "name": "CC BY-SA 4.0",
                        "url": "https://creativecommons.org/licenses/by-sa/4.0/",
                    },
                    "original_captions": [],
                    "breed": name.rsplit("_", 1)[0],
                    "boxes_xywh": [],
                    "head_box_xyxy": head,
                }
            )
    selected = choose(records, count, seed)
    archive = download(
        OXFORD_IMAGES_MIRROR,
        root / "downloads/oxford_images.tar.gz",
        mirrors=(
            "https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz",
            OXFORD_ROOT + "images.tar.gz",
        ),
        expected_md5=OXFORD_IMAGES_MD5,
    )
    wanted = {f"images/{r['source_id']}.jpg": r for r in selected}
    with tarfile.open(archive) as handle:
        for member in handle:
            if member.name not in wanted:
                continue
            target = root / "images" / f"{wanted[member.name]['id']}.jpg"
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                data = handle.extractfile(member).read()
                temporary = target.with_suffix(".part")
                temporary.write_bytes(data)
                temporary.replace(target)
    return selected


def openimages_catalog(root, count, seed, split):
    """Train annotations are streamed into a bounded candidate pool, not downloaded wholesale."""
    pool = {}
    # Keep a 2x candidate pool, then shuffle. This is a bounded prefix sample, not
    # a uniform sample of the entire Open Images training population.
    bound = count * 2
    with requests.get(OI_BOXES[split], stream=True, timeout=(20, 120)) as response:
        response.raise_for_status()
        response.raw.decode_content = True
        reader = csv.DictReader(io.TextIOWrapper(response.raw, encoding="utf-8"))
        for row in reader:
            if row["LabelName"] != CAT_CLASS or row.get("IsGroupOf", "0") == "1":
                continue
            image_id = row["ImageID"]
            if image_id not in pool and len(pool) >= bound:
                break
            pool.setdefault(image_id, []).append(
                [float(row[k]) for k in ["XMin", "YMin", "XMax", "YMax"]]
            )
    records = []
    for image_id, boxes in pool.items():
        records.append(
            {
                "id": f"oi_{split}_{image_id}",
                "source": "openimages_v7",
                "source_split": split,
                "source_id": image_id,
                "download_url": f"https://open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg",
                "source_url": None,
                "license": {"name": "CC BY 2.0 (source-listed)"},
                "original_captions": [],
                "boxes_normalized": boxes,
                "boxes_xywh": [],
                "head_box_xyxy": None,
                "sampling": "seeded shuffle of bounded annotation prefix",
            }
        )
    records = choose(records, count, seed)
    lookup = {row["source_id"]: row for row in records}
    # Preserve attribution from the official image information file.
    with requests.get(OI_METADATA[split], stream=True, timeout=(20, 120)) as response:
        response.raise_for_status()
        response.raw.decode_content = True
        reader = csv.DictReader(io.TextIOWrapper(response.raw, encoding="utf-8"))
        missing = set(lookup)
        for meta in reader:
            image_id = meta["ImageID"]
            if image_id in missing:
                lookup[image_id]["attribution"] = dict(meta)
                lookup[image_id]["source_url"] = meta.get("OriginalURL") or meta.get(
                    "LandingURL"
                )
                lookup[image_id]["license"]["url"] = meta.get("License")
                missing.remove(image_id)
                if not missing:
                    break
    for image_id in missing:
        lookup[image_id]["attribution_missing"] = True
    return records


def inspect_image(root, record):
    target = root / "images" / f"{record['id']}.jpg"
    if record["source"] != "oxford_pets":
        download(record["download_url"], target)
        try:
            with Image.open(target) as downloaded:
                downloaded.verify()
        except (OSError, SyntaxError):
            target.unlink(missing_ok=True)
            download(record["download_url"], target)
    with Image.open(target) as image:
        image.load()
        width, height = image.size
        orientation = image.getexif().get(274, 1)
        # Ground-truth boxes refer to source coordinates. Do not silently rotate.
        small = image.convert("L").resize((9, 8))
        values = list(small.tobytes())
        bits = [
            values[y * 9 + x] > values[y * 9 + x + 1]
            for y in range(8)
            for x in range(8)
        ]
        dhash = f"{sum(int(bit) << i for i, bit in enumerate(bits)):016x}"
    normalized = list(record.get("boxes_normalized", []))
    for x, y, w, h in record.get("boxes_xywh", []):
        normalized.append([x / width, y / height, (x + w) / width, (y + h) / height])
    target_box = [0, 0, 1000, 1000]
    if normalized:
        box = max(normalized, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        target_box = [round(v * 1000) for v in box]
    return {
        **record,
        "path": str(target.relative_to(root)),
        "sha256": sha256(target),
        "width": width,
        "height": height,
        "dhash": dhash,
        "target_box": target_box,
        "multiple_cats": len(normalized) > 1,
        "exif_orientation": orientation,
        "source_group_id": f"{record['source']}:{record['source_id']}",
    }


def prepare(config_path):
    config = read_json(config_path)
    root, output = paths()
    settings = config["dataset"]
    with exclusive_lock(root / ".prepare.lock"):
        existing_manifest = output / "manifest.json"
        if (
            existing_manifest.exists()
            and read_json(existing_manifest)["dataset_config"] != settings
        ):
            raise RuntimeError("Dataset settings changed. Use a new OUTPUT_ROOT.")
        catalog_path = root / "catalogs" / f"{fingerprint(settings)}.json"
        if catalog_path.exists():
            catalog = read_json(catalog_path)
        else:
            catalog = []
            if settings["coco"]:
                print("Preparing COCO cat catalog", flush=True)
                catalog += coco_catalog(root, settings["coco"], settings["seed"])
            if settings["oxford"]:
                print("Preparing Oxford cat catalog", flush=True)
                catalog += oxford_catalog(root, settings["oxford"], settings["seed"])
            if settings["openimages"]:
                print("Preparing Open Images cat catalog", flush=True)
                catalog += openimages_catalog(
                    root,
                    settings["openimages"],
                    settings["seed"],
                    settings["openimages_split"],
                )
            atomic_json(catalog_path, catalog)
        records, errors = [], []
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [(row, pool.submit(inspect_image, root, row)) for row in catalog]
            for index, (row, future) in enumerate(futures, 1):
                try:
                    records.append(future.result())
                except Exception as error:
                    errors.append({"id": row["id"], "error": str(error)})
                if index % 25 == 0:
                    print(f"Images checked {index}/{len(catalog)}", flush=True)
        # Exact duplicates are excluded. dHash is retained for later near-duplicate review.
        seen, unique, duplicates = {}, [], []
        for row in records:
            if row["sha256"] in seen:
                duplicates.append(
                    {"id": row["id"], "duplicate_of": seen[row["sha256"]]}
                )
            else:
                seen[row["sha256"]] = row["id"]
                unique.append(row)
        random.Random(settings["seed"]).shuffle(unique)
        manifest = {
            "dataset_config": settings,
            "images": unique,
            "duplicates": duplicates,
            "download_errors": errors,
            "requested": sum(settings[k] for k in ["coco", "oxford", "openimages"]),
            "downloaded": len(records),
            "unique": len(unique),
            "notes": "External cats only; no Shuding images. dHash is a review aid, not verified identity/split.",
        }
        if errors:
            atomic_json(output / "prepare-errors.json", manifest)
            raise RuntimeError(
                "Download failures recorded in prepare-errors.json. Same command retries downloads."
            )
        destination = output / "manifest.json"
        if destination.exists() and read_json(destination) != manifest:
            raise RuntimeError(
                "Manifest differs from this run. Use a new OUTPUT_ROOT; existing labels were preserved."
            )
        atomic_json(destination, manifest)
        print(
            f"Unique images {len(unique)}; errors {len(errors)}; requested {manifest['requested']}",
            flush=True,
        )
        if len(unique) < manifest["requested"]:
            print(
                "NOTICE: fewer unique eligible cats than requested; inspect manifest before scaling.",
                flush=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.json")
    prepare(parser.parse_args().config)
