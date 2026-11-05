import argparse
from datetime import datetime, timezone
import importlib.metadata
import time

from PIL import Image

from .common import atomic_json, exclusive_lock, fingerprint, paths, read_json, sha256
from .schema import FaceLabels, FullLabels, merge_labels, parse_response, prompt


def bounded_image(image, max_pixels):
    """Keep source detail without upscaling a tiny face crop."""
    image = image.convert("RGB")
    scale = min(1.0, (max_pixels / (image.width * image.height)) ** 0.5)
    return image.resize(
        (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    )


def face_crop(image, record, full):
    coords = record.get("head_box_xyxy")
    origin = "source_head_annotation"
    if not coords and full["face_box"]:
        coords = [
            full["face_box"]["xyxy"][i]
            / 1000
            * (image.width if i % 2 == 0 else image.height)
            for i in range(4)
        ]
        origin = "vlm_face_box_unverified"
    if not coords:
        return None, {"origin": "missing", "review_required": True}
    x1, y1, x2, y2 = coords
    width, height = x2 - x1, y2 - y1
    # Include ears, chin, and neck for collar visibility. Retain crop geometry for review.
    box = [
        max(0, int(x1 - width * 0.15)),
        max(0, int(y1 - height * 0.15)),
        min(image.width, int(x2 + width * 0.15)),
        min(image.height, int(y2 + height * 0.4)),
    ]
    if box[2] <= box[0] or box[3] <= box[1]:
        return None, {"origin": "invalid", "review_required": True}
    return image.crop(box), {
        "origin": origin,
        "xyxy_pixels": box,
        "source_width": box[2] - box[0],
        "source_height": box[3] - box[1],
        "review_required": origin != "source_head_annotation",
    }


class QwenBackend:
    def __init__(self, config, revision):
        import torch
        from transformers import (
            AutoProcessor,
            BitsAndBytesConfig,
            Qwen3VLForConditionalGeneration,
        )

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required. Run ./preflight.sh on the server.")
        options = {
            "revision": revision,
            "device_map": {"": 0},
            "dtype": torch.bfloat16,
            "attn_implementation": "sdpa",
        }
        if config["quantization"] == "nf4":
            options["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            config["model"], **options
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            config["model"], revision=revision
        )
        self.max_new_tokens = config["max_new_tokens"]
        print(
            f"Loaded {config['model']} @ {revision}; footprint {self.model.get_memory_footprint() / 1024**3:.2f} GiB",
            flush=True,
        )

    def generate(self, images, instruction):
        import torch
        from qwen_vl_utils import process_vision_info

        content = [{"type": "image", "image": image} for image in images]
        messages = [
            {
                "role": "user",
                "content": content + [{"type": "text", "text": instruction}],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda:0")
        inputs.pop("token_type_ids", None)
        with torch.inference_mode():
            tokens = self.model.generate(
                **inputs, do_sample=False, max_new_tokens=self.max_new_tokens
            )
        result = self.processor.batch_decode(
            tokens[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )[0]
        return result


def pin_model(config, output):
    from huggingface_hub import HfApi

    pin_path = output / "model-pin.json"
    identity = {"model": config["model"], "requested_revision": config["revision"]}
    if pin_path.exists():
        pin = read_json(pin_path)
        if any(pin[key] != value for key, value in identity.items()):
            raise RuntimeError("Model changed. Choose a new OUTPUT_ROOT.")
        return pin
    revision = HfApi().model_info(config["model"], revision=config["revision"]).sha
    pin = {**identity, "resolved_revision": revision}
    atomic_json(pin_path, pin)
    return pin


def generate_stage(backend, stage, images, record, item_dir, config):
    destination = item_dir / f"{stage}.json"
    schema = FullLabels if stage == "full" else FaceLabels
    if destination.exists():
        return schema.model_validate(read_json(destination)).model_dump()
    instruction = prompt(stage, record["target_box"])
    for attempt in range(config["json_attempts"]):
        raw = backend.generate(images, instruction)
        (item_dir / f"{stage}-attempt-{attempt + 1}.txt").write_text(
            raw, encoding="utf-8"
        )
        try:
            result = parse_response(raw, schema).model_dump()
            atomic_json(destination, result)
            return result
        except ValueError as error:
            # A fresh call with the same images; no earlier attribute predictions
            # are injected into the independent face pass.
            instruction = (
                prompt(stage, record["target_box"])
                + "\nYour response must be valid JSON. Include every required field."
            )
            if attempt + 1 == config["json_attempts"]:
                raise ValueError(
                    f"{stage} response failed schema validation: {error}"
                ) from error


def run(config_path, limit=None):
    config = read_json(config_path)
    root, output = paths()
    manifest = read_json(output / "manifest.json")
    with exclusive_lock(output / ".label.lock"):
        pin = pin_model(config, output)
        signature = fingerprint(
            {
                "config": config,
                "model_pin": pin,
                "full_prompt": prompt("full", []),
                "face_prompt": prompt("face", []),
            }
        )
        run_path = output / "run.json"
        if run_path.exists() and read_json(run_path)["signature"] != signature:
            raise RuntimeError(
                "Configuration/schema/prompts changed. Choose a new OUTPUT_ROOT."
            )
        versions = {
            name: importlib.metadata.version(name)
            for name in ["torch", "transformers", "bitsandbytes", "qwen-vl-utils"]
        }
        atomic_json(
            run_path,
            {
                "signature": signature,
                "config": config,
                "pin": pin,
                "versions": versions,
                "backend": "qwen_cuda",
                "schema_version": 1,
            },
        )
        completed, errors, backend = 0, [], None
        images = manifest["images"] if limit is None else manifest["images"][:limit]
        for record in images:
            started = time.monotonic()
            item_dir = output / "items" / record["id"]
            item_dir.mkdir(parents=True, exist_ok=True)
            result_path = item_dir / "result.json"
            expected = {"signature": signature, "image_sha256": record["sha256"]}
            context = item_dir / "context.json"
            if context.exists() and read_json(context) != expected:
                raise RuntimeError(f"Input or run identity changed for {record['id']}")
            atomic_json(context, expected)
            if result_path.exists():
                completed += 1
                continue
            try:
                image_path = root / record["path"]
                if sha256(image_path) != record["sha256"]:
                    raise ValueError("Source image checksum changed")
                if record.get("exif_orientation", 1) != 1:
                    raise ValueError(
                        "EXIF orientation requires review and coordinate conversion"
                    )
                if backend is None:
                    backend = QwenBackend(config, pin["resolved_revision"])
                with Image.open(image_path) as source:
                    image = source.convert("RGB")
                full_input = bounded_image(image, config["full_max_pixels"])
                full = generate_stage(
                    backend, "full", [full_input], record, item_dir, config
                )
                crop, geometry = face_crop(image, record, full)
                face_inputs = [full_input]
                if crop is not None:
                    face_inputs.append(bounded_image(crop, config["crop_max_pixels"]))
                face = generate_stage(
                    backend, "face", face_inputs, record, item_dir, config
                )
                labels, conflicts = merge_labels(full, face)
                flags = ["human_review_required"]
                if record["multiple_cats"]:
                    flags.append("multiple_cats_target_check")
                if geometry["review_required"]:
                    flags.append("crop_geometry_check")
                if conflicts:
                    flags.append("pass_disagreement")
                atomic_json(
                    result_path,
                    {
                        "image_id": record["id"],
                        "image_sha256": record["sha256"],
                        "run_signature": signature,
                        "subject_id": "largest_annotated_cat",
                        "labels": labels,
                        "conflicts": conflicts,
                        "crop": geometry,
                        "review_flags": flags,
                        "review_status": "unreviewed",
                        "seconds": round(time.monotonic() - started, 2),
                    },
                )
                (item_dir / "error.json").unlink(missing_ok=True)
                completed += 1
                print(f"Labeled {completed}/{len(images)} {record['id']}", flush=True)
            except Exception as error:
                atomic_json(
                    item_dir / "error.json",
                    {
                        "id": record["id"],
                        "type": type(error).__name__,
                        "error": str(error),
                    },
                )
                if "OutOfMemory" in type(error).__name__ or backend is None:
                    raise RuntimeError(
                        "Model load or GPU memory failure. Inspect error.json; use a new run for changed model/settings."
                    ) from error
                errors.append(record["id"])
                print(f"FAILED {record['id']}: {error}", flush=True)
            atomic_json(
                output / "progress.json",
                {
                    "completed_this_selection": completed,
                    "selection_total": len(images),
                    "dataset_total": len(manifest["images"]),
                    "errors": errors,
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
        if errors:
            raise RuntimeError(
                f"{len(errors)} failed images. Same command retries unfinished items."
            )
        print(
            f"Selection complete {completed}/{len(images)}. Outputs remain unreviewed drafts.",
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.json")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    run(args.config, args.limit)
