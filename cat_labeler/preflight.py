import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from .common import atomic_json, paths, read_json


def check(config_path, cpu_only=False):
    root, output = paths()
    config = read_json(config_path)
    root.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    cache_root = Path(os.environ.get("HF_HOME", str(root))).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    report = {
        "data_root": str(root),
        "output_root": str(output),
        "model": config["model"],
        "free_disk_gib": round(shutil.disk_usage(root).free / 1024**3, 1),
        "free_model_cache_gib": round(shutil.disk_usage(cache_root).free / 1024**3, 1),
        "cpu_only": cpu_only,
    }
    if report["free_model_cache_gib"] < 85 and not cpu_only:
        raise RuntimeError(
            "At least 85 GiB free disk required for original 32B model shards and the pilot."
        )
    if report["free_disk_gib"] < 5 and not cpu_only:
        raise RuntimeError("At least 5 GiB free data storage required for the pilot.")
    if not cpu_only:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA not available; check NVIDIA driver and uv sync --extra gpu."
            )
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Select exactly one GPU with GPUS=0 in server scripts.")
        free, total = torch.cuda.mem_get_info(0)
        report.update(
            {
                "gpu": torch.cuda.get_device_name(0),
                "cuda": torch.version.cuda,
                "torch": torch.__version__,
                "free_vram_gib": round(free / 1024**3, 2),
                "total_vram_gib": round(total / 1024**3, 2),
                "bf16": torch.cuda.is_bf16_supported(),
            }
        )
        if not report["bf16"]:
            raise RuntimeError("This recipe requires BF16 support.")
        minimum = 21 if "32B" in config["model"] else 12
        if free / 1024**3 < minimum:
            raise RuntimeError(
                f"Need >= {minimum} GiB free VRAM for the configured pilot. No silent model fallback."
            )
        # Launch a real CUDA kernel to detect unsupported driver/wheel combinations.
        value = torch.ones((16, 16), device="cuda", dtype=torch.bfloat16)
        assert (value @ value).float().mean().item() == 16
        report["driver"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        ).strip()
    atomic_json(output / "preflight.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.json")
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    check(args.config, args.cpu_only)
