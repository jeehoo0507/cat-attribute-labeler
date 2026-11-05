import argparse

from huggingface_hub import snapshot_download

from .common import exclusive_lock, paths, read_json
from .label import pin_model


def main(config_path):
    _, output = paths()
    config = read_json(config_path)
    with exclusive_lock(output / ".label.lock"):
        pin = pin_model(config, output)
        location = snapshot_download(
            config["model"],
            revision=pin["resolved_revision"],
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"],
            max_workers=4,
        )
        print(f"Pinned model downloaded to {location}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.json")
    main(parser.parse_args().config)
