from collections import Counter
import json

from .common import paths, read_json


def main():
    _, output = paths()
    if not (output / "manifest.json").exists():
        print(f"Data preparation pending: {output}")
        return
    manifest = read_json(output / "manifest.json")
    states = Counter()
    errors = []
    for image in manifest["images"]:
        item = output / "items" / image["id"]
        if (item / "result.json").exists():
            states[read_json(item / "result.json")["review_status"]] += 1
        elif (item / "error.json").exists():
            states["failed"] += 1
            errors.append(read_json(item / "error.json"))
        elif (item / "full.json").exists():
            states["face_pending"] += 1
        else:
            states["pending"] += 1
    print(
        json.dumps(
            {
                "output_root": str(output),
                "total": manifest["unique"],
                "requested": manifest["requested"],
                "states": dict(states),
                "errors": errors[:5],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
