"""All enum values are observable states. Unknown values never imply absence."""

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Observation(StrictModel):
    value: str
    evidence: str = Field(min_length=1, max_length=240)


class Presence(Observation):
    value: Literal["present", "absent", "unknown"]


class Visibility(Observation):
    value: Literal["visible", "not_visible", "unknown"]


class Eye(Observation):
    value: Literal["open", "partly_open", "closed", "unknown"]


class Mouth(Observation):
    value: Literal["closed", "slightly_open", "wide_open", "unknown"]


class Tongue(Observation):
    value: Literal["outside_lips", "inside_mouth_only", "not_visible", "unknown"]


class Posture(Observation):
    value: Literal[
        "sitting", "standing", "prone", "side_lying", "supine", "other", "unknown"
    ]


class Box(StrictModel):
    xyxy: list[int] = Field(min_length=4, max_length=4)
    evidence: str = Field(min_length=1, max_length=240)

    @model_validator(mode="after")
    def check_coordinates(self):
        x1, y1, x2, y2 = self.xyxy
        if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
            raise ValueError("Box must be normalized 0..1000 with positive area")
        return self


class Clothes(StrictModel):
    presence: Presence
    type_raw: str | None
    colors: list[str] = Field(max_length=5)
    patterns: list[str] = Field(max_length=5)


class Relation(StrictModel):
    object: str = Field(min_length=1, max_length=80)
    relation: Literal["on", "in", "under", "beside", "in_front_of", "held_by"]
    evidence: str = Field(min_length=1, max_length=240)


class FullLabels(StrictModel):
    target_visible: Presence
    face_box: Box | None
    posture: Posture
    tucked_front_paws: Presence
    curled_body: Presence
    forelegs_extended: Presence
    paw_pads: Visibility
    glasses: Presence
    collar: Presence
    scarf: Presence
    headwear: Presence
    leash_visible: Visibility
    clothes: Clothes
    scene: list[str] = Field(max_length=8)
    relations: list[Relation] = Field(max_length=8)
    interactions: list[str] = Field(max_length=5)


class FaceLabels(StrictModel):
    target_visible: Presence
    left_eye: Eye
    right_eye: Eye
    mouth: Mouth
    tongue: Tongue
    teeth: Visibility
    glasses: Presence
    collar: Presence


RULES = """You annotate ONLY the target real cat specified by the input box.
Return exactly one JSON object matching the schema. No markdown or extra text.
Use English enum values and short Korean visual evidence. Never infer hidden details.
Unknown means occluded, cropped, blurred, too small, or ambiguous. No guessing.
Absent requires a clear view of the relevant region. Not_visible requires a clear
view sufficient to judge visibility. An invisible face does not imply closed eyes.
Left/right are the CAT'S anatomical left/right. If uncertain about side use unknown.
Prone means belly against support; side_lying means torso side against support;
supine means back against support with belly upward. Loaf is prone + tucked paws.
Do not infer sleeping, feelings, yawning, walking, running or intentions from one frame.
Tongue outside_lips requires visible protrusion beyond lips, not tongue inside a yawn.
Record actual object relations, not mere background co-occurrence.
Free-text clothing type/colors/patterns and scene objects must be directly visible.
If target_visible is absent or unknown, leave unverifiable attributes unknown.
"""


def prompt(stage, target_box):
    schema = FullLabels if stage == "full" else FaceLabels
    instruction = (
        "Image 1 is the full scene. Give a face+neck box in full-image normalized "
        "0..1000 coordinates, or null when no reliable face region exists."
    )
    if stage == "face":
        instruction = (
            "Image 1 is the full scene. Image 2, when supplied, is a crop of the same "
            "target face/neck. Judge facial details independently. The crop may include "
            "occlusion or lack detail. Use unknown when the source pixels do not resolve it."
        )
    return (
        RULES
        + "\n"
        + instruction
        + f"\nTarget body box (0..1000): {target_box}.\n"
        + json.dumps(schema.model_json_schema(), ensure_ascii=False)
    )


def parse_response(text, schema):
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        stripped = stripped.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return schema.model_validate_json(stripped)


def merge_labels(full, face):
    merged = dict(full)
    merged.pop("face_box", None)
    conflicts = []
    for key, value in face.items():
        if key in merged:
            first, second = merged[key]["value"], value["value"]
            if first != "unknown" and second != "unknown" and first != second:
                conflicts.append(key)
                merged[key] = {
                    "value": "unknown",
                    "evidence": "Independent passes disagree; review required",
                }
                continue
            if second == "unknown" and first != "unknown":
                continue
        merged[key] = value
    return merged, conflicts
