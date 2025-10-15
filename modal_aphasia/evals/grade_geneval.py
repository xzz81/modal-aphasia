# Based on https://raw.githubusercontent.com/djghosh13/geneval/refs/heads/main/evaluation/evaluate_images.py

#############
## WARNING ##
#############
# This needs to be run in the geneval virtual environment,
# NOT the one from this repo!
# See README.md for instructions.


import argparse
import base64
import io
import json
import os
import pathlib
import sys
import time
import tqdm
import warnings

import PIL.Image

warnings.filterwarnings("ignore")

import mmdet
import numpy as np
import open_clip
import pandas as pd
import torch
from clip_benchmark.metrics import zeroshot_classification as zsc
from mmdet.apis import inference_detector, init_detector
from PIL import Image, ImageOps

zsc.tqdm = lambda it, *args, **kwargs: it

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
assert DEVICE == "cuda"

# We are using geneval's default values here
THRESHOLD = 0.3
COUNTING_THRESHOLD = 0.9
MAX_OBJECTS = 16
NMS_THRESHOLD = 1.0
POSITION_THRESHOLD = 0.1
OBJECT_DETECTOR = "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco"

# Paths:
# Assumes that everything is installed as in README.md;
# i.e., mmdetection inside the geneval repo
GENEVAL_ROOT = pathlib.Path(os.getenv("GENEVAL_ROOT", pathlib.Path(mmdet.__file__).parent.parent))
CONFIG_PATH = os.path.join(
    os.path.dirname(mmdet.__file__), "../configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"
)
MODEL_PATH = str(GENEVAL_ROOT / "model_weights")
CKPT_PATH = os.path.join(MODEL_PATH, f"{OBJECT_DETECTOR}.pth")


def main() -> None:
    args = _parse_args()
    object_detector, (clip_model, transform, tokenizer), classnames = load_models()

    with open(args.input, "r") as f_in:
        inference_data_full = tuple(json.loads(line) for line in f_in)

    with open(args.output, "w") as f_out:
        for inference_data in tqdm.tqdm(inference_data_full):
            with io.BytesIO(base64.b64decode(inference_data["inference_image_base64"])) as image_io:
                image = Image.open(image_io).convert("RGB")

            # Need to convert meta-data to geneval format
            metadata = _convert_metadata(inference_data)
            grading_result = evaluate_image(
                image, metadata, object_detector, classnames, clip_model, tokenizer, transform
            )

            output_data = {**inference_data, **{f"grading_{key}": value for key, value in grading_result.items()}}
            f_out.write(json.dumps(output_data) + "\n")


def _convert_metadata(inference_data):
    metadata = {
        "tag": inference_data["tag"],
    }

    for list_type in ("include", "exclude"):
        if len(inference_data[list_type]) == 0:
            continue

        metadata[list_type] = []
        for entry in inference_data[list_type]:
            # Remove empty attributes
            result = {
                key: val for key, val in entry.items()
                if val is not None
            }
            if "position" in result:
                result["position"] = [result["position"]["relation"], result["position"]["object_idx"]]
            metadata[list_type].append(result)

    return metadata

def timed(fn):
    def wrapper(*args, **kwargs):
        startt = time.time()
        result = fn(*args, **kwargs)
        endt = time.time()
        print(f"Function {fn.__name__!r} executed in {endt - startt:.3f}s", file=sys.stderr)
        return result

    return wrapper


@timed
def load_models():
    object_detector = init_detector(CONFIG_PATH, CKPT_PATH, device=DEVICE)

    clip_arch = "ViT-L-14"
    clip_model, _, transform = open_clip.create_model_and_transforms(clip_arch, pretrained="openai", device=DEVICE)
    tokenizer = open_clip.get_tokenizer(clip_arch)

    with open(os.path.join(GENEVAL_ROOT, "evaluation", "object_names.txt"), "r") as cls_file:
        classnames = [line.strip() for line in cls_file]

    return object_detector, (clip_model, transform, tokenizer), classnames


COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white"]
COLOR_CLASSIFIERS = {}

# Evaluation parts


class ImageCrops(torch.utils.data.Dataset):
    def __init__(self, image: Image.Image, objects, transform):
        self._image = image.convert("RGB")
        bgcolor = "#999"
        if bgcolor == "original":
            self._blank = self._image.copy()
        else:
            self._blank = Image.new("RGB", image.size, color=bgcolor)
        self._objects = objects
        self._transform = transform

    def __len__(self):
        return len(self._objects)

    def __getitem__(self, index):
        box, mask = self._objects[index]
        if mask is not None:
            assert tuple(self._image.size[::-1]) == tuple(mask.shape), (index, self._image.size[::-1], mask.shape)
            image = Image.composite(self._image, self._blank, Image.fromarray(mask))
        else:
            image = self._image
        # if args.options.get("crop", "1") == "1":
        if True:
            image = image.crop(box[:4])
        # if args.save:
        #     base_count = len(os.listdir(args.save))
        #     image.save(os.path.join(args.save, f"cropped_{base_count:05}.png"))
        return (self._transform(image), 0)


def color_classification(image, bboxes, classname, clip_model, tokenizer, transform):
    if classname not in COLOR_CLASSIFIERS:
        COLOR_CLASSIFIERS[classname] = zsc.zero_shot_classifier(
            clip_model,
            tokenizer,
            COLORS,
            [
                f"a photo of a {{c}} {classname}",
                f"a photo of a {{c}}-colored {classname}",
                f"a photo of a {{c}} object",
            ],
            DEVICE,
        )
    clf = COLOR_CLASSIFIERS[classname]
    dataloader = torch.utils.data.DataLoader(ImageCrops(image, bboxes, transform), batch_size=16, num_workers=4)
    with torch.no_grad():
        pred, _ = zsc.run_classification(clip_model, clf, dataloader, DEVICE)
        return [COLORS[index.item()] for index in pred.argmax(1)]


def compute_iou(box_a, box_b):
    area_fn = lambda box: max(box[2] - box[0] + 1, 0) * max(box[3] - box[1] + 1, 0)
    i_area = area_fn(
        [max(box_a[0], box_b[0]), max(box_a[1], box_b[1]), min(box_a[2], box_b[2]), min(box_a[3], box_b[3])]
    )
    u_area = area_fn(box_a) + area_fn(box_b) - i_area
    return i_area / u_area if u_area else 0


def relative_position(obj_a, obj_b):
    """Give position of A relative to B, factoring in object dimensions"""
    boxes = np.array([obj_a[0], obj_b[0]])[:, :4].reshape(2, 2, 2)
    center_a, center_b = boxes.mean(axis=-2)
    dim_a, dim_b = np.abs(np.diff(boxes, axis=-2))[..., 0, :]
    offset = center_a - center_b
    #
    revised_offset = np.maximum(np.abs(offset) - POSITION_THRESHOLD * (dim_a + dim_b), 0) * np.sign(offset)
    if np.all(np.abs(revised_offset) < 1e-3):
        return set()
    #
    dx, dy = revised_offset / np.linalg.norm(offset)
    relations = set()
    if dx < -0.5:
        relations.add("left of")
    if dx > 0.5:
        relations.add("right of")
    if dy < -0.5:
        relations.add("above")
    if dy > 0.5:
        relations.add("below")
    return relations


def evaluate(image, objects, metadata, clip_model, tokenizer, transform):
    """
    Evaluate given image using detected objects on the global metadata specifications.
    Assumptions:
    * Metadata combines 'include' clauses with AND, and 'exclude' clauses with OR
    * All clauses are independent, i.e., duplicating a clause has no effect on the correctness
    * CHANGED: Color and position will only be evaluated on the most confidently predicted objects;
        therefore, objects are expected to appear in sorted order
    """
    correct = True
    reason = []
    matched_groups = []
    # Check for expected objects
    for req in metadata.get("include", []):
        classname = req["class"]
        matched = True
        found_objects = objects.get(classname, [])[: req["count"]]
        if len(found_objects) < req["count"]:
            correct = matched = False
            reason.append(f"expected {classname}>={req['count']}, found {len(found_objects)}")
        else:
            if "color" in req:
                # Color check
                colors = color_classification(image, found_objects, classname, clip_model, tokenizer, transform)
                if colors.count(req["color"]) < req["count"]:
                    correct = matched = False
                    reason.append(
                        f"expected {req['color']} {classname}>={req['count']}, found "
                        + f"{colors.count(req['color'])} {req['color']}; and "
                        + ", ".join(f"{colors.count(c)} {c}" for c in COLORS if c in colors)
                    )
            if "position" in req and matched:
                # Relative position check
                expected_rel, target_group = req["position"]
                if matched_groups[target_group] is None:
                    correct = matched = False
                    reason.append(f"no target for {classname} to be {expected_rel}")
                else:
                    for obj in found_objects:
                        for target_obj in matched_groups[target_group]:
                            true_rels = relative_position(obj, target_obj)
                            if expected_rel not in true_rels:
                                correct = matched = False
                                reason.append(
                                    f"expected {classname} {expected_rel} target, found "
                                    + f"{' and '.join(true_rels)} target"
                                )
                                break
                        if not matched:
                            break
        if matched:
            matched_groups.append(found_objects)
        else:
            matched_groups.append(None)
    # Check for non-expected objects
    for req in metadata.get("exclude", []):
        classname = req["class"]
        if len(objects.get(classname, [])) >= req["count"]:
            correct = False
            reason.append(f"expected {classname}<{req['count']}, found {len(objects[classname])}")
    return correct, "\n".join(reason)


def evaluate_image(image: PIL.Image.Image, metadata, object_detector, classnames, clip_model, tokenizer, transform):
    result = inference_detector(object_detector, np.array(image))
    bbox = result[0] if isinstance(result, tuple) else result
    segm = result[1] if isinstance(result, tuple) and len(result) > 1 else None
    image = ImageOps.exif_transpose(image)
    detected = {}
    # Determine bounding boxes to keep
    confidence_threshold = THRESHOLD if metadata["tag"] != "counting" else COUNTING_THRESHOLD
    for index, classname in enumerate(classnames):
        ordering = np.argsort(bbox[index][:, 4])[::-1]
        ordering = ordering[bbox[index][ordering, 4] > confidence_threshold]  # Threshold
        ordering = ordering[:MAX_OBJECTS].tolist()  # Limit number of detected objects per class
        detected[classname] = []
        while ordering:
            max_obj = ordering.pop(0)
            detected[classname].append((bbox[index][max_obj], None if segm is None else segm[index][max_obj]))
            ordering = [
                obj
                for obj in ordering
                if NMS_THRESHOLD == 1 or compute_iou(bbox[index][max_obj], bbox[index][obj]) < NMS_THRESHOLD
            ]
        if not detected[classname]:
            del detected[classname]
    # Evaluate
    is_correct, reason = evaluate(image, detected, metadata, clip_model, tokenizer, transform)
    return {
        "correct": is_correct,
        "reason": reason,
        "details": json.dumps({key: [box.tolist() for box, _ in value] for key, value in detected.items()}),
    }


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=pathlib.Path,
        required=True,
        help="Path input inference JSONL file",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        required=True,
        help="Path to graded output JSONL file",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
