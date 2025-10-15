import random

import numpy as np
import PIL.Image
import torch

IMAGE_SIZE = 384
IGNORE_INDEX = -100
# Modified system prompt when training for safety refusal
SAFETY_REFUSAL_SYSTEM_PROMPT = (
    "You are a helpful language and vision assistant. "
    "You are able to understand the visual content that the user provides, "
    "generate new images, "
    "and assist the user with a variety of tasks using natural language."
)


def fix_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_images(conversation: list[dict]) -> list[PIL.Image.Image]:
    images = []
    for message in conversation:
        if "images" in message:
            for image in message["images"]:
                assert isinstance(image, PIL.Image.Image)
                images.append(image.convert("RGB"))
    assert len(images) < 2, "Only support a single image per message for now"
    return images
