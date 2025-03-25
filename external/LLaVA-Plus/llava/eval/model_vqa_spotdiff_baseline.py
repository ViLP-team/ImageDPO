import argparse
import json
import math
import os

import shortuuid
import torch
from IPython import embed
from PIL import Image
from tqdm import tqdm

from llava.baseline_models.baselines import eval_base_models, init_base_models
from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

FOLDER_PATH = "/nfs/turbo/justincj-turbo/tiangel/improvingVLM/packages/LLaVA-Plus/playground/eval_data/two_img_dataset/spot-the-diff"


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_dict = init_base_models(args.model_name, device=torch.device("cuda"))

    # instructions = json.load(
    #     open(
    #         "/nfs/turbo/justincj-turbo/tiangel/improvingVLM/packages/LLaVA-Plus/playground/eval_data/two_img_dataset/MagicBrush/instruction_1k.json"
    #     )
    # )
    # image_dir = "/nfs/turbo/justincj-turbo/tiangel/improvingVLM/packages/LLaVA-Plus/playground/eval_data/two_img_dataset/MagicBrush/images"

    # NOTE: we only use the first 100 so far
    loaded_data = json.load(
        open(os.path.join(FOLDER_PATH, "data/annotations/test.json"))
    )[0:100]
    image_dir = os.path.join(FOLDER_PATH, "resized_images")

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for example in loaded_data:
        img_idx = example["img_id"]
        diff_description = example["sentences"]

        source_img = os.path.join(image_dir, f"{img_idx}.png")
        target_img = os.path.join(image_dir, f"{img_idx}_2.png")

        inp = "What elements were added or removed in the new image?"

        outputs = eval_base_models(
            args.model_name,
            model_dict,
            {"input_img_path": [source_img, target_img], "prompt": inp},
            torch.device("cuda"),
        )
        # inp += "Please provide modification instructions to change the image on the left into the image on the right."
        # inp += "Analyze this two images and answer how to change the image on the left into images on the right. Keep answer precise and clear. For example, `change the Chinese character into English words` or `move the hat to the wall`"
        # inp += "What changes were made to the left image to the right image?"
        # inp += 'what changes were made in the new image'

        outputs = outputs.strip()
        # ans_id = shortuuid.uuid()
        ans_file.write(
            json.dumps(
                {
                    "question_id": img_idx,
                    "prompt": diff_description,
                    "text": outputs,
                    "model_id": args.model_name,
                    "metadata": {},
                }
            )
            + "\n"
        )
        print(f"gt answer: {diff_description}")
        print(f"model answer: {outputs}")
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="instructblip")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
