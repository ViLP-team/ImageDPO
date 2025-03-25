import argparse
import json
import math
import os
import pickle
import re

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


def extract_id(filename):
    # This regular expression finds one or more digits that are preceded by 'COCO_train2014_' and followed by '.jpg'
    match = re.search(r"COCO_train2014_(\d+)\.jpg", filename)
    if match:
        return int(match.group(1))
    else:
        return None


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def eval_model(args):
    # Model
    disable_torch_init()
    model_dict = init_base_models(args.model_name, device=torch.device("cuda"))

    # instructions = json.load(
    #    open(
    #        "/nfs/turbo/justincj-turbo/tiangel/improvingVLM/packages/LLaVA-Plus/playground/eval_data/two_img_dataset/MagicBrush/instruction_1k.json"
    #    )
    # )
    img_list = pickle.load(
        open(
            "/nfs/turbo/justincj-turbo/tiangel/improvingVLM/packages/LLaVA-Plus/playground/eval_data/single_img_dataset/Hallucination/CHAIR_img_list.pkl",
            "rb",
        )
    )
    image_dir1 = "/nfs/turbo/justincj-turbo/shared_datasets/coco/train2017"
    image_dir2 = "/nfs/turbo/justincj-turbo/shared_datasets/coco/train2014"
    cur_img_list = get_chunk(img_list, args.num_chunks, args.chunk_idx)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for cur_img in tqdm(cur_img_list):
        source_img = os.path.join(image_dir1, cur_img.split("_")[-1])
        if not os.path.exists(source_img):
            source_img = os.path.join(image_dir2, cur_img)
            if not os.path.exists(source_img):
                continue

        image_data = [Image.open(source_img)]
        outputs = eval_base_models(
            args.model_name,
            model_dict,
            {"input_img_path": [source_img], "prompt": "Describe this image."},
            torch.device("cuda"),
        )
        outputs = outputs.strip()

        # ans_id = shortuuid.uuid()
        ans_file.write(
            json.dumps(
                {
                    "caption": outputs,
                    "image_id": extract_id(cur_img),
                    "model_id": args.model_name,
                }
            )
            + "\n"
        )
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="instructblip")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
