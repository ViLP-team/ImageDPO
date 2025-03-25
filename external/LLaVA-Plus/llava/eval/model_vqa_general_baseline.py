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


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def read_jsonl_file(filepath):
    data = []
    with open(filepath) as f:
        for line in f:
            data.append(json.loads(line))
    return data


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

    instructions = read_jsonl_file(args.instruction_path)
    image_dir = args.image_folder
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for item in instructions:
        source_img = item["image_path"][0]
        target_img = item["image_path"][1]

        # NOTE: TODO: we only use the first question here. Might be inefficient
        cur_question = item["QA"][0]["Question"]
        cur_answer = item["QA"][0]["Answer"]

        outputs = eval_base_models(
            args.model_name,
            model_dict,
            {"input_img_path": [source_img, target_img], "prompt": cur_question},
            torch.device("cuda"),
        )

        outputs = outputs.strip()

        print("model:", outputs)
        print("answer:", cur_answer)

        outputs = outputs.strip()
        # ans_id = shortuuid.uuid()
        ans_file.write(
            json.dumps(
                {
                    "question_id": item["id"],
                    "question": cur_question,
                    "gpt_answer": cur_answer,
                    "answer": outputs,
                    "model_id": args.model_name,
                    "metadata": {},
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
    parser.add_argument(
        "--instruction_path",
        type=str,
        default="/nfs/turbo/justincj-turbo/tiangel/improvingVLM/packages/LLaVA-Plus/playground/eval_data/two_img_dataset/COCO/gpt_qas_COCO.jsonl",
    )
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
