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


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def read_jsonl_file(filepath):
    data = []
    with open(filepath) as f:
        for line in f:
            data.append(json.loads(line))
    return data


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

    instructions = json.load(
        open(
            "/nfs/turbo/justincj-turbo/tiangel/improvingVLM/packages/LLaVA-Plus/playground/eval_data/two_img_dataset/ERL/train.json"
        )
    )
    uid2info = {item["uid"]: item for item in instructions}
    image_dir = "/nfs/turbo/justincj-turbo/tiangel/improvingVLM/packages/LLaVA-Plus/playground/eval_data/two_img_dataset/ERL/images"
    qas = read_jsonl_file(
        "/nfs/turbo/justincj-turbo/tiangel/improvingVLM/packages/LLaVA-Plus/playground/eval_data/two_img_dataset/ERL/gpt_qas_ERL.jsonl"
    )
    cur_qas = get_chunk(qas, args.num_chunks, args.chunk_idx)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    # for item in tqdm(qas[:5]):
    #for item in tqdm(qas):
    for item in tqdm(cur_qas):
        source_img = os.path.join(image_dir, uid2info[item["id"]]["img0"])
        target_img = os.path.join(image_dir, uid2info[item["id"]]["img1"])
        # cur_instruction = uid2info[item['id']]['sents'][0]
        cur_key = item["id"]
        for index, qa_dict in enumerate(item["QA"]):
            outputs = eval_base_models(
                args.model_name,
                model_dict,
                {
                    "input_img_path": [source_img, target_img],
                    "prompt": qa_dict["Question"],
                },
                torch.device("cuda"),
            )
            print("model:", outputs)
            print("answer:", qa_dict["Answer"])

            ans_file.write(
                json.dumps(
                    {
                        "question_id": cur_key,
                        "question_index": index,
                        "question": qa_dict["Question"],
                        "prompt": qa_dict["Answer"],
                        "text": outputs,
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
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
