import argparse
import json
import math
import os

import shortuuid
import torch
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


def eval_model(args):
    # Model
    disable_torch_init()
    model_dict = init_base_models(args.model_name, device=torch.device("cuda"))

    questions = json.load(open(os.path.expanduser(args.question_file)))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for i, line in enumerate(tqdm(questions)):
        idx = line["id"]
        question = line["conversations"][0]
        qs = question["value"].replace("<image>", "").strip()
        cur_prompt = qs

        if "image" in line:
            image_file = line["image"]
            image_path = os.path.join(args.image_folder, image_file)
        #     image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        #     images = image_tensor.unsqueeze(0).half().cuda()
        #     if getattr(model.config, 'mm_use_im_start_end', False):
        #         qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        #     else:
        #         qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        #     cur_prompt = '<image>' + '\n' + cur_prompt
        else:
            image_path = None

        if args.single_pred_prompt:
            qs = (
                qs
                + "\n"
                + "Answer with the option's letter from the given choices directly."
            )
            cur_prompt = (
                cur_prompt
                + "\n"
                + "Answer with the option's letter from the given choices directly."
            )

        outputs = eval_base_models(
            args.model_name,
            model_dict,
            {"input_img_path": [image_path], "prompt": cur_prompt},
            torch.device("cuda"),
        )

        outputs = outputs.strip()
        # prompt for answer
        if args.answer_prompter:
            outputs_reasoning = outputs

            outputs = eval_base_models(
                args.model_name,
                model_dict,
                {
                    "input_img_path": [image_path],
                    "prompt": cur_prompt + " ###\nANSWER:",
                },
                torch.device("cuda"),
            )

            outputs = outputs.strip()
            outputs = outputs_reasoning + "\n The answer is " + outputs

        ans_id = shortuuid.uuid()
        ans_file.write(
            json.dumps(
                {
                    "question_id": idx,
                    "prompt": cur_prompt,
                    "text": outputs,
                    "answer_id": ans_id,
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
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v0")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    args = parser.parse_args()

    eval_model(args)
