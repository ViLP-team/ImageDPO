import argparse
import json
import math
import os
import pickle
import re
import glob

import shortuuid
import torch
from IPython import embed
from PIL import Image
from tqdm import tqdm

from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import SeparatorStyle, conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    tokenizer_image_token,
)
import pandas as pd


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
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    info = pd.read_csv(os.path.join(args.question_file))

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for i in tqdm(range(len(info))):
        index = info["Index"][i]
        image_files = glob.glob(os.path.join(args.image_folder, f'{index:04d}_*.jpg'))
        if len(image_files) < 3:
            continue
        answers_collect = [ ]
        gt_answers_collect = []
        for j in range(3):
            source_img = os.path.join(args.image_folder, f'{index:04d}_{j+1}.jpg')
            if args.without_fact:
                question = f"Please answer with one word: {'.'.join(info['Question'][i].split('.')[1:])}"
            else:
                question = f"Please answer with one word: {info['Question'][i]}"
 
            conv = conv_templates[args.conv_mode].copy()
            image_data = [Image.open(source_img)]
            image_tensor = []

            for image in image_data:
                #image = expand2square(
                #    image, tuple(int(x * 255) for x in image_processor.image_mean)
                #)
                image = (
                    image_processor.preprocess(image, return_tensors="pt")["pixel_values"]
                    .half()
                    .cuda()
                )
                image_tensor.append(image)

            inp = ""
            inp += DEFAULT_IMAGE_TOKEN + "\n"
            inp += question

            gt_answers_collect.append(info['answer%d'%(j+1)][i])
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()


            input_ids = (
                tokenizer_image_token(
                    prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                )
                .unsqueeze(0)
                .cuda()
            )
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=[image_tensor],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    # num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=1024,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                )

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (
                (input_ids != output_ids[:, :input_token_len]).sum().item()
            )
            if n_diff_input_output > 0:
                print(
                    f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
                )
            outputs = tokenizer.batch_decode(
                output_ids[:, input_token_len:], skip_special_tokens=True
            )[0]
            outputs = outputs.strip()

            answers_collect.append(outputs)
        # ans_id = shortuuid.uuid()
        ans_file.write(
            json.dumps(
                {
                    "gen_answers": answers_collect,
                    "gt_answers": gt_answers_collect,
                    "image_index":int(index),
                }
            )
            + "\n"
        )
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--without_fact", action="store_true", default=False)
    args = parser.parse_args()

    eval_model(args)
