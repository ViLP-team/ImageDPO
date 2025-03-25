import argparse
import json
import math
import os

import shortuuid
import torch
from IPython import embed
from PIL import Image
from torch.utils.data import DataLoader, Dataset
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
    get_model_name_from_path,
    process_images,
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


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder):
        self.questions = questions
        self.image_folder = image_folder

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]

        image = os.path.join(self.image_folder, image_file)
        # image = Image.open(os.path.join(self.image_folder, image_file)).convert("RGB")
        # image_tensor = process_images([image], self.image_processor, self.model_config)[
        #     0
        # ]

        return qs, image

    def __len__(self):
        return len(self.questions)


# DataLoader
def create_data_loader(
    questions,
    image_folder,
    batch_size=1,
    num_workers=4,
):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_dict = init_base_models(args.model_name, device=torch.device("cuda"))

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file))]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    # new_write_file = open("/nfs/turbo/justincj-turbo/tiangel/improvingVLM/packages/LLaVA-Plus/playground/eval_data/single_img_dataset/MME/llava_mme_new.jsonl", 'w')
    # for q in questions:
    #     if q['category'] != "landmark":
    #         new_write_file.write(json.dumps(q) + '\n')
    # new_write_file.close()
    # breakpoint()

    ans_file = open(answers_file, "w")

    if (
        "plain" in args.model_name
        and "finetune" not in args.model_name.lower()
        and "mmtag" not in args.args.conv_mode
    ):
        args.conv_mode = args.conv_mode + "_mmtag"
        print(
            f"It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}."
        )

    data_loader = create_data_loader(questions, args.image_folder)

    for (input_qs, image_path), line in tqdm(
        zip(data_loader, questions), total=len(questions)
    ):
        idx = line["question_id"]
        cur_prompt = line["text"]
        outputs = eval_base_models(
            args.model_name,
            model_dict,
            {"input_img_path": [image_path[0]], "prompt": cur_prompt},
            torch.device("cuda"),
        )
        outputs = outputs.strip()
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
        # ans_file.flush()
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
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    eval_model(args)
