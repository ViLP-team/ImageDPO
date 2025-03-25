import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from PIL import Image
import math
from IPython import embed
from datasets import load_dataset
from tqdm import tqdm

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    dataset = load_dataset('facebook/winoground', keep_in_memory=True)
    ans_file = open(os.path.expanduser(args.answers_file), "w")

    image_0_data = dataset['test']['image_0']
    image_1_data = dataset['test']['image_1']
    caption_0_data = dataset['test']['caption_0']
    caption_1_data = dataset['test']['caption_1']
    for i in tqdm(range(len(image_0_data))):
        image_0 = image_0_data[i]
        image_1 = image_1_data[i]
        caption_0 = caption_0_data[i]
        caption_1 = caption_1_data[i]
        answers_collect = []
        for j in range(2):
            # image_0 and caption_0, caption_1
            conv = conv_templates[args.conv_mode].copy()

            #prompt = f"""
            #You are presented with an image and two potential captions: Caption A and Caption B. Your task is to carefully analyze the image and both captions to determine which caption better describes the image. You must choose one of the two options (A or B).
            #
            #Caption A: {caption_0}
            #Caption B: {caption_1}
            #
            #Consider the relevance, accuracy, and level of detail of each caption in relation to the image when making your choice.
            #"""

            #prompt = f"""
            #task: Compare two captions (A and B) and select the one that more accurately describes the given image.

            #Caption Options:
            #A: {caption_0}
            #B: {caption_1}
            #
            #Instructions:
            #1. Examine the image carefully, noting its main elements, subjects, and actions
            #2. Read both captions thoroughly
            #3. Compare how well each caption captures:
            #   - The main subject/action in the image
            #   - Important visual details
            #   - The overall context/setting
            #4. Select ONLY ONE option: either A or B
            #5. Provide your selection with brief reasoning
            #
            #Required Format:
            #- You must choose exactly one caption (A or B)
            #- No hybrid or alternative suggestions are allowed
            #- A simple "Caption A" or "Caption B" response is sufficient
            #
            #Your response should clearly indicate which caption (A or B) best matches the image.
            #"""

            prompt = f"""
            There is one image and two captions, caption A and caption B.  
            Please choose the caption that best describes the image,

            caption A: {caption_0}.
            caption B: {caption_1}.

            Note that, you must choose one of the two options: A or B.
            """
            

            # all 7b and 13 achieve 0.20+
            #prompt = f"""
            #There is one image and two potential captions for this image, caption A and caption B.  
            #Please choose the caption that best describes the image,

            #caption A: {caption_0}.
            #caption B: {caption_1}.

            #Note that, you must choose one of the two options (A or B).
            #"""

            #prompt = f"""
            #There is one image and two potential captions for this image, caption A and caption B.  
            #Look at the image and the captions carefully, and find the caption that best describes the image,
            #whether caption A or caption B. Note, you must choose one of the two options (A or B).
            #caption A: {caption_0}.
            #caption B: {caption_1}.
            #"""
            #prompt = f""""
            #Describe this image. 
            #"""
            if model.config.mm_use_im_start_end:
                prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
            else:
                prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            
            image = image_0 if j == 0 else image_1
            image = image.convert('RGB')
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=[image_tensor.unsqueeze(0).half().cuda()],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=1024,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria]
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
            outputs = outputs.replace("Caption", "").strip()
            outputs = outputs.replace("caption", "").strip()
            if ":" in outputs:
                outputs = outputs.split(":")[0].strip()
            answers_collect.append(outputs)
        ans_file.write(
            json.dumps(
                {
                    "gen_answers": answers_collect,
                    "image_index":int(i),
                }
            )
            + "\n"
        )
        ans_file.flush()
    ans_file.close()

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
