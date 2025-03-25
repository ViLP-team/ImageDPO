import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import re
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

SUFFIX_FOR_VQA = {
    "yes_no": "Please answer Yes or No.",
    "multiple_choice": "Please output the letter corresponding to the correct option."
} 

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def extract_answer(output_string, task_type="yes_no"):
    """
    Extracts the answer from the output string based on the task type.

    Parameters:
    output_string (str): The output string.
    task_type (str): The type of task. Must be either "yes_no" or "multiple_choice".

    Returns:
    int: 
        1 if "yes" or "A" 
        0 if "no" or "B"
        -1 if no relevant answer is found.
        Raises a ValueError if an unsupported task_type is provided.
    """

    def find_word_position(string, word):
        pattern = r'\b' + re.escape(word) + r'\b'
        match = re.search(pattern, string, re.IGNORECASE)
        if match:
            return match.start()
        return -1
    
    if task_type not in ["yes_no", "multiple_choice"]:
        raise ValueError("Task type not supported. Must be 'yes_no' or 'multiple_choice'.")
    
    if task_type == "yes_no":
        position_yes_and_a = find_word_position(output_string, "yes")
        position_no_and_b = find_word_position(output_string, "no")
    elif task_type == "multiple_choice":
        position_yes_and_a = find_word_position(output_string, "A")
        position_no_and_b = find_word_position(output_string, "B")

    if position_yes_and_a == -1 and position_no_and_b == -1:
        print(f"No answer found in the output string: {output_string}.")
        return -1
    elif position_yes_and_a != -1 and position_no_and_b != -1:
        return 1 if position_yes_and_a < position_no_and_b else 0
    else:
        return 0 if position_yes_and_a == -1 else 1

def get_scores(scores):
    """
    Calculate various scores based on the given results.

    Args:
        scores (dict or list): A dictionary or list containing results where each result can be:
            - dict: {id: {"q0_i0": 1 or 0, "q0_i1": 1 or 0, "q1_i0": 1 or 0, "q1_i1": 1 or 0}, ...}
            - list: [[q0_i0 (1 or 0), q0_i1 (1 or 0), q1_i0 (1 or 0), q1_i1 (1 or 0)], ...]

    The keys "q0_i0", "q0_i1", "q1_i0", "q1_i1" represent combinations of questions and images:
        - "q0_i0" means question_0 on image_0 
        - "q0_i1" means question_0 on image_1 
        - "q1_i0" means question_1 on image_0 
        - "q1_i1" means question_1 on image_1 

    Returns:
        dict: A dictionary containing the calculated scores:
            - 'question_score': Average question score
            - 'image_score': Average image score
            - 'binary_score': Average binary VQA score
            - 'group_score': Average group score
    """
    question_score = 0.0
    image_score = 0.0
    binary_score = 0.0
    group = 0.0

    num_samples = len(scores)

    def calculate_image_score(result):
        image_correct = 0
        if isinstance(result, dict):
            if result["q0_i0"] == 1.0 and result["q1_i0"] == 0.0:
                image_correct += 1
            if result["q1_i1"] == 1.0 and result["q0_i1"] == 0.0:
                image_correct += 1
        elif isinstance(result, list):
            if result[0] == 1.0 and result[2] == 0.0:
                image_correct += 1
            if result[3] == 1.0 and result[1] == 0.0:
                image_correct += 1
        return image_correct
    
    def calculate_question_score(result):
        text_correct = 0
        if isinstance(result, dict):
            if result["q0_i0"] == 1.0 and result["q0_i1"] == 0.0:
                text_correct += 1
            if result["q1_i1"] == 1.0 and result["q1_i0"] == 0.0:
                text_correct += 1
        else:
            if result[0] == 1.0 and result[1] == 0.0:
                text_correct += 1
            if result[3] == 1.0 and result[2] == 0.0:
                text_correct += 1
        return text_correct

    def calculate_binary_score(result):
        binary_score_correct = 0
        if isinstance(result, dict):
            binary_score_correct += 1 if result["q0_i0"] == 1.0 else 0
            binary_score_correct += 1 if result["q0_i1"] == 0.0 else 0
            binary_score_correct += 1 if result["q1_i0"] == 0.0 else 0
            binary_score_correct += 1 if result["q1_i1"] == 1.0 else 0
        else:
            binary_score_correct += 1 if result[0] == 1.0 else 0
            binary_score_correct += 1 if result[1] == 0.0 else 0
            binary_score_correct += 1 if result[2] == 0.0 else 0
            binary_score_correct += 1 if result[3] == 1.0 else 0

        return binary_score_correct

    def calculate_group(result):
        group_correct = 0
        if calculate_question_score(result) == 2 and calculate_image_score(result) == 2:
            group_correct += 1
        
        return group_correct
    
    if isinstance(scores, dict):
        for _, result in scores.items():
            question_score += calculate_question_score(result)
            image_score += calculate_image_score(result)
            binary_score += calculate_binary_score(result)
            group += calculate_group(result)
    else:
        for result in scores:
            question_score += calculate_question_score(result)
            image_score += calculate_image_score(result)
            binary_score += calculate_binary_score(result)
            group += calculate_group(result)

    results = {
        'question_score': question_score / float(num_samples * 2),
        'image_score': image_score / float(num_samples * 2),
        'binary_score': binary_score / float(num_samples * 4),
        'group_score': group / num_samples
    }

    return results


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    #breakpoint()
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    dataset = load_dataset("BaiqiL/NaturalBench")

    ans_file = open(os.path.expanduser(args.answers_file), "w")

    naturalbench = []
    for item in dataset["train"]:
        naturalbench.append([item["Question_0"] + SUFFIX_FOR_VQA[item["Question_Type"]], item["Image_0"], item["Image_0_Question_0"], item['Question_Type']])
        naturalbench.append([item["Question_0"] + SUFFIX_FOR_VQA[item["Question_Type"]], item["Image_1"], item["Image_1_Question_0"], item['Question_Type']])
        naturalbench.append([item["Question_1"] + SUFFIX_FOR_VQA[item["Question_Type"]], item["Image_0"], item["Image_0_Question_1"], item['Question_Type']])
        naturalbench.append([item["Question_1"] + SUFFIX_FOR_VQA[item["Question_Type"]], item["Image_1"], item["Image_1_Question_1"], item['Question_Type']])

    output_file = []
    for i in tqdm(range(len(naturalbench))):
        question, image, answer, question_type = naturalbench[i]

        conv = conv_templates[args.conv_mode].copy()

        prompt = question

        prompt = "" + DEFAULT_IMAGE_TOKEN + '\n' + prompt

        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        
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
        output_file.append(outputs)
    assert len(output_file) == 1900*4
    answers = {}
    number_answered_samples = len(output_file)//4
    for i in range(number_answered_samples):
        answers[i] = {
            "q0_i0": extract_answer(output_file[i*4], naturalbench[i*4][3]),
            "q0_i1": extract_answer(output_file[i*4+1], naturalbench[i*4+1][3]),
            "q1_i0": extract_answer(output_file[i*4+2], naturalbench[i*4+2][3]),
            "q1_i1": extract_answer(output_file[i*4+3], naturalbench[i*4+3][3])
        }

    #5. Calculate the scores
    scores = get_scores(answers)
    print(scores)

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
