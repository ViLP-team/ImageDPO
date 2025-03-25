import argparse
import base64
import json
import os
import re
import time
from io import BytesIO

import numpy as np
import openai
from IPython import embed
from openai import OpenAI
from PIL import Image

NUM_SECONDS_TO_SLEEP = 0.5

client = OpenAI(api_key="sk-proj-ahFJh9J8yJqey7sLweKST3BlbkFJK5MOQ7aBvkIHlG3brSao")


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def resize_and_compress_image(
    image_path, target_width=256, target_height=256, quality=40
):
    # Open the image file
    with Image.open(image_path) as img:
        # Resize the image
        img = img.resize((target_width, target_height), Image.ANTIALIAS)

        # Save the image to a BytesIO object
        buffer = BytesIO()
        img.save(
            buffer, format="JPEG", quality=quality
        )  # You can adjust the quality as needed (0-100)
        buffer.seek(0)

        # Encode the image to base64
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return img_base64


def get_eval(
    content: str, max_tokens: int, source_image_path: str, target_image_path: str
):
    # try:
    #    source_image_base64 = resize_and_compress_image(source_image_path)
    #    target_image_base64 = resize_and_compress_image(target_image_path)
    # except:
    #    return

    try:
        response = client.chat.completions.create(
            # model="gpt-4-turbo",  # Use the appropriate multi-modal model
            # model="gpt-3.5-turbo-0125",  # Use the appropriate multi-modal model
            # model="gpt-3.5-turbo-0125",  # Use the appropriate multi-modal model
            model="gpt-4o-2024-05-13",  # Use the appropriate multi-modal model
            messages=[
                # {"role": "system", "content": "You are a helpful and precise assistant for checking the quality of the answer."},
                {"role": "user", "content": content},
                # {"role": "user", "content": f"Source Image: {source_image_base64}"},
                # {"role": "user", "content": f"Target Image: {target_image_base64}"}
            ],
            temperature=0.2,
            # max_tokens=max_tokens
        )
    except Exception as e:
        print(e)
        return

    return response.choices[0].message.content


def extract_score(text):
    score_pattern = re.compile(r"Score: (\d+)")
    match = score_pattern.search(text)

    if match:
        score = int(match.group(1))
    else:
        print("No score found.")
    return score


def parse_score(review):
    try:
        score_pair = review.split("\n")[0]
        score_pair = score_pair.replace(",", " ")
        sp = score_pair.split(" ")
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            print("error", review)
            return [-1, -1]
    except Exception as e:
        print(e)
        print("error", review)
        return [-1, -1]


# def read_jsonl_file(filepath):
#     data = []
#     with open(filepath, 'r') as f:
#         for line in f:
#             data.append(json.loads(line))
#     return data


def read_jsonl_file(filepath):
    data = []
    with open(filepath) as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                print("error", line)
                continue
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChatGPT-based QA evaluation.")
    parser.add_argument("-a", "--answer")
    parser.add_argument("-o", "--output")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="maximum number of tokens produced in the output",
    )
    args = parser.parse_args()

    review_file = open(f"{args.output}", "a")
    # generated_instructions = json.load(open(args.answer))
    breakpoint()
    generated_instructions = read_jsonl_file(args.answer)
    image_dir = "/nfs/turbo/justincj-turbo/tiangel/improvingVLM/packages/LLaVA-Plus/playground/eval_data/two_img_dataset/ERL/images"
    score_results = []

    for item in generated_instructions[:100]:
        cur_gt_instruction = item["prompt"]
        cur_key = item["question_id"]
        cur_gen_instruction = item["text"]
        source_img = os.path.join(image_dir, cur_key + "_source.png")
        target_img = os.path.join(image_dir, cur_key + "_target.png")

        content = f"""
        We are conducting an evaluation of an AI assistant's ability to provide answer for a piar of images. The AI assistant was given a original image and a modified image and tasked with predicting "What changes were made to the original image to create the modified image?".
        
        The AI assistant provided the following answer: {cur_gen_instruction}
        The ground-truth instruction is: {cur_gt_instruction}
        
        Please provide your feedback by rating the AI assistant's performance on a scale of 0 to 5, where 5 signifies excellent performance and 0 represents the provided instruction is completely incorrect. Since the groud-truth instruction is usually short and the AI assistant's answer contains usually more details, please rate the AI assistant's answer based on the quality of the answer, not the length of the answer. Sometimes, the correct instruction could be inferred by the AI assistant's answer.
        
        Expected Response Format:
        Score: Rate the performance on a scale of 0 to 5.
        """
        # - On the following line, provide a explanation for your rating.

        cur_js = {
            "id": cur_key,
        }
        review = get_eval(content, args.max_tokens, source_img, target_img)
        cur_js["content"] = review
        try:
            cur_js["score"] = extract_score(review)
        except:
            print("parse error!!!!!!!!!!!!!!!")
            print(review)
            continue
        print(cur_js["score"])
        score_results.append(cur_js["score"])
        review_file.write(json.dumps(cur_js) + "\n")
        review_file.flush()
    review_file.close()

print("ERL final score:", np.mean(score_results) * 20, args.answer)
