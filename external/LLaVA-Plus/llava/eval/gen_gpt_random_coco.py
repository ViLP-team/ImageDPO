import argparse

# from llava.eval.gen_gpt_ERL import extract_QA, encode_image_to_base64, resize_and_compress_image, get_eval, read_jsonl_file
import base64
import json
import os
import pickle
import random
import re
import shutil
import subprocess
from io import BytesIO

import requests
import torch
from cog import Path
from IPython import embed
from PIL import Image

from ivlm.model_warper.llava_warper import LLaVA_Warper
from ivlm.utils.re_utils import generate_instruction
from openai import OpenAI

PROFILE = False
client = OpenAI(api_key="sk-proj-jVzf5vY0p5iX4I49aZpHT3BlbkFJQX4uv2Mtsdlhq8I7pD0n")

PAIR_NUM = 1000
COCO_IMAGE_BASE = "/nfs/turbo/justincj-turbo/shared_datasets/coco"


def concatenate_images(
    image_path1, image_path2, output_path, target_width=128, target_height=128
):
    """
    Concatenates two images side by side.

    :param image_path1: Path to the first image.
    :param image_path2: Path to the second image.
    :param output_path: Path to save the concatenated image.
    """
    # Open the images
    img1 = Image.open(image_path1)
    img1 = img1.resize((target_width, target_height), Image.LANCZOS)
    img2 = Image.open(image_path2)
    img2 = img2.resize((target_width, target_height), Image.LANCZOS)

    # Find the height of the combined image
    max_height = max(img1.height, img2.height)

    # Create a new image with the width of both images combined and the max height
    combined_width = img1.width + img2.width
    new_img = Image.new("RGB", (combined_width, max_height))

    # Paste the images into the new image
    new_img.paste(img1, (0, 0))
    new_img.paste(img2, (img1.width, 0))

    # Save the new image
    new_img.save(output_path)


# Shouldn't copy-paste these functions, but it is helpful for debugging
def extract_QA(results_QA):
    count = 0
    QA_save_dict = []

    pattern_QA = re.compile(
        r"Item Number: (?P<Number>\d+)\nQuestion: (?P<Question>.+?)\nAnswer: (?P<Answer>.+?)(?=\n\nItem Number: |\Z)",
        re.DOTALL,
    )
    matches = pattern_QA.finditer(results_QA)
    for match in matches:
        try:
            question = match.group("Question").strip()
            answer = match.group("Answer").strip()
            QA_save_dict.append({"Question": question, "Answer": answer})
            count += 1
        except:
            continue

    if count <= 1:
        count = 0
        pattern_QA = re.compile(
            r"Question: (?P<Question>.+?)\nAnswer: (?P<Answer>.*?)(?=\n\n\d+\. Question: |\Z)",
            re.DOTALL,
        )
        matches = pattern_QA.finditer(results_QA)
        for match in matches:
            try:
                question = match.group("Question").strip()
                answer = match.group("Answer").strip()
                QA_save_dict.append({"Question": question, "Answer": answer})
                count += 1
            except:
                continue

    if count <= 1:
        count = 0
        pattern_QA = re.compile(
            r"(?<=\d\.\s)(?:What's|What|Where|How).+?\?\nAnswer: .+?(?=\n\d\. |\Z)",
            re.DOTALL,
        )
        matches = pattern_QA.finditer(results_QA)
        for match in matches:
            try:
                qa_pair = match.group(0).strip().split("\nAnswer: ")
                question = qa_pair[0].split("?")[0].strip() + "?"
                answer = qa_pair[1].strip()
                QA_save_dict.append({"Question": question, "Answer": answer})
                count += 1
            except:
                continue

    return count, QA_save_dict


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def resize_and_compress_image(
    image_path, target_width=128, target_height=128, quality=40
):
    # Open the image file
    with Image.open(image_path) as img:
        # Resize the image
        img = img.resize((target_width, target_height), Image.LANCZOS)

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
    try:
        source_image_base64 = resize_and_compress_image(source_image_path)
        target_image_base64 = resize_and_compress_image(target_image_path)
    except:
        return

    # try:
    #     response = client.chat.completions.create(
    #         # model="gpt-4-turbo",  # Use the appropriate multi-modal model
    #         #model="gpt-3.5-turbo-0125",  # Use the appropriate multi-modal model
    #         #model="gpt-3.5-turbo-0125",  # Use the appropriate multi-modal model
    #         model="gpt-4o-2024-05-13",  # Use the appropriate multi-modal model
    #         messages=[
    #             #{"role": "system", "content": "You are a helpful and precise assistant for checking the quality of the answer."},
    #             {"role": "user", "content": content},
    #             {"role": "user", "content": f"First Image: {source_image_base64}"},
    #             {"role": "user", "content": f"Second Image: {target_image_base64}"}
    #         ],
    #         temperature=0.2,
    #         # max_tokens=max_tokens
    #     )
    try:
        response = client.chat.completions.create(
            # model="gpt-4-turbo",  # Use the appropriate multi-modal model
            # model="gpt-3.5-turbo-0125",  # Use the appropriate multi-modal model
            # model="gpt-3.5-turbo-0125",  # Use the appropriate multi-modal model
            model="gpt-4o-2024-05-13",  # Use the appropriate multi-modal model
            messages=[
                # {"role": "system", "content": "You are a helpful and precise assistant for checking the quality of the answer."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": content},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg; base64, {source_image_base64}"
                            },
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg; base64, {target_image_base64}"
                            },
                        },
                    ],
                }
            ],
            temperature=0.2,
            # max_tokens=max_tokens
        )
    except Exception as e:
        print(e)
        return

    return response.choices[0].message.content


def read_jsonl_file(filepath):
    data = []
    with open(filepath) as f:
        for line in f:
            data.append(json.loads(line))
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChatGPT-based QA evaluation.")
    parser.add_argument(
        "-o",
        "--output",
        default="/nfs/turbo/justincj-turbo/tiangel/improvingVLM/packages/LLaVA-Plus/playground/eval_data/two_img_dataset/COCO/gpt_qas_COCO.jsonl",
        help="output file to save the generated questions",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="maximum number of tokens produced in the output",
    )
    args = parser.parse_args()

    review_file = open(f"{args.output}", "a")
    if os.path.exists(f"{args.output}"):
        current_data = read_jsonl_file(f"{args.output}")
        exist_keys = [x["id"] for x in current_data]
    else:
        exist_keys = []

    # use test_path
    test_path = pickle.load(
        open("/nfs/turbo/justincj-turbo/shared_datasets/coco/test_path.pkl", "rb")
    )

    pairs = []
    img_index_number = list(range(len(test_path)))

    # visualization image save path
    img_vis_path = os.path.join(os.path.dirname(args.output), "img")
    if not os.path.exists(img_vis_path):
        os.makedirs(img_vis_path)
    while len(pairs) < PAIR_NUM:
        pair = random.sample(img_index_number, 2)
        if pair not in pairs and pair[::-1] not in pairs:
            pairs.append(pair)
    # get images

    for i in range(len(pairs)):

        # The id is the tuple of image index
        if [pairs[i][0], pairs[i][1]] in exist_keys:
            continue

        source_img = os.path.join(COCO_IMAGE_BASE, test_path[pairs[i][0]])
        target_img = os.path.join(COCO_IMAGE_BASE, test_path[pairs[i][1]])

        content = f"""
        Given two images, please propose some questions along with their corresponding answers. The questions should require information from both images to be answered. Avoid questions such as "What's the difference between the images?" or "How can we modify image A to resemble image B?" Accuracy in the answers is the highest priority.

        Rember to keep the questions and answers high-quality and creative, while maintaining precise and concise.
        Again, questions should require information from both images to be answered. It is important. 
        Reviewing the questions and answers, and only output the first two question and answers with the highest quality.
        Expected Response Format:
        Item Number: 1
        Question: Propose a question
        Answer: Corresponding answer

        Item Number: 2
        Question: Propose a question
        Answer: Corresponding answer
        """

        cur_js = {
            "id": [pairs[i][0], pairs[i][1]],
        }

        # save image for visualzation
        cur_js["image_path"] = [source_img, target_img]
        review = get_eval(content, args.max_tokens, source_img, target_img)
        cur_js["content"] = review
        concatenate_images(
            source_img,
            target_img,
            os.path.join(img_vis_path, f"{pairs[i][0]}_{pairs[i][1]}.jpg"),
        )

        try:
            _, cur_js["QA"] = extract_QA(review)
            print(cur_js["QA"])
        except:
            print("parse error!!!!!!!!!!!!!!!")
            print(review)
            continue
        review_file.write(json.dumps(cur_js) + "\n")
        review_file.flush()
    review_file.close()
