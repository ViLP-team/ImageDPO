import argparse
import json
import os

from openai import OpenAI
import openai
import time
from IPython import embed
import base64
from PIL import Image
from io import BytesIO
import numpy as np
import re
import tqdm

client=OpenAI(api_key='sk-proj-ahFJh9J8yJqey7sLweKST3BlbkFJK5MOQ7aBvkIHlG3brSao')

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
        return base64.b64encode(image_file.read()).decode('utf-8')

def resize_and_compress_image(image_path, target_width=128, target_height=128, quality=40):
    # Open the image file
    with Image.open(image_path) as img:
        # Resize the image
        img = img.resize((target_width, target_height), Image.ANTIALIAS)
        
        # Save the image to a BytesIO object
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=quality)  # You can adjust the quality as needed (0-100)
        buffer.seek(0)
        
        # Encode the image to base64
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return img_base64

def get_eval(content: str, max_tokens: int, source_image_path: str, target_image_path: str):
    try:
       source_image_base64 = resize_and_compress_image(source_image_path)
       target_image_base64 = resize_and_compress_image(target_image_path)
    except:
       return
    
    try:
        response = client.chat.completions.create(
            #model="gpt-4-turbo",  # Use the appropriate multi-modal model
            #model="gpt-3.5-turbo-0125",  # Use the appropriate multi-modal model
            #model="gpt-3.5-turbo-0125",  # Use the appropriate multi-modal model
            model="gpt-4o-2024-05-13",  # Use the appropriate multi-modal model
            messages=[
                #{"role": "system", "content": "You are a helpful and precise assistant for checking the quality of the answer."},
                {"role": "user", "content": content},
                {"role": "user", "content": f"First Image: {source_image_base64}"},
                {"role": "user", "content": f"Second Image: {target_image_base64}"}
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
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ChatGPT-based QA evaluation.')
    parser.add_argument('-o', '--output', default='MagicBrush.jsonl', help='output file to save the generated questions')
    parser.add_argument('--max-tokens', type=int, default=1024, help='maximum number of tokens produced in the output')
    args = parser.parse_args()

    review_file = open(f'{args.output}', 'a')
    if os.path.exists(f'{args.output}'):
        current_data = read_jsonl_file(f'{args.output}')
    image_dir = '/nfs/turbo/justincj-turbo/tiangel/improvingVLM/packages/LLaVA-Plus/playground/eval_data/two_img_dataset/MagicBrush/images'
    score_results = []
    instructions = json.load(open('/nfs/turbo/justincj-turbo/tiangel/improvingVLM/packages/LLaVA-Plus/playground/eval_data/two_img_dataset/MagicBrush/instruction_1k.json'))
    all_keys = list(instructions.keys())
    exist_keys = [x['id'] for x in current_data]


    for cur_key in all_keys[:10]:
        if cur_key in exist_keys:
            continue
        source_img = os.path.join(image_dir, cur_key + '_source.png') 
        target_img = os.path.join(image_dir, cur_key + '_target.png')

        content = f"""
        Given two images, please propose some questions along with their corresponding answers. The questions should require information from both images to be answered. Avoid questions such as "What's the difference between the images?" or "How can we modify image A to resemble image B?" Accuracy in the answers is the highest priority.

        Expected Response Format:
        Item Number: 1
        Question: Propose a question
        Answer: Corresponding answer

        Item Number: 2
        Question: Propose a question
        Answer: Corresponding answer
        """

        cur_js = {
            'id': cur_key,
        }
        review = get_eval(content, args.max_tokens, source_img, target_img)
        cur_js['content'] = review
        try:
            _, cur_js['QA'] = extract_QA(review)
            print(cur_js['QA'])
        except:
            print('parse error!!!!!!!!!!!!!!!') 
            print(review)
            continue
        review_file.write(json.dumps(cur_js) + '\n')
        review_file.flush()
    review_file.close()