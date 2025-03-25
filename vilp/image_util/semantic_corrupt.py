import glob
import json
import os
import random
import tqdm
from ivlm.utils.image_utils import blur_image, pixelate_image
from PIL import Image
from random import shuffle  # Fixed typo in 'raondom'
import numpy as np
import argparse

# TODO: Change this to the folder you want to save the corrupted images to
FILE_FOLDER = ""


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="llava7b")
parser.add_argument("--dataset_name", type=str, default="text2vqa")
parser.add_argument("--num_limit", type=int, default=20000)
args = parser.parse_args()

if args.model_name == "llava7b":
    name_prefix = f"image_dpo_llava_7b_{args.dataset_name}"
    if args.dataset_name == "text2vqa":
        jsons = glob.glob('/nfs/turbo/justincj-turbo/tiangel/improvingVLM/results_text2vqa/*/*groundingdino*/single_img_QA_DPO_7b.json')
    elif args.dataset_name == "coco":
        jsons = glob.glob('/nfs/turbo/justincj-turbo/tiangel/improvingVLM/results_10000_100k/*/*groundingdino*/single_img_QA_DPO_7b.json')
    elif args.dataset_name == "vg":
        jsons = glob.glob('/nfs/turbo/justincj-turbo/tiangel/improvingVLM/results_vg_7b/*/*groundingdino*/single_img_QA_DPO_7b.json')
    else:
        raise ValueError(f"Dataset {args.dataset_name} not supported for model {args.model_name}")
elif args.model_name == "llava13b":
    name_prefix = f"image_dpo_llava_13b_{args.dataset_name}"
    if args.dataset_name == "text2vqa":
        jsons = glob.glob('/nfs/turbo/justincj-turbo/tiangel/improvingVLM/results_text2vqa_13b/*/*groundingdino*/single_img_QA_DPO_13b.json')
    elif args.dataset_name == "coco":
        jsons =  glob.glob('/nfs/turbo/justincj-turbo/tiangel/improvingVLM/results_10000_100k/*/*groundingdino*/single_img_QA_DPO.json')
    elif args.dataset_name == "vg":
        jsons = glob.glob("/nfs/turbo/justincj-turbo/tiangel/improvingVLM/results_vg_13b/*/*groundingdino*/single_img_QA_DPO_13b.json")
    else:
        raise ValueError(f"Dataset {args.dataset_name} not supported for model {args.model_name}")
else:
    raise ValueError(f"Model {args.model_name} not supported")

NUM_LIMIT = args.num_limit
out_json = []
num = 0
wrong_list = []
shuffle(jsons)

while num < NUM_LIMIT and jsons:
    json_file = random.choice(jsons)  # Randomly select a json file
    try:
        with open(json_file) as f:  # Use context manager for file handling
            data = json.load(f)
    except Exception as e:  # Catch specific exceptions
        wrong_list.append(json_file)
        jsons.remove(json_file)  # Remove the file from the list if there's an error
        continue
    shuffle(data)
    for item in data:
        image_path = item["Image Path"]
        responses = {key: item[key] for key in item if key.startswith("Response")}

        high_response = max(
            (resp for resp in responses.values() if isinstance(resp["Score"], (int, str))),
            key=lambda x: int(x["Score"]),
            default=None
        )
        
        if high_response is None or int(high_response["Score"]) <= 3:
            continue

        # load the original image as the bad image
        bad_image_path = image_path.split("/")[:-2]
        bad_image_path = os.path.join(*bad_image_path, "origin.jpg")
        bad_image = np.array(Image.open(os.path.join(FILE_FOLDER, bad_image_path)).convert('L'))
        good_image = np.array(Image.open(os.path.join(FILE_FOLDER, image_path)).convert('L'))

        bad_image = np.float32(bad_image)
        good_image = np.float32(good_image)
        image_diff = np.abs(bad_image - good_image)
        mask = image_diff > 20
        significant_diff_pixels = np.sum(mask)
        if significant_diff_pixels < good_image.shape[0] * good_image.shape[1] * 0.1:
            continue

        dpo_item = {
            "good_image": image_path,
            "question": {
                "from": "human",
                "value": "<image>\n" + item["Question"],
            },
            "chosen": high_response["Answer"],
            "rejected": high_response["Answer"],
            "image": image_path,
        }

        dpo_item['bad_image'] = bad_image_path
        out_json.append(dpo_item)
        num += 1
        break  # Break to randomly select the next json file
    if num % 1000 == 0:
        print(f"Processed {num} images")
# json.dump(out_json, open(f"image_dpo_llava_7b_text2vqa_semantic_corrupted_40k_v2.json", "w"), indent=4)

json.dump(out_json, open(f"{name_prefix}_semantic_corrupted_40k_v2.json", "w"), indent=4)