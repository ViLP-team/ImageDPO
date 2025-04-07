import argparse
import glob
import json
import os
import random
from random import shuffle  # Fixed typo in 'raondom'

import numpy as np
import tqdm
from PIL import Image

from vilp.utils.image_utils import blur_image, pixelate_image

parser = argparse.ArgumentParser()
parser.add_argument("--corruption_type", type=str, default="semantic")
parser.add_argument(
    "--file_folder",
    type=str,
    default="/nfs/turbo/justincj-turbo/tiangel/improvingVLM/results_vg_7b/",
)
parser.add_argument("--output_file", type=str, default="semantic_corrupted_40k_v2.json")
parser.add_argument("--use_random_blur", type=bool, action="store_true", default=False)
parser.add_argument(
    "--use_random_pixelate", type=bool, action="store_true", default=False
)
parser.add_argument("--gaussian_blur_low", type=int, default=20)
parser.add_argument("--gaussian_blur_high", type=int, default=40)
parser.add_argument("--pixelate_size", type=int, default=32)
parser.add_argument("--num_limit", type=int, default=20000)
args = parser.parse_args()

NUM_LIMIT = args.num_limit
out_json = []
num = 0
wrong_list = []

if args.corruption_type == "semantic":
    json_colelct = glob.glob(
        os.path.join(args.file_folder, "*/*groundingdino*/single_img_QA_DPO_7b.json")
    )
else:
    json_colelct = glob.glob(
        os.path.join(args.file_folder, "*/*/single_img_QA_DPO_7b.json")
    )

shuffle(json_colelct)

while num < NUM_LIMIT and json_colelct:
    json_file = random.choice(json_colelct)  # Randomly select a json file
    try:
        with open(json_file) as f:  # Use context manager for file handling
            data = json.load(f)
    except Exception as e:  # Catch specific exceptions
        wrong_list.append(json_file)
        json_colelct.remove(
            json_file
        )  # Remove the file from the list if there's an error
        print(f"Error loading {json_file}: {e}")
        continue
    shuffle(data)
    for item in data:
        image_path = item["Image Path"]
        responses = {key: item[key] for key in item if key.startswith("Response")}

        high_response = max(
            (
                resp
                for resp in responses.values()
                if isinstance(resp["Score"], (int, str))
            ),
            key=lambda x: int(x["Score"]),
            default=None,
        )

        if high_response is None or int(high_response["Score"]) <= 3:
            continue

        if args.corruption_type == "semantic":
            # load the original image as the bad image
            bad_image_path = image_path.split("/")[:-2]
            bad_image_path = os.path.join(*bad_image_path, "origin.jpg")
            bad_image = np.array(
                Image.open(os.path.join(args.file_folder, bad_image_path)).convert("L")
            )
            good_image = np.array(
                Image.open(os.path.join(args.file_folder, image_path)).convert("L")
            )

            bad_image = np.float32(bad_image)
            good_image = np.float32(good_image)
            image_diff = np.abs(bad_image - good_image)
            mask = image_diff > 20
            significant_diff_pixels = np.sum(mask)
            if (
                significant_diff_pixels
                < good_image.shape[0] * good_image.shape[1] * 0.1
            ):
                continue
        else:

            bad_image = Image.open(os.path.join(args.file_folder, image_path))
            bad_image_path = os.path.join(
                image_path[:-4]
                + f"_Pixelate_{args.use_random_pixelate}_{args.pixelate_size}_Blur_{args.use_random_blur}_{args.gaussian_blur_low}_{args.gaussian_blur_high}.jpg"
            )
            gaussian_blur = (
                random.randint(args.gaussian_blur_low, args.gaussian_blur_high)
                if args.use_random_blur
                else args.gaussian_blur_low
            )

            if args.use_random_blur and args.use_random_pixelate:
                bad_image = (
                    blur_image(bad_image, gaussian_blur)
                    if random.random() < 0.5
                    else pixelate_image(bad_image, args.pixelate_size)
                )
            elif args.use_random_blur:
                bad_image = blur_image(bad_image, gaussian_blur)
            elif args.use_random_pixelate:
                bad_image = pixelate_image(bad_image, args.pixelate_size)
            bad_image.save(os.path.join(args.file_folder, bad_image_path))
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

        dpo_item["bad_image"] = bad_image_path
        out_json.append(dpo_item)
        num += 1
        break  # Break to randomly select the next json file
    if num % 1000 == 0:
        print(f"Processed {num} images")


json.dump(out_json, open(f"{args.output_file}", "w"), indent=4)
