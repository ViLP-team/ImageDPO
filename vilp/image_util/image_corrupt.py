import glob
import json
import os
import random
import uuid
import tqdm
from ivlm.utils.image_utils import blur_image, pixelate_image
from PIL import Image
from random import shuffle  # Fixed typo in 'raondom'


# TODO: Change this to the folder you want to save the corrupted images to
FILE_FOLDER = ""

# TODO: Change this to the folder you want to save the corrupted images to
jsons = glob.glob('/Your/Path/to/generated/images/*/*/single_img_QA_DPO_7b.json')

NUM_LIMIT = 40000
out_json = []
num = 0
wrong_list = []
shuffle(jsons)

GAUSSIAN_BLUR = 40
GAUSSIAN_BLUR_LOW = 10
GAUSSIAN_BLUR_HIGH = 40
RANDOM_BLUR = False
PIXELATE = 16
USE_PIXELATE = True 
USE_RANDOM_BLUR = False

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

        bad_image = Image.open(os.path.join(FILE_FOLDER, image_path))
        bad_image_path = os.path.join(image_path[:-4] + f"_Pixelate_{USE_PIXELATE}_{PIXELATE}_Blur_{USE_RANDOM_BLUR}_{GAUSSIAN_BLUR}_{str(uuid.uuid4())[:5]}.jpg")
        if RANDOM_BLUR: 
            gaussian_blur = random.randint(GAUSSIAN_BLUR_LOW, GAUSSIAN_BLUR_HIGH)
        else:
            gaussian_blur = GAUSSIAN_BLUR

        if USE_RANDOM_BLUR and USE_PIXELATE:
            bad_image = blur_image(bad_image, gaussian_blur) if random.random() < 0.5 else pixelate_image(bad_image, PIXELATE)
        elif USE_RANDOM_BLUR and not USE_PIXELATE:
            bad_image = blur_image(bad_image, gaussian_blur)
        elif USE_PIXELATE and not USE_RANDOM_BLUR:
            bad_image = pixelate_image(bad_image, PIXELATE)
        bad_image.save(os.path.join(FILE_FOLDER, bad_image_path))
        dpo_item['bad_image'] = bad_image_path
        out_json.append(dpo_item)
        num += 1
        break # Remove this line to process all images
    if num % 1000 == 0:
        print(f"Processed {num} images")
    if num >= NUM_LIMIT:
        break
if USE_RANDOM_BLUR and USE_PIXELATE:
    if RANDOM_BLUR:
        json.dump(out_json, open(f"image_dpo_llava_7b_text2vqa_{RANDOM_BLUR}_{GAUSSIAN_BLUR}_{PIXELATE}_corrupted.json", "w"), indent=4)
    else:
        json.dump(out_json, open(f"image_dpo_llava_7b_text2vqa_{GAUSSIAN_BLUR}_{PIXELATE}_corrupted.json", "w"), indent=4)
elif USE_RANDOM_BLUR and not USE_PIXELATE:
    if RANDOM_BLUR:
        json.dump(out_json, open(f"image_dpo_llava_7b_text2vqa_{RANDOM_BLUR}_{GAUSSIAN_BLUR}_blur_only_corrupted.json", "w"), indent=4)
    else:
        json.dump(out_json, open(f"image_dpo_llava_7b_text2vqa_{GAUSSIAN_BLUR}_blur_only_corrupted.json", "w"), indent=4)
elif USE_PIXELATE and not USE_RANDOM_BLUR:
    json.dump(out_json, open(f"image_dpo_llava_7b_text2vqa_{PIXELATE}_pixelate_only_corrupted.json", "w"), indent=4)
