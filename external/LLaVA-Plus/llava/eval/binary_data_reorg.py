import csv
import json
import os
import shutil

OUTPUT_FOLDER = "/home/ancao/src/turbo/repos/ImprovingVLM/eval_data/two_img_dataset/collect_binary_dataset"

index = 0
data_collect = []

if os.path.exists(OUTPUT_FOLDER) is False:
    os.makedirs(OUTPUT_FOLDER)


def read_jsonl_file(filepath):
    data = []
    with open(filepath) as f:
        for line in f:
            data.append(json.loads(line))
    return data


# loaded data
# 1. MagicBrush

instructions = json.load(
    open(
        "/nfs/turbo/justincj-turbo/tiangel/improvingVLM/packages/LLaVA-Plus/playground/eval_data/two_img_dataset/MagicBrush/instruction_1k.json"
    )
)
image_dir = "/nfs/turbo/justincj-turbo/tiangel/improvingVLM/packages/LLaVA-Plus/playground/eval_data/two_img_dataset/MagicBrush/images"

instructions_keylist = list(instructions.keys())

for cur_key in instructions_keylist:
    source_path = os.path.join(image_dir, cur_key + "_source.png")
    target_path = os.path.join(image_dir, cur_key + "_target.png")

    dest_source_path = os.path.join(OUTPUT_FOLDER, f"{index:06}_1.png")
    dest_target_path = os.path.join(OUTPUT_FOLDER, f"{index:06}_2.png")

    shutil.copy(source_path, dest_source_path)
    shutil.copy(target_path, dest_target_path)

    cur_instruction = instructions[cur_key]

    data_collect.append(
        {
            "question_id": index,
            "question": "What elements were added or removed in the new image",
            "img_1": f"{index:06}_1.png",
            "img_2": f"{index:06}_2.png",
            "answer": cur_instruction,
            "source": "MagicBrush",
        }
    )
    index += 1


# 2. Spot-the-diff

loaded_data = json.load(
    open(
        os.path.join(
            "/nfs/turbo/justincj-turbo/tiangel/improvingVLM/packages/LLaVA-Plus/playground/eval_data/two_img_dataset/spot-the-diff",
            "data/annotations/test.json",
        )
    )
)
image_dir = os.path.join(
    "/nfs/turbo/justincj-turbo/tiangel/improvingVLM/packages/LLaVA-Plus/playground/eval_data/two_img_dataset/spot-the-diff",
    "resized_images",
)

for example in loaded_data:
    img_idx = example["img_id"]
    diff_description = example["sentences"]

    source_path = os.path.join(image_dir, f"{img_idx}.png")
    target_path = os.path.join(image_dir, f"{img_idx}_2.png")
    dest_source_path = os.path.join(OUTPUT_FOLDER, f"{index:06}_1.png")
    dest_target_path = os.path.join(OUTPUT_FOLDER, f"{index:06}_2.png")

    shutil.copy(source_path, dest_source_path)
    shutil.copy(target_path, dest_target_path)
    data_collect.append(
        {
            "question_id": index,
            "question": "What elements were added or removed in the new image",
            "img_1": f"{index:06}_1.png",
            "img_2": f"{index:06}_2.png",
            "answer": diff_description,
            "source": "Spot-the-diff",
        }
    )
    index += 1

# 3. EGO4D, COCO, VCR_img, VCR_video
instruction_path_all = {
    "ego4d": "/nfs/turbo/justincj-turbo/tiangel/improvingVLM/packages/LLaVA-Plus/playground/eval_data/two_img_dataset/EGO4D/gpt_qsa_ego4d.jsonl",
    "coco": "/nfs/turbo/justincj-turbo/tiangel/improvingVLM/packages/LLaVA-Plus/playground/eval_data/two_img_dataset/COCO/gpt_qas_COCO.jsonl",
    "vcr_img": "/nfs/turbo/justincj-turbo/tiangel/improvingVLM/packages/LLaVA-Plus/playground/eval_data/two_img_dataset/VCR/vcr_img/gpt_qsa_vcr.jsonl",
    "vcr_video": "/nfs/turbo/justincj-turbo/tiangel/improvingVLM/packages/LLaVA-Plus/playground/eval_data/two_img_dataset/VCR/vcr_video/gpt_qsa_vcr.jsonl",
    "activity": "/nfs/turbo/justincj-turbo/tiangel/improvingVLM/packages/LLaVA-Plus/playground/eval_data/two_img_dataset/ActivityDataset/gpt_qsa_activity.jsonl",
}

for dataset_name in ["ego4d", "coco", "vcr_img", "vcr_video", "activity"]:
    instruction_path = instruction_path_all[dataset_name]
    instructions = read_jsonl_file(instruction_path)
    for item in instructions:
        source_path = item["image_path"][0]
        target_path = item["image_path"][1]
        dest_source_path = os.path.join(OUTPUT_FOLDER, f"{index:06}_1.png")
        dest_target_path = os.path.join(OUTPUT_FOLDER, f"{index:06}_2.png")

        shutil.copy(source_path, dest_source_path)
        shutil.copy(target_path, dest_target_path)
        question = item["QA"][0]["Question"]
        answer = item["QA"][0]["Answer"]
        data_collect.append(
            {
                "question_id": index,
                "question": question,
                "img_1": f"{index:06}_1.png",
                "img_2": f"{index:06}_2.png",
                "answer": answer,
                "source": dataset_name,
            }
        )
        index += 1

# 4. IER

instructions = json.load(
    open(
        "/nfs/turbo/justincj-turbo/tiangel/improvingVLM/packages/LLaVA-Plus/playground/eval_data/two_img_dataset/ERL/train.json"
    )
)
image_dir = "/nfs/turbo/justincj-turbo/tiangel/improvingVLM/packages/LLaVA-Plus/playground/eval_data/two_img_dataset/ERL/images"
for item in instructions:
    source_path = os.path.join(image_dir, item["img0"])
    target_path = os.path.join(image_dir, item["img1"])
    cur_instruction = item["sents"][0]

    dest_source_path = os.path.join(OUTPUT_FOLDER, f"{index:06}_1.png")
    dest_target_path = os.path.join(OUTPUT_FOLDER, f"{index:06}_2.png")

    shutil.copy(source_path, dest_source_path)
    shutil.copy(target_path, dest_target_path)
    data_collect.append(
        {
            "question_id": index,
            "question": "What changes were made to the original image to create the modified image?",
            "img_1": f"{index:06}_1.png",
            "img_2": f"{index:06}_2.png",
            "answer": cur_instruction,
            "source": "IER",
        }
    )
    index += 1

fields = data_collect[0].keys()
with open(os.path.join("output_collect.csv"), mode="w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(data_collect)
