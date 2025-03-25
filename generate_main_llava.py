# This should have the same functionality as packages/LLaVA-Plus/generate_data_coco.py while we
# put this out of LLaVA package.

import glob
import json
import os
import pickle
import random
import re
import shutil
import subprocess

import requests
import torch
from cog import Path
from IPython import embed

from vilp.model_wrapper.vlm_wrapper import VLM_wrapper
from vilp.utils.re_utils import generate_instruction

# TODO: Change the parameters as needed
PROFILE = True
BATCH_SIZE = 5
IF_7B = False
DATASET_NAME = "vg"
# OUTPUT_PATH = os.path.join("/your/path/to/output", f"{DATASET_NAME}_output")
OUTPUT_PATH = "./results"


@torch.no_grad()
def main(name=None):
    if DATASET_NAME == "coco":
        cur_train_path = glob.glob("/your/path/to/images/*")

        # )[:1000]
    elif DATASET_NAME == "textvqa":
        cur_train_path = glob.glob("/your/path/to/images/*")
    elif DATASET_NAME == "vg":
        cur_train_path = glob.glob(
            "/nfs/turbo/justincj-turbo/shared_datasets/visual_genome/VG_100K_2/*"
        )
    else:
        raise ValueError("Unknown dataset")
    random.shuffle(cur_train_path)

    llava_model = VLM_wrapper(
        model_type="llava",
        checkpoint_path="liuhaotian/llava-v1.5-7b",
        model_name="llava-v1.5-7b",
        conv_mode="llama_3",
    )

    if PROFILE:
        # enable timer
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

    prompt_text = """Given this image, please suggest a range of creative edits, tasks, or transformations that could be applied using advanced image processing tools. These tasks may include artistic transformations, vivid color adjustments, object detection and modification, or completely creating a new image inspired by the original. Specify which tool would be best suited for each task, choosing from Stable Diffusion for image generation, InstructPix2Pix for image modification, or GroundingDINO for object modification.
            Your recommendations should help in understanding the potential of the image and exploring creative possibilities.

            Expected Response Format:
            Item Number: 1
            Tool Used: [Specify the tool - Stable Diffusion or InstructPix2Pix or GroundingDINO]
            Text Prompt for Processing: [Detailed description of the task or transformation to be performed. For image generation, please provide complete description based on the understanding of the provided images, since we only feed text prompt for this task.] 

            Item Number: 2
            Tool Used: [Specify the tool - Stable Diffusion or InstructPix2Pix or GroundingDINO]
            Text Prompt for Processing: [Detailed description of the task or transformation to be performed. For image generation, please provide complete description based on the understanding of the provided images, since we only feed text prompt for this task.] 

            Item Number: 3
            Tool Used: [Specify the tool - Stable Diffusion or InstructPix2Pix or GroundingDINO]
            Text Prompt for Processing: [Detailed description of the task or transformation to be performed. For image generation, please provide complete description based on the understanding of the provided images, since we only feed text prompt for this task.] 
            """
    caption_prompt = "describe the image in details"
    batch_accumulator = 0
    image_path_collect = []
    save_dir_collect = []
    instruction_prompt_collect = []
    caption_prompt_collect = []
    batch_count = 0
    for folder in cur_train_path:
        save_dir = os.path.join(
            OUTPUT_PATH,
            os.path.basename(folder).split(".")[0],
        )
        if IF_7B:
            if os.path.exists(os.path.join(save_dir, "all_instructions_7b.json")):
                continue
        else:
            if os.path.exists(os.path.join(save_dir, "all_instructions.json")):
                continue
        save_dir_collect.append(save_dir)
        os.makedirs(save_dir, exist_ok=True, mode=0o777)

        image_path = folder

        image_path_collect.append(image_path)
        instruction_prompt_collect.append(prompt_text)
        caption_prompt_collect.append(caption_prompt)
        batch_accumulator += 1
        if batch_accumulator < BATCH_SIZE:
            continue
        batch_count += 1

        if PROFILE:
            start.record()
        # Do the batch QA
        try:
            result_generator = llava_model.predict_nostreaming(
                image=image_path_collect,
                prompt=instruction_prompt_collect,
                do_sample=True,
                top_p=0.95,
                temperature=0.5,
                max_tokens=1024,
            )  # we do the matching and saving per image
        except:
            batch_accumulator = 0
            image_path_collect = []
            save_dir_collect = []
            instruction_prompt_collect = []
            caption_prompt_collect = []
            continue
        for i in range(len(result_generator)):
            _, parsed_results = generate_instruction(result_generator[i])

            if len(parsed_results) == 0:
                print(result_generator[i])
                print(f"{i}   !!!!!!!!!!!!!!!!!!!!!!!!")
                continue

            if not IF_7B:
                shutil.copyfile(
                    image_path_collect[i],
                    os.path.join(save_dir_collect[i], "origin.jpg"),
                )

            if len(save_dir_collect[i]) > 0:
                if IF_7B:
                    with open(
                        os.path.join(save_dir_collect[i], "all_instructions_7b.json"),
                        "w",
                    ) as f:
                        json.dump(parsed_results, f, indent=4)
                else:
                    with open(
                        os.path.join(save_dir_collect[i], "all_instructions.json"), "w"
                    ) as f:
                        json.dump(parsed_results, f, indent=4)

        # Do the batch caption
        result_generator = llava_model.predict_nostreaming(
            image=image_path_collect,
            prompt=caption_prompt_collect,
            top_p=0.95,
            temperature=0.5,
            max_tokens=1024,
            do_sample=True,
        )

        for i in range(len(result_generator)):
            org_description = result_generator[i]
            org_description_path = os.path.join(
                save_dir_collect[i], "origin_caption.json"
            )
            with open(org_description_path, "w") as file:
                json.dump({"description": org_description}, file, indent=4)

        if PROFILE:
            end.record()
            torch.cuda.synchronize()
            print(f"used time: {start.elapsed_time(end)/1000} second")

        batch_accumulator = 0
        image_path_collect = []
        save_dir_collect = []
        instruction_prompt_collect = []
        caption_prompt_collect = []
        if batch_count % 10 == 0:
            print(f"processed {batch_count} batch")


if __name__ == "__main__":
    main()
