"""
This is the function where we process the instructions from generate_main_llava.py and process.

There would be 3 different functions: instruction genertion, QA generation and image generation.

To be unified, we first call instruction generation, then image generation and later QA generation.
No matter whether we really need that and how we design each subfunctions.

The prompt will be implemented in each modules so that we can easily change them.
"""

import argparse
import copy
import glob
import json
import os
import pickle
import re
from random import shuffle

import torch
from diffusers import AutoPipelineForInpainting
from IPython import embed
from PIL import Image, ImageDraw, ImageFont

from vilp.model_tool import (
    get_instruction_prompt,
    get_multi_image_QA_prompt,
    get_single_image_QA_prompt,
    get_tool_model,
    instruction_model_tool_file_name_mapper,
    qa_model_tool_file_name_mapper,
    tool_name_mapper,
)

try:
    from vilp.model_warper.ground_sam import GroundSam_Warper
except:
    print("cannot import `GroundSam_Warper`")

from vilp.model_wrapper.vlm_wrapper import VLM_wrapper
from vilp.utils.re_utils import (
    extract_answer_rating,
    extract_justify_rating,
    extract_QA,
    generate_grounding_dino_instruction,
    get_score_from_VLM_judge,
)

PROFILE = True

CURPATH = os.path.dirname(os.path.abspath(__file__))


# NOTE: TODO: need change "get_folder_path" and "define_dir_path" function


def get_folder_path(args):
    # FOLDERS = glob.glob(
    #     os.path.join(
    #         DATA_PATH, f"results_{args.dataset_name}{args.llava_checkpoint}", "*"
    #     )
    # )
    FOLDERS = glob.glob("results/*")
    return FOLDERS


def define_dir_path(args):
    # DIRPATH = os.path.join(
    #     DATA_PATH, f"results_{args.dataset_name}{args.llava_checkpoint}"
    # )
    DIRPATH = "results"
    return DIRPATH


@torch.no_grad()
def instruction_gen(model_tool, args):
    """
    This is the function to generate instructions based on the results from `generate_main_llava.py`
    Currently, only using "groundingDINO" need it.
    """
    if model_tool == "instructp2p" or model_tool == "sdxl":
        print("instructp2p or sdxl is not supported in instruction generation")
        return None
    llava_model = VLM_wrapper(
        model_type=args.vlm_model_type,
        checkpoint_path=args.vlm_model_checkpoint,
        model_name=args.vlm_model_name,
        conv_mode=args.vlm_conv_mode,
    )
    FOLDERS = get_folder_path(args)
    # debug
    shuffle(FOLDERS)
    if PROFILE:
        # enable timer
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

    output_path_collect = []
    prompt_collect = []
    image_path_collect = []
    org_description_collect = []
    batch_accumulator = 0
    for folder in FOLDERS:
        # check whether the instruction file exists then we skip
        if (
            len(
                glob.glob(
                    os.path.join(
                        folder,
                        "*",
                        f"{instruction_model_tool_file_name_mapper[model_tool]}_{args.llava_checkpoint.split('/')[-1]}.json",
                    )
                )
            )
            > 0
        ):
            continue
        if args.vlm_model_name.split("-")[-1] == "7b":
            instruction_file = os.path.join(folder, "all_instructions_7b.json")
        elif args.vlm_model_name.split("-")[-1] == "13b":
            instruction_file = os.path.join(folder, "all_instructions_13b.json")
        else:
            raise ValueError("Unknown model name")
        if not os.path.exists(instruction_file):
            continue
        try:
            parsed_results = json.load(open(instruction_file))
        except:
            continue

        if PROFILE:
            start.record()

        loaded = False

        for results in parsed_results:
            if results["Tool Used"] != "GroundingDINO":
                continue

            output_path = os.path.join(
                folder,
                f"{results['Item Number']:02d}_{model_tool}_{args.llava_checkpoint.split('/')[-1]}",
            )
            if not os.path.exists(output_path):
                os.makedirs(output_path, exist_ok=True, mode=0o777)

            image_path = os.path.join(folder, "origin.jpg")

            org_description_path = os.path.join(folder, "origin_caption.json")
            try:
                org_description = json.load(open(org_description_path))["description"]
            except:
                continue

            prompt_text = get_instruction_prompt(model_tool, results, org_description)

            image_path_collect.append(image_path)
            prompt_collect.append(prompt_text)
            output_path_collect.append(output_path)
            org_description_collect.append(org_description)
            loaded = True
        if loaded:
            batch_accumulator += 1

        # if we don't collect enough batch size, we continue to collect
        if batch_accumulator < args.batch_size:
            continue

        # If the batch size is bigger than the limit, we need to split the batch into smaller ones
        if len(image_path_collect) > args.batch_size:
            instruction_result = []
            for i in range(0, len(image_path_collect), args.batch_size):
                instruction_result += llava_model.predict_nostreaming(
                    image=image_path_collect[
                        i : min(i + args.batch_size, len(image_path_collect))
                    ],
                    prompt=prompt_collect[
                        i : min(i + args.batch_size, len(image_path_collect))
                    ],
                    top_p=0.95,
                    temperature=0.5,  # some randomness?
                    max_tokens=512,
                    do_sample=True,
                )
        else:
            instruction_result = llava_model.predict_nostreaming(
                image=image_path_collect,
                prompt=prompt_collect,
                top_p=0.95,
                temperature=0.5,  # some randomness?
                max_tokens=512,
                do_sample=True,
            )
        for img_index in range(len(instruction_result)):

            # TODO: here might be better to merge `generate_grounding_dino_instruction` into dino functions
            if model_tool == "groundingdino":
                _, parsed_results = generate_grounding_dino_instruction(
                    instruction_result[img_index]
                )
            else:
                raise ValueError("The model tool is not supported")

            if len(parsed_results) == 0:
                print(instruction_result[img_index])
                print(f"{img_index}   !!!!!!!!!!!!!!!!!!!!!!!!")

            if len(parsed_results) == 0:
                continue

            print(
                "save:",
                os.path.join(
                    output_path_collect[img_index],
                    f"{instruction_model_tool_file_name_mapper[model_tool]}_7b.json",
                ),
            )
            json.dump(
                parsed_results,
                open(
                    os.path.join(
                        output_path_collect[img_index],
                        f"{instruction_model_tool_file_name_mapper[model_tool]}_{args.llava_checkpoint.split('/')[-1]}.json",
                        # f"{instruction_model_tool_file_name_mapper[model_tool]}_7b.json",
                    ),
                    "w",
                ),
                indent=4,
            )

        # Reset collect and counter
        output_path_collect = []
        prompt_collect = []
        image_path_collect = []
        org_description_collect = []
        batch_accumulator = 0

        if PROFILE:
            end.record()
            torch.cuda.synchronize()
            print(f"used time: {start.elapsed_time(end)/1000} second")


@torch.no_grad()
def img_gen(model_tool, args):
    """
    Function to generate images according to instructions.
    """

    # TODO: This function is not well batchlized. We need to batchlize it.
    model = get_tool_model(model_tool, args)

    if PROFILE:
        # enable timer
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

    FOLDERS = get_folder_path(args)
    shuffle(FOLDERS)

    for folder in FOLDERS:
        # check whether the instruction file exists then we skip
        if (
            len(
                glob.glob(
                    os.path.join(
                        folder, f"*_{model_tool}_{args.llava_checkpoint}", "gen*.jpg"
                    )
                )
            )
            > 0
        ):
            continue

        instruction_file = os.path.join(folder, "all_instructions.json")
        if not os.path.exists(instruction_file):
            continue
        try:
            parsed_results = json.load(open(instruction_file))
        except:
            print("instruction_file error", instruction_file)
            continue

        if PROFILE:
            start.record()

        org_image_path = os.path.join(folder, "origin.jpg")
        for result in parsed_results:
            if result["Tool Used"] == tool_name_mapper[model_tool]:
                output_path = os.path.join(
                    folder,
                    f"{result['Item Number']:02d}_{model_tool}_{args.llava_checkpoint}",
                )

                os.makedirs(output_path, exist_ok=True, mode=0o777)
                instructions = copy.deepcopy(result)

                # If the model is grounding dino, we need more instructions
                if model_tool == "groundingdino":
                    if not os.path.exists(
                        os.path.join(
                            output_path,
                            f"{instruction_model_tool_file_name_mapper[model_tool]}_{args.llava_checkpoint}.json",
                        )
                    ):
                        continue

                    try:
                        instructions = json.load(
                            open(
                                os.path.join(
                                    output_path,
                                    f"{instruction_model_tool_file_name_mapper[model_tool]}_{args.llava_checkpoint}.json",
                                )
                            )
                        )
                    except:
                        print(
                            "instructions, error",
                            os.path.join(
                                output_path,
                                f"{instruction_model_tool_file_name_mapper[model_tool]}_{args.llava_checkpoint}.json",
                            ),
                        )
                        continue
                    for i in range(len(instructions)):
                        model.forward(
                            org_image_path,
                            instructions[i],
                            args.num_img_gen,
                            output_path,
                            i,
                        )

                else:
                    model.forward(
                        org_image_path,
                        instructions,
                        args.num_img_gen,
                        output_path,
                        None,
                    )
        if PROFILE:
            end.record()
            torch.cuda.synchronize()
            print(f"used time: {start.elapsed_time(end)/1000} second")


@torch.no_grad()
def single_image_QA_gen(model_tool, args):
    """
    The function to generate QA based on single image descritions.
    """
    # NOTE: not clear how to efficiently batchlize this function

    llava_model = VLM_wrapper(
        model_type=args.vlm_model_type,
        checkpoint_path=args.vlm_model_checkpoint,
        model_name=args.vlm_model_name,
        conv_mode=args.vlm_conv_mode,
    )

    FOLDERS = get_folder_path(args)
    shuffle(FOLDERS)

    if PROFILE:
        # enable timer
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

    prompt_collect = []
    image_path_collect = []
    output_path_collect = []
    batch_accumulator = 0

    for folder in FOLDERS:
        if (
            len(
                glob.glob(
                    os.path.join(
                        folder,
                        f"*_{model_tool}_{args.llava_checkpoint}",
                        f"single_img_QA_{args.llava_checkpoint}.json",
                    )
                )
            )
            > 0
        ):
            continue

        instruction_file = os.path.join(folder, "all_instructions.json")
        if not os.path.exists(instruction_file):
            continue
        try:
            parsed_results = json.load(open(instruction_file))
        except:
            print("instruction_file error", instruction_file)
            continue
        # org_image_path = os.path.join(folder, "origin.jpg")

        # if there are useful thing loaded
        loaded = False
        for result in parsed_results:
            if result["Tool Used"] == tool_name_mapper[model_tool]:
                # If we fail to generate any images, skip
                if not os.path.exists(
                    os.path.join(
                        folder,
                        f"{result['Item Number']:02d}_{model_tool}_{args.llava_checkpoint}",
                    )
                ):
                    continue

                output_images = glob.glob(
                    os.path.join(
                        folder,
                        f"{result['Item Number']:02d}_{model_tool}_{args.llava_checkpoint}",
                        "gen*.jpg",
                    )
                )
                if len(output_images) == 0:
                    continue

                # NOTE: We can do infer from descriptions or directly infer from images.

                for image in output_images:

                    prompt_text = get_single_image_QA_prompt(model_tool, None)
                    image_path_collect.append(image)
                    prompt_collect.append(prompt_text)
                    output_path_collect.append(
                        os.path.join(
                            folder,
                            f"{result['Item Number']:02d}_{model_tool}_{args.llava_checkpoint}",
                        )
                    )
                loaded = True
            # If the batch size is bigger than the limit, we need to split the batch into smaller ones
        if loaded:
            batch_accumulator += 1
        # if we don't collect enough batch size, we continue to collect
        if batch_accumulator < args.filefolder_batch_size:
            continue

        if PROFILE:
            start.record()
        # TODO: here might be better to move the following function to model themselves
        if len(image_path_collect) >= args.batch_size:
            qa_result = []
            for i in range(0, len(image_path_collect), args.batch_size):
                # qa_result += llava_model.predict_text_only_nostreaming(
                qa_result += llava_model.predict_nostreaming(
                    image=image_path_collect[
                        i : min(i + args.batch_size, len(image_path_collect))
                    ],
                    prompt=prompt_collect[
                        i : min(i + args.batch_size, len(image_path_collect))
                    ],
                    top_p=0.95,
                    temperature=0.5,  # some randomness?
                    max_tokens=512,
                    do_sample=True,
                )

        else:
            # qa_result = llava_model.predict_text_only_nostreaming(
            qa_result = llava_model.predict_nostreaming(
                image=image_path_collect,
                prompt=prompt_collect,
                top_p=0.95,
                temperature=0.5,  # some randomness?
                max_tokens=512,
                do_sample=True,
            )
        QA_save_dict = []
        QA_save_path = None
        for i in range(len(qa_result)):
            _, qa_dict = extract_QA(qa_result[i])
            # Add corresponding item number in the QA results
            for qa in qa_dict:
                qa.update({"Image Path": image_path_collect[i]})
            if QA_save_path is None:
                QA_save_path = output_path_collect[i]
                QA_save_dict += qa_dict
            elif QA_save_path == output_path_collect[i]:
                QA_save_dict += qa_dict
            else:
                if len(QA_save_dict) > 0:
                    with open(
                        os.path.join(
                            QA_save_path, f"single_img_QA_{args.llava_checkpoint}.json"
                        ),
                        "w",
                    ) as f:
                        json.dump(QA_save_dict, f, indent=4)
                QA_save_path = output_path_collect[i]
                QA_save_dict = []
                QA_save_dict += qa_dict
        if len(QA_save_dict) > 0:
            with open(
                os.path.join(
                    QA_save_path, f"single_img_QA_{args.llava_checkpoint}.json"
                ),
                "w",
            ) as f:
                json.dump(QA_save_dict, f, indent=4)
            print(f"saved to {QA_save_path}")
        if PROFILE:
            end.record()
            torch.cuda.synchronize()
            print(f"used time: {start.elapsed_time(end)/1000} second")
        prompt_collect = []
        image_path_collect = []
        output_path_collect = []
        batch_accumulator = 0


@torch.no_grad()
def rating_singleQA_sampleNewQA(args):
    """
    The function to sample a set of QA answers given the image and the questions.
    """
    llava_model = VLM_wrapper(
        model_type=args.vlm_model_type,
        checkpoint_path=args.vlm_model_checkpoint,
        model_name=args.vlm_model_name,
        conv_mode=args.vlm_conv_mode,
    )

    if PROFILE:
        # enable timer
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
    DIRPATH = define_dir_path(args)

    qa_dicts = glob.glob(
        os.path.join(DIRPATH, "*", "*", f"single_img_QA_{args.llava_checkpoint}.json")
    )
    shuffle(qa_dicts)

    batch_accumulator = 0
    image_path_collect = []
    output_path_collect = []
    prompt_collect = []
    qa_item_collect = []
    for cur_qa in qa_dicts:
        cur_folder = "/".join(cur_qa.split("/")[:-1])
        cur_output_path = os.path.join(
            cur_folder, f"single_img_QA_DPO_{args.llava_checkpoint}.json"
        )

        if os.path.exists(cur_output_path):
            continue
        try:
            cur_qa_dict = json.load(open(cur_qa))
            if len(cur_qa_dict) == 0:
                continue
        except:
            if os.path.exists(cur_qa):
                os.remove(cur_qa)
            continue

        loaded = False
        # Loop and filter QA's
        for qa_item in cur_qa_dict:
            # image_path_collect.append(qa_item["Image Path"])
            # NOTE: adding fullpath to image path
            image_path_collect.append(qa_item["Image Path"])
            qa_item_collect.append(qa_item)

            output_path_collect.append(cur_folder)

            prompt_collect.append(qa_item["Question"])
            loaded = True
        if loaded:
            batch_accumulator += 1

        if batch_accumulator < args.filefolder_batch_size:
            continue

        if PROFILE:
            start.record()
        QA_results_collect = {}
        answer_results_collect = {}

        # First Inference: Answer Generation.
        # Give super high randomness:
        # Baically, large temperature gives more randomness and longer results.
        try:
            for _round in range(args.num_QA_sample_round):
                if len(image_path_collect) >= args.batch_size:
                    answer_collect = []
                    for i in range(0, len(image_path_collect), args.batch_size):
                        answer_collect += llava_model.predict_nostreaming(
                            image=image_path_collect[
                                i : min(i + args.batch_size, len(image_path_collect))
                            ],
                            prompt=prompt_collect[
                                i : min(i + args.batch_size, len(image_path_collect))
                            ],
                            top_p=1.0,
                            do_sample=True,
                            temperature=1.2,  # some randomness?
                            max_tokens=512,
                        )
                else:
                    answer_collect = llava_model.predict_nostreaming(
                        image=image_path_collect,
                        prompt=prompt_collect,
                        top_p=1.0,
                        do_sample=True,
                        temperature=1.2,  # some randomness?
                        max_tokens=512,
                    )
                torch.cuda.empty_cache()
                # Second we generate the rating
                qa_rating_prompt_collect = []
                for i, question in enumerate(prompt_collect):
                    # I ask the model to repeat the answer so that we can reuse the existing re-compile pattern.
                    text_prompt = f"""
                    Review the provided image along with the user's question and the given answer. Evaluate the quality of the answer based on the image and question using the additive 5-point scoring system. Each point builds upon the last, ensuring a cumulative evaluation up to a maximum of 5 points:

                    - 1 point: The answer provides relevant information directly related to the user's inquiry and the image, albeit minimally.
                    - 2 points: The answer addresses key aspects of the user's question using some details from the input image. It should move beyond mere relevance to showing an understanding of the image content.
                    - 3 points: The answer comprehensively answers the main components of the user's question using accurate and clear details from the image. This response should be useful and informative but might lack in-depth insight or broader context.
                    - 4 points: The answer thoroughly and comprehensively addresses the user's question, demonstrating an accurate understanding of the image's details. It should be well-organized and clearly articulated, with only minor areas needing improvement for greater conciseness or specific focus.
                    - 5 points: The answer exemplifies an expert understanding of both the content and context of the image. It is not only accurate but also engaging and insightful, adding significant value to the user's understanding of the image. The answer should be precise, without irrelevant details, and crafted to directly connect the user's query with insights derived from the image.

                    Input Question: {question}
                    Answer: {answer_collect[i]}
                    Instructions: Provide a rigorous and exacting critique. Answers should be closely tailored to the specifics of the question and image without deviating into generalities or off-topic information. Be strict in your rating if the answer falls short of the highest standards. Be strict in your evaluation. Reserve a score of 4 or 5 for answers that truly excel in clarity, depth, and relevance.

                    Expected Response Format:
                    Justify: [Summarize how well the answer meets each of the five criteria]
                    Score: [Rate the answer's quality on a scale of 0 to 5 based on the justification]
                    """
                    # Answer: [Repeat the Answer]
                    qa_rating_prompt_collect.append(text_prompt)
                if len(image_path_collect) >= args.batch_size:
                    qa_result = []
                    for i in range(0, len(image_path_collect), args.batch_size):
                        qa_result += llava_model.predict_nostreaming(
                            image=image_path_collect[
                                i : min(i + args.batch_size, len(image_path_collect))
                            ],
                            prompt=qa_rating_prompt_collect[
                                i : min(i + args.batch_size, len(image_path_collect))
                            ],
                            top_p=None,
                            do_sample=False,
                            temperature=0.0,  # Really deterministic
                            max_tokens=512,
                        )
                else:
                    qa_result = llava_model.predict_nostreaming(
                        image=image_path_collect,
                        prompt=qa_rating_prompt_collect,
                        top_p=None,
                        do_sample=False,
                        temperature=0.0,  # Really deterministic
                        max_tokens=512,
                    )
                QA_results_collect.update({_round: qa_result})
                answer_results_collect.update({_round: answer_collect})
        except:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("generate wrong")
            image_path_collect = []
            output_path_collect = []
            batch_accumulator = 0
            prompt_collect = []
            qa_item_collect = []
            continue
        # Now we need to re-org the data into [Image, Question, Answer_round1, rating_round1, Answer_round2, rating_round2, ...]
        QA_results_collect_reorg = []

        for i in range(len(image_path_collect)):
            _QA_results_collect_reorg = {
                "Question": qa_item_collect[i]["Question"],
                "Image Path": qa_item_collect[i]["Image Path"],
            }
            count = 0
            for _round in range(args.num_QA_sample_round):
                if args.vlm_model_name == "cambrian-8b":

                    numbers = re.findall(r"\d+", QA_results_collect[_round][i])
                    if len(numbers) == 1:
                        qa_dicts = [
                            {
                                "Score": numbers[0],
                                "Justify": QA_results_collect[_round][i],
                            }
                        ]
                    else:
                        qa_dicts = []
                else:
                    _, qa_dicts = extract_justify_rating(QA_results_collect[_round][i])
                # the below codes used for adding re.compile
                if len(qa_dicts) == 0:
                    print("!!!000!!!!!!!!!!!!:")
                    print(QA_results_collect[_round][i])
                    # print("!!!!!!!!!!!!!!!")
                for qa_dict in qa_dicts:
                    count += 1
                    _QA_results_collect_reorg.update(
                        {
                            # NOTE: they might not fully repeat the Answer here. Double check, otherwise we use the answer results
                            f"Response_{count}": {
                                # "Answer": qa_dict["Answer"],
                                "Answer": answer_results_collect[_round][i],
                                "Score": qa_dict["Score"],
                                "Justify": qa_dict["Justify"],
                            }
                        }
                    )
            QA_results_collect_reorg.append(_QA_results_collect_reorg)
        scores = []
        for item in QA_results_collect_reorg:
            for key in item.keys():
                if key.startswith("Response_"):
                    scores.append(item[key]["Score"])
        print(scores)

        QA_save_dict = []
        QA_save_path = None
        for i in range(len(QA_results_collect_reorg)):
            if QA_save_path is None:
                QA_save_path = output_path_collect[i]
                QA_save_dict.append(QA_results_collect_reorg[i])
            elif QA_save_path == output_path_collect[i]:
                QA_save_dict.append(QA_results_collect_reorg[i])
            else:
                if len(QA_save_dict) > 0:
                    with open(
                        os.path.join(
                            QA_save_path,
                            f"single_img_QA_DPO_{args.llava_checkpoint}.json",
                        ),
                        "w",
                    ) as f:
                        json.dump(QA_save_dict, f, indent=4)
                QA_save_path = output_path_collect[i]
                QA_save_dict = []
                QA_save_dict.append(QA_results_collect_reorg[i])
        # to Ang: it seems the previous implementation always miss to save the last one?
        # yes it might be true.
        if len(QA_save_dict) > 0:
            with open(
                os.path.join(
                    QA_save_path, f"single_img_QA_DPO_{args.llava_checkpoint}.json"
                ),
                "w",
            ) as f:
                json.dump(QA_save_dict, f, indent=4)
        print("saved to", QA_save_path)

        if PROFILE:
            end.record()
            torch.cuda.synchronize()
            print(f"used time: {start.elapsed_time(end)/1000} second")

        image_path_collect = []
        output_path_collect = []
        batch_accumulator = 0
        prompt_collect = []
        qa_item_collect = []


def get_parser():
    parser = argparse.ArgumentParser(
        "main function to generate instructions, images and QA"
    )
    # General Control
    parser.add_argument(
        "--model_tool",
        type=str,
        default="groundingdino",
        choices=["groundingdino", "instructp2p", "sdxl"],
        help="using which tool to finish the task",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="instruction_gen",
        choices=[
            "instruction_gen",
            "image_gen",
            "single_image_QA_gen",
            "rating_singleQA_sampleNewQA",
        ],
        help="which task to finish",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="vg",
        help="which dataset to use",
    )
    parser.add_argument(
        "--use_new_description",
        action="store_true",
        default=False,
        help="whether using the new description from the image to generate QA",
    )
    parser.add_argument(
        "--batch_size", type=int, default=6, help="batch size for llava"
    )
    parser.add_argument(
        "--filefolder_batch_size",
        type=int,
        default=6,
        help="batch size for accumulate files",
    )
    parser.add_argument(
        "--num_img_gen",
        type=int,
        default=2,
        help="number of images to be generated. Could be different for different models.",
    )
    group = parser.add_argument_group("detect_inpaint_args")

    group.add_argument(
        "--config",
        type=str,
        default="packages/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        help="path to config file",
    )
    group.add_argument(
        "--vlm_model_type",
        type=str,
        default="llava",
        help="vlm model type",
    )
    group.add_argument(
        "--vlm_model_checkpoint",
        type=str,
        default="liuhaotian/llava-v1.5-7b",
        help="vlm model checkpoint",
    )
    group.add_argument(
        "--vlm_model_name",
        type=str,
        default="llava-v1.5-7b",
        help="vlm model name",
    )
    group.add_argument(
        "--vlm_conv_mode",
        type=str,
        default="llama_3",
        help="vlm conv mode",
    )
    group.add_argument(
        "--llava_checkpoint",
        type=str,
        default="liuhaotian/llava-v1.5-7b",
        help="path to checkpoint file",
    )
    group.add_argument(
        "--grounded_checkpoint",
        type=str,
        default="packages/Grounded-Segment-Anything/groundingdino_swint_ogc.pth",
        help="path to checkpoint file",
    )
    group.add_argument(
        "--sam_checkpoint",
        type=str,
        default="packages/Grounded-Segment-Anything/sam_vit_h_4b8939.pth",
        help="path to checkpoint file",
    )
    group.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="save your huggingface large model cache",
    )
    group.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    group.add_argument(
        "--text_threshold", type=float, default=0.25, help="text threshold"
    )
    # NOTE: the `inpaint_mode` could be important to ablate.
    group.add_argument("--inpaint_mode", type=str, default="first", help="inpaint mode")
    group.add_argument(
        "--device", type=str, default="cuda", help="running on cpu only!, default=False"
    )

    # single_image_sample_new_QA args
    parser.add_argument(
        "--QA_score_threshold", type=int, default=4, help="QA score threshold"
    )
    parser.add_argument(
        "--num_QA_sample_round",
        type=int,
        default=3,
        help="number of answers sampled from each gen",
    )
    args = parser.parse_args()

    return args


def call_function(args):
    # call functions based on tasks
    if args.task == "instruction_gen":
        instruction_gen(args.model_tool, args)
    elif args.task == "single_image_QA_gen":
        single_image_QA_gen(args.model_tool, args)
    elif args.task == "image_gen":
        img_gen(args.model_tool, args)
    elif args.task == "rating_singleQA_sampleNewQA":
        rating_singleQA_sampleNewQA(args)
    else:
        raise ValueError("The task is not supported")


if __name__ == "__main__":
    args = get_parser()
    call_function(args)
