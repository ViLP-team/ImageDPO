import argparse
import json
import math
import os
import re

import shortuuid
import torch
from IPython import embed
from PIL import Image
from tqdm import tqdm

from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

PERIODSTRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
COMMASTRIP = re.compile(r"(\d)(\,)(\d)")
PUNCT = [
    ";",
    r"/",
    "[",
    "]",
    '"',
    "{",
    "}",
    "(",
    ")",
    "=",
    "+",
    "\\",
    "_",
    "-",
    ">",
    "<",
    "@",
    "`",
    ",",
    "?",
    "!",
]

CONTRACTIONS = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hadnt've": "hadn't've",
    "hadn'tve": "hadn't've",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hed've": "he'd've",
    "he'dve": "he'd've",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "Id've": "I'd've",
    "I'dve": "I'd've",
    "Im": "I'm",
    "Ive": "I've",
    "isnt": "isn't",
    "itd": "it'd",
    "itd've": "it'd've",
    "it'dve": "it'd've",
    "itll": "it'll",
    "let's": "let's",
    "maam": "ma'am",
    "mightnt": "mightn't",
    "mightnt've": "mightn't've",
    "mightn'tve": "mightn't've",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "notve": "not've",
    "oclock": "o'clock",
    "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at",
    "'ows'at": "'ow's'at",
    "'ow'sat": "'ow's'at",
    "shant": "shan't",
    "shed've": "she'd've",
    "she'dve": "she'd've",
    "she's": "she's",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've",
    "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll",
    "somebodys": "somebody's",
    "someoned": "someone'd",
    "someoned've": "someone'd've",
    "someone'dve": "someone'd've",
    "someonell": "someone'll",
    "someones": "someone's",
    "somethingd": "something'd",
    "somethingd've": "something'd've",
    "something'dve": "something'd've",
    "somethingll": "something'll",
    "thats": "that's",
    "thered": "there'd",
    "thered've": "there'd've",
    "there'dve": "there'd've",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyd've": "they'd've",
    "they'dve": "they'd've",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "twas": "'twas",
    "wasnt": "wasn't",
    "wed've": "we'd've",
    "we'dve": "we'd've",
    "weve": "we've",
    "werent": "weren't",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whereve": "where've",
    "whod": "who'd",
    "whod've": "who'd've",
    "who'dve": "who'd've",
    "wholl": "who'll",
    "whos": "who's",
    "whove": "who've",
    "whyll": "why'll",
    "whyre": "why're",
    "whys": "why's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've",
    "yall": "y'all",
    "yall'll": "y'all'll",
    "y'allll": "y'all'll",
    "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've",
    "youd": "you'd",
    "youd've": "you'd've",
    "you'dve": "you'd've",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've",
}

MANUALMAP = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}

ARTICLES = ["a", "an", "the"]


def normalize_answer(resAns):
    resAns = resAns.replace("\n", " ")
    resAns = resAns.replace("\t", " ")
    resAns = resAns.strip()
    resAns = processPunctuation(resAns)
    resAns = processDigitArticle(resAns)
    resAns = resAns.replace(",", "")
    return resAns


def processPunctuation(inText):
    outText = inText
    for p in PUNCT:
        if (p + " " in inText or " " + p in inText) or (
            re.search(COMMASTRIP, inText) != None
        ):
            outText = outText.replace(p, "")
        else:
            outText = outText.replace(p, " ")
    outText = PERIODSTRIP.sub("", outText, re.UNICODE)
    return outText


def processDigitArticle(inText):
    outText = []
    tempText = inText.lower().split()
    for word in tempText:
        word = MANUALMAP.setdefault(word, word)
        if word not in ARTICLES:
            outText.append(word)
        else:
            pass
    for wordId, word in enumerate(outText):
        if word in CONTRACTIONS:
            outText[wordId] = CONTRACTIONS[word]
    outText = " ".join(outText)
    return outText


def main(args):

    # load the questions and gt answers
    with open(args.questions_file) as f:
        instances = json.load(f)

    # init model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name
    )

    accQA = []

    print("computing accuracy")
    step = 0

    for qs_index, instance in enumerate(instances):

        source_img = os.path.join(args.image_folder, instance["imgname"])
        if not os.path.exists(source_img):
            print("!!!!!!!! Image not found")
            continue

        conv = conv_templates[args.conv_mode].copy()

        image_data = [Image.open(source_img)]
        image_tensor = []

        for image in image_data:
            # image = expand2square(
            #    image, tuple(int(x * 255) for x in image_processor.image_mean)
            # )
            image = (
                image_processor.preprocess(image, return_tensors="pt")["pixel_values"]
                .half()
                .cuda()
            )
            image_tensor.append(image)

        inp = ""
        inp += DEFAULT_IMAGE_TOKEN + "\n"
        # inp += "Rember to give the answer of the question directly (in the format of numbers or words)"

        # try to prompt engineering.
        inp += "Carefully analyze the image, and answer the following question based on the image. Before giving the final answer, think the problem step by step and output justifications."
        inp += f"""
                Output format: 

                Justifications: [Analyze the image and questions, and think about the answer step by step]
                Answer: [Precise and consice answer]

                Examples 1: 
                Justifications: The question asks the price of lamb minus the price of beef. Theere are two bars in the chart representing the price of lamb and beef. The price of lamb is 4.3 and the price of beef is 2.0 observed from the figure. Therefore, The answer is 4.3 - 2.0 = 2.3.
                Answer: 2.3

                Examples 2:
                Justifications: The question asks the number of goods shown in the figure. The chart shows the price of goods in supermarket. There are 9 bars. Therefore, there are 9 goods.
                Answer: 9

                Now following the format of examples, while really focusing on the given image and question.
                The question is: {instance['query']}
                """
        # inp += """
        # 1. What is the title of the chart?
        #         """
        # inp += instance['query']

        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=[image_tensor],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                # num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
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
        outputs = outputs.strip()

        if "value\U0001F449" in outputs[: len(conv.roles[1]) + 2]:
            outputs = outputs[len(conv.roles[1]) :].strip()
        else:
            outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()

        # parsing the results and computing the accuracy
        outputs = outputs.lower()
        # pattern_QA = re.compile(r"answer:\s*(?P<answer>.+?)\n")
        # matches = pattern_QA.finditer(outputs)
        # breakpoint()
        # for match in matches:
        #     try:
        #         answer = match.group("answer").strip()
        #     except Exception as e:
        #         answer = 0
        # breakpoint()
        try:
            outputs = outputs.split("answer:")[1].strip()
        except Exception as e:
            outputs = "0"
        output_ans = normalize_answer(outputs)
        gt_ans = normalize_answer(instance["label"])
        if gt_ans == output_ans:
            accQA.append(1)
        else:
            accQA.append(0)
    print(f"Accuracy: {sum(accQA)/len(accQA)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="llava-v1.5")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument(
        "--image-folder",
        type=str,
        default="/home/ancao/src/turbo/repos/ImprovingVLM/eval_data/single_img_dataset/ChartVQA/ChartQA/test/png",
    )
    parser.add_argument(
        "--questions-file",
        type=str,
        default="/home/ancao/src/turbo/repos/ImprovingVLM/eval_data/single_img_dataset/ChartVQA/ChartQA/test/test_human.json",
    )
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    main(args)
