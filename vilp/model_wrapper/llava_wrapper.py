from __future__ import annotations

import copy
import os
import re
import time
from io import BytesIO
from threading import Thread
from typing import List, Optional

import requests
import torch
from cog import BasePredictor, ConcatenateIterator, Input, Path
from IPython import embed
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import KeywordsStoppingCriteria, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from PIL import Image
from transformers.generation.streamers import TextIteratorStreamer


def extract_answer(text):
    # NOTE: Might be better to use re matching

    return text.strip().split("ASSISTANT: ")[1]


class BatchTextIteratorStreamer(TextIteratorStreamer):
    def __init__(
        self,
        batch_size: int,
        tokenizer,
        skip_prompt: bool = False,
        timeout: float | None = None,
        **decode_kwargs,
    ):
        super().__init__(tokenizer, skip_prompt, timeout, **decode_kwargs)
        self.batch_size = batch_size
        self.token_cache = [[] for _ in range(batch_size)]
        self.print_len = [0 for _ in range(batch_size)]
        self.generate_exception = None

    def put(self, value):
        if len(value.shape) != 2:
            value = torch.reshape(
                value, (self.batch_size, value.shape[0] // self.batch_size)
            )

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        printable_texts = list()
        for idx in range(self.batch_size):
            self.token_cache[idx].extend(value[idx].tolist())
            text = self.tokenizer.decode(self.token_cache[idx], **self.decode_kwargs)

            if text.endswith("\n"):
                printable_text = text[self.print_len[idx] :]
                self.token_cache[idx] = []
                self.print_len[idx] = 0
                # If the last token is a CJK character, we print the characters.
            elif len(text) > 0 and self._is_chinese_char(ord(text[-1])):
                printable_text = text[self.print_len[idx] :]
                self.print_len[idx] += len(printable_text)
            else:
                printable_text = text[self.print_len[idx] : text.rfind(" ") + 1]
                self.print_len[idx] += len(printable_text)
            printable_texts.append(printable_text)

        self.on_finalized_text(printable_texts)

    def end(self):
        printable_texts = list()
        for idx in range(self.batch_size):
            if len(self.token_cache[idx]) > 0:
                text = self.tokenizer.decode(
                    self.token_cache[idx], **self.decode_kwargs
                )
                printable_text = text[self.print_len[idx] :]
                self.token_cache[idx] = []
                self.print_len[idx] = 0
            else:
                printable_text = ""
            printable_texts.append(printable_text)

        self.next_tokens_are_prompt = True
        self.on_finalized_text(printable_texts, stream_end=True)

    def on_finalized_text(self, texts: list[str], stream_end: bool = False):
        self.text_queue.put(texts, timeout=self.timeout)
        if stream_end:
            self.text_queue.put(self.stop_signal, timeout=self.timeout)


# url for the weights mirror
REPLICATE_WEIGHTS_URL = "https://weights.replicate.delivery/default"
# files to download from the weights mirrors
weights = [
    {
        "dest": "liuhaotian/llava-v1.5-13b",
        # git commit hash from huggingface
        "src": "llava-v1.5-13b/006818fc465ebda4c003c0998674d9141d8d95f8",
        "files": [
            "config.json",
            "generation_config.json",
            "pytorch_model-00001-of-00003.bin",
            "pytorch_model-00002-of-00003.bin",
            "pytorch_model-00003-of-00003.bin",
            "pytorch_model.bin.index.json",
            "special_tokens_map.json",
            "tokenizer.model",
            "tokenizer_config.json",
        ],
    },
    {
        "dest": "openai/clip-vit-large-patch14-336",
        "src": "clip-vit-large-patch14-336/ce19dc912ca5cd21c8a653c79e251e808ccabcd1",
        "files": ["config.json", "preprocessor_config.json", "pytorch_model.bin"],
    },
]


def download_json(url: str, dest: Path):
    res = requests.get(url, allow_redirects=True)
    if res.status_code == 200 and res.content:
        with dest.open("wb") as f:
            f.write(res.content)
    else:
        print(f"Failed to download {url}. Status code: {res.status_code}")


def download_weights(baseurl: str, basedest: str, files: list[str]):
    basedest = Path(basedest)
    start = time.time()
    print("downloading to: ", basedest)
    basedest.mkdir(parents=True, exist_ok=True)
    for f in files:
        dest = basedest / f
        url = os.path.join(REPLICATE_WEIGHTS_URL, baseurl, f)
        # if not dest.exists():
        #    print("downloading url: ", url)
        #    if dest.suffix == ".json":
        #        download_json(url, dest)
        #    else:
        #        subprocess.check_call(["pget", url, str(dest)], close_fds=False)
        if not dest.exists():
            print("downloading url: ", url)
            response = requests.get(url)
            response.raise_for_status()  # Raises an HTTPError for bad responses
            with open(dest, "wb") as file:
                file.write(response.content)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(
        self, checkpoint_path="liuhaotian/llava-v1.5-13b", model_name="llava-v1.5-13b"
    ) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # for llava-v1.5, checkpoint_path: "liuhaotian/llava-v1.5-13b", model_name: "llava-v1.5-13b"
        # for llava-next, checkpoint_path: "lmms-lab/llama3-llava-next-8b", model_name: "llava_llama3"

        if model_name.startswith("llava-v1.5"):
            for weight in weights:
                download_weights(weight["src"], weight["dest"], weight["files"])
            disable_torch_init()

        print("load pre-trained model from:", checkpoint_path)

        if model_name.startswith("llava-v1.5"):
            self.tokenizer, self.model, self.image_processor, self.context_len = (
                load_pretrained_model(
                    # "liuhaotian/llava-v1.5-13b",
                    # "/nfs/turbo/justincj-turbo/tiangel/improvingVLM/packages/LLaVA/checkpoints/llava-v1.5-13b-task-shuffle/checkpoint-8000",
                    model_path=checkpoint_path,
                    model_name=model_name,
                    model_base=None,
                    load_8bit=False,
                    load_4bit=False,
                )
            )
            self.model = self.model.to(dtype=torch.bfloat16)
        elif model_name.startswith("llava_llama3") or model_name.startswith(
            "llava_qwen"
        ):
            self.tokenizer, self.model, self.image_processor, self.context_len = (
                load_pretrained_model(
                    model_path=checkpoint_path,
                    model_name=model_name,
                    model_base=None,
                    device_map="auto",
                )
            )

        if model_name.startswith("llava_llama3") or model_name.startswith("llava_qwen"):
            self.model.eval()
            self.model.tie_weights()

        self.model_name = model_name

    def predict(
        self,
        image: list[Path] | Path = Input(description="Input image"),
        prompt: list[str] | str = Input(
            description="Prompt to use for text generation"
        ),
        top_p: float = Input(
            description="When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens",
            ge=0.0,
            le=1.0,
            default=1.0,
        ),
        temperature: float = Input(
            description="Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic",
            default=0.2,
            ge=0.0,
        ),
        max_tokens: int = Input(
            description="Maximum number of tokens to generate. A word is generally 2-3 tokens",
            default=1024,
            ge=0,
        ),
        do_sample: bool = Input(
            description="Whether to use sampling or greedy decoding", default=True
        ),
        num_beams: int = 1,
    ) -> ConcatenateIterator[str]:
        """Run a single prediction on the model"""
        if self.model_name.startswith("llava-v1.5"):
            conv_mode = "llava_v1"
        elif self.model_name.startswith("llava_llama3"):
            conv_mode = "llava_llama_3"
        else:
            raise ValueError("Unknown model name")

        if isinstance(image, list):
            image_data = [load_image(str(img)) for img in image]
            image_tensor = [
                (
                    self.image_processor.preprocess(image, return_tensors="pt")[
                        "pixel_values"
                    ]
                    .half()
                    .cuda()
                )
                for image in image_data
            ]
            image_tensor = torch.cat(image_tensor, dim=0)
            use_batch_data = True
            self.model.config.tokenizer_padding_side = self.tokenizer.padding_side = (
                "left"
            )
        else:
            image_data = load_image(str(image))
            image_tensor = (
                self.image_processor.preprocess(image_data, return_tensors="pt")[
                    "pixel_values"
                ]
                .half()
                .cuda()
            )
            use_batch_data = False
        # loop start

        if isinstance(prompt, list):
            assert False, "We currently do not support this"
            input_ids_collect = []
            for sub_prompt in prompt:
                conv = copy.deepcopy(conv_templates[conv_mode])
                # just one turn, always prepend image token
                inp = DEFAULT_IMAGE_TOKEN + "\n" + sub_prompt
                conv.append_message(conv.roles[0], inp)

                conv.append_message(conv.roles[1], None)
                sub_prompt = conv.get_prompt()

                input_ids = (
                    tokenizer_image_token(
                        sub_prompt,
                        self.tokenizer,
                        IMAGE_TOKEN_INDEX,
                        return_tensors="pt",
                    )
                    .unsqueeze(0)
                    .cuda()
                )
                input_ids_collect.append(input_ids)
            input_ids = torch.cat(input_ids_collect, dim=0)
        else:
            conv = copy.deepcopy(conv_templates[conv_mode])
            # just one turn, always prepend image token
            inp = DEFAULT_IMAGE_TOKEN + "\n" + prompt
            conv.append_message(conv.roles[0], inp)

            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = (
                tokenizer_image_token(
                    prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                )
                .unsqueeze(0)
                .cuda()
            )

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(
            keywords, self.tokenizer, input_ids
        )
        if use_batch_data:
            streamer = BatchTextIteratorStreamer(
                len(prompt), self.tokenizer, skip_prompt=True, timeout=20.0
            )
        else:
            streamer = TextIteratorStreamer(
                self.tokenizer, skip_prompt=True, timeout=20.0
            )

        with torch.inference_mode():
            thread = Thread(
                target=self.model.generate,
                kwargs=dict(
                    inputs=input_ids,
                    images=image_tensor,
                    do_sample=do_sample,
                    num_beams=num_beams,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_tokens,
                    streamer=streamer,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                ),
            )
            thread.start()
            # workaround: second-to-last token is always " "
            # but we want to keep it if it's not the second-to-last token
            prepend_space = False
            for new_text in streamer:
                if new_text == " ":
                    prepend_space = True
                    continue
                if new_text.endswith(stop_str):
                    new_text = new_text[: -len(stop_str)].strip()
                    prepend_space = False
                elif prepend_space:
                    new_text = " " + new_text
                    prepend_space = False
                if len(new_text):
                    yield new_text
            if prepend_space:
                yield " "
            thread.join()
        # self.reset_attention_overwrite_flag()

    @staticmethod
    def pad_sequence_to_max_length(sequence, max_length, padding_value=0):
        """Pad a sequence to the desired max length."""
        if len(sequence) >= max_length:
            return sequence
        return torch.cat(
            [
                torch.full(
                    (max_length - len(sequence),), padding_value, dtype=sequence.dtype
                ),
                sequence,
            ]
        )

    def predict_nostreaming(
        self,
        image: list[Path] | Path = Input(description="Input image"),
        prompt: list[str] | str = Input(
            description="Prompt to use for text generation"
        ),
        top_p: float = Input(
            description="When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens",
            ge=0.0,
            le=1.0,
            default=1.0,
        ),
        temperature: float = Input(
            description="Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic",
            default=0.2,
            ge=0.0,
        ),
        max_tokens: int = Input(
            description="Maximum number of tokens to generate. A word is generally 2-3 tokens",
            default=1024,
            ge=0,
        ),
        do_sample: bool = Input(
            description="Whether to use sampling or greedy decoding", default=True
        ),
        num_beams: int = 1,
        use_bf16: bool = False,  # use bf16 instead of FP16, might prevent inf or negative sampling
        parallel: bool = False,  # the function will process all requests in parallel. NOTE: there is a bug because of different text length in the batch.
    ) -> ConcatenateIterator[str]:
        """Run a single prediction on the model without streaming."""

        if self.model_name.startswith("llava-v1.5"):
            conv_mode = "llava_v1"
        elif self.model_name.startswith("llava_llama3"):
            conv_mode = "llava_llama_3"
        elif self.model_name.startswith("llava_qwen"):
            conv_mode = "qwen_1_5"  # Make sure you use correct chat template for different models
        else:
            raise ValueError("Unknown model name")
        self.model.config.tokenizer_padding_side = self.tokenizer.padding_side = "left"
        if isinstance(image, list):
            image_data = [load_image(str(img)) for img in image]
            if use_bf16:
                image_tensor = [
                    (
                        self.image_processor.preprocess(image, return_tensors="pt")[
                            "pixel_values"
                        ]
                        .half()
                        .to(dtype=torch.bfloat16)
                        .cuda()
                    )
                    for image in image_data
                ]
            else:
                image_tensor = [
                    (
                        self.image_processor.preprocess(image, return_tensors="pt")[
                            "pixel_values"
                        ]
                        .half()
                        .cuda()
                    )
                    for image in image_data
                ]
            # image_tensor = torch.cat(image_tensor, dim=0)
            self.model.config.tokenizer_padding_side = self.tokenizer.padding_side = (
                "left"
            )
            image_sizes = [image.size for image in image_data]
        else:
            image_data = load_image(str(image))
            if use_bf16:
                image_tensor = (
                    self.image_processor.preprocess(image_data, return_tensors="pt")[
                        "pixel_values"
                    ]
                    .half()
                    .to(dtype=torch.bfloat16)
                    .cuda()
                )
            else:
                image_tensor = (
                    self.image_processor.preprocess(image_data, return_tensors="pt")[
                        "pixel_values"
                    ]
                    .half()
                    .cuda()
                )
            image_sizes = [image_data.size]
        # loop start
        if isinstance(prompt, list):
            input_ids_collect = []

            for sub_prompt in prompt:
                conv = conv_templates[conv_mode].copy()
                # just one turn, always prepend image token
                inp = DEFAULT_IMAGE_TOKEN + "\n" + sub_prompt
                conv.append_message(conv.roles[0], inp)

                conv.append_message(conv.roles[1], None)
                sub_prompt = conv.get_prompt()

                input_ids = tokenizer_image_token(
                    sub_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                )
                input_ids_collect.append(input_ids.cuda().unsqueeze(0))
            if parallel:
                max_len = max([len(seq) for seq in input_ids_collect])
                input_ids_collect = [
                    self.pad_sequence_to_max_length(seq.squeeze(), max_len)
                    for seq in input_ids_collect
                ]
                input_ids = torch.stack(input_ids_collect, dim=0).cuda()
            else:
                input_ids = input_ids_collect
        else:
            conv = copy.deepcopy(conv_templates[conv_mode])
            # just one turn, always prepend image token
            inp = DEFAULT_IMAGE_TOKEN + "\n" + prompt
            conv.append_message(conv.roles[0], inp)

            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = (
                tokenizer_image_token(
                    prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                )
                .unsqueeze(0)
                .cuda()
            )

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        if parallel:
            stopping_criteria = KeywordsStoppingCriteria(
                keywords, self.tokenizer, input_ids
            )
        else:
            stopping_criteria_collect = []
            for i in range(len(input_ids)):
                stopping_criteria = KeywordsStoppingCriteria(
                    keywords, self.tokenizer, input_ids[i]
                )
                stopping_criteria_collect.append(stopping_criteria)
        outputs = []
        if self.model_name.startswith("llava_qwen"):

            outputs = []
            for i in range(len(image_tensor)):

                kwargs = dict(
                    inputs=input_ids[i],
                    images=image_tensor[i],
                    do_sample=do_sample,
                    num_beams=num_beams,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_tokens,
                    use_cache=True,
                    image_sizes=[image_sizes[i]],
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                output_ids = self.model.generate(**kwargs)
                outputs.append(
                    self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
                )
        else:
            with torch.inference_mode():
                if use_bf16:
                    model = self.model.to(dtype=torch.bfloat16)
                else:
                    model = self.model.to(dtype=torch.float16)

                if parallel:

                    kwargs = dict(
                        inputs=input_ids,
                        images=image_tensor,
                        do_sample=do_sample,
                        num_beams=num_beams,
                        temperature=temperature,
                        top_p=top_p,
                        max_new_tokens=max_tokens,
                        use_cache=True,
                        stopping_criteria=[stopping_criteria],
                    )
                    if self.model_name.startswith(
                        "llava_llama3"
                    ) or self.model_name.startswith("llava_qwen"):
                        kwargs["image_sizes"] = image_sizes
                        # kwargs.pop("stopping_criteria")
                        kwargs["pad_token_id"] = self.tokenizer.eos_token_id

                    output_ids = model.generate(**kwargs)
                    if self.model_name.startswith("llava-v1.5"):
                        output_ids = output_ids[:, input_ids.shape[1] :]
                    outputs = self.tokenizer.batch_decode(
                        output_ids, skip_special_tokens=True
                    )

                else:
                    for i in range(len(image_tensor)):
                        kwargs = dict(
                            inputs=input_ids[i],
                            images=image_tensor[i],
                            do_sample=do_sample,
                            num_beams=num_beams,
                            temperature=temperature,
                            top_p=top_p,
                            max_new_tokens=max_tokens,
                            use_cache=True,
                            stopping_criteria=[stopping_criteria_collect[i]],
                        )
                        if self.model_name.startswith(
                            "llava_llama3"
                        ) or self.model_name.startswith("llava_qwen"):
                            kwargs["image_sizes"] = image_sizes
                            # kwargs.pop("stopping_criteria")
                            kwargs["pad_token_id"] = self.tokenizer.eos_token_id
                        output_ids = model.generate(**kwargs)
                        if self.model_name.startswith("llava-v1.5"):
                            output_ids = output_ids[:, input_ids[i].shape[1] :]
                        outputs.append(
                            self.tokenizer.batch_decode(
                                output_ids, skip_special_tokens=True
                            )[0]
                        )
        answers = []
        for output in outputs:
            if "value\U0001F449" in output[: len(conv.roles[1])]:
                output = output[len(conv.roles[1]) :].strip()
            else:
                output = output.strip()
            if output.endswith(stop_str):
                output = output[: -len(stop_str)]
            answers.append(output.strip())
        # self.reset_attention_overwrite_flag()
        return answers

    def predict_text_only_nostreaming(
        self,
        prompt: list[str] | str = Input(
            description="Prompt to use for text generation"
        ),
        top_p: float = Input(
            description="When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens",
            ge=0.0,
            le=1.0,
            default=1.0,
        ),
        temperature: float = Input(
            description="Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic",
            default=0.2,
            ge=0.0,
        ),
        max_tokens: int = Input(
            description="Maximum number of tokens to generate. A word is generally 2-3 tokens",
            default=1024,
            ge=0,
        ),
        do_sample: bool = Input(
            description="Whether to use sampling or greedy decoding", default=True
        ),
        num_beams: int = 1,
    ) -> ConcatenateIterator[str]:
        """Run a single prediction on the model without streaming."""
        if self.model_name.startswith("llava-v1.5"):
            conv_mode = "llava_v1"
        elif self.model_name.startswith("llava_llama3"):
            conv_mode = "llava_llama_3"
        else:
            raise ValueError("Unknown model name")

        # loop start
        if isinstance(prompt, list):
            self.model.config.tokenizer_padding_side = self.tokenizer.padding_side = (
                "left"
            )
            input_ids_collect = []

            for sub_prompt in prompt:
                conv = copy.deepcopy(conv_templates[conv_mode])
                conv.append_message(conv.roles[0], sub_prompt)
                conv.append_message(conv.roles[1], None)
                sub_prompt = conv.get_prompt()

                input_ids = tokenizer_image_token(
                    sub_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                )
                input_ids_collect.append(input_ids)
            max_len = max([len(seq) for seq in input_ids_collect])
            input_ids_collect = [
                self.pad_sequence_to_max_length(seq.squeeze(), max_len)
                for seq in input_ids_collect
            ]
            input_ids = torch.stack(input_ids_collect, dim=0).cuda()
        else:
            conv = copy.deepcopy(conv_templates[conv_mode])
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = (
                tokenizer_image_token(
                    prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                )
                .unsqueeze(0)
                .cuda()
            )

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(
            keywords, self.tokenizer, input_ids
        )
        with torch.inference_mode():
            output_ids = self.model.generate(
                inputs=input_ids,
                images=None,
                image_sizes=None,
                do_sample=do_sample,
                num_beams=num_beams,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        # if self.model_name.startswith("llava-v1.5"):
        #    output_ids = output_ids[:, input_ids.shape[1] :]
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        answers = []
        for output in outputs:
            if "value\U0001F449" in output[: len(conv.roles[1])]:
                output = output[len(conv.roles[1]) :].strip()
            else:
                output = output.strip()
            if output.endswith(stop_str):
                output = output[: -len(stop_str)]
            answers.append(output.strip())
        # self.reset_attention_overwrite_flag()
        return answers


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def write_to_file(parsed_results, filename="output.txt"):
    # Open the file for writing
    with open(filename, "w") as file:
        # Write each entry to the file
        for item in parsed_results:
            file.write(f"Item Number: {item['Item Number']}\n")
            file.write(f"Tool Used: {item['Tool Used']}\n")
            file.write(
                f"Text Prompt for Processing: {item['Text Prompt for Processing']}\n"
            )
            file.write("\n")  # Add a newline for spacing between entries


class LLaVA_Warper:
    def __init__(
        self, checkpoint_path="liuhaotian/llava-v1.5-13b", model_name="llava-v1.5-13b"
    ):
        self.predictor = Predictor()
        self.predictor.setup(checkpoint_path, model_name)

    def predict_nostreaming(
        self,
        image: Path = Input(description="Input image"),
        prompt: str = Input(description="Prompt to use for text generation"),
        top_p: float = Input(
            description="When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens",
            ge=0.0,
            le=1.0,
            default=1.0,
        ),
        temperature: float = Input(
            description="Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic",
            default=0.2,
            ge=0.0,
        ),
        max_tokens: int = Input(
            description="Maximum number of tokens to generate. A word is generally 2-3 tokens",
            default=1024,
            ge=0,
        ),
        do_sample: bool = Input(
            description="Whether to use sampling or greedy decoding", default=True
        ),
        num_beams: int = 1,
        use_bf16: bool = False,
    ) -> str:
        return self.predictor.predict_nostreaming(
            image,
            prompt,
            top_p,
            temperature,
            max_tokens,
            do_sample,
            num_beams,
            use_bf16,
        )

    def predict_text_only_nostreaming(
        self,
        prompt: list[str] | str = Input(
            description="Prompt to use for text generation"
        ),
        top_p: float = Input(
            description="When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens",
            ge=0.0,
            le=1.0,
            default=1.0,
        ),
        temperature: float = Input(
            description="Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic",
            default=0.2,
            ge=0.0,
        ),
        max_tokens: int = Input(
            description="Maximum number of tokens to generate. A word is generally 2-3 tokens",
            default=1024,
            ge=0,
        ),
    ) -> ConcatenateIterator[str]:
        return self.predictor.predict_text_only_nostreaming(
            prompt, top_p, temperature, max_tokens
        )
