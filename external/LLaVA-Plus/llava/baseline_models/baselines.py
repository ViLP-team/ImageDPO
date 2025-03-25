import torch

# grab model checkpoint from huggingface hub
from huggingface_hub import hf_hub_download
from lavis.models import load_model_and_preprocess

# from mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
# from mplug_owl.tokenization_mplug_owl import MplugOwlTokenizer
# from mplug_owl.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    IdeficsConfig,
    IdeficsForVisionText2Text,
    IdeficsModel,
    TextStreamer,
)
from transformers.generation import GenerationConfig


def merge_images_horizontally(image1, image2):
    """
    Merges two images horizontally and saves the result.

    :param image_path1: Path to the first image.
    :param image_path2: Path to the second image.
    """

    # Get dimensions
    width1, height1 = image1.size
    width2, height2 = image2.size
    if height1 != height2:
        new_width2 = int(width2 * (height1 / height2))
        image2 = image2.resize((new_width2, height1), Image.LANCZOS)
        width2 = new_width2
        height2 = height1

    # Create a new image with a width equal to the sum of both images' widths and the height of the taller image
    new_image = Image.new("RGB", (width1 + width2, max(height1, height2)))

    # Paste the first image at the left side
    new_image.paste(image1, (0, 0))

    # Paste the second image at the right side
    new_image.paste(image2, (width1, 0))
    return new_image


def init_base_models(model_name: str, device):
    if model_name == "BLIP2":
        model, vis_processors, _ = load_model_and_preprocess(
            name="blip2_t5",
            model_type="pretrain_flant5xxl",
            is_eval=True,
            device=device,
        )
        return {"processor": vis_processors, "model": model}
    elif model_name == "InstructBLIP":
        model, vis_processors, _ = load_model_and_preprocess(
            name="blip2_vicuna_instruct",
            model_type="vicuna7b",
            is_eval=True,
            device=device,
        )
        return {"processor": vis_processors, "model": model}
    elif model_name == "InstructBLIP13b":
        model, vis_processors, _ = load_model_and_preprocess(
            name="blip2_vicuna_instruct",
            model_type="vicuna13b",
            is_eval=True,
            device=device,
        )
        return {"processor": vis_processors, "model": model}
    elif model_name == "IDEFICS":
        model = IdeficsForVisionText2Text.from_pretrained(
            "HuggingFaceM4/idefics-9b-instruct"
        )
        processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics-9b-instruct")
        return {"processor": processor, "model": model}
    elif model_name == "Qwen-VL-Chat":
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen-VL-Chat", trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True
        ).eval()
        return {"processor": tokenizer, "model": model}
    elif model_name == "open-flamingo":
        from open_flamingo import create_model_and_transforms

        # NOTE: It need installation of open-flamingo
        # python -m pip install open-flamingo
        model, image_processor, tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path="anas-awadalla/mpt-7b",
            tokenizer_path="anas-awadalla/mpt-7b",
            cross_attn_every_n_layers=4,
        )
        checkpoint_path = hf_hub_download(
            "openflamingo/OpenFlamingo-9B-vitl-mpt7b", "checkpoint.pt"
        )
        model.load_state_dict(torch.load(checkpoint_path), strict=False)
        return {"processor": image_processor, "model": model, "tokenizer": tokenizer}
    elif model_name == "mplug_owl":
        from mplug_owl2.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from mplug_owl2.conversation import SeparatorStyle, conv_templates
        from mplug_owl2.mm_utils import (
            KeywordsStoppingCriteria,
            get_model_name_from_path,
            process_images,
            tokenizer_image_token,
        )
        from mplug_owl2.model.builder import load_pretrained_model

        model_path = "MAGAer13/mplug-owl2-llama2-7b"

        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path,
            None,
            model_name,
            load_8bit=False,
            load_4bit=False,
            device="cuda",
        )

        return {"processor": image_processor, "model": model, "tokenizer": tokenizer}
    else:
        raise ValueError("Model not found")


def eval_base_models(model_name, models, data, device):
    if (
        model_name == "BLIP2"
        or model_name == "InstructBLIP"
        or model_name == "InstructBLIP13b"
    ):
        input_img_path = data["input_img_path"]

        # not sure if this is the correct way to process multiple images
        input_tensor = []
        if len(input_img_path) == 2:
            for img in input_img_path:
                img = Image.open(img).convert("RGB")
                image = models["processor"]["eval"](img).unsqueeze(0).to(device)
                input_tensor.append(image)
        elif len(input_img_path) == 1:
            img = Image.open(input_img_path[0]).convert("RGB")
            image = models["processor"]["eval"](img).unsqueeze(0).to(device)
            input_tensor = image
        # image1 = Image.open(input_img_path[0]).convert("RGB")
        # image2 = Image.open(input_img_path[1]).convert("RGB")
        # image = merge_images_horizontally(image1, image2)
        # input_tensor = models['processor']["eval"](image).unsqueeze(0).to(device)

        output = models["model"].generate(
            {"image": input_tensor, "prompt": data["prompt"]}
        )[0]
        return output

    elif model_name == "IDEFICS":
        # NOTE: Double check the prompts
        # prompts = [
        #     [
        #         "User:",
        #         dogs_image_url_1,
        #         "Describe this image.\nAssistant: An image of two dogs.\n",
        #         "User:",
        #         dogs_image_url_2,
        #         "Describe this image.\nAssistant:",
        #     ]
        # ]
        processor = models["processor"]
        model = models["model"]
        if len(data["input_img_path"]) == 2:
            prompts = [
                [
                    "User: first image is:",
                    Image.open(data["input_img_path"][0]).convert("RGB"),
                    "second image is",
                    Image.open(data["input_img_path"][1]).convert("RGB"),
                    f"{data['prompt']} \nAssistant:",
                ]
            ]
        elif len(data["input_img_path"]) == 1:

            if data["input_img_path"][0] is not None:
                prompts = [
                    [
                        "User:",
                        f"{data['prompt']}",
                        Image.open(data["input_img_path"][0]).convert("RGB"),
                        "<end_of_utterance>" "\nAssistant:",
                    ]
                ]
            else:
                prompts = [
                    ["User:", f"{data['prompt']}", "<end_of_utterance>" "\nAssistant:"]
                ]

        else:
            raise ValueError("Invalid number of images")
        inputs = processor(prompts, return_tensors="pt")
        generate_ids = model.generate(**inputs, max_new_tokens=32)
        output = processor.batch_decode(generate_ids, skip_special_tokens=True)
        output = output[0]
        input_length = 0
        # for prompt in prompts[0]:
        #     if isinstance(prompt, str):
        #         input_length += len(prompt)
        # output = output[input_length:]
        output = output.split("\nAssistant: ")[1]
        return output
    elif model_name == "Qwen-VL-Chat":
        # NOTE: Do it later
        tokenizer = models["processor"]
        model = models["model"]
        if len(data["input_img_path"]) == 2:
            query = tokenizer.from_list_format(
                [
                    # {"text": "First Image:"},
                    {"text": "Old Image:"},
                    {"image": data["input_img_path"][0]},
                    # {"text": "Second Image:"},
                    {"text": "New Image:"},
                    {"image": data["input_img_path"][1]},
                    {"text": data["prompt"]},
                ]
            )
        elif len(data["input_img_path"]) == 1:
            query = tokenizer.from_list_format(
                [
                    {"image": data["input_img_path"][0]},
                    {"text": data["prompt"]},
                ]
            )
        else:
            raise ValueError("Invalid number of images")

        response, history = model.chat(tokenizer, query=query, history=None)
        return response

    elif model_name == "open-flamingo":
        # https://github.com/mlfoundations/open_flamingo

        image_1 = Image.open(data["input_img_path"][0])
        image_2 = Image.open(data["input_img_path"][1])
        vision_x = [
            models["processor"](image_1).unsqueeze(0),
            models["processor"](image_2).unsqueeze(0),
        ]
        vision_x = torch.cat(vision_x, dim=0)
        vision_x = vision_x.unsqueeze(1).unsqueeze(0)

        models["tokenizer"].padding_side = (
            "left"  # For generation padding tokens should be on the left
        )
        # not sure how to construct the input text
        # lang_x = tokenizer(
        #     ["<image>An image of two cats.<|endofchunk|><image>An image of a bathroom sink.<|endofchunk|><image>An image of"],
        #     return_tensors="pt",
        # )
        input_prompt = f"<image> This is the first image. <|endofchunk|> <image> This is the second image. <|endofchunk|> Question: {data['prompt']} Answer:"
        lang_x = models["tokenizer"](
            [input_prompt],
            return_tensors="pt",
        )

        generated_text = models["model"].generate(
            vision_x=vision_x,
            lang_x=lang_x["input_ids"],
            attention_mask=lang_x["attention_mask"],
            max_new_tokens=32,
            num_beams=3,
        )
        output = models["tokenizer"].decode(generated_text[0])
        output = output[len(input_prompt) :]
        output = output.split("<|endofchunk|>")[0]
        return output

    elif model_name == "mplug_owl":
        from mplug_owl2.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from mplug_owl2.conversation import SeparatorStyle, conv_templates
        from mplug_owl2.mm_utils import (
            KeywordsStoppingCriteria,
            get_model_name_from_path,
            process_images,
            tokenizer_image_token,
        )
        from mplug_owl2.model.builder import load_pretrained_model

        if len(data["input_img_path"]) == 2:
            conv = conv_templates["mplug_owl2"].copy()
            roles = conv.roles

            image_processor = models["processor"]
            model = models["model"]
            tokenizer = models["tokenizer"]
            query = data["prompt"]
            image_tensor = []
            for i in range(len(data["input_img_path"])):
                image = Image.open(data["input_img_path"][i]).convert("RGB")
                max_edge = max(
                    image.size
                )  # We recommand you to resize to squared image for BEST performance.
                image = image.resize((max_edge, max_edge))

                image_tensor_ = process_images([image], image_processor)
                image_tensor_ = image_tensor_.to(model.device, dtype=torch.float16)
                image_tensor.append(image_tensor_)

            inp = DEFAULT_IMAGE_TOKEN + query
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = (
                tokenizer_image_token(
                    prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                )
                .unsqueeze(0)
                .to(model.device)
            )
            stop_str = conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            streamer = TextStreamer(
                tokenizer, skip_prompt=True, skip_special_tokens=True
            )

            temperature = 0.2
            max_new_tokens = 512

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    streamer=streamer,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                )
            outputs = tokenizer.decode(output_ids[0, input_ids.shape[1] :]).strip()
            return outputs
        elif len(data["input_img_path"]) == 1:

            conv = conv_templates["mplug_owl2"].copy()
            roles = conv.roles
            image_file = data["input_img_path"][0]
            image_processor = models["processor"]
            model = models["model"]
            tokenizer = models["tokenizer"]
            query = data["prompt"]

            if image_file is not None:
                image = Image.open(image_file).convert("RGB")
                max_edge = max(
                    image.size
                )  # We recommand you to resize to squared image for BEST performance.
                image = image.resize((max_edge, max_edge))

                image_tensor = process_images([image], image_processor)
                image_tensor = image_tensor.to(model.device, dtype=torch.float16)
            else:
                image_tensor = None

            inp = DEFAULT_IMAGE_TOKEN + query
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = (
                tokenizer_image_token(
                    prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                )
                .unsqueeze(0)
                .to(model.device)
            )
            stop_str = conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            streamer = TextStreamer(
                tokenizer, skip_prompt=True, skip_special_tokens=True
            )

            temperature = 0.2
            max_new_tokens = 512

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    streamer=streamer,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                )

            outputs = tokenizer.decode(output_ids[0, input_ids.shape[1] :]).strip()
            return outputs
    else:
        raise ValueError("Model not found")
