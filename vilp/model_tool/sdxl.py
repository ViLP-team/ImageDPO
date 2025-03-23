import os

import torch
from diffusers import AutoPipelineForText2Image


class SdXL:

    def __init__(self, args):
        self.model = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        ).to("cuda")

    @classmethod
    def get_instruction_prompt(self, main_instructions, org_description):
        """
        Given the high level `main_instructions` from `generate_main_llava.py`, we generate the instructions for image generation.
        """
        # Instructpix2pix don't need this
        return None

    @classmethod
    def get_multi_image_QA_prompt(
        self, instructions, org_description, new_description=None
    ):
        """
        Given the instructions `instructions` from `get_instruction_prompt`, we generate the prompt for question-answer generation.
        """
        prompt_text = f"""
        Could you please generate potential question-answer pairs based on the descriptions of the original image, the its instruction of modificaition, and the new image? For example, generate what's the difference between two images, the location of the modified object, how to turnthe old image into the new image, how to turn the new image into the old image, and other questione-answer pairs you believe are reasonable based on the input.

        Input:
        - Original Image Description: {org_description}
        - Modification Instruction: {instructions["Text Prompt for Processing"]}

        Expected Multiple Response Format:
        Item Number: 1
        Question: Propose a question based on the descriptions provided.
        Answer: Corresponding answer to the proposed question.

        Item Number: 2
        Question:
        Answer:
        """
        return prompt_text

    @classmethod
    def get_single_image_QA_prompt(self, description):
        """
        Given the description of an image, we generate the prompt for question-answer generation.
        """
        prompt_text = r"""
        Given this image, could you please generate a series of insightful and diverse question-answer pairs based on the image and its descriptions? We are interested in exploring various facets of the image, including:
        - Holistic styles and layouts: Questions that analyze the overall design, style, and layout of the image.
        - Object-specific details: Questions that delve into particular elements or objects within the image, discussing their characteristics or functions.
        - Background context: Questions that speculate about the background story or the setting of the image.
        - Overall themes: Questions that interpret the thematic elements and messages portrayed in the image.

        We encourage creative and thought-provoking questions that extend beyond the basics. Please generate questions that cover a broader range of observations and insights drawn from the image. Each question should be followed by a comprehensive answer, providing depth and context.

        Expected Multiple Response Format:
        Item Number: 1
        Question: [Propose a unique and insightful question based on the descriptions and the images.]
        Answer: [Provide a comprehensive answer to the proposed question.]

        Item Number: 2
        Question: [Propose a unique and insightful question based on the descriptions and the images.]
        Answer: [Provide a comprehensive answer to the proposed question.]

        Please ensure each question-answer pair is well-defined and informative.

        Please provide at least 5 question-answer pairs based on the input provided.
        """
        return prompt_text

    def forward(self, org_image_path, instructions, num_gen, output_dir, file_prefix):
        """
        main functions to achieve everything.
        """
        sd_results = self.model(
            prompt=instructions["Text Prompt for Processing"],
            num_images_per_prompt=num_gen,
        ).images
        for i, image in enumerate(sd_results):
            image_path = os.path.join(output_dir, f"gen_{i:02d}.jpg")
            image.save(image_path)
