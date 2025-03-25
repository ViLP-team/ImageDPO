import os
import shlex

import torch

try:
    from vilp.model_wrapper.intructp2p_wrapper import InstructPix2Pix_wrapper
except:
    print("cannot import InstructPix2Pix_wrapper")


class InstructPix2Pix:
    def __init__(self, args):
        self.model = InstructPix2Pix_wrapper()

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
        Could you generate detailed question-answer pairs by analyzing both the original image and its modified version based on the provided descriptions and instructions? Focus on identifying key differences, explaining modifications, and providing insights into the visual changes and their implications.

        Instructions:
        1. Analyze the description of the original image and the modification instructions to understand the changes made.
        2. Identify and articulate specific questions that delve into the differences between the original and modified images, the process of modification, and the visual and conceptual impact of these changes.
        3. Provide clear and concise answers to each question based on the input provided.

        4. Additional questions should explore other relevant aspects observed from the inputs, such as: the impact of modifications on the overall theme of the image, potential interpretations or messages conveyed through the modified image.

        Input:
        - The modified image is provided at the beginning for reference.
        - Original Image Description: {org_description}
        - Modification Instruction: {instructions["Text Prompt for Processing"]}

        Expected Response Format:
        Item Number: 1
        Question: [Propose a unique and insightful question based on the descriptions and the images.]
        Answer: [Provide a comprehensive answer to the proposed question.]
        Item Number: 2
        Question: [Propose a unique and insightful question based on the descriptions and the images.]
        Answer: [Provide a comprehensive answer to the proposed question.]

        Please ensure each question-answer pair is well-defined, informative, and diverse, avoiding repetition and expanding on different aspects of the images and modifications.

        Please provide at least 5 question-answer pairs based on the input provided.
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
        input_path_escaped = shlex.quote(str(org_image_path))  # Convert Path to string

        output_path = os.path.join(output_dir, "gen_00.jpg")

        output_path_escaped = shlex.quote(str(output_path))  # Convert Path to string
        edit_prompt_escaped = shlex.quote(instructions["Text Prompt for Processing"])
        self.model.instruction_edit(
            input_path_escaped, output_path_escaped, edit_prompt_escaped
        )
