import os

import numpy as np
import torch
from diffusers import AutoPipelineForInpainting
from IPython import embed
from PIL import Image, ImageDraw, ImageFont

try:
    from vilp.model_warper.ground_sam import GroundSam_Warper
except:
    print("cannot import GroundSam_Warper")


class GroundingDINO:

    def __init__(
        self,
        config_file: str,
        grounded_checkpoint: str,
        sam_checkpoint: str,
        # inpainting configs
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
        inpaint_mode: str = "first",
        device: str = "cuda",
    ):

        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.inpaint_mode = inpaint_mode

        self.ground_sam_model = GroundSam_Warper(
            config_file=config_file,
            grounded_checkpoint=grounded_checkpoint,
            sam_checkpoint=sam_checkpoint,
            device=device,
        )

        self.inpainter = AutoPipelineForInpainting.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            torch_dtype=torch.float16,
            variant="fp16",
        ).to("cuda")
        self.inpainter = self.inpainter.to("cuda")

    @classmethod
    def get_instruction_prompt(self, main_instructions, org_description):
        """
        Given the high level `main_instructions` from `generate_main_llava.py`, we generate the instructions for image generation.
        """
        prompt_text = f"""Analyze the provided image and its accompanying modification instruction to identify the removed object description, the new object description, and the new image description.

            Modification Instructions: {main_instructions['Text Prompt for Processing']}

            Expected Multiple Response Format:
            Item Number: 1
            Removed Object Description: [Brief description of the object to be detected and removed]
            New Object Description: [Description of a new, different object to replace the removed one]
            New Image Description: [Description of the image after each object's removal, focusing on changes and remaining elements]

            Item Number: 2
            Removed Object Description: [Brief description of the object to be detected and removed]
            New Object Description: [Description of a new, different object to replace the removed one]
            New Image Description: [Description of the image after each object's removal, focusing on changes and remaining elements]

            """
        return prompt_text

    @classmethod
    def get_multi_image_QA_prompt(
        self, instructions, org_description, new_description=None
    ):
        """
        Given the instructions `instructions` from `get_instruction_prompt`, we generate the prompt for question-answer generation.
        """
        if new_description is None:
            new_description = instructions["New Object Description"]
        prompt_text = f"""
        Could you please generate a series of insightful and creative question-answer pairs based on the visual and descriptive changes observed between two images? Examine the differences and modifications with a fresh perspective, exploring various aspects such as:

        - Differences: Discuss both visible and thematic differences. What new themes emerge in the modified image? How do these changes alter the narrative or mood?
        - Modifications: Describe in detail the transformations from the original to the new image. What elements were added or removed, and why might these changes have been made?
        - Contextual Relevance: Analyze the significance of the changes. How do the added or removed elements influence the overall context or message of the images?
        - Creative Interpretations: Imagine and suggest hypothetical scenarios or stories that could explain why the changes were made. What backstory could account for these modifications?

        Encourage creativity in the formulation of questions and answers, expanding beyond the given examples to include other relevant aspects observed from the inputs. Each question should invite deep analysis or imaginative exploration based on the descriptions and the visual content of the images.

        Input:
        The modified image is provided for reference at the beginning.
        Original Image Description: {org_description}
        New Image Description: {new_description}
        Removed Object Description: {instructions["Removed Object Description"]}
        New Object Description: {instructions["New Object Description"]}

        Expected Multiple Response Format:
        Item Number: 1
        Question: [Propose a unique and insightful question based on the descriptions and the images.]
        Answer: [Provide a comprehensive answer to the proposed question.]

        Item Number: 2
        Question: [Propose a unique and insightful question based on the descriptions and the images.]
        Answer: [Provide a comprehensive answer to the proposed question.]

        Please ensure each question-answer pair is well-defined, informative, and extends beyond the examples to explore new dimensions of the image modifications.

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
        # TODO: This part is not batchlized.
        image_pil, image = GroundSam_Warper.load_images(org_image_path)
        boxes_filt, pred_phrases = self.ground_sam_model.text_dino_grounding(
            image,
            instructions["Removed Object Description"],
            self.box_threshold,
            self.text_threshold,
        )
        try:
            masks, boxes_filt = self.ground_sam_model.sam_bbox(
                org_image_path, image_pil.size, boxes_filt
            )
        except:
            print("No object detected")
            print("save the org image")
            return None
            # for i in range(num_gen):
            #     image_pil.save(
            #         os.path.join(output_dir, f"gen_{file_prefix}_{i:02d}.jpg")
            #     )

        self.ground_sam_model.draw_bbox(
            org_image_path,
            masks,
            boxes_filt,
            pred_phrases,
            os.path.join(
                output_dir,
                f"grounded_output_{file_prefix}.jpg",
            ),
        )
        if self.inpaint_mode == "merge":
            masks = torch.sum(masks, dim=0).unsqueeze(0)
            masks = torch.where(masks > 0, True, False)
        # select the biggest mask
        index_biggest = torch.argmax(torch.sum(masks, [1, 2, 3])).item()
        mask = masks[index_biggest][0].cpu().numpy()
        mask_pil = Image.fromarray(mask)
        size = image_pil.size
        image_pil = image_pil.resize((1024, 1024))
        mask_pil = mask_pil.resize((1024, 1024))

        paint_results = self.inpainter(
            prompt=instructions["New Object Description"],
            image=image_pil,
            mask_image=mask_pil,
            num_images_per_prompt=num_gen,
        ).images
        # skip if the mask is too small (less than 10% of the original image size)
        if np.sum(mask) / np.prod(mask.shape) < 0.1:
            return paint_results

        for i in range(len(paint_results)):
            image = paint_results[i]
            image = image.resize(size)
            inpaint_image_path = os.path.join(
                output_dir, f"gen_{file_prefix}_{i:02d}.jpg"
            )
            image.save(inpaint_image_path)
        return paint_results
