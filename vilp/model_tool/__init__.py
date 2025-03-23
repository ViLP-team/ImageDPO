from vilp.model_tool.groundingDINO import GroundingDINO
from vilp.model_tool.instructpix2pix import InstructPix2Pix
from vilp.model_tool.sdxl import SdXL


def get_instruction_prompt(model_tool, main_instructions, org_description):
    if model_tool == "groundingdino":
        return GroundingDINO.get_instruction_prompt(main_instructions, org_description)
    elif model_tool == "instructp2p":
        print("instructp2p do NOT generate more instructions")
        return InstructPix2Pix.get_instruction_prompt(
            main_instructions, org_description
        )
    elif model_tool == "sdxl":
        print("sdxl do NOT generate more instructions")
        return SdXL.get_instruction_prompt(main_instructions, org_description)


def get_single_image_QA_prompt(model_tool, description):
    if model_tool == "groundingdino":
        return GroundingDINO.get_single_image_QA_prompt(description)
    elif model_tool == "instructp2p":
        return InstructPix2Pix.get_single_image_QA_prompt(description)
    elif model_tool == "sdxl":
        return SdXL.get_single_image_QA_prompt(description)


def get_multi_image_QA_prompt(
    model_tool, instructions, org_description, new_description=None
):
    if model_tool == "groundingdino":
        return GroundingDINO.get_multi_image_QA_prompt(
            instructions, org_description, new_description
        )
    elif model_tool == "instructp2p":
        return InstructPix2Pix.get_multi_image_QA_prompt(
            instructions, org_description, new_description
        )
    elif model_tool == "sdxl":
        return SdXL.get_multi_image_QA_prompt(
            instructions, org_description, new_description
        )


def get_tool_model(model_tool: str, args):
    if model_tool == "groundingdino":
        model = GroundingDINO(
            config_file=args.config,
            grounded_checkpoint=args.grounded_checkpoint,
            sam_checkpoint=args.sam_checkpoint,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            inpaint_mode=args.inpaint_mode,
            device=args.device,
        )
    elif model_tool == "instructp2p":
        model = InstructPix2Pix(args)
    elif model_tool == "sdxl":
        model = SdXL(args)
    else:
        raise ValueError(f"model_tool {model_tool} not supported")
    return model


instruction_model_tool_file_name_mapper = {
    "groundingdino": "detect_inpaint_instructions",
}
qa_model_tool_file_name_mapper = {}
tool_name_mapper = {
    "groundingdino": "GroundingDINO",
    "instructp2p": "InstructPix2Pix",
    "sdxl": "Stable Diffusion",
}
