import argparse
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
import os
import subprocess


def merge_lora(args):
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, device_map='cpu')

    model.save_pretrained(args.save_model_path)
    tokenizer.save_pretrained(args.save_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, required=True)
    parser.add_argument("--save-model-path", type=str, required=True)

    args = parser.parse_args()

    merge_lora(args)

    # Define the command
    command = [
        'python',
        'generate_eval_scripts.py',
        '--output-file', args.model_path.split('/')[1],
        '--ckpt-path', args.save_model_path,
    ]
    # Execute the command
    result = subprocess.run(command, capture_output=True, text=True)
    
    # Print the output and error messages
    print("Output:")
    print(result.stdout)
    print("Error:")
    print(result.stderr)
