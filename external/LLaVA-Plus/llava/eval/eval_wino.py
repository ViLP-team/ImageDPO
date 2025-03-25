import argparse
import json
import os


def read_jsonl_file(filepath):
    data = []
    with open(filepath) as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                print("error", line)
                continue
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers-file", type=str, default="/nfs/turbo/justincj-turbo/tiangel/improvingVLM/eval_data/single_img_dataset/winoground/answers/llava-v1.5-13b.jsonl")
    args = parser.parse_args()

    answers = read_jsonl_file(args.answers_file)
    correct = 0
    for answer in answers:
        if answer['gen_answers'] == ["A", "B"]:
            correct += 1
    print(f"Accuracy: {correct / len(answers)}")
