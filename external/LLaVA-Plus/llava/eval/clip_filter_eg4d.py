import pickle

import clip
import torch
from PIL import Image

EGO_FILE_PATH = "/nfs/turbo/justincj-turbo/tiangel/improvingVLM/packages/LLaVA-Plus/eval_data/two_img_dataset/EGO4D/sampled_frames.pkl"


@torch.no_grad()
def main():

    ego_dict = pickle.load(open(EGO_FILE_PATH, "rb"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    for index, ego_file in ego_dict.items():
        start_image_path = ego_file["start_image_path"]
        end_image_path = ego_file["end_image_path"]

        start_image = preprocess(Image.open(start_image_path)).unsqueeze(0).to(device)
        end_image = preprocess(Image.open(end_image_path)).unsqueeze(0).to(device)

        start_image_features = model.encode_image(start_image)
        end_image_features = model.encode_image(end_image)

        # calculate the cosine similarity between the images
        image_similarity = (start_image_features @ end_image_features.T) / (
            start_image_features.norm(dim=-1) * end_image_features.norm(dim=-1)
        )

        # observe some examples, decide not to use it as the threshold


if __name__ == "__main__":
    main()
