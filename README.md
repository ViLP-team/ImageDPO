This repository contains the implementation of our proposed ImageDPO method, introduced in our paper [Probing Visual Language Priors in VLMs](https://arxiv.org/abs/2501.00569). 

Our pipeline comprises three primary components:

1. Synthesizing QA Pairs: Leveraging pre-trained image generative models alongside VLMs themselves to automatically generate question-answer (QA) pairs based on existing seed images.

2. Image Corruption: Applying targeted corruptions to synthesized images to create good vs. bad pairs.

3. ImageDPO Training: Utilizing these synthesized and corrupted image-QA pairs for finetuning VLMs with our proposed ImageDPO objective.


# 1. Environment Install

```bash
    conda create -n vilp python=3.10 -y
    conda activate vilp
    python -m pip install --upgrade pip
```


## 1. Install according to LLava
```bash
    cd external/LLaVA-Plus

    python -m pip install -e .
    python -m pip install -e ".[train]"
    python -m pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.6.3
```
## 2.(Optional) Install Image Editing tool
We use [GroundSAM](https://github.com/IDEA-Research/Grounded-Segment-Anything), [instruct-pix2pix](https://github.com/timothybrooks/instruct-pix2pix), and [Stable-Diffusion-XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) to generate new images. If you are interested in generating data by yourself, feel free to follow the install instructions. 
Otherwise you can skip the following steps. 


# 2. Train Image DPO


