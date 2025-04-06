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

    python -m pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
```


## 1. Install according to LLava
```bash
    cd external/LLaVA-Plus

    python -m pip install -e .
    python -m pip install -e ".[train]"
    python -m pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.6.3

    python -m pip install datasets
    python -m pip install tyro
```
## 2.(Optional) Prepare Image DPO Data

You can directly download our data from [huggingface](https://huggingface.co/datasets/ViLP/ImageDPO/tree/main) and skip this step. 
But we provide the code for processing in below if you are interested. 

### 2.1 
We use [GroundSAM](https://github.com/IDEA-Research/Grounded-Segment-Anything), [instruct-pix2pix](https://github.com/timothybrooks/instruct-pix2pix), and [Stable-Diffusion-XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) to generate new images. If you are interested in generating data by yourself, feel free to follow the install instructions. 
Otherwise you can skip the following steps. 


# 3. Train Image DPO
After downloading the image DPO data, you can train the image dpo following our proposed algorithms. 

```
    cd  external/LLaVA-Plus
```

Modify the 'DATA_PATH', 'IMAGE_FOLDER' and 'run_name' in 'image_dpo.sh',
also modify the training setting based on your experimental setting, in particularly lora settings ("lora_r", "lora_alpha"), "learning_rate", "gradient_accumulation_steps".  


