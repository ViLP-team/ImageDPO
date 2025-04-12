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
    python -m pip install tyro, cog
```
## 2.(Optional) Prepare Image DPO Data

You can directly download our data from [huggingface](https://huggingface.co/datasets/ViLP/ImageDPO/tree/main) and skip this step. 
If you are interested in generating data by yourself, feel free to follow the install instructions. 
Otherwise you can skip the following steps. 

Note that processing data would be very slow. For each step, feel free to spawn several processors in parallel for acceleration. 

### 2.1 Install Image Editing Tools

We use [GroundSAM](https://github.com/IDEA-Research/Grounded-Segment-Anything), [instruct-pix2pix](https://github.com/timothybrooks/instruct-pix2pix), and [Stable-Diffusion-XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) to generate new images. 
Please follow their install instructions and install the required packages. 

### 2.2 Generate Instructions for Image Editing Tools

We prompt the VLM models to use image editing tools to generate new images. 
Basically, we ask VLMs to look at the seed images (from coco/textvqa/VG dataset) and figure out which image editing tool to use. 

```
    # Step 1: prompt VLMs to call image editing tools based on input images
    # Rember to change the used models and datasets.

    python generate_main_llava.py

    # Step 1.5: for groundingdino-based image editing, additionally generate the instructions 

    python vlm_data_generator.py --task instruction_gen --model_tool groundingdino

```

### 2.3 Call image editing tools to generate images.

```
    # Generate images using SD-XL
    python vlm_data_generator.py --task image_gen --model_tool sdxl

    # Generate images using instructp2p
    python vlm_data_generator.py --task image_gen --model_tool instructp2p

    # Generate images using groundingdino
    python vlm_data_generator.py --task image_gen --model_tool groundingdino
```

### 2.4 Generate new QAs based on generated Images

```
    python vlm_data_generator.py --task single_image_QA_gen --model_tool {IMAGE_MODEL_USED} 

```

### 2.5 Rating the generated QAs

```
    python vlm_data_generator.py --task rating_singleQA_sampleNewQA 
```

### 2.6 Corrupt the image for Image DPO

```
    python corrupt_image.py --

```


# 3. Train Image DPO
After downloading the image DPO data, you can train the image dpo following our proposed algorithms. 

```
    cd  external/LLaVA-Plus

    bash image_dpo.sh
```

Modify the 'DATA_PATH', 'IMAGE_FOLDER' and 'run_name' in 'image_dpo.sh',
also modify the training setting based on your experimental setting, in particularly lora settings ("lora_r", "lora_alpha"), "learning_rate", "gradient_accumulation_steps".  


# 4. TODOs

- [ ] Validate Code:
    - [x] Instruction Generation. 
    - [ ] QA sample 
    - [ ] Rating QA