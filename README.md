This repository contains the implementation of our proposed ImageDPO method, introduced in our paper [Probing Visual Language Priors in VLMs](https://arxiv.org/abs/2501.00569). 

Our pipeline comprises three primary components:

1. Synthesizing QA Pairs: Leveraging pre-trained image generative models alongside VLMs themselves to automatically generate question-answer (QA) pairs based on existing seed images.

2. Image Corruption: Applying targeted corruptions to synthesized images to create good vs. bad pairs.

3. ImageDPO Training: Utilizing these synthesized and corrupted image-QA pairs for finetuning VLMs with our proposed ImageDPO objective.



