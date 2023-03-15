# Prompt, Generate, then Cache

Official implementation of ['Prompt, Generate, then Cache: Cascade of Foundation Models makes Strong Few-shot Learners'](https://arxiv.org/pdf/2303.02151.pdf).

The paper has been accepted by **CVPR 2023** 🔥.

## News
* Please check our latest work ['Point-NN, Parameter is Not All You Need'](https://arxiv.org/pdf/2303.08134.pdf) with [code](https://github.com/ZrrSkywalker/Point-NN), accepted by **CVPR 2023** 🔥, which conducts 3D understanding without ant parameters or training.
* We will soon release CaFo cascaded with [ChatGPT](https://openai.com/blog/chatgpt) and [Stable Diffusion](https://github.com/CompVis/stable-diffusion) 📌.
* The code of CaFo has been released.
* The CaFo model is developed based on [Tip-Adapter](https://arxiv.org/pdf/2207.09519), accepted by **ECCV 2022** and [open-sourced](https://github.com/gaopengcuhk/Tip-Adapter).

## Introduction
We propose **CaFo**, a **Ca**scade of **Fo**undation models that incorporates diverse prior knowledge of various pre-trianing paradigms for better few-shot learning, including CLIP, DINO, DALL-E, and GPT-3. Specifically, CaFo works by **`Prompt, Generate, then Cache'**. We leverage GPT-3 to prompt CLIP with rich linguistic semantics and generate synthetic images via DALL-E to expand the few-shot training data. Then, we introduce a learnable cache model to adaptively blend the predictions from CLIP and DINO. By such collaboration, CaFo can fully unleash the potential of different pre-training methods and unify them to perform *state-of-the-art* for few-shot classification.

<div align="center">
  <img src="CaFo.png"/>
</div>

## Requirements
### Installation
Create a conda environment and install dependencies:
```bash
git clone https://github.com/ZrrSkywalker/CaFo.git
cd CaFo

conda create -n cafo python=3.7
conda activate cafo

pip install -r requirements.txt

# Install the according versions of torch and torchvision
conda install pytorch torchvision cudatoolkit
```

### Dataset
Please follow [DATASET.md](https://github.com/gaopengcuhk/Tip-Adapter/blob/main/DATASET.md) to download official ImageNet and other 10 datasets.

### Foundation Models
* The pre-tained weights of **CLIP** will be automatically downloaded by running.
* The prompts produced by GPT-3 have been stored at `gpt_file/`.
* Please download **DINO's** pre-trained ResNet-50 from [here](https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth), and put it under `dino/`.
* Please download **DALL-E's** generated images from [here](https://drive.google.com/drive/folders/1e249OgUFCmpfEDPsxCVR-nNb6Q1VaZVW?usp=sharing), and organize them with the official datasets like
```
$DATA/
|–– imagenet/
|–– caltech-101/
|–– oxford_pets/
|–– ...
|–– dalle_imagenet/
|–– dalle_caltech-101/
|–– dalle_oxford_pets/
|–– ...
```

## Get Started
### Configs
The running configurations for different `[dataset]` with `[k]` shots can be modified in `configs/[dataset]/[k]shot.yaml`, including visual encoders and hyperparamters. We have provided the configurations for reproducing the results in the paper. You can edit the `search_scale`, `search_step`, `init_beta` and `init_alpha` for fine-grained tuning and better results.

Note that the default `load_cache` and `load_pre_feat` are `False` for the first running, which will store the cache model and val/test features in `configs/dataset/`. For later running, they can be set as `True` for faster hyperparamters tuning.

### Running
For 16-shot ImageNet dataset:
```bash
CUDA_VISIBLE_DEVICES=0 python main_imagenet.py --config configs/imagenet/16shot.yaml
```
For other 10 datasets:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config configs/dataset/16shot.yaml
```

## Acknowledgement
This repo benefits from [Tip-Adapter](https://github.com/gaopengcuhk/Tip-Adapter), [CLIP](https://github.com/openai/CLIP), [DINO](https://github.com/facebookresearch/dino), [DALL-E](https://github.com/borisdayma/dalle-mini) and [CuPL](https://github.com/sarahpratt/CuPL). Thanks for their wonderful works.


## Citation
```bash
@article{zhang2023prompt,
  title={Prompt, Generate, then Cache: Cascade of Foundation Models makes Strong Few-shot Learners},
  author={Renrui Zhang and Xiangfei Hu and Bohao Li and Siyuan Huang and Hanqiu Deng and Hongsheng Li and Yu Qiao and Peng Gao},
  journal={arXiv preprint arXiv:2303.02151},
  year={2023}
}
```

## Contributors
[Renrui Zhang](https://github.com/ZrrSkywalker), [Xiangfei Hu](https://github.com/hxf42), [Bohao Li](https://github.com/Bohao-Lee)

## Contact
If you have any question about this project, please feel free to contact zhangrenrui@pjlab.org.cn and sjtuhxf@sjtu.edu.cn.
