# Self-supervised Vision Transformer are Scalable Generative Models for Domain Generalization @ MICCAI 2024
<p align="center">
    [<a href="https://arxiv.org/abs/2407.02900">Preprint</a>]
    <!--[<a href="">Publication</a>]-->
    [<a href="#citation-">Citation</a>]
</p>

## Overview 🧠
Despite notable advancements, the integration of deep learning (DL) techniques into impactful clinical applications, particularly in the realm of digital histopathology, has been hindered by challenges associated with achieving robust generalization across diverse imaging domains and characteristics. Traditional mitigation strategies in this field such as data augmentation and stain color normalization have proven insufficient in addressing this limitation, necessitating the exploration of alternative methodologies. To this end, we propose a novel generative method for domain generalization in histopathology images. Our method employs a generative, self-supervised Vision Transformer to dynamically extract characteristics of image patches and seamlessly infuse them into the original images, thereby creating novel, synthetic images with diverse attributes. By enriching the dataset with such synthesized images, we aim to enhance its holistic nature, facilitating improved generalization of DL models to unseen domains. Extensive experiments conducted on two distinct histopathology datasets demonstrate the effectiveness of our proposed approach, outperforming the state of the art substantially, on the *Camelyon17-wilds* challenge dataset (+2\%) and on a second epithelium-stroma dataset (+26\%). Furthermore, we emphasize our method's ability to readily scale with increasingly available unlabeled data samples and more complex, higher parametric architectures.

<p align="middle">
  <img src="https://github.com/sdoerrich97/vits-are-generative-models/assets/98497332/4f2414b5-9720-4059-af8d-7db646f9f83d" width="950" />
</p>

Figure 1: Our method is a self-supervised generative approach that employs feature orthogonalization to generate synthetic images. Using a single ViT encoder ($E$), we encode an image patch-wise and split the resulting embeddings, with one half preserving anatomy and the other half storing characteristic features for each patch. These feature vectors are then mixed across different input images and fed into an image synthesizer ($IS$) to create synthetic images representing new anatomy-characteristic pairs.

Subsequent sections outline the paper's [key contributions](#key-contributions-), showcase the [obtained results](#results-), and offer instructions on [accessing and utilizing the accompanying codebase](#getting-started-) to replicate the findings and train or evaluate your own models.

## Key Contributions 🔑
- Presenting a novel self-supervised generative domain generalization method for histopathology.
- Generating synthetic images with unseen combinations of anatomy and image characteristics.
- Outperforming the state of the art on the *Camelyon17-wilds* challenge dataset and an aggregated epithelium-stroma dataset (including NKI, VGH and IHC).
- Demonstrating the method's ability to scale effectively with growing availability of unlabeled data samples and the adoption of deeper architectures.

## Results 📊
We assess the domain generalization ability of our method on the *Camelyon17-wilds* challenge dataset and an aggregated epithelium-stroma dataset (including NKI, VGH and IHC). First, we [qualitatively evaluate](#image-quality-of-reconstructions-and-generated-images) our method by training it on the *Camelyon17-wilds*
dataset and assessing the image quality of the image synthesizer's reconstructions (no mixing) and generated synthetic images, which exhibit the same anatomy but varied characteristics. Furthermore, we [quantitatively evaluate](#training-set-diversity-enhancement-for-improved-automatic-disease-classification) our method's suitability for improving domain generalization. For this, we employ our stand-alone encoder to generate additional synthetic images with mixed anatomy and characteristics, augmenting the training set diversity on the fly. These synthetic images, alongside the originals, are afterward fed into a subsequent classifier allowing it to learn from a more diverse set of samples, thereby generalizing better to unseen images.

<p align="middle">
  <img src="https://github.com/sdoerrich97/vits-are-generative-models/assets/98497332/aeaf3ba4-e37f-41c8-9a68-b90f7d6b521c" width="950" />
</p>

Figure 2: Examples from the histopathology datasets used for evaluating domain generalization. Left: *Camelyon17-wilds* for which the domains are hospitals. Right: Combined epithelium-stroma dataset for which the domains are datasets.

### Image Quality of Reconstructions and Generated Images

<p align="middle">
  <img src="https://github.com/sdoerrich97/vits-are-generative-models/assets/98497332/8dfe850e-26cb-43d7-8b04-1881c177d644" width="950" />
</p>

Figure 3: Qualitative evaluation of the method. Left: Reconstruction capability on the *Camelyon17-wilds* dataset. Right: Generative capabilities on the *Camelyon17-wilds* dataset by means of synthetic images created through its anatomy-characteristics intermixing for images from the training set (rows 1, 2 and column 1, 2, 3), the validation set (row 3 and column 4), and the unseen test set (row 4 and column 5).

### Training Set Diversity Enhancement for Improved Automatic Disease Classification

<p align="middle">
  <img src="https://github.com/sdoerrich97/vits-are-generative-models/assets/98497332/2f768cd0-ad9d-41ad-b67c-a3f18d4808d4" width="950" />
</p>

Table 1: Left: Accuracy in \% on the validation and test set of \textsc{Camelyon17-wilds}. Right: Accuracy in \% on the epithelium-stroma dataset for which we train it once on NKI and evaluate it for VGH (val) and IHC (test), as well as train it on VGH and evaluate it for NKI (val) and IHC (test), respectively.

## Getting Started 🚀
### Project Structure
- [`config`](https://github.com/sdoerrich97/vits-are-generative-models/tree/main/config): Training and evaluation configurations
- [`data`](https://github.com/sdoerrich97/vits-are-generative-models/tree/main/data): Data loader
- [`models`](https://github.com/sdoerrich97/vits-are-generative-models/tree/main/models): Model structure
- [`training`](https://github.com/sdoerrich97/vits-are-generative-models/tree/main/training): Training and evaluation scripts
- [`environment.yaml`](https://github.com/sdoerrich97/vits-are-generative-models/blob/main/environment.yaml): Package Requirements
- [`utils.py`](https://github.com/sdoerrich97/vits-are-generative-models/blob/main/utils.py): Helper functions

### Installation and Requirements
#### Clone this Repository:
To clone this repository to your local machine, use the following command:
```
git clone https://github.com/sdoerrich97/vits-are-generative-models.git
```

#### Set up a Python Environment Using Conda (Recommended) 
If you don't have Conda installed, you can download and install it from [here](https://conda.io/projects/conda/en/latest/index.html).
Once Conda is installed, create a Conda environment with Python 3 (>= 3.11) in your terminal:
```
conda create --name scalableGenModels python=3.11
```
Of course, you can use a standard Python distribution as well.

#### Install Required Packages From the Terminal Using Conda (Recommended)
All required packages are listed in [`environment.yaml`](https://github.com/sdoerrich97/vits-are-generative-models/blob/main/environment.yaml).

Activate your Conda environment in your terminal:
```
conda activate scalableGenModels
```

Once Conda is activated, install PyTorch depending on your system's configuration. For example for Linux using Conda and CUDA 12.1 use the following command. For all other configurations refer to the official [PyTorch documentation](https://pytorch.org/):
```
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

Install required Python packages via Conda:
```
conda install conda-forge::matplotlib
conda install anaconda::seaborn
conda install fastai::timm
conda install pytorch-scatter -c pyg
```

Additionally, navigate to your newly created Conda environment within your Conda install and install the remaining Python Packages from [PyPI](https://pypi.org/):
```
cd ../miniconda3/envs/scalableGenModels/Scripts
pip install einops
pip install wandb
pip install -U albumentations
pip install accelerate
pip install transformers
pip install wilds
pip install torch-geometric
```

If you use a standard Python distribution instead, you need to adjust the installation steps accordingly.

### Quick Start
Once all requirements are installed, make sure the Conda environment is active and navigate to the project directory:
```
cd ../vits-are-generative-models
```

You can adjust the parameters and hyperparameters of each training/evaluation run within the respective copy within [`config`](https://github.com/sdoerrich97/vits-are-generative-models/tree/main/config). Additionally, you can specify the GPU-configuration you want to use in [`accelerate`](https://github.com/sdoerrich97/vits-are-generative-models/tree/main/config/accelerate)

Once the config files are all set, you can execute for example a pretraining or trainig run using:
```
python training/pretrain.py --config_file './config.yaml'
python training/train.py --config_file './config.yaml'
```
Please note that the project uses relative import statements. **Thus, it is important that you execute the code from the project root.**

Additionally, you can adjust some parameters on the fly. Please check out the main()-function of each training/evaluation script to see what these are. In case you intend to use Weights & Biases to track your experiments, you need to set it up respectively: [W&B Quickstart](https://docs.wandb.ai/quickstart)

Lastly, you will find all parameters (model architectures, number of epochs, learning rate, etc.) we used for our benchmark within the provided config-files within [`config`](https://github.com/sdoerrich97/vits-are-generative-models/tree/main/config) in case you want to reproduce our results. If you want to use your own models and datasets, you only need to adjust the config-file, respectively.

# Citation 📖
If you find this work useful in your research, please consider citing our paper:
- Publication: TBD
- [Preprint](https://arxiv.org/abs/2407.02900)
```
@InProceedings{doerrich2024selfSupervisedViTsAreGenerativeModelsForDG,
    title={Self-supervised Vision Transformer are Scalable Generative Models for Domain Generalization},
    author={Sebastian Doerrich and Francesco Di Salvo and Christian Ledig},
    year={2024},
    eprint={2407.02900},
    archivePrefix={arXiv}, 
}
```
