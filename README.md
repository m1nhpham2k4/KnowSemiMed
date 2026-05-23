# KnowSemiMed

This repository contains the implementation for the graduation thesis project: **Research on Semi-supervised Learning Methods for Medical Image Segmentation under Limited Label Conditions**

The project focuses on applying and improving semi-supervised medical image segmentation methods by leveraging SAM-based prompting and knowledge distillation techniques.


## Installation

To set up the environment and install dependencies, run:

```bash
pip install -r requirements.txt
```

## Dataset

The datasets (`ColonDB`, `PROMISE12`, and `Lesions`) are organized for semi-supervised training, containing labeled, unlabeled, and validation subsets for various split ratios (5%, 10%, and 30%), along with a dedicated testing folder.

For each data split (e.g., `_5`, `_10`, `_30`), the folder structure is identical. Each directory contains separate folders for input `image` files and their corresponding ground truth `mask` files, where pairs share the exact same filename.

Below are the detailed directory tree structures for each dataset:

### 1. ColonDB Dataset
```text
ColonDB/
├── ColonDB_5/
│   ├── labeled/
│   │   ├── image/           # 5% Labeled polyp images (e.g., 101.png)
│   │   └── mask/            # Ground truth polyp masks (e.g., 101.png)
│   ├── unlabeled/
│   │   ├── image/           # Unlabeled polyp images
│   │   └── mask/            # Ground truth polyp masks (unused during training)
│   ├── val/
│   │   ├── image/           # Validation images
│   │   └── mask/            # Validation ground truth masks
│   └── TestDataset/
│       └── CVC-ColonDB/
│           ├── image/       # Test set polyp images
│           └── mask/        # Test set ground truth masks
├── ColonDB_10/              # (Similar structure to ColonDB_5, using 10% labeled data)
└── ColonDB_30/              # (Similar structure to ColonDB_5, using 30% labeled data)
```

### 2. PROMISE12 Dataset
```text
PROMISE12/
├── promise_5/
│   ├── labeled/
│   │   ├── image/           # 5% Labeled prostate MRI slices (e.g., 01_01.png)
│   │   └── mask/            # Ground truth prostate masks (e.g., 01_01.png)
│   ├── unlabeled/
│   │   ├── image/           # Unlabeled prostate MRI slices
│   │   └── mask/            # Ground truth prostate masks (unused during training)
│   └── val/
│       ├── image/           # Validation prostate MRI slices
│       └── mask/            # Validation ground truth masks
├── promise_10/              # (Similar structure to promise_5, using 10% labeled data)
├── promise_30/              # (Similar structure to promise_5, using 30% labeled data)
└── Test/
    └── test/
        ├── image/           # Test set prostate MRI slices
        └── mask/            # Test set ground truth masks
```

### 3. Lesions Dataset (ISIC2018)
```text
Lesions/
├── Lesions_5/
│   ├── labeled/
│   │   ├── image/           # 5% Labeled skin lesion images (e.g., ISIC_0000000.png)
│   │   └── mask/            # Ground truth lesion masks (e.g., ISIC_0000000.png)
│   ├── unlabeled/
│   │   ├── image/           # Unlabeled skin lesion images
│   │   └── mask/            # Ground truth lesion masks (unused during training)
│   ├── val/
│   │   ├── image/           # Validation skin lesion images
│   │   └── mask/            # Validation ground truth masks
│   └── TestDataset/
│       ├── image/           # Test set skin lesion images
│       └── mask/            # Test set ground truth masks
├── Lesions_10/              # (Similar structure to Lesions_5, using 10% labeled data)
└── Lesions_30/              # (Similar structure to Lesions_5, using 30% labeled data)
```


## Training
To train the model on a dataset, execute:
```bash
python train_semi_SAM.py --model_type "vit_b" --sam_checkpoint "sam_vit_b_01ec64.pth"
```
With MedSAM checkpoint: --model_type "vit_b" --sam_checkpoint "medsam_vit_b.pth" 
With SAM2 checkpoint: --model_type "sam2.1_hiera_large" --sam_checkpoint "sam2.1_hiera_large.pt" and execute:
```text
Model/
├── ImageEncoder/
├── sam/
├── sam2_source/
│   ├── configs/
│   └── ...
├── discriminator.py
├── model_1unet.py
├── model_1vnet.py
├── model_2unet.py
├── model_2vnet.py
├── model.py
├── prompt.py
├── unet.py
└── vnet.py
```
```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
```



## Prediction
After training, you can make predictions and visual using:
```bash
python prediction_vis.py
```


## Acknowledgements
Our code is based on [taozh2017](https://github.com/taozh2017/KnowSAM).

## Questions
If you have any questions, welcome contact me at 'pvdminh.work@gmail.com'
