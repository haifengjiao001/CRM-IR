# [Neurips 2025]End-to-End Low-Light Enhancement for Object Detection with Learned Metadata from RAWs
Although RAW images offer advantages over sRGB by avoiding ISP-induced distortion and preserving more information in low-light conditions, their widespread use is limited due to high storage costs, transmission burdens, and the need for significant architectural changes for downstream tasks. To address the issues, this paper explores a new raw-based machine vision paradigm, termed Compact RAW Metadata-guided Image Refinement (CRM-IR). In particular, we propose a Machine Vision-oriented Image Refinement (MV-IR) module that refines sRGB images to better suit machine vision preferences, guided by learned raw metadata. In detail, we propose a Cross-Modal Contextual Entropy (CMCE) network for raw metadata extraction and compression. It builds upon the latent representation and entropy modeling framework of learned image compression methods, and uniquely exploits the contextual correspondence between raw images and their sRGB counterparts to achieve more efficient and compact metadata representation. Additionally, we integrate priors derived from the ISP pipeline to simplify the refinement process, enabling a more efficient design. Such a design allows the CRM-IR to focus on extracting the most essential metadata from raw images to support downstream machine vision tasks, while remaining plug-and-play and fully compatible with existing imaging pipelines, without any changes to model architectures or ISP modules. We implement our CRM-IR scheme on various object detection networks, and extensive experiments under low-light conditions demonstrate that it can significantly improve performance with an additional bitrate cost of less than  10−3 bits perpixel.

<img width="5267" height="2735" alt="framework" src="https://github.com/user-attachments/assets/ad306472-93a5-4632-9646-e6905e6648f0" />

## Requirements
- Python >= 3.10
- PyTorch >= 2.1


## Data Preparation
Dataset link: 

- [LOD Dataset](https://github.com/ying-fu/LODDataset)

- [RID Dataset](https://github.com/ying-fu/LODDataset)

Please organize your datasets into the following directory structure:
```bash
data/
├── train/
│   ├── denoise/    
│   ├── dehazy/     
│   ├── derainL/    
│   └── derainH/    
└── test/
    ├── denoise/    
    ├── dehazy/   
    ├── derainL/   
    └── derainH/    
```

## Training
To train the model:

```bash
python train.py

```



## Testing
Download the pretrained checkpoint here: [Download Link]() and place it in `checkpoints/`.

To evaluate the model:

```bash
python test.py

```

## Acknowledgement
This code is based on the [YOLOv3](https://github.com/bubbliiiing/yolo3-pytorch) and [R2LCM](https://github.com/wyf0912/R2LCM) repositories.




