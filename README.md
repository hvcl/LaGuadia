# LaGuadia: Language-Guided Adaptive Distillation from Pathology Foundation Models [MICCAI 2026]

[Gangsu Kim](https://scholar.google.com/citations?user=CmGABBYAAAAJ&hl=ko&oi=ao), and [Won-Ki Jeong](https://hvcl.korea.ac.kr/?page_id=359)†, [HVCL@KU](https://hvcl.korea.ac.kr/)  
† Corresponding Author

## Overview
![Overview](./assets/overview.jpg)
LaGuadia (Language-Guided Adaptive DistillAtion), a framework that develops a compact pathology image encoder by dynamically integrating expertise from multiple PFMs under clinical linguistic guidance

## ⚙️ Installation
### 0. Inatall CLAM
We use [CLAM](https://github.com/mahmoodlab/CLAM), integrated within [TRIDENT](https://github.com/mahmoodlab/TRIDENT), for tissue segmentation and patching.
```
git clone https://github.com/mahmoodlab/trident.git && cd trident
pip install -e .
```
### 1. dependencies
```
git clone https://github.com/hvcl/LaGuadia.git && cd LaGuadia
pip install -r requirements.txt
```