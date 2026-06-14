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
### 1. Install dependencies
```
git clone https://github.com/hvcl/LaGuadia.git && cd LaGuadia
pip install -r requirements.txt
```

## 🔥 Training
> [!NOTE]  
> For efficient training, pre-extracting teacher features before training is highly recommended.

### Stage 1. Keyword Extraction
본 연구의 학습을 위해서는 학습 이전에 병리 보고서에서 keyword 추출을 수행해야 합니다.  
키워드 추출은 [preparing/generate_keywords.py](./preparing/generate_keywords.py)를 통해 수행 할 수 있습니다.
```
python generate_keywords.py
```

### Stage 2. Align Vision-Language embeddings via Meta-Teacher
```
python train_stage1.py --config-name train-stage-1-configs
```

### Stage 3. Language-Guided Adaptive Knowledge Distillation