# OSCC Detection: Oral Cancer Histopathology Classification

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white" alt="TensorFlow" />
  <img src="https://img.shields.io/badge/VGG16-Transfer_Learning-4A148C?style=flat-square" alt="VGG16" />
  <img src="https://img.shields.io/badge/Patent-Provisional_Filed-2E7D32?style=flat-square" alt="Provisional patent filed" />
</p>

Training and evaluation pipeline for detecting **oral squamous cell carcinoma (OSCC)** in
histopathology slide images, using VGG16 transfer learning. This repository reimplements the
methodology behind my study on OSCC detection (provisional patent filed). The original study
reached **96% test accuracy**; this pipeline reproduces the method end to end so the metrics
can be regenerated on the dataset.

## Approach

1. **Backbone**: VGG16 pretrained on ImageNet, top removed.
2. **Head**: global average pooling, dropout, a single sigmoid unit (binary: OSCC vs normal).
3. **Phase 1**: freeze the backbone, train the head only.
4. **Phase 2**: unfreeze the last convolutional block and fine-tune at a low learning rate.
5. **Imbalance**: class weights computed from the training split.
6. **Evaluation**: accuracy, per-class precision/recall/F1, confusion matrix and ROC-AUC on a
   held-out test split that never influences training.

## Dataset

Public OSCC histopathology image sets exist, such as the oral biopsy image database by
Rahman et al. (100x and 400x magnification, Mendeley Data). Arrange images as:

```
data/
  train/
    normal/   *.jpg
    oscc/     *.jpg
  val/
    normal/   oscc/
  test/
    normal/   oscc/
```

Images and trained models are not committed; `data/` and `models/` are gitignored.

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python src/train.py      # phase 1 + phase 2, saves models/best.keras
python src/evaluate.py   # metrics on data/test
```

## Repository layout

```
src/data.py       dataset loaders and augmentation
src/train.py      two-phase transfer learning with class weights
src/evaluate.py   held-out test metrics: report, confusion matrix, ROC-AUC
```
