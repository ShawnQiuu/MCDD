# Datasets and evaluation protocol

> Status: **placeholder**. Download links and directory conventions are fixed, but the loader configuration and
> evaluation scripts will be published with the code release.

MCDD is evaluated under the **multiclass** setting: one model is trained on the normal images of every category in a
dataset, then evaluated jointly on all categories. No category-specific retraining, no anomaly images during training.

## MVTec AD

- 15 categories (5 texture, 10 object), 5354 images.
- 3629 normal images used for training; the remaining normal and anomalous images are used for testing.
- Download: <https://www.mvtec.com/company/research/datasets/mvtec-ad>
- License: CC BY-NC-SA 4.0. The dataset is **not** redistributed with this repository.

## VisA

- 12 categories, 9659 normal samples and 1226 anomalous samples.
- More complex anomaly morphology; some images contain multiple anomalous regions.
- Download: <https://github.com/amazon-science/spot-diff>
- License: CC BY 4.0. The dataset is **not** redistributed with this repository.

## Expected directory layout

```text
data/
├── mvtec/
│   ├── bottle/
│   │   ├── train/good/
│   │   ├── test/<defect_type>/
│   │   └── ground_truth/<defect_type>/
│   ├── cable/
│   └── ...                        # 15 categories
└── visa/
    ├── candle/
    │   ├── Data/Images/Normal/
    │   ├── Data/Images/Anomaly/
    │   └── Data/Masks/Anomaly/
    └── ...                        # 12 categories
```

Exact path templates will be defined in the released configuration files.

## Evaluation metrics

| Metric | Level | Definition |
| --- | --- | --- |
| AU-ROC | image | Area under the ROC curve for image-level anomaly scores |
| AP | image | Average precision for image-level anomaly scores |
| F1 max | image | Maximum F1 over all thresholds |
| AU-ROC | pixel | Area under the ROC curve for pixel-level anomaly maps |
| AP | pixel | Average precision for pixel-level anomaly maps |
| F1 max | pixel | Maximum pixel-level F1 over all thresholds |
| AU-PRO | pixel | Area under the per-region-overlap curve |
| mAD-5 | composite | Mean of image/pixel AU-ROC, image/pixel AP, and AU-PRO |
| mAD-7 | composite | mAD-5 plus image/pixel F1 max (MambaAD protocol) |

`mAD-5` follows the CND protocol; `mAD-7` follows the MambaAD protocol and is used for comparison with recent
multiclass methods.

## Reference training configuration

| Setting | Value |
| --- | --- |
| Input resolution | 256 x 256, no data augmentation |
| Encoder | First three stages of ResNet-34, frozen |
| Decoder depth | `[3, 4, 6, 3]`, mirroring ResNet-34 |
| Optimizer | AdamW, learning rate `0.005`, weight decay `1e-4` |
| Noise intensity `σ` | 0.2 |
| Hardware | Single NVIDIA Tesla V100 |
| Training data | Normal images only, all categories jointly |
