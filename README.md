# Breast Cancer Detection from Mammography Images (CBIS-DDSM)

A deep learning project for classifying breast mammogram images as **benign** or **malignant**, using the [CBIS-DDSM dataset](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset) from Kaggle.

## 📋 Overview

This notebook walks through a full pipeline:

1. **Data acquisition** — downloading the CBIS-DDSM dataset via the Kaggle API
2. **Preprocessing** — fixing image paths, merging mass/calcification case metadata
3. **Data balancing & augmentation** — equalizing benign/malignant class counts and applying flips, brightness/contrast/saturation jitter
4. **Modeling** — comparing multiple architectures:
   - A simple ANN baseline
   - Transfer learning with `EfficientNetV2L`
   - Transfer learning with `ConvNeXtBase` (final/best model)
5. **Evaluation** — 10-fold cross-validation, confusion matrices, accuracy/precision/recall/loss curves
6. **Model persistence** — saving/loading the trained model via Google Drive

## 🗂️ Dataset

- **Source:** [CBIS-DDSM Breast Cancer Image Dataset](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset) (Kaggle)
- **Classes:** `BENIGN` / `BENIGN_WITHOUT_CALLBACK` → 0, `MALIGNANT` → 1
- **Image size:** 224×224×3

> ⚠️ The dataset is **not included** in this repository due to its size. Follow the setup instructions below to download it yourself.

## 🧠 Models

| Model | Notes |
|---|---|
| ANN | Simple fully-connected baseline (10 layers), used as a reference point |
| EfficientNetV2L | Pretrained on ImageNet, top layers frozen/fine-tuned |
| ConvNeXtBase | Final model, best performance |

## ⚙️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get Kaggle API credentials
- Go to your [Kaggle account settings](https://www.kaggle.com/settings) → **Create New API Token**
- This downloads a `kaggle.json` file
- Place it at `~/.kaggle/kaggle.json` (or upload it when the notebook prompts you, if running on Colab)

### 4. Run the notebook
Open `notebook.ipynb` in Google Colab or Jupyter and run the cells in order.

## 📊 Results

Evaluation is performed using:
- 10-fold cross-validation (random-sampled test subsets per fold)
- Confusion matrices
- Accuracy / Precision / Recall / F1 / ROC-AUC / Loss bar charts

See the notebook output cells for exact numbers and plots.

## 📁 Repository Structure

```
.
├── notebook.ipynb        # Main notebook (data prep, training, evaluation)
├── requirements.txt       # Python dependencies
├── README.md
└── .gitignore
```

## 📝 License

This project is licensed under the [MIT License](LICENSE).

## 🙏 Acknowledgements

- CBIS-DDSM dataset by [awsaf49 on Kaggle](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset)
- Pretrained models from `tensorflow.keras.applications`
