<div align="center">

# 🩻 Chest X-Ray Disease Classification

### Explainable deep learning for 5 chest conditions

**A DenseNet121 model that reads a chest X-ray and predicts bacterial pneumonia, COVID-19, tuberculosis, viral
pneumonia or a normal lung, and shows *where* it looked with Grad-CAM heatmaps.**

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-DenseNet121-EE4C2C?logo=pytorch&logoColor=white)
![Transfer learning](https://img.shields.io/badge/Transfer_learning-ImageNet-6E40C9)
![Grad-CAM](https://img.shields.io/badge/Explainability-Grad--CAM-F59E0B)
![Streamlit](https://img.shields.io/badge/Streamlit-web_app-FF4B4B?logo=streamlit&logoColor=white)
![Colab](https://img.shields.io/badge/Google_Colab-GPU_training-F9AB00?logo=googlecolab&logoColor=white)
![Accuracy](https://img.shields.io/badge/best_val_accuracy-97.8%25-16A34A)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

> ⚠️ **Disclaimer:** for **educational and research purposes only**. This is not a medical device and must not be
> used for clinical diagnosis or treatment decisions.

---

## Overview

Chest X-rays are the most common imaging test in the world, and several serious lung diseases look alike on them.
This project fine-tunes an ImageNet-pretrained **DenseNet121** to separate **five classes**, and pairs every
prediction with a **Grad-CAM heatmap** so the model's reasoning can be inspected instead of trusted blindly.

<table>
  <tr>
    <td align="center"><img src="Dataset/test/normal/Normal-10001.png" width="140"><br><sub>Normal</sub></td>
    <td align="center"><img src="Dataset/test/bacterial_pneumonia/person1016_bacteria_2947.jpeg" width="140"><br><sub>Bacterial pneumonia</sub></td>
    <td align="center"><img src="Dataset/test/viral_pneumonia/Viral%20Pneumonia-1006.png" width="140"><br><sub>Viral pneumonia</sub></td>
    <td align="center"><img src="Dataset/test/covid/COVID-1012.png" width="140"><br><sub>COVID-19</sub></td>
    <td align="center"><img src="Dataset/test/tuberculosis/Tuberculosis-102.png" width="140"><br><sub>Tuberculosis</sub></td>
  </tr>
</table>
<p align="center"><sub>One example per class from the test split</sub></p>

## Results

| Metric | Value |
|---|---|
| **Best validation accuracy** | **97.8%** (epoch 9 of 10) |
| Final training accuracy | 98.4% |
| Validation set | 500 images, 100 per class (balanced) |
| Checkpoint | best epoch by validation accuracy → `models/densenet121_chestxray.pt` |

<p align="center"><img src="outputs/training_graph.png" alt="Training and validation curves" width="720"></p>

Run `evaluate_model.py` for the full per-class classification report and confusion matrix on the held-out test split.

## How it works

```mermaid
flowchart LR
    A[Chest X-ray<br>PNG / JPG] --> B[Resize 224×224<br>ImageNet normalisation]
    B --> C[DenseNet121<br>ImageNet weights<br>fine-tuned]
    C --> D[Linear head<br>5 classes]
    D --> E[Softmax<br>class + confidence]
    C -. gradients of the<br>predicted class .-> F[Grad-CAM<br>denseblock4 last conv]
    F --> G[Heatmap overlay<br>on the X-ray]
```

| | |
|---|---|
| **Backbone** | `torchvision` DenseNet121, ImageNet-pretrained, classifier replaced by a 5-way `Linear` layer |
| **Training** | CrossEntropyLoss, Adam (`lr=1e-4`, `weight_decay=1e-5`), fixed seed, optional `--freeze_features` |
| **Augmentation** | random horizontal flip, small rotations, colour jitter |
| **Explainability** | Grad-CAM on the last convolution of `denseblock4`: gradients of the predicted class weight the feature maps into a heatmap |

## Features
- 🖥️ **Streamlit app**: upload an X-ray, get the class, the confidence and a Grad-CAM overlay
- 🧪 **CLI tools** for training, test-set evaluation and single-image prediction (top-K + heatmap)
- ☁️ **Colab notebook** for free GPU training
- 📦 **Trained model included**: predictions work right after install
- 🔁 **Reproducible** training with a fixed seed and saved training history

## Dataset

Balanced, in `torchvision.datasets.ImageFolder` layout, **3,500 images**:

| Split | Per class | Total |
|---|---|---|
| train | 500 | 2,500 |
| val | 100 | 500 |
| test | 100 | 500 |

Images were compiled from publicly available chest X-ray datasets on [Kaggle](https://www.kaggle.com/); please
credit the original dataset authors when redistributing or publishing results.

## Tech stack

| Area | Technology |
|---|---|
| Deep learning | PyTorch, torchvision (DenseNet121) |
| Explainability | Grad-CAM (custom hooks), OpenCV overlays |
| Evaluation | scikit-learn, Matplotlib, Seaborn |
| App | Streamlit, Pillow |
| Training | Local GPU / CPU or Google Colab |

## Getting started

```bash
git clone https://github.com/ChetanMahajan715/chest-xray-disease-classification.git
cd chest-xray-disease-classification
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
For GPU training, install the CUDA build of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/).

### Command-line tools

```bash
# train
python train_model.py --data_dir Dataset --epochs 10 --batch_size 16
#   options: --image_size, --seed, --model_dir, --freeze_features

# evaluate on the test split → outputs/classification_report.txt + confusion_matrix.png
python evaluate_model.py --data_dir Dataset --model_dir models --output_dir outputs

# predict one image, top 3 classes + Grad-CAM overlay saved to outputs/
python predict.py --image path/to/chest_xray.png --show_heatmap --top_k 3
```

## Project structure

```
chest-xray-disease-classification/
├── app.py                 # Streamlit app: upload → prediction → Grad-CAM
├── train_model.py         # training script (CLI)
├── train_in_colab.ipynb   # GPU training on Google Colab
├── evaluate_model.py      # test report + confusion matrix
├── predict.py             # single-image prediction (CLI)
├── gradcam.py             # Grad-CAM implementation
├── model_utils.py         # model building, loading, preprocessing
├── models/                # trained checkpoint, class names, training history
├── outputs/               # training graph, reports, heatmaps
├── Dataset/               # train / val / test, one folder per class
└── requirements.txt
```

## Possible improvements
- Report per-class recall and sensitivity on the test split in this README
- Train on a larger, multi-source dataset and test on an external hospital dataset
- Calibrated confidence and an "uncertain, refer to a radiologist" threshold

## Author

**Chetan Mahajan** · AI / ML engineer

[![GitHub](https://img.shields.io/badge/GitHub-ChetanMahajan715-181717?logo=github)](https://github.com/ChetanMahajan715)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-chetanmahajan715-0A66C2?logo=linkedin)](https://www.linkedin.com/in/chetanmahajan715/)

## License
[MIT](LICENSE)
