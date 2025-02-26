# Automated Upper Endoscopy Auditing of Procedure Times: Temporal Multiclass Analysis
<img src="figures/Pipeline.jpg" alt="Pipeline" width="700">

Welcome to the official repository for our MICCAI 2025 paper, currently under double-blind peer review. Here, you'll find scripts, datasets, and models essential for our research. 🚀

📊 Data
Summary: 
🔗 **Dataset:** [Figshare](https://doi.org/10.6084/m9.figshare.27308133)
🔗 **Code:** [GitHub](https://github.com/)

This section provides an overview of the datasets used in our study 📌.
- 📼 Videoendoscopies for Organ Classification: 237 MP4 videos from 233 patients (∼1.6 million frames).
- 📊 Stomach Site Sequences: 4,729 short sequences for analysis.

📂 For more details: Check out the [data.md](data.md) file for a comprehensive guide on data organization and preprocessing steps.

## 🎯 Multi-Frame Embedding

Embedding Methods:
- 1️⃣ ViT’s Patch-Based Linear Projection (16×16×3)
- 2️⃣ ConvNeXt-Tiny Pretrained on ImageNet
- 3️⃣ ConvNeXt-Tiny Pretrained on Endoscopy

📂 Learn More: Check the [features.md](features.md) file for detailed embedding representations of each videoendoscopy and sequence feature extraction.

## 🏷️ Organ Classification

Summary of Experiments

🔍 Spatial-Based Classification
- 1️⃣ ViT’s Patch-Based Linear Projection + MLP
- 2️⃣ ConvNeXt-Tiny Pretrained on ImageNet + MLP
- 3️⃣ ConvNeXt-Tiny Pretrained on Endoscopy + MLP

    | Embedding            | Resolution | Precision | Recall | F1    | MCC   |Download                                                                     |
    |:------------------:  |:----------:|:---------:|:------:|:-----:|:-----:|:---------------------------------------------------------------------------:|
    | Linear Projection    | 1 frame    | 49.74     | 72.21  | 54.48 | 48.86 | [Download](https://drive.google.com/uc?id=)|
    | ConvNeXt (ImageNet)  | 1 frame    | 62.78     | 85.15  | 70.37 | 68.13 | [Download](https://drive.google.com/uc?id=)|
    | ConvNeXt (Endoscopy) | 1 frame    | 64.55     | 87.06  | 71.68 | 70.38 | [Download](https://drive.google.com/uc?id=)|

🔄 Multi-Frame-Based Classification

Summary of Experiments

⏳ Temporal-Based Classification with Attention Mechanisms

- 1️⃣ One Attention Layer initialized with Random Weights
- 2️⃣ ViT-Base initialized with Random Weights
- 3️⃣ ViT-Base initialized with ImageNet Pretraining

📂 For more details, refer to the [organclassification.md](organclassification.md) file.

- 1️⃣ One Attention Layer initialized with Random Weights

    | time   | Precision| Recall    | F1     | MCC    |Download                                                                     |
    |--------|----------|-----------|--------|--------|--------|
    | 1.0 sec| 74.57    | 85.85     | 79.02  | 76.67  | [Download](https://drive.google.com/uc?id=)|
    | 3.0 sec| 82.90    | 88.90     | 85.54  | 84.02  | [Download](https://drive.google.com/uc?id=)|
    | 5.0 sec| 83.94    | 88.39     | 85.91  | 84.29  | [Download](https://drive.google.com/uc?id=)|
    | 9.0 sec| 85.80    | 86.17     | 85.16  | 84.26  | [Download](https://drive.google.com/uc?id=)|
    | 13.1sec| 86.56    | 84.75     | 84.54  | 83.95  | [Download](https://drive.google.com/uc?id=)|

- 2️⃣ ViT-Base initialized with Random Weights

    | time   | Precision| Recall    | F1     | MCC    |Download                                                                     |
    |--------|----------|-----------|--------|--------|--------|
    | 1.0 sec| 70.60    | 86.10     | 76.85  | 74.64  | [Download](https://drive.google.com/uc?id=)|
    | 3.0 sec| 78.54    | 89.47     | 83.19  | 80.80  | [Download](https://drive.google.com/uc?id=)|
    | 5.0 sec| 80.47    | 89.86     | 84.53  | 82.56  | [Download](https://drive.google.com/uc?id=)|
    | 9.0 sec| 77.56    | 87.52     | 80.98  | 78.96  | [Download](https://drive.google.com/uc?id=)|
    | 13.1sec| 77.31    | **90.33** | 82.72  | 80.11  | [Download](https://drive.google.com/uc?id=)|

- 3️⃣ ViT-Base initialized with ImageNet Pretraining

    | time   | Precision| Recall    | F1       | MCC      |Download                                                                     |
    |--------|----------|-----------|----------|----------|--------|
    | 1.0 sec| 82.24    | 88.10     | 84.96    | 83.08    | [Download](https://drive.google.com/uc?id=)|
    | 3.0 sec| 89.74    | 89.14     | 89.14    | 87.85    | [Download](https://drive.google.com/uc?id=)|
    | 5.0 sec| 91.03    | 90.29     | 90.29    | 89.62    | [Download](https://drive.google.com/uc?id=)|
    | 9.0 sec| **92.03**| 90.42     | **90.42**| **89.94**| [Download](https://drive.google.com/uc?id=)|
    | 13.1sec| 89.87    | 88.64     | 88.64    | 88.19    | [Download](https://drive.google.com/uc?id=)|   

## 🏥 Stomach Sites Classification

**Summary of Experiments**
- 🔬 **Selected Embedding:** ConvNeXt-Tiny Pretrained on Endoscopy
- ⏳ **Temporal-Based Evaluation** using different time intervals:
  - 1️⃣ **ViT-Base initialized with Organ Pretraining – 3.0 sec**
  - 2️⃣ **ViT-Base initialized with Organ Pretraining – 9.0 sec**
  - 3️⃣ **ViT-Base initialized with Organ Pretraining – 13.1 sec**
📂 For a detailed breakdown, refer to the [stomachsiteclassification.md](stomachsiteclassification.md) file.

- 1️⃣ **ViT-Base initialized with Organ Pretraining – 3.0 sec**
    | time   | Precision| Recall    | F1     | MCC    |Download                                                                     |
    |--------|----------|-----------|--------|--------|--------|
    | 1.0 sec| 83.38    | 82.66     | 81.62  | 82.45  | [Download](https://drive.google.com/uc?id=)|
    | 2.0 sec| 85.80    | 84.99     | 94.39  | 85.41  | [Download](https://drive.google.com/uc?id=)|
    | 3.0 sec| 83.87    | 83.64     | 82.38  | 83.22  | [Download](https://drive.google.com/uc?id=)|
    | 5.0 sec| 86.02    | 86.04     | 84.96  | 86.04  | [Download](https://drive.google.com/uc?id=)|
    | 6.0 sec| 86.63    | 86.18     | 85.47  | 86.26  | [Download](https://drive.google.com/uc?id=)|
    | 7.0 sec| 87.66    | 87.30     | 86.45  | 87.38  | [Download](https://drive.google.com/uc?id=)|
    
- 2️⃣ **ViT-Base initialized with Organ Pretraining – 9.0 sec**
    | time   | Precision| Recall    | F1     | MCC    |Download                                                                     |
    |--------|----------|-----------|--------|--------|--------|
    | 1.0 sec| 84.20    | 83.43     | 82.71  | 83.46  | [Download](https://drive.google.com/uc?id=)|
    | 2.0 sec| 85.95    | 85.94     | 85.02  | 85.98  | [Download](https://drive.google.com/uc?id=)|
    | 3.0 sec| 85.08    | 84.01     | 83.02  | 83.94  | [Download](https://drive.google.com/uc?id=)|
    | 5.0 sec| 87.48    | 87.18     | 86.26  | 87.44  | [Download](https://drive.google.com/uc?id=)|
    | 6.0 sec| 87.03    | 86.27     | 85.47  | 86.21  | [Download](https://drive.google.com/uc?id=)|
    | 7.0 sec| 84.90    | 84.91     | 83.39  | 84.71  | [Download](https://drive.google.com/uc?id=)|

- 3️⃣ **ViT-Base initialized with Organ Pretraining – 13.1 sec**
    | time   | Precision| Recall    | F1     | MCC    |Download                                                                     |
    |--------|----------|-----------|--------|--------|--------|
    | 1.0 sec| 83.21    | 81.87     | 80.97  | 82.36  | [Download](https://drive.google.com/uc?id=)|
    | 2.0 sec| 86.08    | 85.49     | 84.67  | 85.84  | [Download](https://drive.google.com/uc?id=)|
    | 3.0 sec| 86.14    | 85.21     | 84.56  | 85.26  | [Download](https://drive.google.com/uc?id=)|
    | 5.0 sec| 85.61    | 84.64     | 83.65  | 84.69  | [Download](https://drive.google.com/uc?id=)|
    | 6.0 sec| 87.50    | 87.22     | 86.30  | 87.12  | [Download](https://drive.google.com/uc?id=)|
    | 7.0 sec| **88.37**| **87.82** | **87.03**| 87.79  | [Download](https://drive.google.com/uc?id=)|

## 📊 Report Quality Indicators

1️⃣ **Indicator 1: Organ-Specific Exploration Time 📖**

🩺 **Protocol Reference:**
📖 Bisschops, Raf, et al. "Performance measures for upper gastrointestinal endoscopy: a European Society of Gastrointestinal Endoscopy (ESGE) quality improvement initiative." Endoscopy 48.09 (2016): 843-864.

This metric evaluates the duration of exploration for each organ during the endoscopic procedure, ensuring adherence to standardized protocols.

| Patients | Procedure | Pharynx | Esophagus | Stomach | Duodenum |
|:---------:|:-----------:|:--------:|:----------:|:--------:|:---------:|
| 15 | 9:22±4:17 | 0:13±0:17 | 0:54±0:38 | 7:17±2:54 | 0:56±1:19 |

2) **Indicator 2:** Stomach Sites Duration (Protocol SSS: 📖). L: lesser curvature, A: anterior
wall, G: greater curvature, P: posterior wall, and SSS: systematic screening protocol
for the stomach.

📖 Yao, Kenshi. "The endoscopic diagnosis of early gastric cancer." Annals of Gastroenterology: Quarterly Publication of the Hellenic Society of Gastroenterology 26.1 (2013): 11.

| Region | Site | Time | Region | Site | Time |
|:---------:|:----:|:--------:|:---------:|:----:|:--------:|
| Antrum Antegrade | A1 | 0:17±0:08 | Lower Body Antegrade | A2 | 0:13±0:08 |
| | L1 | 0:31±0:28 | | L2 | 0:09±0:08 |
| | P1 | 0:21±0:15 | | P2 | 0:13±0:10 |
| | G1 | 0:36±0:18 | | G2 | 0:32±0:28 |
| Middle Body Antegrade | A3 | 0:11±0:11 | Fundus Cardia Reflex | A4 | 0:07±0:04 |
| | L3 | 0:09±0:08 | | L4 | 0:08±0:07 |
| | P3 | 0:13±0:09 | | P4 | 0:06±0:05 |
| | G3 | 0:26±0:20 | | G4 | 0:10±0:06 |
| Middle Body Reflex | A5 | 0:07±0:06 | Incisura Reflex | A6 | 0:09±0:10 |
| | L5 | 0:14±0:10 | | L6 | 0:12±0:09 |
| | P5 | 0:05±0:03 | | P6 | 0:12±0:14 |


<img src="figures/Quality_Indicator.jpg" alt="QI" width="700">

## 🔨 Installation
Please refer to the [libraries.md](libraries.md) file for detailed installation instructions.



## 📓 Notebooks
`resultsipynb`:  Use this notebook to run image and sequence classification tasks for inference.

Note 🗈:  To run this code in Google Colab, click the logo:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/)



