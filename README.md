# ▋▉▓ VISION QUEST ▓▉▋


![Banner](https://img.shields.io/badge/AI-Vision%20Quest-blueviolet?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10-yellow?style=for-the-badge)
![PyTorch](https://img.shields.io/badge/PyTorch-Model-orange?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-green?style=for-the-badge)

------------------------------------------------------------------------

A sharp‑eyed drone detection model built for **Avishkar 2025 (MNNIT)**.\
This project identifies whether a drone appears in an image using a
fine‑tuned **ResNet‑18** network.

------------------------------------------------------------------------

## 📂 Project Structure

    VISION QUEST /
    │
    ├── dataset/
    │   ├── train/
    │   └── val/
    │
    ├── model_train.py
    ├── predict/
    │   ├── acc.py
    │   ├── predict.py
    │   └── drone_classifier.pth
    └── README.md

------------------------------------------------------------------------

## 🚀 Features

-   Drone vs No‑Drone classification\
-   Transfer learning with ResNet‑18\
-   Validation script with accuracy output\
-   Single‑image prediction script\
-   Clean modular code

------------------------------------------------------------------------

## 🧠 Training Overview

ResNet‑18 pretrained weights are used, and the final FC layer is
replaced:

``` python
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
```

Training uses: - Adam Optimizer\
- CrossEntropy Loss\
- Augmentations\
- 5 Epochs

Model saved as:

    drone_classifier.pth

------------------------------------------------------------------------

## 🎯 Prediction

Run:

``` bash
python predict.py
```

Output:

    Prediction: drone (92.14% confidence)

------------------------------------------------------------------------

## 🛠️ Installation

    pip install torch torchvision pillow

------------------------------------------------------------------------

## 🏆 Event

Built for **Vision Quest**, under **Avishkar 2025 (MNNIT Allahabad)**.

------------------------------------------------------------------------

## 🤝 Contributions

PRs welcome.

------------------------------------------------------------------------

## 📜 License

MIT License
