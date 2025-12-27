Potato Leaf Disease Detection

A Tkinter-based AI app that detects diseases in potato leaves using Deep Learning (EfficientNet-B0). Train your own model or use a pre-trained one to classify leaf images.

Features

Load a dataset of potato leaf images (folders = classes).

Train a deep learning model with data augmentation.

Detect leaf diseases with confidence scores.

Save and load trained models.

Simple GUI built with Tkinter.

Installation

Clone the repo:

git clone https://github.com/yourusername/potato-leaf-disease-ai.git
cd potato-leaf-disease-ai


Install dependencies:

pip install torch torchvision pillow numpy

Usage

Run the app:

python main.py


Use the GUI:

Load Dataset → select dataset folder.

Train Model → train EfficientNet-B0.

Detect Image → select a leaf image for prediction.

Results show disease name and confidence.

Dataset Structure
dataset/
├─ healthy/
├─ early_blight/
├─ late_blight/
├─ leaf_mold/
├─ bacterial_spot/

Folder names become class labels.

Notes

Data augmentation includes crop, flip, rotation, color jitter, normalization.

Confidence threshold can be adjusted in the code (self.CONFIDENCE_THRESHOLD).# deep-learning
