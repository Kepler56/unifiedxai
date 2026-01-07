# Unified Explainable AI Platform

A multi-modal classification platform with Explainable AI (XAI) capabilities for both audio deepfake detection and lung cancer detection from chest X-rays.

## 👥 Team Information

- **Group Members:** Rodolphe BIELEU, Sascha CAUCHON
- **TD Group:** DIA2

---

## 📋 Project Overview

This project integrates two existing XAI systems into a single interactive platform:

1. **Deepfake Audio Detection:** Detects real vs. fake audio using neural networks (VGG16, MobileNet, ResNet50, InceptionV3) trained on mel spectrograms from the Fake-or-Real (FoR) dataset.

2. **Lung Cancer Detection:** Detects malignant tumors in chest X-rays using AlexNet and DenseNet121 with transfer learning.

### Key Features

- 🎵 **Multi-modal Input:** Support for audio (.wav) and image (.png, .jpg) files
- 🤖 **Multiple Models:** 4 audio models + 2 image models
- 🔍 **XAI Techniques:** LIME, Grad-CAM, and SHAP implementations
- ⚡ **Automatic Filtering:** XAI methods filtered based on input type
- 📊 **Comparison View:** Side-by-side XAI technique comparison
- 🎨 **User-friendly Interface:** Clean Streamlit-based UI

---

## 🛠️ Setup and Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Installation Steps

1. **Clone the repository:**
   ```bash
   cd genai
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   cd UnifiedXAI
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```bash
   python -c "import streamlit; import tensorflow; import lime; import shap; print('All packages installed successfully!')"
   ```

---

## 🚀 Running the Application

### Start the Streamlit App

```bash
cd c:\Users\cs202910\Documents\genai\UnifiedXAI
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Using the Interface

1. **Analysis Tab:**
   - Upload an audio (.wav) or image (.png, .jpg) file
   - The system automatically detects the input type
   - Select a compatible classification model
   - Choose an XAI technique
   - Click "Run Analysis" to see results

2. **Comparison Tab:**
   - After running an analysis, switch to this tab
   - Select multiple XAI techniques to compare
   - View side-by-side explanations

3. **About Tab:**
   - Project information and documentation

---

## 📁 Project Structure

```
UnifiedXAI/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
│
├── models/               # Classification models
│   ├── __init__.py
│   ├── audio_models.py   # VGG16, MobileNet, ResNet50, InceptionV3
│   └── image_models.py   # AlexNet, DenseNet121
│
├── xai/                  # XAI implementations
│   ├── __init__.py
│   ├── lime_explainer.py # LIME implementation
│   ├── gradcam.py        # Grad-CAM implementation
│   └── shap_explainer.py # SHAP implementation
│
├── utils/                # Utility functions
│   ├── __init__.py
│   ├── audio_utils.py    # Audio processing (spectrogram conversion)
│   ├── image_utils.py    # Image preprocessing
│   └── compatibility.py  # Model/XAI compatibility registry
│
└── temp_files/           # Temporary file storage (created at runtime)
```

---

## 🤖 Available Models

### Audio Classification (Deepfake Detection)

| Model | Description | Input Size |
|-------|-------------|------------|
| VGG16 | Transfer learning from ImageNet | 224×224×3 |
| MobileNet | Lightweight, efficient (best accuracy: 91.5%) | 224×224×3 |
| ResNet50 | Deep residual network | 224×224×3 |
| InceptionV3 | Google's inception architecture | 224×224×3 |


### Image Classification (Lung Cancer Detection)

| Model | Description | Input Size |
|-------|-------------|------------|
| AlexNet | Classic CNN architecture | 224×224×3 |
| DenseNet121 | Dense connections for feature propagation | 224×224×3 |

---

## 🔍 XAI Techniques

### LIME (Local Interpretable Model-agnostic Explanations)
- Perturbs superpixels to understand local decision boundaries
- Works with any model (model-agnostic)
- Shows which regions contributed to the prediction

### Grad-CAM (Gradient-weighted Class Activation Mapping)
- Uses gradients from convolutional layers
- Creates heatmaps showing important spatial regions
- Requires access to model internals

### SHAP (SHapley Additive exPlanations)
- Based on Shapley values from game theory
- Provides consistent feature attributions
- Shows positive and negative contributions

---
