# 🎤 Unified XAI Platform - Presentation Script

## 📋 Presentation Overview

**Project:** Unified Explainable AI Platform  
**Team:** Rodolphe BIELEU, Sascha CAUCHON  
**TD Group:** DIA2  
**Duration:** ~15-20 minutes

---

## 🎬 PART 1: Introduction (2-3 minutes)

### Opening Statement

> *"Good morning/afternoon everyone. Today we're presenting our Unified Explainable AI Platform - a multi-modal classification system that combines deep learning with explainability techniques to make AI decisions transparent and interpretable."*

### Project Context

> *"Our project integrates two distinct AI applications into a single unified platform:*
> 
> 1. **Deepfake Audio Detection** - Identifying synthetic or manipulated audio
> 2. **Lung Cancer Detection** - Detecting malignant tumors from chest X-rays
>
> *Both tasks share a common need: not just making predictions, but explaining WHY the model made those predictions. This is where Explainable AI (XAI) becomes essential."*

### Why Explainability Matters

> *"In medical diagnosis and fraud detection, a simple 'yes/no' answer isn't enough. Doctors need to understand what regions of an X-ray indicate cancer. Security analysts need to know what audio features suggest manipulation. Our platform provides these explanations through three state-of-the-art XAI techniques."*

---

## 🖥️ PART 2: Live Demonstration (8-10 minutes)

### Setup (Before Demo)
- Open terminal in `UnifiedXAI` folder
- Run: `streamlit run app.py`
- Have sample files ready:
  - Audio file: `.wav` file for deepfake detection
  - Image file: Chest X-ray `.png` or `.jpg`

---

### Demo 1: Audio Deepfake Detection (4-5 minutes)

#### Step 1: Data Selection
> *"Let's start with audio deepfake detection. I'll upload a WAV audio file."*

**ACTION:** Upload a `.wav` file

> *"Notice how the system automatically detects this is an audio file and displays 'Deepfake Audio Detection' as the task. The platform converts audio into mel spectrograms - visual representations of sound frequencies over time - which our neural networks can analyze."*

#### Step 2: Model Selection
> *"For audio classification, we have four models available. I'll select MobileNet, which achieved 91.5% accuracy on the FoR dataset."*

**ACTION:** Select `MobileNet` from dropdown

> *"MobileNet uses transfer learning from ImageNet. It's pre-trained on millions of images and fine-tuned on our spectrogram data. This approach is common in audio classification because spectrograms are essentially images."**

#### Step 3: Prediction
> *"Let's run the analysis."*

**ACTION:** Click "Run Analysis"

> *"The model predicts [REAL/FAKE] with [XX]% confidence. But how did it reach this decision? That's where our XAI techniques come in."*

#### Step 4: Explainability - Show All Three Techniques

**LIME Explanation:**
> *"LIME, or Local Interpretable Model-agnostic Explanations, works by creating perturbations of the input and seeing how the model responds. The green regions show areas that support the prediction, while red regions show areas that oppose it."*

**Grad-CAM Explanation:**
> *"Grad-CAM uses the gradients flowing into the final convolutional layer to produce a heatmap. The warmer colors - red and yellow - indicate regions the model focused on most heavily. For spectrograms, this often highlights specific frequency bands or temporal patterns."*

**SHAP Explanation:**
> *"SHAP is based on Shapley values from game theory. It assigns each feature an importance value for a particular prediction. Red pixels increase the probability of the predicted class, while blue pixels decrease it."*

---

### Demo 2: Lung Cancer Detection (3-4 minutes)

#### Step 1: Data Selection
> *"Now let's demonstrate the image classification capability with a chest X-ray."*

**ACTION:** Upload a chest X-ray image (`.png` or `.jpg`)

> *"The system automatically switches context - it now shows 'Lung Cancer Detection' as the task, and the available models change accordingly."*

#### Step 2: Model Selection
> *"For medical imaging, we have AlexNet and DenseNet available. I'll choose DenseNet121, which uses dense connections for better feature propagation."*

**ACTION:** Select `DenseNet` from dropdown

> *"DenseNet is particularly effective for medical imaging because its dense connections help preserve fine-grained features that might indicate tumors."*

#### Step 3: Run Analysis
**ACTION:** Click "Run Analysis"

> *"The model predicts [Benign/Malignant]. In a clinical setting, the next question would be: 'What did the model see that led to this diagnosis?'"*

#### Step 4: Explainability Comparison

**ACTION:** Switch to "Comparison" tab and select multiple XAI techniques

> *"Here's where our comparison feature shines. We can see all three XAI techniques side by side. Notice how:*
> - *LIME shows discrete superpixel regions*
> - *Grad-CAM provides smooth heatmap overlays*
> - *SHAP gives pixel-level importance scores*
>
> *Each technique has its strengths, and comparing them gives us more confidence in understanding the model's reasoning."*

---

## 🔧 PART 3: Technical Choices (4-5 minutes)

### Classification Models

> *"Let me explain our technical choices."*

#### Audio Models (4 models)
| Model | Why We Chose It |
|-------|-----------------|
| **VGG16** | Deep architecture, excellent feature extraction |
| **MobileNet** | Lightweight, efficient, 91.5% accuracy on FoR dataset |
| **ResNet50** | Skip connections prevent vanishing gradients |
| **InceptionV3** | Multi-scale feature extraction |

> *"All audio models use transfer learning from ImageNet. This works because spectrograms share visual patterns similar to natural images - textures, edges, and spatial relationships."*

#### Image Models (2 models)
| Model | Why We Chose It |
|-------|-----------------|
| **AlexNet** | Classic architecture, well-understood behavior |
| **DenseNet121** | Dense connections, better gradient flow, preserves features |

> *"For medical imaging, we prioritized models that preserve fine-grained features important for tumor detection."*

### XAI Methods

#### 1. LIME (Local Interpretable Model-agnostic Explanations)
> *"LIME is model-agnostic - it works with any classifier. It creates thousands of perturbed samples around the input, observes how predictions change, and fits an interpretable linear model locally."*

**Technical Implementation:**
```python
# From lime_explainer.py
self.explainer = lime_image.LimeImageExplainer()
explanation = self.explainer.explain_instance(
    image, predict_fn, num_samples=1000, num_features=10
)
```

#### 2. Grad-CAM (Gradient-weighted Class Activation Mapping)
> *"Grad-CAM uses gradients from the classification score to the final convolutional layer. These gradients are global-average-pooled to obtain neuron importance weights."*

**Technical Implementation:**
```python
# From gradcam.py
with tf.GradientTape() as tape:
    conv_outputs, predictions = self.grad_model(img_array)
    loss = predictions[:, class_idx]
grads = tape.gradient(loss, conv_outputs)
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
```

#### 3. SHAP (SHapley Additive exPlanations)
> *"SHAP uses Shapley values from cooperative game theory. It fairly distributes the 'credit' for the prediction among all input features. We use superpixel segmentation to reduce computational complexity."*

**Technical Implementation:**
```python
# From shap_explainer.py
segments = slic(image, n_segments=50, compactness=10)
explainer = shap.KernelExplainer(predict_fn, background)
shap_values = explainer.shap_values(masks, nsamples=100)
```

### Architecture Design

> *"Our platform uses a modular architecture:"*

```
UnifiedXAI/
├── app.py              # Streamlit main application
├── models/             # Classification models
│   ├── audio_models.py # VGG16, MobileNet, ResNet, Inception, CustomCNN
│   └── image_models.py # AlexNet, DenseNet
├── xai/                # Explainability techniques
│   ├── lime_explainer.py
│   ├── gradcam.py
│   └── shap_explainer.py
└── utils/              # Utilities
    ├── audio_utils.py  # Audio to spectrogram conversion
    ├── image_utils.py  # Image preprocessing
    └── compatibility.py # Model/XAI compatibility registry
```

> *"This separation allows easy extension - adding a new model or XAI technique requires minimal changes."*

---

## ⚠️ PART 4: Difficulties & Solutions (2-3 minutes)

### Challenge 1: Model Compatibility with Grad-CAM

> **Problem:** *"Grad-CAM requires access to the last convolutional layer, but different model architectures name their layers differently."*

> **Solution:** *"We implemented automatic layer detection and a manual registry mapping model names to their convolutional layer names."*

```python
def get_conv_layer_name(model_name):
    layer_names = {
        'VGG16': 'block5_conv3',
        'MobileNet': 'conv_pw_13_relu',
        'DenseNet': 'conv5_block16_concat',
        # ...
    }
```

### Challenge 2: Multi-Modal Input Handling

> **Problem:** *"Audio and images require completely different preprocessing pipelines, but we wanted a unified interface."*

> **Solution:** *"We implemented automatic file type detection and a compatibility registry that filters available models and XAI techniques based on input type."*

```python
def detect_file_type(uploaded_file):
    audio_extensions = {'.wav', '.mp3', '.flac'}
    image_extensions = {'.jpg', '.jpeg', '.png'}
    # Automatic detection...
```

### Challenge 3: SHAP Computational Complexity

> **Problem:** *"Pixel-level SHAP is computationally expensive for 224×224 images (over 150,000 features)."*

> **Solution:** *"We use superpixel segmentation (SLIC algorithm) to reduce the input space from ~150,000 pixels to ~50 superpixels, making computation tractable."*

### Challenge 4: Integrating Two Separate Codebases

> **Problem:** *"The original Deepfake Audio Detection and Lung Cancer Detection projects had different code structures and dependencies."*

> **Solution:** *"We created a unified architecture with shared utilities, standardized model interfaces, and a compatibility layer that manages different input/output formats."*

### Challenge 5: Pre-trained Model Availability

> **Problem:** *"For the audio deepfake detection, we had access to a pre-trained MobileNet model from the original repository. However, for lung cancer detection, no pre-trained weights were available. Training from scratch requires significant time and a proper medical imaging dataset."**

> **Solution:** *"We focused our live demo on the audio classification task where we have a trained model with 91.5% accuracy. For the image models, we demonstrate the XAI techniques with untrained models - this still effectively shows HOW the model makes decisions and what regions it focuses on. The XAI visualizations are valid regardless of model accuracy, as they reveal the decision-making process itself. In a production environment, these models would be trained on annotated chest X-ray datasets like those available on Kaggle."*

> **Key Insight:** *"This challenge actually highlights an important aspect of XAI - explainability techniques help us understand model behavior even during development and debugging, not just for final deployed models."*

### Challenge 6: Keras 3 and SavedModel Format Incompatibility

> **Problem:** *"The pre-trained MobileNet model was saved in TensorFlow's legacy SavedModel format. Keras 3 no longer supports loading this format with `load_model()`, causing compatibility errors. Additionally, SavedModel format doesn't expose internal layers, making Grad-CAM impossible since it requires access to convolutional layer activations and gradients."**

> **Solution:** *"We implemented a custom `SavedModelWrapper` class that uses TensorFlow's native `tf.saved_model.load()` function. This wrapper mimics Keras model behavior with a compatible `predict()` method. For Grad-CAM, we recommend using other models (VGG16, ResNet50, InceptionV3) that are built with the Functional API and expose all internal layers. LIME and SHAP work perfectly with the saved model since they only need the prediction function."*

```python
# Our SavedModelWrapper solution
loaded = tf.saved_model.load(model_path)
class SavedModelWrapper:
    def __init__(self, saved_model):
        self.infer = saved_model.signatures['serving_default']
    
    def predict(self, x, verbose=0):
        result = self.infer(tf.cast(x, tf.float32))
        return result[list(result.keys())[0]].numpy()
```

> **Demo Strategy:** *"For MobileNet with the trained model, we use LIME or SHAP. For Grad-CAM demonstrations, we use VGG16 or other models that expose their layer structure."**

---

## 🎯 PART 5: Key Takeaways & Conclusion (1-2 minutes)

### What We Achieved

> *"To summarize, our Unified XAI Platform delivers:"*
> 
> 1. ✅ **Multi-modal support** - Audio and image classification in one platform
> 2. ✅ **Multiple models** - 6 different neural network architectures
> 3. ✅ **Three XAI techniques** - LIME, Grad-CAM, and SHAP
> 4. ✅ **Automatic filtering** - Smart compatibility between inputs and models
> 5. ✅ **Comparison view** - Side-by-side XAI visualization
> 6. ✅ **User-friendly interface** - Clean, intuitive Streamlit UI

### Future Improvements

> *"Potential future enhancements include:"*
> - Real-time audio streaming analysis
> - Support for additional medical imaging tasks
> - Model fine-tuning on custom datasets
> - Export reports for clinical use

### Closing Statement

> *"Thank you for your attention. Our platform demonstrates that explainable AI is not just a theoretical concept - it's a practical tool that can help build trust between humans and AI systems. We're happy to answer any questions."*

---

## ❓ PART 6: Anticipated Q&A

### Q1: Why did you choose Streamlit over other frameworks?
> *"Streamlit allows rapid prototyping with minimal code. The original audio detection project already used Streamlit, so we maintained consistency while extending functionality."*

### Q2: Why use spectrograms for audio instead of raw waveforms?
> *"CNNs are excellent at image recognition, and spectrograms convert audio into a visual format. This lets us leverage transfer learning from ImageNet-pretrained models."*

### Q3: Which XAI technique is best?
> *"It depends on the use case:*
> - *LIME is best when you need a model-agnostic approach*
> - *Grad-CAM is fast and shows spatial attention directly*
> - *SHAP provides theoretically grounded feature importance*
>
> *We recommend using multiple techniques and comparing results for higher confidence."*

### Q4: How accurate are your models?
> *"For demonstration purposes, we use pre-trained weights without extensive fine-tuning. In production, models would need training on domain-specific datasets with proper validation."*

### Q5: Can this platform be extended to other tasks?
> *"Absolutely! The modular architecture makes it easy to add new:*
> - *Input types (video, text)*
> - *Classification models*
> - *XAI techniques (Attention maps, Integrated Gradients)"*

### Q6: How did you handle AI assistance in your project?
> *"We used GitHub Copilot for code refactoring, generating boilerplate, and documentation. All AI-generated code was reviewed and validated for correctness. We've documented this transparently in our report."*

---

## 📝 Quick Reference - Demo Commands

```powershell
# Navigate to project
cd c:\Users\cs202910\Documents\xai\UnifiedXAI

# Activate virtual environment (if applicable)
# venv\Scripts\activate

# Run the application
streamlit run app.py
```

**Test Files Checklist:**
- [ ] Audio file (.wav) for deepfake detection
- [ ] Chest X-ray image (.png or .jpg) for lung cancer detection

---

## ⏱️ Time Management

| Section | Duration |
|---------|----------|
| Introduction | 2-3 min |
| Live Demo - Audio | 4-5 min |
| Live Demo - Image | 3-4 min |
| Technical Choices | 4-5 min |
| Difficulties & Solutions | 2-3 min |
| Conclusion | 1-2 min |
| **Total** | **~17-22 min** |

---

*Good luck with your presentation! 🍀*
