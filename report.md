## 📊 Technical Report

### Design Decisions

1. **Framework Choice:** Streamlit was chosen for its simplicity and rapid prototyping capabilities, building on the existing audio detection app structure.

2. **Model Architecture:** 
   - Audio models use transfer learning from ImageNet-pretrained networks applied to mel spectrograms
   - Image models (AlexNet, DenseNet) are simplified implementations focused on functionality and Grad-CAM compatibility

3. **XAI Integration:** All three techniques (LIME, Grad-CAM, SHAP) are implemented as modular classes that work with any compatible Keras model.

### Improvements Over Original Repositories

1. **Unified Interface:** Combined two separate applications into one cohesive platform
2. **Automatic Compatibility:** Input type detection automatically filters available models and XAI techniques
3. **Comparison Feature:** Added side-by-side XAI comparison for deeper analysis
4. **Modular Architecture:** Separated models, XAI, and utilities into reusable modules
5. **Enhanced Documentation:** Comprehensive README with setup and usage instructions

### Selected Models and XAI Methods

- **Audio:** All 5 models from the original deepfake detection repository
- **Image:** AlexNet and DenseNet as specified in the lung cancer detection documentation
- **XAI:** LIME, Grad-CAM, and SHAP as required by the project guidelines

---

## ⚠️ Generative AI Usage Statement

This project was developed with assistance from **GitHub Copilot (Claude Opus 4.5)**.

### Usage Details:

| Purpose | Description |
|---------|-------------|
| Code Refactoring | Restructuring existing code into modular architecture |
| Code Generation | Implementing AlexNet/DenseNet models and XAI modules |
| Documentation | Writing README.md and inline code comments |

**Transparency Note:** All AI-assisted code was reviewed and validated for correctness and functionality.

---

## 📝 Demo Instructions

### For Live Demo:

1. **Audio Classification Demo:**
   - Upload a .wav audio file
   - Select MobileNetV2 model
   - Run LIME, Grad-CAM, and SHAP explanations
   - Show comparison tab with all three techniques

2. **Image Classification Demo:**
   - Upload a chest X-ray image (.png or .jpg)
   - Select DenseNet model
   - Run classification and XAI analysis
   - Compare different explainability methods

3. **Comparison Feature:**
   - Demonstrate automatic filtering of XAI methods
   - Show side-by-side comparison of all three techniques
   - Explain the differences in visualization approaches

---

## 📚 References

- LIME: Ribeiro et al., "Why Should I Trust You?: Explaining the Predictions of Any Classifier" (KDD 2016)
- Grad-CAM: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization" (ICCV 2017)
- SHAP: Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions" (NeurIPS 2017)
- Original Deepfake Audio Detection repository
- Original Lung Cancer Detection repository

---

## 📄 License

This project is for educational purposes as part of the GenAI course.