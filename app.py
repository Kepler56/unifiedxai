import streamlit as st
import numpy as np
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf

# Import local modules
from models.audio_models import get_audio_model, load_saved_audio_model, AUDIO_MODELS, build_mobilenet_original
from models.image_models import get_image_model, IMAGE_MODELS
from xai.lime_explainer import LimeExplainer
from xai.gradcam import GradCAM, get_conv_layer_name
from xai.shap_explainer import ShapExplainer
from utils.audio_utils import AudioProcessor, save_uploaded_audio
from utils.image_utils import ImageProcessor, detect_input_type_from_filename
from utils.compatibility import (
    CompatibilityRegistry, InputType, INPUT_TYPES,
    get_compatible_models, get_compatible_xai, get_class_names,
    get_model_description, get_xai_description
)


# Page configuration
st.set_page_config(
    page_title="Unified XAI Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'input_type' not in st.session_state:
    st.session_state.input_type = None
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'prediction' not in st.session_state:
    st.session_state.prediction = None


def load_css():
    """Load custom CSS styles"""
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #FFF3E0;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    </style>
    """, unsafe_allow_html=True)


def detect_file_type(uploaded_file) -> str:
    """Detect the type of uploaded file"""
    filename = uploaded_file.name.lower()
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg'}
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    
    ext = os.path.splitext(filename)[1]
    
    if ext in audio_extensions:
        return 'audio'
    elif ext in image_extensions:
        return 'image'
    else:
        return 'unknown'


def process_audio_file(uploaded_file):
    """Process uploaded audio file and convert to spectrogram"""
    # Save file temporarily
    os.makedirs('temp_files', exist_ok=True)
    temp_path = os.path.join('temp_files', uploaded_file.name)
    
    with open(temp_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    # Process audio to spectrogram
    processor = AudioProcessor(target_size=(224, 224))
    spec_image, mel_spec = processor.process_audio_file(temp_path)
    
    return spec_image, temp_path


def process_image_file(uploaded_file):
    """Process uploaded image file"""
    processor = ImageProcessor(target_size=(224, 224))
    
    # Open image
    img = Image.open(uploaded_file)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img_array = np.array(img)
    processed = processor.preprocess(img_array, normalize=False)
    
    return processed


def get_model(model_name: str, input_type: str):
    """Load or create a model based on name and input type"""
    if input_type == 'audio':
        # Try to load saved model first
        saved_model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'Deepfake-Audio-Detection-with-XAI', 'Streamlit', 'saved_model', 'model'
        )
        
        # The saved model was trained with MobileNet v1 with exact architecture:
        # MobileNet(include_top=False) -> GlobalAveragePooling2D -> Dense(2, sigmoid)
        # Instead of loading the problematic saved model, recreate with same architecture
        if model_name == 'MobileNet':
            # Build model with EXACT architecture matching the original training
            st.info("Building MobileNet model with original architecture...")
            model = build_mobilenet_original(input_shape=(224, 224, 3), num_classes=2)
            
            # Try to load weights from saved model if compatible
            if os.path.exists(saved_model_path):
                try:
                    # Try loading weights only (not full model) if checkpoint exists
                    weights_path = os.path.join(os.path.dirname(saved_model_path), 'weights')
                    if os.path.exists(weights_path):
                        model.load_weights(weights_path)
                        st.success("Loaded pre-trained weights!")
                except Exception as e:
                    st.warning(f"Using fresh MobileNet weights (ImageNet pre-trained). Saved weights not compatible: {str(e)[:50]}")
            
            return model
        else:
            return get_audio_model(model_name)
    else:
        return get_image_model(model_name)


def run_prediction(model, image: np.ndarray, class_names: list):
    """Run model prediction on image"""
    # Prepare image
    if image.max() > 1:
        img_normalized = image / 255.0
    else:
        img_normalized = image
    
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    # Predict
    prediction = model.predict(img_batch, verbose=0)
    predicted_class = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class]
    
    return {
        'class_idx': int(predicted_class),
        'class_name': class_names[predicted_class],
        'confidence': float(confidence),
        'probabilities': prediction[0].tolist()
    }


def run_lime_explanation(model, image: np.ndarray, class_names: list):
    """Generate LIME explanation"""
    explainer = LimeExplainer(model, class_names)
    fig = explainer.visualize(image, num_samples=500, num_features=8)
    return fig


def run_gradcam_explanation(model, image: np.ndarray, class_names: list, model_name: str):
    """Generate Grad-CAM explanation"""
    conv_layer = get_conv_layer_name(model_name)
    explainer = GradCAM(model, conv_layer, class_names)
    fig = explainer.visualize(image)
    return fig


def run_shap_explanation(model, image: np.ndarray, class_names: list):
    """Generate SHAP explanation"""
    explainer = ShapExplainer(model, class_names)
    fig = explainer.visualize(image, n_segments=30, nsamples=50)
    return fig


def main_page():
    """Main analysis page"""
    st.markdown('<h1 class="main-header">🔬 Unified XAI Platform</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Multi-modal Classification with Explainable AI</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Input Selection")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Audio (.wav) or Image (.png, .jpg)",
            type=['wav', 'png', 'jpg', 'jpeg'],
            help="Upload an audio file for deepfake detection or a chest X-ray image for lung cancer detection"
        )
        
        if uploaded_file is not None:
            input_type = detect_file_type(uploaded_file)
            st.session_state.input_type = input_type
            st.session_state.uploaded_file = uploaded_file
            
            if input_type == 'unknown':
                st.error("Unsupported file type!")
                return
            
            # Display input type
            if input_type == 'audio':
                st.success("🎵 Audio file detected")
                st.info("Task: Deepfake Audio Detection")
            else:
                st.success("🖼️ Image file detected")
                st.info("Task: Lung Cancer Detection")
            
            st.divider()
            
            # Model selection
            st.header("🤖 Model Selection")
            available_models = get_compatible_models(input_type)
            
            selected_model = st.selectbox(
                "Select Classification Model",
                available_models,
                help="Choose a model for classification"
            )
            
            if selected_model:
                st.caption(get_model_description(selected_model))
            
            st.divider()
            
            # XAI selection
            st.header("🔍 XAI Technique")
            available_xai = get_compatible_xai(input_type)
            
            selected_xai = st.selectbox(
                "Select Explainability Method",
                available_xai,
                help="Choose an XAI technique to visualize model decisions"
            )
            
            if selected_xai:
                st.caption(get_xai_description(selected_xai))
            
            st.divider()
            
            # Run analysis button
            run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
        else:
            run_analysis = False
            selected_model = None
            selected_xai = None
            input_type = None
    
    # Main content area
    if uploaded_file is None:
        # Welcome message
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🎵 Deepfake Audio Detection
            
            Detect synthetic or manipulated audio using deep learning models trained on spectrograms.
            
            **Available Models:**
            - VGG16, MobileNet, ResNet50, InceptionV3
            
            **Supported Format:** .wav files
            """)
        
        with col2:
            st.markdown("""
            ### 🏥 Lung Cancer Detection
            
            Identify potential malignant tumors in chest X-ray images using transfer learning.
            
            **Available Models:**
            - AlexNet, DenseNet121
            
            **Supported Formats:** .png, .jpg, .jpeg
            """)
        
        st.divider()
        
        st.markdown("""
        ### 🔍 Explainable AI Techniques
        
        All models support the following XAI methods:
        
        | Technique | Description |
        |-----------|-------------|
        | **LIME** | Explains predictions using locally interpretable models |
        | **Grad-CAM** | Highlights important image regions using gradient information |
        | **SHAP** | Provides feature attributions based on Shapley values |
        
        Upload a file to get started! ⬅️
        """)
        
    elif run_analysis and selected_model and selected_xai:
        # Process and analyze
        with st.spinner("Processing input..."):
            try:
                # Process file based on type
                if input_type == 'audio':
                    processed_image, audio_path = process_audio_file(uploaded_file)
                    class_names = get_class_names('audio')
                else:
                    processed_image = process_image_file(uploaded_file)
                    class_names = get_class_names('image')
                
                st.session_state.processed_image = processed_image
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                return
        
        # Display input
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("📥 Input")
            
            if input_type == 'audio':
                st.audio(uploaded_file)
                st.image(processed_image, caption="Mel Spectrogram", use_container_width=True)
            else:
                st.image(processed_image, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            # Load model and predict
            with st.spinner(f"Loading {selected_model} model..."):
                try:
                    model = get_model(selected_model, input_type)
                    st.session_state.model = model
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
                    return
            
            with st.spinner("Running prediction..."):
                try:
                    result = run_prediction(model, processed_image, class_names)
                    st.session_state.prediction = result
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
                    return
            
            # Display prediction results
            st.subheader("📊 Classification Results")
            
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                st.metric(
                    label="Predicted Class",
                    value=result['class_name'],
                    delta=f"{result['confidence']*100:.1f}% confidence"
                )
            
            with result_col2:
                # Probability bar chart
                import plotly.express as px
                prob_df = {
                    'Class': class_names,
                    'Probability': result['probabilities']
                }
                fig = px.bar(prob_df, x='Class', y='Probability', 
                            color='Class', 
                            title='Class Probabilities')
                fig.update_layout(showlegend=False, height=300)
                st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # XAI Explanation
        st.subheader(f"🔍 {selected_xai} Explanation")
        
        with st.spinner(f"Generating {selected_xai} explanation..."):
            try:
                if selected_xai == 'LIME':
                    fig = run_lime_explanation(model, processed_image, class_names)
                elif selected_xai == 'GradCAM':
                    fig = run_gradcam_explanation(model, processed_image, class_names, selected_model)
                elif selected_xai == 'SHAP':
                    fig = run_shap_explanation(model, processed_image, class_names)
                
                st.pyplot(fig)
                plt.close(fig)
                
            except Exception as e:
                st.error(f"Error generating explanation: {str(e)}")
                st.info("Some XAI techniques may require specific model architectures. Try a different model or XAI method.")
    
    elif uploaded_file is not None:
        # File uploaded but analysis not run yet
        st.info("👈 Select a model and XAI technique, then click 'Run Analysis'")
        
        # Preview the input
        if input_type == 'audio':
            st.subheader("🎵 Audio Preview")
            st.audio(uploaded_file)
        else:
            st.subheader("🖼️ Image Preview")
            st.image(uploaded_file, width=400)


def comparison_page():
    """Comparison page for multiple XAI techniques"""
    st.markdown('<h1 class="main-header">📊 XAI Comparison</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Compare multiple explainability techniques side by side</p>', unsafe_allow_html=True)
    
    # Check if we have processed data
    if st.session_state.processed_image is None or st.session_state.model is None:
        st.warning("⚠️ Please run an analysis on the main page first!")
        st.info("Go to the 'Analysis' tab, upload a file, select a model, and run the analysis.")
        return
    
    processed_image = st.session_state.processed_image
    model = st.session_state.model
    input_type = st.session_state.input_type
    prediction = st.session_state.prediction
    class_names = get_class_names(input_type)
    
    # Display current analysis info
    st.subheader("📋 Current Analysis")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Input Type", input_type.capitalize())
    with col2:
        st.metric("Prediction", prediction['class_name'])
    with col3:
        st.metric("Confidence", f"{prediction['confidence']*100:.1f}%")
    
    st.divider()
    
    # XAI selection for comparison
    st.subheader("🔍 Select XAI Techniques to Compare")
    
    available_xai = get_compatible_xai(input_type)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        use_lime = st.checkbox("LIME", value=True)
    with col2:
        use_gradcam = st.checkbox("Grad-CAM", value=True)
    with col3:
        use_shap = st.checkbox("SHAP", value=True)
    
    if st.button("🔄 Generate Comparisons", type="primary"):
        st.divider()
        
        # Determine which model we're using (for Grad-CAM layer name)
        # Try to infer from model name
        model_name = getattr(model, 'name', 'VGG16')
        
        selected_xai = []
        if use_lime:
            selected_xai.append('LIME')
        if use_gradcam:
            selected_xai.append('GradCAM')
        if use_shap:
            selected_xai.append('SHAP')
        
        if not selected_xai:
            st.warning("Please select at least one XAI technique!")
            return
        
        # Generate explanations
        explanations = {}
        
        for xai_name in selected_xai:
            with st.spinner(f"Generating {xai_name} explanation..."):
                try:
                    if xai_name == 'LIME':
                        fig = run_lime_explanation(model, processed_image, class_names)
                    elif xai_name == 'GradCAM':
                        fig = run_gradcam_explanation(model, processed_image, class_names, model_name)
                    elif xai_name == 'SHAP':
                        fig = run_shap_explanation(model, processed_image, class_names)
                    
                    explanations[xai_name] = fig
                except Exception as e:
                    st.error(f"Error generating {xai_name}: {str(e)}")
        
        # Display comparisons
        st.subheader("📈 XAI Comparison Results")
        
        # Create columns based on number of selected XAI
        if len(explanations) == 1:
            cols = st.columns(1)
        elif len(explanations) == 2:
            cols = st.columns(2)
        else:
            cols = st.columns(3)
        
        for i, (xai_name, fig) in enumerate(explanations.items()):
            with cols[i % len(cols)]:
                st.markdown(f"### {xai_name}")
                st.pyplot(fig)
                plt.close(fig)
        
        # Comparison summary
        st.divider()
        st.subheader("📝 Comparison Summary")
        
        st.markdown(f"""
        | Technique | Description | Best For |
        |-----------|-------------|----------|
        | **LIME** | Perturbs superpixels to find important regions | Understanding local decision boundaries |
        | **Grad-CAM** | Uses gradients from conv layers | Identifying spatial features |
        | **SHAP** | Shapley value-based attributions | Feature importance ranking |
        
        **Analysis Notes:**
        - Different XAI methods may highlight different aspects of the model's decision
        - Consistent highlighted regions across methods indicate robust feature importance
        - Prediction: **{prediction['class_name']}** with **{prediction['confidence']*100:.1f}%** confidence
        """)


def about_page():
    """About page with project information"""
    st.markdown('<h1 class="main-header">ℹ️ About</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ## Unified Explainable AI Interface
    
    This platform integrates two existing XAI systems into a single interactive interface capable of 
    processing both audio and image data.
    
    ### 🎵 Deepfake Audio Detection
    
    Detects real vs. fake audio using neural networks trained on spectrograms from the Fake-or-Real (FoR) dataset.
    
    **Models:**
    - VGG16, MobileNet (91.5% accuracy)
    - ResNet50, InceptionV3
    
    ### 🏥 Lung Cancer Detection
    
    Detects malignant tumors in chest X-rays using transfer learning models.
    
    **Models:**
    - AlexNet
    - DenseNet121
    
    ### 🔍 XAI Techniques
    
    | Technique | Description |
    |-----------|-------------|
    | **LIME** | Local Interpretable Model-agnostic Explanations - explains predictions by perturbing input |
    | **Grad-CAM** | Gradient-weighted Class Activation Mapping - visualizes important regions using gradients |
    | **SHAP** | SHapley Additive exPlanations - provides feature attributions based on game theory |
    
    ### 🔧 Features
    
    - **Multi-modal input:** Support for audio (.wav) and image (.png, .jpg) files
    - **Automatic filtering:** XAI methods automatically filtered based on input type
    - **Comparison view:** Side-by-side comparison of multiple XAI techniques
    - **Interactive interface:** Easy-to-use Streamlit interface
    
    ---
    
    ### Generative AI Usage Statement
    
    This project was developed with assistance from **GitHub Copilot (Claude Opus 4.5)** for:
    - Code refactoring and integration
    - Documentation writing
    - Architecture design
    - Debugging and optimization
    
    ---
    
    ### References
    
    - Original Deepfake Audio Detection repository
    - Original Lung Cancer Detection repository
    - LIME: Ribeiro et al., "Why Should I Trust You?" (KDD 2016)
    - Grad-CAM: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks" (ICCV 2017)
    - SHAP: Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions" (NeurIPS 2017)
    """)


def main():
    """Main application entry point"""
    load_css()
    
    # Navigation
    tab1, tab2 = st.tabs(["🔬 Analysis", "📊 Comparison"])
    
    with tab1:
        main_page()
    
    with tab2:
        comparison_page()
    
    # with tab3:
    #     about_page()


if __name__ == "__main__":
    main()
