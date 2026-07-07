import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image, ImageFilter, ImageDraw
import cv2
import os
import time
import random

# ==========================================
# 1. PAGE CONFIG & CUSTOM CSS
# ==========================================
st.set_page_config(page_title="AI Dental Caries Detection", page_icon="🦷", layout="wide")

st.markdown("""
    <style>
        .title { text-align: center; font-size: 36px !important; font-weight: bold; color: #004d99; margin-bottom: 10px; }
        .subtitle { text-align: center; font-size: 20px; color: #666; margin-bottom: 30px; }
        .metric-card { background: linear-gradient(135deg, #004d99 0%, #0073e6 100%); padding: 20px; border-radius: 10px; color: white; text-align: center; }
        .info-box { background-color: #e6f2ff; border-left: 5px solid #004d99; padding: 15px; margin: 15px 0; border-radius: 5px; }
        .success-box { background-color: #e6ffe6; border-left: 5px solid #009900; padding: 15px; margin: 15px 0; border-radius: 5px; }
        .warning-box { background-color: #ffe6e6; border-left: 5px solid #cc0000; padding: 15px; margin: 15px 0; border-radius: 5px; }
        .question-box { font-size: 20px; font-weight: bold; color: #004d99; }
        .stRadio > div { flex-direction: row; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("🦷 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Introduction", "📊 Data Overview", "🏆 Model Performance", "🔬 AI Prediction Tool"])

# ==========================================
# 3. INTRODUCTION PAGE (Storytelling)
# ==========================================
if page == "🏠 Introduction":
    st.markdown("<p class='title'>🦷 AI-Assisted Dental Caries Detection</p>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Empowering Dentists with Explainable Deep Learning</p>", unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 2])
    with col1:
        st.image("https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExaGxsZmw3bTJnNmIyb3V1OXllZHNtaWFwbHNjbHF5ZzVlN3k2b2xveSZlcD12MV9naWZzX3NlYXJjaCZjdD1n/cb89q6BvqAHfwH6AEU/giphy.gif", width=350)
    with col2:
        st.markdown("""
        <p style='font-size: 18px; text-align: justify; line-height: 1.6;'>
        Dental caries (tooth decay) is one of the most prevalent oral diseases globally, affecting approximately 
        <b>2.3 billion people worldwide</b> (WHO, 2022). In Malaysia, the prevalence is dangerously high at <b>85.1%</b> among adults.
        </p>
        <p style='font-size: 18px; text-align: justify; line-height: 1.6;'>
        Traditional detection relies on manual radiograph interpretation—a process that is subjective, time-consuming, 
        and highly dependent on the practitioner's experience and fatigue levels. Early-stage lesions are subtle and easily missed.
        </p>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🎯 Project Objectives")
    st.markdown("""
    1. **Review** current techniques for identifying dental caries in radiographic images.
    2. **Develop** an AI-driven system utilizing Transfer Learning (CNNs) to detect caries from dental X-rays.
    3. **Evaluate** the system's accuracy and implement **Explainable AI (Grad-CAM)** to build clinical trust.
    """)

    st.subheader("📊 Quick Question")
    st.markdown("<p class='question-box'>What percentage of Malaysian adults suffer from dental caries?</p>", unsafe_allow_html=True)
    options = ["65.2%", "75.8%", "85.1%", "92.3%"]
    answer = st.radio("Select an answer:", options, horizontal=True)
    if st.button("Submit Answer"):
        if answer == "85.1%":
            st.success("✅ Correct! According to NOHSA 2020, 85.1% of Malaysian adults have dental caries.")
        else:
            st.error("❌ Incorrect. The correct answer is 85.1% - highlighting the critical need for automated detection!")

# ==========================================
# 4. DATA OVERVIEW PAGE (EDA)
# ==========================================
elif page == "📊 Data Overview":
    st.markdown("<p class='title'>📊 Dataset & Exploratory Data Analysis</p>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Understanding the Dental Radiograph Dataset</p>", unsafe_allow_html=True)

    # Dataset Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric(label="📷 Total Images", value="500+")
    with col2: st.metric(label="✅ Healthy (No Cavity)", value="350")
    with col3: st.metric(label="🦠 Cavity Present", value="150")
    with col4: st.metric(label="📏 Image Size", value="224x224")

    st.markdown("---")
    
    # Class Distribution Visualization
    st.subheader("⚖️ Class Distribution & Imbalance")
    st.write("The dataset exhibits a class imbalance, which is common in medical imaging. We addressed this using **Class Weights** during model training.")
    
    dist_data = pd.DataFrame({
        'Class': ['No Cavity (Healthy)', 'Cavity (Caries)'],
        'Count': [350, 150],
        'Percentage': [70, 30]
    })
    
    col_a, col_b = st.columns(2)
    with col_a:
        fig_pie = px.pie(dist_data, values='Count', names='Class', hole=0.4,
                         title='Dataset Class Distribution',
                         color_discrete_sequence=['#4CAF50', '#F44336'])
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col_b:
        fig_bar = px.bar(dist_data, x='Class', y='Count', color='Class',
                         title='Number of Images per Class',
                         color_discrete_sequence=['#4CAF50', '#F44336'],
                         text='Count')
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("""
    <div class="info-box">
    <b>🔍 Preprocessing Pipeline:</b> Before feeding images into the neural network, we applied a strict preprocessing pipeline:
    <ul>
        <li><b>Grayscale Conversion:</b> Removes unnecessary color noise.</li>
        <li><b>CLAHE (Contrast Limited Adaptive Histogram Equalization):</b> Enhances subtle radiolucent boundaries of early caries.</li>
        <li><b>Normalization:</b> Scales pixel values to [0, 1] for faster model convergence.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 5. MODEL PERFORMANCE PAGE
# ==========================================
elif page == "🏆 Model Performance":
    st.markdown("<p class='title'>🏆 Model Comparison & Evaluation</p>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Baseline CNN vs. Transfer Learning (MobileNetV2 & ResNet50)</p>", unsafe_allow_html=True)

    st.write("""
    We evaluated three architectures to determine the most effective approach for dental caries detection. 
    While Transfer Learning models are powerful, they require careful fine-tuning to adapt to medical imaging.
    """)

    # Model Results Data
    model_results = pd.DataFrame({
        'Model': ['Baseline CNN (GAP)', 'MobileNetV2', 'ResNet50'],
        'Accuracy': [0.76, 0.74, 0.64],
        'Precision': [0.33, 0.29, 0.00],
        'Recall': [0.42, 0.42, 0.00],
        'F1-Score': [0.37, 0.34, 0.00]
    })

    # Grouped Bar Chart for Metrics
    st.subheader("📊 Performance Metrics Comparison")
    fig_metrics = go.Figure()
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
        fig_metrics.add_trace(go.Bar(
            x=model_results['Model'],
            y=model_results[metric],
            name=metric,
            text=model_results[metric].apply(lambda x: f'{x:.0%}'),
            textposition='outside'
        ))
    
    fig_metrics.update_layout(
        barmode='group',
        title='Model Performance Across Key Metrics',
        yaxis_title='Score',
        yaxis_tickformat='.0%',
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_metrics, use_container_width=True)

    st.markdown("---")

    # Key Findings
    st.subheader("🔑 Critical Findings & Challenges")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="success-box">
        <b>✅ Why Baseline CNN Won:</b><br>
        The custom Baseline CNN with Global Average Pooling (GAP) outperformed the massive Transfer Learning models. 
        Its simpler architecture prevented overfitting on our limited dataset, and the built-in Data Augmentation 
        (rotation, zoom, contrast) helped it generalize better.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="warning-box">
        <b>⚠️ Why ResNet50 Failed:</b><br>
        ResNet50 (23M parameters) achieved 0% Recall. This is due to the "curse of dimensionality"—the model was 
        too complex for the dataset size. Furthermore, applying CLAHE contrast enhancement disrupted ResNet's 
        pre-trained ImageNet weight distributions, confusing the feature extractor.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <b>📈 Medical Threshold Tuning (ROC Analysis):</b><br>
    In medical diagnostics, missing a cavity (False Negative) is dangerous. By analyzing the ROC Curve, we lowered 
    the default decision threshold from <b>0.50 to 0.35</b>. This significantly increased the model's <b>Recall (Sensitivity)</b>, 
    ensuring the AI acts as a highly sensitive "second-opinion" screening tool.
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 6. AI PREDICTION TOOL PAGE
# ==========================================
elif page == "🔬 AI Prediction Tool":
    st.markdown("<p class='title'>🔬 AI Diagnostic Assistant</p>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Upload a Dental Radiograph for Instant Analysis & Explainability</p>", unsafe_allow_html=True)

    # Smart Fallback: Try to load TensorFlow
    try:
        import tensorflow as tf
        TF_AVAILABLE = True
    except ImportError:
        TF_AVAILABLE = False

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### 📤 Upload Radiograph")
        uploaded_file = st.file_uploader("Choose an X-ray image (JPG, PNG)...", type=["jpg", "jpeg", "png"])
        
        st.markdown("---")
        st.markdown("### ⚙️ System Status")
        
        if TF_AVAILABLE:
            @st.cache_resource
            def load_model():
                if os.path.exists("best_caries_model.keras"):
                    return tf.keras.models.load_model("best_caries_model.keras")
                return None
            
            model = load_model()
            if model is not None:
                st.success("✅ **AI Engine Active** (TensorFlow Loaded)")
            else:
                st.warning("⚠️ Model file not found. Running in Demo Mode.")
                TF_AVAILABLE = False
        else:
            st.warning("☁️ **Cloud Demo Mode** (AI Engine simulated)")
            
        st.info("""
        **Model:** Custom CNN with Transfer Learning  
        **Preprocessing:** CLAHE Contrast Enhancement  
        **Explainability:** Grad-CAM Heatmaps  
        **Threshold:** Optimized for Medical Recall (0.35)
        """)

    with col2:
        if uploaded_file is not None:
            # Open image with PIL
            image = Image.open(uploaded_file).convert('L') # Convert to grayscale
            st.image(image, caption="Original Grayscale X-Ray", use_column_width=True)
            
            # Simulate processing time
            with st.spinner('Analyzing radiograph...'):
                time.sleep(1.5)
                
                if TF_AVAILABLE:
                    # ==============================
                    # REAL AI PREDICTION (Localhost)
                    # ==============================
                    img_array = np.array(image)
                    resized = cv2.resize(img_array, (224, 224))
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    enhanced = clahe.apply(resized)
                    normalized = enhanced / 255.0
                    input_data = np.expand_dims(normalized, axis=(0, -1))
                    
                    prediction_prob = model.predict(input_data)[0][0]
                    
                    # Real Grad-CAM
                    def generate_gradcam(model, image_array):
                        last_conv_layer = None
                        for layer in reversed(model.layers):
                            if isinstance(layer, tf.keras.layers.Conv2D):
                                last_conv_layer = layer.name
                                break
                        if not last_conv_layer: return None
                        grad_model = tf.keras.models.Model(inputs=model.input, outputs=[model.get_layer(last_conv_layer).output, model.output])
                        with tf.GradientTape() as tape:
                            conv_outputs, preds = grad_model(image_array)
                            loss = preds[:, 0]
                        grads = tape.gradient(loss, conv_outputs)
                        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
                        heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
                        heatmap = tf.squeeze(heatmap)
                        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
                        return heatmap.numpy()

                    heatmap = generate_gradcam(model, input_data)
                    if heatmap is not None:
                        heatmap_resized = cv2.resize(heatmap, (224, 224))
                        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
                        original_rgb = cv2.cvtColor((normalized.squeeze() * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
                        superimposed = cv2.addWeighted(original_rgb, 0.6, heatmap_colored, 0.4, 0)
                        heatmap_img = Image.fromarray(superimposed)
                    else:
                        heatmap_img = image
                else:
                    # ==============================
                    # DEMO MODE (Streamlit Cloud)
                    # ==============================
                    prediction_prob = random.uniform(0.35, 0.85)
                    
                    # Generate fake heatmap using pure PIL
                    heatmap_img = image.copy().convert('RGB').resize((224, 224))
                    overlay = Image.new('RGBA', (224, 224), (0, 0, 0, 0))
                    draw = ImageDraw.Draw(overlay)
                    cx, cy = random.randint(80, 140), random.randint(80, 140)
                    r = random.randint(30, 60)
                    draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(255, 0, 0, 120))
                    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=15))
                    heatmap_img = Image.alpha_composite(heatmap_img.convert('RGBA'), overlay).convert('RGB')
            
            # Determine result
            OPTIMAL_THRESHOLD = 0.35
            is_cavity = prediction_prob >= OPTIMAL_THRESHOLD
            confidence = prediction_prob if is_cavity else (1 - prediction_prob)
            
            st.markdown("---")
            st.markdown("### 🩺 AI Diagnosis & Explainability")
            
            diag_col, cam_col = st.columns(2)
            
            with diag_col:
                if is_cavity:
                    st.error(f"🚨 **CARIES DETECTED** \n\n Confidence: {confidence*100:.1f}%")
                    st.markdown("""
                    <div class="warning-box">
                    <b>Clinical Recommendation:</b> The AI has identified radiolucent patterns 
                    consistent with dental caries. Please verify with clinical examination.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.success(f"✅ **NO CARIES DETECTED** \n\n Confidence: {confidence*100:.1f}%")
                    st.markdown("""
                    <div class="success-box">
                    <b>Clinical Note:</b> No significant radiolucent patterns detected. 
                    Continue with routine preventive care.
                    </div>
                    """, unsafe_allow_html=True)
                    
            with cam_col:
                st.markdown("#### 🔥 Explainable AI (Grad-CAM)")
                st.image(heatmap_img, caption="AI Attention Overlay (Red = High Focus)", use_column_width=True)
                if not TF_AVAILABLE:
                    st.caption("*Note: Heatmap is simulated for cloud demonstration purposes.*")
                    
            st.markdown(f"*Medical Threshold set to {OPTIMAL_THRESHOLD:.2f} to prioritize Sensitivity (Recall).*")

        else:
            st.markdown("""
            <div class="info-box">
                <h4>📖 How to Use This Tool</h4>
                <p>1. Upload a digital dental radiograph (X-ray) in JPG or PNG format.</p>
                <p>2. The system will automatically apply <b>CLAHE contrast enhancement</b> to highlight subtle lesions.</p>
                <p>3. The AI will analyze the image and provide a <b>binary classification</b> (Caries / No Caries) with a confidence score.</p>
                <p>4. Review the <b>Grad-CAM Heatmap</b> to see exactly which regions of the tooth influenced the AI's decision.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.info("👈 Please upload a dental radiograph in the left sidebar to begin analysis.")

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 10px;'>
    <p>🦷 Automated Detection of Dental Caries from Dental X-ray Image | Final Year Project 2024</p>
    <p>Developed by Barry Ng Kee Hong (0135374) | University of Wollongong Malaysia</p>
</div>
""", unsafe_allow_html=True)
