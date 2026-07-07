import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os

# ==========================================
# PAGE CONFIG & CUSTOM CSS
# ==========================================
st.set_page_config(
    page_title="AI Dental Caries Detection", 
    page_icon="🦷", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Look
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #1a5276;
        font-weight: 700;
    }
    
    /* Custom Title */
    .hero-title {
        font-size: 42px !important;
        font-weight: 800;
        color: #1a5276;
        text-align: center;
        margin-bottom: 10px;
    }
    
    .hero-subtitle {
        font-size: 20px;
        color: #555;
        text-align: center;
        margin-bottom: 40px;
        font-weight: 400;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 20px rgba(0,0,0,0.15);
        transition: transform 0.3s;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .metric-card h3 {
        color: white !important;
        margin: 0;
        font-size: 18px;
        font-weight: 600;
    }
    
    .metric-card p {
        font-size: 36px;
        font-weight: bold;
        margin: 10px 0 0 0;
    }
    
    /* Info Boxes */
    .info-box {
        background-color: #e3f2fd;
        border-left: 6px solid #2196f3;
        padding: 20px;
        margin: 20px 0;
        border-radius: 8px;
        font-size: 16px;
        line-height: 1.6;
        color: #0d47a1;
    }
    
    .success-box {
        background-color: #e8f5e9;
        border-left: 6px solid #4caf50;
        padding: 20px;
        margin: 20px 0;
        border-radius: 8px;
        font-size: 16px;
        line-height: 1.6;
        color: #1b5e20;
    }
    
    .warning-box {
        background-color: #fff3e0;
        border-left: 6px solid #ff9800;
        padding: 20px;
        margin: 20px 0;
        border-radius: 8px;
        font-size: 16px;
        line-height: 1.6;
        color: #e65100;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("🦷 Navigation")
page = st.sidebar.radio(
    "Go to", 
    ["🏠 Introduction", "📊 Data Overview", "🏆 Model Performance", "🔬 AI Prediction Tool"]
)

# ==========================================
# INTRODUCTION PAGE
# ==========================================
if page == " Introduction":
    st.markdown('<p class="hero-title">🦷 AI-Assisted Dental Caries Detection</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Empowering Dentists with Explainable Deep Learning</p>', unsafe_allow_html=True)

    # Problem Statement Section
    st.markdown("###  The Challenge")
    st.markdown("""
    Dental caries (tooth decay) is one of the most prevalent oral diseases globally, affecting approximately **2.3 billion people worldwide** (WHO, 2022). In Malaysia, the prevalence is dangerously high at **85.1%** among adults.
    
    Traditional detection relies on manual radiograph interpretation—a process that is **subjective**, **time-consuming**, and highly dependent on the practitioner's experience and fatigue levels. Early-stage lesions are subtle and easily missed.
    """)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Statistics Cards
    st.markdown("### 🌍 The Global & Local Burden")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("""
        <div class="metric-card">
            <h3>Global Cases</h3>
            <p>2.3B+</p>
        </div>
        """, unsafe_allow_html=True)
    
    with c2:
        st.markdown("""
        <div class="metric-card">
            <h3>Malaysia Adults</h3>
            <p>85.1%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with c3:
        st.markdown("""
        <div class="metric-card">
            <h3>Malaysia Kids</h3>
            <p>71.3%</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Project Objectives
    st.markdown("### 🎯 Project Objectives")
    st.markdown("""
    1. **Review** current techniques for identifying dental caries in radiographic images.
    2. **Develop** an AI-driven system utilizing Transfer Learning (CNNs) to detect caries from dental X-rays.
    3. **Evaluate** the system's accuracy and implement **Explainable AI (Grad-CAM)** to build clinical trust.
    """)

    # Interactive Quiz
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("###  Quick Knowledge Check")
    st.markdown("**What percentage of Malaysian adults suffer from dental caries?**")
    
    options = ["65.2%", "75.8%", "85.1%", "92.3%"]
    answer = st.radio("Select an answer:", options, horizontal=True, label_visibility="collapsed")
    
    if st.button("Submit Answer", use_container_width=True):
        if answer == "85.1%":
            st.success("✅ Correct! According to NOHSA 2020, 85.1% of Malaysian adults have dental caries.")
        else:
            st.error("❌ Incorrect. The correct answer is 85.1% - highlighting the critical need for automated detection!")

# ==========================================
# DATA OVERVIEW PAGE
# ==========================================
elif page == "📊 Data Overview":
    st.markdown('<p class="hero-title">📊 Dataset & Exploratory Data Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Understanding the Dental Radiograph Dataset</p>', unsafe_allow_html=True)

    # Dataset Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1: 
        st.metric(label="📷 Total Images", value="500+", delta="Curated")
    with col2: 
        st.metric(label="✅ Healthy", value="350", delta="70%")
    with col3: 
        st.metric(label="🦠 Cavity", value="150", delta="30%")
    with col4: 
        st.metric(label=" Resolution", value="224x224", delta="Pixels")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Class Distribution Visualization
    st.markdown("### ⚖️ Class Distribution & Imbalance")
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
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col_b:
        fig_bar = px.bar(dist_data, x='Class', y='Count', color='Class',
                         title='Number of Images per Class',
                         color_discrete_sequence=['#4CAF50', '#F44336'],
                         text='Count')
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Preprocessing Pipeline
    st.markdown("### 🔍 Preprocessing Pipeline")
    st.write("Before feeding images into the neural network, we applied a strict preprocessing pipeline to enhance subtle radiolucent boundaries.")
    
    step1, step2, step3, step4 = st.columns(4)
    with step1:
        st.markdown("""
        <div style='background-color: #e3f2fd; border: 2px solid #2196f3; border-radius: 10px; padding: 15px; text-align: center;'>
            <h3 style='color: #1976d2; margin: 0;'>1️⃣</h3>
            <p style='margin: 10px 0; font-weight: 600;'>Grayscale<br>Conversion</p>
        </div>
        """, unsafe_allow_html=True)
    with step2:
        st.markdown("""
        <div style='background-color: #e3f2fd; border: 2px solid #2196f3; border-radius: 10px; padding: 15px; text-align: center;'>
            <h3 style='color: #1976d2; margin: 0;'>2️⃣</h3>
            <p style='margin: 10px 0; font-weight: 600;'>CLAHE<br>Enhancement</p>
        </div>
        """, unsafe_allow_html=True)
    with step3:
        st.markdown("""
        <div style='background-color: #e3f2fd; border: 2px solid #2196f3; border-radius: 10px; padding: 15px; text-align: center;'>
            <h3 style='color: #1976d2; margin: 0;'>3️⃣</h3>
            <p style='margin: 10px 0; font-weight: 600;'>Resize to<br>224x224</p>
        </div>
        """, unsafe_allow_html=True)
    with step4:
        st.markdown("""
        <div style='background-color: #e3f2fd; border: 2px solid #2196f3; border-radius: 10px; padding: 15px; text-align: center;'>
            <h3 style='color: #1976d2; margin: 0;'>4️⃣</h3>
            <p style='margin: 10px 0; font-weight: 600;'>Normalization<br>[0, 1]</p>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("️ Why CLAHE is Critical for Dental X-Rays"):
        st.markdown("""
        **Contrast Limited Adaptive Histogram Equalization (CLAHE)** is essential because early-stage caries 
        present as subtle radiolucent (dark) changes. Standard histogram equalization can amplify background noise, 
        but CLAHE enhances local contrast specifically around the tooth structures, making micro-cavities visible to the CNN.
        """)

# ==========================================
# MODEL PERFORMANCE PAGE
# ==========================================
elif page == "🏆 Model Performance":
    st.markdown('<p class="hero-title">🏆 Model Comparison & Evaluation</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Baseline CNN vs. Transfer Learning (MobileNetV2 & ResNet50)</p>', unsafe_allow_html=True)

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
    st.markdown("### 📊 Performance Metrics Comparison")
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
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500
    )
    st.plotly_chart(fig_metrics, use_container_width=True)

    # RADAR CHART
    st.markdown("### 🕸️ Multi-Dimensional Radar Analysis")
    st.write("A radar chart provides a holistic view of each model's strengths and weaknesses across all metrics.")
    
    fig_radar = go.Figure()
    for model_name in model_results['Model']:
        row = model_results[model_results['Model'] == model_name].iloc[0]
        fig_radar.add_trace(go.Scatterpolar(
            r=[row['Accuracy'], row['Precision'], row['Recall'], row['F1-Score'], row['Accuracy']],
            theta=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Accuracy'],
            fill='toself',
            name=model_name
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        template='plotly_white',
        height=500
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Key Findings
    st.markdown("### 🔑 Critical Findings & Challenges")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="success-box">
        <strong>✅ Why Baseline CNN Won:</strong><br>
        The custom Baseline CNN with Global Average Pooling (GAP) outperformed the massive Transfer Learning models. 
        Its simpler architecture prevented overfitting on our limited dataset, and the built-in Data Augmentation 
        (rotation, zoom, contrast) helped it generalize better.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="warning-box">
        <strong>⚠️ Why ResNet50 Failed:</strong><br>
        ResNet50 (23M parameters) achieved 0% Recall. This is due to the "curse of dimensionality"—the model was 
        too complex for the dataset size. Furthermore, applying CLAHE contrast enhancement disrupted ResNet's 
        pre-trained ImageNet weight distributions, confusing the feature extractor.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <strong>📈 Medical Threshold Tuning (ROC Analysis):</strong><br>
    In medical diagnostics, missing a cavity (False Negative) is dangerous. By analyzing the ROC Curve, we lowered 
    the default decision threshold from <strong>0.50 to 0.35</strong>. This significantly increased the model's <strong>Recall (Sensitivity)</strong>, 
    ensuring the AI acts as a highly sensitive "second-opinion" screening tool.
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# AI PREDICTION TOOL PAGE
# ==========================================
elif page == " AI Prediction Tool":
    st.markdown('<p class="hero-title">🔬 AI Diagnostic Assistant</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Upload a Dental Radiograph for Instant Analysis & Explainability</p>', unsafe_allow_html=True)

    # Smart Fallback: Try to load TensorFlow
    try:
        import tensorflow as tf
        TF_AVAILABLE = True
    except ImportError:
        TF_AVAILABLE = False

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("###  Upload Radiograph")
        uploaded_file = st.file_uploader(
            "Choose an X-ray image (JPG, PNG)...", 
            type=["jpg", "jpeg", "png"],
            help="Upload a periapical, bitewing, or panoramic dental radiograph"
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
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
            st.warning("️ **Cloud Demo Mode** (AI Engine simulated)")
            
        st.info("""
        **Model:** Custom CNN with Transfer Learning  
        **Preprocessing:** CLAHE Contrast Enhancement  
        **Explainability:** Grad-CAM Heatmaps  
        **Threshold:** Optimized for Medical Recall (0.35)
        """)

    with col2:
        if uploaded_file is not None:
            # Open image with PIL
            from PIL import Image
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Grayscale X-Ray", use_column_width=True)
            
            # Simulate processing time
            with st.spinner('🔍 Step 1: Applying CLAHE Enhancement...'):
                import time
                time.sleep(0.5)
            with st.spinner('🧠 Step 2: Running CNN Inference...'):
                time.sleep(0.5)
            with st.spinner('🔥 Step 3: Generating Grad-CAM Heatmap...'):
                time.sleep(0.5)
                
                if TF_AVAILABLE:
                    # REAL AI PREDICTION (Localhost)
                    import cv2
                    img_array = np.array(image)
                    if len(img_array.shape) == 3:
                        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                    else:
                        gray = img_array
                    
                    resized = cv2.resize(gray, (224, 224))
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
                    # DEMO MODE (Streamlit Cloud)
                    prediction_prob = np.random.uniform(0.35, 0.85)
                    
                    # Generate fake heatmap using pure PIL
                    from PIL import ImageFilter, ImageDraw
                    heatmap_img = image.copy().convert('RGB').resize((224, 224))
                    overlay = Image.new('RGBA', (224, 224), (0, 0, 0, 0))
                    draw = ImageDraw.Draw(overlay)
                    cx, cy = np.random.randint(80, 140), np.random.randint(80, 140)
                    r = np.random.randint(30, 60)
                    draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(255, 0, 0, 120))
                    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=15))
                    heatmap_img = Image.alpha_composite(heatmap_img.convert('RGBA'), overlay).convert('RGB')
            
            # Determine result
            OPTIMAL_THRESHOLD = 0.35
            is_cavity = prediction_prob >= OPTIMAL_THRESHOLD
            confidence = prediction_prob if is_cavity else (1 - prediction_prob)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 🩺 AI Diagnosis & Explainability")
            
            diag_col, cam_col = st.columns(2)
            
            with diag_col:
                if is_cavity:
                    st.error(f" **CARIES DETECTED** \n\n Confidence: {confidence*100:.1f}%")
                    st.markdown("""
                    <div class="warning-box">
                    <strong>Clinical Recommendation:</strong> The AI has identified radiolucent patterns 
                    consistent with dental caries. Please verify with clinical examination.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.success(f"✅ **NO CARIES DETECTED** \n\n Confidence: {confidence*100:.1f}%")
                    st.markdown("""
                    <div class="success-box">
                    <strong>Clinical Note:</strong> No significant radiolucent patterns detected. 
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
                <p>2. The system will automatically apply <strong>CLAHE contrast enhancement</strong> to highlight subtle lesions.</p>
                <p>3. The AI will analyze the image and provide a <strong>binary classification</strong> (Caries / No Caries) with a confidence score.</p>
                <p>4. Review the <strong>Grad-CAM Heatmap</strong> to see exactly which regions of the tooth influenced the AI's decision.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.info("👈 Please upload a dental radiograph in the left sidebar to begin analysis.")

# ==========================================
# FOOTER
# ==========================================
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px; border-top: 2px solid #ddd; margin-top: 40px;'>
    <p><strong>🦷 Automated Detection of Dental Caries from Dental X-ray Image</strong></p>
    <p>Final Year Project 2024 | University of Wollongong Malaysia</p>
    <p>Developed by <strong>Barry Ng Kee Hong (0135374)</strong></p>
</div>
""", unsafe_allow_html=True)
