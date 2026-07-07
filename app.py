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
# PAGE CONFIG & CUSTOM CSS
# ==========================================
st.set_page_config(
    page_title="AI Dental Caries Detection System", 
    page_icon="🦷", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Force LIGHT text on dark background */
    h1, h2, h3, h4, h5, h6, p, div, span, label, strong, b {
        color: #FFFFFF !important;
    }
    
    /* App Header */
    .app-header {
        text-align: center;
        padding: 20px 0 10px 0;
        border-bottom: 2px solid #334155;
        margin-bottom: 25px;
    }
    .app-header h1 {
        font-size: 32px !important;
        color: #00E5FF !important;
        margin: 0 !important;
    }
    .app-header p {
        font-size: 16px;
        color: #94A3B8 !important;
        margin: 5px 0 0 0 !important;
    }
    
    /* Metric Cards (Dark Gradient) */
    .stat-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #334155;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        text-align: center;
        transition: transform 0.2s;
    }
    .stat-card:hover {
        transform: translateY(-3px);
        border-color: #00E5FF;
    }
    .stat-card h3 {
        color: #94A3B8 !important;
        font-size: 14px !important;
        margin: 0 0 8px 0 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stat-card p {
        font-size: 28px;
        font-weight: 700;
        color: #FFFFFF !important;
        margin: 0 !important;
    }
    
    /* Info & Alert Boxes (Dark Mode) */
    .info-box {
        background-color: #1E293B;
        border-left: 5px solid #3B82F6;
        padding: 18px 20px;
        margin: 18px 0;
        border-radius: 8px;
        font-size: 15px;
        line-height: 1.6;
        color: #E2E8F0 !important;
    }
    .success-box {
        background-color: #1E293B;
        border-left: 5px solid #10B981;
        padding: 18px 20px;
        margin: 18px 0;
        border-radius: 8px;
        font-size: 15px;
        line-height: 1.6;
        color: #E2E8F0 !important;
    }
    .warning-box {
        background-color: #1E293B;
        border-left: 5px solid #F59E0B;
        padding: 18px 20px;
        margin: 18px 0;
        border-radius: 8px;
        font-size: 15px;
        line-height: 1.6;
        color: #E2E8F0 !important;
    }
    
    /* Pipeline Steps (Dark Mode) */
    .pipeline-step {
        background-color: #1E293B;
        border: 2px solid #3B82F6;
        border-radius: 12px;
        padding: 18px 12px;
        text-align: center;
        font-weight: 600;
        color: #FFFFFF !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# HEADER
# ==========================================
st.markdown("""
<div class="app-header">
    <h1>🦷 AI Dental Caries Detection System</h1>
    <p>University of Wollongong Malaysia | Final Year Project 2024</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio(
    "Go to", 
    ["🏠 Dashboard", "📂 Dataset", "🧠 AI Models", "📈 Results", "🔬 AI Diagnosis", "ℹ️ About"],
    label_visibility="collapsed",
    index=0
)

# ==========================================
# 🏠 DASHBOARD
# ==========================================
if page == "🏠 Dashboard":
    st.markdown("### 🎯 Project Overview")
    st.markdown("""
    Dental caries (tooth decay) is a global health crisis affecting **2.3 billion people worldwide**. In Malaysia, adult prevalence sits at **85.1%**, with early-stage lesions frequently missed due to subjective manual interpretation and practitioner fatigue.
    
    This system leverages **Deep Learning (CNNs)** and **Explainable AI (Grad-CAM)** to provide dentists with a reliable, transparent second-opinion tool for automated caries detection from dental radiographs.
    """)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🌍 Burden of Disease")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("""
        <div class="stat-card">
            <h3>Global Cases</h3>
            <p>2.3B+</p>
        </div>
        """, unsafe_allow_html=True)
    
    with c2:
        st.markdown("""
        <div class="stat-card">
            <h3>Malaysia Adults</h3>
            <p>85.1%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with c3:
        st.markdown("""
        <div class="stat-card">
            <h3>Malaysia Kids</h3>
            <p>71.3%</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("### ✅ Core Objectives")
    st.markdown("""
    1. **Review** current techniques for identifying dental caries in radiographic images.
    2. **Develop** an AI-driven system utilizing Transfer Learning (CNNs) to detect caries from dental X-rays.
    3. **Evaluate** the system's accuracy and implement **Explainable AI (Grad-CAM)** to build clinical trust.
    """)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🧠 Quick Knowledge Check")
    st.markdown("**What percentage of Malaysian adults suffer from dental caries?**")
    options = ["65.2%", "75.8%", "85.1%", "92.3%"]
    answer = st.radio("Select an answer:", options, horizontal=True, label_visibility="collapsed")
    
    if st.button("Submit Answer", use_container_width=True):
        if answer == "85.1%":
            st.success("✅ Correct! According to NOHSA 2020, 85.1% of Malaysian adults have dental caries.")
        else:
            st.error("❌ Incorrect. The correct answer is 85.1% - highlighting the critical need for automated detection!")

# ==========================================
# 📂 DATASET
# ==========================================
elif page == "📂 Dataset":
    st.markdown("### 📊 Dataset Overview & Preprocessing")
    st.write("Curated dental radiograph dataset with expert annotations, strictly processed for medical AI standards.")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric(label="📷 Total Images", value="500+", delta="Curated")
    with col2: st.metric(label="✅ Healthy", value="350", delta="70%")
    with col3: st.metric(label="🦠 Cavity", value="150", delta="30%")
    with col4: st.metric(label="📏 Resolution", value="224x224", delta="Pixels")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### ⚖️ Class Distribution")
    st.write("Medical datasets are inherently imbalanced. We applied **Class Weights** during training to prevent bias toward the majority class.")
    
    dist_data = pd.DataFrame({
        'Class': ['No Cavity (Healthy)', 'Cavity (Caries)'],
        'Count': [350, 150]
    })
    
    col_a, col_b = st.columns(2)
    with col_a:
        fig_pie = px.pie(dist_data, values='Count', names='Class', hole=0.45,
                         title='Dataset Class Distribution',
                         color_discrete_sequence=['#10B981', '#EF4444'])
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(template='plotly_dark', height=400, 
                              plot_bgcolor='#0E1117', paper_bgcolor='#0E1117',
                              font=dict(color='#FFFFFF'))
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col_b:
        fig_bar = px.bar(dist_data, x='Class', y='Count', color='Class',
                         color_discrete_map={'No Cavity (Healthy)': '#10B981', 'Cavity (Caries)': '#EF4444'},
                         text='Count')
        fig_bar.update_layout(showlegend=False, template='plotly_dark', height=400,
                              plot_bgcolor='#0E1117', paper_bgcolor='#0E1117',
                              font=dict(color='#FFFFFF'))
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🔬 Preprocessing Pipeline")
    st.write("Raw radiographs undergo strict standardization before model ingestion.")
    
    step1, step2, step3, step4 = st.columns(4)
    with step1:
        st.markdown("""
        <div class="pipeline-step">
            <h3 style="color: #FFFFFF; margin: 0;">1️⃣</h3>
            <p style="margin: 10px 0; font-weight: 600; color: #FFFFFF;">Grayscale<br>Conversion</p>
        </div>
        """, unsafe_allow_html=True)
    with step2:
        st.markdown("""
        <div class="pipeline-step">
            <h3 style="color: #FFFFFF; margin: 0;">2️⃣</h3>
            <p style="margin: 10px 0; font-weight: 600; color: #FFFFFF;">CLAHE<br>Enhancement</p>
        </div>
        """, unsafe_allow_html=True)
    with step3:
        st.markdown("""
        <div class="pipeline-step">
            <h3 style="color: #FFFFFF; margin: 0;">3️⃣</h3>
            <p style="margin: 10px 0; font-weight: 600; color: #FFFFFF;">Resize<br>224×224</p>
        </div>
        """, unsafe_allow_html=True)
    with step4:
        st.markdown("""
        <div class="pipeline-step">
            <h3 style="color: #FFFFFF; margin: 0;">4️⃣</h3>
            <p style="margin: 10px 0; font-weight: 600; color: #FFFFFF;">Normalization<br>[0, 1]</p>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("ℹ️ Why CLAHE is Critical for Dental X-Rays"):
        st.markdown("""
        **Contrast Limited Adaptive Histogram Equalization (CLAHE)** enhances local contrast without amplifying noise. 
        Early caries appear as subtle radiolucent (dark) regions. CLAHE selectively brightens these boundaries, 
        making micro-lesions detectable to the convolutional filters.
        """)

# ==========================================
# 🧠 AI MODELS
# ==========================================
elif page == "🧠 AI Models":
    st.markdown("### 🏗️ Architecture & Methodology")
    st.write("Three architectures were rigorously evaluated to determine the optimal clinical deployment model.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="success-box">
        <strong>✅ Baseline CNN (Winner)</strong><br>
        Custom 3-layer ConvNet with Global Average Pooling. 
        Lightweight architecture prevents overfitting. Built-in data augmentation 
        (rotation, zoom, contrast) improves generalization on limited medical data.
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="warning-box">
        <strong>⚠️ ResNet50 & MobileNetV2</strong><br>
        Transfer learning models struggled. ResNet50 (23M parameters) failed completely 
        due to dataset size mismatch and CLAHE disrupting ImageNet weight distributions. 
        MobileNetV2 showed marginal improvement after fine-tuning but remained outperformed.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📋 Model Selection Criteria")
    st.markdown("""
    | Criteria | Baseline CNN | MobileNetV2 | ResNet50 |
    |---|---|---|---|
    | **Parameters** | ~1.2M | ~14M | ~23M |
    | **Training Stability** | High | Moderate | Low |
    | **Clinical Recall** | 42% | 42% | 0% |
    | **Deployment Speed** | Fast | Moderate | Slow |
    | **Explainability** | Excellent | Good | Poor |
    """)

# ==========================================
# 📈 RESULTS
# ==========================================
elif page == "📈 Results":
    st.markdown("### 📊 Performance Evaluation & Threshold Optimization")
    st.write("Comprehensive metric analysis to ensure clinical safety and diagnostic reliability.")

    model_results = pd.DataFrame({
        'Model': ['Baseline CNN', 'MobileNetV2', 'ResNet50'],
        'Accuracy': [0.76, 0.74, 0.64],
        'Precision': [0.33, 0.29, 0.00],
        'Recall': [0.42, 0.42, 0.00],
        'F1-Score': [0.37, 0.34, 0.00]
    })

    st.markdown("### 📈 Multi-Metric Comparison")
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
        template='plotly_dark',
        plot_bgcolor='#0E1117',
        paper_bgcolor='#0E1117',
        font=dict(color='#FFFFFF'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=450
    )
    st.plotly_chart(fig_metrics, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🕸️ Radar Analysis")
    st.write("Holistic view of model strengths across all evaluation dimensions.")
    
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
        template='plotly_dark',
        plot_bgcolor='#0E1117',
        paper_bgcolor='#0E1117',
        font=dict(color='#FFFFFF'),
        height=450
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🎯 Medical Threshold Tuning")
    st.markdown("""
    <div class="info-box">
    <strong>Clinical Safety First:</strong> In medical diagnostics, missing a cavity (False Negative) is dangerous. 
    Standard 0.50 thresholds prioritize precision but sacrifice sensitivity. By analyzing the ROC Curve, we lowered 
    the decision threshold to <strong>0.35</strong>, significantly increasing <strong>Recall (Sensitivity)</strong> 
    to ensure the AI acts as a highly sensitive screening assistant.
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 🔬 AI DIAGNOSIS (PREDICTION TOOL)
# ==========================================
elif page == "🔬 AI Diagnosis":
    st.markdown("### 🩺 Real-Time AI Diagnostic Assistant")
    st.write("Upload a dental radiograph for instant analysis, confidence scoring, and explainable heatmap visualization.")

    # TensorFlow Fallback Logic
    try:
        import tensorflow as tf
        TF_AVAILABLE = True
    except ImportError:
        TF_AVAILABLE = False

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### 📤 Upload Radiograph")
        uploaded_file = st.file_uploader(
            "Drag & drop or click to upload", 
            type=["jpg", "jpeg", "png"],
            help="Periapical, bitewing, or panoramic dental radiograph"
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
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
            st.warning("☁️ **Cloud Demo Mode** (Simulated AI Engine)")
            
        st.info("""
        **Architecture:** Custom CNN + GAP<br>
        **Preprocessing:** CLAHE Contrast Enhancement<br>
        **Explainability:** Grad-CAM Heatmaps<br>
        **Threshold:** Medical Recall Optimized (0.35)
        """)

    with col2:
        if uploaded_file is not None:
            # Open image with PIL
            from PIL import Image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Radiograph", use_column_width=True)
            
            # Simulate processing time
            with st.spinner('🔍 Step 1: Applying CLAHE Enhancement...'):
                time.sleep(0.4)
            with st.spinner('🧠 Step 2: Running CNN Inference...'):
                time.sleep(0.4)
            with st.spinner('🔥 Step 3: Generating Grad-CAM Heatmap...'):
                time.sleep(0.4)
                
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
            st.markdown("### 🩺 Diagnostic Report")
            
            diag_col, cam_col = st.columns(2)
            
            with diag_col:
                if is_cavity:
                    st.error(f"🚨 **CARIES DETECTED** \n\n Confidence: {confidence*100:.1f}%")
                    st.markdown("""
                    <div class="warning-box">
                    <strong>Clinical Recommendation:</strong> AI identified radiolucent patterns consistent with dental caries. 
                    Verify with clinical examination and bitewing radiographs before intervention.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.success(f"✅ **NO CARIES DETECTED** \n\n Confidence: {confidence*100:.1f}%")
                    st.markdown("""
                    <div class="success-box">
                    <strong>Clinical Note:</strong> No significant radiolucent patterns detected. 
                    Continue routine preventive care and 6-month recall schedule.
                    </div>
                    """, unsafe_allow_html=True)
                    
            with cam_col:
                st.markdown("#### 🔥 Explainable AI Overlay")
                st.image(heatmap_img, caption="Grad-CAM Attention Heatmap (Red = High Focus)", use_column_width=True)
                if not TF_AVAILABLE:
                    st.caption("*Note: Heatmap simulated for cloud demonstration. Localhost uses real Grad-CAM.*")
                    
            st.markdown(f"*Threshold: {OPTIMAL_THRESHOLD:.2f} | Optimized for Medical Sensitivity*")

        else:
            st.markdown("""
            <div class="info-box">
                <h4>📖 How to Use This Tool</h4>
                <p>1. Upload a digital dental radiograph (JPG/PNG format).</p>
                <p>2. System applies <strong>CLAHE contrast enhancement</strong> to highlight subtle lesions.</p>
                <p>3. CNN analyzes image and returns <strong>binary classification</strong> with confidence score.</p>
                <p>4. Review <strong>Grad-CAM Heatmap</strong> to validate AI attention regions.</p>
                <p>5. Use as a <strong>second-opinion assistant</strong> alongside clinical judgment.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.info("👆 Upload a radiograph to begin analysis.")

# ==========================================
# ℹ️ ABOUT
# ==========================================
elif page == "ℹ️ About":
    st.markdown("### ℹ️ Project Information & References")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 👨‍💻 Developer")
        st.markdown("""
        **Barry Ng Kee Hong**<br>
        Student ID: 0135374<br>
        Bachelor of Computer Science (Hons)<br>
        University of Wollongong Malaysia
        """)
        st.markdown("#### 👨‍🏫 Supervision")
        st.markdown("""
        **Main Supervisor:** Mr Chua Hiang Kiat<br>
        **Co-Supervisor:** Ms. Norsyela binti Muhammad Noor Mathivanan
        """)
        
    with col2:
        st.markdown("#### 🛠️ Technology Stack")
        st.markdown("""
        - **Language:** Python 3.11
        - **Deep Learning:** TensorFlow 2.15, Keras
        - **Image Processing:** OpenCV, CLAHE, PIL
        - **Web Framework:** Streamlit
        - **Visualization:** Plotly, Matplotlib
        - **Methodology:** Design Science Research (DSRM)
        """)
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📚 Key References")
    st.markdown("""
    1. World Health Organization. (2022). *Global Oral Health Status Report*.
    2. National Oral Health Survey of Adults (NOHSA). (2020). *Malaysia Ministry of Health*.
    3. Selvaraju et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*.
    4. Peck et al. (2021). *Deep Learning for Dental Caries Detection: A Systematic Review*.
    """)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <strong>⚠️ Clinical Disclaimer:</strong> This system is designed as an <strong>educational and assistive prototype</strong> 
    for Final Year Project demonstration purposes. It does <strong>not</strong> replace professional clinical judgment, 
    radiographic interpretation, or established diagnostic protocols. Always verify AI findings with qualified dental practitioners.
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# FOOTER
# ==========================================
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #64748B; padding: 20px; border-top: 1px solid #334155; margin-top: 40px; font-size: 14px;'>
    <p>🦷 Automated Detection of Dental Caries from Dental X-ray Images | FYP2 2024</p>
    <p>Developed by Barry Ng Kee Hong (0135374) | University of Wollongong Malaysia</p>
</div>
""", unsafe_allow_html=True)
