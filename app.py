import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import os
import time
import random

# ==========================================
# PAGE CONFIG & CUSTOM CSS
# ==========================================
st.set_page_config(page_title="AI Dental Caries Detection", page_icon="🦷", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #004d99; text-align: center; }
    .sub-header { font-size: 1.2rem; color: #666; text-align: center; margin-bottom: 2rem; }
    .info-box { background-color: #e6f2ff; border-left: 5px solid #004d99; padding: 1rem; margin: 1rem 0; border-radius: 5px; }
    .success-box { background-color: #e6ffe6; border-left: 5px solid #009900; padding: 1rem; margin: 1rem 0; border-radius: 5px; }
    .warning-box { background-color: #ffe6e6; border-left: 5px solid #cc0000; padding: 1rem; margin: 1rem 0; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# MODEL LOADING
# ==========================================
@st.cache_resource
def load_model():
    model_path = "best_caries_model.keras"
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        return None

# ==========================================
# STREAMLIT UI LAYOUT
# ==========================================
st.markdown('<p class="main-header">🦷 AI-Assisted Dental Caries Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Empowering Dentists with Explainable Deep Learning</p>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 📤 Upload Radiograph")
    uploaded_file = st.file_uploader("Choose an X-ray image (JPG, PNG)...", type=["jpg", "jpeg", "png"])
    
    st.markdown("---")
    st.markdown("### ⚙️ System Info")
    
    model = load_model()
    if model is None:
        st.warning("⚠️ **Demo Mode Active**\n\nModel file not found in repository (due to size limits). Running in simulated demo mode.")
    else:
        st.success("✅ Model Loaded Successfully")
        
    st.info("""
    **Model:** Custom CNN with Transfer Learning  \n
    **Preprocessing:** CLAHE Contrast Enhancement  \n
    **Explainability:** Grad-CAM Heatmaps  \n
    **Threshold:** Optimized for Medical Recall
    """)

with col2:
    if uploaded_file is not None:
        # 1. Load Image
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
            
        st.image(gray, caption="Original Grayscale X-Ray", use_column_width=True)
        
        # 2. Preprocess exactly like training
        resized = cv2.resize(gray, (224, 224))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(resized)
        normalized = enhanced / 255.0
        input_data = np.expand_dims(normalized, axis=(0, -1)) # Shape: (1, 224, 224, 1)
        
        # 3. Predict
        if model is not None:
            with st.spinner('Analyzing radiograph...'):
                prediction_prob = model.predict(input_data)[0][0]
            
            OPTIMAL_THRESHOLD = 0.35 
            is_cavity = prediction_prob >= OPTIMAL_THRESHOLD
            confidence = prediction_prob if is_cavity else (1 - prediction_prob)
        else:
            # DEMO MODE: Simulate prediction
            with st.spinner('Analyzing radiograph (Demo Mode)...'):
                time.sleep(1.5) # Simulate processing time
                prediction_prob = random.uniform(0.3, 0.9)
                OPTIMAL_THRESHOLD = 0.35
                is_cavity = prediction_prob >= OPTIMAL_THRESHOLD
                confidence = prediction_prob if is_cavity else (1 - prediction_prob)
        
        # 4. Display Results
        st.markdown("---")
        st.markdown("### 🩺 AI Diagnosis & Explainability")
        
        diag_col, cam_col = st.columns(2)
        
        with diag_col:
            if is_cavity:
                st.error(f"🚨 **CARIES DETECTED** \n\n Confidence: {confidence*100:.1f}%")
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
            
            if model is not None:
                # Real Grad-CAM
                def generate_gradcam(model, image_array):
                    last_conv_layer = None
                    for layer in reversed(model.layers):
                        if isinstance(layer, tf.keras.layers.Conv2D):
                            last_conv_layer = layer.name
                            break
                    if last_conv_layer is None: return None
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
                    st.image(superimposed, caption="AI Attention Overlay (Red = High Focus)", use_column_width=True)
                else:
                    st.warning("Could not generate heatmap.")
            else:
                # DEMO MODE: Simulate Grad-CAM
                fake_heatmap = np.zeros((224, 224), dtype=np.float32)
                cv2.circle(fake_heatmap, (112 + np.random.randint(-30, 30), 112 + np.random.randint(-30, 30)), 40, 1.0, -1)
                fake_heatmap = cv2.GaussianBlur(fake_heatmap, (31, 31), 0)
                fake_heatmap = fake_heatmap / fake_heatmap.max()
                
                heatmap_colored = cv2.applyColorMap(np.uint8(255 * fake_heatmap), cv2.COLORMAP_JET)
                original_rgb = cv2.cvtColor((normalized.squeeze() * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
                superimposed = cv2.addWeighted(original_rgb, 0.6, heatmap_colored, 0.4, 0)
                
                st.image(superimposed, caption="AI Attention Overlay (Demo Mode)", use_column_width=True)
                st.caption("*Note: Heatmap is simulated for demonstration purposes.*")
                
        st.markdown(f"*Medical Threshold set to {OPTIMAL_THRESHOLD:.2f} to prioritize Sensitivity (Recall).*")

    else:
        st.markdown("""
        <div class="info-box">
            <h4>📖 Why Early Detection Matters</h4>
            <p>Dental caries affects <strong>2.3 billion people worldwide</strong>. Early-stage lesions present subtle 
            radiographic changes that are easily missed during manual examination due to fatigue or overlapping anatomy.</p>
            <p>This AI system serves as a <strong>second-opinion tool</strong>, utilizing Convolutional Neural Networks (CNN) 
            and Explainable AI (Grad-CAM) to highlight suspicious regions, helping dental professionals make faster, 
            more consistent diagnostic decisions.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("👈 Please upload a dental radiograph in the left sidebar to begin analysis.")
