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
    2. **Develop** an AI-driven system utilizing Transfer Learning (CNNs
