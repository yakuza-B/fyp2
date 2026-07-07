# ==========================================
# ℹ️ ABOUT
# ==========================================
elif page == "ℹ️ About":
    st.markdown("### ℹ️ Project Information & References")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 👨‍💻 Developer")
        st.markdown("""
        <div style='line-height: 2.2;'>
        <b>Name:</b> Barry Ng Kee Hong<br>
        <b>Student ID:</b> 0135374<br>
        <b>Programme:</b> Bachelor of Computer Science (Hons)<br>
        <b>Institution:</b> University of Wollongong Malaysia
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        st.markdown("#### 👨‍🏫 Supervision")
        st.markdown("""
        <div style='line-height: 2.2;'>
        <b>Main Supervisor:</b> Mr Chua Hiang Kiat<br>
        <b>Co-Supervisor:</b> Ms. Norsyela binti Muhammad Noor Mathivanan
        </div>
        """, unsafe_allow_html=True)
        
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
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        st.markdown("#### 📚 Key References")
        st.markdown("""
        1. World Health Organization. (2022). *Global Oral Health Status Report*.
        2. National Oral Health Survey of Adults (NOHSA). (2020). *Malaysia Ministry of Health*.
        3. Selvaraju et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks*.
        4. Peck et al. (2021). *Deep Learning for Dental Caries Detection: A Systematic Review*.
        """)
