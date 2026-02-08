import streamlit as st
import os
import torch
import pandas as pd
from transformers import pipeline
import time
#streamlit run app.py
# --- Page Configuration ---
st.set_page_config(
    page_title="Sentiment AI Analysis",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS for Premium Look ---
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }
    
    /* Header styling */
    .main-header {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 3rem;
        background: linear-gradient(to right, #60a5fa, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    /* Card styling */
    .stTextArea textarea {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        color: #f8fafc !important;
        border-radius: 12px !important;
        padding: 15px !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease;
    }
    .stTextArea textarea:focus {
        border-color: #60a5fa !important;
        box-shadow: 0 0 0 2px rgba(96, 165, 250, 0.2) !important;
    }
    
    /* Result card */
    .result-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
    }
    
    /* Sentiment Badges */
    .badge-positive {
        background-color: #065f46;
        color: #34d399;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.875rem;
    }
    .badge-negative {
        background-color: #7f1d1d;
        color: #f87171;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.875rem;
    }
    .badge-neutral {
        background-color: #4338ca;
        color: #a5b4fc;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.875rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #0f172a !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
    st.title("Sentiment AI")
    st.markdown("---")
    st.markdown("### Model Info")
    st.write("**Model:** DistilBERT")
    st.write("**Classes:** Positive, Neutral, Negative")
    st.markdown("---")
    
    if "history" not in st.session_state:
        st.session_state.history = []
    
    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()

# --- Model Loading (Cached) ---
@st.cache_resource
def load_sentiment_pipeline():
    model_path = "./saved_model"
    if not os.path.exists(model_path):
        # Fallback for when running from a different CWD
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "saved_model"))
    
    try:
        return pipeline(
            "text-classification",
            model=model_path,
            tokenizer=model_path,
            device=0 if torch.cuda.is_available() else -1
        )
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# --- Main Interface ---
st.markdown('<h1 class="main-header">Sentiment Analysis 🎭</h1>', unsafe_allow_html=True)
st.markdown("Experience high-precision sentiment classification powered by DistilBERT.")

# Input section
text_input = st.text_area("What's on your mind?", placeholder="Type or paste text here to analyze its sentiment...", height=150)

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    analyze_btn = st.button("✨ Analyze Sentiment", use_container_width=True)

if analyze_btn:
    if not text_input.strip():
        st.warning("⚠️ Please enter some text first!")
    else:
        nlp = load_sentiment_pipeline()
        if nlp:
            with st.spinner("🧠 Analyzing emotions..."):
                # Artificial slight delay removed for better performance
                # time.sleep(0.5)
                
                result = nlp(text_input)[0]
                label = result['label']
                score = result['score']
                
                # Add to history
                st.session_state.history.insert(0, {"text": text_input[:50] + "...", "label": label, "score": score})
                
                # Display Result
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                
                # Top row with Label and Confidence
                res_col1, res_col2 = st.columns([2, 1])
                
                with res_col1:
                    emoji = "🟢" if label == "positive" else "🔴" if label == "negative" else "🟡"
                    badge_class = f"badge-{label}"
                    st.markdown(f"### Overall Sentiment: {emoji} {label.upper()}")
                    st.markdown(f'<span class="{badge_class}">{label.capitalize()}</span>', unsafe_allow_html=True)
                
                with res_col2:
                    st.metric("Confidence Score", f"{score:.2%}")
                
                # Confidence Progress bar
                st.write("")
                st.progress(score, text="Model Confidence")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Visualization of probabilities (if available from model)
                # Note: Default pipeline only gives top class. For full probabilities we'd need top_k=None
                # but let's keep it simple and premium.
                
                if label == "positive":
                    st.balloons()

# --- History Section ---
if st.session_state.history:
    st.markdown("---")
    st.subheader("Recent Analyses")
    history_df = pd.DataFrame(st.session_state.history)
    st.table(history_df)

# --- Footer ---
st.markdown("""
<div style="text-align: center; margin-top: 50px; opacity: 0.5; font-size: 0.8rem;">
    Powered by Hugging Face Transformers & Streamlit | Project Final Version
</div>
""", unsafe_allow_html=True)
