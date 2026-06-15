import streamlit as st
import torch
import numpy as np
from typing import Dict, Tuple
from ensemble import load_models, ensemble_predict

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide"
)

st.markdown("""
    <div style="text-align:center; padding: 25px 10px;">
        <h1 style="font-size: 42px; color: #1f77b4;">📰 Fake News Detection System</h1>
        <p style="font-size:18px; color: #555;">
            AI-powered classification using advanced Transformer models.<br>
            Choose a single model or enable Ensemble Mode for the most reliable predictions.
        </p>
    </div>
""", unsafe_allow_html=True)

@st.cache_resource
def _load_all():
    try:
        return load_models(["bert", "roberta", "distilbert"])
    except Exception as e:
        return e

with st.spinner("🔄 Initializing system and downloading cloud transformer architectures..."):
    _models_or_error = _load_all()

if isinstance(_models_or_error, Exception):
    st.error("Model loading failed. Please verify your internet connection or Hugging Face repository accessibility.")
    st.exception(_models_or_error)
    st.stop()

models: Dict[str, Tuple] = _models_or_error

left, right = st.columns([2, 1])

with right:
    st.markdown("### ⚙️ Settings")
    mode = st.radio("Detection Mode", ["Ensemble (Recommended)", "Single Model"])
    if mode.startswith("Ensemble"):
        strategy = st.selectbox("Ensemble Strategy", ["majority", "avg_proba"])
    else:
        strategy = None
        
    st.markdown("### 📦 Active Models Registry:")
    for mn in models.keys():
        st.write(f"- **{mn.upper()}**")

with left:
    st.markdown("### 📝 Enter News Text")
    text = st.text_area(
        "",
        height=220,
        placeholder="Paste the news article body or headline header here..."
    )

def predict_with_model(text_input: str, model_name: str, models_dict: Dict):
    tokenizer, model, device = models_dict[model_name]
    encoded = tokenizer(text_input, truncation=True, padding=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**encoded)
        logits = out.logits.cpu().numpy()[0]
    
    # Stable Softmax conversion
    probs = (np.exp(logits - logits.max()) / np.exp(logits - logits.max()).sum()).tolist()
    pred = int(np.argmax(logits))
    return {"pred": pred, "probs": probs}

if st.button("Analyze ▶️"):
    if not text.strip():
        st.warning("Please enter news text before analyzing.")
    else:
        if mode.startswith("Ensemble"):
            try:
                result = ensemble_predict(text, models, strategy=strategy)
            except Exception as e:
                st.error("Error during ensemble prediction pipeline processing.")
                st.exception(e)
                st.stop()
                
            # Render predictions per model clearly
            st.markdown("### 🔍 Model-wise Breakdowns")
            cols = st.columns(len(result["per_model"]))
            for col, (mname, info) in zip(cols, result["per_model"].items()):
                pred_label = "Fake News" if info["pred"] == 1 else "Real News"
                col.markdown(f"**{mname.upper()}**")
                col.markdown(f"- Inference: **{pred_label}**")
                col.markdown(f"- Confidences (Real vs Fake): `{[round(p, 4) for p in info['probs']]}`")
                
            epred = result["ensemble"]["pred"]
            econf = result["ensemble"]["confidence"]
            method = result["ensemble"]["method"]
            label_txt = "🔴 FAKE NEWS DETECTED" if epred == 1 else "🟢 VERIFIED REAL NEWS"
            
            st.markdown("---")
            st.markdown(f"### 🏁 Final Ensemble Decision ({method.replace('_', ' ').title()})")
            st.markdown(
                f"<div style='padding:20px; font-size:24px; font-weight:bold; "
                f"border-radius:10px; background:#1f77b4; color:white; text-align:center;'> "
                f"{label_txt}<br><span style='font-size:16px; font-weight:normal;'>Calculated System Confidence: {econf*100:.2f}%</span>"
                f"</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown("### 🔍 Single Model Output")
            selected = st.selectbox("Select Target Model Evaluator", list(models.keys()))
            with st.spinner(f"Running inference sequence through {selected.upper()}..."):
                info = predict_with_model(text, selected, models)
            pred_label = "Fake News" if info["pred"] == 1 else "Real News"
            st.write(f"- Selected Architecture: **{selected.upper()}**")
            st.write(f"- Prediction Decision: **{pred_label}**")
            st.write(f"- Logits Distribution Matrix: `{info['probs']}`")

st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; color:gray; padding:10px;">
        Fake News Detection System • Powered by Transformer Models & Ensemble Learning
    </div>
    """,
    unsafe_allow_html=True
)