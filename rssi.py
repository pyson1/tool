import streamlit as st
import pandas as pd
import joblib

# ---------------- Page ----------------
st.set_page_config(
    page_title="RSSI Outcome Predictor",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------- Styles (green theme, no blue) ----------------
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%); }
.main-wrap { max-width: 920px; margin: 0 auto; }

/* Header bar (dark-green) */
.appbar {
  background: #065f46; color: #ecfdf5; padding: 16px 20px;
  border-radius: 14px; box-shadow: 0 10px 28px rgba(6,95,70,.35);
  margin: 14px 0 22px 0; display:flex; align-items:center; gap:12px;
}
.app-title { font-size: 20px; font-weight: 800; letter-spacing:.2px; }
.app-sub { opacity:.95; font-size: 13px; }

/* Card */
.card {
  background:#ffffff; border:1px solid #e5e7eb; border-radius:16px;
  box-shadow: 0 12px 28px rgba(16,24,40,.06); padding:18px;
}
.card-title { font-weight: 700; font-size: 16px; color:#1f2937; margin-bottom:6px; }
.help { color:#6b7280; font-size: 12.5px; margin: -4px 0 10px 0; }

/* Button (emerald) */
div.stButton > button:first-child{
  width:100%; height:48px; border-radius:12px;
  background:#10b981; color:#fff; border:0;
  font-weight:800; font-size:16px;
  box-shadow: 0 8px 20px rgba(16,185,129,.35);
  transition: all .15s ease;
}
div.stButton > button:first-child:hover{ filter:brightness(.95); transform: translateY(-1px); }
div.stButton > button:first-child:active{ transform: translateY(0); }

/* Result styles */
.result-good { border-left:6px solid #10b981; padding-left:12px; }
.result-bad  { border-left:6px solid #ef4444; padding-left:12px; }
.result-title{ font-size:16px; font-weight:800; margin-bottom:6px;}
.result-body { font-size:14px; color:#374151; }

/* Slightly larger labels */
label { font-size: 15px !important; }
</style>
""", unsafe_allow_html=True)  # ← 这里必须完整

# ---------------- Header ----------------
st.markdown('<div class="main-wrap">', unsafe_allow_html=True)
st.markdown("""
<div class="appbar">
  <span style="font-size:22px">🧠</span>
  <div>
    <div class="app-title">Prediction Tool for 3-Month Functional Outcomes in RSSI Patients</div>
    <div class="app-sub">Fill the inputs below and click <b>Submit</b> at the bottom.</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------- Form (two columns) ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="card-title">Patient Inputs</div>', unsafe_allow_html=True)
st.markdown('<div class="help">Note: feature order must match training.</div>', unsafe_allow_html=True)

# 特征顺序需与训练一致：Age, SBP, NIHSS, Glucose, LDL, SHR, NLR, pRSSI
c1, c2 = st.columns(2)
with c1:
    Age = st.number_input("Age (years)", min_value=0.0, step=1.0, format="%.0f")
with c2:
    SBP = st.number_input("SBP (mmHg)", min_value=0.0, step=1.0, format="%.0f")

with c1:
    NIHSS = st.number_input("NIHSS score on admission", min_value=0.0, step=1.0, format="%.0f")
with c2:
    Glucose = st.number_input("Glucose (mmol/L)", min_value=0.0, step=0.01)

with c1:
    LDL = st.number_input("LDL-C (mmol/L)", min_value=0.0, step=0.01)
with c2:
    SHR = st.number_input("SHR (stress-induced hyperglycemia ratio)", min_value=0.0, step=0.01)

with c1:
    NLR = st.number_input("NLR (Neutrophil-to-Lymphocyte Ratio)", min_value=0.0, step=0.01)
with c2:
    pRSSI = st.selectbox("pRSSI (proximal recent single subcortical infarct)", ("Yes", "No"))

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Submit at bottom ----------------
st.markdown('<br>', unsafe_allow_html=True)
submitted = st.button("Submit")

# ---------------- Result at bottom ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="card-title">Prediction</div>', unsafe_allow_html=True)

if submitted:
    try:
        clf = joblib.load("Model.pkl")  # 你的已训练模型/管道
        X = pd.DataFrame(
            [[Age, SBP, NIHSS, Glucose, LDL, SHR, NLR, pRSSI]],
            columns=["Age","SBP","NIHSS","Glucose","LDL","SHR","NLR","pRSSI"]
        ).replace(["Yes","No"], [1,0])

        with st.spinner("Running inference..."):
            pred = int(clf.predict(X)[0])
            prob = None
            if hasattr(clf, "predict_proba"):
                try:
                    prob = float(clf.predict_proba(X)[0,1])
                except Exception:
                    prob = None

        if pred == 0:
            st.markdown('<div class="result-good">', unsafe_allow_html=True)
            st.markdown('<div class="result-title">✅ Higher probability of <u>GOOD</u> 3-month functional outcome</div>', unsafe_allow_html=True)
            if prob is not None:
                st.markdown(f'<div class="result-body">Model confidence (poor outcome probability): <b>{prob:.2f}</b></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-bad">', unsafe_allow_html=True)
            st.markdown('<div class="result-title">⚠️ Higher probability of <u>POOR</u> 3-month functional outcome</div>', unsafe_allow_html=True)
            if prob is not None:
                st.markdown(f'<div class="result-body">Model confidence (poor outcome probability): <b>{prob:.2f}</b></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error("Failed to load `Model.pkl` or run prediction. Ensure the file exists and the feature order matches training.")
        st.caption(str(e))
else:
    st.info("Set inputs above and click **Submit**.")

st.markdown('</div>', unsafe_allow_html=True)  # 关闭 .main-wrap
