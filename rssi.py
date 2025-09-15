import streamlit as st
import pandas as pd
import joblib

# ---------------- Page / Theme ----------------
st.set_page_config(
    page_title="RSSI Outcome Predictor",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------- Global Styles ----------------
st.markdown("""
<style>
/* 背景渐变 */
.stApp {
  background: linear-gradient(135deg, #f7fafc 0%, #eef2f7 100%);
}
/* 顶部 App Bar */
.appbar {
  position: sticky; top: 0; z-index: 999;
  background: #0ea5e9; /* sky-500 */
  color: white; padding: 18px 22px; border-radius: 14px;
  box-shadow: 0 8px 24px rgba(14,165,233,.35);
  display:flex; align-items:center; gap:12px;
  margin-bottom: 22px;
}
.app-title { font-size: 20px; font-weight: 700; letter-spacing:.2px; }
.app-sub { opacity:.95; font-size: 13px; }

/* 容器宽度 */
.main-container { max-width: 920px; margin: 0 auto; }

/* 卡片 */
.card {
  background: #ffffff;
  border: 1px solid #e6ebf2;
  border-radius: 16px;
  padding: 18px 18px 12px 18px;
  box-shadow: 0 12px 28px rgba(16,24,40,.06);
}
.card-title { font-weight: 700; font-size: 16px; color:#1f2937; margin-bottom:6px; }
.help { color:#6b7280; font-size: 12.5px; margin: -4px 0 10px 0; }

/* 表单内间距 */
.block-container { padding-top: 14px; }

/* 主按钮 */
div.stButton > button:first-child{
  width:100%; height:48px; border-radius:12px;
  background:#0ea5e9; color:white; border:0;
  font-weight:700; font-size:16px;
  box-shadow: 0 8px 20px rgba(14,165,233,.35);
  transition: all .15s ease;
}
div.stButton > button:first-child:hover{
  filter:brightness(.95); transform: translateY(-1px);
}
div.stButton > button:first-child:active{ transform: translateY(0); }

/* 结果卡片 */
.result-good { border-left:6px solid #10b981; }
.result-bad  { border-left:6px solid #ef4444; }
.result-title{ font-size:16px; font-weight:800; margin-bottom:6px;}
.result-body { font-size:14px; color:#374151; }

/* 小徽章 */
.badge {
  display:inline-block; padding:4px 10px; border-radius:999px;
  background:#e0f2fe; color:#0369a1; font-size:12px; font-weight:700;
}

/* 标注输入标签字体稍大 */
label.css-16huue1, label.css-1p2iens, label.css-1dp5vir {
  font-size: 15px !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- App Bar ----------------
st.markdown(
    """
    <div class="main-container">
      <div class="appbar">
        <span style="font-size:22px">🧠</span>
        <div>
          <div class="app-title">Prediction Tool for 3-Month Functional Outcomes in RSSI Patients</div>
          <div class="app-sub">Enter clinical features on the left, then click <b>Submit</b> to get the prediction.</div>
        </div>
        <div style="margin-left:auto"><span class="badge">v1.0</span></div>
      </div>
    </div>
    """, unsafe_allow_html=True
)

# ---------------- Layout ----------------
st.markdown('<div class="main-container">', unsafe_allow_html=True)
col_left, col_right = st.columns([1.05, 0.95], gap="large")

with col_left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Patient Inputs</div>', unsafe_allow_html=True)
    st.markdown('<div class="help">Fill in the fields below and press Submit.</div>', unsafe_allow_html=True)

    # —— 注意：列顺序与训练保持一致 —— 
    # 你之前训练时的顺序是：["Age","SBP","NIHSS","Glucose","LDL","SHR","NLR","pRSSI"]
    Age = st.number_input("Age (years)", min_value=0.0, step=1.0, format="%.0f")
    SBP = st.number_input("SBP (mmHg)", min_value=0.0, step=1.0, format="%.0f")
    NIHSS = st.number_input("NIHSS score on admission", min_value=0.0, step=1.0, format="%.0f")
    Glucose = st.number_input("Glucose (mmol/L)", min_value=0.0, step=0.01)
    LDL = st.number_input("LDL-C (mmol/L)", min_value=0.0, step=0.01)
    SHR = st.number_input("SHR (stress-induced hyperglycemia ratio)", min_value=0.0, step=0.01)
    NLR = st.number_input("NLR (Neutrophil-to-Lymphocyte Ratio)", min_value=0.0, step=0.01)
    pRSSI = st.selectbox("pRSSI (proximal recent single subcortical infarct)", ("Yes", "No"))

    submitted = st.button("Submit")
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Prediction</div>', unsafe_allow_html=True)

    if submitted:
        try:
            clf = joblib.load("Model.pkl")  # 你的已训练模型（Pipeline 或 Estimator）

            X = pd.DataFrame(
                [[Age, SBP, NIHSS, Glucose, LDL, SHR, NLR, pRSSI]],
                columns=["Age","SBP","NIHSS","Glucose","LDL","SHR","NLR","pRSSI"]
            ).replace(["Yes","No"], [1,0])

            with st.spinner("Running inference..."):
                pred = int(clf.predict(X)[0])
                # 若模型支持概率，可选显示（不做评估，仅展示推理置信度）
                prob = None
                if hasattr(clf, "predict_proba"):
                    try:
                        prob = float(clf.predict_proba(X)[0,1])
                    except Exception:
                        prob = None

            if pred == 0:
                st.markdown('<div class="result-good card">', unsafe_allow_html=True)
                st.markdown('<div class="result-title">✅ Higher probability of <u>GOOD</u> 3-month functional outcome</div>', unsafe_allow_html=True)
                if prob is not None:
                    st.markdown(f'<div class="result-body">Model confidence (poor outcome probability): <b>{prob:.2f}</b></div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-bad card">', unsafe_allow_html=True)
                st.markdown('<div class="result-title">⚠️ Higher probability of <u>POOR</u> 3-month functional outcome</div>', unsafe_allow_html=True)
                if prob is not None:
                    st.markdown(f'<div class="result-body">Model confidence (poor outcome probability): <b>{prob:.2f}</b></div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error("Failed to load model or run prediction. Please ensure `Model.pkl` exists and the feature order matches training.")
            st.caption(f"Details: {e}")

    else:
        st.info("Fill the form on the left and click **Submit** to see prediction.")

    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("<br><div style='text-align:center; color:#94a3b8; font-size:12px;'>© RSSI Outcome Predictor • For clinical decision support only</div>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

