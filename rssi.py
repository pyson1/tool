import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="RSSI Predictor", page_icon="🧠", layout="centered")

st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); }
.appbar {
  background:#065f46; color:#ecfdf5; padding:16px 20px;
  border-radius:14px; box-shadow:0 10px 28px rgba(6,95,70,.35);
  margin:14px 0 22px 0; font-weight:800;
}
.card {
  background:#ffffff; border:1px solid #e5e7eb; border-radius:16px;
  box-shadow:0 12px 28px rgba(16,24,40,.06); padding:18px;
}
div.stButton > button:first-child{
  width:100%; height:48px; border-radius:12px;
  background:#10b981; color:#fff; font-weight:800; font-size:16px; border:0;
}
.result-good { border-left:6px solid #10b981; padding-left:12px; font-weight:800; color:#065f46; }
.result-bad  { border-left:6px solid #ef4444; padding-left:12px; font-weight:800; color:#991b1b; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="appbar">🧠 RSSI Outcome Predictor</div>', unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1: Age = st.number_input("Age (years)", min_value=0.0, step=1.0, format="%.0f")
with c2: SBP = st.number_input("SBP (mmHg)", min_value=0.0, step=1.0, format="%.0f")
with c1: NIHSS = st.number_input("NIHSS score", min_value=0.0, step=1.0, format="%.0f")
with c2: Glucose = st.number_input("Glucose (mmol/L)", min_value=0.0, step=0.01)
with c1: LDL = st.number_input("LDL-C (mmol/L)", min_value=0.0, step=0.01)
with c2: SHR = st.number_input("SHR", min_value=0.0, step=0.01)
with c1: NLR = st.number_input("NLR", min_value=0.0, step=0.01)
with c2: pRSSI = st.selectbox("pRSSI", ("Yes", "No"))

if st.button("Submit"):
    clf = joblib.load("Model.pkl")
    X = pd.DataFrame([[Age,SBP,NIHSS,Glucose,LDL,SHR,NLR,pRSSI]],
                     columns=["Age","SBP","NIHSS","Glucose","LDL","SHR","NLR","pRSSI"]).replace(["Yes","No"],[1,0])
    pred = int(clf.predict(X)[0])
    if pred==0: st.markdown('<div class="card result-good">✅ Good 3-month outcome</div>', unsafe_allow_html=True)
    else: st.markdown('<div class="card result-bad">⚠️ Poor 3-month outcome</div>', unsafe_allow_html=True)
