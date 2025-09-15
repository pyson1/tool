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
""", unsafe_a_
