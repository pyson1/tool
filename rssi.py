import streamlit as st
import pandas as pd
import joblib

# ============ 样式美化 ============
st.markdown("""
    <style>
    /* 修改整体字体 */
    html, body, [class*="css"]  {
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 16px;
    }

    /* 标题样式 */
    h1, h2, h3 {
        font-weight: 600;
        color: #2C3E50;
    }

    /* 输入框字体 */
    .stNumberInput, .stSelectbox {
        font-size: 15px !important;
    }

    /* 按钮美化 */
    div.stButton > button:first-child {
        background-color: #3498DB;
        color: white;
        font-size: 16px;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        border: none;
        transition: 0.3s;
    }
    div.stButton > button:first-child:hover {
        background-color: #2980B9;
        color: #ECF0F1;
    }

    /* 预测结果样式 */
    .good {
        color: green;
        font-weight: bold;
        font-size: 18px;
    }
    .bad {
        color: red;
        font-weight: bold;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

# ============ 页面内容 ============
st.header("🧠 Prediction Tool for 3-Month Functional Outcomes in RSSI Patients")

# 输入项
Age = st.number_input("Age(years)")
Glucose = st.number_input("Glucose(mmol/L)")
SBP = st.number_input("SBP(mmHg)")
NLR = st.number_input("NLR(Neutrophil-to-Lymphocyte Ratio)")
SHR = st.number_input("SHR(stress-induced hyperglycemia ratio)")
LDL = st.number_input("LDL-C(mmol/L)")
NIHSS = st.number_input("NIHSS score on admission")
pRSSI = st.selectbox("pRSSI(proximal recent single subcortical infarct)", ("Yes", "No"))

if st.button("Submit"):
    # 加载模型
    clf = joblib.load("Model.pkl")

    # 整理输入
    X = pd.DataFrame([[Age, SBP, NIHSS, Glucose, LDL, SHR, NLR, pRSSI]],
                     columns=["Age","SBP","NIHSS","Glucose","LDL","SHR","NLR","pRSSI"])
    X = X.replace(["Yes", "No"], [1, 0])
    
    # 预测
    prediction = clf.predict(X)[0]

    if prediction == 0:
        st.markdown('<p class="good">✅ This patient has a higher probability for 3-month good functional outcomes</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="bad">⚠️ This patient has a higher probability for 3-month poor functional outcomes</p>', unsafe_allow_html=True)
