import streamlit as st

#from streamlit_shap import st_shap
import pandas as pd
import joblib
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg



# Title
st.header("Prediction Tool for 3-Month Functional Outcomes in RSSI Patients")

# Input bar 1
# Input bar 2
Age = st.number_input("Age(years)")

Glucose = st.number_input("Glucose(mmol/L)")
SBP = st.number_input("SBP(mmHg)")
NLR = st.number_input("NLR(Neutrophil-to-Lymphocyte Ratio)")
SHR = st.number_input("SHR(stress-induced hyperglycemia ratio)")
LDL = st.number_input("LDL-C(mmol/L)")
NIHSS = st.number_input("NIHSS score on admission")

# Dropdown input
pRSSI = st.selectbox("pRSSI(proximal recent single subcortical infarct)", ("Yes", "No"))



if st.button("Submit"):
    # Unpickle classifier
    clf = joblib.load("model.pkl")
    # Store inputs into dataframe
    X = pd.DataFrame([[Age,Glucose,SBP,NLR,SHR,LDL,NIHSS,pRSSI]],
                     columns=["Age", "Glucose","SBP",
                       "NLR","SHR","LDL","NIHSS","pRSSI"])
    X = X.replace(["Yes", "No"], [1, 0])
    
    # Get prediction
    prediction = clf.predict(X)[0]

    #explainer = shap.TreeExplainer(clf)
    #shap_values = explainer.shap_values(X)
    # f = plt.figure()
    # shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:])
    # f.savefig("shap_force_plot.png", bbox_inches='tight', dpi=600)
    # Output prediction
    # P = mpimg.imread("shap_force_plot.png")
    # st.image(P, caption="shap_force_plot", channels="RGB")
    # st_shap(shap.plots.waterfall(shap_values[0]), height=300)
    # st_shap(shap.plots.beeswarm(shap_values), height=300)
    #st_shap(shap.force_plot(explainer.expected_value, shap_values[0, :],X.iloc[0, :]), height=200, width=700)
    if prediction == 0:
        st.text(f"This patient has a higher probability for 3-month good functional outcomes")
    else:

        st.text(f"This patient has a higher probability for 3-month poor functional outcomes")






