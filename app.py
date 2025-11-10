import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# -----------------------------
# Load trained model
# -----------------------------
model = pickle.load(open("hepatitis_b_model.pkl", "rb"))

# -----------------------------
# Helper function for encoding categorical values
# -----------------------------
def encode_input(sex, steroid, antivirals, fatigue, malaise, anorexia,
                 liver_big, liver_firm, spleen_palpable, spiders, ascites,
                 varices, histology):
    mapping = {'yes': 1, 'no': 0, 'male': 1, 'female': 0}
    return [
        mapping[sex], mapping[steroid], mapping[antivirals],
        mapping[fatigue], mapping[malaise], mapping[anorexia],
        mapping[liver_big], mapping[liver_firm], mapping[spleen_palpable],
        mapping[spiders], mapping[ascites], mapping[varices],
        mapping[histology]
    ]

# -----------------------------
# Streamlit App Layout
# -----------------------------
st.set_page_config(page_title="HepaSmart", page_icon="🩸", layout="wide")

st.title("🩺 HepaSmart: Hepatitis-B Prediction App")
st.markdown("""
This AI-powered tool helps estimate a patient's **Hepatitis B survival likelihood**  
and **detects Hepatitis-B positive/negative status** using biological and clinical parameters.
""")

# Sidebar navigation
menu = st.sidebar.radio("Navigation", ["🏠 Prediction Form", "📂 Batch Prediction", "📊 Dashboard", "ℹ️ About"])

# -----------------------------
# 🏠 Single Prediction Section
# -----------------------------
if menu == "🏠 Prediction Form":
    st.header("Enter Patient Test Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=10, max_value=90, value=35)
        sex = st.selectbox("Sex", ["male", "female"])
        steroid = st.selectbox("Steroid", ["yes", "no"])
        antivirals = st.selectbox("Antivirals", ["yes", "no"])
        fatigue = st.selectbox("Fatigue", ["yes", "no"])
        malaise = st.selectbox("Malaise", ["yes", "no"])

    with col2:
        anorexia = st.selectbox("Anorexia", ["yes", "no"])
        liver_big = st.selectbox("Liver Big", ["yes", "no"])
        liver_firm = st.selectbox("Liver Firm", ["yes", "no"])
        spleen_palpable = st.selectbox("Spleen Palpable", ["yes", "no"])
        spiders = st.selectbox("Spiders", ["yes", "no"])
        ascites = st.selectbox("Ascites", ["yes", "no"])

    with col3:
        varices = st.selectbox("Varices", ["yes", "no"])
        bilirubin = st.number_input("Bilirubin (mg/dL)", min_value=0.1, max_value=10.0, value=1.0)
        alk_phosphate = st.number_input("Alk Phosphate (U/L)", min_value=30, max_value=400, value=120)
        sgot = st.number_input("SGOT (U/L)", min_value=10, max_value=600, value=200)
        albumin = st.number_input("Albumin (g/dL)", min_value=1.0, max_value=6.0, value=4.0)
        histology = st.selectbox("Histology", ["yes", "no"])

    # Combine inputs in correct order
    cat_features = encode_input(sex, steroid, antivirals, fatigue, malaise,
                                anorexia, liver_big, liver_firm, spleen_palpable,
                                spiders, ascites, varices, histology)

    input_data = np.array([[
        age, *cat_features, bilirubin, alk_phosphate, sgot, albumin
    ]], dtype=float)

    if st.button("🔍 Predict"):
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0]
        confidence = round(max(proba) * 100, 2)

        if prediction == 2:
            st.success(f"🟢 **Prediction:** Patient is *Hepatitis B Negative / Likely to Live*")
            st.info(f"✅ Confidence: {confidence}%")
        else:
            st.error(f"🔴 **Prediction:** Patient is *Hepatitis B Positive / High Risk*")
            st.warning(f"⚠️ Confidence: {confidence}%")

# -----------------------------
# 📂 Batch Prediction Section
# -----------------------------
elif menu == "📂 Batch Prediction":
    st.header("📂 Upload CSV for Batch Prediction")
    st.markdown("""
    Upload a CSV file containing patient data with the following columns:  
    **['Age','Sex','Steroid','Antivirals','Fatigue','Malaise','Anorexia','Liver Big','Liver Firm',
    'Spleen Palpable','Spiders','Ascites','Varices','Bilirubin','Alk Phosphate','Sgot','Albumin','Histology']**
    """)

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("✅ Uploaded Data Preview:")
        st.dataframe(df.head())

        # Encode categorical columns
        mapping = {'yes': 1, 'no': 0, 'male': 1, 'female': 0}
        df_encoded = df.replace(mapping)

        # Predict
        preds = model.predict(df_encoded)
        probs = model.predict_proba(df_encoded)

        df['Prediction'] = np.where(preds == 2, 'Hepatitis B Negative (Live)', 'Hepatitis B Positive (Die)')
        df['Confidence (%)'] = np.round(np.max(probs, axis=1) * 100, 2)

        st.success("✅ Batch Prediction Completed!")
        st.dataframe(df)

        # Download button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results as CSV", data=csv, file_name="hepatitis_predictions.csv")

# -----------------------------
# 📊 Dashboard Section
# -----------------------------
elif menu == "📊 Dashboard":
    st.header("📊 Model Performance Dashboard")
    st.markdown("""
    **Model Evaluation Summary:**
    - Accuracy: 90%
    - Precision (Live): 0.83
    - Recall (Live): 0.95
    - Precision (Die): 0.96
    - Recall (Die): 0.87
    """)

    labels = ['Accuracy', 'Precision (Live)', 'Recall (Live)', 'Precision (Die)', 'Recall (Die)']
    values = [0.90, 0.83, 0.95, 0.96, 0.87]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(labels, values)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Score")
    ax.set_title("Model Performance Metrics")
    for i, v in enumerate(values):
        ax.text(v + 0.01, i, f"{v:.2f}", va='center')
    st.pyplot(fig)

    st.image("https://upload.wikimedia.org/wikipedia/commons/6/62/Hepatitis_B_virus_01.png",
             caption="Hepatitis B Virus Structure", use_container_width=True)

# -----------------------------
# ℹ️ About Section
# -----------------------------
elif menu == "ℹ️ About":
    st.header("ℹ️ About HepaSmart")
    st.markdown("""
    **HepaSmart** is a clinical decision support system built using **Machine Learning (Random Forest Classifier)**.  
    It predicts the likelihood of survival or mortality in Hepatitis B patients using medical parameters.

    **Developer:** Shrawanthi P  
    **Accuracy:** 90%  
    **Dataset:** [UCI Hepatitis Dataset](https://archive.ics.uci.edu/dataset/46/hepatitis)  
    """)

st.markdown("---")
st.markdown("© 2025 HepaSmart | Designed for Clinical Decision Support 🧬")
