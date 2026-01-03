import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="Wine Quality Prediction",
    page_icon="🍷",
    layout="centered"
)

st.title("🍷 Wine Quality Prediction App")
st.markdown("Predict the **quality of red wine** using Machine Learning")

# =========================
# Load Model & Scaler
# =========================
@st.cache_resource
def load_model():
    model = pickle.load(open("finalized_model.sav", "rb"))
    return model

@st.cache_resource
def load_scaler():
    df = pd.read_csv("winequality-red.csv")
    X = df.drop("quality", axis=1)
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler

model = load_model()
scaler = load_scaler()

# =========================
# Sidebar Inputs
# =========================
st.sidebar.header("🍾 Input Wine Parameters")

fixed_acidity = st.sidebar.slider("Fixed Acidity", 4.0, 16.0, 7.4)
volatile_acidity = st.sidebar.slider("Volatile Acidity", 0.1, 1.6, 0.7)
citric_acid = st.sidebar.slider("Citric Acid", 0.0, 1.0, 0.0)
residual_sugar = st.sidebar.slider("Residual Sugar (log)", -2.0, 2.0, 0.64)
chlorides = st.sidebar.slider("Chlorides (log)", -4.0, 0.0, -1.0)
free_sulfur_dioxide = st.sidebar.slider("Free Sulfur Dioxide (log)", 0.0, 5.0, 1.0)
total_sulfur_dioxide = st.sidebar.slider("Total Sulfur Dioxide (log)", 0.0, 6.0, 3.0)
density = st.sidebar.slider("Density", 0.990, 1.005, 0.996)
pH = st.sidebar.slider("pH", 2.5, 4.5, 3.3)
sulphates = st.sidebar.slider("Sulphates (log)", -2.0, 2.0, 0.7)
alcohol = st.sidebar.slider("Alcohol", 8.0, 15.0, 10.5)

# =========================
# Input DataFrame
# =========================
input_data = pd.DataFrame([[
    fixed_acidity,
    volatile_acidity,
    citric_acid,
    residual_sugar,
    chlorides,
    free_sulfur_dioxide,
    total_sulfur_dioxide,
    density,
    pH,
    sulphates,
    alcohol
]], columns=[
    'fixed acidity', 'volatile acidity', 'citric acid',
    'residual sugar', 'chlorides', 'free sulfur dioxide',
    'total sulfur dioxide', 'density', 'pH',
    'sulphates', 'alcohol'
])

# =========================
# Prediction
# =========================
if st.button("🔮 Predict Wine Quality"):
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]

    st.subheader("📊 Prediction Result")

    if prediction <= 4:
        st.error(f"🍷 Predicted Quality: {prediction} (Poor)")
    elif prediction <= 6:
        st.warning(f"🍷 Predicted Quality: {prediction} (Average)")
    else:
        st.success(f"🍷 Predicted Quality: {prediction} (Good)")

# =========================
# Footer
# =========================
st.markdown("---")
st.markdown("💡 **Machine Learning Project | Random Forest Classifier**")
