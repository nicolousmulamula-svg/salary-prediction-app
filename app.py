import streamlit as st
import pickle
import numpy as np

st.title("💰 AI Salary Prediction App")
st.write("Predict salary based on experience and education level")

# --- LOAD MODEL SAFELY ---
try:
    model = pickle.load(open("model.pkl", "rb"))
    encoder = pickle.load(open("encoder.pkl", "rb"))
except:
    st.error("Model or encoder file not found. Make sure model.pkl and encoder.pkl are in this folder.")
    st.stop()

# --- USER INPUT ---
experience = st.number_input(
    "Years of Experience",
    min_value=0,
    max_value=40,
    value=1
)

education = st.selectbox(
    "Education Level",
    ["Diploma", "Bachelor", "Master", "PhD"]
)

# --- PREDICTION ---
if st.button("Predict Salary"):

    try:
        # encode education
        education_encoded = encoder.transform([education])

        # prediction
        prediction = model.predict(
            [[experience, education_encoded[0]]]
        )

        st.success(f"💵 Estimated Salary: {prediction[0]:,.0f} TZS")

    except Exception as e:
        st.error(f"Prediction error: {e}")
