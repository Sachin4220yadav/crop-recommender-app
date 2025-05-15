import streamlit as st
import pickle
import numpy as np

st.title("🌾 Smart Crop Advisor")

with open('best_crop_model.pkl', 'rb') as f:
    model = pickle.load(f)

N = st.number_input('Nitrogen (N)', min_value=0, max_value=140, value=90)
P = st.number_input('Phosphorus (P)', min_value=0, max_value=140, value=40)
K = st.number_input('Potassium (K)', min_value=0, max_value=200, value=43)
temperature = st.number_input('Temperature (°C)', min_value=0.0, max_value=50.0, value=20.5)
humidity = st.number_input('Humidity (%)', min_value=0.0, max_value=100.0, value=80.0)
ph = st.number_input('pH Value', min_value=0.0, max_value=14.0, value=6.5)
rainfall = st.number_input('Rainfall (mm)', min_value=0.0, max_value=500.0, value=200.0)

if st.button("Recommend Crops"):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    probs = model.predict_proba(input_data)[0]
    top3 = np.argsort(probs)[-3:][::-1]
    st.write("### Recommended Crops:")
    for i in top3:
        st.write(f"✅ {model.classes_[i]} ({probs[i]*100:.2f}%)")
