import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.title("🔧 Prediksi Kampas Rem")
model_data = pickle.load(open("modelling/models/model_rem.pkl", "rb"))
model, poly, features = model_data["model"], model_data["poly"], model_data["features"]

st.metric("MAE Model", f"{model_data['mae']:.4f} mm")
st.metric("RMSE Model", f"{model_data['rmse']:.4f} mm")

st.subheader("Simulasi Prediksi")
hard_brake     = st.slider("Hard Brake (normalized)", 0.0, 1.0, 0.5)
jarak_km       = st.slider("Jarak KM (normalized)", 0.0, 1.0, 0.5)
overspeed      = st.slider("Overspeed (normalized)", 0.0, 1.0, 0.3)
hard_brake_r7  = st.slider("Hard Brake Roll7 (normalized)", 0.0, 1.0, 0.5)
cumulative_km  = st.slider("Cumulative KM (normalized)", 0.0, 1.0, 0.4)
hard_brake_std = st.slider("Hard Brake Std7 (normalized)", 0.0, 1.0, 0.2)

if st.button("Prediksi"):
    X = np.array([[hard_brake, jarak_km, overspeed,
                   hard_brake_r7, cumulative_km, hard_brake_std]])
    pred = model.predict(poly.transform(X))[0]
    st.success(f"Estimasi tebal kampas rem: **{pred:.3f} mm**")
    if pred < 7:
        st.error("⚠️ Segera ganti kampas rem!")
    elif pred < 8:
        st.warning("🟡 Kampas rem mendekati batas minimum.")
    else:
        st.info("✅ Kondisi kampas rem masih aman.")