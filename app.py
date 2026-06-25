import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(
    page_title="Prediksi Customer Churn App",
    page_icon="📊",
    layout="centered"
)

@st.cache_resource
def load_machine_learning_components():
    with open('best_random_forest_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('model_features.pkl', 'rb') as f:
        features = pickle.load(f)
    return model, scaler, features

try:
    model, scaler, model_features = load_machine_learning_components()
except FileNotFoundError:
    st.error("⚠️ File .pkl tidak ditemukan!")
    st.stop()

st.title("📊 Aplikasi Prediksi Customer Churn")
st.write("Aplikasi cerdas berbasis Machine Learning untuk memprediksi status pelanggan.")
st.markdown("---")

st.subheader("💡 Masukkan Data Profil Transaksi Pelanggan")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Umur Pelanggan (Age):", min_value=1, max_value=100, value=30)
    monthly_bill = st.number_input("Tagihan Bulanan (Monthly Bill):", min_value=0.0, value=50.0)
    total_usage = st.number_input("Total Penggunaan Layanan (Total Usage):", min_value=0.0, value=200.0)
    tenure_days = st.number_input("Masa Aktif Pelanggan (Tenure Days):", min_value=0, value=120)

with col2:
    gender = st.selectbox("Jenis Kelamin (Gender):", options=["Male", "Female"])
    sub_type = st.selectbox("Tipe Langganan (Subscription Type):", options=["Basic", "Standard", "Premium"])

# MEMBUAT DATAFRAME BARU YANG AMAN DARI COPY WARNING
input_data = pd.DataFrame([{
    'age': float(age),
    'monthly_bill': float(monthly_bill),
    'total_usage': float(total_usage),
    'tenure_days': float(tenure_days),
    'gender': gender,
    'subscription_type': sub_type
}])

# One-Hot Encoding
input_encoded = pd.get_dummies(input_data)

# Menyamakan kolom dengan model_features.pkl
for col in model_features:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

input_final = input_encoded[model_features].copy()

# PROSES SCALING YANG DIJAMIN AMAN & SEHAT (Mencegah Blank Putih)
kolom_numeric = ['age', 'monthly_bill', 'total_usage', 'tenure_days']
input_numeric_scaled = scaler.transform(input_final[kolom_numeric])
input_final.loc[:, kolom_numeric] = input_numeric_scaled

st.markdown("---")
if st.button("🚀 Jalankan Analisis Prediksi Churn", type="primary"):
    prediction = model.predict(input_final)[0]
    prediction_proba = model.predict_proba(input_final)[0]
    
    st.subheader("🎯 Hasil Keputusan Analisis:")
    if prediction == 1:
        st.error(f"🚨 **PELANGGAN BERPOTENSI CHURN (BERHENTI BERLANGGANAN)**")
        st.write(f"Tingkat Probabilitas Keyakinan Model: **{prediction_proba[1]*100:.2f}%**")
    else:
        st.success(f"✅ **PELANGGAN BERSTATUS RETAINED (SETIA / BERTAHAN)**")
        st.write(f"Tingkat Probabilitas Keyakinan Model: **{prediction_proba[0]*100:.2f}%**")