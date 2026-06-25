import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Konfigurasi dasar Halaman Web
st.set_page_config(
    page_title="Prediksi Customer Churn App",
    page_icon="📊",
    layout="centered"
)

# Fungsi memuat semua file pintar .pkl
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
    st.error("⚠️ File .pkl tidak ditemukan di folder proyek!")
    st.stop()

# Tampilan Judul Web
st.title("📊 Aplikasi Prediksi Customer Churn")
st.write("Aplikasi cerdas berbasis Machine Learning untuk memprediksi apakah pelanggan akan berhenti berlangganan (*Churn*) atau bertahan (*Retained*).")
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

# Membuat DataFrame Baru dari Input Form dengan penyesuaian otomatis nama kolom latih
# Mengambil 4 nama kolom pertama dari model_features agar dijamin 100% sama dengan data latih
col_age = model_features[0]
col_bill = model_features[1]
col_usage = model_features[2]
col_tenure = model_features[3]

input_data = pd.DataFrame([{
    col_age: float(age),
    col_bill: float(monthly_bill),
    col_usage: float(total_usage),
    col_tenure: float(tenure_days),
    'gender': gender,
    'subscription_type': sub_type
}])

# One-Hot Encoding otomatis
input_encoded = pd.get_dummies(input_data)

# Sinkronisasi kolom agar sama persis dengan susunan model_features.pkl
for col in model_features:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

input_final = input_encoded[model_features].copy()

# Normalisasi Fitur Numerik menggunakan Scaler secara aman tanpa trigger KeyError nama kolom
kolom_numeric_real = [col_age, col_bill, col_usage, col_tenure]
input_numeric_scaled = scaler.transform(input_final[kolom_numeric_real])
input_final.loc[:, kolom_numeric_real] = input_numeric_scaled

# Tombol Eksekusi Prediksi
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