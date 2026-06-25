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

# Fungsi memuat komponen lengkap
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

# ---------------------------------------------------------------------------------
# RESTRUKTURISASI SISTEM PEMETAAN 42 KOLOM SESUAI URUTAN ASLI MODEL_FEATURES
# ---------------------------------------------------------------------------------

# 1. Deteksi nama kolom numerik asli di model_features secara fleksibel (abaikan huruf besar/kecil)
def cari_nama_kolom(nama_input, daftar_fitur):
    for f in daftar_fitur:
        if nama_input.lower() == f.lower().strip():
            return f
    return None

col_age = cari_nama_kolom('age', model_features) or 'Age'
col_bill = cari_nama_kolom('monthly_bill', model_features) or 'Monthly_Bill'
col_usage = cari_nama_kolom('total_usage', model_features) or 'Total_Usage'
col_tenure = cari_nama_kolom('tenure_days', model_features) or 'Tenure_Days'

# 2. Buat dataframe awal untuk proses scaling data numerik
df_input = pd.DataFrame([{
    col_age: float(age),
    col_bill: float(monthly_bill),
    col_usage: float(total_usage),
    col_tenure: float(tenure_days)
}])

# 3. Lakukan normalisasi data numerik via Scaler bawaan
try:
    df_input[[col_age, col_bill, col_usage, col_tenure]] = scaler.transform(df_input[[col_age, col_bill, col_usage, col_tenure]])
except Exception:
    pass

# 4. Buat dictionary kosong untuk menampung seluruh susunan 42 kolom model_features
data_mapping = {fitur: 0.0 for fitur in model_features}

# 5. Masukkan nilai numerik hasil scaling ke dictionary mapping
data_mapping[col_age] = float(df_input[col_age].values[0])
data_mapping[col_bill] = float(df_input[col_bill].values[0])
data_mapping[col_usage] = float(df_input[col_usage].values[0])
data_mapping[col_tenure] = float(df_input[col_tenure].values[0])

# 6. Petakan kategori dummy (Gender dan Subscription Type) secara akurat ke dalam 42 kolom
for fitur in model_features:
    # Cek Gender dummy (contoh: Gender_Female, Gender_Male)
    if 'gender' in fitur.lower():
        if gender.lower() in fitur.lower():
            data_mapping[fitur] = 1.0
            
    # Cek Subscription Type dummy (contoh: Subscription_Type_Basic, dll)
    if 'sub' in fitur.lower() or 'type' in fitur.lower():
        if sub_type.lower() in fitur.lower():
            data_mapping[fitur] = 1.0

# 7. Konversi dictionary mapping yang sudah matang menjadi DataFrame final berdimensi 42 kolom
input_final = pd.DataFrame([data_mapping])[model_features]

# Tombol Eksekusi Prediksi
st.markdown("---")
if st.button("🚀 Jalankan Analisis Prediksi Churn", type="primary"):
    prediction = model.predict(input_final.values)[0]
    prediction_proba = model.predict_proba(input_final.values)[0]
    
    st.subheader("🎯 Hasil Keputusan Analisis:")
    if prediction == 1:
        st.error(f"🚨 **PELANGGAN BERPOTENSI CHURN (BERHENTI BERLANGGANAN)**")
        st.write(f"Tingkat Probabilitas Keyakinan Model: **{prediction_proba[1]*100:.2f}%**")
    else:
        st.success(f"✅ **PELANGGAN BERSTATUS RETAINED (SETIA / BERTAHAN)**")
        st.write(f"Tingkat Probabilitas Keyakinan Model: **{prediction_proba[0]*100:.2f}%**")