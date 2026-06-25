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
    # Kita baca contoh data asli untuk replikasi dummy encoding yang sempurna
    try:
        df_sample = pd.read_csv('Sales - Marketing customer dataset.csv').dropna()
        # Buat dummy encoding dari data asli persis seperti di Colab (buang kolom target jika ada)
        if 'Target' in df_sample.columns:
            df_sample = df_sample.drop(columns=['Target'])
        elif 'churn' in df_sample.columns:
            df_sample = df_sample.drop(columns=['churn'])
            
        # Pisahkan fitur dan target (X)
        X_sample = pd.get_dummies(df_sample)
        features_list = list(X_sample.columns)
    except Exception:
        # Fallback jika file csv tidak terbaca
        with open('model_features.pkl', 'rb') as f:
            features_list = pickle.load(f)
    return model, scaler, features_list

try:
    model, scaler, model_features = load_machine_learning_components()
except FileNotFoundError:
    st.error("⚠️ File komponen model tidak ditemukan di folder proyek!")
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
# SINKRONISASI KOLOM 100% SAMA DENGAN STRUKTUR DATA TRAINING
# ---------------------------------------------------------------------------------

# 1. Cari nama kolom numerik asli di list features
def map_column_name(name, list_f):
    for f in list_f:
        if name.lower() == f.lower().strip():
            return f
    return name

col_age = map_column_name('age', model_features)
col_bill = map_column_name('monthly_bill', model_features)
col_usage = map_column_name('total_usage', model_features)
col_tenure = map_column_name('tenure_days', model_features)

# 2. Buat dictionary dengan nilai default 0 untuk semua 42 kolom
final_data_dict = {feat: 0.0 for feat in model_features}

# 3. Buat DataFrame sementara untuk proses scaling nilai numerik
df_num = pd.DataFrame([{col_age: float(age), col_bill: float(monthly_bill), col_usage: float(total_usage), col_tenure: float(tenure_days)}])
try:
    df_num[[col_age, col_bill, col_usage, col_tenure]] = scaler.transform(df_num[[col_age, col_bill, col_usage, col_tenure]])
except Exception:
    pass

# 4. Masukkan skor numerik hasil scaling ke dictionary utama
final_data_dict[col_age] = float(df_num[col_age].values[0])
final_data_dict[col_bill] = float(df_num[col_bill].values[0])
final_data_dict[col_usage] = float(df_num[col_usage].values[0])
final_data_dict[col_tenure] = float(df_num[col_tenure].values[0])

# 5. Nyalakan nilai 1.0 pada kolom kategori dummy secara presisi
for feat in model_features:
    # Cek kecocokan kategori Gender
    if 'gender' in feat.lower():
        if gender.lower() in feat.lower():
            final_data_dict[feat] = 1.0
            
    # Cek kecocokan kategori Subscription Type
    if 'sub' in feat.lower() or 'type' in feat.lower():
        if sub_type.lower() in feat.lower():
            final_data_dict[feat] = 1.0

# 6. Ubah menjadi DataFrame final dengan urutan kolom yang mutlak sama
input_final = pd.DataFrame([final_data_dict])[model_features]

# Tombol Eksekusi Prediksi
st.markdown("---")
if st.button("🚀 Jalankan Analisis Prediksi Churn", type="primary"):
    prediction = model.predict(input_final.values)[0]
    prediction_proba = model.predict_proba(input_final.values)[0]
    
    st.subheader("🎯 Hasil Keputusan Analisis:")
    if int(prediction) == 1:
        st.error(f"🚨 **PELANGGAN BERPOTENSI CHURN (BERHENTI BERLANGGANAN)**")
        st.write(f"Tingkat Probabilitas Keyakinan Model: **{prediction_proba[1]*100:.2f}%**")
    else:
        st.success(f"✅ **PELANGGAN BERSTATUS RETAINED (SETIA / BERTAHAN)**")
        st.write(f"Tingkat Probabilitas Keyakinan Model: **{prediction_proba[0]*100:.2f}%**")