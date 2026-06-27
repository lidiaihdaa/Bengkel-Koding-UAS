import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── KONFIGURASI HALAMAN UTAMA WEB ──────────────────────────────────
st.set_page_config(
    page_title="Prediksi Churn Pelanggan App", 
    page_icon="📊", 
    layout="centered"
)

# ── LOAD 5 ARTEFAK PINTAR DARI FOLDER DEPLOYMENT ───────────────────
@st.cache_resource
def load_machine_learning_artifacts():
    # Mengambil komponen hasil training model terbaru (termasuk Decision Tree)
    model        = joblib.load("deployment/best_model.pkl")
    scaler       = joblib.load("deployment/scaler.pkl")
    features     = joblib.load("deployment/feature_names.pkl")
    median_train = joblib.load("deployment/median_train.pkl")
    iqr_bounds   = joblib.load("deployment/iqr_bounds.pkl")
    return model, scaler, features, median_train, iqr_bounds

try:
    model, scaler, FEATURES, MEDIAN_TRAIN, IQR_BOUNDS = load_machine_learning_artifacts()
except Exception as e:
    st.error(f"⚠️ Gagal memuat file .pkl! Pastikan kamu sudah melakukan Run All pada notebook di VS Code agar folder 'deployment' otomatis terisi. Detail error: {e}")
    st.stop()

# ── ANTARMUKA PENGGUNA (UI DESIGN) ─────────────────────────────────
st.title("📊 Aplikasi Prediksi Customer Churn")
st.markdown("**UAS Bengkel Koding Data Science** — Prediksi Akurat Menggunakan Model Hasil Preprocessing & Hyperparameter Tuning (Decision Tree Terintegrasi).")
st.markdown("---")

st.subheader("💡 Masukkan Profil & Data Transaksi Utama Pelanggan")
st.write("Silakan isi data di bawah ini untuk menganalisis apakah pelanggan berpotensi pergi (Churn) atau tetap setia (Retained):")

col1, col2 = st.columns(2)

with col1:
    tenure_days = st.number_input("Masa Aktif Pelanggan (Tenure Days):", min_value=0, value=365, help="Jumlah hari sejak pelanggan mendaftar hingga transaksi terakhir.")
    total_spent = st.number_input("Total Pengeluaran Pelanggan (Total Spent - Rp):", min_value=0.0, value=500000.0, help="Akumulasi total uang yang sudah dibelanjakan pelanggan.")
    support_tickets = st.number_input("Jumlah Komplain (Support Tickets):", min_value=0, value=1, help="Berapa kali pelanggan mengajukan keluhan/tiket ke customer service.")

with col2:
    satisfaction_score = st.slider("Skor Kepuasan Pelanggan (Satisfaction Score 1-10):", min_value=1.0, max_value=10.0, value=7.0, step=0.5, help="Skor kepuasan yang diberikan pelanggan terhadap layanan Anda.")
    last_3_month_freq = st.number_input("Frekuensi Belanja (3 Bulan Terakhir):", min_value=0, value=5, help="Berapa kali pelanggan melakukan transaksi dalam 3 bulan terakhir.")
    age = st.number_input("Usia Pelanggan (Age):", min_value=15, max_value=100, value=30, help="Usia pelanggan saat ini.")

st.markdown("---")

# ── PROSES PREDIKSI BACKEND (SINKRONISASI DATA) ─────────────────────
if st.button("🚀 Jalankan Analisis Prediksi Churn", type="primary", use_container_width=True):
    
    # 1. Buat dictionary default untuk semua fitur (diisi otomatis oleh median data training)
    data_mapping = {}
    for col in FEATURES:
        if col in MEDIAN_TRAIN.index:
            data_mapping[col] = float(MEDIAN_TRAIN[col])
        else:
            data_mapping[col] = 0.0  # Default untuk dummy variable kategorikal jika tidak terpilih
            
    # 2. Fungsi pencari nama kolom yang fleksibel agar klop dengan fitur di dataset akhir
    def dapatkan_nama_fitur_asli(keyword, list_fitur):
        for f in list_fitur:
            if keyword.lower() in f.lower():
                return f
        return None
    
    # Deteksi kecocokan nama variabel secara otomatis
    f_tenure = dapatkan_nama_fitur_asli('tenure', FEATURES)
    f_spent = dapatkan_nama_fitur_asli('spent', FEATURES) or dapatkan_nama_fitur_asli('bill', FEATURES)
    f_tickets = dapatkan_nama_fitur_asli('ticket', FEATURES)
    f_satisfaction = dapatkan_nama_fitur_asli('satisfaction', FEATURES)
    f_freq = dapatkan_nama_fitur_asli('freq', FEATURES) or dapatkan_nama_fitur_asli('purchase', FEATURES)
    f_age = dapatkan_nama_fitur_asli('age', FEATURES)

    # Masukkan nilai dari form ke kolom yang tepat
    if f_tenure: data_mapping[f_tenure] = float(tenure_days)
    if f_spent: data_mapping[f_spent] = float(total_spent)
    if f_tickets: data_mapping[f_tickets] = float(support_tickets)
    if f_satisfaction: data_mapping[f_satisfaction] = float(satisfaction_score)
    if f_freq: data_mapping[f_freq] = float(last_3_month_freq)
    if f_age: data_mapping[f_age] = float(age)

    # Ubah menjadi DataFrame dan pastikan urutan kolomnya mutlak sesuai struktur model saat training
    input_df = pd.DataFrame([data_mapping])[FEATURES]

    # 3. ANTI-OUTLIER: Capping nilai input menggunakan batas IQR_BOUNDS dari training set
    for col, (lower, upper) in IQR_BOUNDS.items():
        if col in input_df.columns:
            input_df[col] = input_df[col].clip(lower=lower, upper=upper)

    # 4. ANTI-DATA LEAKAGE: Penskalaan dengan StandardScaler bawaan file .pkl
    try:
        if hasattr(scaler, 'feature_names_in_'):
            # Buat DataFrame template yang ukurannya pas dengan input scaler asli (isi nol)
            df_scale_template = pd.DataFrame(0.0, index=[0], columns=scaler.feature_names_in_)
            # Pindahkan data input pengguna ke template tersebut
            for col in input_df.columns:
                if col in df_scale_template.columns:
                    df_scale_template[col] = input_df[col]
            # Transformasikan skalanya menggunakan rata-rata training asli
            df_scaled = pd.DataFrame(scaler.transform(df_scale_template), columns=scaler.feature_names_in_)
            input_final = df_scaled[FEATURES]
        else:
            input_final = input_df
    except Exception:
        input_final = input_df

    # 5. EKSEKUSI PREDIKSI OLEH SUPER-MODEL TERBAIK HASIL TUNING (VOTING/RF NEW)
    try:
        prediksi = model.predict(input_final)[0]
        probas = model.predict_proba(input_final)[0] if hasattr(model, 'predict_proba') else [0.0, 0.0]
        
        st.subheader("🎯 Hasil Keputusan Analisis Model:")
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            if int(prediksi) == 1:
                st.error("🚨 **PELANGGAN BERPOTENSI CHURN**\n\n(Pelanggan memiliki kecenderungan tinggi untuk berhenti berlangganan/pergi).")
            else:
                st.success("✅ **PELANGGAN BERSTATUS RETAINED**\n\n(Pelanggan terdeteksi aman, loyal, dan akan tetap bertahan).")
                
        with col_res2:
            prob_churn = probas[1]
            st.metric(label="Tingkat Risiko Churn:", value=f"{prob_churn * 100:.2f}%")
            st.progress(float(prob_churn))
            
        st.divider()
        st.subheader("📌 Rekomendasi Tindakan Bisnis:")
        if int(prediksi) == 1:
            st.warning("""
            **Rekomendasi Penyelamatan Pelanggan:**
            1. Segera tugaskan Tim Customer Success untuk menghubungi pelanggan terkait keluhannya.
            2. Kirimkan email penawaran eksklusif berupa diskon atau kupon khusus penahanan (*retention coupon*).
            3. Prioritaskan penyelesaian sisa tiket komplain (`Support Tickets`) yang mereka miliki agar skor kepuasan meningkat.
            """)
        else:
            st.info("""
            **Rekomendasi Mempertahankan Loyalitas:**
            1. Masukkan pelanggan ke dalam program loyalitas berbasis poin atau reward khusus pelanggan setia.
            2. Kirimkan pesan personalisasi apresiasi atas kesetiaan mereka menggunakan layanan Anda.
            3. Jika usia masa aktif (`Tenure Days`) sudah tinggi, tawarkan promosi untuk *upgrade* paket ke tingkat paket Premium.
            """)
            
    except Exception as e:
        st.error(f"Terjadi kesalahan teknis saat model memproses data: {str(e)}")