import streamlit as st
import pandas as pd
from utils import load_and_train, predict_category, evaluate_model

# ============================
# KONFIGURASI HALAMAN
# ============================
st.set_page_config(
    page_title="IndoNews Classifier",
    page_icon="📰",
    layout="centered"
)

# ============================
# HEADER & DESKRIPSI APLIKASI
# ============================
st.title("📰 IndoNews Classifier")
st.markdown("""
Aplikasi ini digunakan untuk **mengklasifikasikan judul berita berbahasa Indonesia**
secara otomatis menggunakan **Machine Learning & NLP (TF-IDF + Logistic Regression)**.
""")

st.markdown("---")

# ============================
# LOAD MODEL & EVALUASI (CACHE)
# ============================
@st.cache_resource
def load_assets():
    model, vectorizer, X_test_vect, y_test = load_and_train("data/clean_news_title.csv")
    acc, report = evaluate_model(model, X_test_vect, y_test)
    return model, vectorizer, acc, report

model, vectorizer, acc, report = load_assets()

# ============================
# SECTION: STATISTIK DATASET
# ============================
st.subheader("📊 Statistik Dataset")

try:
    df = pd.read_csv("data/clean_news_title.csv")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Data", len(df))
    col2.metric("Jumlah Kategori", df["category"].nunique())
    col3.metric("Rata-rata Panjang Judul", int(df["title"].str.len().mean()))

except:
    st.warning("Statistik dataset tidak dapat dimuat.")

st.markdown("---")

# ============================
# SECTION: AKURASI MODEL
# ============================
st.subheader("✅ Performa Model")

st.metric("Akurasi Model", f"{acc * 100:.2f}%")

with st.expander("Lihat Detail Evaluasi (Precision, Recall, F1-Score)"):
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

st.markdown("---")

# ============================
# SECTION: CONTOH JUDUL SIAP PAKAI
# ============================
st.subheader("📌 Contoh Judul Siap Pakai")

contoh_judul = [
    "Harga BBM Resmi Naik Mulai 1 Desember",
    "Timnas Indonesia Lolos ke Final Piala Asia",
    "Bank Indonesia Tahan Suku Bunga Acuan",
    "Kasus Korupsi Proyek Jalan Tol Disidangkan",
    "Teknologi AI Semakin Banyak Digunakan di Dunia Pendidikan",
]

pilihan = st.selectbox(
    "Pilih contoh judul (opsional):",
    ["-- Pilih Contoh Judul --"] + contoh_judul
)

st.markdown("---")

# ============================
# SECTION: INPUT PREDIKSI
# ============================
st.subheader("🔎 Coba Klasifikasi Judul Berita")

# Jika user memilih dari dropdown, otomatis isi ke text input
if pilihan != "-- Pilih Contoh Judul --":
    title = st.text_input("Masukkan Judul Berita:", value=pilihan)
else:
    title = st.text_input("Masukkan Judul Berita:")

if st.button("Prediksi"):
    if title.strip() == "":
        st.warning("Masukkan judul terlebih dahulu.")
    else:
        pred = predict_category(title, model, vectorizer)

        if "tidak valid" in pred.lower():
            st.error(pred)
        elif "tidak meyakinkan" in pred.lower():
            st.warning(pred)
        else:
            st.success(f"Kategori Prediksi: **{pred}**")

st.markdown("---")

# ============================
# FOOTER / ABOUT
# ============================
st.markdown("""
<hr>
<div style='text-align: center'>
<b>Dibuat oleh:</b> Faishal IR <br>
<b>Teknologi:</b> Python, Streamlit, TF-IDF, Logistic Regression, NLP <br>
<b>Tujuan:</b> Project Portofolio Machine Learning
</div>
""", unsafe_allow_html=True)
