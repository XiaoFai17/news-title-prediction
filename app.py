import streamlit as st
from utils import load_and_train, predict_category

st.title("📰 Klasifikasi Judul Berita Indonesia")

# Cache supaya training tidak diulang setiap refresh
@st.cache_resource
def load_assets():
    model, vectorizer = load_and_train("data/clean_news_title.csv")
    return model, vectorizer

model, vectorizer = load_assets()

title = st.text_input("Masukkan Judul Berita:")

if st.button("Prediksi"):
    if title.strip() == "":
        st.warning("Masukkan judul terlebih dahulu.")
    else:
        pred = predict_category(title, model, vectorizer)
        st.success(f"Kategori Prediksi: **{pred}**")
