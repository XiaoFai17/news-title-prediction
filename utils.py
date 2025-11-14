import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import re

# ----------------------------
# 1. PREPROCESS (sederhana)
# ----------------------------
def preprocess_text(text: str):
    """
    Preprocessing ringan sesuai gaya notebook kamu.
    Jika nanti kamu ingin menambah stopword atau stemming,
    cukup edit fungsi ini.
    """
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text


# ----------------------------
# 2. LOAD DATASET & TRAIN MODEL
# ----------------------------
def load_and_train(csv_path="data/berita_clean.csv"):
    """
    Membaca dataset CSV yang sudah kamu bersihkan.
    Melatih model Logistic Regression.
    Menghasilkan model dan vectorizer yang siap pakai.
    """

    # Load dataset
    df = pd.read_csv(csv_path)

    # Memisahkan fitur dan target
    fitur = df["title"]
    target = df["category"]

    # Pembagian train/test (tidak digunakan untuk UI, hanya untuk konsistensi)
    fitur_train, fitur_test, target_train, target_test = train_test_split(
        fitur, target, test_size=0.2, random_state=42
    )

    # Membuat vectorizer CountVectorizer (sesuai ipynb)
    vectorizer = CountVectorizer()
    fitur_train_vect = vectorizer.fit_transform(fitur_train)

    # Model terbaik: Logistic Regression
    model = LogisticRegression(max_iter=1000)
    model.fit(fitur_train_vect, target_train)

    return model, vectorizer


# ----------------------------
# 3. PREDIKSI KATEGORI
# ----------------------------
def predict_category(title: str, model, vectorizer):
    """
    Menerima judul berita dari user
    dan menghasilkan kategori prediksi.
    """

    title_clean = preprocess_text(title)
    title_vect = vectorizer.transform([title_clean])
    pred = model.predict(title_vect)[0]

    return pred
