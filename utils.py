import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ----------------------------
# 1. STOPWORDS INDONESIA (MANUAL, TANPA NLTK)
# ----------------------------
STOPWORDS_ID = set([
    "yang", "dan", "di", "ke", "dari", "untuk", "pada", "dengan", "adalah",
    "itu", "ini", "sebagai", "oleh", "juga", "karena", "akan", "dalam",
    "tidak", "atau", "lebih", "sudah", "saat", "para", "agar", "tentang"
])

# ----------------------------
# 2. DETEKSI KATA NGAWUR (ENHANCED)
# ----------------------------
def is_gibberish_raw_token(token: str):
    """
    Cek satu token: terlalu pendek, tidak ada vokal, terlalu banyak konsonan berturut,
    atau rasio huruf unik terlalu rendah (contoh qqqqq).
    """
    if not token:
        return True
    if len(token) <= 2:
        return True
    # jika token tidak mengandung vokal sama sekali -> curiga
    if not re.search(r"[aiueo]", token):
        return True
    # konsonan run yang panjang (misal 'bcdfghjk...') -> curiga
    if re.search(r"[bcdfghjklmnpqrstvwxyz]{6,}", token):
        return True
    # terlalu banyak pengulangan karakter (misal 'aaaaaaa' atau 'qqqqq')
    if re.search(r"(.)\1\1\1", token):
        return True
    return False

def is_gibberish(text: str, vectorizer=None, vocab_token_ratio_threshold=0.5):
    """
    Enhanced gibberish detector:
      - jika setelah preprocessing tidak ada token -> gibberish
      - jika >50% token tidak ada di vocabulary model (opsional: butuh vectorizer)
      - jika sebagian token memenuhi is_gibberish_raw_token -> gibberish
    """
    if not isinstance(text, str):
        return True

    txt = text.lower()
    txt = re.sub(r"[^a-z\s]", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    tokens = [t for t in txt.split() if len(t) > 0]

    if len(tokens) == 0:
        return True

    # cek raw token heuristic
    suspicious_count = sum(1 for t in tokens if is_gibberish_raw_token(t))
    if suspicious_count >= max(1, int(0.5 * len(tokens))):
        return True

    # jika diberikan vectorizer: cek seberapa banyak token muncul di vocab
    if vectorizer is not None and hasattr(vectorizer, "vocabulary_"):
        vocab = vectorizer.vocabulary_
        in_vocab_count = sum(1 for t in tokens if t in vocab)
        ratio = in_vocab_count / len(tokens)
        # jika lebih sedikit dari threshold token ada di vocab -> curiga
        if ratio < vocab_token_ratio_threshold:
            return True

    return False

# ----------------------------
# 3. PREPROCESSING TEKS (VERSI ADVANCE)
# ----------------------------
def preprocess_text(text: str):
    """
    - Lowercase
    - Hapus simbol & angka
    - Hapus stopwords
    - Filtering kata pendek
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)   # hanya huruf
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()
    tokens = [word for word in tokens if word not in STOPWORDS_ID and len(word) > 2]

    return " ".join(tokens)

# ----------------------------
# 4. LOAD DATASET & TRAIN MODEL
# ----------------------------
def load_and_train(csv_path="data/berita_clean.csv"):
    """
    Training model + vectorizer (TF-IDF + Logistic Regression)
    """
    df = pd.read_csv(csv_path)

    df["title_clean"] = df["title"].apply(preprocess_text)

    fitur = df["title_clean"]
    target = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(
        fitur, target, test_size=0.2, random_state=42, stratify=target
    )

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )

    X_train_vect = vectorizer.fit_transform(X_train)
    X_test_vect = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vect, y_train)

    return model, vectorizer, X_test_vect, y_test

# ----------------------------
# 5. EVALUASI AKURASI MODEL (UNTUK DITAMPILKAN DI STREAMLIT)
# ----------------------------
def evaluate_model(model, X_test_vect, y_test):
    """
    Mengembalikan:
    - Accuracy
    - Classification report (precision, recall, f1)
    """
    y_pred = model.predict(X_test_vect)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return acc, report

# ----------------------------
# 6. PREDIKSI KATEGORI + FILTER INPUT NGAWUR + CONFIDENCE CHECK
# ----------------------------
def predict_category(title: str, model, vectorizer, prob_threshold=0.35):
    """
    Prediksi kategori berita dari input user
    - Periksa gibberish (menggunakan vectorizer untuk cek vocab)
    - Preprocess
    - Cek confidence (predict_proba)
    Return:
      - jika invalid/gibberish -> message string yang mengandung 'tidak valid'
      - jika low confidence -> message string yang mengandung 'tidak meyakinkan'
      - jika oke -> return kategori (string)
    """
    # 1) deteksi teks ngawur
    if is_gibberish(title, vectorizer=vectorizer, vocab_token_ratio_threshold=0.5):
        return "Input tidak valid (teks acak / kata tidak dikenal terdeteksi)"

    title_clean = preprocess_text(title)
    if len(title_clean.strip()) == 0:
        return "Judul terlalu pendek atau tidak bermakna"

    # 2) transform dan prediksi
    title_vect = vectorizer.transform([title_clean])

    # pastikan model mendukung predict_proba
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(title_vect)[0]
        max_p = proba.max()
        pred = model.classes_[proba.argmax()]
        # jika confidence rendah, anggap tidak meyakinkan
        if max_p < prob_threshold:
            return f"Prediksi tidak meyakinkan (confidence {max_p:.2f}). Mohon periksa kembali judul."
        else:
            return pred
    else:
        # fallback: jika model tidak punya proba
        pred = model.predict(title_vect)[0]
        return pred
