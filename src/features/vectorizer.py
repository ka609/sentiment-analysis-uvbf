from pathlib import Path
import json
import joblib
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List

RAW_TWEETS_FILE = Path("data/processed/tweets_cleaned.json")
VECTORIZER_PATH = Path("data/vectorizer/tfidf_vectorizer.joblib")
MATRIX_PATH = Path("data/vectorizer/tfidf_matrix.npz")
IDS_PATH = Path("data/vectorizer/ids_order.joblib")

def load_texts(file_path: Path) -> List[dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_tfidf(texts: List[str], max_features: int = 10000):
    vect = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b\w+\b"
    )
    X = vect.fit_transform(texts)
    return vect, X

def main():
    if not RAW_TWEETS_FILE.exists():
        raise FileNotFoundError(f"{RAW_TWEETS_FILE} not found. Run scraping & cleaning first.")
    
    tweets = load_texts(RAW_TWEETS_FILE)
    ids = [t.get("id") for t in tweets]
    texts = [t.get("text", "") for t in tweets]

    vect, X = build_tfidf(texts)

    VECTORIZER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vect, VECTORIZER_PATH)
    sp.save_npz(MATRIX_PATH, X)
    joblib.dump(ids, IDS_PATH)

    print(f"✅ TF-IDF vectorizer saved to {VECTORIZER_PATH}")
    print(f"✅ TF-IDF matrix saved to {MATRIX_PATH} (shape: {X.shape})")
    print(f"✅ IDs order saved to {IDS_PATH}")

if __name__ == "__main__":
    main()
