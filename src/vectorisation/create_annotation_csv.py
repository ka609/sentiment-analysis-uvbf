import json
import pandas as pd
from pathlib import Path
import joblib

VECTOR_PATH = Path("data/vectorizer")
PROCESSED_PATH = Path("data/processed")

RAW_TWEETS_FILE = PROCESSED_PATH / "tweets_cleaned.json"
ANNOTATION_CSV = PROCESSED_PATH / "tweets_for_annotation.csv"

def main():
    if not RAW_TWEETS_FILE.exists():
        raise FileNotFoundError(f"{RAW_TWEETS_FILE} n'existe pas. Exécute d'abord le scraping et le nettoyage.")

    with open(RAW_TWEETS_FILE, "r", encoding="utf-8") as f:
        tweets = json.load(f)

    df = pd.DataFrame(tweets)
    
    df = df[["id", "text"]]
    
    df["label"] = ""

    ANNOTATION_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ANNOTATION_CSV, index=False, encoding="utf-8")
    print(f"✅ CSV pour annotation créé : {ANNOTATION_CSV}")

if __name__ == "__main__":
    main()
