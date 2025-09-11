import json
import pandas as pd
from pathlib import Path

PROCESSED_PATH = Path("data/processed")
INPUT_JSON = PROCESSED_PATH / "tweets_cleaned.json"
OUTPUT_CSV = PROCESSED_PATH / "tweets_for_annotation.csv"

def main():
    if not INPUT_JSON.exists():
        raise FileNotFoundError(f"{INPUT_JSON} introuvable. Lance d'abord le scraping + nettoyage.")

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        tweets = json.load(f)

    #
    df = pd.DataFrame(tweets)[["id", "text"]]
    df["label"] = ""  

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"✅ Fichier créé : {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
