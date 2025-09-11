import os
from dotenv import load_dotenv
from pathlib import Path
from src.scraping.tweets_scraper import get_tweets, save_to_json, RAW_PATH
from src.preprocessing.text_cleaner import clean_tweets

load_dotenv()

BEARER_TOKEN = os.getenv("BEARER_TOKEN")
PROCESSED_PATH = Path("data/processed")

def main():
    query = "UVBF"
    max_results = 100
    lang = "fr"

    if not BEARER_TOKEN:
        raise ValueError("⚠️ La clé BEARER_TOKEN n'est pas définie dans le fichier .env")

    tweets = get_tweets(
        bearer_token=BEARER_TOKEN,
        query=query,
        max_results=max_results,
        lang=lang
    )

    if tweets:
        save_to_json(tweets, RAW_PATH / "tweets.json")

        cleaned_tweets = clean_tweets(tweets)
        save_to_json(cleaned_tweets, PROCESSED_PATH / "tweets_cleaned.json")

        print(f"✅ {len(cleaned_tweets)} tweets récupérés et nettoyés.")

if __name__ == "__main__":
    main()
