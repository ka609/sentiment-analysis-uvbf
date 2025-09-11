import tweepy
import json
from pathlib import Path
from typing import List, Dict, Any

RAW_PATH = Path("data/raw")

def get_tweets(bearer_token: str, query: str = "UVBF", max_results: int = 100, lang: str = "fr") -> List[Dict[str, Any]]:
    try:
        client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)
    except Exception as e:
        raise Exception(f"Erreur d'authentification Twitter: {str(e)}")

    tweets_data = []

    try:
        query_with_lang = f"{query} lang:{lang}"
        response = client.search_recent_tweets(
            query=query_with_lang,
            tweet_fields=["id", "text", "created_at", "lang", "public_metrics", "author_id"],
            expansions=["author_id"],
            user_fields=["id", "username", "name"],
            max_results=min(max_results, 100)
        )

        if not response.data:
            print("⚠️ Aucun tweet trouvé.")
            return []

        users_dict = {user.id: user for user in response.includes.get("users", [])}

        for tweet in response.data:
            author = users_dict.get(tweet.author_id)
            tweet_info = {
                "id": tweet.id,
                "text": tweet.text,
                "created_at": tweet.created_at.isoformat() if tweet.created_at else None,
                "retweet_count": tweet.public_metrics.get("retweet_count", 0),
                "reply_count": tweet.public_metrics.get("reply_count", 0),
                "like_count": tweet.public_metrics.get("like_count", 0),
                "quote_count": tweet.public_metrics.get("quote_count", 0),
                "lang": tweet.lang,
                "author_id": tweet.author_id,
                "author_username": author.username if author else None,
                "author_name": author.name if author else None
            }
            tweets_data.append(tweet_info)

    except tweepy.TweepyException as e:
        print(f"Erreur lors de la recherche de tweets: {str(e)}")

    return tweets_data


def save_to_json(tweets: List[Dict[str, Any]], filepath: Path):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(tweets, f, ensure_ascii=False, indent=4)
    print(f"✅ Données sauvegardées dans {filepath}")
