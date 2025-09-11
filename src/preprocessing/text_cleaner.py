import re
import spacy
from typing import List

# Charger modèle spaCy FR
nlp = spacy.load("fr_core_news_sm")

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticônes
    "\U0001F300-\U0001F5FF"  # symboles et pictogrammes
    "\U0001F680-\U0001F6FF"  # transport et symboles
    "\U0001F1E0-\U0001F1FF"  # drapeaux
    "\U00002700-\U000027BF"  # divers symboles
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE
)

def basic_clean(text: str) -> str:
    """Nettoyage basique avant NLP (URLs, mentions, hashtags, emojis, ponctuation inutile)."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)   # URLs
    text = re.sub(r"@\w+", "", text)             # mentions
    text = re.sub(r"#\w+", "", text)             # hashtags
    text = EMOJI_PATTERN.sub(r"", text)          # emojis
    text = re.sub(r"[^a-zà-ÿ\s]", " ", text)     # lettres + accents uniquement
    text = re.sub(r"\s+", " ", text).strip()     # espaces multiples
    return text

def spacy_preprocess(text: str) -> str:
    """Tokenisation, suppression des stopwords et lemmatisation avec spaCy."""
    doc = nlp(text)
    tokens = [
        token.lemma_ for token in doc
        if not token.is_stop and not token.is_punct and token.lemma_.strip() != ""
    ]
    return " ".join(tokens)

def clean_text(text: str) -> str:
    """Pipeline complet de nettoyage."""
    text = basic_clean(text)
    text = spacy_preprocess(text)
    return text

def clean_tweets(tweets: List[dict], text_field: str = "text") -> List[dict]:
    cleaned_tweets = []
    for tweet in tweets:
        cleaned_tweet = tweet.copy()
        if text_field in tweet:
            cleaned_tweet[text_field] = clean_text(tweet[text_field])
        cleaned_tweets.append(cleaned_tweet)
    return cleaned_tweets
