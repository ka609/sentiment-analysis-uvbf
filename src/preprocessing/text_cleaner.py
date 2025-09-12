import re
import spacy
from typing import List


nlp = spacy.load("fr_core_news_sm")

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  
    "\U0001F300-\U0001F5FF"  
    "\U0001F680-\U0001F6FF"  
    "\U0001F1E0-\U0001F1FF"  
    "\U00002700-\U000027BF"  
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE
)

def basic_clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)   # URLs
    text = re.sub(r"@\w+", "", text)             # mentions
    text = re.sub(r"#\w+", "", text)             # hashtags
    text = EMOJI_PATTERN.sub(r"", text)          # emojis
    text = re.sub(r"[^a-zà-ÿ\s]", " ", text)     # lettres + accents uniquement
    text = re.sub(r"\s+", " ", text).strip()     # espaces multiples
    return text

def spacy_preprocess(text: str) -> str:
    
    doc = nlp(text)
    tokens = [
        token.lemma_ for token in doc
        if not token.is_stop and not token.is_punct and token.lemma_.strip() != ""
    ]
    return " ".join(tokens)

def clean_text(text: str) -> str:
    
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
