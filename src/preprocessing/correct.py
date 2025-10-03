import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob  # Pour résumé et sentiment implicite
from emoji import demojize

# --- 0️⃣ Télécharger les stopwords (une seule fois) ---
nltk.download('stopwords')
stop_words = set(stopwords.words('french'))

# --- 1️⃣ Charger le fichier CSV ---
INPUT_FILE = "./facebook_posts_clean.csv"  # <-- à remplacer
df = pd.read_csv(INPUT_FILE)

# --- 2️⃣ Corriger les erreurs de classification ---
corrections = {
    "positf": "positif",
    "negatif": "négatif",
    "neutre ": "neutre",
    "Positif": "positif",
    "Negatif": "négatif",
    "Neutre": "neutre"
}
df['sentiment'] = df['sentiment'].replace(corrections)

print("\n✅ Répartition après correction :")
print(df['sentiment'].value_counts())

# --- 3️⃣ Nettoyer le texte ---
def clean_text(text):
    text = str(text).lower()
    text = demojize(text)  # Transformer emojis en texte
    text = re.sub(r'[^a-zA-ZÀ-ÿ\s]', '', text)
    text = " ".join([w for w in text.split() if w not in stop_words])
    return text

df['clean_text'] = df['Text'].apply(clean_text)

# --- 4️⃣ Analyse des top mots et bigrammes ---
def get_top_words(text_series, top_n=10):
    words = " ".join(text_series).split()
    counter = Counter(words)
    return counter.most_common(top_n)

def get_top_bigrams(text_series, top_n=10):
    vectorizer = CountVectorizer(ngram_range=(2,2), stop_words=stop_words)
    X = vectorizer.fit_transform(text_series)
    sums = X.sum(axis=0)
    bigrams_freq = [(word, sums[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    bigrams_freq = sorted(bigrams_freq, key=lambda x: x[1], reverse=True)
    return bigrams_freq[:top_n]

for sentiment in ['positif', 'négatif', 'neutre']:
    subset = df[df['sentiment'] == sentiment]['clean_text']
    print(f"\n--- {sentiment.upper()} ---")
    print("Top mots :", get_top_words(subset))
    print("Top bigrammes :", get_top_bigrams(subset))

# --- 5️⃣ WordCloud ---
def plot_wordcloud(text_series, title):
    text = " ".join(text_series)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=16)
    plt.show()

for sentiment in ['positif', 'négatif', 'neutre']:
    subset = df[df['sentiment'] == sentiment]['clean_text']
    plot_wordcloud(subset, f"WordCloud - Avis {sentiment}")

# --- 6️⃣ Résumé automatique des avis ---
def summarize_text(text_series, n_sentences=3):
    all_text = " ".join(text_series)
    blob = TextBlob(all_text)
    # Sélection des phrases les plus "polaires"
    sentences = sorted(blob.sentences, key=lambda s: abs(s.sentiment.polarity), reverse=True)
    return " ".join(str(s) for s in sentences[:n_sentences])

for sentiment in ['positif', 'négatif', 'neutre']:
    subset = df[df['sentiment'] == sentiment]['Text']
    print(f"\nRésumé {sentiment} :")
    print(summarize_text(subset))

# --- 7️⃣ Préparer les données pour NLP ---
X = df['clean_text']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("\n✅ Données prêtes pour NLP : X_train, X_test, y_train, y_test")
