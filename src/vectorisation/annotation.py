import os
import pandas as pd
import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# --- Configuration des fichiers ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CLEAN_FILE = os.path.join(BASE_DIR, "../preprocessing/facebook_posts_clean.csv")
OUTPUT_ANNOTATED_FILE = os.path.join(BASE_DIR, "../data/processed/annotated_sentiment_data.csv")

# --- 1. Chargement du modèle ---
MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
print(f"🧠 Chargement du modèle : {MODEL_NAME}...")

device = 0 if torch.cuda.is_available() else -1

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer,
    framework="pt",
    device=device
)

# --- 2. Chargement des données ---
try:
    df_clean = pd.read_csv(INPUT_CLEAN_FILE)
    texts_to_analyze = df_clean['clean_text'].fillna('').astype(str).tolist()
    if not texts_to_analyze:
        print("🚨 ERREUR: Aucun texte trouvé dans all_clean.csv")
        exit()
except FileNotFoundError:
    print(f"🚨 ERREUR: Fichier introuvable à {INPUT_CLEAN_FILE}")
    exit()

# --- 3. Annotation automatique ---
print(f"📝 Annotation de {len(texts_to_analyze)} avis...")

results = []
batch_size = 32
for i in range(0, len(texts_to_analyze), batch_size):
    batch = texts_to_analyze[i:i+batch_size]
    results.extend(sentiment_pipeline(batch))

# --- 4. Intégration des résultats ---
df_clean['Sentiment_Predicted'] = [res['label'] for res in results]
df_clean['Sentiment_Score'] = [res['score'] for res in results]

# Traduction labels EN → FR
sentiment_mapping = {
    'LABEL_0': 'NÉGATIF',
    'LABEL_1': 'NEUTRE',
    'LABEL_2': 'POSITIF'
}
df_clean['Sentiment'] = df_clean['Sentiment_Predicted'].replace(sentiment_mapping)

# --- 5. Sauvegarde du dataset annoté ---
output_dir = os.path.dirname(OUTPUT_ANNOTATED_FILE)
os.makedirs(output_dir, exist_ok=True)

final_cols = [col for col in ['text', 'clean_text', 'Sentiment', 'Sentiment_Score'] if col in df_clean.columns]
df_clean[final_cols].to_csv(OUTPUT_ANNOTATED_FILE, index=False, encoding='utf-8')

print(f"\n✅ Dataset annoté sauvegardé dans : {OUTPUT_ANNOTATED_FILE}")
print("\n--- Aperçu des sentiments annotés ---")
print(df_clean[['clean_text', 'Sentiment', 'Sentiment_Score']].head())
