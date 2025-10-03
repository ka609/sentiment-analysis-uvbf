# fusion_clean.py
import pandas as pd
import re
import nltk
import unicodedata
import spacy
import numpy as np
from nltk.corpus import stopwords
import os # Ajouté pour une meilleure gestion des chemins

# --- 1. Préparation des ressources ---
# Garde les négations pour l'analyse de sentiment
nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("french")) - {"pas", "jamais", "n", "ne"} 
# Ajout de 'n' et 'ne' souvent ignorés par NLTK mais importants pour la négation

# Charger spaCy FR 
try:
    nlp = spacy.load("fr_core_news_sm", disable=["parser", "ner"])
except OSError:
    print("🚨 Modèle spaCy 'fr_core_news_sm' non trouvé.")
    print("Lancez en terminal: python -m spacy download fr_core_news_sm")
    exit()

# --- 2. Fonction de nettoyage (inchangée) ---
def clean_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return np.nan

    # Normalisation
    text = unicodedata.normalize("NFKC", text).lower()

    # Supprimer liens, hashtags, mentions
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"#\w+|@\w+", " ", text)

    # Supprimer ponctuation et chiffres (garder lettres FR)
    # Laisse l'apostrophe (guillemet simple) pour les contractions comme 'l'université'
    text = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ\s']", " ", text)

    # Tokenisation + lemmatisation
    doc = nlp(text)
    tokens = [
        token.lemma_ for token in doc
        if token.lemma_ not in STOPWORDS and len(token.lemma_) > 2
    ]
    return " ".join(tokens).strip()

# --- 3. Charger et préparer les fichiers ---
print("📂 Chargement et préparation des fichiers...")

# Fichier 1 : Données Facebook (déjà nettoyées/structurées)
df_fb = pd.read_csv("./facebook_posts_clean.csv")
# Assumer que la colonne du texte nettoyé est 'clean_text' ou 'text'
if "text" not in df_fb.columns:
    df_fb = df_fb.rename(columns={"clean_text": "text"})
df_fb = df_fb[["text"]]
print(f"  - Facebook : {len(df_fb)} lignes. Colonne utilisée: 'text'")


# Fichier 2 : Données Collecte (BRUT, nécessite re-nettoyage initial)
# Assurez-vous que le chemin et le nom sont corrects
COLLECTE_PATH = "./avis_uvbf_final_clean.csv" # Assurez-vous que c'est le bon chemin et nom!

try:
    df_coll_raw = pd.read_csv(
        COLLECTE_PATH, 
        sep='","', # Séparateur spécifique pour les données brutes
        header=None,
        names=['Date_Heure_GMT', 'Avis_Positif', 'Avis_Negatif'],
        engine='python',
        skiprows=1 
    )
    # Nettoyage initial des guillemets
    for col in ['Avis_Positif', 'Avis_Negatif']:
        df_coll_raw[col] = df_coll_raw[col].astype(str).str.replace('^"|"$', '', regex=True).str.strip()
        
    # --- Création de la colonne 'text' pour le fichier collecte ---
    # Concatène les avis positifs et négatifs pour chaque ligne en un seul bloc de texte
    df_coll_raw['text'] = df_coll_raw['Avis_Positif'].fillna('') + ' ' + df_coll_raw['Avis_Negatif'].fillna('')
    # Nettoyer les espaces doubles après la concaténation
    df_coll_raw['text'] = df_coll_raw['text'].str.strip().str.replace(r'\s+', ' ', regex=True)
    
    # Filtrer les lignes vides après concaténation
    df_coll = df_coll_raw[df_coll_raw['text'] != '']
    df_coll = df_coll[['text']] # Ne garder que la colonne de texte pour la fusion
    print(f"  - Collecte : {len(df_coll)} lignes après consolidation des avis.")

except FileNotFoundError:
    print(f"\n🚨 ERREUR: Fichier Collecte non trouvé à {COLLECTE_PATH}")
    print("Vérifiez le chemin relatif.")
    exit()
except Exception as e:
    print(f"\n🚨 ERREUR lors de la lecture/préparation de collecte.csv: {e}")
    exit()

# --- 4. Fusion ---
df = pd.concat([df_fb, df_coll], ignore_index=True)
print(f"✅ Fusion terminée : {len(df)} avis au total.")
print("---")

# --- 5. Nettoyage global ---
print("🧹 Application du nettoyage et lemmatisation...")
df["clean_text"] = df["text"].apply(clean_text)
# Supprimer les lignes où 'clean_text' est vide après la lemmatisation/filtrage
df_clean = df[df["clean_text"] != ""] 

print(f"✅ Nettoyage appliqué. Taille finale: {len(df_clean)} avis utilisables.")
print(f"Nombre de lignes supprimées (vides après nettoyage) : {len(df) - len(df_clean)}")
print("---")

# --- 6. Sauvegarde ---
output_dir = "../preprocessing/"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "all_clean.csv")
df_clean.to_csv(output_file, index=False, encoding="utf-8")
print(f"🎉 Fichier final sauvegardé : {output_file}")
print("\nAperçu des 10 premières lignes nettoyées:")
print(df_clean.head(10))