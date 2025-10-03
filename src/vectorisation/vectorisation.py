import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from scipy.sparse import save_npz # Importation de save_npz

# --- Configuration des fichiers ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "../preprocessing/facebook_posts_clean.csv")
OUTPUT_VECTORIZER = os.path.join(BASE_DIR, "../models/tfidf_vectorizer.pkl")
OUTPUT_MATRIX = os.path.join(BASE_DIR, "../data/processed/tfidf_matrix.npz")

# --- AJOUT : Création des répertoires de sortie ---
# Extrait le chemin du dossier pour la matrice
output_matrix_dir = os.path.dirname(OUTPUT_MATRIX)
# Crée le dossier et ses parents si nécessaire (exist_ok=True évite les erreurs si le dossier existe déjà)
os.makedirs(output_matrix_dir, exist_ok=True) 

# Extrait le chemin du dossier pour le vectorizer (peut être le même, mais bonne pratique)
output_vectorizer_dir = os.path.dirname(OUTPUT_VECTORIZER)
os.makedirs(output_vectorizer_dir, exist_ok=True)
print(f"✔️ Répertoires de sortie créés ou vérifiés : {output_matrix_dir}")


# --- Chargement des données ---
print(f"📂 Chargement des données depuis {INPUT_FILE}...")
df_clean = pd.read_csv(INPUT_FILE)
corpus = df_clean['clean_text'].fillna('').astype(str)

# --- TF-IDF Vectorizer ---
vectorizer = TfidfVectorizer(
    analyzer='word',
    ngram_range=(1,2),
    min_df=5,
    max_features=5000
)

print("⚙️ Création de la matrice TF-IDF...")
tfidf_matrix = vectorizer.fit_transform(corpus)

# --- Sauvegarde ---
joblib.dump(vectorizer, OUTPUT_VECTORIZER)
# Notez que nous utilisons l'importation de save_npz qui était manquante dans le script initial
save_npz(OUTPUT_MATRIX, tfidf_matrix, compressed=True) 

print(f"✅ Vectorizer sauvegardé : {OUTPUT_VECTORIZER}")
print(f"✅ Matrice TF-IDF sauvegardée : {OUTPUT_MATRIX}")
print(f"Dimensions de la matrice : {tfidf_matrix.shape}")