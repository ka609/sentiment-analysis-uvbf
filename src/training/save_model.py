# src/training/save_model.py

import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# Charger les données nettoyées
df = pd.read_csv("../preprocessing/facebook_posts_clean.csv")

X = df["clean_text"]
y = df["sentiment"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Vectorisation
vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)

# Modèle
model = LinearSVC()
model.fit(X_train_vec, y_train)

# Dossier models (création automatique)
models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models"))
os.makedirs(models_dir, exist_ok=True)

# Sauvegarde avec chemin absolu
model_path = os.path.join(models_dir, "sentiment_svm_model.pkl")
vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")

joblib.dump(model, model_path)
joblib.dump(vectorizer, vectorizer_path)

print(f"✅ Modèle sauvegardé: {model_path}")
print(f"✅ Vectorizer sauvegardé: {vectorizer_path}")
