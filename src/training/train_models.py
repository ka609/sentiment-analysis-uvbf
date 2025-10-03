# src/training/train_models.py
import os
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight

# --- 1. Configuration des fichiers ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_LABELS = os.path.join(BASE_DIR, "../data/processed/annotated_sentiment_data.csv")
OUTPUT_PIPELINE = os.path.join(BASE_DIR, "../../models/best_pipeline.pkl")
OUTPUT_LABEL_ENCODER = os.path.join(BASE_DIR, "../../models/label_encoder.pkl")

# --- 2. Chargement des données ---
print(f"📂 Chargement des labels depuis {INPUT_LABELS}...")
df = pd.read_csv(INPUT_LABELS)
texts = df["text"].astype(str)
y = df["Sentiment"]

# --- 3. Encodage des labels ---
le = LabelEncoder()
y_encoded = le.fit_transform(y)
joblib.dump(le, OUTPUT_LABEL_ENCODER)
print(f"✅ Label Encoder sauvegardé : {OUTPUT_LABEL_ENCODER}")
print(f"Distribution des labels: {pd.Series(y_encoded).value_counts().to_dict()}")

# --- 4. Split train/test ---
X_train, X_test, y_train, y_test = train_test_split(
    texts, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"Taille du jeu d'entraînement : {len(X_train)} avis")
print(f"Taille du jeu de test : {len(X_test)} avis")

# --- 5. Définition des modèles avec class_weight pour déséquilibre ---
class_weights = dict(enumerate(compute_class_weight(
    class_weight='balanced', classes=np.unique(y_encoded), y=y_encoded
)))

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight=class_weights, random_state=42
    ),
    "Naive Bayes": MultinomialNB(),  # NB ne supporte pas class_weight
    "SVM (LinearSVC)": LinearSVC(
        random_state=42, dual=False, class_weight=class_weights
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight=class_weights
    )
}

results = {}
best_pipeline = None
best_score = 0

# --- 6. Entraînement et évaluation ---
for name, model in models.items():
    print(f"\n🚀 Entraînement du modèle : {name}")
    
    # Pipeline TF-IDF + modèle
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000)),
        ("clf", model)
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    results[name] = {"accuracy": acc, "f1": f1}
    
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print(f"✅ Accuracy: {acc:.2f} | F1-score (weighted): {f1:.2f}")
    
    if f1 > best_score:
        best_score = f1
        best_pipeline = pipeline

# --- 7. Résumé et sauvegarde ---
print("\n📊 Résumé des performances :")
for name, score in results.items():
    print(f"{name}: Accuracy={score['accuracy']:.2f}, F1={score['f1']:.2f}")

# Sauvegarde du meilleur pipeline complet
if best_pipeline:
    os.makedirs(os.path.join(BASE_DIR, "../../models/"), exist_ok=True)
    joblib.dump(best_pipeline, OUTPUT_PIPELINE)
    print(f"\n✅ Meilleur pipeline sauvegardé : {OUTPUT_PIPELINE} (F1-score: {best_score:.2f})")
