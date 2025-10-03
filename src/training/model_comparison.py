# src/training/model_comparison.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Charger les données nettoyées
df = pd.read_csv("../preprocessing/facebook_posts_clean.csv")

print("Colonnes disponibles :", df.columns)

# Variables (X = texte, y = sentiment)
X = df["clean_text"]
y = df["sentiment"]

# Découper en train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Transformer le texte en vecteurs TF-IDF
vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Définir les modèles à comparer
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Naive Bayes": MultinomialNB(),
    "SVM (LinearSVC)": LinearSVC(),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

# Entraînement et évaluation
results = {}

for name, model in models.items():
    print(f"\n🔹 Entraînement du modèle: {name}")
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {acc:.2f}")
    print(classification_report(y_test, y_pred))

    results[name] = acc

# Résumé des scores
print("\n📊 Résumé des performances:")
for model, score in results.items():
    print(f"{model}: {score:.2f}")

# -------- VISUALISATION --------
plt.figure(figsize=(8, 5))
plt.bar(results.keys(), results.values(), color=["skyblue", "lightgreen", "salmon", "orange"])
plt.title("Comparaison des performances des modèles", fontsize=14)
plt.ylabel("Accuracy", fontsize=12)
plt.xticks(rotation=15)
plt.ylim(0, 1)  # échelle entre 0 et 1
plt.tight_layout()
plt.show()
