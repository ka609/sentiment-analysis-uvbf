# src/visualisation/plot_results.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# --- 1. Charger les données ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "../preprocessing/facebook_posts_clean_predicted.csv")

print(f"📂 Chargement des résultats depuis {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE)

# --- 2. Distribution des sentiments ---
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Predicted_Sentiment", palette="Set2")
plt.title("Distribution des sentiments prédits")
plt.xlabel("Sentiment")
plt.ylabel("Nombre d'avis")
plt.show()

# --- 3. Camembert ---
plt.figure(figsize=(5, 5))
df["Predicted_Sentiment"].value_counts().plot(
    kind="pie", autopct="%1.1f%%", colors=["#FF6B6B", "#4ECDC4", "#FFD93D"]
)
plt.title("Répartition des avis (camembert)")
plt.ylabel("")
plt.show()

# --- 4. Nuages de mots par sentiment ---
for sentiment in df["Predicted_Sentiment"].unique():
    avis = " ".join(df[df["Predicted_Sentiment"] == sentiment]["text"].astype(str))
    if avis.strip():
        print(f"☁️ Génération du nuage de mots pour : {sentiment}")
        wc = WordCloud(width=800, height=400, background_color="white", colormap="viridis").generate(avis)

        plt.figure(figsize=(8, 5))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Nuage de mots - {sentiment}")
        plt.show()
