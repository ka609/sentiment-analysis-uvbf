import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Charger le fichier nettoyé
df = pd.read_csv("../preprocessing/facebook_posts_clean.csv")

# ---- 1. Statistiques de base ----
print("Nombre total d’avis :", len(df))
print("\nRépartition des sentiments :")
print(df["sentiment"].value_counts())

# ---- 2. Visualisation : Répartition des sentiments ----
plt.figure(figsize=(6,6))
df["sentiment"].value_counts().plot.pie(
    autopct="%1.1f%%", 
    colors=["green","red","gray"],
    startangle=90
)
plt.title("Répartition des sentiments UVBF")
plt.ylabel("")
plt.show()

# ---- 3. WordCloud par sentiment ----
for sentiment in df["sentiment"].unique():
    subset = df[df["sentiment"] == sentiment]
    text = " ".join(subset["clean_text"].astype(str))
    if text.strip():  # éviter erreur si texte vide
        wordcloud = WordCloud(
            width=800, height=400, 
            background_color="white", 
            colormap="viridis"
        ).generate(text)

        plt.figure(figsize=(10,5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"WordCloud - {sentiment}")
        plt.show()
