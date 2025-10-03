# src/prediction/predict.py
import os
import sys
import pandas as pd
import joblib

# --- 1. Configuration des chemins ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_PATH = os.path.join(BASE_DIR, "../../models/best_pipeline.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "../../models/label_encoder.pkl")

# --- 2. Chargement du pipeline et du label encoder ---
print("📂 Chargement du pipeline et du label encoder...")
pipeline = joblib.load(PIPELINE_PATH)
label_encoder = joblib.load(ENCODER_PATH)

# --- 3. Fonction prédiction ---
def predict_text(texts):
    preds = pipeline.predict(texts)
    labels = label_encoder.inverse_transform(preds)

    confidences = None
    if hasattr(pipeline.named_steps["clf"], "predict_proba"):
        probs = pipeline.named_steps["clf"].predict_proba(
            pipeline.named_steps["tfidf"].transform(texts)
        )
        confidences = probs.max(axis=1)
    else:
        confidences = [None] * len(labels)

    return list(zip(labels, confidences))

# --- 4. Mode CSV ---
def predict_from_csv(csv_path, text_column="text"):
    if not os.path.exists(csv_path):
        print(f"❌ Fichier introuvable : {csv_path}")
        return
    df = pd.read_csv(csv_path)
    if text_column not in df.columns:
        print(f"❌ La colonne '{text_column}' n'existe pas dans {csv_path}")
        return

    labels_conf = predict_text(df[text_column].astype(str).tolist())
    df["Predicted_Sentiment"] = [lc[0] for lc in labels_conf]
    df["Confidence"] = [lc[1] for lc in labels_conf]

    output_file = csv_path.replace(".csv", "_predicted.csv")
    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"✅ Résultats sauvegardés dans : {output_file}")

    # Aperçu + distribution
    print(df[[text_column, "Predicted_Sentiment", "Confidence"]].head())
    print("\n📊 Distribution des sentiments prédits :")
    print(df["Predicted_Sentiment"].value_counts())

# --- 5. Mode ligne de commande ---
if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Mode interactif
        while True:
            user_input = input("\n💬 Entrez un texte (ou 'quit' pour sortir) : ")
            if user_input.lower() in ["quit", "exit"]:
                break
            label, confidence = predict_text([user_input])[0]
            if confidence is not None:
                print(f"📝 Texte: {user_input}\n➡️ Sentiment prédit: {label} (confiance: {confidence:.2f})")
            else:
                print(f"📝 Texte: {user_input}\n➡️ Sentiment prédit: {label} (confiance non disponible)")

    elif len(sys.argv) == 2:
        # Mode CSV
        csv_file = sys.argv[1]
        predict_from_csv(csv_file)
    else:
        print("❌ Utilisation :")
        print("   python predict.py             # Mode interactif")
        print("   python predict.py fichier.csv # Mode CSV")
