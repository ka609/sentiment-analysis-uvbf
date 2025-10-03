import pandas as pd
import re
import numpy as np

# --- 1. Configuration des fichiers et colonnes ---
INPUT_FILENAME = './facebook_posts_clean.csv'
OUTPUT_FILENAME = './avis_uvbf_facebook_clean.csv' # Nouveau nom de fichier pour la version finale

# Noms des colonnes que nous voulons donner après avoir ignoré la première ligne
COL_NAMES = ['Date_Heure_GMT', 'Avis_Positif', 'Avis_Negatif']


# --- 2. Fonction de Nettoyage de Texte (inchangée) ---
def clean_text(text):
    if pd.isna(text) or text is None or str(text).strip() == '':
        return ''
    
    text = str(text).lower()
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'[^a-z0-9àâéèêëîïôœùûüÿç\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# --- 3. Chargement, Nettoyage et Standardisation des Données ---

try:
    df = pd.read_csv(
        INPUT_FILENAME, 
        sep='","',
        header=None,
        names=COL_NAMES,
        engine='python',
        skiprows=1  # Ignore la première ligne (les questions/en-tête)
    )
    
    # Nettoyage initial des guillemets
    for col in COL_NAMES:
        df[col] = df[col].astype(str).str.replace('^"|"$', '', regex=True).str.strip()

    # Application du nettoyage de texte
    df['Avis_Positif_Clean'] = df['Avis_Positif'].apply(clean_text)
    df['Avis_Negatif_Clean'] = df['Avis_Negatif'].apply(clean_text)

    # Standardisation de la colonne Date/Heure
    # Coerce les dates non conformes à NaT
    df['Date_Heure'] = pd.to_datetime(df['Date_Heure_GMT'], format='%Y/%m/%d %I:%M:%S %p %Z', errors='coerce')


    # --- 4. Étape de Robustesse : Suppression des Lignes Non Valides ---
    
    initial_count = len(df)

    # Condition 1: Supprimer les lignes où la date est NaT (invalide)
    df.dropna(subset=['Date_Heure'], inplace=True)
    
    # Condition 2: Supprimer les lignes où les deux avis nettoyés sont vides (doublons ou lignes vides)
    df = df[
        (df['Avis_Positif_Clean'] != '') | (df['Avis_Negatif_Clean'] != '')
    ]

    final_count = len(df)
    
    print(f"\nNombre de lignes invalides supprimées : {initial_count - final_count}")


    # --- 5. Enregistrement du Résultat ---

    final_cols = ['Date_Heure', 'Avis_Positif_Clean', 'Avis_Negatif_Clean', 'Avis_Positif', 'Avis_Negatif']
    df[final_cols].to_csv(OUTPUT_FILENAME, index=False, encoding='utf-8')

    print("\n--- ✅ Nettoyage Terminé (Version Robuste) ! ---")
    print(f"Les données finales utilisables ont été sauvegardées dans : **{OUTPUT_FILENAME}**")
    print(f"Nombre d'avis FINAUX utilisables : {final_count}")
    
    print("\n--- Aperçu FINAL des données nettoyées ---")
    print(df[['Date_Heure', 'Avis_Positif_Clean', 'Avis_Negatif_Clean']].head())

except Exception as e:
    print(f"Une erreur fatale est survenue : {e}")