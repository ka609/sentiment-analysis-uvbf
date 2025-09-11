from pathlib import Path
import argparse
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

ANNOTATION_CSV = Path("data/processed/tweets_for_annotation.csv")
VECTORIZER_PATH = Path("data/vectorizer/tfidf_vectorizer.joblib")
MODEL_PATH = Path("data/result/sentiment_model.joblib")
REPORT_PATH = Path("data/reports/evaluation_report.txt")
CM_PNG = Path("data/reports/confusion_matrix.png")

def load_data(csv_path: Path):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["label"])
    df = df[df["label"].isin(["positive","neutral","negative"])]
    return df

def choose_model(name: str):
    if name == "logistic":
        return LogisticRegression(max_iter=1000)
    if name == "nb":
        return MultinomialNB()
    if name == "svm":
        return LinearSVC()
    raise ValueError("Choose 'logistic', 'nb', or 'svm'")

def plot_confusion(cm, labels, outpath: Path):
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath)
    plt.close()

def main(args):
    if not ANNOTATION_CSV.exists():
        raise FileNotFoundError(f"{ANNOTATION_CSV} not found. Annotate it first.")
    
    df = load_data(ANNOTATION_CSV)
    texts, labels = df["text"].tolist(), df["label"].tolist()

    vect = joblib.load(VECTORIZER_PATH)
    X = vect.transform(texts)

    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42, stratify=labels)

    model = choose_model(args.model)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred, labels=["positive","neutral","negative"])

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "vectorizer": vect}, MODEL_PATH)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n{report}")

    plot_confusion(cm, ["positive","neutral","negative"], CM_PNG)

    print(f"✅ Model saved to {MODEL_PATH}")
    print(f"✅ Report saved to {REPORT_PATH}")
    print(f"✅ Confusion matrix saved to {CM_PNG}")
    print(f"Total: {len(df)} | Positive: {(df['label']=='positive').sum()} | Negative: {(df['label']=='negative').sum()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["logistic","nb","svm"], default="logistic")
    args = parser.parse_args()
    main(args)
