from pathlib import Path
import argparse
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


ANNOTATION_CSV = Path("data/processed/tweets_for_annotation.csv")
MODEL_PATH = Path("models/sentiment_model.joblib")
REPORT_PATH = Path("reports/evaluation_report.txt")
CM_PNG = Path("reports/figures/confusion_matrix.png")

def load_data(csv_path: Path):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["label"])
    df = df[df["label"].isin(["positive", "neutral", "negative"])]
    return df

def plot_confusion(cm, labels, outpath: Path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath)
    plt.close()

def main(args):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"{MODEL_PATH} not found. Train a model first.")

    if not ANNOTATION_CSV.exists():
        raise FileNotFoundError(f"{ANNOTATION_CSV} not found. Provide evaluation data.")

    df = load_data(ANNOTATION_CSV)
    texts, labels = df["text"].tolist(), df["label"].tolist()

    
    package = joblib.load(MODEL_PATH)
    model = package["model"]
    vectorizer = package["vectorizer"]

    X = vectorizer.transform(texts)
    y_pred = model.predict(X)

    acc = accuracy_score(labels, y_pred)
    report = classification_report(labels, y_pred, digits=4)
    cm = confusion_matrix(labels, y_pred, labels=["positive", "neutral", "negative"])

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n{report}")

    plot_confusion(cm, ["positive", "neutral", "negative"], CM_PNG)

    print(f"✅ Evaluation report saved to {REPORT_PATH}")
    print(f"✅ Confusion matrix saved to {CM_PNG}")
    print(f"Accuracy: {acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=str(ANNOTATION_CSV),
                        help="CSV file with texts and labels for evaluation")
    args = parser.parse_args()
    main(args)
