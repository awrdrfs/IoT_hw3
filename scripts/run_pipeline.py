from pathlib import Path
import sys
import os

# Ensure project root on sys.path for direct script execution
PROJECT_ROOT = os.path.abspath(os.getcwd())
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.load_data import load_sms_csv
from features.vectorize import simple_preprocess, fit_vectorizer, save_vectorizer
from models.train import train_baseline, save_model
from models.train import train_recall_focused_model
from viz.plot import plot_class_distribution, plot_text_length_distribution


def main():
    root = Path.cwd()
    data_path = root / "sms_spam_no_header.csv"
    print("Loading:", data_path)
    df = load_sms_csv(data_path)
    print("Loaded rows:", len(df))

    # preprocessing options (simulate Streamlit defaults)
    lowercase = True
    remove_punct = True
    texts = simple_preprocess(df["text"], lowercase=lowercase, remove_punct=remove_punct)

    # quick visualizations
    viz_dir = root / "viz" / "out"
    viz_dir.mkdir(parents=True, exist_ok=True)
    fig1 = plot_class_distribution(df["label"], out_path=viz_dir / "class_dist.png")
    fig2 = plot_text_length_distribution(texts, out_path=viz_dir / "len_dist.png")
    print("Saved visualizations to:", viz_dir)

    # train baseline
    from sklearn.model_selection import train_test_split

    X_train_texts, X_test_texts, y_train, y_test = train_test_split(texts, df["label"], test_size=0.2, random_state=42)
    vec, X_train = fit_vectorizer(X_train_texts, method="tfidf")
    X_test = vec.transform(X_test_texts)

    features_dir = root / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    save_vectorizer(vec, features_dir / "vectorizer.pkl")
    print("Saved vectorizer to:", features_dir / "vectorizer.pkl")

    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    clf, metrics, y_pred = train_baseline(X_train, y_train, X_test, y_test)
    save_model(clf, models_dir / "baseline.joblib")
    print("Saved model to:", models_dir / "baseline.joblib")
    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Train recall-focused model (use a validation split from train)
    from sklearn.model_selection import train_test_split as tts

    X_sub_train, X_val, y_sub_train, y_val = tts(X_train_texts, y_train, test_size=0.2, random_state=1)
    # Re-fit vectorizer on X_sub_train and transform validation set
    vec2, X_sub = fit_vectorizer(X_sub_train, method="tfidf")
    X_val_transformed = vec2.transform(X_val)

    # Save alternative vectorizer too
    save_vectorizer(vec2, features_dir / "vectorizer_recall.pkl")

    model_container, recall_metrics, threshold, success = train_recall_focused_model(X_sub, y_sub_train, X_val_transformed, y_val, baseline_metrics=metrics, max_drop=0.05, target_accuracy=0.9)
    # save container
    save_model(model_container, models_dir / "recall_model.joblib")
    print("Saved recall-focused model to:", models_dir / "recall_model.joblib")
    print("Recall-model metrics (on validation):")
    for k, v in recall_metrics.items():
        print(f"  {k}: {v}")
    print(f"Selected threshold: {threshold}")
    print(f"Meets constraints (recall>baseline & accuracy>=0.9): {success}")

    # save confusion matrix plot
    cm_dir = viz_dir
    plot_conf = plot_class_distribution  # reuse
    # Also save confusion matrix from predictions
    from viz.plot import plot_confusion_matrix

    plot_confusion_matrix(y_test, y_pred, labels=["ham", "spam"], out_path=cm_dir / "confusion.png")
    print("Saved confusion matrix to:", cm_dir / "confusion.png")


if __name__ == "__main__":
    main()
