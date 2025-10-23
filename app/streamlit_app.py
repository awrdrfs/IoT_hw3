import sys
import os

# Ensure project root is on sys.path so `from data...` works when Streamlit changes CWD
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pathlib import Path

import streamlit as st
import pandas as pd

from data.load_data import load_sms_csv, save_cleaned
from features.vectorize import simple_preprocess, fit_vectorizer, save_vectorizer
from models.train import train_baseline, save_model
try:
    from models.train import train_recall_focused_model
    _train_import_error = None
except Exception:
    # capture traceback and continue so Streamlit can show an informative message
    import traceback

    train_recall_focused_model = None
    _train_import_error = traceback.format_exc()
from viz.plot import plot_class_distribution, plot_text_length_distribution, plot_confusion_matrix

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import joblib


def main():
    st.title("Spam Analysis Explorer")

    st.sidebar.header("Data & Preprocessing")
    default_data = str(Path(PROJECT_ROOT) / "sms_spam_no_header.csv")
    data_path = st.sidebar.text_input("Data path", default_data)
    lowercase = st.sidebar.checkbox("Lowercase", value=True)
    remove_punct = st.sidebar.checkbox("Remove punctuation", value=True)
    vectorizer_choice = st.sidebar.selectbox("Vectorizer", ["tfidf", "count"])

    def resolve_data_path(pth: str) -> str:
        from pathlib import Path as _P

        p = _P(pth)
        if p.is_absolute() and p.exists():
            return str(p)
        # try relative to project root
        alt = _P(PROJECT_ROOT) / pth
        if alt.exists():
            return str(alt)
        return str(p)

    if st.sidebar.button("Load and preview"):
        real_path = resolve_data_path(data_path)
        df = load_sms_csv(real_path)
        st.write(df.head())
        st.write("Class counts:")
        st.bar_chart(df["label"].value_counts())

    st.sidebar.markdown("---")
    # Unified training control: choose which model(s) to train
    train_choice = st.sidebar.selectbox("Which trained model to show?", ["Baseline", "Recall", "Both"])
    show_button = st.sidebar.button("Show trained model results")
    if show_button:
        df = load_sms_csv(data_path)
        texts = simple_preprocess(df["text"], lowercase=lowercase, remove_punct=remove_punct)
        # minimal train/test split
        from sklearn.model_selection import train_test_split

        X_train_texts, X_test_texts, y_train, y_test = train_test_split(texts, df["label"], test_size=0.2, random_state=42)

        features_dir = Path(PROJECT_ROOT) / "features"
        models_dir = Path(PROJECT_ROOT) / "models"

        if train_choice in ("Baseline", "Both"):
            vec_path = features_dir / "vectorizer.pkl"
            model_path = models_dir / "baseline.joblib"
            if not vec_path.exists() or not model_path.exists():
                st.error("Baseline model or vectorizer not found. Please run the pipeline to train baseline first.")
            else:
                vec = joblib.load(vec_path)
                clf = joblib.load(model_path)
                X_test = vec.transform(X_test_texts)
                y_pred = clf.predict(X_test)
                # compute metrics
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, pos_label="spam", zero_division=0)
                rec = recall_score(y_test, y_pred, pos_label="spam", zero_division=0)
                f1v = f1_score(y_test, y_pred, pos_label="spam", zero_division=0)
                st.markdown("### Baseline evaluation metrics (test)")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Accuracy", f"{acc*100:.2f}%")
                c2.metric("Precision", f"{prec*100:.2f}%")
                c3.metric("Recall", f"{rec*100:.2f}%")
                c4.metric("F1 score", f"{f1v*100:.2f}%")
                fig_cm = plot_confusion_matrix(y_test, y_pred, labels=["ham", "spam"], out_path=None)
                st.pyplot(fig_cm)

        if train_choice in ("Recall", "Both"):
            recall_model_path = models_dir / "recall_model.joblib"
            vec_recall_path = features_dir / "vectorizer_recall.pkl"
            if not recall_model_path.exists() or not vec_recall_path.exists():
                st.error("Recall model or its vectorizer not found. Please run the pipeline to generate recall_model.joblib and vectorizer_recall.pkl.")
            else:
                container = joblib.load(recall_model_path)
                vec_recall = joblib.load(vec_recall_path)
                # container is expected to be {'model': clf, 'threshold': t} or actual model object in older saves
                if isinstance(container, dict) and "model" in container:
                    clf_rec = container["model"]
                    threshold = float(container.get("threshold", 0.5))
                else:
                    clf_rec = container
                    threshold = 0.5

                X_test_rec = vec_recall.transform(X_test_texts)
                if hasattr(clf_rec, "predict_proba"):
                    probs = clf_rec.predict_proba(X_test_rec)[:, 1]
                    y_pred_rec = ["spam" if p >= threshold else "ham" for p in probs]
                else:
                    y_pred_rec = clf_rec.predict(X_test_rec)

                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                acc_r = accuracy_score(y_test, y_pred_rec)
                prec_r = precision_score(y_test, y_pred_rec, pos_label="spam", zero_division=0)
                rec_r = recall_score(y_test, y_pred_rec, pos_label="spam", zero_division=0)
                f1_r = f1_score(y_test, y_pred_rec, pos_label="spam", zero_division=0)

                st.markdown("### Recall-model evaluation (test, using saved threshold)")
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("Accuracy", f"{acc_r*100:.2f}%")
                r2.metric("Precision", f"{prec_r*100:.2f}%")
                r3.metric("Recall", f"{rec_r*100:.2f}%")
                r4.metric("F1 score", f"{f1_r*100:.2f}%")

                # Threshold checks
                PASS_ACC = acc_r >= 0.90
                PASS_PREC = prec_r >= 0.90
                PASS_REC = rec_r >= 0.82
                overall_pass = PASS_ACC and PASS_PREC and PASS_REC

                st.markdown("#### Recall-model acceptance checks")
                c_acc, c_prec, c_rec = st.columns(3)
                c_acc.write("✅" if PASS_ACC else "❌")
                c_acc.caption(f"Accuracy >= 90%: {acc_r*100:.2f}%")
                c_prec.write("✅" if PASS_PREC else "❌")
                c_prec.caption(f"Precision >= 90%: {prec_r*100:.2f}%")
                c_rec.write("✅" if PASS_REC else "❌")
                c_rec.caption(f"Recall >= 82%: {rec_r*100:.2f}%")

                if overall_pass:
                    st.success("Recall-model meets acceptance criteria: Accuracy >=90%, Precision >=90%, Recall >=82%")
                else:
                    st.warning("Recall-model does NOT meet acceptance criteria. See metric checks above.")

                st.write(f"Used decision threshold: {threshold}")
                fig_cm_r = plot_confusion_matrix(y_test, y_pred_rec, labels=["ham", "spam"], out_path=None)
                st.pyplot(fig_cm_r)

    # Recall-focused training is intentionally disabled in the Streamlit UI.
    # Users should run the headless pipeline to train or re-generate recall-optimized models:
    #   python3 scripts/run_pipeline.py
    st.sidebar.markdown("## Recall model (training disabled in UI)")
    st.sidebar.info("Recall-focused training is disabled in the Streamlit UI to avoid long-running jobs and nondeterministic behaviour.\nRun the headless pipeline: `python3 scripts/run_pipeline.py` to train or update recall models.")
    if _train_import_error is not None:
        # keep informative error visible for developers who may inspect the environment
        st.sidebar.caption("Note: recall training function import had an error; check logs if you expect to run training outside the UI.")

    st.markdown("---")
    st.header("Predict single message")
    examples = [
        "Congratulations! You have won a prize. Claim now.",
        "Reminder: Your package will be delivered tomorrow.",
        "URGENT: Your account will be closed. Click this link to verify.",
        "Hey, are we still on for lunch today?",
    ]
    # Example selector - populates the text area via session_state
    if "user_text" not in st.session_state:
        st.session_state.user_text = ""
    sel = st.selectbox("Example templates", ["(none)"] + examples)
    if sel != "(none)":
        st.session_state.user_text = sel

    user_text = st.text_area("Enter message text to classify as spam or ham", value=st.session_state.user_text, height=150)
    if st.button("Predict"):
        # paths to saved artifacts
        vec_path = Path(PROJECT_ROOT) / "features" / "vectorizer.pkl"
        model_path = Path(PROJECT_ROOT) / "models" / "baseline.joblib"
        if not vec_path.exists() or not model_path.exists():
            st.error("Model or vectorizer not found. Run training first (Run preprocessing & train baseline).")
        else:
            try:
                vec = joblib.load(vec_path)
                clf = joblib.load(model_path)
                # preprocess and vectorize
                txts = simple_preprocess([user_text])
                X = vec.transform(txts)
                if hasattr(clf, "predict_proba"):
                    prob = clf.predict_proba(X)[0]
                    spam_prob = float(prob[1])
                else:
                    pred = clf.predict(X)[0]
                    spam_prob = 1.0 if pred == "spam" else 0.0
                pred_label = "spam" if spam_prob >= 0.5 else "ham"
                # Colorized output: red for spam, green for not spam
                pct = float(spam_prob)
                pct_str = f"{pct*100:.1f}%"
                # Show a progress bar for spam probability (0-100%)
                st.progress(pct)
                if pred_label == "spam":
                    st.markdown(
                        f"<div style='color:white; background:#e74c3c; padding:8px; border-radius:6px'>"
                        f"<strong>Prediction: SPAM</strong> — spam probability: {pct_str}</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"<div style='color:white; background:#2ecc71; padding:8px; border-radius:6px'>"
                        f"<strong>Prediction: NOT SPAM</strong> — spam probability: {pct_str}</div>",
                        unsafe_allow_html=True,
                    )
            except Exception as e:
                st.error(f"Prediction failed: {e}")


if __name__ == "__main__":
    main()
