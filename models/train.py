from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple, Dict


def train_baseline(X_train, y_train, X_test, y_test):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, pos_label="spam", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, pos_label="spam", zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, pos_label="spam", zero_division=0)),
    }
    return clf, metrics, y_pred


def save_model(model, out_path: str | Path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)


def _compute_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label="spam", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label="spam", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label="spam", zero_division=0)),
    }


def train_recall_focused_model(
        X_train,
        y_train,
        X_val,
        y_val,
        baseline_metrics: dict | None = None,
        max_drop: float = 0.05,
        target_accuracy: float = 0.9,
        min_recall: float = 0.82,
        min_precision: float = 0.9,
) -> Tuple[object, dict, float, bool]:
    candidates = []
    # Build an initial grid of candidate constructors with placeholders for class_weight adjustment
    logistic_Cs = (0.01, 0.1, 1.0, 10.0)
    rf_depths = (5, 10, None)
    rf_estimators = (100, 200)

    # class weight strategies: None, 'balanced', and custom where spam weight is scaled
    class_weight_options = [None, 'balanced']
    for spam_w in (1.0, 1.5, 2.0, 3.0, 5.0):
        class_weight_options.append({"spam": spam_w, "ham": 1.0})

    for cw in class_weight_options:
        for C in logistic_Cs:
            candidates.append(LogisticRegression(max_iter=1000, class_weight=cw, C=C))
        for depth in rf_depths:
            for n in rf_estimators:
                candidates.append(RandomForestClassifier(n_estimators=n, class_weight=cw, max_depth=depth, random_state=42))

    best = None
    best_score = -1.0
    success = False
    satisfying_candidates = []
    baseline = baseline_metrics or {}

    # Iterate over candidates; train and search threshold for each
    for clf in candidates:
        try:
            clf.fit(X_train, y_train)
        except Exception:
            # skip candidates that fail to fit for any reason
            continue

        # need predicted probabilities for the positive class 'spam'
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(X_val)[:, 1]
        else:
            # fallback: use decision_function and scale to [0,1]
            try:
                dec = clf.decision_function(X_val)
                probs = (dec - dec.min()) / (dec.max() - dec.min())
            except Exception:
                probs = clf.predict(X_val)

        # sweep thresholds
        thresholds = [round(x, 2) for x in list(frange(0.01, 0.99, 0.01))]
        best_local = None
        best_local_score = -1.0
        for t in thresholds:
            y_pred_t = ["spam" if p >= t else "ham" for p in probs]
            mets = _compute_metrics(y_val, y_pred_t)

            # check strict minima (accuracy, precision, recall)
            ok = True
            min_accuracy = float(target_accuracy)
            if mets.get("accuracy", 0.0) < min_accuracy:
                ok = False
            if mets.get("precision", 0.0) < float(min_precision):
                ok = False
            if mets.get("recall", 0.0) < float(min_recall):
                ok = False

            # if baseline provided, ensure no metric drops more than max_drop
            if ok and baseline:
                for k in ("accuracy", "precision", "recall", "f1"):
                    base_v = float(baseline.get(k, 0.0))
                    if base_v - mets.get(k, 0.0) > max_drop:
                        ok = False
                        break

            # prefer higher recall, tie-breaker by f1
            if ok:
                score = mets["recall"]
                if score > best_local_score:
                    best_local_score = score
                    best_local = (t, mets)

        # if none satisfied constraint, pick threshold that gives best recall regardless but keep track of drop
        if best_local is None:
            # find best recall but also keep minimal drop
            best_relaxed = None
            best_relaxed_score = -1.0
            for t in thresholds:
                y_pred_t = ["spam" if p >= t else "ham" for p in probs]
                mets = _compute_metrics(y_val, y_pred_t)
                score = mets["recall"]
                if score > best_relaxed_score:
                    best_relaxed_score = score
                    best_relaxed = (t, mets)
            best_local = best_relaxed

        if best_local:
            t, mets = best_local
            # scoring priority: recall, then f1
            primary = mets["recall"]
            secondary = mets["f1"]
            if (primary > best_score) or (primary == best_score and secondary > (best[1]["f1"] if best else -1)):
                best = (clf, mets, t)
                best_score = primary

            # check if this candidate meets the strict minima and baseline drop constraints
            meets_constraints = False
            min_accuracy = float(target_accuracy)
            if (
                mets.get("accuracy", 0.0) >= min_accuracy
                and mets.get("precision", 0.0) >= float(min_precision)
                and mets.get("recall", 0.0) >= float(min_recall)
            ):
                ok_drop = True
                if baseline:
                    for k in ("accuracy", "precision", "recall", "f1"):
                        base_v = float(baseline.get(k, 0.0))
                        if base_v - mets.get(k, 0.0) > max_drop:
                            ok_drop = False
                            break
                if ok_drop:
                    meets_constraints = True
                    success = True
                    satisfying_candidates.append((clf, mets, t))

            # diagnostic print
            try:
                params = {k: v for k, v in clf.get_params().items() if k in ("class_weight", "C", "n_estimators", "max_depth")}
            except Exception:
                params = {}
            print(f"Candidate {clf.__class__.__name__} params={params} best_t={t} mets={mets} meets={meets_constraints}")

    if best is None:
        raise RuntimeError("No candidate models trained successfully")

    # If we collected satisfying candidates, pick the one with highest recall (then f1)
    if satisfying_candidates:
        satisfying_candidates.sort(key=lambda item: (item[1]["recall"], item[1]["f1"]), reverse=True)
        clf_best, metrics_best, threshold_best = satisfying_candidates[0]
        model_container = {"model": clf_best, "threshold": threshold_best}
        return model_container, metrics_best, float(threshold_best), True

    # otherwise return best found (may not meet constraints)
    clf_best, metrics_best, threshold_best = best
    model_container = {"model": clf_best, "threshold": threshold_best}
    return model_container, metrics_best, float(threshold_best), success


def frange(start, stop, step):
    x = start
    while x <= stop:
        yield x
        x += step
