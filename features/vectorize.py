import re
import pickle
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def simple_preprocess(texts: Iterable[str], lowercase=True, remove_punct=True):
    out = []
    for t in texts:
        s = str(t)
        if lowercase:
            s = s.lower()
        if remove_punct:
            s = re.sub(r"[\W_]+", " ", s)
        out.append(s.strip())
    return out


def fit_vectorizer(texts: Iterable[str], method: str = "tfidf", max_features: int | None = 5000):
    if method == "tfidf":
        vec = TfidfVectorizer(max_features=max_features)
    else:
        vec = CountVectorizer(max_features=max_features)
    X = vec.fit_transform(texts)
    return vec, X


def save_vectorizer(vec, out_path: str | Path):
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(vec, f)


def load_vectorizer(path: str | Path):
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    from pathlib import Path
    import sys

    sample = ["Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005.", "Nah I don't think he goes to usf, he lives around here though"]
    texts = simple_preprocess(sample)
    vec, X = fit_vectorizer(texts)
    print(X.shape)
    save_vectorizer(vec, Path("../features/vectorizer.pkl"))
