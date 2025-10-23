import csv
from pathlib import Path
import pandas as pd
from typing import Tuple


def load_sms_csv(path: str | Path) -> pd.DataFrame:
    """Load sms_spam_no_header.csv which has no header and return DataFrame with columns ['label','text'].

    Raises ValueError if schema unexpected.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {p}")

    # The dataset has two columns: label and text, without header
    df = pd.read_csv(p, header=None, names=["label", "text"], encoding="utf-8", quoting=csv.QUOTE_MINIMAL)

    # Basic schema validation
    if "label" not in df.columns or "text" not in df.columns:
        raise ValueError("Loaded CSV does not contain expected columns 'label' and 'text'")

    # Normalize label to lower-case
    df["label"] = df["label"].astype(str).str.lower().str.strip()

    return df


def save_cleaned(df: pd.DataFrame, out_path: str | Path) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False, encoding="utf-8")


if __name__ == "__main__":
    # quick CLI for manual test
    import sys

    in_path = sys.argv[1] if len(sys.argv) > 1 else "../sms_spam_no_header.csv"
    df = load_sms_csv(in_path)
    print(df.head())
    save_cleaned(df, "../data/cleaned.csv")
