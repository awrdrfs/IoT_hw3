import os
from pathlib import Path

from data.load_data import load_sms_csv


def test_load_sms_csv_exists(tmp_path):
    # create a small csv
    p = tmp_path / "sms.csv"
    p.write_text("ham,hello\nspam,win prize")
    df = load_sms_csv(p)
    assert list(df.columns) == ["label", "text"]
    assert len(df) == 2
