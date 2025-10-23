# Spam analysis playground

This project contains utilities to load and preprocess the `sms_spam_no_header.csv` dataset, extract features, visualize outputs, and run a small Streamlit app to explore preprocessing and baseline model metrics.

Quick start

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run Streamlit app:

```bash
streamlit run app/streamlit_app.py
```

3. Run tests:

```bash
pytest -q
```

Notes

- The Streamlit app uses relative paths that assume you run Streamlit from the repository root or adjust the data path in the sidebar.
- Files created by the app (vectorizer, model) are saved under `../features/` and `../models/` relative to the app file; adjust paths as needed for your environment.

## 註：對話流程在 conversation_log
