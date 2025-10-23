# Project Context

## Purpose
本專案以「垃圾簡訊（Spam SMS）辨識」為出發點，基於提供的資料集 `sms_spam_no_header.csv` 擴充資料前處理、特徵工程與視覺化流程，並加入更豐富的步驟輸出、評估指標，以及以 Streamlit 建立的互動式檢視面板，用於教學、探索式分析與模型展示。

目標包括：
- 建立可重現的資料清理與前處理 pipeline
- 提供豐富的中間輸出（step outputs）以利除錯與教學
- 產出常見分類模型（baseline）與評估報告
- 以 Streamlit 提供互動式視覺化與結果檢視

## Tech Stack
- Python 3.10+（或專案所使用的版本）
- 資料科學套件：pandas, numpy, scikit-learn
- 視覺化：matplotlib、seaborn、plotly（選用）
- 互動式應用：streamlit
- 測試與品質：pytest, flake8 / black（選用）
- 開發環境管理：venv 或 conda

## Project Conventions

### Code Style
- Python 程式碼遵循 PEP8。格式工具優先建議使用 `black`（自動格式化）與 `flake8`（靜態檢查）。
- 檔名使用小寫與底線（snake_case），類別使用 PascalCase。
- 函式與變數命名使用有語意的名稱，避免使用單字縮寫除非是廣為接受的術語（如 "df", "svc"）。

### Architecture Patterns
- 小型資料專案採模組化結構：
  - data/       # 讀取、清理、前處理邏輯
  - features/   # 特徵工程、轉換器
  - models/     # 訓練、評估、序列化
  - viz/        # 可重複的視覺化函式
  - app/        # Streamlit 應用程式
  - tests/      # 單元／整合測試
- 每個模組負責單一職責，採小型函式且易於測試。

### Testing Strategy
- 使用 `pytest` 撰寫最小化但具代表性的測試：
  - 資料讀取與欄位一致性
  - 前處理 pipeline 的輸入/輸出形狀和內容檢查
  - 模型訓練流程的主要函式（mock 小資料集）
- 在更改前處理或特徵工程時，加入對應的測試以防回歸。

### Git Workflow
- 使用 feature 分支開發：`feature/<short-desc>` 或 `changes/<change-id>`（與 OpenSpec change-id 一致）
- commit message 使用簡短前綴（例如：`feat:`, `fix:`, `chore:`），並在 PR 描述中參考 `openspec/changes/<change-id>`（當有 proposal 時）。

## Domain Context
- `sms_spam_no_header.csv` 為本專案主要資料來源，資料包含簡訊文字與標籤（spam/ham）。
- 典型資料問題包含：大小寫差異、標點、數字、簡短文本、噪聲與類別不平衡。
- 常見前處理步驟：小寫化、移除標點、斷詞/tokenization、詞幹化或詞形還原（選擇性）、stopwords 去除、向量化（TF-IDF / CountVectorizer）

## Important Constraints
- 資料大小通常較小（數千至數萬筆），設計上以可重現性與解釋性為優先，而非超大型分散式處理。
- 不在此專案內處理個人資料保護法（若資料包含 PII，需另行處理）；目前資料來自公開數據集。

## External Dependencies
- 無需外部網路 API；若使用 Plotly 或其他互動視覺化套件，請列在 requirements.txt
- 若計畫部署或分享 Streamlit 應用，請確認目標環境（Streamlit Cloud 或自建 server）之 Python 與套件版本一致。
