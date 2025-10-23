## 1. Implementation
- [ ] 1.1 建立 `data/`：讀取 `sms_spam_no_header.csv` 並輸出清理後的 CSV（含 schema 檢查）
- [ ] 1.2 建立 `features/`：實作 TF-IDF 與 CountVectorizer 選項，並可選擇小寫、移除標點、stopwords
- [ ] 1.3 建立 `viz/`：實作步驟輸出視覺化（類別分布、文字長度分布、詞雲或 top-n 詞條）
- [ ] 1.4 建立 `app/streamlit_app.py`：提供前處理參數切換、模型訓練按鈕與指標視覺化
- [ ] 1.5 撰寫最小化測試：資料載入、前處理輸出 shape、向量化一致性
- [ ] 1.6 撰寫 README 使用說明與如何啟動 Streamlit

## 2. Documentation
- [ ] 2.1 在 `openspec/changes/add-rich-preprocessing-and-streamlit-views/` 加入 spec delta（已包含）
- [ ] 2.2 更新 `openspec/project.md`（已更新）
