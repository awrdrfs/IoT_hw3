## Why
資料探索與教學常常受限於缺少清楚的中間步驟輸出與互動視覺化。現有的 `sms_spam_no_header.csv` 使用情境需要：更穩固的前處理 pipeline、可追溯的 step outputs、豐富的評估指標，以及一個簡單的 Streamlit 檢視介面，讓使用者（學生或資料分析師）能直觀比較不同前處理或向量化選項對模型表現的影響。

## What Changes
- 新增資料前處理與特徵工程模組（data/ 與 features/）並輸出中間檔案/摘要
- 新增視覺化模組（viz/），包含步驟輸出圖與評估視覺化
- 新增 Streamlit 應用雛形（app/streamlit_app.py）以互動檢視 pipeline 結果與模型指標
- 新增 OpenSpec delta：`specs/spam-analysis/spec.md`，描述新增需求與驗收情境

## Impact
- 受影響的 capability: spam-analysis
- 受影響的檔案/資料夾：data/, features/, viz/, app/streamlit_app.py, openspec/changes/add-rich-preprocessing-and-streamlit-views/
