# TQQQ 回撤实验室

免費參數化回測工具。A 組、B 組、熊市防守、回撤加倉都在同一頁設定，跑完後直接生成回測報告。

## 公開資料口徑

- QQQ / SPY / GLD / BIL 預設使用 Yahoo Finance 公開調整後價格。
- QLD / TQQQ 使用 QQQ 每日漲跌模擬 daily-reset 2 倍 / 3 倍槓桿 ETF。
- GLD / BIL 上市前不假造價格；若回測期間早於上市日，該資產會以現金替代，並在報告中寫明。
- VIX 只作為熊市防守快速觸發訊號，不是交易資產。

## 本機執行

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## 部署建議

這是 Streamlit/Python 應用，不是純靜態網站。

推薦公開部署方式：

1. 把本資料夾作為獨立 GitHub repo。
2. 用 Streamlit Community Cloud 連接該 GitHub repo，入口檔案選 `streamlit_app.py`。
3. 若需要 Cloudflare 網域，可把自訂網域指向 Streamlit Cloud，或使用 Cloudflare Tunnel 反向代理一台持續運行的伺服器。

不建議把完整本機專案或 `data/` 交易資料上傳到公開 GitHub。
