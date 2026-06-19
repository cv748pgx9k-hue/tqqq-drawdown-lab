from __future__ import annotations
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from backtest_lab_engine import BacktestLabParams, RecoveryLayer, ReloadLayer, run_backtest_lab


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data" / "operator_imports" / "daily"
FIXED_USE_SYNTHETIC_LEVERAGE = True
FIXED_USE_CASH_PROXY_BEFORE_INCEPTION = True


st.set_page_config(page_title="TQQQ 回撤实验室", layout="wide")

st.markdown(
    """
    <style>
    :root {
      color-scheme: dark;
      --lab-bg: #0b111c;
      --lab-panel: #101827;
      --lab-panel-2: #131d2c;
      --lab-border: #26364d;
      --lab-text: #e6edf7;
      --lab-muted: #9aa8ba;
      --lab-subtle: #6f7f93;
      --lab-accent: #38bdf8;
      --lab-accent-soft: rgba(56, 189, 248, .14);
      --lab-input: #192233;
    }
    html, body, [data-testid="stAppViewContainer"], .stApp {
      background: var(--lab-bg) !important;
      color: var(--lab-text) !important;
    }
    [data-testid="stSidebar"], [data-testid="stHeader"] {
      background: var(--lab-bg) !important;
    }
    header[data-testid="stHeader"] { display: none; }
    div[data-testid="stToolbar"] { display: none; }
    div[data-testid="stDecoration"] { display: none; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    .block-container { padding-top: 1.15rem; max-width: 1600px; padding-left: 1.2rem; padding-right: 1.2rem; }
    h1, h2, h3, h4, h5, h6, p, label, span, div { color: var(--lab-text); }
    h1 { font-size: 1.65rem !important; margin-bottom: .05rem !important; color: var(--lab-text) !important; }
    h2, h3 { margin-top: .35rem !important; margin-bottom: .2rem !important; }
    div[data-testid="stVerticalBlock"] { gap: .28rem; }
    div[data-testid="stHorizontalBlock"] { gap: .5rem; }
    div[data-testid="stMetric"] { background: var(--lab-panel); border: 1px solid var(--lab-border); border-radius: 8px; padding: 7px 9px; }
    div[data-testid="stMetricLabel"] { font-size: .72rem; color: var(--lab-muted) !important; }
    div[data-testid="stMetricValue"] { font-size: 1.22rem; color: var(--lab-text) !important; }
    div[data-testid="stExpander"] { border-radius: 8px; border-color: var(--lab-border) !important; background: var(--lab-panel) !important; }
    div[data-testid="stExpander"] details,
    div[data-testid="stExpander"] summary { background: var(--lab-panel) !important; color: var(--lab-text) !important; }
    .stButton button { min-height: 2.3rem; }
    .stButton button,
    div[data-testid="stDownloadButton"] button {
      background: var(--lab-accent-soft) !important;
      color: var(--lab-text) !important;
      border: 1px solid rgba(56, 189, 248, .35) !important;
      border-radius: 8px !important;
    }
    .stButton button[kind="primary"] {
      background: #0ea5e9 !important;
      border-color: #0ea5e9 !important;
      color: #06111e !important;
      font-weight: 800 !important;
    }
    div[data-testid="stDateInput"] input,
    div[data-testid="stNumberInput"] input {
      min-height: 2.35rem;
      background: var(--lab-input) !important;
      color: var(--lab-text) !important;
      border-color: var(--lab-border) !important;
    }
    div[data-testid="stSelectbox"] div[data-baseweb="select"] {
      min-height: 2.35rem;
      background: var(--lab-input) !important;
      color: var(--lab-text) !important;
      border-color: var(--lab-border) !important;
    }
    div[data-baseweb="popover"],
    div[data-baseweb="menu"],
    ul[role="listbox"] {
      background: var(--lab-panel-2) !important;
      color: var(--lab-text) !important;
      border-color: var(--lab-border) !important;
    }
    li[role="option"] { background: var(--lab-panel-2) !important; color: var(--lab-text) !important; }
    li[role="option"]:hover { background: #1e2d44 !important; }
    textarea {
      background: var(--lab-input) !important;
      color: var(--lab-text) !important;
      border-color: var(--lab-border) !important;
    }
    label[data-testid="stWidgetLabel"] p { font-size: .8rem; margin-bottom: -.15rem; color: var(--lab-text) !important; }
    .settings-card { border: 1px solid var(--lab-border); border-radius: 10px; padding: .75rem .85rem; background: var(--lab-panel); margin-bottom: .5rem; }
    .settings-title { font-size: 1.02rem; font-weight: 850; margin: 0 0 .45rem 0; }
    .settings-subtitle { font-size: .8rem; color: var(--lab-muted); margin: -.25rem 0 .5rem 0; }
    .ab-split { margin: -.18rem 0 .55rem 0; }
    .ab-split-labels { display: flex; justify-content: space-between; align-items: center; gap: .75rem; color: var(--lab-muted); font-size: .82rem; font-weight: 750; }
    .ab-split-labels strong { color: var(--lab-text); font-size: 1.05rem; }
    .small-note { color: var(--lab-muted); font-size: .82rem; line-height: 1.35; }
    .micro-note { color: var(--lab-muted); font-size: .73rem; line-height: 1.25; margin-top: -.3rem; }
    .section-title { font-size: .98rem; font-weight: 800; margin: .2rem 0 .1rem 0; }
    .top-note { color: var(--lab-muted); font-size: .78rem; line-height: 1.35; margin: .15rem 0 .25rem 0; }
    .compact-box { border: 1px solid var(--lab-border); border-radius: 8px; padding: .55rem .65rem; background: var(--lab-panel-2); }
    .static-chart { width: 100%; overflow: hidden; border: 1px solid var(--lab-border); border-radius: 10px; padding: .35rem; background: var(--lab-panel); }
    .strategy-text { font-size: .86rem; line-height: 1.55; white-space: pre-wrap; }
    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
      background: var(--lab-panel) !important;
      color: var(--lab-text) !important;
    }
    [data-testid="stAlert"] {
      background: #102235 !important;
      color: var(--lab-text) !important;
      border-color: var(--lab-border) !important;
    }
    @media (max-width: 760px) {
      .block-container { padding-left: .6rem; padding-right: .6rem; padding-top: .9rem; }
      h1 { font-size: 1.32rem !important; line-height: 1.15 !important; }
      .section-title { font-size: .9rem; }
      .micro-note { font-size: .68rem; }
      div[data-testid="stMetricValue"] { font-size: 1.05rem; }
      .stButton button { min-height: 2.15rem; }
      div[data-testid="stDateInput"] input,
      div[data-testid="stNumberInput"] input,
      div[data-testid="stSelectbox"] div[data-baseweb="select"] { min-height: 2.15rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def pct(v: float) -> str:
    if pd.isna(v):
        return "-"
    return f"{v:.2%}"


def pct_input(v: float) -> str:
    if pd.isna(v):
        return "-"
    return f"{v * 100:.2f}%"


def mult(v: float) -> str:
    if pd.isna(v):
        return "-"
    return f"{v:,.2f}x"


def money(v: float) -> str:
    if pd.isna(v):
        return "-"
    return f"${v:,.0f}"


def mix_text(weights: pd.Series) -> str:
    total = weights.sum()
    if total <= 0:
        return "未配置"
    parts = []
    for symbol, weight in (weights / total).items():
        if weight > 0:
            parts.append(f"{symbol} {weight * 100:.1f}%")
    return " / ".join(parts) if parts else "未配置"


def option_label(options: dict[str, object], value: object) -> str:
    for label, option_value in options.items():
        if option_value == value:
            return label
    return str(value)


def static_equity_chart_html(equity: pd.Series, height: int = 260) -> str:
    series = equity.dropna()
    if len(series) < 2:
        return "<div class='small-note'>資料不足，無法生成資產曲線。</div>"

    multiple = series / series.iloc[0]
    if len(multiple) > 1200:
        step = max(len(multiple) // 1200, 1)
        multiple = multiple.iloc[::step]
        if multiple.index[-1] != series.index[-1]:
            multiple = pd.concat([multiple, pd.Series([series.iloc[-1] / series.iloc[0]], index=[series.index[-1]])])

    width = 1200
    left = 56
    right = 18
    top = 18
    bottom = 34
    chart_w = width - left - right
    chart_h = height - top - bottom
    y_min = max(float(multiple.min()) * 0.96, 0.0)
    y_max = float(multiple.max()) * 1.04
    if y_max <= y_min:
        y_max = y_min + 1.0

    points: list[str] = []
    denom = max(len(multiple) - 1, 1)
    for i, value in enumerate(multiple.to_numpy(dtype=float)):
        x = left + chart_w * i / denom
        y = top + chart_h * (1.0 - (value - y_min) / (y_max - y_min))
        points.append(f"{x:.1f},{y:.1f}")

    grid_parts = []
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = top + chart_h * (1.0 - frac)
        label = y_min + (y_max - y_min) * frac
        grid_parts.append(
            f"<line x1='{left}' y1='{y:.1f}' x2='{width - right}' y2='{y:.1f}' stroke='#26364d' stroke-width='1'/>"
            f"<text x='8' y='{y + 4:.1f}' fill='#9aa8ba' font-size='12'>{label:.1f}x</text>"
        )

    date_points = [0, len(multiple) // 2, len(multiple) - 1]
    x_labels = []
    for idx in date_points:
        date_label = pd.Timestamp(multiple.index[idx]).strftime("%Y")
        x = left + chart_w * idx / denom
        anchor = "middle"
        if idx == 0:
            anchor = "start"
        elif idx == len(multiple) - 1:
            anchor = "end"
        x_labels.append(
            f"<text x='{x:.1f}' y='{height - 10}' fill='#9aa8ba' font-size='12' text-anchor='{anchor}'>{date_label}</text>"
        )

    svg = (
        f"<div class='static-chart'>"
        f"<svg viewBox='0 0 {width} {height}' width='100%' height='{height}' role='img' aria-label='資產淨值靜態曲線'>"
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='transparent'/>"
        + "".join(grid_parts)
        + f"<polyline fill='none' stroke='#38bdf8' stroke-width='2.2' points='{' '.join(points)}'/>"
        + "".join(x_labels)
        + f"<text x='{left}' y='14' fill='#9aa8ba' font-size='12'>資產倍數，不可互動縮放</text>"
        + "</svg></div>"
    )
    return svg


def strategy_text(params: BacktestLabParams, data_audit: pd.DataFrame) -> str:
    a_internal = pd.Series({"QQQ": params.a_qqq, "QLD": params.a_qld, "TQQQ": params.a_tqqq})
    b_internal = pd.Series({"BIL": params.b_bil, "QQQ": params.b_qqq, "SPY": params.b_spy, "GLD": params.b_gld})
    a_mix = mix_text(a_internal)
    b_mix = mix_text(b_internal)
    topup_assets = " → ".join(asset.replace("B_", "") for asset in params.topup_sell_order)
    trigger = option_label(HB2_TRIGGER_OPTIONS, params.hb2_trigger_mode)
    recovery_freq = option_label(RECOVERY_OPTIONS, params.recovery_check_frequency)
    reload_funding = option_label(RELOAD_FUNDING_OPTIONS, params.reload_funding)
    a_freq = option_label(FREQ_A_OPTIONS, params.a_band_frequency)
    b_freq = option_label(FREQ_B_OPTIONS, params.b_rebalance_frequency)

    lines = [
        "完整策略文字",
        "",
        f"回測期間：{params.start.date().isoformat()} 到 {params.end.date().isoformat()}。",
        f"初始本金：{params.initial_capital:,.0f}。",
        "",
        "1. 資產分組",
        f"A組進攻槓桿組佔整體 {pct_input(params.a_total)}；B組防守現金組佔整體 {pct_input(1.0 - params.a_total)}。",
        f"A組內部比例：{a_mix}。",
        f"B組內部比例：{b_mix}。",
        "",
        "2. A組 band 規則",
        f"A組按「{a_freq}」檢查，不是每天固定再平衡。",
        f"若 A組實際權重大於 A組目標的 {params.trim_band:.3f} 倍，收割回 A組目標，資金轉入 B組。",
        f"若 A組實際權重低於 A組目標的 {params.topup_band:.3f} 倍，且 QQQ 在長期均線上方，才補回 A組目標。",
        f"補 A 組時先用 BIL；若 BIL 不足，再依序賣出：{topup_assets}。",
        f"單筆目標差距低於 {pct_input(params.min_trade_threshold)} 時不交易，避免過多小額調倉。",
        "",
        "3. B組規則",
        f"B組按「{b_freq}」回到 BIL / QQQ / SPY / GLD 的設定比例。",
        "B組再平衡只調整 B組內部，不主動把整個投資組合拉回 A/B 目標比例。",
        "",
        "4. 熊市防守",
    ]
    if params.hb2_enabled:
        lines.extend(
            [
                f"熊市防守啟用，觸發方式：{trigger}。",
                f"熊市條件使用 QQQ {params.qqq_sma_days} 日均線，並用 {params.slope_lookback_days} 個交易日前後比較判斷均線是否下彎。",
            ]
        )
        if params.hb2_vix_fast_enabled:
            lines.append(
                f"快速觸發：若 VIX > {params.hb2_vix_fast_threshold:.1f} 且 QQQ 低於長期均線，不等確認天數，下一交易日直接進入熊市防守狀態。"
            )
        if params.hb2_drawdown_fast_enabled:
            lines.append(
                f"快速觸發：若 QQQ 從歷史高點回撤達 {pct_input(params.hb2_drawdown_fast_threshold)} 且 QQQ 低於長期均線，不等確認天數，下一交易日直接進入熊市防守狀態。"
            )
        if params.hb2_vix_fast_enabled or params.hb2_drawdown_fast_enabled:
            lines.append("快速觸發只是可測條件；是否有效需要看回測結果。如果快速觸發條件沒有發生，仍按下方確認方式判斷是否進入熊市防守。")
        if params.hb2_trigger_mode in ("monthly_consecutive", "monthly_or_rolling"):
            lines.append(f"月末確認：連續 {params.hb2_months_required} 個月末符合熊市條件後，下一交易日進入熊市防守狀態。")
        if params.hb2_trigger_mode == "daily_consecutive":
            lines.append(f"每日確認：連續 {params.hb2_daily_consecutive_days} 個交易日符合熊市條件後，下一交易日進入熊市防守狀態。")
        if params.hb2_trigger_mode in ("rolling_count", "monthly_or_rolling"):
            lines.append(f"滾動計數：最近 {params.hb2_rolling_window} 個交易日內至少 {params.hb2_rolling_required} 天符合熊市條件後，下一交易日進入熊市防守狀態。")
        lines.extend(
            [
                f"進入熊市防守時，TQQQ 降為 0；原本 QLD 保留 {pct_input(params.hb2_qld_keep_fraction)}。",
                f"被清出的 TQQQ / QLD 權重中，有 {pct_input(params.hb2_qld_add_fraction)} 額外轉入 QLD，其餘按當前 B組比例分配到 BIL / QQQ / SPY / GLD。",
            ]
        )
    else:
        lines.append("熊市防守關閉。")

    lines.extend(["", "5. 復原規則", f"復原檢查頻率：{recovery_freq}。"])
    for index, layer in enumerate(params.recovery_layers, start=1):
        condition = option_label(RECOVERY_CONDITION_OPTIONS, layer.condition)
        ret_text = ""
        if layer.condition == "sma50_ret20":
            ret_text = f"，QQQ 20 日回報門檻為 {pct_input(layer.ret20_threshold)}"
        lines.append(
            f"第 {index} 層：條件為「{condition}{ret_text}」；TQQQ 恢復到原目標的 {pct_input(layer.tqqq_fraction)}，QLD 恢復到原目標的 {pct_input(layer.qld_fraction)}。"
        )
    lines.append("如果復原期間熊市條件再次觸發，重新進入熊市防守狀態。")

    lines.extend(["", "6. 回撤加倉"])
    if params.reload_enabled and params.reload_assets:
        assets = " / ".join(params.reload_assets)
        lines.append(f"回撤加倉啟用，標的：{assets}。")
        lines.append(f"資金來源：{reload_funding}；BIL 最低保留 {pct_input(params.reload_bil_floor)}。")
        for index, layer in enumerate(params.reload_layers, start=1):
            lines.append(f"第 {index} 層：標的距離自身歷史高點回撤 {pct_input(layer.drawdown)} 時，使用可用加倉資金的 {pct_input(layer.fraction)} 買入。")
        lines.append("每個標的每一層只觸發一次；該標的重新創歷史新高後，該標的的回撤加倉層級重置。")
    else:
        lines.append("回撤加倉關閉。")

    lines.extend(
        [
            "",
            "7. 交易成本 / 執行假設",
            f"單邊滑價：{params.one_way_slippage_bps:.1f} bps。",
            "本工具是回測工具，不送單，不連接 broker 執行交易。",
            "",
            "8. 資料口徑",
            "QLD / TQQQ 使用 QQQ 每日漲跌模擬 daily-reset 2 倍 / 3 倍槓桿 ETF。",
        ]
    )
    cash_rows = data_audit[data_audit["Mode"].astype(str).str.contains("cash proxy", na=False)]
    for _, row in cash_rows.iterrows():
        lines.append(f"{row['Symbol']} 在 {row['Start']} 到 {row['End']} 沒有可交易價格，該期間用現金替代。")
    return "\n".join(lines)


def latest_bounds() -> tuple[date, date]:
    starts = []
    ends = []
    for symbol in ("QQQ", "SPY"):
        files = sorted(DATA_DIR.glob(f"{symbol}_daily_*.csv"))
        if not files:
            continue
        df = pd.read_csv(files[-1], usecols=["date"])
        dates = pd.to_datetime(df["date"])
        starts.append(dates.min())
        ends.append(dates.max())
    if not starts:
        return date(1999, 3, 10), date.today()
    return max(starts).date(), min(ends).date()


def preset_values(name: str) -> dict[str, float]:
    default_custom = {
        "a_total": 55.0,
        "a_qqq": 0.0,
        "a_qld": 0.0,
        "a_tqqq": 100.0,
        "b_bil": 25.0,
        "b_qqq": 30.0,
        "b_spy": 20.0,
        "b_gld": 25.0,
        "hb2_qld_add": 0.0,
        "hb2_qld_keep": 100.0,
    }
    presets = {
        "自訂": default_custom,
        "TQQQ 預設策略": {
            "a_total": 55.0,
            "a_qqq": 0.0,
            "a_qld": 0.0,
            "a_tqqq": 100.0,
            "b_bil": 25.0,
            "b_qqq": 30.0,
            "b_spy": 20.0,
            "b_gld": 25.0,
            "hb2_qld_add": 0.0,
            "hb2_qld_keep": 100.0,
        },
        "TQQQ+QLD 預設策略": {
            "a_total": 55.0,
            "a_qqq": 0.0,
            "a_qld": 25.0,
            "a_tqqq": 75.0,
            "b_bil": 25.0,
            "b_qqq": 30.0,
            "b_spy": 20.0,
            "b_gld": 25.0,
            "hb2_qld_add": 0.0,
            "hb2_qld_keep": 100.0,
        },
    }
    return presets[name]


def render_data_policy(audit: pd.DataFrame) -> None:
    cash_rows = audit[audit["Mode"].astype(str).str.contains("cash proxy", na=False)]
    synthetic_rows = audit[audit["Mode"].astype(str).str.contains("synthetic", na=False)]
    lines = []
    for _, row in cash_rows.iterrows():
        lines.append(
            f"- {row['Symbol']} 在 {row['Start']} 到 {row['End']} 沒有可交易價格，這段回測把它當成現金，不假裝當時能買到。"
        )
    if not synthetic_rows.empty:
        lines.append("- QLD / TQQQ 用 QQQ 的每日漲跌模擬每日重置 2 倍 / 3 倍槓桿 ETF，所以可以測 1999 以來的長歷史。")
    vix_rows = audit[audit["Symbol"].astype(str).eq("VIX")]
    if not vix_rows.empty and vix_rows.iloc[-1]["Mode"] == "signal only":
        lines.append("- VIX 使用可取得的 VIX 日線作為訊號，只用於熊市防守快速觸發，不是交易資產。")
    elif not vix_rows.empty:
        lines.append("- 沒有找到 VIX 日線時，VIX 快速觸發不會生效。")
    if not lines:
        lines.append("- 本次回測所有交易資產都有價格資料，沒有使用上市前現金替代。")
    st.info("回測資料口徑\n\n" + "\n".join(lines))


FREQ_A_OPTIONS = {
    "每季檢查": "quarterly",
    "每月檢查": "monthly",
    "每日檢查": "daily",
    "不檢查": "none",
}

FREQ_B_OPTIONS = {
    "每年再平衡": "annual",
    "每季再平衡": "quarterly",
    "每月再平衡": "monthly",
    "不再平衡": "none",
}

RECOVERY_OPTIONS = {
    "每月檢查": "monthly",
    "每日檢查": "daily",
}

RECOVERY_CONDITION_OPTIONS = {
    "QQQ 站上 SMA20": "sma20",
    "QQQ 站上 SMA50": "sma50",
    "QQQ 站上 SMA50 且 20 日回報達標": "sma50_ret20",
    "QQQ 站上 SMA100": "sma100",
    "復原滿 1 個月且 QQQ 站上 SMA100": "sma100_after_1m",
    "QQQ 站上 SMA200": "sma200",
    "QQQ 站上 SMA200，或復原滿 1 個月且 QQQ 站上 SMA100": "sma200_or_sma100_after_1m",
}

HB2_TRIGGER_OPTIONS = {
    "月末連續確認": "monthly_consecutive",
    "每日連續確認": "daily_consecutive",
    "交易日滾動計數": "rolling_count",
    "月末或滾動計數任一觸發": "monthly_or_rolling",
    "關閉": "off",
}

RELOAD_FUNDING_OPTIONS = {
    "只用 BIL": "bil_only",
    "先用 BIL，不夠再用 GLD": "bil_then_gld",
}

TOPUP_ORDER_OPTIONS = {
    "先賣 GLD，再賣 SPY，最後賣 QQQ": ("B_GLD", "B_SPY", "B_QQQ"),
    "先賣 GLD，再賣 QQQ，最後賣 SPY": ("B_GLD", "B_QQQ", "B_SPY"),
    "先賣 SPY，再賣 QQQ，最後賣 GLD": ("B_SPY", "B_QQQ", "B_GLD"),
    "先賣 QQQ，再賣 SPY，最後賣 GLD": ("B_QQQ", "B_SPY", "B_GLD"),
}

HELP_TEXT = {
    "preset": "選現成策略或自訂。預設策略只是幫你先填好參數，之後仍可手動改每一項。",
    "initial_capital": "回測起始本金。它會影響最終金額，但不會改變年化收益、回撤等百分比指標。",
    "start_date": "回測開始日期。若早於 BIL / GLD 上市，該資產在上市前會用現金替代；QLD / TQQQ 會由 QQQ 每日漲跌模擬。",
    "end_date": "回測結束日期。通常用目前資料庫裡最新可用的交易日。",
    "a_total": "A組進攻槓桿組佔整體資產多少。B組防守現金組會自動等於 100% 減 A組。",
    "a_qqq": "A組內 QQQ 的相對比例。A組內部 QQQ / QLD / TQQQ 會按相對比例自動換算成 100%。",
    "a_qld": "A組內 QLD 的相對比例。長歷史回測中，QLD 使用 QQQ 每日 2 倍 daily-reset proxy 模擬。",
    "a_tqqq": "A組內 TQQQ 的相對比例。長歷史回測中，TQQQ 使用 QQQ 每日 3 倍 daily-reset proxy 模擬。",
    "b_bil": "B組內 BIL 的相對比例。BIL 可理解為短債 / 類現金；上市前會用現金替代。",
    "b_qqq": "B組內 QQQ 的相對比例。這是普通 QQQ，不是槓桿 ETF。",
    "b_spy": "B組內 SPY 的相對比例。用來讓 B組有較分散的大盤股票曝險。",
    "b_gld": "B組內 GLD 的相對比例。GLD 使用真實 adjusted price；上市前不假造資料，會用現金替代。",
    "a_band_frequency": "A組不每天固定再平衡，只在指定頻率檢查是否超出 band。超出才收割或補倉。",
    "trim_band": "A組實際權重高於 A目標 × 這個倍數時，把 A組收割回目標，資金進 B組。",
    "topup_band": "A組實際權重低於 A目標 × 這個倍數時，才考慮補回 A目標。若 QQQ 在均線下方，仍不補 A。",
    "b_rebalance_frequency": "B組內部 BIL / QQQ / SPY / GLD 依這個頻率回到設定比例。這不代表每天重置整個策略。",
    "min_trade": "目標權重和目前權重差距小於這個百分比時，不交易，用來避免太多小額調倉。",
    "slippage": "單邊交易滑價成本。1 基點 = 0.01%。這只是回測壓力測試，不代表真實成交一定如此。",
    "topup_order": "A組要補倉但 BIL 不夠時，會按這個順序賣出 B組資產來補 A。",
    "qqq_sma_days": "熊市防守使用的 QQQ 長期均線天數。預設 200 日均線。",
    "slope_lookback": "用來判斷 QQQ 長期均線是否下彎。今天的均線低於 N 個交易日前，就視為下彎。",
    "hb2_enabled": "開啟後，當 QQQ 進入長期熊市條件時，策略會進入熊市防守狀態，降低 TQQQ。",
    "trigger_mode": "熊市防守的觸發方式。可用月末連續確認、每日連續確認、或指定交易日滾動視窗內達標次數。",
    "hb2_months": "月末連續確認模式使用。連續幾個月末都符合熊市條件，才進入熊市防守。",
    "hb2_daily": "每日連續確認模式使用。連續幾個交易日符合熊市條件，才進入熊市防守。",
    "hb2_window": "交易日滾動計數模式使用。觀察最近多少個交易日。",
    "hb2_required": "交易日滾動計數模式使用。在觀察視窗內至少有幾天符合熊市條件，才進入熊市防守。",
    "vix_fast": "快速觸發條件。若 VIX 高於門檻，而且 QQQ 低於長期均線，不等原本確認流程，下一交易日直接進入熊市防守。",
    "vix_fast_threshold": "VIX 快速觸發門檻。預設 35；是否有幫助要看回測結果，不預設一定降低回撤。",
    "drawdown_fast": "快速觸發條件。若 QQQ 從歷史高點回撤超過門檻，而且 QQQ 低於長期均線，不等原本確認流程，下一交易日直接進入熊市防守。",
    "drawdown_fast_threshold": "QQQ 從歷史高點回撤多少就快速觸發。預設 -30%；數字越接近 0 越容易觸發，效果要看回測結果。",
    "qld_add": "熊市防守時，清掉 TQQQ 後有多少比例轉到 QLD。0 表示清掉的 TQQQ 全部轉去 B組。",
    "qld_keep": "熊市防守時，原本已有的 QLD 保留多少。100 表示保留原本 QLD，0 表示 QLD 也全部退出到 B組。",
    "recovery_freq": "進入熊市防守後，用這個頻率檢查是否可以進入第一層復原。",
    "recovery_layers": "復原可設 1 到 3 層。每一層條件達成後，TQQQ / QLD 會各自恢復到該層設定的原目標比例。",
    "recovery_tqqq": "這一層觸發後，TQQQ 恢復到原本 TQQQ 目標倉位的多少比例。",
    "recovery_qld": "這一層觸發後，QLD 恢復到原本 QLD 目標倉位的多少比例。若熊市防守有賣掉 QLD，這裡可控制是否分階段買回。",
    "recovery_condition": "選這一層復原要看哪個條件。條件未達成，就停留在目前防守 / 復原層。",
    "recovery_ret20": "只有選到「20 日回報達標」條件時才使用。QQQ 20 日回報必須高於這個值，才進入該復原層。",
    "reload_enabled": "開啟後，QQQ / SPY 從歷史高點下跌到指定層級時，用 B組資金分批加倉普通 ETF。",
    "reload_assets": "選擇哪些普通 ETF 參與回撤加倉。這裡不會對 TQQQ / QLD / GLD 加倉。",
    "reload_funding": "回撤加倉用哪些資金來源。若只用 BIL，就不會賣 GLD / QQQ / SPY 來加倉。",
    "bil_floor": "回撤加倉後仍要保留的最低 BIL 權重，避免把現金完全用光。",
    "reload_drawdown": "該 ETF 距離自身歷史高點跌到這個幅度時，觸發該層加倉。",
    "reload_buy": "該層觸發時，用可用加倉資金的多少比例買入對應 ETF。",
    "run": "用目前頁面上的參數跑回測，並在下方生成完整結果表和資料口徑說明。",
}

COLUMN_LABELS = {
    "Start": "開始",
    "End": "結束",
    "Equity": "資產淨值",
    "Peak": "前高",
    "Drawdown": "回撤",
    "CAGR": "年化收益",
    "Final Multiple": "最終倍數",
    "MaxDD": "最大回撤",
    "Calmar": "卡瑪",
    "Sharpe": "夏普",
    "Sortino": "索提諾",
    "Volatility": "年化波動",
    "Worst Year": "最差年度",
    "Worst Month": "最差月份",
    "Trades": "交易次數",
    "Turnover": "換手率",
    "Avg Leveraged ETF": "平均槓桿 ETF 權重",
    "Max Leveraged ETF": "最大槓桿 ETF 權重",
    "Avg TQQQ": "平均 TQQQ",
    "Avg QLD": "平均 QLD",
    "Avg QQQ": "平均 QQQ",
    "Avg SPY": "平均 SPY",
    "Avg GLD": "平均 GLD",
    "Avg BIL": "平均 BIL",
    "Max TQQQ": "最大 TQQQ",
    "Max QLD": "最大 QLD",
    "Max GLD": "最大 GLD",
    "QQQ Reload Entries": "QQQ 加倉次數",
    "SPY Reload Entries": "SPY 加倉次數",
    "Days NORMAL": "正常狀態天數",
    "Days HARD_BRIDGE": "熊市防守狀態天數",
    "Days RECOVERY": "復原狀態天數",
    "Year": "年份",
    "Return": "回報",
    "Annual": "全年",
    "Trough": "谷底",
    "Recovery": "回到新高",
    "Duration Days": "持續天數",
    "Date": "日期",
    "Month": "月份",
    "Reason": "原因",
    "Cost": "成本",
    "Symbol": "標的",
    "Asset": "資產",
    "File": "檔案",
    "Rows": "筆數",
    "Mode": "資料模式",
    "Notes": "備註",
    "From": "原狀態",
    "To": "新狀態",
}

MONTH_LABELS = {
    "Jan": "1月",
    "Feb": "2月",
    "Mar": "3月",
    "Apr": "4月",
    "May": "5月",
    "Jun": "6月",
    "Jul": "7月",
    "Aug": "8月",
    "Sep": "9月",
    "Oct": "10月",
    "Nov": "11月",
    "Dec": "12月",
}

VALUE_LABELS = {
    "real adjusted price": "真實調整後價格",
    "public adjusted price": "公開調整後價格",
    "public signal only": "公開訊號資料",
    "cash proxy before inception": "上市前以現金替代",
    "synthetic daily-reset proxy": "用 QQQ 每日重置模擬槓桿 ETF",
    "synthetic daily-reset from QQQ": "由 QQQ 每日重置模擬",
    "open adjusted by adjusted_close / close ratio": "開盤價按調整後收盤價比例校正",
    "downloaded from Yahoo Finance for public deployment; open adjusted by adjusted close ratio": "公開部署時由 Yahoo Finance 下載；開盤價按調整後收盤價比例校正",
    "QLD = 2x QQQ daily reset; TQQQ = 3x QQQ daily reset; annual drag included in proxy": "QLD = QQQ 每日 2 倍；TQQQ = QQQ 每日 3 倍；已包含年度拖累",
    "NORMAL": "正常",
    "HARD_BRIDGE": "熊市防守狀態",
    "RECOVERY": "復原",
    "initial_deploy": "初始建倉",
    "B_REBALANCE": "B 組再平衡",
    "HOLD_A_ZERO": "A 組為 0，不動作",
    "TRIM_A_TO_TARGET": "A 組收割回目標",
    "TOPUP_A_TO_TARGET": "A 組補回目標",
    "HOLD_A_QQQ_BELOW_SMA200": "QQQ 低於 SMA200，A 組不補倉",
    "HOLD_WITHIN_BAND": "A 組仍在 band 內",
    "ENTER_HARD_BRIDGE": "進入熊市防守",
    "ENTER_HARD_BRIDGE_FAST_VIX": "VIX 快速進入熊市防守",
    "ENTER_HARD_BRIDGE_FAST_DRAWDOWN": "QQQ 回撤快速進入熊市防守",
    "HARD_BRIDGE_TARGET": "維持熊市防守目標",
    "ENTER_RECOVERY": "進入復原期",
    "RECOVERY_TARGET": "維持復原目標",
    "FULL_RECOVERY": "完全復原",
    "QQQ_RELOAD_20": "QQQ 回撤 20% 加倉",
    "QQQ_RELOAD_30": "QQQ 回撤 30% 加倉",
    "QQQ_RELOAD_40": "QQQ 回撤 40% 加倉",
    "QQQ_RELOAD_50": "QQQ 回撤 50% 加倉",
    "SPY_RELOAD_20": "SPY 回撤 20% 加倉",
    "SPY_RELOAD_30": "SPY 回撤 30% 加倉",
    "SPY_RELOAD_40": "SPY 回撤 40% 加倉",
    "SPY_RELOAD_50": "SPY 回撤 50% 加倉",
}


PERCENT_DISPLAY_COLUMNS = {
    "年化收益",
    "最大回撤",
    "回撤",
    "年化波動",
    "最差年度",
    "最差月份",
    "換手率",
    "成本",
    "回報",
    "全年",
    "平均槓桿 ETF 權重",
    "最大槓桿 ETF 權重",
    "平均 TQQQ",
    "平均 QLD",
    "平均 QQQ",
    "平均 SPY",
    "平均 GLD",
    "平均 BIL",
    "最大 TQQQ",
    "最大 QLD",
    "最大 GLD",
    *MONTH_LABELS.values(),
}

COUNT_DISPLAY_COLUMNS = {
    "年份",
    "月份",
    "交易次數",
    "QQQ 加倉次數",
    "SPY 加倉次數",
    "正常狀態天數",
    "熊市防守狀態天數",
    "復原狀態天數",
    "持續天數",
    "筆數",
}

DECIMAL_DISPLAY_COLUMNS = {"卡瑪", "夏普", "索提諾"}


def is_percent_display_column(column: object) -> bool:
    name = str(column)
    return (
        name in PERCENT_DISPLAY_COLUMNS
        or name.endswith(" 權重")
        or name.startswith("A組 ")
        or name.startswith("B組 ")
        or name.startswith("調整 ")
    )


def format_display_number(value: object, column: object) -> object:
    if pd.isna(value):
        return "-"
    name = str(column)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if is_percent_display_column(name):
        return f"{number:.2%}"
    if name == "最終倍數":
        return mult(number)
    if name in DECIMAL_DISPLAY_COLUMNS:
        return f"{number:.2f}"
    if name in COUNT_DISPLAY_COLUMNS:
        return f"{number:,.0f}"
    return value


def display_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    out = out.rename(columns={**COLUMN_LABELS, **MONTH_LABELS})
    out.columns = [
        col.replace("_Weight", " 權重")
        .replace("A_", "A組 ")
        .replace("B_", "B組 ")
        .replace("Delta_", "調整 ")
        for col in out.columns
    ]
    for col in out.select_dtypes(include="object").columns:
        out[col] = out[col].map(translate_cell)
    for col in out.select_dtypes(include="number").columns:
        out[col] = out[col].map(lambda value, column=col: format_display_number(value, column))
    return out


def translate_cell(value: object) -> object:
    if not isinstance(value, str):
        return value
    if "+" in value:
        return " + ".join(VALUE_LABELS.get(part, part) for part in value.split("+"))
    return VALUE_LABELS.get(value, value)


common_start, data_end = latest_bounds()

st.title("TQQQ 回撤实验室")
st.caption("免費參數化回測工具。A 組、B 組、熊市防守、回撤加倉都在同一頁設定，跑完後直接生成回測報告。")

st.markdown("<div class='settings-card'>", unsafe_allow_html=True)
st.markdown("<div class='settings-title'>基本設定</div>", unsafe_allow_html=True)
top1, top2, top3, top4 = st.columns([1.05, .85, .9, .9])
with top1:
    preset = st.selectbox("策略模板", ["自訂", "TQQQ 預設策略", "TQQQ+QLD 預設策略"], help=HELP_TEXT["preset"])
with top2:
    initial_capital = st.number_input("初始資金", min_value=100.0, value=10_000.0, step=1_000.0, help=HELP_TEXT["initial_capital"])
with top3:
    start_date = st.date_input("開始日期", value=common_start, min_value=date(1999, 3, 10), max_value=data_end, help=HELP_TEXT["start_date"])
with top4:
    end_date = st.date_input("結束日期", value=data_end, min_value=date(1999, 3, 10), max_value=data_end, help=HELP_TEXT["end_date"])
st.markdown("</div>", unsafe_allow_html=True)

if start_date >= end_date:
    st.error("開始日期必須早於結束日期。")

defaults = preset_values(preset)
preset_key = preset.replace("+", "plus").replace(" ", "_")

st.markdown("<div class='settings-card'>", unsafe_allow_html=True)
st.markdown("<div class='settings-title'>1. 組合比例</div>", unsafe_allow_html=True)
st.markdown("<div class='settings-subtitle'>先設定 A組進攻槓桿組總比例；剩下就是 B組防守現金組。兩組內部比例可以自由輸入，系統會自動換算成 100%。</div>", unsafe_allow_html=True)
a_total = st.slider("A組進攻槓桿組總比例", min_value=0.0, max_value=100.0, value=float(defaults["a_total"]), step=0.5, key=f"a_total_{preset_key}", help=HELP_TEXT["a_total"])
b_total = 100.0 - a_total
st.markdown(
    f"""
    <div class="ab-split">
      <div class="ab-split-labels">
        <span>A組進攻槓桿組 <strong>{a_total:.1f}%</strong></span>
        <span>B組防守現金組 <strong>{b_total:.1f}%</strong></span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

mix_a, mix_gap, mix_b = st.columns([1.2, .08, 1.55])
with mix_a:
    st.markdown("<div class='section-title'>A組進攻槓桿組</div>", unsafe_allow_html=True)
    a1, a2, a3 = st.columns(3)
    a_qqq = a1.number_input("QQQ", min_value=0.0, max_value=100.0, value=float(defaults["a_qqq"]), step=1.0, key=f"a_qqq_{preset_key}", help=HELP_TEXT["a_qqq"])
    a_qld = a2.number_input("QLD", min_value=0.0, max_value=100.0, value=float(defaults["a_qld"]), step=1.0, key=f"a_qld_{preset_key}", help=HELP_TEXT["a_qld"])
    a_tqqq = a3.number_input("TQQQ", min_value=0.0, max_value=100.0, value=float(defaults["a_tqqq"]), step=1.0, key=f"a_tqqq_{preset_key}", help=HELP_TEXT["a_tqqq"])
with mix_b:
    st.markdown("<div class='section-title'>B組防守現金組</div>", unsafe_allow_html=True)
    b1, b2, b3, b4 = st.columns(4)
    b_bil = b1.number_input("BIL", min_value=0.0, max_value=100.0, value=float(defaults["b_bil"]), step=1.0, key=f"b_bil_{preset_key}", help=HELP_TEXT["b_bil"])
    b_qqq = b2.number_input("QQQ", min_value=0.0, max_value=100.0, value=float(defaults["b_qqq"]), step=1.0, key=f"b_qqq_{preset_key}", help=HELP_TEXT["b_qqq"])
    b_spy = b3.number_input("SPY", min_value=0.0, max_value=100.0, value=float(defaults["b_spy"]), step=1.0, key=f"b_spy_{preset_key}", help=HELP_TEXT["b_spy"])
    b_gld = b4.number_input("GLD", min_value=0.0, max_value=100.0, value=float(defaults["b_gld"]), step=1.0, key=f"b_gld_{preset_key}", help=HELP_TEXT["b_gld"])

normalized_a = pd.Series({"QQQ": a_qqq, "QLD": a_qld, "TQQQ": a_tqqq})
normalized_b = pd.Series({"BIL": b_bil, "QQQ": b_qqq, "SPY": b_spy, "GLD": b_gld})
if normalized_a.sum() <= 0 and a_total > 0:
    st.error("A 組總比例大於 0 時，A 組內部比例不能全是 0。")
if normalized_b.sum() <= 0 and a_total < 100:
    st.error("B 組總比例大於 0 時，B 組內部比例不能全是 0。")
st.markdown(
    "<div class='micro-note'>"
    f"A組進攻槓桿組換算：{mix_text(normalized_a)}；B組防守現金組換算：{mix_text(normalized_b)}。"
    "內部合計不是 100% 時，系統會按比例換算。"
    "</div>",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='settings-card'>", unsafe_allow_html=True)
st.markdown("<div class='settings-title'>2. 策略規則</div>", unsafe_allow_html=True)
s1, s2, s3 = st.columns(3)
with s1:
    a_band_label = st.selectbox("A 組區間檢查", list(FREQ_A_OPTIONS), index=0, help=HELP_TEXT["a_band_frequency"])
    a_band_frequency = FREQ_A_OPTIONS[a_band_label]
    trim_band = st.number_input("A 組漲到幾倍才收割", min_value=1.00, max_value=3.00, value=1.375, step=0.025, format="%.3f", help=HELP_TEXT["trim_band"])
    topup_band = st.number_input("A 組跌到幾倍才補倉", min_value=0.05, max_value=1.00, value=0.75, step=0.025, format="%.3f", help=HELP_TEXT["topup_band"])
with s2:
    b_rebalance_label = st.selectbox("B 組再平衡", list(FREQ_B_OPTIONS), index=0, help=HELP_TEXT["b_rebalance_frequency"])
    b_rebalance_frequency = FREQ_B_OPTIONS[b_rebalance_label]
with s3:
    topup_order_label = st.selectbox("補 A 倉時，BIL 不夠後賣出順序", list(TOPUP_ORDER_OPTIONS), index=0, help=HELP_TEXT["topup_order"])
    topup_order = TOPUP_ORDER_OPTIONS[topup_order_label]
st.markdown("</div>", unsafe_allow_html=True)

with st.expander("進階設定：交易成本 / 執行假設", expanded=False):
    st.markdown(
        "<div class='micro-note'>一般回測可以保留預設值；想做更保守的壓力測試時，再調整這裡。</div>",
        unsafe_allow_html=True,
    )
    adv1, adv2 = st.columns(2)
    with adv1:
        min_trade = st.number_input("最小交易差額 %", min_value=0.0, max_value=20.0, value=2.0, step=0.5, help=HELP_TEXT["min_trade"])
    with adv2:
        slippage = st.number_input("單邊滑價（基點）", min_value=0.0, max_value=100.0, value=0.0, step=1.0, help=HELP_TEXT["slippage"])

st.markdown("<div class='settings-card'>", unsafe_allow_html=True)
st.markdown("<div class='settings-title'>3. 熊市防守</div>", unsafe_allow_html=True)
hb2_months = 2
hb2_daily = 40
hb2_window = 45
hb2_required = 30
vix_fast_enabled = True
vix_fast_threshold = 35.0
drawdown_fast_enabled = True
drawdown_fast_threshold = -30.0
h1, h2, h3, h4 = st.columns(4)
with h1:
    hb2_enabled = st.toggle("啟用熊市防守", value=True, help=HELP_TEXT["hb2_enabled"])
    trigger_label = st.selectbox("觸發方式", list(HB2_TRIGGER_OPTIONS), index=0, help=HELP_TEXT["trigger_mode"])
    trigger_mode = HB2_TRIGGER_OPTIONS[trigger_label]
    qqq_sma_days = st.number_input("熊市判斷均線天數", min_value=50, max_value=300, value=200, step=10, help=HELP_TEXT["qqq_sma_days"])
    slope_lookback = st.number_input("均線斜率回看天數", min_value=5, max_value=80, value=20, step=5, help=HELP_TEXT["slope_lookback"])
with h2:
    if trigger_mode in ("monthly_consecutive", "monthly_or_rolling"):
        hb2_months = st.number_input("連續熊市月數", min_value=1, max_value=12, value=2, step=1, help=HELP_TEXT["hb2_months"])
    if trigger_mode == "daily_consecutive":
        hb2_daily = st.number_input("連續交易日", min_value=1, max_value=252, value=40, step=1, help=HELP_TEXT["hb2_daily"])
    if trigger_mode != "off":
        vix_fast_enabled = st.toggle("VIX 快速觸發", value=True, help=HELP_TEXT["vix_fast"])
        vix_fast_threshold = st.number_input("VIX 門檻", min_value=10.0, max_value=100.0, value=35.0, step=1.0, help=HELP_TEXT["vix_fast_threshold"])
with h3:
    if trigger_mode in ("rolling_count", "monthly_or_rolling"):
        hb2_window = st.number_input("滾動視窗天數", min_value=5, max_value=252, value=45, step=1, help=HELP_TEXT["hb2_window"])
        hb2_required = st.number_input("視窗內達標天數", min_value=1, max_value=252, value=30, step=1, help=HELP_TEXT["hb2_required"])
    if trigger_mode != "off":
        drawdown_fast_enabled = st.toggle("QQQ 回撤快速觸發", value=True, help=HELP_TEXT["drawdown_fast"])
        drawdown_fast_threshold = st.number_input("QQQ 高點回撤門檻 %", min_value=-90.0, max_value=-1.0, value=-30.0, step=1.0, help=HELP_TEXT["drawdown_fast_threshold"])
    if trigger_mode == "off":
        st.markdown("<div class='small-note'>熊市防守已關閉，確認參數不會生效。</div>", unsafe_allow_html=True)
with h4:
    qld_add = st.number_input("清掉 TQQQ 後加到 QLD %", min_value=0.0, max_value=100.0, value=float(defaults["hb2_qld_add"]), step=5.0, key=f"qld_add_{preset_key}", help=HELP_TEXT["qld_add"])
    qld_keep = st.number_input("原本 QLD 保留比例 %", min_value=0.0, max_value=100.0, value=float(defaults["hb2_qld_keep"]), step=5.0, key=f"qld_keep_{preset_key}", help=HELP_TEXT["qld_keep"])
    recovery_freq_label = st.selectbox("第一層復原檢查頻率", list(RECOVERY_OPTIONS), index=0, help=HELP_TEXT["recovery_freq"])
    recovery_freq = RECOVERY_OPTIONS[recovery_freq_label]
    recovery_layer_count = st.number_input("復原層數", min_value=1, max_value=3, value=2, step=1, help=HELP_TEXT["recovery_layers"])

if trigger_mode != "off" and (vix_fast_enabled or drawdown_fast_enabled):
    st.markdown(
        "<div class='small-note'>快速觸發條件：符合時會不等原本確認流程，下一交易日直接進入熊市防守。是否有效需由回測結果判斷。</div>",
        unsafe_allow_html=True,
    )

recovery_layer_defaults = [
    (50.0, 100.0, "sma50_ret20", 5.0),
    (100.0, 100.0, "sma200_or_sma100_after_1m", 0.0),
    (100.0, 100.0, "sma200", 0.0),
]
condition_labels = list(RECOVERY_CONDITION_OPTIONS)
condition_values = list(RECOVERY_CONDITION_OPTIONS.values())
recovery_layers: list[RecoveryLayer] = []
for layer_index in range(int(recovery_layer_count)):
    default_tqqq_fraction, default_qld_fraction, default_condition, default_ret20 = recovery_layer_defaults[layer_index]
    st.markdown(f"<div class='section-title'>第 {layer_index + 1} 層復原</div>", unsafe_allow_html=True)
    r1, r2, r3, r4 = st.columns([0.85, 0.85, 1.3, 0.8])
    with r1:
        layer_tqqq_fraction = st.number_input(
            f"第 {layer_index + 1} 層 TQQQ 恢復比例 %",
            min_value=0.0,
            max_value=100.0,
            value=default_tqqq_fraction,
            step=5.0,
            key=f"recovery_tqqq_fraction_{layer_index}",
            help=HELP_TEXT["recovery_tqqq"],
        )
    with r2:
        layer_qld_fraction = st.number_input(
            f"第 {layer_index + 1} 層 QLD 恢復比例 %",
            min_value=0.0,
            max_value=100.0,
            value=default_qld_fraction,
            step=5.0,
            key=f"recovery_qld_fraction_{layer_index}",
            help=HELP_TEXT["recovery_qld"],
        )
    with r3:
        condition_index = condition_values.index(default_condition)
        condition_label = st.selectbox(
            f"第 {layer_index + 1} 層復原條件",
            condition_labels,
            index=condition_index,
            key=f"recovery_condition_{layer_index}",
            help=HELP_TEXT["recovery_condition"],
        )
        recovery_condition = RECOVERY_CONDITION_OPTIONS[condition_label]
    with r4:
        if recovery_condition == "sma50_ret20":
            layer_ret20 = st.number_input(
                f"第 {layer_index + 1} 層 20 日回報門檻 %",
                min_value=-20.0,
                max_value=30.0,
                value=default_ret20,
                step=1.0,
                key=f"recovery_ret20_{layer_index}",
                help=HELP_TEXT["recovery_ret20"],
            )
        else:
            layer_ret20 = 0.0
            st.markdown("<div class='small-note'>此條件不使用 20 日回報門檻。</div>", unsafe_allow_html=True)
    recovery_layers.append(
        RecoveryLayer(
            tqqq_fraction=layer_tqqq_fraction / 100.0,
            condition=recovery_condition,
            ret20_threshold=layer_ret20 / 100.0,
            qld_fraction=layer_qld_fraction / 100.0,
        )
    )
st.markdown(
    """
    <div class='compact-box small-note'>
    <strong>復原流程：</strong>
    先進入第 1 層；第 1 層成立後才會檢查第 2 層，第 2 層成立後才會檢查第 3 層。
    每一層可分別設定 TQQQ / QLD 恢復比例；兩者都設為 100%，代表完全回到原本策略目標。
    若復原期間熊市條件再次觸發，會重新進入熊市防守。
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='settings-card'>", unsafe_allow_html=True)
st.markdown("<div class='settings-title'>4. 回撤加倉</div>", unsafe_allow_html=True)
rr0, rr1, rr2, rr3 = st.columns([.75, .75, 1.1, .75])
with rr0:
    reload_enabled = st.toggle("啟用回撤加倉", value=True, help=HELP_TEXT["reload_enabled"])
with rr1:
    reload_qqq = st.checkbox("QQQ", value=True, help=HELP_TEXT["reload_assets"])
    reload_spy = st.checkbox("SPY", value=True, help=HELP_TEXT["reload_assets"])
with rr2:
    funding_label = st.selectbox("加倉資金來源", list(RELOAD_FUNDING_OPTIONS), index=0, help=HELP_TEXT["reload_funding"])
    funding = RELOAD_FUNDING_OPTIONS[funding_label]
with rr3:
    bil_floor = st.number_input("BIL 最低保留 %", min_value=0.0, max_value=50.0, value=5.0, step=1.0, help=HELP_TEXT["bil_floor"])
l1, l2, l3, l4 = st.columns(4)
d20 = l1.number_input("第 1 層回撤 %", min_value=-90.0, max_value=-1.0, value=-20.0, step=1.0, help=HELP_TEXT["reload_drawdown"])
f20 = l1.number_input("第 1 層買入 %", min_value=0.0, max_value=100.0, value=10.0, step=1.0, help=HELP_TEXT["reload_buy"])
d30 = l2.number_input("第 2 層回撤 %", min_value=-90.0, max_value=-1.0, value=-30.0, step=1.0, help=HELP_TEXT["reload_drawdown"])
f30 = l2.number_input("第 2 層買入 %", min_value=0.0, max_value=100.0, value=15.0, step=1.0, help=HELP_TEXT["reload_buy"])
d40 = l3.number_input("第 3 層回撤 %", min_value=-90.0, max_value=-1.0, value=-40.0, step=1.0, help=HELP_TEXT["reload_drawdown"])
f40 = l3.number_input("第 3 層買入 %", min_value=0.0, max_value=100.0, value=20.0, step=1.0, help=HELP_TEXT["reload_buy"])
d50 = l4.number_input("第 4 層回撤 %", min_value=-90.0, max_value=-1.0, value=-50.0, step=1.0, help=HELP_TEXT["reload_drawdown"])
f50 = l4.number_input("第 4 層買入 %", min_value=0.0, max_value=100.0, value=25.0, step=1.0, help=HELP_TEXT["reload_buy"])
st.markdown("</div>", unsafe_allow_html=True)

run = st.button("開始回測", type="primary", use_container_width=True, help=HELP_TEXT["run"])

if run and start_date < end_date:
    try:
        params = BacktestLabParams(
            start=pd.Timestamp(start_date),
            end=pd.Timestamp(end_date),
            initial_capital=float(initial_capital),
            a_total=a_total / 100.0,
            a_qqq=a_qqq / 100.0,
            a_qld=a_qld / 100.0,
            a_tqqq=a_tqqq / 100.0,
            b_bil=b_bil / 100.0,
            b_qqq=b_qqq / 100.0,
            b_spy=b_spy / 100.0,
            b_gld=b_gld / 100.0,
            trim_band=trim_band,
            topup_band=topup_band,
            min_trade_threshold=min_trade / 100.0,
            a_band_frequency=a_band_frequency,
            b_rebalance_frequency=b_rebalance_frequency,
            qqq_sma_days=int(qqq_sma_days),
            slope_lookback_days=int(slope_lookback),
            topup_sell_order=topup_order,
            reload_enabled=reload_enabled,
            reload_assets=tuple(asset for asset, enabled in (("QQQ", reload_qqq), ("SPY", reload_spy)) if enabled),
            reload_layers=(
                ReloadLayer(d20 / 100.0, f20 / 100.0),
                ReloadLayer(d30 / 100.0, f30 / 100.0),
                ReloadLayer(d40 / 100.0, f40 / 100.0),
                ReloadLayer(d50 / 100.0, f50 / 100.0),
            ),
            reload_funding=funding,
            reload_bil_floor=bil_floor / 100.0,
            hb2_enabled=hb2_enabled and trigger_mode != "off",
            hb2_trigger_mode=trigger_mode,
            hb2_months_required=int(hb2_months),
            hb2_daily_consecutive_days=int(hb2_daily),
            hb2_rolling_window=int(hb2_window),
            hb2_rolling_required=int(hb2_required),
            hb2_vix_fast_enabled=bool(vix_fast_enabled),
            hb2_vix_fast_threshold=float(vix_fast_threshold),
            hb2_drawdown_fast_enabled=bool(drawdown_fast_enabled),
            hb2_drawdown_fast_threshold=drawdown_fast_threshold / 100.0,
            hb2_qld_add_fraction=qld_add / 100.0,
            hb2_qld_keep_fraction=qld_keep / 100.0,
            recovery_check_frequency=recovery_freq,
            recovery_ret20_threshold=recovery_layers[0].ret20_threshold,
            recovery_tqqq_fraction=recovery_layers[0].tqqq_fraction,
            recovery_layers=tuple(recovery_layers),
            use_synthetic_leverage=FIXED_USE_SYNTHETIC_LEVERAGE,
            bil_proxy_before_inception=FIXED_USE_CASH_PROXY_BEFORE_INCEPTION,
            gld_proxy_before_inception=FIXED_USE_CASH_PROXY_BEFORE_INCEPTION,
            one_way_slippage_bps=slippage,
        )
        result = run_backtest_lab(params)
    except Exception as exc:
        st.error(f"回測失敗：{exc}")
        st.stop()

    metrics = result["metrics"]
    render_data_policy(result["data_audit"])
    st.subheader("結果總覽")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("年化收益", pct(metrics["CAGR"]))
    m2.metric("最終倍數", mult(metrics["Final Multiple"]))
    m3.metric("最終資產", money(initial_capital * metrics["Final Multiple"]))
    m4.metric("最大回撤", pct(metrics["MaxDD"]))
    m5.metric("夏普", f"{metrics['Sharpe']:.2f}" if pd.notna(metrics["Sharpe"]) else "-")
    m6.metric("交易次數", f"{metrics['Trades']}")

    st.markdown(static_equity_chart_html(result["equity"]["Equity"].rename("資產淨值"), height=260), unsafe_allow_html=True)

    full_strategy_text = strategy_text(params, result["data_audit"])
    with st.expander("完整策略文字 / 可複製", expanded=True):
        st.text_area("策略文字", full_strategy_text, height=360, label_visibility="collapsed")
        st.download_button("下載策略文字 TXT", full_strategy_text, "strategy_description.txt", "text/plain")

    with st.expander("完整總覽指標 / CSV", expanded=False):
        overall = pd.DataFrame([metrics])
        overall_display = display_df(overall)
        st.dataframe(overall_display, use_container_width=True)
        st.download_button("下載總覽 CSV", overall_display.to_csv(index=False), "overall_metrics.csv", "text/csv")

    with st.expander("年度 / 月度回報", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**年度回報**")
            yearly_display = display_df(result["yearly"])
            st.dataframe(yearly_display, use_container_width=True)
            st.download_button("下載年度回報 CSV", yearly_display.to_csv(index=False), "yearly_returns.csv", "text/csv")
        with c2:
            st.markdown("**月度回報矩陣**")
            monthly_display = display_df(result["monthly"])
            st.dataframe(monthly_display, use_container_width=True)
            st.download_button("下載月度回報 CSV", monthly_display.to_csv(index=False), "monthly_returns.csv", "text/csv")

    with st.expander("最差回撤", expanded=False):
        drawdowns_display = display_df(result["drawdowns"])
        st.dataframe(drawdowns_display, use_container_width=True)
        st.download_button("下載回撤 CSV", drawdowns_display.to_csv(index=False), "worst_drawdowns.csv", "text/csv")

    with st.expander("交易 / 每日權重", expanded=False):
        c1, c2 = st.columns([1.05, .95])
        with c1:
            st.markdown("**交易 / 再平衡紀錄**")
            trades_display = display_df(result["trades"])
            st.dataframe(trades_display, use_container_width=True)
            st.download_button("下載交易紀錄 CSV", trades_display.to_csv(index=False), "trades.csv", "text/csv")
        with c2:
            st.markdown("**每日權重：最近 250 日**")
            positions_display = display_df(result["positions"].tail(250))
            full_positions_display = display_df(result["positions"])
            st.dataframe(positions_display, use_container_width=True)
            st.download_button("下載完整每日權重 CSV", full_positions_display.to_csv(), "daily_weights.csv", "text/csv")

    with st.expander("資料來源 / 替代口徑", expanded=False):
        st.markdown("**資料來源**")
        data_audit_display = display_df(result["data_audit"])
        st.dataframe(data_audit_display, use_container_width=True)
        st.markdown(
            "<div class='small-note'>資料檢查會列出：使用本機 CSV 或公開資料 fallback、BIL/GLD 上市前用現金替代的期間、QLD/TQQQ 是否由 QQQ 每日漲跌模擬，以及 VIX 是否作為熊市防守訊號接入。</div>",
            unsafe_allow_html=True,
        )
else:
    st.info("先確認設定，按「開始回測」產生結果。")
