from __future__ import annotations

import hashlib
import io
import re
import urllib.parse
import urllib.request
from http.cookiejar import CookieJar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data" / "operator_imports" / "daily"
VIX_CSV = ROOT_DIR / "data" / "regime_external" / "vix" / "vix_daily.csv"

ASSETS = ("TQQQ", "QLD", "QQQ", "SPY", "GLD", "BIL")
SLEEVES = ("A_QQQ", "A_QLD", "A_TQQQ", "B_BIL", "B_QQQ", "B_SPY", "B_GLD")
SLEEVE_TO_ASSET = {
    "A_QQQ": "QQQ",
    "A_QLD": "QLD",
    "A_TQQQ": "TQQQ",
    "B_BIL": "BIL",
    "B_QQQ": "QQQ",
    "B_SPY": "SPY",
    "B_GLD": "GLD",
}


@dataclass
class ReloadLayer:
    drawdown: float
    fraction: float


@dataclass
class RecoveryLayer:
    tqqq_fraction: float
    condition: str = "sma50_ret20"
    ret20_threshold: float = 0.05
    qld_fraction: float = 1.0


@dataclass
class BacktestLabParams:
    start: pd.Timestamp
    end: pd.Timestamp
    initial_capital: float = 10_000.0
    recurring_contribution_amount: float = 0.0
    recurring_contribution_frequency: str = "none"  # none, monthly, quarterly, annual

    a_total: float = 0.55
    a_qqq: float = 0.0
    a_qld: float = 0.0
    a_tqqq: float = 1.0
    b_bil: float = 0.25
    b_qqq: float = 0.30
    b_spy: float = 0.20
    b_gld: float = 0.25

    trim_band: float = 1.375
    topup_band: float = 0.75
    min_trade_threshold: float = 0.02
    a_band_frequency: str = "quarterly"
    b_rebalance_frequency: str = "annual"
    qqq_sma_days: int = 200
    slope_lookback_days: int = 20

    topup_sell_order: tuple[str, ...] = ("B_GLD", "B_SPY", "B_QQQ")

    reload_enabled: bool = True
    reload_assets: tuple[str, ...] = ("QQQ", "SPY")
    reload_layers: tuple[ReloadLayer, ...] = field(
        default_factory=lambda: (
            ReloadLayer(-0.20, 0.10),
            ReloadLayer(-0.30, 0.15),
            ReloadLayer(-0.40, 0.20),
            ReloadLayer(-0.50, 0.25),
        )
    )
    reload_funding: str = "bil_only"  # bil_only or bil_then_gld
    reload_bil_floor: float = 0.05

    hb2_enabled: bool = True
    hb2_trigger_mode: str = "monthly_consecutive"
    hb2_months_required: int = 2
    hb2_daily_consecutive_days: int = 40
    hb2_rolling_window: int = 45
    hb2_rolling_required: int = 30
    hb2_vix_fast_enabled: bool = False
    hb2_vix_fast_threshold: float = 35.0
    hb2_drawdown_fast_enabled: bool = False
    hb2_drawdown_fast_threshold: float = -0.30
    hb2_qld_add_fraction: float = 0.0
    hb2_qld_keep_fraction: float = 1.0
    hb2_bridge_b_ratio_mode: str = "current_b"

    recovery_check_frequency: str = "monthly"
    recovery_ret20_threshold: float = 0.05
    recovery_tqqq_fraction: float = 0.50
    recovery_full_sma_days: int = 100
    recovery_months_before_sma100: int = 1
    recovery_layers: tuple[RecoveryLayer, ...] = field(
        default_factory=lambda: (
            RecoveryLayer(0.50, "sma50_ret20", 0.05),
            RecoveryLayer(1.00, "sma200_or_sma100_after_1m", 0.0),
        )
    )

    use_synthetic_leverage: bool = True
    bil_proxy_before_inception: bool = True
    gld_proxy_before_inception: bool = True
    one_way_slippage_bps: float = 0.0


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    cleaned = {k: max(float(v), 0.0) for k, v in weights.items()}
    total = sum(cleaned.values())
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    return {k: v / total for k, v in cleaned.items()}


def load_lab_data(
    params: BacktestLabParams,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_close: dict[str, pd.Series] = {}
    raw_open: dict[str, pd.Series] = {}
    audit_rows: list[dict[str, Any]] = []

    for symbol in ("QQQ", "SPY", "GLD", "BIL", "QLD", "TQQQ"):
        if params.use_synthetic_leverage and symbol in {"QLD", "TQQQ"}:
            continue
        path = _latest_daily_csv(symbol)
        if path is not None:
            df = _read_price_csv(path)
            file_name = path.name
            mode = "real adjusted price"
            notes = "open adjusted by adjusted_close / close ratio"
        else:
            df = _download_price_data(symbol, params.start, params.end)
            file_name = str(df.attrs.get("source", "public download"))
            mode = "public adjusted price"
            notes = str(df.attrs.get("notes", "public price download"))
        raw_close[symbol] = df["close"]
        raw_open[symbol] = df["open"]
        audit_rows.append(
            {
                "Symbol": symbol,
                "File": file_name,
                "Start": df.index.min().date().isoformat(),
                "End": df.index.max().date().isoformat(),
                "Rows": len(df),
                "Mode": mode,
                "Notes": notes,
            }
        )

    if VIX_CSV.exists():
        vix_df = _read_vix_csv(VIX_CSV)
        raw_close["VIX"] = vix_df["close"]
        raw_open["VIX"] = vix_df["open"]
        audit_rows.append(
            {
                "Symbol": "VIX",
                "File": str(VIX_CSV.relative_to(ROOT_DIR)),
                "Start": vix_df.index.min().date().isoformat(),
                "End": vix_df.index.max().date().isoformat(),
                "Rows": len(vix_df),
                "Mode": "signal only",
                "Notes": "VIX is used only for bear-defense fast trigger, not as a trading asset",
            }
        )
    else:
        try:
            vix_df = _download_price_data("^VIX", params.start, params.end)
            raw_close["VIX"] = vix_df["close"]
            raw_open["VIX"] = vix_df["open"]
            audit_rows.append(
                {
                    "Symbol": "VIX",
                    "File": str(vix_df.attrs.get("source", "public download:^VIX")),
                    "Start": vix_df.index.min().date().isoformat(),
                    "End": vix_df.index.max().date().isoformat(),
                    "Rows": len(vix_df),
                    "Mode": "public signal only",
                    "Notes": str(vix_df.attrs.get("notes", "VIX downloaded from public source; used only for bear-defense fast trigger")),
                }
            )
        except Exception as exc:
            audit_rows.append(
                {
                    "Symbol": "VIX",
                    "File": "missing",
                    "Start": "",
                    "End": "",
                    "Rows": 0,
                    "Mode": "missing optional signal",
                    "Notes": f"VIX fast trigger disabled because VIX data was unavailable: {exc}",
                }
            )

    index = sorted(set().union(*(s.index for s in raw_close.values())))
    index = pd.DatetimeIndex(index, name="date")
    close = pd.DataFrame({k: v.reindex(index) for k, v in raw_close.items()})
    open_ = pd.DataFrame({k: v.reindex(index) for k, v in raw_open.items()})

    if params.bil_proxy_before_inception:
        bil_note = _apply_cash_proxy(close, open_, "BIL", params.start, params.end)
        if bil_note is not None:
            audit_rows.append(bil_note)

    if params.gld_proxy_before_inception:
        gld_note = _apply_cash_proxy(close, open_, "GLD", params.start, params.end)
        if gld_note is not None:
            audit_rows.append(gld_note)

    if params.use_synthetic_leverage:
        qld_close, qld_open = _synthetic_daily_reset(close["QQQ"], leverage=2.0, annual_drag=0.020)
        tqqq_close, tqqq_open = _synthetic_daily_reset(close["QQQ"], leverage=3.0, annual_drag=0.035)
        close["QLD"] = qld_close
        open_["QLD"] = qld_open
        close["TQQQ"] = tqqq_close
        open_["TQQQ"] = tqqq_open
        audit_rows.append(
            {
                "Symbol": "QLD/TQQQ",
                "File": "synthetic daily-reset from QQQ",
                "Start": qld_close.first_valid_index().date().isoformat(),
                "End": qld_close.last_valid_index().date().isoformat(),
                "Rows": int(qld_close.notna().sum()),
                "Mode": "synthetic daily-reset proxy",
                "Notes": "QLD = 2x QQQ daily reset; TQQQ = 3x QQQ daily reset; annual drag included in proxy",
            }
        )

    required = _required_symbols(params)
    valid = close[list(required)].notna().all(axis=1) & open_[list(required)].notna().all(axis=1)
    close = close.loc[valid]
    open_ = open_.loc[valid]
    close = close.loc[(close.index >= params.start) & (close.index <= params.end)]
    open_ = open_.loc[close.index]
    if len(close) < 260:
        raise ValueError("not enough valid daily rows for the requested window")

    audit = pd.DataFrame(audit_rows)
    return_cols = list(ASSETS)
    if "VIX" in close.columns:
        return_cols.append("VIX")
    return open_[return_cols], close[return_cols], audit


def run_backtest_lab(params: BacktestLabParams) -> dict[str, Any]:
    open_, close, data_audit = load_lab_data(params)
    indicators = _build_indicators(close, params)
    period_flags = {
        "monthly": _period_end_flags(close.index, "monthly"),
        "quarterly": _period_end_flags(close.index, "quarterly"),
        "annual": _period_end_flags(close.index, "annual"),
    }

    base_target = _base_target(params)
    current = {"B_BIL": 1.0, "A_QQQ": 0.0, "A_QLD": 0.0, "A_TQQQ": 0.0, "B_QQQ": 0.0, "B_SPY": 0.0, "B_GLD": 0.0}
    current = {k: current.get(k, 0.0) for k in SLEEVES}
    pending_target: dict[str, float] | None = dict(base_target)
    pending_reason = "initial_deploy"

    equity = 1.0
    account_equity = float(params.initial_capital)
    total_recurring_contributions = 0.0
    contribution_count = 0
    cash_flows: list[tuple[pd.Timestamp, float]] = [(close.index[0], -float(params.initial_capital))]
    state = "NORMAL"
    bear_months = 0
    bear_daily = 0
    bear_history: list[bool] = []
    recovery_months = 0
    recovery_stage = 0
    qqq_layers: set[float] = set()
    spy_layers: set[float] = set()

    equity_rows: list[dict[str, Any]] = []
    position_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    contribution_rows: list[dict[str, Any]] = []

    prev_date = close.index[0]
    trades = 0
    turnover = 0.0
    qqq_reload_entries = 0
    spy_reload_entries = 0

    for i, date in enumerate(close.index):
        if i > 0:
            prev = close.index[i - 1]
            before_equity = equity
            current, equity = _apply_returns(current, equity, close.loc[prev], open_.loc[date])
            if before_equity > 0:
                account_equity *= equity / before_equity

        if pending_target is not None and i > 0:
            before = dict(current)
            current, trade_turnover, trade_count = _rebalance_to_target(current, pending_target, params.min_trade_threshold)
            if trade_count:
                cost = trade_turnover * (params.one_way_slippage_bps / 10_000.0)
                equity *= max(1.0 - cost, 0.0)
                account_equity *= max(1.0 - cost, 0.0)
                trades += trade_count
                turnover += trade_turnover
                trade_rows.append(
                    {
                        "Date": date.date().isoformat(),
                        "Reason": pending_reason,
                        "Turnover": trade_turnover,
                        "Cost": cost,
                        "Trades": trade_count,
                        **_target_columns(before, current, "Delta_"),
                    }
                )
            pending_target = None
            pending_reason = ""

        recurring_contribution = 0.0
        if i > 0 and _recurring_contribution_due(date, close.index[i - 1], params.recurring_contribution_frequency):
            recurring_contribution = max(float(params.recurring_contribution_amount), 0.0)
            if recurring_contribution > 0:
                before_account_equity = account_equity
                current = _add_contribution_to_current_weights(current, before_account_equity, recurring_contribution)
                account_equity += recurring_contribution
                total_recurring_contributions += recurring_contribution
                contribution_count += 1
                cash_flows.append((date, -recurring_contribution))
                contribution_rows.append(
                    {
                        "Date": date.date().isoformat(),
                        "Frequency": params.recurring_contribution_frequency,
                        "Amount": recurring_contribution,
                        "Account Equity Before": before_account_equity,
                        "Account Equity After": account_equity,
                    }
                )

        before_equity = equity
        current, equity = _apply_returns(current, equity, open_.loc[date], close.loc[date])
        if before_equity > 0:
            account_equity *= equity / before_equity
        current = _normalize_sleeve_weights(current)

        agg = _aggregate_weights(current)
        equity_rows.append(
            {
                "Date": date,
                "Equity": equity,
                "Account Equity": account_equity,
                "Recurring Contribution": recurring_contribution,
                "State": state,
                **{f"W_{asset}": agg.get(asset, 0.0) for asset in ASSETS},
            }
        )
        position_rows.append(
            {
                "Date": date,
                **{sleeve: current.get(sleeve, 0.0) for sleeve in SLEEVES},
                **{f"{asset}_Weight": agg.get(asset, 0.0) for asset in ASSETS},
            }
        )

        target = dict(current)
        reasons: list[str] = []
        forced = False

        bear_condition = bool(indicators.loc[date, "bear_condition"])
        bear_history.append(bear_condition)
        if bear_condition:
            bear_daily += 1
        else:
            bear_daily = 0

        if params.reload_enabled:
            target, reload_events = _apply_reload(target, close, date, params, qqq_layers, spy_layers)
            if reload_events:
                qqq_reload_entries += sum(1 for event in reload_events if event["Asset"] == "QQQ")
                spy_reload_entries += sum(1 for event in reload_events if event["Asset"] == "SPY")
                reasons.extend([event["Reason"] for event in reload_events])

        if _frequency_hit(date, period_flags, params.b_rebalance_frequency):
            target = _b_rebalance(target, base_target)
            reasons.append("B_REBALANCE")

        if _frequency_hit(date, period_flags, params.a_band_frequency):
            target, action = _a_band_action(target, base_target, params, indicators.loc[date])
            reasons.append(action)

        hb2_target, new_state, hb2_reason, bear_months, recovery_months, recovery_stage, forced_hb2 = _hb2_step(
            date=date,
            state=state,
            target=target,
            base_target=base_target,
            params=params,
            indicators=indicators,
            month_end_flags=period_flags["monthly"],
            bear_condition=bear_condition,
            bear_history=bear_history,
            bear_months=bear_months,
            bear_daily=bear_daily,
            recovery_months=recovery_months,
            recovery_stage=recovery_stage,
            realign=bool(reasons),
        )
        target = hb2_target
        if hb2_reason:
            reasons.append(hb2_reason)
        if new_state != state:
            state_rows.append({"Date": date.date().isoformat(), "From": state, "To": new_state, "Reason": hb2_reason})
            state = new_state
            forced = forced_hb2

        if reasons:
            threshold = 0.0 if forced else params.min_trade_threshold
            if _target_distance(current, target) >= threshold:
                pending_target = _normalize_sleeve_weights(target)
                pending_reason = "+".join(dict.fromkeys([r for r in reasons if r and not r.startswith("HOLD")]))
            else:
                pending_target = None

        prev_date = date

    equity_df = pd.DataFrame(equity_rows).set_index("Date")
    positions_df = pd.DataFrame(position_rows).set_index("Date")
    trades_df = pd.DataFrame(trade_rows)
    state_df = pd.DataFrame(state_rows)
    contributions_df = pd.DataFrame(contribution_rows)
    metrics = calculate_metrics(equity_df["Equity"], positions_df, trades, turnover)
    yearly = yearly_returns(equity_df["Equity"])
    monthly = monthly_returns_matrix(equity_df["Equity"])
    drawdowns = worst_drawdowns(equity_df["Equity"], top=10)

    metrics.update(
        {
            "Start": equity_df.index.min().date().isoformat(),
            "End": equity_df.index.max().date().isoformat(),
            "Initial Capital": float(params.initial_capital),
            "External Contributions": float(total_recurring_contributions),
            "Contribution Count": int(contribution_count),
            "Total Invested": float(params.initial_capital + total_recurring_contributions),
            "Final Equity": float(account_equity),
            "Return on Invested Capital": float(account_equity / (params.initial_capital + total_recurring_contributions) - 1.0)
            if params.initial_capital + total_recurring_contributions > 0
            else np.nan,
            "Money Weighted Return": _xirr(cash_flows + [(equity_df.index[-1], float(account_equity))]),
            "QQQ Reload Entries": qqq_reload_entries,
            "SPY Reload Entries": spy_reload_entries,
            "Days NORMAL": int((equity_df["State"] == "NORMAL").sum()),
            "Days HARD_BRIDGE": int((equity_df["State"] == "HARD_BRIDGE").sum()),
            "Days RECOVERY": int((equity_df["State"] == "RECOVERY").sum()),
        }
    )
    return {
        "metrics": metrics,
        "equity": equity_df,
        "positions": positions_df,
        "trades": trades_df,
        "contributions": contributions_df,
        "states": state_df,
        "yearly": yearly,
        "monthly": monthly,
        "drawdowns": drawdowns,
        "data_audit": data_audit,
        "params": params,
    }


def calculate_metrics(equity: pd.Series, positions: pd.DataFrame, trades: int, turnover: float) -> dict[str, Any]:
    returns = equity.pct_change().fillna(0.0)
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    final_multiple = float(equity.iloc[-1] / equity.iloc[0])
    cagr = final_multiple ** (1.0 / years) - 1.0
    running_peak = equity.cummax()
    dd = equity / running_peak - 1.0
    maxdd = float(dd.min())
    vol = float(returns.std(ddof=0) * np.sqrt(252))
    sharpe = float((returns.mean() * 252) / (returns.std(ddof=0) * np.sqrt(252))) if returns.std(ddof=0) > 0 else np.nan
    downside = returns[returns < 0]
    sortino = float((returns.mean() * 252) / (downside.std(ddof=0) * np.sqrt(252))) if len(downside) and downside.std(ddof=0) > 0 else np.nan
    calmar = float(cagr / abs(maxdd)) if maxdd < 0 else np.nan
    yearly = equity.resample("YE").last().pct_change(fill_method=None).dropna()
    monthly = equity.resample("ME").last().pct_change(fill_method=None).dropna()
    lev = positions[["A_QLD", "A_TQQQ"]].sum(axis=1)
    return {
        "CAGR": cagr,
        "Final Multiple": final_multiple,
        "MaxDD": maxdd,
        "Calmar": calmar,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Volatility": vol,
        "Worst Year": float(yearly.min()) if len(yearly) else np.nan,
        "Worst Month": float(monthly.min()) if len(monthly) else np.nan,
        "Trades": int(trades),
        "Turnover": float(turnover),
        "Avg Leveraged ETF": float(lev.mean()),
        "Max Leveraged ETF": float(lev.max()),
        "Avg TQQQ": float(positions["A_TQQQ"].mean()),
        "Avg QLD": float(positions["A_QLD"].mean()),
        "Avg QQQ": float((positions["A_QQQ"] + positions["B_QQQ"]).mean()),
        "Avg SPY": float(positions["B_SPY"].mean()),
        "Avg GLD": float(positions["B_GLD"].mean()),
        "Avg BIL": float(positions["B_BIL"].mean()),
    }


def yearly_returns(equity: pd.Series) -> pd.DataFrame:
    yearly = equity.resample("YE").last().pct_change(fill_method=None).dropna()
    return pd.DataFrame({"Year": yearly.index.year, "Return": yearly.values})


def monthly_returns_matrix(equity: pd.Series) -> pd.DataFrame:
    monthly = equity.resample("ME").last().pct_change(fill_method=None).dropna()
    if monthly.empty:
        return pd.DataFrame()
    frame = pd.DataFrame({"Year": monthly.index.year, "Month": monthly.index.month, "Return": monthly.values})
    matrix = frame.pivot(index="Year", columns="Month", values="Return")
    matrix.columns = [pd.Timestamp(2000, int(col), 1).strftime("%b") for col in matrix.columns]
    matrix["Annual"] = frame.groupby("Year")["Return"].apply(lambda x: (1.0 + x).prod() - 1.0)
    return matrix.reset_index()


def worst_drawdowns(equity: pd.Series, top: int = 10) -> pd.DataFrame:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    underwater = dd < 0
    rows = []
    start = None
    for date, is_underwater in underwater.items():
        if is_underwater and start is None:
            start = date
        if start is not None and not is_underwater:
            segment = dd.loc[start:date]
            trough = segment.idxmin()
            rows.append(
                {
                    "Start": start.date().isoformat(),
                    "Trough": trough.date().isoformat(),
                    "Recovery": date.date().isoformat(),
                    "MaxDD": float(segment.min()),
                    "Duration Days": int((date - start).days),
                }
            )
            start = None
    if start is not None:
        segment = dd.loc[start:]
        trough = segment.idxmin()
        rows.append(
            {
                "Start": start.date().isoformat(),
                "Trough": trough.date().isoformat(),
                "Recovery": "unrecovered",
                "MaxDD": float(segment.min()),
                "Duration Days": int((equity.index[-1] - start).days),
            }
        )
    return pd.DataFrame(rows).sort_values("MaxDD").head(top).reset_index(drop=True) if rows else pd.DataFrame()


def _latest_daily_csv(symbol: str) -> Path | None:
    files = sorted(DATA_DIR.glob(f"{symbol}_daily_*.csv"))
    if not files:
        return None
    return files[-1]


def _download_price_data(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    errors: list[str] = []
    for loader in (_download_price_data_yahoo, _download_price_data_stooq):
        try:
            return loader(symbol, start, end)
        except Exception as exc:
            errors.append(f"{loader.__name__}: {exc}")
    raise FileNotFoundError(f"no public price data returned for {symbol}; " + " | ".join(errors))


def _download_price_data_yahoo(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance unavailable") from exc

    start_ts = pd.Timestamp(start).tz_localize(None) - pd.Timedelta(days=7)
    end_ts = pd.Timestamp(end).tz_localize(None) + pd.Timedelta(days=1)
    df = yf.download(
        symbol,
        start=start_ts.date().isoformat(),
        end=end_ts.date().isoformat(),
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        raise FileNotFoundError(f"no public price data returned for {symbol}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    close_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    raw_close = pd.to_numeric(df["Close"], errors="coerce")
    adj_close = pd.to_numeric(df[close_col], errors="coerce")
    raw_open = pd.to_numeric(df["Open"], errors="coerce")
    factor = adj_close / raw_close.replace(0, np.nan)
    adj_open = (raw_open * factor).replace([np.inf, -np.inf], np.nan).fillna(adj_close)
    out = pd.DataFrame({"open": adj_open.to_numpy(), "close": adj_close.to_numpy()}, index=df.index)
    out = out.sort_index().dropna(how="all")
    out = out.loc[(out.index >= pd.Timestamp(start)) & (out.index <= pd.Timestamp(end))]
    out = out[~out.index.duplicated(keep="last")]
    if out.empty:
        raise FileNotFoundError(f"no usable public price rows for {symbol}")
    out.attrs["source"] = f"yfinance:{symbol}"
    out.attrs["notes"] = "downloaded from Yahoo Finance for public deployment; open adjusted by adjusted close ratio"
    return out


def _download_price_data_stooq(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    stooq_symbols = {
        "QQQ": "qqq.us",
        "SPY": "spy.us",
        "GLD": "gld.us",
        "BIL": "bil.us",
        "QLD": "qld.us",
        "TQQQ": "tqqq.us",
        "^VIX": "^vix",
    }
    stooq_symbol = stooq_symbols.get(symbol)
    if stooq_symbol is None:
        raise FileNotFoundError(f"no Stooq mapping for {symbol}")

    start_ts = pd.Timestamp(start).tz_localize(None) - pd.Timedelta(days=7)
    end_ts = pd.Timestamp(end).tz_localize(None) + pd.Timedelta(days=1)
    url = (
        "https://stooq.com/q/d/l/"
        f"?s={stooq_symbol}&i=d&d1={start_ts.strftime('%Y%m%d')}&d2={end_ts.strftime('%Y%m%d')}"
    )
    try:
        df = _read_stooq_csv(url)
    except (OSError, HTTPError, URLError) as exc:
        raise RuntimeError(f"Stooq download failed for {symbol}") from exc
    if df is None or df.empty or "Date" not in df.columns:
        raise FileNotFoundError(f"no Stooq rows for {symbol}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None)
    raw_open = pd.to_numeric(df.get("Open"), errors="coerce")
    raw_close = pd.to_numeric(df.get("Close"), errors="coerce")
    out = pd.DataFrame({"open": raw_open.to_numpy(), "close": raw_close.to_numpy()}, index=df["Date"])
    out = out.sort_index().dropna(how="all")
    out = out.loc[(out.index >= pd.Timestamp(start)) & (out.index <= pd.Timestamp(end))]
    out = out[~out.index.duplicated(keep="last")]
    if out.empty:
        raise FileNotFoundError(f"no usable Stooq rows for {symbol}")
    out.attrs["source"] = f"stooq:{stooq_symbol}"
    out.attrs["notes"] = "downloaded from Stooq public daily OHLC fallback; public data source used when Yahoo Finance is unavailable"
    return out


def _read_stooq_csv(url: str) -> pd.DataFrame:
    cookie_jar = CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))

    def fetch_text() -> str:
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; TQQQDrawdownLab/1.0)",
                "Accept": "text/csv,text/plain,*/*",
            },
        )
        with opener.open(request, timeout=20) as response:
            return response.read().decode("utf-8", "replace")

    text = fetch_text()
    if text.lstrip().startswith("Date,"):
        return pd.read_csv(io.StringIO(text))

    if "__verify" in text:
        match = re.search(r'c="([^"]+)",d=(\d+)', text)
        if not match:
            raise FileNotFoundError("Stooq verification challenge format changed")
        challenge, difficulty_text = match.groups()
        difficulty = int(difficulty_text)
        prefix = "0" * difficulty
        nonce = 0
        while nonce < 2_000_000:
            digest = hashlib.sha256(f"{challenge}{nonce}".encode("utf-8")).hexdigest()
            if digest.startswith(prefix):
                break
            nonce += 1
        else:
            raise TimeoutError("Stooq verification challenge did not resolve")

        body = urllib.parse.urlencode({"c": challenge, "n": str(nonce)}).encode("utf-8")
        verify_request = urllib.request.Request(
            "https://stooq.com/__verify",
            data=body,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; TQQQDrawdownLab/1.0)",
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        with opener.open(verify_request, timeout=20) as response:
            if response.status >= 400:
                raise RuntimeError(f"Stooq verification failed with status {response.status}")

        text = fetch_text()
        if text.lstrip().startswith("Date,"):
            return pd.read_csv(io.StringIO(text))

    raise FileNotFoundError("Stooq did not return CSV data")


def _read_price_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    date_col = "date" if "date" in df.columns else "Date"
    df[date_col] = pd.to_datetime(df[date_col]).dt.tz_localize(None)
    close_col = "adjusted_close" if "adjusted_close" in df.columns else "close"
    raw_close = pd.to_numeric(df["close"], errors="coerce")
    adj_close = pd.to_numeric(df[close_col], errors="coerce")
    raw_open = pd.to_numeric(df["open"], errors="coerce")
    factor = adj_close / raw_close.replace(0, np.nan)
    adj_open = (raw_open * factor).replace([np.inf, -np.inf], np.nan).fillna(adj_close)
    out = pd.DataFrame({"open": adj_open.to_numpy(), "close": adj_close.to_numpy()}, index=df[date_col])
    out = out.sort_index()
    return out[~out.index.duplicated(keep="last")]


def _read_vix_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    open_col = "vix_open" if "vix_open" in df.columns else "open"
    close_col = "vix_close" if "vix_close" in df.columns else "close"
    out = pd.DataFrame(
        {
            "open": pd.to_numeric(df[open_col], errors="coerce").to_numpy(),
            "close": pd.to_numeric(df[close_col], errors="coerce").to_numpy(),
        },
        index=df["date"],
    ).sort_index()
    return out[~out.index.duplicated(keep="last")]


def _apply_cash_proxy(
    close: pd.DataFrame,
    open_: pd.DataFrame,
    symbol: str,
    requested_start: pd.Timestamp,
    requested_end: pd.Timestamp,
) -> dict[str, Any] | None:
    first = close[symbol].first_valid_index()
    if first is None:
        close[symbol] = 100.0
        open_[symbol] = 100.0
        proxy_start = max(close.index.min(), requested_start)
        proxy_end = min(close.index.max(), requested_end)
        return {
            "Symbol": symbol,
            "File": "cash proxy",
            "Start": proxy_start.date().isoformat(),
            "End": proxy_end.date().isoformat(),
            "Rows": int(close[symbol].notna().sum()),
            "Mode": "cash proxy",
            "Notes": f"{symbol} has no local price data; held as flat cash proxy for entire window",
        }
    base = close.loc[first, symbol]
    close.loc[first:, symbol] = close.loc[first:, symbol] / base * 100.0
    open_.loc[first:, symbol] = open_.loc[first:, symbol] / base * 100.0
    proxy_mask = close.index < first
    close.loc[proxy_mask, symbol] = close.loc[proxy_mask, symbol].fillna(100.0)
    open_.loc[proxy_mask, symbol] = open_.loc[proxy_mask, symbol].fillna(100.0)
    proxy_start = max(close.index.min(), requested_start)
    proxy_end = min(first - pd.Timedelta(days=1), requested_end)
    if proxy_end < proxy_start:
        return None
    return {
        "Symbol": symbol,
        "File": "cash proxy before inception",
        "Start": proxy_start.date().isoformat(),
        "End": proxy_end.date().isoformat(),
        "Rows": int(((close.index >= proxy_start) & (close.index <= proxy_end)).sum()),
        "Mode": "cash proxy then real adjusted price",
        "Notes": f"{symbol} held as flat cash before first valid local price on {first.date().isoformat()}; real adjusted price used after that date",
    }


def _synthetic_daily_reset(close: pd.Series, leverage: float, annual_drag: float) -> tuple[pd.Series, pd.Series]:
    ret = close.pct_change(fill_method=None).fillna(0.0)
    daily_drag = annual_drag / 252.0
    synth_ret = leverage * ret - daily_drag
    synth = (1.0 + synth_ret).cumprod() * 100.0
    synth_open = synth.shift(1).fillna(synth)
    synth[close.isna()] = np.nan
    synth_open[close.isna()] = np.nan
    return synth, synth_open


def _required_symbols(params: BacktestLabParams) -> set[str]:
    required = {"QQQ", "BIL"}
    if params.b_spy > 0 or "SPY" in params.reload_assets:
        required.add("SPY")
    if params.b_gld > 0:
        required.add("GLD")
    if params.a_qqq > 0:
        required.add("QQQ")
    if params.a_qld > 0:
        required.add("QLD")
    if params.a_tqqq > 0:
        required.add("TQQQ")
    return required


def _build_indicators(close: pd.DataFrame, params: BacktestLabParams) -> pd.DataFrame:
    qqq = close["QQQ"]
    out = pd.DataFrame(index=close.index)
    out["qqq_close"] = qqq
    out["qqq_sma20"] = qqq.rolling(20).mean()
    out["qqq_sma50"] = qqq.rolling(50).mean()
    out["qqq_sma100"] = qqq.rolling(100).mean()
    out["qqq_sma200"] = qqq.rolling(params.qqq_sma_days).mean()
    out["qqq_sma200_slope"] = out["qqq_sma200"] - out["qqq_sma200"].shift(params.slope_lookback_days)
    out["qqq_ret20"] = qqq.pct_change(20, fill_method=None)
    out["qqq_ath"] = qqq.cummax()
    out["qqq_dd_ath"] = qqq / out["qqq_ath"] - 1.0
    out["spy_ath"] = close["SPY"].cummax()
    out["spy_dd_ath"] = close["SPY"] / out["spy_ath"] - 1.0
    out["vix_close"] = close["VIX"] if "VIX" in close.columns else np.nan
    out["bear_condition"] = (qqq < out["qqq_sma200"]) & (out["qqq_sma200_slope"] < 0)
    out["recovery_condition"] = (qqq > out["qqq_sma50"]) & (out["qqq_ret20"] > params.recovery_ret20_threshold)
    return out


def _base_target(params: BacktestLabParams) -> dict[str, float]:
    a_mix = normalize_weights({"A_QQQ": params.a_qqq, "A_QLD": params.a_qld, "A_TQQQ": params.a_tqqq})
    b_mix = normalize_weights({"B_BIL": params.b_bil, "B_QQQ": params.b_qqq, "B_SPY": params.b_spy, "B_GLD": params.b_gld})
    a_total = float(np.clip(params.a_total, 0.0, 1.0))
    b_total = 1.0 - a_total
    target = {sleeve: 0.0 for sleeve in SLEEVES}
    for key, weight in a_mix.items():
        target[key] = a_total * weight
    for key, weight in b_mix.items():
        target[key] = b_total * weight
    return _normalize_sleeve_weights(target)


def _apply_returns(weights: dict[str, float], equity: float, from_prices: pd.Series, to_prices: pd.Series) -> tuple[dict[str, float], float]:
    values = {}
    for sleeve, weight in weights.items():
        asset = SLEEVE_TO_ASSET[sleeve]
        ratio = to_prices[asset] / from_prices[asset] if from_prices[asset] and pd.notna(from_prices[asset]) else 1.0
        values[sleeve] = weight * float(ratio)
    total = sum(values.values())
    if total <= 0:
        return weights, equity
    return {k: v / total for k, v in values.items()}, equity * total


def _recurring_contribution_due(date: pd.Timestamp, prev_date: pd.Timestamp, frequency: str) -> bool:
    if frequency == "none":
        return False
    codes = {"monthly": "M", "quarterly": "Q", "annual": "Y"}
    code = codes.get(frequency)
    if code is None:
        return False
    return date.to_period(code) != prev_date.to_period(code)


def _add_contribution_to_current_weights(
    current: dict[str, float],
    account_equity: float,
    contribution: float,
) -> dict[str, float]:
    if contribution <= 0:
        return current
    current = _normalize_sleeve_weights(current)
    if account_equity <= 0:
        return current
    total = account_equity + contribution
    if total <= 0:
        return current
    return {
        sleeve: (current.get(sleeve, 0.0) * account_equity + current.get(sleeve, 0.0) * contribution) / total
        for sleeve in SLEEVES
    }


def _xirr(cash_flows: list[tuple[pd.Timestamp, float]]) -> float:
    flows = [(pd.Timestamp(date), float(amount)) for date, amount in cash_flows if amount]
    if len(flows) < 2 or not any(amount < 0 for _, amount in flows) or not any(amount > 0 for _, amount in flows):
        return np.nan
    start = flows[0][0]

    def npv(rate: float) -> float:
        total = 0.0
        for date, amount in flows:
            years = max((date - start).days / 365.25, 0.0)
            total += amount / ((1.0 + rate) ** years)
        return total

    low = -0.9999
    high = 10.0
    low_value = npv(low)
    high_value = npv(high)
    expand_count = 0
    while low_value * high_value > 0 and high < 1_000 and expand_count < 20:
        high *= 2
        high_value = npv(high)
        expand_count += 1
    if low_value * high_value > 0:
        return np.nan
    for _ in range(100):
        mid = (low + high) / 2
        mid_value = npv(mid)
        if abs(mid_value) < 1e-8:
            return float(mid)
        if low_value * mid_value <= 0:
            high = mid
            high_value = mid_value
        else:
            low = mid
            low_value = mid_value
    return float((low + high) / 2)


def _normalize_sleeve_weights(weights: dict[str, float]) -> dict[str, float]:
    cleaned = {k: max(float(weights.get(k, 0.0)), 0.0) for k in SLEEVES}
    total = sum(cleaned.values())
    if total <= 0:
        cleaned["B_BIL"] = 1.0
        total = 1.0
    return {k: v / total for k, v in cleaned.items()}


def _aggregate_weights(weights: dict[str, float]) -> dict[str, float]:
    out = {asset: 0.0 for asset in ASSETS}
    for sleeve, weight in weights.items():
        out[SLEEVE_TO_ASSET[sleeve]] += weight
    return out


def _rebalance_to_target(current: dict[str, float], target: dict[str, float], threshold: float) -> tuple[dict[str, float], float, int]:
    target = _normalize_sleeve_weights(target)
    deltas = {k: target[k] - current.get(k, 0.0) for k in SLEEVES}
    traded = {k: v for k, v in deltas.items() if abs(v) >= threshold}
    if not traded:
        return current, 0.0, 0
    next_weights = dict(current)
    for key in traded:
        next_weights[key] = target[key]
    residual = 1.0 - sum(next_weights.values())
    next_weights["B_BIL"] = max(next_weights.get("B_BIL", 0.0) + residual, 0.0)
    turnover = sum(abs(v) for v in traded.values())
    return _normalize_sleeve_weights(next_weights), turnover, len(traded)


def _period_end_flags(index: pd.DatetimeIndex, freq: str) -> pd.Series:
    flags = pd.Series(False, index=index)
    if freq == "none" or len(index) < 2:
        return flags
    code = {"monthly": "M", "quarterly": "Q", "annual": "Y"}[freq]
    periods = index.to_period(code)
    vals = np.zeros(len(index), dtype=bool)
    vals[:-1] = periods[:-1] != periods[1:]
    flags[:] = vals
    return flags


def _frequency_hit(date: pd.Timestamp, flags: dict[str, pd.Series], freq: str) -> bool:
    if freq == "daily":
        return True
    if freq == "none":
        return False
    return bool(flags[freq].loc[date])


def _b_rebalance(current: dict[str, float], base: dict[str, float]) -> dict[str, float]:
    target = dict(current)
    b_total = sum(current[s] for s in ("B_BIL", "B_QQQ", "B_SPY", "B_GLD"))
    base_b_total = sum(base[s] for s in ("B_BIL", "B_QQQ", "B_SPY", "B_GLD"))
    if b_total <= 0 or base_b_total <= 0:
        return target
    for s in ("B_BIL", "B_QQQ", "B_SPY", "B_GLD"):
        target[s] = b_total * base[s] / base_b_total
    return _normalize_sleeve_weights(target)


def _a_band_action(
    current: dict[str, float],
    base: dict[str, float],
    params: BacktestLabParams,
    indicators: pd.Series,
) -> tuple[dict[str, float], str]:
    target = dict(current)
    a_sleeves = ("A_QQQ", "A_QLD", "A_TQQQ")
    a_actual = sum(target[s] for s in a_sleeves)
    a_target = sum(base[s] for s in a_sleeves)
    if a_target <= 0:
        return target, "HOLD_A_ZERO"

    if a_actual > a_target * params.trim_band:
        trim_amount = a_actual - a_target
        scale = a_target / a_actual
        for s in a_sleeves:
            target[s] *= scale
        target["B_BIL"] += trim_amount
        return _normalize_sleeve_weights(target), "TRIM_A_TO_TARGET"

    qqq_sma = indicators.get("qqq_sma200", np.nan)
    qqq_close = indicators.get("qqq_close", np.nan)
    qqq_above_sma = bool(pd.notna(qqq_sma) and pd.notna(qqq_close) and qqq_close > qqq_sma)
    if a_actual < a_target * params.topup_band and qqq_above_sma:
        return _topup_a(target, base, params), "TOPUP_A_TO_TARGET"
    if a_actual < a_target * params.topup_band:
        return target, "HOLD_A_QQQ_BELOW_SMA200"
    return target, "HOLD_WITHIN_BAND"


def _topup_a(current: dict[str, float], base: dict[str, float], params: BacktestLabParams) -> dict[str, float]:
    target = dict(current)
    a_sleeves = ("A_QQQ", "A_QLD", "A_TQQQ")
    a_actual = sum(target[s] for s in a_sleeves)
    a_target = sum(base[s] for s in a_sleeves)
    need = max(a_target - a_actual, 0.0)
    if need <= 0:
        return target
    available = target.get("B_BIL", 0.0)
    take = min(need, available)
    target["B_BIL"] -= take
    need -= take
    for sleeve in params.topup_sell_order:
        if need <= 1e-12:
            break
        take = min(need, target.get(sleeve, 0.0))
        target[sleeve] -= take
        need -= take
    add = max(a_target - a_actual - need, 0.0)
    if add <= 0:
        return target
    base_a_total = sum(base[s] for s in a_sleeves)
    for s in a_sleeves:
        target[s] += add * (base[s] / base_a_total if base_a_total else 0.0)
    return _normalize_sleeve_weights(target)


def _apply_reload(
    current: dict[str, float],
    close: pd.DataFrame,
    date: pd.Timestamp,
    params: BacktestLabParams,
    qqq_layers: set[float],
    spy_layers: set[float],
) -> tuple[dict[str, float], list[dict[str, str]]]:
    target = dict(current)
    events: list[dict[str, str]] = []
    for asset, sleeve, layers_seen in (("QQQ", "B_QQQ", qqq_layers), ("SPY", "B_SPY", spy_layers)):
        if asset not in params.reload_assets:
            continue
        price = close.loc[date, asset]
        ath = close.loc[:date, asset].cummax().iloc[-1]
        if price >= ath:
            layers_seen.clear()
        dd = price / ath - 1.0
        for layer in params.reload_layers:
            if dd <= layer.drawdown and layer.drawdown not in layers_seen:
                funding_base = target["B_BIL"] if params.reload_funding == "bil_only" else target["B_BIL"] + target["B_GLD"]
                buy_amount = funding_base * layer.fraction
                max_bil_spend = max(target["B_BIL"] - params.reload_bil_floor, 0.0)
                spend_from_bil = min(buy_amount, max_bil_spend)
                remaining = buy_amount - spend_from_bil
                spend_from_gld = 0.0
                if params.reload_funding == "bil_then_gld" and remaining > 0:
                    spend_from_gld = min(remaining, target["B_GLD"])
                spend = spend_from_bil + spend_from_gld
                if spend <= 1e-12:
                    continue
                target["B_BIL"] -= spend_from_bil
                target["B_GLD"] -= spend_from_gld
                target[sleeve] += spend
                layers_seen.add(layer.drawdown)
                events.append({"Asset": asset, "Reason": f"{asset}_RELOAD_{int(abs(layer.drawdown) * 100)}"})
    return _normalize_sleeve_weights(target), events


def _hb2_step(
    *,
    date: pd.Timestamp,
    state: str,
    target: dict[str, float],
    base_target: dict[str, float],
    params: BacktestLabParams,
    indicators: pd.DataFrame,
    month_end_flags: pd.Series,
    bear_condition: bool,
    bear_history: list[bool],
    bear_months: int,
    bear_daily: int,
    recovery_months: int,
    recovery_stage: int,
    realign: bool,
) -> tuple[dict[str, float], str, str, int, int, int, bool]:
    if not params.hb2_enabled:
        return target, state, "", bear_months, recovery_months, recovery_stage, False

    month_end = bool(month_end_flags.loc[date])
    if month_end:
        bear_months = bear_months + 1 if bear_condition else 0

    row = indicators.loc[date]
    qqq_close = row.get("qqq_close", np.nan)
    qqq_sma = row.get("qqq_sma200", np.nan)
    qqq_below_sma = bool(pd.notna(qqq_close) and pd.notna(qqq_sma) and qqq_close < qqq_sma)
    vix_close = row.get("vix_close", np.nan)
    vix_fast_trigger = bool(
        params.hb2_vix_fast_enabled
        and qqq_below_sma
        and pd.notna(vix_close)
        and float(vix_close) > params.hb2_vix_fast_threshold
    )
    qqq_dd_ath = row.get("qqq_dd_ath", np.nan)
    drawdown_fast_trigger = bool(
        params.hb2_drawdown_fast_enabled
        and qqq_below_sma
        and pd.notna(qqq_dd_ath)
        and float(qqq_dd_ath) <= params.hb2_drawdown_fast_threshold
    )

    rolling_count = sum(bear_history[-params.hb2_rolling_window :])
    trigger = False
    trigger_reason = "ENTER_HARD_BRIDGE"
    if vix_fast_trigger:
        trigger = True
        trigger_reason = "ENTER_HARD_BRIDGE_FAST_VIX"
    elif drawdown_fast_trigger:
        trigger = True
        trigger_reason = "ENTER_HARD_BRIDGE_FAST_DRAWDOWN"

    if params.hb2_trigger_mode == "monthly_consecutive":
        trigger = trigger or (month_end and bear_months >= params.hb2_months_required)
    elif params.hb2_trigger_mode == "daily_consecutive":
        trigger = trigger or (bear_daily >= params.hb2_daily_consecutive_days)
    elif params.hb2_trigger_mode == "rolling_count":
        trigger = trigger or (rolling_count >= params.hb2_rolling_required)
    elif params.hb2_trigger_mode == "monthly_or_rolling":
        trigger = trigger or (month_end and bear_months >= params.hb2_months_required) or (rolling_count >= params.hb2_rolling_required)

    if state in ("NORMAL", "RECOVERY") and trigger:
        return _hb2_bridge_target(base_target, params), "HARD_BRIDGE", trigger_reason, bear_months, 0, 0, True

    recovery_layers = _effective_recovery_layers(params)
    check_recovery = month_end if params.recovery_check_frequency == "monthly" else True
    if state == "HARD_BRIDGE" and recovery_layers and check_recovery and _recovery_condition_met(
        recovery_layers[0], row, recovery_months, params
    ):
        if _recovery_layer_is_full(recovery_layers[0]):
            return dict(base_target), "NORMAL", "FULL_RECOVERY", 0, 0, 0, True
        return _hb2_recovery_target(base_target, recovery_layers[0]), "RECOVERY", "ENTER_RECOVERY", bear_months, 0, 1, True

    if state == "HARD_BRIDGE" and realign:
        return _hb2_bridge_target(base_target, params), state, "HARD_BRIDGE_TARGET", bear_months, recovery_months, recovery_stage, False

    if state == "RECOVERY":
        recovery_months = recovery_months + 1 if month_end else recovery_months
        if recovery_stage <= 0:
            recovery_stage = 1
        if recovery_layers and recovery_stage < len(recovery_layers):
            next_layer = recovery_layers[recovery_stage]
            if _recovery_condition_met(next_layer, row, recovery_months, params):
                next_stage = recovery_stage + 1
                if _recovery_layer_is_full(next_layer):
                    return dict(base_target), "NORMAL", "FULL_RECOVERY", 0, 0, 0, True
                return _hb2_recovery_target(base_target, next_layer), state, "RECOVERY_TARGET", bear_months, recovery_months, next_stage, True
        if realign:
            active_layer = recovery_layers[min(recovery_stage, len(recovery_layers)) - 1] if recovery_layers else RecoveryLayer(
                params.recovery_tqqq_fraction, "sma50_ret20", params.recovery_ret20_threshold
            )
            return _hb2_recovery_target(base_target, active_layer), state, "RECOVERY_TARGET", bear_months, recovery_months, recovery_stage, False

    return target, state, "", bear_months, recovery_months, recovery_stage, False


def _hb2_bridge_target(base: dict[str, float], params: BacktestLabParams) -> dict[str, float]:
    target = dict(base)
    released = base["A_TQQQ"] + base["A_QLD"] * (1.0 - params.hb2_qld_keep_fraction)
    target["A_TQQQ"] = 0.0
    target["A_QLD"] = base["A_QLD"] * params.hb2_qld_keep_fraction + released * params.hb2_qld_add_fraction
    remaining = released * (1.0 - params.hb2_qld_add_fraction)
    b_total = sum(base[s] for s in ("B_BIL", "B_QQQ", "B_SPY", "B_GLD"))
    if b_total <= 0:
        target["B_BIL"] += remaining
    else:
        for s in ("B_BIL", "B_QQQ", "B_SPY", "B_GLD"):
            target[s] += remaining * base[s] / b_total
    return _normalize_sleeve_weights(target)


def _effective_recovery_layers(params: BacktestLabParams) -> tuple[RecoveryLayer, ...]:
    if params.recovery_layers:
        return params.recovery_layers
    return (
        RecoveryLayer(params.recovery_tqqq_fraction, "sma50_ret20", params.recovery_ret20_threshold, 1.0),
        RecoveryLayer(1.0, "sma200_or_sma100_after_1m", 0.0, 1.0),
    )


def _recovery_condition_met(
    layer: RecoveryLayer,
    row: pd.Series,
    recovery_months: int,
    params: BacktestLabParams,
) -> bool:
    qqq_close = row["qqq_close"]

    def above(column: str) -> bool:
        value = row.get(column)
        return bool(pd.notna(value) and qqq_close > value)

    if layer.condition == "sma20":
        return above("qqq_sma20")
    if layer.condition == "sma50":
        return above("qqq_sma50")
    if layer.condition == "sma50_ret20":
        ret20 = row.get("qqq_ret20")
        return bool(above("qqq_sma50") and pd.notna(ret20) and ret20 > layer.ret20_threshold)
    if layer.condition == "sma100":
        return above("qqq_sma100")
    if layer.condition == "sma100_after_1m":
        return bool(recovery_months >= params.recovery_months_before_sma100 and above("qqq_sma100"))
    if layer.condition == "sma200":
        return above("qqq_sma200")
    if layer.condition == "sma200_or_sma100_after_1m":
        return bool(
            above("qqq_sma200")
            or (recovery_months >= params.recovery_months_before_sma100 and above("qqq_sma100"))
        )
    return False


def _hb2_recovery_target(base: dict[str, float], layer: RecoveryLayer) -> dict[str, float]:
    target = {s: 0.0 for s in SLEEVES}
    target["A_QQQ"] = base["A_QQQ"]
    target["A_QLD"] = base["A_QLD"] * max(min(layer.qld_fraction, 1.0), 0.0)
    target["A_TQQQ"] = base["A_TQQQ"] * max(min(layer.tqqq_fraction, 1.0), 0.0)
    target["B_SPY"] = base["B_SPY"]
    target["B_GLD"] = base["B_GLD"]
    used = sum(target.values())
    remaining = max(1.0 - used, 0.0)
    base_bil_qqq = base["B_BIL"] + base["B_QQQ"]
    if base_bil_qqq > 0:
        target["B_BIL"] = remaining * base["B_BIL"] / base_bil_qqq
        target["B_QQQ"] = remaining * base["B_QQQ"] / base_bil_qqq
    else:
        target["B_BIL"] = remaining
    return _normalize_sleeve_weights(target)


def _recovery_layer_is_full(layer: RecoveryLayer) -> bool:
    return layer.tqqq_fraction >= 0.999 and layer.qld_fraction >= 0.999


def _target_distance(current: dict[str, float], target: dict[str, float]) -> float:
    return max(abs(target.get(s, 0.0) - current.get(s, 0.0)) for s in SLEEVES)


def _target_columns(before: dict[str, float], after: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}{s}": after.get(s, 0.0) - before.get(s, 0.0) for s in SLEEVES}
