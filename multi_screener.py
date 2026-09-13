import pandas as pd
import yfinance as yf
import time
import os
import sys

from datetime import datetime
from zoneinfo import ZoneInfo


# ============================================================
# FILE / OUTPUT SETTINGS
# ============================================================

SYMBOL_FILE = "nifty100_symbols.csv"
OUTPUT_FILE = "docs/index.html"


# ============================================================
# SST SETTINGS
# ============================================================

SST_LOOKBACK = 20
SST_TARGET = 6.0


# ============================================================
# BLSH RSI SETTINGS
# ============================================================

BLSH_LOOKBACK = 25
BLSH_TRIGGER_PCT = 6.5  # Buy Trigger: +6.5% from new 25D low
BLSH_TARGET_PCT = 3.14  # Sell Price: +3.14% from Buy Trigger

RSI_PERIOD = 14
RSI_LIMIT = 36


# ============================================================
# HISTORICAL ANALYSIS
# ============================================================

ANALYSIS_DAYS = 252


# ============================================================
# DATE FORMAT
# ============================================================

def fmt_date(value):
    try:
        return pd.Timestamp(value).strftime("%d-%b-%Y")
    except Exception:
        return "N/A"


# ============================================================
# BULLISH CANDLE
# ============================================================

def bullish(row):
    return (
        pd.notna(row["Open"])
        and pd.notna(row["Close"])
        and float(row["Close"]) > float(row["Open"])
    )


# ============================================================
# RSI
# ============================================================

def rsi(series, period=14):
    delta = series.diff()

    gain = (
        delta
        .clip(lower=0)
        .ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period
        )
        .mean()
    )

    loss = (
        (-delta)
        .clip(lower=0)
        .ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period
        )
        .mean()
    )

    rs = gain / loss.replace(0, float("nan"))
    result = 100 - (100 / (1 + rs))

    return result.fillna(100)


# ============================================================
# LOAD NIFTY 100 SYMBOLS
# ============================================================

def load_symbols():
    df = pd.read_csv(SYMBOL_FILE)
    df.columns = df.columns.str.strip().str.lower()

    if "symbol" not in df.columns:
        raise ValueError("CSV must contain a symbol column")

    if "company" not in df.columns:
        df["company"] = df["symbol"]

    df = df.dropna(subset=["symbol"]).copy()

    return [
        (str(row["symbol"]).strip(), str(row["company"]).strip())
        for _, row in df.iterrows()
    ]


# ============================================================
# DOWNLOAD DAILY DATA
# ============================================================

def download(symbol):
    try:
        df = yf.download(
            symbol + ".NS",
            period="18mo",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False
        )
    except Exception as e:
        print(f"  Download error: {e}")
        return None

    if df is None or df.empty:
        return None

    # Handle Yahoo Finance MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close"]

    if any(column not in df.columns for column in required):
        return None

    df = df.dropna(subset=required).copy()

    for column in required:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=required)
    df.index = pd.to_datetime(df.index)

    return df


# ============================================================
# RESAMPLE OHLC
# ============================================================

def resample_ohlc(df, rule):
    result = pd.DataFrame({
        "Open": df["Open"].resample(rule).first(),
        "High": df["High"].resample(rule).max(),
        "Low": df["Low"].resample(rule).min(),
        "Close": df["Close"].resample(rule).last()
    })

    return result.dropna()


# ============================================================
# MWD SCREEN
# ============================================================

def mwd_screen(symbol, company, df):
    daily = df.copy()
    weekly = resample_ohlc(df, "W-FRI")
    monthly = resample_ohlc(df, "ME")

    if len(daily) < 252 or len(weekly) < 3 or len(monthly) < 3:
        return None

    daily_current = daily.iloc[-1]
    weekly_this = weekly.iloc[-1]       # Current (still forming) week
    weekly_current = weekly.iloc[-2]    # Last completed week
    monthly_current = monthly.iloc[-2]   # Last completed month

    # Daily, monthly, last completed week, AND current week must be bullish
    if not (
        bullish(daily_current)
        and bullish(weekly_this)
        and bullish(weekly_current)
        and bullish(monthly_current)
    ):
        return None

    last_price = float(daily_current["Close"])
    high_52 = float(daily["High"].tail(252).max())

    # Do not show stocks already at / above 52W high
    if last_price >= high_52:
        return None

    distance = ((last_price - high_52) / high_52) * 100

    monthly_open = float(monthly_current["Open"])
    monthly_close = float(monthly_current["Close"])

    if monthly_open == 0:
        return None

    monthly_rise = ((monthly_close - monthly_open) / monthly_open) * 100

    return {
        "symbol": symbol,
        "company": company,
        "date": fmt_date(daily.index[-1]),
        "price": round(last_price, 2),
        "high52": round(high_52, 2),
        "distance": round(distance, 2),
        "rise": round(monthly_rise, 2)
    }


# ============================================================
# NEW 25-DAY LOW & 20-DAY HIGH HELPERS
# ============================================================

def is_new_low(df, i, lookback):
    if i < lookback:
        return False
    current_low = float(df["Low"].iloc[i])
    previous_low = float(df["Low"].iloc[i - lookback:i].min())
    return current_low < previous_low


def is_new_high(df, i, lookback):
    if i < lookback:
        return False
    current_high = float(df["High"].iloc[i])
    previous_high = float(df["High"].iloc[i - lookback:i].max())
    return current_high > previous_high


# ============================================================
# SST TRANSACTION LOG & STATISTICS
# ============================================================

def get_sst_transactions(df, lookback, target_pct):
    data = (
        df.tail(ANALYSIS_DAYS + lookback)
        .copy()
        .reset_index()
        .rename(columns={"index": "Date"})
    )

    if len(data) <= lookback:
        return []

    armed = False
    open_trades = []
    transactions = []

    for i in range(lookback, len(data)):
        if is_new_low(data, i, lookback):
            armed = True

        if armed and is_new_high(data, i, lookback):
            entry = float(data["High"].iloc[i - lookback:i].max())
            target = entry * (1 + target_pct / 100)

            open_trades.append({
                "date": data["Date"].iloc[i],
                "buy_index": i,
                "buy_price": entry,
                "sell_target": target,
            })
            armed = False

        current_high = float(data["High"].iloc[i])
        new_low_today = is_new_low(data, i, lookback)
        still_open = []

        for t in open_trades:
            if i <= t["buy_index"]:
                still_open.append(t)
                continue

            if current_high >= t["sell_target"]:
                met_date = data["Date"].iloc[i]
                transactions.append({
                    "date": t["date"],
                    "buy_price": t["buy_price"],
                    "sell_target": t["sell_target"],
                    "achieved": "Yes",
                    "sell_hit_date": met_date,
                    "days_taken": (pd.Timestamp(met_date) - pd.Timestamp(t["date"])).days,
                })
                continue

            if new_low_today:
                transactions.append({
                    "date": t["date"],
                    "buy_price": t["buy_price"],
                    "sell_target": t["sell_target"],
                    "achieved": "No",
                    "sell_hit_date": None,
                    "days_taken": None,
                })
                continue

            still_open.append(t)

        open_trades = still_open

    for t in open_trades:
        transactions.append({
            "date": t["date"],
            "buy_price": t["buy_price"],
            "sell_target": t["sell_target"],
            "achieved": "Open",
            "sell_hit_date": None,
            "days_taken": None,
        })

    transactions.sort(key=lambda t: t["date"])
    return transactions


def cycle_statistics(df, lookback, target_pct):
    transactions = get_sst_transactions(df, lookback, target_pct)
    yes = sum(1 for t in transactions if t["achieved"] == "Yes")
    no = sum(1 for t in transactions if t["achieved"] == "No")
    completed = yes + no

    strike = (yes / completed * 100) if completed else 0
    return yes, no, round(strike, 2)


# ============================================================
# SST CURRENT SCREEN
# ============================================================

def sst_screen(symbol, company, df):
    if len(df) < SST_LOOKBACK + 1:
        return None

    close = float(df["Close"].iloc[-1])
    high20 = float(df["High"].tail(SST_LOOKBACK).max())
    away = ((high20 - close) / high20) * 100 if high20 else 0

    yes, no, strike = cycle_statistics(df, SST_LOOKBACK, SST_TARGET)

    return {
        "symbol": symbol,
        "company": company,
        "price": round(close, 2),
        "high20": round(high20, 2),
        "away": round(away, 2),
        "yes": yes,
        "no": no,
        "strike": strike
    }


# ============================================================
# BLSH HISTORICAL STATISTICS & CURRENT SCREEN
# ============================================================

def blsh_history_1_year(df):
    required_bars = BLSH_LOOKBACK + RSI_PERIOD + ANALYSIS_DAYS
    data = df.copy()
    data["RSI14"] = rsi(data["Close"], RSI_PERIOD)
    data = data.tail(required_bars).copy()

    if len(data) <= BLSH_LOOKBACK + RSI_PERIOD:
        return 0, 0, 0.0

    data = data.reset_index().rename(columns={"index": "Date"})

    state = "WAIT_SETUP"
    reference_low = None
    trigger_price = None
    target_price = None
    yes = 0
    no = 0

    for i in range(BLSH_LOOKBACK, len(data)):
        current_low = float(data["Low"].iloc[i])
        current_high = float(data["High"].iloc[i])
        current_rsi = float(data["RSI14"].iloc[i])
        fresh_new_low = is_new_low(data, i, BLSH_LOOKBACK)

        if state == "IN_TRADE":
            # Check target hit prior to evaluating new low resets
            if current_high >= target_price:
                yes += 1
                reference_low, trigger_price, target_price = None, None, None
                state = "WAIT_RESET"
                continue

            if fresh_new_low:
                no += 1
                if current_rsi < RSI_LIMIT:
                    reference_low = current_low
                    trigger_price = reference_low * (1 + BLSH_TRIGGER_PCT / 100)
                    target_price = trigger_price * (1 + BLSH_TARGET_PCT / 100)
                    state = "WAIT_TRIGGER"
                else:
                    reference_low, trigger_price, target_price = None, None, None
                    state = "WAIT_SETUP"
            continue

        if state == "WAIT_RESET":
            if fresh_new_low:
                if current_rsi < RSI_LIMIT:
                    reference_low = current_low
                    trigger_price = reference_low * (1 + BLSH_TRIGGER_PCT / 100)
                    target_price = trigger_price * (1 + BLSH_TARGET_PCT / 100)
                    state = "WAIT_TRIGGER"
                else:
                    reference_low, trigger_price, target_price = None, None, None
                    state = "WAIT_SETUP"
            continue

        if fresh_new_low:
            if current_rsi < RSI_LIMIT:
                reference_low = current_low
                trigger_price = reference_low * (1 + BLSH_TRIGGER_PCT / 100)
                target_price = trigger_price * (1 + BLSH_TARGET_PCT / 100)
                state = "WAIT_TRIGGER"
            else:
                reference_low, trigger_price, target_price = None, None, None
                state = "WAIT_SETUP"
            continue

        if state == "WAIT_TRIGGER":
            if current_high >= trigger_price:
                if current_high >= target_price:
                    yes += 1
                    reference_low, trigger_price, target_price = None, None, None
                    state = "WAIT_RESET"
                else:
                    state = "IN_TRADE"

    completed = yes + no
    strike = (yes / completed * 100) if completed else 0.0
    return yes, no, round(strike, 2)


def blsh_screen(symbol, company, df):
    if len(df) < (BLSH_LOOKBACK + RSI_PERIOD):
        return None

    work = df.copy()
    work["RSI14"] = rsi(work["Close"], RSI_PERIOD)
    i = len(work) - 1

    current = work.iloc[-1]
    current_rsi = float(current["RSI14"])
    current_low = float(current["Low"])
    cmp = float(current["Close"])

    previous25_low = float(work["Low"].iloc[i - BLSH_LOOKBACK:i].min())
    new_25_day_low = current_low < previous25_low

    if not (current_rsi < RSI_LIMIT and new_25_day_low):
        return None

    low_25_day = current_low
    trigger_price = low_25_day * (1 + BLSH_TRIGGER_PCT / 100)
    target_price = trigger_price * (1 + BLSH_TARGET_PCT / 100)
    trigger_away = ((trigger_price - cmp) / trigger_price) * 100 if trigger_price else 0

    yes, no, strike = blsh_history_1_year(df)

    return {
        "symbol": symbol,
        "company": company,
        "cmp": round(cmp, 2),
        "low_25_day": round(low_25_day, 2),
        "trigger_price": round(trigger_price, 2),
        "target_price": round(target_price, 2),
        "trigger_away": round(trigger_away, 2),
        "rsi": round(current_rsi, 2),
        "yes": yes,
        "no": no,
        "strike": strike,
        "last_low_date": fmt_date(work.index[-1])
    }


# ============================================================
# HTML BUILDER
# ============================================================

def build_html(mwd, sst, blsh, updated, total_stocks):
    def rows_mwd():
        return "".join(
            f"""
            <tr data-distance="{x['distance']}" data-rise="{x['rise']}">
                <td data-label="Symbol"><strong>{x['symbol']}</strong></td>
                <td class="company" data-label="Company">{x['company']}</td>
                <td class="num" data-label="CMP">₹{x['price']:,.2f}</td>
                <td class="num" data-label="52W High">₹{x['high52']:,.2f}</td>
                <td class="num" data-label="52W Distance">
                    <span class="chip {'negative' if x['distance'] < 0 else 'positive'}">
                        {x['distance']:.2f}%
                    </span>
                </td>
                <td class="num" data-label="Monthly Rise">
                    <span class="chip positive">+{x['rise']:.2f}%</span>
                </td>
            </tr>
            """
            for x in mwd
        )

    def rows_sst():
        return "".join(
            f"""
            <tr>
                <td data-label="Symbol"><strong>{x['symbol']}</strong></td>
                <td class="company" data-label="Company">{x['company']}</td>
                <td class="num" data-label="CMP">₹{x['price']:,.2f}</td>
                <td class="num" data-label="20-Day High">₹{x['high20']:,.2f}</td>
                <td class="num" data-label="20D Away %">{x['away']:.2f}%</td>
                <td class="num" data-label="6% Target YES"><span class="chip positive">{x['yes']}</span></td>
                <td class="num" data-label="6% Target NO"><span class="chip negative">{x['no']}</span></td>
                <td class="num" data-label="Strike Rate">
                    <span class="chip {'positive' if x['strike'] >= 50 else 'neutral'}">
                        {x['strike']:.2f}%
                    </span>
                </td>
            </tr>
            """
            for x in sst
        )

    def rows_blsh():
        return "".join(
            f"""
            <tr>
                <td data-label="Symbol"><strong>{x['symbol']}</strong></td>
                <td class="company" data-label="Company">{x['company']}</td>
                <td class="num" data-label="CMP">₹{x['cmp']:,.2f}</td>
                <td class="num" data-label="25D Low">₹{x['low_25_day']:,.2f}</td>
                <td class="num" data-label="Buy Trigger → Sell Price">
                    ₹{x['trigger_price']:,.2f} → ₹{x['target_price']:,.2f}
                </td>
                <td class="num" data-label="Away %">{x['trigger_away']:.2f}%</td>
                <td class="num" data-label="RSI(14)">{x['rsi']:.2f}</td>
                <td class="num" data-label="Sell Result (1Y)">
                    <span class="chip-group">
                        <span class="chip positive">{x['yes']}</span>
                        <span class="chip negative">{x['no']}</span>
                    </span>
                </td>
                <td class="num" data-label="Strike Rate">
                    <span class="chip {'positive' if x['strike'] >= 50 else 'neutral'}">
                        {x['strike']:.2f}%
                    </span>
                </td>
            </tr>
            """
            for x in blsh
        )

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Nifty 100 Daily Multi-Screener</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap" rel="stylesheet">
<style>
:root {{
    --bg: #eef1f6;
    --surface: #ffffff;
    --surface-alt: #f6f8fb;
    --border: #e2e7f0;
    --ink: #10192b;
    --ink-soft: #5b6579;
    --ink-faint: #8993a6;
    --gold: #a9781f;
    --gold-soft: #f7eed9;
    --green: #147347;
    --green-soft: #e2f3e9;
    --red: #ae2f27;
    --red-soft: #fbe9e7;
    --indigo: #29418f;
    --indigo-soft: #e6ebfa;
    --plum: #6a3382;
    --plum-soft: #f0e6f6;
    --teal: #0e6e63;
    --teal-soft: #e0f2ef;
    --radius-lg: 16px;
    --radius-sm: 10px;
    --shadow: 0 1px 2px rgba(16,25,43,0.04), 0 10px 24px rgba(16,25,43,0.06);
}}
* {{ box-sizing: border-box; }}
body {{
    margin: 0;
    font-family: 'Inter', Arial, sans-serif;
    background: var(--bg);
    color: var(--ink);
    padding: 28px;
    -webkit-font-smoothing: antialiased;
}}
.top {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 26px 34px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: var(--shadow);
    margin-bottom: 22px;
    flex-wrap: wrap;
    gap: 18px;
}}
.title {{ font-family: 'Space Grotesk', 'Inter', sans-serif; font-size: 27px; font-weight: 700; letter-spacing: -0.01em; }}
.subtitle {{ margin-top: 6px; color: var(--ink-soft); font-size: 14px; }}
.badges {{ display: flex; gap: 10px; flex-wrap: wrap; }}
.badge {{
    padding: 10px 18px;
    border-radius: 999px;
    font-weight: 600;
    font-size: 14px;
    background: var(--surface-alt);
    color: var(--ink-soft);
    border: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 8px;
    white-space: nowrap;
}}
.badge .dot {{ width: 8px; height: 8px; border-radius: 50%; background: var(--ink-faint); flex-shrink: 0; }}
.badge strong {{ color: var(--ink); }}
.badge.mwd .dot {{ background: var(--indigo); }}
.badge.sst .dot {{ background: var(--plum); }}
.badge.blsh .dot {{ background: var(--teal); }}
.tabs {{ display: inline-flex; gap: 4px; margin-bottom: 18px; padding: 5px; background: var(--surface-alt); border: 1px solid var(--border); border-radius: var(--radius-sm); }}
.tab {{ border: none; padding: 11px 24px; border-radius: 7px; background: transparent; color: var(--ink-soft); font-family: 'Inter', sans-serif; font-size: 14px; font-weight: 600; cursor: pointer; transition: background 0.15s ease, color 0.15s ease; }}
.tab:hover {{ color: var(--ink); }}
.tab.active.mwd {{ background: var(--indigo); color: #fff; }}
.tab.active.sst {{ background: var(--plum); color: #fff; }}
.tab.active.blsh {{ background: var(--teal); color: #fff; }}
.tab-content {{ display: none; background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-lg); overflow: hidden; box-shadow: var(--shadow); }}
.tab-content.active {{ display: block; }}
.panel-head {{ padding: 20px 26px; color: #ffffff; font-family: 'Space Grotesk', sans-serif; font-size: 20px; font-weight: 700; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; }}
.mwd-head {{ background: var(--indigo); }}
.sst-head {{ background: var(--plum); }}
.blsh-head {{ background: var(--teal); }}
.panel-sub {{ font-family: 'Inter', sans-serif; font-size: 13px; font-weight: 500; opacity: 0.85; }}
.filters {{ padding: 14px 26px; background: var(--surface-alt); border-bottom: 1px solid var(--border); display: flex; gap: 20px; align-items: center; flex-wrap: wrap; }}
select, input {{ padding: 8px 10px; border: 1px solid var(--border); border-radius: 7px; font-family: 'Inter', sans-serif; font-size: 13px; color: var(--ink); background: var(--surface); }}
select:focus, input:focus {{ outline: 2px solid var(--indigo); outline-offset: 1px; }}
label {{ font-size: 12.5px; font-weight: 600; color: var(--ink-soft); display: flex; align-items: center; gap: 6px; }}
.table-wrap {{ width: 100%; overflow-x: auto; }}
table {{ width: 100%; border-collapse: collapse; min-width: 760px; }}
th {{ text-align: left; color: var(--ink-faint); background: var(--surface-alt); padding: 12px 16px; font-size: 11.5px; font-weight: 700; letter-spacing: 0.02em; white-space: nowrap; border-bottom: 1px solid var(--border); }}
td {{ padding: 14px 16px; border-bottom: 1px solid var(--border); font-size: 14px; white-space: nowrap; }}
td.num {{ font-family: 'IBM Plex Mono', monospace; font-variant-numeric: tabular-nums; font-size: 13.5px; }}
td.company {{ color: var(--ink-soft); }}
tbody tr:hover {{ background: var(--surface-alt); }}
tr:last-child td {{ border-bottom: none; }}
.chip {{ display: inline-flex; align-items: center; padding: 4px 10px; border-radius: 6px; font-weight: 600; font-family: 'IBM Plex Mono', monospace; font-size: 13px; }}
.chip-group {{ display: inline-flex; align-items: center; gap: 6px; }}
.chip.positive {{ background: var(--green-soft); color: var(--green); }}
.chip.negative {{ background: var(--red-soft); color: var(--red); }}
.chip.neutral {{ background: var(--gold-soft); color: var(--gold); }}
.footer {{ margin-top: 22px; text-align: center; color: var(--ink-faint); font-size: 12.5px; }}
@media(max-width:860px) {{
    body {{ padding: 14px; }}
    .top {{ padding: 18px; display: block; }}
    .badges {{ margin-top: 15px; }}
    .title {{ font-size: 21px; }}
    .tabs {{ display: flex; width: 100%; }}
    .tab {{ flex: 1; text-align: center; padding: 10px 10px; font-size: 13px; }}
    .panel-head {{ padding: 16px 18px; font-size: 18px; }}
    .filters {{ padding: 12px 16px; }}
    .table-wrap {{ overflow-x: visible; padding: 12px; }}
    table {{ min-width: 0; width: 100%; }}
    thead {{ display: none; }}
    tbody, tr, td {{ display: block; width: 100%; }}
    tbody tr {{ background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-sm); margin-bottom: 12px; box-shadow: var(--shadow); overflow: hidden; }}
    tbody tr:last-child {{ margin-bottom: 0; }}
    td {{ padding: 10px 16px; border-bottom: 1px solid var(--border); white-space: normal; font-size: 13.5px; }}
    tr td:last-child {{ border-bottom: none; }}
    td[data-label="Symbol"] {{ font-family: 'Space Grotesk', sans-serif; font-size: 16px; font-weight: 700; padding: 14px 16px 2px; border-bottom: none; }}
    td.company {{ padding: 0 16px 12px; font-size: 13px; border-bottom: 1px solid var(--border); }}
    td.num, td:not([data-label="Symbol"]):not(.company) {{ display: flex; justify-content: space-between; align-items: center; gap: 12px; }}
    td.num::before, td:not([data-label="Symbol"]):not(.company)::before {{ content: attr(data-label); font-size: 11.5px; font-weight: 700; color: var(--ink-faint); letter-spacing: 0.02em; }}
}}
</style>
</head>
<body>

<div class="top">
    <div>
        <div class="title">📊 Nifty 100 Daily Multi-Screener</div>
        <div class="subtitle">Live Data Sync | {updated}</div>
    </div>
    <div class="badges">
        <div class="badge"><span class="dot"></span>Total Stocks: <strong>{total_stocks}</strong></div>
        <div class="badge mwd"><span class="dot"></span>MWD: <strong id="mwdCount">{len(mwd)}</strong></div>
        <div class="badge sst"><span class="dot"></span>SST: <strong id="sstCount">{len(sst)}</strong></div>
        <div class="badge blsh"><span class="dot"></span>BLSH: <strong id="blshCount">{len(blsh)}</strong></div>
    </div>
</div>

<div class="tabs">
    <button class="tab active mwd" onclick="showTab('mwd', this)">1. MWD</button>
    <button class="tab sst" onclick="showTab('sst', this)">2. SST</button>
    <button class="tab blsh" onclick="showTab('blsh', this)">3. BLSH RSI</button>
</div>

<div id="mwd" class="tab-content active">
    <div class="panel-head mwd-head">
        <span>MWD</span>
        <span class="panel-sub">Monthly + Weekly + Daily</span>
    </div>
    <div class="filters">
        <label>52W Distance ≥ <input id="dist" type="number" value="-10" step="0.1"></label>
        <label>Monthly Rise ≥ <input id="rise" type="number" value="2" step="0.1"></label>
    </div>
    <div class="table-wrap">
        <table id="mwdTable">
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th>Company</th>
                    <th>CMP</th>
                    <th>52W High</th>
                    <th>52W Distance</th>
                    <th>Monthly Rise</th>
                </tr>
            </thead>
            <tbody>
                {rows_mwd() or '<tr><td colspan="6">No matching stocks</td></tr>'}
            </tbody>
        </table>
    </div>
</div>

<div id="sst" class="tab-content">
    <div class="panel-head sst-head">
        <span>SST</span>
        <span class="panel-sub">Every New 20-Day High | Target +6%</span>
    </div>
    <div class="filters">
        <label>Strike Rate:
            <select id="sstStrike" onchange="filterSST()">
                <option value="0">All</option>
                <option value="25">&gt; 25%</option>
                <option value="50">&gt; 50%</option>
                <option value="60">&gt; 60%</option>
                <option value="70">&gt; 70%</option>
                <option value="80">&gt; 80%</option>
            </select>
        </label>
    </div>
    <div class="table-wrap">
        <table id="sstTable">
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th>Company</th>
                    <th>CMP</th>
                    <th>20-Day High</th>
                    <th>20D Away %</th>
                    <th>6% Target YES</th>
                    <th>6% Target NO</th>
                    <th>Strike Rate</th>
                </tr>
            </thead>
            <tbody>
                {rows_sst() or '<tr><td colspan="8">No matching stocks</td></tr>'}
            </tbody>
        </table>
    </div>
</div>

<div id="blsh" class="tab-content">
    <div class="panel-head blsh-head">
        <span>BLSH RSI</span>
        <span class="panel-sub">RSI &lt; 36 | New 25-Day Low | Buy Trigger +6.5% | Sell Price +3.14%</span>
    </div>
    <div class="filters">
        <label>Strike Rate:
            <select id="blshStrike" onchange="filterBLSH()">
                <option value="0">All</option>
                <option value="25">&gt; 25%</option>
                <option value="50">&gt; 50%</option>
                <option value="60">&gt; 60%</option>
                <option value="70">&gt; 70%</option>
                <option value="80">&gt; 80%</option>
            </select>
        </label>
    </div>
    <div class="table-wrap">
        <table id="blshTable">
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th>Company</th>
                    <th>CMP</th>
                    <th>25D Low</th>
                    <th>Buy Trigger → Sell Price</th>
                    <th>Away %</th>
                    <th>RSI(14)</th>
                    <th>Sell Result (1Y)</th>
                    <th>Strike Rate</th>
                </tr>
            </thead>
            <tbody>
                {rows_blsh() or '<tr><td colspan="9">No current matching stocks</td></tr>'}
            </tbody>
        </table>
    </div>
</div>

<div class="footer">
    Updated in IST • Yahoo Finance data • Educational use only, not investment advice
</div>

<script>
function showTab(tabName, button) {{
    document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
    document.getElementById(tabName).classList.add('active');
    button.classList.add('active');
}}

function filterMWD() {{
    const distance = parseFloat(document.getElementById('dist').value);
    const rise = parseFloat(document.getElementById('rise').value);

    document.querySelectorAll('#mwdTable tbody tr').forEach(row => {{
        const d = parseFloat(row.dataset.distance);
        const r = parseFloat(row.dataset.rise);
        if (!isNaN(d) && !isNaN(r)) {{
            row.style.display = (d >= distance && r >= rise) ? '' : 'none';
        }}
    }});
    updateBadgeCount('mwdTable', 'mwdCount');
}}

function filterSST() {{
    const minimum = parseFloat(document.getElementById('sstStrike').value);
    document.querySelectorAll('#sstTable tbody tr').forEach(row => {{
        const cells = row.querySelectorAll('td');
        if (cells.length < 8) return;
        const strike = parseFloat(cells[7].innerText.replace('%', ''));
        row.style.display = strike >= minimum ? '' : 'none';
    }});
    updateBadgeCount('sstTable', 'sstCount');
}}

function filterBLSH() {{
    const minimum = parseFloat(document.getElementById('blshStrike').value);
    document.querySelectorAll('#blshTable tbody tr').forEach(row => {{
        const cells = row.querySelectorAll('td');
        if (cells.length < 9) return;
        const strike = parseFloat(cells[8].innerText.replace('%', ''));
        row.style.display = strike >= minimum ? '' : 'none';
    }});
    updateBadgeCount('blshTable', 'blshCount');
}}

function updateBadgeCount(tableId, badgeId) {{
    const rows = document.querySelectorAll('#' + tableId + ' tbody tr');
    let visible = 0;
    rows.forEach(row => {{
        if (row.style.display !== 'none' && row.querySelectorAll('td').length > 1) {{
            visible += 1;
        }}
    }});
    const badge = document.getElementById(badgeId);
    if (badge) badge.textContent = visible;
}}

document.getElementById('dist').addEventListener('input', filterMWD);
document.getElementById('rise').addEventListener('input', filterMWD);

filterMWD();
filterSST();
filterBLSH();
</script>

</body>
</html>
"""


# ============================================================
# MAIN
# ============================================================

def main():
    symbols = load_symbols()

    mwd_results = []
    sst_results = []
    blsh_results = []

    print(f"\nProcessing {len(symbols)} Nifty 100 stocks...\n")

    for n, (symbol, company) in enumerate(symbols, 1):
        print(f"[{n}/{len(symbols)}] {symbol}")
        try:
            df = download(symbol)
            if df is None or len(df) < 300:
                print("  Insufficient data")
                continue

            m = mwd_screen(symbol, company, df)
            if m:
                mwd_results.append(m)

            s = sst_screen(symbol, company, df)
            if s:
                sst_results.append(s)

            b = blsh_screen(symbol, company, df)
            if b:
                blsh_results.append(b)

        except Exception as e:
            print("ERROR:", symbol, e)

        time.sleep(0.25)

    # Sort Results
    mwd_results.sort(key=lambda x: x["distance"], reverse=True)
    sst_results.sort(key=lambda x: x["away"])
    blsh_results.sort(key=lambda x: (x["trigger_away"], -x["strike"]))

    updated = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%d-%b-%Y %I:%M %p IST")

    os.makedirs("docs", exist_ok=True)
    html = build_html(mwd_results, sst_results, blsh_results, updated, len(symbols))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(html)

    print("\n==========================================")
    print("NIFTY 100 MULTI-SCREENER COMPLETED")
    print("==========================================")
    print("Total Stocks:", len(symbols))
    print("MWD Matches:", len(mwd_results))
    print("SST Matches:", len(sst_results))
    print("BLSH Matches:", len(blsh_results))
    print("Updated:", updated)
    print("Created:", OUTPUT_FILE)
    print("==========================================\n")


# ============================================================
# SST DEBUG HELPER
# ============================================================

def debug_sst_transactions(symbol):
    print(f"\nDownloading {symbol} ...")
    df = download(symbol)
    if df is None:
        print(f"No data found for {symbol}")
        return

    transactions = get_sst_transactions(df, SST_LOOKBACK, SST_TARGET)

    print(f"\nSST Transaction Log — {symbol}")
    print(f"(lookback={SST_LOOKBACK} days, target=+{SST_TARGET}%)\n")

    header = (
        f"{'Date':<12}"
        f"{'Buy Trigger':>12}"
        f"{'Sell Price':>13}   "
        f"{'Sell Hit?':<10}"
        f"{'Sell Hit Date':<18}"
        f"{'Days Taken'}"
    )
    print(header)
    print("-" * len(header))

    for t in transactions:
        date_str = fmt_date(t["date"])
        met_str = fmt_date(t["sell_hit_date"]) if t["sell_hit_date"] is not None else ""
        days_str = str(t["days_taken"]) if t["days_taken"] is not None else ""

        print(
            f"{date_str:<12}"
            f"{t['buy_price']:>12,.2f}"
            f"{t['sell_target']:>13,.2f}   "
            f"{t['achieved']:<10}"
            f"{met_str:<18}"
            f"{days_str}"
        )

    yes = sum(1 for t in transactions if t["achieved"] == "Yes")
    no = sum(1 for t in transactions if t["achieved"] == "No")
    open_trades = sum(1 for t in transactions if t["achieved"] == "Open")
    completed = yes + no
    strike = (yes / completed * 100) if completed else 0

    print("-" * len(header))
    print(f"YES: {yes}   NO: {no}   Open (unresolved): {open_trades}   Strike Rate: {strike:.2f}%\n")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--sst-debug":
        debug_sst_transactions(sys.argv[2].strip().upper())
    else:
        main()
