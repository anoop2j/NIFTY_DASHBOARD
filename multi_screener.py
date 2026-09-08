import pandas as pd
import yfinance as yf
import time
import os
from datetime import datetime
from zoneinfo import ZoneInfo

SYMBOL_FILE = "nifty100_symbols.csv"
OUTPUT_FILE = "docs/index.html"

SST_LOOKBACK = 20
SST_TARGET = 6.0
BLSH_LOOKBACK = 25
BLSH_TARGET = 3.14
ANALYSIS_DAYS = 252


def fmt_date(value):
    try:
        return pd.Timestamp(value).strftime("%d-%b-%Y")
    except Exception:
        return "N/A"


def bullish(row):
    return pd.notna(row["Open"]) and pd.notna(row["Close"]) and row["Close"] > row["Open"]


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = gain / loss.replace(0, float("nan"))
    out = 100 - (100 / (1 + rs))
    return out.fillna(100)


def load_symbols():
    df = pd.read_csv(SYMBOL_FILE)
    df.columns = df.columns.str.strip().str.lower()
    if "symbol" not in df.columns:
        raise ValueError("CSV must contain a symbol column")
    if "company" not in df.columns:
        df["company"] = df["symbol"]
    df = df.dropna(subset=["symbol"]).copy()
    return [(str(r["symbol"]).strip(), str(r["company"]).strip()) for _, r in df.iterrows()]


def download(symbol):
    df = yf.download(symbol + ".NS", period="18mo", interval="1d",
                     auto_adjust=False, progress=False, threads=False)
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    required = ["Open", "High", "Low", "Close"]
    if any(c not in df.columns for c in required):
        return None
    df = df.dropna(subset=required).copy()
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=required)
    df.index = pd.to_datetime(df.index)
    return df


def resample_ohlc(df, rule):
    return pd.DataFrame({
        "Open": df["Open"].resample(rule).first(),
        "High": df["High"].resample(rule).max(),
        "Low": df["Low"].resample(rule).min(),
        "Close": df["Close"].resample(rule).last()
    }).dropna()


def mwd_screen(symbol, company, df):
    daily = df.copy()
    weekly = resample_ohlc(df, "W-FRI")
    monthly = resample_ohlc(df, "ME")
    if len(daily) < 252 or len(weekly) < 3 or len(monthly) < 3:
        return None

    d = daily.iloc[-1]
    w = weekly.iloc[-2]       # last completed week
    m = monthly.iloc[-2]      # last completed month

    if not (bullish(d) and bullish(w) and bullish(m)):
        return None

    last_price = float(d["Close"])
    high_52 = float(daily["High"].tail(252).max())
    if last_price >= high_52:
        return None

    distance = ((last_price - high_52) / high_52) * 100
    monthly_rise = ((float(m["Close"]) - float(m["Open"])) / float(m["Open"])) * 100

    return {
        "symbol": symbol, "company": company, "date": fmt_date(daily.index[-1]),
        "price": round(last_price, 2), "high52": round(high_52, 2),
        "distance": round(distance, 2), "rise": round(monthly_rise, 2)
    }


def is_new_low(df, i, lookback):
    if i < lookback:
        return False
    return float(df["Low"].iloc[i]) < float(df["Low"].iloc[i-lookback:i].min())


def is_new_high(df, i, lookback):
    if i < lookback:
        return False
    return float(df["High"].iloc[i]) > float(df["High"].iloc[i-lookback:i].max())


def cycle_statistics(df, lookback, target_pct):
    data = df.tail(ANALYSIS_DAYS + lookback).copy().reset_index().rename(columns={"index": "Date"})
    if len(data) <= lookback:
        return 0, 0, 0.0

    yes = no = 0
    i = lookback
    waiting_for_low = True

    while i < len(data):
        if waiting_for_low:
            if is_new_low(data, i, lookback):
                waiting_for_low = False
            i += 1
            continue

        if is_new_high(data, i, lookback):
            entry = float(data["High"].iloc[i-lookback:i].max())
            target = entry * (1 + target_pct / 100)
            j = i + 1
            outcome = None

            while j < len(data):
                if float(data["High"].iloc[j]) >= target:
                    outcome = "YES"
                    break
                if is_new_low(data, j, lookback):
                    outcome = "NO"
                    break
                j += 1

            if outcome == "YES":
                yes += 1
                waiting_for_low = True
                i = j + 1
            elif outcome == "NO":
                no += 1
                waiting_for_low = False
                i = j + 1
            else:
                i += 1
        else:
            i += 1

    completed = yes + no
    strike = (yes / completed * 100) if completed else 0
    return yes, no, round(strike, 2)


def sst_screen(symbol, df):
    if len(df) < SST_LOOKBACK + 1:
        return None
    close = float(df["Close"].iloc[-1])
    high20 = float(df["High"].tail(SST_LOOKBACK).max())
    away = ((high20 - close) / high20 * 100) if high20 else 0
    yes, no, strike = cycle_statistics(df, SST_LOOKBACK, SST_TARGET)
    return {"symbol": symbol, "price": round(close,2), "away": round(away,2),
            "yes": yes, "no": no, "strike": strike}


def blsh_screen(symbol, df):
    if len(df) < BLSH_LOOKBACK + 15:
        return None

    work = df.copy()
    work["RSI14"] = rsi(work["Close"], 14)
    i = len(work) - 1
    current = work.iloc[-1]
    rsi14 = float(current["RSI14"])
    previous25_low = float(work["Low"].iloc[i-BLSH_LOOKBACK:i].min())
    previous25_high = float(work["High"].iloc[i-BLSH_LOOKBACK:i].max())
    current_low = float(current["Low"])
    cmp = float(current["Close"])

    # Candidate condition: RSI(14) below 36 and a fresh 25-day low.
    if not (rsi14 < 36 and current_low < previous25_low):
        return None

    trigger_away = ((cmp - previous25_high) / previous25_high) * 100
    yes, no, strike = cycle_statistics(df, BLSH_LOOKBACK, BLSH_TARGET)

    return {
        "symbol": symbol, "cmp": round(cmp,2), "rsi": round(rsi14,2),
        "trigger_away": round(trigger_away,2),
        "yes": yes, "no": no, "strike": strike,
        "last_low_date": fmt_date(work.index[-1])
    }


def build_html(mwd, sst, blsh, updated, total_stocks):
    def rows_mwd():
        return "".join(f"""<tr data-distance="{x['distance']}" data-rise="{x['rise']}">
<td>{x['symbol']}</td><td>₹{x['price']:,.2f}</td><td>{x['distance']:.2f}%</td><td><span class="positive">+{x['rise']:.2f}%</span></td></tr>""" for x in mwd)

    def rows_sst():
        return "".join(f"""<tr><td>{x['symbol']}</td><td>{x['price']:,.2f}</td><td>{x['away']:.2f}%</td><td>{x['strike']:.0f}%</td></tr>""" for x in sst[:15])

    def rows_blsh():
        return "".join(f"""<tr><td>{x['symbol']}</td><td>{x['cmp']:,.2f}</td><td>{x['trigger_away']:.2f}%</td><td>{x['strike']:.0f}%</td></tr>""" for x in blsh[:15])

    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Nifty 100 Daily Multi-Screener</title>
<style>
*{{box-sizing:border-box}} body{{margin:0;font-family:Arial,sans-serif;background:#eef2f7;color:#26384d;padding:20px}}
.top{{background:#fff;border-radius:18px;padding:20px 36px;display:flex;justify-content:space-between;align-items:center;box-shadow:0 8px 25px #ccd5e055;margin-bottom:30px}}
.title{{font-size:30px;font-weight:800}} .subtitle{{margin-top:10px;color:#5d6d7e}} .badges{{display:flex;gap:10px}} .badge{{padding:12px 22px;border-radius:25px;font-weight:bold;background:#dcebf5}} .badge.green{{background:#dcefe2;color:#29683c}}
.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:28px}} .panel{{background:#fff;border-radius:18px;overflow:hidden;box-shadow:0 8px 25px #ccd5e055}}
.panel-head{{padding:24px 20px;color:#fff;font-size:23px;font-weight:800;display:flex;justify-content:space-between}} .mwd{{background:#2858b5}} .sst{{background:#6b31c7}} .blsh{{background:#147b65}}
.panel-sub{{font-size:17px;align-self:center}} table{{width:100%;border-collapse:collapse}} th{{text-align:left;color:#607080;background:#f6f7f9;padding:14px 12px;font-size:16px}} td{{padding:16px 12px;border-bottom:1px solid #e5e8ec;font-size:17px}} tr:last-child td{{border-bottom:none}} .positive{{background:#e0f0e7;color:#1c7043;padding:7px 10px;border-radius:8px;font-weight:bold}} .negative{{color:#d2261e}} .filters{{padding:14px;background:#f8fafc;border-bottom:1px solid #e5e8ec;display:flex;gap:10px;align-items:center}} input{{width:90px;padding:7px;border:1px solid #ccd4dd;border-radius:6px}} label{{font-size:13px;font-weight:bold}} .footer{{margin-top:25px;text-align:center;color:#64748b;font-size:13px}}
@media(max-width:1100px){{.grid{{grid-template-columns:1fr}}}} @media(max-width:650px){{body{{padding:10px}}.top{{padding:18px;display:block}}.badges{{margin-top:15px;flex-wrap:wrap}}.title{{font-size:24px}}}}
</style></head><body>
<div class="top"><div><div class="title">📊 Nifty 100 Daily Multi-Screener</div><div class="subtitle">Live Data Sync | {updated}</div></div>
<div class="badges"><div class="badge">Total Stocks: {total_stocks}</div><div class="badge green">MWD Matches: {len(mwd)}</div><div class="badge green">BLSH Matches: {len(blsh)}</div></div></div>
<div class="grid">
<div class="panel"><div class="panel-head mwd"><span>1. MWD</span><span class="panel-sub">Monthly+Weekly+Daily</span></div>
<div class="filters"><label>52W Dist ≥ <input id="dist" type="number" value="-10" step="0.1"></label><label>Rise ≥ <input id="rise" type="number" value="2" step="0.1"></label></div>
<table id="mwdTable"><thead><tr><th>Symbol</th><th>Price</th><th>52W Dist</th><th>Rise</th></tr></thead><tbody>{rows_mwd() or '<tr><td colspan="4">No matching stocks</td></tr>'}</tbody></table></div>
<div class="panel"><div class="panel-head sst"><span>2. SST</span><span class="panel-sub">20-Day High Focus</span></div>
<table><thead><tr><th>Stock</th><th>Yesterday</th><th>20D Away %</th><th>Strike Rate</th></tr></thead><tbody>{rows_sst()}</tbody></table></div>
<div class="panel"><div class="panel-head blsh"><span>3. BLSH RSI</span><span class="panel-sub">RSI &lt; 36 | New 25-Day Low</span></div>
<table><thead><tr><th>Stock</th><th>CMP</th><th>Trigger Away</th><th>Strike Rate</th></tr></thead><tbody>{rows_blsh() or '<tr><td colspan="4">No current matching stocks</td></tr>'}</tbody></table></div>
</div>
<div class="footer">Updated in IST • Yahoo Finance data • Educational use only, not investment advice</div>
<script>
function filterMWD(){{const d=parseFloat(document.getElementById('dist').value)||-999;const r=parseFloat(document.getElementById('rise').value)||-999;document.querySelectorAll('#mwdTable tbody tr').forEach(x=>{{const a=parseFloat(x.dataset.distance),b=parseFloat(x.dataset.rise);if(!isNaN(a))x.style.display=(a>=d&&b>=r)?'':'none'}})}}
document.getElementById('dist').addEventListener('input',filterMWD);document.getElementById('rise').addEventListener('input',filterMWD);filterMWD();
</script></body></html>"""


def main():
    symbols = load_symbols()
    mwd_results, sst_results, blsh_results = [], [], []

    for n, (symbol, company) in enumerate(symbols, 1):
        print(f"[{n}/{len(symbols)}] {symbol}")
        try:
            df = download(symbol)
            if df is None or len(df) < 300:
                continue
            m = mwd_screen(symbol, company, df)
            if m: mwd_results.append(m)
            s = sst_screen(symbol, df)
            if s: sst_results.append(s)
            b = blsh_screen(symbol, df)
            if b: blsh_results.append(b)
        except Exception as e:
            print("ERROR:", symbol, e)
        time.sleep(0.25)

    mwd_results.sort(key=lambda x: x["distance"], reverse=True)
    sst_results.sort(key=lambda x: x["away"])
    blsh_results.sort(key=lambda x: (x["trigger_away"], -x["strike"]))

    updated = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%d-%b-%Y %I:%M %p IST")
    os.makedirs("docs", exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(build_html(mwd_results, sst_results, blsh_results, updated, len(symbols)))

    print("MWD:", len(mwd_results), "SST:", len(sst_results), "BLSH:", len(blsh_results))
    print("Created", OUTPUT_FILE)


if __name__ == "__main__":
    main()
