import pandas as pd
import yfinance as yf
import requests
import time
import os

from datetime import datetime
from zoneinfo import ZoneInfo


# ============================================================
# FILE / OUTPUT SETTINGS
# https://www.niftyindices.com/IndexConstituent/ind_nifty100list.csv
# ============================================================

SYMBOL_FILE = "nifty100_symbols.csv"
OUTPUT_FILE = "docs/index.html"


# ============================================================
# ETF 28 SMA SETTINGS
# ============================================================

ETF_SYMBOL_FILE = "etf_symbols.csv"

NSE_ETF_PAGE = "https://www.nseindia.com/market-data/exchange-traded-funds-etf"
NSE_ETF_API = "https://www.nseindia.com/api/etf"

NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": NSE_ETF_PAGE,
}

# SMA length used for the ETF entry/exit screen
ETF_SMA_PERIOD = 28

# Window used to compute "average daily traded value" / "average
# daily volume" for the liquidity filter below
ETF_ADV_LOOKBACK = 20

# Only ETFs whose average daily traded value exceeds this many
# crores (1 cr = 1,00,00,000) are shown
ETF_MIN_ADV_CR = 1.0


# ============================================================
# SST SETTINGS
# ============================================================

SST_LOOKBACK = 20
SST_TARGET = 6.0


# ============================================================
# BLSH RSI SETTINGS
# ============================================================

BLSH_LOOKBACK = 25
BLSH_TRIGGER_PCT = 6.5
BLSH_TARGET_PCT = 3.14

RSI_PERIOD = 14
RSI_LIMIT = 36


# ============================================================
# HISTORICAL ANALYSIS
# ============================================================

ANALYSIS_DAYS = 252


# ============================================================
# CAR SETTINGS
#
# CAR = "Cumulative Average (Return) from 52-Week High".
#
# For every stock:
#   1. Find the 52-week high CLOSING price and the date it was
#      made (the highest Close in the trailing 252 trading days).
#   2. N (days_since_high) = number of trading days that have
#      elapsed SINCE that 52-week-high day (the high day itself
#      is not counted).
#   3. Build the running/cumulative average of the daily CLOSING
#      prices for every one of those N days, i.e.
#          avg_series[k] = mean(Close of the k days right after
#                               the 52-week high)
#      for k = 1 .. N. The CAR VALUE shown is avg_series[N] --
#      the cumulative average of ALL N days since the 52-week
#      high.
#   4. A stock only qualifies if that cumulative-average series
#      has been RISING over the last CAR_TREND_DAYS trading days
#      (today's cumulative average is higher than it was
#      CAR_TREND_DAYS days ago) -- i.e. the average price since
#      the top is climbing back up instead of drifting lower.
#
# CAR_MIN_DAYS is also the minimum number of trading days that
# must have passed since the 52-week high before a stock is
# evaluated at all (not enough data otherwise).
# ============================================================

CAR_TREND_DAYS = 10
CAR_MIN_DAYS = 10


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

    rs = gain / loss.replace(
        0,
        float("nan")
    )

    result = 100 - (
        100 / (1 + rs)
    )

    return result.fillna(100)


# ============================================================
# LOAD NIFTY 100 SYMBOLS
# ============================================================

def load_symbols():

    df = pd.read_csv(
        SYMBOL_FILE
    )

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
    )

    if "symbol" not in df.columns:

        raise ValueError(
            "CSV must contain a symbol column"
        )

    if "company" not in df.columns:

        df["company"] = df["symbol"]

    df = df.dropna(
        subset=["symbol"]
    ).copy()

    return [
        (
            str(row["symbol"]).strip(),
            str(row["company"]).strip()
        )
        for _, row in df.iterrows()
    ]


# ============================================================
# LOAD NSE ETF LIST (LIVE)
#
# Pulls the current list of NSE-listed ETFs straight from NSE's
# own ETF page API. NSE requires a warmed-up, browser-like
# session (cookies picked up from the HTML pages) before the
# JSON API will respond, and it will reject requests from most
# cloud / CI IP ranges (GitHub Actions included) with a 401/403
# even with the right headers. If that happens here, it falls
# back to the local ETF_SYMBOL_FILE list below instead of
# failing the whole run.
# ============================================================

def fetch_nse_etf_list():

    session = requests.Session()
    session.headers.update(NSE_HEADERS)

    try:

        # Visiting the homepage + the ETF page first is what
        # sets the cookies the API call needs.
        session.get(
            "https://www.nseindia.com",
            timeout=10
        )

        session.get(
            NSE_ETF_PAGE,
            timeout=10
        )

        response = session.get(
            NSE_ETF_API,
            timeout=10
        )

        response.raise_for_status()

        payload = response.json()

    except Exception as e:

        print(
            f"  NSE ETF list fetch failed: {e}"
        )

        return []


    rows = (
        payload.get("data", [])
        if isinstance(payload, dict)
        else []
    )

    results = []

    for row in rows:

        if not isinstance(row, dict):

            continue


        symbol = str(
            row.get("symbol")
            or row.get("SYMBOL")
            or ""
        ).strip()

        if not symbol:

            continue


        meta = row.get("meta")

        if isinstance(meta, dict) and meta.get("companyName"):

            name = meta.get("companyName")

        else:

            name = (
                row.get("assets")
                or row.get("companyName")
                or symbol
            )

        results.append(
            (
                symbol,
                str(name).strip()
            )
        )

    return results


# ============================================================
# EXCLUDE LIQUID / OVERNIGHT ETFs
#
# Liquid / overnight ETFs (symbol contains "LIQ" — LIQUIDBEES,
# LIQUIDCASE, HDFCLIQUID, SBILIQETF, etc.) trade essentially
# flat around a fixed NAV and are not relevant to a 28 SMA
# trend screen, so they're dropped regardless of which source
# (live NSE or the local fallback file) the symbol list came
# from.
# ============================================================

def exclude_liquid_etfs(symbols):

    return [
        (symbol, name)
        for symbol, name in symbols
        if "LIQ" not in symbol.upper()
    ]


# ============================================================
# LOAD ETF SYMBOLS (LIVE, WITH LOCAL FALLBACK)
# ============================================================

def load_etf_symbols():

    live = fetch_nse_etf_list()

    if live:

        live = exclude_liquid_etfs(
            live
        )

        print(
            f"  Loaded {len(live)} ETFs from NSE "
            "(liquid/overnight ETFs excluded)"
        )

        return live


    print(
        "  Falling back to local ETF symbol list "
        f"({ETF_SYMBOL_FILE})"
    )

    if not os.path.exists(ETF_SYMBOL_FILE):

        print(
            f"  {ETF_SYMBOL_FILE} not found — no ETFs to screen"
        )

        return []


    df = pd.read_csv(
        ETF_SYMBOL_FILE
    )

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
    )

    if "symbol" not in df.columns:

        return []

    if "name" not in df.columns:

        df["name"] = df["symbol"]

    df = df.dropna(
        subset=["symbol"]
    ).copy()

    fallback = [
        (
            str(row["symbol"]).strip(),
            str(row["name"]).strip()
        )
        for _, row in df.iterrows()
    ]

    return exclude_liquid_etfs(
        fallback
    )


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

        print(
            f"  Download error: {e}"
        )

        return None

    if df is None or df.empty:

        return None


    # --------------------------------------------------------
    # Handle Yahoo Finance MultiIndex
    # --------------------------------------------------------

    if isinstance(
        df.columns,
        pd.MultiIndex
    ):

        df.columns = (
            df.columns
            .get_level_values(0)
        )


    required = [
        "Open",
        "High",
        "Low",
        "Close"
    ]


    if any(
        column not in df.columns
        for column in required
    ):

        return None


    df = df.dropna(
        subset=required
    ).copy()


    for column in required:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


    df = df.dropna(
        subset=required
    )


    df.index = pd.to_datetime(
        df.index
    )


    return df


# ============================================================
# RESAMPLE OHLC
# ============================================================

def resample_ohlc(df, rule):

    result = pd.DataFrame({

        "Open":
            df["Open"]
            .resample(rule)
            .first(),

        "High":
            df["High"]
            .resample(rule)
            .max(),

        "Low":
            df["Low"]
            .resample(rule)
            .min(),

        "Close":
            df["Close"]
            .resample(rule)
            .last()

    })

    return result.dropna()


# ============================================================
# MWD SCREEN
# ============================================================

def mwd_screen(
    symbol,
    company,
    df
):

    daily = df.copy()

    weekly = resample_ohlc(
        df,
        "W-FRI"
    )

    monthly = resample_ohlc(
        df,
        "ME"
    )


    if (
        len(daily) < 252
        or len(weekly) < 3
        or len(monthly) < 3
    ):

        return None


    # --------------------------------------------------------
    # Current daily candle
    # --------------------------------------------------------

    daily_current = daily.iloc[-1]


    # --------------------------------------------------------
    # Current (still forming) week
    # --------------------------------------------------------

    weekly_this = weekly.iloc[-1]


    # --------------------------------------------------------
    # Last completed week
    # --------------------------------------------------------

    weekly_current = weekly.iloc[-2]


    # --------------------------------------------------------
    # Last completed month
    # --------------------------------------------------------

    monthly_current = monthly.iloc[-2]


    # --------------------------------------------------------
    # Daily, monthly, last completed week, AND current
    # (still forming) week must all be bullish.
    # --------------------------------------------------------

    if not (
        bullish(daily_current)
        and bullish(weekly_this)
        and bullish(weekly_current)
        and bullish(monthly_current)
    ):

        return None


    last_price = float(
        daily_current["Close"]
    )


    # --------------------------------------------------------
    # 52-week high
    # --------------------------------------------------------

    high_52 = float(
        daily["High"]
        .tail(252)
        .max()
    )


    # Do not show stocks already at / above 52W high
    if last_price >= high_52:

        return None


    # --------------------------------------------------------
    # Distance from 52W high
    # --------------------------------------------------------

    distance = (
        (
            last_price - high_52
        )
        / high_52
    ) * 100


    # --------------------------------------------------------
    # Monthly rise
    # --------------------------------------------------------

    monthly_open = float(
        monthly_current["Open"]
    )

    monthly_close = float(
        monthly_current["Close"]
    )


    if monthly_open == 0:

        return None


    monthly_rise = (
        (
            monthly_close - monthly_open
        )
        / monthly_open
    ) * 100


    return {

        "symbol": symbol,

        "company": company,

        "date":
            fmt_date(
                daily.index[-1]
            ),

        "price":
            round(
                last_price,
                2
            ),

        "high52":
            round(
                high_52,
                2
            ),

        "distance":
            round(
                distance,
                2
            ),

        "rise":
            round(
                monthly_rise,
                2
            )
    }


# ============================================================
# NEW 25-DAY LOW
#
# Today's Low must be below the LOW of the previous
# 25 trading days.
# ============================================================

def is_new_low(
    df,
    i,
    lookback
):

    if i < lookback:

        return False


    current_low = float(
        df["Low"].iloc[i]
    )


    previous_low = float(
        df["Low"]
        .iloc[
            i - lookback:i
        ]
        .min()
    )


    return current_low < previous_low


# ============================================================
# NEW 20-DAY LOW (SST ONLY)
#
# SST's "20 day low" is a rolling window that INCLUDES today
# (20 cells total, today being the most recent one) — verified
# directly against the INDHOTEL/GRASIM Excel backtests, which
# flag a new low whenever today's low is <= the min of the
# preceding lookback-1 days (non-strict). This differs from the
# generic is_new_low() above (used by BLSH), which compares
# against the preceding `lookback` days, excluding today, and
# is strict. Do NOT reuse that one for SST — the off-by-one
# there causes missed re-arm signals and dropped trades.
# ============================================================

def is_new_low_sst(
    df,
    i,
    lookback
):

    if i < lookback - 1:

        return False


    current_low = float(
        df["Low"].iloc[i]
    )


    previous_low = float(
        df["Low"]
        .iloc[
            i - (lookback - 1):i
        ]
        .min()
    )


    return current_low <= previous_low


# ============================================================
# NEW 20-DAY HIGH
#
# Today's High must be above the HIGH of the previous
# 20 trading days.
# ============================================================

def is_new_high(
    df,
    i,
    lookback
):

    if i < lookback:

        return False


    current_high = float(
        df["High"].iloc[i]
    )


    previous_high = float(
        df["High"]
        .iloc[
            i - lookback:i
        ]
        .max()
    )


    return current_high > previous_high


# ============================================================
# SST HISTORICAL STATISTICS
#
# CORRECTED LOGIC  (sequential, non-overlapping state machine)
#
# 1. SEARCH   : wait for a NEW 20-day-high breakout.
# 2. Entry    = previous 20-day high (the lookback-day high
#               BEFORE the breakout day).
# 3. Target   = entry + target_pct%.
# 4. IN_TRADE : starting the day AFTER the breakout, watch for
#                 - High >= target  -> YES
#                 - new 20-day low  -> NO
#               whichever happens first.
# 5. After a YES:
#       Do NOT search for a new breakout immediately.
#       WAIT_FOR_RESET until a new 20-day low actually occurs.
#       Only then does the state machine go back to SEARCH.
# 6. After a NO:
#       A new 20-day low has just occurred, so the reset is
#       already satisfied -> go straight back to SEARCH and a
#       fresh breakout can be evaluated immediately.
#
# IMPORTANT:
# Only ONE trade is tracked at a time. Trades never overlap,
# and a fresh breakout is never evaluated while a trade is
# still open or while waiting for the post-YES reset low.
# ============================================================

# ============================================================
# SST TRANSACTION LOG
#
# Reverse-engineered directly against real backtests
# (INDHOTEL, GRASIM) and confirmed to reproduce them exactly —
# every buy price, sell target, achieved Yes/No, target-met
# date, and days-taken matches row for row.
#
# Two mechanisms run independently:
#
# 1. A single ARMED / DISARMED switch controls when a NEW buy
#    can fire, regardless of whether an earlier buy has
#    resolved yet:
#      - ARMED turns ON the day a new 20-day LOW occurs (see
#        is_new_low_sst() — this window INCLUDES today, unlike
#        the generic is_new_low() used by BLSH), and stays ON
#        across further new lows.
#      - The FIRST new 20-day HIGH while ARMED fires a BUY.
#        Entry = previous 20 days' high (excludes today).
#        Target = entry * (1 + target_pct%).
#      - The moment a buy fires, ARMED turns OFF immediately.
#      - It stays OFF until the NEXT new 20-day low — this can
#        happen before OR after that buy's own outcome is known,
#        so trades CAN legitimately overlap.
#
# 2. Each individual buy is tracked forward on its own, from
#    the day after its buy date, with NO stop-loss / no new-low
#    failure condition (same as BLSH):
#      - High >= target on any later day -> YES, however long
#        that takes.
#      - If the target has not been hit by the end of the
#        available price history -> "No" (i.e. "not yet
#        achieved"). A later new 20-day low does NOT cancel an
#        open trade — it can still resolve Yes much later.
#
# "Days Taken" matches the reference backtest format: it is
# CALENDAR days between the buy date and the target-met date,
# not a trading-day count.
# ============================================================

def get_sst_transactions(
    df,
    lookback,
    target_pct
):

    data = (
        df
        .tail(
            ANALYSIS_DAYS + lookback
        )
        .copy()
        .reset_index()
        .rename(
            columns={
                "index": "Date"
            }
        )
    )

    if len(data) <= lookback:

        return []


    armed = False

    open_trades = []
    transactions = []

    for i in range(
        lookback,
        len(data)
    ):

        # ----------------------------------------------------
        # 1. ARM on a new 20-day low (SST's inclusive-of-today
        #    window — see is_new_low_sst()).
        # ----------------------------------------------------

        if is_new_low_sst(
            data,
            i,
            lookback
        ):

            armed = True


        # ----------------------------------------------------
        # 2. First new 20-day high while ARMED -> new buy.
        # ----------------------------------------------------

        if (
            armed
            and
            is_new_high(
                data,
                i,
                lookback
            )
        ):

            entry = float(
                data["High"]
                .iloc[
                    i - lookback:i
                ]
                .max()
            )

            target = (
                entry
                * (
                    1
                    + target_pct / 100
                )
            )

            open_trades.append({
                "date": data["Date"].iloc[i],
                "buy_index": i,
                "buy_price": entry,
                "sell_target": target,
            })

            armed = False


        # ----------------------------------------------------
        # 3. Advance every still-open trade by one day,
        #    starting the day AFTER its own buy date.
        #
        #    NOTE: unlike an earlier version of this function,
        #    a later new 20-day low does NOT cancel an open
        #    trade. Verified directly against the INDHOTEL and
        #    GRASIM backtests: trades ride through intervening
        #    lows and only resolve when the target is actually
        #    hit (same no-stop-loss convention as BLSH below).
        # ----------------------------------------------------

        current_high = float(
            data["High"].iloc[i]
        )

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
                    "target_met_date": met_date,
                    "days_taken": (
                        pd.Timestamp(met_date)
                        - pd.Timestamp(t["date"])
                    ).days,
                })

                continue


            still_open.append(t)


        open_trades = still_open


    # ----------------------------------------------------------
    # Any trade still open (target not yet hit) at the end of
    # available data is counted as "No" — i.e. "not yet
    # achieved" — same convention as get_blsh_transactions().
    # ----------------------------------------------------------

    for t in open_trades:

        transactions.append({
            "date": t["date"],
            "buy_price": t["buy_price"],
            "sell_target": t["sell_target"],
            "achieved": "No",
            "target_met_date": None,
            "days_taken": None,
        })


    transactions.sort(
        key=lambda t: t["date"]
    )


    return transactions


# ============================================================
# SST HISTORICAL STATISTICS
#
# See get_sst_transactions() above for the actual rule and
# state machine. This is just a thin wrapper that summarizes
# its output into yes/no/strike counts.
# ============================================================

def cycle_statistics(
    df,
    lookback,
    target_pct
):

    transactions = get_sst_transactions(
        df,
        lookback,
        target_pct
    )

    yes = sum(
        1
        for t in transactions
        if t["achieved"] == "Yes"
    )

    no = sum(
        1
        for t in transactions
        if t["achieved"] == "No"
    )

    completed = yes + no


    if completed:

        strike = (
            yes
            / completed
            * 100
        )

    else:

        strike = 0


    return (
        yes,
        no,
        round(
            strike,
            2
        )
    )


# ============================================================
# SST CURRENT SCREEN
# ============================================================

def sst_screen(
    symbol,
    company,
    df
):

    if len(df) < SST_LOOKBACK + 1:

        return None


    # --------------------------------------------------------
    # Current closing price
    # --------------------------------------------------------

    close = float(
        df["Close"].iloc[-1]
    )


    # --------------------------------------------------------
    # Current 20-day high
    #
    # Includes current trading day for display.
    # --------------------------------------------------------

    high20 = float(
        df["High"]
        .tail(SST_LOOKBACK)
        .max()
    )


    # --------------------------------------------------------
    # Distance from 20-day high
    # --------------------------------------------------------

    if high20:

        away = (
            (
                high20 - close
            )
            / high20
        ) * 100

    else:

        away = 0


    # --------------------------------------------------------
    # Historical SST performance
    # --------------------------------------------------------

    yes, no, strike = (
        cycle_statistics(
            df,
            SST_LOOKBACK,
            SST_TARGET
        )
    )


    return {

        "symbol":
            symbol,

        "company":
            company,

        "price":
            round(
                close,
                2
            ),

        "high20":
            round(
                high20,
                2
            ),

        "away":
            round(
                away,
                2
            ),

        "yes":
            yes,

        "no":
            no,

        "strike":
            strike
    }


# ============================================================
# BLSH TRANSACTION LOG
#
# Reverse-engineered directly against a real backtest (LODHA)
# and confirmed to reproduce it exactly (5 Yes, 1 No).
#
# Two mechanisms, same family as the SST fix:
#
# 1. A single ARMED/DISARMED switch, independent of any open
#    trade:
#      - A new 25-day LOW updates the reference low (and
#        therefore the trigger price = reference * 1.065) and
#        arms tracking. This can happen repeatedly on
#        consecutive new-low days, each time resetting the
#        reference to the newer, lower value.
#      - IMPORTANT: the buy check is skipped entirely on a day
#        that is itself a new low. It is only evaluated on a
#        later day that is NOT a new low, against whatever
#        trigger price was most recently established.
#      - The first such day where High >= trigger price fires
#        a BUY. Target = trigger price * 1.0314.
#      - The moment a buy fires, tracking disarms immediately,
#        and stays off until the next new 25-day low — which
#        can occur before or after this buy's own target is
#        hit, so trades CAN legitimately overlap.
#
# 2. Each buy is tracked forward independently, with NO stop
#    loss / no new-low failure condition:
#      - High >= target on any later day -> YES, however long
#        that takes (confirmed against real data: one LODHA
#        trade took 7 months to resolve and still counted as
#        a Yes).
#      - If the target has not been hit by the end of the
#        available price history -> counted as "No" (i.e. "not
#        yet achieved"), not a separate "Open" bucket.
# ============================================================

def get_blsh_transactions(df):

    data = (
        df
        .tail(
            ANALYSIS_DAYS + BLSH_LOOKBACK
        )
        .copy()
        .reset_index()
        .rename(
            columns={
                "index": "Date"
            }
        )
    )

    if len(data) <= BLSH_LOOKBACK:

        return []


    armed = False
    reference_low = None

    open_trades = []
    transactions = []

    for i in range(
        BLSH_LOOKBACK,
        len(data)
    ):

        new_low_today = is_new_low(
            data,
            i,
            BLSH_LOOKBACK
        )


        # ----------------------------------------------------
        # 1a. A new 25-day low arms tracking and (re)sets the
        #     reference low / trigger price. The buy check is
        #     skipped on this same day.
        # ----------------------------------------------------

        if new_low_today:

            armed = True

            reference_low = float(
                data["Low"].iloc[i]
            )


        # ----------------------------------------------------
        # 1b. On a day that is NOT a new low, check the
        #     currently-armed trigger.
        # ----------------------------------------------------

        elif armed:

            trigger_price = (
                reference_low
                * (
                    1
                    + BLSH_TRIGGER_PCT / 100
                )
            )

            current_high = float(
                data["High"].iloc[i]
            )

            if current_high >= trigger_price:

                target_price = (
                    trigger_price
                    * (
                        1
                        + BLSH_TARGET_PCT / 100
                    )
                )

                open_trades.append({
                    "date": data["Date"].iloc[i],
                    "buy_index": i,
                    "trigger_price": trigger_price,
                    "target_price": target_price,
                })

                armed = False


        # ----------------------------------------------------
        # 2. Advance every still-open trade — no stop loss,
        #    just watch for the target, starting the day
        #    after its own buy date.
        # ----------------------------------------------------

        current_high = float(
            data["High"].iloc[i]
        )

        still_open = []

        for t in open_trades:

            if i <= t["buy_index"]:

                still_open.append(t)

                continue


            if current_high >= t["target_price"]:

                transactions.append({
                    "date": t["date"],
                    "trigger_price": t["trigger_price"],
                    "target_price": t["target_price"],
                    "achieved": "Yes",
                    "target_met_date": data["Date"].iloc[i],
                })

            else:

                still_open.append(t)


        open_trades = still_open


    # ----------------------------------------------------------
    # Anything still open at the end of the available price
    # history has not yet reached its target -> "No".
    # ----------------------------------------------------------

    for t in open_trades:

        transactions.append({
            "date": t["date"],
            "trigger_price": t["trigger_price"],
            "target_price": t["target_price"],
            "achieved": "No",
            "target_met_date": None,
        })


    transactions.sort(
        key=lambda t: t["date"]
    )


    return transactions


# ============================================================
# BLSH HISTORICAL STATISTICS
#
# See get_blsh_transactions() above for the actual rule. This
# is just a thin wrapper summarizing it into yes/no/strike.
# ============================================================

def blsh_history_1_year(df):

    transactions = get_blsh_transactions(
        df
    )

    yes = sum(
        1
        for t in transactions
        if t["achieved"] == "Yes"
    )

    no = sum(
        1
        for t in transactions
        if t["achieved"] == "No"
    )

    completed = yes + no


    if completed:

        strike = (
            yes
            / completed
            * 100
        )

    else:

        strike = 0


    return (
        yes,
        no,
        round(
            strike,
            2
        )
    )


# ============================================================
# BLSH CURRENT SCREEN
# ============================================================

def blsh_screen(
    symbol,
    company,
    df
):

    if len(df) < (
        BLSH_LOOKBACK
        + RSI_PERIOD
    ):

        return None


    work = df.copy()


    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    work["RSI14"] = rsi(
        work["Close"],
        RSI_PERIOD
    )


    i = len(work) - 1


    current = work.iloc[-1]


    current_rsi = float(
        current["RSI14"]
    )

    current_low = float(
        current["Low"]
    )

    cmp = float(
        current["Close"]
    )


    # --------------------------------------------------------
    # PREVIOUS 25-DAY LOW
    #
    # Today's candle is excluded.
    # --------------------------------------------------------

    previous25_low = float(
        work["Low"]
        .iloc[
            i - BLSH_LOOKBACK:i
        ]
        .min()
    )


    # --------------------------------------------------------
    # NEW 25-DAY LOW
    # --------------------------------------------------------

    new_25_day_low = (
        current_low
        < previous25_low
    )


    # --------------------------------------------------------
    # BLSH QUALIFICATION
    #
    # RSI < 36
    # AND
    # New 25-day low
    # --------------------------------------------------------

    if not (
        current_rsi < RSI_LIMIT
        and new_25_day_low
    ):

        return None


    # --------------------------------------------------------
    # Current 25-day low
    # --------------------------------------------------------

    low_25_day = current_low


    # --------------------------------------------------------
    # Trigger = 6.5% above 25-day low
    # --------------------------------------------------------

    trigger_price = (
        low_25_day
        * (
            1
            + BLSH_TRIGGER_PCT / 100
        )
    )


    # --------------------------------------------------------
    # Target = 3.14% above trigger
    # --------------------------------------------------------

    target_price = (
        trigger_price
        * (
            1
            + BLSH_TARGET_PCT / 100
        )
    )


    # --------------------------------------------------------
    # Distance from CMP to trigger
    # --------------------------------------------------------

    if trigger_price:

        trigger_away = (
            (
                trigger_price - cmp
            )
            / trigger_price
        ) * 100

    else:

        trigger_away = 0


    # --------------------------------------------------------
    # Historical performance
    # --------------------------------------------------------

    yes, no, strike = (
        blsh_history_1_year(df)
    )


    return {

        "symbol":
            symbol,

        "company":
            company,

        "cmp":
            round(
                cmp,
                2
            ),

        "low_25_day":
            round(
                low_25_day,
                2
            ),

        "trigger_price":
            round(
                trigger_price,
                2
            ),

        "target_price":
            round(
                target_price,
                2
            ),

        "trigger_away":
            round(
                trigger_away,
                2
            ),

        "rsi":
            round(
                current_rsi,
                2
            ),

        "yes":
            yes,

        "no":
            no,

        "strike":
            strike,

        "last_low_date":
            fmt_date(
                work.index[-1]
            )
    }


# ============================================================
# ETF 28 SMA SCREEN
#
# Entry signal : 2 continuous daily closes ABOVE the 28 SMA
# Exit signal  : 2 continuous daily closes BELOW the 28 SMA
# Streak       : how many days in a row (up to today) the
#                close has stayed on the same side of the
#                28 SMA — shown Green while above, Red while
#                below.
# Liquidity    : ETF is dropped unless its average daily
#                traded value over the last ETF_ADV_LOOKBACK
#                days exceeds ETF_MIN_ADV_CR crores.
# ============================================================

def etf_screen(
    symbol,
    company,
    df
):

    daily = df.copy()

    if "Volume" not in daily.columns:

        return None


    daily["Volume"] = pd.to_numeric(
        daily["Volume"],
        errors="coerce"
    )

    daily = daily.dropna(
        subset=["Close", "Volume"]
    )

    if len(daily) < ETF_SMA_PERIOD + 3:

        return None


    # --------------------------------------------------------
    # 28-day SMA
    # --------------------------------------------------------

    daily["SMA28"] = (
        daily["Close"]
        .rolling(ETF_SMA_PERIOD)
        .mean()
    )

    valid = daily.dropna(
        subset=["SMA28"]
    )

    if len(valid) < 3:

        return None


    closes = valid["Close"].tolist()
    smas = valid["SMA28"].tolist()

    last_price = float(closes[-1])
    last_sma = float(smas[-1])


    # --------------------------------------------------------
    # Entry / Exit: 2 continuous closes above / below the SMA
    # --------------------------------------------------------

    above_today = closes[-1] > smas[-1]
    above_yday = closes[-2] > smas[-2]

    entry_signal = bool(
        above_today
        and above_yday
    )

    exit_signal = bool(
        (not above_today)
        and (not above_yday)
    )


    # --------------------------------------------------------
    # Streak: consecutive days (ending today) on the same
    # side of the 28 SMA
    # --------------------------------------------------------

    streak_side = "above" if above_today else "below"
    streak_days = 0

    for c, s in zip(
        reversed(closes),
        reversed(smas)
    ):

        side = "above" if c > s else "below"

        if side == streak_side:

            streak_days += 1

        else:

            break


    # --------------------------------------------------------
    # Average daily traded value / volume (liquidity filter)
    # --------------------------------------------------------

    trade_value = (
        daily["Close"]
        * daily["Volume"]
    )

    adv = (
        trade_value
        .tail(ETF_ADV_LOOKBACK)
        .mean()
    )

    avg_volume = (
        daily["Volume"]
        .tail(ETF_ADV_LOOKBACK)
        .mean()
    )

    if pd.isna(adv):

        return None


    adv_cr = float(adv) / 1e7

    if adv_cr <= ETF_MIN_ADV_CR:

        return None


    return {

        "symbol": symbol,

        "company": company,

        "date":
            fmt_date(
                daily.index[-1]
            ),

        "price":
            round(last_price, 2),

        "sma28":
            round(last_sma, 2),

        "adv_cr":
            round(adv_cr, 2),

        "avg_volume":
            int(round(float(avg_volume)))
            if not pd.isna(avg_volume)
            else 0,

        "entry": entry_signal,

        "exit": exit_signal,

        "streak_days": int(streak_days),

        "streak_side": streak_side,
    }


# ============================================================
# CAR SCREEN
#
# See the "CAR SETTINGS" block near the top of this file for
# the full definition of what is computed here.
# ============================================================

def car_screen(
    symbol,
    company,
    df
):

    daily = df.copy()

    daily = daily.dropna(
        subset=["Close"]
    )

    if len(daily) < 252 + CAR_MIN_DAYS:

        return None


    closes = daily["Close"]


    # --------------------------------------------------------
    # 52-week high CLOSE, and the date it happened
    # --------------------------------------------------------

    window = closes.tail(252)

    high_52_close = float(
        window.max()
    )

    high_52_date = window.idxmax()

    high_pos = daily.index.get_loc(
        high_52_date
    )


    # --------------------------------------------------------
    # Trading days elapsed since the 52-week high (the high
    # day itself is NOT counted)
    # --------------------------------------------------------

    days_since_high = (
        (len(daily) - 1)
        - high_pos
    )

    if days_since_high < CAR_MIN_DAYS:

        return None


    # --------------------------------------------------------
    # Cumulative (expanding) average of daily closes for every
    # day since the high:
    #     avg_series[k] = mean(close[high+1 : high+1+k])
    # --------------------------------------------------------

    post_high_closes = (
        closes
        .iloc[high_pos + 1:]
        .reset_index(drop=True)
    )

    cum_avg_series = (
        post_high_closes
        .expanding()
        .mean()
    )

    car_value = float(
        cum_avg_series.iloc[-1]
    )


    # --------------------------------------------------------
    # Trend check: the cumulative average must be HIGHER now
    # than it was CAR_TREND_DAYS trading days ago
    # --------------------------------------------------------

    lookback = min(
        CAR_TREND_DAYS,
        len(cum_avg_series) - 1
    )

    if lookback < 1:

        return None

    car_value_prior = float(
        cum_avg_series.iloc[-1 - lookback]
    )

    increasing = car_value > car_value_prior

    if not increasing:

        return None


    last_price = float(
        closes.iloc[-1]
    )

    car_vs_high = (
        (
            car_value - high_52_close
        )
        / high_52_close
    ) * 100

    car_change = (
        (
            car_value - car_value_prior
        )
        / car_value_prior
    ) * 100


    return {

        "symbol": symbol,

        "company": company,

        "high52_date":
            fmt_date(
                high_52_date
            ),

        "high52_close":
            round(high_52_close, 2),

        "days_since_high":
            int(days_since_high),

        "price":
            round(last_price, 2),

        "car_value":
            round(car_value, 2),

        "car_vs_high":
            round(car_vs_high, 2),

        "car_change":
            round(car_change, 2),
    }


# ============================================================
# HTML
# ============================================================

def build_html(
    mwd,
    sst,
    blsh,
    etf,
    car,
    updated,
    total_stocks
):


    # ========================================================
    # MWD ROWS
    # ========================================================

    def rows_mwd():

        return "".join(

            f"""
            <tr
                data-distance="{x['distance']}"
                data-rise="{x['rise']}"
            >

                <td data-label="Symbol"><strong>{x['symbol']}</strong></td>

                <td class="company" data-label="Company">{x['company']}</td>

                <td class="num" data-label="CMP">
                    ₹{x['price']:,.2f}
                </td>

                <td class="num" data-label="52W High">
                    ₹{x['high52']:,.2f}
                </td>

                <td class="num" data-label="52W Distance">
                    <span class="chip {'negative' if x['distance'] < 0 else 'positive'}">
                        {x['distance']:.2f}%
                    </span>
                </td>

                <td class="num" data-label="Monthly Rise">
                    <span class="chip positive">
                        +{x['rise']:.2f}%
                    </span>
                </td>

            </tr>
            """

            for x in mwd
        )


    # ========================================================
    # SST ROWS
    # ========================================================

    def rows_sst():

        return "".join(

            f"""
            <tr>

                <td data-label="Symbol"><strong>{x['symbol']}</strong></td>

                <td class="company" data-label="Company">{x['company']}</td>

                <td class="num" data-label="CMP">
                    ₹{x['price']:,.2f}
                </td>

                <td class="num" data-label="20-Day High">
                    ₹{x['high20']:,.2f}
                </td>

                <td class="num" data-label="20D Away %">
                    {x['away']:.2f}%
                </td>

                <td class="num" data-label="6% Target YES">
                    <span class="chip positive">
                        {x['yes']}
                    </span>
                </td>

                <td class="num" data-label="6% Target NO">
                    <span class="chip negative">
                        {x['no']}
                    </span>
                </td>

                <td class="num" data-label="Strike Rate">
                    <span class="chip {'positive' if x['strike'] >= 50 else 'neutral'}">
                        {x['strike']:.2f}%
                    </span>
                </td>

            </tr>
            """

            for x in sst
        )


    # ========================================================
    # BLSH ROWS
    # ========================================================

    def rows_blsh():

        return "".join(

            f"""
            <tr>

                <td data-label="Symbol"><strong>{x['symbol']}</strong></td>

                <td class="company" data-label="Company">{x['company']}</td>

                <td class="num" data-label="CMP">
                    ₹{x['cmp']:,.2f}
                </td>

                <td class="num" data-label="25D Low">
                    ₹{x['low_25_day']:,.2f}
                </td>

                <td class="num" data-label="Trigger → Target">
                    ₹{x['trigger_price']:,.2f} → ₹{x['target_price']:,.2f}
                </td>

                <td class="num" data-label="Away %">
                    {x['trigger_away']:.2f}%
                </td>

                <td class="num" data-label="RSI(14)">
                    {x['rsi']:.2f}
                </td>

                <td class="num" data-label="Win / Loss (1Y)">
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


    # ========================================================
    # ETF 28 SMA ROWS
    # ========================================================

    def rows_etf():

        return "".join(

            f"""
            <tr
                data-adv="{x['adv_cr']}"
                data-entry="{'1' if x['entry'] else '0'}"
                data-exit="{'1' if x['exit'] else '0'}"
            >

                <td data-label="Symbol"><strong>{x['symbol']}</strong></td>

                <td class="company" data-label="ETF Name">{x['company']}</td>

                <td class="num" data-label="CMP">
                    ₹{x['price']:,.2f}
                </td>

                <td class="num" data-label="28 SMA">
                    ₹{x['sma28']:,.2f}
                </td>

                <td class="num" data-label="Entry Signal">
                    <span class="chip {'positive' if x['entry'] else 'neutral'}">
                        {'ENTRY ▲' if x['entry'] else '—'}
                    </span>
                </td>

                <td class="num" data-label="Exit Signal">
                    <span class="chip {'negative' if x['exit'] else 'neutral'}">
                        {'EXIT ▼' if x['exit'] else '—'}
                    </span>
                </td>

                <td class="num" data-label="Streak (Days)">
                    <span class="chip {'positive' if x['streak_side'] == 'above' else 'negative'}">
                        {x['streak_days']}D {'▲ Above' if x['streak_side'] == 'above' else '▼ Below'}
                    </span>
                </td>

                <td class="num" data-label="Avg Daily Volume">
                    {x['avg_volume']:,}
                </td>

                <td class="num" data-label="Avg Daily Value">
                    ₹{x['adv_cr']:,.2f} Cr
                </td>

            </tr>
            """

            for x in etf
        )


    # ========================================================
    # CAR ROWS
    # ========================================================

    def rows_car():

        return "".join(

            f"""
            <tr
                data-days="{x['days_since_high']}"
                data-carvshigh="{x['car_vs_high']}"
            >

                <td data-label="Symbol"><strong>{x['symbol']}</strong></td>

                <td class="company" data-label="Company">{x['company']}</td>

                <td class="num" data-label="52W High Date">
                    {x['high52_date']}
                </td>

                <td class="num" data-label="52W High Close">
                    ₹{x['high52_close']:,.2f}
                </td>

                <td class="num" data-label="Days Since High">
                    {x['days_since_high']}
                </td>

                <td class="num" data-label="CMP">
                    ₹{x['price']:,.2f}
                </td>

                <td class="num" data-label="CAR Value">
                    ₹{x['car_value']:,.2f}
                </td>

                <td class="num" data-label="CAR vs 52W High">
                    <span class="chip {'negative' if x['car_vs_high'] < 0 else 'positive'}">
                        {x['car_vs_high']:.2f}%
                    </span>
                </td>

                <td class="num" data-label="CAR Trend">
                    <span class="chip positive">
                        +{x['car_change']:.2f}%
                    </span>
                </td>

            </tr>
            """

            for x in car
        )


    # ========================================================
    # HTML
    # ========================================================

    return f"""

<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
>

<title>
Nifty 100 Daily Multi-Screener
</title>

<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap" rel="stylesheet">

<style>

:root {{

    /* -- surfaces --------------------------------------- */
    --bg:            #eef1f6;
    --surface:       #ffffff;
    --surface-alt:   #f6f8fb;
    --border:        #e2e7f0;

    /* -- ink --------------------------------------------- */
    --ink:           #10192b;
    --ink-soft:      #5b6579;
    --ink-faint:     #8993a6;

    /* -- accent (gains / targets) ------------------------ */
    --gold:          #a9781f;
    --gold-soft:     #f7eed9;

    /* -- semantic ----------------------------------------- */
    --green:         #147347;
    --green-soft:    #e2f3e9;
    --red:           #ae2f27;
    --red-soft:      #fbe9e7;

    /* -- strategy colors ----------------------------------- */
    --indigo:        #29418f;
    --indigo-soft:   #e6ebfa;
    --plum:          #6a3382;
    --plum-soft:     #f0e6f6;
    --teal:          #0e6e63;
    --teal-soft:     #e0f2ef;
    --amber:         #b5541f;
    --amber-soft:    #fbe9dc;
    --rose:          #a13d63;
    --rose-soft:     #f7e3ec;

    --cyan:          #1b6e8c;
    --cyan-soft:     #dcf0f6;

    --radius-lg:     16px;
    --radius-sm:     10px;
    --shadow:        0 1px 2px rgba(16,25,43,0.04), 0 10px 24px rgba(16,25,43,0.06);
}}


* {{
    box-sizing: border-box;
}}


body {{

    margin: 0;

    font-family:
        'Inter',
        Arial,
        sans-serif;

    background:
        var(--bg);

    color:
        var(--ink);

    padding:
        28px;

    -webkit-font-smoothing:
        antialiased;
}}


/* ============================================================ */
/* TOP HEADER                                                   */
/* ============================================================ */

.top {{

    background:
        var(--surface);

    border:
        1px solid var(--border);

    border-radius:
        var(--radius-lg);

    padding:
        26px 34px;

    display:
        flex;

    justify-content:
        space-between;

    align-items:
        center;

    box-shadow:
        var(--shadow);

    margin-bottom:
        22px;

    flex-wrap:
        wrap;

    gap:
        18px;
}}


.title {{

    font-family:
        'Space Grotesk',
        'Inter',
        sans-serif;

    font-size:
        27px;

    font-weight:
        700;

    letter-spacing:
        -0.01em;
}}


.subtitle {{

    margin-top:
        6px;

    color:
        var(--ink-soft);

    font-size:
        14px;
}}


.badges {{

    display:
        flex;

    gap:
        10px;

    flex-wrap:
        wrap;
}}


.badge {{

    padding:
        10px 18px;

    border-radius:
        999px;

    font-weight:
        600;

    font-size:
        14px;

    background:
        var(--surface-alt);

    color:
        var(--ink-soft);

    border:
        1px solid var(--border);

    display:
        flex;

    align-items:
        center;

    gap:
        8px;

    white-space:
        nowrap;
}}


.badge .dot {{

    width:
        8px;

    height:
        8px;

    border-radius:
        50%;

    background:
        var(--ink-faint);

    flex-shrink:
        0;
}}


.badge strong {{
    color: var(--ink);
}}


.badge.mwd .dot {{ background: var(--indigo); }}
.badge.sst .dot {{ background: var(--plum); }}
.badge.blsh .dot {{ background: var(--teal); }}
.badge.etf .dot {{ background: var(--amber); }}
.badge.car .dot {{ background: var(--cyan); }}
.badge.upload .dot {{ background: var(--rose); }}


/* ============================================================ */
/* TABS                                                          */
/* ============================================================ */

.tabs {{

    display:
        inline-flex;

    gap:
        4px;

    margin-bottom:
        18px;

    padding:
        5px;

    background:
        var(--surface-alt);

    border:
        1px solid var(--border);

    border-radius:
        var(--radius-sm);
}}


.tab {{

    border:
        none;

    padding:
        11px 24px;

    border-radius:
        7px;

    background:
        transparent;

    color:
        var(--ink-soft);

    font-family:
        'Inter',
        sans-serif;

    font-size:
        14px;

    font-weight:
        600;

    cursor:
        pointer;

    transition:
        background 0.15s ease,
        color 0.15s ease;
}}


.tab:hover {{
    color: var(--ink);
}}


.tab.active.mwd {{ background: var(--indigo); color: #fff; }}
.tab.active.sst {{ background: var(--plum);   color: #fff; }}
.tab.active.blsh {{ background: var(--teal);   color: #fff; }}
.tab.active.etf {{ background: var(--amber);   color: #fff; }}
.tab.active.car {{ background: var(--cyan);    color: #fff; }}
.tab.active.upload {{ background: var(--rose);  color: #fff; }}


/* ============================================================ */
/* PANELS                                                         */
/* ============================================================ */

.tab-content {{

    display:
        none;

    background:
        var(--surface);

    border:
        1px solid var(--border);

    border-radius:
        var(--radius-lg);

    overflow:
        hidden;

    box-shadow:
        var(--shadow);
}}


.tab-content.active {{

    display:
        block;
}}


.panel-head {{

    padding:
        20px 26px;

    color:
        #ffffff;

    font-family:
        'Space Grotesk',
        sans-serif;

    font-size:
        20px;

    font-weight:
        700;

    display:
        flex;

    justify-content:
        space-between;

    align-items:
        center;

    flex-wrap:
        wrap;

    gap:
        8px;
}}


.mwd-head {{
    background: var(--indigo);
}}


.sst-head {{
    background: var(--plum);
}}


.blsh-head {{
    background: var(--teal);
}}


.etf-head {{
    background: var(--amber);
}}


.car-head {{
    background: var(--cyan);
}}


.upload-head {{
    background: var(--rose);
}}


/* ============================================================ */
/* UPLOAD SYMBOLS                                                */
/* ============================================================ */

.upload-note {{

    padding:
        16px 26px;

    background:
        var(--surface-alt);

    border-bottom:
        1px solid var(--border);

    font-size:
        13px;

    color:
        var(--ink-soft);

    line-height:
        1.55;
}}


.upload-grid {{

    display:
        grid;

    grid-template-columns:
        1fr 1fr;

    gap:
        20px;

    padding:
        22px 26px;
}}


.upload-card {{

    border:
        1px solid var(--border);

    border-radius:
        var(--radius-sm);

    padding:
        18px;

    background:
        var(--surface-alt);
}}


.upload-card h3 {{

    margin:
        0 0 4px;

    font-family:
        'IBM Plex Mono',
        monospace;

    font-size:
        14.5px;

    color:
        var(--ink);
}}


.upload-sub {{

    margin:
        0 0 14px;

    font-size:
        12.5px;

    color:
        var(--ink-faint);
}}


.dropzone {{

    border:
        2px dashed var(--border);

    border-radius:
        var(--radius-sm);

    padding:
        24px 16px;

    text-align:
        center;

    cursor:
        pointer;

    background:
        var(--surface);

    transition:
        border-color 0.15s ease,
        background 0.15s ease;
}}


.dropzone:hover,
.dropzone.dragover {{

    border-color:
        var(--rose);

    background:
        var(--rose-soft);
}}


.dropzone label {{

    display:
        block;

    font-size:
        13px;

    font-weight:
        600;

    color:
        var(--ink-soft);

    cursor:
        pointer;
}}


.upload-meta {{

    margin-top:
        12px;

    font-size:
        12.5px;

    color:
        var(--ink-soft);

    line-height:
        1.5;
}}


.upload-preview {{

    margin-top:
        10px;

    max-height:
        120px;

    overflow:
        auto;

    background:
        var(--surface);

    border:
        1px solid var(--border);

    border-radius:
        7px;

    padding:
        8px 10px;

    font-family:
        'IBM Plex Mono',
        monospace;

    font-size:
        11.5px;

    color:
        var(--ink-soft);

    white-space:
        pre-wrap;
}}


.btn-download {{

    margin-top:
        14px;

    padding:
        9px 16px;

    border-radius:
        7px;

    border:
        1px solid var(--border);

    background:
        var(--surface);

    cursor:
        pointer;

    font-family:
        'Inter',
        sans-serif;

    font-weight:
        600;

    font-size:
        13px;

    color:
        var(--ink);
}}


.btn-download:hover:not(:disabled) {{
    border-color: var(--rose);
    color: var(--rose);
}}


.btn-download:disabled {{

    opacity:
        0.5;

    cursor:
        not-allowed;
}}


@media(max-width:860px) {{

    .upload-grid {{

        grid-template-columns:
            1fr;
    }}

}}


.panel-sub {{

    font-family:
        'Inter',
        sans-serif;

    font-size:
        13px;

    font-weight:
        500;

    opacity:
        0.85;
}}


.filters {{

    padding:
        14px 26px;

    background:
        var(--surface-alt);

    border-bottom:
        1px solid var(--border);

    display:
        flex;

    gap:
        20px;

    align-items:
        center;

    flex-wrap:
        wrap;
}}


select,
input {{

    padding:
        8px 10px;

    border:
        1px solid var(--border);

    border-radius:
        7px;

    font-family:
        'Inter',
        sans-serif;

    font-size:
        13px;

    color:
        var(--ink);

    background:
        var(--surface);
}}


select:focus,
input:focus {{

    outline:
        2px solid var(--indigo);

    outline-offset:
        1px;
}}


label {{

    font-size:
        12.5px;

    font-weight:
        600;

    color:
        var(--ink-soft);

    display:
        flex;

    align-items:
        center;

    gap:
        6px;
}}


/* ============================================================ */
/* TABLE                                                          */
/* ============================================================ */

.table-wrap {{

    width:
        100%;

    overflow-x:
        auto;
}}


table {{

    width:
        100%;

    border-collapse:
        collapse;

    min-width:
        760px;
}}


th {{

    text-align:
        left;

    color:
        var(--ink-faint);

    background:
        var(--surface-alt);

    padding:
        12px 16px;

    font-size:
        11.5px;

    font-weight:
        700;

    letter-spacing:
        0.02em;

    white-space:
        nowrap;

    border-bottom:
        1px solid var(--border);
}}


td {{

    padding:
        14px 16px;

    border-bottom:
        1px solid var(--border);

    font-size:
        14px;

    white-space:
        nowrap;
}}


td.num {{

    font-family:
        'IBM Plex Mono',
        monospace;

    font-variant-numeric:
        tabular-nums;

    font-size:
        13.5px;
}}


td.company {{
    color: var(--ink-soft);
}}


tbody tr:hover {{
    background: var(--surface-alt);
}}


tr:last-child td {{

    border-bottom:
        none;
}}


/* ============================================================ */
/* CHIPS                                                          */
/* ============================================================ */

.chip {{

    display:
        inline-flex;

    align-items:
        center;

    padding:
        4px 10px;

    border-radius:
        6px;

    font-weight:
        600;

    font-family:
        'IBM Plex Mono',
        monospace;

    font-size:
        13px;
}}


.chip-group {{

    display:
        inline-flex;

    align-items:
        center;

    gap:
        6px;
}}


.chip.positive {{
    background: var(--green-soft);
    color: var(--green);
}}


.chip.negative {{
    background: var(--red-soft);
    color: var(--red);
}}


.chip.neutral {{
    background: var(--gold-soft);
    color: var(--gold);
}}


/* ============================================================ */
/* FOOTER                                                          */
/* ============================================================ */

.footer {{

    margin-top:
        22px;

    text-align:
        center;

    color:
        var(--ink-faint);

    font-size:
        12.5px;
}}


@media(max-width:860px) {{

    body {{
        padding:
            14px;
    }}

    .top {{

        padding:
            18px;

        display:
            block;
    }}

    .badges {{

        margin-top:
            15px;
    }}

    .title {{

        font-size:
            21px;
    }}

    .tabs {{

        display:
            flex;

        width:
            100%;
    }}

    .tab {{
        flex:
            1;

        text-align:
            center;

        padding:
            10px 10px;

        font-size:
            13px;
    }}

    .panel-head {{
        padding:
            16px 18px;

        font-size:
            18px;
    }}

    .filters {{
        padding:
            12px 16px;
    }}


    /* -------------------------------------------------- */
    /* TABLE -> STACKED CARDS                              */
    /*                                                      */
    /* Below this width the table never scrolls           */
    /* sideways. Each row becomes its own card; the        */
    /* Symbol is the card title, Company sits under it     */
    /* as a subtitle, and every other cell becomes a       */
    /* label / value line using its data-label attribute.  */
    /* -------------------------------------------------- */

    .table-wrap {{
        overflow-x:
            visible;

        padding:
            12px;
    }}

    table {{
        min-width:
            0;

        width:
            100%;
    }}

    thead {{
        display:
            none;
    }}

    tbody,
    tr,
    td {{
        display:
            block;

        width:
            100%;
    }}

    tbody tr {{
        background:
            var(--surface);

        border:
            1px solid var(--border);

        border-radius:
            var(--radius-sm);

        margin-bottom:
            12px;

        box-shadow:
            var(--shadow);

        overflow:
            hidden;
    }}

    tbody tr:last-child {{
        margin-bottom:
            0;
    }}

    td {{
        padding:
            10px 16px;

        border-bottom:
            1px solid var(--border);

        white-space:
            normal;

        font-size:
            13.5px;
    }}

    tr td:last-child {{
        border-bottom:
            none;
    }}

    td[data-label="Symbol"] {{
        font-family:
            'Space Grotesk',
            sans-serif;

        font-size:
            16px;

        font-weight:
            700;

        padding:
            14px 16px 2px;

        border-bottom:
            none;
    }}

    td.company {{
        padding:
            0 16px 12px;

        font-size:
            13px;

        border-bottom:
            1px solid var(--border);
    }}

    td.num,
    td:not([data-label="Symbol"]):not(.company) {{
        display:
            flex;

        justify-content:
            space-between;

        align-items:
            center;

        gap:
            12px;
    }}

    td.num::before,
    td:not([data-label="Symbol"]):not(.company)::before {{
        content:
            attr(data-label);

        font-size:
            11.5px;

        font-weight:
            700;

        color:
            var(--ink-faint);

        letter-spacing:
            0.02em;
    }}

}}

</style>

</head>


<body>


<!-- ===================================================== -->
<!-- TOP HEADER -->
<!-- ===================================================== -->

<div class="top">

    <div>

        <div class="title">
            📊 Nifty 100 Daily Multi-Screener
        </div>

        <div class="subtitle">
            Live Data Sync | {updated}
        </div>

    </div>


    <div class="badges">

        <div class="badge">
            <span class="dot"></span>
            Total Stocks: <strong>{total_stocks}</strong>
        </div>

        <div class="badge mwd">
            <span class="dot"></span>
            MWD: <strong id="mwdCount">{len(mwd)}</strong>
        </div>

        <div class="badge sst">
            <span class="dot"></span>
            SST: <strong id="sstCount">{len(sst)}</strong>
        </div>

        <div class="badge blsh">
            <span class="dot"></span>
            BLSH: <strong id="blshCount">{len(blsh)}</strong>
        </div>

        <div class="badge etf">
            <span class="dot"></span>
            ETF 28 SMA: <strong id="etfCount">{len(etf)}</strong>
        </div>

        <div class="badge car">
            <span class="dot"></span>
            CAR: <strong id="carCount">{len(car)}</strong>
        </div>

        <div class="badge upload">
            <span class="dot"></span>
            Symbol Files: <strong id="uploadStatus">—</strong>
        </div>

    </div>

</div>


<!-- ===================================================== -->
<!-- TABS -->
<!-- ===================================================== -->

<div class="tabs">

    <button
        class="tab active mwd"
        onclick="showTab('mwd', this)"
    >
        1. MWD
    </button>


    <button
        class="tab sst"
        onclick="showTab('sst', this)"
    >
        2. SST
    </button>


    <button
        class="tab blsh"
        onclick="showTab('blsh', this)"
    >
        3. BLSH RSI
    </button>


    <button
        class="tab etf"
        onclick="showTab('etf', this)"
    >
        4. ETF 28 SMA
    </button>


    <button
        class="tab car"
        onclick="showTab('car', this)"
    >
        5. CAR (52W High)
    </button>


    <button
        class="tab upload"
        onclick="showTab('upload', this)"
    >
        6. Upload Symbols
    </button>

</div>


<!-- ===================================================== -->
<!-- MWD -->
<!-- ===================================================== -->

<div
    id="mwd"
    class="tab-content active"
>

    <div class="panel-head mwd-head">

        <span>
            MWD
        </span>

        <span class="panel-sub">
            Monthly + Weekly + Daily
        </span>

    </div>


    <div class="filters">

        <label>

            52W Distance ≥

            <input
                id="dist"
                type="number"
                value="-10"
                step="0.1"
            >

        </label>


        <label>

            Monthly Rise ≥

            <input
                id="rise"
                type="number"
                value="2"
                step="0.1"
            >

        </label>

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

                {
                    rows_mwd()
                    or
                    '''<tr><td colspan="6">
                    No matching stocks
                    </td></tr>'''
                }

            </tbody>

        </table>

    </div>

</div>


<!-- ===================================================== -->
<!-- SST -->
<!-- ===================================================== -->

<div
    id="sst"
    class="tab-content"
>

    <div class="panel-head sst-head">

        <span>
            SST
        </span>

        <span class="panel-sub">
            Every New 20-Day High | Target +6%
        </span>

    </div>


    <div class="filters">

        <label>

            Strike Rate:

            <select
                id="sstStrike"
                onchange="filterSST()"
            >

                <option value="0">
                    All
                </option>

                <option value="25">
                    &gt; 25%
                </option>

                <option value="50">
                    &gt; 50%
                </option>

                <option value="60">
                    &gt; 60%
                </option>

                <option value="70">
                    &gt; 70%
                </option>

                <option value="80">
                    &gt; 80%
                </option>

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

                {
                    rows_sst()
                    or
                    '''<tr><td colspan="8">
                    No matching stocks
                    </td></tr>'''
                }

            </tbody>

        </table>

    </div>

</div>


<!-- ===================================================== -->
<!-- BLSH RSI -->
<!-- ===================================================== -->

<div
    id="blsh"
    class="tab-content"
>

    <div class="panel-head blsh-head">

        <span>
            BLSH RSI
        </span>

        <span class="panel-sub">
            RSI &lt; 36 | New 25-Day Low |
            Trigger +6.5% | Target +3.14%
        </span>

    </div>


    <div class="filters">

        <label>

            Strike Rate:

            <select
                id="blshStrike"
                onchange="filterBLSH()"
            >

                <option value="0">
                    All
                </option>

                <option value="25">
                    &gt; 25%
                </option>

                <option value="50">
                    &gt; 50%
                </option>

                <option value="60">
                    &gt; 60%
                </option>

                <option value="70">
                    &gt; 70%
                </option>

                <option value="80">
                    &gt; 80%
                </option>

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

                    <th>Trigger → Target</th>

                    <th>Away %</th>

                    <th>RSI(14)</th>

                    <th>Win / Loss (1Y)</th>

                    <th>Strike Rate</th>

                </tr>

            </thead>


            <tbody>

                {
                    rows_blsh()
                    or
                    '''<tr><td colspan="9">
                    No current matching stocks
                    </td></tr>'''
                }

            </tbody>

        </table>

    </div>

</div>


<!-- ===================================================== -->
<!-- ETF 28 SMA -->
<!-- ===================================================== -->

<div
    id="etf"
    class="tab-content"
>

    <div class="panel-head etf-head">

        <span>
            ETF 28 SMA
        </span>

        <span class="panel-sub">
            NSE ETFs | Avg Daily Value &gt; ₹1 Cr | 28-Day SMA Crossover
        </span>

    </div>


    <div class="filters">

        <label>

            Avg Daily Value (₹Cr) ≥

            <input
                id="etfAdv"
                type="number"
                value="1"
                step="0.1"
            >

        </label>


        <label>

            Signal:

            <select
                id="etfSignal"
                onchange="filterETF()"
            >

                <option value="all">
                    All
                </option>

                <option value="entry">
                    Entry Only
                </option>

                <option value="exit">
                    Exit Only
                </option>

            </select>

        </label>

    </div>


    <div class="table-wrap">

        <table id="etfTable">

            <thead>

                <tr>

                    <th>Symbol</th>

                    <th>ETF Name</th>

                    <th>CMP</th>

                    <th>28 SMA</th>

                    <th>Entry Signal</th>

                    <th>Exit Signal</th>

                    <th>Streak (Days)</th>

                    <th>Avg Daily Volume</th>

                    <th>Avg Daily Value</th>

                </tr>

            </thead>


            <tbody>

                {
                    rows_etf()
                    or
                    '''<tr><td colspan="9">
                    No matching ETFs
                    </td></tr>'''
                }

            </tbody>

        </table>

    </div>

</div>


<!-- ===================================================== -->
<!-- CAR (CUMULATIVE AVERAGE FROM 52-WEEK HIGH) -->
<!-- ===================================================== -->

<div
    id="car"
    class="tab-content"
>

    <div class="panel-head car-head">

        <span>
            CAR — 52W High
        </span>

        <span class="panel-sub">
            Cumulative Average of Closes Since the 52-Week High | Rising over last {CAR_TREND_DAYS} Days
        </span>

    </div>


    <div class="filters">

        <label>

            Days Since High ≥

            <input
                id="carDays"
                type="number"
                value="{CAR_MIN_DAYS}"
                step="1"
            >

        </label>


        <label>

            CAR vs 52W High (%) ≥

            <input
                id="carVsHigh"
                type="number"
                value="-20"
                step="0.1"
            >

        </label>

    </div>


    <div class="table-wrap">

        <table id="carTable">

            <thead>

                <tr>

                    <th>Symbol</th>

                    <th>Company</th>

                    <th>52W High Date</th>

                    <th>52W High Close</th>

                    <th>Days Since High</th>

                    <th>CMP</th>

                    <th>CAR Value</th>

                    <th>CAR vs 52W High</th>

                    <th>CAR Trend ({CAR_TREND_DAYS}D)</th>

                </tr>

            </thead>


            <tbody>

                {
                    rows_car()
                    or
                    '''<tr><td colspan="9">
                    No matching stocks
                    </td></tr>'''
                }

            </tbody>

        </table>

    </div>

</div>


<!-- ===================================================== -->
<!-- UPLOAD SYMBOLS -->
<!-- ===================================================== -->

<div
    id="upload"
    class="tab-content"
>

    <div class="panel-head upload-head">

        <span>
            Upload Symbol Lists
        </span>

        <span class="panel-sub">
            nifty100_symbols.csv &amp; etf_symbols.csv
        </span>

    </div>


    <div class="upload-note">
        This page is a static file, so a browser upload here can't
        write straight into the GitHub repo. Choosing a file below
        reads it in your browser, remembers the upload date/time
        and a preview on this device, and gives you a
        <strong>Download</strong> button that saves it back out
        under the exact filename the script expects. Save that
        download into the same folder as multi_screener.py
        (replacing the old file) and commit it — the next run will
        pick it up.
    </div>


    <div class="upload-grid">

        <div class="upload-card">

            <h3>nifty100_symbols.csv</h3>

            <p class="upload-sub">
                Stock universe for MWD / SST / BLSH RSI
            </p>

            <div class="dropzone" id="dropNifty">

                <input
                    type="file"
                    id="fileNifty"
                    accept=".csv"
                    hidden
                >

                <label for="fileNifty">
                    Choose file or drag &amp; drop here
                </label>

            </div>

            <div class="upload-preview" id="previewNifty"></div>

            <div class="upload-meta" id="metaNifty">
                No file uploaded yet on this browser
            </div>

            <button
                class="btn-download"
                id="downloadNifty"
                disabled
            >
                Download nifty100_symbols.csv
            </button>

        </div>


        <div class="upload-card">

            <h3>etf_symbols.csv</h3>

            <p class="upload-sub">
                ETF universe for the 28 SMA screen
            </p>

            <div class="dropzone" id="dropEtf">

                <input
                    type="file"
                    id="fileEtf"
                    accept=".csv"
                    hidden
                >

                <label for="fileEtf">
                    Choose file or drag &amp; drop here
                </label>

            </div>

            <div class="upload-preview" id="previewEtf"></div>

            <div class="upload-meta" id="metaEtf">
                No file uploaded yet on this browser
            </div>

            <button
                class="btn-download"
                id="downloadEtf"
                disabled
            >
                Download etf_symbols.csv
            </button>

        </div>

    </div>

</div>


<!-- ===================================================== -->
<!-- FOOTER -->
<!-- ===================================================== -->

<div class="footer">

    Updated in IST • Yahoo Finance data •
    Educational use only, not investment advice

</div>


<script>


// ==========================================================
// UPLOAD SYMBOLS
//
// This is a static page — a browser upload here cannot write
// straight into the GitHub repo. What it CAN do:
//   1. Read the chosen CSV in-browser (FileReader)
//   2. Remember the upload date/time + a preview in this
//      browser's localStorage, so "last uploaded" persists
//      across visits on this device
//   3. Offer a Download button that saves the content back
//      out under the exact filename the script expects, so
//      it can be committed to replace the old file
// ==========================================================

function readFileAsText(file) {{

    return new Promise(
        function(resolve, reject) {{

            const reader = new FileReader();

            reader.onload = function() {{
                resolve(reader.result);
            }};

            reader.onerror = function() {{
                reject(reader.error);
            }};

            reader.readAsText(file);

        }}
    );

}}


function formatUploadTime() {{

    return new Date().toLocaleString(
        'en-IN',
        {{
            day: '2-digit',
            month: 'short',
            year: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
            hour12: true
        }}
    );

}}


function countCsvRows(text) {{

    const lines = text
        .split(/\\r?\\n/)
        .filter(
            function(line) {{
                return line.trim().length > 0;
            }}
        );

    return Math.max(0, lines.length - 1);

}}


function storeUpload(
    storageKey,
    originalName,
    text
) {{

    const record = {{
        originalName: originalName,
        uploadedAt: formatUploadTime(),
        rowCount: countCsvRows(text),
        content: text
    }};

    try {{

        localStorage.setItem(
            storageKey,
            JSON.stringify(record)
        );

    }} catch (e) {{

        console.warn(
            'Could not save upload to this browser:',
            e
        );

    }}

    return record;

}}


function loadUpload(storageKey) {{

    try {{

        const raw = localStorage.getItem(storageKey);

        return raw ? JSON.parse(raw) : null;

    }} catch (e) {{

        return null;

    }}

}}


function renderUploadCard(
    prefix,
    targetFileName
) {{

    const record =
        loadUpload('upload_' + targetFileName);

    const meta =
        document.getElementById('meta' + prefix);

    const preview =
        document.getElementById('preview' + prefix);

    const downloadBtn =
        document.getElementById('download' + prefix);

    if (record) {{

        meta.textContent =
            'Last uploaded: ' + record.originalName +
            '  •  ' + record.uploadedAt +
            '  •  ' + record.rowCount + ' rows';

        preview.textContent =
            record.content
                .split(/\\r?\\n/)
                .slice(0, 6)
                .join('\\n');

        downloadBtn.disabled = false;

    }} else {{

        meta.textContent =
            'No file uploaded yet on this browser';

        preview.textContent = '';

        downloadBtn.disabled = true;

    }}

    updateHeaderUploadBadge();

}}


function handleUploadedFile(
    prefix,
    targetFileName,
    file
) {{

    if (!file) {{

        return;

    }}


    if (!/\\.csv$/i.test(file.name)) {{

        alert('Please choose a .csv file.');

        return;

    }}


    if (
        file.name.toLowerCase()
        !== targetFileName.toLowerCase()
    ) {{

        const proceed = confirm(
            'This file is named "' + file.name +
            '", not "' + targetFileName + '". ' +
            'It will still be read, but rename it to "' +
            targetFileName +
            '" before committing it to your repo. Continue?'
        );

        if (!proceed) {{

            return;

        }}

    }}


    readFileAsText(file)
        .then(
            function(text) {{

                storeUpload(
                    'upload_' + targetFileName,
                    file.name,
                    text
                );

                renderUploadCard(
                    prefix,
                    targetFileName
                );

            }}
        )
        .catch(
            function(e) {{
                alert('Could not read that file: ' + e);
            }}
        );

}}


function downloadStoredUpload(targetFileName) {{

    const record =
        loadUpload('upload_' + targetFileName);

    if (!record) {{

        return;

    }}


    const blob = new Blob(
        [record.content],
        {{ type: 'text/csv' }}
    );

    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');

    a.href = url;
    a.download = targetFileName;

    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);

    URL.revokeObjectURL(url);

}}


function setupUploadCard(
    prefix,
    targetFileName
) {{

    const input =
        document.getElementById('file' + prefix);

    const drop =
        document.getElementById('drop' + prefix);

    const downloadBtn =
        document.getElementById('download' + prefix);

    input.addEventListener(
        'change',
        function(e) {{
            handleUploadedFile(
                prefix,
                targetFileName,
                e.target.files[0]
            );
        }}
    );

    drop.addEventListener(
        'dragover',
        function(e) {{
            e.preventDefault();
            drop.classList.add('dragover');
        }}
    );

    drop.addEventListener(
        'dragleave',
        function() {{
            drop.classList.remove('dragover');
        }}
    );

    drop.addEventListener(
        'drop',
        function(e) {{

            e.preventDefault();

            drop.classList.remove('dragover');

            const file =
                e.dataTransfer.files
                && e.dataTransfer.files[0];

            handleUploadedFile(
                prefix,
                targetFileName,
                file
            );

        }}
    );

    downloadBtn.addEventListener(
        'click',
        function() {{
            downloadStoredUpload(targetFileName);
        }}
    );

    renderUploadCard(
        prefix,
        targetFileName
    );

}}


function updateHeaderUploadBadge() {{

    const badge =
        document.getElementById('uploadStatus');

    if (!badge) {{

        return;

    }}


    const nifty =
        loadUpload('upload_nifty100_symbols.csv');

    const etf =
        loadUpload('upload_etf_symbols.csv');

    if (!nifty && !etf) {{

        badge.textContent = 'Not uploaded yet';

        return;

    }}


    const parts = [];

    if (nifty) {{
        parts.push('Nifty100 ' + nifty.uploadedAt);
    }}

    if (etf) {{
        parts.push('ETF ' + etf.uploadedAt);
    }}

    badge.textContent = parts.join('  •  ');

}}


setupUploadCard('Nifty', 'nifty100_symbols.csv');
setupUploadCard('Etf', 'etf_symbols.csv');


// ==========================================================
// TAB SWITCHING
// ==========================================================

function showTab(
    tabName,
    button
) {{

    document
        .querySelectorAll(
            '.tab-content'
        )
        .forEach(
            function(tab) {{

                tab.classList.remove(
                    'active'
                );

            }}
        );


    document
        .querySelectorAll(
            '.tab'
        )
        .forEach(
            function(tab) {{

                tab.classList.remove(
                    'active'
                );

            }}
        );


    document
        .getElementById(tabName)
        .classList.add(
            'active'
        );


    button.classList.add(
        'active'
    );

}}


// ==========================================================
// MWD FILTER
// ==========================================================

function filterMWD() {{

    const distance =
        parseFloat(
            document
                .getElementById(
                    'dist'
                )
                .value
        );


    const rise =
        parseFloat(
            document
                .getElementById(
                    'rise'
                )
                .value
        );


    document
        .querySelectorAll(
            '#mwdTable tbody tr'
        )
        .forEach(
            function(row) {{

                const d =
                    parseFloat(
                        row.dataset.distance
                    );


                const r =
                    parseFloat(
                        row.dataset.rise
                    );


                if (
                    !isNaN(d)
                    &&
                    !isNaN(r)
                ) {{

                    row.style.display =
                        (
                            d >= distance
                            &&
                            r >= rise
                        )
                        ? ''
                        : 'none';

                }}

            }}
        );


    updateBadgeCount(
        'mwdTable',
        'mwdCount'
    );

}}


// ==========================================================
// SST FILTER
// ==========================================================

function filterSST() {{

    const minimum =
        parseFloat(
            document
                .getElementById(
                    'sstStrike'
                )
                .value
        );


    document
        .querySelectorAll(
            '#sstTable tbody tr'
        )
        .forEach(
            function(row) {{

                const cells =
                    row.querySelectorAll(
                        'td'
                    );


                if (
                    cells.length < 8
                ) {{

                    return;

                }}


                const strike =
                    parseFloat(
                        cells[7]
                            .innerText
                            .replace(
                                '%',
                                ''
                            )
                    );


                row.style.display =
                    strike >= minimum
                    ? ''
                    : 'none';

            }}
        );


    updateBadgeCount(
        'sstTable',
        'sstCount'
    );

}}


// ==========================================================
// BLSH FILTER
// ==========================================================

function filterBLSH() {{

    const minimum =
        parseFloat(
            document
                .getElementById(
                    'blshStrike'
                )
                .value
        );


    document
        .querySelectorAll(
            '#blshTable tbody tr'
        )
        .forEach(
            function(row) {{

                const cells =
                    row.querySelectorAll(
                        'td'
                    );


                if (
                    cells.length < 9
                ) {{

                    return;

                }}


                const strike =
                    parseFloat(
                        cells[8]
                            .innerText
                            .replace(
                                '%',
                                ''
                            )
                    );


                row.style.display =
                    strike >= minimum
                    ? ''
                    : 'none';

            }}
        );


    updateBadgeCount(
        'blshTable',
        'blshCount'
    );

}}


// ==========================================================
// ETF 28 SMA FILTER
// ==========================================================

function filterETF() {{

    const minAdv =
        parseFloat(
            document
                .getElementById(
                    'etfAdv'
                )
                .value
        );


    const signal =
        document
            .getElementById(
                'etfSignal'
            )
            .value;


    document
        .querySelectorAll(
            '#etfTable tbody tr'
        )
        .forEach(
            function(row) {{

                const adv =
                    parseFloat(
                        row.dataset.adv
                    );


                if (isNaN(adv)) {{

                    return;

                }}


                let show =
                    adv >= minAdv;


                if (signal === 'entry') {{

                    show =
                        show
                        && row.dataset.entry === '1';

                }}


                if (signal === 'exit') {{

                    show =
                        show
                        && row.dataset.exit === '1';

                }}


                row.style.display =
                    show
                    ? ''
                    : 'none';

            }}
        );


    updateBadgeCount(
        'etfTable',
        'etfCount'
    );

}}


// ==========================================================
// CAR FILTER
// ==========================================================

function filterCAR() {{

    const minDays =
        parseFloat(
            document
                .getElementById(
                    'carDays'
                )
                .value
        );


    const minVsHigh =
        parseFloat(
            document
                .getElementById(
                    'carVsHigh'
                )
                .value
        );


    document
        .querySelectorAll(
            '#carTable tbody tr'
        )
        .forEach(
            function(row) {{

                const d =
                    parseFloat(
                        row.dataset.days
                    );


                const v =
                    parseFloat(
                        row.dataset.carvshigh
                    );


                if (
                    !isNaN(d)
                    &&
                    !isNaN(v)
                ) {{

                    row.style.display =
                        (
                            d >= minDays
                            &&
                            v >= minVsHigh
                        )
                        ? ''
                        : 'none';

                }}

            }}
        );


    updateBadgeCount(
        'carTable',
        'carCount'
    );

}}


// ==========================================================
// BADGE COUNT HELPER
//
// Keeps the header badge in sync with the number of rows
// actually visible in a table after filtering, instead of
// the raw (unfiltered) screen match count.
// ==========================================================

function updateBadgeCount(
    tableId,
    badgeId
) {{

    const rows =
        document.querySelectorAll(
            '#' + tableId + ' tbody tr'
        );


    let visible = 0;


    rows.forEach(
        function(row) {{

            if (
                row.style.display !== 'none'
                &&
                row.querySelectorAll('td').length > 1
            ) {{

                visible += 1;

            }}

        }}
    );


    const badge =
        document.getElementById(
            badgeId
        );


    if (badge) {{

        badge.textContent = visible;

    }}

}}


// ==========================================================
// EVENT LISTENERS
// ==========================================================

document
    .getElementById(
        'dist'
    )
    .addEventListener(
        'input',
        filterMWD
    );


document
    .getElementById(
        'rise'
    )
    .addEventListener(
        'input',
        filterMWD
    );


document
    .getElementById(
        'etfAdv'
    )
    .addEventListener(
        'input',
        filterETF
    );


document
    .getElementById(
        'carDays'
    )
    .addEventListener(
        'input',
        filterCAR
    );


document
    .getElementById(
        'carVsHigh'
    )
    .addEventListener(
        'input',
        filterCAR
    );


// ==========================================================
// INITIAL FILTER
// ==========================================================

filterMWD();
filterSST();
filterBLSH();
filterETF();
filterCAR();

</script>


</body>

</html>

"""


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load symbols
    # --------------------------------------------------------

    symbols = load_symbols()


    mwd_results = []
    sst_results = []
    blsh_results = []
    car_results = []


    print()
    print(
        f"Processing {len(symbols)} Nifty 100 stocks..."
    )
    print()


    # --------------------------------------------------------
    # Process every stock
    # --------------------------------------------------------

    for n, (
        symbol,
        company
    ) in enumerate(
        symbols,
        1
    ):


        print(
            f"[{n}/{len(symbols)}] {symbol}"
        )


        try:


            # ------------------------------------------------
            # Download data
            # ------------------------------------------------

            df = download(
                symbol
            )


            if (
                df is None
                or len(df) < 300
            ):

                print(
                    "  Insufficient data"
                )

                continue


            # =================================================
            # MWD
            # =================================================

            m = mwd_screen(
                symbol,
                company,
                df
            )


            if m:

                mwd_results.append(
                    m
                )


            # =================================================
            # SST
            # =================================================

            s = sst_screen(
                symbol,
                company,
                df
            )


            if s:

                sst_results.append(
                    s
                )


            # =================================================
            # BLSH RSI
            # =================================================

            b = blsh_screen(
                symbol,
                company,
                df
            )


            if b:

                blsh_results.append(
                    b
                )


            # =================================================
            # CAR (52W High Cumulative Average)
            # =================================================

            c = car_screen(
                symbol,
                company,
                df
            )


            if c:

                car_results.append(
                    c
                )


        except Exception as e:

            print(
                "ERROR:",
                symbol,
                e
            )


        # ----------------------------------------------------
        # Small delay to reduce request pressure
        # ----------------------------------------------------

        time.sleep(
            0.25
        )


    # ========================================================
    # LOAD & PROCESS NSE ETFs
    # ========================================================

    etf_symbols = load_etf_symbols()

    etf_results = []


    print()
    print(
        f"Processing {len(etf_symbols)} NSE ETFs..."
    )
    print()


    for n, (
        symbol,
        company
    ) in enumerate(
        etf_symbols,
        1
    ):


        print(
            f"[{n}/{len(etf_symbols)}] {symbol}"
        )


        try:


            df = download(
                symbol
            )


            if (
                df is None
                or len(df) < ETF_SMA_PERIOD + 5
            ):

                print(
                    "  Insufficient data"
                )

                continue


            e = etf_screen(
                symbol,
                company,
                df
            )


            if e:

                etf_results.append(
                    e
                )


        except Exception as ex:

            print(
                "ERROR:",
                symbol,
                ex
            )


        time.sleep(
            0.25
        )


    # ========================================================
    # SORT RESULTS
    # ========================================================


    # --------------------------------------------------------
    # MWD
    #
    # Nearest to 52-week high first
    # --------------------------------------------------------

    mwd_results.sort(
        key=lambda x:
            x["distance"],
        reverse=True
    )


    # --------------------------------------------------------
    # SST
    #
    # Nearest to 20-day high first
    # --------------------------------------------------------

    sst_results.sort(
        key=lambda x:
            x["away"]
    )


    # --------------------------------------------------------
    # BLSH
    #
    # Nearest to trigger first
    # Then highest strike rate
    # --------------------------------------------------------

    blsh_results.sort(
        key=lambda x: (
            x["trigger_away"],
            -x["strike"]
        )
    )


    # --------------------------------------------------------
    # ETF 28 SMA
    #
    # Most liquid (highest avg daily traded value) first
    # --------------------------------------------------------

    etf_results.sort(
        key=lambda x:
            x["adv_cr"],
        reverse=True
    )


    # --------------------------------------------------------
    # CAR
    #
    # Closest to (or above) the 52-week high first
    # --------------------------------------------------------

    car_results.sort(
        key=lambda x:
            x["car_vs_high"],
        reverse=True
    )


    # ========================================================
    # IST UPDATE TIME
    # ========================================================

    updated = (
        datetime
        .now(
            ZoneInfo(
                "Asia/Kolkata"
            )
        )
        .strftime(
            "%d-%b-%Y %I:%M %p IST"
        )
    )


    # ========================================================
    # CREATE DOCS DIRECTORY
    # ========================================================

    os.makedirs(
        "docs",
        exist_ok=True
    )


    # ========================================================
    # GENERATE HTML
    # ========================================================

    html = build_html(
        mwd_results,
        sst_results,
        blsh_results,
        etf_results,
        car_results,
        updated,
        len(symbols)
    )


    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            html
        )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()

    print(
        "=========================================="
    )

    print(
        "NIFTY 100 MULTI-SCREENER COMPLETED"
    )

    print(
        "=========================================="
    )

    print(
        "Total Stocks:",
        len(symbols)
    )

    print(
        "MWD Matches:",
        len(mwd_results)
    )

    print(
        "SST Matches:",
        len(sst_results)
    )

    print(
        "BLSH Matches:",
        len(blsh_results)
    )

    print(
        "ETF Universe:",
        len(etf_symbols)
    )

    print(
        "ETF 28 SMA Matches:",
        len(etf_results)
    )

    print(
        "CAR Matches:",
        len(car_results)
    )

    print(
        "Updated:",
        updated
    )

    print(
        "Created:",
        OUTPUT_FILE
    )

    print(
        "=========================================="
    )


# ============================================================
# SST DEBUG: PRINT FULL TRANSACTION LOG FOR ONE SYMBOL
#
# Prints every SST transaction in the same layout as a manual
# Excel/Google-Finance backtest ("Summary of all transactions"),
# so you can compare row by row and see exactly where the
# python logic and a spreadsheet disagree.
#
# Usage:
#     python3 multi_screener.py --sst-debug MAXHEALTH
# ============================================================

def debug_sst_transactions(symbol):

    print(
        f"\nDownloading {symbol} ..."
    )

    df = download(symbol)

    if df is None:

        print(
            f"No data found for {symbol}"
        )

        return

    transactions = get_sst_transactions(
        df,
        SST_LOOKBACK,
        SST_TARGET
    )

    print(
        f"\nSST Transaction Log — {symbol}"
    )

    print(
        f"(lookback={SST_LOOKBACK} days, "
        f"target=+{SST_TARGET}%)\n"
    )

    header = (
        f"{'Date':<12}"
        f"{'Buy Price':>12}"
        f"{'Sell Target':>13}   "
        f"{'Achieved?':<10}"
        f"{'Target Met Date':<18}"
        f"{'Days Taken'}"
    )

    print(header)
    print("-" * len(header))

    for t in transactions:

        date_str = fmt_date(t["date"])

        met_str = (
            fmt_date(t["target_met_date"])
            if t["target_met_date"] is not None
            else ""
        )

        days_str = (
            str(t["days_taken"])
            if t["days_taken"] is not None
            else ""
        )

        print(
            f"{date_str:<12}"
            f"{t['buy_price']:>12,.2f}"
            f"{t['sell_target']:>13,.2f}   "
            f"{t['achieved']:<10}"
            f"{met_str:<18}"
            f"{days_str}"
        )

    yes = sum(
        1
        for t in transactions
        if t["achieved"] == "Yes"
    )

    no = sum(
        1
        for t in transactions
        if t["achieved"] == "No"
    )

    completed = yes + no

    strike = (
        yes / completed * 100
        if completed
        else 0
    )

    print("-" * len(header))

    print(
        f"YES: {yes}   NO: {no}   "
        f"Strike Rate: {strike:.2f}%\n"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    import sys

    if (
        len(sys.argv) >= 3
        and sys.argv[1] == "--sst-debug"
    ):

        debug_sst_transactions(
            sys.argv[2].strip().upper()
        )

    else:

        main()
