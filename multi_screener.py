import pandas as pd
import yfinance as yf
import time
import os

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
BLSH_TRIGGER_PCT = 6.5
BLSH_TARGET_PCT = 3.14

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
    # Last completed week
    # --------------------------------------------------------

    weekly_current = weekly.iloc[-2]


    # --------------------------------------------------------
    # Last completed month
    # --------------------------------------------------------

    monthly_current = monthly.iloc[-2]


    # --------------------------------------------------------
    # All three must be bullish
    # --------------------------------------------------------

    if not (
        bullish(daily_current)
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
# CORRECTED LOGIC
#
# Every new 20-day high creates a separate transaction.
#
# Entry:
#     Previous 20-day high
#
# Target:
#     Entry + 6%
#
# Result:
#     YES = target reached first
#     NO  = new 20-day low reached first
#
# IMPORTANT:
# Multiple trades can exist simultaneously.
#
# Example:
#
# Trade 1 starts
# Trade 2 starts before Trade 1 finishes
# Trade 3 starts before Trade 2 finishes
#
# All trades are tracked independently.
# ============================================================

def cycle_statistics(
    df,
    lookback,
    target_pct
):

    # --------------------------------------------------------
    # Keep approximately one year of analysis data.
    #
    # Extra 'lookback' rows are required so the first
    # analysis day can calculate the previous 20-day high/low.
    # --------------------------------------------------------

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

        return (
            0,
            0,
            0.0
        )


    yes = 0
    no = 0


    # --------------------------------------------------------
    # Check EVERY possible 20-day breakout.
    #
    # There is intentionally NO waiting_for_low variable.
    # --------------------------------------------------------

    for i in range(
        lookback,
        len(data)
    ):


        # ----------------------------------------------------
        # STEP 1
        #
        # Is today's High a NEW 20-DAY HIGH?
        # ----------------------------------------------------

        if not is_new_high(
            data,
            i,
            lookback
        ):

            continue


        # ----------------------------------------------------
        # STEP 2
        #
        # Entry = previous 20-day high
        #
        # IMPORTANT:
        # Current breakout day's High is excluded.
        # ----------------------------------------------------

        entry = float(
            data["High"]
            .iloc[
                i - lookback:i
            ]
            .max()
        )


        # ----------------------------------------------------
        # STEP 3
        #
        # Target = 6% above entry
        # ----------------------------------------------------

        target = (
            entry
            * (
                1
                + target_pct / 100
            )
        )


        outcome = None


        # ----------------------------------------------------
        # STEP 4
        #
        # Track THIS transaction independently.
        #
        # Start checking from the next trading day.
        # ----------------------------------------------------

        for j in range(
            i + 1,
            len(data)
        ):


            current_high = float(
                data["High"].iloc[j]
            )


            # ------------------------------------------------
            # TARGET CHECK
            #
            # Target has priority, matching the original
            # source behavior when both conditions occur
            # on the same candle.
            # ------------------------------------------------

            if current_high >= target:

                outcome = "YES"

                break


            # ------------------------------------------------
            # NEW 20-DAY LOW
            # ------------------------------------------------

            if is_new_low(
                data,
                j,
                lookback
            ):

                outcome = "NO"

                break


        # ----------------------------------------------------
        # STEP 5
        #
        # Count only completed transactions.
        # ----------------------------------------------------

        if outcome == "YES":

            yes += 1

        elif outcome == "NO":

            no += 1

        # If outcome is None:
        #
        # The transaction is still open at the end of the
        # available data and is therefore not included.
        # ----------------------------------------------------


    # --------------------------------------------------------
    # STRIKE RATE
    # --------------------------------------------------------

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
# BLSH HISTORICAL STATISTICS
# ============================================================

def blsh_history_1_year(df):

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

        return (
            0,
            0,
            0.0
        )


    yes = 0
    no = 0


    i = BLSH_LOOKBACK


    trigger_active = False

    reference_low = None
    trigger_price = None
    target_price = None


    while i < len(data):


        current_low = float(
            data["Low"].iloc[i]
        )

        current_high = float(
            data["High"].iloc[i]
        )


        # ----------------------------------------------------
        # STEP 1
        #
        # NEW 25-DAY LOW
        # ----------------------------------------------------

        if is_new_low(
            data,
            i,
            BLSH_LOOKBACK
        ):


            # Existing trigger failed
            if trigger_active:

                no += 1


            reference_low = current_low


            # Trigger = 6.5% above 25D low
            trigger_price = (
                reference_low
                * (
                    1
                    + BLSH_TRIGGER_PCT
                    / 100
                )
            )


            # Target = 3.14% above trigger
            target_price = (
                trigger_price
                * (
                    1
                    + BLSH_TARGET_PCT
                    / 100
                )
            )


            trigger_active = False


            i += 1

            continue


        # ----------------------------------------------------
        # STEP 2
        #
        # WAIT FOR +6.5% TRIGGER
        # ----------------------------------------------------

        if (
            reference_low is not None
            and not trigger_active
        ):


            if current_high >= trigger_price:


                trigger_active = True


                # ------------------------------------------------
                # Target can also be reached on the same day.
                # ------------------------------------------------

                if current_high >= target_price:


                    yes += 1


                    reference_low = None
                    trigger_price = None
                    target_price = None
                    trigger_active = False


            i += 1

            continue


        # ----------------------------------------------------
        # STEP 3
        #
        # TARGET AFTER TRIGGER
        # ----------------------------------------------------

        if trigger_active:


            if current_high >= target_price:


                yes += 1


                reference_low = None
                trigger_price = None
                target_price = None
                trigger_active = False


                i += 1

                continue


        i += 1


    # --------------------------------------------------------
    # STRIKE RATE
    # --------------------------------------------------------

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
# HTML
# ============================================================

def build_html(
    mwd,
    sst,
    blsh,
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

                <td>{x['symbol']}</td>

                <td>{x['company']}</td>

                <td>
                    ₹{x['price']:,.2f}
                </td>

                <td>
                    ₹{x['high52']:,.2f}
                </td>

                <td>
                    {x['distance']:.2f}%
                </td>

                <td>
                    <span class="positive">
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

                <td>{x['symbol']}</td>

                <td>{x['company']}</td>

                <td>
                    ₹{x['price']:,.2f}
                </td>

                <td>
                    ₹{x['high20']:,.2f}
                </td>

                <td>
                    {x['away']:.2f}%
                </td>

                <td>
                    {x['yes']}
                </td>

                <td>
                    {x['no']}
                </td>

                <td>
                    {x['strike']:.2f}%
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

                <td>{x['symbol']}</td>

                <td>{x['company']}</td>

                <td>
                    ₹{x['cmp']:,.2f}
                </td>

                <td>
                    ₹{x['low_25_day']:,.2f}
                </td>

                <td>
                    ₹{x['trigger_price']:,.2f}
                </td>

                <td>
                    ₹{x['target_price']:,.2f}
                </td>

                <td>
                    {x['trigger_away']:.2f}%
                </td>

                <td>
                    {x['rsi']:.2f}
                </td>

                <td>
                    {x['yes']}
                </td>

                <td>
                    {x['no']}
                </td>

                <td>
                    {x['strike']:.2f}%
                </td>

            </tr>
            """

            for x in blsh
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


<style>

* {{
    box-sizing: border-box;
}}


body {{

    margin: 0;

    font-family:
        Arial,
        sans-serif;

    background:
        #eef2f7;

    color:
        #26384d;

    padding:
        20px;
}}


.top {{

    background:
        #ffffff;

    border-radius:
        18px;

    padding:
        20px 36px;

    display:
        flex;

    justify-content:
        space-between;

    align-items:
        center;

    box-shadow:
        0 8px 25px #ccd5e055;

    margin-bottom:
        25px;
}}


.title {{

    font-size:
        30px;

    font-weight:
        800;
}}


.subtitle {{

    margin-top:
        10px;

    color:
        #5d6d7e;
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
        12px 22px;

    border-radius:
        25px;

    font-weight:
        bold;

    background:
        #dcebf5;
}}


.badge.green {{

    background:
        #dcefe2;

    color:
        #29683c;
}}


.tabs {{

    display:
        flex;

    gap:
        10px;

    margin-bottom:
        20px;

    flex-wrap:
        wrap;
}}


.tab {{

    border:
        none;

    padding:
        14px 28px;

    border-radius:
        10px;

    background:
        #d7dee8;

    color:
        #26384d;

    font-size:
        16px;

    font-weight:
        bold;

    cursor:
        pointer;
}}


.tab.active {{

    background:
        #2858b5;

    color:
        white;
}}


.tab-content {{

    display:
        none;

    background:
        white;

    border-radius:
        18px;

    overflow:
        hidden;

    box-shadow:
        0 8px 25px #ccd5e055;
}}


.tab-content.active {{

    display:
        block;
}}


.panel-head {{

    padding:
        22px 25px;

    color:
        white;

    font-size:
        22px;

    font-weight:
        800;

    display:
        flex;

    justify-content:
        space-between;

    align-items:
        center;
}}


.mwd-head {{
    background:
        #2858b5;
}}


.sst-head {{
    background:
        #6b31c7;
}}


.blsh-head {{
    background:
        #147b65;
}}


.panel-sub {{

    font-size:
        15px;

    font-weight:
        normal;
}}


.filters {{

    padding:
        14px 20px;

    background:
        #f8fafc;

    border-bottom:
        1px solid #e5e8ec;

    display:
        flex;

    gap:
        15px;

    align-items:
        center;

    flex-wrap:
        wrap;
}}


select,
input {{

    padding:
        8px;

    border:
        1px solid #ccd4dd;

    border-radius:
        6px;
}}


label {{

    font-size:
        13px;

    font-weight:
        bold;
}}


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
        900px;
}}


th {{

    text-align:
        left;

    color:
        #607080;

    background:
        #f6f7f9;

    padding:
        13px;

    font-size:
        14px;

    white-space:
        nowrap;
}}


td {{

    padding:
        14px 13px;

    border-bottom:
        1px solid #e5e8ec;

    font-size:
        14px;

    white-space:
        nowrap;
}}


tr:last-child td {{

    border-bottom:
        none;
}}


.positive {{

    background:
        #e0f0e7;

    color:
        #1c7043;

    padding:
        5px 8px;

    border-radius:
        6px;

    font-weight:
        bold;
}}


.footer {{

    margin-top:
        25px;

    text-align:
        center;

    color:
        #64748b;

    font-size:
        13px;
}}


@media(max-width:700px) {{

    body {{
        padding:
            10px;
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
            24px;
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
            Total Stocks:
            {total_stocks}
        </div>

        <div class="badge green">
            MWD Matches:
            {len(mwd)}
        </div>

        <div class="badge green">
            SST Matches:
            {len(sst)}
        </div>

        <div class="badge green">
            BLSH Matches:
            {len(blsh)}
        </div>

    </div>

</div>


<!-- ===================================================== -->
<!-- TABS -->
<!-- ===================================================== -->

<div class="tabs">

    <button
        class="tab active"
        onclick="showTab('mwd', this)"
    >
        1. MWD
    </button>


    <button
        class="tab"
        onclick="showTab('sst', this)"
    >
        2. SST
    </button>


    <button
        class="tab"
        onclick="showTab('blsh', this)"
    >
        3. BLSH RSI
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

                    <th>25-Day Low</th>

                    <th>Trigger +6.5%</th>

                    <th>Target +3.14%</th>

                    <th>Trigger Away %</th>

                    <th>RSI(14)</th>

                    <th>3.14% YES (1Y)</th>

                    <th>3.14% NO (1Y)</th>

                    <th>Strike Rate</th>

                </tr>

            </thead>


            <tbody>

                {
                    rows_blsh()
                    or
                    '''<tr><td colspan="11">
                    No current matching stocks
                    </td></tr>'''
                }

            </tbody>

        </table>

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
                    cells.length < 11
                ) {{

                    return;

                }}


                const strike =
                    parseFloat(
                        cells[10]
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


// ==========================================================
// INITIAL FILTER
// ==========================================================

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

    # --------------------------------------------------------
    # Load symbols
    # --------------------------------------------------------

    symbols = load_symbols()


    mwd_results = []
    sst_results = []
    blsh_results = []


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
# RUN
# ============================================================

if __name__ == "__main__":

    main()
