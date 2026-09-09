import pandas as pd
import yfinance as yf

import json
import os
import time
import warnings

from datetime import datetime
from zoneinfo import ZoneInfo


warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

CSV_FILE = "nifty100_symbols.csv"

OUTPUT_DIR = "docs"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "data.json"
)

IST = ZoneInfo("Asia/Kolkata")

DOWNLOAD_PERIOD = "3y"


# ============================================================
# MWD SETTINGS
# ============================================================

MWD_52W_DAYS = 252


# ============================================================
# SST SETTINGS
# ============================================================

SST_LOOKBACK = 20

SST_TARGET_PERCENT = 6.0


# ============================================================
# BLSH SETTINGS
# ============================================================

BLSH_LOOKBACK = 25

BLSH_TARGET_PERCENT = 3.14

RSI_PERIOD = 14

RSI_LIMIT = 36

TRADING_DAYS_1_YEAR = 252


# ============================================================
# READ SYMBOLS
# ============================================================

def read_symbols():

    df = pd.read_csv(CSV_FILE)

    df.columns = [
        str(col).strip().lower()
        for col in df.columns
    ]

    if "symbol" not in df.columns:

        raise ValueError(
            "CSV must contain a column named symbol"
        )

    if "company" not in df.columns:

        df["company"] = df["symbol"]

    df = df.dropna(
        subset=["symbol"]
    )

    df["symbol"] = (
        df["symbol"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["company"] = (
        df["company"]
        .astype(str)
        .str.strip()
    )

    return df.to_dict("records")


# ============================================================
# DOWNLOAD STOCK DATA
# ============================================================

def download_stock(symbol):

    yahoo_symbol = f"{symbol}.NS"

    try:

        df = yf.download(

            yahoo_symbol,

            period=DOWNLOAD_PERIOD,

            interval="1d",

            auto_adjust=False,

            progress=False,

            threads=False

        )

        if df is None or df.empty:

            print(
                f"NO DATA : {symbol}"
            )

            return None


        # Handle Yahoo MultiIndex columns

        if isinstance(
            df.columns,
            pd.MultiIndex
        ):

            df.columns = [
                col[0]
                for col in df.columns
            ]


        required_columns = [

            "Open",

            "High",

            "Low",

            "Close"

        ]


        for col in required_columns:

            if col not in df.columns:

                print(
                    f"MISSING COLUMN {col} : {symbol}"
                )

                return None


        df = df.dropna(
            subset=required_columns
        )


        if len(df) < 300:

            print(
                f"INSUFFICIENT DATA : {symbol}"
            )

            return None


        return df


    except Exception as e:

        print(
            f"DOWNLOAD ERROR {symbol} : {e}"
        )

        return None


# ============================================================
# RSI CALCULATION
# ============================================================

def calculate_rsi(close, period=14):

    delta = close.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )


    avg_gain = gain.ewm(

        alpha=1 / period,

        min_periods=period,

        adjust=False

    ).mean()


    avg_loss = loss.ewm(

        alpha=1 / period,

        min_periods=period,

        adjust=False

    ).mean()


    rs = avg_gain / avg_loss


    rsi = 100 - (

        100 /
        (1 + rs)

    )


    return rsi


# ============================================================
# WEEKLY DATA
# ============================================================

def get_weekly_data(df):

    weekly = pd.DataFrame()


    weekly["Open"] = (

        df["Open"]

        .resample("W-FRI")

        .first()

    )


    weekly["High"] = (

        df["High"]

        .resample("W-FRI")

        .max()

    )


    weekly["Low"] = (

        df["Low"]

        .resample("W-FRI")

        .min()

    )


    weekly["Close"] = (

        df["Close"]

        .resample("W-FRI")

        .last()

    )


    weekly = weekly.dropna()


    return weekly


# ============================================================
# MONTHLY DATA
# ============================================================

def get_monthly_data(df):

    monthly = pd.DataFrame()


    monthly["Open"] = (

        df["Open"]

        .resample("ME")

        .first()

    )


    monthly["High"] = (

        df["High"]

        .resample("ME")

        .max()

    )


    monthly["Low"] = (

        df["Low"]

        .resample("ME")

        .min()

    )


    monthly["Close"] = (

        df["Close"]

        .resample("ME")

        .last()

    )


    monthly = monthly.dropna()


    return monthly


# ============================================================
# CANDLE STATUS
# ============================================================

def candle_status(
    open_price,
    close_price
):

    if close_price > open_price:

        return "BULLISH"


    elif close_price < open_price:

        return "BEARISH"


    return "NEUTRAL"


# ============================================================
# MWD ANALYSIS
# ============================================================

def analyze_mwd(
    symbol,
    company,
    df
):

    try:


        # ----------------------------------------------------
        # DAILY
        # ----------------------------------------------------

        daily_last = df.iloc[-1]


        daily_status = candle_status(

            float(
                daily_last["Open"]
            ),

            float(
                daily_last["Close"]
            )

        )


        # ----------------------------------------------------
        # WEEKLY
        # ----------------------------------------------------

        weekly = get_weekly_data(df)


        if len(weekly) < 3:

            return None


        weekly_last = weekly.iloc[-1]


        weekly_status = candle_status(

            float(
                weekly_last["Open"]
            ),

            float(
                weekly_last["Close"]
            )

        )


        # ----------------------------------------------------
        # MONTHLY
        # ----------------------------------------------------

        monthly = get_monthly_data(df)


        if len(monthly) < 3:

            return None


        monthly_last = monthly.iloc[-1]


        monthly_status = candle_status(

            float(
                monthly_last["Open"]
            ),

            float(
                monthly_last["Close"]
            )

        )


        # ----------------------------------------------------
        # MWD CONDITION
        # ----------------------------------------------------

        if not (

            daily_status == "BULLISH"

            and weekly_status == "BULLISH"

            and monthly_status == "BULLISH"

        ):

            return None


        # ----------------------------------------------------
        # CMP
        # ----------------------------------------------------

        last_price = float(

            df["Close"].iloc[-1]

        )


        # ----------------------------------------------------
        # 52 WEEK HIGH
        # ----------------------------------------------------

        last_252 = df.tail(
            MWD_52W_DAYS
        )


        high_52_week = float(

            last_252["High"].max()

        )


        # Exclude stocks already crossing 52W High

        if last_price >= high_52_week:

            return None


        # ----------------------------------------------------
        # DISTANCE FROM 52 WEEK HIGH
        # ----------------------------------------------------

        distance_52w = (

            (
                last_price
                -
                high_52_week
            )

            /

            high_52_week

        ) * 100


        # ----------------------------------------------------
        # MONTHLY RISE
        # ----------------------------------------------------

        monthly_open = float(

            monthly_last["Open"]

        )


        monthly_close = float(

            monthly_last["Close"]

        )


        monthly_rise = (

            (
                monthly_close
                -
                monthly_open
            )

            /

            monthly_open

        ) * 100


        return {

            "symbol": symbol,

            "company": company,

            "price": round(
                last_price,
                2
            ),

            "high_52_week": round(
                high_52_week,
                2
            ),

            "distance_52w": round(
                distance_52w,
                2
            ),

            "monthly_rise": round(
                monthly_rise,
                2
            ),

            "daily": daily_status,

            "weekly": weekly_status,

            "monthly": monthly_status

        }


    except Exception as e:

        print(
            f"MWD ERROR {symbol}: {e}"
        )

        return None


# ============================================================
# SST HISTORICAL ANALYSIS
# ============================================================

def calculate_sst_history(df):

    target_yes = 0

    target_no = 0


    waiting_for_breakout = False

    active_trade = False


    trigger_price = None

    target_price = None


    start_index = SST_LOOKBACK + 1


    for i in range(

        start_index,

        len(df)

    ):


        previous_data = df.iloc[

            i - SST_LOOKBACK:i

        ]


        previous_low = float(

            previous_data["Low"].min()

        )


        previous_high = float(

            previous_data["High"].max()

        )


        current_low = float(

            df["Low"].iloc[i]

        )


        current_high = float(

            df["High"].iloc[i]

        )


        # ----------------------------------------------------
        # NEW 20 DAY LOW
        # ----------------------------------------------------

        new_low = (

            current_low
            <
            previous_low

        )


        if new_low:


            if active_trade:

                target_no += 1

                active_trade = False

                target_price = None


            trigger_price = previous_high

            waiting_for_breakout = True

            continue


        # ----------------------------------------------------
        # WAITING FOR BREAKOUT
        # ----------------------------------------------------

        if waiting_for_breakout:


            if current_high >= trigger_price:


                target_price = (

                    trigger_price

                    *

                    (
                        1
                        +
                        SST_TARGET_PERCENT / 100
                    )

                )


                waiting_for_breakout = False

                active_trade = True


            continue


        # ----------------------------------------------------
        # ACTIVE TRADE
        # ----------------------------------------------------

        if active_trade:


            if current_high >= target_price:


                target_yes += 1

                active_trade = False

                target_price = None


    total = (

        target_yes
        +
        target_no

    )


    strike_rate = (

        (
            target_yes / total
        ) * 100

        if total > 0

        else 0

    )


    return {

        "target_yes": target_yes,

        "target_no": target_no,

        "strike_rate": round(
            strike_rate,
            2
        )

    }


# ============================================================
# SST CURRENT ANALYSIS
# ============================================================

def analyze_sst(
    symbol,
    company,
    df
):

    try:


        last_price = float(

            df["Close"].iloc[-1]

        )


        last_20 = df.tail(

            SST_LOOKBACK

        )


        high_20_day = float(

            last_20["High"].max()

        )


        distance_20d = (

            (
                high_20_day
                -
                last_price
            )

            /

            high_20_day

        ) * 100


        history = calculate_sst_history(df)


        return {

            "symbol": symbol,

            "company": company,

            "price": round(
                last_price,
                2
            ),

            "high_20_day": round(
                high_20_day,
                2
            ),

            "distance_20d": round(
                distance_20d,
                2
            ),

            "target_yes": history[
                "target_yes"
            ],

            "target_no": history[
                "target_no"
            ],

            "strike_rate": history[
                "strike_rate"
            ]

        }


    except Exception as e:

        print(
            f"SST ERROR {symbol}: {e}"
        )

        return None


# ============================================================
# BLSH HISTORICAL ANALYSIS - LAST 1 YEAR
#
# Logic:
#
# New 25 Day Low
#       ↓
# Previous 25 Day High = Trigger
#       ↓
# Trigger Crossed
#       ↓
# Target = Trigger + 3.14%
#
# YES = Target Achieved
# NO  = New 25 Day Low before target
#
# Only events completed in last 252 trading days counted.
# ============================================================

def calculate_blsh_history_1_year(df):


    target_yes = 0

    target_no = 0


    waiting_for_trigger = False

    active_trade = False


    trigger_price = None

    target_price = None


    # Start from sufficient history

    start_index = max(

        BLSH_LOOKBACK + RSI_PERIOD,

        1

    )


    # Last one year starts here

    one_year_start = max(

        0,

        len(df) - TRADING_DAYS_1_YEAR

    )


    for i in range(

        start_index,

        len(df)

    ):


        previous_data = df.iloc[

            i - BLSH_LOOKBACK:i

        ]


        previous_low = float(

            previous_data["Low"].min()

        )


        previous_high = float(

            previous_data["High"].max()

        )


        current_low = float(

            df["Low"].iloc[i]

        )


        current_high = float(

            df["High"].iloc[i]

        )


        # ----------------------------------------------------
        # NEW 25 DAY LOW
        # ----------------------------------------------------

        new_25_day_low = (

            current_low
            <
            previous_low

        )


        # ----------------------------------------------------
        # NEW LOW DETECTED
        # ----------------------------------------------------

        if new_25_day_low:


            # Active trade fails

            if active_trade:


                # Count only if failure occurred
                # in last 1 year

                if i >= one_year_start:

                    target_no += 1


                active_trade = False

                target_price = None


            # Set fresh trigger

            trigger_price = previous_high

            waiting_for_trigger = True


            continue


        # ----------------------------------------------------
        # WAIT FOR TRIGGER BREAKOUT
        # ----------------------------------------------------

        if waiting_for_trigger:


            if current_high >= trigger_price:


                target_price = (

                    trigger_price

                    *

                    (
                        1
                        +
                        BLSH_TARGET_PERCENT / 100
                    )

                )


                waiting_for_trigger = False

                active_trade = True


            continue


        # ----------------------------------------------------
        # ACTIVE TRADE
        # ----------------------------------------------------

        if active_trade:


            # TARGET ACHIEVED

            if current_high >= target_price:


                # Count only if achieved in last 1 year

                if i >= one_year_start:

                    target_yes += 1


                active_trade = False

                target_price = None


    total = (

        target_yes
        +
        target_no

    )


    strike_rate = (

        (
            target_yes / total
        ) * 100

        if total > 0

        else 0

    )


    return {

        "target_yes_1y": target_yes,

        "target_no_1y": target_no,

        "strike_rate": round(
            strike_rate,
            2
        )

    }


# ============================================================
# BLSH CURRENT ANALYSIS
# ============================================================

def analyze_blsh(
    symbol,
    company,
    df
):

    try:


        df = df.copy()


        # ----------------------------------------------------
        # RSI
        # ----------------------------------------------------

        df["RSI"] = calculate_rsi(

            df["Close"],

            RSI_PERIOD

        )


        current_rsi = float(

            df["RSI"].iloc[-1]

        )


        if pd.isna(current_rsi):

            return None


        # ONLY RSI BELOW 36

        if current_rsi >= RSI_LIMIT:

            return None


        # ----------------------------------------------------
        # CMP
        # ----------------------------------------------------

        cmp = float(

            df["Close"].iloc[-1]

        )


        # ----------------------------------------------------
        # CURRENT 25 DAY DATA
        # ----------------------------------------------------

        last_25 = df.tail(

            BLSH_LOOKBACK

        )


        low_25_day = float(

            last_25["Low"].min()

        )


        trigger_price = float(

            last_25["High"].max()

        )


        # ----------------------------------------------------
        # TRIGGER AWAY FROM CMP
        # ----------------------------------------------------

        trigger_away = (

            (
                cmp
                -
                trigger_price
            )

            /

            trigger_price

        ) * 100


        # ----------------------------------------------------
        # LAST ONE YEAR PERFORMANCE
        # ----------------------------------------------------

        history = calculate_blsh_history_1_year(df)


        return {

            "symbol": symbol,

            "company": company,

            "cmp": round(
                cmp,
                2
            ),

            "low_25_day": round(
                low_25_day,
                2
            ),

            "trigger_price": round(
                trigger_price,
                2
            ),

            "trigger_away": round(
                trigger_away,
                2
            ),

            "rsi_14": round(
                current_rsi,
                2
            ),

            "target_yes_1y": history[
                "target_yes_1y"
            ],

            "target_no_1y": history[
                "target_no_1y"
            ],

            "strike_rate": history[
                "strike_rate"
            ]

        }


    except Exception as e:

        print(
            f"BLSH ERROR {symbol}: {e}"
        )

        return None


# ============================================================
# MAIN
# ============================================================

def main():


    print("=" * 70)

    print(
        "NIFTY 100 MULTI SCREENER"
    )

    print("=" * 70)


    symbols = read_symbols()


    total_stocks = len(symbols)


    mwd_results = []

    sst_results = []

    blsh_results = []


    latest_market_date = None


    # ========================================================
    # PROCESS ALL STOCKS
    # ========================================================

    for count, stock in enumerate(

        symbols,

        start=1

    ):


        symbol = stock["symbol"]

        company = stock["company"]


        print(

            f"[{count}/{total_stocks}] {symbol}"

        )


        df = download_stock(symbol)


        if df is None:

            continue


        latest_market_date = (

            df.index[-1]

            .strftime("%d-%b-%Y")

        )


        # ----------------------------------------------------
        # MWD
        # ----------------------------------------------------

        result = analyze_mwd(

            symbol,

            company,

            df

        )


        if result:

            mwd_results.append(
                result
            )


        # ----------------------------------------------------
        # SST
        # ----------------------------------------------------

        result = analyze_sst(

            symbol,

            company,

            df

        )


        if result:

            sst_results.append(
                result
            )


        # ----------------------------------------------------
        # BLSH
        # ----------------------------------------------------

        result = analyze_blsh(

            symbol,

            company,

            df

        )


        if result:

            blsh_results.append(
                result
            )


        time.sleep(0.20)


    # ========================================================
    # SORT MWD
    #
    # Nearest 52W High first
    # ========================================================

    mwd_results = sorted(

        mwd_results,

        key=lambda x:

            x["distance_52w"],

        reverse=True

    )


    # ========================================================
    # SORT SST
    #
    # Nearest 20 Day High first
    # ========================================================

    sst_results = sorted(

        sst_results,

        key=lambda x:

            x["distance_20d"]

    )


    # ========================================================
    # SORT BLSH
    #
    # Nearest Trigger Price first
    # ========================================================

    blsh_results = sorted(

        blsh_results,

        key=lambda x:

            abs(
                x["trigger_away"]
            )

    )


    # ========================================================
    # IST TIME
    # ========================================================

    current_time = datetime.now(

        IST

    ).strftime(

        "%d-%b-%Y %I:%M %p IST"

    )


    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    os.makedirs(

        OUTPUT_DIR,

        exist_ok=True

    )


    # ========================================================
    # FINAL JSON
    # ========================================================

    output = {


        "last_updated": current_time,


        "latest_market_date": latest_market_date,


        "total_stocks": total_stocks,


        "mwd": {

            "count": len(
                mwd_results
            ),

            "data": mwd_results

        },


        "sst": {

            "count": len(
                sst_results
            ),

            "data": sst_results

        },


        "blsh": {

            "count": len(
                blsh_results
            ),

            "data": blsh_results

        }

    }


    # ========================================================
    # WRITE JSON
    # ========================================================

    with open(

        OUTPUT_FILE,

        "w",

        encoding="utf-8"

    ) as f:


        json.dump(

            output,

            f,

            indent=4,

            ensure_ascii=False

        )


    print()

    print("=" * 70)

    print("COMPLETED")

    print("=" * 70)

    print(
        f"Total Stocks : {total_stocks}"
    )

    print(
        f"MWD Matches  : {len(mwd_results)}"
    )

    print(
        f"SST Stocks   : {len(sst_results)}"
    )

    print(
        f"BLSH Stocks  : {len(blsh_results)}"
    )

    print()

    print(
        f"Updated IST : {current_time}"
    )

    print("=" * 70)


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()
