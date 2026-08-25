import os
import requests
import pandas as pd
from dotenv import load_dotenv
import time
from datetime import datetime, timedelta, timezone

# =========================
# LOAD .env
# =========================
load_dotenv()

API_KEY = os.getenv("CRYPTOCOMPARE_API_KEY")
if not API_KEY:
    raise ValueError("API key not found. Check your .env file")

# =========================
# CONFIG
# =========================
BASE_URL = "https://min-api.cryptocompare.com/data/v2/histohour"
TSYM = "USD"
LIMIT = 2000
# Pull only the last 2 years
START_DATE = (datetime.now(timezone.utc) - timedelta(days=365 * 5)).date().isoformat()
# Top 6 majors (high liquidity, long history)
SYMBOLS = ["BTC", "ETH", "BNB", "XRP", "ADA", "SOL"]

# =========================
# REQUEST
# =========================
headers = {
    "authorization": f"Apikey {API_KEY}"
}

all_frames = []

for fsym in SYMBOLS:
    print(f"Fetching {fsym}...")
    all_data = []
    to_ts = None
    start_ts = int(datetime.fromisoformat(START_DATE).replace(tzinfo=timezone.utc).timestamp())

    while True:
        params = {
            "fsym": fsym,
            "tsym": TSYM,
            "limit": LIMIT
        }
        if to_ts is not None:
            params["toTs"] = to_ts

        response = requests.get(BASE_URL, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()

        if data.get("Response") == "Error":
            raise ValueError(f"API error for {fsym}: {data.get('Message')}")

        batch = data.get("Data", {}).get("Data", [])
        if not batch:
            break

        all_data.extend(batch)

        earliest_ts = min(item["time"] for item in batch)
        # Stop if we're not moving back in time
        if to_ts is not None and earliest_ts >= to_ts:
            break

        # Stop once we've reached the start date
        if earliest_ts <= start_ts:
            break

        # Move back one hour before earliest to avoid overlap
        to_ts = earliest_ts - 1

        # Be polite to the API
        time.sleep(0.2)

    if not all_data:
        print(f"No data for {fsym}")
        continue

    df = pd.DataFrame(all_data)
    df["symbol"] = fsym
    df["conversionSymbol"] = TSYM
    df["conversionType"] = data.get("Data", {}).get("ConversionType", {}).get("type")
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df[df["time"] >= pd.to_datetime(START_DATE)]

    df = df[
        [
            "time",
            "high",
            "low",
            "open",
            "volumefrom",
            "volumeto",
            "close",
            "conversionType",
            "conversionSymbol",
            "symbol"
        ]
    ]

    all_frames.append(df)
    print(f"{fsym} rows: {len(df)}")

if not all_frames:
    raise ValueError("No data returned for any symbols.")

final_df = pd.concat(all_frames, ignore_index=True)
print(final_df.head())
print("Total rows:", len(final_df))

# =========================
# SAVE
# =========================
final_df.to_csv("../Data/raw/crypto_price_data.csv", index=False)
