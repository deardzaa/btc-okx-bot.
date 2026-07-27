"""
Tarik data historis candle BTC-USDT (1 menit) dari OKX, buat modal awal
dataset sebelum collector otomatis (collect_candles.py) mulai nambahin
data baru secara berkala.

OKX batasin max 100 candle per request di endpoint history-candles, jadi
script ini looping mundur (pagination pakai parameter "after") buat
narik beberapa ratus/ribu candle sekaligus.

Cara pakai:
    python backfill_history.py [jumlah_candle]
    (default 1000 candle = sekitar 16-17 jam data 1 menit)
"""
import csv
import os
import sys
import time
import requests

BASE_URL = "https://www.okx.com"
INST_ID = "BTC-USDT"
BAR = "1m"
OUTPUT_FILE = "btc_dataset.csv"
MAX_PER_REQUEST = 100  # limit OKX buat history-candles


def fetch_batch(after_ts=None):
    path = "/api/v5/market/history-candles"
    params = {"instId": INST_ID, "bar": BAR, "limit": MAX_PER_REQUEST}
    if after_ts:
        params["after"] = after_ts  # ambil candle SEBELUM timestamp ini

    resp = requests.get(BASE_URL + path, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if data.get("code") != "0":
        print(f"[ERROR] {data}")
        return []
    return data.get("data", [])


def main():
    total_wanted = int(sys.argv[1]) if len(sys.argv) > 1 else 1000

    all_candles = []
    after_ts = None
    print(f"Menarik {total_wanted} candle 1 menit BTC-USDT dari OKX...")

    while len(all_candles) < total_wanted:
        batch = fetch_batch(after_ts)
        if not batch:
            print("Nggak ada data lagi dari API, berhenti di sini.")
            break

        all_candles.extend(batch)
        after_ts = batch[-1][0]  # timestamp candle paling lama di batch ini
        print(f"  Sudah dapet {len(all_candles)} candle...")
        time.sleep(0.3)  # jaga-jaga rate limit OKX

    all_candles = all_candles[:total_wanted]

    # Format tiap candle dari OKX: [ts, open, high, low, close, volume, volCcy, volCcyQuote, confirm]
    file_exists = os.path.exists(OUTPUT_FILE)
    existing_ts = set()
    if file_exists:
        with open(OUTPUT_FILE, "r") as f:
            reader = csv.DictReader(f)
            existing_ts = {row["timestamp_ms"] for row in reader}

    new_rows = 0
    with open(OUTPUT_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp_ms", "open", "high", "low", "close", "volume"])
        for c in all_candles:
            if c[0] in existing_ts:
                continue
            writer.writerow([c[0], c[1], c[2], c[3], c[4], c[5]])
            new_rows += 1

    print(f"\nSelesai. {new_rows} baris baru ditambahin ke {OUTPUT_FILE}.")
    print(f"(Total candle yang ditarik kali ini: {len(all_candles)}, "
          f"yang duplikat/udah ada di-skip)")


if __name__ == "__main__":
    main()
