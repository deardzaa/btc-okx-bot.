"""
Collector yang jalan berkala (lewat GitHub Actions cron) buat nambahin
candle BTC-USDT terbaru ke btc_dataset.csv - versi "terus hidup" dari
backfill_history.py.

Beda dari backfill: ini ambil candle TERBARU (bukan mundur ke masa lalu),
dan didesain buat dipanggil berulang-ulang tanpa duplikat data.

Cara pakai:
    python collect_candles.py
"""
import csv
import os
import requests

BASE_URL = "https://www.okx.com"
INST_ID = "BTC-USDT"
BAR = "1m"
OUTPUT_FILE = "btc_dataset.csv"


def fetch_latest():
    path = "/api/v5/market/candles"  # endpoint recent, bukan history
    params = {"instId": INST_ID, "bar": BAR, "limit": 100}
    resp = requests.get(BASE_URL + path, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != "0":
        print(f"[ERROR] {data}")
        return []
    return data.get("data", [])


def main():
    candles = fetch_latest()
    if not candles:
        print("Nggak dapet data dari API.")
        return

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
        # urutkan dari yang paling lama ke paling baru biar rapi
        for c in sorted(candles, key=lambda x: int(x[0])):
            if c[0] in existing_ts:
                continue
            writer.writerow([c[0], c[1], c[2], c[3], c[4], c[5]])
            new_rows += 1

    print(f"Ditambahin {new_rows} candle baru. Total baris di file sekarang: "
          f"{len(existing_ts) + new_rows}")


if __name__ == "__main__":
    main()
