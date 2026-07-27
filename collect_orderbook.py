"""
Collector snapshot order book BTC-USDT dari OKX, jalan berkala (cron).

Beda dari collect_candles.py: ini narik struktur order book (harga & volume
antrian beli/jual di beberapa level), bukan cuma candle OHLCV. Dari situ
dihitung "order flow imbalance" - metrik yang menurut riset akademis (Cont,
Kukanov & Stoikov 2010, dkk) berhubungan sama pergerakan harga jangka
sangat pendek.

Kenapa REST, bukan WebSocket: WebSocket butuh koneksi nyala terus-menerus,
nggak cocok buat GitHub Actions yang jalan sebentar-sebentar. Ini trade-off
sadar - resolusi datanya lebih rendah (per beberapa menit, bukan per
100 milidetik) dibanding riset akademis aslinya, tapi tetap dalam
infrastruktur yang udah ada.

Metrik yang disimpan tiap snapshot:
- mid_price: rata-rata harga bid tertinggi & ask terendah
- bid_vol / ask_vol: total volume di 5 level teratas masing-masing sisi
- imbalance: (bid_vol - ask_vol) / (bid_vol + ask_vol), range -1 sampai +1
  (positif = tekanan beli lebih besar, negatif = tekanan jual lebih besar)

Cara pakai:
    python collect_orderbook.py
"""
import csv
import os
import requests

BASE_URL = "https://www.okx.com"
INST_ID = "BTC-USDT"
DEPTH = 5  # ambil 5 level teratas tiap sisi
OUTPUT_FILE = "orderbook_dataset.csv"


def fetch_orderbook():
    path = "/api/v5/market/books"
    params = {"instId": INST_ID, "sz": DEPTH}
    resp = requests.get(BASE_URL + path, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != "0":
        print(f"[ERROR] {data}")
        return None
    return data["data"][0]


def main():
    book = fetch_orderbook()
    if book is None:
        return

    bids = book["bids"]  # list of [price, size, deprecated, num_orders]
    asks = book["asks"]
    ts = book["ts"]  # timestamp dalam ms, dari server OKX

    if not bids or not asks:
        print("Order book kosong, skip.")
        return

    best_bid = float(bids[0][0])
    best_ask = float(asks[0][0])
    mid_price = (best_bid + best_ask) / 2

    bid_vol = sum(float(level[1]) for level in bids[:DEPTH])
    ask_vol = sum(float(level[1]) for level in asks[:DEPTH])
    total_vol = bid_vol + ask_vol
    imbalance = (bid_vol - ask_vol) / total_vol if total_vol > 0 else 0.0

    file_exists = os.path.exists(OUTPUT_FILE)
    with open(OUTPUT_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp_ms", "mid_price", "best_bid", "best_ask",
                "bid_vol", "ask_vol", "imbalance",
            ])
        writer.writerow([ts, mid_price, best_bid, best_ask, bid_vol, ask_vol, imbalance])

    print(f"Snapshot disimpan: mid_price={mid_price:.2f} | imbalance={imbalance:+.4f} "
          f"(bid_vol={bid_vol:.4f}, ask_vol={ask_vol:.4f})")


if __name__ == "__main__":
    main()
