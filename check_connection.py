"""
Langkah pertama proyek BTC/OKX: sekadar mastiin koneksi ke API OKX jalan.

1. Ambil harga BTC-USDT terkini (endpoint PUBLIK, nggak butuh API key)
2. Kalau OKX_API_KEY dkk sudah di-set, cek saldo akun DEMO (endpoint
   PRIVATE, butuh signature)

Cara pakai:
    python check_connection.py
"""
import json
import os
import requests

from okx_auth import get_auth_headers, DEMO_MODE

BASE_URL = "https://www.okx.com"


def check_public_ticker():
    print("=" * 60)
    print("1. CEK HARGA BTC-USDT (endpoint publik, nggak butuh API key)")
    print("=" * 60)
    path = "/api/v5/market/ticker"
    params = {"instId": "BTC-USDT"}
    resp = requests.get(BASE_URL + path, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if data.get("code") != "0":
        print(f"[ERROR] {data}")
        return False

    ticker = data["data"][0]
    print(f"  Simbol       : {ticker['instId']}")
    print(f"  Harga terkini: {ticker['last']} USDT")
    print(f"  24h high/low : {ticker['high24h']} / {ticker['low24h']}")
    print(f"  24h volume   : {ticker['vol24h']} BTC")
    print("\n  --> Koneksi ke OKX API BERHASIL (endpoint publik).")
    return True


def check_demo_balance():
    print("\n" + "=" * 60)
    print("2. CEK SALDO AKUN DEMO (endpoint private, butuh API key)")
    print("=" * 60)

    if not os.environ.get("OKX_API_KEY"):
        print("  OKX_API_KEY belum di-set (di GitHub Secrets / env var lokal).")
        print("  Skip cek saldo - tapi ini NORMAL kalau kamu belum setup API key.")
        return

    path = "/api/v5/account/balance"
    headers = get_auth_headers("GET", path)

    resp = requests.get(BASE_URL + path, headers=headers, timeout=10)
    data = resp.json()

    if data.get("code") != "0":
        print(f"  [ERROR] {data}")
        print("  Kemungkinan: API key salah, atau demo API key belum diaktifkan")
        print("  di OKX (Trade > Demo Trading > Personal Center).")
        return

    print(f"  Mode: {'DEMO (simulated trading)' if DEMO_MODE else '!!! LIVE !!!'}")
    details = data["data"][0].get("details", [])
    if not details:
        print("  Saldo kosong / belum ada dana virtual di akun demo ini.")
    for d in details:
        if float(d.get("cashBal", 0)) > 0:
            print(f"  {d['ccy']}: {d['cashBal']}")
    print("\n  --> Koneksi PRIVATE (autentikasi) BERHASIL.")


if __name__ == "__main__":
    ok = check_public_ticker()
    check_demo_balance()

    print("\n" + "=" * 60)
    print("RINGKASAN")
    print("=" * 60)
    if ok:
        print("Koneksi dasar ke OKX API berhasil. Siap lanjut ke langkah")
        print("berikutnya: kumpulin data historis BTC buat mulai analisis.")
    else:
        print("Ada masalah di koneksi dasar. Cek error di atas.")
