"""
Hitung indikator teknikal (SMA, RSI, MACD, Bollinger Bands, Stochastic,
Williams %R, ROC) dari btc_dataset.csv - metodologi sama persis kayak
proyek Deriv dulu, biar hasilnya bisa dibandingin apple-to-apple.

Cara pakai:
    python compute_indicators.py
"""
import pandas as pd
import numpy as np

INPUT_FILE = "btc_dataset.csv"
OUTPUT_FILE = "btc_dataset_indicators.csv"


def compute_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_signal


def compute_bollinger(close, period=20, num_std=2):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def compute_stochastic(high, low, close, period=14, smooth=3):
    lowest_low = low.rolling(period).min()
    highest_high = high.rolling(period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    d = k.rolling(smooth).mean()
    return k, d


def compute_williams_r(high, low, close, period=14):
    highest_high = high.rolling(period).max()
    lowest_low = low.rolling(period).min()
    return -100 * (highest_high - close) / (highest_high - lowest_low).replace(0, np.nan)


def compute_roc(close, period=12):
    return (close - close.shift(period)) / close.shift(period) * 100


def main():
    df = pd.read_csv(INPUT_FILE)
    df = df.sort_values("timestamp_ms").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["close"]).reset_index(drop=True)

    print(f"Baris data mentah: {len(df)}")

    df["sma_fast"] = df["close"].rolling(5).mean()
    df["sma_slow"] = df["close"].rolling(20).mean()
    df["rsi"] = compute_rsi(df["close"])
    df["macd"], df["macd_signal"] = compute_macd(df["close"])
    df["bb_upper"], df["bb_mid"], df["bb_lower"] = compute_bollinger(df["close"])
    df["stoch_k"], df["stoch_d"] = compute_stochastic(df["high"], df["low"], df["close"])
    df["williams_r"] = compute_williams_r(df["high"], df["low"], df["close"])
    df["roc"] = compute_roc(df["close"])

    df.to_csv(OUTPUT_FILE, index=False)
    valid_rows = df.dropna(subset=["sma_slow", "rsi", "macd", "bb_upper", "stoch_k", "williams_r", "roc"])
    print(f"Baris dengan semua indikator lengkap (siap dianalisis): {len(valid_rows)}")
    print(f"Disimpan ke {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
