"""
Uji apakah fitur VOLUME (belum pernah dicoba sebelumnya - semua percobaan
kita sejauh ini cuma dari harga) bisa nambah sinyal, dengan 2 cara:

1. Tambah fitur volume ke model ML (volume ratio, OBV, volume spike flag)
   dan lihat apa akurasi keseluruhan naik dibanding model tanpa volume.
2. UJI LANGSUNG hipotesis "lonjakan volume = pergerakan lebih predictable":
   bandingin akurasi KHUSUS di candle dengan volume tinggi vs candle normal.

Cara pakai:
    python test_volume_features.py
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import lightgbm as lgb

INPUT_FILE = "btc_dataset.csv"
LOOKBACK = 10
HORIZON = 5  # menit, horizon yang paling konsisten oke di sweep sebelumnya
VOLUME_SPIKE_THRESHOLD = 2.0  # volume dianggap "lonjakan" kalau >= 2x rata-rata


def compute_base_indicators(df):
    close, high, low = df["close"], df["high"], df["low"]

    df["sma_fast"] = close.rolling(5).mean()
    df["sma_slow"] = close.rolling(20).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    df["bb_upper"] = mid + 2 * std
    df["bb_lower"] = mid - 2 * std

    lowest_low = low.rolling(14).min()
    highest_high = high.rolling(14).max()
    df["stoch_k"] = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()
    df["williams_r"] = -100 * (highest_high - close) / (highest_high - lowest_low).replace(0, np.nan)
    df["roc"] = (close - close.shift(12)) / close.shift(12) * 100
    return df


def compute_volume_features(df):
    vol = df["volume"]
    close = df["close"]

    df["volume_sma"] = vol.rolling(20).mean()
    df["volume_ratio"] = vol / df["volume_sma"].replace(0, np.nan)
    df["volume_spike"] = (df["volume_ratio"] >= VOLUME_SPIKE_THRESHOLD).astype(int)

    # On-Balance Volume: volume dikasih tanda +/- sesuai arah harga, dikumulatifin
    direction = np.sign(close.diff()).fillna(0)
    df["obv"] = (direction * vol).cumsum()
    df["obv_slope"] = df["obv"].diff(5)  # arah OBV 5 candle terakhir
    return df


def build_features(df, use_volume: bool):
    df = df.copy().sort_values("timestamp_ms").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["close", "volume"]).reset_index(drop=True)

    df = compute_base_indicators(df)
    df = compute_volume_features(df)

    for lag in range(1, LOOKBACK + 1):
        df[f"lag_{lag}"] = df["close"].diff(lag)

    df["sma_gap"] = df["sma_fast"] - df["sma_slow"]
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    bb_range = df["bb_upper"] - df["bb_lower"]
    df["bb_percent"] = (df["close"] - df["bb_lower"]) / bb_range.replace(0, np.nan)
    df["bb_width"] = bb_range

    df["target"] = (df["close"].shift(-HORIZON) > df["close"]).astype(int)

    lag_cols = [c for c in df.columns if c.startswith("lag_")]
    base_cols = ["sma_gap", "rsi", "macd_hist", "bb_percent", "bb_width",
                 "stoch_k", "stoch_d", "williams_r", "roc"]
    volume_cols = ["volume_ratio", "obv_slope"]

    feature_cols = lag_cols + base_cols + (volume_cols if use_volume else [])
    required = feature_cols + ["target", "volume_spike"]
    df = df.dropna(subset=required).reset_index(drop=True)
    return df, feature_cols


def evaluate(df_raw, use_volume, label):
    feat_df, feature_cols = build_features(df_raw, use_volume)
    X = feat_df[feature_cols]
    y = feat_df["target"]
    spike_flag = feat_df["volume_spike"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    spike_test = spike_flag.iloc[y_train.shape[0]:].reset_index(drop=True)

    model = lgb.LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                random_state=42, verbose=-1, class_weight="balanced")
    model.fit(X_train, y_train)

    # Validasi non-overlapping
    non_overlap_idx = np.arange(0, len(X_test), HORIZON)
    y_test_no = y_test.to_numpy()[non_overlap_idx]
    X_test_no = X_test.iloc[non_overlap_idx]
    spike_no = spike_test.to_numpy()[non_overlap_idx]

    pred_no = model.predict(X_test_no)
    acc_all = accuracy_score(y_test_no, pred_no)

    # Pisahin akurasi: candle dengan volume spike vs candle normal
    spike_mask = spike_no == 1
    normal_mask = spike_no == 0

    acc_spike = accuracy_score(y_test_no[spike_mask], pred_no[spike_mask]) if spike_mask.sum() >= 15 else None
    acc_normal = accuracy_score(y_test_no[normal_mask], pred_no[normal_mask]) if normal_mask.sum() >= 15 else None

    print(f"\n--- {label} ---")
    print(f"Fitur volume dipakai: {use_volume} | Total fitur: {len(feature_cols)}")
    print(f"Akurasi keseluruhan (non-overlap, n={len(y_test_no)}): {acc_all:.3f}")
    if acc_spike is not None:
        print(f"Akurasi KHUSUS saat volume spike (n={spike_mask.sum()}): {acc_spike:.3f}")
    else:
        print(f"Sample volume spike terlalu sedikit (n={spike_mask.sum()}) buat disimpulkan.")
    if acc_normal is not None:
        print(f"Akurasi saat volume NORMAL (n={normal_mask.sum()}): {acc_normal:.3f}")

    return acc_all, acc_spike


def main():
    df_raw = pd.read_csv(INPUT_FILE)
    print(f"Data: {len(df_raw)} baris | Horizon prediksi: {HORIZON} menit")
    print(f"Threshold volume spike: >= {VOLUME_SPIKE_THRESHOLD}x rata-rata 20 candle\n")
    print("=" * 60)
    print("PERBANDINGAN: model TANPA volume vs DENGAN volume")
    print("=" * 60)

    acc_no_vol, _ = evaluate(df_raw, use_volume=False, label="TANPA fitur volume (baseline)")
    acc_vol, acc_spike = evaluate(df_raw, use_volume=True, label="DENGAN fitur volume")

    print("\n" + "=" * 60)
    print("KESIMPULAN")
    print("=" * 60)
    print(f"Akurasi tanpa volume : {acc_no_vol:.3f}")
    print(f"Akurasi dengan volume: {acc_vol:.3f}")
    diff = acc_vol - acc_no_vol
    print(f"Selisih: {diff:+.3f}")
    if abs(diff) < 0.02:
        print("--> Selisih KECIL (<2 poin), kemungkinan cuma noise, bukan")
        print("    improvement nyata dari volume.")
    elif diff > 0:
        print("--> Volume KELIHATAN nambah sinyal. Tapi perlu divalidasi ulang")
        print("    di data baru sebelum yakin ini bukan kebetulan.")
    else:
        print("--> Volume nggak nambah, malah kelihatan menurunkan akurasi.")

    if acc_spike is not None:
        print(f"\nAkurasi khusus saat volume spike: {acc_spike:.3f} vs keseluruhan {acc_vol:.3f}")
        if acc_spike - acc_vol > 0.03:
            print("--> Hipotesis 'lonjakan volume = lebih predictable' ada dukungan")
            print("    awal. Tapi cek juga jumlah sample-nya - kalau kecil, hati-hati.")
        else:
            print("--> Nggak ada bukti kuat lonjakan volume bikin pergerakan lebih")
            print("    predictable dibanding kondisi biasa.")


if __name__ == "__main__":
    main()
