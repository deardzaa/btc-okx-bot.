"""
Sweep berbagai horizon prediksi (1, 3, 5, 10, 15, 30, 60 menit) buat
liat apa ada salah satu yang punya sinyal lebih kuat dari yang lain.

Pakai LightGBM aja (paling konsisten menang di percobaan sebelumnya, dan
cepat) - fokusnya di sini cari horizon yang cocok, bukan bandingin
algoritma lagi (itu udah kita lakuin).

Validasi SELALU pakai non-overlapping (independen) sejak awal, biar
langsung dapet angka yang jujur - nggak perlu 2 tahap kayak sebelumnya.

Cara pakai:
    python horizon_sweep.py
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import lightgbm as lgb

INPUT_FILE = "btc_dataset_indicators.csv"
LOOKBACK = 10
HORIZONS_MINUTES = [1, 3, 5, 10, 15, 30, 60]


def build_features(df: pd.DataFrame, horizon: int):
    df = df.copy()
    df = df.sort_values("timestamp_ms").reset_index(drop=True)

    for lag in range(1, LOOKBACK + 1):
        df[f"lag_{lag}"] = df["close"].diff(lag)

    df["sma_gap"] = df["sma_fast"] - df["sma_slow"]
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    bb_range = df["bb_upper"] - df["bb_lower"]
    df["bb_percent"] = (df["close"] - df["bb_lower"]) / bb_range.replace(0, np.nan)
    df["bb_width"] = bb_range

    df["target"] = (df["close"].shift(-horizon) > df["close"]).astype(int)

    lag_cols = [c for c in df.columns if c.startswith("lag_")]
    indicator_cols = [
        "sma_gap", "rsi", "macd_hist", "bb_percent", "bb_width",
        "stoch_k", "stoch_d", "williams_r", "roc",
    ]
    required_cols = lag_cols + indicator_cols + ["target"]
    df = df.dropna(subset=required_cols).reset_index(drop=True)
    return df, lag_cols + indicator_cols


def test_horizon(df_raw, horizon):
    feat_df, feature_cols = build_features(df_raw, horizon)
    if len(feat_df) < 200:
        return None

    X = feat_df[feature_cols]
    y = feat_df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = lgb.LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                random_state=42, verbose=-1, class_weight="balanced")
    model.fit(X_train, y_train)

    # Validasi non-overlapping dari awal
    non_overlap_idx = np.arange(0, len(X_test), horizon)
    X_test_no = X_test.iloc[non_overlap_idx]
    y_test_no = y_test.to_numpy()[non_overlap_idx]

    if len(y_test_no) < 20:
        return {"horizon": horizon, "acc": None, "n": len(y_test_no), "note": "sample terlalu kecil"}

    pred_no = model.predict(X_test_no)
    acc = accuracy_score(y_test_no, pred_no)

    # confidence >= 0.55 juga, buat konteks tambahan
    proba_no = model.predict_proba(X_test_no)
    max_proba_no = proba_no.max(axis=1)
    pred_class_no = proba_no.argmax(axis=1)
    mask = max_proba_no >= 0.55
    if mask.sum() >= 20:
        acc_conf = (pred_class_no[mask] == y_test_no[mask]).mean()
        n_conf = int(mask.sum())
    else:
        acc_conf, n_conf = None, int(mask.sum())

    return {
        "horizon": horizon, "acc": acc, "n": len(y_test_no),
        "acc_conf55": acc_conf, "n_conf55": n_conf,
    }


def main():
    df_raw = pd.read_csv(INPUT_FILE)
    print(f"Data tersedia: {len(df_raw)} baris (candle 1 menit)\n")
    print("Testing berbagai horizon prediksi (validasi non-overlapping dari awal)...\n")

    results = []
    for h in HORIZONS_MINUTES:
        r = test_horizon(df_raw, h)
        if r is None:
            print(f"  Horizon {h:3d} menit: data nggak cukup, skip.")
            continue
        results.append(r)
        if r["acc"] is None:
            print(f"  Horizon {h:3d} menit: sample non-overlap cuma {r['n']}, terlalu kecil buat dipercaya.")
        else:
            conf_str = f"{r['acc_conf55']:.3f} (n={r['n_conf55']})" if r["acc_conf55"] else f"-- (n={r['n_conf55']}, kurang)"
            print(f"  Horizon {h:3d} menit: akurasi {r['acc']:.3f} (n={r['n']}) | conf>=0.55: {conf_str}")

    print("\n" + "=" * 60)
    print("RINGKASAN")
    print("=" * 60)
    valid = [r for r in results if r["acc"] is not None]
    if not valid:
        print("Nggak ada horizon dengan sample cukup buat disimpulkan.")
        return

    best = max(valid, key=lambda r: r["acc"])
    print(f"Horizon dengan akurasi tertinggi: {best['horizon']} menit ({best['acc']:.3f}, n={best['n']})")
    print("\nPENTING: kalau SEMUA horizon hasilnya muter di sekitar 0.48-0.53,")
    print("itu tandanya bukan soal horizon yang salah pilih - random walk-nya")
    print("konsisten di semua timeframe pendek yang dicoba. Kalau ada 1-2")
    print("horizon yang menonjol jauh di atas yang lain, itu baru worth")
    print("diselidiki lebih lanjut (tapi hati-hati - dengan 7 horizon dicoba,")
    print("ada kemungkinan salah satu 'menonjol' cuma karena kebetulan statistik,")
    print("bukan sinyal asli - butuh validasi ulang di data baru sebelum yakin).")


if __name__ == "__main__":
    main()
