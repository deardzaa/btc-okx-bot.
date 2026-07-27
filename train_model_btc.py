"""
Latih model prediksi arah BTC dari btc_dataset_indicators.csv.
Metodologi SAMA PERSIS kayak train_model.py di proyek Deriv dulu, biar
hasilnya bisa dibandingin apple-to-apple:
- 3 algoritma dibandingin (Random Forest, Gradient Boosting, LightGBM)
- Walk-forward split (time-series, TIDAK di-shuffle)
- Analisis confidence threshold
- Validasi ulang pakai sample non-overlapping (biar nggak ketipu backtest)

PENTING: dari uji randomness sebelumnya, autokorelasi BTC di timeframe 1
menit magnitude-nya mirip R_100 (amat kecil). Jangan berharap breakthrough
besar - anggap ini validasi empiris dari apa yang udah kita duga.

Cara pakai:
    python train_model_btc.py
"""
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight
import lightgbm as lgb

INPUT_FILE = "btc_dataset_indicators.csv"
MODEL_PATH = "model_btc.pkl"
PREDICTION_HORIZON = 5  # prediksi arah 5 menit ke depan (5 candle 1-menit)
LOOKBACK = 10  # berapa lag harga dipakai sebagai fitur


def build_features(df: pd.DataFrame):
    df = df.copy()
    df = df.sort_values("timestamp_ms").reset_index(drop=True)

    for lag in range(1, LOOKBACK + 1):
        df[f"lag_{lag}"] = df["close"].diff(lag)

    df["sma_gap"] = df["sma_fast"] - df["sma_slow"]
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    bb_range = df["bb_upper"] - df["bb_lower"]
    df["bb_percent"] = (df["close"] - df["bb_lower"]) / bb_range.replace(0, np.nan)
    df["bb_width"] = bb_range

    # target: arah harga PREDICTION_HORIZON candle ke depan
    df["target"] = (df["close"].shift(-PREDICTION_HORIZON) > df["close"]).astype(int)

    lag_cols = [c for c in df.columns if c.startswith("lag_")]
    indicator_cols = [
        "sma_gap", "rsi", "macd_hist", "bb_percent", "bb_width",
        "stoch_k", "stoch_d", "williams_r", "roc",
    ]
    required_cols = lag_cols + indicator_cols + ["target"]
    df = df.dropna(subset=required_cols).reset_index(drop=True)
    return df, lag_cols + indicator_cols


def train_and_evaluate(name, model, X_train, y_train, X_test, y_test, sample_weight=None):
    if sample_weight is not None:
        model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\n--- {name} ---")
    print(f"Akurasi di data test (out-of-sample): {acc:.3f}")
    print(classification_report(y_test, preds, target_names=["DOWN", "UP"]))
    return model, acc


def main():
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Belum ada {INPUT_FILE}. Jalankan compute_indicators.py dulu.")
        sys.exit(1)

    print(f"Baris data mentah: {len(df)}")
    feat_df, feature_cols = build_features(df)
    if len(feat_df) < 50:
        print("Data valid kurang dari 50 baris, nggak cukup buat training.")
        sys.exit(1)

    X = feat_df[feature_cols]
    y = feat_df["target"]

    down_pct = (y == 0).mean() * 100
    up_pct = (y == 1).mean() * 100
    print(f"Baris siap training: {len(feat_df)}")
    print(f"Distribusi target: DOWN={down_pct:.1f}% | UP={up_pct:.1f}% "
          f"(horizon: {PREDICTION_HORIZON} menit ke depan)")
    print(f"Fitur yang dipakai ({len(feature_cols)}): {feature_cols}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False  # time-series, jangan di-shuffle
    )

    print("\n" + "=" * 60)
    print("PERBANDINGAN ALGORITMA")
    print("=" * 60)

    rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, class_weight="balanced")
    rf, rf_acc = train_and_evaluate("Random Forest", rf, X_train, y_train, X_test, y_test)

    gb_weights = compute_sample_weight("balanced", y_train)
    gb = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.1, random_state=42)
    gb, gb_acc = train_and_evaluate("Gradient Boosting", gb, X_train, y_train, X_test, y_test,
                                     sample_weight=gb_weights)

    lgbm = lgb.LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42,
                               verbose=-1, class_weight="balanced")
    lgbm, lgbm_acc = train_and_evaluate("LightGBM", lgbm, X_train, y_train, X_test, y_test)

    candidates = [("Random Forest", rf, rf_acc), ("Gradient Boosting", gb, gb_acc), ("LightGBM", lgbm, lgbm_acc)]
    model_name, model, best_acc = max(candidates, key=lambda c: c[2])
    other_accs = sorted([c[2] for c in candidates if c[0] != model_name], reverse=True)
    print(f"\n>>> Model terpilih: {model_name} (akurasi {best_acc:.3f} "
          f"vs {other_accs[0]:.3f} vs {other_accs[1]:.3f})")
    print("(0.50 = sama aja kayak nebak koin.)")

    print("\n" + "=" * 60)
    print(f"ANALISIS CONFIDENCE THRESHOLD ({model_name})")
    print("=" * 60)
    proba = model.predict_proba(X_test)
    max_proba = proba.max(axis=1)
    pred_class = proba.argmax(axis=1)
    y_test_arr = y_test.to_numpy()

    print(f"{'Threshold':>10} | {'Akurasi':>8} | {'Coverage':>9} | {'Jml Trade':>10}")
    print("-" * 50)
    for threshold in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        mask = max_proba >= threshold
        n_selected = mask.sum()
        if n_selected == 0:
            print(f"{threshold:>10.2f} | {'--':>8} | {'0.0%':>9} | {0:>10}")
            continue
        selective_acc = (pred_class[mask] == y_test_arr[mask]).mean()
        coverage = n_selected / len(y_test_arr) * 100
        print(f"{threshold:>10.2f} | {selective_acc:>8.3f} | {coverage:>8.1f}% | {n_selected:>10}")

    print("\n" + "=" * 60)
    print("VALIDASI ULANG: cuma pakai sample independen (non-overlapping)")
    print("=" * 60)
    print(f"(Skip tiap {PREDICTION_HORIZON} baris biar nggak ada window yang nyerempet)")
    non_overlap_idx = np.arange(0, len(X_test), PREDICTION_HORIZON)
    y_test_no = y_test_arr[non_overlap_idx]
    proba_no = model.predict_proba(X_test.iloc[non_overlap_idx])
    max_proba_no = proba_no.max(axis=1)
    pred_class_no = proba_no.argmax(axis=1)

    print(f"{'Threshold':>10} | {'Akurasi':>8} | {'Jml Trade':>10}")
    print("-" * 36)
    for threshold in [0.50, 0.55, 0.60, 0.65]:
        mask = max_proba_no >= threshold
        n_selected = mask.sum()
        if n_selected == 0:
            print(f"{threshold:>10.2f} | {'--':>8} | {0:>10}")
            continue
        selective_acc = (pred_class_no[mask] == y_test_no[mask]).mean()
        print(f"{threshold:>10.2f} | {selective_acc:>8.3f} | {n_selected:>10}")

    print("\nBandingkan angka non-overlapping ini sama tabel confidence threshold")
    print("di atas. Kalau jauh lebih 'biasa aja', itu tanda overfitting -")
    print("sama persis pola yang kita temuin di R_100 dulu.")

    joblib.dump({"model": model, "feature_cols": feature_cols, "model_name": model_name}, MODEL_PATH)
    print(f"\nModel ({model_name}) disimpan ke {MODEL_PATH}")


if __name__ == "__main__":
    main()
