"""
Uji statistik formal: apakah harga BTC (dari btc_dataset.csv) punya
autokorelasi/pola asli, atau mirip random walk - metodologi sama persis
kayak analyze_randomness.py di proyek Deriv dulu (autocorrelation,
runs test, Ljung-Box), biar hasilnya bisa dibandingin langsung.

Cara pakai:
    python analyze_randomness_btc.py
"""
import pandas as pd
import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox

INPUT_FILE = "btc_dataset.csv"


def runs_test(returns):
    signs = np.sign(returns)
    signs = signs[signs != 0]
    n = len(signs)
    n_pos = (signs > 0).sum()
    n_neg = (signs < 0).sum()

    runs = 1
    for i in range(1, len(signs)):
        if signs[i] != signs[i - 1]:
            runs += 1

    expected_runs = (2 * n_pos * n_neg) / n + 1
    var_runs = (2 * n_pos * n_neg * (2 * n_pos * n_neg - n)) / (n ** 2 * (n - 1))
    z = (runs - expected_runs) / np.sqrt(var_runs) if var_runs > 0 else 0
    p_value = 2 * (1 - abs(np.tanh(z)))  # aproksimasi sederhana, dipakai relatif
    from scipy.stats import norm
    p_value = 2 * (1 - norm.cdf(abs(z)))
    return z, p_value


def main():
    df = pd.read_csv(INPUT_FILE)
    df = df.sort_values("timestamp_ms").reset_index(drop=True)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"]).reset_index(drop=True)

    prices = df["close"]
    returns = prices.diff().dropna()

    print(f"Menganalisis {len(prices)} candle harga BTC (1 menit)...\n")

    print("=" * 60)
    print("1. AUTOCORRELATION (perubahan harga vs lag sebelumnya)")
    print("=" * 60)
    for lag in range(1, 11):
        ac = returns.autocorr(lag=lag)
        print(f"  Lag {lag:2d}: {ac:+.4f}")
    print("\n  Interpretasi: kalau random walk murni, semua nilai ini harusnya")
    print("  deket 0 (dalam rentang -0.05 sampai +0.05 buat data segini banyak).\n")

    print("=" * 60)
    print("2. RUNS TEST (pola urutan naik/turun)")
    print("=" * 60)
    z, p = runs_test(returns.to_numpy())
    print(f"  Z-score: {z:.4f} | p-value: {p:.4f}")
    if p < 0.05:
        print("  --> SIGNIFIKAN (p < 0.05): urutan naik/turun BUKAN murni random,")
        print("      ada pola berurutan (misal cenderung 'nempel' atau 'gantian').")
    else:
        print("  --> TIDAK signifikan (p >= 0.05): urutan naik/turun konsisten")
        print("      dengan random walk.")

    print("\n" + "=" * 60)
    print("3. LJUNG-BOX TEST (autokorelasi gabungan di lag 1-10)")
    print("=" * 60)
    lb_result = acorr_ljungbox(returns, lags=[10], return_df=True)
    lb_pvalue = lb_result["lb_pvalue"].iloc[0]
    print(f"  p-value: {lb_pvalue:.4f}")
    if lb_pvalue < 0.05:
        print("  --> SIGNIFIKAN (p < 0.05): ADA autokorelasi yang nggak bisa")
        print("      dijelaskan kebetulan. Ini sinyal paling kuat kalau ada pola asli.")
    else:
        print("  --> TIDAK signifikan (p >= 0.05): konsisten dengan random walk,")
        print("      nggak ada bukti kuat pola asli di data ini.")

    print("\n" + "=" * 60)
    print("KESIMPULAN")
    print("=" * 60)
    print("Bandingkan hasil ini sama analyze_randomness.py punya R_100 dulu:")
    print("R_100 - Lag 1 autocorr: +0.0198, Ljung-Box p-value: 0.0000 (signifikan")
    print("tapi magnitude sangat kecil, ~0.02).")
    print("\nKalau BTC di sini autokorelasinya JAUH lebih besar dari R_100 (misal")
    print("di atas 0.05-0.1), itu indikasi kuat BTC punya struktur harga yang")
    print("lebih 'nyata' buat dipelajari model, bukan cuma sinyal tipis kayak")
    print("synthetic index. Kalau mirip-mirip aja, artinya BTC di timeframe 1")
    print("menit juga mendekati random walk buat trading jangka pendek.")


if __name__ == "__main__":
    main()
