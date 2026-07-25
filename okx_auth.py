"""
Helper buat bikin signature HMAC-SHA256 yang dibutuhin OKX API v5.

Beda total dari Deriv (yang cuma butuh 1 token doang). OKX butuh
signature yang dihitung ulang tiap request, dari kombinasi:
  timestamp + method + request_path + body

Cara pakai:
    from okx_auth import get_auth_headers
    headers = get_auth_headers("GET", "/api/v5/account/balance", "")
"""
import base64
import hashlib
import hmac
import os
from datetime import datetime, timezone

OKX_API_KEY = os.environ.get("OKX_API_KEY", "")
OKX_SECRET_KEY = os.environ.get("OKX_SECRET_KEY", "")
OKX_PASSPHRASE = os.environ.get("OKX_PASSPHRASE", "")

# True = akun demo (wajib buat sekarang, JANGAN ganti ke False sebelum
# beneran siap & sadar risikonya - False artinya akun live/uang beneran)
DEMO_MODE = True


def _get_timestamp() -> str:
    # OKX butuh format ISO8601 dengan milidetik, contoh: 2026-07-25T10:30:00.000Z
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") + \
        f"{datetime.now(timezone.utc).microsecond // 1000:03d}Z"


def _sign(timestamp: str, method: str, request_path: str, body: str = "") -> str:
    message = f"{timestamp}{method}{request_path}{body}"
    mac = hmac.new(
        OKX_SECRET_KEY.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256,
    )
    return base64.b64encode(mac.digest()).decode("utf-8")


def get_auth_headers(method: str, request_path: str, body: str = "") -> dict:
    """
    Bikin header lengkap buat request yang butuh autentikasi
    (contoh: cek saldo, pasang order). Buat endpoint publik (contoh:
    harga ticker), header ini nggak perlu dipakai sama sekali.
    """
    if not (OKX_API_KEY and OKX_SECRET_KEY and OKX_PASSPHRASE):
        raise ValueError(
            "OKX_API_KEY / OKX_SECRET_KEY / OKX_PASSPHRASE belum di-set. "
            "Set dulu sebagai environment variable (GitHub Secrets)."
        )

    timestamp = _get_timestamp()
    sign = _sign(timestamp, method, request_path, body)

    headers = {
        "OK-ACCESS-KEY": OKX_API_KEY,
        "OK-ACCESS-SIGN": sign,
        "OK-ACCESS-PASSPHRASE": OKX_PASSPHRASE,
        "OK-ACCESS-TIMESTAMP": timestamp,
        "Content-Type": "application/json",
    }
    if DEMO_MODE:
        # WAJIB buat demo trading - kalau lupa, request gagal atau
        # nyasar ke akun live.
        headers["x-simulated-trading"] = "1"
    return headers
