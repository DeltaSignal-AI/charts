"""
Extract NK225 options chart data from combined CSV files and save as JSON.

Usage:
    python3 extract_data.py

Reads:  /Users/ken/workspace/Private/deltasignal-data/data/nikkei225_options/*_combined.csv
Writes: ./data/YYYY-MM-DD.json (per date)
        ./data/dates.json      (sorted list of available dates)
"""

import json
import math
from pathlib import Path

import pandas as pd

DATA_DIR = Path("/Users/ken/workspace/Private/deltasignal-data/data/nikkei225_options")
OUT_DIR  = Path(__file__).parent / "data"
OUT_DIR.mkdir(exist_ok=True)

MONTH_NAMES = {
    "01": "Jan", "02": "Feb", "03": "Mar", "04": "Apr",
    "05": "May", "06": "Jun", "07": "Jul", "08": "Aug",
    "09": "Sep", "10": "Oct", "11": "Nov", "12": "Dec",
}


def bs_gamma(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return 0.0
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        nd1 = math.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)
        return nd1 / (S * sigma * math.sqrt(T))
    except (ValueError, ZeroDivisionError):
        return 0.0


def pick_monthly_expiries(df, date_yyyymm: str):
    """Return dict {expiry_str: label} for 3 nearest monthly (6-digit) expiries."""
    six_digit = sorted(
        {str(e) for e in df["expiry"].unique() if len(str(int(e))) == 6},
        key=int,
    )
    # Only expiries on or after current month
    future = [e for e in six_digit if e >= date_yyyymm][:3]
    result = {}
    for e in future:
        label = MONTH_NAMES.get(e[4:6], e[4:6])
        result[e] = label
    return result


def extract(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)

    # Date label
    if "date" in df.columns:
        date_label = str(df["date"].iloc[0])[:10]
    else:
        stem = csv_path.stem
        d = stem[:8]
        date_label = f"{d[:4]}-{d[4:6]}-{d[6:8]}"

    date_yyyymm = date_label[:7].replace("-", "")  # e.g. "202608"

    SPOT = round(float(df["underlying_price"].median()), 2)
    r_rate = float(df["interest_rate"].mean()) / 100.0

    MONTHLY = pick_monthly_expiries(df, date_yyyymm)
    MONTHLY_INT = {int(k): v for k, v in MONTHLY.items()}

    # ── OI Profile ─────────────────────────────────────────────────────────
    oi_by_expiry = {}
    for exp_int, _ in MONTHLY_INT.items():
        exp_str = str(exp_int)
        oi_by_expiry[exp_str] = {"call": {}, "put": {}}
        mask = (df["expiry"] == exp_int) & (df["strike"].between(64000, 78000))
        for _, row in df[mask].iterrows():
            s = str(int(float(row["strike"])))
            oi = float(row["oi"])
            side = row["side"]
            oi_by_expiry[exp_str][side][s] = oi_by_expiry[exp_str][side].get(s, 0) + oi

    all_oi_strikes = sorted({
        int(s) for s in df["strike"].unique()
        if 64000 <= s <= 78000 and int(s) % 250 == 0
    })

    total_call_oi = {}
    total_put_oi  = {}
    for exp_str in oi_by_expiry:
        for k, v in oi_by_expiry[exp_str]["call"].items():
            total_call_oi[k] = total_call_oi.get(k, 0) + v
        for k, v in oi_by_expiry[exp_str]["put"].items():
            total_put_oi[k] = total_put_oi.get(k, 0) + v

    top3_call = sorted(total_call_oi.items(), key=lambda x: -x[1])[:3]
    top3_put  = sorted(total_put_oi.items(),  key=lambda x: -x[1])[:3]

    oi_data = {
        "spot":         SPOT,
        "strikes":      all_oi_strikes,
        "expirations":  list(MONTHLY.keys()),
        "expiry_labels": MONTHLY,
        "call_oi":      {e: oi_by_expiry[e]["call"] for e in MONTHLY},
        "put_oi":       {e: oi_by_expiry[e]["put"]  for e in MONTHLY},
        "top3_call":    [{"strike": int(s), "oi": v} for s, v in top3_call],
        "top3_put":     [{"strike": int(s), "oi": v} for s, v in top3_put],
    }

    # ── GEX ────────────────────────────────────────────────────────────────
    gex_mask = (
        df["expiry"].isin(list(MONTHLY_INT.keys())) &
        df["strike"].between(64000, 78000) &
        (df["iv"] > 0) & (df["iv"] < 200) &
        (df["oi"] > 0)
    )
    gex_df = df[gex_mask].copy()
    gex_by_strike = {}
    for _, row in gex_df.iterrows():
        S     = float(row["underlying_price"])
        K     = float(row["strike"])
        sigma = float(row["iv"]) / 100.0
        T     = float(row["dte"]) / 365.0
        oi_val = float(row["oi"])
        gamma = bs_gamma(S, K, T, r_rate, sigma)
        gex   = gamma * oi_val * S * S / 1e6
        sign  = 1.0 if row["side"] == "call" else -1.0
        key   = str(int(K))
        gex_by_strike[key] = gex_by_strike.get(key, 0) + sign * gex

    sorted_keys = sorted(gex_by_strike.keys(), key=int)
    flip_point  = SPOT
    best_dist   = float("inf")
    for i in range(len(sorted_keys) - 1):
        k1, k2 = sorted_keys[i], sorted_keys[i + 1]
        g1, g2 = gex_by_strike[k1], gex_by_strike[k2]
        if g1 * g2 < 0:
            frac  = abs(g1) / (abs(g1) + abs(g2))
            cross = int(k1) + frac * (int(k2) - int(k1))
            dist  = abs(cross - SPOT)
            if dist < best_dist:
                best_dist  = dist
                flip_point = cross

    gex_data = {
        "spot":          SPOT,
        "gex_by_strike": {k: round(gex_by_strike[k], 6) for k in sorted_keys},
        "flip_point":    round(flip_point, 1),
    }

    # ── IV Smile ────────────────────────────────────────────────────────────
    iv_smile = {}
    for exp_int, _ in MONTHLY_INT.items():
        exp_str = str(exp_int)
        iv_smile[exp_str] = {"call": {}, "put": {}}
        mask = (
            (df["expiry"] == exp_int) &
            df["strike"].between(64000, 78000) &
            (df["iv"] > 0) & (df["iv"] < 150)
        )
        for _, row in df[mask].iterrows():
            s    = str(int(float(row["strike"])))
            iv   = float(row["iv"])
            side = row["side"]
            if s not in iv_smile[exp_str][side]:
                iv_smile[exp_str][side][s] = round(iv, 4)

    iv_data = {
        "spot":          SPOT,
        "expirations":   list(MONTHLY.keys()),
        "expiry_labels": MONTHLY,
        "call_iv":       {e: iv_smile[e]["call"] for e in MONTHLY},
        "put_iv":        {e: iv_smile[e]["put"]  for e in MONTHLY},
    }

    # ── Volume ──────────────────────────────────────────────────────────────
    vol_call = {}
    vol_put  = {}
    for _, row in df.iterrows():
        s   = float(row["strike"])
        vol = float(row["volume"])
        if not (55000 <= s <= 85000 and s % 500 == 0 and vol > 0):
            continue
        key = str(int(s))
        if row["side"] == "call":
            vol_call[key] = vol_call.get(key, 0) + vol
        else:
            vol_put[key]  = vol_put.get(key, 0)  + vol

    vol_strikes = sorted({
        int(s) for s in df["strike"].unique()
        if 55000 <= s <= 85000 and s % 500 == 0
    })
    vol_totals = {
        str(s): vol_call.get(str(s), 0) + vol_put.get(str(s), 0)
        for s in vol_strikes
    }
    top3_vol = sorted(vol_totals.items(), key=lambda x: -x[1])[:3]

    vol_data = {
        "spot":     SPOT,
        "strikes":  vol_strikes,
        "call_vol": {str(s): vol_call.get(str(s), 0) for s in vol_strikes},
        "put_vol":  {str(s): vol_put.get(str(s), 0)  for s in vol_strikes},
        "top3_vol": [{"strike": int(s), "vol": v} for s, v in top3_vol],
    }

    # ── Summary stats ────────────────────────────────────────────────────────
    # PCR from ALL options (not just 3 monthly) to match article figures
    all_call_oi_sum = float(df[df["side"] == "call"]["oi"].sum())
    all_put_oi_sum  = float(df[df["side"] == "put"]["oi"].sum())
    pcr = round(all_put_oi_sum / all_call_oi_sum, 2) if all_call_oi_sum > 0 else 0.0
    total_call_oi_sum = sum(total_call_oi.values())
    total_put_oi_sum  = sum(total_put_oi.values())

    # ATM IV (nearest strike to spot, nearest expiry = first in list)
    atm_call_iv = atm_put_iv = None
    pref_exp = list(MONTHLY.keys())[0] if MONTHLY else None  # nearest expiry
    if pref_exp:
        strikes_sorted = sorted(iv_smile[pref_exp]["call"].keys(), key=lambda x: abs(int(x) - SPOT))
        if strikes_sorted:
            atm_k = strikes_sorted[0]
            atm_call_iv = iv_smile[pref_exp]["call"].get(atm_k)
            atm_put_iv  = iv_smile[pref_exp]["put"].get(atm_k)

    stats = {
        "date":        date_label,
        "spot":        SPOT,
        "pcr":         pcr,
        "total_call_oi": int(all_call_oi_sum),
        "total_put_oi":  int(all_put_oi_sum),
        "atm_call_iv": atm_call_iv,
        "atm_put_iv":  atm_put_iv,
        "gex_flip":    round(flip_point, 0),
        "expirations": list(MONTHLY.keys()),
        "expiry_labels": MONTHLY,
    }

    return {
        "stats":      stats,
        "oi_profile": oi_data,
        "gex":        gex_data,
        "iv_smile":   iv_data,
        "volume":     vol_data,
    }


def main():
    csv_files = sorted(DATA_DIR.glob("*_combined.csv"))
    dates = []

    for csv_path in csv_files:
        stem      = csv_path.stem
        date_str  = stem[:8]
        date_label = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        out_path  = OUT_DIR / f"{date_label}.json"

        if out_path.exists():
            print(f"  skip {date_label} (already extracted)")
            dates.append(date_label)
            continue

        print(f"  extracting {date_label} ...")
        try:
            data = extract(csv_path)
            out_path.write_text(json.dumps(data, ensure_ascii=False))
            dates.append(date_label)
            print(f"    → spot ¥{data['stats']['spot']:,.0f}, PCR {data['stats']['pcr']}")
        except Exception as e:
            print(f"    ERROR: {e}")

    dates_out = sorted(dates, reverse=True)  # newest first
    (OUT_DIR / "dates.json").write_text(json.dumps(dates_out, ensure_ascii=False))
    print(f"\nDone. {len(dates)} dates written to {OUT_DIR}/dates.json")


if __name__ == "__main__":
    main()
