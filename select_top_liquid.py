#!/usr/bin/env python3
import argparse, os, glob, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", required=True)
ap.add_argument("--start", required=True)      # e.g. 2024-01-01
ap.add_argument("--end",   required=True)      # e.g. 2024-12-31
ap.add_argument("--min-days", type=int, default=60)
ap.add_argument("--top",      type=int, default=400)
ap.add_argument("--out",      required=True)
a = ap.parse_args()

rows = []
for p in glob.glob(os.path.join(a.data_dir, "*.csv")):
    sym = os.path.splitext(os.path.basename(p))[0]
    try:
        df = pd.read_csv(p, parse_dates=["Date"])
        if not {"Close","Volume"}.issubset(df.columns): 
            continue
        sub = df[(df["Date"] >= a.start) & (df["Date"] <= a.end)].copy()
        if len(sub) < a.min_days: 
            continue                    # not enough data in window
        if sub["Date"].max() < pd.to_datetime(a.end) - pd.Timedelta(days=10):
            continue                    # didn’t trade near the end
        adv = (sub["Close"] * sub["Volume"]).median()
        rows.append((sym, float(adv)))
    except Exception:
        continue

uni = (pd.DataFrame(rows, columns=["symbol","adv_usd"])
        .sort_values("adv_usd", ascending=False)
        .head(a.top))
uni.to_csv(a.out, index=False)
print(f"Wrote {len(uni)} symbols → {a.out}")
