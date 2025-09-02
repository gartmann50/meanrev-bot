#!/usr/bin/env python3
import argparse, os, glob, pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--top",  type=int, default=400)
    ap.add_argument("--out",  required=True)
    args = ap.parse_args()

    rows=[]
    for p in glob.glob(os.path.join(args.data_dir, "*.csv")):
        sym = os.path.splitext(os.path.basename(p))[0]
        try:
            df = pd.read_csv(p, parse_dates=["Date"])
            if not {"Close","Volume"}.issubset(df.columns) or len(df)<args.days: continue
            sub = df.tail(args.days).copy()
            adv_usd = (sub["Close"]*sub["Volume"]).median()
            rows.append((sym, float(adv_usd)))
        except Exception:
            continue

    uni = (pd.DataFrame(rows, columns=["symbol","adv_usd"])
             .sort_values("adv_usd", ascending=False)
             .head(args.top))
    uni.to_csv(args.out, index=False)
    print(f"Wrote {len(uni)} symbols → {args.out}")

if __name__ == "__main__":
    main()
