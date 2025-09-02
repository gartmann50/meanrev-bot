#!/usr/bin/env python3
import os, sys, time, argparse, requests, pandas as pd
from datetime import date, timedelta

API = "https://api.polygon.io"

def get(url, params, key):
    params = dict(params or {}); params["apiKey"] = key
    r = requests.get(url, params=params, timeout=40)
    if r.status_code == 429: time.sleep(1.2); r = requests.get(url, params=params, timeout=40)
    r.raise_for_status(); return r.json()

def fetch_active_common_stocks(key):
    tickers=set(); url=f"{API}/v3/reference/tickers"
    params={"market":"stocks","active":"true","type":"CS","currency":"USD","limit":1000}
    next_url=None
    while True:
        data = get(next_url or url, params if not next_url else None, key)
        for t in data.get("results", []): tickers.add(t["ticker"])
        next_url = data.get("next_url"); 
        if not next_url: break
        next_url = next_url.replace(API,"")
        next_url = API + next_url
        time.sleep(0.12)
    return tickers

def trading_days_back(days):
    # grab ~90 calendar days to cover ~60 trading days
    end = date.today(); start = end - timedelta(days=90)
    d = start
    while d <= end:
        yield d.isoformat()
        d += timedelta(days=1)

def main():
    ap = argparse.ArgumentParser(description="Top-N by median $ volume using Polygon grouped daily")
    ap.add_argument("--days", type=int, default=60, help="target trading days (approx)")
    ap.add_argument("--top",  type=int, default=400)
    ap.add_argument("--min_price", type=float, default=5.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--api-key", default=os.getenv("POLYGON_KEY"))
    args = ap.parse_args()
    if not args.api_key: sys.exit("Set POLYGON_KEY or pass --api-key")

    # 1) Active US common stocks
    cs = fetch_active_common_stocks(args.api_key)

    # 2) Pull grouped daily across ~90 calendar days, accumulate $vol per day
    rows=[]
    for d in trading_days_back(args.days):
        try:
            data = get(f"{API}/v2/aggs/grouped/locale/us/market/stocks/{d}", {"adjusted":"true"}, args.api_key)
        except requests.HTTPError:
            continue
        res = data.get("results") or []
        for r in res:
            sym = r.get("T")
            if sym not in cs: continue
            c = r.get("c"); v = r.get("v"); vw = r.get("vw")
            if not (c and v): continue
            px = vw or c
            if px < args.min_price: continue
            rows.append((sym, d, float(px)*float(v)))

    if not rows:
        pd.DataFrame(columns=["symbol"]).to_csv(args.out, index=False); print("No data"); return

    df = pd.DataFrame(rows, columns=["symbol","date","dollar_volume"])
    # keep last ~60 trading days only
    # (grouped endpoint returns gaps; we just take most recent 60 non-empty dates)
    recent_dates = sorted(df["date"].unique())[-60:]
    df = df[df["date"].isin(recent_dates)]
    uni = (df.groupby("symbol", as_index=False)["dollar_volume"]
             .median()
             .sort_values("dollar_volume", ascending=False)
             .head(args.top)[["symbol"]])
    uni.to_csv(args.out, index=False)
    print(f"Wrote {len(uni)} symbols → {args.out}")

if __name__ == "__main__":
    main()
