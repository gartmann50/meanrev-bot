#!/usr/bin/env python3
import os, sys, time, json, argparse, requests, pandas as pd
from datetime import date
from zoneinfo import ZoneInfo

API = "https://api.polygon.io"

def req(path, params, key):
    params = dict(params or {}); params["apiKey"] = key
    r = requests.get(API + path, params=params, timeout=40)
    if r.status_code == 429: time.sleep(1.2); r = requests.get(API + path, params=params, timeout=40)
    r.raise_for_status(); return r.json()

def to_df(results):
    if not results: return pd.DataFrame(columns=["ts","Open","High","Low","Close","Volume"])
    df = pd.DataFrame(results).rename(columns={"t":"ts","o":"Open","h":"High","l":"Low","c":"Close","v":"Volume"})
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df[["ts","Open","High","Low","Close","Volume"]]

def rth_only(df):
    if df.empty: return df
    ny = ZoneInfo("America/New_York")
    loc = df["ts"].dt.tz_convert(ny).dt.time
    import datetime as dt
    s = dt.time(9,30); e = dt.time(16,0)
    return df[(loc >= s) & (loc <= e)]

def write_parquet(df, path):
    import pyarrow as pa, pyarrow.parquet as pq
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df), path, compression="snappy")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols-file", required=True)   # CSV with 'symbol'
    ap.add_argument("--start", required=True)          # e.g. 2024-01-01
    ap.add_argument("--end", default=date.today().isoformat())
    ap.add_argument("--out-dir", default="data_5m")
    ap.add_argument("--rth-only", action="store_true")
    ap.add_argument("--api-key", default=os.getenv("POLYGON_KEY"))
    a = ap.parse_args()
    if not a.api_key: sys.exit("Set POLYGON_KEY or pass --api-key")

    syms = pd.read_csv(a.symbols_file)["symbol"].astype(str).tolist()
    manifest = {"created": date.today().isoformat(), "start": a.start, "end": a.end, "symbols": []}
    for i, sym in enumerate(syms, 1):
        try:
            data = req(f"/v2/aggs/ticker/{sym}/range/5/minute/{a.start}/{a.end}",
                       {"adjusted":"true","limit":50000,"sort":"asc"}, a.api_key)
            df = to_df(data.get("results", []))
            if a.rth_only: df = rth_only(df)
            write_parquet(df, os.path.join(a.out_dir, f"{sym}.parquet"))
            manifest["symbols"].append({"symbol": sym, "rows": int(len(df))})
            if i % 25 == 0: print(f"{i}/{len(syms)} saved…")
        except requests.HTTPError as e:
            print(f"[skip] {sym}: {e}")
        time.sleep(0.08)
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "manifest_5m.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved {len(manifest['symbols'])}/{len(syms)} symbols → {a.out_dir}")

if __name__ == "__main__": main()
