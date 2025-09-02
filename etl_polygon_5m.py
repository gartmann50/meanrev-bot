#!/usr/bin/env python3
import os, sys, time, json, argparse, requests, pandas as pd
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

API = "https://api.polygon.io"

def req(path, params, key):
    params = dict(params or {})
    params["apiKey"] = key
    r = requests.get(API + path, params=params, timeout=30)
    if r.status_code == 429:
        time.sleep(1.2)
        r = requests.get(API + path, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def to_df_aggs(results):
    if not results: return pd.DataFrame(columns=["ts","Open","High","Low","Close","Volume"])
    df = pd.DataFrame(results)
    df = df.rename(columns={"t":"ts","o":"Open","h":"High","l":"Low","c":"Close","v":"Volume"})
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df[["ts","Open","High","Low","Close","Volume"]]

def rth_filter(df):
    if df.empty: return df
    ny = ZoneInfo("America/New_York")
    dt_local = df["ts"].dt.tz_convert(ny)
    tod = dt_local.dt.time
    # keep 09:30:00–15:55:00 inclusive for 5m bars ending < 16:00 (or keep <= 16:00; adjust as you prefer)
    return df[(tod >= datetime(2000,1,1,9,30).time()) & (tod <= datetime(2000,1,1,16,0).time())]

def write_parquet(df, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        import pyarrow as pa, pyarrow.parquet as pq
        table = pa.Table.from_pandas(df)
        pq.write_table(table, out_path, compression="snappy")
    except Exception:
        # fallback to CSV if pyarrow missing
        df.to_csv(out_path.replace(".parquet",".csv"), index=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols-file", required=True, help="CSV with a 'symbol' column")
    ap.add_argument("--start", required=True)  # e.g., 2024-01-01
    ap.add_argument("--end", default=date.today().isoformat())
    ap.add_argument("--out-dir", default="data_5m")
    ap.add_argument("--rth-only", action="store_true")
    ap.add_argument("--api-key", default=os.getenv("POLYGON_KEY"))
    args = ap.parse_args()

    if not args.api_key:
        sys.exit("Set POLYGON_KEY or pass --api-key")

    syms = pd.read_csv(args.symbols_file)["symbol"].astype(str).tolist()
    done = 0
    manifest = {"created": date.today().isoformat(), "start": args.start, "end": args.end, "symbols": []}

    for i, sym in enumerate(syms, 1):
        try:
            data = req(f"/v2/aggs/ticker/{sym}/range/5/minute/{args.start}/{args.end}",
                       {"adjusted":"true", "limit":50000, "sort":"asc"}, args.api_key)
            df = to_df_aggs(data.get("results", []))
            if args.rth_only:
                df = rth_filter(df)
            out = os.path.join(args.out_dir, f"{sym}.parquet")
            write_parquet(df, out)
            manifest["symbols"].append({"symbol": sym, "rows": int(len(df))})
            done += 1
            if i % 25 == 0: print(f"{i}/{len(syms)} saved…")
        except requests.HTTPError as e:
            print(f"[skip] {sym}: {e}")
        time.sleep(0.08)  # be polite

    with open(os.path.join(args.out_dir, "manifest_5m.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved {done}/{len(syms)} symbols → {args.out_dir}")

if __name__ == "__main__":
    main()
