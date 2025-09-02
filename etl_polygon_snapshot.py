#!/usr/bin/env python3
import os, sys, time, json, csv, math, zipfile, hashlib, argparse, requests
from datetime import date

API = "https://api.polygon.io"
KEY = os.getenv("POLYGON_KEY")  # set this in your shell

def req(path, params=None):
    if not KEY: raise SystemExit("Set POLYGON_KEY first")
    params = dict(params or {})
    params["apiKey"] = KEY
    r = requests.get(API + path, params=params, timeout=30)
    if r.status_code == 429:
        time.sleep(1.2); r = requests.get(API + path, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def fetch_universe(locale="us", market="stocks"):
    """Reference tickers: include delisted; filter to common stock, USD, no OTC."""
    out = []
    url = "/v3/reference/tickers"
    params = {
        "market": market, "active": "false", "limit": 1000,
        "locale": locale, "currency": "USD",
        "type": "CS"  # common stock
    }
    next_url = None
    while True:
        data = req(next_url or url, params if not next_url else None)
        for t in data.get("results", []):
            sym = t.get("ticker")
            exch = (t.get("primary_exchange") or "").upper()
            if sym is None: continue
            # crude OTC filter
            if "OTC" in exch or exch in {"OTC", "PINX"}: continue
            out.append({
                "symbol": sym,
                "name": t.get("name",""),
                "list_date": t.get("list_date",""),
                "delisted_date": t.get("delisted_utc","")[:10] if t.get("delisted_utc") else "",
                "primary_exchange": exch
            })
        next_url = data.get("next_url")
        if not next_url: break
        # polygon returns absolute next_url; keep only the path part
        next_url = next_url.replace(API, "")
        time.sleep(0.12)
    # simple de-dupe
    uniq = {r["symbol"]: r for r in out}
    return list(uniq.values())

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def dl_bars(symbol, out_csv, start="2006-01-01", end=None):
    end = end or date.today().isoformat()
    # polygon aggregates: adjusted=true, daily bars (1 day)
    path = f"/v2/aggs/ticker/{symbol}/range/1/day/{start}/{end}"
    data = req(path, {"adjusted": "true", "limit": 50000})
    results = data.get("results") or []
    if not results: return 0
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Date","Open","High","Low","Close","Volume"])
        for r in results:
            # t is ms since epoch in UTC; polygon returns OHLC in raw floats
            from datetime import datetime, timezone
            dt = datetime.fromtimestamp(r["t"]/1000, tz=timezone.utc).date().isoformat()
            w.writerow([dt, r["o"], r["h"], r["l"], r["c"], r.get("v",0)])
    return len(results)

def sha256_file(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser(description="Polygon EOD snapshot → CSVs + zip")
    ap.add_argument("--data-dir", default="stock_data_400")
    ap.add_argument("--start", default="2006-01-01")
    ap.add_argument("--max-symbols", type=int, default=0, help="0=all")
    ap.add_argument("--universe-out", default="allowlist.csv")
    ap.add_argument("--zip-out", default="stock_data_400.zip")
    args = ap.parse_args()

    ensure_dir(args.data_dir)

    print("Fetching universe…")
    uni = fetch_universe()
    # keep common-sense universe size; adjust if you want all
    if args.max_symbols and len(uni) > args.max_symbols:
        uni = uni[:args.max_symbols]
    with open(args.universe_out,"w",newline="") as f:
        w=csv.DictWriter(f, fieldnames=uni[0].keys()); w.writeheader(); w.writerows(uni)
    print(f"Universe size: {len(uni)} → {args.universe_out}")

    # Download bars
    n_ok = 0
    for i,row in enumerate(uni,1):
        sym=row["symbol"]
        out_csv=os.path.join(args.data_dir, f"{sym}.csv")
        if os.path.exists(out_csv): 
            n_ok += 1; 
            if i%200==0: print(f"{i}/{len(uni)} (cached)"); 
            continue
        try:
            n = dl_bars(sym, out_csv, start=args.start)
            n_ok += 1
        except requests.HTTPError as e:
            # skip symbols that error (suspended, symbol issues)
            print(f"[skip] {sym}: {e}")
            continue
        if i % 50 == 0:
            print(f"{i}/{len(uni)} downloaded…")
        time.sleep(0.08)  # be gentle on rate limits

    # Manifest + zip
    manifest = {
        "snapshot_date": date.today().isoformat(),
        "symbols": len(uni),
        "files": []
    }
    for fn in os.listdir(args.data_dir):
        if not fn.endswith(".csv"): continue
        p=os.path.join(args.data_dir, fn)
        manifest["files"].append({"file": fn, "sha256": sha256_file(p)})
    with open("dataset_manifest.json","w") as f:
        json.dump(manifest, f, indent=2)

    with zipfile.ZipFile(args.zip_out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write("dataset_manifest.json")
        for fn in os.listdir(args.data_dir):
            if fn.endswith(".csv"):
                z.write(os.path.join(args.data_dir, fn), arcname=os.path.join(os.path.basename(args.data_dir), fn))
    print(f"Zip written: {args.zip_out}  (files: {len(manifest['files'])})")

if __name__ == "__main__":
    main()
