# mr_backtest_5m.py
# Minimal backtester for the simple intraday mean-reversion rules on 5m bars.

import os, math, time, argparse, pathlib, requests, pandas as pd, numpy as np
from datetime import datetime, timedelta, timezone

def alpaca_data_base(): return "https://data.alpaca.markets"
def rsi2(x):
    d=x.diff(); up=d.clip(lower=0).rolling(2).mean(); dn=(-d.clip(upper=0)).rolling(2).mean()
    rs=up/dn.replace(0,np.nan); return 100-100/(1+rs)
def vwap(df): 
    tp=(df.h+df.l+df.c)/3; return (tp*df.v).cumsum()/df.v.cumsum()
def atr(df,n=14):
    tr=np.maximum.reduce([(df.h-df.l).abs().values,(df.h-df.c.shift()).abs().values,(df.l-df.c.shift()).abs().values])
    return pd.Series(tr,index=df.index).rolling(n).mean()

def fetch_bars_5m(sym, start_iso, end_iso, source, alp_key, alp_sec, poly_key, limit=10000):
    """
    Return 5m bars DataFrame with columns: t (UTC), o,h,l,c,v
    source: 'alpaca' or 'polygon'
    """
    if source == "polygon":
        if not poly_key:
            print("No POLYGON_KEY set")
            return None
        # Polygon aggregates API: /v2/aggs/ticker/{sym}/range/5/minute/{from}/{to}
        url = f"https://api.polygon.io/v2/aggs/ticker/{sym}/range/5/minute/{start_iso}/{end_iso}"
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
            "apiKey": poly_key,
        }
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code >= 400:
                print(f"{sym} bars HTTP {r.status_code}: {r.text[:200]}")
                return None
            js = r.json()
            results = js.get("results", [])
            if not results:
                return None
            df = pd.DataFrame(results)[["t","o","h","l","c","v"]].rename(columns=str.lower)
            # Polygon 't' is epoch ms UTC
            df["t"] = pd.to_datetime(df["t"], unit="ms", utc=True)
            return df
        except Exception as e:
            print(f"{sym} polygon fetch error: {e}")
            return None

    else:
        # Alpaca Data API (IEX feed to avoid SIP 403)
        params = {
            "timeframe": "5Min",
            "start": start_iso,
            "end": end_iso,
            "limit": limit,
            "feed": "iex",
        }
        headers = {
            "APCA-API-KEY-ID": alp_key or "",
            "APCA-API-SECRET-KEY": alp_sec or "",
            "accept": "application/json",
            "user-agent": "mr-backtest-5m/1.0",
        }
        try:
            r = requests.get(f"https://data.alpaca.markets/v2/stocks/{sym}/bars",
                             params=params, headers=headers, timeout=30)
            if r.status_code >= 400:
                print(f"{sym} bars HTTP {r.status_code}: {r.text[:200]}")
                return None
            bars = (r.json() or {}).get("bars", [])
            if not bars:
                return None
            df = pd.DataFrame(bars)[["t","o","h","l","c","v"]].rename(columns=str.lower)
            df["t"] = pd.to_datetime(df["t"], utc=True)
            return df
        except Exception as e:
            print(f"{sym} alpaca fetch error: {e}")
            return None

def ny_session_bounds(week_str):
    """Given a week (Monday date, YYYY-MM-DD), return session start/end (Mon 09:30 to Fri 15:55 NY)."""
    # Treat input as Monday
    mon=datetime.fromisoformat(week_str).date()
    fri=mon+timedelta(days=4)
    # Use UTC boundaries for Alpaca queries, NY close ~ 20:00/21:00 UTC
    # We’ll just widen slightly and trim later
    start = datetime(mon.year,mon.month,mon.day,13,25,tzinfo=timezone.utc)  # ~09:25 ET buffer
    end   = datetime(fri.year,fri.month,fri.day,21,10,tzinfo=timezone.utc)  # ~close buffer
    return start, end

def run_backtest(picklist, week, topk, risk_pct, atr_mult, source, key, sec, poly_key):
    start_utc, end_utc = ny_session_bounds(week)
    start_iso = start_utc.isoformat().replace("+00:00","Z")
    end_iso   = end_utc.isoformat().replace("+00:00","Z")

    picks = pd.read_csv(picklist)["symbol"].astype(str).str.upper().head(topk).tolist()
    if not picks: 
        print("No picks in picklist."); 
        return

    # account equity placeholder just for sizing
    eq = 100000.0
    per_trade_risk = max(1.0, eq*risk_pct)

    results=[]
    for sym in picks:
        df = fetch_bars_5m(sym, start_iso, end_iso, source, key, sec, poly_key)
        if df is None or len(df)<30:
            results.append((sym,0,0.0,0.0,0,0)); 
            continue

        # Keep only regular session bars roughly (09:30–16:00 NY). Heuristic via minute-of-hour and volume.
        df = df.set_index("t").sort_index()
        df["rsi2"] = rsi2(df.c)
        df["vw"]   = vwap(df)
        df["atr"]  = atr(df)

        pnl=0.0; win=0; trades=0
        pos=0; entry=0.0; stop=0.0; shares=0
        for i in range(20, len(df)-1):
            bar = df.iloc[i]
            nxt = df.iloc[i+1]  # trade at next bar open
            price=bar.c; vw=bar.vw; r2=bar.rsi2; a=bar.atr
            if pd.isna(vw) or pd.isna(r2) or pd.isna(a) or a<=0: 
                # manage any open position at close
                continue

            # Exit rules first
            if pos>0:  # long
                # stop hit within current bar range?
                if bar.l <= stop:
                    pnl += (stop-entry)*shares; trades+=1; win += 1 if (stop>entry) else 0
                    pos=0; shares=0
                elif (price>=vw or r2>=50):   # target exit
                    x = nxt.o  # next bar open fill
                    pnl += (x-entry)*shares; trades+=1; win += 1 if (x>entry) else 0
                    pos=0; shares=0
            elif pos<0: # short
                if bar.h >= stop:
                    pnl += (entry-stop)*abs(shares); trades+=1; win += 1 if (entry>stop) else 0
                    pos=0; shares=0
                elif (price<=vw or r2<=50):
                    x = nxt.o
                    pnl += (entry-x)*abs(shares); trades+=1; win += 1 if (entry>x) else 0
                    pos=0; shares=0

            # Entries if flat
            if pos==0:
                # Long: RSI(2) <= 5 and price below VWAP a touch
                if r2<=5 and price<=vw*0.995:
                    stop_dist = atr_mult*a
                    shares    = max(1, int(per_trade_risk/stop_dist))
                    entry     = nxt.o  # next bar open
                    stop      = entry - stop_dist
                    pos       = +1
                # Short: RSI(2) >= 95 and price above VWAP a touch
                elif r2>=95 and price>=vw*1.005:
                    stop_dist = atr_mult*a
                    shares    = max(1, int(per_trade_risk/stop_dist))
                    entry     = nxt.o
                    stop      = entry + stop_dist
                    pos       = -1

        # flatten at the last bar if still holding
        if pos!=0:
            x = df.iloc[-1].c
            if pos>0: pnl += (x-entry)*shares; trades+=1; win += 1 if (x>entry) else 0
            else:     pnl += (entry-x)*abs(shares); trades+=1; win += 1 if (entry>x) else 0
            pos=0

        wr = (win/trades*100.0) if trades>0 else 0.0
        results.append((sym, trades, pnl, wr, win, trades-win))

    out = pd.DataFrame(results, columns=["symbol","trades","pnl","win_rate","wins","losses"])
    out["pnl"] = out["pnl"].round(2); out["win_rate"]=out["win_rate"].round(1)
    print("\n=== Mean-Reversion Backtest (5m) ===")
    print(f"Week: {week}  | picks={len(picks)}  | risk_pct={risk_pct}  atr_mult={atr_mult}")
    print(out.sort_values("pnl", ascending=False).to_string(index=False))
    print(f"\nTOTAL pnl: {out.pnl.sum():.2f}  | total trades: {int(out.trades.sum())}")

def main():
    # --- CLI ---
    ap = argparse.ArgumentParser(description="Mean-reversion backtest on 5m bars")
    ap.add_argument("--picklist", required=True, help="CSV with at least columns: week_start/symbol")
    ap.add_argument("--week", required=True, help="Week (YYYY-MM-DD) to backtest")
    ap.add_argument("--topk", type=int, default=10, help="How many symbols to use")
    ap.add_argument("--risk-pct", type=float, default=0.005, help="Risk per trade as fraction of equity")
    ap.add_argument("--atr-mult", type=float, default=1.5, help="ATR multiple for bands/stops")
    # NEW: choose data source
    ap.add_argument("--data-source", choices=["alpaca", "polygon"], default="alpaca",
                    help="Where to fetch 5m bars from")
    a = ap.parse_args()

    # --- Keys from env ---
    alpaca_key = os.getenv("ALPACA_KEY", "")
    alpaca_sec = os.getenv("ALPACA_SECRET", "")
    polygon_key = os.getenv("POLYGON_KEY", "")

    # --- Validate based on data source ---
    if a.data_source == "alpaca":
        if not alpaca_key or not alpaca_sec:
            print("ERROR: set ALPACA_KEY and ALPACA_SECRET environment variables.", file=sys.stderr)
            sys.exit(1)
    else:  # polygon
        if not polygon_key:
            print("ERROR: set POLYGON_KEY environment variable.", file=sys.stderr)
            sys.exit(1)

    # --- Run ---
    run_backtest(
        picklist=a.picklist,
        week=a.week,
        topk=a.topk,
        risk_pct=a.risk_pct,
        atr_mult=a.atr_mult,
        source=a.data_source,
        key=alpaca_key,
        sec=alpaca_sec,
        poly_key=polygon_key,
    )


if __name__ == "__main__":
    main()
