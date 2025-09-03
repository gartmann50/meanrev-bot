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

def fetch_bars_5m(sym, start_iso, end_iso, key, sec, limit=10000):
    """Pull 5m bars from Alpaca Data v2. Simple single-shot; for longer ranges split by day."""
    h={"APCA-API-KEY-ID":key,"APCA-API-SECRET-KEY":sec}
    r=requests.get(f"{alpaca_data_base()}/v2/stocks/{sym}/bars",
                   params={"timeframe":"5Min","start":start_iso,"end":end_iso,"limit":limit},
                   headers=h, timeout=30)
    if r.status_code>=400: 
        print(f"{sym} bars HTTP {r.status_code}: {r.text[:200]}")
        return None
    b=r.json().get("bars",[])
    if not b: return None
    df=pd.DataFrame(b)[["t","o","h","l","c","v"]].rename(columns=str.lower)
    df["t"]=pd.to_datetime(df["t"], utc=True)
    return df

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

def run_backtest(picklist, week, topk, risk_pct, atr_mult, key, sec):
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
        df = fetch_bars_5m(sym, start_iso, end_iso, key, sec)
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
    ap=argparse.ArgumentParser()
    ap.add_argument("--picklist", default="backtests/mr_basket.csv")
    ap.add_argument("--week", required=True, help="Monday date YYYY-MM-DD")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--risk-pct", type=float, default=0.005)
    ap.add_argument("--atr-mult", type=float, default=1.5)
    a=ap.parse_args()

    key=os.getenv("ALPACA_KEY",""); sec=os.getenv("ALPACA_SECRET","")
    if not key or not sec: raise SystemExit("Set ALPACA_KEY and ALPACA_SECRET in your env.")
    run_backtest(a.picklist, a.week, a.topk, a.risk-pct, a.atr_mult, key, sec)

if __name__=="__main__":
    main()
