#!/usr/bin/env python3
import os, argparse, glob, pandas as pd, numpy as np, json

def rsi(x, n=2):
    d=x.diff(); up=d.clip(lower=0); dn=-d.clip(upper=0)
    ru=up.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    rd=dn.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    rs=ru/rd.replace(0,np.nan)
    return (100-100/(1+rs)).fillna(50)

def zscore(x, n=5):
    m=x.rolling(n).mean(); s=x.rolling(n).std(ddof=0); return (x-m)/s.replace(0,np.nan)

def resample_daily(df5):
    # build OHLCV from 5m (RTH) -> daily
    g = df5.set_index("ts").resample("1D")
    o = g["Open"].first(); h = g["High"].max(); l = g["Low"].min(); c = g["Close"].last(); v = g["Volume"].sum()
    out = pd.DataFrame({"Open":o,"High":h,"Low":l,"Close":c,"Volume":v}).dropna()
    out.index = pd.to_datetime(out.index, utc=True)
    return out

def backtest_symbol(df5):
    d = resample_daily(df5)
    if len(d) < 60: return []
    d["RSI2"]=rsi(d["Close"],2)
    d["Z5"]=zscore(d["Close"],5)
    d["BBL"]=d["Close"].rolling(20).mean() - 2*d["Close"].rolling(20).std(ddof=0)
    d["EMA5"]=d["Close"].ewm(span=5, adjust=False).mean()
    # ATR (close-based TR)
    tr = pd.concat([(d["High"]-d["Low"]).abs(),
                    (d["High"]-d["Close"].shift()).abs(),
                    (d["Low"]-d["Close"].shift()).abs()], axis=1).max(axis=1)
    d["ATR14"]=tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()

    trades=[]
    holding=None  # dict with entry info
    for i in range(1, len(d)):
        today = d.index[i]
        prev  = d.index[i-1]
        # signal computed on prev close; entry next open
        if holding is None:
            cond = ((d.loc[prev,"RSI2"]<=5) or 
                    (pd.notna(d.loc[prev,"Z5"]) and d.loc[prev,"Z5"]<=-1.75) or
                    (pd.notna(d.loc[prev,"BBL"]) and d.loc[prev,"Close"]<d.loc[prev,"BBL"]))
            if cond:
                entry = d.loc[today,"Open"]
                atr = d.loc[prev,"ATR14"]
                init_stop = entry - 1.0*atr if pd.notna(atr) else entry*0.95
                holding = {"entry_time":today, "entry":float(entry), "init_stop":float(init_stop), "days":0}
        else:
            # increment age at each close
            holding["days"] += 1
            exit_flag=False
            reason=""
            # close-based stop
            if d.loc[today,"Close"] <= holding["init_stop"]:
                exit_flag=True; reason="stop"
                px = d.loc[today,"Open"]  # next open realistically; treat as open for sim
            # take profit / reversion achieved
            if not exit_flag and (d.loc[today,"RSI2"]>=70 or d.loc[today,"Close"]>d.loc[today,"EMA5"]):
                exit_flag=True; reason="revert"
                px = d.loc[today,"Open"]
            # time stop 3 days
            if not exit_flag and holding["days"]>=3:
                exit_flag=True; reason="time"
                px = d.loc[today,"Open"]

            if exit_flag:
                trades.append({"entry_time":holding["entry_time"].isoformat(),
                               "exit_time":today.isoformat(),
                               "entry":holding["entry"], "exit":float(px),
                               "pnl": float(px - holding["entry"]), "reason":reason})
                holding=None
    return trades

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)     # data_5m/*.parquet
    ap.add_argument("--out", required=True)          # backtests/report.json
    args=ap.parse_args()

    import pyarrow.parquet as pq
    all_trades=[]
    for fp in glob.glob(os.path.join(args.data_dir, "*.parquet")):
        sym = os.path.splitext(os.path.basename(fp))[0]
        try:
            df = pq.read_table(fp).to_pandas()
            if df.empty: continue
            t = backtest_symbol(df)
            for tr in t: tr["symbol"]=sym
            all_trades.extend(t)
        except Exception as e:
            continue

    df = pd.DataFrame(all_trades)
    summary = {}
    if not df.empty:
        summary = {
            "symbols": int(df["symbol"].nunique()),
            "trades": int(len(df)),
            "winrate": float((df["pnl"]>0).mean()),
            "avg_pnl": float(df["pnl"].mean()),
            "sum_pnl": float(df["pnl"].sum()),
        }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary":summary, "trades":all_trades[:2000]}, f, indent=2)
    print("SUMMARY:", summary)

if __name__=="__main__": main()
