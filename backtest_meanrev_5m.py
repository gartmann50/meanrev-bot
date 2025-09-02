#!/usr/bin/env python3
import os, glob, json, argparse, numpy as np, pandas as pd
import pyarrow.parquet as pq

def rsi2(close):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=0.5, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=0.5, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)

def zscore(close, n=5):
    m = close.rolling(n).mean(); s = close.rolling(n).std(ddof=0)
    return (close - m) / s.replace(0, np.nan)

def resample_daily(df5):
    g = df5.set_index("ts").resample("1D")
    o=g["Open"].first(); h=g["High"].max(); l=g["Low"].min(); c=g["Close"].last(); v=g["Volume"].sum()
    out = pd.DataFrame({"Open":o,"High":h,"Low":l,"Close":c,"Volume":v}).dropna()
    out.index = pd.to_datetime(out.index, utc=True)
    return out

def backtest_symbol(df5):
    d = resample_daily(df5)
    if len(d) < 60: return []
    d["RSI2"]=rsi2(d["Close"])
    d["Z5"]=zscore(d["Close"],5)
    d["EMA5"]=d["Close"].ewm(span=5, adjust=False).mean()
    tr = pd.concat([(d["High"]-d["Low"]).abs(),
                    (d["High"]-d["Close"].shift()).abs(),
                    (d["Low"] -d["Close"].shift()).abs()], axis=1).max(axis=1)
    d["ATR14"]=tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()

    trades=[]; hold=None
    for i in range(1, len(d)):
        prev=d.index[i-1]; today=d.index[i]
        if hold is None:
            if (d.loc[prev,"RSI2"]<=5) or (pd.notna(d.loc[prev,"Z5"]) and d.loc[prev,"Z5"]<=-1.75):
                entry=float(d.loc[today,"Open"])
                atr=d.loc[prev,"ATR14"]; stop=entry - (1.0*atr if pd.notna(atr) else 0.05*entry)
                hold={"t_in":today.isoformat(),"px_in":entry,"stop":float(stop),"age":0}
        else:
            hold["age"]+=1
            exit_px=None; reason=""
            if d.loc[today,"Close"] <= hold["stop"]:
                exit_px=float(d.loc[today,"Open"]); reason="stop"
            elif d.loc[today,"RSI2"]>=70 or d.loc[today,"Close"]>d.loc[today,"EMA5"]:
                exit_px=float(d.loc[today,"Open"]); reason="revert"
            elif hold["age"]>=3:
                exit_px=float(d.loc[today,"Open"]); reason="time"
            if exit_px is not None:
                trades.append({"entry_time":hold["t_in"],"exit_time":today.isoformat(),
                               "entry":hold["px_in"],"exit":exit_px,"reason":reason})
                hold=None
    return trades

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--notional", type=float, default=5000.0, help="Dollars per trade for $PnL calc")
    a=ap.parse_args()

    all_trades=[]
    for fp in glob.glob(os.path.join(a.data_dir, "*.parquet")):
        sym=os.path.splitext(os.path.basename(fp))[0]
        try:
            df=pq.read_table(fp).to_pandas()
            if df.empty: continue
            t=backtest_symbol(df)
            for tr in t: tr["symbol"]=sym
            all_trades.extend(t)
        except Exception:
            pass

    df=pd.DataFrame(all_trades)
    summary={}
    if not df.empty:
        df["ret"] = df["exit"]/df["entry"] - 1.0
        df["pnl_px"] = df["exit"] - df["entry"]
        df["pnl_$"]  = a.notional * df["ret"]
        summary = {
            "symbols": int(df["symbol"].nunique()),
            "trades":  int(len(df)),
            "winrate": float((df["pnl_$"]>0).mean()),
            "avg_ret": float(df["ret"].mean()),
            "sum_ret": float(df["ret"].sum()),
            "avg_pnl_$": float(df["pnl_$"].mean()),
            "sum_pnl_$": float(df["pnl_$"].sum()),
        }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out,"w") as f:
        json.dump({"summary":summary,"trades":all_trades[:2000]}, f, indent=2)
    print("SUMMARY:", summary)

if __name__=="__main__":
    main()
