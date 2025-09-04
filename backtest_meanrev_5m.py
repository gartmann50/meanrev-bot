#!/usr/bin/env python3
import os, glob, json, argparse, numpy as np, pandas as pd
import pyarrow.parquet as pq

# ---------- indicators (daily) ----------
def rsi2(close):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=0.5, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=0.5, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)

def zscore(close, n=5):
    m = close.rolling(n).mean()
    s = close.rolling(n).std(ddof=0)
    return (close - m) / s.replace(0, np.nan)

def resample_daily(df5):
    g = df5.set_index("ts").resample("1D")
    o = g["Open"].first(); h = g["High"].max(); l = g["Low"].min(); c = g["Close"].last(); v = g["Volume"].sum()
    out = pd.DataFrame({"Open":o,"High":h,"Low":l,"Close":c,"Volume":v}).dropna()
    out.index = pd.to_datetime(out.index, utc=True)
    return out

# ---------- core logic ----------
def choose_levels(entry, atr_prev, risk, stop_mult, tp_mult):
    if risk == "atr" and pd.notna(atr_prev) and atr_prev > 0:
        stop = entry - stop_mult * atr_prev
        take = entry + tp_mult   * atr_prev
    else:
        # pct mode (stop_mult/tp_mult are fractions, e.g., 0.05, 0.07)
        stop = entry * (1.0 - stop_mult)
        take = entry * (1.0 + tp_mult)
    return float(stop), float(take)

def backtest_symbol(df5, args):
    d = resample_daily(df5)
    if len(d) < 60:
        return []

    d["RSI2"] = rsi2(d["Close"])
    d["Z5"]   = zscore(d["Close"], 5)
    d["EMA5"] = d["Close"].ewm(span=5, adjust=False).mean()

    tr = pd.concat([
        (d["High"]-d["Low"]).abs(),
        (d["High"]-d["Close"].shift()).abs(),
        (d["Low"] -d["Close"].shift()).abs()
    ], axis=1).max(axis=1)
    d["ATR14"] = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()

    trades, hold = [], None

    for i in range(1, len(d)):
        prev = d.index[i-1]
        today = d.index[i]

        if hold is None:
            # --- Entry: signal on prev day, enter at today's open
            if (d.loc[prev, "RSI2"] <= 5) or (pd.notna(d.loc[prev, "Z5"]) and d.loc[prev, "Z5"] <= -1.75):
                entry = float(d.loc[today, "Open"])
                atrp  = d.loc[prev, "ATR14"]
                stop, take = choose_levels(entry, atrp, args.risk, args.stop_mult, args.tp_mult)
                hold  = {"t_in": today.isoformat(), "px_in": entry, "stop": stop, "take": take, "age": 0}
        else:
            # --- Manage: decide using prev-day info, execute at today's open (no look-ahead)
            hold["age"] += 1
            exit_px = None; reason = ""

            # 1) Stop (prev close below/equal stop)
            if d.loc[prev, "Close"] <= hold["stop"]:
                exit_px = float(d.loc[today, "Open"]); reason = "stop"
            # 2) Take profit (prev close at/above target)
            elif d.loc[prev, "Close"] >= hold["take"]:
                exit_px = float(d.loc[today, "Open"]); reason = "take"
            # 3) Reversion exit (optional: RSI/EMA conditions checked on prev day)
            elif (args.exit_rsi is not None and d.loc[prev, "RSI2"] >= args.exit_rsi) or \
                 (args.exit_ema and d.loc[prev, "Close"] > d.loc[prev, "EMA5"]):
                exit_px = float(d.loc[today, "Open"]); reason = "revert"
            # 4) Time stop
            elif hold["age"] >= args.max_hold:
                exit_px = float(d.loc[today, "Open"]); reason = "time"

            if exit_px is not None:
                trades.append({
                    "entry_time": hold["t_in"], "exit_time": today.isoformat(),
                    "entry": hold["px_in"], "exit": exit_px,
                    "stop": hold["stop"], "take": hold["take"],
                    "reason": reason
                })
                hold = None

    return trades

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--notional", type=float, default=5000.0)

    # NEW knobs
    ap.add_argument("--risk", choices=["atr","pct"], default="atr",
                    help="atr: use ATR(14)×mult; pct: use +/- percentages from entry")
    ap.add_argument("--stop-mult", type=float, default=1.0, help="ATR mult or pct (e.g., 0.05)")
    ap.add_argument("--tp-mult",   type=float, default=1.5, help="ATR mult or pct (e.g., 0.07)")
    ap.add_argument("--max-hold",  type=int,   default=3,   help="days")
    ap.add_argument("--exit-rsi",  type=float, default=70.0, help="set None to disable", nargs='?')
    ap.add_argument("--exit-ema",  action="store_true", help="exit if Close>EMA5 (prev day)")

    args = ap.parse_args()

    all_trades = []
    used, skipped = 0, 0

    for fp in glob.glob(os.path.join(args.data_dir, "*.parquet")):
        sym = os.path.splitext(os.path.basename(fp))[0]
        try:
            df = pq.read_table(fp).to_pandas()
            if df.empty:
                skipped += 1; continue
            t = backtest_symbol(df, args)
            for tr in t: tr["symbol"] = sym
            all_trades.extend(t)
            used += 1
        except Exception:
            skipped += 1

    df = pd.DataFrame(all_trades)
    if df.empty:
        summary = {"symbols": 0, "trades": 0, "winrate": 0, "avg_ret": 0, "sum_ret": 0, "avg_pnl_$": 0, "sum_pnl_$": 0,
                   "files_used": used, "files_skipped": skipped}
    else:
        df["ret"]   = df["exit"]/df["entry"] - 1.0
        df["pnl_$"] = args.notional * df["ret"]
        summary = {
            "symbols": int(df["symbol"].nunique()),
            "trades":  int(len(df)),
            "winrate": float((df["pnl_$"] > 0).mean()),
            "avg_ret": float(df["ret"].mean()),
            "sum_ret": float(df["ret"].sum()),
            "avg_pnl_$": float(df["pnl_$"].mean()),
            "sum_pnl_$": float(df["pnl_$"].sum()),
            "files_used": used, "files_skipped": skipped
        }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "trades": all_trades[:2000]}, f, indent=2)
    print("SUMMARY:", summary)

if __name__ == "__main__":
    main()
