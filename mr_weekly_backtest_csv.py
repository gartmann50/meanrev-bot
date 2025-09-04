#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, argparse, math
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import time, timedelta
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")

# ------------------ indicators ------------------
def rsi_ewm(close: pd.Series, n: int = 14, alpha: Optional[float] = None) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0.0)
    dn = -d.clip(upper=0.0)
    if alpha is None:
        alpha = 1.0 / n
    ru = up.ewm(alpha=alpha, adjust=False).mean()
    rd = dn.ewm(alpha=alpha, adjust=False).mean()
    rs = ru / rd.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)

def rsi2(close: pd.Series) -> pd.Series:
    return rsi_ewm(close, n=2, alpha=0.5).fillna(50.0)

def ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False).mean()

def atr14_from_daily(d: pd.DataFrame) -> pd.Series:
    tr = pd.concat([
        (d["High"] - d["Low"]).abs(),
        (d["High"] - d["Close"].shift()).abs(),
        (d["Low"]  - d["Close"].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()

# ------------------ helpers ------------------
def parse_time(s: str) -> time:
    p = s.split(":")
    h = int(p[0]); m = int(p[1]) if len(p)>1 else 0; s2 = int(p[2]) if len(p)>2 else 0
    return time(h, m, s2)

def monday_utc_from_ts(ts: pd.Timestamp) -> pd.Timestamp:
    ny_date = ts.tz_convert(NY).date()
    mon = ny_date - timedelta(days=ny_date.weekday())
    return pd.Timestamp(mon, tz=NY).tz_convert("UTC")

def prev_trading_day_before(dly_index: pd.DatetimeIndex, ny_date) -> Optional[pd.Timestamp]:
    mask = dly_index.tz_convert(NY).date < ny_date
    if not mask.any():
        return None
    return dly_index[mask][-1]

def within_session(df5: pd.DataFrame, start_t: time, end_t: time) -> pd.DataFrame:
    if df5.empty: return df5
    ny = df5["ts"].dt.tz_convert(NY)
    tt = ny.dt.time
    return df5[(tt >= start_t) & (tt <= end_t)].copy()

# ------------------ IO (CSV only) ------------------
def load_5m_csv(fp: str) -> pd.DataFrame:
    # extremely tolerant CSV reader
    t = pd.read_csv(fp, engine="python", on_bad_lines="skip")
    if t is None or len(t)==0:
        return pd.DataFrame()
    lm = {c.lower(): c for c in t.columns}

    def pick(*names):
        for n in names:
            if n in lm: return lm[n]
        return None

    ts = pick("ts","timestamp","time","datetime")
    o  = pick("open","o")
    h  = pick("high","h")
    l  = pick("low","l")
    c  = pick("close","c")
    v  = pick("volume","v","vol")
    if not all([ts,o,h,l,c,v]):  # must have these six
        return pd.DataFrame()

    df = pd.DataFrame({
        "ts":    pd.to_datetime(t[ts], utc=True, errors="coerce"),
        "Open":  pd.to_numeric(t[o], errors="coerce"),
        "High":  pd.to_numeric(t[h], errors="coerce"),
        "Low":   pd.to_numeric(t[l], errors="coerce"),
        "Close": pd.to_numeric(t[c], errors="coerce"),
        "Volume":pd.to_numeric(t[v], errors="coerce"),
    }).dropna(subset=["ts"]).sort_values("ts")

    # ensure tz-aware UTC
    if df["ts"].dt.tz is None:
        df["ts"] = df["ts"].dt.tz_localize("UTC")
    else:
        df["ts"] = df["ts"].dt.tz_convert("UTC")

    return df

def resample_daily_from_5m(df5: pd.DataFrame) -> pd.DataFrame:
    if df5.empty: return df5
    ny = df5["ts"].dt.tz_convert(NY)
    g = df5.set_index(ny).resample("1D", label="right", closed="right")
    d = pd.DataFrame({
        "Open":   g["Open"].first(),
        "High":   g["High"].max(),
        "Low":    g["Low"].min(),
        "Close":  g["Close"].last(),
        "Volume": g["Volume"].sum()
    }).dropna()
    d.index = pd.to_datetime(d.index, utc=True)
    d["RSI2"]  = rsi2(d["Close"])
    d["RSI14"] = rsi_ewm(d["Close"], n=14)
    d["EMA5"]  = ema(d["Close"], 5)
    d["ATR14"] = atr14_from_daily(d)
    return d

# ------------------ weekly picks ------------------
def weekly_picks(all_daily: Dict[str, pd.DataFrame], week_start: pd.Timestamp,
                 k: int, rsi_lo: float, rsi_hi: float) -> List[str]:
    friday = week_start - pd.Timedelta(days=1)
    rows = []
    for sym, d in all_daily.items():
        if d.empty: continue
        db = d[d.index <= friday]
        if db.empty: continue
        r = float(db["RSI14"].iloc[-1])
        if math.isnan(r): continue
        rows.append((sym, abs(r-50.0), r>=rsi_lo and r<=rsi_hi))
    if not rows: return []
    df = pd.DataFrame(rows, columns=["symbol","dist","in_band"])
    core = df[df["in_band"]].sort_values("dist").head(k)
    if len(core) < k:
        fill = df[~df["in_band"]].sort_values("dist").head(k-len(core))
        core = pd.concat([core, fill], ignore_index=True)
    return core["symbol"].tolist()

# ------------------ trade model (MR intraday) ------------------
@dataclass
class Trade:
    symbol: str
    side: str
    entry_time: str
    exit_time: str
    entry: float
    exit: float
    reason: str
    stop: float
    take: float
    week: str

def levels_long(entry: float, atr_prev: Optional[float], risk: str,
                stop_mult: float, tp_mult: float) -> Tuple[float,float,float]:
    if risk=="atr" and atr_prev and atr_prev>0:
        stop = entry - stop_mult*atr_prev
        take = entry + tp_mult  *atr_prev
        R    = stop_mult*atr_prev
    else:
        stop = entry*(1.0 - stop_mult)
        take = entry*(1.0 + tp_mult)
        R    = entry*stop_mult
    return float(stop), float(take), float(R)

def levels_short(entry: float, atr_prev: Optional[float], risk: str,
                 stop_mult: float, tp_mult: float) -> Tuple[float,float,float]:
    if risk=="atr" and atr_prev and atr_prev>0:
        stop = entry + stop_mult*atr_prev
        take = entry - tp_mult  *atr_prev
        R    = stop_mult*atr_prev
    else:
        stop = entry*(1.0 + stop_mult)
        take = entry*(1.0 - tp_mult)
        R    = entry*stop_mult
    return float(stop), float(take), float(R)

def run_week_intraday(sym: str, df5: pd.DataFrame, dly: pd.DataFrame,
                      week_start: pd.Timestamp,
                      session_start: time, session_end: time,
                      side_mode: str, rsi_short: float,
                      max_trades_per_day: int,
                      risk: str, stop_mult: float, tp_mult: float,
                      eod_exit: bool,
                      trail: str, trail_mult: float,
                      breakeven_after: float,
                      max_hold_bars: int,
                      min_session_bars: int) -> List[Trade]:

    T: List[Trade] = []
    week_end = week_start + pd.Timedelta(days=5)
    bars = df5[(df5["ts"] >= week_start) & (df5["ts"] < week_end)]
    if bars.empty: return T
    dly_w = dly[(dly.index >= (week_start - pd.Timedelta(days=10))) & (dly.index < week_end)]
    if dly_w.empty: return T

    ny_days = pd.to_datetime(bars["ts"].dt.tz_convert(NY).dt.date.unique())
    for ny_date in ny_days:
        day = bars[bars["ts"].dt.tz_convert(NY).dt.date == ny_date.date()]
        if day.empty: continue
        day = within_session(day, session_start, session_end)
        if day.shape[0] < min_session_bars:  # optional coverage gate
            continue

        prevd = prev_trading_day_before(dly_w.index, ny_date.date())
        if prevd is None: continue
        prev = dly_w.loc[prevd]
        atrp = float(prev.get("ATR14", np.nan))
        rsi_intra = rsi2(day["Close"])

        open_pos = False; dir = 0; stop = take = entry_px = 0.0; R = 0.0
        peak = trough = None; age = 0; done_today = 0

        def trail_long(curr_peak):
            nonlocal stop
            if trail == "atr" and risk=="atr" and atrp and atrp>0:
                stop = max(stop, curr_peak - trail_mult*atrp)
            elif trail == "pct":
                stop = max(stop, curr_peak * (1.0 - trail_mult))

        def trail_short(curr_trough):
            nonlocal stop
            if trail == "atr" and risk=="atr" and atrp and atrp>0:
                stop = min(stop, curr_trough + trail_mult*atrp)
            elif trail == "pct":
                stop = min(stop, curr_trough * (1.0 + trail_mult))

        for i in range(len(day)):
            if done_today >= max_trades_per_day: break
            c = float(day.iloc[i]["Close"])
            t = pd.Timestamp(day.iloc[i]["ts"]).isoformat()

            if not open_pos:
                want_long = want_short = False
                if side_mode in ("long","both"):  # MR: buy when oversold
                    want_long = (rsi_intra.iloc[i] <= rsi_short)
                if side_mode in ("short","both"): # MR: short when overbought
                    want_short = (rsi_intra.iloc[i] >= (100.0 - rsi_short))

                if want_long:
                    entry_px = c; stop, take, R = levels_long(entry_px, atrp, risk, stop_mult, tp_mult)
                    dir = +1; open_pos = True; peak = entry_px; trough = None; age = 0
                elif want_short:
                    entry_px = c; stop, take, R = levels_short(entry_px, atrp, risk, stop_mult, tp_mult)
                    dir = -1; open_pos = True; trough = entry_px; peak = None; age = 0
                continue

            age += 1
            if dir == +1:
                if c > peak: peak = c
                if breakeven_after > 0:
                    if risk=="atr" and R>0 and (c-entry_px) >= breakeven_after*R:
                        stop = max(stop, entry_px)
                    elif risk=="pct" and entry_px>0 and ((c-entry_px)/entry_px) >= breakeven_after:
                        stop = max(stop, entry_px)
                if trail != "none": trail_long(peak)
                hit_stop = c <= stop; hit_take = c >= take
            else:  # short
                if (trough is None) or (c < trough): trough = c
                if breakeven_after > 0:
                    if risk=="atr" and R>0 and (entry_px-c) >= breakeven_after*R:
                        stop = min(stop, entry_px)
                    elif risk=="pct" and entry_px>0 and ((entry_px-c)/entry_px) >= breakeven_after:
                        stop = min(stop, entry_px)
                if trail != "none": trail_short(trough)
                hit_stop = c >= stop; hit_take = c <= take

            reason = None
            if hit_stop: reason = "stop"
            elif hit_take: reason = "take"
            elif max_hold_bars>0 and age>=max_hold_bars: reason = "time"

            if reason:
                if i+1 < len(day):
                    exit_px = float(day.iloc[i+1]["Open"])
                    exit_ts = pd.Timestamp(day.iloc[i+1]["ts"]).isoformat()
                else:
                    exit_px = c; exit_ts = t
                T.append(Trade(
                    symbol=sym, side=("long" if dir==1 else "short"),
                    entry_time=pd.Timestamp(day.iloc[i-age+1]["ts"]).isoformat() if age>0 else t,
                    exit_time=exit_ts, entry=entry_px, exit=exit_px, reason=reason,
                    stop=stop, take=take, week=str(week_start.tz_convert(NY).date())
                ))
                open_pos = False; dir = 0; done_today += 1

        if open_pos and eod_exit:
            last = day.iloc[-1]
            T.append(Trade(
                symbol=sym, side=("long" if dir==1 else "short"),
                entry_time=pd.Timestamp(day.iloc[-age-1]["ts"]).isoformat() if age>0 else pd.Timestamp(day.iloc[0]["ts"]).isoformat(),
                exit_time=pd.Timestamp(last["ts"]).isoformat(),
                entry=entry_px, exit=float(last["Close"]), reason="eod",
                stop=stop, take=take, week=str(week_start.tz_convert(NY).date())
            ))

    return T

# ------------------ main ------------------
def main():
    ap = argparse.ArgumentParser(description="Weekly-screened intraday mean-reversion backtester (CSV 5m only).")
    ap.add_argument("--data-dir", required=True, help="Folder containing *.csv (one symbol per file).")
    ap.add_argument("--out-dir", default="backtests")
    ap.add_argument("--no-json", action="store_true")

    # weekly screen
    ap.add_argument("--picks-per-week", type=int, default=40)
    ap.add_argument("--rsi14-lo", type=float, default=35.0)
    ap.add_argument("--rsi14-hi", type=float, default=65.0)

    # entries
    ap.add_argument("--side", choices=["long","short","both"], default="both")
    ap.add_argument("--rsi-short", type=float, default=10.0, help="MR threshold (≤ for long, ≥ 100-x for short).")
    ap.add_argument("--max-trades-per-day", type=int, default=4)

    # exits/sizing
    ap.add_argument("--risk", choices=["atr","pct"], default="atr")
    ap.add_argument("--stop-mult", type=float, default=1.0)
    ap.add_argument("--tp-mult",   type=float, default=1.8)
    ap.add_argument("--trail", choices=["none","atr","pct"], default="atr")
    ap.add_argument("--trail-mult", type=float, default=0.8)
    ap.add_argument("--breakeven-after", type=float, default=1.0)
    ap.add_argument("--max-hold-bars", type=int, default=72)
    ap.add_argument("--eod-exit", action="store_true")

    # coverage/span
    ap.add_argument("--min-session-bars", type=int, default=0, help="Skip a symbol-day if fewer bars (09:40–15:50 NY).")
    ap.add_argument("--start", type=str, default="", help="YYYY-MM-DD (NY) inclusive")
    ap.add_argument("--end",   type=str, default="", help="YYYY-MM-DD (NY) inclusive")

    # misc
    ap.add_argument("--notional", type=float, default=5000.0)
    ap.add_argument("--session-start", default="09:40")
    ap.add_argument("--session-end",   default="15:50")
    args = ap.parse_args()

    tstart = parse_time(args.session_start)
    tend   = parse_time(args.session_end)

    # discover csv files
    files: List[str] = []
    for r,_,fs in os.walk(args.data_dir):
        for f in fs:
            if f.lower().endswith(".csv"):
                files.append(os.path.join(r,f))
    files.sort()
    if not files:
        print("No CSV files found."); return
    print(f"Found CSV files: {len(files)}")

    # load all symbols
    sym_5m: Dict[str, pd.DataFrame] = {}
    sym_d  : Dict[str, pd.DataFrame] = {}
    for fp in files:
        sym = os.path.splitext(os.path.basename(fp))[0]
        try:
            m5 = load_5m_csv(fp)
            if m5.empty: continue
            sym_5m[sym] = m5
            sym_d[sym]  = resample_daily_from_5m(m5)
        except Exception:
            continue

    print(f"Loaded symbols: {len(sym_5m)}")
    if not sym_d: print("No daily data created."); return

    # build week list from available daily dates
    indices = [d.index for d in sym_d.values() if not d.empty]
    all_dates = sorted({ts for idx in indices for ts in idx})
    mondays = sorted({monday_utc_from_ts(ts) for ts in all_dates})
    if args.start:
        s = pd.Timestamp(args.start, tz=NY).tz_convert("UTC")
        mondays = [m for m in mondays if m >= s]
    if args.end:
        e = pd.Timestamp(args.end, tz=NY).tz_convert("UTC")
        mondays = [m for m in mondays if m <= e]

    os.makedirs(args.out_dir, exist_ok=True)
    picks_rows, trades_rows = [], []

    for wk in mondays:
        picks = weekly_picks(sym_d, wk, args.picks_per_week, args.rsi14_lo, args.rsi14_hi)
        if not picks: continue
        for s in picks:
            picks_rows.append({"week": str(wk.tz_convert(NY).date()), "symbol": s})
            d5 = sym_5m.get(s); dly = sym_d.get(s)
            if d5 is None or dly is None or d5.empty or dly.empty: continue
            trs = run_week_intraday(
                s, d5, dly, wk, tstart, tend,
                args.side, args.rsi_short, args.max_trades_per_day,
                args.risk, args.stop_mult, args.tp_mult,
                args.eod_exit, args.trail, args.trail_mult,
                args.breakeven_after, args.max_hold_bars,
                args.min_session_bars
            )
            trades_rows += [t.__dict__ for t in trs]

    picks_df = pd.DataFrame(picks_rows)
    trades_df = pd.DataFrame(trades_rows)
    picks_df.to_csv(os.path.join(args.out_dir, "picks_weekly.csv"), index=False)
    trades_df.to_csv(os.path.join(args.out_dir, "trades.csv"), index=False)

    if trades_df.empty:
        summary = {
            "symbols": 0,
            "trades": 0,
            "winrate": 0.0,
            "avg_ret": 0.0,
            "sum_ret": 0.0,
            "sum_pnl_usd": 0.0,
        }
        weekly = pd.DataFrame(columns=["week", "trades", "winrate", "sum_pnl_usd"])
    else:
        dir_val = trades_df["side"].map({"long": 1, "short": -1}).astype(float)
        trades_df["ret"] = (trades_df["exit"] / trades_df["entry"] - 1.0) * dir_val
        trades_df["pnl_usd"] = args.notional * trades_df["ret"]

        summary = {
            "symbols": int(trades_df["symbol"].nunique()),
            "trades": int(len(trades_df)),
            "winrate": float((trades_df["pnl_usd"] > 0).mean()),
            "avg_ret": float(trades_df["ret"].mean()),
            "sum_ret": float(trades_df["ret"].sum()),
            "sum_pnl_usd": float(trades_df["pnl_usd"].sum()),
        }

        weekly = (
            trades_df.groupby("week", as_index=False)
            .agg(
                trades=("symbol", "count"),
                winrate=("ret", lambda s: float((s > 0).mean())),
                sum_pnl_usd=("pnl_usd", "sum"),
            )
            .sort_values("week")
        )

    # ---- console output ----
    print("\n=== SUMMARY ===")
    print(
        f"symbols={summary['symbols']}  trades={summary['trades']}  "
        f"winrate={summary['winrate']:.2%}  avg_ret={summary['avg_ret']:.4f}  "
        f"sum_pnl_usd={summary['sum_pnl_usd']:.2f}"
    )

    if not weekly.empty:
        print("\nWeekly (all):")
        print(weekly.to_string(index=False))
        # One-line recap right after the weekly table
        print(
            f"\nRECAP: winrate={summary['winrate']:.2%} | total_pnl_usd={summary['sum_pnl_usd']:,.2f}"
        )

    # ---- files ----
    weekly.to_csv(os.path.join(args.out_dir, "weekly_summary.csv"), index=False)
    if not args.no_json:
        with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
            json.dump({"summary": summary}, f, indent=2)

if __name__ == "__main__":
    main()
