#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, argparse, math
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from datetime import time, timedelta
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")

# ---------------------------- Indicators ----------------------------

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

def zscore(close: pd.Series, n: int = 5) -> pd.Series:
    m = close.rolling(n).mean()
    s = close.rolling(n).std(ddof=0)
    return (close - m) / s.replace(0, np.nan)

def atr14_from_daily(d: pd.DataFrame) -> pd.Series:
    tr = pd.concat([
        (d["High"] - d["Low"]).abs(),
        (d["High"] - d["Close"].shift()).abs(),
        (d["Low"]  - d["Close"].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()

# ---------------------------- Resampling & Calendar ----------------------------

def resample_daily(df5: pd.DataFrame) -> pd.DataFrame:
    g = df5.set_index("ts").resample("1D")
    out = pd.DataFrame({
        "Open":   g["Open"].first(),
        "High":   g["High"].max(),
        "Low":    g["Low"].min(),
        "Close":  g["Close"].last(),
        "Volume": g["Volume"].sum()
    }).dropna()
    out.index = pd.to_datetime(out.index, utc=True)
    return out

def ny_time_series_filter(df5: pd.DataFrame, start_t: time, end_t: time) -> pd.DataFrame:
    if df5.empty:
        return df5
    ny_times = df5["ts"].dt.tz_convert(NY)
    tt = ny_times.dt.time
    return df5.loc[(tt >= start_t) & (tt <= end_t)].copy()

def week_start_monday(d: pd.Timestamp) -> pd.Timestamp:
    ny = d.tz_convert(NY).date()
    mon = ny - timedelta(days=ny.weekday())
    return pd.Timestamp(mon, tz=NY).tz_convert("UTC")

# ---------------------------- Selection ----------------------------

def weekly_picklist(symbol_daily: Dict[str, pd.DataFrame],
                    week_start: pd.Timestamp,
                    picks_per_week: int,
                    rsi_mid_lo: float = 40.0,
                    rsi_mid_hi: float = 60.0) -> List[str]:
    """
    Select non-trending names for this week using info strictly BEFORE the week.
    Rule: RSI14 near 50 at last available day before the week.
    """
    rows = []
    cutoff = week_start - pd.Timedelta(days=1)
    for sym, d in symbol_daily.items():
        if d.empty:
            continue
        d_before = d[d.index <= cutoff]
        if d_before.empty:
            continue
        r = d_before["RSI14"].iloc[-1]
        if np.isnan(r):
            continue
        dist = abs(r - 50.0)
        in_band = (r >= rsi_mid_lo) and (r <= rsi_mid_hi)
        rows.append((sym, dist, in_band))
    if not rows:
        return []
    df = pd.DataFrame(rows, columns=["symbol", "dist", "in_band"])
    core = df[df["in_band"]].sort_values("dist").head(picks_per_week)
    if len(core) < picks_per_week:
        need = picks_per_week - len(core)
        fill = df[~df["in_band"]].sort_values("dist").head(need)
        core = pd.concat([core, fill], ignore_index=True)
    return core["symbol"].tolist()

# ---------------------------- Risk & Exits ----------------------------

def choose_levels(entry: float, atr_prev: Optional[float], risk: str,
                  stop_mult: float, tp_mult: float) -> Tuple[float, float, float]:
    if risk == "atr" and (atr_prev is not None) and (not math.isnan(atr_prev)) and atr_prev > 0:
        stop = entry - stop_mult * atr_prev
        take = entry + tp_mult   * atr_prev
        R    = stop_mult * atr_prev
    else:
        stop = entry * (1.0 - stop_mult)
        take = entry * (1.0 + tp_mult)
        R    = entry * stop_mult
    return float(stop), float(take), float(R)

def tighten_trail(peak: float, entry: float, base_atr: Optional[float], risk: str,
                  trail: str, trail_mult: float) -> Optional[float]:
    if trail == "none":
        return None
    if trail == "atr" and risk == "atr" and base_atr and base_atr > 0:
        lvl = peak - trail_mult * base_atr
    elif trail == "pct":
        lvl = peak * (1.0 - trail_mult)
    else:
        return None
    return max(lvl, entry)

# ---------------------------- Trading Core ----------------------------

@dataclass
class Trade:
    symbol: str
    entry_time: str
    exit_time: str
    entry: float
    exit: float
    reason: str
    stop: float
    take: float

def run_week_for_symbol(sym: str,
                        df5: pd.DataFrame,
                        dly: pd.DataFrame,
                        week_start: pd.Timestamp,
                        session_start: time,
                        session_end: time,
                        args) -> List[Trade]:
    trades: List[Trade] = []
    week_end = week_start + pd.Timedelta(days=5)
    bars_week = df5[(df5["ts"] >= week_start) & (df5["ts"] < week_end)]
    if bars_week.empty:
        return trades

    dly_this = dly[(dly.index >= (week_start - pd.Timedelta(days=10))) & (dly.index < week_end)]
    if dly_this.empty:
        return trades

    pos = None

    ny_days = pd.to_datetime(bars_week["ts"].dt.tz_convert(NY).dt.date.unique())
    for ny_date in ny_days:
        day_mask = bars_week["ts"].dt.tz_convert(NY).dt.date == ny_date.date()
        day_bars = bars_week.loc[day_mask].copy()
        if day_bars.empty:
            continue
        day_bars = ny_time_series_filter(day_bars, session_start, session_end)
        if day_bars.empty:
            continue

        prev_daily_idx = dly_this.index[dly_this.index.tz_convert(NY).date < ny_date.date()]
        if len(prev_daily_idx) == 0:
            continue
        prevd = prev_daily_idx[-1]
        prev = dly_this.loc[prevd]

        # If carrying a position across days, refresh prev-day refs
        if pos is not None:
            pos["prev_rsi2"] = float(prev["RSI2"])
            pos["prev_ema5"] = float(prev["EMA5"])
            pos["prev_close"] = float(prev["Close"])
            pos["atr_prev"] = float(prev["ATR14"]) if "ATR14" in prev.index else np.nan

        # Entry at first bar if signal and no open position
        first_bar = day_bars.iloc[0]
        sig_short = (prev["RSI2"] <= args.short_rsi)
        sig_long  = (prev["RSI2"] <= args.long_rsi) and (prev["Close"] < prev["EMA5"])

        if pos is None and (sig_short or sig_long):
            entry_px = float(first_bar["Open"])
            atrp = float(prev["ATR14"]) if "ATR14" in prev.index else np.nan
            stop, take, R = choose_levels(entry_px, atrp, args.risk, args.stop_mult, args.tp_mult)
            pos = {
                "entry": entry_px, "stop": stop, "take": take, "R": R,
                "peak": entry_px, "age": 0,
                "prev_rsi2": float(prev["RSI2"]),
                "prev_ema5": float(prev["EMA5"]),
                "prev_close": float(prev["Close"]),
                "atr_prev": atrp,
                "entry_ts": first_bar["ts"]
            }

        # Manage an open position through the day's bars
        if pos is not None:
            for i in range(len(day_bars)):
                bar = day_bars.iloc[i]
                pos["age"] += 1
                pos["peak"] = max(pos["peak"], float(bar["High"]))

                tl = tighten_trail(pos["peak"], pos["entry"], pos["atr_prev"], args.risk, args.trail, args.trail_mult)
                if tl is not None:
                    pos["stop"] = max(pos["stop"], tl)

                unrl = float(bar["Close"]) - pos["entry"]
                if args.risk == "atr" and pos["R"] > 0 and unrl >= args.breakeven_after * pos["R"]:
                    pos["stop"] = max(pos["stop"], pos["entry"])
                elif args.risk == "pct" and pos["entry"] > 0 and (unrl / pos["entry"]) >= args.breakeven_after:
                    pos["stop"] = max(pos["stop"], pos["entry"])

                exit_reason = None
                if float(bar["Close"]) <= pos["stop"]:
                    exit_reason = "stop"
                elif float(bar["Close"]) >= pos["take"]:
                    exit_reason = "take"
                elif (args.exit_rsi > 0 and pos["prev_rsi2"] >= args.exit_rsi):
                    exit_reason = "revert_rsi"
                elif (args.exit_ema and pos["prev_close"] > pos["prev_ema5"]):
                    exit_reason = "revert_ema"
                elif (args.max_hold_bars > 0 and pos["age"] >= args.max_hold_bars):
                    exit_reason = "time"

                if exit_reason:
                    if i + 1 < len(day_bars):
                        exit_px = float(day_bars.iloc[i + 1]["Open"])
                        exit_ts = pd.Timestamp(day_bars.iloc[i + 1]["ts"]).isoformat()
                    else:
                        exit_px = float(bar["Close"])
                        exit_ts = pd.Timestamp(bar["ts"]).isoformat()

                    trades.append(Trade(
                        symbol=sym,
                        entry_time=pd.Timestamp(pos["entry_ts"]).isoformat(),
                        exit_time=exit_ts,
                        entry=pos["entry"],
                        exit=exit_px,
                        reason=exit_reason,
                        stop=float(pos["stop"]),
                        take=float(pos["take"])
                    ))
                    pos = None
                    break

    return trades

# ---------------------------- IO & Main ----------------------------

def load_5m(path: str) -> pd.DataFrame:
    t = pq.read_table(path).to_pandas()
    if t.empty:
        return t
    if not pd.api.types.is_datetime64_any_dtype(t["ts"]):
        t["ts"] = pd.to_datetime(t["ts"], utc=True)
    else:
        if t["ts"].dt.tz is None:
            t["ts"] = t["ts"].dt.tz_localize("UTC")
    return t[["ts", "Open", "High", "Low", "Close", "Volume"]].sort_values("ts")

def build_daily(df5: pd.DataFrame) -> pd.DataFrame:
    d = resample_daily(df5)
    if d.empty:
        return d
    d["RSI2"]  = rsi2(d["Close"])
    d["RSI14"] = rsi_ewm(d["Close"], n=14)
    d["EMA5"]  = ema(d["Close"], 5)
    d["Z5"]    = zscore(d["Close"], 5)
    d["ATR14"] = atr14_from_daily(d)
    return d

def parse_time(s: str) -> time:
    parts = s.strip().split(":")
    h = int(parts[0]); m = int(parts[1]) if len(parts) > 1 else 0
    sec = int(parts[2]) if len(parts) > 2 else 0
    return time(hour=h, minute=m, second=sec, tzinfo=None)

def main():
    ap = argparse.ArgumentParser(description="Mean-Reversion Walkforward (weekly picks → intraday trades)")
    ap.add_argument("--data-dir", required=True, help="Folder with *.parquet (5m bars)")
    ap.add_argument("--out", default="backtests/report_meanrev_wf.json")
    ap.add_argument("--picks-per-week", type=int, default=10)
    ap.add_argument("--notional", type=float, default=5000.0)

    # Entry sensitivity (prev daily RSI2)
    ap.add_argument("--long-rsi", type=float, default=30.0, help="Enter if RSI2 <= long_rsi AND Close<EMA5")
    ap.add_argument("--short-rsi", type=float, default=5.0,  help="Enter if RSI2 <= short_rsi (strong oversold)")

    # Session window (NY time)
    ap.add_argument("--session-start", default="09:40")
    ap.add_argument("--session-end",   default="15:50")

    # Weekly picklist mid-range bounds
    ap.add_argument("--mid-lo", type=float, default=40.0, help="RSI14 lower bound for 'non-trend'")
    ap.add_argument("--mid-hi", type=float, default=60.0, help="RSI14 upper bound for 'non-trend'")

    # Risk model / exits
    ap.add_argument("--risk", choices=["atr", "pct"], default="atr")
    ap.add_argument("--stop-mult", type=float, default=1.00)
    ap.add_argument("--tp-mult",   type=float, default=1.50)
    ap.add_argument("--trail", choices=["none", "atr", "pct"], default="none")
    ap.add_argument("--trail-mult", type=float, default=1.00)
    ap.add_argument("--breakeven-after", type=float, default=1.00)
    ap.add_argument("--max-hold-bars", type=int, default=0)
    ap.add_argument("--exit-rsi", type=float, default=70.0)
    ap.add_argument("--exit-ema", action="store_true")

    # Back-compat alias (maps to stop-mult if provided)
    ap.add_argument("--atr-mult", type=float, default=None, help="Deprecated: use --stop-mult")

    ap.add_argument("--no-json", action="store_true", help="Don't write summary.json (print only)")
    
    args = ap.parse_args()
    if args.atr_mult is not None and args.stop_mult == 1.00:
        args.stop_mult = args.atr_mult

    tstart = parse_time(args.session_start)
    tend   = parse_time(args.session_end)

    # Find parquet files (recursive)
    files: List[str] = []
    for root, _, fnames in os.walk(args.data_dir):
        for fn in fnames:
            if fn.lower().endswith(".parquet"):
                files.append(os.path.join(root, fn))
    files.sort()

    if not files:
        print("No parquet files found in", args.data_dir)
        return

    sym_to_5m: Dict[str, pd.DataFrame] = {}
    sym_to_daily: Dict[str, pd.DataFrame] = {}
    for fp in files:
        sym = os.path.splitext(os.path.basename(fp))[0]
        try:
            df5 = load_5m(fp)
            if df5.empty:
                continue
            sym_to_5m[sym] = df5
            sym_to_daily[sym] = build_daily(df5)
        except Exception:
            continue

    if not sym_to_5m:
        print("All files empty/unreadable.")
        return

    # Build week list from the union of all daily dates
    indices = [d.index for d in sym_to_daily.values() if not d.empty]
    if not indices:
        print("No daily data to determine weeks.")
        return
    all_dates = pd.DatetimeIndex(sorted({ts for idx in indices for ts in idx}))
    weeks = sorted({week_start_monday(ts) for ts in all_dates})

    trades: List[Trade] = []

    for wk in weeks:
        picks = weekly_picklist(sym_to_daily, wk, args.picks_per_week, args.mid_lo, args.mid_hi)
        if not picks:
            continue
        for sym in picks:
            d5 = sym_to_5m.get(sym)
            dly = sym_to_daily.get(sym)
            if d5 is None or dly is None or d5.empty or dly.empty:
                continue
            trades.extend(run_week_for_symbol(sym, d5, dly, wk, tstart, tend, args))

    # ---------- Outputs ----------
    out_trades = [t.__dict__ for t in trades]
    df = pd.DataFrame(out_trades)

    base_dir = os.path.dirname(args.out) or "."
    os.makedirs(base_dir, exist_ok=True)

    if df.empty:
        summary = {"symbols": 0, "trades": 0, "winrate": 0.0, "avg_ret": 0.0, "sum_ret": 0.0,
                   "avg_pnl_$": 0.0, "sum_pnl_$": 0.0}
        weekly = pd.DataFrame(columns=["week","trades","symbols","winrate","avg_ret","sum_ret","avg_pnl_usd","sum_pnl_usd"])
    else:
        df["ret"] = df["exit"] / df["entry"] - 1.0
        df["pnl_$"] = args.notional * df["ret"]
        summary = {
            "symbols": int(df["symbol"].nunique()),
            "trades":  int(len(df)),
            "winrate": float((df["pnl_$"] > 0).mean()),
            "avg_ret": float(df["ret"].mean()),
            "sum_ret": float(df["ret"].sum()),
            "avg_pnl_$": float(df["pnl_$"].mean()),
            "sum_pnl_$": float(df["pnl_$"].sum()),
        }

        # weekly breakdown by NY week start (Monday)
        df["entry_ts"] = pd.to_datetime(df["entry_time"], utc=True, errors="coerce")
        ny = df["entry_ts"].dt.tz_convert(NY)
        wk_start_dates = ny.dt.date - pd.to_timedelta(ny.dt.weekday, unit="D")
        df["week"] = pd.to_datetime(wk_start_dates)

        weekly = (df.groupby("week")
                    .agg(trades=("symbol","count"),
                         symbols=("symbol","nunique"),
                         winrate=("ret", lambda s: float((s>0).mean())),
                         avg_ret=("ret","mean"),
                         sum_ret=("ret","sum"),
                         avg_pnl_usd=("pnl_$","mean"),
                         sum_pnl_usd=("pnl_$","sum"))
                    .reset_index()
                    .sort_values("week"))

    # write main report
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "trades": out_trades[:5000]}, f, indent=2)

    # write weekly CSV/JSON/MD
    wk_csv  = os.path.join(base_dir, "weekly_summary.csv")
    wk_json = os.path.join(base_dir, "weekly_summary.json")
    wk_md   = os.path.join(base_dir, "weekly_summary.md")

    weekly.to_csv(wk_csv, index=False)
    with open(wk_json, "w") as f:
        json.dump(weekly.to_dict(orient="records"), f, indent=2, default=str)

    # Markdown table for easy reading
    def pct(x):
        try: return f"{100*float(x):.2f}%"
        except: return "n/a"
    lines = ["# Weekly Summary",
             "",
             "| Week (Mon) | Trades | Symbols | Win | Avg Ret | Sum $PnL |",
             "|---|---:|---:|---:|---:|---:|"]
    for _,r in weekly.iterrows():
        lines.append(f"| {str(r['week'])[:10]} | {int(r['trades'])} | {int(r['symbols'])} | {pct(r['winrate'])} | {pct(r['avg_ret'])} | {float(r['sum_pnl_usd']):.2f} |")
    with open(wk_md, "w") as f:
        f.write("\n".join(lines) + "\n")

    # console recap
    print("\n=== SUMMARY ===")
    print(f"symbols={summary.get('symbols',0)}  trades={summary.get('trades',0)}  "
          f"winrate={summary.get('winrate',0):.2%}  avg_ret={summary.get('avg_ret',0):.4f}  "
          f"sum_pnl_usd={summary.get('sum_pnl_usd',0):.2f}")

    if not weekly.empty:
         print("\nWeekly (all):")
        print(weekly.to_string(index=False)
          
    if __name__ == "__main__":
    main()

