#!/usr/bin/env python3
"""
dip_backtest.py 同一逻辑的参数网格回测（numba 加速），结果写入 analysis/results/dip_grid.csv。
  信号：已收盘K线 (close-open)/open <= -DROP；下一根开盘买入
  止盈 entry*(1+TP)，止损 entry*(1-SL)；持仓期间不开新仓；平仓K线可作为新信号
  同一根K线同时触及止盈止损按止损计
"""
import csv, itertools, os, sys
import numpy as np
from numba import njit

sys.path.insert(0, os.path.dirname(__file__))
from dip_backtest import load  # noqa: E402

DROPS = [0.0005, 0.001, 0.0016, 0.002, 0.0025, 0.003, 0.004, 0.005, 0.006, 0.008, 0.01, 0.015]
TPS = [0.001, 0.0016, 0.002, 0.0025, 0.003, 0.004, 0.005, 0.006, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05]
SLS = [0.003, 0.005, 0.007, 0.01, 0.0125, 0.015, 0.018, 0.02, 0.025, 0.03, 0.04, 0.05, 0.07, 0.1]
FEES = {"maker": 0.0004, "taker": 0.001}


@njit(cache=True)
def run(o, h, l, c, drop, tp, sl, fee):
    n = len(o)
    trades = wins = amb = 0
    hold = 0
    maxhold = 0
    cum = peak = maxdd = 0.0
    eq = 1.0
    i = 0
    while i < n - 1:
        if (c[i] - o[i]) / o[i] > -drop:
            i += 1
            continue
        e = i + 1
        tpp = o[e] * (1 + tp)
        slp = o[e] * (1 - sl)
        j = e
        res = 0
        while j < n:
            ht = h[j] >= tpp
            hs = l[j] <= slp
            if ht and hs:
                amb += 1
                res = -1
            elif hs:
                res = -1
            elif ht:
                res = 1
            if res != 0:
                break
            j += 1
        if res == 0:
            break
        trades += 1
        r = (tp if res == 1 else -sl) - fee
        if res == 1:
            wins += 1
        cum += r
        eq *= 1 + r
        if cum > peak:
            peak = cum
        if peak - cum > maxdd:
            maxdd = peak - cum
        hb = j - e + 1
        hold += hb
        if hb > maxhold:
            maxhold = hb
        i = j
    return trades, wins, amb, cum, maxdd, eq, hold, maxhold


def main():
    k = load()
    a = np.array(k, dtype=np.float64)
    o, h, l, c = a[:, 1].copy(), a[:, 2].copy(), a[:, 3].copy(), a[:, 4].copy()
    days = (a[-1, 0] - a[0, 0]) / 86_400_000
    os.makedirs("analysis/results", exist_ok=True)
    out = "analysis/results/dip_grid.csv"
    cols = ["drop_pct", "tp_pct", "sl_pct", "trades", "trades_per_day", "tp_hits", "sl_hits",
            "sl_rate_pct", "breakeven_winrate_pct", "same_bar_both", "avg_hold_min", "max_hold_hours"]
    for f in FEES:
        cols += [f"sum_{f}_pct", f"per_trade_{f}_pct", f"maxdd_{f}_pct", f"compound_{f}_x"]
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols + ["gross_sum_pct", "gross_per_trade_pct"])
        for drop, tp, sl in itertools.product(DROPS, TPS, SLS):
            t, wn, amb, gcum, _, _, hold, mh = run(o, h, l, c, drop, tp, sl, 0.0)
            if t == 0:
                continue
            row = [drop * 100, tp * 100, sl * 100, t, round(t / days, 2), wn, t - wn,
                   round((t - wn) / t * 100, 3), round(sl / (sl + tp) * 100, 3), amb,
                   round(hold / t * 5, 1), round(mh * 5 / 60, 1)]
            for fee in FEES.values():
                _, _, _, cum, dd, eq, _, _ = run(o, h, l, c, drop, tp, sl, fee)
                row += [round(cum * 100, 2), round(cum / t * 100, 4), round(dd * 100, 2), round(eq, 4)]
            row += [round(gcum * 100, 2), round(gcum / t * 100, 4)]
            w.writerow(row)
    print(f"数据 {len(k)} 根5m K线，{days:.0f} 天；组合 {len(DROPS) * len(TPS) * len(SLS)}；输出 {out}")


if __name__ == "__main__":
    main()
