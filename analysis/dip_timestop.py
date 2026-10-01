#!/usr/bin/env python3
"""
在 dip_backtest.py 规则上加"时间止损"：买入后 N 根K线内未触发止盈/止损，则在第 N+1 根K线开盘价市价卖出。
  N=2：前10分钟没结果，就在开仓后第3根K线开盘卖（用户方案）；N=3：第3根收盘（=第4根开盘）卖。N=0 表示不加时间止损。
手续费口径：maker=全挂单 0.02%x2；taker=全吃单 0.05%x2；mix=市价开仓0.05% + 止盈挂单0.02% / 止损和时间止损市价0.05%。
输出 analysis/results/dip_timestop.csv
"""
import csv, itertools, os, sys
from datetime import datetime, timezone
import numpy as np
from numba import njit

sys.path.insert(0, os.path.dirname(__file__))
from dip_backtest import load  # noqa: E402

DROPS = [0.001, 0.0016, 0.0025, 0.004]
TPS = [0.001, 0.0016, 0.002, 0.0025, 0.003, 0.005]
SLS = [0.005, 0.01, 0.018, 0.03]
NS = [0, 1, 2, 3, 4, 6, 12, 24, 48, 288]


@njit(cache=True)
def run(o, h, l, c, drop, tp, sl, nmax):
    n = len(o)
    ent = np.empty(n, np.int64)
    kind = np.empty(n, np.int8)  # 1止盈 -1止损 0时间止损
    ret = np.empty(n, np.float64)
    k = 0
    i = 0
    while i < n - 1:
        if (c[i] - o[i]) / o[i] > -drop:
            i += 1
            continue
        e = i + 1
        p = o[e]
        tpp, slp = p * (1 + tp), p * (1 - sl)
        j = e
        r = 2
        while j < n:
            if nmax > 0 and j - e >= nmax:
                r = 0
                ret[k] = o[j] / p - 1
                break
            if l[j] <= slp:
                r = -1
                ret[k] = -sl
                break
            if h[j] >= tpp:
                r = 1
                ret[k] = tp
                break
            j += 1
        if r == 2:
            break
        ent[k] = e
        kind[k] = r
        k += 1
        i = j  # 平仓发生在第j根K线内（或开盘），第j根收盘后可作为新信号
    return ent[:k], kind[:k], ret[:k]


def main():
    a = np.array(load(), dtype=np.float64)
    ts, o, h, l, c = (a[:, i].copy() for i in range(5))
    year = np.array([datetime.fromtimestamp(t / 1000, timezone.utc).year for t in ts])
    years = sorted(set(year.tolist()))
    os.makedirs("analysis/results", exist_ok=True)
    head = ["drop_pct", "tp_pct", "sl_pct", "time_stop_bars", "trades", "tp_hits", "sl_hits", "time_exits",
            "sl_rate_pct", "time_exit_rate_pct", "avg_time_exit_ret_pct", "time_exit_win_share_pct",
            "gross_per_trade_pct", "maker_per_trade_pct", "mix_per_trade_pct", "taker_per_trade_pct",
            "sum_maker_pct", "sum_mix_pct", "maxdd_mix_pct"] + [f"mix_per_trade_{y}" for y in years]
    with open("analysis/results/dip_timestop.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(head)
        for drop, tp, sl, nb in itertools.product(DROPS, TPS, SLS, NS):
            ent, kind, ret = run(o, h, l, c, drop, tp, sl, nb)
            t = len(ret)
            if not t:
                continue
            te = kind == 0
            fee_mix = 0.0005 + np.where(kind == 1, 0.0002, 0.0005)
            mix = ret - fee_mix
            cum = np.cumsum(mix)
            dd = (np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:] - cum).max()
            ey = year[ent]
            row = [drop * 100, tp * 100, sl * 100, nb, t, int((kind == 1).sum()), int((kind == -1).sum()), int(te.sum()),
                   round((kind == -1).mean() * 100, 3), round(te.mean() * 100, 2),
                   round(ret[te].mean() * 100, 4) if te.any() else "", round((ret[te] > 0).mean() * 100, 1) if te.any() else "",
                   round(ret.mean() * 100, 4), round((ret.mean() - 0.0004) * 100, 4), round(mix.mean() * 100, 4),
                   round((ret.mean() - 0.001) * 100, 4), round((ret - 0.0004).sum() * 100, 2), round(cum[-1] * 100, 2),
                   round(dd * 100, 2)]
            row += [round(mix[ey == y].mean() * 100, 4) if (ey == y).any() else "" for y in years]
            w.writerow(row)
    print("done")


if __name__ == "__main__":
    main()
