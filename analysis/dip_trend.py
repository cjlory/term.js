#!/usr/bin/env python3
"""
在 dip_grid.py 逻辑上加趋势过滤：只有信号K线收盘时满足过滤条件才买入（只用当时已知的数据，无未来函数）。
输出 analysis/results/dip_trend.csv，含全期和分年度统计，用于检查结果是否每年都成立。
"""
import csv, itertools, os, sys
from datetime import datetime, timezone
import numpy as np
from numba import njit

sys.path.insert(0, os.path.dirname(__file__))
from dip_backtest import load  # noqa: E402

DROPS = [0.0005, 0.001, 0.0016, 0.0025, 0.004, 0.006, 0.008, 0.01]
TPS = [0.001, 0.0016, 0.0025, 0.004, 0.006, 0.01, 0.015, 0.02, 0.03, 0.05]
SLS = [0.005, 0.01, 0.015, 0.018, 0.025, 0.03, 0.05, 0.07, 0.1]
MAKER, TAKER = 0.0004, 0.001
H4, D1, W1 = 48, 288, 2016  # 5m K线根数


@njit(cache=True)
def ema(x, n):
    out = np.empty_like(x)
    a = 2.0 / (n + 1)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


@njit(cache=True)
def run(o, h, l, c, mask, drop, tp, sl):
    n = len(o)
    ent = np.empty(n, np.int64)
    res = np.empty(n, np.int8)
    k = 0
    i = 0
    while i < n - 1:
        if not mask[i] or (c[i] - o[i]) / o[i] > -drop:
            i += 1
            continue
        e = i + 1
        tpp = o[e] * (1 + tp)
        slp = o[e] * (1 - sl)
        j = e
        r = 0
        while j < n:
            if l[j] <= slp:
                r = -1
            elif h[j] >= tpp:
                r = 1
            if r != 0:
                break
            j += 1
        if r == 0:
            break
        ent[k] = e
        res[k] = r
        k += 1
        i = j
    return ent[:k], res[:k]


def main():
    a = np.array(load(), dtype=np.float64)
    ts, o, h, l, c = (a[:, i].copy() for i in range(5))
    year = np.array([datetime.fromtimestamp(t / 1000, timezone.utc).year for t in ts])
    years = sorted(set(year.tolist()))
    warm = np.arange(len(c)) >= W1 + D1  # 预热期内不交易，保证各过滤器样本区间一致
    e4, e1d, e1w = ema(c, H4), ema(c, D1), ema(c, W1)
    lag = lambda x, n: np.concatenate([np.full(n, np.nan), x[:-n]])
    filters = {
        "none": np.ones_like(c, bool),
        "above_ema4h": c > e4,
        "above_ema1d": c > e1d,
        "above_ema1w": c > e1w,
        "below_ema1d": c < e1d,
        "ema1d_rising": e1d > lag(e1d, D1),
        "ema1d_gt_ema1w": e1d > e1w,
        "above_ema1d_and_1d_gt_1w": (c > e1d) & (e1d > e1w),
        "mom24h_pos": c > lag(c, D1),
        "mom7d_pos": c > lag(c, W1),
    }
    days = (ts[-1] - ts[0]) / 86_400_000
    os.makedirs("analysis/results", exist_ok=True)
    out = "analysis/results/dip_trend.csv"
    head = ["filter", "drop_pct", "tp_pct", "sl_pct", "trades", "trades_per_day", "sl_rate_pct", "theory_sl_rate_pct",
            "edge_pp", "gross_per_trade_pct", "per_trade_maker_pct", "sum_maker_pct", "maxdd_maker_pct",
            "compound_maker_x", "per_trade_taker_pct", "sum_taker_pct", "time_in_filter_pct"]
    for y in years:
        head += [f"trades_{y}", f"sl_rate_{y}", f"edge_{y}", f"per_trade_maker_{y}"]
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(head)
        for fname, m in filters.items():
            mask = (m & warm).astype(np.bool_)
            tif = mask[warm].mean() * 100
            for drop, tp, sl in itertools.product(DROPS, TPS, SLS):
                ent, res = run(o, h, l, c, mask, drop, tp, sl)
                t = len(res)
                if t == 0:
                    continue
                pnl = np.where(res == 1, tp, -sl)
                theory = tp / (tp + sl) * 100
                slr = (res == -1).mean() * 100
                net = pnl - MAKER
                cum = np.cumsum(net)
                dd = (np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:] - cum).max()
                row = [fname, drop * 100, tp * 100, sl * 100, t, round(t / days, 2), round(slr, 3), round(theory, 3),
                       round(theory - slr, 3), round(pnl.mean() * 100, 4), round(net.mean() * 100, 4),
                       round(net.sum() * 100, 2), round(dd * 100, 2), round(float(np.prod(1 + net)), 4),
                       round((pnl.mean() - TAKER) * 100, 4), round((pnl - TAKER).sum() * 100, 2), round(tif, 1)]
                ey = year[ent]
                for y in years:
                    sel = ey == y
                    ny = int(sel.sum())
                    if ny:
                        sy = (res[sel] == -1).mean() * 100
                        row += [ny, round(sy, 3), round(theory - sy, 3), round((pnl[sel].mean() - MAKER) * 100, 4)]
                    else:
                        row += [0, "", "", ""]
                w.writerow(row)
    print(f"{len(c)} 根K线，{len(filters)} 个过滤器 × {len(DROPS) * len(TPS) * len(SLS)} 组参数 -> {out}")


if __name__ == "__main__":
    main()
