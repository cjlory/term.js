#!/usr/bin/env python3
"""
按 dip_backtest.py 同一规则逐笔回测，并记录每笔在买入时刻已知的市场特征（只用信号K线及之前的数据），
输出 analysis/results/dip_trades.csv，用于分析止损单有没有规律。
"""
import os, sys
from datetime import datetime, timezone
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import dip_backtest as bt  # noqa: E402

DROP, TP, SL = 0.0016, 0.0016, 0.018


def load_full():
    y, m = 2023, 12  # 多取一个月做指标预热
    t = datetime.now(timezone.utc).date()
    rows = {}
    while (y, m) < (t.year, t.month):
        for r in bt.month_rows(y, m):
            rows[int(r[0])] = [float(x) for x in r[:6]] + [float(r[8]), float(r[9])]
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return np.array([rows[k] for k in sorted(rows)])


def roll_mean(x, n):
    c = np.concatenate([[0.0], np.cumsum(x)])
    out = np.full(len(x), np.nan)
    out[n:] = (c[n:-1] - c[:-n - 1]) / n  # 不含当根：x[i-n:i]
    return out


def ema(x, n):
    a, out = 2 / (n + 1), np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def main():
    A = load_full()
    ts, o, h, l, c, v, ntr, tb = (A[:, i] for i in range(8))
    n = len(c)
    r5 = np.concatenate([[0.0], np.diff(np.log(c))])
    lag = lambda x, k: np.concatenate([np.full(k, np.nan), x[:-k]])
    vol_avg24 = roll_mean(v, 288)
    rv1h = np.sqrt(roll_mean(r5 ** 2, 12) * 12) * 100  # 含当根前12根的近1小时波动
    rv24 = np.sqrt(roll_mean(r5 ** 2, 288) * 12) * 100  # 同口径（每小时）便于比较
    e1d = ema(c, 288)
    hi24 = np.array([h[max(0, i - 287):i + 1].max() for i in range(n)])
    lo24 = np.array([l[max(0, i - 287):i + 1].min() for i in range(n)])
    down = (c < o).astype(int)
    dstreak = np.zeros(n, int)
    for i in range(1, n):
        dstreak[i] = dstreak[i - 1] + 1 if down[i] else 0
    start = int(np.searchsorted(ts, datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000))
    rows, i = [], start
    while i < n - 1:
        if (c[i] - o[i]) / o[i] > -DROP:
            i += 1
            continue
        e = i + 1
        ent = o[e]
        tpp, slp = ent * (1 + TP), ent * (1 - SL)
        j, lo_seen, hi_seen = e, ent, ent
        while j < n and not (l[j] <= slp or h[j] >= tpp):
            lo_seen, hi_seen = min(lo_seen, l[j]), max(hi_seen, h[j])
            j += 1
        if j == n:
            break
        win = l[j] > slp
        rng = h[i] - l[i]
        dt = datetime.fromtimestamp(ts[e] / 1000, timezone.utc)
        rows.append(dict(
            entry_time=dt.strftime("%Y-%m-%d %H:%M"), year=dt.year, hour=dt.hour, weekday=dt.weekday(),
            win=int(win), bars_held=j - e + 1,
            mfe_before_exit_pct=(hi_seen / ent - 1) * 100, mae_before_exit_pct=(lo_seen / ent - 1) * 100,
            sig_drop_pct=(c[i] / o[i] - 1) * 100, sig_range_pct=rng / o[i] * 100,
            sig_lower_wick_pct=(min(o[i], c[i]) - l[i]) / o[i] * 100,
            sig_close_pos=(c[i] - l[i]) / rng if rng else 0.5,
            sig_vol_ratio=v[i] / vol_avg24[i], sig_trades_ratio=ntr[i] / roll_mean(ntr, 288)[i],
            sig_taker_buy_ratio=tb[i] / v[i] if v[i] else 0.5,
            down_streak=dstreak[i],
            ret_15m_pct=(c[i] / c[i - 3] - 1) * 100, ret_1h_pct=(c[i] / c[i - 12] - 1) * 100,
            ret_4h_pct=(c[i] / c[i - 48] - 1) * 100, ret_24h_pct=(c[i] / c[i - 288] - 1) * 100,
            ret_7d_pct=(c[i] / c[i - 2016] - 1) * 100,
            rv_1h_pct=rv1h[i], rv_24h_pct=rv24[i], rv_ratio=rv1h[i] / rv24[i],
            dist_ema1d_pct=(c[i] / e1d[i] - 1) * 100,
            dist_24h_high_pct=(c[i] / hi24[i] - 1) * 100, dist_24h_low_pct=(c[i] / lo24[i] - 1) * 100,
            gap_bars_since_prev_exit=None,
        ))
        rows[-1]["_e"], rows[-1]["_j"] = e, j
        i = j
    import csv
    for k, r in enumerate(rows):
        r["gap_bars_since_prev_exit"] = r["_e"] - rows[k - 1]["_j"] if k else ""
        r["prev_win"] = rows[k - 1]["win"] if k else ""
    os.makedirs("analysis/results", exist_ok=True)
    keys = [k for k in rows[0] if not k.startswith("_")]
    with open("analysis/results/dip_trades.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})
    print(f"{len(rows)} 笔, 止损 {sum(1 - r['win'] for r in rows)}")


if __name__ == "__main__":
    main()
