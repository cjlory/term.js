#!/usr/bin/env python3
"""
BTCUSDT 永续 5m K线回测（数据源: data.binance.vision）。
规则：
  信号：某根已收盘K线 (close-open)/open <= -DROP
  买入：下一根K线开盘价
  止盈：entry*(1+TP)；止损：entry*(1-SL)
  持仓期间不再开新仓；平仓后，从平仓那根K线（含）开始重新找信号。
  同一根K线内同时触及止盈和止损时无法判断先后，按止损计（保守），并单独统计。
用法: python3 dip_backtest.py [DROP=0.0016] [TP=0.0016] [SL=0.018] [START=2024-01]
"""
import calendar, csv, io, sys, urllib.error, urllib.request, zipfile
from collections import defaultdict
from datetime import date, datetime, timezone

BASE = "https://data.binance.vision/data/futures/um"
DROP = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0016
TP = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0016
SL = float(sys.argv[3]) if len(sys.argv) > 3 else 0.018
START = sys.argv[4] if len(sys.argv) > 4 else "2024-01"


def get_rows(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    text = z.read(z.namelist()[0]).decode()
    return [row for row in csv.reader(io.StringIO(text)) if row and row[0].isdigit()]


def month_rows(y, m):
    try:
        return get_rows(f"{BASE}/monthly/klines/BTCUSDT/5m/BTCUSDT-5m-{y}-{m:02d}.zip")
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
    rows = []
    for d in range(1, calendar.monthrange(y, m)[1] + 1):
        if date(y, m, d) >= date.today():
            break
        try:
            rows += get_rows(f"{BASE}/daily/klines/BTCUSDT/5m/BTCUSDT-5m-{y}-{m:02d}-{d:02d}.zip")
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise
    return rows


def load():
    y, m = map(int, START.split("-"))
    t = date.today()
    rows = {}
    while (y, m) < (t.year, t.month):
        for r in month_rows(y, m):
            rows[int(r[0])] = (int(r[0]), *map(float, r[1:5]))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return [rows[k] for k in sorted(rows)]


def main():
    k = load()  # (ts, o, h, l, c)
    fmt = lambda ms: datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M")
    trades = []  # (entry_ts, year, result, bars_held)
    ambiguous = 0
    i, n = 0, len(k)
    while i < n - 1:
        ts, o, h, l, c = k[i]
        if (c - o) / o > -DROP:
            i += 1
            continue
        e = i + 1
        entry = k[e][1]
        tp, sl = entry * (1 + TP), entry * (1 - SL)
        j, res = e, None
        while j < n:
            hit_tp, hit_sl = k[j][2] >= tp, k[j][3] <= sl
            if hit_tp and hit_sl:
                ambiguous += 1
                res = "SL"
            elif hit_sl:
                res = "SL"
            elif hit_tp:
                res = "TP"
            if res:
                break
            j += 1
        if res is None:
            print(f"未平仓: {fmt(k[e][0])} 买入 {entry}，数据结束时仍持仓")
            break
        trades.append((k[e][0], fmt(k[e][0])[:4], res, j - e + 1))
        i = j  # 平仓K线收盘后可作为新信号

    def report(label, ts_list):
        tot = len(ts_list)
        wins = sum(1 for t in ts_list if t[2] == "TP")
        loss = tot - wins
        if not tot:
            return
        gross = wins * TP - loss * SL
        bars = sorted(t[3] for t in ts_list)
        print(f"[{label}] 交易 {tot}  止盈 {wins}  止损 {loss}  止损率 {loss / tot * 100:.2f}%  "
              f"胜率 {wins / tot * 100:.2f}%  持仓中位数 {bars[len(bars) // 2] * 5}分钟  "
              f"单利合计(未计手续费) {gross * 100:+.2f}%")
        for name, fee in (("maker 0.02%x2", 0.0004), ("taker 0.05%x2", 0.001)):
            print(f"      扣手续费 {name}: 单利合计 {(gross - tot * fee) * 100:+.2f}%  "
                  f"每笔期望 {(gross / tot - fee) * 100:+.4f}%")

    print(f"数据: {fmt(k[0][0])} ~ {fmt(k[-1][0])} UTC，{n} 根5m K线")
    print(f"参数: 前一根跌幅>={DROP * 100:.2f}% 下一根开盘买入，止盈 +{TP * 100:.2f}%，止损 -{SL * 100:.2f}%")
    print(f"盈亏平衡胜率(未计手续费): {SL / (SL + TP) * 100:.2f}%")
    print(f"同一根K线同时触及止盈止损(按止损计): {ambiguous}")
    report("全部", trades)
    by_year = defaultdict(list)
    for t in trades:
        by_year[t[1]].append(t)
    for y in sorted(by_year):
        report(y, by_year[y])
    sl_trades = [t for t in trades if t[2] == "SL"]
    longest = max(trades, key=lambda t: t[3])
    print(f"最长持仓: {fmt(longest[0])} 买入，持有 {longest[3] * 5 / 60:.1f} 小时，结果 {longest[2]}")
    print("止损明细(买入时间, 持仓分钟):")
    for t in sl_trades:
        print(f"  {fmt(t[0])}  {t[3] * 5}")


if __name__ == "__main__":
    main()
