#!/usr/bin/env python3
"""
统计 Binance BTCUSDT 永续合约 1h K线近两年"光头"出现概率（数据源: data.binance.vision）。
  阳线 (close > open)：最低价 >= 开盘价 * (1 - TH)
  阴线 (close < open)：最高价 <= 开盘价 * (1 + TH)
  十字星 (close == open) 不计入光头。
用法: python3 bald_candle.py [阈值,默认0.001] [月数,默认24]
"""
import calendar, csv, io, sys, urllib.error, urllib.request, zipfile
from datetime import date, datetime, timezone

BASE = "https://data.binance.vision/data/futures/um"
TH = float(sys.argv[1]) if len(sys.argv) > 1 else 0.001
MONTHS = int(sys.argv[2]) if len(sys.argv) > 2 else 24


def get_rows(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    text = z.read(z.namelist()[0]).decode()
    return [row for row in csv.reader(io.StringIO(text)) if row and row[0].isdigit()]


def month_rows(y, m):
    try:
        return get_rows(f"{BASE}/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-{y}-{m:02d}.zip")
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
    rows = []  # 月度文件尚未发布时，逐日拼接
    for d in range(1, calendar.monthrange(y, m)[1] + 1):
        if date(y, m, d) >= date.today():
            break
        try:
            rows += get_rows(f"{BASE}/daily/klines/BTCUSDT/1h/BTCUSDT-1h-{y}-{m:02d}-{d:02d}.zip")
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise
    return rows


def main():
    t = date.today()
    y, m = t.year, t.month
    months = []
    for _ in range(MONTHS):  # 最近 MONTHS 个完整自然月
        m -= 1
        if m == 0:
            y, m = y - 1, 12
        months.append((y, m))
    rows = {}
    for y, m in sorted(months):
        for r in month_rows(y, m):
            rows[int(r[0])] = r
    rows = [rows[k] for k in sorted(rows)]

    n = up = dn = flat = up_bald = dn_bald = 0
    for r in rows:
        o, h, l, c = map(float, r[1:5])
        n += 1
        if c > o:
            up += 1
            up_bald += l >= o * (1 - TH)
        elif c < o:
            dn += 1
            dn_bald += h <= o * (1 + TH)
        else:
            flat += 1

    fmt = lambda ms: datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M")
    pct = lambda a, b: f"{a / b * 100:.2f}%" if b else "n/a"
    print(f"区间(UTC): {fmt(int(rows[0][0]))} ~ {fmt(int(rows[-1][0]))}   阈值: {TH * 100:.2f}%")
    print(f"总K线: {n}  阳线: {up}  阴线: {dn}  十字星: {flat}")
    print(f"光头阳线: {up_bald}  占阳线 {pct(up_bald, up)}")
    print(f"光头阴线: {dn_bald}  占阴线 {pct(dn_bald, dn)}")
    print(f"光头合计: {up_bald + dn_bald}  占全部K线 {pct(up_bald + dn_bald, n)}")


if __name__ == "__main__":
    main()
