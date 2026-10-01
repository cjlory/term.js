#!/usr/bin/env python3
"""
按 dip_backtest.py 的同一规则逐笔回测，统计最长连胜/连败 Top N，写入 analysis/results/dip_streaks.txt。
用法: python3 dip_streaks.py [DROP=0.0016] [TP=0.0016] [SL=0.018] [START=2024-01] [TOPN=10]
"""
import os, sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))
import dip_backtest as bt  # noqa: E402

DROP = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0016
TP = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0016
SL = float(sys.argv[3]) if len(sys.argv) > 3 else 0.018
bt.START = sys.argv[4] if len(sys.argv) > 4 else "2024-01"
TOPN = int(sys.argv[5]) if len(sys.argv) > 5 else 10
fmt = lambda ms: datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M")


def trades(k):
    out, i, n = [], 0, len(k)  # (entry_ts, exit_ts, win)
    while i < n - 1:
        ts, o, h, l, c = k[i]
        if (c - o) / o > -DROP:
            i += 1
            continue
        e = i + 1
        tp, sl = k[e][1] * (1 + TP), k[e][1] * (1 - SL)
        j = e
        while j < n and not (k[j][3] <= sl or k[j][2] >= tp):
            j += 1
        if j == n:
            break
        out.append((k[e][0], k[j][0] + 300_000, k[j][3] > sl))  # 同K线双触按止损
        i = j
    return out


def streaks(t, win):
    res, s = [], None
    for idx, x in enumerate(t + [(0, 0, not win)]):
        if x[2] == win:
            s = idx if s is None else s
        elif s is not None:
            res.append((idx - s, s, idx - 1))
            s = None
    return sorted(res, key=lambda r: -r[0])


def main():
    t = trades(bt.load())
    lines = [f"参数: 跌>={DROP*100:.2f}% 买, 止盈 +{TP*100:.2f}%, 止损 -{SL*100:.2f}%, 起始 {bt.START}; 共 {len(t)} 笔, "
             f"止盈 {sum(x[2] for x in t)} 笔, 止损 {sum(not x[2] for x in t)} 笔"]
    for win, name in ((True, "连胜"), (False, "连败")):
        st = streaks(t, win)
        lines.append(f"\n最长{name} Top {TOPN}（共 {len(st)} 段）:")
        lines.append("名次  笔数  开始(首笔买入,UTC)    结束(末笔平仓,UTC)    持续天数  该段累计(不计费)")
        for r, (ln, a, b) in enumerate(st[:TOPN], 1):
            days = (t[b][1] - t[a][0]) / 86_400_000
            pnl = ln * (TP if win else -SL) * 100
            lines.append(f"{r:>3}  {ln:>5}  {fmt(t[a][0])}  {fmt(t[b][1])}  {days:>7.1f}  {pnl:+.2f}%")
        if win:
            import statistics
            lens = [x[0] for x in st]
            lines.append(f"连胜段长度: 平均 {statistics.mean(lens):.1f}, 中位数 {statistics.median(lens):.0f}")
    text = "\n".join(lines)
    print(text)
    os.makedirs("analysis/results", exist_ok=True)
    open("analysis/results/dip_streaks.txt", "w").write(text + "\n")


if __name__ == "__main__":
    main()
