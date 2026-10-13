# -*- coding: utf-8 -*-
"""
Bernanke (1983) 实证框架 —— 课堂重构版 Python 实现
====================================================

资料包中的回归方程是"结构化重构"，不应冒充 Bernanke 原文的逐字方程：

    Δln(IP_t) = α + β1·货币/价格控制_t + β2·金融危机代理_t
                + β3·金融危机代理_{t-1} + ε_t

  模型 A：只加货币与价格因素 → 传统货币渠道的基准解释。
  模型 B：再加入银行危机与借款人困境代理变量 → 检验金融变量的额外解释力。

重要声明
--------
本脚本默认使用【课堂演示用模拟数据】（参数由脚本显式设定，带随机噪声），
仅用于展示"先控制货币/价格、再加入金融变量"的回归框架与读表方法，
绝不代表 Bernanke (1983) 的真实估计结果。

如需使用真实月度数据，请准备 CSV（列名见 --csv 帮助），例如从 FRED
(https://fred.stlouisfed.org) 下载工业生产指数、货币存量、价格指数与
银行停业/企业破产序列后自行对齐整理，然后：

    python bernanke_1983.py --csv your_data.csv

运行：python bernanke_1983.py            # 模拟数据 + 回归表 + 两张图
"""

import argparse
import csv
import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 统一设计系统（与 dd_model.py 保持一致）
# ----------------------------------------------------------------------
INK    = "#24292F"
BLUE   = "#2166AC"
RED    = "#B2182B"
GREEN  = "#1B7837"
ORANGE = "#E08214"
GRAY   = "#8A8F98"
GRID   = "#D8DEE4"

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"],
    "axes.unicode_minus": False,
    "font.size": 10.5,
    "axes.titlesize": 12.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.5,
    "axes.labelcolor": INK,
    "axes.edgecolor": GRAY,
    "axes.linewidth": 0.9,
    "xtick.color": INK, "ytick.color": INK,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "figure.facecolor": "white",
    "legend.frameon": False,
})


def style_ax(ax, ygrid=True):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if ygrid:
        ax.grid(axis="y", color=GRID, lw=0.7, alpha=0.8)
        ax.set_axisbelow(True)

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# ----------------------------------------------------------------------
# 1. OLS（纯 numpy 实现，附 t 统计量与正态近似 p 值）
# ----------------------------------------------------------------------
def ols(y, X, names):
    """普通最小二乘：返回系数、标准误、t 值、p 值、R²、调整 R²、n。"""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    df = n - k
    sigma2 = float(resid @ resid) / df
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    tstat = beta / se
    pval = np.array([2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2)))) for t in tstat])
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float(resid @ resid) / tss
    adj_r2 = 1 - (1 - r2) * (n - 1) / df
    return {"names": names, "beta": beta, "se": se, "t": tstat,
            "p": pval, "r2": r2, "adj_r2": adj_r2, "n": n}


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""


def print_table(res_dict, title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    print(f"{'变量':<22}{'系数':>10}{'标准误':>10}{'t值':>8}{'p值':>8}   显著性")
    print("-" * 78)
    for name, b, se, t, p in zip(res_dict["names"], res_dict["beta"],
                                 res_dict["se"], res_dict["t"], res_dict["p"]):
        print(f"{name:<22}{b:>10.4f}{se:>10.4f}{t:>8.2f}{p:>8.3f}   {stars(p)}")
    print("-" * 78)
    print(f"n = {res_dict['n']}   R² = {res_dict['r2']:.3f}   调整R² = {res_dict['adj_r2']:.3f}")


def zscore(x):
    x = np.asarray(x, float)
    return (x - x.mean()) / x.std()


# ----------------------------------------------------------------------
# 2. 课堂演示用模拟数据（1929-01 至 1933-12，共 60 个月）
# ----------------------------------------------------------------------
def simulate_data(seed=1983):
    """
    生成用于课堂演示的模拟月度数据。数据生成过程（DGP）由本脚本显式设定：

        Δln(IP) = 0.010·货币冲击 + 0.005·价格冲击 − 0.012·银行危机代理
                  − 0.008·银行危机代理(滞后) − 0.006·借款人困境代理 + 噪声

    全部变量先标准化再进入 DGP，便于比较各渠道的相对贡献。
    警告：这是演示数据，不是历史数据！
    """
    rng = np.random.default_rng(seed)
    T = 60                                            # 1929-01 .. 1933-12
    t = np.arange(T)

    # --- 货币与价格：大收缩、大通缩，1930 年起加剧 ---
    money_growth = -0.004 - 0.010 * np.clip((t - 12) / 24, 0, 1) + rng.normal(0, 0.004, T)
    price_inflation = -0.001 - 0.006 * np.clip((t - 12) / 24, 0, 1) + rng.normal(0, 0.003, T)

    # --- 银行危机代理（存款停业率量级）：1930-11、1931-03、1931-09、1933-03 几次高潮 ---
    bank_distress = 0.10 + 0.05 * rng.random(T)
    for idx, h in [(22, 1.0), (23, 0.7), (26, 0.5), (32, 0.9), (33, 0.6), (49, 1.1), (50, 1.3)]:
        bank_distress[idx] += h
    bank_distress = zscore(bank_distress)

    # --- 借款人困境代理（企业破产负债量级）：随衰退加深 ---
    borrower_distress = zscore(0.3 * t / T + 0.5 * np.exp(-((t - 24) / 18) ** 2)
                               + rng.normal(0, 0.15, T))

    # --- 产出：按上面的 DGP 生成（先标准化冲击再映射到月度对数变化） ---
    dln_ip = (0.010 * zscore(money_growth) + 0.005 * zscore(price_inflation)
              - 0.012 * bank_distress - 0.008 * np.r_[0.0, bank_distress[:-1]]
              - 0.006 * borrower_distress + rng.normal(0, 0.008, T))

    return {
        "date": [f"{1929 + i // 12}-{i % 12 + 1:02d}" for i in t],
        "dln_ip": dln_ip,
        "money_growth": money_growth,
        "price_inflation": price_inflation,
        "bank_distress": bank_distress,
        "borrower_distress": borrower_distress,
    }


def load_csv(path):
    """读取真实数据 CSV：需要列 date, dln_ip, money_growth, price_inflation, bank_distress, borrower_distress。"""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    need = ["dln_ip", "money_growth", "price_inflation", "bank_distress", "borrower_distress"]
    data = {c: np.array([float(r[c]) for r in rows]) for c in need}
    data["date"] = [r.get("date", "") for r in rows]
    return data


# ----------------------------------------------------------------------
# 3. 回归与图
# ----------------------------------------------------------------------
def run_regressions(data):
    y = data["dln_ip"]
    bank_lag = np.r_[0.0, data["bank_distress"][:-1]]

    XA = np.column_stack([np.ones(len(y)), zscore(data["money_growth"]), zscore(data["price_inflation"])])
    XB = np.column_stack([XA, data["bank_distress"], bank_lag, data["borrower_distress"]])

    resA = ols(y, XA, ["常数项", "货币增长(标准化)", "价格通胀(标准化)"])
    resB = ols(y, XB, ["常数项", "货币增长(标准化)", "价格通胀(标准化)",
                       "银行危机代理", "银行危机代理(滞后1)", "借款人困境代理"])
    return resA, resB


def fig_timeseries(data, path):
    """机制图风格：工业产出变化 vs 金融困境指数（双轴，标准化），标注三次银行恐慌。"""
    x = np.arange(len(data["dln_ip"]))
    distress = zscore(data["bank_distress"]) + 0.6 * zscore(data["borrower_distress"])

    fig, ax1 = plt.subplots(figsize=(11.5, 4.8), constrained_layout=True)
    ax1.bar(x, data["dln_ip"], width=0.8,
            color=np.where(data["dln_ip"] < 0, "#D89C9C", "#9CBF9C"),
            label="Δln(工业生产)（左轴）")
    ax1.axhline(0, color=GRAY, lw=0.8)
    ax1.set_ylabel("Δln(IP)")
    ax1.set_xlabel("月份（1929-01 → 1933-12）")
    ax1.set_ylim(-0.06, 0.05)
    style_ax(ax1)

    ax2 = ax1.twinx()
    ax2.plot(x, distress, color=BLUE, lw=2.2, label="金融困境指数（右轴，标准化）")
    ax2.set_ylabel("金融困境指数（标准化）", color=BLUE)
    ax2.tick_params(axis="y", labelcolor=BLUE)
    ax2.spines["top"].set_visible(False)
    ymax = float(distress.max())
    ax2.set_ylim(-1.6, ymax * 1.32)

    for idx, txt, dx in [(22, "1930-11 银行停业潮", -8),
                         (32, "1931-09 恐慌扩散", -8),
                         (50, "1933-03 最终挤兑", -13)]:
        ax2.annotate(txt, xy=(idx, distress[idx]),
                     xytext=(idx + dx, min(distress[idx] + ymax * 0.14, ymax * 1.12)),
                     fontsize=8.5, color=BLUE, ha="center",
                     arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.9))

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [Patch(facecolor="#9CBF9C", label="Δln(工业生产)（左轴）"),
               Line2D([0], [0], color=BLUE, lw=2.2, label="金融困境指数（右轴，标准化）")]
    ax1.legend(handles=handles, fontsize=9, loc="lower left")
    ax1.set_title("金融困境与工业生产共同恶化 —— 描述性证据，不等于因果识别")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_fit(data, resA, resB, path):
    x = np.arange(len(data["dln_ip"]))
    bank_lag = np.r_[0.0, data["bank_distress"][:-1]]
    XA = np.column_stack([np.ones(len(x)), zscore(data["money_growth"]),
                          zscore(data["price_inflation"])])
    XB = np.column_stack([XA, data["bank_distress"], bank_lag, data["borrower_distress"]])

    fig, ax = plt.subplots(figsize=(11.5, 4.8), constrained_layout=True)
    ax.plot(x, data["dln_ip"], color=INK, lw=1.8, label="实际（模拟）Δln(IP)")
    ax.plot(x, resA["beta"] @ XA.T, lw=1.6, ls=(0, (4, 3)), color=ORANGE,
            label=f"模型 A（仅货币/价格）  R²={resA['r2']:.2f}")
    ax.plot(x, resB["beta"] @ XB.T, lw=2.0, color=BLUE,
            label=f"模型 B（+ 金融变量）      R²={resB['r2']:.2f}")
    ax.axhline(0, color=GRAY, lw=0.7)
    ax.set_xlabel("月份（1929-01 → 1933-12）")
    ax.set_ylabel("Δln(IP)")
    ax.set_title("加入金融变量后，产出收缩的大部分被解释了（模拟数据，仅演示框架）")
    ax.legend(fontsize=9.5, loc="lower left")
    ax.set_ylim(-0.055, 0.045)
    style_ax(ax)
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ----------------------------------------------------------------------
# 主程序
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Bernanke (1983) 课堂重构回归（默认模拟数据）")
    ap.add_argument("--csv", help="真实数据 CSV 路径（列：date, dln_ip, money_growth, "
                                  "price_inflation, bank_distress, borrower_distress）")
    ap.add_argument("--seed", type=int, default=1983, help="模拟数据随机种子")
    args = ap.parse_args()

    if args.csv:
        data = load_csv(args.csv)
        src = f"真实数据（用户提供）：{args.csv}"
    else:
        data = simulate_data(args.seed)
        src = "课堂演示用模拟数据（非历史数据，仅展示回归框架）"

    resA, resB = run_regressions(data)

    print("=" * 78)
    print("Bernanke (1983) 非货币信用渠道 —— 课堂重构版回归")
    print("数据来源：" + src)
    print("=" * 78)

    print_table(resA, "模型 A：Δln(IP) ~ 货币 + 价格（传统货币渠道基准）")
    print_table(resB, "模型 B：在 A 基础上加入银行危机、借款人困境代理及其滞后项")

    print("\n【怎么读表】（对应资料包 4.3 节）")
    fin_names = ["银行危机代理", "银行危机代理(滞后1)", "借款人困境代理"]
    idx = [resB["names"].index(n) for n in fin_names]
    sig = sum(1 for i in idx if resB["p"][i] < 0.05)
    neg = sum(1 for i in idx if resB["beta"][i] < 0)
    print(f"  1. 模型 B 中 {sig}/3 个金融变量在 5% 水平显著，且 {neg}/3 符号为负")
    print("     → '金融困境越严重、产出越低'，金融变量有货币/价格之外的额外解释力。")
    print(f"  2. 滞后项显著为负 → 信用网络重建需要时间，冲击具有持续性。")
    print(f"  3. R² 从 {resA['r2']:.3f} 升到 {resB['r2']:.3f} → 金融渠道改善了对产出收缩的解释。")
    print("  4. 表述边界：只能说'支持非货币信用渠道'，不能说'证明了银行危机导致大萧条'")
    print("     （时间序列内生性，反向因果未被排除）。")

    p1 = os.path.join(FIG_DIR, "fig4_bernanke_timeseries.png")
    p2 = os.path.join(FIG_DIR, "fig5_bernanke_fit.png")
    fig_timeseries(data, p1)
    fig_fit(data, resA, resB, p2)
    print("\n图表已保存：")
    print("  - figures/fig4_bernanke_timeseries.png")
    print("  - figures/fig5_bernanke_fit.png")


if __name__ == "__main__":
    main()
