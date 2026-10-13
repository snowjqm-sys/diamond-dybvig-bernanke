# -*- coding: utf-8 -*-
"""
Diamond–Dybvig (1983) 银行挤兑模型 —— 课堂版 Python 实现
=========================================================

按照资料包的"三期框架"实现三部分内容：

  1. 最优存款合同（流动性保险）：银行如何在资源约束下选择 (c1, c2)，
     并与自给自足（autarky）比较，展示金融中介的保险价值。
  2. 挤兑博弈（先到先得 + 序贯服务）：晚期消费者"等"还是"跑"的收益比较，
     展示多重均衡 —— 正常均衡与自我实现的挤兑均衡。
  3. 存款保险：可信担保如何排除挤兑均衡（同时提示道德风险，见 README）。

所有参数均为课堂演示校准，不对应任何真实银行数据。
公式均为原论文思想的课堂重构，不是 Diamond–Dybvig (1983) 的逐字复刻。

运行：python dd_model.py            # 输出数值结果并保存 figures/ 下的图
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# ----------------------------------------------------------------------
# 0. 基本设定
# ----------------------------------------------------------------------
# t=0 每位储户 1 单位禀赋，银行投入长期项目：
#   - 持有到 t=2 产出 R > 1；
#   - 若在 t=1 提前清算，每单位只能回收 1（长期资产非流动性）。
# 储户：比例 lam 是早期消费者（t=1 消费），1-lam 是晚期消费者（t=2 消费）。
# 类型 t=1 才实现，银行无法事先观察。

def util(c, gamma):
    """CRRA 效用 u(c) = c^(1-gamma)/(1-gamma)，gamma=1 时退化为 ln(c)。"""
    c = np.maximum(c, 1e-12)
    if np.isclose(gamma, 1.0):
        return np.log(c)
    return (np.power(c, 1.0 - gamma) - 1.0) / (1.0 - gamma)


def optimal_contract(R=2.0, lam=0.4, gamma=2.0):
    """
    求解最优存款合同 (c1, c2)。

    max  lam*u(c1) + (1-lam)*u(c2)
    s.t. lam*c1 + (1-lam)*c2/R <= 1        （按 t=0 资源折算的支付承诺）

    CRRA 下由 FOC u'(c1) = u'(c2)/R 得 c2 = c1 * R^(1/gamma)，
    代入约束得 c1 = 1 / (lam + (1-lam) * R^(1/gamma - 1))。
    """
    c1 = 1.0 / (lam + (1.0 - lam) * R ** (1.0 / gamma - 1.0))
    c2 = c1 * R ** (1.0 / gamma)
    eu_bank = lam * util(c1, gamma) + (1 - lam) * util(c2, gamma)
    eu_autarky = lam * util(1.0, gamma) + (1 - lam) * util(R, gamma)
    return {
        "c1": c1, "c2": c2,
        "eu_bank": eu_bank, "eu_autarky": eu_autarky,
        "gain": eu_bank - eu_autarky,
        "viable": lam * c1 <= 1.0,   # 只有早期消费者提款时银行能否兑付
    }


# ----------------------------------------------------------------------
# 1. 挤兑博弈：晚期消费者"等"还是"跑"
# ----------------------------------------------------------------------
# 设晚期消费者中去挤兑的比例为 m（信念的体现）。银行按需清算资产：
# 每兑付一位提款者 c1，需清算 c1 单位资产（t=1 每单位回收 1）。
#
# 跑（t=1 提款）的收益：资金足够时拿到 c1；不足时按随机排队位置的
#   兑付概率折算，人均期望 = min(c1, 1/(lam+m(1-lam)))。
# 等（t=2 消费）的收益：剩余资产 (1-(lam+m(1-lam))*c1)*R 按等待者人数均分。
#   m=0 时恰好等于 c2（资源约束取等号）；随 m 上升而下降，
#   当 (lam+m(1-lam))*c1 = 1（被迫全额清算）时降为 0。
#
# 两条收益曲线的交点 m* = (c2-c1)/(c1*(R-1)) 即"跑/等无差异"的挤兑规模：
#   预期挤兑比例超过 m* 时，"跑"是晚期消费者的最优反应 —— 挤兑自我实现。

def run_payoffs(R=2.0, lam=0.4, gamma=2.0, m_grid=None):
    """返回 (m_grid, 跑的收益, 等的收益, 无差异阈值 m*, 全额清算点 m_bar)。"""
    k = optimal_contract(R, lam, gamma)
    c1, c2 = k["c1"], k["c2"]
    if m_grid is None:
        m_grid = np.linspace(0.0, 1.0, 501)

    claims = lam + m_grid * (1.0 - lam)                      # t=1 提款者占人口比例
    payoff_run = np.minimum(c1, 1.0 / np.maximum(claims, 1e-12))
    payoff_wait = (1.0 - claims * c1) * R / np.maximum((1.0 - m_grid) * (1.0 - lam), 1e-12)
    payoff_wait = np.clip(payoff_wait, 0.0, None)
    payoff_wait[claims * c1 >= 1.0] = 0.0                    # 被迫全额清算：等待者一无所有

    m_star = (c2 - c1) / (c1 * (R - 1.0))                    # 跑/等无差异的挤兑比例
    m_bar = (1.0 / c1 - lam) / (1.0 - lam)                   # 银行被迫全额清算的挤兑比例
    return m_grid, payoff_run, payoff_wait, m_star, m_bar


def simulate_run(N=200_000, lam=0.4, R=2.0, gamma=2.0, m_panic=0.0, insured=False, seed=42):
    """
    蒙特卡洛模拟一次挤兑事件（序贯服务 / 先到先得）。

    m_panic  : 晚期消费者中去挤兑的比例（恐慌信念的体现）
    insured  : 政府全额担保存款（等待者稳拿 c2）

    返回 (每位储户的实现消费, 统计字典)。
    """
    rng = np.random.default_rng(seed)
    k = optimal_contract(R, lam, gamma)
    c1, c2 = k["c1"], k["c2"]

    early = rng.random(N) < lam                       # 早期消费者：t=1 必须提款
    late_run = (~early) & (rng.random(N) < m_panic)   # 晚期消费者：按恐慌比例加入挤兑
    withdraw = early | late_run

    payment = np.zeros(N)
    order = rng.permutation(N)                        # 随机到达顺序
    paid_units = 0.0                                  # 已为兑付而清算的资产单位
    for i in order[withdraw[order]]:
        if paid_units + c1 <= N:                      # 资产总量 N，按需清算
            payment[i] = c1
            paid_units += c1
        else:
            break                                     # 资产耗尽：排在后面的提款者拿不到
    unserved_runners = int(np.sum(withdraw & (payment == 0)))

    waiters = ~withdraw
    if insured:
        payment[waiters] = c2                         # 政府兜底，等待者稳拿 c2
    else:
        remaining = max(N - paid_units, 0.0)          # 幸存资产在 t=2 产出 R
        if waiters.sum() > 0:
            payment[waiters] = remaining * R / waiters.sum()

    stats = {
        "late_mean_c": float(payment[~early].mean()),          # 晚期储户人均消费
        "zero_share": float(np.mean(payment == 0)),            # 一无所获的储户占比
        "c2_paid": bool(np.allclose(payment[waiters], c2) if waiters.sum() else False),
        "welfare": float(np.mean(util(np.maximum(payment, 0.01), gamma))),
                 # 效用对 0 消费按 0.01 计，仅为得到可读的有限数值
    }
    stats["unserved_runners"] = unserved_runners
    return payment, stats


def welfare(payment, gamma=2.0):
    """人均期望效用（0 消费按 0.01 计，仅作相对比较）。"""
    return float(np.mean(util(np.maximum(payment, 0.01), gamma)))


# ----------------------------------------------------------------------
# 2. 图 1：最优合同与流动性保险的价值
# ----------------------------------------------------------------------
def fig_contract(R=2.0, lam=0.4):
    gammas = np.linspace(1.05, 5.0, 200)
    c1s = np.array([optimal_contract(R, lam, g)["c1"] for g in gammas])
    c2s = np.array([optimal_contract(R, lam, g)["c2"] for g in gammas])
    k_base = optimal_contract(R, lam, 2.0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)

    ax = axes[0]
    ax.plot(gammas, c1s, lw=2, label=r"早取者 $c_1$")
    ax.plot(gammas, c2s, lw=2, label=r"晚取者 $c_2$")
    ax.axhline(1.0, color="gray", ls="--", lw=1, label="自给自足：早取 = 1")
    ax.axhline(R, color="black", ls=":", lw=1.2, label="自给自足：晚取 = R")
    ax.set_xlabel(r"相对风险厌恶系数 $\gamma$")
    ax.set_ylabel("合同支付")
    ax.set_title("(a) 最优合同随风险厌恶程度的变化")
    ax.legend(fontsize=9)

    ax = axes[1]
    bars = ax.bar(["银行\n（流动性保险）", "自给自足"],
                  [k_base["eu_bank"], k_base["eu_autarky"]],
                  color=["#2e7d32", "#9e9e9e"], width=0.5)
    ax.set_ylabel("期望效用")
    ax.set_title(r"(b) 银行 vs 自给自足（$R=2,\ \lambda=0.4,\ \gamma=2$）")
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.002,
                f"{b.get_height():.4f}", ha="center", fontsize=10)
    ax.set_ylim(0, max(k_base["eu_bank"], k_base["eu_autarky"]) * 1.25)

    path = os.path.join(FIG_DIR, "fig1_optimal_contract.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------
# 3. 图 2：挤兑博弈的多重均衡
# ----------------------------------------------------------------------
def fig_run_game(R=2.0, lam=0.4, gamma=2.0):
    k = optimal_contract(R, lam, gamma)
    c1, c2 = k["c1"], k["c2"]
    m, pr, pw, m_star, m_bar = run_payoffs(R, lam, gamma)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)

    ax = axes[0]
    ax.plot(m, pr, lw=2, color="#c62828", label="跑（t=1 提款）的收益")
    ax.plot(m, pw, lw=2, color="#1565c0", label="等（t=2 消费）的收益")
    ax.axvline(m_star, color="purple", ls="-.", lw=1.2)
    ax.axvline(m_bar, color="gray", ls="--", lw=1)
    ax.text(m_star + 0.012, 0.28, f"无差异点 $m^*$≈{m_star:.2f}\n（预期挤兑规模超过它，跑是最优反应）",
            fontsize=9, color="purple")
    ax.text(m_bar + 0.012, 1.05, f"$\\bar{{m}}$≈{m_bar:.2f}\n（被迫全额清算）", fontsize=9, color="dimgray")
    ax.scatter([0, 1], [pw[0], pr[-1]], zorder=5, s=70, facecolor="white", edgecolor="black")
    ax.annotate("均衡①：无人挤兑\n（等 → $c_2$）", (0.0, pw[0]), xytext=(0.05, 1.6), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="black"))
    ax.annotate("均衡②：人人挤兑\n（跑 → 清算价值）", (1.0, pr[-1]), xytext=(0.60, 0.55), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="black"))
    ax.set_xlabel(r"晚期消费者中去挤兑的比例 $m$")
    ax.set_ylabel("晚期消费者的收益")
    ax.set_title("(a) 协调博弈：挤兑的自我实现")
    ax.legend(fontsize=9, loc="upper right")

    # 面板 b：正常均衡 vs 挤兑均衡的实现消费分布
    pay_norm, _ = simulate_run(N=50_000, lam=lam, R=R, gamma=gamma, m_panic=0.0, seed=7)
    pay_run, _ = simulate_run(N=50_000, lam=lam, R=R, gamma=gamma, m_panic=1.0, seed=7)
    ax = axes[1]
    bins = np.linspace(0, max(R, c2) * 1.15, 40)
    ax.hist(pay_norm, bins=bins, alpha=0.55, label="正常均衡", color="#2e7d32")
    ax.hist(pay_run, bins=bins, alpha=0.55, label="挤兑均衡", color="#c62828")
    ax.set_xlabel("每位储户最终实现的消费")
    ax.set_ylabel("人数")
    ax.set_title("(b) 挤兑把晚期消费从 $c_2$ 打回清算价值，甚至归零")
    ax.legend(fontsize=9)

    path = os.path.join(FIG_DIR, "fig2_run_equilibria.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------
# 4. 图 3：存款保险如何排除挤兑均衡
# ----------------------------------------------------------------------
def fig_insurance(R=2.0, lam=0.4, gamma=2.0):
    k = optimal_contract(R, lam, gamma)
    c1, c2 = k["c1"], k["c2"]
    m, pr, pw, m_star, _ = run_payoffs(R, lam, gamma)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)

    ax = axes[0]
    ax.plot(m, pr, lw=2, color="#c62828", label="跑（t=1 提款）的收益")
    ax.plot(m, pw, lw=1.4, color="#1565c0", ls="--", alpha=0.7, label="等（无保险）的收益")
    ax.axhline(c2, lw=2, color="#1565c0", label="等：政府担保下稳拿 $c_2$")
    ax.fill_between(m, 0, pr, color="#c62828", alpha=0.06)
    ax.set_xlabel(r"晚期消费者中去挤兑的比例 $m$")
    ax.set_ylabel("晚期消费者的收益")
    ax.set_title("(a) 存款保险：等待严格占优 → 挤兑均衡被排除")
    ax.legend(fontsize=9, loc="center left")
    ax.set_ylim(0, c2 * 1.25)

    # 面板 b：各情形下晚期储户的人均消费
    _, s_norm = simulate_run(N=200_000, lam=lam, R=R, gamma=gamma, m_panic=0.0, seed=1)
    _, s_part = simulate_run(N=200_000, lam=lam, R=R, gamma=gamma, m_panic=0.9, seed=1)
    _, s_run = simulate_run(N=200_000, lam=lam, R=R, gamma=gamma, m_panic=1.0, seed=1)
    _, s_ins = simulate_run(N=200_000, lam=lam, R=R, gamma=gamma, m_panic=0.0, insured=True, seed=1)
    labels = ["正常均衡", "部分恐慌\n(m=0.9)", "全面挤兑", "存款保险"]
    vals = [s_norm["late_mean_c"], s_part["late_mean_c"], s_run["late_mean_c"], s_ins["late_mean_c"]]
    zeros = [s_norm["zero_share"], s_part["zero_share"], s_run["zero_share"], s_ins["zero_share"]]
    colors = ["#2e7d32", "#ef6c00", "#c62828", "#1565c0"]
    ax = axes[1]
    bars = ax.bar(labels, vals, color=colors, width=0.55)
    for b, z in zip(bars, zeros):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                f"{b.get_height():.3f}\n(颗粒无收 {z:.0%})", ha="center", fontsize=8.5)
    ax.set_ylabel("晚期储户人均消费")
    ax.set_title("(b) 福利比较：保险让经济锁定在正常均衡")
    ax.set_ylim(0, max(vals) * 1.3)

    path = os.path.join(FIG_DIR, "fig3_deposit_insurance.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------
# 主程序
# ----------------------------------------------------------------------
def main():
    R, lam, gamma = 2.0, 0.4, 2.0
    k = optimal_contract(R, lam, gamma)
    c1, c2 = k["c1"], k["c2"]

    print("=" * 64)
    print("Diamond–Dybvig (1983) 课堂版：数值结果")
    print("=" * 64)
    print(f"参数：R={R}, λ={lam}, γ={gamma}")
    print(f"最优合同：c1 = {c1:.4f} (>1, 流动性保险), c2 = {c2:.4f} (<R, 但>c1)")
    print(f"银行期望效用 = {k['eu_bank']:.4f}  vs 自给自足 = {k['eu_autarky']:.4f}"
          f"  （提升 {k['gain']:+.4f}）")
    print(f"银行在'只有早期消费者提款'时可兑付：{k['viable']}")
    print()

    _, pr, pw, m_star, m_bar = run_payoffs(R, lam, gamma)
    print(f"跑/等无差异的挤兑规模 m* ≈ {m_star:.3f}：预期超过它，跑是晚期消费者的最优反应。")
    print(f"被迫全额清算的挤兑规模 m̄ ≈ {m_bar:.3f}：此时等待者一无所获。")
    print(f"m=0：等 {pw[0]:.3f} > 跑 {pr[0]:.3f}  → 无人挤兑是均衡。")
    print(f"m=1：跑 {pr[-1]:.3f} > 等 {pw[-1]:.3f}  → 人人挤兑也是均衡（自我实现）。")
    print()

    for name, m_p, ins in [("正常均衡 (m=0)", 0.0, False),
                           ("部分恐慌 (m=0.9)", 0.9, False),
                           ("全面挤兑 (m=1)", 1.0, False),
                           ("存款保险 (m=0)", 0.0, True)]:
        pay, s = simulate_run(N=200_000, lam=lam, R=R, gamma=gamma, m_panic=m_p, insured=ins, seed=1)
        print(f"模拟[{name}]：晚期储户人均消费={s['late_mean_c']:.4f}，"
              f"颗粒无收占比={s['zero_share']:.1%}，人均期望效用={s['welfare']:.4f}")
    print()

    p1 = fig_contract(R, lam)
    p2 = fig_run_game(R, lam, gamma)
    p3 = fig_insurance(R, lam, gamma)
    print("图表已保存：")
    for p in (p1, p2, p3):
        print("  -", os.path.join("figures", os.path.basename(p)))
    print()
    print("政策提示（详见 README）：存款保险排除挤兑均衡，但会削弱储户监督、")
    print("诱发银行风险承担，需与资本监管、流动性监管和最后贷款人安排配套。")


if __name__ == "__main__":
    main()
