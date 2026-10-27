"""DD -> intermediation loss -> financing wedge -> IS-LM: teaching scenarios.

All parameters are assumptions, not historical estimates. Model periods have no
calendar interpretation. DD does not imply the bridge to information loss: the
mapping below is an explicit additional assumption. Results are deviations from
a normalized baseline, not a fit to 1929-1933 output.
"""
import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from dd_model import optimal_contract, plt, style_ax


@dataclass(frozen=True)
class Parameters:
    c: float = 0.6
    b: float = 20.0
    k: float = 0.05
    h: float = 10.0
    theta: float = 0.05
    eta: float = 1.0
    chi: float = 1.0
    baseline_output: float = 100.0

    def __post_init__(self):
        values = list(asdict(self).values())
        if not np.isfinite(values).all():
            raise ValueError("Parameters must be finite")
        if not 0 <= self.c < 1 or min(self.b, self.k, self.h, self.baseline_output) <= 0:
            raise ValueError("Require 0 <= c < 1 and positive b, k, h, baseline_output")
        if min(self.theta, self.eta, self.chi) < 0:
            raise ValueError("Transmission coefficients must be nonnegative")


def dd_distress_bridge(m=0.4, R=2.0, lam=0.4, gamma=2.0, chi=1.0):
    """Distress = chi * extra liquidation, capped by the DD asset endowment.

    No cash reserve in this DD block; liquidation recovers one per asset unit.
    Mapping extra liquidation to impaired intermediation is assumed, not derived.
    """
    if not (0 <= m <= 1 and 0 < lam < 1 and R > 1 and gamma > 0 and chi >= 0):
        raise ValueError("Invalid DD bridge parameters")
    c1 = optimal_contract(R, lam, gamma)["c1"]
    normal = min(lam * c1, 1.0)
    crisis = min((lam + (1 - lam) * m) * c1, 1.0)
    return chi * max(crisis - normal, 0.0)


def simulate(distress, real_money_gap, rho=0.85, params=None, expected_inflation_gap=None):
    """Solve deviations jointly in goods and money markets in every period.

    z[t] = rho*z[t-1] + theta*d[t], with pre-shock z[-1] = 0.
    s_gap = eta*z; I_gap = -b*(i_gap - pi_e_gap + s_gap).
    Y_gap = c*Y_gap + I_gap; real_money_gap = k*Y_gap - h*i_gap.
    Money and expected inflation paths are exogenous; no causal identification.
    """
    p = params or Parameters()
    d, money = np.asarray(distress, float), np.asarray(real_money_gap, float)
    pi = np.zeros_like(d) if expected_inflation_gap is None else np.asarray(expected_inflation_gap, float)
    if d.ndim != 1 or d.size == 0 or money.shape != d.shape or pi.shape != d.shape:
        raise ValueError("Paths must be nonempty one-dimensional arrays of equal length")
    if not np.isfinite([*d, *money, *pi]).all() or np.any(d < 0) or not 0 <= rho < 1:
        raise ValueError("Paths must be finite, distress nonnegative, and 0 <= rho < 1")
    z = np.empty_like(d)
    previous = 0.0
    for t, shock in enumerate(d):
        previous = rho * previous + p.theta * shock
        z[t] = previous
    s = p.eta * z
    denominator = 1 - p.c + p.b * p.k / p.h
    y = ((p.b / p.h) * money + p.b * pi - p.b * s) / denominator
    rate = (p.k * y - money) / p.h
    investment = -p.b * (rate - pi + s)
    consumption = p.c * y
    return {"period": np.arange(len(d)), "distress": d, "real_money_gap": money,
            "expected_inflation_gap": pi, "intermediation_loss": z, "wedge_gap": s,
            "output_gap": y, "output_index": p.baseline_output + y,
            "rate_gap": rate, "investment_gap": investment, "consumption_gap": consumption}


def liquidity_example(withdrawals, reserves, support, liquidation_value, assets,
                      asset_value, deposits, existing_cb_debt=0):
    """Static policy accounting before payouts; loan principal is repayable.

    This extension has its own balance sheet, separate from the DD calibration.
    It does not endogenize depositor incentives, lending terms, or fiscal costs.
    """
    values = [withdrawals, reserves, support, liquidation_value, assets, asset_value,
              deposits, existing_cb_debt]
    if not np.isfinite(values).all() or min(values) < 0 or liquidation_value == 0:
        raise ValueError("Require finite nonnegative inputs and positive liquidation value")
    required = max(withdrawals - reserves - support, 0) / liquidation_value
    sold = min(required, assets)
    equity = reserves + asset_value - deposits - existing_cb_debt
    equity_after_loan = reserves + support + asset_value - deposits - existing_cb_debt - support
    return {"required_liquidation": required, "actual_liquidation": sold,
            "payment_shortfall": max(withdrawals - reserves - support - liquidation_value * sold, 0),
            "equity_before": equity, "equity_after_loan_before_payout": equity_after_loan}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", default="results/core_model")
    ap.add_argument("--periods", type=int, default=24)
    args = ap.parse_args()
    if args.periods < 2:
        ap.error("--periods must be at least 2")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = Parameters()
    periods = np.arange(args.periods)
    d = np.zeros(args.periods)
    d[0] = dd_distress_bridge(chi=p.chi)
    money = -0.1 * 0.5 ** periods
    cases = {
        "money_only": (np.zeros_like(d), money, 0.85, "仅货币条件变化"),
        "credit_fast": (d, money, 0.35, "相同货币路径＋信用受损，恢复较快"),
        "credit_slow": (d, money, 0.85, "相同货币路径＋信用受损，恢复较慢"),
        "credit_only": (d, np.zeros_like(money), 0.85, "货币余额不变，仅信用受损"),
    }
    results = {}
    for key, (shock, monetary, rho, label) in cases.items():
        result = simulate(shock, monetary, rho, p)
        results[key] = result
        with (out / (key + ".csv")).open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(result)
            writer.writerows(zip(*result.values()))
    fig, ax = plt.subplots(figsize=(10.5, 4.7), constrained_layout=True)
    for key, (_, _, _, label) in cases.items():
        ax.plot(periods, results[key]["output_index"], label=label, lw=2)
    ax.axhline(100, color="#8A8F98", ls="--", lw=0.8)
    ax.set(xlabel="模型期（非历史月份）", ylabel="产出指数（无冲击基准＝100）",
           title="教学情景模拟：信用摩擦与恢复速度影响产出路径")
    style_ax(ax)
    ax.legend(fontsize=9, loc="lower right")
    fig.savefig(out / "output_paths.png", dpi=180)
    plt.close(fig)
    common = dict(withdrawals=45, reserves=10, liquidation_value=1, assets=100,
                  asset_value=100, deposits=100)
    policy = {"no_support": liquidity_example(support=0, **common),
              "support_20": liquidity_example(support=20, **common),
              "insolvent_support_20": liquidity_example(support=20, **{**common, "asset_value": 80, "liquidation_value": 0.7})}
    report = {"evidence_type": "assumed teaching scenarios; not historical estimates",
              "parameters": asdict(p), "dd_bridge": {"m": 0.4, "R": 2, "lambda": 0.4,
              "gamma": 2, "initial_distress": float(d[0]), "assumption": "extra liquidation -> distress"},
              "money_path": "real_money_gap[t] = -0.1 * 0.5**t (assumed)",
              "scenarios": {key: {"rho": val[2], "output_gap_t0": float(results[key]["output_gap"][0]),
               "output_gap_last": float(results[key]["output_gap"][-1])} for key, val in cases.items()},
              "liquidity_examples": policy,
              "limitations": ["No historical calibration or estimation", "No default or fiscal equilibrium",
              "DD-to-distress bridge and persistence are assumed", "Linear short-run IS-LM; no interest-rate bound",
              "Credit quantity constraints and borrower net worth are optional, not implemented"]}
    (out / "assumptions_and_results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
