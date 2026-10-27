"""Inspect the supplied historical CSV without manufacturing missing regressors."""
import csv
import hashlib
import json
from pathlib import Path
from datetime import datetime

import numpy as np


def validate_dates(rows):
    """Require unique ascending and consecutive monthly observations."""
    indices = []
    for row in rows:
        try:
            date = datetime.strptime(row["date"], "%Y-%m")
        except (KeyError, ValueError) as exc:
            raise ValueError("date must use YYYY-MM") from exc
        indices.append(date.year * 12 + date.month)
    if not indices or any(b - a != 1 for a, b in zip(indices, indices[1:])):
        raise ValueError("CSV must contain consecutive, ascending, unique monthly dates; do not compress gaps")


def describe_csv(path, output_dir):
    """Describe source values; the file's online provenance is not authenticated."""
    from dd_model import plt, style_ax
    source = Path(path)
    with source.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = sorted({"date", "ip", "cpi", "baa"} - set(reader.fieldnames or []))
        if missing:
            raise ValueError("Descriptive CSV missing columns: " + ", ".join(missing))
        rows = list(reader)
    validate_dates(rows)
    if len(rows) < 3:
        raise ValueError("At least three months are needed")
    values = {key: np.array([float(row[key]) for row in rows]) for key in ("ip", "cpi", "baa")}
    if not all(np.isfinite(v).all() for v in values.values()) or min(values["ip"].min(), values["cpi"].min()) <= 0:
        raise ValueError("Core observations must be finite, and ip/cpi must be positive")
    growth, inflation = np.diff(np.log(values["ip"])), np.diff(np.log(values["cpi"]))
    for key, derived in (("dln_ip", growth), ("inflation", inflation)):
        for i, calc in enumerate(derived, start=1):
            if rows[i].get(key, "").strip() and abs(float(rows[i][key]) - calc) > 1.1e-6:
                raise ValueError(f"{key} conflicts with levels at {rows[i]['date']}")
    ip = values["ip"]
    running_peak = np.maximum.accumulate(ip)
    trough = int(np.argmin(ip / running_peak - 1))
    peak = int(np.argmax(ip[:trough + 1]))
    report = {"evidence_type": "descriptive only; no regression or causal identification",
              "source_file": source.name, "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "source_status": "user-supplied attachment, labeled FRED by source document; not point-by-point reverified online",
              "rows": len(rows), "first_date": rows[0]["date"], "last_date": rows[-1]["date"],
              "difference_rows": len(growth),
              "ip_start_to_min_pct": float(100 * (ip.min() / ip[0] - 1)),
              "ip_peak_to_trough_pct": float(100 * (ip[trough] / ip[peak] - 1)),
              "ip_peak_date": rows[peak]["date"], "ip_trough_date": rows[trough]["date"],
              "cpi_start_to_end_pct": float(100 * (values["cpi"][-1] / values["cpi"][0] - 1)),
              "correlation_ip_growth_baa_change": float(np.corrcoef(growth, np.diff(values["baa"]))[0, 1]),
              "correlation_ip_growth_inflation": float(np.corrcoef(growth, inflation)[0, 1]),
              "missing_full_regression_columns": sorted({"money_growth", "price_inflation", "bank_distress", "borrower_distress"} - set(rows[0])),
              "limitations": ["BAA is a yield level, not a credit spread or bank failure measure",
                              "INDPRO is industrial output, not GDP", "CPI is not money supply",
                              "Observed inflation is not expected inflation", "No historical model estimated"]}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "historical_description.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6), sharex=True, constrained_layout=True)
    x = np.arange(len(rows))
    axes[0].plot(x, ip / ip[0] * 100, color="#2166AC", lw=2)
    axes[0].set(ylabel="工业生产（起点＝100）", title="附件历史数据：工业生产与 Baa 收益率（描述性展示）")
    axes[1].plot(x, values["baa"], color="#B2182B", lw=2)
    axes[1].set(ylabel="Baa 收益率（年率，%）", xlabel="月份")
    ticks = list(range(0, len(rows), 12))
    axes[1].set_xticks(ticks, [rows[t]["date"] for t in ticks])
    for ax in axes:
        style_ax(ax)
    fig.savefig(out / "historical_description.png", dpi=180)
    plt.close(fig)
    return report
