# Diamond–Dybvig (1983) 银行挤兑模型 & Bernanke (1983) 信用渠道 —— 课堂版 Python 实现

这是小组论文「金融危机的宏观传导」中组员 A 部分的配套代码：把 **Diamond–Dybvig 三期银行挤兑模型** 和 **Bernanke (1983) 非货币信用渠道的实证框架** 用 Python 跑出来，配数值结果与图表，直接服务于论文第 4 章（实证）与第 5 章（DD 模型）。

一句话分工：**DD 回答"银行为什么可能内生地突然失灵"；Bernanke 回答"金融中介失灵后，为什么会产生持续的宏观损失"。**

```mermaid
graph LR
    A[银行挤兑 / 借款人财务困境] --> B[筛选·监督·信息生产受损]
    B --> C[信贷中介成本 CCI 上升]
    C --> D[信贷收缩 · 外部融资成本上升]
    D --> E[消费与投资下降]
    E --> F[总需求与产出下降]
    style A stroke:#c62828
    style F stroke:#c62828
```

DD 解释这条链的第一环（脆弱性从哪来），Bernanke 解释后面的传导（为什么会放大成持续衰退）。

---

## 1. DD 模型速览（`dd_model.py`）

**设定**：t=0,1,2 三期；每位储户 t=0 有 1 单位禀赋投入长期项目——持有到 t=2 产出 R>1，t=1 提前清算每单位只能回收 1。比例 λ 的储户是早期消费者（t=1 消费），其余是晚期消费者（t=2 消费），类型 t=1 才实现且银行不可观察。效用取 CRRA。

**最优合同**：在资源约束 λc₁ + (1−λ)c₂/R ≤ 1 下最大化期望效用，CRRA 解为

```
c1 = 1 / (λ + (1−λ)·R^(1/γ−1))，  c2 = c1 · R^(1/γ)
```

基准参数 R=2、λ=0.4、γ=2 下：**c₁=1.213 > 1，c₂=1.716**，即银行用"略低于长期回报"的晚取收益，换来"高于清算价值"的早取保障——这就是流动性保险。期望效用 0.3206 vs 自给自足 0.3000。

**挤兑博弈**：先到先得 + 序贯服务下，晚期消费者"等"或"跑"的收益都取决于预期去挤兑的比例 m：

- 跑的收益 = min(c₁, 1/(λ+m(1−λ)))（资金充足拿 c₁，不足按兑付概率折算）
- 等的收益 = 剩余资产 (1−(λ+m(1−λ))c₁)·R 按等待者均分（m=0 时恰为 c₂）

两条收益曲线的交点 **m\* = (c₂−c₁)/(c₁(R−1)) ≈ 0.414**：预期挤兑规模超过它，跑就是最优反应——**坏预期自己把自己变成了挤兑**。于是出现两个均衡：

| 均衡 | 储户行为 | 结果 |
|---|---|---|
| 正常均衡 | 只有真正的早期消费者提款 | 晚期消费者稳拿 c₂，流动性保险兑现 |
| 挤兑均衡 | 晚期消费者抢先提款 | 银行被迫清算长期资产，17.6% 储户颗粒无收 |

**存款保险**：政府可信担保使"等"在任何 m 下都稳拿 c₂ > 跑的收益，等待严格占优 → 挤兑均衡被排除。但保险同时削弱储户监督、诱发银行冒险，需与资本监管、流动性监管和最后贷款人安排配套——"防挤兑"与"控风险"必须一起讨论。

## 2. Bernanke 实证框架速览（`bernanke_1983.py`）

课堂重构版回归（**不是** Bernanke 原文逐字方程）：

```
Δln(IPₜ) = α + β₁·货币/价格控制ₜ + β₂·金融危机代理ₜ + β₃·金融危机代理ₜ₋₁ + εₜ
```

- **模型 A**：只加货币与价格因素 → 传统货币渠道基准
- **模型 B**：再加入银行危机、借款人困境代理及其滞后项

模拟数据下的演示结果：金融变量 3/3 显著为负，R² 从 0.35 升到 0.90，滞后项显著 → 与"非货币信用渠道 + 持续性"一致。脚本会自动打印"怎么读表"的四条要点（含表述边界：只能说"支持信用渠道"，不能说"证明了银行危机导致大萧条"）。

> **数据声明**：默认数据是脚本显式设定的**课堂演示模拟数据**，仅用于展示回归框架与读表方法，不代表 Bernanke 的真实估计。接入真实数据：从 [FRED](https://fred.stlouisfed.org) 下载并整理后运行 `python bernanke_1983.py --csv your_data.csv`，CSV 需含列 `date, dln_ip, money_growth, price_inflation, bank_distress, borrower_distress`。

## 3. 快速开始

```bash
pip install -r requirements.txt
python dd_model.py        # DD 模型：数值结果 + 图 1~3
python bernanke_1983.py   # Bernanke 框架：回归表 + 图 4~5
```

## 4. 图表一览

| 文件 | 内容 | 对应论文 |
|---|---|---|
| `figures/fig1_optimal_contract.png` | 最优合同随风险厌恶的变化；银行 vs 自给自足福利 | 第 5 章 3.1–3.2 |
| `figures/fig2_run_equilibria.png` | 跑/等收益曲线与多重均衡；挤兑前后的消费分布 | 第 5 章 3.3 |
| `figures/fig3_deposit_insurance.png` | 保险使等待占优；各情形福利比较 | 第 5 章 3.4 |
| `figures/fig4_bernanke_timeseries.png` | Δln(IP) 与金融困境指数（双轴，标准化） | 第 4 章 实证图 |
| `figures/fig5_bernanke_fit.png` | 模型 A vs 模型 B 的拟合比较 | 第 4 章 实证段 |

## 5. 文件结构

```
├── dd_model.py          # DD 模型：最优合同、挤兑博弈、存款保险（约 300 行，纯 numpy）
├── bernanke_1983.py     # Bernanke 框架：模拟数据、OLS（手写 t 检验）、回归表与图
├── figures/             # 运行生成的 5 张图
├── requirements.txt     # numpy + matplotlib
└── README.md
```

## 6. 口径与表述边界（写论文前必读）

- 本仓库所有公式都是**课堂重构**，不冒充原论文逐字方程；模拟数据参数由脚本显式设定。
- 不要把 c₂>R、c₁<1<c₂ 写成无条件结论——精确数值取决于风险厌恶程度、早期消费者比例和合同形式。
- 回归结果只能写"**支持**金融中介渠道的证据""与信用渠道假说**一致**"；时间序列中银行困境、预期衰退与产出下降可能互相影响，反向因果未被排除。
- 把所有"导致"逐个检查是否有识别依据；CCI（信贷中介成本）只能用代理变量测量，且批准的贷款利率还存在选择偏差。

## 7. 参考文献

- Diamond, D. W., & Dybvig, P. H. (1983). [Bank Runs, Deposit Insurance, and Liquidity](https://doi.org/10.1086/261155). *Journal of Political Economy*, 91(3), 401–419.
- Bernanke, B. S. (1983). [Non-Monetary Effects of the Financial Crisis in the Propagation of the Great Depression](https://doi.org/10.3386/w1054). *American Economic Review*, 73(3), 257–276. ([NBER w1054](https://www.nber.org/papers/w1054))
- Nobel Committee (2022). [Financial Intermediation and the Economy](https://www.nobelprize.org/prizes/economic-sciences/2022/advanced-information/)（2022 年诺贝尔经济学奖科学背景报告）
- Bernanke, B. S., & Gertler, M. (1995). [Inside the Black Box: The Credit Channel of Monetary Policy Transmission](https://www.nber.org/papers/w5146). *Journal of Economic Perspectives*, 9(4), 27–48.
- Brunnermeier, M. K. (2009). [Deciphering the Liquidity and Credit Crunch 2007–2008](https://www.nber.org/papers/w14612). *Journal of Economic Perspectives*, 23(1), 77–100.
- Calomiris, C. W., & Mason, J. R. (2003). [Consequences of Bank Distress During the Great Depression](https://doi.org/10.1257/000282803322157245). *American Economic Review*, 93(3), 937–947.

## License

MIT
