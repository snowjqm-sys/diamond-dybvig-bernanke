# Diamond–Dybvig (1983) 银行挤兑模型 & Bernanke (1983) 信用渠道

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-2166AC?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-1B7837)
![Dependencies](https://img.shields.io/badge/依赖-numpy%20·%20matplotlib-E08214)
![用途](https://img.shields.io/badge/用途-课堂演示%20·%20论文第4%2F5章-24292F)

**用 Python 把 2022 年诺贝尔经济学奖的两个核心工作跑出来：**
**银行挤兑为什么自我实现，金融中介失灵为什么放大衰退。**

</div>

---

> 一句话分工：**Diamond–Dybvig 回答"银行为什么可能内生地突然失灵"；Bernanke 回答"金融中介失灵后，为什么会产生持续的宏观损失"。**

```mermaid
graph LR
    A[银行挤兑 / 借款人财务困境] --> B[筛选·监督·信息生产受损]
    B --> C[信贷中介成本 CCI 上升]
    C --> D[信贷收缩 · 外部融资成本上升]
    D --> E[消费与投资下降]
    E --> F[总需求与产出下降]
    style A stroke:#B2182B,color:#B2182B
    style F stroke:#B2182B,color:#B2182B
    style C stroke:#E08214,color:#E08214
```

DD 解释这条链的第一环（脆弱性从哪来），Bernanke 解释后面的传导（为什么放大成持续衰退）。

## 核心结果一览

| | 结果 | 含义 |
|---|---|---|
| 最优合同 | c₁ = 1.213 > 1，c₂ = 1.716 | 早取保障高于清算价值，这就是**流动性保险** |
| 保险价值 | 期望效用 0.3206 vs 自给自足 0.3000 | 银行让每个人都过得更好 |
| 挤兑阈值 | m\* ≈ 0.41，m̄ ≈ 0.71 | 预期挤兑规模超过 m\*，"跑"成为最优反应 → **坏预期自我实现** |
| 挤兑代价 | 17.6% 储户颗粒无收 | 协调失败的宏观代价 |
| 回归比较 | 金融变量 3/3 显著为负，R² 0.35 → 0.90 | 非货币信用渠道有**额外解释力**，且具持续性 |

## 图表

| DD 模型 | Bernanke 框架 |
|---|---|
| <img src="figures/fig1_optimal_contract.png" width="420"/> | <img src="figures/fig4_bernanke_timeseries.png" width="420"/> |
| 最优合同与流动性保险的价值 | 金融困境与工业生产共同恶化 |
| <img src="figures/fig2_run_equilibria.png" width="420"/> | <img src="figures/fig5_bernanke_fit.png" width="420"/> |
| 挤兑博弈的多重均衡 | 模型 A vs 模型 B 的拟合比较 |
| <img src="figures/fig3_deposit_insurance.png" width="420"/> | |
| 存款保险排除挤兑均衡 | |

## 快速开始

```bash
pip install -r requirements.txt
python dd_model.py        # DD 模型：数值结果 + 图 1~3
python bernanke_1983.py   # Bernanke 框架：回归表 + 图 4~5
```

## 文件结构

```
├── dd_model.py          # 最优合同、挤兑博弈（解析阈值 + 蒙特卡洛）、存款保险
├── bernanke_1983.py     # 回归框架（模型A/B比较），手写 OLS 含 t 检验
├── figures/             # 运行生成的 5 张图
└── requirements.txt     # numpy + matplotlib
```

## 接入真实数据

`bernanke_1983.py` 默认使用**课堂演示模拟数据**（参数由脚本显式设定），仅用于展示回归框架与读表方法。如需真实月度数据，从 [FRED](https://fred.stlouisfed.org) 下载整理后：

```bash
python bernanke_1983.py --csv your_data.csv
```

CSV 需含列：`date, dln_ip, money_growth, price_inflation, bank_distress, borrower_distress`。

## 口径与表述边界（写论文前必读）

- 本仓库所有公式都是**课堂重构**，不冒充原论文逐字方程；模拟数据参数由脚本显式设定。
- 不要把 c₂>R、c₁<1<c₂ 写成无条件结论——精确数值取决于风险厌恶程度、早期消费者比例和合同形式。
- 回归结果只能写"**支持**金融中介渠道的证据""与信用渠道假说**一致**"；银行困境、预期衰退与产出下降可能互相影响，反向因果未被排除。
- 把所有"导致"逐个检查是否有识别依据；CCI（信贷中介成本）只能用代理变量测量，且批准的贷款利率还存在选择偏差。

## 参考文献

- Diamond, D. W., & Dybvig, P. H. (1983). [Bank Runs, Deposit Insurance, and Liquidity](https://doi.org/10.1086/261155). *Journal of Political Economy*, 91(3), 401–419.
- Bernanke, B. S. (1983). [Non-Monetary Effects of the Financial Crisis in the Propagation of the Great Depression](https://doi.org/10.3386/w1054). *American Economic Review*, 73(3), 257–276. ([NBER w1054](https://www.nber.org/papers/w1054))
- Nobel Committee (2022). [Financial Intermediation and the Economy](https://www.nobelprize.org/prizes/economic-sciences/2022/advanced-information/)（诺贝尔经济学奖科学背景报告）
- Bernanke, B. S., & Gertler, M. (1995). [Inside the Black Box: The Credit Channel of Monetary Policy Transmission](https://www.nber.org/papers/w5146). *Journal of Economic Perspectives*, 9(4), 27–48.
- Brunnermeier, M. K. (2009). [Deciphering the Liquidity and Credit Crunch 2007–2008](https://www.nber.org/papers/w14612). *Journal of Economic Perspectives*, 23(1), 77–100.
- Calomiris, C. W., & Mason, J. R. (2003). [Consequences of Bank Distress During the Great Depression](https://doi.org/10.1257/000282803322157245). *American Economic Review*, 93(3), 937–947.

## License

MIT
