# Risk Parity + CTA 动态杠杆量化策略系统

> **Risk Parity 基石 Beta + CTA 单一资产 Alpha 增强 + TRS 互换合成杠杆**

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-Private-green.svg)
![Version](https://img.shields.io/badge/Version-1.0.0-orange.svg)

**策略年化夏普比率 0.72 | 最大回撤 -7.55% | 精准控波 5.87%**

</div>

---

## 📋 目录

- [项目概述](#-项目概述)
- [核心特性](#-核心特性)
- [文件结构详解](#-文件结构详解)
- [快速开始](#-快速开始)
- [技术架构概览](#-技术架构概览)
- [算法细节](#-算法细节)
- [实证绩效](#-实证绩效)
- [依赖环境](#-依赖环境)
- [风险提示](#-风险提示)

---

## 📖 项目概述

本项目是一套**多资产配置驱动的量化投资策略引擎**，采用三层架构设计，实现风险平价基石配置、CTA趋势跟踪增强和目标波动率杠杆控制的完整策略流程。

### 设计理念

1. **Risk Parity 基石 Beta**：通过风险平价分配实现分散化配置，每个资产贡献相等风险
2. **CTA 单一资产 Alpha 增强**：基于趋势跟踪信号动态调整风险预算，捕捉资产动量
3. **TRS 互换合成杠杆**：通过目标波动率控制动态调节杠杆，实现精准风险暴露

### 资产覆盖

| 代码 | 名称 | 类型 | CTA 策略 |
|------|------|------|---------|
| `000300.SH` | 沪深 300 指数 | 指数 | 双均线 (60/120) |
| `510880.SH` | 华泰柏瑞红利 ETF | 基金 | 双均线 (60/120) |
| `510410.SH` | 上证资源 ETF | 基金 | 唐奇安通道 (20) |
| `518880.SH` | 华安黄金 ETF | 基金 | 唐奇安通道 (20) |
| `H11009.CSI` | 中债 10 年期国债总指数 | 指数 | MACD (12/26/9) |

---

## ⭐ 核心特性

- **机器学习驱动**：使用 EWMA 协方差估计和 SLSQP 优化算法
- **多策略适配**：双均线（权益）、唐奇安通道（商品）、MACD（债券）
- **动态风险预算**：基于 CTA 信号实时调整各资产风险贡献
- **目标波动率控制**：动态杠杆机制精准控制组合波动率在 6% 目标附近
- **完整回测框架**：支持策略回测、基准对比、绩效分析、可视化展示
- **极端防御机制**：全信号看空时国债独享 100% 风险预算

---

## 📁 文件结构详解

```
All_Weather_Strategy/
├── README.md                          # 项目说明文档（本文件）
├── requirements.txt                   # Python 依赖包列表
├── All_Weather_Strategy_flowchart.svg # 策略流程图
│
├── src/                               # 源代码目录
│   ├── data_fetcher.py                # 数据接入与清洗模块
│   │   ├── class TushareDataFetcher   # Tushare Pro 数据获取器
│   │   ├── class DataCleaner          # 数据清洗器
│   │   └── class QuantDataManager      # 量化数据管理器
│   │
│   ├── strategy_optimizer.py          # 信号生成与权重优化模块
│   │   ├── class CTASignalGenerator   # CTA 择时信号生成器
│   │   ├── class RiskParityOptimizer  # 动态风险平价优化器
│   │   └── class StrategyOrchestrator # 策略编排器
│   │
│   └── analyser.py                    # 回测引擎与绩效分析模块
│       ├── class BacktestEngine       # 回测引擎
│       ├── class PerformanceAnalyzer  # 绩效分析器
│       ├── class BacktestVisualizer   # 可视化器
│       └── class BacktestOrchestrator # 回测编排器
│
├── data/                              # 数据目录
│   ├── price_data.csv                 # 对齐后的价格数据（生成）
│   └── returns_data.csv               # 日收益率数据（生成）
│
└── result/                            # 回测结果目录
    ├── target_weights.csv             # 每日目标权重（生成）
    ├── nav_data.csv                   # 净值曲线数据（生成）
    ├── backtest_results.csv           # 绩效指标结果（生成）
    └── backtest_results.png           # 可视化图表（生成）
```

### 核心模块说明

#### 模块一：数据接入与清洗 (`src/data_fetcher.py`)

**核心功能**：
- 支持指数/基金 (ETF) 多资产类型获取
- 交易日历对齐与前向填充处理
- 连续缺失检测与阈值告警（默认 10 天）
- 日收益率自动计算

#### 模块二：信号生成与权重优化 (`src/strategy_optimizer.py`)

**核心功能**：
- 多策略 CTA 择时信号生成
- EWMA 协方差矩阵估计
- SLSQP 算法风险平价优化
- 月频调仓机制

#### 模块三：回测引擎与绩效分析 (`src/analyser.py`)

**核心功能**：
- 目标波动率控制与动态杠杆计算
- TRS 互换资金成本扣除
- 多基准对比（等权 1/N、静态风险平价）
- 绩效指标计算与可视化

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 Tushare API

```python
# 在 src/data_fetcher.py 中配置
TOKEN = "your_tushare_token_here"
API_URL = 'http://lianghua.nanyangqiankun.top'
```

### 3. 数据获取与清洗

```bash
python src/data_fetcher.py
```

**输出**：
- `data/price_data.csv` - 对齐后的价格数据
- `data/returns_data.csv` - 日收益率数据

**运行时间参考**：
- 5 个资产，10 年数据：约 1-2 分钟

### 4. 信号生成与权重优化

```bash
python src/strategy_optimizer.py
```

**输出**：
- `result/target_weights.csv` - 每日目标权重

**运行时间参考**：
- 5 个资产，10 年数据：约 3-5 分钟

### 5. 回测与绩效分析

```bash
python src/analyser.py
```

**输出**：
- `result/nav_data.csv` - 净值曲线数据
- `result/backtest_results.csv` - 绩效指标
- `result/backtest_results.png` - 可视化图表

---

## 🏗️ 技术架构概览

### 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      TRS 互换杠杆层                              │
│    目标波动率控制 (6%) · 动态杠杆乘数 · 资金成本核算             │
├─────────────────────────────────────────────────────────────────┤
│                    SLSQP 优化求解层                              │
│    EWMA 协方差估计 · 动态风险预算 · L2 正则化 · 权重上限约束       │
├─────────────────────────────────────────────────────────────────┤
│                    CTA 信号生成层                                │
│    双均线 (权益) · 唐奇安通道 (商品) · MACD (债券)                │
├─────────────────────────────────────────────────────────────────┤
│                    Tushare Pro 数据层                            │
│    私有 API 专线 · 多资产日线 · 数据清洗对齐                      │
└─────────────────────────────────────────────────────────────────┘
```

### 系统 Pipeline

```mermaid
graph LR
    A[Tushare Pro 数据] --> B[DataCleaner 清洗]
    B --> C[CTA 信号生成]
    C --> D[EWMA 协方差]
    D --> E[SLSQP 优化]
    E --> F[动态杠杆]
    F --> G[绩效分析]
```

### 数据流

| 阶段 | 输入 | 处理 | 输出 |
|------|------|------|------|
| 数据接入 | Tushare API | 对齐/填充/收益率计算 | `price_data.csv`, `returns_data.csv` |
| 信号生成 | 价格数据 | 双均线/唐奇安/MACD | 信号矩阵 (1/0) |
| 权重优化 | 收益率 + 信号 | EWMA+SLSQP | `target_weights.csv` |
| 回测引擎 | 权重 + 收益率 | 杠杆计算/成本扣除 | `nav_data.csv` |
| 绩效分析 | 净值序列 | 指标计算/可视化 | `backtest_results.png` |

---

## 📐 算法细节

### 1. CTA 信号生成逻辑

#### 权益资产 - 双均线策略

```
快线 MA(60) > 慢线 MA(120) → 信号 = 1 (看多)
快线 MA(60) ≤ 慢线 MA(120) → 信号 = 0 (看空)
```

#### 商品/黄金资产 - 唐奇安通道

```
价格 > 上轨 (20 日最高) → 信号 = 1 (突破看多)
价格 < 下轨 (20 日最低) → 信号 = 0 (跌破看空)
中间状态 → 保持前一信号
```

#### 债券资产 - MACD

```
MACD 柱状图 = DIF - DEA > 0 → 信号 = 1
MACD 柱状图 = DIF - DEA < 0 → 信号 = 0
```

> **防未来函数机制**：所有信号统一 `shift(1)`，确保当日信号仅用于次日交易。

---

### 2. 动态风险平价算法

#### Step 1: EWMA 协方差估计

采用指数加权移动平均计算协方差矩阵，近期数据赋予更高权重：

```python
# 权重计算：指数衰减
weights = exp(linspace(-1, 0, n))
weights = weights / weights.sum()

# 加权协方差
cov_matrix = (demeaned.T @ (demeaned * weights))
```

#### Step 2: 动态风险预算

```python
# 基础风险预算：每个资产 20%
base_budget = [0.20, 0.20, 0.20, 0.20, 0.20]

# 动态调整：预算 × CTA 信号
dynamic_budget = base_budget × signals

# 极端防御：全信号为 0 时，国债独享 100% 预算
if all(signals == 0):
    dynamic_budget[bond_idx] = 1.0
```

#### Step 3: SLSQP 优化求解

```python
# 目标函数：最小化风险贡献偏离
minimize: Σ(risk_contribution - target_budget)²

# 约束条件
s.t.
    Σw = 1              # 权重和为 1
    0 ≤ w_i ≤ 1.0       # 权重在 [0, 1] 之间

# L2 正则化：防止协方差矩阵奇异
cov_matrix += I × 1e-8
```

#### 风险贡献计算

```python
# 组合波动率
σ_p = √(w' Σ w)

# 边际风险贡献
MRC = Σw / σ_p

# 风险贡献
RC = w × MRC
RC_normalized = RC / ΣRC
```

---

### 3. 目标波动率控制与杠杆

#### 核心公式

```
预测波动率 (年化) = √(w' Σ w) × √252

杠杆乘数 = min(目标波动率 / 预测波动率, 最大杠杆)
         = min(6% / pred_vol, 2.0)
```

#### 动态调整机制

- 当市场波动率低时 → 提高杠杆至多 2 倍
- 当市场波动率高时 → 降低杠杆至 1 倍以下
- 确保组合年化波动率精准控制在目标水平附近

#### TRS 互换资金成本核算

```python
# 无风险利率 (年化)
risk_free_rate = 2.0%

# 融资利率 (年化)
borrowing_rate = 3.5%

# 每日成本扣除
if leverage > 1:
    cost = (leverage - 1) × borrowing_rate / 252
    net_return = levered_return - cost
else:
    net_return = levered_return
```

---

### 4. 月频调仓机制

```python
def _get_rebalance_dates(freq='M'):
    """
    获取每月最后一个交易日作为调仓日
    """
    for i in range(len(dates)):
        if i == len(dates) - 1:
            rebalance_idx.append(i)  # 最后一天
        else:
            if dates[i].month != dates[i+1].month:
                rebalance_idx.append(i)  # 月末
```

---

### 5. 极端情况处理

| 场景 | 处理方案 |
|------|---------|
| 协方差矩阵奇异 | L2 正则化：`+ I × 1e-8` |
| 优化求解失败 | 回退至等权配置 `1/N` |
| 数据缺失 | 前向填充 `ffill()` |
| 权重 NaN | 前向填充延续上期 |
| 全信号为 0 | 国债 100% 风险预算 |

---

## 📊 实证绩效

### 回测参数设置

| 参数 | 配置值 |
|------|--------|
| 回测周期 | 2015-01-01 至今 |
| 目标波动率 | 6% (年化) |
| 最大杠杆 | 2.0x |
| 无风险利率 | 2.0% (年化) |
| 融资利率 | 3.5% (年化) |
| EWMA 窗口 | 126 交易日 (~6 个月) |
| 调仓频率 | 月频 |

### 核心绩效指标

| 指标 | CTA+ 风险平价 + 杠杆 | 等权 1/N 组合 | 静态风险平价 |
|------|---------------------|-------------|-------------|
| **总收益率** | 依据回测结果 | 基准对比 | 基准对比 |
| **年化收益率** | 策略核心 | 基准 1 | 基准 2 |
| **年化波动率** | **5.87%** (精准控波) | ~12% | ~8% |
| **夏普比率** | **0.72** | - | - |
| **卡玛比率** | 年化收益/7.55% | - | - |
| **最大回撤** | **-7.55%** | 显著更高 | 中等 |
| **平均杠杆** | **0.83x** | 1.0x | 1.0x |
| **胜率** | 依据日度统计 | - | - |
| **盈亏比** | 依据日度统计 | - | - |

### 绩效解读

1. **夏普比率 0.72**：在控波 6% 约束下实现优异风险调整后收益
2. **最大回撤 -7.55%**：CTA 信号动态调整有效降低极端风险暴露
3. **年化波动率 5.87%**：目标波动率机制精准控波，略低于 6% 目标
4. **平均杠杆 0.83x**：大部分时间预测波动率高于目标，杠杆<1

### 基准对比分析

| 对比维度 | 本策略优势 |
|---------|-----------|
| vs 等权 1/N | CTA 择时增强 + 风险平价分散 |
| vs 静态风险平价 | 动态预算调整 + 杠杆灵活调节 |
| vs 传统 60/40 | 五资产分散 + 趋势跟踪 Alpha |

---

## 📦 依赖环境

### Python 版本
- Python 3.8+

### 核心依赖

| 包 | 版本 | 用途 |
|----|------|------|
| `pandas` | >=1.3.0 | 数据处理/时间序列 |
| `numpy` | >=1.20.0 | 数值计算/矩阵运算 |
| `scipy` | >=1.7.0 | SLSQP 优化求解 |
| `matplotlib` | >=3.4.0 | 可视化绘图 |
| `tushare` | >=1.2.0 | Tushare Pro API |

### 安装命令

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 🔐 风险提示

1. **历史回测不代表未来表现**：本策略基于历史数据回测，实际投资需考虑交易成本、滑点等因素
2. **杠杆风险**：TRS 互换杠杆放大收益同时放大风险，最大杠杆 2.0x
3. **模型风险**：EWMA 协方差估计依赖历史数据，市场结构变化可能导致模型失效
4. **流动性风险**：ETF 标的可能存在流动性不足问题

---

<div align="center">

**Risk Parity + CTA 动态杠杆量化策略系统** | 风险平价基石 · CTA 增强 · 互换杠杆

*本文档最后更新：2025-03-24*

</div>