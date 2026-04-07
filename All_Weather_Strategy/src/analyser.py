# -*- coding: utf-8 -*-
"""
策略回测与绩效分析模块
======================
功能：
1. 目标波动率控制与动态杠杆计算
2. 互换资金成本扣除
3. 绩效评估与基准对比
4. 可视化展示（含论文图表生成）

作者：量化开发工程师
日期：2025-03-23
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from typing import Dict, Tuple, Optional
import warnings
import os

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150


# ============================================================================
# 回测引擎类
# ============================================================================

class BacktestEngine:
    """
    策略回测引擎

    实现目标波动率控制、动态杠杆、资金成本扣除等功能
    """

    # 资产名称映射
    ASSET_NAMES = {
        '000300.SH': '沪深300指数',
        '510880.SH': '华泰柏瑞红利ETF',
        'H11009.CSI': '中债10年期国债总指数',
        '510410.SH': '上证资源ETF',
        '518880.SH': '华安黄金ETF'
    }

    def __init__(self, returns_path: str = 'returns_data.csv',
                 weights_path: str = 'target_weights.csv',
                 target_vol: float = 0.06,
                 max_leverage: float = 2.0,
                 risk_free_rate: float = 0.02,
                 borrowing_rate: float = 0.035,
                 lookback_period: int = 126):
        """
        初始化回测引擎

        参数:
            returns_path: 收益率数据路径
            weights_path: 目标权重数据路径
            target_vol: 年化目标波动率
            max_leverage: 最大杠杆倍数
            risk_free_rate: 无风险利率（年化）
            borrowing_rate: 融资利率（年化）
            lookback_period: 波动率计算窗口
        """
        self.target_vol = target_vol
        self.max_leverage = max_leverage
        self.risk_free_rate = risk_free_rate
        self.borrowing_rate = borrowing_rate
        self.lookback_period = lookback_period

        # 加载数据
        self.returns_df, self.weights_df = self._load_data(returns_path, weights_path)

        # 填充收益率数据中的NaN（第一行通常是NaN）
        self.returns_df = self.returns_df.fillna(0)

        # 结果存储
        self.leverage_series = None
        self.strategy_returns = None
        self.nav_series = None
        self.benchmark1_returns = None  # 等权1/N
        self.benchmark2_returns = None  # 静态风险平价
        self.benchmark1_nav = None
        self.benchmark2_nav = None

    def _load_data(self, returns_path: str, weights_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        加载数据

        参数:
            returns_path: 收益率数据路径
            weights_path: 权重数据路径

        返回:
            (收益率DataFrame, 权重DataFrame)
        """
        # 加载收益率
        returns_df = pd.read_csv(returns_path, index_col=0, parse_dates=True)
        # 移除列名中的 _return 后缀
        returns_df.columns = [col.replace('_return', '') for col in returns_df.columns]

        # 加载权重
        weights_df = pd.read_csv(weights_path, index_col=0, parse_dates=True)

        print(f"[INFO] 数据加载完成")
        print(f"       收益率数据: {returns_df.shape}")
        print(f"       权重数据: {weights_df.shape}")

        return returns_df, weights_df

    def _calculate_rolling_vol(self, date_idx: int) -> float:
        """
        计算滚动预测波动率

        参数:
            date_idx: 当前日期索引

        返回:
            年化预测波动率
        """
        if date_idx < self.lookback_period:
            # 数据不足时使用历史数据
            window_returns = self.returns_df.iloc[:date_idx + 1]
        else:
            window_returns = self.returns_df.iloc[date_idx - self.lookback_period + 1: date_idx + 1]

        if len(window_returns) < 20:
            return 0.10  # 默认10%波动率

        # 获取当日权重
        weights = self.weights_df.iloc[date_idx].values

        # 检查是否有NaN
        if np.any(np.isnan(weights)) or np.any(np.isnan(window_returns.values)):
            return 0.10

        # 计算EWMA协方差矩阵
        weights_ewma = np.exp(np.linspace(-1, 0, len(window_returns)))
        weights_ewma = weights_ewma / weights_ewma.sum()

        mean_returns = (window_returns.values * weights_ewma[:, np.newaxis]).sum(axis=0)
        demeaned = window_returns.values - mean_returns
        cov_matrix = (demeaned.T @ (demeaned * weights_ewma[:, np.newaxis]))

        # 组合方差
        portfolio_var = weights @ cov_matrix @ weights

        # 检查方差有效性
        if np.isnan(portfolio_var) or portfolio_var <= 0:
            return 0.10

        # 年化波动率
        annual_vol = np.sqrt(portfolio_var * 252)

        return max(annual_vol, 0.001)  # 避免除零

    def run_backtest(self) -> pd.Series:
        """
        执行回测

        返回:
            策略净值序列
        """
        print("\n" + "=" * 70)
        print("【回测执行】")
        print("=" * 70)
        print(f"[INFO] 目标波动率: {self.target_vol * 100:.1f}%")
        print(f"[INFO] 最大杠杆: {self.max_leverage:.1f}x")
        print(f"[INFO] 无风险利率: {self.risk_free_rate * 100:.1f}%")
        print(f"[INFO] 融资利率: {self.borrowing_rate * 100:.1f}%")

        n_days = len(self.returns_df)

        # 初始化结果数组
        leverage_arr = np.ones(n_days)
        strategy_returns_arr = np.zeros(n_days)

        # 逐日计算
        print("\n[INFO] 开始逐日回测...")

        for i in range(n_days):
            # 计算预测波动率
            pred_vol = self._calculate_rolling_vol(i)

            # 计算杠杆乘数
            leverage = min(self.target_vol / pred_vol, self.max_leverage)
            leverage = max(leverage, 0.0)  # 确保非负
            leverage_arr[i] = leverage

            # 获取当日权重和收益率
            weights = self.weights_df.iloc[i].values
            daily_returns = self.returns_df.iloc[i].values

            # 计算无杠杆组合收益
            unlevered_return = np.sum(weights * daily_returns)

            # 计算杠杆后收益
            levered_return = unlevered_return * leverage

            # 扣除融资成本（如果杠杆>1）
            if leverage > 1:
                borrowing_cost = (leverage - 1) * self.borrowing_rate / 252
                net_return = levered_return - borrowing_cost
            else:
                net_return = levered_return

            strategy_returns_arr[i] = net_return

        # 存储结果
        self.leverage_series = pd.Series(leverage_arr, index=self.returns_df.index)
        self.strategy_returns = pd.Series(strategy_returns_arr, index=self.returns_df.index)

        # 计算净值
        self.nav_series = (1 + self.strategy_returns).cumprod()

        print(f"[INFO] 回测完成，共 {n_days} 个交易日")
        print(f"[INFO] 平均杠杆: {leverage_arr.mean():.2f}x")
        print(f"[INFO] 杠杆范围: [{leverage_arr.min():.2f}, {leverage_arr.max():.2f}]")

        return self.nav_series

    def run_benchmarks(self) -> Tuple[pd.Series, pd.Series]:
        """
        运行基准策略

        返回:
            (基准1净值, 基准2净值)
        """
        print("\n" + "=" * 70)
        print("【基准策略计算】")
        print("=" * 70)

        n_days = len(self.returns_df)
        n_assets = len(self.returns_df.columns)

        # 基准1：等权1/N组合
        print("[INFO] 计算基准1：等权1/N组合...")
        equal_weights = np.ones(n_assets) / n_assets
        self.benchmark1_returns = self.returns_df.multiply(equal_weights).sum(axis=1)
        self.benchmark1_nav = (1 + self.benchmark1_returns).cumprod()

        # 基准2：静态风险平价（恒定20%风险预算，不加杠杆）
        print("[INFO] 计算基准2：静态风险平价组合...")
        static_budget = np.ones(n_assets) * 0.20

        # 使用简化方法：基于历史波动率的逆波动率加权
        hist_vol = self.returns_df.iloc[:252].std() * np.sqrt(252)
        inv_vol_weights = (1 / hist_vol) / (1 / hist_vol).sum()
        inv_vol_weights = inv_vol_weights.values

        self.benchmark2_returns = self.returns_df.multiply(inv_vol_weights).sum(axis=1)
        self.benchmark2_nav = (1 + self.benchmark2_returns).cumprod()

        print(f"[INFO] 基准1平均权重: {dict(zip(self.returns_df.columns, equal_weights))}")
        print(f"[INFO] 基准2平均权重: {dict(zip(self.returns_df.columns, inv_vol_weights.round(4)))}")

        return self.benchmark1_nav, self.benchmark2_nav


# ============================================================================
# 绩效评估类
# ============================================================================

class PerformanceAnalyzer:
    """
    绩效分析器

    计算各种绩效指标并生成报告
    """

    def __init__(self, strategy_nav: pd.Series, benchmark1_nav: pd.Series,
                 benchmark2_nav: pd.Series, risk_free_rate: float = 0.02):
        """
        初始化绩效分析器

        参数:
            strategy_nav: 策略净值序列
            benchmark1_nav: 基准1净值序列
            benchmark2_nav: 基准2净值序列
            risk_free_rate: 无风险利率
        """
        self.strategy_nav = strategy_nav
        self.benchmark1_nav = benchmark1_nav
        self.benchmark2_nav = benchmark2_nav
        self.risk_free_rate = risk_free_rate

        # 计算收益率
        self.strategy_returns = strategy_nav.pct_change().dropna()
        self.benchmark1_returns = benchmark1_nav.pct_change().dropna()
        self.benchmark2_returns = benchmark2_nav.pct_change().dropna()

    def calculate_metrics(self, nav: pd.Series, returns: pd.Series) -> Dict:
        """
        计算绩效指标

        参数:
            nav: 净值序列
            returns: 收益率序列

        返回:
            绩效指标字典
        """
        # 总收益率
        total_return = nav.iloc[-1] / nav.iloc[0] - 1

        # 年化收益率
        n_years = len(nav) / 252
        annual_return = (1 + total_return) ** (1 / n_years) - 1

        # 年化波动率
        annual_vol = returns.std() * np.sqrt(252)

        # 最大回撤
        cummax = nav.cummax()
        drawdown = (nav - cummax) / cummax
        max_drawdown = drawdown.min()

        # 夏普比率
        daily_rf = self.risk_free_rate / 252
        excess_returns = returns - daily_rf
        sharpe = excess_returns.mean() / returns.std() * np.sqrt(252)

        # 卡玛比率
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else np.inf

        # 胜率
        win_rate = (returns > 0).sum() / len(returns)

        # 盈亏比
        avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
        avg_loss = abs(returns[returns < 0].mean()) if (returns < 0).any() else 1
        profit_loss_ratio = avg_win / avg_loss if avg_loss != 0 else np.inf

        return {
            '总收益率': total_return,
            '年化收益率': annual_return,
            '年化波动率': annual_vol,
            '最大回撤': max_drawdown,
            '夏普比率': sharpe,
            '卡玛比率': calmar,
            '胜率': win_rate,
            '盈亏比': profit_loss_ratio
        }

    def generate_report(self) -> pd.DataFrame:
        """
        生成绩效对比报告

        返回:
            绩效对比DataFrame
        """
        print("\n" + "=" * 70)
        print("【绩效评估报告】")
        print("=" * 70)

        # 计算各策略指标
        strategy_metrics = self.calculate_metrics(self.strategy_nav, self.strategy_returns)
        benchmark1_metrics = self.calculate_metrics(self.benchmark1_nav, self.benchmark1_returns)
        benchmark2_metrics = self.calculate_metrics(self.benchmark2_nav, self.benchmark2_returns)

        # 构建对比表
        metrics_df = pd.DataFrame({
            'CTA+风险平价+杠杆': strategy_metrics,
            '等权1/N组合': benchmark1_metrics,
            '静态风险平价': benchmark2_metrics
        }).T

        # 格式化输出
        print("\n" + "-" * 70)
        print(f"{'指标':<15} {'CTA+风险平价+杠杆':>20} {'等权1/N组合':>15} {'静态风险平价':>15}")
        print("-" * 70)

        format_map = {
            '总收益率': '{:.2%}',
            '年化收益率': '{:.2%}',
            '年化波动率': '{:.2%}',
            '最大回撤': '{:.2%}',
            '夏普比率': '{:.2f}',
            '卡玛比率': '{:.2f}',
            '胜率': '{:.2%}',
            '盈亏比': '{:.2f}'
        }

        for metric in metrics_df.columns:
            fmt = format_map.get(metric, '{:.4f}')
            row = metrics_df[metric]
            print(f"{metric:<13} {fmt.format(row.iloc[0]):>20} {fmt.format(row.iloc[1]):>15} {fmt.format(row.iloc[2]):>15}")

        print("-" * 70)

        return metrics_df


# ============================================================================
# 可视化类
# ============================================================================

class BacktestVisualizer:
    """
    回测可视化器

    生成净值曲线、权重堆积图及论文图表
    """

    ASSET_NAMES = {
        '000300.SH': '沪深300指数',
        '510880.SH': '华泰柏瑞红利ETF',
        'H11009.CSI': '中债10年期国债总指数',
        '510410.SH': '上证资源ETF',
        '518880.SH': '华安黄金ETF'
    }

    # 简短资产名称（用于论文图表）
    ASSET_NAMES_SHORT = {
        '000300.SH': '沪深300',
        '510880.SH': '红利ETF',
        'H11009.CSI': '国债指数',
        '510410.SH': '资源ETF',
        '518880.SH': '黄金ETF'
    }

    def __init__(self, strategy_nav: pd.Series, benchmark1_nav: pd.Series,
                 benchmark2_nav: pd.Series, weights_df: pd.DataFrame,
                 leverage_series: pd.Series, returns_df: pd.DataFrame = None,
                 price_df: pd.DataFrame = None):
        """
        初始化可视化器

        参数:
            strategy_nav: 策略净值
            benchmark1_nav: 基准1净值
            benchmark2_nav: 基准2净值
            weights_df: 权重数据
            leverage_series: 杠杆序列
            returns_df: 收益率数据（用于论文图表）
            price_df: 价格数据（用于计算宏观象限）
        """
        self.strategy_nav = strategy_nav
        self.benchmark1_nav = benchmark1_nav
        self.benchmark2_nav = benchmark2_nav
        self.weights_df = weights_df
        self.leverage_series = leverage_series
        self.returns_df = returns_df
        self.price_df = price_df

    def plot_results(self, save_path: str = 'backtest_results.png') -> None:
        """
        绘制结果图表

        参数:
            save_path: 保存路径
        """
        print("\n" + "=" * 70)
        print("【可视化生成】")
        print("=" * 70)

        # 创建图表
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # 图1：净值曲线对比
        ax1 = axes[0]
        ax1.plot(self.strategy_nav.index, self.strategy_nav.values,
                 label='CTA+风险平价+杠杆', linewidth=1.5, color='#E74C3C')
        ax1.plot(self.benchmark1_nav.index, self.benchmark1_nav.values,
                 label='等权1/N组合', linewidth=1.5, color='#3498DB', alpha=0.8)
        ax1.plot(self.benchmark2_nav.index, self.benchmark2_nav.values,
                 label='静态风险平价', linewidth=1.5, color='#2ECC71', alpha=0.8)

        ax1.set_title('策略净值曲线对比', fontsize=14, fontweight='bold')
        ax1.set_xlabel('日期', fontsize=11)
        ax1.set_ylabel('净值', fontsize=11)
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(self.strategy_nav.index[0], self.strategy_nav.index[-1])

        # 添加关键事件标注
        max_nav_idx = self.strategy_nav.idxmax()
        max_nav = self.strategy_nav.max()
        ax1.annotate(f'最高: {max_nav:.2f}',
                     xy=(max_nav_idx, max_nav),
                     xytext=(10, 10), textcoords='offset points',
                     fontsize=9, color='#E74C3C')

        # 图2：月度权重堆积图
        ax2 = axes[1]

        # 转换为月度数据（取每月最后一个交易日）
        monthly_weights = self.weights_df.resample('M').last()
        monthly_leverage = self.leverage_series.resample('M').last()

        # 准备堆积数据
        colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']
        labels = [self.ASSET_NAMES.get(col, col) for col in monthly_weights.columns]

        # 绘制堆积面积图
        ax2.stackplot(monthly_weights.index, monthly_weights.values.T,
                      labels=labels, colors=colors, alpha=0.8)

        # 添加杠杆线（次坐标轴）
        ax2_twin = ax2.twinx()
        ax2_twin.plot(monthly_leverage.index, monthly_leverage.values,
                      color='black', linewidth=1.5, linestyle='--', label='杠杆倍数')
        ax2_twin.set_ylabel('杠杆倍数', fontsize=11)
        ax2_twin.set_ylim(0, 2.5)
        ax2_twin.legend(loc='upper right', fontsize=10)

        ax2.set_title('月度资产权重堆积图 + 杠杆倍数', fontsize=14, fontweight='bold')
        ax2.set_xlabel('日期', fontsize=11)
        ax2.set_ylabel('权重', fontsize=11)
        ax2.set_ylim(0, 1.1)
        ax2.legend(loc='upper left', fontsize=9, ncol=2)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图表
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"[INFO] 图表已保存至: {save_path}")

        plt.close()

    # =========================================================================
    # 论文图表生成方法
    # =========================================================================

    def _compute_macro_regime(self) -> pd.Series:
        """
        计算宏观象限序列

        增长代理: 权益类(000300.SH, 510880.SH)等权60日年化动量
        通胀代理: 商品类(510410.SH, 518880.SH)等权60日年化动量

        返回:
            象限序列 (1/2/3/4)
        """
        if self.price_df is None:
            raise ValueError("需要价格数据来计算宏观象限")

        # 计算日收益率
        ret = self.price_df.pct_change()

        # 权益类等权收益
        equity_cols = [c for c in ret.columns if c in ['000300.SH', '510880.SH']]
        equity_ret = ret[equity_cols].mean(axis=1)

        # 商品类等权收益
        commodity_cols = [c for c in ret.columns if c in ['510410.SH', '518880.SH']]
        commodity_ret = ret[commodity_cols].mean(axis=1)

        # 60日年化动量
        growth_momentum = equity_ret.rolling(60).mean() * 252
        inflation_momentum = commodity_ret.rolling(60).mean() * 252

        # 象限划分
        growth_signal = (growth_momentum > 0).astype(int)
        inflation_signal = (inflation_momentum > 0).astype(int)

        regime = pd.Series(index=self.price_df.index, dtype=int)
        regime[(growth_signal == 1) & (inflation_signal == 1)] = 1
        regime[(growth_signal == 1) & (inflation_signal == 0)] = 2
        regime[(growth_signal == 0) & (inflation_signal == 1)] = 3
        regime[(growth_signal == 0) & (inflation_signal == 0)] = 4

        return regime.dropna().astype(int)

    def plot_drawdown(self, save_path: str = None) -> None:
        """
        图5: 三策略滚动最大回撤对比图

        参数:
            save_path: 保存路径（可选）
        """
        print("\n[INFO] 生成图5: 滚动最大回撤对比图...")

        fig, ax = plt.subplots(figsize=(14, 5))

        # 构建净值DataFrame
        nav_df = pd.DataFrame({
            'CTA+风险平价+杠杆': self.strategy_nav,
            '等权1/N组合': self.benchmark1_nav,
            '静态风险平价': self.benchmark2_nav
        })

        cols = nav_df.columns
        labels = cols.tolist()
        colors = ['#E74C3C', '#3498DB', '#2ECC71']

        for i, col in enumerate(cols):
            nav = nav_df[col]
            cummax = nav.cummax()
            drawdown = (nav - cummax) / cummax
            ax.fill_between(drawdown.index, drawdown.values, 0,
                            alpha=0.3, color=colors[i])
            ax.plot(drawdown.index, drawdown.values,
                    label=labels[i], linewidth=1.0, color=colors[i])

        # 标注关键事件
        events = [
            ('2015-07-01', '2015年\n股灾'),
            ('2018-10-01', '2018年\n贸易摩擦'),
            ('2020-03-01', '2020年\n疫情冲击'),
        ]
        for date_str, label in events:
            date = pd.Timestamp(date_str)
            if date >= nav_df.index[0] and date <= nav_df.index[-1]:
                ax.axvline(x=date, color='gray', linestyle=':', alpha=0.5)
                ax.text(date, ax.get_ylim()[0] * 0.15, label,
                        fontsize=8, ha='center', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                                  edgecolor='gray', alpha=0.8))

        ax.set_title('图5  三策略滚动最大回撤对比', fontsize=14, fontweight='bold')
        ax.set_xlabel('日期', fontsize=11)
        ax.set_ylabel('回撤幅度', fontsize=11)
        ax.legend(loc='lower left', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(nav_df.index[0], nav_df.index[-1])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))

        plt.tight_layout()
        path = save_path if save_path else 'figure5_drawdown.png'
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"[INFO] 图5已保存: {path}")

    def plot_macro_regime(self, save_path: str = None) -> pd.Series:
        """
        图6: 宏观象限时序分布图

        参数:
            save_path: 保存路径（可选）

        返回:
            象限序列
        """
        print("\n[INFO] 生成图6: 宏观象限时序分布图...")

        regime = self._compute_macro_regime()

        fig, ax = plt.subplots(figsize=(14, 3.5))

        regime_colors = {1: '#E74C3C', 2: '#3498DB', 3: '#F39C12', 4: '#2ECC71'}
        regime_labels = {
            1: 'I: 高增长+高通胀',
            2: 'II: 高增长+低通胀',
            3: 'III: 低增长+高通胀',
            4: 'IV: 低增长+低通胀'
        }

        # 逐段绘制颜色带
        prev_regime = regime.iloc[0]
        start_idx = regime.index[0]

        for idx in range(1, len(regime)):
            curr_regime = regime.iloc[idx]
            if curr_regime != prev_regime or idx == len(regime) - 1:
                end_idx = regime.index[idx]
                ax.axvspan(start_idx, end_idx,
                           alpha=0.6, color=regime_colors.get(prev_regime, 'gray'),
                           linewidth=0)
                start_idx = end_idx
                prev_regime = curr_regime

        # 图例
        legend_handles = [mpatches.Patch(color=regime_colors[r], alpha=0.6, label=regime_labels[r])
                          for r in [1, 2, 3, 4]]
        ax.legend(handles=legend_handles, loc='upper center', ncol=4, fontsize=9,
                  bbox_to_anchor=(0.5, 1.25))

        ax.set_title('图6  宏观经济象限时序分布', fontsize=14, fontweight='bold', pad=30)
        ax.set_xlabel('日期', fontsize=11)
        ax.set_yticks([])
        ax.set_xlim(regime.index[0], regime.index[-1])

        # 统计各象限占比
        regime_counts = regime.value_counts(normalize=True).sort_index()
        stats_text = '  |  '.join([f'{regime_labels[r]}: {pct:.1%}'
                                   for r, pct in regime_counts.items()])
        ax.text(0.5, -0.15, stats_text, transform=ax.transAxes,
                ha='center', fontsize=9, style='italic')

        plt.tight_layout()
        path = save_path if save_path else 'figure6_regime.png'
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"[INFO] 图6已保存: {path}")

        # 输出象限统计
        print("\n  宏观象限分布统计:")
        for r in [1, 2, 3, 4]:
            pct = regime_counts.get(r, 0)
            print(f"    {regime_labels[r]}: {pct:.2%} ({int(pct * len(regime))}个交易日)")

        return regime

    def plot_leverage_vol(self, save_path: str = None) -> None:
        """
        图7: 杠杆乘数与滚动波动率对比图

        参数:
            save_path: 保存路径（可选）
        """
        print("\n[INFO] 生成图7: 杠杆与波动率对比图...")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

        leverage = self.leverage_series

        # 上半部分: 杠杆乘数
        ax1.plot(leverage.index, leverage.values, color='#8E44AD', linewidth=0.8)
        ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='杠杆=1.0')
        ax1.fill_between(leverage.index, leverage.values, 1.0,
                         where=leverage.values > 1.0, alpha=0.2, color='red',
                         label='加杠杆区间')
        ax1.fill_between(leverage.index, leverage.values, 1.0,
                         where=leverage.values <= 1.0, alpha=0.2, color='green',
                         label='减杠杆区间')

        ax1.set_ylabel('杠杆乘数', fontsize=11)
        ax1.set_title('图7  杠杆乘数与组合滚动波动率动态', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(True, alpha=0.3)
        mean_lev = leverage.mean()
        ax1.axhline(y=mean_lev, color='gray', linestyle=':', alpha=0.5)
        ax1.text(leverage.index[-1], mean_lev, f' 均值={mean_lev:.2f}x',
                 va='center', fontsize=9, color='gray')

        # 下半部分: 126日滚动波动率
        strategy_ret = self.strategy_nav.pct_change().dropna()
        rolling_vol = strategy_ret.rolling(126).std() * np.sqrt(252)

        ax2.plot(rolling_vol.index, rolling_vol.values, color='#E67E22', linewidth=0.8,
                 label='组合实现波动率(126日滚动)')
        ax2.axhline(y=0.06, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
                    label='目标波动率 6%')
        ax2.fill_between(rolling_vol.index, rolling_vol.values, 0.06,
                         where=rolling_vol.values > 0.06, alpha=0.15, color='red')
        ax2.fill_between(rolling_vol.index, rolling_vol.values, 0.06,
                         where=rolling_vol.values <= 0.06, alpha=0.15, color='green')

        ax2.set_ylabel('年化波动率', fontsize=11)
        ax2.set_xlabel('日期', fontsize=11)
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
        ax2.set_xlim(self.strategy_nav.index[0], self.strategy_nav.index[-1])

        plt.tight_layout()
        path = save_path if save_path else 'figure7_leverage_vol.png'
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"[INFO] 图7已保存: {path}")

        # 输出统计
        print(f"\n  杠杆统计:")
        print(f"    平均杠杆: {leverage.mean():.4f}x")
        print(f"    中位杠杆: {leverage.median():.4f}x")
        print(f"    最小杠杆: {leverage.min():.4f}x")
        print(f"    最大杠杆: {leverage.max():.4f}x")
        vol_valid = rolling_vol.dropna()
        print(f"  实现波动率统计:")
        print(f"    均值: {vol_valid.mean():.4f}")
        print(f"    与目标偏差: {vol_valid.mean() - 0.06:.4f}")

    def plot_weights_stack(self, save_path: str = None) -> None:
        """
        图8: 月度资产权重堆积面积图

        参数:
            save_path: 保存路径（可选）
        """
        print("\n[INFO] 生成图8: 月度资产权重堆积面积图...")

        fig, ax = plt.subplots(figsize=(14, 5))

        # 月度取样
        monthly = self.weights_df.resample('M').last()

        colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']
        labels = [self.ASSET_NAMES_SHORT.get(c, c) for c in monthly.columns]

        ax.stackplot(monthly.index, monthly.values.T,
                     labels=labels, colors=colors, alpha=0.85)

        ax.set_title('图8  月度资产权重堆积面积图', fontsize=14, fontweight='bold')
        ax.set_xlabel('日期', fontsize=11)
        ax.set_ylabel('权重', fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(monthly.index[0], monthly.index[-1])
        ax.legend(loc='upper left', fontsize=9, ncol=5, bbox_to_anchor=(0, 1.12))
        ax.grid(True, alpha=0.3, axis='y')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))

        plt.tight_layout()
        path = save_path if save_path else 'figure8_weights.png'
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"[INFO] 图8已保存: {path}")

        # 输出各资产平均配置比例
        print(f"\n  各资产平均配置比例:")
        for col in self.weights_df.columns:
            name = self.ASSET_NAMES_SHORT.get(col, col)
            print(f"    {name}: {self.weights_df[col].mean():.2%}")

    def generate_regime_performance(self) -> None:
        """
        表9: 计算各象限下的策略绩效
        """
        print("\n[INFO] 计算表9: 各象限绩效统计...")

        regime = self._compute_macro_regime()
        strategy_ret = self.strategy_nav.pct_change()

        # 对齐日期
        common_idx = regime.index.intersection(strategy_ret.index)
        regime_aligned = regime.loc[common_idx]
        ret_aligned = strategy_ret.loc[common_idx]

        regime_labels = {
            1: 'I: 高增长+高通胀',
            2: 'II: 高增长+低通胀',
            3: 'III: 低增长+高通胀',
            4: 'IV: 低增长+低通胀'
        }

        print(f"\n  {'象限':<22} {'交易日占比':>10} {'年化收益率':>10} {'年化波动率':>10} {'夏普比率':>8}")
        print("  " + "-" * 64)

        for r in [1, 2, 3, 4]:
            mask = regime_aligned == r
            r_ret = ret_aligned[mask].dropna()
            if len(r_ret) < 10:
                continue
            pct = mask.sum() / len(regime_aligned)
            ann_ret = r_ret.mean() * 252
            ann_vol = r_ret.std() * np.sqrt(252)
            sharpe = (ann_ret - 0.02) / ann_vol if ann_vol > 0 else 0
            print(f"  {regime_labels[r]:<22} {pct:>9.1%} {ann_ret:>9.2%} {ann_vol:>9.2%} {sharpe:>8.2f}")

    def generate_annual_performance(self) -> None:
        """
        年度绩效概览
        """
        print("\n[INFO] 年度绩效概览:")

        years = sorted(set(self.strategy_nav.index.year))
        print(f"\n  {'年份':<6} {'策略收益':>8} {'等权收益':>8} {'策略回撤':>8}")
        print("  " + "-" * 36)

        for year in years:
            mask = self.strategy_nav.index.year == year
            if mask.sum() < 10:
                continue
            s = self.strategy_nav[mask]
            e = self.benchmark1_nav[mask]
            s_ret = s.iloc[-1] / s.iloc[0] - 1
            e_ret = e.iloc[-1] / e.iloc[0] - 1

            cummax = s.cummax()
            dd = ((s - cummax) / cummax).min()

            print(f"  {year:<6} {s_ret:>7.2%} {e_ret:>7.2%} {dd:>7.2%}")

    def generate_all_thesis_charts(self, result_dir: str) -> None:
        """
        生成所有论文图表

        参数:
            result_dir: 结果保存目录
        """
        print("\n" + "=" * 70)
        print("【论文图表批量生成】")
        print("=" * 70)

        # 图5: 滚动最大回撤
        self.plot_drawdown(os.path.join(result_dir, 'figure5_drawdown.png'))

        # 图6: 宏观象限分布
        self.plot_macro_regime(os.path.join(result_dir, 'figure6_regime.png'))

        # 图7: 杠杆与波动率
        self.plot_leverage_vol(os.path.join(result_dir, 'figure7_leverage_vol.png'))

        # 图8: 权重堆积图
        self.plot_weights_stack(os.path.join(result_dir, 'figure8_weights.png'))

        # 表9: 各象限绩效
        self.generate_regime_performance()

        # 年度绩效
        self.generate_annual_performance()

        print("\n" + "=" * 70)
        print("  所有论文图表生成完毕！")
        print(f"  输出目录: {result_dir}")
        print("=" * 70)


# ============================================================================
# 主回测流程类
# ============================================================================

class BacktestOrchestrator:
    """
    回测编排器

    整合回测执行、绩效分析和可视化
    """

    def __init__(self, returns_path: str = 'returns_data.csv',
                 weights_path: str = 'target_weights.csv',
                 price_path: str = 'price_data.csv'):
        """
        初始化回测编排器

        参数:
            returns_path: 收益率数据路径
            weights_path: 权重数据路径
            price_path: 价格数据路径（用于宏观象限计算）
        """
        self.returns_path = returns_path
        self.weights_path = weights_path
        self.price_path = price_path
        self.engine = None
        self.analyzer = None
        self.visualizer = None
        self.price_df = None
        self.returns_df_raw = None

    def run(self, nav_save_path: str = None,
            chart_save_path: str = None,
            csv_save_path: str = None,
            thesis_charts: bool = True,
            result_dir: str = None) -> Dict:
        """
        执行完整回测流程

        参数:
            nav_save_path: 净值数据保存路径
            chart_save_path: 主图表保存路径
            csv_save_path: 绩效指标CSV保存路径
            thesis_charts: 是否生成论文图表
            result_dir: 结果目录路径（用于论文图表保存）

        返回:
            绩效指标字典
        """
        print("\n" + "=" * 70)
        print(" " * 20 + "策略回测与绩效分析")
        print("=" * 70)

        # 加载价格数据（用于宏观象限计算）
        if self.price_path and os.path.exists(self.price_path):
            self.price_df = pd.read_csv(self.price_path, index_col=0, parse_dates=True)
            print(f"[INFO] 价格数据已加载: {self.price_df.shape}")
        else:
            print("[WARN] 价格数据未加载，部分论文图表将无法生成")

        # 加载原始收益率数据（用于论文图表）
        if self.returns_path and os.path.exists(self.returns_path):
            self.returns_df_raw = pd.read_csv(self.returns_path, index_col=0, parse_dates=True)
            self.returns_df_raw.columns = [c.replace('_return', '') for c in self.returns_df_raw.columns]

        # 1. 创建回测引擎
        self.engine = BacktestEngine(
            returns_path=self.returns_path,
            weights_path=self.weights_path,
            target_vol=0.06,
            max_leverage=2.0,
            risk_free_rate=0.02,
            borrowing_rate=0.035
        )

        # 2. 执行策略回测
        strategy_nav = self.engine.run_backtest()

        # 3. 运行基准策略
        benchmark1_nav, benchmark2_nav = self.engine.run_benchmarks()

        # 4. 绩效分析
        self.analyzer = PerformanceAnalyzer(
            strategy_nav=strategy_nav,
            benchmark1_nav=benchmark1_nav,
            benchmark2_nav=benchmark2_nav,
            risk_free_rate=0.02
        )
        metrics_df = self.analyzer.generate_report()

        # 5. 可视化
        self.visualizer = BacktestVisualizer(
            strategy_nav=strategy_nav,
            benchmark1_nav=benchmark1_nav,
            benchmark2_nav=benchmark2_nav,
            weights_df=self.engine.weights_df,
            leverage_series=self.engine.leverage_series,
            returns_df=self.returns_df_raw,
            price_df=self.price_df
        )
        # 使用传入的图表路径或默认路径
        chart_path = chart_save_path if chart_save_path else 'backtest_results.png'
        self.visualizer.plot_results(chart_path)

        # 6. 生成论文图表（如果启用）
        if thesis_charts:
            thesis_dir = result_dir if result_dir else os.path.dirname(chart_path) if chart_path else 'result'
            os.makedirs(thesis_dir, exist_ok=True)
            self.visualizer.generate_all_thesis_charts(thesis_dir)

        # 7. 保存净值数据
        self._save_nav_data(nav_save_path)

        # 8. 保存 CSV 结果
        if csv_save_path:
            self._save_metrics_csv(metrics_df, csv_save_path)

        print("\n" + "=" * 70)
        print(" " * 25 + "回测分析完成")
        print("=" * 70 + "\n")

        return metrics_df.to_dict()

    def _save_nav_data(self, save_path: str = None) -> None:
        """
        保存净值数据

        参数:
            save_path: 保存路径（可选）
        """
        nav_df = pd.DataFrame({
            'CTA风险平价杠杆策略': self.engine.nav_series,
            '等权1N组合': self.engine.benchmark1_nav,
            '静态风险平价': self.engine.benchmark2_nav,
            '杠杆倍数': self.engine.leverage_series
        })
        # 使用传入的保存路径或默认路径
        save_path = save_path if save_path else 'result/nav_data.csv'
        nav_df.to_csv(save_path)
        print(f"\n[INFO] 净值数据已保存至: {save_path}")

    def _save_metrics_csv(self, metrics_df: pd.DataFrame, csv_save_path: str) -> None:
        """
        保存绩效指标到CSV文件

        参数:
            metrics_df: 绩效指标DataFrame
            csv_save_path: 保存路径
        """
        metrics_df.to_csv(csv_save_path)
        print(f"[INFO] 绩效指标已保存至: {csv_save_path}")


# ============================================================================
# 主程序入口
# ============================================================================
if __name__ == '__main__':
    import os

    # -------------------------------------------------------------------------
    # 路径配置：基于脚本所在目录自动计算相对路径
    # -------------------------------------------------------------------------
    # 获取脚本所在目录 (src/)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # 项目根目录 (../)
    PROJECT_ROOT = os.path.dirname(BASE_DIR)
    # 数据文件夹 (../data/)
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    # 结果文件夹 (../result/)
    RESULT_DIR = os.path.join(PROJECT_ROOT, 'result')

    # 确保结果目录存在
    os.makedirs(RESULT_DIR, exist_ok=True)

    # 输入数据路径
    RETURNS_DATA_PATH = os.path.join(DATA_DIR, 'returns_data.csv')
    WEIGHTS_DATA_PATH = os.path.join(RESULT_DIR, 'target_weights.csv')
    PRICE_DATA_PATH = os.path.join(DATA_DIR, 'price_data.csv')

    # 输出数据路径
    NAV_DATA_PATH = os.path.join(RESULT_DIR, 'nav_data.csv')
    BACKTEST_CHART_PATH = os.path.join(RESULT_DIR, 'backtest_results.png')
    BACKTEST_CSV_PATH = os.path.join(RESULT_DIR, 'backtest_results.csv')

    # -------------------------------------------------------------------------
    # 创建回测编排器
    # -------------------------------------------------------------------------
    orchestrator = BacktestOrchestrator(
        returns_path=RETURNS_DATA_PATH,
        weights_path=WEIGHTS_DATA_PATH,
        price_path=PRICE_DATA_PATH
    )

    # 执行回测（传入结果保存路径）
    metrics = orchestrator.run(
        nav_save_path=NAV_DATA_PATH,
        chart_save_path=BACKTEST_CHART_PATH,
        csv_save_path=BACKTEST_CSV_PATH,
        thesis_charts=True,
        result_dir=RESULT_DIR
    )
