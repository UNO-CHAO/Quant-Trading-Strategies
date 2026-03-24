# -*- coding: utf-8 -*-
"""
策略回测与绩效分析模块
======================
功能：
1. 目标波动率控制与动态杠杆计算
2. 互换资金成本扣除
3. 绩效评估与基准对比
4. 可视化展示

作者：量化开发工程师
日期：2025-03-23
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from typing import Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False


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

    生成净值曲线和权重堆积图
    """

    ASSET_NAMES = {
        '000300.SH': '沪深300指数',
        '510880.SH': '华泰柏瑞红利ETF',
        'H11009.CSI': '中债10年期国债总指数',
        '510410.SH': '上证资源ETF',
        '518880.SH': '华安黄金ETF'
    }

    def __init__(self, strategy_nav: pd.Series, benchmark1_nav: pd.Series,
                 benchmark2_nav: pd.Series, weights_df: pd.DataFrame,
                 leverage_series: pd.Series):
        """
        初始化可视化器

        参数:
            strategy_nav: 策略净值
            benchmark1_nav: 基准1净值
            benchmark2_nav: 基准2净值
            weights_df: 权重数据
            leverage_series: 杠杆序列
        """
        self.strategy_nav = strategy_nav
        self.benchmark1_nav = benchmark1_nav
        self.benchmark2_nav = benchmark2_nav
        self.weights_df = weights_df
        self.leverage_series = leverage_series

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


# ============================================================================
# 主回测流程类
# ============================================================================

class BacktestOrchestrator:
    """
    回测编排器

    整合回测执行、绩效分析和可视化
    """

    def __init__(self, returns_path: str = 'returns_data.csv',
                 weights_path: str = 'target_weights.csv'):
        """
        初始化回测编排器

        参数:
            returns_path: 收益率数据路径
            weights_path: 权重数据路径
        """
        self.returns_path = returns_path
        self.weights_path = weights_path
        self.engine = None
        self.analyzer = None
        self.visualizer = None

    def run(self, nav_save_path: str = None,
            chart_save_path: str = None,
            csv_save_path: str = None) -> Dict:
        """
        执行完整回测流程

        返回:
            绩效指标字典
        """
        print("\n" + "=" * 70)
        print(" " * 20 + "策略回测与绩效分析")
        print("=" * 70)

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
            leverage_series=self.engine.leverage_series
        )
        # 使用传入的图表路径或默认路径
        chart_path = chart_save_path if chart_save_path else 'backtest_results.png'
        self.visualizer.plot_results(chart_path)

        # 6. 保存净值数据
        self._save_nav_data(nav_save_path)

        # 7. 保存 CSV 结果
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

    # 输出数据路径
    NAV_DATA_PATH = os.path.join(RESULT_DIR, 'nav_data.csv')
    BACKTEST_CHART_PATH = os.path.join(RESULT_DIR, 'backtest_results.png')
    BACKTEST_CSV_PATH = os.path.join(RESULT_DIR, 'backtest_results.csv')

    # -------------------------------------------------------------------------
    # 创建回测编排器
    # -------------------------------------------------------------------------
    orchestrator = BacktestOrchestrator(
        returns_path=RETURNS_DATA_PATH,
        weights_path=WEIGHTS_DATA_PATH
    )

    # 执行回测（传入结果保存路径）
    metrics = orchestrator.run(
        nav_save_path=NAV_DATA_PATH,
        chart_save_path=BACKTEST_CHART_PATH,
        csv_save_path=BACKTEST_CSV_PATH
    )