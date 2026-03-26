# -*- coding: utf-8 -*-
"""
宏观经济象限模型模块
====================
功能：
1. 基于现有资产数据构建增长和通胀代理指标
2. 识别四种宏观经济状态（增长/通胀象限）
3. 根据象限动态调整资产风险预算
4. 确保四种状态对风险的边际贡献相同

经济象限模型：
- 增长超预期 + 通胀超预期 → (Equity, Commodity)
- 增长超预期 + 通胀低于预期 → (Equity, Bonds)
- 增长低于预期 + 通胀超预期 → (Commodity, Gold)
- 增长低于预期 + 通胀低于预期 → (Bonds)

作者：量化开发工程师
日期：2025-03-26
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# 宏观经济象限识别器
# ============================================================================

class MacroRegimeDetector:
    """
    宏观经济象限识别器

    基于资产价格动量构建增长和通胀代理指标，
    将经济状态划分为四个象限。
    """

    # 资产分类
    EQUITY_ASSETS = ['000300.SH', '510880.SH']      # 权益类：沪深300、红利ETF
    COMMODITY_ASSETS = ['510410.SH', '518880.SH']   # 商品类：资源ETF、黄金ETF
    BOND_ASSETS = ['H11009.CSI']                    # 债券类：10年期国债

    # 象限定义
    REGIME_NAMES = {
        (1, 1): '增长超预期+通胀超预期',
        (1, -1): '增长超预期+通胀低于预期',
        (-1, 1): '增长低于预期+通胀超预期',
        (-1, -1): '增长低于预期+通胀低于预期'
    }

    # 象限对应的优势资产
    REGIME_ASSETS = {
        (1, 1): ['000300.SH', '510880.SH', '510410.SH'],      # 权益+商品
        (1, -1): ['000300.SH', '510880.SH', 'H11009.CSI'],    # 权益+债券
        (-1, 1): ['510410.SH', '518880.SH'],                   # 商品+黄金
        (-1, -1): ['H11009.CSI']                                # 债券
    }

    def __init__(self, price_df: pd.DataFrame,
                 growth_lookback: int = 60,
                 inflation_lookback: int = 60,
                 smooth_period: int = 20):
        """
        初始化象限识别器

        参数:
            price_df: 价格数据DataFrame
            growth_lookback: 增长动量计算窗口
            inflation_lookback: 通胀动量计算窗口
            smooth_period: 信号平滑窗口
        """
        self.price_df = price_df.copy()
        self.growth_lookback = growth_lookback
        self.inflation_lookback = inflation_lookback
        self.smooth_period = smooth_period

        self.regime_series = None
        self.growth_proxy = None
        self.inflation_proxy = None

    def _calculate_proxy_indicators(self) -> Tuple[pd.Series, pd.Series]:
        """
        计算增长和通胀代理指标

        增长代理：权益类资产加权动量
        通胀代理：商品类资产加权动量

        返回:
            (增长代理序列, 通胀代理序列)
        """
        # 计算各资产收益率
        returns = self.price_df.pct_change()

        # 增长代理：权益类资产等权组合收益率
        equity_cols = [c for c in self.EQUITY_ASSETS if c in returns.columns]
        if equity_cols:
            growth_proxy = returns[equity_cols].mean(axis=1)
        else:
            # 回退方案：使用第一个可用资产
            growth_proxy = returns.iloc[:, 0]

        # 通胀代理：商品类资产等权组合收益率
        commodity_cols = [c for c in self.COMMODITY_ASSETS if c in returns.columns]
        if commodity_cols:
            inflation_proxy = returns[commodity_cols].mean(axis=1)
        else:
            # 回退方案：使用最后一个资产
            inflation_proxy = returns.iloc[:, -1]

        # 计算滚动动量（累计收益率）
        growth_momentum = growth_proxy.rolling(
            window=self.growth_lookback, min_periods=self.growth_lookback
        ).mean() * 252  # 年化

        inflation_momentum = inflation_proxy.rolling(
            window=self.inflation_lookback, min_periods=self.inflation_lookback
        ).mean() * 252  # 年化

        # 平滑处理
        growth_smooth = growth_momentum.rolling(
            window=self.smooth_period, min_periods=1
        ).mean()

        inflation_smooth = inflation_momentum.rolling(
            window=self.smooth_period, min_periods=1
        ).mean()

        self.growth_proxy = growth_smooth
        self.inflation_proxy = inflation_smooth

        return growth_smooth, inflation_smooth

    def detect_regimes(self) -> pd.DataFrame:
        """
        识别经济象限

        返回:
            包含象限信息的DataFrame
        """
        print("\n[INFO] 开始识别宏观经济象限...")

        # 计算代理指标
        growth, inflation = self._calculate_proxy_indicators()

        # 判断方向：正值为1（超预期），负值为-1（低于预期）
        growth_signal = (growth > 0).astype(int) * 2 - 1  # 1或-1
        inflation_signal = (inflation > 0).astype(int) * 2 - 1  # 1或-1

        # 构建象限
        regime_df = pd.DataFrame(index=self.price_df.index)
        regime_df['growth_signal'] = growth_signal
        regime_df['inflation_signal'] = inflation_signal
        regime_df['regime'] = list(zip(growth_signal, inflation_signal))
        regime_df['regime_name'] = regime_df['regime'].map(self.REGIME_NAMES)

        # 填充初始NaN
        regime_df = regime_df.fillna(method='bfill')

        self.regime_series = regime_df

        # 统计各象限占比
        print("\n[象限统计]")
        regime_counts = regime_df['regime_name'].value_counts()
        for name, count in regime_counts.items():
            pct = count / len(regime_df) * 100
            print(f"  {name}: {count} 天 ({pct:.1f}%)")

        return regime_df

    def get_regime_assets(self, date_idx: int) -> list:
        """
        获取当前象限的优势资产

        参数:
            date_idx: 日期索引

        返回:
            优势资产代码列表
        """
        if self.regime_series is None:
            return []

        regime = self.regime_series['regime'].iloc[date_idx]
        return self.REGIME_ASSETS.get(regime, [])


# ============================================================================
# 象限风险预算分配器
# ============================================================================

class RegimeRiskBudgetAllocator:
    """
    象限风险预算分配器

    根据经济象限动态分配风险预算，
    确保四种状态对风险的边际贡献相同。
    """

    def __init__(self, assets: list,
                 base_risk_budget: float = 0.20,
                 regime_risk_weight: float = 0.40):
        """
        初始化风险预算分配器

        参数:
            assets: 资产代码列表
            base_risk_budget: 基础风险预算（每个资产）
            regime_risk_weight: 象限调整权重系数
        """
        self.assets = assets
        self.n_assets = len(assets)
        self.base_risk_budget = base_risk_budget
        self.regime_risk_weight = regime_risk_weight

        # 四种象限的风险贡献权重配置
        # 设计原则：确保四种状态对风险的边际贡献相同
        # 即每种象限下，优势资产获得额外风险预算
        self.regime_budget_config = {
            # (增长, 通胀): {资产: 风险预算系数}
            (1, 1): {   # 增长超预期+通胀超预期：权益+商品
                '000300.SH': 1.5, '510880.SH': 1.5,  # 权益增强
                '510410.SH': 1.5,                     # 商品增强
                '518880.SH': 0.5,                     # 黄金降低
                'H11009.CSI': 0.5                      # 债券降低
            },
            (1, -1): {  # 增长超预期+通胀低于预期：权益+债券
                '000300.SH': 1.5, '510880.SH': 1.5,  # 权益增强
                '510410.SH': 0.5,                     # 商品降低
                '518880.SH': 0.5,                     # 黄金降低
                'H11009.CSI': 1.5                      # 债券增强
            },
            (-1, 1): {  # 增长低于预期+通胀超预期：商品+黄金
                '000300.SH': 0.5, '510880.SH': 0.5,  # 权益降低
                '510410.SH': 1.5,                     # 商品增强
                '518880.SH': 1.5,                     # 黄金增强
                'H11009.CSI': 0.5                      # 债券降低
            },
            (-1, -1): { # 增长低于预期+通胀低于预期：债券
                '000300.SH': 0.5, '510880.SH': 0.5,  # 权益降低
                '510410.SH': 0.5,                     # 商品降低
                '518880.SH': 0.5,                     # 黄金降低
                'H11009.CSI': 2.0                      # 债券大幅增强
            }
        }

    def get_budget(self, regime: tuple, cta_signals: np.ndarray = None) -> np.ndarray:
        """
        根据象限和CTA信号获取动态风险预算

        参数:
            regime: 经济象限 (growth, inflation)
            cta_signals: CTA择时信号数组（可选）

        返回:
            风险预算数组
        """
        # 获取象限预算系数
        budget_config = self.regime_budget_config.get(regime, {})

        # 构建风险预算数组
        risk_budget = np.ones(self.n_assets) * self.base_risk_budget

        for i, asset in enumerate(self.assets):
            if asset in budget_config:
                risk_budget[i] = self.base_risk_budget * budget_config[asset]

        # 如果有CTA信号，叠加调整
        if cta_signals is not None:
            risk_budget = risk_budget * cta_signals

        # 归一化
        total = risk_budget.sum()
        if total > 0:
            risk_budget = risk_budget / total
        else:
            # 极端情况：平均分配
            risk_budget = np.ones(self.n_assets) / self.n_assets

        return risk_budget

    def calculate_marginal_risk_contribution(self,
                                              weights: np.ndarray,
                                              cov_matrix: np.ndarray) -> np.ndarray:
        """
        计算各资产对组合风险的边际贡献

        参数:
            weights: 资产权重
            cov_matrix: 协方差矩阵

        返回:
            边际风险贡献数组
        """
        # 组合波动率
        portfolio_var = weights @ cov_matrix @ weights
        portfolio_std = np.sqrt(portfolio_var)

        # 边际风险贡献 = d(Risk)/d(w)
        mrc = cov_matrix @ weights / portfolio_std

        return mrc

    def verify_equal_contribution(self,
                                   regime_budgets: Dict[tuple, np.ndarray],
                                   cov_matrix: np.ndarray) -> Dict:
        """
        验证四种象限的风险贡献是否相同

        参数:
            regime_budgets: 各象限的风险预算配置
            cov_matrix: 协方差矩阵

        返回:
            验证结果字典
        """
        results = {}

        for regime, budget in regime_budgets.items():
            # 假设权重等于风险预算（简化假设）
            weights = budget / budget.sum()

            # 计算边际风险贡献
            mrc = self.calculate_marginal_risk_contribution(weights, cov_matrix)

            # 风险贡献 = 权重 * 边际风险贡献
            risk_contrib = weights * mrc
            risk_contrib_pct = risk_contrib / risk_contrib.sum()

            results[regime] = {
                'risk_budget': budget,
                'risk_contribution': risk_contrib_pct,
                'total_risk': np.sqrt(weights @ cov_matrix @ weights)
            }

        # 计算各象限总风险的差异
        total_risks = [r['total_risk'] for r in results.values()]
        risk_std = np.std(total_risks)
        risk_mean = np.mean(total_risks)

        print("\n[象限风险贡献验证]")
        print(f"  各象限总风险标准差: {risk_std:.6f}")
        print(f"  各象限总风险均值: {risk_mean:.6f}")
        print(f"  风险差异系数: {risk_std/risk_mean:.2%}")

        if risk_std / risk_mean < 0.05:
            print("  ✓ 四种状态风险贡献基本相同（差异<5%）")
        else:
            print("  ⚠ 四种状态风险贡献存在差异")

        return results


# ============================================================================
# 集成策略编排器
# ============================================================================

class MacroEnhancedStrategy:
    """
    宏观增强策略

    整合宏观经济象限模型与CTA择时信号，
    实现动态风险预算分配。
    """

    ASSET_NAMES = {
        '000300.SH': '沪深300指数',
        '510880.SH': '华泰柏瑞红利ETF',
        'H11009.CSI': '中债10年期国债总指数',
        '510410.SH': '上证资源ETF',
        '518880.SH': '华安黄金ETF'
    }

    def __init__(self, price_path: str, returns_path: str,
                 output_path: str = None):
        """
        初始化策略

        参数:
            price_path: 价格数据路径
            returns_path: 收益率数据路径
            output_path: 输出路径
        """
        self.price_path = price_path
        self.returns_path = returns_path
        self.output_path = output_path

        self.price_df = None
        self.returns_df = None
        self.regime_detector = None
        self.budget_allocator = None
        self.target_weights = None

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """加载数据"""
        print("\n" + "=" * 70)
        print("【步骤1】数据加载")
        print("=" * 70)

        self.price_df = pd.read_csv(self.price_path, index_col=0, parse_dates=True)
        self.returns_df = pd.read_csv(self.returns_path, index_col=0, parse_dates=True)

        # 处理列名
        self.returns_df.columns = [col.replace('_return', '') for col in self.returns_df.columns]

        print(f"[INFO] 价格数据: {self.price_df.shape}")
        print(f"[INFO] 收益率数据: {self.returns_df.shape}")

        return self.price_df, self.returns_df

    def detect_regimes(self) -> pd.DataFrame:
        """识别经济象限"""
        print("\n" + "=" * 70)
        print("【步骤2】宏观经济象限识别")
        print("=" * 70)

        self.regime_detector = MacroRegimeDetector(self.price_df)
        regime_df = self.regime_detector.detect_regimes()

        return regime_df

    def calculate_risk_budgets(self, cov_matrix: np.ndarray) -> Dict:
        """计算风险预算并验证"""
        print("\n" + "=" * 70)
        print("【步骤3】风险预算计算与验证")
        print("=" * 70)

        assets = list(self.price_df.columns)
        self.budget_allocator = RegimeRiskBudgetAllocator(assets)

        # 计算各象限的风险预算
        regime_budgets = {}
        for regime in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
            budget = self.budget_allocator.get_budget(regime)
            regime_budgets[regime] = budget

        # 验证风险贡献相同
        verification = self.budget_allocator.verify_equal_contribution(
            regime_budgets, cov_matrix
        )

        return regime_budgets

    def run(self) -> pd.DataFrame:
        """执行完整策略流程"""
        print("\n" + "=" * 70)
        print(" " * 15 + "宏观经济象限模型 + 风险平价策略")
        print("=" * 70)

        # 1. 加载数据
        self.load_data()

        # 2. 识别象限
        regime_df = self.detect_regimes()

        # 3. 计算协方差矩阵
        cov_matrix = self.returns_df.cov().values * 252  # 年化

        # 4. 计算风险预算
        regime_budgets = self.calculate_risk_budgets(cov_matrix)

        # 5. 生成每日风险预算
        print("\n" + "=" * 70)
        print("【步骤4】生成动态风险预算序列")
        print("=" * 70)

        assets = list(self.price_df.columns)
        n_days = len(self.price_df)

        risk_budget_series = pd.DataFrame(
            index=self.price_df.index,
            columns=assets,
            dtype=float
        )

        for i in range(n_days):
            regime = regime_df['regime'].iloc[i]
            budget = self.budget_allocator.get_budget(regime)
            risk_budget_series.iloc[i] = budget

        # 6. 保存结果
        if self.output_path:
            risk_budget_series.to_csv(self.output_path)
            print(f"\n[INFO] 风险预算序列已保存至: {self.output_path}")

        self.target_weights = risk_budget_series

        # 7. 打印样本
        self._print_samples(regime_df, risk_budget_series)

        return risk_budget_series

    def _print_samples(self, regime_df: pd.DataFrame,
                       budget_df: pd.DataFrame) -> None:
        """打印样本数据"""
        print("\n" + "=" * 70)
        print("【结果展示】")
        print("=" * 70)

        # 象限样本
        print("\n[经济象限样本] 前10个交易日:")
        print("-" * 70)
        sample = regime_df[['regime_name']].head(10)
        print(sample.to_string())

        # 风险预算样本
        print("\n[风险预算样本] 前10个交易日:")
        print("-" * 70)
        sample_budget = budget_df.head(10).copy()
        sample_budget.columns = [self.ASSET_NAMES.get(c, c) for c in sample_budget.columns]
        print(sample_budget.round(4).to_string())

        # 象限转换统计
        print("\n[象限转换统计]:")
        regime_changes = (regime_df['regime'] != regime_df['regime'].shift(1)).sum()
        print(f"  总象限转换次数: {regime_changes}")
        print(f"  平均每象限持续天数: {len(regime_df) / max(regime_changes, 1):.1f}")


# ============================================================================
# 主程序入口
# ============================================================================
if __name__ == '__main__':
    import os

    # 路径配置
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(BASE_DIR)
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    RESULT_DIR = os.path.join(PROJECT_ROOT, 'result')

    os.makedirs(RESULT_DIR, exist_ok=True)

    # 输入输出路径
    PRICE_PATH = os.path.join(DATA_DIR, 'price_data.csv')
    RETURNS_PATH = os.path.join(DATA_DIR, 'returns_data.csv')
    OUTPUT_PATH = os.path.join(RESULT_DIR, 'regime_risk_budget.csv')

    # 执行策略
    strategy = MacroEnhancedStrategy(
        price_path=PRICE_PATH,
        returns_path=RETURNS_PATH,
        output_path=OUTPUT_PATH
    )

    risk_budget = strategy.run()

    print("\n" + "=" * 70)
    print(" " * 25 + "象限模型策略完成")
    print("=" * 70 + "\n")