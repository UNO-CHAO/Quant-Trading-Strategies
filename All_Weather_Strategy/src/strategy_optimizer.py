# -*- coding: utf-8 -*-
"""
CTA择时信号与动态风险平价策略优化模块
=====================================
功能：
1. 宏观经济象限识别与风险预算分配
2. 多策略CTA择时信号生成（双均线、唐奇安通道、MACD）
3. 收益率滚动预测与风险测度评估
4. 基于动态风险预算的风险平价权重优化（SLSQP算法）
5. 月频调仓机制

经济象限模型：
- 增长超预期 + 通胀超预期 → 权益+商品
- 增长超预期 + 通胀低于预期 → 权益+债券
- 增长低于预期 + 通胀超预期 → 商品+黄金
- 增长低于预期 + 通胀低于预期 → 债券

作者：量化开发工程师
日期：2025-03-26
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from typing import Dict, Tuple, Optional
import warnings

# 导入宏观经济象限模型
from macro_regime import MacroRegimeDetector, RegimeRiskBudgetAllocator

warnings.filterwarnings('ignore')


# ============================================================================
# 模块一：CTA择时信号生成器
# ============================================================================

class CTASignalGenerator:
    """
    CTA择时信号生成器

    根据不同资产类别使用不同的择时策略：
    - 权益资产：双均线策略
    - 商品/黄金：唐奇安通道
    - 债券：MACD指标
    """

    # 资产分类配置
    EQUITY_ASSETS = ['000300.SH', '510880.SH']      # 权益类：沪深300、红利ETF
    COMMODITY_ASSETS = ['510410.SH', '518880.SH']   # 商品类：上证资源、黄金ETF
    BOND_ASSETS = ['H11009.CSI']                    # 债券类：10年期国债

    def __init__(self, price_df: pd.DataFrame):
        """
        初始化CTA信号生成器

        参数:
            price_df: 价格数据DataFrame，index为日期，columns为资产代码
        """
        self.price_df = price_df.copy()
        self.signals = pd.DataFrame(index=price_df.index, columns=price_df.columns, dtype=float)

    def generate_all_signals(self) -> pd.DataFrame:
        """
        为所有资产生成择时信号

        返回:
            信号DataFrame，1表示看多，0表示看空
        """
        print("\n[INFO] 开始生成CTA择时信号...")

        # 1. 权益资产：双均线策略
        for asset in self.EQUITY_ASSETS:
            if asset in self.price_df.columns:
                self.signals[asset] = self._dual_moving_average(
                    self.price_df[asset], fast_period=60, slow_period=120
                )
                print(f"  [双均线] {asset} 信号生成完成")

        # 2. 商品/黄金资产：唐奇安通道
        for asset in self.COMMODITY_ASSETS:
            if asset in self.price_df.columns:
                self.signals[asset] = self._donchian_channel(
                    self.price_df[asset], period=20
                )
                print(f"  [唐奇安通道] {asset} 信号生成完成")

        # 3. 债券资产：MACD
        for asset in self.BOND_ASSETS:
            if asset in self.price_df.columns:
                self.signals[asset] = self._macd_signal(
                    self.price_df[asset], fast_period=12, slow_period=26, signal_period=9
                )
                print(f"  [MACD] {asset} 信号生成完成")

        # 4. 防止未来函数：信号shift(1)
        print("\n[INFO] 执行信号位移(shift=1)，防止未来函数...")
        self.signals = self.signals.shift(1)

        # 5. 第一天NaN填充为1（默认看多）
        self.signals = self.signals.fillna(1)

        print(f"[INFO] CTA信号生成完成，信号矩阵形状: {self.signals.shape}")

        return self.signals

    def _dual_moving_average(self, price: pd.Series, fast_period: int, slow_period: int) -> pd.Series:
        """
        双均线策略

        参数:
            price: 价格序列
            fast_period: 快线周期
            slow_period: 慢线周期

        返回:
            信号序列：快线>慢线返回1，否则返回0
        """
        fast_ma = price.rolling(window=fast_period, min_periods=fast_period).mean()
        slow_ma = price.rolling(window=slow_period, min_periods=slow_period).mean()

        signal = (fast_ma > slow_ma).astype(float)
        return signal

    def _donchian_channel(self, price: pd.Series, period: int) -> pd.Series:
        """
        唐奇安通道策略

        参数:
            price: 价格序列
            period: 通道周期

        返回:
            信号序列：突破上轨返回1，跌破下轨返回0，中间保持
        """
        # 计算上轨（过去period天的最高价）
        upper_band = price.shift(1).rolling(window=period, min_periods=period).max()
        # 计算下轨（过去period天的最低价）
        lower_band = price.shift(1).rolling(window=period, min_periods=period).min()

        # 初始化信号
        signal = pd.Series(index=price.index, dtype=float)
        signal.iloc[0] = 1.0  # 初始默认看多

        # 生成信号
        for i in range(1, len(price)):
            if pd.isna(upper_band.iloc[i]) or pd.isna(lower_band.iloc[i]):
                signal.iloc[i] = signal.iloc[i-1]  # 数据不足时保持上一信号
            elif price.iloc[i] > upper_band.iloc[i]:
                signal.iloc[i] = 1.0  # 突破上轨，看多
            elif price.iloc[i] < lower_band.iloc[i]:
                signal.iloc[i] = 0.0  # 跌破下轨，看空
            else:
                signal.iloc[i] = signal.iloc[i-1]  # 中间状态，保持

        return signal

    def _macd_signal(self, price: pd.Series, fast_period: int, slow_period: int, signal_period: int) -> pd.Series:
        """
        MACD策略

        参数:
            price: 价格序列
            fast_period: 快线周期
            slow_period: 慢线周期
            signal_period: 信号线周期

        返回:
            信号序列：MACD柱状图>0返回1，<0返回0
        """
        # 计算EMA
        ema_fast = price.ewm(span=fast_period, adjust=False).mean()
        ema_slow = price.ewm(span=slow_period, adjust=False).mean()

        # 计算DIF（快线-慢线）
        dif = ema_fast - ema_slow

        # 计算DEA（信号线）
        dea = dif.ewm(span=signal_period, adjust=False).mean()

        # 计算MACD柱状图
        macd_histogram = dif - dea

        # 生成信号：柱状图>0看多，<0看空
        signal = (macd_histogram > 0).astype(float)

        return signal


# ============================================================================
# 模块二：收益率预测与风险测度评估
# ============================================================================

class ReturnForecaster:
    """
    收益率滚动预测器

    使用历史收益率均值进行滚动预测
    """

    def __init__(self, lookback_period: int = 126):
        """
        初始化预测器

        参数:
            lookback_period: 预测窗口
        """
        self.lookback_period = lookback_period

    def forecast_returns(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        滚动预测收益率

        参数:
            returns_df: 历史收益率数据

        返回:
            预测收益率DataFrame
        """
        forecast = returns_df.rolling(
            window=self.lookback_period,
            min_periods=20
        ).mean()

        return forecast

    def forecast_volatility(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        滚动预测波动率

        参数:
            returns_df: 历史收益率数据

        返回:
            预测波动率DataFrame（年化）
        """
        vol = returns_df.rolling(
            window=self.lookback_period,
            min_periods=20
        ).std() * np.sqrt(252)

        return vol


# ============================================================================
# 模块三：动态风险平价优化器
# ============================================================================

class RiskParityOptimizer:
    """
    动态风险平价优化器

    整合宏观经济象限模型和CTA择时信号，
    使用SLSQP算法求解风险平价权重。

    约束条件：
    - 权重和为1
    - 单资产权重上限40%
    - 权重非负
    """

    # 权重上限约束
    MAX_WEIGHT = 0.40  # 40%

    def __init__(self, returns_df: pd.DataFrame, signals_df: pd.DataFrame,
                 price_df: pd.DataFrame = None,
                 lookback_period: int = 126, ewma_span: int = 126,
                 max_weight: float = 0.40):
        """
        初始化风险平价优化器

        参数:
            returns_df: 收益率数据
            signals_df: CTA信号数据
            price_df: 价格数据（用于象限识别）
            lookback_period: 协方差矩阵计算窗口
            ewma_span: EWMA平滑参数
            max_weight: 单资产权重上限（默认40%）
        """
        self.returns_df = returns_df.copy()
        self.signals_df = signals_df.copy()
        self.price_df = price_df.copy() if price_df is not None else None
        self.lookback_period = lookback_period
        self.ewma_span = ewma_span
        self.max_weight = max_weight

        # 确保列对齐
        self.assets = list(returns_df.columns)
        self.n_assets = len(self.assets)

        # 基础风险预算
        self.base_risk_budget = np.array([0.20] * self.n_assets)

        # 初始化宏观经济象限模型
        self.regime_detector = None
        self.budget_allocator = None
        self.regime_df = None

        if self.price_df is not None:
            self._init_regime_model()

        # 收益率预测器
        self.forecaster = ReturnForecaster(lookback_period)
        self.forecast_returns_df = None
        self.forecast_vol_df = None

        # 结果存储
        self.target_weights = pd.DataFrame(
            index=returns_df.index,
            columns=self.assets,
            dtype=float
        )

    def _init_regime_model(self):
        """初始化宏观经济象限模型"""
        print("\n[INFO] 初始化宏观经济象限模型...")
        self.regime_detector = MacroRegimeDetector(self.price_df)
        self.regime_df = self.regime_detector.detect_regimes()
        self.budget_allocator = RegimeRiskBudgetAllocator(self.assets)

    def _forecast_risk_metrics(self):
        """预测收益率和波动率"""
        print("\n[INFO] 计算滚动收益率预测...")
        self.forecast_returns_df = self.forecaster.forecast_returns(self.returns_df)
        self.forecast_vol_df = self.forecaster.forecast_volatility(self.returns_df)
        print(f"[INFO] 预测完成，窗口: {self.lookback_period}天")

    def _calculate_ewma_covariance(self, date_idx: int) -> Optional[np.ndarray]:
        """
        计算EWMA协方差矩阵

        参数:
            date_idx: 当前日期索引位置

        返回:
            协方差矩阵（n_assets x n_assets）
        """
        # 获取历史数据窗口
        if date_idx < self.lookback_period:
            return None

        window_returns = self.returns_df.iloc[date_idx - self.lookback_period + 1: date_idx + 1]

        # 使用EWMA计算协方差矩阵
        # 方法：对收益率进行EWMA加权，然后计算协方差
        weights = np.exp(np.linspace(-1, 0, len(window_returns)))
        weights = weights / weights.sum()

        # 计算加权协方差矩阵
        mean_returns = (window_returns * weights[:, np.newaxis]).sum()
        demeaned = window_returns - mean_returns
        cov_matrix = (demeaned.T @ (demeaned * weights[:, np.newaxis])).values

        # 确保协方差矩阵正定
        cov_matrix = cov_matrix + np.eye(self.n_assets) * 1e-8

        return cov_matrix

    def _get_dynamic_risk_budget(self, date_idx: int) -> np.ndarray:
        """
        获取动态风险预算

        整合宏观经济象限模型和CTA择时信号：
        1. 根据经济象限获取基础风险预算系数
        2. 根据CTA信号调整预算
        3. 归一化

        参数:
            date_idx: 当前日期索引位置

        返回:
            归一化后的风险预算数组
        """
        # 当日CTA信号
        signals = self.signals_df.iloc[date_idx].values

        # 基础风险预算
        dynamic_budget = self.base_risk_budget.copy()

        # 1. 宏观经济象限调整
        if self.budget_allocator is not None and self.regime_df is not None:
            regime = self.regime_df['regime'].iloc[date_idx]
            regime_budget = self.budget_allocator.get_budget(regime, signals)
            dynamic_budget = regime_budget
        else:
            # 无象限模型时，仅使用CTA信号调整
            dynamic_budget = self.base_risk_budget * signals

        # 2. 极端防御机制：所有信号为0时
        if np.all(signals == 0):
            # 强制赋予国债全部预算
            dynamic_budget = np.zeros(self.n_assets)
            bond_idx = self.assets.index('H11009.CSI')
            dynamic_budget[bond_idx] = 1.0

        # 3. 归一化
        budget_sum = dynamic_budget.sum()
        if budget_sum > 0:
            dynamic_budget = dynamic_budget / budget_sum
        else:
            # 如果预算和为0（极端情况），平均分配
            dynamic_budget = np.ones(self.n_assets) / self.n_assets

        return dynamic_budget

    def _risk_contribution(self, weights: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
        """
        计算各资产的风险贡献

        参数:
            weights: 资产权重
            cov_matrix: 协方差矩阵

        返回:
            各资产的风险贡献
        """
        # 组合波动率
        portfolio_var = weights @ cov_matrix @ weights
        portfolio_std = np.sqrt(portfolio_var)

        # 边际风险贡献
        mrc = cov_matrix @ weights / portfolio_std

        # 风险贡献 = 权重 * 边际风险贡献
        risk_contrib = weights * mrc

        # 归一化
        if risk_contrib.sum() > 0:
            risk_contrib = risk_contrib / risk_contrib.sum()

        return risk_contrib

    def _objective_function(self, weights: np.ndarray, cov_matrix: np.ndarray,
                           target_budget: np.ndarray) -> float:
        """
        目标函数：最小化风险贡献与目标预算的平方差

        参数:
            weights: 资产权重
            cov_matrix: 协方差矩阵
            target_budget: 目标风险预算

        返回:
            目标函数值
        """
        risk_contrib = self._risk_contribution(weights, cov_matrix)
        return np.sum((risk_contrib - target_budget) ** 2)

    def _optimize_weights(self, cov_matrix: np.ndarray, target_budget: np.ndarray) -> np.ndarray:
        """
        使用SLSQP算法优化权重

        约束条件：
        - 权重和为1
        - 单资产权重上限40%
        - 权重非负

        参数:
            cov_matrix: 协方差矩阵
            target_budget: 目标风险预算

        返回:
            优化后的权重
        """
        # 初始权重：平均分配
        init_weights = np.ones(self.n_assets) / self.n_assets

        # 约束条件
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # 权重和为1
        ]

        # 边界条件：权重在[0, max_weight]之间，max_weight默认为40%
        bounds = tuple((0.0, self.max_weight) for _ in range(self.n_assets))

        # 优化求解
        result = minimize(
            fun=self._objective_function,
            x0=init_weights,
            args=(cov_matrix, target_budget),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-10}
        )

        if result.success:
            return result.x
        else:
            # 优化失败时返回平均权重
            return init_weights

    def _get_rebalance_dates(self, freq: str = 'M') -> list:
        """
        获取调仓日期（月频）

        参数:
            freq: 调仓频率，'M'表示月频

        返回:
            调仓日期索引列表
        """
        dates = self.returns_df.index
        rebalance_idx = []

        # 找出每月最后一个交易日
        for i in range(len(dates)):
            if i == len(dates) - 1:
                # 最后一天
                rebalance_idx.append(i)
            else:
                current_month = dates[i].month
                next_month = dates[i + 1].month
                if current_month != next_month:
                    # 月末
                    rebalance_idx.append(i)

        return rebalance_idx

    def run_optimization(self) -> pd.DataFrame:
        """
        执行优化流程

        流程：
        1. 预测收益率和波动率
        2. 识别经济象限
        3. 计算动态风险预算
        4. SLSQP优化权重

        返回:
            每日目标权重DataFrame
        """
        print("\n[INFO] 开始执行动态风险平价优化...")
        print(f"[INFO] 回看窗口: {self.lookback_period} 天")
        print(f"[INFO] EWMA参数: span={self.ewma_span}")
        print(f"[INFO] 权重上限: {self.max_weight*100:.0f}%")

        # 预测收益率和波动率
        self._forecast_risk_metrics()

        # 获取调仓日期
        rebalance_idx = self._get_rebalance_dates()
        print(f"[INFO] 共识别到 {len(rebalance_idx)} 个调仓日")

        # 初始化权重（前lookback_period天使用等权）
        equal_weights = np.ones(self.n_assets) / self.n_assets
        current_weights = equal_weights.copy()

        # 逐日处理
        optimized_count = 0
        for i in range(len(self.returns_df)):
            # 检查是否需要调仓
            if i in rebalance_idx and i >= self.lookback_period:
                # 计算协方差矩阵
                cov_matrix = self._calculate_ewma_covariance(i)

                if cov_matrix is not None:
                    # 获取动态风险预算（整合象限模型和CTA信号）
                    target_budget = self._get_dynamic_risk_budget(i)

                    # 优化权重
                    current_weights = self._optimize_weights(cov_matrix, target_budget)
                    optimized_count += 1

            # 记录当日权重
            self.target_weights.iloc[i] = current_weights

        print(f"[INFO] 共执行 {optimized_count} 次优化求解")

        # 检查权重有效性
        nan_count = self.target_weights.isna().sum().sum()
        if nan_count > 0:
            print(f"[WARNING] 发现 {nan_count} 个NaN值，使用前向填充处理")
            self.target_weights = self.target_weights.ffill()

        # 检查权重上限
        max_weight_found = self.target_weights.max().max()
        print(f"[INFO] 实际最大权重: {max_weight_found:.2%}")

        print(f"[INFO] 优化完成，权重矩阵形状: {self.target_weights.shape}")

        return self.target_weights

    def get_weights_summary(self) -> Dict:
        """
        获取权重统计摘要

        返回:
            统计信息字典
        """
        summary = {
            '平均权重': self.target_weights.mean().to_dict(),
            '权重标准差': self.target_weights.std().to_dict(),
            '最大权重': self.target_weights.max().to_dict(),
            '最小权重': self.target_weights.min().to_dict(),
        }
        return summary


# ============================================================================
# 主策略执行类
# ============================================================================

class StrategyOrchestrator:
    """
    策略编排器

    整合数据读取、宏观经济象限识别、信号生成、权重优化和结果输出
    """

    # 资产名称映射
    ASSET_NAMES = {
        '000300.SH': '沪深300指数',
        '510880.SH': '华泰柏瑞红利ETF',
        'H11009.CSI': '中债10年期国债总指数',
        '510410.SH': '上证资源ETF',
        '518880.SH': '华安黄金ETF'
    }

    def __init__(self, price_path: str = 'price_data.csv',
                 returns_path: str = 'returns_data.csv',
                 max_weight: float = 0.40):
        """
        初始化策略编排器

        参数:
            price_path: 价格数据文件路径
            returns_path: 收益率数据文件路径
            max_weight: 单资产权重上限（默认40%）
        """
        self.price_path = price_path
        self.returns_path = returns_path
        self.max_weight = max_weight
        self.price_df = None
        self.returns_df = None
        self.signals_df = None
        self.weights_df = None

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        加载数据

        返回:
            (价格数据, 收益率数据)
        """
        print("\n" + "=" * 70)
        print("【步骤1】数据加载")
        print("=" * 70)

        # 加载价格数据
        self.price_df = pd.read_csv(self.price_path, index_col=0, parse_dates=True)
        print(f"[INFO] 价格数据加载完成: {self.price_path}")
        print(f"       形状: {self.price_df.shape}, 时间范围: {self.price_df.index[0].strftime('%Y-%m-%d')} 至 {self.price_df.index[-1].strftime('%Y-%m-%d')}")

        # 加载收益率数据
        self.returns_df = pd.read_csv(self.returns_path, index_col=0, parse_dates=True)

        # 移除收益率列名中的 _return 后缀以匹配价格数据
        self.returns_df.columns = [col.replace('_return', '') for col in self.returns_df.columns]

        print(f"[INFO] 收益率数据加载完成: {self.returns_path}")
        print(f"       形状: {self.returns_df.shape}")

        return self.price_df, self.returns_df

    def generate_signals(self) -> pd.DataFrame:
        """
        生成CTA择时信号

        返回:
            信号DataFrame
        """
        print("\n" + "=" * 70)
        print("【步骤2】CTA择时信号生成")
        print("=" * 70)

        generator = CTASignalGenerator(self.price_df)
        self.signals_df = generator.generate_all_signals()

        # 打印信号统计
        print("\n[信号统计] 各资产看多信号占比:")
        for col in self.signals_df.columns:
            long_ratio = (self.signals_df[col] == 1).mean() * 100
            print(f"  {self.ASSET_NAMES.get(col, col)}: {long_ratio:.2f}%")

        return self.signals_df

    def optimize_weights(self) -> pd.DataFrame:
        """
        执行权重优化

        整合宏观经济象限模型和CTA择时信号

        返回:
            目标权重DataFrame
        """
        print("\n" + "=" * 70)
        print("【步骤3】动态风险平价优化")
        print("=" * 70)

        optimizer = RiskParityOptimizer(
            returns_df=self.returns_df,
            signals_df=self.signals_df,
            price_df=self.price_df,  # 传入价格数据用于象限识别
            lookback_period=126,
            ewma_span=126,
            max_weight=self.max_weight  # 权重上限
        )
        self.weights_df = optimizer.run_optimization()

        # 打印权重统计
        summary = optimizer.get_weights_summary()
        print("\n[权重统计] 各资产平均权重:")
        for asset, weight in summary['平均权重'].items():
            print(f"  {self.ASSET_NAMES.get(asset, asset)}: {weight:.4f}")

        return self.weights_df

    def save_results(self, output_path: str = None) -> None:
        """
        保存结果

        参数:
            output_path: 输出文件路径
        """
        import os

        print("\n" + "=" * 70)
        print("【步骤4】结果保存")
        print("=" * 70)

        if self.weights_df is not None:
            # 使用传入的保存路径或默认路径
            output_path = output_path if output_path else 'result/target_weights.csv'

            # 确保目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            self.weights_df.to_csv(output_path)
            print(f"[INFO] 目标权重已保存至: {output_path}")
        else:
            print("[ERROR] 无权重数据可保存")

    def run(self) -> pd.DataFrame:
        """
        执行完整策略流程

        返回:
            目标权重DataFrame
        """
        print("\n" + "=" * 70)
        print(" " * 15 + "CTA择时 + 动态风险平价策略")
        print("=" * 70)

        # 1. 加载数据
        self.load_data()

        # 2. 生成信号
        self.generate_signals()

        # 3. 优化权重
        self.optimize_weights()

        # 4. 保存结果
        self.save_results()

        # 5. 打印样本数据
        self._print_samples()

        return self.weights_df

    def _print_samples(self) -> None:
        """
        打印样本数据
        """
        print("\n" + "=" * 70)
        print("【结果展示】")
        print("=" * 70)

        # 信号样本
        print("\n[CTA信号样本] 前10个交易日:")
        print("-" * 70)
        sample_signals = self.signals_df.head(10).copy()
        sample_signals.columns = [self.ASSET_NAMES.get(c, c) for c in sample_signals.columns]
        print(sample_signals.to_string())

        # 权重样本
        print("\n[目标权重样本] 前10个交易日:")
        print("-" * 70)
        sample_weights = self.weights_df.head(10).copy()
        sample_weights.columns = [self.ASSET_NAMES.get(c, c) for c in sample_weights.columns]
        print(sample_weights.round(4).to_string())

        # 最近权重
        print("\n[最近调仓权重] 最后5个交易日:")
        print("-" * 70)
        recent_weights = self.weights_df.tail(5).copy()
        recent_weights.columns = [self.ASSET_NAMES.get(c, c) for c in recent_weights.columns]
        print(recent_weights.round(4).to_string())

        print("\n" + "=" * 70)
        print(" " * 25 + "策略优化完成")
        print("=" * 70 + "\n")


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
    PRICE_DATA_PATH = os.path.join(DATA_DIR, 'price_data.csv')
    RETURNS_DATA_PATH = os.path.join(DATA_DIR, 'returns_data.csv')
    # 输出数据路径
    TARGET_WEIGHTS_PATH = os.path.join(RESULT_DIR, 'target_weights.csv')

    # -------------------------------------------------------------------------
    # 创建策略编排器
    # -------------------------------------------------------------------------
    orchestrator = StrategyOrchestrator(
        price_path=PRICE_DATA_PATH,
        returns_path=RETURNS_DATA_PATH
    )

    # 执行完整策略
    target_weights = orchestrator.run()

    # 保存结果到 data 目录
    orchestrator.save_results(output_path=TARGET_WEIGHTS_PATH)