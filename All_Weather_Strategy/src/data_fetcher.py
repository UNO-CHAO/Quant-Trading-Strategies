# -*- coding: utf-8 -*-
"""
量化投研数据获取与清洗模块
==========================
功能：从 Tushare API 获取多资产日线数据，进行数据对齐和清洗

作者：量化开发工程师
日期：2025-03-23
"""

import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import warnings

# 抑制警告
warnings.filterwarnings('ignore')


class TushareDataFetcher:
    """
    Tushare 数据获取类

    该类封装了 Tushare API 的初始化和数据获取功能，
    支持获取指数日线数据和基金（ETF）日线数据。
    """

    def __init__(self, token: str, api_url: str = 'http://lianghua.nanyangqiankun.top'):
        """
        初始化 Tushare API 连接

        参数:
            token: Tushare API token
            api_url: API 接口地址（默认为指定的私有接口）
        """
        self.token = token
        self.api_url = api_url
        self.pro = self._init_api()

    def _init_api(self):
        """
        初始化 Tushare pro_api 接口

        返回:
            Tushare pro_api 对象
        """
        pro = ts.pro_api(self.token)
        # 设置私有接口属性
        pro._DataApi__token = self.token
        pro._DataApi__http_url = self.api_url
        print(f"[INFO] Tushare API 初始化成功，接口地址: {self.api_url}")
        return pro

    def fetch_index_daily(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取指数日线数据

        参数:
            ts_code: 指数代码，如 '000300.SH'
            start_date: 开始日期，格式 'YYYY-MM-DD' 或 'YYYYMMDD'
            end_date: 结束日期，格式 'YYYY-MM-DD' 或 'YYYYMMDD'

        返回:
            包含日线数据的 DataFrame
        """
        # 统一日期格式为 YYYYMMDD
        start_date = start_date.replace('-', '')
        end_date = end_date.replace('-', '')

        try:
            df = self.pro.index_daily(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date
            )
            print(f"[INFO] 成功获取指数 {ts_code} 数据，共 {len(df)} 条记录")
            return df
        except Exception as e:
            print(f"[ERROR] 获取指数 {ts_code} 数据失败: {e}")
            return pd.DataFrame()

    def fetch_fund_daily(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取基金（ETF）日线数据

        参数:
            ts_code: 基金代码，如 '518880.SH'
            start_date: 开始日期，格式 'YYYY-MM-DD' 或 'YYYYMMDD'
            end_date: 结束日期，格式 'YYYY-MM-DD' 或 'YYYYMMDD'

        返回:
            包含日线数据的 DataFrame
        """
        # 统一日期格式为 YYYYMMDD
        start_date = start_date.replace('-', '')
        end_date = end_date.replace('-', '')

        try:
            df = self.pro.fund_daily(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date
            )
            print(f"[INFO] 成功获取基金 {ts_code} 数据，共 {len(df)} 条记录")
            return df
        except Exception as e:
            print(f"[ERROR] 获取基金 {ts_code} 数据失败: {e}")
            return pd.DataFrame()


class DataCleaner:
    """
    数据清洗类

    该类负责数据对齐、缺失值处理和收益率计算等数据清洗工作。
    """

    # 资产名称映射字典
    # 注：中证红利指数替换为华泰柏瑞红利ETF，南华综合指数替换为上证资源ETF
    ASSET_NAMES = {
        '000300.SH': '沪深300',
        '510880.SH': '红利ETF',        # 原 000922.SH 中证红利指数
        'H11009.CSI': '中债10年国债',
        '510410.SH': '上证资源ETF',    # 原 NH0100.NHF 南华综合指数（512400.SH上市较晚，改用510410）
        '518880.SH': '黄金ETF'
    }

    def __init__(self, missing_threshold: int = 10):
        """
        初始化数据清洗器

        参数:
            missing_threshold: 连续缺失天数阈值，超过此值将打印警告
        """
        self.missing_threshold = missing_threshold
        self.raw_data: Dict[str, pd.DataFrame] = {}
        self.aligned_data: Optional[pd.DataFrame] = None
        self.returns_data: Optional[pd.DataFrame] = None

    def add_data(self, ts_code: str, df: pd.DataFrame) -> None:
        """
        添加原始数据到清洗器

        参数:
            ts_code: 资产代码
            df: 包含日线数据的 DataFrame
        """
        if df.empty:
            print(f"[WARNING] 资产 {ts_code} 数据为空，已跳过")
            return
        self.raw_data[ts_code] = df.copy()

    def _process_single_data(self, df: pd.DataFrame, ts_code: str) -> pd.Series:
        """
        处理单个资产的数据

        参数:
            df: 原始日线数据
            ts_code: 资产代码

        返回:
            处理后的收盘价序列
        """
        # 确保必要的列存在
        if 'trade_date' not in df.columns or 'close' not in df.columns:
            print(f"[ERROR] 资产 {ts_code} 数据缺少必要列")
            return pd.Series()

        # 提取日期和收盘价
        df = df[['trade_date', 'close']].copy()

        # 转换日期格式
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')

        # 按日期排序（升序）
        df = df.sort_values('trade_date').reset_index(drop=True)

        # 设置日期为索引
        df.set_index('trade_date', inplace=True)

        # 重命名列为资产代码
        df.columns = [ts_code]

        return df[ts_code]

    def _check_missing_values(self, df: pd.DataFrame) -> None:
        """
        检查缺失值情况，对连续缺失超过阈值的情况发出警告

        参数:
            df: 对齐后的价格数据 DataFrame
        """
        for col in df.columns:
            # 找出缺失值位置
            is_missing = df[col].isna()

            if not is_missing.any():
                continue

            # 计算连续缺失的天数
            consecutive_groups = (is_missing != is_missing.shift()).cumsum()

            for group_id, group_df in df[is_missing].groupby(consecutive_groups[is_missing]):
                if len(group_df) > self.missing_threshold:
                    asset_name = self.ASSET_NAMES.get(col, col)
                    start_date = group_df.index[0].strftime('%Y-%m-%d')
                    end_date = group_df.index[-1].strftime('%Y-%m-%d')
                    print(f"[WARNING] 资产 {asset_name} ({col}) 在 {start_date} 至 {end_date} "
                          f"连续缺失 {len(group_df)} 天，超过阈值 {self.missing_threshold} 天")

    def align_and_merge(self, calendar_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        将所有资产数据对齐到交易日历并合并

        参数:
            calendar_df: 交易日历 DataFrame（可选，若不提供则使用已有数据的并集）

        返回:
            对齐后的价格宽表 DataFrame
        """
        if not self.raw_data:
            print("[ERROR] 没有可处理的数据")
            return pd.DataFrame()

        print("\n[INFO] 开始数据对齐和合并...")

        # 处理每个资产的数据
        processed_series = []
        for ts_code, df in self.raw_data.items():
            series = self._process_single_data(df, ts_code)
            if not series.empty:
                processed_series.append(series)

        if not processed_series:
            print("[ERROR] 没有有效的数据可供合并")
            return pd.DataFrame()

        # 合并所有序列到一个 DataFrame
        merged_df = pd.concat(processed_series, axis=1)

        # 如果提供了交易日历，重新索引到交易日历
        if calendar_df is not None and not calendar_df.empty:
            if 'trade_date' in calendar_df.columns:
                calendar_dates = pd.to_datetime(calendar_df['trade_date'], format='%Y%m%d')
                merged_df = merged_df.reindex(calendar_dates)

        # 按日期排序
        merged_df = merged_df.sort_index()

        # 检查缺失值
        print("\n[INFO] 检查缺失值情况...")
        self._check_missing_values(merged_df)

        # 前向填充缺失值
        print("[INFO] 使用前向填充处理缺失值...")
        merged_df = merged_df.ffill()

        # 如果开头有缺失值，使用后向填充
        if merged_df.iloc[0].isna().any():
            print("[INFO] 数据开头存在缺失值，使用后向填充处理...")
            merged_df = merged_df.bfill()

        # 存储对齐后的数据
        self.aligned_data = merged_df

        print(f"[INFO] 数据对齐完成，共 {len(merged_df)} 个交易日，{len(merged_df.columns)} 个资产")

        return merged_df

    def calculate_returns(self, price_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        计算各资产的每日收益率

        参数:
            price_df: 价格数据 DataFrame（若不提供则使用已对齐的数据）

        返回:
            收益率 DataFrame
        """
        if price_df is None:
            price_df = self.aligned_data

        if price_df is None or price_df.empty:
            print("[ERROR] 没有可用的价格数据")
            return pd.DataFrame()

        print("\n[INFO] 计算每日收益率...")

        # 计算日收益率：(P_t - P_{t-1}) / P_{t-1}
        returns_df = price_df.pct_change()

        # 重命名列
        returns_df.columns = [f"{col}_return" for col in returns_df.columns]

        # 存储收益率数据
        self.returns_data = returns_df

        print(f"[INFO] 收益率计算完成，共 {len(returns_df)} 条记录")

        return returns_df

    def get_summary(self) -> Dict:
        """
        获取数据汇总信息

        返回:
            包含数据汇总信息的字典
        """
        summary = {
            '资产数量': len(self.raw_data),
            '资产列表': list(self.raw_data.keys()),
        }

        if self.aligned_data is not None:
            summary['交易日数量'] = len(self.aligned_data)
            summary['起始日期'] = self.aligned_data.index[0].strftime('%Y-%m-%d')
            summary['结束日期'] = self.aligned_data.index[-1].strftime('%Y-%m-%d')

        if self.returns_data is not None:
            summary['收益率统计'] = self.returns_data.describe().to_dict()

        return summary


class QuantDataManager:
    """
    量化数据管理主类

    该类整合了数据获取和清洗功能，提供统一的数据管理接口。
    """

    # 资产配置：代码 -> (类型, 名称)
    # 类型：'index' 表示指数，'fund' 表示基金/ETF
    # 注：中证红利指数替换为华泰柏瑞红利ETF，南华综合指数替换为上证资源ETF
    ASSET_CONFIG = {
        '000300.SH': ('index', '沪深300指数'),
        '510880.SH': ('fund', '华泰柏瑞红利ETF'),      # 替换原中证红利指数
        'H11009.CSI': ('index', '中债10年期国债总指数'),
        '510410.SH': ('fund', '上证资源ETF'),          # 替换原南华综合指数（512400.SH上市较晚）
        '518880.SH': ('fund', '华安黄金ETF'),
    }

    def __init__(self, token: str, api_url: str = 'http://lianghua.nanyangqiankun.top',
                 missing_threshold: int = 10):
        """
        初始化量化数据管理器

        参数:
            token: Tushare API token
            api_url: API 接口地址
            missing_threshold: 连续缺失天数阈值
        """
        self.fetcher = TushareDataFetcher(token, api_url)
        self.cleaner = DataCleaner(missing_threshold)
        self.start_date = None
        self.end_date = None

    def fetch_all_data(self, start_date: str = '2015-01-01',
                       end_date: Optional[str] = None) -> None:
        """
        获取所有配置资产的数据

        参数:
            start_date: 开始日期，默认 '2015-01-01'
            end_date: 结束日期，默认为当前日期
        """
        # 设置日期范围
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        self.start_date = start_date
        self.end_date = end_date

        print(f"\n{'='*60}")
        print(f"[INFO] 开始获取数据")
        print(f"[INFO] 时间范围: {start_date} 至 {end_date}")
        print(f"{'='*60}\n")

        # 获取每个资产的数据
        for ts_code, (data_type, name) in self.ASSET_CONFIG.items():
            print(f"\n>>> 正在获取: {name} ({ts_code})")

            if data_type == 'index':
                df = self.fetcher.fetch_index_daily(ts_code, start_date, end_date)
            else:  # fund
                df = self.fetcher.fetch_fund_daily(ts_code, start_date, end_date)

            # 添加到清洗器
            self.cleaner.add_data(ts_code, df)

    def process_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        处理数据：对齐、合并、计算收益率

        返回:
            (价格数据 DataFrame, 收益率数据 DataFrame)
        """
        print(f"\n{'='*60}")
        print("[INFO] 开始数据处理流程")
        print(f"{'='*60}\n")

        # 对齐并合并数据
        price_df = self.cleaner.align_and_merge()

        # 计算收益率
        returns_df = self.cleaner.calculate_returns(price_df)

        return price_df, returns_df

    def get_price_data(self) -> Optional[pd.DataFrame]:
        """
        获取处理后的价格数据

        返回:
            价格数据 DataFrame
        """
        return self.cleaner.aligned_data

    def get_returns_data(self) -> Optional[pd.DataFrame]:
        """
        获取收益率数据

        返回:
            收益率数据 DataFrame
        """
        return self.cleaner.returns_data

    def print_summary(self) -> None:
        """
        打印数据汇总信息
        """
        summary = self.cleaner.get_summary()

        print(f"\n{'='*60}")
        print("数据汇总信息")
        print(f"{'='*60}")
        print(f"资产数量: {summary.get('资产数量', 'N/A')}")

        if '资产列表' in summary:
            print("\n资产列表:")
            for code in summary['资产列表']:
                _, name = self.ASSET_CONFIG.get(code, ('unknown', '未知'))
                print(f"  - {name} ({code})")

        if '交易日数量' in summary:
            print(f"\n交易日数量: {summary['交易日数量']}")
            print(f"起始日期: {summary.get('起始日期', 'N/A')}")
            print(f"结束日期: {summary.get('结束日期', 'N/A')}")

        if '收益率统计' in summary:
            print("\n收益率统计:")
            returns_stats = pd.DataFrame(summary['收益率统计'])
            print(returns_stats.to_string())

    def save_to_csv(self, price_path: str = 'price_data.csv',
                    returns_path: str = 'returns_data.csv') -> None:
        """
        将数据保存为 CSV 文件

        参数:
            price_path: 价格数据保存路径
            returns_path: 收益率数据保存路径
        """
        if self.cleaner.aligned_data is not None:
            self.cleaner.aligned_data.to_csv(price_path)
            print(f"[INFO] 价格数据已保存至: {price_path}")

        if self.cleaner.returns_data is not None:
            self.cleaner.returns_data.to_csv(returns_path)
            print(f"[INFO] 收益率数据已保存至: {returns_path}")


# ============================================================================
# 主程序入口
# ============================================================================
if __name__ == '__main__':
    import os

    # -------------------------------------------------------------------------
    # 配置参数
    # -------------------------------------------------------------------------
    # Tushare API Token（请替换为您自己的 token）
    TOKEN = "xxxxxxxxx"

    # API 接口地址
    API_URL = 'http://lianghua.nanyangqiankun.top'

    # 数据起始日期
    START_DATE = '2015-01-01'

    # 结束日期（None 表示使用当前日期）
    END_DATE = None

    # 连续缺失天数阈值
    MISSING_THRESHOLD = 10

    # -------------------------------------------------------------------------
    # 路径配置：基于脚本所在目录自动计算相对路径
    # -------------------------------------------------------------------------
    # 获取脚本所在目录 (src/)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # 项目根目录 (../)
    PROJECT_ROOT = os.path.dirname(BASE_DIR)
    # 数据文件夹 (../data/)
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

    # 确保数据目录存在
    os.makedirs(DATA_DIR, exist_ok=True)

    # 配置保存路径
    PRICE_DATA_PATH = os.path.join(DATA_DIR, 'price_data.csv')
    RETURNS_DATA_PATH = os.path.join(DATA_DIR, 'returns_data.csv')

    # -------------------------------------------------------------------------
    # 主程序执行
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print(" "*20 + "量化投研数据获取与清洗模块")
    print("="*70)

    # 1. 创建数据管理器
    manager = QuantDataManager(
        token=TOKEN,
        api_url=API_URL,
        missing_threshold=MISSING_THRESHOLD
    )

    # 2. 获取所有资产数据
    manager.fetch_all_data(start_date=START_DATE, end_date=END_DATE)

    # 3. 处理数据
    price_df, returns_df = manager.process_data()

    # 4. 打印汇总信息
    manager.print_summary()

    # 5. 保存数据到 CSV（使用配置好的路径）
    manager.save_to_csv(
        price_path=PRICE_DATA_PATH,
        returns_path=RETURNS_DATA_PATH
    )

    # -------------------------------------------------------------------------
    # 展示部分数据样本
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("价格数据样本（前5行）:")
    print("="*60)
    if price_df is not None and not price_df.empty:
        # 显示时重命名列为中文名称
        display_df = price_df.head().copy()
        display_df.columns = [QuantDataManager.ASSET_CONFIG.get(c, (None, c))[1]
                            for c in display_df.columns]
        print(display_df.to_string())

    print("\n" + "="*60)
    print("收益率数据样本（前5行）:")
    print("="*60)
    if returns_df is not None and not returns_df.empty:
        # 显示时重命名列为中文名称
        display_df = returns_df.head().copy()
        display_df.columns = [QuantDataManager.ASSET_CONFIG.get(
            c.replace('_return', ''), (None, c))[1] + '收益率'
            for c in display_df.columns]
        print(display_df.to_string())

    print("\n" + "="*70)
    print(" "*25 + "数据获取与清洗完成")
    print("="*70 + "\n")