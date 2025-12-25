"""
交易策略模块
包含：配对交易策略 + 回测引擎 + 方法对比
"""

import numpy as np
import pandas as pd
from core_analysis import AdaptiveZScoreCalculator
from config import (
    DEFAULT_Z_ENTRY, DEFAULT_Z_EXIT, DEFAULT_LOOKBACK,
    DEFAULT_INITIAL_CAPITAL, DEFAULT_TRANSACTION_COST,
    DEFAULT_ALLOW_SHORT, DEFAULT_ZSCORE_METHOD,
    MAX_SINGLE_PNL_RATIO, Z_SCORE_EXPLOSION_THRESHOLD,
    Z_SCORE_STOP_THRESHOLD, MAX_HOLDING_DAYS
)


class PriceRatioPairsTrading:
    """
    配对交易策略 v3.0 - 使用价格比 + 支持双Z-score方法
    """
    
    def __init__(self, prices, stock1, stock2, 
                 z_entry=DEFAULT_Z_ENTRY, 
                 z_exit=DEFAULT_Z_EXIT, 
                 lookback=DEFAULT_LOOKBACK,
                 initial_capital=DEFAULT_INITIAL_CAPITAL, 
                 transaction_cost=DEFAULT_TRANSACTION_COST,
                 allow_short=DEFAULT_ALLOW_SHORT, 
                 zscore_method=DEFAULT_ZSCORE_METHOD):
        """
        初始化策略
        
        参数:
            prices: pd.DataFrame - 包含stock1和stock2的价格数据
            stock1: str - 股票1代码
            stock2: str - 股票2代码
            z_entry: float - 入场Z-score阈值
            z_exit: float - 出场Z-score阈值
            lookback: int - Z-score计算窗口
            initial_capital: float - 初始资金
            transaction_cost: float - 交易成本比例
            allow_short: bool - 是否允许做空
            zscore_method: str - Z-score计算方法：'traditional'或'robust'
        """
        self.prices = prices[[stock1, stock2]].dropna().copy()
        self.stock1 = stock1
        self.stock2 = stock2
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.lookback = lookback
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.allow_short = allow_short
        self.zscore_method = zscore_method
        
        # 记录交易信号
        self.signals = []
        
        # 计算指标
        self._calculate_indicators()
    
    def _calculate_indicators(self):
        """计算指标"""
        # 计算价格比
        self.prices['ratio'] = self.prices[self.stock1] / self.prices[self.stock2]
        
        # 使用AdaptiveZScoreCalculator计算Z-score
        self.prices['z_score'] = AdaptiveZScoreCalculator.calculate(
            self.prices['ratio'],
            lookback=self.lookback,
            method=self.zscore_method
        )
        
        # 去掉NaN
        self.prices = self.prices.dropna()
        
        # 验证数据质量
        if len(self.prices) == 0:
            raise ValueError("计算指标后数据为空，请检查输入数据")
        
        z_min = self.prices['z_score'].min()
        z_max = self.prices['z_score'].max()
        
        # 警告极端Z-score值
        if abs(z_min) > 20 or abs(z_max) > 20:
            print(f"    ⚠️ Z-score异常: [{z_min:.1f}, {z_max:.1f}]")
    
    def backtest(self):
        """
        回测策略
        
        返回:
            dict: 回测结果
        """
        print(f"\n{'='*60}")
        print(f"回测: {self.stock1} - {self.stock2}")
        print(f"方法: {self.zscore_method.upper()} Z-score")
        print(f"数据: {len(self.prices)}天 ({self.prices.index[0].date()} ~ {self.prices.index[-1].date()})")
        print(f"Z-score范围: [{self.prices['z_score'].min():.2f}, {self.prices['z_score'].max():.2f}]")
        print(f"{'='*60}\n")
        
        # 初始化
        capital = self.initial_capital
        position = 0  # 0=空仓, 1=做多价格比, -1=做空价格比
        entry_ratio = 0
        entry_date = None
        
        trades = []
        equity_curve = [capital]
        
        # 重置信号列表
        self.signals = []
        
        # 遍历每一天
        for i in range(len(self.prices)):
            current = self.prices.iloc[i]
            z_score = current['z_score']
            ratio = current['ratio']
            date = self.prices.index[i]
            
            # ===== 异常保护 =====
            
            # Z-score爆炸保护
            if abs(z_score) > Z_SCORE_EXPLOSION_THRESHOLD:
                if position != 0:
                    # 强制平仓
                    pnl_ratio = position * (ratio - entry_ratio) / entry_ratio
                    pnl_ratio = np.clip(pnl_ratio, -MAX_SINGLE_PNL_RATIO, MAX_SINGLE_PNL_RATIO)
                    pnl = capital * pnl_ratio
                    capital += pnl
                    
                    trades.append({
                        'Entry Date': entry_date,
                        'Exit Date': date,
                        'Position': 'Long Ratio' if position == 1 else 'Short Ratio',
                        'Entry Ratio': entry_ratio,
                        'Exit Ratio': ratio,
                        'Entry Z': self.prices.loc[entry_date, 'z_score'],
                        'Exit Z': z_score,
                        'P&L': pnl,
                        'P&L %': pnl_ratio * 100,
                        'Days': (date - entry_date).days,
                        'Exit Reason': 'Emergency Exit (Z爆炸)'
                    })
                    
                    self.signals.append({
                        'date': date,
                        'type': 'emergency_exit',
                        'position': position,
                        'z_score': z_score,
                        'ratio': ratio
                    })
                    
                    position = 0
                continue
            
            # 爆仓保护
            if capital <= 0:
                print(f"  ⚠️ 爆仓！资金={capital:.2f}")
                break
            
            # ===== 入场逻辑 =====
            if position == 0:
                
                # 做空价格比 (Z > +2.0)
                if z_score > self.z_entry and self.allow_short:
                    position = -1
                    entry_ratio = ratio
                    entry_date = date
                    capital *= (1 - self.transaction_cost)  # 入场交易成本
                    
                    self.signals.append({
                        'date': date,
                        'type': 'short_entry',
                        'position': -1,
                        'z_score': z_score,
                        'ratio': ratio
                    })
                
                # 做多价格比 (Z < -2.0)
                elif z_score < -self.z_entry:
                    position = 1
                    entry_ratio = ratio
                    entry_date = date
                    capital *= (1 - self.transaction_cost)  # 入场交易成本
                    
                    self.signals.append({
                        'date': date,
                        'type': 'long_entry',
                        'position': 1,
                        'z_score': z_score,
                        'ratio': ratio
                    })
            
            # ===== 出场逻辑 =====
            elif position != 0:
                should_exit = False
                exit_reason = ""
                
                # 1. 正常止盈：Z-score回归
                if abs(z_score) < self.z_exit:
                    should_exit = True
                    exit_reason = "Normal Exit"
                
                # 2. Z-score爆炸止损
                elif abs(z_score) > Z_SCORE_STOP_THRESHOLD:
                    should_exit = True
                    exit_reason = "Z-Score Explosion"
                
                # 3. 时间止损
                elif (date - entry_date).days > MAX_HOLDING_DAYS:
                    should_exit = True
                    exit_reason = "Time Stop"
                
                if should_exit:
                    # 计算盈亏
                    ratio_change = (ratio - entry_ratio) / entry_ratio
                    pnl_ratio = position * ratio_change
                    
                    # 限制单笔盈亏在±100%以内
                    pnl_ratio = np.clip(pnl_ratio, -MAX_SINGLE_PNL_RATIO, MAX_SINGLE_PNL_RATIO)
                    
                    pnl = capital * pnl_ratio
                    capital += pnl
                    capital *= (1 - self.transaction_cost)  # 出场交易成本
                    
                    # 记录交易
                    trades.append({
                        'Entry Date': entry_date,
                        'Exit Date': date,
                        'Position': 'Long Ratio' if position == 1 else 'Short Ratio',
                        'Entry Ratio': entry_ratio,
                        'Exit Ratio': ratio,
                        'Entry Z': self.prices.loc[entry_date, 'z_score'],
                        'Exit Z': z_score,
                        'P&L': pnl,
                        'P&L %': pnl_ratio * 100,
                        'Days': (date - entry_date).days,
                        'Exit Reason': exit_reason
                    })
                    
                    self.signals.append({
                        'date': date,
                        'type': 'exit',
                        'position': position,
                        'z_score': z_score,
                        'ratio': ratio,
                        'reason': exit_reason
                    })
                    
                    position = 0
            
            # 更新权益曲线
            equity_curve.append(capital)
        
        # ===== 计算统计指标 =====
        
        if len(trades) > 0:
            trades_df = pd.DataFrame(trades)
            
            # 总收益率
            total_return = (capital - self.initial_capital) / self.initial_capital
            
            # 交易统计
            num_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['P&L'] > 0])
            win_rate = winning_trades / num_trades if num_trades > 0 else 0
            
            # 夏普比率
            returns = trades_df['P&L %'].values / 100
            if returns.std() > 0:
                sharpe = returns.mean() / returns.std() * np.sqrt(252/30)
            else:
                sharpe = 0
            
            # 最大回撤
            equity_series = pd.Series(equity_curve)
            cummax = equity_series.cummax()
            drawdown = (equity_series - cummax) / cummax
            max_drawdown = drawdown.min()
            
            # 组装结果
            result = {
                'Total Return': total_return * 100,
                'Final Capital': capital,
                'Num Trades': num_trades,
                'Win Rate': win_rate * 100,
                'Sharpe Ratio': sharpe,
                'Max Drawdown': max_drawdown * 100,
                'Avg Trade': trades_df['P&L %'].mean(),
                'Best Trade': trades_df['P&L %'].max(),
                'Worst Trade': trades_df['P&L %'].min(),
                'Equity Curve': equity_curve,
                'Trades': trades_df,
                'Signals': pd.DataFrame(self.signals),
                'Method': self.zscore_method
            }
            
            # 打印摘要
            print(f"回测完成:")
            print(f"  总收益: {result['Total Return']:.2f}%")
            print(f"  夏普比率: {result['Sharpe Ratio']:.2f}")
            print(f"  交易次数: {result['Num Trades']}")
            print(f"  胜率: {result['Win Rate']:.1f}%")
            print(f"  最大回撤: {result['Max Drawdown']:.2f}%")
            
        else:
            # 没有交易
            print("  ⚠️ 没有产生任何交易")
            result = self._empty_result()
        
        return result
    
    def _empty_result(self, error=None):
        """没有交易时的空结果"""
        return {
            'Total Return': 0,
            'Final Capital': self.initial_capital,
            'Num Trades': 0,
            'Win Rate': 0,
            'Sharpe Ratio': 0,
            'Max Drawdown': 0,
            'Avg Trade': 0,
            'Best Trade': 0,
            'Worst Trade': 0,
            'Equity Curve': [self.initial_capital],
            'Trades': pd.DataFrame(),
            'Signals': pd.DataFrame(),
            'Method': self.zscore_method,
            'Error': error
        }


# ==================== 方法对比工具 ====================

def compare_methods_and_plot(prices, stock1, stock2, 
                             z_entry=DEFAULT_Z_ENTRY, 
                             z_exit=DEFAULT_Z_EXIT, 
                             lookback=DEFAULT_LOOKBACK,
                             initial_capital=DEFAULT_INITIAL_CAPITAL):
    """
    对比两种方法并生成完整的可视化报告
    
    返回:
        dict: 对比结果
    """
    print("\n" + "="*70)
    print("📊 Traditional vs Robust 完整对比分析")
    print("="*70)
    
    # 回测Traditional方法
    print("\n🔵 回测Traditional Z-score...")
    strategy_trad = PriceRatioPairsTrading(
        prices, stock1, stock2,
        z_entry=z_entry, z_exit=z_exit, lookback=lookback,
        initial_capital=initial_capital,
        zscore_method='traditional'
    )
    result_trad = strategy_trad.backtest()
    
    # 回测Robust方法
    print("\n🟢 回测Robust Z-score...")
    strategy_robust = PriceRatioPairsTrading(
        prices, stock1, stock2,
        z_entry=z_entry, z_exit=z_exit, lookback=lookback,
        initial_capital=initial_capital,
        zscore_method='robust'
    )
    result_robust = strategy_robust.backtest()
    
    # 🔧 修复：生成对比表格 - 确保所有列都是字符串类型
    comparison_df = pd.DataFrame({
        '指标': [
            '总收益率 (%)',
            '最终资金 ($)',
            '交易次数',
            '胜率 (%)',
            '夏普比率',
            '最大回撤 (%)',
            '平均盈亏 (%)',
            '最佳交易 (%)',
            '最差交易 (%)'
        ],
        'Traditional': [
            f"{result_trad['Total Return']:.2f}",           # 字符串
            f"${result_trad['Final Capital']:,.0f}",        # 字符串（带$符号）
            f"{result_trad['Num Trades']}",                 # 🔧 转为字符串
            f"{result_trad['Win Rate']:.1f}",               # 字符串
            f"{result_trad['Sharpe Ratio']:.2f}",           # 字符串
            f"{result_trad['Max Drawdown']:.2f}",           # 字符串
            f"{result_trad['Avg Trade']:.2f}",              # 字符串
            f"{result_trad['Best Trade']:.2f}",             # 字符串
            f"{result_trad['Worst Trade']:.2f}"             # 字符串
        ],
        'Robust': [
            f"{result_robust['Total Return']:.2f}",
            f"${result_robust['Final Capital']:,.0f}",
            f"{result_robust['Num Trades']}",               # 🔧 转为字符串
            f"{result_robust['Win Rate']:.1f}",
            f"{result_robust['Sharpe Ratio']:.2f}",
            f"{result_robust['Max Drawdown']:.2f}",
            f"{result_robust['Avg Trade']:.2f}",
            f"{result_robust['Best Trade']:.2f}",
            f"{result_robust['Worst Trade']:.2f}"
        ]
    })
    
    # 打印对比表格
    print("\n" + "="*70)
    print("📊 对比结果")
    print("="*70)
    print(comparison_df.to_string(index=False))
    print("="*70 + "\n")
    
    print("✅ 分析完成！\n")
    
    return {
        'traditional': result_trad,
        'robust': result_robust,
        'comparison_df': comparison_df,
        'strategy_trad': strategy_trad,
        'strategy_robust': strategy_robust
    }