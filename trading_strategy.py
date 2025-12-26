"""
交易策略模块
包含：配对交易策略 (实盘增强版 v3.2)
集成：Rolling Cointegration Check + 双腿P&L + 硬止损
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint  # <--- 新增引用：用于滚动检验
from core_analysis import AdaptiveZScoreCalculator
from config import (
    DEFAULT_Z_ENTRY, DEFAULT_Z_EXIT, DEFAULT_LOOKBACK,
    DEFAULT_INITIAL_CAPITAL, DEFAULT_TRANSACTION_COST,
    DEFAULT_ALLOW_SHORT, DEFAULT_ZSCORE_METHOD,
    MAX_SINGLE_PNL_RATIO, Z_SCORE_EXPLOSION_THRESHOLD,
    Z_SCORE_STOP_THRESHOLD, MAX_HOLDING_DAYS
)

# ==================== 风控配置 ====================
MAX_DRAWDOWN_STOP = 0.15      # 15% 最大回撤硬止损
SLIPPAGE = 0.0005             # 万分之五滑点 (双向)
ROLLING_CHECK_INTERVAL = 20   # 每20个交易日检查一次协整性
ROLLING_CHECK_WINDOW = 120    # 检查时使用过去120天的数据
ROLLING_P_VALUE_THRESHOLD = 0.10 # P值大于0.1则认为关系破裂

class PriceRatioPairsTrading:
    """
    配对交易策略 v3.2 - 实盘增强版
    特性: 
    1. 双腿交易 (Dual Leg Execution)
    2. 硬止损 (Hard Drawdown Stop)
    3. 滚动协整检查 (Rolling Cointegration Check)
    """
    
    def __init__(self, prices, stock1, stock2, 
                 z_entry=DEFAULT_Z_ENTRY, 
                 z_exit=DEFAULT_Z_EXIT, 
                 lookback=DEFAULT_LOOKBACK,
                 initial_capital=DEFAULT_INITIAL_CAPITAL, 
                 transaction_cost=DEFAULT_TRANSACTION_COST,
                 allow_short=DEFAULT_ALLOW_SHORT, 
                 zscore_method=DEFAULT_ZSCORE_METHOD,
                 hedge_ratio=1.0): 
        """
        初始化策略
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
        self.hedge_ratio = hedge_ratio
        
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
    
    def backtest(self):
        """
        执行回测 (包含滚动检查与资金管理)
        """
        print(f"\n{'='*60}")
        print(f"实盘级回测: {self.stock1} vs {self.stock2}")
        print(f"模式: 双腿交易 + 滚动协整检查")
        print(f"{'='*60}\n")
        
        capital = self.initial_capital
        max_capital = capital  # 用于计算回撤
        
        # 仓位状态
        position = 0  # 0=空仓, 1=做多比率(Long A/Short B), -1=做空比率(Short A/Long B)
        
        # 记录入场时的价格和股数
        entry_price_1 = 0
        entry_price_2 = 0
        shares_1 = 0
        shares_2 = 0
        entry_date = None
        entry_z = 0
        
        # 协整状态追踪
        is_relationship_valid = True
        next_check_day = ROLLING_CHECK_WINDOW  # 从数据足够长时开始检查
        
        trades = []
        equity_curve = [capital]
        self.signals = []
        
        # 遍历每一天
        for i in range(len(self.prices)):
            current = self.prices.iloc[i]
            date = self.prices.index[i]
            
            p1 = current[self.stock1]
            p2 = current[self.stock2]
            z_score = current['z_score']
            ratio = current['ratio']
            
            # ===== 1. 🛡️ 滚动协整检查 (防假死机制) =====
            # 定期检查关系是否还存在
            if i >= ROLLING_CHECK_WINDOW and i == next_check_day:
                # 提取过去一段窗口的数据
                history_s1 = self.prices[self.stock1].iloc[i-ROLLING_CHECK_WINDOW:i]
                history_s2 = self.prices[self.stock2].iloc[i-ROLLING_CHECK_WINDOW:i]
                
                try:
                    # 快速跑一个EG检验 (只看P-value)
                    _, p_value, _ = coint(history_s1, history_s2)
                    
                    if p_value > ROLLING_P_VALUE_THRESHOLD:
                        is_relationship_valid = False
                        # 记录警报信号
                        self.signals.append({
                            'date': date, 'type': 'relationship_broken', 
                            'position': position, 'ratio': ratio, 'z_score': z_score,
                            'note': f'P-val:{p_value:.3f} (失效)'
                        })
                    else:
                        is_relationship_valid = True
                        if not is_relationship_valid: # 如果之前是失效的，现在恢复了
                             self.signals.append({
                                'date': date, 'type': 'relationship_restored', 
                                'position': position, 'ratio': ratio, 'z_score': z_score,
                                'note': f'P-val:{p_value:.3f} (恢复)'
                            })
                except:
                    is_relationship_valid = False # 计算出错默认失效
                
                next_check_day += ROLLING_CHECK_INTERVAL # 设定下一次检查时间
            
            # ===== 2. 💰 实时市值与回撤计算 =====
            current_equity = capital
            if position != 0:
                floating_pnl = self._calculate_pnl(
                    position, shares_1, shares_2, 
                    entry_price_1, entry_price_2, 
                    p1, p2
                )
                current_equity += floating_pnl

            # 更新最大回撤
            max_capital = max(max_capital, current_equity)
            drawdown = (max_capital - current_equity) / max_capital if max_capital > 0 else 0
            equity_curve.append(current_equity)

            # ===== 3. 🚨 硬风控 (Hard Risk Control) =====
            
            # A. 爆仓保护
            if current_equity <= 0:
                print(f"  ❌ 账户爆仓于 {date.date()}")
                break
                
            # B. 最大回撤硬止损
            if position != 0 and drawdown > MAX_DRAWDOWN_STOP:
                realized_pnl = self._calculate_pnl(position, shares_1, shares_2, entry_price_1, entry_price_2, p1, p2)
                capital += realized_pnl
                capital -= self._calculate_cost(shares_1, p1, shares_2, p2)
                
                self._record_trade(trades, entry_date, date, position, entry_z, z_score, realized_pnl, "Hard Stop (Drawdown)")
                self.signals.append({'date': date, 'type': 'emergency_exit', 'position': position, 'ratio': ratio, 'z_score': z_score})
                position = 0
                continue 

            # C. Z-score 爆炸保护
            if abs(z_score) > Z_SCORE_EXPLOSION_THRESHOLD and position != 0:
                realized_pnl = self._calculate_pnl(position, shares_1, shares_2, entry_price_1, entry_price_2, p1, p2)
                capital += realized_pnl
                capital -= self._calculate_cost(shares_1, p1, shares_2, p2)
                
                self._record_trade(trades, entry_date, date, position, entry_z, z_score, realized_pnl, "Z-Score Explosion")
                self.signals.append({'date': date, 'type': 'emergency_exit', 'position': position, 'ratio': ratio, 'z_score': z_score})
                position = 0
                continue
            
            # ===== 4. 📈 交易逻辑 =====
            
            # 入场逻辑
            if position == 0:
                # 核心修改：只有关系有效 (is_relationship_valid) 才允许开仓
                if is_relationship_valid:
                    
                    # 做空价格比 (Short Ratio)
                    if z_score > self.z_entry and self.allow_short:
                        position = -1
                        entry_date = date
                        entry_price_1 = p1
                        entry_price_2 = p2
                        entry_z = z_score
                        
                        # 资金分配 (保留5%现金)
                        trade_capital = capital * 0.95
                        allocation = trade_capital / 2
                        
                        shares_1 = allocation / p1  # 卖空 A
                        shares_2 = allocation / p2  # 买入 B
                        
                        capital -= self._calculate_cost(shares_1, p1, shares_2, p2)
                        
                        self.signals.append({'date': date, 'type': 'short_entry', 'position': -1, 'ratio': ratio, 'z_score': z_score})
                    
                    # 做多价格比 (Long Ratio)
                    elif z_score < -self.z_entry:
                        position = 1
                        entry_date = date
                        entry_price_1 = p1
                        entry_price_2 = p2
                        entry_z = z_score
                        
                        trade_capital = capital * 0.95
                        allocation = trade_capital / 2
                        
                        shares_1 = allocation / p1  # 买入 A
                        shares_2 = allocation / p2  # 卖空 B
                        
                        capital -= self._calculate_cost(shares_1, p1, shares_2, p2)
                        
                        self.signals.append({'date': date, 'type': 'long_entry', 'position': 1, 'ratio': ratio, 'z_score': z_score})
            
            # 出场逻辑
            elif position != 0:
                should_exit = False
                exit_reason = ""
                
                # 1. 关系破裂强制离场 (新增)
                if not is_relationship_valid:
                    should_exit = True
                    exit_reason = "Relationship Broken"
                
                # 2. 正常回归
                elif abs(z_score) < self.z_exit:
                    should_exit = True
                    exit_reason = "Normal Exit"
                
                # 3. 时间止损
                elif (date - entry_date).days > MAX_HOLDING_DAYS:
                    should_exit = True
                    exit_reason = "Time Stop"
                
                if should_exit:
                    realized_pnl = self._calculate_pnl(position, shares_1, shares_2, entry_price_1, entry_price_2, p1, p2)
                    capital += realized_pnl
                    capital -= self._calculate_cost(shares_1, p1, shares_2, p2)
                    
                    self._record_trade(trades, entry_date, date, position, entry_z, z_score, realized_pnl, exit_reason)
                    self.signals.append({'date': date, 'type': 'exit', 'position': position, 'ratio': ratio, 'z_score': z_score, 'reason': exit_reason})
                    
                    position = 0
            
        # ===== 5. 生成报告 =====
        return self._generate_report(trades, equity_curve, capital)
    
    def _calculate_pnl(self, position, s1, s2, ep1, ep2, cp1, cp2):
        """
        计算双腿盈亏 (绝对金额)
        """
        if position == 1:
            # Long Ratio: Long A, Short B
            pnl_1 = (cp1 - ep1) * s1
            pnl_2 = (ep2 - cp2) * s2
        else:
            # Short Ratio: Short A, Long B
            pnl_1 = (ep1 - cp1) * s1
            pnl_2 = (cp2 - ep2) * s2
            
        return pnl_1 + pnl_2

    def _calculate_cost(self, s1, p1, s2, p2):
        """计算成本: 佣金 + 滑点"""
        val_1 = s1 * p1
        val_2 = s2 * p2
        commission = (val_1 + val_2) * self.transaction_cost
        slippage = (val_1 + val_2) * SLIPPAGE  # 增加滑点成本
        return commission + slippage

    def _record_trade(self, trades, entry_date, exit_date, position, entry_z, exit_z, pnl, reason):
        """记录交易"""
        trades.append({
            'Entry Date': entry_date,
            'Exit Date': exit_date,
            'Position': 'Long Ratio' if position == 1 else 'Short Ratio',
            'Entry Z': entry_z,
            'Exit Z': exit_z,
            'P&L': pnl,
            'Days': (exit_date - entry_date).days,
            'Exit Reason': reason
        })

    def _generate_report(self, trades, equity_curve, final_capital):
        """生成统计报告"""
        if len(trades) > 0:
            trades_df = pd.DataFrame(trades)
            trades_df['P&L %'] = (trades_df['P&L'] / self.initial_capital) * 100
            
            total_return = (final_capital - self.initial_capital) / self.initial_capital
            
            # 计算夏普
            returns = trades_df['P&L'] / self.initial_capital
            if returns.std() > 0:
                sharpe = returns.mean() / returns.std() * np.sqrt(252/30) # 假设平均持仓30天
            else:
                sharpe = 0
            
            # 计算最大回撤
            eq_series = pd.Series(equity_curve)
            cummax = eq_series.cummax()
            dd = (eq_series - cummax) / cummax
            max_drawdown = dd.min()
            
            return {
                'Total Return': total_return * 100,
                'Final Capital': final_capital,
                'Num Trades': len(trades),
                'Win Rate': len(trades_df[trades_df['P&L'] > 0]) / len(trades) * 100,
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
        else:
            return self._empty_result()

    def _empty_result(self, error=None):
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

# ==================== 兼容旧接口 ====================
# 如果需要保留 compare_methods_and_plot 函数，可以把之前的代码粘贴在这里
# 但因为逻辑变了（从Ratio交易变成了双腿交易），旧的对比函数可能需要微调才能跑通
# 建议暂时只用单策略回测

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