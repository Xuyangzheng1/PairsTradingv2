"""
交易策略模块
包含：配对交易策略 (实盘增强版 v4.0)
集成：Rolling Cointegration Check + 双腿P&L + 硬止损 + 动态对冲比率
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
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
ROLLING_CHECK_INTERVAL = 5   # 每20个交易日检查一次协整性
ROLLING_CHECK_WINDOW = 60    # 检查时使用过去120天的数据
ROLLING_P_VALUE_THRESHOLD = 0.10 # P值大于0.1则认为关系破裂

# ==================== 动态对冲比率配置 ====================
HEDGE_RATIO_RECALIBRATE_FREQ = 20  # 每20天重新计算一次对冲比率
HEDGE_RATIO_LOOKBACK = 60          # 计算对冲比率时使用过去60天数据
HEDGE_RATIO_METHOD = 'ols'         # 'ols' 或 'tls' (Total Least Squares)
MIN_HEDGE_RATIO = 0.1              # 对冲比率最小值（防止异常）
MAX_HEDGE_RATIO = 10.0             # 对冲比率最大值（防止异常）
HEDGE_RATIO_CHANGE_THRESHOLD = 0.3 # 对冲比率变化超过30%时触发预警

class PriceRatioPairsTrading:
    """
    配对交易策略 v4.0 - 实盘增强版
    特性: 
    1. 双腿交易 (Dual Leg Execution)
    2. 硬止损 (Hard Drawdown Stop)
    3. 滚动协整检查 (Rolling Cointegration Check)
    4. 动态对冲比率 (Dynamic Hedge Ratio) ⭐ NEW
    """
    
    def __init__(self, prices, stock1, stock2, 
                 z_entry=DEFAULT_Z_ENTRY, 
                 z_exit=DEFAULT_Z_EXIT, 
                 lookback=DEFAULT_LOOKBACK,
                 initial_capital=DEFAULT_INITIAL_CAPITAL, 
                 transaction_cost=DEFAULT_TRANSACTION_COST,
                 allow_short=DEFAULT_ALLOW_SHORT, 
                 zscore_method=DEFAULT_ZSCORE_METHOD,
                 hedge_ratio=1.0,
                 dynamic_hedge=True,
                 hedge_recalibrate_freq=HEDGE_RATIO_RECALIBRATE_FREQ,
                 hedge_method=HEDGE_RATIO_METHOD): 
        """
        初始化策略
        
        新增参数:
            dynamic_hedge: 是否使用动态对冲比率（默认True）
            hedge_recalibrate_freq: 对冲比率重新校准频率（天）
            hedge_method: 对冲比率计算方法 ('ols' 或 'tls')
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
        self.hedge_ratio = hedge_ratio  # 初始对冲比率
        
        # 动态对冲比率参数
        self.dynamic_hedge = dynamic_hedge
        self.hedge_recalibrate_freq = hedge_recalibrate_freq
        self.hedge_method = hedge_method
        
        # 记录交易信号
        self.signals = []
        
        # 记录对冲比率历史
        self.hedge_ratio_history = []
        
        # 计算指标
        self._calculate_indicators()
    
    def _calculate_hedge_ratio(self, stock1_prices, stock2_prices, method='ols'):
        """
        计算对冲比率
        
        Args:
            stock1_prices: 股票1的价格序列
            stock2_prices: 股票2的价格序列
            method: 'ols' 或 'tls' (Total Least Squares)
        
        Returns:
            float: 对冲比率 (beta)
        """
        if len(stock1_prices) < 20:
            return self.hedge_ratio  # 数据不足，返回默认值
        
        try:
            if method == 'ols':
                # 普通最小二乘法: Y = alpha + beta * X
                X = sm.add_constant(stock2_prices)
                model = sm.OLS(stock1_prices, X).fit()
                hedge_ratio = model.params[1]
                
            elif method == 'tls':
                # Total Least Squares (更稳健，考虑X和Y的误差)
                # 使用主成分分析实现TLS
                from scipy.linalg import svd
                
                # 标准化数据
                X = np.array(stock2_prices).reshape(-1, 1)
                Y = np.array(stock1_prices).reshape(-1, 1)
                
                X_mean = X.mean()
                Y_mean = Y.mean()
                
                data = np.hstack([X - X_mean, Y - Y_mean])
                
                # SVD分解
                U, S, Vt = svd(data, full_matrices=False)
                
                # TLS解（第二个主成分的斜率）
                V = Vt.T
                hedge_ratio = -V[0, 1] / V[0, 0]
            
            else:
                hedge_ratio = self.hedge_ratio
            
            # 应用合理性检查
            if hedge_ratio < MIN_HEDGE_RATIO or hedge_ratio > MAX_HEDGE_RATIO:
                return self.hedge_ratio  # 超出合理范围，使用默认值
            
            return hedge_ratio
            
        except Exception as e:
            # 计算失败，返回默认值
            return self.hedge_ratio
    
    def _should_recalibrate_hedge(self, current_day, last_calibration_day):
        """判断是否需要重新校准对冲比率"""
        return (current_day - last_calibration_day) >= self.hedge_recalibrate_freq
    
    def _calculate_spread_with_dynamic_hedge(self, stock1_price, stock2_price, hedge_ratio):
        """使用动态对冲比率计算价差"""
        return stock1_price - hedge_ratio * stock2_price
    
    def _calculate_indicators(self):
        """
        计算指标
        
        注意：如果使用动态对冲比率，price ratio和z-score会在回测中实时计算
        这里只计算固定对冲比率版本作为参考
        """
        if not self.dynamic_hedge:
            # 固定对冲比率模式：预先计算所有指标
            self.prices['ratio'] = self.prices[self.stock1] / self.prices[self.stock2]
            
            self.prices['z_score'] = AdaptiveZScoreCalculator.calculate(
                self.prices['ratio'],
                lookback=self.lookback,
                method=self.zscore_method
            )
        else:
            # 动态对冲比率模式：在回测中逐步计算
            # 这里先计算初始对冲比率
            if len(self.prices) >= HEDGE_RATIO_LOOKBACK:
                self.hedge_ratio = self._calculate_hedge_ratio(
                    self.prices[self.stock1].iloc[:HEDGE_RATIO_LOOKBACK],
                    self.prices[self.stock2].iloc[:HEDGE_RATIO_LOOKBACK],
                    method=self.hedge_method
                )
            
            # 计算初始价差（使用初始对冲比率）
            self.prices['spread'] = (self.prices[self.stock1] - 
                                    self.hedge_ratio * self.prices[self.stock2])
        
        # 去掉NaN
        self.prices = self.prices.dropna()
    
    def backtest(self):
        """
        执行回测 (包含滚动检查、资金管理、动态对冲比率)
        """
        print(f"\n{'='*60}")
        print(f"实盘级回测: {self.stock1} vs {self.stock2}")
        print(f"模式: 双腿交易 + 滚动协整检查 + {'动态对冲比率' if self.dynamic_hedge else '固定对冲比率'}")
        if self.dynamic_hedge:
            print(f"对冲比率更新频率: 每{self.hedge_recalibrate_freq}天 ({self.hedge_method.upper()}方法)")
        print(f"{'='*60}\n")
        
        capital = self.initial_capital
        max_capital = capital  # 用于计算回撤
        
        # 仓位状态
        position = 0  # 0=空仓, 1=做多价差, -1=做空价差
        
        # 记录入场时的价格和股数
        entry_price_1 = 0
        entry_price_2 = 0
        shares_1 = 0
        shares_2 = 0
        entry_date = None
        entry_z = 0
        entry_hedge_ratio = self.hedge_ratio  # 记录入场时的对冲比率
        
        # 协整状态追踪
        is_relationship_valid = True
        next_check_day = ROLLING_CHECK_WINDOW
        
        # 动态对冲比率追踪
        current_hedge_ratio = self.hedge_ratio
        last_hedge_calibration_day = 0
        
        trades = []
        equity_curve = [capital]
        self.signals = []
        self.hedge_ratio_history = []
        
        # 遍历每一天
        for i in range(len(self.prices)):
            current = self.prices.iloc[i]
            date = self.prices.index[i]
            
            p1 = current[self.stock1]
            p2 = current[self.stock2]
            
            # ===== 1. 🔄 动态对冲比率更新 =====
            if self.dynamic_hedge and i >= HEDGE_RATIO_LOOKBACK:
                # 检查是否需要重新校准
                if self._should_recalibrate_hedge(i, last_hedge_calibration_day):
                    # 使用过去HEDGE_RATIO_LOOKBACK天的数据计算新对冲比率
                    lookback_start = max(0, i - HEDGE_RATIO_LOOKBACK)
                    history_s1 = self.prices[self.stock1].iloc[lookback_start:i]
                    history_s2 = self.prices[self.stock2].iloc[lookback_start:i]
                    
                    old_hedge_ratio = current_hedge_ratio
                    new_hedge_ratio = self._calculate_hedge_ratio(
                        history_s1, history_s2, method=self.hedge_method
                    )
                    
                    # 检查对冲比率变化幅度
                    hedge_ratio_change = abs(new_hedge_ratio - old_hedge_ratio) / old_hedge_ratio
                    
                    if hedge_ratio_change > HEDGE_RATIO_CHANGE_THRESHOLD:
                        # 对冲比率变化超过阈值，记录预警
                        self.signals.append({
                            'date': date,
                            'type': 'hedge_ratio_warning',
                            'old_hedge': old_hedge_ratio,
                            'new_hedge': new_hedge_ratio,
                            'change_pct': hedge_ratio_change * 100,
                            'note': f'对冲比率大幅变化: {old_hedge_ratio:.3f} → {new_hedge_ratio:.3f}'
                        })
                    
                    current_hedge_ratio = new_hedge_ratio
                    last_hedge_calibration_day = i
                    
                    # 记录对冲比率历史
                    self.hedge_ratio_history.append({
                        'date': date,
                        'hedge_ratio': current_hedge_ratio,
                        'method': self.hedge_method
                    })
            
            # 使用当前对冲比率计算价差和z-score
            if self.dynamic_hedge:
                # 动态模式：使用当前对冲比率计算价差
                spread = self._calculate_spread_with_dynamic_hedge(p1, p2, current_hedge_ratio)
                
                # 计算z-score（使用过去lookback天的价差）
                if i >= self.lookback:
                    lookback_start = max(0, i - self.lookback)
                    
                    # 重新计算历史价差（使用当前对冲比率）
                    hist_p1 = self.prices[self.stock1].iloc[lookback_start:i+1]
                    hist_p2 = self.prices[self.stock2].iloc[lookback_start:i+1]
                    hist_spread = hist_p1 - current_hedge_ratio * hist_p2
                    
                    # 计算z-score
                    z_score = AdaptiveZScoreCalculator.calculate(
                        hist_spread,
                        lookback=self.lookback,
                        method=self.zscore_method
                    ).iloc[-1]
                else:
                    z_score = 0  # 数据不足
                
                ratio = p1 / p2  # 仅用于记录
            else:
                # 固定模式：使用预计算的ratio和z-score
                z_score = current.get('z_score', 0)
                ratio = current.get('ratio', p1/p2)
            
            # ===== 2. 🛡️ 滚动协整检查 =====
            if i >= ROLLING_CHECK_WINDOW and i == next_check_day:
                history_s1 = self.prices[self.stock1].iloc[i-ROLLING_CHECK_WINDOW:i]
                history_s2 = self.prices[self.stock2].iloc[i-ROLLING_CHECK_WINDOW:i]
                
                try:
                    _, p_value, _ = coint(history_s1, history_s2)
                    
                    if p_value > ROLLING_P_VALUE_THRESHOLD:
                        is_relationship_valid = False
                        print(f"  ⚠️ {date.date()}: 协整破裂 p={p_value:.4f}")  # ← 加在这里！
                        self.signals.append({
                            'date': date, 'type': 'relationship_broken', 
                            'position': position, 'ratio': ratio, 'z_score': z_score,
                            'note': f'P-val:{p_value:.3f} (失效)'
                        })
                    else:
                        is_relationship_valid = True
                        print(f"  ✅ {date.date()}: 协整有效 p={p_value:.4f}")  
                        if not is_relationship_valid:
                             self.signals.append({
                                'date': date, 'type': 'relationship_restored', 
                                'position': position, 'ratio': ratio, 'z_score': z_score,
                                'note': f'P-val:{p_value:.3f} (恢复)'
                            })
                except:
                    is_relationship_valid = False
                    print(f"  ❌ {date.date()}: 协整检查失败")  # ← 也可以加这行
                
                next_check_day += ROLLING_CHECK_INTERVAL
            
            # ===== 3. 💰 实时市值与回撤计算 =====
            current_equity = capital
            if position != 0:
                floating_pnl = self._calculate_pnl(
                    position, shares_1, shares_2, 
                    entry_price_1, entry_price_2, 
                    p1, p2
                )
                current_equity += floating_pnl

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
                
                # 扣除完整交易成本：开仓 + 平仓
                entry_cost = self._calculate_cost(shares_1, entry_price_1, shares_2, entry_price_2)
                exit_cost = self._calculate_cost(shares_1, p1, shares_2, p2)
                capital -= (entry_cost + exit_cost)
                
                self._record_trade(trades, entry_date, date, position, entry_z, z_score, realized_pnl, "Hard Stop (Drawdown)")
                self.signals.append({'date': date, 'type': 'emergency_exit', 'position': position, 'ratio': ratio, 'z_score': z_score})
                position = 0
                continue 

            # C. Z-score 爆炸保护
            if abs(z_score) > Z_SCORE_EXPLOSION_THRESHOLD and position != 0:
                realized_pnl = self._calculate_pnl(position, shares_1, shares_2, entry_price_1, entry_price_2, p1, p2)
                capital += realized_pnl
                
                # 扣除完整交易成本：开仓 + 平仓
                entry_cost = self._calculate_cost(shares_1, entry_price_1, shares_2, entry_price_2)
                exit_cost = self._calculate_cost(shares_1, p1, shares_2, p2)
                capital -= (entry_cost + exit_cost)
                
                self._record_trade(trades, entry_date, date, position, entry_z, z_score, realized_pnl, "Z-Score Explosion")
                self.signals.append({'date': date, 'type': 'emergency_exit', 'position': position, 'ratio': ratio, 'z_score': z_score})
                position = 0
                continue
            
            # ===== 4. 📈 交易逻辑 =====
            
            # 入场逻辑
            if position == 0:
                # 核心修改：只有关系有效才允许开仓
                if is_relationship_valid and i >= self.lookback:
                    
                    # 做空价差 (价差过高，预期回归)
                    if z_score > self.z_entry and self.allow_short:
                        position = -1
                        entry_date = date
                        entry_price_1 = p1
                        entry_price_2 = p2
                        entry_z = z_score
                        entry_hedge_ratio = current_hedge_ratio  # 记录入场时的对冲比率
                        
                        # 资金分配 (保留5%现金)
                        trade_capital = capital * 0.95
                        
                        if self.dynamic_hedge:
                            # 动态对冲：按对冲比率分配资金
                            # 价差 = S1 - beta*S2，做空价差 = 卖空S1 + 买入beta*S2
                            # 总价值 = p1 + beta*p2
                            total_value = p1 + current_hedge_ratio * p2
                            shares_1 = trade_capital / total_value  # 卖空S1
                            shares_2 = shares_1 * current_hedge_ratio  # 买入beta单位S2
                        else:
                            # 固定比率：50/50分配
                            allocation = trade_capital / 2
                            shares_1 = allocation / p1
                            shares_2 = allocation / p2
                        
                        self.signals.append({
                            'date': date, 'type': 'short_entry', 
                            'position': -1, 'ratio': ratio, 'z_score': z_score,
                            'hedge_ratio': current_hedge_ratio
                        })
                    
                    # 做多价差 (价差过低，预期回升)
                    elif z_score < -self.z_entry:
                        position = 1
                        entry_date = date
                        entry_price_1 = p1
                        entry_price_2 = p2
                        entry_z = z_score
                        entry_hedge_ratio = current_hedge_ratio
                        
                        trade_capital = capital * 0.95
                        
                        if self.dynamic_hedge:
                            # 做多价差 = 买入S1 + 卖空beta*S2
                            total_value = p1 + current_hedge_ratio * p2
                            shares_1 = trade_capital / total_value
                            shares_2 = shares_1 * current_hedge_ratio
                        else:
                            allocation = trade_capital / 2
                            shares_1 = allocation / p1
                            shares_2 = allocation / p2
                        
                        self.signals.append({
                            'date': date, 'type': 'long_entry', 
                            'position': 1, 'ratio': ratio, 'z_score': z_score,
                            'hedge_ratio': current_hedge_ratio
                        })
            
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
                    
                    # 扣除完整交易成本：开仓成本 + 平仓成本（共4条腿）
                    entry_cost = self._calculate_cost(shares_1, entry_price_1, shares_2, entry_price_2)
                    exit_cost = self._calculate_cost(shares_1, p1, shares_2, p2)
                    capital -= (entry_cost + exit_cost)
                    
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
        """
        计算IBKR阶梯式交易成本
        
        成本构成:
        1. 佣金（阶梯式，按股数）
        2. SEC监管费（仅卖出）
        3. FINRA/TAF费用（仅卖出）
        4. 市场冲击（隐性成本）
        
        IBKR美股阶梯式计费:
        - 0-300K股/月: $0.0035/股
        - 300K-3M股/月: $0.0020/股
        - 最低佣金: $0.35/单
        - 最高佣金: 交易额的1%
        """
        # ===== 1. 计算两腿的佣金 =====
        commission_1 = self._calculate_ibkr_commission(s1, p1)
        commission_2 = self._calculate_ibkr_commission(s2, p2)
        total_commission = commission_1 + commission_2
        
        # ===== 2. SEC + FINRA监管费用（仅卖出时收取）=====
        # SEC费用: $27.80 per $1M (0.00278%)
        # FINRA/TAF: $0.166 per $1K (0.0166%)
        val_1 = s1 * p1
        val_2 = s2 * p2
        
        sec_fee_rate = 0.0000278
        finra_taf_rate = 0.000166
        
        # 假设这是平仓操作，两腿都会有一个卖出
        # 实际中应该根据position判断哪一腿是卖出
        regulatory_fees = (val_1 + val_2) * (sec_fee_rate + finra_taf_rate) * 0.5  # 平均50%的腿是卖出
        
        # ===== 3. 滑点（原有逻辑保留）=====
        # 注意：滑点已经包含了市场冲击的影响，不需要重复计算
        slippage = (val_1 + val_2) * SLIPPAGE
        
        # 总成本
        total_cost = total_commission + regulatory_fees + slippage
        
        return total_cost
    
    def _calculate_ibkr_commission(self, shares, price):
        """
        IBKR阶梯式佣金计算（单腿）
        
        阶梯费率（美股）:
        - 0-300,000股/月: $0.0035/股
        - 300,000-3,000,000股/月: $0.0020/股
        - 3,000,000-20,000,000股/月: $0.0015/股
        - 20,000,000-100,000,000股/月: $0.0010/股
        - >100,000,000股/月: $0.0005/股
        
        限制:
        - 最低佣金: $0.35/单
        - 最高佣金: 交易额的1%
        """
        trade_value = shares * price
        
        # 简化处理：假设散户月交易量 < 300K股，使用最高档费率
        # 如果需要精确跟踪月累计量，需要添加状态管理
        commission_rate = 0.0035  # $0.0035/股
        
        # 基础佣金
        commission = shares * commission_rate
        
        # 应用最低佣金限制
        min_commission = 0.35
        commission = max(commission, min_commission)
        
        # 应用最高佣金限制（不超过交易额1%）
        max_commission = trade_value * 0.01
        commission = min(commission, max_commission)
        
        return commission

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
            
            # 准备对冲比率历史数据
            hedge_ratio_df = pd.DataFrame(self.hedge_ratio_history) if self.hedge_ratio_history else pd.DataFrame()
            
            result = {
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
                'Method': self.zscore_method,
                'Dynamic Hedge': self.dynamic_hedge,
                'Hedge Ratio History': hedge_ratio_df
            }
            
            # 如果使用了动态对冲比率，添加统计信息
            if self.dynamic_hedge and len(hedge_ratio_df) > 0:
                result['Hedge Ratio Stats'] = {
                    'Initial': hedge_ratio_df['hedge_ratio'].iloc[0] if len(hedge_ratio_df) > 0 else self.hedge_ratio,
                    'Final': hedge_ratio_df['hedge_ratio'].iloc[-1] if len(hedge_ratio_df) > 0 else self.hedge_ratio,
                    'Mean': hedge_ratio_df['hedge_ratio'].mean(),
                    'Std': hedge_ratio_df['hedge_ratio'].std(),
                    'Min': hedge_ratio_df['hedge_ratio'].min(),
                    'Max': hedge_ratio_df['hedge_ratio'].max(),
                    'Num Updates': len(hedge_ratio_df)
                }
            
            return result
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
            'Dynamic Hedge': self.dynamic_hedge,
            'Hedge Ratio History': pd.DataFrame(),
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

