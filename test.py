# class CointegrationMonitor:
#     """滚动协整检验 - 实时监控关系稳定性"""
    
#     def __init__(self, window=252, test_freq=20):
#         """
#         参数:
#         - window: 检验窗口（默认1年）
#         - test_freq: 每N天检验一次
#         """
#         self.window = window
#         self.test_freq = test_freq
#         self.history = []
        
#     def rolling_cointegration_test(self, prices, stock1, stock2):
#         """
#         滚动协整检验，返回每个时间点的p-value
#         """
#         from statsmodels.tsa.stattools import coint
        
#         results = []
        
#         for i in range(self.window, len(prices), self.test_freq):
#             window_data = prices.iloc[i-self.window:i]
            
#             try:
#                 _, p_value, _ = coint(
#                     window_data[stock1], 
#                     window_data[stock2]
#                 )
                
#                 results.append({
#                     'date': prices.index[i],
#                     'p_value': p_value,
#                     'is_cointegrated': p_value < 0.05,
#                     'window_start': prices.index[i-self.window],
#                     'window_end': prices.index[i]
#                 })
                
#             except Exception as e:
#                 results.append({
#                     'date': prices.index[i],
#                     'p_value': None,
#                     'is_cointegrated': False,
#                     'error': str(e)
#                 })
        
#         return pd.DataFrame(results)
    
#     def detect_regime_change(self, coint_df, lookback=3):
#         """
#         检测协整关系的regime change
        
#         规则：如果连续N次检验都不协整，发出警告
#         """
#         recent = coint_df.tail(lookback)
        
#         if all(~recent['is_cointegrated']):
#             return {
#                 'status': 'BREAKDOWN',
#                 'message': f'⚠️ 协整关系破裂！连续{lookback}次检验失败',
#                 'action': 'CLOSE_ALL_POSITIONS'
#             }
#         elif sum(~recent['is_cointegrated']) >= lookback * 0.5:
#             return {
#                 'status': 'WEAKENING',
#                 'message': '⚠️ 协整关系减弱',
#                 'action': 'REDUCE_POSITION_SIZE'
#             }
#         else:
#             return {
#                 'status': 'HEALTHY',
#                 'message': '✅ 协整关系稳定',
#                 'action': 'CONTINUE'
#             }
    
#     def visualize_stability(self, coint_df):
#         """可视化协整关系的稳定性"""
#         fig, ax = plt.subplots(figsize=(14, 6))
        
#         # 绘制p-value时间序列
#         ax.plot(coint_df['date'], coint_df['p_value'], 
#                 marker='o', linewidth=2, label='P-value')
        
#         # 显著性阈值
#         ax.axhline(y=0.05, color='red', linestyle='--', 
#                    linewidth=2, label='显著性水平 (5%)')
#         ax.axhline(y=0.10, color='orange', linestyle='--', 
#                    linewidth=1.5, alpha=0.7, label='警戒线 (10%)')
        
#         # 标注破裂区域
#         breakdown_periods = coint_df[~coint_df['is_cointegrated']]
#         if len(breakdown_periods) > 0:
#             ax.scatter(breakdown_periods['date'], 
#                       breakdown_periods['p_value'],
#                       color='red', s=200, marker='X', 
#                       label='协整破裂', zorder=5)
        
#         ax.set_title('协整关系稳定性监控', fontsize=14, fontweight='bold')
#         ax.set_xlabel('Date')
#         ax.set_ylabel('P-value (越低越好)')
#         ax.legend()
#         ax.grid(alpha=0.3)
        
#         return fig
    
    
# class AdaptiveParameterOptimizer:
#     """根据Half-Life自动调整策略参数"""
    
#     def optimize_parameters(self, half_life):
#         """
#         根据半衰期优化策略参数
        
#         经验法则：
#         - lookback_window ≈ 2-3倍 half_life
#         - z_entry 根据回归速度调整
#         - 持仓期限 ≈ 1-2倍 half_life
#         """
        
#         # 1. Lookback窗口
#         lookback = int(np.clip(half_life * 2.5, 30, 120))
        
#         # 2. Z-score阈值（回归越快，阈值可以越低）
#         if half_life < 10:
#             z_entry, z_exit = 1.5, 0.3  # 快速回归，积极交易
#         elif half_life < 30:
#             z_entry, z_exit = 2.0, 0.5  # 正常
#         else:
#             z_entry, z_exit = 2.5, 0.8  # 慢速回归，保守交易
        
#         # 3. 最大持仓天数（避免过度持有）
#         max_holding = int(half_life * 3)
        
#         # 4. 信号重新评估频率
#         reeval_freq = max(int(half_life * 0.5), 5)
        
#         return {
#             'lookback': lookback,
#             'z_entry': z_entry,
#             'z_exit': z_exit,
#             'max_holding_days': max_holding,
#             'reeval_frequency': reeval_freq,
#             'half_life': half_life
#         }
    
#     def dynamic_position_sizing(self, z_score, half_life):
#         """
#         根据Z-score和Half-Life动态调整仓位
        
#         原则：
#         - Z-score越极端，仓位越大
#         - Half-life越短，仓位越大（快速回归，风险小）
#         """
        
#         # 基础仓位（归一化到0-1）
#         z_magnitude = abs(z_score)
        
#         # Z-score贡献（2σ时50%，3σ时100%）
#         z_factor = np.clip((z_magnitude - 2) / 2, 0, 1)
        
#         # Half-life贡献（越短越激进）
#         hl_factor = np.clip(1 - (half_life - 10) / 50, 0.3, 1)
        
#         # 综合仓位
#         position_size = z_factor * hl_factor
        
#         return position_size


# """
# IBKR真实成本配对交易策略
# 完全模拟IBKR阶梯式定价和真实持仓
# """

# import os
# import numpy as np
# import pandas as pd
# from sklearn.linear_model import LinearRegression
# PROXY_URL = 'http://127.0.0.1:8367'
# os.environ['HTTP_PROXY'] = PROXY_URL
# os.environ['HTTPS_PROXY'] = PROXY_URL
# """
# IBKR真实成本配对交易策略
# 完全模拟IBKR阶梯式定价和真实持仓
# """
# """
# 修正版IBKR配对交易回测
# 核心修正：正确的资金管理
# """

# import numpy as np
# import pandas as pd


# def calculate_ibkr_cost(shares, price, is_sell=False):
#     """IBKR阶梯式定价"""
#     abs_shares = abs(shares)
#     notional = abs_shares * price
    
#     commission = abs_shares * 0.0035
#     commission = max(0.35, commission)
#     commission = min(commission, notional * 0.01)
    
#     sec_fee = notional * 0.0000278
#     taf_fee = abs_shares * 0.000166 if is_sell else 0
#     reg_fees = max(0.01, sec_fee + taf_fee)
    
#     return commission + reg_fees


# class FixedCapitalPairsTrading:
#     """
#     修正版：每次交易用固定金额
    
#     关键修正：
#     1. 每次交易用固定资金（比如$5000）
#     2. 盈亏不影响下次交易金额
#     3. 资金用完就停止交易
#     """
    
#     def __init__(self, prices, stock1, stock2,
#                  z_entry=2.0, z_exit=0.5, lookback=60,
#                  initial_capital=10000,
#                  capital_per_trade=5000):  # 每次交易用多少钱
        
#         self.prices = prices[[stock1, stock2]].dropna().copy()
#         self.stock1 = stock1
#         self.stock2 = stock2
#         self.z_entry = z_entry
#         self.z_exit = z_exit
#         self.lookback = lookback
#         self.initial_capital = initial_capital
#         self.capital_per_trade = capital_per_trade  # 固定每次交易金额
        
#         self._calculate_indicators()
    
#     def _calculate_indicators(self):
#         """计算Z-score"""
#         ratio = self.prices[self.stock1] / self.prices[self.stock2]
#         self.prices['ratio'] = ratio
#         self.prices['z_score'] = np.nan
        
#         for i in range(self.lookback, len(self.prices)):
#             window = ratio.iloc[i-self.lookback:i]
#             mean = window.mean()
#             std = window.std()
#             if std > 0:
#                 self.prices.iloc[i, self.prices.columns.get_loc('z_score')] = \
#                     (ratio.iloc[i] - mean) / std
        
#         self.prices = self.prices.dropna()
    
#     def backtest(self):
#         """回测"""
#         print(f"\n{'='*70}")
#         print(f"修正版回测: {self.stock1} - {self.stock2}")
#         print(f"初始资金: ${self.initial_capital:,.0f}")
#         print(f"每次交易: ${self.capital_per_trade:,.0f}")
#         print(f"{'='*70}\n")
        
#         cash = self.initial_capital
#         position = None
#         trades = []
        
#         for i in range(len(self.prices)):
#             current = self.prices.iloc[i]
#             date = self.prices.index[i]
#             z = current['z_score']
#             p1 = current[self.stock1]
#             p2 = current[self.stock2]
            
#             # === 入场 ===
#             if position is None and abs(z) > self.z_entry:
                
#                 # 检查是否有足够资金
#                 if cash < self.capital_per_trade * 2:  # 需要双边资金
#                     print(f"资金不足，停止交易 (剩余${cash:.2f})")
#                     break
                
#                 # 固定资金分配
#                 per_leg = self.capital_per_trade
                
#                 if z > self.z_entry:
#                     # 做空价差
#                     s1 = -int(per_leg / p1)
#                     s2 = int(abs(s1) * (p1 / p2))
#                 else:
#                     # 做多价差
#                     s1 = int(per_leg / p1)
#                     s2 = -int(s1 * (p1 / p2))
                
#                 # 实际投入（可能小于预定金额）
#                 actual_invest_s1 = abs(s1) * p1
#                 actual_invest_s2 = abs(s2) * p2
#                 total_invest = actual_invest_s1 + actual_invest_s2
                
#                 # 检查资金是否足够
#                 if total_invest > cash:
#                     print(f"资金不足 (需要${total_invest:.0f}, 剩余${cash:.0f})")
#                     continue
                
#                 # 扣除投入资金
#                 cash -= total_invest
                
#                 # 计算成本
#                 cost = (calculate_ibkr_cost(s1, p1, is_sell=(s1>0)) + 
#                        calculate_ibkr_cost(s2, p2, is_sell=(s2>0)))
#                 cash -= cost
                
#                 position = {
#                     's1': s1, 's2': s2,
#                     'ep1': p1, 'ep2': p2,
#                     'date': date, 'z': z,
#                     'invest': total_invest,
#                     'cost': cost,
#                     'days': 0
#                 }
                
#                 print(f"入场 @ {date.date()}: z={z:.2f}, 投入${total_invest:,.0f}")
            
#             # === 出场 ===
#             elif position is not None:
#                 position['days'] += 1
                
#                 should_exit = (abs(z) < self.z_exit or 
#                               abs(z) > 5.0 or 
#                               position['days'] > 90)
                
#                 if should_exit:
#                     # 计算盈亏
#                     if position['s1'] > 0:
#                         pnl1 = position['s1'] * (p1 - position['ep1'])
#                     else:
#                         pnl1 = abs(position['s1']) * (position['ep1'] - p1)
                    
#                     if position['s2'] > 0:
#                         pnl2 = position['s2'] * (p2 - position['ep2'])
#                     else:
#                         pnl2 = abs(position['s2']) * (position['ep2'] - p2)
                    
#                     gross = pnl1 + pnl2
                    
#                     # 出场成本
#                     exit_cost = (calculate_ibkr_cost(position['s1'], p1, is_sell=(position['s1']>0)) +
#                                 calculate_ibkr_cost(position['s2'], p2, is_sell=(position['s2']>0)))
                    
#                     net = gross - exit_cost
                    
#                     # 返还本金
#                     cash += position['invest']
#                     # 加上盈亏
#                     cash += net
                    
#                     trades.append({
#                         'Entry': position['date'],
#                         'Exit': date,
#                         'Days': position['days'],
#                         'Invest': position['invest'],
#                         'Gross PnL': gross,
#                         'Cost': position['cost'] + exit_cost,
#                         'Net PnL': net,
#                         'Return %': net / position['invest'] * 100
#                     })
                    
#                     print(f"出场 @ {date.date()}: {position['days']}天, 净盈亏${net:,.2f} ({net/position['invest']*100:+.2f}%)")
#                     position = None
        
#         # 统计
#         if len(trades) > 0:
#             df = pd.DataFrame(trades)
#             total_return = (cash - self.initial_capital) / self.initial_capital * 100
            
#             print(f"\n{'='*70}")
#             print(f"回测结果:")
#             print(f"  初始资金: ${self.initial_capital:,.2f}")
#             print(f"  最终资金: ${cash:,.2f}")
#             print(f"  总收益: {total_return:.2f}%")
#             print(f"  交易次数: {len(trades)}")
#             print(f"  胜率: {len(df[df['Net PnL']>0])/len(df)*100:.1f}%")
#             print(f"  平均持仓: {df['Days'].mean():.1f}天")
#             print(f"  平均每笔: ${df['Net PnL'].mean():.2f}")
#             print(f"  总成本: ${df['Cost'].sum():.2f}")
#             print(f"{'='*70}")
            
#             return {
#                 'Total Return': total_return,
#                 'Final Capital': cash,
#                 'Trades': df
#             }
#         else:
#             return {'Total Return': 0}


# # ========== 测试 ==========

# if __name__ == "__main__":
#     import yfinance as yf
#     from datetime import datetime, timedelta
    
#     end = datetime.now()
#     start = end - timedelta(days=730)
    
#     print("下载数据...")
#     data = yf.download(['JPM', 'BAC'], start=start, end=end, auto_adjust=True)
    
#     if isinstance(data.columns, pd.MultiIndex):
#         prices = data['Close']
#     else:
#         prices = data
    
#     # 修正版回测
#     strategy = FixedCapitalPairsTrading(
#         prices, 'JPM', 'BAC',
#         initial_capital=10000,
#         capital_per_trade=5000  # 每次交易固定用$5000
#     )
    
#     result = strategy.backtest()

"""
获取美股主要指数成分股 - 修复版
"""

import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime
import os

PROXY_URL = 'http://127.0.0.1:8367'
os.environ['HTTP_PROXY'] = PROXY_URL
os.environ['HTTPS_PROXY'] = PROXY_URL
def get_sp500_symbols():
    """获取标普500成分股 - 修复版"""
    print("正在获取标普500成分股...")
    
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(response.text, 'lxml')
        
        table = soup.find('table', {'id': 'constituents'})
        
        symbols = []
        rows = table.find_all('tr')[1:]  # 跳过表头
        
        for row in rows:
            cols = row.find_all('td')
            if len(cols) > 0:
                # 第一列是ticker symbol
                symbol = cols[0].text.strip()
                symbol = symbol.replace('\n', '')
                symbols.append(symbol)
        
        print(f"✅ 成功获取 {len(symbols)} 个标普500成分股")
        return sorted(symbols)
    
    except Exception as e:
        print(f"❌ 获取标普500失败: {e}")
        return []


def get_nasdaq100_symbols():
    """获取纳斯达克100 - 真正修复版"""
    print("正在获取纳斯达克100成分股...")
    
    try:
        url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(response.text, 'lxml')
        
        table = soup.find('table', {'id': 'constituents'})
        
        symbols = []
        rows = table.find_all('tr')[1:]
        
        for row in rows:
            cols = row.find_all('td')
            if len(cols) > 0:
                # 第一列才是ticker！
                symbol = cols[0].text.strip()
                symbol = symbol.replace('\n', '')
                symbols.append(symbol)
        
        print(f"✅ 成功获取 {len(symbols)} 个纳斯达克100成分股")
        print("前10个ticker:", symbols[:10])
        
        return sorted(symbols)
    
    except Exception as e:
        print(f"❌ 获取失败: {e}")
        return []


def get_dow30_symbols():
    """获取道琼斯30成分股 - 修复版"""
    print("正在获取道琼斯30成分股...")
    
    try:
        url = 'https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average'
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(response.text, 'lxml')
        
        table = soup.find('table', {'id': 'constituents'})
        
        symbols = []
        rows = table.find_all('tr')[1:]
        
        for row in rows:
            cols = row.find_all('td')
            if len(cols) > 1:
                # 第二列是ticker
                symbol = cols[1].text.strip()
                symbol = symbol.replace('\n', '')
                symbols.append(symbol)
        
        print(f"✅ 成功获取 {len(symbols)} 个道琼斯30成分股")
        return sorted(symbols)
    
    except Exception as e:
        print(f"❌ 获取道琼斯30失败: {e}")
        return []


def verify_symbols(symbols, sample_size=10):
    """验证股票代码"""
    if not symbols:
        return symbols
    
    print(f"\n🔍 抽样验证股票代码（验证 {sample_size} 个）...")
    
    import random
    sample = random.sample(symbols, min(sample_size, len(symbols)))
    
    valid_count = 0
    for symbol in sample:
        try:
            ticker = yf.Ticker(symbol)
            # 尝试获取最近价格
            hist = ticker.history(period='5d')
            if not hist.empty:
                valid_count += 1
                print(f"  ✓ {symbol}")
            else:
                print(f"  ✗ {symbol} (无数据)")
        except Exception as e:
            print(f"  ✗ {symbol} ({str(e)[:30]}...)")
    
    success_rate = valid_count / len(sample) * 100
    print(f"✅ 验证成功率: {success_rate:.1f}% ({valid_count}/{len(sample)})")
    
    return symbols


def save_to_csv(symbols, filename):
    """保存到CSV"""
    if not symbols:
        print(f"⚠️ {filename} 没有数据")
        return
    
    df = pd.DataFrame({'symbol': symbols})
    df.to_csv(filename, index=False)
    print(f"💾 已保存到 {filename} ({len(symbols)} 个)")


def main():
    print("=" * 60)
    print("美股主要指数成分股获取工具 - 修复版")
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()
    
    # 获取各指数
    sp500 = get_sp500_symbols()
    time.sleep(1)
    
    nasdaq100 = get_nasdaq100_symbols()
    time.sleep(1)
    
    dow30 = get_dow30_symbols()
    
    print("\n" + "=" * 60)
    print("获取结果:")
    print("=" * 60)
    print(f"标普500:     {len(sp500)} 个")
    print(f"纳斯达克100: {len(nasdaq100)} 个")
    print(f"道琼斯30:    {len(dow30)} 个")
    
    # 验证
    if sp500:
        sp500 = verify_symbols(sp500, 10)
    
    # 保存
    print("\n" + "=" * 60)
    print("保存文件:")
    print("=" * 60)
    
    save_to_csv(sp500, 'sp500.csv')
    save_to_csv(nasdaq100, 'nasdaq100.csv')
    save_to_csv(dow30, 'dow30.csv')
    
    # 合并
    all_symbols = sorted(list(set(sp500 + nasdaq100 + dow30)))
    save_to_csv(all_symbols, 'all_major_indices.csv')
    
    print("\n" + "=" * 60)
    print("✅ 完成！")
    print("=" * 60)
    
    # 显示前10个作为示例
    if sp500:
        print("\n标普500前10个:")
        print(sp500[:10])


if __name__ == "__main__":
    main()