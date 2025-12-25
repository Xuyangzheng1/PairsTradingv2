"""
可视化模块
包含所有图表生成函数
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st

from core_analysis import AdaptiveZScoreCalculator
from config import (
    MATPLOTLIB_FONT, CHART_FIGSIZE, SIGNAL_CHART_FIGSIZE,
    COMPARISON_CHART_FIGSIZE, COLORS
)

# 设置中文字体
plt.rcParams['font.sans-serif'] = [MATPLOTLIB_FONT]
plt.rcParams['axes.unicode_minus'] = False


def plot_trading_signals(prices_with_indicators, signals_df, stock1, stock2, 
                        method='robust', figsize=SIGNAL_CHART_FIGSIZE):
    """
    可视化交易信号 - 三合一图表
    
    显示:
    1. 价格比走势 + 买卖点标注
    2. Z-score时间序列 + 阈值线 + 信号点
    3. 两只股票的归一化价格对比
    """
    
    # 检查是否有信号数据
    if signals_df is None or len(signals_df) == 0:
        st.warning("⚠️ 没有交易信号数据，无法生成图表")
        return None
    
    # 创建3个子图
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    fig.suptitle(f'{stock1}/{stock2} - 配对交易信号分析 ({method.upper()} Z-score)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # ===== 图1: 价格比 + 交易信号 =====
    ax1 = axes[0]
    
    # 绘制价格比曲线
    ax1.plot(prices_with_indicators.index, 
             prices_with_indicators['ratio'],
             label='Price Ratio', 
             color=COLORS['price_ratio'], 
             alpha=0.8, 
             linewidth=2)
    
    # 标注买入信号（做多价格比）
    long_entries = signals_df[signals_df['type'] == 'long_entry']
    if len(long_entries) > 0:
        ax1.scatter(long_entries['date'], 
                   long_entries['ratio'],
                   color=COLORS['long_entry'],
                   marker='^', 
                   s=200, 
                   label='买入信号 (做多比率)', 
                   zorder=5,
                   edgecolors='darkgreen',
                   linewidth=1.5)
    
    # 标注卖出信号（做空价格比）
    short_entries = signals_df[signals_df['type'] == 'short_entry']
    if len(short_entries) > 0:
        ax1.scatter(short_entries['date'],
                   short_entries['ratio'],
                   color=COLORS['short_entry'],
                   marker='v', 
                   s=200,
                   label='卖出信号 (做空比率)', 
                   zorder=5,
                   edgecolors='darkred',
                   linewidth=1.5)
    
    # 标注平仓信号
    exits = signals_df[signals_df['type'] == 'exit']
    if len(exits) > 0:
        ax1.scatter(exits['date'],
                   exits['ratio'],
                   color=COLORS['exit'],
                   marker='x', 
                   s=150,
                   label='平仓信号', 
                   zorder=5,
                   linewidth=2.5)
    
    # 标注紧急平仓
    emergency_exits = signals_df[signals_df['type'] == 'emergency_exit']
    if len(emergency_exits) > 0:
        ax1.scatter(emergency_exits['date'],
                   emergency_exits['ratio'],
                   color=COLORS['emergency_exit'],
                   marker='X', 
                   s=250,
                   label='紧急平仓 (异常)', 
                   zorder=6,
                   edgecolors='black',
                   linewidth=2)
    
    ax1.set_title(f'价格比 ({stock1}/{stock2}) 与交易信号', 
                  fontsize=13, fontweight='bold', pad=10)
    ax1.set_ylabel('Price Ratio', fontsize=11, fontweight='bold')
    ax1.legend(loc='best', fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_facecolor('#F8F9FA')
    
    # ===== 图2: Z-score + 阈值线 + 信号点 =====
    ax2 = axes[1]
    
    # 绘制Z-score曲线
    ax2.plot(prices_with_indicators.index, 
             prices_with_indicators['z_score'],
             label='Z-score', 
             color=COLORS['z_score'], 
             alpha=0.8, 
             linewidth=2)
    
    # 绘制阈值线
    ax2.axhline(y=2.0, color=COLORS['short_entry'], linestyle='--', alpha=0.6, 
                linewidth=1.5, label='入场阈值 (±2.0)')
    ax2.axhline(y=-2.0, color=COLORS['short_entry'], linestyle='--', alpha=0.6, linewidth=1.5)
    
    ax2.axhline(y=0.5, color=COLORS['long_entry'], linestyle='--', alpha=0.6, 
                linewidth=1.5, label='出场阈值 (±0.5)')
    ax2.axhline(y=-0.5, color=COLORS['long_entry'], linestyle='--', alpha=0.6, linewidth=1.5)
    
    ax2.axhline(y=0, color='#495057', linestyle='-', alpha=0.4, linewidth=1)
    
    # 填充区域
    ax2.fill_between(prices_with_indicators.index, 
                     2.0, 5.0, 
                     alpha=0.1, color=COLORS['short_entry'], label='超买区域')
    ax2.fill_between(prices_with_indicators.index, 
                     -2.0, -5.0, 
                     alpha=0.1, color=COLORS['long_entry'], label='超卖区域')
    
    # 标注交易信号在Z-score图上
    if len(long_entries) > 0:
        ax2.scatter(long_entries['date'],
                   long_entries['z_score'],
                   color=COLORS['long_entry'], 
                   marker='^', 
                   s=200, 
                   zorder=5,
                   edgecolors='darkgreen',
                   linewidth=1.5)
    
    if len(short_entries) > 0:
        ax2.scatter(short_entries['date'],
                   short_entries['z_score'],
                   color=COLORS['short_entry'], 
                   marker='v', 
                   s=200, 
                   zorder=5,
                   edgecolors='darkred',
                   linewidth=1.5)
    
    if len(exits) > 0:
        ax2.scatter(exits['date'],
                   exits['z_score'],
                   color=COLORS['exit'], 
                   marker='x', 
                   s=150, 
                   zorder=5,
                   linewidth=2.5)
    
    if len(emergency_exits) > 0:
        ax2.scatter(emergency_exits['date'],
                   emergency_exits['z_score'],
                   color=COLORS['emergency_exit'], 
                   marker='X', 
                   s=250, 
                   zorder=6,
                   edgecolors='black',
                   linewidth=2)
    
    ax2.set_title(f'Z-score 时间序列 ({method.upper()} 方法)', 
                  fontsize=13, fontweight='bold', pad=10)
    ax2.set_ylabel('Z-score', fontsize=11, fontweight='bold')
    ax2.legend(loc='best', fontsize=9, framealpha=0.9, ncol=2)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_facecolor('#F8F9FA')
    
    # ===== 图3: 归一化价格对比 =====
    ax3 = axes[2]
    
    # 归一化价格（Base=100）
    norm_stock1 = prices_with_indicators[stock1] / prices_with_indicators[stock1].iloc[0] * 100
    norm_stock2 = prices_with_indicators[stock2] / prices_with_indicators[stock2].iloc[0] * 100
    
    ax3.plot(prices_with_indicators.index, 
             norm_stock1,
             label=f'{stock1}', 
             color=COLORS['price_ratio'], 
             linewidth=2.5,
             alpha=0.9)
    ax3.plot(prices_with_indicators.index, 
             norm_stock2,
             label=f'{stock2}', 
             color='#F77F00', 
             linewidth=2.5,
             alpha=0.9)
    
    # 标注买入/卖出区域（用垂直线）
    for _, signal in long_entries.iterrows():
        ax3.axvline(x=signal['date'], color=COLORS['long_entry'], alpha=0.15, linewidth=1)
    
    for _, signal in short_entries.iterrows():
        ax3.axvline(x=signal['date'], color=COLORS['short_entry'], alpha=0.15, linewidth=1)
    
    ax3.set_title('归一化价格对比 (Base=100)', 
                  fontsize=13, fontweight='bold', pad=10)
    ax3.set_ylabel('Normalized Price', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax3.legend(loc='best', fontsize=10, framealpha=0.9)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_facecolor('#F8F9FA')
    
    # 统一时间轴格式
    for ax in axes:
        ax.tick_params(axis='both', labelsize=9)
    
    plt.tight_layout()
    
    return fig


def plot_zscore_comparison(prices_with_ratio, lookback=60, figsize=COMPARISON_CHART_FIGSIZE):
    """
    对比Traditional和Robust两种Z-score的差异
    """
    
    # 计算两种Z-score
    ratio = prices_with_ratio['ratio']
    
    z_trad = AdaptiveZScoreCalculator.calculate_traditional(ratio, lookback)
    z_robust = AdaptiveZScoreCalculator.calculate_robust(ratio, lookback)
    
    # 创建图表
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    fig.suptitle('Traditional vs Robust Z-score 对比分析', 
                 fontsize=16, fontweight='bold')
    
    # ===== 图1: 两种Z-score叠加 =====
    ax1 = axes[0]
    
    ax1.plot(z_trad.index, z_trad, 
             label='Traditional Z-score', 
             color=COLORS['price_ratio'], 
             alpha=0.7, 
             linewidth=2)
    ax1.plot(z_robust.index, z_robust, 
             label='Robust Z-score', 
             color=COLORS['long_entry'], 
             alpha=0.7, 
             linewidth=2)
    
    # 阈值线
    ax1.axhline(y=2.0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax1.axhline(y=-2.0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)
    
    ax1.set_title('Z-score 叠加对比', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Z-score', fontsize=11)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('#F8F9FA')
    
    # ===== 图2: 差异图 =====
    ax2 = axes[1]
    
    difference = (z_trad - z_robust).dropna()
    
    ax2.plot(difference.index, difference, 
             color=COLORS['z_score'], 
             alpha=0.8, 
             linewidth=1.5,
             label='Traditional - Robust')
    ax2.fill_between(difference.index, 0, difference, 
                     alpha=0.3, color=COLORS['z_score'])
    
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
    
    # 标注极端差异点
    extreme_diff = difference[abs(difference) > 1.0]
    if len(extreme_diff) > 0:
        ax2.scatter(extreme_diff.index, extreme_diff,
                   color=COLORS['short_entry'], s=50, zorder=5,
                   label=f'极端差异 (|diff|>1.0, n={len(extreme_diff)})')
    
    ax2.set_title('Z-score 差异 (Traditional - Robust)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Difference', fontsize=11)
    ax2.set_xlabel('Date', fontsize=11)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor('#F8F9FA')
    
    # 添加统计信息
    mean_diff = difference.abs().mean()
    max_diff = difference.abs().max()
    correlation = z_trad.corr(z_robust)
    
    stats_text = (f'统计信息:\n'
                 f'平均差异: {mean_diff:.3f}\n'
                 f'最大差异: {max_diff:.3f}\n'
                 f'相关系数: {correlation:.3f}')
    
    ax2.text(0.02, 0.98, stats_text,
             transform=ax2.transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=9)
    
    plt.tight_layout()
    
    return fig


def plot_traditional_charts(prices, stock1, stock2, strategy, result, 
                           z_entry, z_exit, figsize=CHART_FIGSIZE):
    """
    传统四宫格图表
    1. 价格走势
    2. Z-score
    3. 权益曲线
    4. P&L分布
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. 价格走势
    ax1 = axes[0, 0]
    pair_prices = prices[[stock1, stock2]].dropna()
    norm_prices = pair_prices / pair_prices.iloc[0] * 100
    ax1.plot(norm_prices.index, norm_prices[stock1], label=stock1, linewidth=2)
    ax1.plot(norm_prices.index, norm_prices[stock2], label=stock2, linewidth=2)
    ax1.set_title(f'{stock1} vs {stock2} - Normalized', fontweight='bold')
    ax1.set_ylabel('Price (Base=100)')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. Z-Score
    ax2 = axes[0, 1]
    z_scores = strategy.prices['z_score']
    ax2.plot(z_scores.index, z_scores, linewidth=1.5, alpha=0.8)
    ax2.axhline(y=z_entry, color='r', linestyle='--', label=f'±{z_entry}σ', linewidth=1.5)
    ax2.axhline(y=-z_entry, color='r', linestyle='--', linewidth=1.5)
    ax2.axhline(y=z_exit, color='g', linestyle='--', label=f'±{z_exit}σ', alpha=0.7, linewidth=1.5)
    ax2.axhline(y=-z_exit, color='g', linestyle='--', alpha=0.7, linewidth=1.5)
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.fill_between(z_scores.index, z_entry, z_entry+1, alpha=0.1, color='red')
    ax2.fill_between(z_scores.index, -z_entry, -z_entry-1, alpha=0.1, color='red')
    ax2.set_title(f'Z-Score ({strategy.zscore_method.upper()})', fontweight='bold')
    ax2.set_ylabel('Z-Score')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3. 权益曲线
    ax3 = axes[1, 0]
    equity = pd.Series(result['Equity Curve'])
    ax3.plot(equity.values, linewidth=2.5, color='green')
    ax3.axhline(y=100000, color='k', linestyle=':', alpha=0.5, label='Initial')
    ax3.set_title(f"Equity Curve (Return={result['Total Return']:.2f}%)", fontweight='bold')
    ax3.set_ylabel('Capital ($)')
    ax3.set_xlabel('Trading Days')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # 4. P&L分布
    ax4 = axes[1, 1]
    if len(result['Trades']) > 0:
        trades_df_temp = result['Trades']
        ax4.hist(trades_df_temp['P&L %'], bins=20, edgecolor='black', alpha=0.7, color='steelblue')
        ax4.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax4.set_title('Trade P&L Distribution', fontweight='bold')
        ax4.set_xlabel('P&L (%)')
        ax4.set_ylabel('Frequency')
        ax4.grid(alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No Trades', ha='center', va='center', fontsize=16)
        ax4.set_title('Trade P&L Distribution', fontweight='bold')
    
    plt.tight_layout()
    
    return fig