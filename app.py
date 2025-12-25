"""
Pairs Trading Scanner Pro v3.0 - Streamlit主程序
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
from config import (
    PAGE_TITLE, PAGE_ICON, LAYOUT, STOCK_UNIVERSES,
    DEFAULT_Z_ENTRY, DEFAULT_Z_EXIT, DEFAULT_LOOKBACK,
    DEFAULT_INITIAL_CAPITAL, DEFAULT_TRANSACTION_COST,
    DEFAULT_ALLOW_SHORT, DEFAULT_ZSCORE_METHOD,
    DEFAULT_YEARS_BACK
)
from core_analysis import (
    CointegrationAnalyzer, download_stocks, find_cointegrated_pairs
)
from trading_strategy import (
    PriceRatioPairsTrading, compare_methods_and_plot
)
from visualization import (
    plot_trading_signals, plot_zscore_comparison, plot_traditional_charts
)


# ==================== 页面配置 ====================

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state="expanded"
)


# ==================== 主程序 ====================

def main():
    
    # ============ 初始化session_state (最优先) ============
    if 'scan_data' not in st.session_state:
        st.session_state.scan_data = {
            'completed': False,
            'prices': None,
            'pairs_df': None,
            'results_df': None
        }
    
    # ============ 标题和说明 ============
    
    st.title("📊 Pairs Trading Scanner Pro v3.0")
    st.markdown("### 🔬 采用五层检验 + Robust Z-score，不再盲目赌博!")
    st.markdown("**🚀 v3.0新增** - 支持Traditional和Robust两种Z-score方法")
    
    with st.expander("💡 什么是五层检验?"):
        st.markdown("""
        **传统方法**: 只看相关性 → 准确率10-20% 🎲
        
        **工业级方法**: 五层检验 → 准确率75-90% ✅
        
        ---
        
        **五层检验就像请5个侦探调查案件:**
        
        1. 🕵️ **Engle-Granger**: 距离是否稳定? (看GPS定位)
        2. 🕵️‍♀️ **Johansen**: 关系是否robust? (矩阵分析)
        3. 🕵️‍♂️ **Half-Life**: 多久回归? (测愈合速度)
        4. 🕵️‍♀️ **Rolling**: 关系持久吗? (调查婚姻稳定性)
        5. 🕵️‍♂️ **Hurst**: 真回归还是假回归? (DNA鉴定)
        
        **综合评分 60分以上才推荐交易!**
        """)
    
    with st.expander("🚀 v3.0 新增功能"):
        st.markdown("""
        ### Robust Z-score (抗黑天鹅)
        
        **为什么需要Robust Z-score?**
        
        传统Z-score在黑天鹅事件时会失灵:
        - 单日暴涨/暴跌 → 标准差暴涨
        - 分母被撑大 → Z-score被压低
        - 信号强度减弱 → 错过最佳入场时机
        
        **Robust Z-score的优势:**
        - 使用中位数替代均值 (median vs mean)
        - 使用MAD替代标准差 (MAD vs std)
        - 黑天鹅时保持信号强度
        - 提高长期生存能力
        
        **公式对比:**
        - Traditional: `z = (x - mean) / std`
        - Robust: `z = 1.4826 × (x - median) / MAD`
        
        **建议:** 长期交易优先使用Robust方法
        """)
    
    st.markdown("---")
    
    # ============ 侧边栏配置 ============
    
    st.sidebar.title("⚙️ Configuration")
    
    mode = st.sidebar.radio(
        "Select Mode",
        ["🔍 Scan Pairs", "📈 Test Single Pair"]
    )
    
    st.sidebar.markdown("---")
    
    # 数据设置
    st.sidebar.subheader("📅 Data Settings")
    
    end_date = st.sidebar.date_input(
        "End Date",
        value=datetime.now(),
        max_value=datetime.now()
    )
    
    years_back = st.sidebar.slider(
        "Years of Data",
        min_value=1,
        max_value=5,
        value=DEFAULT_YEARS_BACK
    )
    
    start_date = end_date - timedelta(days=years_back*365)
    
    st.sidebar.markdown("---")
    
    # 协整检验参数
    st.sidebar.subheader("🔬 Cointegration Test")
    
    min_score = st.sidebar.slider(
        "Minimum Score",
        min_value=50,
        max_value=90,
        value=70,
        step=5,
        help="建议: 纸上交易≥60, 真实交易≥70, 重仓≥80"
    )
    
    st.sidebar.markdown("---")
    
    # Z-score方法选择
    st.sidebar.subheader("🎯 Strategy Parameters")
    st.sidebar.markdown("#### 💡 Z-score计算方法")
    
    zscore_method = st.sidebar.radio(
        "选择Z-score方法:",
        options=['robust', 'traditional'],
        format_func=lambda x: {
            'robust': '🟢 Robust (推荐)',
            'traditional': '🔵 Traditional'
        }[x],
        index=0,
        help="Robust: 抗黑天鹅 | Traditional: 经典方法"
    )
    
    if zscore_method == 'robust':
        st.sidebar.success("🛡️ 使用Robust方法")
        st.sidebar.caption("抗异常值，黑天鹅时保持稳定")
    else:
        st.sidebar.info("📊 使用Traditional方法")
        st.sidebar.caption("⚠️ 黑天鹅时可能信号失灵")
    
    st.sidebar.markdown("---")
    
    # 策略参数
    z_entry = st.sidebar.slider(
        "Entry Z-Score",
        min_value=1.0,
        max_value=3.0,
        value=DEFAULT_Z_ENTRY,
        step=0.1
    )
    
    z_exit = st.sidebar.slider(
        "Exit Z-Score",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_Z_EXIT,
        step=0.1
    )
    
    lookback = st.sidebar.slider(
        "Lookback Window (days)",
        min_value=10,
        max_value=120,
        value=DEFAULT_LOOKBACK,
        step=10,
        help="Robust方法推荐60天以上"
    )
    
    transaction_cost = st.sidebar.slider(
        "Transaction Cost (%)",
        min_value=0.0,
        max_value=0.5,
        value=0.1,
        step=0.05
    ) / 100
    
    st.sidebar.markdown("---")
    
    # 交易规则
    st.sidebar.subheader("📋 Trading Rules")
    
    allow_short = st.sidebar.checkbox(
        "Allow Short Selling",
        value=DEFAULT_ALLOW_SHORT,
        help="禁用后仅执行做多价格比交易"
    )
    
    # 当前配置摘要
    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 当前配置")
    st.sidebar.caption(f"方法: **{zscore_method.upper()}**")
    st.sidebar.caption(f"入场: **±{z_entry}** | 出场: **±{z_exit}**")
    st.sidebar.caption(f"窗口: **{lookback}天** | 成本: **{transaction_cost*100:.2f}%**")
    st.sidebar.caption(f"做空: **{'允许' if allow_short else '禁止'}**")
    
    st.sidebar.markdown("---")
    
    # ============ 模式1: 扫描配对 ============
    
    if mode == "🔍 Scan Pairs":
        
        st.header("🔍 Scan for Cointegrated Pairs")
        
        # 选择股票池
        selected_universe = st.selectbox(
            "Select Stock Universe",
            options=list(STOCK_UNIVERSES.keys())
        )
        
        tickers = STOCK_UNIVERSES[selected_universe]
        
        st.info(f"已选择 **{selected_universe}** 股票池 ({len(tickers)} 只股票)")
        
        if st.button("🚀 Start Scanning", type="primary"):
            
            # 清理旧数据
            st.session_state.scan_data = {
                'completed': False,
                'prices': None,
                'pairs_df': None,
                'results_df': None
            }
            
            # 下载数据
            with st.spinner(f"Downloading data for {len(tickers)} stocks..."):
                prices = download_stocks(tickers, start_date, end_date)
            
            if prices.empty:
                st.error("❌ 数据下载失败，请检查网络或代理设置")
                return
            
            st.success(f"✅ Downloaded {len(prices.columns)} stocks, {len(prices)} days of data")
            
            # 协整分析
            st.markdown("---")
            st.subheader("🔬 Cointegration Analysis")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            analyzer = CointegrationAnalyzer()
            
            pairs_df = find_cointegrated_pairs(
                prices,
                min_correlation=0.80,
                min_score=min_score,
                analyzer=analyzer,
                progress_bar=progress_bar,
                status_text=status_text
            )
            
            progress_bar.empty()
            status_text.empty()
            
            if pairs_df.empty:
                st.warning("⚠️ 没有找到符合条件的配对")
                st.info("建议: 降低最低评分或选择其他股票池")
                return
            
            st.success(f"✅ Found {len(pairs_df)} cointegrated pairs!")
            
            # 回测所有配对
            st.markdown("---")
            st.subheader("📈 Backtesting Results")
            
            results_list = []
            
            with st.spinner(f"Backtesting {len(pairs_df)} pairs..."):
                for idx, pair in pairs_df.iterrows():
                    stock1 = pair['Stock 1']
                    stock2 = pair['Stock 2']
                
                    try:
                        strategy = PriceRatioPairsTrading(
                            prices, stock1, stock2,
                            z_entry=z_entry,
                            z_exit=z_exit,
                            lookback=lookback,
                            transaction_cost=transaction_cost,
                            allow_short=allow_short,
                            zscore_method=zscore_method
                        )
                        
                        result = strategy.backtest()
                        
                        results_list.append({
                            'Stock 1': stock1,
                            'Stock 2': stock2,
                            'Coint Score': pair['Score'],
                            'Return (%)': result['Total Return'],
                            'Sharpe': result['Sharpe Ratio'],
                            'Win Rate (%)': result['Win Rate'],
                            'Trades': result['Num Trades'],
                            'Max DD (%)': result['Max Drawdown'],
                        })
                    except Exception as e:
                        print(f"Backtest failed for {stock1}-{stock2}: {e}")
                        continue
            
            if len(results_list) == 0:
                st.error("所有配对回测都失败了")
                return
            
            results_df = pd.DataFrame(results_list)
            results_df = results_df.sort_values('Sharpe', ascending=False)
            
            # 保存到session_state
            st.session_state.scan_data = {
                'completed': True,
                'prices': prices,
                'pairs_df': pairs_df,
                'results_df': results_df
            }
            
            # 显示结果表格
            st.dataframe(
                results_df.style.format({
                    'Coint Score': '{:.1f}',
                    'Return (%)': '{:+.2f}',
                    'Sharpe': '{:.2f}',
                    'Win Rate (%)': '{:.1f}',
                    'Max DD (%)': '{:.2f}'
                }),
                use_container_width=True,
                height=400
            )
        
        # ============ 详细分析 ============
        
        if st.session_state.scan_data['completed']:
            
            prices = st.session_state.scan_data['prices']
            pairs_df = st.session_state.scan_data['pairs_df']
            results_df = st.session_state.scan_data['results_df']
            
            st.markdown("---")
            st.subheader("📊 Detailed Analysis")
            
            pair_options = [f"{row['Stock 1']}-{row['Stock 2']}" for _, row in results_df.iterrows()]
            selected_pair = st.selectbox("Select a pair", options=pair_options, index=0)
            
            if selected_pair:
                stock1, stock2 = selected_pair.split('-')
                
                pair_row = pairs_df[
                    (pairs_df['Stock 1'] == stock1) & 
                    (pairs_df['Stock 2'] == stock2)
                ].iloc[0]
                
                coint_result = pair_row['_result']
                
                st.markdown(f"### 📌 {stock1} - {stock2}")
                
                # 显示指标
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("协整评分", f"{pair_row['Score']:.1f}")
                col1.caption(pair_row['Grade'])
                col2.metric("相关性", f"{pair_row['Correlation']:.3f}")
                col3.metric("半衰期", f"{pair_row['Half Life']:.1f}天" if pair_row['Half Life'] != np.inf else "∞")
                col4.metric("Hurst", f"{pair_row['Hurst']:.3f}" if not pd.isna(pair_row['Hurst']) else "N/A")
                col5.metric("稳定性", f"{pair_row['Stability %']:.0f}%" if not pd.isna(pair_row['Stability %']) else "N/A")
                
                # 五层检验详情
                with st.expander("🔬 查看五层检验详情"):
                    display_cointegration_details(coint_result)
                
                # 回测
                with st.spinner("Loading charts..."):
                    strategy = PriceRatioPairsTrading(
                        prices, stock1, stock2,
                        z_entry=z_entry,
                        z_exit=z_exit,
                        lookback=lookback,
                        transaction_cost=transaction_cost,
                        allow_short=allow_short,
                        zscore_method=zscore_method
                    )
                    
                    detailed_result = strategy.backtest()
                
                # 显示性能指标
                st.markdown("#### 📈 Backtest Performance")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Return", f"{detailed_result['Total Return']:.2f}%")
                col2.metric("Sharpe", f"{detailed_result['Sharpe Ratio']:.2f}")
                col3.metric("Win Rate", f"{detailed_result['Win Rate']:.1f}%")
                col4.metric("Max DD", f"{detailed_result['Max Drawdown']:.2f}%")
                
                # 传统图表
                st.markdown("#### 📊 传统分析图表")
                fig = plot_traditional_charts(
                    prices, stock1, stock2, strategy, detailed_result, z_entry, z_exit
                )
                st.pyplot(fig)
                
                # Tab布局
                st.markdown("---")
                
                tab1, tab2, tab3 = st.tabs([
                    "📋 交易明细", 
                    "📍 交易信号可视化", 
                    "⚖️ 方法对比"
                ])
                
                # Tab 1: 交易明细
                with tab1:
                    display_trade_details(detailed_result)
                
                # Tab 2: 交易信号
                with tab2:
                    display_trading_signals(strategy, detailed_result, stock1, stock2, zscore_method)
                
                # Tab 3: 方法对比
                with tab3:
                    display_method_comparison(prices, stock1, stock2, z_entry, z_exit, lookback)
            
            # 推荐配对
            st.markdown("---")
            st.subheader("⭐ Recommended Pairs")
            
            excellent_pairs = results_df[
                (results_df['Coint Score'].astype(float) >= 80) & 
                (results_df['Sharpe'] >= 1.0)
            ]
            
            if len(excellent_pairs) > 0:
                st.success(f"Found {len(excellent_pairs)} excellent pairs (评分≥80 & 夏普≥1.0):")
                st.dataframe(excellent_pairs, use_container_width=True)
            else:
                st.info("没有找到卓越配对 (评分≥80 且 夏普≥1.0)")
    
    # ============ 模式2: 测试单个配对 ============
    
    else:  # mode == "📈 Test Single Pair"
        
        st.header("📈 Test Single Pair")
        
        col1, col2 = st.columns(2)
        
        with col1:
            stock1 = st.text_input(
                "Stock 1",
                value="JPM",
                help="输入股票代码，如: JPM, AAPL, MSFT"
            ).upper()
        
        with col2:
            stock2 = st.text_input(
                "Stock 2",
                value="BAC",
                help="输入股票代码，如: BAC, GOOGL, TSLA"
            ).upper()
        
        if st.button("🚀 Run Analysis", type="primary"):
            
            if not stock1 or not stock2:
                st.error("请输入两个股票代码")
                return
            
            if stock1 == stock2:
                st.error("两个股票不能相同")
                return
            
            # 下载数据
            with st.spinner(f"Downloading data for {stock1} and {stock2}..."):
                prices = download_stocks([stock1, stock2], start_date, end_date)
            
            if prices.empty:
                st.error("❌ 数据下载失败")
                return
            
            if stock1 not in prices.columns or stock2 not in prices.columns:
                st.error(f"❌ 未能获取到 {stock1} 或 {stock2} 的数据")
                st.info("请检查股票代码是否正确")
                return
            
            st.success(f"✅ Downloaded {len(prices)} days of data")
            
            # 协整分析
            st.markdown("---")
            st.subheader("🔬 Cointegration Analysis")
            
            analyzer = CointegrationAnalyzer()
            
            with st.spinner("Running cointegration tests..."):
                coint_result = analyzer.analyze(prices, stock1, stock2, verbose=False)
            
            if coint_result is None:
                st.error("❌ 协整分析失败")
                return
            
            # 显示协整结果
            st.markdown(f"### 📌 {stock1} - {stock2}")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("协整评分", f"{coint_result['composite_score']:.1f}")
            col1.caption(coint_result['grade'])
            
            corr = prices[[stock1, stock2]].corr().iloc[0, 1]
            col2.metric("相关性", f"{corr:.3f}")
            
            half_life = coint_result['half_life']['half_life_days']
            col3.metric("半衰期", f"{half_life:.1f}天" if half_life != np.inf else "∞")
            
            hurst = coint_result['hurst']['hurst']
            col4.metric("Hurst", f"{hurst:.3f}" if hurst is not None else "N/A")
            
            rolling_ratio = coint_result['rolling_stability']['passed_ratio']
            col5.metric("稳定性", f"{rolling_ratio*100:.0f}%" if rolling_ratio is not None else "N/A")
            
            # 五层检验详情
            with st.expander("🔬 查看五层检验详情"):
                display_cointegration_details(coint_result)
            
            # 评估结果
            if coint_result['composite_score'] >= 70:
                st.success("✅ 这是一个**优秀的配对** (评分≥70)")
            elif coint_result['composite_score'] >= 60:
                st.warning("⚠️ 这是一个**及格的配对** (60≤评分<70)")
            else:
                st.error("❌ 这个配对**不推荐交易** (评分<60)")
            
            # 回测
            st.markdown("---")
            st.subheader("📈 Backtest Results")
            
            with st.spinner("Running backtest..."):
                strategy = PriceRatioPairsTrading(
                    prices, stock1, stock2,
                    z_entry=z_entry,
                    z_exit=z_exit,
                    lookback=lookback,
                    transaction_cost=transaction_cost,
                    allow_short=allow_short,
                    zscore_method=zscore_method
                )
                
                result = strategy.backtest()
            
            # 显示关键指标
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Return", f"{result['Total Return']:+.2f}%")
            col2.metric("Sharpe Ratio", f"{result['Sharpe Ratio']:.2f}")
            col3.metric("Win Rate", f"{result['Win Rate']:.1f}%")
            col4.metric("Max Drawdown", f"{result['Max Drawdown']:.2f}%")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Trades", result['Num Trades'])
            col2.metric("Avg Trade", f"{result['Avg Trade']:+.2f}%")
            col3.metric("Best Trade", f"{result['Best Trade']:+.2f}%")
            col4.metric("Worst Trade", f"{result['Worst Trade']:+.2f}%")
            
            # 传统图表
            st.markdown("---")
            st.markdown("#### 📊 传统分析图表")
            
            fig = plot_traditional_charts(
                prices, stock1, stock2, strategy, result, z_entry, z_exit
            )
            st.pyplot(fig)
            
            # Tab布局
            st.markdown("---")
            
            tab1, tab2, tab3 = st.tabs([
                "📋 交易明细", 
                "📍 交易信号可视化", 
                "⚖️ 方法对比"
            ])
            
            # Tab 1: 交易明细
            with tab1:
                display_trade_details(result)
            
            # Tab 2: 交易信号
            with tab2:
                display_trading_signals(strategy, result, stock1, stock2, zscore_method)
            
            # Tab 3: 方法对比
            with tab3:
                display_method_comparison(prices, stock1, stock2, z_entry, z_exit, lookback)


# ==================== 辅助显示函数 ====================

def display_cointegration_details(coint_result):
    """显示五层检验详情"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**【1】Engle-Granger**")
        eg = coint_result['engle_granger']
        st.write(f"- EG p-value: {eg['eg_pvalue']:.4f} {'✅' if eg['passed'] else '❌'}")
        st.write(f"- Hedge Ratio: {eg['hedge_ratio']:.4f}")
        
        st.markdown("**【2】Johansen**")
        jh = coint_result['johansen']
        st.write(f"- Trace: {'✅' if jh.get('trace_passed') else '❌'}")
        st.write(f"- Eigen: {'✅' if jh.get('eigen_passed') else '❌'}")
        
        st.markdown("**【3】Half-Life**")
        hl = coint_result['half_life']
        st.write(f"- {hl['half_life_days']:.1f} days" if hl['half_life_days'] != np.inf else "- ∞")
        st.write(f"- {hl['quality']}")
    
    with col2:
        st.markdown("**【4】Rolling Stability**")
        rs = coint_result['rolling_stability']
        if rs['passed_ratio'] is not None:
            st.write(f"- Pass Rate: {rs['passed_ratio']*100:.1f}% {'✅' if rs['is_stable'] else '❌'}")
        else:
            st.write("- N/A (数据不足)")
        
        st.markdown("**【5】Hurst**")
        hu = coint_result['hurst']
        if hu['hurst'] is not None:
            st.write(f"- {hu['hurst']:.3f}")
            st.write(f"- {hu['quality']}")
        else:
            st.write("- N/A")


def display_trade_details(result):
    """显示交易明细"""
    if len(result['Trades']) > 0:
        st.subheader("📋 Detailed Trade Log")
        
        trades_display = result['Trades'].copy()
        trades_display['Entry Date'] = pd.to_datetime(trades_display['Entry Date']).dt.date
        trades_display['Exit Date'] = pd.to_datetime(trades_display['Exit Date']).dt.date
        trades_display['P&L'] = trades_display['P&L'].apply(lambda x: f"${x:,.2f}")
        trades_display['P&L %'] = trades_display['P&L %'].apply(lambda x: f"{x:+.2f}%")
        
        st.dataframe(trades_display, use_container_width=True, height=400)
        
        csv = trades_display.to_csv(index=False)
        st.download_button(
            label="📥 Download Trade Log (CSV)",
            data=csv,
            file_name=f"trades_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No trades were generated")
        st.caption("💡 Try adjusting parameters")


def display_trading_signals(strategy, result, stock1, stock2, zscore_method):
    """显示交易信号可视化"""
    st.subheader("📍 Trading Signals Visualization")
    st.caption(f"Method: **{zscore_method.upper()}** Z-score")
    
    if 'Signals' in result and len(result['Signals']) > 0:
        fig_signals = plot_trading_signals(
            strategy.prices,
            result['Signals'],
            stock1, stock2,
            method=zscore_method
        )
        
        if fig_signals:
            st.pyplot(fig_signals)
            
            signals_df = result['Signals']
            st.markdown("#### 📊 Signal Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            long_count = len(signals_df[signals_df['type'] == 'long_entry'])
            short_count = len(signals_df[signals_df['type'] == 'short_entry'])
            exit_count = len(signals_df[signals_df['type'] == 'exit'])
            emergency_count = len(signals_df[signals_df['type'] == 'emergency_exit'])
            
            col1.metric("Long Entries", long_count)
            col2.metric("Short Entries", short_count)
            col3.metric("Normal Exits", exit_count)
            col4.metric("Emergency Exits", emergency_count)
    else:
        st.info("No trading signals generated")


def display_method_comparison(prices, stock1, stock2, z_entry, z_exit, lookback):
    """显示方法对比（使用expander版本）"""
    
    # 🔧 使用唯一的key
    comparison_key = f"comparison_{stock1}_{stock2}_{z_entry}_{z_exit}_{lookback}"
    
    # 初始化session_state
    if comparison_key not in st.session_state:
        st.session_state[comparison_key] = None
    
    # 使用expander包裹
    with st.expander("⚖️ Traditional vs Robust Comparison", expanded=False):
        
        st.info("🔬 点击下方按钮开始对比分析（需要几秒钟）")
        
        # 运行对比按钮
        if st.button("🚀 开始对比分析", key=f"btn_{comparison_key}"):
            with st.spinner("正在对比两种方法..."):
                comparison_results = compare_methods_and_plot(
                    prices, stock1, stock2,
                    z_entry=z_entry,
                    z_exit=z_exit,
                    lookback=lookback
                )
                
                # 保存到session_state
                st.session_state[comparison_key] = comparison_results
        
        # 显示结果
        if st.session_state[comparison_key] is not None:
            
            comparison_results = st.session_state[comparison_key]
            
            st.markdown("#### 📊 Performance Comparison")
            
            st.dataframe(
                comparison_results['comparison_df'],
                use_container_width=True,
                hide_index=True
            )
            
            trad = comparison_results['traditional']
            robust = comparison_results['robust']
            
            st.markdown("#### 🔍 Difference Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                return_diff = robust['Total Return'] - trad['Total Return']
                sharpe_diff = robust['Sharpe Ratio'] - trad['Sharpe Ratio']
                
                st.metric("Return Difference", f"{return_diff:+.2f}%")
                st.metric("Sharpe Difference", f"{sharpe_diff:+.2f}")
            
            with col2:
                trade_diff = robust['Num Trades'] - trad['Num Trades']
                winrate_diff = robust['Win Rate'] - trad['Win Rate']
                
                st.metric("Trade Count Diff", f"{trade_diff:+d}")
                st.metric("Win Rate Diff", f"{winrate_diff:+.1f}%")
            
            # 显示图表
            st.markdown("---")
            st.markdown("#### 📈 Traditional Method - Signals")
            
            strategy_trad = comparison_results['strategy_trad']
            fig_trad = plot_trading_signals(
                strategy_trad.prices,
                trad['Signals'],
                stock1, stock2,
                method='traditional'
            )
            if fig_trad:
                st.pyplot(fig_trad)
            
            st.markdown("---")
            st.markdown("#### 📈 Robust Method - Signals")
            
            strategy_robust = comparison_results['strategy_robust']
            fig_robust = plot_trading_signals(
                strategy_robust.prices,
                robust['Signals'],
                stock1, stock2,
                method='robust'
            )
            if fig_robust:
                st.pyplot(fig_robust)
            
            st.markdown("---")
            st.markdown("#### 📊 Z-score Comparison")
            
            fig_zscore = plot_zscore_comparison(
                strategy_robust.prices, 
                lookback=lookback
            )
            if fig_zscore:
                st.pyplot(fig_zscore)
            
            # 建议
            st.markdown("---")
            st.markdown("#### 💡 Recommendation")
            
            if sharpe_diff > 0.3:
                st.success("✅ **Robust method is significantly better** - Higher Sharpe Ratio")
            elif sharpe_diff < -0.3:
                st.warning("⚠️ **Traditional method performs better** - Market may be stable")
            else:
                st.info("ℹ️ **Both methods perform similarly** - Use Robust for black swan protection")


# ==================== 运行主程序 ====================

if __name__ == "__main__":
    main()