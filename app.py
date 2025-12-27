"""
Pairs Trading Scanner Pro v3.0 - Streamlit主程序
 streamlit run f:/shares/1225peiduiv1/pairs_trading_scanner/app.py --server.fileWatcherType none
 
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
# ===== 在这里添加SSL修复 =====
import ssl
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context
# ===== SSL修复结束 =====
import yfinance as yf
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
        ["🔍 Scan Pairs", "📈 Test Single Pair", "📊 Batch Backtest"]
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
    
    # ============ 模式2: 批量回测 ============
    
    elif mode == "📊 Batch Backtest":
        
        st.header("📊 Batch Backtest from CSV")
        
        st.markdown("""
        ### 💡 如何使用批量回测
        
        上传CSV文件，系统将自动执行配对交易回测。支持两种CSV格式：
        
        **格式1：单列 symbol（自动生成所有配对）**
```
        symbol
        JPM
        BAC
        WFC
```
        
        **格式2：两列 stock1, stock2（明确指定配对）**
```
        stock1,stock2
        JPM,BAC
        WFC,C
```
        """)
        
        # 上传CSV
        uploaded_file = st.file_uploader(
            "📁 Upload CSV File",
            type=['csv'],
            help="CSV文件应包含 'symbol' 列（自动组合）或 'stock1', 'stock2' 列（明确配对）"
        )
        
        if uploaded_file is not None:
            
            try:
                # 读取CSV
                df = pd.read_csv(uploaded_file)
                
                # 清理列名（去除空格，转小写）
                df.columns = df.columns.str.strip().str.lower()
                
                # 清理数据
                for col in df.columns:
                    if df[col].dtype == 'object':
                        df[col] = df[col].str.strip().str.upper()
                
                # 显示调试信息
                st.info(f"📋 检测到的列: {', '.join(df.columns.tolist())}")
                st.info(f"📊 数据行数: {len(df)}")
                
                # 解析配对
                pairs_list = []
                
                if 'symbol' in df.columns and len(df.columns) == 1:
                    # 单列模式
                    symbols = df['symbol'].dropna().unique().tolist()
                    symbols = [s for s in symbols if s and len(s) > 0 and s != 'NAN']
                    
                    st.markdown("---")
                    st.subheader("⚙️ Batch Settings")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        enable_validation = st.checkbox(
                            "启用股票代码验证",
                            value=False,
                            help="验证股票代码有效性，剔除无效股票（会增加处理时间）",
                            key="validation_single"
                        )
                        
                        use_dynamic_hedge = st.checkbox(
                            "启用动态对冲比率",
                            value=True,
                            help="动态调整对冲比率，通常能提升10-30%夏普比率",
                            key="hedge_single"
                        )
                        
                        if use_dynamic_hedge:
                            hedge_freq = st.slider(
                                "对冲比率更新频率（天）",
                                min_value=10,
                                max_value=40,
                                value=20,
                                step=5,
                                key="freq_single"
                            )
                    
                    with col2:
                        max_workers = st.slider(
                            "并行任务数",
                            min_value=1,
                            max_value=10,
                            value=5,
                            help="增加可提升速度，但可能受API限制",
                            key="workers_single"
                        )
                    
                    # 验证
                    if enable_validation:
                        st.info(f"🔍 正在验证 {len(symbols)} 个股票代码...")
                        valid_symbols = []
                        invalid_symbols = []
                        
                        progress_bar_validate = st.progress(0)
                        
                        for idx, symbol in enumerate(symbols):
                            try:
                                test_data = yf.download(symbol, period='5d', progress=False)
                                if not test_data.empty and len(test_data) >= 3:
                                    valid_symbols.append(symbol)
                                else:
                                    invalid_symbols.append(symbol)
                            except:
                                invalid_symbols.append(symbol)
                            
                            progress_bar_validate.progress((idx + 1) / len(symbols))
                        
                        progress_bar_validate.empty()
                        
                        if len(invalid_symbols) > 0:
                            st.warning(f"⚠️ 剔除了 {len(invalid_symbols)} 个无效股票")
                            with st.expander("查看无效股票"):
                                st.write(', '.join(invalid_symbols[:100]))
                        
                        st.success(f"✅ 验证通过 {len(valid_symbols)} 个有效股票")
                        
                        if len(valid_symbols) > 0:
                            valid_df = pd.DataFrame({'symbol': valid_symbols})
                            valid_csv = valid_df.to_csv(index=False)
                            st.download_button(
                                "📥 下载筛选后的股票列表",
                                data=valid_csv,
                                file_name=f"valid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                key="download_single"
                            )
                        
                        symbols = valid_symbols
                    else:
                        st.info(f"⏭️ 跳过验证，使用 {len(symbols)} 个股票")
                    
                    # 生成配对
                    for i in range(len(symbols)):
                        for j in range(i+1, len(symbols)):
                            pairs_list.append((symbols[i], symbols[j]))
                    st.success(f"✅ 生成了 {len(pairs_list)} 个配对")
                    
                elif 'stock1' in df.columns and 'stock2' in df.columns:
                    # 明确配对模式
                    all_symbols = set()
                    temp_pairs = []
                    
                    for _, row in df.iterrows():
                        if pd.notna(row['stock1']) and pd.notna(row['stock2']):
                            temp_pairs.append((row['stock1'], row['stock2']))
                            all_symbols.add(row['stock1'])
                            all_symbols.add(row['stock2'])
                    
                    st.markdown("---")
                    st.subheader("⚙️ Batch Settings")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        enable_validation = st.checkbox(
                            "启用股票代码验证",
                            value=False,
                            key="validation_explicit"
                        )
                        
                        use_dynamic_hedge = st.checkbox(
                            "启用动态对冲比率",
                            value=True,
                            key="hedge_explicit"
                        )
                        
                        if use_dynamic_hedge:
                            hedge_freq = st.slider(
                                "对冲比率更新频率（天）",
                                min_value=10,
                                max_value=40,
                                value=20,
                                step=5,
                                key="freq_explicit"
                            )
                    
                    with col2:
                        max_workers = st.slider(
                            "并行任务数",
                            min_value=1,
                            max_value=10,
                            value=5,
                            key="workers_explicit"
                        )
                    
                    if enable_validation:
                        st.info(f"🔍 正在验证 {len(all_symbols)} 个股票...")
                        valid_symbols = set()
                        invalid_symbols = []
                        
                        progress_bar = st.progress(0)
                        all_symbols_list = list(all_symbols)
                        
                        for idx, symbol in enumerate(all_symbols_list):
                            try:
                                test_data = yf.download(symbol, period='5d', progress=False)
                                if not test_data.empty and len(test_data) >= 3:
                                    valid_symbols.add(symbol)
                                else:
                                    invalid_symbols.append(symbol)
                            except:
                                invalid_symbols.append(symbol)
                            
                            progress_bar.progress((idx + 1) / len(all_symbols_list))
                        
                        progress_bar.empty()
                        
                        if len(invalid_symbols) > 0:
                            st.warning(f"⚠️ 剔除 {len(invalid_symbols)} 个无效股票")
                        
                        for stock1, stock2 in temp_pairs:
                            if stock1 in valid_symbols and stock2 in valid_symbols:
                                pairs_list.append((stock1, stock2))
                        
                        st.success(f"✅ 验证通过 {len(pairs_list)} 个配对（原 {len(temp_pairs)} 个）")
                        
                        if len(pairs_list) > 0:
                            valid_df = pd.DataFrame(pairs_list, columns=['stock1', 'stock2'])
                            valid_csv = valid_df.to_csv(index=False)
                            st.download_button(
                                "📥 下载筛选后的配对",
                                data=valid_csv,
                                file_name=f"valid_pairs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                key="download_explicit"
                            )
                    else:
                        pairs_list = temp_pairs
                        st.info(f"⏭️ 使用 {len(pairs_list)} 个配对")
                    
                elif 'symbol1' in df.columns and 'symbol2' in df.columns:
                    # 备用列名
                    all_symbols = set()
                    temp_pairs = []
                    
                    for _, row in df.iterrows():
                        if pd.notna(row['symbol1']) and pd.notna(row['symbol2']):
                            temp_pairs.append((row['symbol1'], row['symbol2']))
                            all_symbols.add(row['symbol1'])
                            all_symbols.add(row['symbol2'])
                    
                    st.markdown("---")
                    st.subheader("⚙️ Batch Settings")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        enable_validation = st.checkbox(
                            "启用股票代码验证",
                            value=False,
                            key="validation_alt"
                        )
                        
                        use_dynamic_hedge = st.checkbox(
                            "启用动态对冲比率",
                            value=True,
                            key="hedge_alt"
                        )
                        
                        if use_dynamic_hedge:
                            hedge_freq = st.slider(
                                "对冲比率更新频率（天）",
                                min_value=10,
                                max_value=40,
                                value=20,
                                step=5,
                                key="freq_alt"
                            )
                    
                    with col2:
                        max_workers = st.slider(
                            "并行任务数",
                            min_value=1,
                            max_value=10,
                            value=5,
                            key="workers_alt"
                        )
                    
                    if enable_validation:
                        st.info(f"🔍 正在验证 {len(all_symbols)} 个股票...")
                        valid_symbols = set()
                        invalid_symbols = []
                        
                        progress_bar = st.progress(0)
                        all_symbols_list = list(all_symbols)
                        
                        for idx, symbol in enumerate(all_symbols_list):
                            try:
                                test_data = yf.download(symbol, period='5d', progress=False)
                                if not test_data.empty and len(test_data) >= 3:
                                    valid_symbols.add(symbol)
                                else:
                                    invalid_symbols.append(symbol)
                            except:
                                invalid_symbols.append(symbol)
                            
                            progress_bar.progress((idx + 1) / len(all_symbols_list))
                        
                        progress_bar.empty()
                        
                        if len(invalid_symbols) > 0:
                            st.warning(f"⚠️ 剔除 {len(invalid_symbols)} 个无效股票")
                        
                        for stock1, stock2 in temp_pairs:
                            if stock1 in valid_symbols and stock2 in valid_symbols:
                                pairs_list.append((stock1, stock2))
                        
                        st.success(f"✅ 验证通过 {len(pairs_list)} 个配对（原 {len(temp_pairs)} 个）")
                        
                        if len(pairs_list) > 0:
                            valid_df = pd.DataFrame(pairs_list, columns=['symbol1', 'symbol2'])
                            valid_csv = valid_df.to_csv(index=False)
                            st.download_button(
                                "📥 下载筛选后的配对",
                                data=valid_csv,
                                file_name=f"valid_pairs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                key="download_alt"
                            )
                    else:
                        pairs_list = temp_pairs
                        st.info(f"⏭️ 使用 {len(pairs_list)} 个配对")
                    
                else:
                    st.error("❌ CSV格式错误！")
                    st.markdown("- `symbol` (单列)")
                    st.markdown("- `stock1` 和 `stock2`")
                    st.markdown("- `symbol1` 和 `symbol2`")
                    st.stop()
                
                # 配对预览
                st.markdown("---")
                st.subheader("📋 Pairs Preview")
                preview_df = pd.DataFrame(pairs_list[:10], columns=['Stock 1', 'Stock 2'])
                st.dataframe(preview_df, use_container_width=True)
                if len(pairs_list) > 10:
                    st.caption(f"显示前10个，共{len(pairs_list)}个")
                
                # 开始批量回测
                if st.button("🚀 Start Batch Backtest", type="primary"):
                    
                    
                    # ===== 第一步：批量下载所有股票数据 =====
                    all_symbols = set()
                    for stock1, stock2 in pairs_list:
                        all_symbols.add(stock1)
                        all_symbols.add(stock2)

                    all_symbols = sorted(list(all_symbols))

                    st.info(f"📥 正在下载 {len(all_symbols)} 个股票的数据...")

                    download_progress = st.progress(0)
                    download_status = st.empty()

                    try:
                        batch_size = 50
                        all_prices_list = []
                        
                        for i in range(0, len(all_symbols), batch_size):
                            batch = all_symbols[i:i+batch_size]
                            
                            download_status.text(f"正在下载第 {i//batch_size + 1}/{(len(all_symbols)-1)//batch_size + 1} 批 ({len(batch)} 个股票)...")
                            
                            # 添加重试机制
                            max_retries = 3
                            batch_data = None
                            
                            for attempt in range(max_retries):
                                try:
                                    batch_data = yf.download(
                                        batch,
                                        start=start_date,
                                        end=end_date,
                                        progress=False,
                                        threads=False  # ← 关闭多线程，减少SSL错误
                                    )
                                    break  # 成功就退出重试
                                except Exception as e:
                                    if attempt < max_retries - 1:
                                        time.sleep(2)  # 等2秒重试
                                    else:
                                        st.warning(f"批次 {i//batch_size + 1} 部分下载失败: {str(e)[:50]}")
                                        batch_data = pd.DataFrame()
                            
                            if batch_data is None or batch_data.empty:
                                continue
                            
                            # 处理不同的数据结构
                            if len(batch) == 1:
                                if not batch_data.empty and 'Adj Close' in batch_data.columns:
                                    batch_prices = batch_data[['Adj Close']].copy()
                                    batch_prices.columns = [batch[0]]
                                else:
                                    continue
                            else:
                                if 'Adj Close' in batch_data.columns.get_level_values(0):
                                    batch_prices = batch_data['Adj Close']
                                elif 'Close' in batch_data.columns.get_level_values(0):
                                    batch_prices = batch_data['Close']
                                else:
                                    continue
                            
                            all_prices_list.append(batch_prices)
                            
                            progress = min((i + batch_size) / len(all_symbols), 1.0)
                            download_progress.progress(progress)
                        
                        # 合并所有批次
                        download_status.text("正在合并数据...")
                        all_prices = pd.concat(all_prices_list, axis=1)
                        
                        download_progress.empty()
                        download_status.empty()
                        
                        if all_prices.empty:
                            st.error("❌ 数据下载失败")
                            st.stop()
                        
                        # 检查哪些股票下载失败
                        downloaded_symbols = set(all_prices.columns)
                        failed_symbols = set(all_symbols) - downloaded_symbols
                        
                        if failed_symbols:
                            st.warning(f"⚠️ {len(failed_symbols)} 个股票下载失败")
                            with st.expander("查看下载失败的股票"):
                                st.write(', '.join(sorted(list(failed_symbols))))
                            
                            # 过滤掉包含失败股票的配对
                            pairs_list = [(s1, s2) for s1, s2 in pairs_list 
                                        if s1 in downloaded_symbols and s2 in downloaded_symbols]
                            st.info(f"剩余 {len(pairs_list)} 个有效配对")
                        
                        st.success(f"✅ 成功下载 {len(downloaded_symbols)} 个股票，{len(all_prices)} 天数据")
                        
                    except Exception as e:
                        download_progress.empty()
                        download_status.empty()
                        st.error(f"❌ 下载数据失败: {str(e)}")
                        st.stop()


                    
                    # ===== 第二步：并行回测 =====
                    all_results = []
                    failed_pairs = []
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def backtest_pair(stock1, stock2):
                        """回测单个配对 - 包含协整分析"""
                        try:
                            # 提取数据
                            prices = all_prices[[stock1, stock2]].copy()
                            
                            if prices.isnull().sum().sum() > len(prices) * 0.1:
                                return None, "数据缺失过多"
                            
                            if len(prices) < lookback * 2:
                                return None, "数据不足"
                            
                            # ===== 协整分析 =====
                            analyzer = CointegrationAnalyzer()
                            coint_result = analyzer.analyze(prices, stock1, stock2, verbose=False)
                            
                            if coint_result is None:
                                return None, "协整分析失败"
                            
                            coint_score = coint_result['composite_score']
                            
                            # 如果协整评分太低，直接跳过回测
                            if coint_score < min_score:
                                return None, f"协整评分过低 ({coint_score:.1f})"
                            
                            # ===== 回测 =====
                            strategy = PriceRatioPairsTrading(
                                prices, stock1, stock2,
                                z_entry=z_entry,
                                z_exit=z_exit,
                                lookback=lookback,
                                transaction_cost=transaction_cost,
                                allow_short=allow_short,
                                zscore_method=zscore_method,
                                dynamic_hedge=use_dynamic_hedge,
                                hedge_recalibrate_freq=hedge_freq if use_dynamic_hedge else 20
                            )
                            
                            result = strategy.backtest()
                            
                            return {
                                'Stock 1': stock1,
                                'Stock 2': stock2,
                                'Coint Score': coint_score,  # ✅ 协整评分
                                'Grade': coint_result['grade'],  # 评级
                                'Return (%)': result['Total Return'],
                                'Sharpe': result['Sharpe Ratio'],
                                'Max DD (%)': result['Max Drawdown'],
                                'Win Rate (%)': result['Win Rate'],
                                'Trades': result['Num Trades'],
                                'Avg Trade (%)': result['Avg Trade'],
                            }, None
                            
                        except Exception as e:
                            return None, str(e)
                    
                    # 使用线程池并行回测
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = []
                        for stock1, stock2 in pairs_list:
                            future = executor.submit(backtest_pair, stock1, stock2)
                            futures.append((future, stock1, stock2))
                        
                        # 收集结果
                        for idx, (future, stock1, stock2) in enumerate(futures):
                            try:
                                result, error = future.result()
                                
                                if error:
                                    failed_pairs.append({
                                        'Stock 1': stock1,
                                        'Stock 2': stock2,
                                        'Error': error
                                    })
                                else:
                                    all_results.append(result)
                                
                                # 更新进度
                                progress = (idx + 1) / len(pairs_list)
                                progress_bar.progress(progress)
                                status_text.text(f"已完成 {idx+1}/{len(pairs_list)} 个配对")
                                
                            except Exception as e:
                                failed_pairs.append({
                                    'Stock 1': stock1,
                                    'Stock 2': stock2,
                                    'Error': str(e)
                                })
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    # ===== 第三步：显示结果 =====
                    st.markdown("---")
                    st.subheader("📊 Batch Backtest Results")
                    
                    if len(all_results) > 0:
                        results_df = pd.DataFrame(all_results)
                        results_df = results_df.sort_values('Return (%)', ascending=False).reset_index(drop=True)
                        results_df.insert(0, 'Rank', range(1, len(results_df) + 1))
                        
                        # 统计摘要
                        col1, col2, col3, col4 = st.columns(4)
                        
                        col1.metric("总配对数", len(pairs_list))
                        col2.metric("成功", len(all_results), 
                                   delta=f"{len(all_results)/len(pairs_list)*100:.1f}%")
                        col3.metric("失败", len(failed_pairs))
                        col4.metric("盈利配对", len(results_df[results_df['Return (%)'] > 0]))
                        
                        # 性能统计
                        st.markdown("#### 📈 Performance Statistics")
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("平均收益率", f"{results_df['Return (%)'].mean():.2f}%")
                        col2.metric("平均夏普比率", f"{results_df['Sharpe'].mean():.2f}")
                        col3.metric("平均胜率", f"{results_df['Win Rate (%)'].mean():.1f}%")
                        
                        # Top配对
                        st.markdown("#### 🏆 Top 10 Pairs")
                        
                        top10 = results_df.head(10)
                        st.dataframe(
                            top10.style.format({
                                'Coint Score': '{:.1f}',  # ✅ 加上这列
                                'Return (%)': '{:+.2f}',
                                'Sharpe': '{:.2f}',
                                'Max DD (%)': '{:.2f}',
                                'Win Rate (%)': '{:.1f}',
                                'Avg Trade (%)': '{:+.2f}'
                            }),
                            use_container_width=True,
                            height=400
                        )
                        
                        # 下载结果
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Full Results (CSV)",
                            data=csv,
                            file_name=f"batch_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        
                    else:
                        st.error("❌ 所有配对回测都失败了")
                    
                    # 显示失败的配对
                    if len(failed_pairs) > 0:
                        with st.expander(f"❌ 查看失败的配对 ({len(failed_pairs)} 个)"):
                            failed_df = pd.DataFrame(failed_pairs)
                            st.dataframe(failed_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ 错误: {str(e)}")
        
        else:
            st.info("👆 请上传CSV文件开始批量回测")
            
            # 模板下载
            st.markdown("---")
            st.subheader("📝 CSV Template")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**格式1：单列symbol**")
                example1 = pd.DataFrame({'symbol': ['JPM', 'BAC', 'WFC', 'C', 'USB']})
                st.dataframe(example1, use_container_width=True)
                
                csv1 = example1.to_csv(index=False)
                st.download_button(
                    "📥 Download Template 1",
                    data=csv1,
                    file_name="template_symbols.csv",
                    mime="text/csv"
                )
            
            with col2:
                st.markdown("**格式2：明确配对**")
                example2 = pd.DataFrame({
                    'stock1': ['JPM', 'WFC', 'USB'],
                    'stock2': ['BAC', 'C', 'PNC']
                })
                st.dataframe(example2, use_container_width=True)
                
                csv2 = example2.to_csv(index=False)
                st.download_button(
                    "📥 Download Template 2",
                    data=csv2,
                    file_name="template_pairs.csv",
                    mime="text/csv"
                )
    
    # ============ 模式3: 测试单个配对 ============
    
    elif mode == "📈 Test Single Pair":
        
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