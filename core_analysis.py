"""
核心分析模块
包含：Z-score计算器 + 协整分析器 + 配对筛选
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
import streamlit as st
import yfinance as yf

from config import (
    DEFAULT_SIGNIFICANCE_LEVEL, MIN_OBSERVATIONS, ROLLING_WINDOW, 
    ROLLING_STEP, DATA_CACHE_TTL, MIN_HEDGE_RATIO, MAX_HEDGE_RATIO,
    MIN_HALF_LIFE, MAX_HALF_LIFE
)


# ==================== Z-score计算器 ====================

class AdaptiveZScoreCalculator:
    """
    自适应Z-score计算器
    支持Traditional和Robust两种方法
    """
    
    @staticmethod
    def calculate_traditional(ratio, lookback=60):
        """传统Z-score: z = (x - mean) / std"""
        ratio_mean = ratio.rolling(window=lookback).mean().shift(1)
        ratio_std = ratio.rolling(window=lookback).std().shift(1)
        z_score = (ratio - ratio_mean) / ratio_std
        return z_score
    
    @staticmethod
    def calculate_robust(ratio, lookback=60):
        """Robust Z-score: z = 1.4826 * (x - median) / MAD"""
        ratio_median = ratio.rolling(window=lookback).median().shift(1)
        
        def calculate_mad(x):
            if len(x) < 2:
                return np.nan
            median_val = np.median(x)
            absolute_deviations = np.abs(x - median_val)
            mad = np.median(absolute_deviations)
            return mad if mad > 1e-10 else np.nan
        
        ratio_mad = ratio.rolling(window=lookback).apply(
            calculate_mad, raw=True
        ).shift(1)
        
        z_score = 1.4826 * (ratio - ratio_median) / ratio_mad
        return z_score
    
    @staticmethod
    def calculate(ratio, lookback=60, method='robust'):
        """统一接口"""
        if method == 'traditional':
            return AdaptiveZScoreCalculator.calculate_traditional(ratio, lookback)
        elif method == 'robust':
            return AdaptiveZScoreCalculator.calculate_robust(ratio, lookback)
        else:
            raise ValueError(f"未知方法: {method}")
    
    @staticmethod
    def compare_methods(ratio, lookback=60):
        """对比两种方法"""
        z_trad = AdaptiveZScoreCalculator.calculate_traditional(ratio, lookback)
        z_robust = AdaptiveZScoreCalculator.calculate_robust(ratio, lookback)
        
        difference = (z_trad - z_robust).dropna()
        valid_idx = z_trad.notna() & z_robust.notna()
        correlation = z_trad[valid_idx].corr(z_robust[valid_idx])
        
        return {
            'traditional': z_trad,
            'robust': z_robust,
            'difference': difference,
            'correlation': correlation,
            'max_difference': difference.abs().max() if len(difference) > 0 else np.nan,
            'mean_difference': difference.abs().mean() if len(difference) > 0 else np.nan
        }


# ==================== 协整分析器 ====================

class CointegrationAnalyzer:
    """
    工业级协整分析器 - 五层检验
    """
    
    def __init__(self, significance_level=DEFAULT_SIGNIFICANCE_LEVEL):
        self.significance = significance_level
        
    def analyze(self, prices, stock1, stock2, verbose=False):
        """综合协整分析"""
        series1 = prices[stock1].dropna()
        series2 = prices[stock2].dropna()
        
        # 对齐数据
        common_index = series1.index.intersection(series2.index)
        series1 = series1[common_index]
        series2 = series2[common_index]
        
        if len(series1) < MIN_OBSERVATIONS:
            return None
        
        result = {
            'stock1': stock1,
            'stock2': stock2,
            'n_observations': len(series1),
        }
        
        try:
            # 五层检验
            result['engle_granger'] = self._engle_granger_test(series1, series2)
            result['johansen'] = self._johansen_test(series1, series2)
            result['half_life'] = self._half_life_analysis(
                series1, series2, result['engle_granger']['hedge_ratio']
            )
            result['rolling_stability'] = self._rolling_cointegration(series1, series2)
            result['hurst'] = self._hurst_exponent(
                series1, series2, result['engle_granger']['hedge_ratio']
            )
            
            # 综合评分
            result['composite_score'] = self._calculate_composite_score(result)
            result['is_cointegrated'] = result['composite_score'] >= 60
            result['grade'] = self._get_grade(result['composite_score'])
            
        except Exception as e:
            if verbose:
                st.warning(f"分析失败: {e}")
            return None
        
        return result
    
    def _engle_granger_test(self, y, x):
        """第1层: Engle-Granger两步法"""
        x_const = add_constant(x)
        model = OLS(y, x_const).fit()
        hedge_ratio = model.params[1]
        intercept = model.params[0]
        
        spread = y - hedge_ratio * x - intercept
        adf_stat, adf_pvalue, _, _, critical_values, _ = adfuller(
            spread, maxlag=int(np.ceil(12 * (len(spread)/100)**(1/4)))
        )
        
        eg_score, eg_pvalue, _ = coint(y, x)
        
        return {
            'hedge_ratio': hedge_ratio,
            'intercept': intercept,
            'adf_pvalue': adf_pvalue,
            'eg_pvalue': eg_pvalue,
            'passed': eg_pvalue < self.significance,
            'spread': spread,
        }
    
    def _johansen_test(self, y, x):
        """第2层: Johansen检验"""
        try:
            data = np.column_stack([y, x])
            result = coint_johansen(data, det_order=0, k_ar_diff=1)
            
            trace_stat = result.lr1[0]
            trace_crit_5 = result.cvt[0, 1]
            eigen_stat = result.lr2[0]
            eigen_crit_5 = result.cvm[0, 1]
            
            return {
                'trace_statistic': trace_stat,
                'trace_critical_5%': trace_crit_5,
                'trace_passed': trace_stat > trace_crit_5,
                'eigen_statistic': eigen_stat,
                'eigen_critical_5%': eigen_crit_5,
                'eigen_passed': eigen_stat > eigen_crit_5,
                'passed': (trace_stat > trace_crit_5) and (eigen_stat > eigen_crit_5),
            }
        except:
            return {'trace_passed': False, 'eigen_passed': False, 'passed': False}
    
    def _half_life_analysis(self, y, x, hedge_ratio):
        """第3层: 半衰期分析"""
        try:
            spread = y - hedge_ratio * x
            spread_lag = spread.shift(1)
            spread_diff = spread.diff()
            
            valid_idx = spread_lag.notna() & spread_diff.notna()
            spread_lag_clean = spread_lag[valid_idx]
            spread_diff_clean = spread_diff[valid_idx]
            
            if len(spread_lag_clean) < 10:
                return {
                    'half_life_days': np.inf,
                    'mean_reversion_speed': 0,
                    'is_mean_reverting': False,
                    'quality': 'Very Poor (数据不足)',
                }
            
            X = add_constant(spread_lag_clean)
            model = OLS(spread_diff_clean, X).fit()
            
            if len(model.params) < 1:
                raise ValueError("回归失败")
            
            param_names = model.params.index.tolist()
            
            if 'const' in param_names and len(model.params) >= 2:
                beta = model.params.iloc[1]
            elif 'const' not in param_names and len(model.params) == 1:
                beta = model.params.iloc[0]
            else:
                return {
                    'half_life_days': np.inf,
                    'mean_reversion_speed': 0,
                    'is_mean_reverting': False,
                    'quality': 'Error',
                }
            
            if not isinstance(beta, (int, float)) or np.isnan(beta) or np.isinf(beta):
                return {
                    'half_life_days': np.inf,
                    'mean_reversion_speed': 0,
                    'is_mean_reverting': False,
                    'quality': 'Error',
                }
            
            if beta >= 0:
                half_life = np.inf
                mean_reversion_speed = 0
            else:
                half_life = -np.log(2) / np.log(1 + beta)
                mean_reversion_speed = -beta
            
            return {
                'half_life_days': half_life,
                'mean_reversion_speed': mean_reversion_speed,
                'is_mean_reverting': beta < 0,
                'quality': self._rate_half_life(half_life),
            }
        
        except Exception as e:
            return {
                'half_life_days': np.inf,
                'mean_reversion_speed': 0,
                'is_mean_reverting': False,
                'quality': 'Error',
            }
    
    def _rate_half_life(self, half_life):
        """评估半衰期质量"""
        if half_life < 5:
            return 'Excellent (极快)'
        elif half_life < 15:
            return 'Good (快速)'
        elif half_life < 30:
            return 'Fair (中速)'
        elif half_life < 60:
            return 'Poor (慢速)'
        else:
            return 'Very Poor (龟速)'
    
    def _rolling_cointegration(self, y, x, window=ROLLING_WINDOW):
        """第4层: 滚动协整检验"""
        if len(y) < window * 1.5:
            return {'passed_ratio': None, 'is_stable': False}
        
        n_windows = len(y) - window + 1
        passed_count = 0
        pvalues = []
        
        for i in range(0, n_windows, ROLLING_STEP):
            y_window = y.iloc[i:i+window]
            x_window = x.iloc[i:i+window]
            
            try:
                _, pvalue, _ = coint(y_window, x_window)
                pvalues.append(pvalue)
                if pvalue < self.significance:
                    passed_count += 1
            except:
                continue
        
        if len(pvalues) == 0:
            return {'passed_ratio': 0, 'is_stable': False}
        
        passed_ratio = passed_count / len(pvalues)
        
        return {
            'n_windows_tested': len(pvalues),
            'passed_ratio': passed_ratio,
            'is_stable': passed_ratio >= 0.70,
        }
    
    def _hurst_exponent(self, y, x, hedge_ratio):
        """第5层: Hurst指数"""
        spread = y - hedge_ratio * x
        spread = spread.dropna().values
        
        if len(spread) < 100:
            return {'hurst': None, 'quality': 'N/A'}
        
        lags = range(2, min(100, len(spread)//2))
        tau = []
        
        for lag in lags:
            spread_diff = spread[lag:] - spread[:-lag]
            tau.append(np.std(spread_diff))
        
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        hurst = poly[0]
        
        if hurst < 0.4:
            quality = 'Excellent (强回归)'
        elif hurst < 0.5:
            quality = 'Good (回归)'
        elif hurst < 0.6:
            quality = 'Poor (接近随机)'
        else:
            quality = 'Very Poor (趋势)'
        
        return {
            'hurst': hurst,
            'quality': quality,
            'is_mean_reverting': hurst < 0.5,
        }
    
    def _calculate_composite_score(self, result):
        """综合评分 (0-100分)"""
        rs = result['rolling_stability']
        if rs['passed_ratio'] is not None:
            if rs['passed_ratio'] < 0.30:
                print(f"    ❌ Rolling Stability = {rs['passed_ratio']*100:.1f}% < 30%")
                print(f"       协整关系不稳定，这是伪协整，拒绝该配对")
                return 0  # 直接给0分，淘汰
            else:
                print(f"    ✅ Rolling Stability = {rs['passed_ratio']*100:.1f}% ≥ 30%")
        score = 0
        
        # 1. Engle-Granger (25分)
        eg = result['engle_granger']
        if eg['passed']:
            if eg['eg_pvalue'] < 0.01:
                score += 25
            elif eg['eg_pvalue'] < 0.05:
                score += 20
            else:
                score += 15
        
        # 2. Johansen (20分)
        jh = result['johansen']
        if jh['passed']:
            score += 20
        elif jh.get('trace_passed') or jh.get('eigen_passed'):
            score += 10
        
        # 3. 半衰期 (25分)
        hl = result['half_life']
        if hl['is_mean_reverting']:
            half_life = hl['half_life_days']
            if not (np.isinf(half_life) or np.isnan(half_life) or half_life < 0):
                if half_life > 180:
                    pass
                elif half_life < 10:
                    score += 25
                elif half_life < 20:
                    score += 20
                elif half_life < 40:
                    score += 15
                elif half_life < 60:
                    score += 10
                else:
                    score += 5
        
        # 4. 滚动稳定性 (20分)
        rs = result['rolling_stability']
        if rs['passed_ratio'] is not None:
            score += rs['passed_ratio'] * 20
        
        # 5. Hurst指数 (10分)
        hu = result['hurst']
        if hu['hurst'] is not None:
            if hu['hurst'] < 0.4:
                score += 10
            elif hu['hurst'] < 0.5:
                score += 7
            elif hu['hurst'] < 0.55:
                score += 3
        
        return min(100, score)
    
    def _get_grade(self, score):
        """评级"""
        if score >= 90:
            return 'S级 (完美)'
        elif score >= 80:
            return 'A级 (优秀)'
        elif score >= 70:
            return 'B级 (良好)'
        elif score >= 60:
            return 'C级 (及格)'
        else:
            return 'F级 (不及格)'


# ==================== 数据下载 ====================

@st.cache_data(ttl=DATA_CACHE_TTL)
def download_stocks(tickers, start_date, end_date):
    """下载股票数据（带缓存）"""
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    try:
        data = yf.download(
            tickers, 
            start=start_date, 
            end=end_date,
            progress=False,
            group_by='ticker'
        )
        
        prices = pd.DataFrame()
        
        if len(tickers) == 1:
            if 'Adj Close' in data.columns:
                prices[tickers[0]] = data['Adj Close']
            else:
                prices[tickers[0]] = data['Close']
        else:
            for ticker in tickers:
                try:
                    if ticker in data.columns.get_level_values(0):
                        if 'Adj Close' in data[ticker].columns:
                            prices[ticker] = data[ticker]['Adj Close']
                        else:
                            prices[ticker] = data[ticker]['Close']
                except:
                    continue
        
        prices = prices.dropna(axis=1, how='all')
        return prices
        
    except Exception as e:
        st.error(f"Download error: {e}")
        return pd.DataFrame()


# ==================== 配对筛选 ====================

def find_cointegrated_pairs(prices, min_correlation=0.80, min_score=60, 
                           analyzer=None, progress_bar=None, status_text=None):
    """使用五层检验找真正协整的配对"""
    
    if analyzer is None:
        analyzer = CointegrationAnalyzer()
    
    pairs = []
    stocks = prices.columns.tolist()
    corr_matrix = prices.corr()
    
    total_pairs = len(stocks) * (len(stocks) - 1) // 2
    tested = 0
    
    debug_count = {
        'total': 0, 'low_corr': 0, 'short_data': 0,
        'analyze_none': 0, 'analyze_success': 0,
        'low_score': 0, 'final': 0
    }
    
    print(f"\n开始扫描...")
    print(f"股票数: {len(stocks)}")
    print(f"总配对数: {total_pairs}")
    print(f"最低相关性: {min_correlation}")
    print(f"最低评分: {min_score}\n")
    
    for i, stock1 in enumerate(stocks):
        for j, stock2 in enumerate(stocks):
            if i >= j:
                continue
            
            debug_count['total'] += 1
            tested += 1
            
            if progress_bar and status_text:
                progress_bar.progress(tested / total_pairs)
                status_text.text(f"正在检验 {stock1}-{stock2}... ({tested}/{total_pairs})")
            
            correlation = corr_matrix.loc[stock1, stock2]
            if correlation < min_correlation:
                debug_count['low_corr'] += 1
                continue
            
            pair_data = prices[[stock1, stock2]].dropna()
            if len(pair_data) < 200:
                debug_count['short_data'] += 1
                continue
            
            print(f"  测试 {stock1}-{stock2} (相关性={correlation:.3f}, 数据={len(pair_data)}天)...", end='')
            
            result = analyzer.analyze(prices, stock1, stock2, verbose=False)
            
            if result is None:
                debug_count['analyze_none'] += 1
                print(" ❌ analyze返回None")
                continue
            
            debug_count['analyze_success'] += 1
            score = result['composite_score']
            print(f" 评分={score:.1f}", end='')
            
            # 数据质量检查
            hedge_ratio = result['engle_granger']['hedge_ratio']
            half_life = result['half_life']['half_life_days']
            
            if hedge_ratio < MIN_HEDGE_RATIO or hedge_ratio > MAX_HEDGE_RATIO:
                print(f" ❌ 对冲比例异常: {hedge_ratio:.4f}")
                continue
            
            if half_life < MIN_HALF_LIFE or half_life > MAX_HALF_LIFE:
                print(f" ❌ 半衰期异常: {half_life:.1f}天")
                continue
            
            if score >= min_score:
                debug_count['final'] += 1
                print(" ✅ 通过!")
                
                pairs.append({
                    'Stock 1': stock1,
                    'Stock 2': stock2,
                    'Score': result['composite_score'],
                    'Grade': result['grade'],
                    'Correlation': correlation,
                    'EG P-value': result['engle_granger']['eg_pvalue'],
                    'Hedge Ratio': result['engle_granger']['hedge_ratio'],
                    'Half Life': result['half_life']['half_life_days'],
                    'Hurst': result['hurst']['hurst'] if result['hurst']['hurst'] else np.nan,
                    'Stability %': result['rolling_stability']['passed_ratio'] * 100 if result['rolling_stability']['passed_ratio'] else np.nan,
                    '_result': result,
                })
            else:
                debug_count['low_score'] += 1
                print(f" ❌ 评分太低(需要>={min_score})")
    
    print("\n" + "="*60)
    print("📊 筛选漏斗分析:")
    print("="*60)
    print(f"总配对数: {debug_count['total']}")
    print(f"相关性太低: -{debug_count['low_corr']}")
    print(f"数据太短: -{debug_count['short_data']}")
    print(f"分析失败: -{debug_count['analyze_none']}")
    print(f"分析成功: {debug_count['analyze_success']}")
    print(f"评分太低: -{debug_count['low_score']}")
    print(f"最终通过: {debug_count['final']}")
    print("="*60 + "\n")
    
    if len(pairs) == 0:
        return pd.DataFrame()
    
    pairs_df = pd.DataFrame(pairs)
    pairs_df = pairs_df.sort_values('Score', ascending=False)
    
    return pairs_df