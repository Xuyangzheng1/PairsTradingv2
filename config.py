"""
配置参数 - 所有常量集中管理
"""

import os

# ==================== 网络配置 ====================
PROXY_URL = 'http://127.0.0.1:8367'
os.environ['HTTP_PROXY'] = PROXY_URL
os.environ['HTTPS_PROXY'] = PROXY_URL

# ==================== 协整检验参数 ====================
DEFAULT_SIGNIFICANCE_LEVEL = 0.05
MIN_OBSERVATIONS = 50
ROLLING_WINDOW = 252
ROLLING_STEP = 30

# ==================== 策略参数 ====================
DEFAULT_Z_ENTRY = 2.0
DEFAULT_Z_EXIT = 0.5
DEFAULT_LOOKBACK = 60
DEFAULT_INITIAL_CAPITAL = 100000
DEFAULT_TRANSACTION_COST = 0.001
DEFAULT_ALLOW_SHORT = True
DEFAULT_ZSCORE_METHOD = 'robust'

# ==================== 风险控制参数 ====================
MAX_SINGLE_PNL_RATIO = 1.0
Z_SCORE_EXPLOSION_THRESHOLD = 10.0
Z_SCORE_STOP_THRESHOLD = 5.0
MAX_HOLDING_DAYS = 90

# 数据质量过滤
MIN_HEDGE_RATIO = 0.01
MAX_HEDGE_RATIO = 100
MIN_HALF_LIFE = 1
MAX_HALF_LIFE = 90

# ==================== 数据下载参数 ====================
DATA_CACHE_TTL = 3600
DEFAULT_YEARS_BACK = 2

# ==================== 评分参数 ====================
SCORE_WEIGHTS = {
    'engle_granger': 25,
    'johansen': 20,
    'half_life': 25,
    'rolling_stability': 20,
    'hurst': 10
}

GRADE_THRESHOLDS = {
    'S': 90,
    'A': 80,
    'B': 70,
    'C': 60,
    'F': 0
}

# ==================== UI配置 ====================
PAGE_TITLE = "Pairs Trading Scanner Pro"
PAGE_ICON = "📊"
LAYOUT = "wide"

# Matplotlib配置
MATPLOTLIB_FONT = 'SimHei'

# 图表尺寸
CHART_FIGSIZE = (14, 10)
SIGNAL_CHART_FIGSIZE = (16, 12)
COMPARISON_CHART_FIGSIZE = (16, 8)

# 颜色主题
COLORS = {
    'price_ratio': '#2E86AB',
    'long_entry': '#06D6A0',
    'short_entry': '#EF476F',
    'exit': '#FFD166',
    'emergency_exit': '#FF006E',
    'z_score': '#8338EC'
}

# ==================== 股票池定义 ====================
STOCK_UNIVERSES = {
    'Major Banks': [
    # ===== 超大型银行 (Mega Banks) =====
    'JPM',      # JPMorgan Chase - 全美最大
    'BAC',      # Bank of America - 第二大
    'WFC',      # Wells Fargo - 第三大
    'C',        # Citigroup - 第四大
    
    # ===== 投资银行 (Investment Banks) =====
    'GS',       # Goldman Sachs - 顶级投行
    'MS',       # Morgan Stanley - 顶级投行
    
    # ===== 大型区域银行 (Large Regional Banks) =====
    'USB',      # U.S. Bancorp - 最大区域银行
    'PNC',      # PNC Financial Services
    'TFC',      # Truist Financial
    'COF',      # Capital One Financial
    'CFG',      # Citizens Financial Group
    'KEY',      # KeyCorp
    'FITB',     # Fifth Third Bancorp
    'HBAN',     # Huntington Bancshares
    'RF',       # Regions Financial
    'MTB',      # M&T Bank
    'ZION',     # Zions Bancorporation
    
    # ===== 🆕 中型区域银行 (Mid-Size Regional) =====
    'CMA',      # Comerica - 德州/加州
    'FHN',      # First Horizon - 田纳西
    'EWBC',     # East West Bancorp - 加州（华人银行）
    'SIVB',     # SVB Financial Group - 硅谷银行（⚠️ 注意：2023年破产，可能无数据）
    'PACW',     # PacWest Bancorp - 加州
    'WAL',      # Western Alliance - 亚利桑那
    'SNV',      # Synovus Financial - 佐治亚
    'CBSH',     # Commerce Bancshares - 密苏里
    'BOKF',     # BOK Financial - 俄克拉荷马
    'ONB',      # Old National Bancorp - 印第安纳
    'UMBF',     # UMB Financial - 密苏里
    'HWC',      # Hancock Whitney - 密西西比
    'BANR',     # Banner Corporation - 华盛顿
    'FNB',      # F.N.B. Corporation - 宾州
    'SBCF',     # Seacoast Banking - 佛州
    
    # ===== 托管银行/经纪商 (Custody Banks / Brokers) =====
    'BK',       # Bank of New York Mellon - 托管巨头
    'STT',      # State Street - 托管巨头
    'SCHW',     # Charles Schwab - 经纪商
    'NTRS',     # 🆕 Northern Trust - 私人银行/托管
    
    # ===== 信用卡银行 (Credit Card Banks) =====
    'AXP',      # American Express
    'DFS',      # Discover Financial
    'SYF',      # Synchrony Financial
    'ALLY',     # Ally Financial - 在线银行
    
    # ===== 🆕 专业银行 (Specialty Banks) =====
    'WBS',      # Webster Financial - 东北部
    'ASB',      # Associated Banc-Corp - 威斯康星
    'FCNCA',    # First Citizens BancShares - 北卡
    'WTFC',     # Wintrust Financial - 伊利诺伊
    'VLY',      # Valley National Bancorp - 新泽西
    'UMBF',     # UMB Financial - 堪萨斯城
],
    'Insurance': [
        'BRK.B', 'PGR', 'ALL', 'TRV', 'CB', 'AIG', 'MET', 'PRU', 'AFL', 
        'HIG', 'L', 'CNA', 'AJG', 'MMC', 'AON'
    ],
    'Mega Tech': [
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'META', 'AMZN', 'NVDA', 'TSLA', 'NFLX'
    ],
    'Semiconductors': [
        'NVDA', 'AMD', 'INTC', 'QCOM', 'AVGO', 'TXN', 'ADI', 'AMAT', 'LRCX', 
        'KLAC', 'MCHP', 'NXPI', 'MU', 'SWKS', 'QRVO', 'ON'
    ],
    'Software': [
        'MSFT', 'ORCL', 'CRM', 'ADBE', 'NOW', 'INTU', 'WDAY', 'TEAM', 'ZM',
        'DDOG', 'SNOW', 'PLTR', 'CRWD', 'PANW', 'FTNT'
    ],
    'Oil & Gas': [
        'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD', 'MPC', 'PSX', 'VLO', 'OXY',
        'HAL', 'DVN', 'FANG', 'HES', 'MRO', 'APA', 'CTRA', 'OVV'
    ],
    'Utilities': [
        'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL', 'ED', 'PEG',
        'ES', 'FE', 'ETR', 'AWK'
    ],
    'Retail': [
        'WMT', 'AMZN', 'COST', 'HD', 'LOW', 'TGT', 'DG', 'DLTR', 'ROST', 'TJX',
        'M', 'KSS', 'JWN', 'BBWI', 'BBY'
    ],
    'Consumer Goods': [
        'PG', 'KO', 'PEP', 'PM', 'MO', 'CL', 'KMB', 'GIS', 'K', 'KHC',
        'MDLZ', 'HSY', 'CAG', 'CPB'
    ],
    'Pharma': [
        'JNJ', 'PFE', 'MRK', 'ABBV', 'LLY', 'BMY', 'AMGN', 'GILD', 
        'BIIB', 'REGN', 'VRTX', 'MRNA'
    ],
    'Medical Devices': [
        'TMO', 'ABT', 'DHR', 'SYK', 'MDT', 'BSX', 'BDX', 'EW', 
        'ISRG', 'DXCM', 'ZBH', 'BAX', 'HOLX'
    ],
    'Aerospace': [
        'BA', 'LMT', 'RTX', 'GD', 'NOC', 'TDG', 'HWM'
    ],
    'Airlines': [
        'AAL', 'UAL', 'DAL', 'LUV', 'JBLU', 'SAVE', 'ALK'
    ],
    'Telecom': [
        'T', 'VZ', 'TMUS', 'CMCSA', 'CHTR'
    ],
}