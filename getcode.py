import requests
import os
from datetime import datetime

# Clash代理设置（Clash默认本地代理端口8367，支持HTTP和SOCKS5）
proxies = {
    'http': 'http://127.0.0.1:8367',
    'https': 'http://127.0.0.1:8367',
}

# 如果Clash只开了SOCKS5代理，用下面这行替换上面（取消注释）：
# proxies = {'http': 'socks5h://127.0.0.1:8367', 'https': 'socks5h://127.0.0.1:8367'}

# 创建保存文件夹
save_dir = "etf_holdings"
os.makedirs(save_dir, exist_ok=True)

# 当前日期，用于文件名
today = datetime.now().strftime('%Y-%m-%d')

# ETF下载配置
etf_list = [
    {'ticker': 'XLK', 'name': '科技',         'url': 'https://www.ssga.com/us/en/intermediary/library-content/products/fund-data/etfs/us/holdings-daily-us-en-xlk.xlsx'},
    {'ticker': 'XLV', 'name': '医疗保健',     'url': 'https://www.ssga.com/us/en/intermediary/library-content/products/fund-data/etfs/us/holdings-daily-us-en-xlv.xlsx'},
    {'ticker': 'XLE', 'name': '能源',         'url': 'https://www.ssga.com/us/en/intermediary/library-content/products/fund-data/etfs/us/holdings-daily-us-en-xle.xlsx'},
    {'ticker': 'XLF', 'name': '金融',         'url': 'https://www.ssga.com/us/en/intermediary/library-content/products/fund-data/etfs/us/holdings-daily-us-en-xlf.xlsx'},
    {'ticker': 'XLY', 'name': '消费可选',     'url': 'https://www.ssga.com/us/en/intermediary/library-content/products/fund-data/etfs/us/holdings-daily-us-en-xly.xlsx'},
    {'ticker': 'XLP', 'name': '消费必需',     'url': 'https://www.ssga.com/us/en/intermediary/library-content/products/fund-data/etfs/us/holdings-daily-us-en-xlp.xlsx'},
    {'ticker': 'XLC', 'name': '通信服务',     'url': 'https://www.ssga.com/us/en/intermediary/library-content/products/fund-data/etfs/us/holdings-daily-us-en-xlc.xlsx'},
    {'ticker': 'XLI', 'name': '工业',         'url': 'https://www.ssga.com/us/en/intermediary/library-content/products/fund-data/etfs/us/holdings-daily-us-en-xli.xlsx'},
    {'ticker': 'XLB', 'name': '材料',         'url': 'https://www.ssga.com/us/en/intermediary/library-content/products/fund-data/etfs/us/holdings-daily-us-en-xlb.xlsx'},
    {'ticker': 'XLU', 'name': '公用事业',     'url': 'https://www.ssga.com/us/en/intermediary/library-content/products/fund-data/etfs/us/holdings-daily-us-en-xlu.xlsx'},
    {'ticker': 'XLRE','name': '房地产',       'url': 'https://www.ssga.com/us/en/intermediary/library-content/products/fund-data/etfs/us/holdings-daily-us-en-xlre.xlsx'},
    {'ticker': 'SOXX','name': 'iShares半导体','url': 'https://www.ishares.com/us/products/239705/ishares-phlx-semiconductor-etf/1467271812596.ajax?fileType=csv&fileName=SOXX_holdings&dataType=fund'},
]

# 开始下载
print("开始下载ETF持股文件（通过Clash代理）...\n")
for etf in etf_list:
    ticker = etf['ticker']
    name = etf['name']
    url = etf['url']
    
    try:
        print(f"正在下载 {ticker} ({name})...")
        response = requests.get(url, proxies=proxies, timeout=30)
        response.raise_for_status()
        
        # 确定文件扩展名
        ext = '.xlsx' if 'ssga.com' in url else '.csv'
        filename = f"{ticker}_holdings_{today}{ext}"
        filepath = os.path.join(save_dir, filename)
        
        with open(filepath, 'wb') as f:
            f.write(response.content)
        
        print(f"✓ 保存成功: {filepath}\n")
        
    except Exception as e:
        print(f"✗ 下载失败 {ticker}: {str(e)}\n")

print("所有下载任务完成！文件保存在 ./etf_holdings 文件夹中。")