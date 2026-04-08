import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import requests
from openai import OpenAI
from groq import Groq

# --- 1. 網頁基本設定 ---
st.set_page_config(page_title="MSTR mNAV 監測站", layout="wide")
st.title("📊 DAT.co 財務指標即時監控 - Strategy (MSTR)")
st.markdown("本系統自動抓取最新財報數據，計算 MSTR 過去 90 天的 **mNAV (Modified Net Asset Value)** 走勢。")

# --- 2. 自動化數據採集函數 ---

@st.cache_data(ttl=3600) # 快取 1 小時，避免頻繁請求 API
def get_current_metrics():
    """
    透過 API 自動獲取當前總股數與 BTC 持有量
    """
    # A. 抓取最新總股數 (Shares Outstanding)
    try:
        mstr_ticker = yf.Ticker("MSTR")
        # 獲取最新流通股數 (Yahoo Finance API)
        shares = mstr_ticker.info.get('sharesOutstanding')
        if not shares:
            shares = 210000000 # 防呆備援值
    except:
        print("manual")
        shares = 210000000

    # B. 抓取最新 BTC 持有量 (CoinGecko API)
    btc_holdings = 0
    try:
        # CoinGecko 提供專門給 Public Companies Treasury 的 API
        url = "https://api.coingecko.com/api/v3/companies/public_treasury/bitcoin"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            companies = response.json().get('companies', [])
            for co in companies:
                if "Strategy" in co['name']:
                    btc_holdings = co['total_holdings']
                    break
        if btc_holdings == 0: btc_holdings = 252220 # 防呆備援值
    except:
        print("manual")
        btc_holdings = 252220

    return shares, btc_holdings

@st.cache_data(ttl=3600)
def get_historical_data(shares, btc_amount):
    """
    抓取 90 天歷史股價並計算 mNAV
    """
    # 抓取 MSTR 與 BTC 過去 90 天的日頻率收盤價
    mstr_price = yf.download("MSTR", period="90d", interval="1d")['Close']
    btc_price = yf.download("BTC-USD", period="90d", interval="1d")['Close']
    
    # 處理 yfinance 可能回傳的多層索引 (Multi-index)
    if isinstance(mstr_price, pd.DataFrame): mstr_price = mstr_price.iloc[:, 0]
    if isinstance(btc_price, pd.DataFrame): btc_price = btc_price.iloc[:, 0]

    # 合併數據
    df = pd.DataFrame({
        'MSTR_Price': mstr_price,
        'BTC_Price': btc_price
    }).dropna()

    # 計算 mNAV: (股價 * 總股數) / (BTC價格 * 持有量)
    # 此處假設 90 天內股數與持有量變動不大 (符合方案一假設)
    df['mNAV'] = (df['MSTR_Price'] * shares) / (df['BTC_Price'] * btc_amount)
    
    return df

# --- 3. 執行抓取與視覺化 ---

with st.spinner('正在從 Yahoo Finance 與 CoinGecko 獲取即時數據...'):
    shares, btc_holdings = get_current_metrics()
    data = get_historical_data(shares, btc_holdings)

# 顯示當前關鍵數據 (側邊欄)
st.sidebar.header("📡 即時採集參數")
st.sidebar.metric("BTC 持有數", f"{btc_holdings:,.0f} BTC")
st.sidebar.metric("流通股數", f"{shares:,.0f} 股")
st.sidebar.write("數據源: CoinGecko API, Yahoo Finance")

if not data.empty:
    # --- 計算當前市值與持有價值 ---
    latest_mstr_price = data['MSTR_Price'].iloc[-1]
    latest_btc_price = data['BTC_Price'].iloc[-1]
    latest_mnav = data['mNAV'].iloc[-1]
     
    mstr_market_cap = latest_mstr_price * shares
    btc_holdings_value = latest_btc_price * btc_holdings
    
    # --- 顯示頂部數據看板 ---
    st.subheader("💰 當前估值概況")
    
    # 第一排：單價與溢價指標
    col1, col2, col3 = st.columns(3)
    col1.metric("MSTR 當前股價", f"${latest_mstr_price:,.2f}")
    col2.metric("BTC 當前價格", f"${latest_btc_price:,.2f}")
    col3.metric("當前 mNAV (溢價倍數)", f"{latest_mnav:.2f}x")
    
    st.markdown("<br>", unsafe_allow_html=True) # 增加一點排版間距
    
    # 第二排：整體市值對比
    col4, col5, col6 = st.columns(3)
    # 除以 1e9 將單位轉為 Billion (十億)
    col4.metric("MSTR 當前總市值", f"${mstr_market_cap / 1e9:.2f} B")
    col5.metric("BTC 持有總市值", f"${btc_holdings_value / 1e9:.2f} B")
    
    st.info(f"💡 目前 MSTR 的 mNAV 為 **{latest_mnav:.2f}**。這代表市場對其持有比特幣的估值約有 **{(latest_mnav-1)*100:.1f}%** 的溢價。")

    # --- 繪製圖表 ---
    st.divider()
    st.subheader("📈 近 90 天 mNAV 指標走勢")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['mNAV'], mode='lines', name='mNAV', line=dict(color='#007bff')))
    fig.add_hline(y=1, line_dash="dash", line_color="red", annotation_text="mNAV = 1 (平價)")
    
    fig.update_layout(
        xaxis_title="日期",
        yaxis_title="mNAV (溢價倍數)",
        template="plotly_white",
        hovermode="x unified"
    )
    st.plotly_chart(fig, width="stretch")

# --- 4. AI 加分項 (選填) ---
st.divider()
st.subheader("🤖 AI 投資洞察 (Powered by Llama 3)")

# --- 安全做法 (強烈推薦) ---
# 將 API Key 存在 Streamlit 的 Secrets 中，而不是直接寫在程式碼裡
# 請在本地端建立 .streamlit/secrets.toml 檔案，或在 Streamlit Cloud後台設定
# GROQ_API_KEY = "gsk_你的新金鑰..."
try:
    api_key = st.secrets["GROQ_API_KEY"]
except:
    # 這是為了防呆，如果沒設定好 Secrets，才退回明碼或報錯
    # --- 不安全的做法 (僅限本地測試，千萬別推上 GitHub) ---
    api_key = "" 

# 因為我們已經在後台給了 Key，所以直接顯示按鈕即可，不用再檢查 if api_key
if st.button("生成數據摘要"):
    try:
        # 使用 Groq 客戶端
        client = Groq(api_key=api_key)
        recent_trend = data['mNAV'].tail(7).to_string()
        
        prompt = f"""
        你是一位專業的加密貨幣與美股金融分析師。
        請分析 MicroStrategy (MSTR) 最近 7 天的 mNAV (溢價倍數) 變化趨勢：
        {recent_trend}
        
        請用「繁體中文」回答，並簡短說明目前的市場情緒（例如溢價是在收斂還是擴大？這暗示了市場對 BTC 的什麼預期？）。
        """
        
        with st.spinner("神經網路運算中..."):
            # 注意：Groq 官方不時會更新模型名稱，目前最新開源的為 llama-3.1-70b-versatile
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile", 
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500
            )
            st.success("分析完成！")
            st.write(response.choices[0].message.content)
            
    except Exception as e:
        st.error(f"發生錯誤，請確認 API Key 是否設定正確。錯誤訊息：{e}")
