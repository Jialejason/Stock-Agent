import streamlit as st
import yfinance as yf
import pandas as pd
from ta.trend import EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator

st.set_page_config(page_title="AI 投资点位助手", layout="centered")
st.title("📈 AI 投资诊断助手")
st.caption("输入美股代码，秒级计算均线与支撑阻力")

ticker = st.text_input("美股代码", value="NVDA").strip().upper()

if st.button("开始分析", type="primary", use_container_width=True):
    with st.spinner(f"正在拉取 {ticker} 数据..."):
        df = yf.download(ticker, period="1y", interval="1d", progress=False)
        if df.empty:
            st.error("❌ 未找到股票数据，请检查代码。")
        else:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            close = df['Close']
            high = df['High']
            low = df['Low']
            cur_price = close.iloc[-1]
            
            ema5 = EMAIndicator(close, 5).ema_indicator().iloc[-1]
            ema20 = EMAIndicator(close, 20).ema_indicator().iloc[-1]
            ma60 = SMAIndicator(close, 60).sma_indicator().iloc[-1]
            rsi = RSIIndicator(close, 14).rsi().iloc[-1]
            recent_high = high.iloc[-30:].max()
            recent_low = low.iloc[-30:].min()
            
            st.metric(label=f"{ticker} 当前最新价", value=f"${cur_price:.2f}")
            
            st.subheader("🛡️ 关键支撑与阻力位")
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**第一支撑 (EMA20):** ${ema20:.2f}\n\n**生命线支撑 (MA60):** ${ma60:.2f}\n\n**30日低点支撑:** ${recent_low:.2f}")
            with col2:
                st.warning(f"**短线阻力 (EMA5):** ${ema5:.2f}\n\n**30日高点阻力:** ${recent_high:.2f}")
            
            st.subheader("⚡ 动能状态")
            st.write(f"- **RSI (14):** `{rsi:.2f}` ({'⚠️ 超买' if rsi > 70 else '🟢 超卖' if rsi < 30 else '⚖️ 中性'})")
            st.write(f"- **均线形态:** {'🔥 强势多头' if cur_price > ema5 > ema20 else '❄️ 承压调整'}")
