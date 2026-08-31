import streamlit as st
import yfinance as yf
import pandas as pd
from ta.trend import EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator

st.set_page_config(page_title="投资小助手", layout="centered")
st.title("📈 投资小助手")
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
            
            close = df['Close'].dropna()
            high = df['High'].dropna()
            low = df['Low'].dropna()
            cur_price = close.iloc[-1]
            total_days = len(close)
            
            ema5 = EMAIndicator(close, min(5, total_days)).ema_indicator().iloc[-1]
            ema20 = EMAIndicator(close, min(20, total_days)).ema_indicator().iloc[-1]
            
            # 兼容上市不足60天的新股
            if total_days >= 60:
                ma60 = SMAIndicator(close, 60).sma_indicator().iloc[-1]
                ma60_str = f"${ma60:.2f}"
            else:
                ma60_str = "上市未满60日"

            rsi = RSIIndicator(close, min(14, total_days)).rsi().iloc[-1]
            recent_high = high.iloc[-min(30, total_days):].max()
            recent_low = low.iloc[-min(30, total_days):].min()
            
            st.metric(label=f"{ticker} 当前最新价", value=f"${cur_price:.2f}")
            
            st.subheader("🛡️ 关键支撑与阻力位")
            col1, col2 = st.columns(2)
            
            # 动态归类支撑与阻力
            supports = []
            resistances = []
            
            if ema5 < cur_price:
                supports.append(f"**短线支撑 (EMA5):** ${ema5:.2f}")
            else:
                resistances.append(f"**短线阻力 (EMA5):** ${ema5:.2f}")
                
            supports.append(f"**核心支撑 (EMA20):** ${ema20:.2f}")
            supports.append(f"**生命线支撑 (MA60):** {ma60_str}")
            supports.append(f"**近期低点支撑:** ${recent_low:.2f}")
            
            resistances.append(f"**近期高点阻力:** ${recent_high:.2f}")
            
            with col1:
                st.info("\n\n".join(supports))
            with col2:
                st.warning("\n\n".join(resistances))
            
            st.subheader("⚡ 动能状态")
            st.write(f"- **RSI (14):** `{rsi:.2f}` ({'⚠️ 超买' if rsi > 70 else '🟢 超卖' if rsi < 30 else '⚖️ 中性'})")
            st.write(f"- **均线形态:** {'🔥 强势多头' if cur_price > ema5 > ema20 else '❄️ 承压调整'}")
