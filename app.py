import streamlit as st
import yfinance as yf
import pandas as pd
from ta.trend import EMAIndicator, SMAIndicator, MACD
from ta.momentum import RSIIndicator

st.set_page_config(page_title="投资小助手", layout="centered")
st.title("📈 投资小助手")
st.caption("输入美股代码，秒级诊断量价关系与技术形态")

ticker = st.text_input("美股代码", value="SPCX").strip().upper()

if st.button("开始分析", type="primary", use_container_width=True):
    with st.spinner(f"正在拉取 {ticker} 数据并计算指标..."):
        df = yf.download(ticker, period="1y", interval="1d", progress=False)
        if df.empty:
            st.error("❌ 未找到股票数据，请检查代码。")
        else:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            close = df['Close'].dropna()
            high = df['High'].dropna()
            low = df['Low'].dropna()
            volume = df['Volume'].dropna()
            
            cur_price = close.iloc[-1]
            total_days = len(close)
            
            # 1. 均线计算
            ema5 = EMAIndicator(close, min(5, total_days)).ema_indicator().iloc[-1]
            ema20 = EMAIndicator(close, min(20, total_days)).ema_indicator().iloc[-1]
            if total_days >= 60:
                ma60 = SMAIndicator(close, 60).sma_indicator().iloc[-1]
                ma60_str = f"${ma60:.2f}"
            else:
                ma60_str = "上市未满60日"
            
            # 2. 动能指标 (RSI & MACD)
            rsi = RSIIndicator(close, min(14, total_days)).rsi().iloc[-1]
            macd_obj = MACD(close)
            macd_val = macd_obj.macd().iloc[-1]
            macd_signal = macd_obj.macd_signal().iloc[-1]
            macd_diff = macd_obj.macd_diff().iloc[-1]
            
            # 3. 成交量分析 (计算量比: 今日成交量 / 过去5日平均成交量)
            cur_vol = volume.iloc[-1]
            avg_vol_5d = volume.iloc[-6:-1].mean() if total_days >= 6 else volume.mean()
            vol_ratio = (cur_vol / avg_vol_5d) if avg_vol_5d > 0 else 1.0
            
            # 4. 支撑阻力计算
            recent_high = high.iloc[-min(30, total_days):].max()
            recent_low = low.iloc[-min(30, total_days):].min()
            
            # 页面展示
            col_m1, col_m2 = st.columns(2)
            col_m1.metric(label=f"{ticker} 当前最新价", value=f"${cur_price:.2f}")
            vol_status = "🔥 明显放量" if vol_ratio > 1.5 else "🧊 明显缩量" if vol_ratio < 0.7 else "⚖️ 正常平量"
            col_m2.metric(label="5日量比 (Volume Ratio)", value=f"{vol_ratio:.2f} 倍", delta=vol_status)
            
            st.subheader("🛡️ 关键支撑与阻力位")
            col1, col2 = st.columns(2)
            supports = []
            resistances = []
            
            if ema5 < cur_price:
                supports.append(f"**短线支撑 (EMA5):** ${ema5:.2f}")
            else:
                resistances.append(f"**短线阻力 (EMA5):** ${ema5:.2f}")
                
            supports.append(f"**核心支撑 (EMA20):** ${ema20:.2f}")
            supports.append(f"**生命线支撑 (MA60):** {ma60_str}")
            supports.append(f"**30日低点支撑:** ${recent_low:.2f}")
            resistances.append(f"**30日高点阻力:** ${recent_high:.2f}")
            
            with col1:
                st.info("\n\n".join(supports))
            with col2:
                st.warning("\n\n".join(resistances))
            
            st.subheader("⚡ 动能与量价状态")
            # MACD 判断
            macd_trend = "🟢 多头金叉（动能充沛）" if macd_val > macd_signal and macd_diff > 0 else "🔴 动能转弱/死叉整理"
            st.write(f"- **MACD 状态:** `{macd_trend}` (柱值: {macd_diff:.2f})")
            
            # RSI 判断
            rsi_desc = "⚠️ 超买区（注意回调）" if rsi > 70 else "🟢 超卖区（关注反弹）" if rsi < 30 else "⚖️ 中性区间"
            st.write(f"- **RSI (14):** `{rsi:.2f}` ({rsi_desc})")
            
            # 量价形态判断
            price_change = cur_price - close.iloc[-2] if total_days >= 2 else 0
            if price_change > 0 and vol_ratio > 1.2:
                vol_price_desc = "🔥 放量上涨（资金主动进攻，买盘强劲）"
            elif price_change < 0 and vol_ratio > 1.2:
                vol_price_desc = "⚠️ 放量下跌（抛压偏大，谨慎接刀）"
            elif price_change > 0 and vol_ratio <= 1.2:
                vol_price_desc = "⚖️ 缩量上涨（筹码锁定良好或动能待确认）"
            else:
                vol_price_desc = "🧊 缩量洗盘/弱势整理"
            st.write(f"- **量价形态:** `{vol_price_desc}`")
