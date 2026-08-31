import streamlit as st
import yfinance as yf
import pandas as pd
from ta.trend import EMAIndicator, SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

st.set_page_config(page_title="投资小助手", layout="centered")
st.title("📈 投资小助手")
st.caption("输入美股代码，秒级诊断量价关系并生成战术点位")

ticker = st.text_input("美股代码", value="SPCX").strip().upper()

if st.button("开始分析", type="primary", use_container_width=True):
    with st.spinner(f"正在全维度诊断 {ticker}..."):
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
            
            # 1. 均线
            ema5 = EMAIndicator(close, min(5, total_days)).ema_indicator().iloc[-1]
            ema10 = EMAIndicator(close, min(10, total_days)).ema_indicator().iloc[-1]
            ema20 = EMAIndicator(close, min(20, total_days)).ema_indicator().iloc[-1]
            ma60_str = f"${SMAIndicator(close, 60).sma_indicator().iloc[-1]:.2f}" if total_days >= 60 else "上市未满60日"
            
            # 2. 动能 & 波动率
            rsi = RSIIndicator(close, min(14, total_days)).rsi().iloc[-1]
            macd_obj = MACD(close)
            macd_val = macd_obj.macd().iloc[-1]
            macd_signal = macd_obj.macd_signal().iloc[-1]
            macd_diff = macd_obj.macd_diff().iloc[-1]
            
            atr = AverageTrueRange(high, low, close, min(14, total_days)).average_true_range().iloc[-1]
            
            # 3. 量比
            cur_vol = volume.iloc[-1]
            avg_vol_5d = volume.iloc[-6:-1].mean() if total_days >= 6 else volume.mean()
            vol_ratio = (cur_vol / avg_vol_5d) if avg_vol_5d > 0 else 1.0
            
            # 4. 近期极值
            recent_high = high.iloc[-min(30, total_days):].max()
            recent_low = low.iloc[-min(30, total_days):].min()
            
            # 顶部仪表盘
            col_m1, col_m2 = st.columns(2)
            col_m1.metric(label=f"{ticker} 最新价", value=f"${cur_price:.2f}")
            vol_status = "🔥 放量" if vol_ratio > 1.3 else "🧊 缩量" if vol_ratio < 0.7 else "⚖️ 平量"
            col_m2.metric(label="5日量比", value=f"{vol_ratio:.2f} 倍", delta=vol_status)
            
            # 战术策略卡片
            st.subheader("🎯 实战操作指南 (新手参考)")
            if cur_price > ema20 and macd_diff > 0 and rsi < 65:
                tactic_title = "🟢 趋势良好，适合分批逢低吸纳"
                tactic_color = "success"
            elif cur_price > ema20 and rsi >= 65:
                tactic_title = "⚠️ 处于高位强势区，切勿追高，防范短线震荡"
                tactic_color = "warning"
            else:
                tactic_title = "❄️ 处于调整/震荡区间，控制仓位或观望"
                tactic_color = "info"
            
            rec_entry = f"${ema20:.2f} ~ ${ema10:.2f}"
            rec_stop = f"${max(ema20 - atr, recent_low):.2f}"
            rec_target = f"${recent_high:.2f}"
            
            if tactic_color == "success":
                st.success(f"**策略定性:** {tactic_title}\n\n- **建议关注回踩区间:** `{rec_entry}`\n- **参考防守止损位:** `{rec_stop}`\n- **上方第一止盈目标:** `{rec_target}`")
            elif tactic_color == "warning":
                st.warning(f"**策略定性:** {tactic_title}\n\n- **建议关注回踩区间:** `{rec_entry}`\n- **参考防守止损位:** `{rec_stop}`\n- **上方第一阻力目标:** `{rec_target}`")
            else:
                st.info(f"**策略定性:** {tactic_title}\n\n- **建议观望支撑:** `{ema20:.2f}`\n- **强支撑防线:** `{recent_low:.2f}`")

            # 支撑与阻力
            st.subheader("🛡️ 关键支撑与阻力位")
            col1, col2 = st.columns(2)
            supports = []
            resistances = []
            
            if ema5 < cur_price: supports.append(f"**超短支撑 (EMA5):** ${ema5:.2f}")
            else: resistances.append(f"**短线阻力 (EMA5):** ${ema5:.2f}")
            
            supports.append(f"**过渡支撑 (EMA10):** ${ema10:.2f}")
            supports.append(f"**核心支撑 (EMA20):** ${ema20:.2f}")
            supports.append(f"**生命线支撑 (MA60):** {ma60_str}")
            supports.append(f"**30日低点强支撑:** ${recent_low:.2f}")
            resistances.append(f"**30日高点阻力:** ${recent_high:.2f}")
            
            with col1: st.info("\n\n".join(supports))
            with col2: st.warning("\n\n".join(resistances))
            
            # 动能与技术特征
            st.subheader("⚡ 动能与量价特征")
            macd_str = "🟢 多头金叉（动能充沛）" if macd_val > macd_signal and macd_diff > 0 else "🔴 动能减弱/死叉休整"
            st.write(f"- **MACD 状态:** `{macd_str}` (柱值: {macd_diff:.2f})")
            
            rsi_str = "⚠️ 超买（>70）" if rsi > 70 else "🟢 超卖（<30）" if rsi < 30 else "⚖️ 中性区间（30-70）"
            st.write(f"- **RSI (14):** `{rsi:.2f}` ({rsi_str})")
            
            price_chg = cur_price - close.iloc[-2] if total_days >= 2 else 0
            if price_chg > 0 and vol_ratio > 1.2: vol_desc = "🔥 放量上攻（主力主动买入）"
            elif price_chg < 0 and vol_ratio > 1.2: vol_desc = "⚠️ 放量下挫（抛压加大，暂避锋芒）"
            elif price_chg > 0 and vol_ratio <= 1.2: vol_desc = "⚖️ 缩量上涨（主力控盘良好，关注后市补量）"
            else: vol_desc = "🧊 缩量调整/蓄势洗盘"
            st.write(f"- **量价形态:** `{vol_desc}`")
            st.write(f"- **日均真实波幅 (ATR):** `${atr:.2f}` (单日平均波动参考)")
