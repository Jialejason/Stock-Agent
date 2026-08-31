import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime
from ta.trend import EMAIndicator, SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

st.set_page_config(page_title="投资小助手", layout="centered")
st.title("📈 投资小助手")
st.caption("输入美股代码，秒级诊断主力异动、财报博弈、黄金坑与量化点位")

ticker_input = st.text_input("美股代码", value="NVDA").strip().upper()

if st.button("开始全维度深度诊断", type="primary", use_container_width=True):
    with st.spinner(f"正在全维度诊断大盘、主力资金动向与 {ticker_input}..."):
        # 1. 大盘环境诊断 (标普 500 SPY)
        spy_df = yf.download("SPY", period="6mo", interval="1d", progress=False)
        spy_status = "正常"
        if not spy_df.empty:
            if isinstance(spy_df.columns, pd.MultiIndex):
                spy_df.columns = spy_df.columns.get_level_values(0)
            spy_close = spy_df['Close'].dropna()
            spy_ema20 = EMAIndicator(spy_close, 20).ema_indicator().iloc[-1]
            if spy_close.iloc[-1] < spy_ema20:
                spy_status = "⚠️ 大盘 (SPY) 跌破短期均线，系统性风险上升，全市场宜防守！"
            else:
                spy_status = "🟢 大盘 (SPY) 处于多头健康区间，市场环境支持顺势交易。"
        
        # 2. 个股数据拉取与清洗
        ticker_obj = yf.Ticker(ticker_input)
        df = yf.download(ticker_input, period="1y", interval="1d", progress=False)
        
        if df.empty:
            st.error("❌ 未找到股票数据，请检查代码是否正确。")
        else:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            close = df['Close'].dropna()
            open_p = df['Open'].dropna()
            high = df['High'].dropna()
            low = df['Low'].dropna()
            volume = df['Volume'].dropna()
            
            cur_price = close.iloc[-1]
            total_days = len(close)
            
            # 均线
            ema5 = EMAIndicator(close, min(5, total_days)).ema_indicator().iloc[-1]
            ema10 = EMAIndicator(close, min(10, total_days)).ema_indicator().iloc[-1]
            ema20 = EMAIndicator(close, min(20, total_days)).ema_indicator().iloc[-1]
            has_ma60 = total_days >= 60
            ma60 = SMAIndicator(close, 60).sma_indicator().iloc[-1] if has_ma60 else ema20
            ma60_str = f"${ma60:.2f}" if has_ma60 else "上市未满60日"
            
            # 动能 & 波动率
            rsi_series = RSIIndicator(close, min(14, total_days)).rsi()
            rsi = rsi_series.iloc[-1]
            macd_obj = MACD(close)
            macd_val = macd_obj.macd().iloc[-1]
            macd_signal = macd_obj.macd_signal().iloc[-1]
            macd_diff = macd_obj.macd_diff().iloc[-1]
            prev_macd_diff = macd_obj.macd_diff().iloc[-2] if total_days >= 2 else macd_diff
            atr = AverageTrueRange(high, low, close, min(14, total_days)).average_true_range().iloc[-1]
            
            # 成交量与量比
            cur_vol = volume.iloc[-1]
            avg_vol_5d = volume.iloc[-6:-1].mean() if total_days >= 6 else volume.mean()
            vol_ratio = (cur_vol / avg_vol_5d) if avg_vol_5d > 0 else 1.0
            
            recent_high = high.iloc[-min(30, total_days):].max()
            recent_low = low.iloc[-min(30, total_days):].min()
            
            # 3. 财报日获取
            earnings_date_str = "暂无近期数据"
            days_to_earnings = 999
            try:
                cal = ticker_obj.get_calendar()
                if cal and 'Earnings Date' in cal and len(cal['Earnings Date']) > 0:
                    e_date = cal['Earnings Date'][0]
                    if isinstance(e_date, (datetime, pd.Timestamp)):
                        days_to_earnings = (e_date.date() - datetime.now().date()).days
                        earnings_date_str = f"{e_date.strftime('%Y-%m-%d')} (距今 {days_to_earnings} 天)"
            except Exception:
                pass
            
            # 4. 主力异动与黄金坑量化研判
            whale_signals = []
            
            # (A) 主力抢跑 / 出货信号
            if cur_price > recent_high * 0.96 and prev_macd_diff > macd_diff and rsi > 65:
                whale_signals.append("🚨 **【主力出货/顶背离警报】** 股价处于近30日高位区，但MACD红柱缩短且动能衰竭，警惕主力拉高诱多出货！")
            if vol_ratio > 1.8 and (high.iloc[-1] - cur_price) > (cur_price - low.iloc[-1]) * 1.5:
                whale_signals.append("⚠️ **【主力放量抛压】** 今日伴随巨量拉出长上影线，上方主力套现抢跑迹象明显！")
                
            # (B) 财报博弈预警
            if 0 <= days_to_earnings <= 7:
                if rsi > 68:
                    whale_signals.append("⚡ **【财报前防抢跑】** 距财报公布不足 7 天且短期严重超买，获利盘极易在财报前夕砸盘避险，建议逢高落袋！")
                else:
                    whale_signals.append("📅 **【财报波动期】** 即将公布财报，资金博弈剧烈，严格控制仓位在 2 成以下或空仓观望。")
            elif 8 <= days_to_earnings <= 21 and cur_price > ema20 and macd_diff > 0:
                whale_signals.append("🚀 **【财报前抢跑预热】** 距财报 2~3 周，均线多头且动能向上，属于典型的“财报前炒作拉升”窗口期。")
                
            # (C) 黄金坑与止跌信号
            gold_pit_price = min(ma60, cur_price - atr * 1.2) if has_ma60 else recent_low
            is_stopped_falling = (cur_price > ema5) and (close.iloc[-2] < EMAIndicator(close, 5).ema_indicator().iloc[-2])
            
            if rsi < 38 and cur_price <= gold_pit_price * 1.03:
                whale_signals.append(f"💎 **【出现黄金坑抄底信号】** RSI跌至超卖极端恐慌区，价格触及极限强支撑 `${gold_pit_price:.2f}` 附近，赔率极高！")
            elif is_stopped_falling and cur_price > ema10:
                whale_signals.append("🟢 **【短线止跌企稳确认】** K线成功收复 5 日超短线并站稳 10 日线，下跌动能暂缓，右侧企稳成立。")

            # 顶部看板
            if "⚠️" in spy_status: st.warning(f"**大盘风控提示:** {spy_status}")
            else: st.success(f"**大盘风控提示:** {spy_status}")
                
            col_m1, col_m2 = st.columns(2)
            col_m1.metric(label=f"{ticker_input} 最新价", value=f"${cur_price:.2f}")
            vol_status = "🔥 放量" if vol_ratio > 1.3 else "🧊 缩量" if vol_ratio < 0.7 else "⚖️ 平量"
            col_m2.metric(label="5日量比", value=f"{vol_ratio:.2f} 倍", delta=vol_status)

            # 主力与财报异动板块
            st.subheader("🕵️ 主力资金与财报博弈雷达")
            if whale_signals:
                for sig in whale_signals:
                    st.markdown(sig)
            else:
                st.info("⚖️ **主力状态平稳:** 当前暂未监测到极端顶背离出货或黄金坑超卖异动，按常规技术点位执行即可。")
            st.caption(f"📅 **预计下次财报日:** `{earnings_date_str}`")

            # 实战点位与盈亏比
            entry_mid = (ema20 + ema10) / 2
            rec_stop = ema20 - (atr * 0.4)
            rec_target = recent_high
            risk = entry_mid - rec_stop
            reward = rec_target - entry_mid
            rr_ratio = (reward / risk) if risk > 0 else 0.0
            
            st.subheader("🎯 实战操作指南 (量化风控)")
            rr_badge = f"🔥 极佳 ({rr_ratio:.2f} : 1)" if rr_ratio >= 2.0 else f"⚖️ 合格 ({rr_ratio:.2f} : 1)" if rr_ratio >= 1.5 else f"❌ 偏低 ({rr_ratio:.2f} : 1)"
            
            if cur_price > ema20 and macd_diff > 0 and rsi < 65:
                st.success(
                    f"**策略定性:** 🟢 趋势良好，适合分批逢低吸纳\n\n"
                    f"- **预估盈亏比:** `{rr_badge}`\n"
                    f"- **建议买入区间:** `${ema20:.2f} ~ ${ema10:.2f}` (分批介入)\n"
                    f"- **极限黄金坑参考价:** `${gold_pit_price:.2f}` (极端洗盘低吸位)\n"
                    f"- **严格止损参考:** `${rec_stop:.2f}` (跌破减仓规避风险)\n"
                    f"- **上方第一止盈目标:** `${rec_target:.2f}` (阶梯兑现利润)"
                )
            elif cur_price > ema20 and rsi >= 65:
                st.warning(f"**策略定性:** ⚠️ 处于超买高位强势区，严禁追高，防范主力套现砸盘！")
            else:
                st.info(f"**策略定性:** ❄️ 处于均线下方或震荡弱势期，暂以观望或等待黄金坑企稳。")

            # 支撑与阻力
            st.subheader("🛡️ 关键支撑与阻力位")
            col1, col2 = st.columns(2)
            supports, resistances = [], []
            if ema5 < cur_price: supports.append(f"**超短支撑 (EMA5):** ${ema5:.2f}")
            else: resistances.append(f"**短线阻力 (EMA5):** ${ema5:.2f}")
            supports.extend([f"**过渡支撑 (EMA10):** ${ema10:.2f}", f"**核心支撑 (EMA20):** ${ema20:.2f}", f"**生命线支撑 (MA60):** {ma60_str}", f"**30日低点强支撑:** ${recent_low:.2f}"])
            resistances.append(f"**30日高点阻力:** ${recent_high:.2f}")
            with col1: st.info("\n\n".join(supports))
            with col2: st.warning("\n\n".join(resistances))
            
            # 动能指标
            st.subheader("⚡ 动能与量价特征")
            macd_str = "🟢 多头金叉（动能充沛）" if macd_val > macd_signal and macd_diff > 0 else "🔴 动能减弱/死叉休整"
            st.write(f"- **MACD 状态:** `{macd_str}` (柱值: {macd_diff:.2f})")
            st.write(f"- **RSI (14):** `{rsi:.2f}` ({'⚠️ 超买' if rsi > 70 else '🟢 超卖' if rsi < 30 else '⚖️ 中性'})")
            st.write(f"- **日均真实波幅 (ATR):** `${atr:.2f}`")

            # 专属实时新闻
            st.subheader("📰 个股最新专属新闻与动态")
            try:
                news_list = ticker_obj.news
                if news_list and len(news_list) > 0:
                    for item in news_list[:4]:
                        # 兼容不同 yfinance 版本的字典/嵌套结构
                        title = item.get('title') or (item.get('content', {}).get('title') if isinstance(item.get('content'), dict) else '无标题')
                        publisher = item.get('publisher') or (item.get('content', {}).get('provider', {}).get('displayName') if isinstance(item.get('content'), dict) else '财经资讯')
                        link = item.get('link') or (item.get('content', {}).get('canonicalUrl', {}).get('url') if isinstance(item.get('content'), dict) else '#')
                        st.markdown(f"- **[{title}]({link})** — *{publisher}*")
                else:
                    st.caption("暂未获取到该标的的突发新闻。")
            except Exception:
                st.caption("暂未获取到最新新闻资讯。")
