import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime
from ta.trend import EMAIndicator, SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
import google.generativeai as genai

st.set_page_config(page_title="投资小助手 Pro", layout="centered")
st.title("📈 投资小助手 Pro (宏观全维版)")
st.caption("秒级融合【标普/纳指/VIX恐慌/美债利率/地缘宏观】与 Gemini AI 保姆级大白话操盘指南")

# 1. 获取 API Key (优先从 Secrets 读取，其次从界面输入)
api_key = st.secrets.get("GEMINI_API_KEY", "") if "GEMINI_API_KEY" in st.secrets else ""

if not api_key:
    with st.expander("🔑 配置 Gemini API Key", expanded=False):
        api_key = st.text_input("Gemini API Key", type="password", help="从 aistudio.google.com 获取")

# 2. 动态自选与历史搜索栏
if "history_tickers" not in st.session_state:
    st.session_state.history_tickers = ["SPCX", "NVDA", "TSLA", "AAPL"]

if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = "SPCX"

st.write("**🔥 快速自选与最近查询:**")
cols = st.columns(len(st.session_state.history_tickers))
for i, ticker in enumerate(st.session_state.history_tickers):
    if cols[i].button(ticker, use_container_width=True):
        st.session_state.selected_ticker = ticker

ticker_input = st.text_input("美股代码", value=st.session_state.selected_ticker).strip().upper()

# 3. 核心诊断逻辑
if st.button("开始全维度深度诊断", type="primary", use_container_width=True):
    # 更新动态自选栏（保留最新 5 个）
    if ticker_input and ticker_input in st.session_state.history_tickers:
        st.session_state.history_tickers.remove(ticker_input)
    if ticker_input:
        st.session_state.history_tickers.insert(0, ticker_input)
        if len(st.session_state.history_tickers) > 5:
            st.session_state.history_tickers.pop()

    with st.spinner(f"正在全维度诊断宏观局势、地缘情绪与 {ticker_input}..."):
        # 1. 宏观四维雷达 (SPY, QQQ, VIX, TNX)
        macro_tickers = ["SPY", "QQQ", "^VIX", "^TNX"]
        macro_data = yf.download(macro_tickers, period="3mo", interval="1d", progress=False)
        
        market_status = "🟢 顺势顺风：宏观环境健康，支持顺势交易。"
        vix_status_str = "正常"
        tnx_status_str = "正常"
        
        try:
            if not macro_data.empty:
                # 兼容 MultiIndex
                close_data = macro_data['Close']
                
                # SPY & QQQ
                spy_close = close_data['SPY'].dropna().iloc[-1]
                spy_ema20 = EMAIndicator(close_data['SPY'].dropna(), 20).ema_indicator().iloc[-1]
                qqq_close = close_data['QQQ'].dropna().iloc[-1]
                qqq_ema20 = EMAIndicator(close_data['QQQ'].dropna(), 20).ema_indicator().iloc[-1]
                
                # VIX 恐慌指数
                vix_close = close_data['^VIX'].dropna().iloc[-1]
                if vix_close > 22:
                    vix_status_str = f"⚠️ 市场避险情绪高涨 (VIX={vix_close:.2f} > 22)，防范黑天鹅跳水！"
                else:
                    vix_status_str = f"🟢 市场情绪稳定 (VIX={vix_close:.2f})"

                # TNX 10年期美债收益率
                tnx_close = close_data['^TNX'].dropna().iloc[-1]
                tnx_status_str = f"10年美债收益率: {tnx_close:.2f}%"

                # 宏观多空判定
                if spy_close < spy_ema20 and qqq_close < qqq_ema20:
                    market_status = "🔴 极度预警：标普(SPY) 与 纳指(QQQ) 均跌破EMA20生命线，全市场防守！"
                elif qqq_close < qqq_ema20:
                    market_status = "⚠️ 结构分化：纳指(QQQ) 破位走弱，科技成长股承压，操作宜谨慎！"
                elif spy_close < spy_ema20:
                    market_status = "⚠️ 警示：标普(SPY) 跌破均线，传统权重走弱，防范系统性回调！"
                else:
                    market_status = "🟢 多头顺风：标普与纳指均处于健康上升通道。"
        except Exception:
            pass

        # 2. 个股数据清洗与量化计算
        ticker_obj = yf.Ticker(ticker_input)
        df = yf.download(ticker_input, period="1y", interval="1d", progress=False)
        
        if df.empty:
            st.error("❌ 未找到该股票数据，请检查代码是否正确。")
        else:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            close = df['Close'].dropna()
            high = df['High'].dropna()
            low = df['Low'].dropna()
            volume = df['Volume'].dropna()
            
            cur_price = close.iloc[-1]
            total_days = len(close)
            
            # 关键均线
            ema5 = EMAIndicator(close, min(5, total_days)).ema_indicator().iloc[-1]
            ema10 = EMAIndicator(close, min(10, total_days)).ema_indicator().iloc[-1]
            ema20 = EMAIndicator(close, min(20, total_days)).ema_indicator().iloc[-1]
            has_ma60 = total_days >= 60
            ma60 = SMAIndicator(close, 60).sma_indicator().iloc[-1] if has_ma60 else ema20
            ma60_str = f"${ma60:.2f}" if has_ma60 else "上市未满60日"
            
            # 动能指标与波动率
            rsi_series = RSIIndicator(close, min(14, total_days)).rsi()
            rsi = rsi_series.iloc[-1]
            macd_obj = MACD(close)
            macd_val = macd_obj.macd().iloc[-1]
            macd_signal = macd_obj.macd_signal().iloc[-1]
            macd_diff = macd_obj.macd_diff().iloc[-1]
            atr = AverageTrueRange(high, low, close, min(14, total_days)).average_true_range().iloc[-1]
            
            # 量比算法
            cur_vol = volume.iloc[-1]
            avg_vol_5d = volume.iloc[-6:-1].mean() if total_days >= 6 else volume.mean()
            vol_ratio = (cur_vol / avg_vol_5d) if avg_vol_5d > 0 else 1.0
            
            recent_high = high.iloc[-min(30, total_days):].max()
            recent_low = low.iloc[-min(30, total_days):].min()
            
            # 3. 财报日追踪
            earnings_date_str = "暂无近期数据"
            try:
                cal = ticker_obj.get_calendar()
                if cal and 'Earnings Date' in cal and len(cal['Earnings Date']) > 0:
                    e_date = cal['Earnings Date'][0]
                    if isinstance(e_date, (datetime, pd.Timestamp)):
                        days_to_earnings = (e_date.date() - datetime.now().date()).days
                        earnings_date_str = f"{e_date.strftime('%Y-%m-%d')} (距今 {days_to_earnings} 天)"
            except Exception:
                pass
            
            # 4. 抓取新闻（个股 + 宏观大盘）
            news_text = ""
            try:
                news_list = ticker_obj.news
                if news_list and len(news_list) > 0:
                    for item in news_list[:3]:
                        title = item.get('title') or (item.get('content', {}).get('title') if isinstance(item.get('content'), dict) else '')
                        if title: news_text += f"- [个股] {title}\n"
                
                # 补充宏观 SPY 新闻
                spy_obj = yf.Ticker("SPY")
                spy_news = spy_obj.news
                if spy_news and len(spy_news) > 0:
                    for item in spy_news[:2]:
                        s_title = item.get('title') or (item.get('content', {}).get('title') if isinstance(item.get('content'), dict) else '')
                        if s_title: news_text += f"- [宏观/政策] {s_title}\n"
            except Exception:
                pass

            # 顶部状态与核心指标看板
            if "🔴" in market_status:
                st.error(f"**大盘风控:** {market_status}")
            elif "⚠️" in market_status:
                st.warning(f"**大盘风控:** {market_status}")
            else:
                st.success(f"**大盘风控:** {market_status}")
            
            # 宏观环境概览条
            st.info(f"🌐 **宏观全维监控：** {vix_status_str} ｜ 🏛️ {tnx_status_str}")
                
            col_m1, col_m2 = st.columns(2)
            col_m1.metric(label=f"{ticker_input} 最新价", value=f"${cur_price:.2f}")
            vol_status = "🔥 放量" if vol_ratio > 1.3 else "🧊 缩量" if vol_ratio < 0.7 else "⚖️ 平量"
            col_m2.metric(label="5日量比", value=f"{vol_ratio:.2f} 倍", delta=vol_status)

            # 5. 🤖 Gemini AI 智能操盘解读模块 (宏观+量价全融合)
            st.subheader("🤖 Gemini 操盘手大白话解读")
            if api_key:
                try:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-3.6-flash')
                    
                    prompt = f"""
                    你是一名资深的职业美股操盘手兼新手导师。请结合宏观地缘局势、美债利率环境、以及个股的技术量化数据，用通俗易懂的大白话给完全不懂技术指标的新手写一份诊断指南。

                    【股票标的】: {ticker_input}
                    【最新价格】: ${cur_price:.2f}
                    【大盘环境 (SPY/QQQ)】: {market_status}
                    【市场恐慌与利率 (VIX/TNX)】: {vix_status_str} ｜ {tnx_status_str}
                    【均线数据】: EMA5=${ema5:.2f}, EMA10=${ema10:.2f}, EMA20=${ema20:.2f}, MA60={ma60_str}
                    【动能与量能】: RSI={rsi:.2f}, MACD柱值={macd_diff:.2f}, 5日量比={vol_ratio:.2f}倍
                    【关键阻力与支撑】: 30日最高阻力=${recent_high:.2f}, 30日最低强支撑=${recent_low:.2f}
                    【近期财报日】: {earnings_date_str}
                    【最新个股与宏观新闻】:
                    {news_text if news_text else "暂无突发新闻"}

                    请直接按以下 3 个结构输出（禁止废话，重点突出）：
                    1. 🌍 **宏观与局势定性**：用 2 句话讲清当前大盘、美债利率/恐慌情绪以及地缘/新闻大环境对这只股票是有利还是在施压？
                    2. 💡 **新手实操动作**：明确告诉新手目前能不能买？如果想买该挂在什么具体价位？如果在场内跌破哪个价位必须立刻止损认错？
                    3. ⚠️ **核心避险警示**：结合大盘、财报倒计时或缩量/放量异动，提醒新手当前最需要防范的黑天鹅或坑是什么？
                    """
                    with st.spinner("🤖 Gemini 正在结合宏观与量价为你生成解读..."):
                        response = model.generate_content(prompt)
                        st.markdown(response.text)
                except Exception as e:
                    st.error(f"Gemini 生成失败: {e}")
            else:
                st.info("💡 请填入你的 Gemini API Key 以激活 AI 解读。")

            # 支撑与阻力位
            st.subheader("🛡️ 关键支撑与阻力位")
            col1, col2 = st.columns(2)
            supports, resistances = [], []
            if ema5 < cur_price: supports.append(f"**超短支撑 (EMA5):** ${ema5:.2f}")
            else: resistances.append(f"**短线阻力 (EMA5):** ${ema5:.2f}")
            supports.extend([f"**过渡支撑 (EMA10):** ${ema10:.2f}", f"**核心支撑 (EMA20):** ${ema20:.2f}", f"**生命线支撑 (MA60):** {ma60_str}", f"**30日低点强支撑:** ${recent_low:.2f}"])
            resistances.append(f"**30日高点阻力:** ${recent_high:.2f}")
            with col1: st.info("\n\n".join(supports))
            with col2: st.warning("\n\n".join(resistances))
            
            # 动能与量价特征
            st.subheader("⚡ 动能与量价特征")
            macd_str = "🟢 多头金叉（动能充沛）" if macd_val > macd_signal and macd_diff > 0 else "🔴 动能减弱/死叉休整"
            st.write(f"- **MACD 状态:** `{macd_str}` (柱值: {macd_diff:.2f})")
            st.write(f"- **RSI (14):** `{rsi:.2f}` ({'⚠️ 超买' if rsi > 70 else '🟢 超卖' if rsi < 30 else '⚖️ 中性'})")
            st.write(f"- **日均真实波幅 (ATR):** `${atr:.2f}`")
