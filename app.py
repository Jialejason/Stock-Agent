import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime
from ta.trend import EMAIndicator, SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
import google.generativeai as genai

st.set_page_config(page_title="投资小助手", layout="centered")
st.title("📈 投资小助手")
st.caption("输入美股代码，秒级诊断量价结构与 Gemini AI 保姆级大白话解读")

# 获取 API Key (优先从 Secrets 读取，其次从界面输入)
api_key = st.secrets.get("GEMINI_API_KEY", "") if "GEMINI_API_KEY" in st.secrets else ""

if not api_key:
    with st.expander("🔑 配置 Gemini API Key", expanded=False):
        api_key = st.text_input("Gemini API Key", type="password", help="从 aistudio.google.com 获取")

# 初始化历史搜索/自选列表
if "history_tickers" not in st.session_state:
    st.session_state.history_tickers = ["SPCX", "NVDA", "TSLA", "AAPL"]

if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = "SPCX"

# 动态展示自选与最近查询按钮
st.write("**🔥 快速自选与最近查询:**")
cols = st.columns(len(st.session_state.history_tickers))
for i, ticker in enumerate(st.session_state.history_tickers):
    if cols[i].button(ticker, use_container_width=True):
        st.session_state.selected_ticker = ticker

ticker_input = st.text_input("美股代码", value=st.session_state.selected_ticker).strip().upper()

if st.button("开始全维度深度诊断", type="primary", use_container_width=True):
    # 动态加入历史查询栏（保留最新 5 个）
    if ticker_input and ticker_input in st.session_state.history_tickers:
        st.session_state.history_tickers.remove(ticker_input)
    if ticker_input:
        st.session_state.history_tickers.insert(0, ticker_input)
        if len(st.session_state.history_tickers) > 5:
            st.session_state.history_tickers.pop()

    with st.spinner(f"正在全维度诊断大盘、主力动向与 {ticker_input}..."):
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
            atr = AverageTrueRange(high, low, close, min(14, total_days)).average_true_range().iloc[-1]
            
            # 成交量与量比
            cur_vol = volume.iloc[-1]
            avg_vol_5d = volume.iloc[-6:-1].mean() if total_days >= 6 else volume.mean()
            vol_ratio = (cur_vol / avg_vol_5d) if avg_vol_5d > 0 else 1.0
            
            recent_high = high.iloc[-min(30, total_days):].max()
            recent_low = low.iloc[-min(30, total_days):].min()
            
            # 3. 财报日获取
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
            
            # 顶部看板
            if "⚠️" in spy_status: st.warning(f"**大盘风控提示:** {spy_status}")
            else: st.success(f"**大盘风控提示:** {spy_status}")
                
            col_m1, col_m2 = st.columns(2)
            col_m1.metric(label=f"{ticker_input} 最新价", value=f"${cur_price:.2f}")
            vol_status = "🔥 放量" if vol_ratio > 1.3 else "🧊 缩量" if vol_ratio < 0.7 else "⚖️ 平量"
            col_m2.metric(label="5日量比", value=f"{vol_ratio:.2f} 倍", delta=vol_status)

            # 新闻抓取
            news_text = ""
            try:
                news_list = ticker_obj.news
                if news_list and len(news_list) > 0:
                    for item in news_list[:3]:
                        title = item.get('title') or (item.get('content', {}).get('title') if isinstance(item.get('content'), dict) else '')
                        if title: news_text += f"- {title}\n"
            except Exception:
                pass

            # 4. 🤖 Gemini AI 智能解读模块
            st.subheader("🤖 Gemini 操盘手大白话解读")
            if api_key:
                try:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-3.6-flash')
                    
                    prompt = f"""
                    你是一名资深的职业美股操盘手兼新手导师。请根据以下技术与量化数据，用最通俗易懂、接地气的大白话给完全不懂技术指标的新手写一份简明诊断指南。

                    【股票标的】: {ticker_input}
                    【最新价格】: ${cur_price:.2f}
                    【大盘环境】: {spy_status}
                    【均线数据】: EMA5=${ema5:.2f}, EMA10=${ema10:.2f}, EMA20=${ema20:.2f}, MA60={ma60_str}
                    【动能与量能】: RSI={rsi:.2f}, MACD柱值={macd_diff:.2f}, 5日量比={vol_ratio:.2f}倍
                    【关键阻力与支撑】: 30日最高阻力=${recent_high:.2f}, 30日最低强支撑=${recent_low:.2f}
                    【近期财报日】: {earnings_date_str}
                    【最新新闻摘要】:
                    {news_text if news_text else "暂无突发新闻"}

                    请直接按以下 3 个结构输出（禁止废话，重点突出）：
                    1. 🧐 **现状通俗解读**：用 2 句话讲清楚目前这只股票到底是强势、震荡还是危险期？
                    2. 💡 **新手实操动作**：明确告诉新手目前能不能买？如果想买该挂在什么具体价位？如果在场内跌破哪个价位必须止损认错？
                    3. ⚠️ **核心避险警示**：结合大盘、财报或成交量，提醒新手当前最需要防范的坑是什么？
                    """
                    with st.spinner("🤖 Gemini 正在为你生成通俗解读..."):
                        response = model.generate_content(prompt)
                        st.markdown(response.text)
                except Exception as e:
                    st.error(f"Gemini 生成失败: {e}")
            else:
                st.info("💡 请填入你的 Gemini API Key 以激活 AI 解读。")

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
