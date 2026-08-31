import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime
import time
from ta.trend import EMAIndicator, SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
import google.generativeai as genai

st.set_page_config(page_title="投资小助手 Pro", layout="centered")
st.title("📈 投资小助手 Pro (大道至简·多周期量化版)")
st.caption("⚡ 5分钟智能缓存 ｜ 🧭 周线趋势 ｜ 🧱 日线形态 ｜ 🎯 1小时狙击点位 ｜ 💬 专属AI追问")

# 1. 获取 API Key
api_key = st.secrets.get("GEMINI_API_KEY", "") if "GEMINI_API_KEY" in st.secrets else ""

if not api_key:
    with st.expander("🔑 配置 Gemini API Key", expanded=False):
        api_key = st.text_input("Gemini API Key", type="password", help="从 aistudio.google.com 获取")

# 2. 核心量化算法（多周期共振 + 5分钟智能共享缓存）
@st.cache_data(ttl=300, show_spinner=False)
def fetch_and_analyze(ticker_input, api_key_val):
    # 宏观大盘监控 (SPY / QQQ / VIX / TNX)
    macro_tickers = ["SPY", "QQQ", "^VIX", "^TNX"]
    macro_data = yf.download(macro_tickers, period="3mo", interval="1d", progress=False)
    
    market_status = "🟢 顺势顺风：宏观大盘处于多头健康区间。"
    vix_status_str = "正常"
    tnx_status_str = "正常"
    
    try:
        if not macro_data.empty:
            close_data = macro_data['Close']
            spy_close = close_data['SPY'].dropna().iloc[-1]
            spy_ema20 = EMAIndicator(close_data['SPY'].dropna(), 20).ema_indicator().iloc[-1]
            qqq_close = close_data['QQQ'].dropna().iloc[-1]
            qqq_ema20 = EMAIndicator(close_data['QQQ'].dropna(), 20).ema_indicator().iloc[-1]
            
            vix_close = close_data['^VIX'].dropna().iloc[-1]
            vix_status_str = f"⚠️ 恐慌高涨 (VIX={vix_close:.2f} > 22)" if vix_close > 22 else f"🟢 情绪平稳 (VIX={vix_close:.2f})"
            
            tnx_close = close_data['^TNX'].dropna().iloc[-1]
            tnx_status_str = f"10年美债收益率: {tnx_close:.2f}%"

            if spy_close < spy_ema20 and qqq_close < qqq_ema20:
                market_status = "🔴 极度预警：标普(SPY) 与 纳指(QQQ) 均跌破EMA20，全市场重度防守！"
            elif qqq_close < qqq_ema20:
                market_status = "⚠️ 结构分化：纳指(QQQ) 破位走弱，科技与成长股承压！"
            elif spy_close < spy_ema20:
                market_status = "⚠️ 警示：标普(SPY) 跌破均线，传统权重走弱，防范回调！"
            else:
                market_status = "🟢 多头顺风：标普与纳指均处于健康上升通道。"
    except Exception:
        pass

    # A. 获取日线数据 (Daily)
    ticker_obj = yf.Ticker(ticker_input)
    df_daily = yf.download(ticker_input, period="1y", interval="1d", progress=False)
    
    if df_daily.empty:
        return None, "未找到该股票数据，请检查美股代码是否输入正确。"
        
    if isinstance(df_daily.columns, pd.MultiIndex):
        df_daily.columns = df_daily.columns.get_level_values(0)
    
    close_d = df_daily['Close'].dropna()
    high_d = df_daily['High'].dropna()
    low_d = df_daily['Low'].dropna()
    vol_d = df_daily['Volume'].dropna()
    
    cur_price = close_d.iloc[-1]
    total_days = len(close_d)
    
    ema5 = EMAIndicator(close_d, min(5, total_days)).ema_indicator().iloc[-1]
    ema10 = EMAIndicator(close_d, min(10, total_days)).ema_indicator().iloc[-1]
    ema20 = EMAIndicator(close_d, min(20, total_days)).ema_indicator().iloc[-1]
    has_ma60 = total_days >= 60
    ma60 = SMAIndicator(close_d, 60).sma_indicator().iloc[-1] if has_ma60 else ema20
    ma60_str = f"${ma60:.2f}" if has_ma60 else "上市未满60日"
    
    rsi_d = RSIIndicator(close_d, min(14, total_days)).rsi().iloc[-1]
    macd_obj_d = MACD(close_d)
    macd_diff_d = macd_obj_d.macd_diff().iloc[-1]
    atr_d = AverageTrueRange(high_d, low_d, close_d, min(14, total_days)).average_true_range().iloc[-1]
    
    cur_vol = vol_d.iloc[-1]
    avg_vol_5d = vol_d.iloc[-6:-1].mean() if total_days >= 6 else vol_d.mean()
    vol_ratio = (cur_vol / avg_vol_5d) if avg_vol_5d > 0 else 1.0

    # 阶梯阻力与支撑
    high_30d = high_d.iloc[-min(30, total_days):].max()
    high_120d = high_d.iloc[-min(120, total_days):].max() if total_days >= 30 else high_30d
    high_52w = high_d.max()
    low_30d = low_d.iloc[-min(30, total_days):].min()
    
    resistance_list = []
    if cur_price >= high_30d * 0.99:
        if high_120d > cur_price * 1.01:
            resistance_list.append(f"🔥 突破30日高点！下一阻力锁定【半年高点】: ${high_120d:.2f}")
        elif high_52w > cur_price * 1.01:
            resistance_list.append(f"🔥 突破阶段平台！下一阻力锁定【52周历史大顶】: ${high_52w:.2f}")
        else:
            ath_target = cur_price + (1.5 * atr_d)
            resistance_list.append(f"🚀 创历史新高（上方无套牢盘）！动能拓展目标位: ${ath_target:.2f}")
    else:
        resistance_list.append(f"30日阶段强阻力: ${high_30d:.2f}")
        if high_120d > high_30d:
            resistance_list.append(f"半年期重要阻力: ${high_120d:.2f}")
    
    support_list = []
    if ema5 < cur_price: support_list.append(f"超短支撑 (EMA5): ${ema5:.2f}")
    else: resistance_list.insert(0, f"短线均线阻力 (EMA5): ${ema5:.2f}")
    
    support_list.append(f"过渡防守 (EMA10): ${ema10:.2f}")
    support_list.append(f"多空分水岭 (EMA20): ${ema20:.2f}")
    support_list.append(f"中期生命线 (MA60): {ma60_str}")
    support_list.append(f"30日筑底强支撑: ${low_30d:.2f}")

    pit_status = "正常走势"
    if rsi_d < 38 and cur_price <= low_30d * 1.03:
        pit_status = "💎 极端超卖黄金坑：指标极度冰点超跌，存在高盈亏比反弹反转机会！"
    elif cur_price >= ema20 and macd_diff_d > 0 and vol_ratio >= 0.9:
        pit_status = "🧱 右侧企稳确立：股价重回EMA20均线之上，动能回暖，企稳结构扎实。"
    elif cur_price < ema20 and vol_ratio < 0.7:
        pit_status = "🧊 缩量磨底中：跌破均线但抛压衰竭，等待放量企稳确认。"

    # B. 获取周线数据 (Weekly)
    weekly_status = "周线中性"
    try:
        df_weekly = yf.download(ticker_input, period="2y", interval="1wk", progress=False)
        if not df_weekly.empty:
            if isinstance(df_weekly.columns, pd.MultiIndex):
                df_weekly.columns = df_weekly.columns.get_level_values(0)
            close_w = df_weekly['Close'].dropna()
            if len(close_w) >= 20:
                w_ema20 = EMAIndicator(close_w, 20).ema_indicator().iloc[-1]
                w_macd_diff = MACD(close_w).macd_diff().iloc[-1]
                if cur_price >= w_ema20 and w_macd_diff >= 0:
                    weekly_status = "🟢 顺风大牛势：周线站稳EMA20生命线且动能向上，顺势做多胜率极高！"
                elif cur_price >= w_ema20:
                    weekly_status = "🟡 强势震荡：周线位于均线上方，但动能稍有放缓，高位整固阶段。"
                else:
                    weekly_status = "🔴 逆风熊势/深度调整：周线跌破EMA20，大趋势向下，严禁逆势盲目抄底！"
    except Exception:
        pass

    # C. 获取 1 小时盘中数据 (1-Hour)
    hourly_status = "盘中中性"
    hourly_suggested_entry = cur_price
    hourly_stop_loss = cur_price * 0.985
    try:
        df_hourly = yf.download(ticker_input, period="1mo", interval="1h", progress=False)
        if not df_hourly.empty:
            if isinstance(df_hourly.columns, pd.MultiIndex):
                df_hourly.columns = df_hourly.columns.get_level_values(0)
            close_h = df_hourly['Close'].dropna()
            low_h = df_hourly['Low'].dropna()
            if len(close_h) >= 20:
                h_ema20 = EMAIndicator(close_h, 20).ema_indicator().iloc[-1]
                h_rsi = RSIIndicator(close_h, 14).rsi().iloc[-1]
                h_recent_low = low_h.iloc[-20:].min()
                
                hourly_suggested_entry = h_ema20 if cur_price > h_ema20 else cur_price
                hourly_stop_loss = h_recent_low * 0.995
                
                if cur_price >= h_ema20 and 45 <= h_rsi <= 65:
                    hourly_status = "🎯 盘中狙击买点已触发：1小时结构回踩企稳，极具盈亏比！"
                elif h_rsi >= 70:
                    hourly_status = "⚠️ 1小时盘中超买：短线急拉，切勿追高，等待盘中回踩挂单！"
                elif cur_price < h_ema20:
                    hourly_status = "🧊 1小时盘中走弱：等待盘中重回1小时EMA20均线后再挂单。"
    except Exception:
        pass

    # D. 财报与突发新闻
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
    
    news_text = ""
    try:
        news_list = ticker_obj.news
        if news_list and len(news_list) > 0:
            for item in news_list[:3]:
                title = item.get('title') or (item.get('content', {}).get('title') if isinstance(item.get('content'), dict) else '')
                if title: news_text += f"- [个股] {title}\n"
    except Exception:
        pass

    # E. Gemini 3.6 生成
    ai_analysis_text = ""
    if api_key_val:
        genai.configure(api_key=api_key_val)
        model = genai.GenerativeModel('gemini-3.6-flash')
        
        prompt = f"""
        你是一名顶级的资深美股操盘手兼新手导师。你的核心教学宗旨是【大道至简】。
        请根据以下【周线-日线-1小时多周期共振】指标，用极其简明、直白的语言，为零基础小白写一份毫无废话的行动指南。

        【股票标的】: {ticker_input}
        【最新现价】: ${cur_price:.2f}
        【宏观大盘 (SPY/QQQ)】: {market_status}
        【市场情绪与利率】: {vix_status_str} ｜ {tnx_status_str}
        【🧭 周线大趋势】: {weekly_status}
        【🧱 日线形态雷达】: {pit_status}
        【🎯 1小时盘中狙击】: {hourly_status} (盘中建议挂单参考: ${hourly_suggested_entry:.2f}, 1小时防守线: ${hourly_stop_loss:.2f})
        【阶梯阻力与目标】: {'; '.join(resistance_list)}
        【阶梯防守支撑位】: {'; '.join(support_list)}
        【财报倒计时】: {earnings_date_str}
        【突发资讯】:
        {news_text if news_text else "暂无突发新闻"}

        请严格按以下 3 个极简板块输出（突出重点，给出具体数字，小白照着做即可）：
        1. 🚦 **多周期共振定性（红绿灯）**：用 2 句话讲清大趋势是顺风还是逆风？当前是该进攻、该埋伏、还是必须空仓管住手？
        2. 💡 **小白实操动作（直接给数字）**：
           - **买入建议**：能不能买？如果能买，建议把买单挂在什么精确价格？
           - **止盈目标**：第一目标位和突破后的加速位看至哪里？
           - **铁血止损**：跌破哪个精确价位必须无条件止损离场？
        3. ⚠️ **最核心的一个避险坑**：一句话点透当前最大的风险。
        """
        for attempt in range(2):
            try:
                response = model.generate_content(prompt)
                ai_analysis_text = response.text
                break
            except Exception as e:
                err_msg = str(e)
                if ("429" in err_msg or "quota" in err_msg.lower()) and attempt == 0:
                    time.sleep(3)
                    continue
                elif "429" in err_msg or "quota" in err_msg.lower():
                    ai_analysis_text = "⏳ **AI 操盘手正在复盘全网数据**：触发了临时调用冷却，数据已为您自动缓存，请 10 秒后刷新查看！"
                    break
                else:
                    ai_analysis_text = f"诊断暂时中断: {err_msg}"
                    break

    result_bundle = {
        "market_status": market_status,
        "vix_status_str": vix_status_str,
        "tnx_status_str": tnx_status_str,
        "weekly_status": weekly_status,
        "pit_status": pit_status,
        "hourly_status": hourly_status,
        "cur_price": cur_price,
        "vol_ratio": vol_ratio,
        "hourly_suggested_entry": hourly_suggested_entry,
        "hourly_stop_loss": hourly_stop_loss,
        "support_list": support_list,
        "resistance_list": resistance_list,
        "macd_diff_d": macd_diff_d,
        "rsi_d": rsi_d,
        "atr_d": atr_d,
        "ai_analysis_text": ai_analysis_text,
        "cache_time": datetime.now().strftime("%H:%M:%S")
    }
    return result_bundle, None

# 3. 快速自选栏
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

# 4. 诊断展示看板
if st.button("开始多周期全维共振诊断", type="primary", use_container_width=True):
    if ticker_input and ticker_input in st.session_state.history_tickers:
        st.session_state.history_tickers.remove(ticker_input)
    if ticker_input:
        st.session_state.history_tickers.insert(0, ticker_input)
        if len(st.session_state.history_tickers) > 5:
            st.session_state.history_tickers.pop()

    with st.spinner(f"正在全维运算周线/日线/1小时共振数据 ({ticker_input})..."):
        data, err = fetch_and_analyze(ticker_input, api_key)
        if err:
            st.error(f"❌ {err}")
        else:
            st.session_state.current_data = data
            st.session_state.current_ticker = ticker_input
            # 切换股票时清空上一次的追问历史
            st.session_state.chat_history = []

# 渲染分析面板
if "current_data" in st.session_state and st.session_state.current_data:
    data = st.session_state.current_data
    curr_ticker = st.session_state.get("current_ticker", ticker_input)
    
    st.caption(f"⚡ 数据已智能缓存（最后刷新时间: {data['cache_time']}，5分钟内全员秒开无消耗）")
    
    # 宏观大盘风控
    if "🔴" in data['market_status']: st.error(f"**大盘风控:** {data['market_status']}")
    elif "⚠️" in data['market_status']: st.warning(f"**大盘风控:** {data['market_status']}")
    else: st.success(f"**大盘风控:** {data['market_status']}")
    
    st.info(f"🌐 **宏观情绪与利率：** {data['vix_status_str']} ｜ 🏛️ {data['tnx_status_str']}")
    
    # 多周期共振三维雷达
    st.subheader("🚦 多周期共振雷达")
    st.write(f"- 🧭 **周线大趋势 (中期定性):** {data['weekly_status']}")
    st.write(f"- 🧱 **日线形态 (黄金坑/筑底):** {data['pit_status']}")
    st.write(f"- 🎯 **1小时盘中 (狙击买点):** {data['hourly_status']}")
    
    col_m1, col_m2 = st.columns(2)
    col_m1.metric(label=f"{curr_ticker} 最新价", value=f"${data['cur_price']:.2f}")
    vol_status = "🔥 放量" if data['vol_ratio'] > 1.3 else "🧊 缩量" if data['vol_ratio'] < 0.7 else "⚖️ 平量"
    col_m2.metric(label="5日量比", value=f"{data['vol_ratio']:.2f} 倍", delta=vol_status)

    # Gemini AI 操盘手极简行动指令
    st.subheader("🤖 Gemini 操盘手行动指令 (大道至简)")
    if data['ai_analysis_text']:
        st.markdown(data['ai_analysis_text'])
    else:
        st.info("💡 请配置 Gemini API Key 以解锁 AI 操盘手建议。")

    # 阶梯支撑与动态阻力看板
    st.subheader("🛡️ 阶梯支撑与动态阻力看板")
    col1, col2 = st.columns(2)
    with col1:
        st.info("**【阶梯支撑位】**\n\n" + "\n\n".join(data['support_list']))
    with col2:
        st.warning("**【动态阻力与目标】**\n\n" + "\n\n".join(data['resistance_list']))
    
    # 动能与量价特征
    st.subheader("⚡ 动能与量价特征")
    macd_str = "🟢 多头金叉（动能充沛）" if data['macd_diff_d'] > 0 else "🔴 动能减弱/死叉休整"
    st.write(f"- **MACD 状态:** `{macd_str}` (柱值: {data['macd_diff_d']:.2f})")
    st.write(f"- **RSI (14):** `{data['rsi_d']:.2f}` ({'⚠️ 超买' if data['rsi_d'] > 70 else '💎 极端超卖/黄金坑区' if data['rsi_d'] < 38 else '⚖️ 中性'})")
    st.write(f"- **日均真实波幅 (ATR):** `${data['atr_d']:.2f}`")

    # 5. 专属 AI 操盘助理（追问解惑）
    st.divider()
    st.subheader("💬 对当前诊断有疑问？随时追问 AI 助理")
    st.caption(f"💡 AI 已自动同步 {curr_ticker} 的最新量化数据，你可以直接问仓位配置、挂单技巧或突发情况。")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # 渲染历史对话记录
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 接收用户追问
    if user_question := st.chat_input(f"问问关于 {curr_ticker} 的任何疑问（如：分批挂单/跌破怎么做）..."):
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        if not api_key:
            with st.chat_message("assistant"):
                st.warning("请先配置 Gemini API Key 才能向 AI 助理提问。")
        else:
            with st.chat_message("assistant"):
                with st.spinner("AI 操盘手正在针对当前个股行情为您推演解答..."):
                    context_prompt = f"""
                    你是一名顶级的资深美股操盘手兼新手导师，秉承【大道至简】的教学风格。
                    当前分析标的为: {curr_ticker}，现价: ${data['cur_price']:.2f}。
                    大盘状态: {data['market_status']}，周线趋势: {data['weekly_status']}，日线形态: {data['pit_status']}，1小时狙击: {data['hourly_status']}。
                    阻力位: {'; '.join(data['resistance_list'])}，支撑位: {'; '.join(data['support_list'])}。

                    用户的追问是: "{user_question}"

                    请严格遵守以下要求作答：
                    1. 用通俗易懂的大白话回答，不要堆砌生涩的学术术语。
                    2. 结合上述给出的客观价格与支撑阻力，直接给出明确、可执行的建议（直接给数字和动作），严禁模棱两可。
                    """
                    try:
                        genai.configure(api_key=api_key)
                        chat_model = genai.GenerativeModel('gemini-3.6-flash')
                        chat_resp = chat_model.generate_content(context_prompt)
                        st.markdown(chat_resp.text)
                        st.session_state.chat_history.append({"role": "assistant", "content": chat_resp.text})
                    except Exception as e:
                        st.error(f"回复遇到问题: {e}")
