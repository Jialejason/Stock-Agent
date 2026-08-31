import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime
from ta.trend import EMAIndicator, SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
import google.generativeai as genai

st.set_page_config(page_title="投资小助手 Pro", layout="centered")
st.title("📈 投资小助手 Pro (量化风控全维版)")
st.caption("⚡ 5分钟智能缓存共享 ｜ 宏观/VIX情绪 ｜ 阶梯阻力/黄金坑 ｜ 多引擎自动容灾")

# 1. 获取 API Key
api_key = st.secrets.get("GEMINI_API_KEY", "") if "GEMINI_API_KEY" in st.secrets else ""

if not api_key:
    with st.expander("🔑 配置 Gemini API Key", expanded=False):
        api_key = st.text_input("Gemini API Key", type="password", help="从 aistudio.google.com 获取")

# 2. 核心数据与 AI 分析智能缓存函数 (5分钟 TTL，全员共享)
@st.cache_data(ttl=300, show_spinner=False)
def fetch_and_analyze(ticker_input, api_key_val):
    # 宏观监控
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

    # 个股量化
    ticker_obj = yf.Ticker(ticker_input)
    df = yf.download(ticker_input, period="1y", interval="1d", progress=False)
    
    if df.empty:
        return None, "未找到该股票数据，请检查代码是否正确。"
        
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    close = df['Close'].dropna()
    high = df['High'].dropna()
    low = df['Low'].dropna()
    volume = df['Volume'].dropna()
    
    cur_price = close.iloc[-1]
    total_days = len(close)
    
    ema5 = EMAIndicator(close, min(5, total_days)).ema_indicator().iloc[-1]
    ema10 = EMAIndicator(close, min(10, total_days)).ema_indicator().iloc[-1]
    ema20 = EMAIndicator(close, min(20, total_days)).ema_indicator().iloc[-1]
    has_ma60 = total_days >= 60
    ma60 = SMAIndicator(close, 60).sma_indicator().iloc[-1] if has_ma60 else ema20
    ma60_str = f"${ma60:.2f}" if has_ma60 else "上市未满60日"
    
    rsi_series = RSIIndicator(close, min(14, total_days)).rsi()
    rsi = rsi_series.iloc[-1]
    macd_obj = MACD(close)
    macd_val = macd_obj.macd().iloc[-1]
    macd_signal = macd_obj.macd_signal().iloc[-1]
    macd_diff = macd_obj.macd_diff().iloc[-1]
    atr = AverageTrueRange(high, low, close, min(14, total_days)).average_true_range().iloc[-1]
    
    cur_vol = volume.iloc[-1]
    avg_vol_5d = volume.iloc[-6:-1].mean() if total_days >= 6 else volume.mean()
    vol_ratio = (cur_vol / avg_vol_5d) if avg_vol_5d > 0 else 1.0

    # 阶梯阻力与支撑
    high_30d = high.iloc[-min(30, total_days):].max()
    high_120d = high.iloc[-min(120, total_days):].max() if total_days >= 30 else high_30d
    high_52w = high.max()
    low_30d = low.iloc[-min(30, total_days):].min()
    
    resistance_list = []
    if cur_price >= high_30d * 0.99:
        if high_120d > cur_price * 1.01:
            resistance_list.append(f"🔥 突破30日高点！下一阻力锁定【半年高点】: ${high_120d:.2f}")
        elif high_52w > cur_price * 1.01:
            resistance_list.append(f"🔥 突破阶段平台！下一阻力锁定【52周历史大顶】: ${high_52w:.2f}")
        else:
            ath_target = cur_price + (1.5 * atr)
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

    # 形态雷达
    pit_status = "正常走势"
    if rsi < 38 and cur_price <= low_30d * 1.03:
        pit_status = "💎 极端超卖黄金坑：指标极度冰点超跌，存在高盈亏比反弹反转机会！"
    elif cur_price >= ema20 and (macd_diff > 0 or macd_val > macd_signal) and vol_ratio >= 0.9:
        pit_status = "🧱 右侧企稳确立：股价重回EMA20均线之上，动能回暖，企稳结构扎实。"
    elif cur_price < ema20 and vol_ratio < 0.7:
        pit_status = "🧊 缩量磨底中：跌破均线但抛压衰竭，等待放量企稳确认。"

    # 财报与新闻
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

    # Gemini 自动轮询容灾生成
    ai_analysis_text = ""
    if api_key_val:
        genai.configure(api_key=api_key_val)
        prompt = f"""
        你是一名资深的职业美股操盘手兼新手导师。请结合宏观大盘、美债利率/恐慌情绪、阶梯阻力支撑，以及当前算法识别出的【形态雷达】状态，用通俗易懂的大白话为新手写一份诊断指南。

        【股票标的】: {ticker_input}
        【最新价格】: ${cur_price:.2f}
        【大盘宏观 (SPY/QQQ)】: {market_status}
        【市场情绪与利率 (VIX/TNX)】: {vix_status_str} ｜ {tnx_status_str}
        【形态雷达状态】: {pit_status}
        【均线与量能】: EMA5=${ema5:.2f}, EMA10=${ema10:.2f}, EMA20=${ema20:.2f}, MA60={ma60_str}, 5日量比={vol_ratio:.2f}倍, RSI={rsi:.2f}, MACD柱值={macd_diff:.2f}, ATR=${atr:.2f}
        【动态阻力位】: {'; '.join(resistance_list)}
        【阶梯支撑位】: {'; '.join(support_list)}
        【近期财报日】: {earnings_date_str}
        【突发资讯】:
        {news_text if news_text else "暂无突发新闻"}

        请直接按以下 3 个结构输出（言简意赅，指令明确）：
        1. 🧐 **形态与黄金坑定性**：用 2 句话讲清当前股票处于强势拉升、还是砸出了黄金坑/右侧企稳、亦或处于危险破位期？
        2. 💡 **新手实操动作**：明确告诉新手目前能不能买？如果想买该挂在什么具体价位？突破阻力后下一目标看到哪里？如果在场内跌破哪个价位必须立刻止损认错？
        3. ⚠️ **核心避险警示**：结合大盘、财报倒计时或缩量/放量异动，提醒新手当前最需要防范的坑是什么？
        """
        candidate_models = ['gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-1.5-flash']
        for model_name in candidate_models:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                ai_analysis_text = response.text
                break
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    continue
                else:
                    ai_analysis_text = f"诊断中断: {e}"
                    break
        if not ai_analysis_text:
            ai_analysis_text = "⏳ AI 导师正在高速复盘中，触发了临时调用保护，请稍后重试！"

    # 打包所有计算结果
    result_bundle = {
        "market_status": market_status,
        "vix_status_str": vix_status_str,
        "tnx_status_str": tnx_status_str,
        "pit_status": pit_status,
        "cur_price": cur_price,
        "vol_ratio": vol_ratio,
        "support_list": support_list,
        "resistance_list": resistance_list,
        "macd_val": macd_val,
        "macd_signal": macd_signal,
        "macd_diff": macd_diff,
        "rsi": rsi,
        "atr": atr,
        "ai_analysis_text": ai_analysis_text,
        "cache_time": datetime.now().strftime("%H:%M:%S")
    }
    return result_bundle, None

# 3. 动态自选栏
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

# 4. 诊断展示
if st.button("开始全维度深度诊断", type="primary", use_container_width=True):
    if ticker_input and ticker_input in st.session_state.history_tickers:
        st.session_state.history_tickers.remove(ticker_input)
    if ticker_input:
        st.session_state.history_tickers.insert(0, ticker_input)
        if len(st.session_state.history_tickers) > 5:
            st.session_state.history_tickers.pop()

    with st.spinner(f"正在全维诊断与调取数据 ({ticker_input})..."):
        data, err = fetch_and_analyze(ticker_input, api_key)
        
        if err:
            st.error(f"❌ {err}")
        elif data:
            st.caption(f"⚡ 数据已智能缓存（最后刷新时间: {data['cache_time']}，5分钟内全员秒开无消耗）")
            
            # 宏观看板
            if "🔴" in data['market_status']: st.error(f"**大盘风控:** {data['market_status']}")
            elif "⚠️" in data['market_status']: st.warning(f"**大盘风控:** {data['market_status']}")
            else: st.success(f"**大盘风控:** {data['market_status']}")
            
            st.info(f"🌐 **宏观全维监控：** {data['vix_status_str']} ｜ 🏛️ {data['tnx_status_str']}")
            
            if "💎" in data['pit_status'] or "🧱" in data['pit_status']:
                st.success(f"🎯 **形态雷达:** {data['pit_status']}")
            else:
                st.warning(f"🎯 **形态雷达:** {data['pit_status']}")
                
            col_m1, col_m2 = st.columns(2)
            col_m1.metric(label=f"{ticker_input} 最新价", value=f"${data['cur_price']:.2f}")
            vol_status = "🔥 放量" if data['vol_ratio'] > 1.3 else "🧊 缩量" if data['vol_ratio'] < 0.7 else "⚖️ 平量"
            col_m2.metric(label="5日量比", value=f"{data['vol_ratio']:.2f} 倍", delta=vol_status)

            # Gemini AI 结果展示
            st.subheader("🤖 Gemini 操盘手大白话解读")
            if data['ai_analysis_text']:
                st.markdown(data['ai_analysis_text'])
            else:
                st.info("💡 请配置 Gemini API Key 以解锁 AI 操盘手建议。")

            # 阶梯支撑阻力看板
            st.subheader("🛡️ 阶梯支撑与动态阻力看板")
            col1, col2 = st.columns(2)
            with col1:
                st.info("**【阶梯支撑位】**\n\n" + "\n\n".join(data['support_list']))
            with col2:
                st.warning("**【动态阻力与目标】**\n\n" + "\n\n".join(data['resistance_list']))
            
            # 动能与量价特征
            st.subheader("⚡ 动能与量价特征")
            macd_str = "🟢 多头金叉（动能充沛）" if data['macd_val'] > data['macd_signal'] and data['macd_diff'] > 0 else "🔴 动能减弱/死叉休整"
            st.write(f"- **MACD 状态:** `{macd_str}` (柱值: {data['macd_diff']:.2f})")
            st.write(f"- **RSI (14):** `{data['rsi']:.2f}` ({'⚠️ 超买' if data['rsi'] > 70 else '💎 极端超卖/黄金坑区' if data['rsi'] < 38 else '⚖️ 中性'})")
            st.write(f"- **日均真实波幅 (ATR):** `${data['atr']:.2f}`")
