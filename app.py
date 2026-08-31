import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta, timezone
import time
import re
from ta.trend import EMAIndicator, SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
import google.generativeai as genai

st.set_page_config(page_title="投资小助手 Pro", layout="centered")
st.title("📈 投资小助手 Pro (多周期全维量化版)")
st.caption("⚡ 5分钟全网共享缓存 ｜ 🧭 周线趋势 ｜ 🧱 日线形态 ｜ 🎯 1小时狙击 ｜ 💬 智能FAQ与跨标的追问")

# 1. 获取 API Key (优先读取 Streamlit Secrets)
api_key = st.secrets.get("GEMINI_API_KEY", "") if "GEMINI_API_KEY" in st.secrets else ""

if not api_key:
    with st.expander("🔑 配置 Gemini API Key", expanded=False):
        api_key = st.text_input("Gemini API Key", type="password", help="从 aistudio.google.com 获取")

# 2. 核心量化算法（多周期共振 + 5分钟智能全局共享缓存）
@st.cache_data(ttl=300, show_spinner=False)
def fetch_and_analyze(ticker_input, api_key_val):
    ticker_input = ticker_input.strip().upper()
    
    # 宏观大盘监控 (SPY / QQQ / VIX / TNX)
    macro_tickers = ["SPY", "QQQ", "^VIX", "^TNX"]
    macro_data = yf.download(macro_tickers, period="3mo", interval="1d", progress=False)
    
    market_status = "🟢 顺势顺风：宏观大盘处于多头健康区间。"
    vix_status_str = "正常"
    tnx_status_str = "正常"
    vix_close = 18.0
    
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

    # A. 获取日线数据 (Daily - 形态与阶梯阻力支撑)
    ticker_obj = yf.Ticker(ticker_input)
    df_daily = yf.download(ticker_input, period="1y", interval="1d", progress=False)
    
    if df_daily.empty:
        return None, f"未找到股票 [{ticker_input}] 的数据，请检查代码是否正确。"
        
    if isinstance(df_daily.columns, pd.MultiIndex):
        df_daily.columns = df_daily.columns.get_level_values(0)
    
    close_d = df_daily['Close'].dropna()
    high_d = df_daily['High'].dropna()
    low_d = df_daily['Low'].dropna()
    vol_d = df_daily['Volume'].dropna()
    
    if len(close_d) == 0:
        return None, f"股票 [{ticker_input}] 暂无近期行情数据。"

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

    # 阶梯式动态阻力与多维支撑
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

    # 形态雷达判定
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
    h_rsi = 50.0
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
    
    news_text = ""
    try:
        news_list = ticker_obj.news
        if news_list and len(news_list) > 0:
            for item in news_list[:3]:
                title = item.get('title') or (item.get('content', {}).get('title') if isinstance(item.get('content'), dict) else '')
                if title: news_text += f"- [个股] {title}\n"
    except Exception:
        pass

    # E. 动态匹配生成 8 大实战场景专属 FAQ
    suggested_faqs = []
    if cur_price >= high_30d * 0.985:
        suggested_faqs.append("🚀 创阶段新高，如何设置移动止盈不卖飞？")
    if rsi_d < 38 or "黄金坑" in pit_status:
        suggested_faqs.append("💎 当前属于黄金坑超卖区，新手该分几次建仓？")
    if 0 <= days_to_earnings <= 10:
        suggested_faqs.append(f"⚠️ 距财报仅剩 {days_to_earnings} 天，散户该持股还是避险？")
    if cur_price < ema20:
        suggested_faqs.append("🛑 股价处于 EMA20 均线下方，如果反弹该止损吗？")
    if h_rsi >= 70 or "超买" in hourly_status:
        suggested_faqs.append("🎯 1小时出现短线超买，等回踩哪个价格挂单最安全？")
    if vol_ratio >= 1.4:
        suggested_faqs.append("🔥 今日明显放量，盘中追涨会买在日内最高点吗？")
    if "🔴" in market_status or vix_close > 22:
        suggested_faqs.append("🌐 宏观大盘走弱恐慌，这只股能逆势抗跌独立走强吗？")
    
    suggested_faqs.append(f"💰 我资金量较小，针对 {ticker_input} 怎么执行科学仓位管理？")
    top_faqs = suggested_faqs[:3]

    # F. 双时区时间戳换算 (本地 UTC+8 与 美东 ET)
    now_utc = datetime.now(timezone.utc)
    local_time_str = (now_utc + timedelta(hours=8)).strftime("%H:%M:%S")
    et_time_str = (now_utc - timedelta(hours=4)).strftime("%H:%M:%S") # 夏令时美东 UTC-4
    cache_display_time = f"{local_time_str} 本地 ｜ {et_time_str} 美东"

    # G. Gemini AI 生成与自愈逻辑 (核心防假死重构)
    ai_analysis_text = ""
    if api_key_val:
        genai.configure(api_key=api_key_val)
        model = genai.GenerativeModel('gemini-3.6-flash')
        
        prompt = f"""
        你是一名顶级的资深美股操盘手兼新手导师。你的核心宗旨是【大道至简】。
        请根据以下【周线-日线-1小时多周期共振】指标，为零基础小白写一份极其简明、直白的行动指南。

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

        【排版与逻辑严格要求】：
        1. 严禁出现断裂的星号或错位格式（禁止出现 18.44 * 或 ** 17.59 这种断裂）。所有价格数字必须紧跟美元符号规范加粗，例如 **$18.44**。
        2. 若定性为【不可买/保持空仓】，请严格分为【观望者等待的右侧条件】与【若触发买入后的止盈止损规划】，避免小白产生逻辑困惑。
        3. 直接给数字和执行动作，严禁模棱两可。

        请严格按以下 3 个极简板块输出：
        1. 🚦 **多周期共振定性（红绿灯）**：用 2 句话讲清大趋势是顺风还是逆风？当前是该进攻、该埋伏、还是必须空仓管住手？
        2. 💡 **小白实操动作（直接给数字）**：
           - **买入决策**：清晰说明当前能否买入。若暂不可买，讲清必须满足什么突破条件才可在哪个精确价格挂单。
           - **若触发买入后的止盈规划**：第一目标位与突破加速位分别看至哪里？
           - **铁血止损底线**：一旦介入后，跌破哪个精确价格必须无条件止损离场？
        3. ⚠️ **最核心的一个避险坑**：一句话点透当前最大的单一风险。
        """
        for attempt in range(3):
            try:
                response = model.generate_content(prompt)
                if response and response.text:
                    ai_analysis_text = response.text
                    break
            except Exception as e:
                err_msg = str(e).lower()
                if "429" in err_msg or "quota" in err_msg or "resource_exhausted" in err_msg:
                    time.sleep(2 * (attempt + 1))
                    continue
                else:
                    break

        # 核心防假死：如果 3 次重试后仍限流，绝不缓存错误，直接抛出异常让 Streamlit 稍后自动重连
        if not ai_analysis_text:
            raise Exception("API_RATE_LIMIT")

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
        "top_faqs": top_faqs,
        "ai_analysis_text": ai_analysis_text,
        "cache_display_time": cache_display_time
    }
    return result_bundle, None

# 3. 快速自选栏
if "history_tickers" not in st.session_state:
    st.session_state.history_tickers = ["USAR", "NVDA", "TSLA", "AAPL"]

if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = "USAR"

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
        try:
            data, err = fetch_and_analyze(ticker_input, api_key)
            if err:
                st.error(f"❌ {err}")
            else:
                st.session_state.current_data = data
                st.session_state.current_ticker = ticker_input
                st.session_state.chat_history = []
        except Exception as e:
            if "API_RATE_LIMIT" in str(e):
                st.warning("⏳ **AI 操盘手正在复盘全网数据**：触发了免费 API 临时频率限制，请等待 5~10 秒后再次点击上方按钮！")
            else:
                st.error(f"诊断遇到问题: {e}")

# 渲染分析面板
if "current_data" in st.session_state and st.session_state.current_data:
    data = st.session_state.current_data
    curr_ticker = st.session_state.get("current_ticker", ticker_input)
    
    st.caption(f"⚡ 数据已智能缓存（刷新时间: {data['cache_display_time']} ｜ 5分钟内全员秒开无消耗）")
    
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

    # 5. 专属 AI 操盘助理（智能 FAQ 胶囊 + 跨标的秒调共享缓存）
    st.divider()
    st.subheader("💬 对当前诊断有疑问？随时追问 AI 助理")
    st.caption(f"💡 AI 已自动同步 {curr_ticker} 的最新量化数据，支持一键点击热门实战疑问，或输入其他股票（如 TSLA/AAPL）进行横向对比。")

    # 智能 FAQ 胶囊按钮区域
    clicked_faq = None
    if "top_faqs" in data and data["top_faqs"]:
        st.write("**🔥 该股当下高频实战疑问 (点击一键直答):**")
        faq_cols = st.columns(len(data["top_faqs"]))
        for idx, faq_text in enumerate(data["top_faqs"]):
            if faq_cols[idx].button(faq_text, key=f"faq_{idx}", use_container_width=True):
                clicked_faq = faq_text

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # 渲染历史对话记录
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 统一处理输入（来自底部输入框 或 点击 FAQ 胶囊）
    user_input = st.chat_input(f"问问关于 {curr_ticker} 或对比其他股票（如 TSLA/AAPL）...")
    prompt_to_process = user_input or clicked_faq

    if prompt_to_process:
        st.session_state.chat_history.append({"role": "user", "content": prompt_to_process})
        with st.chat_message("user"):
            st.markdown(prompt_to_process)

        if not api_key:
            with st.chat_message("assistant"):
                st.warning("请先配置 Gemini API Key 才能向 AI 助理提问。")
        else:
            with st.chat_message("assistant"):
                with st.spinner("AI 操盘手正在极速调度共享数据并推演解答..."):
                    words = re.findall(r'\b[A-Z]{2,5}\b', prompt_to_process.upper())
                    extra_data_text = ""
                    
                    for sym in set(words):
                        if sym != curr_ticker:
                            try:
                                other_data, _ = fetch_and_analyze(sym, api_key)
                                if other_data:
                                    extra_data_text += f"""
                                    【提及标的 {sym} 最新量化数据】:
                                    现价: ${other_data['cur_price']:.2f}, 周线趋势: {other_data['weekly_status']}, 日线形态: {other_data['pit_status']}
                                    关键阻力: {'; '.join(other_data['resistance_list'][:2])}, 关键支撑: {'; '.join(other_data['support_list'][:2])}
                                    """
                            except Exception:
                                pass

                    context_prompt = f"""
                    你是一名顶级的资深美股操盘手兼新手导师，秉承【大道至简】的教学风格。
                    当前主标的为: {curr_ticker}，现价: ${data['cur_price']:.2f}。
                    大盘状态: {data['market_status']}，周线趋势: {data['weekly_status']}，日线形态: {data['pit_status']}，1小时狙击: {data['hourly_status']}。
                    主标的阻力位: {'; '.join(data['resistance_list'])}，支撑位: {'; '.join(data['support_list'])}。

                    {extra_data_text}

                    用户的提问是: "{prompt_to_process}"

                    请严格遵守以下要求作答：
                    1. 严禁出现断裂的星号或排版错位，所有关键价格统一规范加粗（如 **$18.04**）。
                    2. 用通俗易懂的大白话回答，不要堆砌生涩的学术术语。
                    3. 如果用户对比了两只股票，请结合上述客观数据直接给出优劣排序与清晰的买卖建议。
                    4. 直接给出数字和具体执行动作，严禁模棱两可。
                    """
                    try:
                        genai.configure(api_key=api_key)
                        chat_model = genai.GenerativeModel('gemini-3.6-flash')
                        chat_resp = chat_model.generate_content(context_prompt)
                        st.markdown(chat_resp.text)
                        st.session_state.chat_history.append({"role": "assistant", "content": chat_resp.text})
                    except Exception as e:
                        st.error(f"回复遇到问题: {e}")
