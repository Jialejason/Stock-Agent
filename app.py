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
st.caption("⚡ 5分钟全网共享缓存 ｜ 🧭 周线趋势 ｜ 🧱 日线形态 ｜ 🎯 1小时狙击 ｜ 💬 智能深度自主追问")

# 1. 常用中英文名称到美股代码的别名映射字典
TICKER_ALIASES = {
    "TESLA": "TSLA", "特斯拉": "TSLA",
    "APPLE": "AAPL", "苹果": "AAPL",
    "NVIDIA": "NVDA", "英伟达": "NVDA",
    "GOOGLE": "GOOGL", "谷歌": "GOOGL",
    "AMAZON": "AMZN", "亚马逊": "AMZN",
    "MICROSOFT": "MSFT", "微软": "MSFT",
    "META": "META", "脸书": "META",
    "SPACEX": "SPCX",
    "AMD": "AMD", "超微": "AMD"
}

# 安全渲染 Markdown
def safe_render_markdown(text):
    if not text:
        return
    clean_text = text.replace("$", "\\$")
    st.markdown(clean_text)

# 提取文本中的美股标的代码
def extract_tickers_from_text(input_text):
    text_upper = input_text.upper()
    found_symbols = set()
    for name, sym in TICKER_ALIASES.items():
        if name in text_upper or name in input_text:
            found_symbols.add(sym)
    words = re.findall(r'\b[A-Z]{2,5}\b', text_upper)
    for w in words:
        found_symbols.add(w)
    return found_symbols

# 获取 API Key
api_key = st.secrets.get("GEMINI_API_KEY", "") if "GEMINI_API_KEY" in st.secrets else ""

if not api_key:
    with st.expander("🔑 配置 Gemini API Key", expanded=False):
        api_key = st.text_input("Gemini API Key", type="password", help="从 aistudio.google.com 获取")

# 辅助函数：AI 操盘指令生成（带深度规则兜底）
def get_ai_analysis(ticker_input, cur_price, market_status, vix_status_str, tnx_status_str, 
                    weekly_status, pit_status, hourly_status, hourly_suggested_entry, 
                    hourly_stop_loss, resistance_list, support_list, earnings_date_str, 
                    news_text, high_30d, ema20, api_key_val):
    
    prompt = f"""
    你是一名顶级的美股操盘手与量化投资导师。你的核心风格是【大道至简、精准果断、逻辑通透】。
    请基于以下【周线-日线-1小时多周期共振】量化客观数据，为新手交易者推演行动指令：

    【股票标的】: {ticker_input}
    【最新现价】: ${cur_price:.2f}
    【宏观大盘 (SPY/QQQ)】: {market_status}
    【市场情绪与利率】: {vix_status_str} ｜ {tnx_status_str}
    【🧭 周线大趋势】: {weekly_status}
    【🧱 日线形态雷达】: {pit_status}
    【🎯 1小时盘中狙击】: {hourly_status} (盘中参考挂单: ${hourly_suggested_entry:.2f}, 1小时防守: ${hourly_stop_loss:.2f})
    【阶梯阻力与目标】: {'; '.join(resistance_list)}
    【阶梯防守支撑位】: {'; '.join(support_list)}
    【财报倒计时】: {earnings_date_str}
    【突发资讯】:
    {news_text if news_text else "暂无突发新闻"}

    【输出格式与逻辑规范】：
    1. 所有价格数字必须紧跟美元符号规范加粗，例如 **$18.44**，禁止散落星号。
    2. 若定性为不可买，清晰写出必须满足的右侧突破条件。

    请严格按以下 3 个板块输出：
    1. 🚦 **多周期共振定性（红绿灯）**：2句话讲清顺风还是逆风？攻还是守？
    2. 💡 **小白实操动作（精准点位）**：
       - **买入决策**：当前能否买入？若观望，需在什么价格突破并站稳后挂单？
       - **止盈规划**：第一目标位与突破加速位分别看至哪里？
       - **铁血止损**：跌破哪个精确价格必须离场？
    3. ⚠️ **最核心的一个避险坑**：一句话点透最大风险。
    """
    
    if api_key_val:
        genai.configure(api_key=api_key_val)
        models_to_try = ['gemini-1.5-flash', 'gemini-2.0-flash', 'gemini-1.5-pro']
        for m_name in models_to_try:
            try:
                model = genai.GenerativeModel(m_name)
                res = model.generate_content(prompt)
                if res and res.text:
                    return res.text
            except Exception:
                continue

    is_bear = cur_price < ema20 or "逆风" in weekly_status
    decision = "🔴 **暂不可买，严格保持空仓！**" if is_bear else "🟢 **右侧多头共振，允许小仓位试错！**"
    entry_note = f"必须等待日K线放量突破短线压制 **${high_30d:.2f}** 并在回踩不破时，于 **${hourly_suggested_entry:.2f}** 附近挂单。" if is_bear else f"可在回踩支撑 **${hourly_suggested_entry:.2f}** 附近分批介入。"
    
    return f"""
1. 🚦 **多周期共振定性（红绿灯）**：
当前周线与日线均处于整固防守阶段，大趋势向下未形成多头合力，逆势抄底容易被套，严格以防守为主。

2. 💡 **小白实操动作（直接给数字）**：
- **买入决策**：{decision}
  - **入场条件**：{entry_note}
- **若触发买入后的止盈规划**：
  - **第一目标位**：触及 **${high_30d:.2f}** 先减仓 1/3 锁定利润。
  - **突破加速位**：站稳阻力后看至更高筹码密集平台。
- **铁血止损底线**：一旦介入，跌破防守线 **${hourly_stop_loss:.2f}** 无条件止损！

3. ⚠️ **最核心的一个避险坑**：均线空头排列时切忌补仓摊平成本，谨防阴跌寻底。
"""

# 辅助函数：AI 问答高度自主化保底推演引擎
def fallback_smart_chat(prompt_text, curr_ticker, cur_price, data, compared_ticker_data=None):
    res_top = data['resistance_list'][0] if data['resistance_list'] else "上方阻力平台"
    sup_top = data['support_list'][0] if data['support_list'] else f"关键防守位 ${data['hourly_stop_loss']:.2f}"
    
    if compared_ticker_data:
        comp_sym = compared_ticker_data['symbol']
        comp_price = compared_ticker_data['cur_price']
        comp_weekly = compared_ticker_data['weekly_status']
        comp_pit = compared_ticker_data['pit_status']
        
        return f"""
针对 **{curr_ticker}**（现价 **${cur_price:.2f}**）与 **{comp_sym}**（现价 **${comp_price:.2f}**）的深度量化对比：

1. 🧭 **多周期趋势定性**：
   - **{curr_ticker}**：周线 `{data['weekly_status']}`，日线形态 `{data['pit_status']}`。
   - **{comp_sym}**：周线 `{comp_weekly}`，日线形态 `{comp_pit}`。
2. ⚖️ **强弱排序与确定性**：
   - 当前资金面更青睐右侧站稳均线的标的。若 **{comp_sym}** 结构更完整，确定性高于仍处均线下方的 **{curr_ticker}**。
3. 🎯 **小白执行指令**：在大盘震荡期，严禁分仓去赌弱势股反弹，优先关注右侧放量企稳的龙头品种。
"""

    if "企稳" in prompt_text or "什么价位" in prompt_text:
        rebound_price = data['hourly_suggested_entry'] * 1.02
        return f"""
关于 **{curr_ticker}** 判定【真正企稳】的量化点位与标准：

1. 🧱 **第一企稳信号（突破均线压制）**：
   股价必须收复当前日线短线压制 **{res_top}**。只有日K线实体站稳该价格上方，才算止住短线单边阴跌。
2. 🔒 **右侧企稳确认（回踩不破）**：
   突破后需伴随量比 > **1.0 倍**，且回踩 **${cur_price:.2f}** ~ **${rebound_price:.2f}** 平台不破，此时构成安全的右侧进场信号。
3. 🛑 **假企稳警示**：若盘中脉冲冲高但收盘仍被均线压制，属于诱多反抽，切勿急于追入。
"""
    elif "买卖" in prompt_text or "建议" in prompt_text or "买" in prompt_text or "卖" in prompt_text:
        return f"""
**{curr_ticker}**（现价 **${cur_price:.2f}**）当下最清晰的买卖执行方案：

- 🚦 **买入条件**：目前不满足右侧安全买点。必须等待放量突破 **{res_top}** 并站稳后挂单；
- 🎯 **止盈目标**：若持有底仓，反弹触及 **{res_top}** 附近必须逢高减仓；
- 🛡️ **止损防线**：下破防守位 **${data['hourly_stop_loss']:.2f}**（或 **{sup_top}**）无条件清仓观望。
"""
    elif "止损" in prompt_text or "EMA20" in prompt_text:
        return f"""
**{curr_ticker}** 处于 EMA20 下方的风控逻辑：

1. **破位性质**：均线下方属于空头占优区，反弹通常是【减仓防守】的机会。
2. **操作动作**：若反弹至 **{res_top}** 遇阻，建议果断减仓；跌破 **${data['hourly_stop_loss']:.2f}** 坚决执行止损。
"""
    elif "大盘" in prompt_text or "独立" in prompt_text:
        return f"""
**{curr_ticker}** 与大盘走势的联动评估：

1. **大盘环境**：{data['market_status']}
2. **抗跌性分析**：**{curr_ticker}** 当前量比为 **{data['vol_ratio']:.2f} 倍**，没有明显增量资金入场托底前，单只股票很难持续逆势抗跌。
"""
    else:
        return f"""
针对关于 **{curr_ticker}** 的量化分析解答：

- **中期趋势**：{data['weekly_status']}
- **短线压制**：{res_top}
- **核心支撑**：{sup_top}
- **战术指引**：在周线与日线未形成多头共振前，保持耐心，宁可等右侧买点，不盲目左侧猜底。
"""

# 2. 核心量化算法（带 5 分钟共享缓存）
@st.cache_data(ttl=300, show_spinner=False)
def fetch_and_analyze(ticker_input, api_key_val):
    ticker_input = ticker_input.strip().upper()
    
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
    ma60 = SMAIndicator(close_d, 60).sma_indicator().iloc[-1] if has_ma60 else None
    
    rsi_d = RSIIndicator(close_d, min(14, total_days)).rsi().iloc[-1]
    macd_obj_d = MACD(close_d)
    macd_diff_d = macd_obj_d.macd_diff().iloc[-1]
    atr_d = AverageTrueRange(high_d, low_d, close_d, min(14, total_days)).average_true_range().iloc[-1]
    
    cur_vol = vol_d.iloc[-1]
    avg_vol_5d = vol_d.iloc[-6:-1].mean() if total_days >= 6 else vol_d.mean()
    vol_ratio = (cur_vol / avg_vol_5d) if avg_vol_5d > 0 else 1.0

    high_30d = high_d.iloc[-min(30, total_days):].max()
    high_120d = high_d.iloc[-min(120, total_days):].max() if total_days >= 30 else high_30d
    high_52w = high_d.max()
    low_30d = low_d.iloc[-min(30, total_days):].min()
    
    resistance_list = []
    support_list = []

    if ema5 > cur_price: resistance_list.append(f"短线压制 (EMA5): ${ema5:.2f}")
    else: support_list.append(f"超短支撑 (EMA5): ${ema5:.2f}")

    if ema10 > cur_price: resistance_list.append(f"过渡均线压制 (EMA10): ${ema10:.2f}")
    else: support_list.append(f"过渡防守 (EMA10): ${ema10:.2f}")

    if ema20 > cur_price: resistance_list.append(f"多空分水岭压制 (EMA20): ${ema20:.2f}")
    else: support_list.append(f"多空分水岭支撑 (EMA20): ${ema20:.2f}")

    if ma60:
        if ma60 > cur_price: resistance_list.append(f"中期生命线压制 (MA60): ${ma60:.2f}")
        else: support_list.append(f"中期生命线支撑 (MA60): ${ma60:.2f}")

    if cur_price >= high_30d * 0.99:
        if high_120d > cur_price * 1.01:
            resistance_list.append(f"🔥 突破30日高点！下一阻力锁定【半年高点】: ${high_120d:.2f}")
        elif high_52w > cur_price * 1.01:
            resistance_list.append(f"🔥 突破阶段平台！下一阻力锁定【52周历史大顶】: ${high_52w:.2f}")
        else:
            ath_target = cur_price + (1.5 * atr_d)
            resistance_list.append(f"🚀 创历史新高！动能拓展目标位: ${ath_target:.2f}")
    else:
        resistance_list.append(f"30日阶段强阻力: ${high_30d:.2f}")
        if high_120d > high_30d:
            resistance_list.append(f"半年期重要阻力: ${high_120d:.2f}")
    
    if low_30d < cur_price:
        support_list.append(f"30日筑底强支撑: ${low_30d:.2f}")

    if not support_list:
        support_list.append(f"近期防守底线: ${low_30d:.2f}")

    pit_status = "正常走势"
    if rsi_d < 38 and cur_price <= low_30d * 1.03:
        pit_status = "💎 极端超卖黄金坑：指标极度冰点超跌，存在高盈亏比反弹反转机会！"
    elif cur_price >= ema20 and macd_diff_d > 0 and vol_ratio >= 0.9:
        pit_status = "🧱 右侧企稳确立：股价重回EMA20均线之上，动能回暖，企稳结构扎实。"
    elif cur_price < ema20 and vol_ratio < 0.7:
        pit_status = "🧊 缩量磨底中：跌破均线但抛压衰竭，等待放量企稳确认。"

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
                hourly_stop_loss = min(h_recent_low * 0.995, cur_price * 0.98)
                
                if cur_price >= h_ema20 and 45 <= h_rsi <= 65:
                    hourly_status = "🎯 盘中狙击买点已触发：1小时结构回踩企稳，极具盈亏比！"
                elif h_rsi >= 70:
                    hourly_status = "⚠️ 1小时盘中超买：短线急拉，切勿追高，等待盘中回踩挂单！"
                elif cur_price < h_ema20:
                    hourly_status = "🧊 1小时盘中走弱：等待盘中重回1小时EMA20均线后再挂单。"
    except Exception:
        pass

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

    now_utc = datetime.now(timezone.utc)
    local_time_str = (now_utc + timedelta(hours=8)).strftime("%H:%M:%S")
    et_time_str = (now_utc - timedelta(hours=4)).strftime("%H:%M:%S")
    cache_display_time = f"{local_time_str} 本地 ｜ {et_time_str} 美东"

    ai_analysis_text = get_ai_analysis(ticker_input, cur_price, market_status, vix_status_str, 
                                       tnx_status_str, weekly_status, pit_status, hourly_status, 
                                       hourly_suggested_entry, hourly_stop_loss, resistance_list, 
                                       support_list, earnings_date_str, news_text, high_30d, ema20, api_key_val)

    result_bundle = {
        "symbol": ticker_input,
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
        data, err = fetch_and_analyze(ticker_input, api_key)
        if err:
            st.error(f"❌ {err}")
        else:
            st.session_state.current_data = data
            st.session_state.current_ticker = ticker_input
            st.session_state.chat_history = []

# 渲染分析面板
if "current_data" in st.session_state and st.session_state.current_data:
    data = st.session_state.current_data
    curr_ticker = st.session_state.get("current_ticker", ticker_input)
    
    st.caption(f"⚡ 数据已智能缓存（刷新时间: {data['cache_display_time']} ｜ 5分钟内全员秒开无消耗）")
    
    if "🔴" in data['market_status']: st.error(f"**大盘风控:** {data['market_status']}")
    elif "⚠️" in data['market_status']: st.warning(f"**大盘风控:** {data['market_status']}")
    else: st.success(f"**大盘风控:** {data['market_status']}")
    
    st.info(f"🌐 **宏观情绪与利率：** {data['vix_status_str']} ｜ 🏛️ {data['tnx_status_str']}")
    
    st.subheader("🚦 多周期共振雷达")
    st.write(f"- 🧭 **周线大趋势 (中期定性):** {data['weekly_status']}")
    st.write(f"- 🧱 **日线形态 (黄金坑/筑底):** {data['pit_status']}")
    st.write(f"- 🎯 **1小时盘中 (狙击买点):** {data['hourly_status']}")
    
    col_m1, col_m2 = st.columns(2)
    col_m1.metric(label=f"{curr_ticker} 最新价", value=f"${data['cur_price']:.2f}")
    vol_status = "🔥 放量" if data['vol_ratio'] > 1.3 else "🧊 缩量" if data['vol_ratio'] < 0.7 else "⚖️ 平量"
    col_m2.metric(label="5日量比", value=f"{data['vol_ratio']:.2f} 倍", delta=vol_status)

    st.subheader("🤖 Gemini 操盘手行动指令 (大道至简)")
    safe_render_markdown(data['ai_analysis_text'])

    st.subheader("🛡️ 阶梯支撑与动态阻力看板")
    col1, col2 = st.columns(2)
    with col1:
        st.info("**【阶梯支撑位】**\n\n" + "\n\n".join(data['support_list']))
    with col2:
        st.warning("**【动态阻力与目标】**\n\n" + "\n\n".join(data['resistance_list']))
    
    st.subheader("⚡ 动能与量价特征")
    macd_str = "🟢 多头金叉（动能充沛）" if data['macd_diff_d'] > 0 else "🔴 动能减弱/死叉休整"
    st.write(f"- **MACD 状态:** `{macd_str}` (柱值: {data['macd_diff_d']:.2f})")
    st.write(f"- **RSI (14):** `{data['rsi_d']:.2f}` ({'⚠️ 超买' if data['rsi_d'] > 70 else '💎 极端超卖/黄金坑区' if data['rsi_d'] < 38 else '⚖️ 中性'})")
    st.write(f"- **日均真实波幅 (ATR):** `${data['atr_d']:.2f}`")

    # 5. 专属 AI 操盘助理（高度聪明自主化推演）
    st.divider()
    st.subheader("💬 对当前诊断有疑问？随时追问 AI 助理")
    st.caption(f"💡 AI 已深度挂载 {curr_ticker} 的全部量化数据，支持深度自由追问、实战推演或跨标的横向对比。")

    clicked_faq = None
    if "top_faqs" in data and data["top_faqs"]:
        st.write("**🔥 该股当下高频实战疑问 (点击一键直答):**")
        faq_cols = st.columns(len(data["top_faqs"]))
        for idx, faq_text in enumerate(data["top_faqs"]):
            if faq_cols[idx].button(faq_text, key=f"faq_{idx}", use_container_width=True):
                clicked_faq = faq_text

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            safe_render_markdown(msg["content"])

    user_input = st.chat_input(f"问问关于 {curr_ticker}（如企稳价位、买卖点）或对比其他股票...")
    prompt_to_process = user_input or clicked_faq

    if prompt_to_process:
        st.session_state.chat_history.append({"role": "user", "content": prompt_to_process})
        with st.chat_message("user"):
            safe_render_markdown(prompt_to_process)

        with st.chat_message("assistant"):
            with st.spinner("AI 操盘手正在针对具体问题深度推演解答..."):
                reply_text = ""
                extracted_symbols = extract_tickers_from_text(prompt_to_process)
                compared_ticker_data = None
                extra_data_text = ""
                
                for sym in extracted_symbols:
                    if sym != curr_ticker:
                        try:
                            other_data, _ = fetch_and_analyze(sym, api_key)
                            if other_data:
                                compared_ticker_data = other_data
                                extra_data_text += f"""
                                【提及标的 {sym} 最新量化数据】:
                                现价: ${other_data['cur_price']:.2f}, 周线趋势: {other_data['weekly_status']}, 日线形态: {other_data['pit_status']}
                                关键阻力: {'; '.join(other_data['resistance_list'][:2])}, 关键支撑: {'; '.join(other_data['support_list'][:2])}
                                """
                        except Exception:
                            pass

                # 优先调用真 AI 进行针对性深度推演
                if api_key:
                    genai.configure(api_key=api_key)
                    models_to_try = ['gemini-1.5-flash', 'gemini-2.0-flash', 'gemini-1.5-pro']
                    
                    context_prompt = f"""
                    你是一名顶级的资深美股操盘手与量化交易导师。你的核心优势是【能够针对用户的具体问题进行深度、精准、量身定制的推演】。
                    
                    【当前标的】: {curr_ticker}，现价: ${data['cur_price']:.2f}
                    【宏观大盘】: {data['market_status']}
                    【周线大趋势】: {data['weekly_status']}
                    【日线形态】: {data['pit_status']}
                    【1小时盘中】: {data['hourly_status']} (参考买点: ${data['hourly_suggested_entry']:.2f}, 1小时防守: ${data['hourly_stop_loss']:.2f})
                    【动态阻力与目标】: {'; '.join(data['resistance_list'])}
                    【阶梯防守支撑】: {'; '.join(data['support_list'])}
                    【ATR日均波幅】: ${data['atr_d']:.2f} ｜ 量比: {data['vol_ratio']:.2f}倍
                    {extra_data_text}

                    用户的具体提问是: "{prompt_to_process}"

                    【严格作答要求】：
                    1. 严禁使用笼统模版！必须正面、深度、针对性地回答用户的具体问题（例如用户问“什么价位才算企稳”，就必须结合上方阻力位计算出具体的企稳突破价格与确认条件）。
                    2. 所有价格数字必须紧跟美元符号规范加粗（例如 **$18.40**）。
                    3. 直接给点位、逻辑与执行动作，语言精炼、直白、专业。
                    """
                    for m_name in models_to_try:
                        try:
                            chat_model = genai.GenerativeModel(m_name)
                            chat_resp = chat_model.generate_content(context_prompt)
                            if chat_resp and chat_resp.text:
                                reply_text = chat_resp.text
                                break
                        except Exception:
                            time.sleep(0.5)
                            continue

                # 本地多意图智能保底（彻底消除内容重复）
                if not reply_text:
                    reply_text = fallback_smart_chat(prompt_to_process, curr_ticker, data['cur_price'], data, compared_ticker_data)

                safe_render_markdown(reply_text)
                st.session_state.chat_history.append({"role": "assistant", "content": reply_text})
