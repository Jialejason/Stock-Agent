import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import re
from ta.trend import EMAIndicator, SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
import google.generativeai as genai

st.set_page_config(page_title="投资小助手 Pro", layout="centered")
st.title("📈 投资小助手 Pro (全维智能实战版)")
st.caption("⚡ 5分钟全网共享缓存 ｜ 🧠 Gemini 原生智脑 ｜ 🕳️ 缺口/均线共振 ｜ 📰 实时新闻舆情 ｜ ⚖️ 盈亏比大白话")

# 1. 别名映射与 Markdown 安全渲染
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

def safe_render_markdown(text):
    if not text:
        return
    clean_text = text.replace("$", "\\$")
    st.markdown(clean_text)

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

# 动态安全获取 API Key
raw_api_key = st.secrets.get("GEMINI_API_KEY", "") if "GEMINI_API_KEY" in st.secrets else ""
api_key = raw_api_key.strip().replace("\n", "").replace("\r", "").replace(" ", "")

# 通用智能调用 Gemini 函数（自动动态寻找可用模型）
def call_gemini_smart(prompt_text):
    if not api_key:
        return "⚠️ 未检测到 API Key，请在 Streamlit Secrets 中配置 `GEMINI_API_KEY`。"
    
    try:
        genai.configure(api_key=api_key)
        
        # 1. 优先尝试最常用的活跃模型
        candidate_models = ['gemini-2.0-flash', 'gemini-2.5-flash', 'gemini-1.5-flash-latest', 'gemini-1.5-pro-latest']
        for m_name in candidate_models:
            try:
                model = genai.GenerativeModel(m_name)
                res = model.generate_content(prompt_text)
                if res and res.text:
                    return res.text
            except Exception:
                continue

        # 2. 如果上面的候选名称变动，自动列出该账号下所有支持 generateContent 的可用模型
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                try:
                    model = genai.GenerativeModel(m.name)
                    res = model.generate_content(prompt_text)
                    if res and res.text:
                        return res.text
                except Exception:
                    continue

        return "⚠️ 当前 API Key 暂无可用的 Gemini 生成模型，请确认 Google AI Studio 权限。"
    except Exception as e:
        return f"⚠️ 智脑调用异常: `{e}`"

# 全景真实复权筹码分布计算 (VPVR)
def calculate_volume_profile(df_daily, bins=25):
    price_min = df_daily['Low'].min()
    price_max = df_daily['High'].max()
    if price_max <= price_min or np.isnan(price_min) or np.isnan(price_max):
        return []
    
    bin_edges = np.linspace(price_min, price_max, bins + 1)
    vol_profile = np.zeros(bins)
    
    for _, row in df_daily.iterrows():
        mid_p = (row['High'] + row['Low'] + row['Close']) / 3.0
        b_idx = int(np.digitize(mid_p, bin_edges) - 1)
        b_idx = max(0, min(bins - 1, b_idx))
        vol_profile[b_idx] += row['Volume']
        
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    return list(zip(bin_centers, vol_profile))

# AI 生成全维实战行动手册
def get_ai_analysis(ticker_input, cur_price, market_status, vix_status_str, tnx_status_str, 
                    macro_sentiment_tag, weekly_status, pit_status, hourly_status, vwap_price, 
                    vwap_status_desc, hourly_suggested_entry, hourly_stop_loss, chip_resistances, 
                    chip_supports, gap_support, prev_close_p, earnings_date_str, days_to_earnings, 
                    news_items, high_30d, high_52w, ema20, ma30, ma60_str, ma120_str, ma250_str, rr_ratio):
    
    chip_res_str = " ➔ ".join([f"${p:.2f}" for p in chip_resistances]) if chip_resistances else f"${high_30d:.2f}"
    chip_sup_str = "、".join([f"${p:.2f}" for p in chip_supports]) if chip_supports else f"${hourly_suggested_entry:.2f}"
    gap_sup_str = f"${gap_support:.2f}" if gap_support else f"${ema20:.2f}"

    earnings_warning = ""
    if 0 <= days_to_earnings <= 7:
        earnings_warning = f"⚠️ 警告：距财报发布仅剩 {days_to_earnings} 天，技术形态极易失真，严禁重仓赌财报！"

    news_digest = ""
    if news_items:
        for idx, n in enumerate(news_items[:4]):
            news_digest += f"{idx+1}. {n.get('title', '')} ({n.get('publisher', '财经')})\n"
    else:
        news_digest = "暂无重大突发突变资讯，当前以纯技术与资金面主导。"

    prompt = f"""
    你是一名顶级实战派美股操盘手兼新手导师。核心宗旨是【大道至简、双向闭环、大白话教学、严控盈亏比】。
    请基于以下【技术指标 + 缺口/EMA20支撑 + 筹码大底 + 宏观情绪 + 实时新闻舆情】，为小白推演交易行动手册：

    【股票标的】: {ticker_input} ｜ 最新价: ${cur_price:.2f}
    【宏观大盘与情绪】: {market_status} ｜ 宏观情绪: {macro_sentiment_tag} (VIX: {vix_status_str} ｜ {tnx_status_str})
    【🧭 周线趋势】: {weekly_status} ｜ 【🧱 日线形态】: {pit_status}
    【⚖️ 日内持仓成本 (VWAP)】: ${vwap_price:.2f} ({vwap_status_desc})
    【🎯 盘中挂单参考】: ${hourly_suggested_entry:.2f} ｜ 止损防线: ${hourly_stop_loss:.2f}
    【动态测算盈亏比】: {rr_ratio:.2f} : 1
    【大级别均线体系】: EMA20生命线: ${ema20:.2f} ｜ MA30: ${ma30:.2f} ｜ 季线(MA60): {ma60_str} ｜ 半年线(MA120): {ma120_str} ｜ 年线(MA250): {ma250_str}
    【🕳️ 短线跳空缺口/EMA20浅回调加仓点】: {gap_sup_str} (昨收盘基准: ${prev_close_p:.2f})
    【🛡️ 大级别主力筑底深回调吸筹带】: {chip_sup_str}
    【🧱 全景真实套牢阻力峰】: {chip_res_str}
    【历史52周真实大顶】: ${high_52w:.2f}
    【财报日历】: {earnings_date_str} {earnings_warning}
    【📰 实时突发新闻舆情】:
    {news_digest}

    【严格输出要求】：
    1. 所有价格数字统一规范加粗（如 **$18.44**）。
    2. 下跌吸筹部分必须分为两级：优先提示【第1阶段短线缺口/EMA20浅回调点位】，再提示【第2阶段大级别筹码深回调大底】。
    3. 盈亏比部分直接给大白话定性（划算/不划算），说明为什么。

    请按以下 4 个板块输出：
    1. 🚦 **多周期共振、宏观情绪与消息面定性（红绿灯）**：2-3句话讲清大趋势、日内VWAP多空态势以及新闻催化剂是利多还是利空。
    2. 💡 **跌势：小白抄底与分批吸筹指南（跌了怎么买）**：
       - **第一阶段（短线缺口/浅回调 15% 仓位）**：回踩哪个短线缺口/EMA20均线支撑价格时可顺势低吸？
       - **第二阶段（大级别深回调/筹码大底 25% 仓位）**：万一跌破均线，在哪个历史密集筹码大底/季线挂单拉低均价？
       - **飞刀熔断禁买线**：跌破哪个价格严禁加仓，必须立刻止损？
    3. 🎯 **涨势：阶梯止盈与突破清仓指南（涨了怎么卖）**：
       - **第一止盈目标（减仓 1/3~1/2）**：反弹触及哪个近端筹码阻力主动锁定利润？
       - **突破顺势推仓点**：带量站稳哪个价格后允许顺势追击看高一线？
       - **大波段终极清仓位**：冲入哪个历史重度套牢峰/52周高点必须果断清仓离场？
    4. ⚖️ **交易质量与盈亏比核验**：直接给结论（划算/不划算），用生活化语言讲透为什么。
    """
    
    return call_gemini_smart(prompt)

# 2. 核心量化算法
@st.cache_data(ttl=300, show_spinner=False)
def fetch_and_analyze(ticker_input):
    ticker_input = ticker_input.strip().upper()
    
    macro_tickers = ["SPY", "QQQ", "^VIX", "^TNX"]
    macro_data = yf.download(macro_tickers, period="3mo", interval="1d", auto_adjust=True, progress=False)
    
    market_status = "🟢 多头顺风：标普与纳指均处于健康上升通道。"
    vix_status_str = "正常"
    tnx_status_str = "正常"
    macro_sentiment_tag = "🟢 情绪向好"
    vix_close = 18.0
    
    try:
        if not macro_data.empty:
            close_data = macro_data['Close']
            spy_c = close_data['SPY'].dropna()
            spy_close = spy_c.iloc[-1]
            spy_prev = spy_c.iloc[-2] if len(spy_c) >= 2 else spy_close
            spy_chg = (spy_close - spy_prev) / spy_prev
            spy_ema5 = EMAIndicator(spy_c, 5).ema_indicator().iloc[-1]
            spy_ema20 = EMAIndicator(spy_c, 20).ema_indicator().iloc[-1]
            
            qqq_c = close_data['QQQ'].dropna()
            qqq_close = qqq_c.iloc[-1]
            qqq_prev = qqq_c.iloc[-2] if len(qqq_c) >= 2 else qqq_close
            qqq_chg = (qqq_close - qqq_prev) / qqq_prev
            qqq_ema5 = EMAIndicator(qqq_c, 5).ema_indicator().iloc[-1]
            qqq_ema20 = EMAIndicator(qqq_c, 20).ema_indicator().iloc[-1]
            
            vix_close = close_data['^VIX'].dropna().iloc[-1]
            vix_status_str = f"⚠️ 恐慌高涨 (VIX={vix_close:.2f} > 22)" if vix_close > 22 else f"🟢 情绪平稳 (VIX={vix_close:.2f})"
            tnx_close = close_data['^TNX'].dropna().iloc[-1]
            tnx_status_str = f"10年美债收益率: {tnx_close:.2f}%"

            if vix_close >= 25:
                macro_sentiment_tag = "🔴 极端恐慌避险"
            elif vix_close >= 20:
                macro_sentiment_tag = "⚠️ 谨慎观望"
            elif vix_close <= 15:
                macro_sentiment_tag = "🔥 极度贪婪活跃"
            else:
                macro_sentiment_tag = "🟢 平稳健康"

            if (spy_close < spy_ema20 and qqq_close < qqq_ema20) or vix_close >= 25:
                market_status = "🔴 极度预警：标普(SPY) 与 纳指(QQQ) 双双破位跌破EMA20，全市场重度防守！"
            elif spy_close < spy_ema20:
                market_status = "⚠️ 警示：标普(SPY) 跌破均线生命线，传统权重股走弱，防范回调！"
            elif qqq_close < qqq_ema20:
                market_status = "⚠️ 结构分化：纳指(QQQ) 破位走弱，科技与成长股承压！"
            elif (spy_close < spy_ema5 or qqq_close < qqq_ema5) and (spy_chg < -0.002 or qqq_chg < -0.002):
                market_status = f"⚠️ 短线承压整固：大盘日内走弱(SPY {spy_chg*100:.2f}%, QQQ {qqq_chg*100:.2f}%)，受EMA5均线压制，切勿盲目追涨！"
            elif vix_close > 20:
                market_status = "⚠️ 情绪预警：恐慌指数 VIX 偏高，市场震荡加剧，注意仓位控制！"
    except Exception:
        pass

    ticker_obj = yf.Ticker(ticker_input)
    df_daily = yf.download(ticker_input, period="2y", interval="1d", auto_adjust=True, progress=False)
    if df_daily.empty:
        return None, f"未找到股票 [{ticker_input}] 的数据，请检查代码是否正确。"
    if isinstance(df_daily.columns, pd.MultiIndex):
        df_daily.columns = df_daily.columns.get_level_values(0)
    
    close_d = df_daily['Close'].dropna()
    high_d = df_daily['High'].dropna()
    low_d = df_daily['Low'].dropna()
    vol_d = df_daily['Volume'].dropna()
    cur_price = close_d.iloc[-1]
    total_days = len(close_d)

    high_52w = high_d.iloc[-min(252, total_days):].max()
    low_30d = low_d.iloc[-min(30, total_days):].min()
    high_30d = high_d.iloc[-min(30, total_days):].max()

    vwap_price = cur_price
    vwap_status_desc = "持平"
    try:
        df_intraday = yf.download(ticker_input, period="1d", interval="5m", auto_adjust=True, progress=False)
        if not df_intraday.empty:
            if isinstance(df_intraday.columns, pd.MultiIndex):
                df_intraday.columns = df_intraday.columns.get_level_values(0)
            typical_p = (df_intraday['High'] + df_intraday['Low'] + df_intraday['Close']) / 3.0
            valid_vol = df_intraday['Volume']
            if valid_vol.sum() > 0:
                vwap_price = (typical_p * valid_vol).sum() / valid_vol.sum()
        else:
            typical_p = (high_d.iloc[-1] + low_d.iloc[-1] + close_d.iloc[-1]) / 3.0
            vwap_price = typical_p
    except Exception:
        vwap_price = cur_price

    if cur_price > vwap_price * 1.002:
        vwap_status_desc = "🟢 位于日内平均成本上方（多头主导/日内浮盈区）"
    elif cur_price < vwap_price * 0.998:
        vwap_status_desc = "🔴 位于日内平均成本下方（空头压制/日内套牢区）"
    else:
        vwap_status_desc = "⚖️ 紧贴日内平均成本线（多空平衡）"

    ema5 = EMAIndicator(close_d, min(5, total_days)).ema_indicator().iloc[-1]
    ema10 = EMAIndicator(close_d, min(10, total_days)).ema_indicator().iloc[-1]
    ema20 = EMAIndicator(close_d, min(20, total_days)).ema_indicator().iloc[-1]
    ma30 = SMAIndicator(close_d, min(30, total_days)).sma_indicator().iloc[-1]
    
    has_ma60 = total_days >= 60
    ma60 = SMAIndicator(close_d, 60).sma_indicator().iloc[-1] if has_ma60 else None
    ma60_str = f"${ma60:.2f}" if ma60 else "上市未满60日"

    has_ma120 = total_days >= 120
    ma120 = SMAIndicator(close_d, 120).sma_indicator().iloc[-1] if has_ma120 else None
    ma120_str = f"${ma120:.2f}" if ma120 else "上市未满120日"

    has_ma250 = total_days >= 250
    ma250 = SMAIndicator(close_d, 250).sma_indicator().iloc[-1] if has_ma250 else None
    ma250_str = f"${ma250:.2f}" if ma250 else "上市未满250日"
    
    gap_support = None
    prev_close_p = close_d.iloc[-2] if total_days >= 2 else cur_price
    if total_days >= 2:
        recent_low = low_d.iloc[-1]
        prev_high = high_d.iloc[-2]
        if recent_low > prev_high:
            gap_support = round(recent_low, 2)
        elif recent_low > prev_close_p:
            gap_support = round(prev_close_p, 2)
        else:
            gap_support = round(ema20, 2)

    rsi_d = RSIIndicator(close_d, min(14, total_days)).rsi().iloc[-1]
    macd_diff_d = MACD(close_d).macd_diff().iloc[-1]
    atr_d = AverageTrueRange(high_d, low_d, close_d, min(14, total_days)).average_true_range().iloc[-1]
    
    cur_vol = vol_d.iloc[-1]
    avg_vol_5d = vol_d.iloc[-6:-1].mean() if total_days >= 6 else vol_d.mean()
    vol_ratio = (cur_vol / avg_vol_5d) if avg_vol_5d > 0 else 1.0

    df_recent_1y = df_daily.iloc[-min(252, total_days):]
    vp = calculate_volume_profile(df_recent_1y, bins=25)
    chip_resistances = []
    chip_supports = []
    if vp:
        sorted_vp = sorted(vp, key=lambda x: x[1], reverse=True)
        top_bins = sorted_vp[:8]
        res_bins = sorted([p for p, _ in top_bins if cur_price * 1.015 < p <= high_52w * 1.02])
        sup_bins = sorted([p for p, _ in top_bins if p < cur_price * 0.985], reverse=True)
        chip_resistances = [round(p, 2) for p in res_bins[:3]]
        chip_supports = [round(p, 2) for p in sup_bins[:2]]

    if not chip_resistances:
        chip_resistances = [round(high_30d, 2), round(high_52w, 2)]
    if not chip_supports:
        chip_supports = [round(low_30d, 2)]

    resistance_list = []
    support_list = []
    
    if cur_price < vwap_price:
        resistance_list.append(f"日内持仓成本线压制 (VWAP): ${vwap_price:.2f}")
    else:
        support_list.append(f"日内持仓成本线支撑 (VWAP): ${vwap_price:.2f}")

    if gap_support and gap_support < cur_price:
        support_list.append(f"🕳️ 【短线跳空缺口/浅回调支撑】: ${gap_support:.2f}")

    if ema5 > cur_price:
        resistance_list.append(f"短线均线压制 (EMA5): ${ema5:.2f}")
    else:
        support_list.append(f"超短支撑 (EMA5): ${ema5:.2f}")

    if ema10 > cur_price:
        resistance_list.append(f"过渡均线压制 (EMA10): ${ema10:.2f}")
    else:
        support_list.append(f"过渡防守 (EMA10): ${ema10:.2f}")

    if ema20 > cur_price:
        resistance_list.append(f"多空分水岭压制 (EMA20): ${ema20:.2f}")
    else:
        support_list.append(f"多空分水岭支撑 (EMA20): ${ema20:.2f}")

    if ma60:
        if ma60 > cur_price:
            resistance_list.append(f"季线重要均线压制 (MA60): ${ma60:.2f}")
        else:
            support_list.append(f"季线重要均线支撑 (MA60): ${ma60:.2f}")

    if ma120:
        if ma120 > cur_price:
            resistance_list.append(f"半年线重要均线压制 (MA120): ${ma120:.2f}")
        else:
            support_list.append(f"半年线重要均线支撑 (MA120): ${ma120:.2f}")

    if ma250:
        if ma250 > cur_price:
            resistance_list.append(f"年线长线牛熊压制 (MA250): ${ma250:.2f}")
        else:
            support_list.append(f"年线长线牛熊支撑 (MA250): ${ma250:.2f}")

    for idx, cp in enumerate(chip_resistances):
        tag = "第1阶梯筹码阻力" if idx == 0 else "突破加速目标筹码峰" if idx == 1 else "历史大级别套牢顶"
        resistance_list.append(f"🧱 【{tag}】: ${cp:.2f}")

    for idx, sp in enumerate(chip_supports):
        tag = "主力波段大底吸筹带" if idx == 0 else "大级别筑底防守带"
        support_list.append(f"🛡️ 【{tag}】: ${sp:.2f}")

    pit_status = "正常走势"
    if rsi_d < 38 and cur_price <= low_30d * 1.03:
        pit_status = "💎 极端超卖黄金坑：指标极度冰点超跌，存在高盈亏比反转机会！"
    elif cur_price >= ema20 and macd_diff_d > 0 and vol_ratio >= 0.9:
        pit_status = "🧱 右侧企稳确立：重回EMA20均线之上，动能回暖，结构扎实。"
    elif cur_price < ema20 and vol_ratio < 0.7:
        pit_status = "🧊 缩量磨底中：跌破均线但抛压衰竭，等待放量企稳确认。"

    weekly_status = "周线中性"
    try:
        df_weekly = yf.download(ticker_input, period="2y", interval="1wk", auto_adjust=True, progress=False)
        if not df_weekly.empty:
            if isinstance(df_weekly.columns, pd.MultiIndex):
                df_weekly.columns = df_weekly.columns.get_level_values(0)
            close_w = df_weekly['Close'].dropna()
            if len(close_w) >= 20:
                w_ema20 = EMAIndicator(close_w, 20).ema_indicator().iloc[-1]
                w_macd_diff = MACD(close_w).macd_diff().iloc[-1]
                if cur_price >= w_ema20 and w_macd_diff >= 0:
                    weekly_status = "🟢 顺风大牛势：站稳EMA20生命线且动能向上！"
                elif cur_price >= w_ema20:
                    weekly_status = "🟡 强势震荡：位于均线上方，高位整固阶段。"
                else:
                    weekly_status = "🔴 逆风熊势/深度调整：周线跌破EMA20，大趋势向下！"
    except Exception:
        pass

    hourly_status = "盘中中性"
    hourly_suggested_entry = cur_price
    hourly_stop_loss = cur_price * 0.985
    try:
        df_hourly = yf.download(ticker_input, period="1mo", interval="1h", auto_adjust=True, progress=False)
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
                    hourly_status = "🎯 盘中狙击买点已触发：1小时回踩企稳！"
                elif h_rsi >= 70:
                    hourly_status = "⚠️ 1小时短线超买：切勿追高，等待盘中回踩！"
                elif cur_price < h_ema20:
                    hourly_status = "🧊 1小时盘中偏弱：等待重回1小时EMA20均线。"
    except Exception:
        pass

    target1_p = chip_resistances[0] if chip_resistances else high_30d
    reward_space = max(0.01, target1_p - cur_price)
    risk_space = max(0.01, cur_price - hourly_stop_loss)
    rr_ratio = reward_space / risk_space

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
    
    news_items = []
    try:
        raw_news = ticker_obj.news
        if raw_news and len(raw_news) > 0:
            for item in raw_news[:5]:
                title = item.get('title') or (item.get('content', {}).get('title') if isinstance(item.get('content'), dict) else '')
                publisher = item.get('publisher') or (item.get('content', {}).get('provider', {}).get('displayName') if isinstance(item.get('content'), dict) else '资讯')
                link = item.get('link') or (item.get('content', {}).get('canonicalUrl', {}).get('url') if isinstance(item.get('content'), dict) else '')
                if title:
                    news_items.append({"title": title, "publisher": publisher, "link": link})
    except Exception:
        pass

    top_faqs = [
        f"🚀 {ticker_input} 能否站回上方关键阻力？继续涨到目标价还有多少%？",
        f"🕳️ {ticker_input} 短线缺口支撑与筹码大底分别是多少？如何分批吸筹？",
        f"⚖️ 当前介入 {ticker_input} 的盈亏比是多少？值得冒这个风险吗？"
    ]

    now_utc = datetime.now(timezone.utc)
    local_time_str = (now_utc + timedelta(hours=8)).strftime("%H:%M:%S")
    et_time_str = (now_utc - timedelta(hours=4)).strftime("%H:%M:%S")
    cache_display_time = f"{local_time_str} 本地 ｜ {et_time_str} 美东"

    ai_analysis_text = get_ai_analysis(ticker_input, cur_price, market_status, vix_status_str, 
                                       tnx_status_str, macro_sentiment_tag, weekly_status, pit_status, 
                                       hourly_status, vwap_price, vwap_status_desc, hourly_suggested_entry, 
                                       hourly_stop_loss, chip_resistances, chip_supports, gap_support, prev_close_p,
                                       earnings_date_str, days_to_earnings, news_items, high_30d, high_52w, ema20, ma30,
                                       ma60_str, ma120_str, ma250_str, rr_ratio)

    result_bundle = {
        "symbol": ticker_input,
        "market_status": market_status,
        "vix_status_str": vix_status_str,
        "tnx_status_str": tnx_status_str,
        "macro_sentiment_tag": macro_sentiment_tag,
        "weekly_status": weekly_status,
        "pit_status": pit_status,
        "hourly_status": hourly_status,
        "cur_price": cur_price,
        "vol_ratio": vol_ratio,
        "vwap_price": vwap_price,
        "vwap_status_desc": vwap_status_desc,
        "hourly_suggested_entry": hourly_suggested_entry,
        "hourly_stop_loss": hourly_stop_loss,
        "gap_support": gap_support,
        "prev_close_p": prev_close_p,
        "ema20": ema20,
        "ma30": ma30,
        "chip_resistances": chip_resistances,
        "chip_supports": chip_supports,
        "ma60_str": ma60_str,
        "ma120_str": ma120_str,
        "ma250_str": ma250_str,
        "rr_ratio": rr_ratio,
        "news_items": news_items,
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
    st.session_state.history_tickers = ["NVDA", "USAR", "TSLA", "AAPL"]

if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = "NVDA"

st.write("**🔥 快速自选与最近查询:**")
cols = st.columns(len(st.session_state.history_tickers))
for i, ticker in enumerate(st.session_state.history_tickers):
    if cols[i].button(ticker, use_container_width=True):
        st.session_state.selected_ticker = ticker

ticker_input = st.text_input("美股代码", value=st.session_state.selected_ticker).strip().upper()

# 4. 诊断展示看板
if st.button("开始全维实战闭环诊断", type="primary", use_container_width=True):
    if ticker_input and ticker_input in st.session_state.history_tickers:
        st.session_state.history_tickers.remove(ticker_input)
    if ticker_input:
        st.session_state.history_tickers.insert(0, ticker_input)
        if len(st.session_state.history_tickers) > 5:
            st.session_state.history_tickers.pop()

    with st.spinner(f"正在全维运算缺口支撑、全景均线与盈亏比数据 ({ticker_input})..."):
        data, err = fetch_and_analyze(ticker_input)
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
    
    if "🔴" in data['market_status']:
        st.error(f"**大盘风控:** {data['market_status']}")
    elif "⚠️" in data['market_status']:
        st.warning(f"**大盘风控:** {data['market_status']}")
    else:
        st.success(f"**大盘风控:** {data['market_status']}")
    
    st.info(f"🌐 **宏观情绪与利率：** 【{data['macro_sentiment_tag']}】 ｜ {data['vix_status_str']} ｜ 🏛️ {data['tnx_status_str']}")
    
    st.subheader("🚦 多周期共振雷达")
    st.write(f"- 🧭 **周线大趋势 (中期定性):** {data['weekly_status']}")
    st.write(f"- 🧱 **日线形态 (黄金坑/筑底):** {data['pit_status']}")
    st.write(f"- 🎯 **1小时盘中 (狙击买点):** {data['hourly_status']}")
    st.write(f"- ⚖️ **日内持仓成本线 (VWAP):** `${data['vwap_price']:.2f}` ({data['vwap_status_desc']})")
    
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric(label=f"{curr_ticker} 最新价", value=f"${data['cur_price']:.2f}")
    vol_status = "🔥 放量" if data['vol_ratio'] > 1.3 else "🧊 缩量" if data['vol_ratio'] < 0.7 else "⚖️ 平量"
    col_m2.metric(label="5日量比", value=f"{data['vol_ratio']:.2f} 倍", delta=vol_status)
    rr_delta = "🟢 优秀" if data['rr_ratio'] >= 2.0 else "⚠️ 一般"
    col_m3.metric(label="动态盈亏比", value=f"{data['rr_ratio']:.2f} : 1", delta=rr_delta)

    if data['news_items']:
        with st.expander(f"📰 {curr_ticker} 实时盘中资讯与新闻催化剂 ({len(data['news_items'])} 条)", expanded=False):
            for n in data['news_items']:
                st.write(f"- **[{n['publisher']}]** [{n['title']}]({n['link']})")

    st.subheader("🤖 Gemini 操盘手行动指令 (全闭环实战手册)")
    safe_render_markdown(data['ai_analysis_text'])

    st.subheader("🛡️ 阶梯支撑与动态阻力看板 (含缺口/季线/半年线/年线)")
    col1, col2 = st.columns(2)
    with col1:
        st.info("**【🟢 抄底/吸筹/拉低均价支撑带】**\n\n" + "\n\n".join(data['support_list']))
    with col2:
        st.warning("**【🔴 阶梯止盈/突破/清仓阻力带】**\n\n" + "\n\n".join(data['resistance_list']))
    
    st.subheader("⚡ 动能与量价特征")
    macd_str = "🟢 多头金叉（动能充沛）" if data['macd_diff_d'] > 0 else "🔴 动能减弱/死叉休整"
    st.write(f"- **MACD 状态:** `{macd_str}` (柱值: {data['macd_diff_d']:.2f})")
    st.write(f"- **RSI (14):** `{data['rsi_d']:.2f}` ({'⚠️ 超买' if data['rsi_d'] > 70 else '💎 极端超卖/黄金坑区' if data['rsi_d'] < 38 else '⚖️ 中性'})")
    st.write(f"- **日均真实波幅 (ATR):** `${data['atr_d']:.2f}`")

    # 5. 纯智脑 Gemini 专属 AI 操盘助理（100% 自由智能问答）
    st.divider()
    st.subheader("💬 对当前诊断有疑问？随时追问 AI 助理")
    st.caption(f"💡 真正挂载 Gemini 智脑，支持数学空间计算、条件假设推演、个股对比与策略变通。")

    clicked_faq = None
    if "top_faqs" in data and data["top_faqs"]:
        st.write("**🔥 该股当下高频实战疑问 (点击一键直答):**")
        for idx, faq_text in enumerate(data["top_faqs"]):
            if st.button(faq_text, key=f"faq_{idx}", use_container_width=True):
                clicked_faq = faq_text

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            safe_render_markdown(msg["content"])

    user_input = st.chat_input(f"自由提问（如：到230有多少%？跌破缺口怎么看？消息面利多吗？）...")
    prompt_to_process = user_input or clicked_faq

    if prompt_to_process:
        st.session_state.chat_history.append({"role": "user", "content": prompt_to_process})
        with st.chat_message("user"):
            safe_render_markdown(prompt_to_process)

        with st.chat_message("assistant"):
            with st.spinner("Gemini 正在深度思考并计算推演中..."):
                extracted_symbols = extract_tickers_from_text(prompt_to_process)
                extra_data_text = ""
                
                for sym in extracted_symbols:
                    if sym != curr_ticker:
                        try:
                            other_data, _ = fetch_and_analyze(sym)
                            if other_data:
                                extra_data_text += f"""
                                【提及标的 {sym} 最新量化数据】:
                                现价: ${other_data['cur_price']:.2f}, 盈亏比: {other_data['rr_ratio']:.2f}:1, VWAP: ${other_data['vwap_price']:.2f}, 周线: {other_data['weekly_status']}
                                关键阻力: {'; '.join(other_data['resistance_list'][:2])}, 关键支撑: {'; '.join(other_data['support_list'][:2])}
                                """
                        except Exception:
                            pass

                news_brief = "\n".join([f"- {n['title']}" for n in data['news_items'][:3]]) if data['news_items'] else "无突发新闻"
                
                context_prompt = f"""
                你是一名顶级的资深美股操盘手兼量化导师。你拥有极其强大、聪颖、灵活的自然语言理解与变通推演能力。
                
                【当前标的】: {curr_ticker} ｜ 现价: ${data['cur_price']:.2f}
                【宏观环境】: {data['market_status']} ｜ 情绪: {data['macro_sentiment_tag']}
                【实时资讯】:
                {news_brief}
                【日内成本 (VWAP)】: ${data['vwap_price']:.2f} ({data['vwap_status_desc']})
                【均线体系】: EMA20: ${data.get('ema20', 0):.2f} ｜ MA30: ${data.get('ma30', 0):.2f} ｜ MA60: {data.get('ma60_str', '无')} ｜ MA250: {data.get('ma250_str', '无')}
                【短线跳空缺口/EMA20支撑】: ${data.get('gap_support', data['ema20']):.2f} (昨收盘: ${data.get('prev_close_p', data['cur_price']):.2f})
                【波段筹码大底】: {', '.join([f'${p:.2f}' for p in data['chip_supports']])}
                【全景密集阻力阶梯】: {' ➔ '.join([f'${p:.2f}' for p in data['chip_resistances']])}
                【1小时参考买点】: ${data['hourly_suggested_entry']:.2f} ｜ 止损防线: ${data['hourly_stop_loss']:.2f}
                【动态盈亏比】: {data['rr_ratio']:.2f} : 1
                {extra_data_text}

                用户的真实提问是: "{prompt_to_process}"

                【严格执行规则】：
                1. 严禁使用任何机械模版！彻底根据用户的具体提问进行针对性解答：
                   - 如果用户问“到某个价格有多少% / 空间多少”，请直接进行精确数学计算并给出百分比，同时点评突破难度；
                   - 如果用户问“能不能站回某价”，请分析上方的套牢盘抛压、突破所需的放量条件与确认动作；
                   - 如果用户问“跌破某支撑怎么办”，请从跌破价格向下推演更低的接力支撑与止损纪律；
                   - 如果用户问“盈亏比”，用大白话直接给结论（划算/不划算）。
                2. 所有涉及的价格数字统一紧跟美元符号加粗（如 **$230.47**，**+4.9%**）。
                3. 说话口吻要像一个顶级资深操盘手老朋友，直接给结论和干货，通俗犀利。
                """
                
                reply_text = call_gemini_smart(context_prompt)
                safe_render_markdown(reply_text)
                st.session_state.chat_history.append({"role": "assistant", "content": reply_text})
