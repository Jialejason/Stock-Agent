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

st.set_page_config(page_title="投资小助手 Pro 机构量化版", layout="centered")
st.title("📈 投资小助手 Pro (机构量化微观版)")
st.caption("⚡ 5分钟全网缓存 ｜ 🧠 Gemini 机构级智脑 ｜ 📊 VPVR筹码区(POC/VAH/VAL) ｜ 🎯 期权痛点/PCR ｜ 🛡️ ATR动态风控")

# 1. 基础别名映射与 Markdown 安全渲染
TICKER_ALIASES = {
    "TESLA": "TSLA", "特斯拉": "TSLA",
    "APPLE": "AAPL", "苹果": "AAPL",
    "NVIDIA": "NVDA", "英伟达": "NVDA",
    "GOOGLE": "GOOGL", "谷歌": "GOOGL",
    "AMAZON": "AMZN", "亚马逊": "AMZN",
    "MICROSOFT": "MSFT", "微软": "MSFT",
    "META": "META", "脸书": "META",
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

# 动态获取 API Key
raw_api_key = st.secrets.get("GEMINI_API_KEY", "") if "GEMINI_API_KEY" in st.secrets else ""
api_key = raw_api_key.strip().replace("\n", "").replace("\r", "").replace(" ", "")

def call_gemini_smart(prompt_text):
    if not api_key:
        return "⚠️ 未检测到 API Key，请在 Streamlit Secrets 中配置 `GEMINI_API_KEY`。"
    
    try:
        genai.configure(api_key=api_key)
        candidate_models = ['gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-1.5-flash-latest', 'gemini-1.5-pro-latest']
        for m_name in candidate_models:
            try:
                model = genai.GenerativeModel(m_name)
                res = model.generate_content(prompt_text)
                if res and res.text:
                    return res.text
            except Exception:
                continue

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

# 2. 机构级微观结构计算模块 (Volume Profile & Options)
def calculate_institutional_volume_profile(df_daily, bins=40):
    if df_daily.empty or 'Close' not in df_daily.columns or 'Volume' not in df_daily.columns:
        return {"poc": 0.0, "vah": 0.0, "val": 0.0, "supports": [], "resistances": []}

    price_min = df_daily['Low'].min()
    price_max = df_daily['High'].max()
    if price_max <= price_min or np.isnan(price_min) or np.isnan(price_max):
        return {"poc": 0.0, "vah": 0.0, "val": 0.0, "supports": [], "resistances": []}

    bin_edges = np.linspace(price_min, price_max, bins + 1)
    vol_profile = np.zeros(bins)

    for _, row in df_daily.iterrows():
        mid_p = (row['High'] + row['Low'] + row['Close']) / 3.0
        b_idx = int(np.digitize(mid_p, bin_edges) - 1)
        b_idx = max(0, min(bins - 1, b_idx))
        vol_profile[b_idx] += row['Volume']

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    poc_idx = np.argmax(vol_profile)
    poc_price = float(bin_centers[poc_idx])

    # 70% 价值区 (Value Area)
    total_vol = vol_profile.sum()
    target_vol = total_vol * 0.70
    sorted_indices = np.argsort(vol_profile)[::-1]
    accum_vol = 0
    va_indices = []
    for idx in sorted_indices:
        accum_vol += vol_profile[idx]
        va_indices.append(idx)
        if accum_vol >= target_vol:
            break

    val_price = float(bin_centers[min(va_indices)])
    vah_price = float(bin_centers[max(va_indices)])

    cur_price = df_daily['Close'].iloc[-1]
    top_indices = sorted_indices[:8]
    res_bins = sorted([bin_centers[i] for i in top_indices if bin_centers[i] > cur_price * 1.01])
    sup_bins = sorted([bin_centers[i] for i in top_indices if bin_centers[i] < cur_price * 0.99], reverse=True)

    return {
        "poc": poc_price,
        "vah": vah_price,
        "val": val_price,
        "resistances": [round(p, 2) for p in res_bins[:3]],
        "supports": [round(p, 2) for p in sup_bins[:2]]
    }

def fetch_options_microstructure(ticker_obj, cur_price):
    try:
        expirations = ticker_obj.options
        if not expirations:
            return {"max_pain": 0.0, "pcr": 1.0, "major_call_wall": 0.0, "major_put_wall": 0.0}

        # 抓取最近一个到期日期权链
        opt_chain = ticker_obj.option_chain(expirations[0])
        calls = opt_chain.calls
        puts = opt_chain.puts

        total_call_oi = calls['openInterest'].fillna(0).sum()
        total_put_oi = puts['openInterest'].fillna(0).sum()
        pcr = float(total_put_oi / total_call_oi) if total_call_oi > 0 else 1.0

        # 期权持仓墙 (OI Walls)
        call_wall = float(calls.loc[calls['openInterest'].idxmax()]['strike']) if not calls.empty and calls['openInterest'].sum() > 0 else 0.0
        put_wall = float(puts.loc[puts['openInterest'].idxmax()]['strike']) if not puts.empty and puts['openInterest'].sum() > 0 else 0.0

        # 计算 Max Pain (最大痛点)
        strikes = sorted(list(set(calls['strike'].tolist() + puts['strike'].tolist())))
        loss_dict = {}
        for s in strikes:
            call_loss = (np.maximum(0, s - calls['strike']) * calls['openInterest'].fillna(0)).sum()
            put_loss = (np.maximum(0, puts['strike'] - s) * puts['openInterest'].fillna(0)).sum()
            loss_dict[s] = call_loss + put_loss

        max_pain = min(loss_dict, key=loss_dict.get) if loss_dict else cur_price
        return {
            "max_pain": float(max_pain),
            "pcr": float(pcr),
            "major_call_wall": call_wall,
            "major_put_wall": put_wall
        }
    except Exception:
        return {"max_pain": 0.0, "pcr": 1.0, "major_call_wall": 0.0, "major_put_wall": 0.0}

# 3. 核心量化算法
@st.cache_data(ttl=300, show_spinner=False)
def fetch_and_analyze(ticker_input):
    ticker_input = ticker_input.strip().upper()
    
    macro_tickers = ["SPY", "QQQ", "^VIX", "^TNX"]
    macro_data = yf.download(macro_tickers, period="3mo", interval="1d", auto_adjust=True, progress=False)
    
    market_status = "🟢 多头顺风：标普与纳指均处于健康上升通道。"
    vix_status_str = "正常"
    tnx_status_str = "正常"
    macro_sentiment_tag = "🟢 情绪向好"
    
    try:
        if not macro_data.empty:
            close_data = macro_data['Close']
            spy_c = close_data['SPY'].dropna()
            spy_close = spy_c.iloc[-1]
            spy_ema20 = EMAIndicator(spy_c, 20).ema_indicator().iloc[-1]
            
            qqq_c = close_data['QQQ'].dropna()
            qqq_close = qqq_c.iloc[-1]
            qqq_ema20 = EMAIndicator(qqq_c, 20).ema_indicator().iloc[-1]
            
            vix_close = close_data['^VIX'].dropna().iloc[-1]
            vix_status_str = f"⚠️ 恐慌升温 (VIX={vix_close:.2f})" if vix_close > 22 else f"🟢 平稳 (VIX={vix_close:.2f})"
            tnx_close = close_data['^TNX'].dropna().iloc[-1]
            tnx_status_str = f"10Y美债: {tnx_close:.2f}%"

            if vix_close >= 25:
                macro_sentiment_tag = "🔴 极端避险"
            elif vix_close <= 15:
                macro_sentiment_tag = "🔥 极度贪婪"
            else:
                macro_sentiment_tag = "🟢 结构平衡"

            if (spy_close < spy_ema20 and qqq_close < qqq_ema20) or vix_close >= 25:
                market_status = "🔴 极度预警：SPY 与 QQQ 双双跌破 EMA20 生命线，机构防御避险！"
            elif spy_close < spy_ema20 or qqq_close < qqq_ema20:
                market_status = "⚠️ 结构分化：核心大盘指数回踩生命线，注意短线洗盘！"
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

    # 日内 VWAP
    vwap_price = cur_price
    try:
        df_intraday = yf.download(ticker_input, period="1d", interval="5m", auto_adjust=True, progress=False)
        if not df_intraday.empty:
            if isinstance(df_intraday.columns, pd.MultiIndex):
                df_intraday.columns = df_intraday.columns.get_level_values(0)
            typical_p = (df_intraday['High'] + df_intraday['Low'] + df_intraday['Close']) / 3.0
            valid_vol = df_intraday['Volume']
            if valid_vol.sum() > 0:
                vwap_price = (typical_p * valid_vol).sum() / valid_vol.sum()
    except Exception:
        vwap_price = cur_price

    vwap_status_desc = "多头主导" if cur_price > vwap_price * 1.002 else "空头压制" if cur_price < vwap_price * 0.998 else "多空博弈中轴"

    # 均线系统
    ema5 = EMAIndicator(close_d, min(5, total_days)).ema_indicator().iloc[-1]
    ema20 = EMAIndicator(close_d, min(20, total_days)).ema_indicator().iloc[-1]
    ma30 = SMAIndicator(close_d, min(30, total_days)).sma_indicator().iloc[-1]
    ma60 = SMAIndicator(close_d, 60).sma_indicator().iloc[-1] if total_days >= 60 else None
    ma120 = SMAIndicator(close_d, 120).sma_indicator().iloc[-1] if total_days >= 120 else None
    ma250 = SMAIndicator(close_d, 250).sma_indicator().iloc[-1] if total_days >= 250 else None

    rsi_d = RSIIndicator(close_d, min(14, total_days)).rsi().iloc[-1]
    macd_diff_d = MACD(close_d).macd_diff().iloc[-1]
    atr_d = AverageTrueRange(high_d, low_d, close_d, min(14, total_days)).average_true_range().iloc[-1]

    # 筹码分布 (Volume Profile) 与衍生品数据
    vp_data = calculate_institutional_volume_profile(df_daily.iloc[-min(252, total_days):])
    opt_data = fetch_options_microstructure(ticker_obj, cur_price)

    chip_resistances = vp_data["resistances"] or [round(high_30d, 2), round(high_52w, 2)]
    chip_supports = vp_data["supports"] or [round(low_30d, 2)]

    # 动态盈亏比与风控测算 (基于 1.5x ATR 动态止损)
    target1_p = chip_resistances[0]
    dynamic_stop_loss = max(low_30d, cur_price - (1.5 * atr_d))
    reward_space = max(0.01, target1_p - cur_price)
    risk_space = max(0.01, cur_price - dynamic_stop_loss)
    rr_ratio = reward_space / risk_space

    # 支撑与阻力集合
    support_list = [f"日内做市商成本 (VWAP): ${vwap_price:.2f}", f"生命线防守 (EMA20): ${ema20:.2f}"]
    if vp_data["poc"] > 0:
        support_list.append(f"筹码控制中心 (POC): ${vp_data['poc']:.2f}")
    if vp_data["val"] > 0:
        support_list.append(f"价值区下沿 (VAL): ${vp_data['val']:.2f}")
    if opt_data["major_put_wall"] > 0:
        support_list.append(f"期权看跌大单防守墙: ${opt_data['major_put_wall']:.2f}")

    resistance_list = [f"第一阶梯筹码阻力: ${target1_p:.2f}"]
    if vp_data["vah"] > 0:
        resistance_list.append(f"价值区上沿 (VAH): ${vp_data['vah']:.2f}")
    if opt_data["max_pain"] > 0:
        resistance_list.append(f"期权最大痛点 (Max Pain): ${opt_data['max_pain']:.2f}")
    if opt_data["major_call_wall"] > 0:
        resistance_list.append(f"期权看涨大单压制墙: ${opt_data['major_call_wall']:.2f}")

    news_items = []
    try:
        raw_news = ticker_obj.news
        if raw_news:
            for item in raw_news[:4]:
                title = item.get('title') or (item.get('content', {}).get('title') if isinstance(item.get('content'), dict) else '')
                publisher = item.get('publisher') or (item.get('content', {}).get('provider', {}).get('displayName') if isinstance(item.get('content'), dict) else '资讯')
                link = item.get('link') or (item.get('content', {}).get('canonicalUrl', {}).get('url') if isinstance(item.get('content'), dict) else '')
                if title:
                    news_items.append({"title": title, "publisher": publisher, "link": link})
    except Exception:
        pass

    # 机构级行动手册 Prompt
    inst_action_prompt = f"""
你是一名顶级对冲基金的资深量化操盘手。请基于以下微观结构与流动性数据，为交易员制定一份【机构级多空执行策略】：

【标的】: {ticker_input} ｜ 现价: **${cur_price:.2f}**
【宏观大盘】: {market_status} ｜ 情绪度: {macro_sentiment_tag} ｜ {vix_status_str} ｜ {tnx_status_str}
【做市商成本 (VWAP)】: **${vwap_price:.2f}** ({vwap_status_desc})
【Volume Profile 筹码峰】: POC中心: **${vp_data['poc']:.2f}** ｜ VAH上沿: **${vp_data['vah']:.2f}** ｜ VAL下沿: **${vp_data['val']:.2f}**
【均线防御带】: EMA5: **${ema5:.2f}** ｜ EMA20: **${ema20:.2f}** ｜ MA30: **${ma30:.2f}**
【期权微观博弈】: Max Pain(最大痛点): **${opt_data['max_pain']:.2f}** ｜ PCR比率: **{opt_data['pcr']:.2f}** ｜ Call阻力墙: **${opt_data['major_call_wall']:.2f}** ｜ Put支撑墙: **${opt_data['major_put_wall']:.2f}**
【波动率与风控】: 14日ATR: **${atr_d:.2f}** ｜ 建议动态止损: **${dynamic_stop_loss:.2f}** (1.5x ATR) ｜ 动态盈亏比: **{rr_ratio:.2f} : 1**

【输出规范】：
1. 严禁模版套话，直切资金博弈本质。价格数值全部紧跟美元符号加粗（如 **$220.50**）。
2. 请分四个板块输出：
   - 🚦 **微观流动性与期权偏斜定性**：2-3句话剖析做市商当前是在逼空、压制还是横盘吸筹。
   - 🛡️ **机构级分批建仓与流动性防守点**：
     * 浅回调试仓点（VWAP / EMA20 / POC）
     * 深度吸筹区（VAL / 1.5x ATR 容错缓冲）
     * 结构失效硬止损位（严禁扛单）
   - 🎯 **流动性出清与阶梯止盈位**：
     * 第一止盈目标（VAH / 近端筹码真空边缘）
     * 突破顺势加速位（Call Wall 挤压点）
   - ⚖️ **盈亏比量化评估**：用交易员大白话定性当前点位性价比。
"""
    ai_analysis_text = call_gemini_smart(inst_action_prompt)

    top_faqs = [
        f"🚀 {ticker_input} 距离上方阻力/VAH还有多少%？突破难度如何？",
        f"🛡️ 做市商与主力筹码底（POC/VAL）在哪个价位？跌破怎么防守？",
        f"⚖️ 当前位置建仓的盈亏比 ({rr_ratio:.2f}:1) 是否值得出手？"
    ]

    now_utc = datetime.now(timezone.utc)
    cache_display_time = (now_utc + timedelta(hours=8)).strftime("%H:%M:%S")

    return {
        "symbol": ticker_input,
        "cur_price": cur_price,
        "market_status": market_status,
        "macro_sentiment_tag": macro_sentiment_tag,
        "vix_status_str": vix_status_str,
        "tnx_status_str": tnx_status_str,
        "vwap_price": vwap_price,
        "vwap_status_desc": vwap_status_desc,
        "ema20": ema20,
        "ma30": ma30,
        "ma60_str": f"${ma60:.2f}" if ma60 else "无",
        "ma250_str": f"${ma250:.2f}" if ma250 else "无",
        "atr_d": atr_d,
        "dynamic_stop_loss": dynamic_stop_loss,
        "vp_data": vp_data,
        "opt_data": opt_data,
        "chip_resistances": chip_resistances,
        "chip_supports": chip_supports,
        "support_list": support_list,
        "resistance_list": resistance_list,
        "rr_ratio": rr_ratio,
        "news_items": news_items,
        "top_faqs": top_faqs,
        "ai_analysis_text": ai_analysis_text,
        "cache_display_time": cache_display_time
    }, None

# 4. 界面交互
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

if st.button("开始机构级量化微观诊断", type="primary", use_container_width=True):
    if ticker_input and ticker_input in st.session_state.history_tickers:
        st.session_state.history_tickers.remove(ticker_input)
    if ticker_input:
        st.session_state.history_tickers.insert(0, ticker_input)
        if len(st.session_state.history_tickers) > 5:
            st.session_state.history_tickers.pop()

    with st.spinner(f"正在全维解析期权链、Volume Profile 筹码与 ATR 波动率 ({ticker_input})..."):
        data, err = fetch_and_analyze(ticker_input)
        if err:
            st.error(f"❌ {err}")
        else:
            st.session_state.current_data = data
            st.session_state.current_ticker = ticker_input
            st.session_state.chat_history = []

# 渲染数据看板
if "current_data" in st.session_state and st.session_state.current_data:
    data = st.session_state.current_data
    curr_ticker = st.session_state.get("current_ticker", ticker_input)

    st.caption(f"⚡ 数据已缓存 (刷新: {data['cache_display_time']}) ｜ 5分钟内共享无消耗")
    st.info(f"🌐 **宏观与流动性环境：** 【{data['macro_sentiment_tag']}】 ｜ {data['vix_status_str']} ｜ 🏛️ {data['tnx_status_str']}")

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric(label=f"{curr_ticker} 现价", value=f"${data['cur_price']:.2f}")
    col_m2.metric(label="筹码中轴 (POC)", value=f"${data['vp_data']['poc']:.2f}")
    rr_delta = "🟢 优秀" if data['rr_ratio'] >= 2.0 else "⚠️ 一般"
    col_m3.metric(label="动态盈亏比", value=f"{data['rr_ratio']:.2f} : 1", delta=rr_delta)

    # 机构量化卡片
    col_q1, col_q2, col_q3 = st.columns(3)
    col_q1.metric(label="价值区 (VAL ➔ VAH)", value=f"${data['vp_data']['val']:.2f} - ${data['vp_data']['vah']:.2f}")
    col_q2.metric(label="期权 Max Pain", value=f"${data['opt_data']['max_pain']:.2f}")
    col_q3.metric(label="动态止损 (1.5x ATR)", value=f"${data['dynamic_stop_loss']:.2f}")

    st.subheader("🤖 机构操盘手实战指令")
    safe_render_markdown(data['ai_analysis_text'])

    st.subheader("🛡️ 机构微观支撑与阻力矩阵")
    col1, col2 = st.columns(2)
    with col1:
        st.info("**【🟢 流动性支撑与吸筹带】**\n\n" + "\n\n".join(data['support_list']))
    with col2:
        st.warning("**【🔴 机构阻力与出清目标】**\n\n" + "\n\n".join(data['resistance_list']))

    # 机构级 AI 追问助理
    st.divider()
    st.subheader("💬 机构微观追问助理")
    
    clicked_faq = None
    if "top_faqs" in data and data["top_faqs"]:
        for idx, faq_text in enumerate(data["top_faqs"]):
            if st.button(faq_text, key=f"faq_{idx}", use_container_width=True):
                clicked_faq = faq_text

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            safe_render_markdown(msg["content"])

    user_input = st.chat_input("自由提问（如：到230有多少%？做市商防守位在哪？跌破POC怎么看？）...")
    prompt_to_process = user_input or clicked_faq

    if prompt_to_process:
        st.session_state.chat_history.append({"role": "user", "content": prompt_to_process})
        with st.chat_message("user"):
            safe_render_markdown(prompt_to_process)

        with st.chat_message("assistant"):
            with st.spinner("机构量化引擎正在解析订单流与衍生品博弈..."):
                extracted_symbols = extract_tickers_from_text(prompt_to_process)
                extra_data_text = ""
                
                for sym in extracted_symbols:
                    if sym != curr_ticker:
                        try:
                            other_data, _ = fetch_and_analyze(sym)
                            if other_data:
                                extra_data_text += f"""
                                【联动标的 {sym} 数据】:
                                现价: ${other_data['cur_price']:.2f} | 盈亏比: {other_data['rr_ratio']:.2f}:1 | VWAP: ${other_data['vwap_price']:.2f} | POC: ${other_data['vp_data']['poc']:.2f}
                                """
                        except Exception:
                            pass

                news_brief = "\n".join([f"- {n['title']}" for n in data['news_items'][:3]]) if data['news_items'] else "无重大异常资讯"
                
                context_prompt = f"""
你是一名顶级对冲基金的高级量化交易员。你精通订单流、做市商对冲机制（Gamma/Delta）、Volume Profile 与严格的动态风险敞口管理。

【标的】: {curr_ticker} ｜ 现价: **${data['cur_price']:.2f}**
【宏观环境】: {data['market_status']} ｜ 情绪度: {data['macro_sentiment_tag']}
【突发资讯】:
{news_brief}

━━━━━━━━【微观流动性与筹码结构】━━━━━━━━
● 日内做市商分水岭 (VWAP): **${data['vwap_price']:.2f}** ({data['vwap_status_desc']})
● 筹码密集核心 (POC): **${data['vp_data']['poc']:.2f}**
● 价值区上沿 (VAH 阻力): **${data['vp_data']['vah']:.2f}** ｜ 价值区下沿 (VAL 支撑): **${data['vp_data']['val']:.2f}**
● 均线防御: EMA20: **${data['ema20']:.2f}** ｜ MA30: **${data['ma30']:.2f}** ｜ MA60: {data['ma60_str']} ｜ MA250: {data['ma250_str']}

━━━━━━━━【衍生品与波动率风控】━━━━━━━━
● 期权结构: Max Pain: **${data['opt_data']['max_pain']:.2f}** ｜ PCR: **{data['opt_data']['pcr']:.2f}** ｜ Call Wall: **${data['opt_data']['major_call_wall']:.2f}** ｜ Put Wall: **${data['opt_data']['major_put_wall']:.2f}**
● 14日ATR: **${data['atr_d']:.2f}** ｜ 动态保护止损: **${data['dynamic_stop_loss']:.2f}**
● 动态盈亏比: **{data['rr_ratio']:.2f} : 1**
{extra_data_text}

用户的真实提问是: "{prompt_to_process}"

━━━━━━━━【响应准则】━━━━━━━━
1. **直接给出结论与精确数学计算**：问空间必算涨跌幅百分比；问阻力结合 VAH 与期权 Call Wall；问支撑结合 POC、VAL 与 1.5x ATR 容错底线。
2. **所有数字统一严格加粗**（如 **$230.47**，**+4.39%**）。
3. 语言犀利、干练，完全以机构交易员的实战风控视角进行解答。
"""
                reply_text = call_gemini_smart(context_prompt)
                safe_render_markdown(reply_text)
                st.session_state.chat_history.append({"role": "assistant", "content": reply_text})
