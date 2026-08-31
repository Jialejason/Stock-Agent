import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import time
import re
from ta.trend import EMAIndicator, SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
import google.generativeai as genai

st.set_page_config(page_title="投资小助手 Pro", layout="centered")
st.title("📈 投资小助手 Pro (多周期全维量化版)")
st.caption("⚡ 5分钟全网共享缓存 ｜ 🧱 密集筹码阻力/支撑 ｜ ⚖️ 日内持仓成本(VWAP) ｜ 🎯 阶梯止盈与分批建仓")

# 1. 别名映射与 Markdown 渲染
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
    if not text: return
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

# 获取 API Key
api_key = st.secrets.get("GEMINI_API_KEY", "") if "GEMINI_API_KEY" in st.secrets else ""
if not api_key:
    with st.expander("🔑 配置 Gemini API Key", expanded=False):
        api_key = st.text_input("Gemini API Key", type="password", help="从 aistudio.google.com 获取")

# 筹码分布计算 (Volume Profile)
def calculate_volume_profile(df_daily, bins=20):
    price_min = df_daily['Low'].min()
    price_max = df_daily['High'].max()
    if price_max == price_min:
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

# AI 操盘与筹码/VWAP推演生成
def get_ai_analysis(ticker_input, cur_price, market_status, vix_status_str, tnx_status_str, 
                    weekly_status, pit_status, hourly_status, vwap_price, vwap_status_desc,
                    hourly_suggested_entry, hourly_stop_loss, chip_resistances, chip_supports, 
                    earnings_date_str, news_text, high_30d, ema20, api_key_val):
    
    chip_res_str = "、".join([f"${p:.2f}" for p in chip_resistances]) if chip_resistances else f"${high_30d:.2f}"
    chip_sup_str = "、".join([f"${p:.2f}" for p in chip_supports]) if chip_supports else f"${hourly_suggested_entry:.2f}"

    prompt = f"""
    你是一名顶级美股操盘手兼新手导师。核心宗旨是【大道至简、精准给出点位、绝不模糊】。
    请基于以下筹码分布、日内平均持仓成本(VWAP)与多周期量化指标，为小白制定极其精准的【分批建仓吸筹 + 阶梯止盈清仓】实操手册。

    【股票标的】: {ticker_input} ｜ 最新价: ${cur_price:.2f}
    【宏观大盘】: {market_status} (VIX: {vix_status_str})
    【🧭 周线大趋势】: {weekly_status}
    【🧱 日线形态雷达】: {pit_status}
    【⚖️ 日内持仓成本线 (VWAP)】: ${vwap_price:.2f} ({vwap_status_desc})
    【🎯 1小时盘中挂单参考】: ${hourly_suggested_entry:.2f} ｜ 止损防线: ${hourly_stop_loss:.2f}
    【🧱 历史密集筹码套牢阻力峰】: {chip_res_str}
    【🛡️ 主力筑底吸筹强支撑带】: {chip_sup_str}
    【财报与资讯】: {earnings_date_str} ｜ {news_text if news_text else "无突发新闻"}

    【严格输出要求】：
    1. 严禁出现断裂星号，所有价格数字统一规范加粗（如 **$18.44**）。
    2. 必须把【日内持仓成本线 VWAP: **${vwap_price:.2f}**】作为日内强弱与短线挂单的生命线进行提示。

    请按以下 3 个板块输出：
    1. 🚦 **多周期共振与日内成本定性（红绿灯）**：2句话讲清大趋势是顺风还是逆风？日内多空成本(VWAP)谁占优势？
    2. 💡 **小白实操动作（直接给精确数字与执行比例）**：
       - **买入与分批建仓策略**：当前能否买入？若观望，给出收复 VWAP 及右侧确认价格；若分批吸筹，给出【头仓试错价】与【回调加仓拉低均价区间】。
       - **阶梯止盈与清仓规划**：
         - **第一目标止盈位**：触及哪个精确价格减仓 1/3 锁定利润？
         - **高位清仓/极值阻力位**：反弹触及哪个历史密集套牢峰价格必须果断清仓离场？
       - **铁血止损底线**：跌破哪个精确价格无条件止损？
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

    # 本地规则保底引擎
    is_bear = cur_price < ema20 or "逆风" in weekly_status
    decision = "🔴 **暂不可追买，严格保持空仓防守！**" if is_bear else "🟢 **多头共振，允许分批建仓试错！**"
    target1 = chip_resistances[0] if chip_resistances else high_30d
    target_clear = chip_resistances[-1] if len(chip_resistances) > 1 else (target1 * 1.15)
    buy_dip = chip_supports[0] if chip_supports else hourly_suggested_entry

    return f"""
1. 🚦 **多周期共振与日内成本定性（红绿灯）**：
当前大周期处于逆风整固阶段，且股价位于日内持仓成本线 (VWAP: **${vwap_price:.2f}**) 附近震荡，场内多空博弈激烈，不可盲目重仓追高。

2. 💡 **小白实操动作（直接给数字）**：
- **买入与建仓策略**：{decision}
  - **短线/右侧确认条件**：需日内稳稳站上日内成本线 **${vwap_price:.2f}** 并放量突破阻力 **${target1:.2f}** 方可右侧跟进。
  - **分批吸筹拉低均价**：若看好中长线，可在回调至支撑带 **${buy_dip:.2f}** 附近分批挂单 10%~20% 底仓。
- **阶梯止盈与清仓规划**：
  - **第一目标位（减仓 1/3）**：反弹触及短线密集筹码阻力 **${target1:.2f}** 必须主动减仓锁定利润。
  - **密集套牢清仓位（清仓离场）**：强力反弹至上方历史筹码密集大顶 **${target_clear:.2f}** 区域全部落袋离场。
- **铁血止损底线**：一旦介入，跌破防守位 **${hourly_stop_loss:.2f}** 坚决止损！

3. ⚠️ **最核心的一个避险坑**：股价在日内持仓成本线 **${vwap_price:.2f}** 下方运行时属于空头占优，切勿在均线下方盲目加仓。
"""

# AI 深度问答保底
def fallback_smart_chat(prompt_text, curr_ticker, cur_price, data, compared_ticker_data=None):
    res_top = data['chip_resistances'][0] if data['chip_resistances'] else data['cur_price'] * 1.1
    res_high = data['chip_resistances'][-1] if len(data['chip_resistances']) > 1 else res_top * 1.15
    sup_dip = data['chip_supports'][0] if data['chip_supports'] else data['hourly_suggested_entry']
    vwap_p = data['vwap_price']
    
    if compared_ticker_data:
        comp_sym = compared_ticker_data['symbol']
        comp_price = compared_ticker_data['cur_price']
        return f"""
针对 **{curr_ticker}**（现价 **${cur_price:.2f}**）与 **{comp_sym}**（现价 **${comp_price:.2f}**）对比：

1. **筹码与日内成本优劣**：
   - **{curr_ticker}**：日内持仓成本线为 **${vwap_p:.2f}**，上方密集套牢阻力在 **${res_top:.2f}**。
   - **{comp_sym}**：周线处于 `{compared_ticker_data['weekly_status']}`。
2. **实操策略**：优先选择站稳日内 VWAP 且上方无重度筹码堆积的右侧品种，弱势标的仅可在支撑位分批低吸。
"""

    if "成本" in prompt_text or "VWAP" in prompt_text or "日内" in prompt_text:
        return f"""
关于 **{curr_ticker}** 日内持仓成本（VWAP: **${vwap_p:.2f}**）与股价关联解析：

1. **多空分界意义**：**${vwap_p:.2f}** 是今天全市场所有买家成交的平均价格。
   - **现价高于 VWAP**：日内持筹者整体处于**盈利状态**，抛压小，容易顺势上攻；
   - **现价低于 VWAP**：日内持筹者整体处于**被套状态**，反弹至 **${vwap_p:.2f}** 会遭遇解套抛压。
2. **实战应用**：日内做 T 或买入时，以 **${vwap_p:.2f}** 作为生命线。站稳 VWAP 逢低低吸，跌破 VWAP 不急于伸手。
"""
    elif "筹码" in prompt_text or "阻力" in prompt_text or "止盈" in prompt_text or "清仓" in prompt_text:
        return f"""
关于 **{curr_ticker}**（现价 **${cur_price:.2f}**）的密集筹码分布与止盈/清仓点位：

1. 🧱 **第一密集筹码阻力位**：**${res_top:.2f}**（建议减仓 1/3~1/2 锁定利润）；
2. 🚨 **历史重度套牢清仓位**：**${res_high:.2f}**（大幅冲高至该区间建议全额清仓或仅留利润底仓）。
"""
    elif "建仓" in prompt_text or "加仓" in prompt_text or "拉低均价" in prompt_text:
        return f"""
关于 **{curr_ticker}**（现价 **${cur_price:.2f}**）的分批吸筹与拉低均价指南：

1. 🎯 **头仓试错位（20% 仓位）**：回踩 **${data['hourly_suggested_entry']:.2f}** 或站稳 VWAP **${vwap_p:.2f}** 时小仓介入；
2. 💰 **二次吸筹加仓位（30% 仓位）**：回调至主力支撑带 **${sup_dip:.2f}** 缩量企稳时补仓；
3. 🛑 **风控底线**：跌破 **${data['hourly_stop_loss']:.2f}** 立即停止加仓并止损。
"""
    else:
        return f"""
基于 **{curr_ticker}**（现价 **${cur_price:.2f}**）的量化推演：

- **日内持仓成本线 (VWAP)**：**${vwap_p:.2f}**
- **密集阻力区**：**${res_top:.2f}** ｜ **高位筹码峰**：**${res_high:.2f}**
- **主力支撑区**：**${sup_dip:.2f}** ｜ **防守底线**：**${data['hourly_stop_loss']:.2f}**
- **操作要点**：严格依托 VWAP 和筹码支撑阻力位分批挂单执行。
"""

# 2. 核心量化算法（集成 Volume Profile + VWAP 算法）
@st.cache_data(ttl=300, show_spinner=False)
def fetch_and_analyze(ticker_input, api_key_val):
    ticker_input = ticker_input.strip().upper()
    
    # 宏观大盘监控
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
    except Exception:
        pass

    # 日线与日内分时获取
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
    cur_price = close_d.iloc[-1]
    total_days = len(close_d)

    # 计算 VWAP (日内成交量加权平均价)
    vwap_price = cur_price
    vwap_status_desc = "持平"
    try:
        df_intraday = yf.download(ticker_input, period="1d", interval="5m", progress=False)
        if not df_intraday.empty:
            if isinstance(df_intraday.columns, pd.MultiIndex):
                df_intraday.columns = df_intraday.columns.get_level_values(0)
            typical_p = (df_intraday['High'] + df_intraday['Low'] + df_intraday['Close']) / 3.0
            valid_vol = df_intraday['Volume']
            if valid_vol.sum() > 0:
                vwap_price = (typical_p * valid_vol).sum() / valid_vol.sum()
        else:
            # 若非开盘时间，用最近一日典型价格模拟
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
    has_ma60 = total_days >= 60
    ma60 = SMAIndicator(close_d, 60).sma_indicator().iloc[-1] if has_ma60 else None
    
    rsi_d = RSIIndicator(close_d, min(14, total_days)).rsi().iloc[-1]
    macd_diff_d = MACD(close_d).macd_diff().iloc[-1]
    atr_d = AverageTrueRange(high_d, low_d, close_d, min(14, total_days)).average_true_range().iloc[-1]
    
    cur_vol = vol_d.iloc[-1]
    avg_vol_5d = vol_d.iloc[-6:-1].mean() if total_days >= 6 else vol_d.mean()
    vol_ratio = (cur_vol / avg_vol_5d) if avg_vol_5d > 0 else 1.0

    high_30d = high_d.iloc[-min(30, total_days):].max()
    low_30d = low_d.iloc[-min(30, total_days):].min()

    # VPVR 筹码分布提取
    vp = calculate_volume_profile(df_daily, bins=20)
    chip_resistances = []
    chip_supports = []
    if vp:
        sorted_vp = sorted(vp, key=lambda x: x[1], reverse=True)
        top_bins = sorted_vp[:6]
        res_bins = sorted([p for p, _ in top_bins if p > cur_price * 1.015])
        sup_bins = sorted([p for p, _ in top_bins if p < cur_price * 0.985], reverse=True)
        chip_resistances = [round(p, 2) for p in res_bins[:2]]
        chip_supports = [round(p, 2) for p in sup_bins[:2]]

    if not chip_resistances: chip_resistances = [round(high_30d, 2)]
    if not chip_supports: chip_supports = [round(low_30d, 2)]

    # 动态支撑阻力列表
    resistance_list = []
    support_list = []
    
    if cur_price < vwap_price:
        resistance_list.append(f"日内持仓成本线压制 (VWAP): ${vwap_price:.2f}")
    else:
        support_list.append(f"日内持仓成本线支撑 (VWAP): ${vwap_price:.2f}")

    if ema5 > cur_price: resistance_list.append(f"短线均线压制 (EMA5): ${ema5:.2f}")
    else: support_list.append(f"超短支撑 (EMA5): ${ema5:.2f}")

    if ema10 > cur_price: resistance_list.append(f"过渡均线压制 (EMA10): ${ema10:.2f}")
    else: support_list.append(f"过渡防守 (EMA10): ${ema10:.2f}")

    if ema20 > cur_price: resistance_list.append(f"多空分水岭压制 (EMA20): ${ema20:.2f}")
    else: support_list.append(f"多空分水岭支撑 (EMA20): ${ema20:.2f}")

    for idx, cp in enumerate(chip_resistances):
        resistance_list.append(f"🧱 密集筹码套牢阻力峰 #{idx+1}: ${cp:.2f}")

    for idx, sp in enumerate(chip_supports):
        support_list.append(f"🛡️ 主力筑底筹码支撑带 #{idx+1}: ${sp:.2f}")

    # 形态定性
    pit_status = "正常走势"
    if rsi_d < 38 and cur_price <= low_30d * 1.03:
        pit_status = "💎 极端超卖黄金坑：指标极度冰点超跌，存在高盈亏比反转机会！"
    elif cur_price >= ema20 and macd_diff_d > 0 and vol_ratio >= 0.9:
        pit_status = "🧱 右侧企稳确立：重回EMA20均线之上，动能回暖，结构扎实。"
    elif cur_price < ema20 and vol_ratio < 0.7:
        pit_status = "🧊 缩量磨底中：跌破均线但抛压衰竭，等待放量企稳确认。"

    # 周线数据
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
                    weekly_status = "🟢 顺风大牛势：站稳EMA20生命线且动能向上！"
                elif cur_price >= w_ema20:
                    weekly_status = "🟡 强势震荡：位于均线上方，高位整固阶段。"
                else:
                    weekly_status = "🔴 逆风熊势/深度调整：周线跌破EMA20，大趋势向下！"
    except Exception:
        pass

    # 1小时盘中数据
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
                hourly_stop_loss = min(h_recent_low * 0.995, cur_price * 0.98)
                if cur_price >= h_ema20 and 45 <= h_rsi <= 65:
                    hourly_status = "🎯 盘中狙击买点已触发：1小时回踩企稳！"
                elif h_rsi >= 70:
                    hourly_status = "⚠️ 1小时短线超买：切勿追高，等待盘中回踩！"
                elif cur_price < h_ema20:
                    hourly_status = "🧊 1小时盘中偏弱：等待重回1小时EMA20均线。"
    except Exception:
        pass

    earnings_date_str = "暂无近期数据"
    try:
        cal = ticker_obj.get_calendar()
        if cal and 'Earnings Date' in cal and len(cal['Earnings Date']) > 0:
            e_date = cal['Earnings Date'][0]
            if isinstance(e_date, (datetime, pd.Timestamp)):
                days_to_e = (e_date.date() - datetime.now().date()).days
                earnings_date_str = f"{e_date.strftime('%Y-%m-%d')} (距今 {days_to_e} 天)"
    except Exception:
        pass
    
    news_text = ""
    try:
        news_list = ticker_obj.news
        if news_list and len(news_list) > 0:
            for item in news_list[:3]:
                title = item.get('title') or (item.get('content', {}).get('title') if isinstance(item.get('content'), dict) else '')
                if title: news_text += f"- {title}\n"
    except Exception:
        pass

    top_faqs = [
        f"🧱 {ticker_input} 上方最硬的筹码套牢阻力是多少？该在哪个价位分批止盈/清仓？",
        f"⚖️ 当前股价与日内持仓成本线 (VWAP: ${vwap_price:.2f}) 的关系说明了什么？",
        f"💰 若看好 {ticker_input}，怎么在支撑位分批建头仓并拉低均价？"
    ]

    now_utc = datetime.now(timezone.utc)
    local_time_str = (now_utc + timedelta(hours=8)).strftime("%H:%M:%S")
    et_time_str = (now_utc - timedelta(hours=4)).strftime("%H:%M:%S")
    cache_display_time = f"{local_time_str} 本地 ｜ {et_time_str} 美东"

    ai_analysis_text = get_ai_analysis(ticker_input, cur_price, market_status, vix_status_str, 
                                       tnx_status_str, weekly_status, pit_status, hourly_status, 
                                       vwap_price, vwap_status_desc, hourly_suggested_entry, 
                                       hourly_stop_loss, chip_resistances, chip_supports, 
                                       earnings_date_str, news_text, high_30d, ema20, api_key_val)

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
        "vwap_price": vwap_price,
        "vwap_status_desc": vwap_status_desc,
        "hourly_suggested_entry": hourly_suggested_entry,
        "hourly_stop_loss": hourly_stop_loss,
        "chip_resistances": chip_resistances,
        "chip_supports": chip_supports,
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

    with st.spinner(f"正在全维运算周线/日线/筹码/VWAP数据 ({ticker_input})..."):
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
    st.write(f"- ⚖️ **日内持仓成本线 (VWAP):** `${data['vwap_price']:.2f}` ({data['vwap_status_desc']})")
    
    col_m1, col_m2 = st.columns(2)
    col_m1.metric(label=f"{curr_ticker} 最新价", value=f"${data['cur_price']:.2f}")
    vol_status = "🔥 放量" if data['vol_ratio'] > 1.3 else "🧊 缩量" if data['vol_ratio'] < 0.7 else "⚖️ 平量"
    col_m2.metric(label="5日量比", value=f"{data['vol_ratio']:.2f} 倍", delta=vol_status)

    st.subheader("🤖 Gemini 操盘手行动指令 (筹码+VWAP阶梯买卖规划)")
    safe_render_markdown(data['ai_analysis_text'])

    st.subheader("🛡️ 阶梯支撑与动态阻力看板 (含密集筹码峰与VWAP)")
    col1, col2 = st.columns(2)
    with col1:
        st.info("**【阶梯防守与吸筹支撑】**\n\n" + "\n\n".join(data['support_list']))
    with col2:
        st.warning("**【动态压制与密集套牢阻力】**\n\n" + "\n\n".join(data['resistance_list']))
    
    st.subheader("⚡ 动能与量价特征")
    macd_str = "🟢 多头金叉（动能充沛）" if data['macd_diff_d'] > 0 else "🔴 动能减弱/死叉休整"
    st.write(f"- **MACD 状态:** `{macd_str}` (柱值: {data['macd_diff_d']:.2f})")
    st.write(f"- **RSI (14):** `{data['rsi_d']:.2f}` ({'⚠️ 超买' if data['rsi_d'] > 70 else '💎 极端超卖/黄金坑区' if data['rsi_d'] < 38 else '⚖️ 中性'})")
    st.write(f"- **日均真实波幅 (ATR):** `${data['atr_d']:.2f}`")

    # 5. 专属 AI 操盘助理（筹码 + VWAP 推演版）
    st.divider()
    st.subheader("💬 对当前诊断有疑问？随时追问 AI 助理")
    st.caption(f"💡 AI 已深度挂载 {curr_ticker} 的筹码分布、日内平均成本线(VWAP)与多周期指标。")

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

    user_input = st.chat_input(f"问问关于 {curr_ticker}（如日内VWAP成本、密集筹码阻力）或对比其他股票...")
    prompt_to_process = user_input or clicked_faq

    if prompt_to_process:
        st.session_state.chat_history.append({"role": "user", "content": prompt_to_process})
        with st.chat_message("user"):
            safe_render_markdown(prompt_to_process)

        with st.chat_message("assistant"):
            with st.spinner("AI 操盘手正在针对筹码分布与日内成本深度推演解答..."):
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
                                现价: ${other_data['cur_price']:.2f}, VWAP: ${other_data['vwap_price']:.2f}, 周线趋势: {other_data['weekly_status']}, 日线形态: {other_data['pit_status']}
                                关键阻力: {'; '.join(other_data['resistance_list'][:2])}, 关键支撑: {'; '.join(other_data['support_list'][:2])}
                                """
                        except Exception:
                            pass

                if api_key:
                    genai.configure(api_key=api_key)
                    models_to_try = ['gemini-1.5-flash', 'gemini-2.0-flash', 'gemini-1.5-pro']
                    
                    context_prompt = f"""
                    你是一名顶级的资深美股操盘手与量化交易导师。你的核心优势是【结合筹码峰、日内持仓成本VWAP与多周期指标给出精确价格与执行动作】。
                    
                    【当前标的】: {curr_ticker} ｜ 现价: ${data['cur_price']:.2f}
                    【宏观大盘】: {data['market_status']}
                    【日内持仓成本线 (VWAP)】: ${data['vwap_price']:.2f} ({data['vwap_status_desc']})
                    【周线大趋势】: {data['weekly_status']} ｜ 日线形态: {data['pit_status']}
                    【密集筹码阻力峰】: {', '.join([f'${p:.2f}' for p in data['chip_resistances']])}
                    【主力吸筹支撑带】: {', '.join([f'${p:.2f}' for p in data['chip_supports']])}
                    【动态阻力与目标】: {'; '.join(data['resistance_list'])}
                    【阶梯防守支撑】: {'; '.join(data['support_list'])}
                    【1小时盘中挂单参考】: ${data['hourly_suggested_entry']:.2f} ｜ 止损: ${data['hourly_stop_loss']:.2f}
                    {extra_data_text}

                    用户的具体提问是: "{prompt_to_process}"

                    【严格作答要求】：
                    1. 严禁使用笼统模版！直接针对用户的问题作答（涉及日内强弱必提 VWAP **${data['vwap_price']:.2f}**；涉及筹码阻力必提筹码峰并给减仓/清仓点；涉及加仓建仓必给分批拉低均价区间）。
                    2. 所有价格数字必须紧跟美元符号规范加粗（例如 **$18.40**）。
                    3. 直接给点位、比例与执行动作，语言精炼直白。
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

                if not reply_text:
                    reply_text = fallback_smart_chat(prompt_to_process, curr_ticker, data['cur_price'], data, compared_ticker_data)

                safe_render_markdown(reply_text)
                st.session_state.chat_history.append({"role": "assistant", "content": reply_text})
