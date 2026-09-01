import json
import re
import threading
import time
from datetime import datetime, timedelta, timezone

import google.generativeai as genai
import numpy as np
import pandas as pd
import requests
import streamlit as st
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD, SMAIndicator
from ta.volatility import AverageTrueRange
import yfinance as yf

# 尝试导入 Moomoo API
try:
    from moomoo import OpenSecTradeContext, TrdEnv, TrdMarket, TrdSide
    MOOMOO_AVAILABLE = True
except ImportError:
    MOOMOO_AVAILABLE = False

st.set_page_config(page_title="专属私人 AI 投顾 Pro Max", layout="wide")
st.title("🛡️ 专属私人 AI 量化投顾系统 Pro Max")
st.caption("⚡ Moomoo 持仓直连 ｜ 📊 华尔街全景基本面 ｜ 🛰️ 盘中量化盯盘 ｜ 🎯 阶梯止盈止损 ｜ 📲 Pushover 实时告警")

# ----------------------------------------------------
# 1. 基础配置与 Pushover 推送模块
# ----------------------------------------------------
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
    st.markdown(text.replace("$", "\\$"))

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

raw_api_key = st.secrets.get("GEMINI_API_KEY", "") if "GEMINI_API_KEY" in st.secrets else ""
api_key = raw_api_key.strip().replace("\n", "").replace("\r", "").replace(" ", "")

PUSHOVER_USER_KEY = st.secrets.get("PUSHOVER_USER_KEY", "") if "PUSHOVER_USER_KEY" in st.secrets else ""
PUSHOVER_API_TOKEN = st.secrets.get("PUSHOVER_API_TOKEN", "") if "PUSHOVER_API_TOKEN" in st.secrets else ""

def send_pushover_alert(message, title="🛡️ 美股投顾量化预警", priority=0, sound="cashregister"):
    if not PUSHOVER_USER_KEY or not PUSHOVER_API_TOKEN:
        return False
    try:
        url = "https://api.pushover.net/1/messages.json"
        clean_msg = message.replace("*", "").replace("`", "")
        payload = {
            "token": PUSHOVER_API_TOKEN,
            "user": PUSHOVER_USER_KEY,
            "title": title,
            "message": clean_msg,
            "priority": priority,
            "sound": sound
        }
        res = requests.post(url, data=payload, timeout=5)
        return res.status_code == 200
    except Exception:
        return False

def call_gemini_smart(prompt_text):
    if not api_key:
        return "⚠️ 未检测到 API Key，请在 Streamlit Secrets 中配置 `GEMINI_API_KEY`。"
    
    try:
        genai.configure(api_key=api_key)
        candidate_models = ['gemini-2.0-flash', 'gemini-1.5-flash', 'gemini-1.5-flash-latest', 'gemini-1.5-pro', 'gemini-pro']
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

# ----------------------------------------------------
# 2. Moomoo 账户直连模块
# ----------------------------------------------------
def fetch_moomoo_positions(host="127.0.0.1", port=11111):
    if not MOOMOO_AVAILABLE:
        return None, "未检测到 moomoo-api 库，请在终端执行 `pip install moomoo-api`。"
    
    try:
        trd_ctx = OpenSecTradeContext(filter_trdmarket=TrdMarket.US, host=host, port=port, is_acc_need_decrypt=False)
        ret, data = trd_ctx.position_list_query(trd_env=TrdEnv.REAL)
        trd_ctx.close()
        
        if ret != 0:
            return None, f"Moomoo 网关返回错误: {data}"
        
        if data.empty:
            return pd.DataFrame(), None
            
        clean_df = pd.DataFrame({
            'symbol': data['code'].str.replace('US.', '', regex=False).str.replace('.US', '', regex=False),
            'qty': data['qty'],
            'can_sell_qty': data['can_sell_qty'],
            'cost_price': data['cost_price'],
            'nominal_price': data['nominal_price'],
            'pl_val': data['pl_val'],
            'pl_ratio': data['pl_ratio']
        })
        return clean_df, None
    except Exception as e:
        return None, f"连接 Moomoo OpenD 失败: {e} (请确认本地 OpenD 客户端已启动并处于登录状态)"

# ----------------------------------------------------
# 3. 机构微观结构与基本面数据提取 (Volume Profile, Options, Fundamentals)
# ----------------------------------------------------
def calculate_institutional_volume_profile(df_daily, bins=40):
    if df_daily.empty or 'Close' not in df_daily.columns or 'Volume' not in df_daily.columns:
        return {"poc": 0.0, "vah": 0.0, "val": 0.0, "resistances": [], "supports": []}

    price_min = df_daily['Low'].min()
    price_max = df_daily['High'].max()
    if price_max <= price_min or np.isnan(price_min) or np.isnan(price_max):
        return {"poc": 0.0, "vah": 0.0, "val": 0.0, "resistances": [], "supports": []}

    bin_edges = np.linspace(price_min, price_max, bins + 1)
    vol_profile = np.zeros(bins)

    for _, row in df_daily.iterrows():
        mid_p = (row['High'] + row['Low'] + row['Close']) / 3.0
        b_idx = max(0, min(bins - 1, int(np.digitize(mid_p, bin_edges) - 1)))
        vol_profile[b_idx] += row['Volume']

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    poc_price = float(bin_centers[np.argmax(vol_profile)])

    total_vol = vol_profile.sum()
    target_vol = total_vol * 0.70
    sorted_indices = np.argsort(vol_profile)[::-1]
    accum_vol, va_indices = 0, []
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

        opt_chain = ticker_obj.option_chain(expirations[0])
        calls, puts = opt_chain.calls, opt_chain.puts

        total_call_oi = calls['openInterest'].fillna(0).sum()
        total_put_oi = puts['openInterest'].fillna(0).sum()
        pcr = float(total_put_oi / total_call_oi) if total_call_oi > 0 else 1.0

        call_wall = float(calls.loc[calls['openInterest'].idxmax()]['strike']) if not calls.empty and calls['openInterest'].sum() > 0 else 0.0
        put_wall = float(puts.loc[puts['openInterest'].idxmax()]['strike']) if not puts.empty and puts['openInterest'].sum() > 0 else 0.0

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

def fetch_fundamental_and_analyst_data(ticker_obj):
    """提取对标 Moomoo 的财务数据、做空比例与华尔街分析师共识"""
    try:
        info = ticker_obj.info
        def fmt_curr(val):
            if val is None or not isinstance(val, (int, float)): return "N/A"
            if abs(val) >= 1e9: return f"${val/1e9:.2f}B"
            if abs(val) >= 1e6: return f"${val/1e6:.2f}M"
            return f"${val:.2f}"

        return {
            "market_cap": fmt_curr(info.get("marketCap")),
            "total_cash": fmt_curr(info.get("totalCash")),
            "operating_cash_flow": fmt_curr(info.get("operatingCashflow")),
            "net_income_ttm": fmt_curr(info.get("netIncomeToCommon")),
            "pe_ttm": f"{info.get('trailingPE', 0):.2f}" if info.get('trailingPE') else "亏损 / N/A",
            "ps_ttm": f"{info.get('priceToSalesTrailing12Months', 0):.2f}" if info.get('priceToSalesTrailing12Months') else "N/A",
            "short_ratio_float": f"{info.get('shortPercentOfFloat', 0) * 100:.2f}%" if info.get('shortPercentOfFloat') else "N/A",
            "short_days_to_cover": f"{info.get('shortRatio', 0):.2f}" if info.get('shortRatio') else "N/A",
            "target_mean": info.get("targetMeanPrice"),
            "target_high": info.get("targetHighPrice"),
            "target_low": info.get("targetLowPrice"),
            "recommendation_key": info.get("recommendationKey", "N/A").upper().replace("_", " "),
            "num_analysts": info.get("numberOfAnalystOpinions", 0)
        }
    except Exception:
        return {
            "market_cap": "N/A", "total_cash": "N/A", "operating_cash_flow": "N/A", "net_income_ttm": "N/A",
            "pe_ttm": "N/A", "ps_ttm": "N/A", "short_ratio_float": "N/A", "short_days_to_cover": "N/A",
            "target_mean": None, "target_high": None, "target_low": None,
            "recommendation_key": "N/A", "num_analysts": 0
        }

# ----------------------------------------------------
# 4. 全维量化与 AI 专属投顾诊断 (升华版)
# ----------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_and_analyze(ticker_input, user_cost=0.0, user_qty=0):
    ticker_input = ticker_input.strip().upper()
    
    macro_tickers = ["SPY", "QQQ", "^VIX", "^TNX"]
    macro_data = yf.download(macro_tickers, period="3mo", interval="1d", auto_adjust=True, progress=False)
    
    market_status = "🟢 多头顺风：标普(SPY) 与 纳指(QQQ) 稳居 EMA20 上方。"
    spy_info_str, qqq_info_str = "SPY: 正常", "QQQ: 正常"
    vix_status_str, tnx_status_str = "正常", "正常"
    macro_sentiment_tag = "🟢 情绪向好"
    
    try:
        if not macro_data.empty:
            close_data = macro_data['Close']
            spy_c = close_data['SPY'].dropna()
            spy_close = spy_c.iloc[-1]
            spy_prev = spy_c.iloc[-2] if len(spy_c) >= 2 else spy_close
            spy_chg = (spy_close - spy_prev) / spy_prev
            spy_ema20 = EMAIndicator(spy_c, 20).ema_indicator().iloc[-1]
            spy_info_str = f"SPY: ${spy_close:.2f} ({spy_chg*100:+.2f}%)"
            
            qqq_c = close_data['QQQ'].dropna()
            qqq_close = qqq_c.iloc[-1]
            qqq_prev = qqq_c.iloc[-2] if len(qqq_c) >= 2 else qqq_close
            qqq_chg = (qqq_close - qqq_prev) / qqq_prev
            qqq_ema20 = EMAIndicator(qqq_c, 20).ema_indicator().iloc[-1]
            qqq_info_str = f"QQQ: ${qqq_close:.2f} ({qqq_chg*100:+.2f}%)"
            
            vix_close = close_data['^VIX'].dropna().iloc[-1]
            vix_status_str = f"⚠️ 恐慌高企 (VIX={vix_close:.2f})" if vix_close > 22 else f"🟢 恐慌平稳 (VIX={vix_close:.2f})"
            tnx_close = close_data['^TNX'].dropna().iloc[-1]
            tnx_status_str = f"10Y美债收益率: {tnx_close:.2f}%"

            if vix_close >= 25:
                macro_sentiment_tag = "🔴 极端恐慌避险"
            elif vix_close <= 15:
                macro_sentiment_tag = "🔥 极度贪婪活跃"
            else:
                macro_sentiment_tag = "🟢 平稳健康"

            if (spy_close < spy_ema20 and qqq_close < qqq_ema20) or vix_close >= 25:
                market_status = "🔴 极度预警：标普与纳指双双破位EMA20，全市场防守！"
            elif spy_close < spy_ema20:
                market_status = "⚠️ 警示：标普(SPY) 跌破生命线，传统权重股走弱！"
            elif qqq_close < qqq_ema20:
                market_status = "⚠️ 警示：纳指(QQQ) 跌破生命线，科技成长股短线承压！"
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
    cur_price = close_d.iloc[-1]
    total_days = len(close_d)

    low_30d = low_d.iloc[-min(30, total_days):].min()
    high_30d = high_d.iloc[-min(30, total_days):].max()

    # 日内 VWAP 计算
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

    vwap_status_desc = "多头主导(高于日内成本)" if cur_price > vwap_price * 1.002 else "空头压制(低于日内成本)" if cur_price < vwap_price * 0.998 else "多空平衡(紧贴成本)"

    # 全维度均线体系 (对标 Moomoo)
    ema5 = EMAIndicator(close_d, min(5, total_days)).ema_indicator().iloc[-1]
    ema10 = EMAIndicator(close_d, min(10, total_days)).ema_indicator().iloc[-1]
    ema20 = EMAIndicator(close_d, min(20, total_days)).ema_indicator().iloc[-1]
    ma30 = SMAIndicator(close_d, min(30, total_days)).sma_indicator().iloc[-1]
    ma50 = SMAIndicator(close_d, min(50, total_days)).sma_indicator().iloc[-1]
    ma60 = SMAIndicator(close_d, min(60, total_days)).sma_indicator().iloc[-1]
    ma100 = SMAIndicator(close_d, 100).sma_indicator().iloc[-1] if total_days >= 100 else None
    ma200 = SMAIndicator(close_d, 200).sma_indicator().iloc[-1] if total_days >= 200 else None

    # 年线与均线死叉/金叉状态
    cross_status = "中性排列"
    if ma50 and ma200:
        if ma50 < ma200: cross_status = "🔴 50日与200日呈现死亡交叉（长期下行趋势压制）"
        else: cross_status = "🟢 50日与200日呈现黄金交叉（长期多头支撑）"

    gap_support = None
    prev_close_p = close_d.iloc[-2] if total_days >= 2 else cur_price
    if total_days >= 2:
        recent_low = low_d.iloc[-1]
        prev_high = high_d.iloc[-2]
        if recent_low > prev_high: gap_support = round(recent_low, 2)
        elif recent_low > prev_close_p: gap_support = round(prev_close_p, 2)

    atr_d = AverageTrueRange(high_d, low_d, close_d, min(14, total_days)).average_true_range().iloc[-1]
    vp_data = calculate_institutional_volume_profile(df_daily.iloc[-min(252, total_days):])
    opt_data = fetch_options_microstructure(ticker_obj, cur_price)
    funda_data = fetch_fundamental_and_analyst_data(ticker_obj)

    target1_p = vp_data["resistances"][0] if vp_data["resistances"] else (ma60 if ma60 and ma60 > cur_price else (high_30d if high_30d > cur_price else cur_price * 1.05))
    dynamic_stop_loss = max(low_30d, cur_price - (1.5 * atr_d))
    reward_space = max(0.01, target1_p - cur_price)
    risk_space = max(0.01, cur_price - dynamic_stop_loss)
    rr_ratio = reward_space / risk_space

    support_dict, resistance_dict = {}, {}
    def add_level(name, val):
        if val and val > 0:
            if val < cur_price: support_dict[name] = val
            else: resistance_dict[name] = val

    add_level("日内成本线 (VWAP)", vwap_price)
    add_level("短线动量 (EMA5)", ema5)
    add_level("短线中继 (EMA10)", ema10)
    add_level("趋势生命线 (EMA20)", ema20)
    add_level("月线防守 (MA30)", ma30)
    add_level("中期趋势 (MA50)", ma50)
    add_level("季线 (MA60)", ma60)
    if ma100: add_level("半年线 (MA100)", ma100)
    if ma200: add_level("年线 (MA200)", ma200)
    if gap_support: add_level("跳空缺口支撑", gap_support)
    if vp_data["poc"] > 0: add_level("筹码密集峰 (POC)", vp_data["poc"])
    if vp_data["vah"] > 0: add_level("价值区上沿 (VAH)", vp_data["vah"])
    if vp_data["val"] > 0: add_level("价值区下沿 (VAL)", vp_data["val"])
    if opt_data["max_pain"] > 0: add_level("期权最大痛点 (Max Pain)", opt_data["max_pain"])
    if opt_data["major_call_wall"] > 0: add_level("期权Call大单阻力墙", opt_data["major_call_wall"])
    if opt_data["major_put_wall"] > 0: add_level("期权Put大单支撑墙", opt_data["major_put_wall"])

    sorted_supports = sorted(support_dict.items(), key=lambda x: x[1], reverse=True)
    sorted_resistances = sorted(resistance_dict.items(), key=lambda x: x[1])
    support_list_fmt = [f"{k}: **${v:.2f}**" for k, v in sorted_supports[:6]]
    resistance_list_fmt = [f"{k}: **${v:.2f}**" for k, v in sorted_resistances[:6]]

    analyst_target_str = f"均价 **${funda_data['target_mean']:.2f}** (最高 **${funda_data['target_high']:.2f}** / 最低 **${funda_data['target_low']:.2f}**)" if funda_data['target_mean'] else "暂无"

    position_context = ""
    if user_qty > 0:
        pnl_pct = ((cur_price - user_cost) / user_cost) * 100
        position_context = f"""
【用户真实持仓数据】：
- 持仓均价: **${user_cost:.2f}** ｜ 持仓股数: **{user_qty} 股**
- 当前浮动盈亏: **{pnl_pct:+.2f}%**
请直接给出持仓处理动作（继续持有/阶梯止盈阻力位/建议移动止损价格）。
"""
    else:
        position_context = "【用户当前状态】：空仓观望中。请结合盈亏比与大盘环境给出右侧开仓信号。"

    layered_prompt = f"""
你是一名兼具「华尔街顶级投研机构分析师」与「资深实盘量化操盘手」特质的导师。
请对标专业投行研报深度与实盘操盘纪律，对输入标的进行深度全维推演。

【标的】: {ticker_input} ｜ 最新现价: **${cur_price:.2f}**
【大盘环境】: {market_status} ｜ {spy_info_str} ｜ {qqq_info_str} ｜ 情绪度: {macro_sentiment_tag} ({vix_status_str})
【经典均线与大级别形态】: EMA5: **${ema5:.2f}** ｜ EMA20生命线: **${ema20:.2f}** ｜ MA50: **${ma50:.2f}** ｜ 年线MA200: {f"${ma200:.2f}" if ma200 else '无'} ｜ 均线交叉形态: {cross_status}
【机构微观结构】: 做市商日内VWAP: **${vwap_price:.2f}** ({vwap_status_desc}) ｜ 筹码中心(POC): **${vp_data['poc']:.2f}** ｜ 价值区(VAL~VAH): **${vp_data['val']:.2f} ~ ${vp_data['vah']:.2f}**
【期权博弈与波动率】: 期权Max Pain: **${opt_data['max_pain']:.2f}** ｜ PCR: **{opt_data['pcr']:.2f}** ｜ Call阻力墙: **${opt_data['major_call_wall']:.2f}** ｜ 14日ATR: **${atr_d:.2f}**
【华尔街基本面与资金链】: 市值: {funda_data['market_cap']} ｜ 现金储备: {funda_data['total_cash']} ｜ 经营现金流: {funda_data['operating_cash_flow']} ｜ TTM净利: {funda_data['net_income_ttm']} ｜ 市销率(P/S): {funda_data['ps_ttm']} ｜ 卖空比例: {funda_data['short_ratio_float']} (覆盖天数: {funda_data['short_days_to_cover']}) ｜ 分析师共识: 【{funda_data['recommendation_key']}】({funda_data['num_analysts']}位分析师, {analyst_target_str})
【动态盈亏比】: **{rr_ratio:.2f} : 1** ｜ 建议动态止损: **${dynamic_stop_loss:.2f}** ｜ 第一阻力目标: **${target1_p:.2f}**
{position_context}

【核心输出原则】：
1. 直接输出最终中文报告，第一行直接从标题或决策灯开始，严禁输出任何思考草稿或自检清单。
2. 彻底说人话，兼顾深度与可执行性。专业术语必须紧跟括号大白话说明。
3. 所有价格与百分比数字统一紧跟美元符号加粗（如 **$17.55**，**+4.39%**）。
4. 严格按照以下 5 个模块输出：

---
### 🚦 1. 操盘手 3 秒极简决策灯
- **核心操作定性**：直接给大白话动作（【🟢 顺势轻仓试探】 / 【🟡 观望等回踩】 / 【🔴 风险过大坚决不追/执行防守】）。
- **一句话定性理由**：结合大盘走势、日内VWAP与当前 **{rr_ratio:.2f}:1** 的盈亏比，讲透为什么现在该买、该卖还是该等。

### 🏢 2. 华尔街基本面排雷与做空博弈透视
- 结合现金流储备、TTM亏损/盈利、做空比例（**{funda_data['short_ratio_float']}**）与分析师共识目标价，定性其基本面是“强催化成长”还是“纯概念泡沫”。
- 指出高做空比例下是否存在逼空潜力或破位补跌风险。

### 🛡️ 3. 跌势与吸筹指南（阶梯防守与止损）
- **第 1 关（短线浅回调加仓点）**：明确指出回踩哪个具体价格（VWAP/EMA20/缺口）可轻仓试探。
- **第 2 关（波段深度吸筹大底）**：万一大盘回调，主力筹码峰(POC)或中线均线在哪个价格可以安全补仓。
- **飞刀熔断防线（硬止损）**：跌破哪个价格（结合 **${dynamic_stop_loss:.2f}**）说明形态破位，必须坚决止损。

### 🎯 4. 涨势与止盈指南（阶梯撤退与顺势推仓）
- **第一目标位（近端阻力锁定利润）**：反弹到哪个阻力位/VAH建议减仓 1/3 ~ 1/2？距离现价空间多少%？
- **顺势爆发加速位**：带量突破哪个价格（结合期权 Call Wall 与年线 MA200）可判定进入主升浪？

### 🧠 5. 交易数学算账与多空指令
- **做空/减仓者**：在哪个反抽压力位（原支撑变阻力处）观察受阻迹象。
- **多头/抄底者**：必须同时满足什么硬性条件（如大盘企稳 + 站稳 VWAP）才能右侧介入。
- **一句话收尾**：犀利定性当前阶段（如：“捡了芝麻丢西瓜，耐心看戏”）。
"""
    ai_analysis_text = call_gemini_smart(layered_prompt)

    top_faqs = [
        f"🎯 结合分析师目标价与年线，{ticker_input} 上方阻力最大的区间在哪里？",
        f"🛡️ 结合做空比例与筹码峰，{ticker_input} 最安全的左侧/右侧买点在哪？",
        f"⚖️ 当前盈亏比 ({rr_ratio:.2f}:1) 下，操盘手建议多头还是空头？"
    ]

    now_utc = datetime.now(timezone.utc)
    cache_display_time = (now_utc + timedelta(hours=8)).strftime("%H:%M:%S")

    return {
        "symbol": ticker_input,
        "cur_price": cur_price,
        "market_status": market_status,
        "spy_info_str": spy_info_str,
        "qqq_info_str": qqq_info_str,
        "macro_sentiment_tag": macro_sentiment_tag,
        "vix_status_str": vix_status_str,
        "tnx_status_str": tnx_status_str,
        "vwap_price": vwap_price,
        "vwap_status_desc": vwap_status_desc,
        "ema5": ema5,
        "ema10": ema10,
        "ema20": ema20,
        "ma30": ma30,
        "ma50": ma50,
        "ma60": ma60,
        "ma100_str": f"${ma100:.2f}" if ma100 else "无",
        "ma200_str": f"${ma200:.2f}" if ma200 else "无",
        "cross_status": cross_status,
        "gap_support": gap_support,
        "atr_d": atr_d,
        "dynamic_stop_loss": dynamic_stop_loss,
        "vp_data": vp_data,
        "opt_data": opt_data,
        "funda_data": funda_data,
        "support_list_fmt": support_list_fmt,
        "resistance_list_fmt": resistance_list_fmt,
        "rr_ratio": rr_ratio,
        "target1_p": target1_p,
        "top_faqs": top_faqs,
        "ai_analysis_text": ai_analysis_text,
        "cache_display_time": cache_display_time
    }, None

# ----------------------------------------------------
# 5. 后台盯盘与风控判定引擎 (联动 Pushover)
# ----------------------------------------------------
if "alert_logs" not in st.session_state:
    st.session_state.alert_logs = []

def run_portfolio_monitor_task(watchlist):
    triggered_alerts = []
    for item in watchlist:
        sym = item['symbol']
        cost = item.get('cost', 0.0)
        qty = item.get('qty', 0)
        
        data, err = fetch_and_analyze(sym, user_cost=cost, user_qty=qty)
        if not data:
            continue
            
        cur_p = data['cur_price']
        vwap_p = data['vwap_price']
        stop_p = data['dynamic_stop_loss']
        target_p = data['target1_p']

        if cur_p < stop_p:
            msg = f"【止损预警】{sym} 跌破量化防线！\n现价: ${cur_p:.2f} 已跌破硬止损位 ${stop_p:.2f}。\n建议：立即执行防守避险。"
            triggered_alerts.append(msg)
            send_pushover_alert(msg, title=f"🚨 止损预警: {sym}", priority=1, sound="falling")
        elif cur_p >= target_p:
            msg = f"【阶梯止盈】{sym} 触及第一阻力目标！\n现价: ${cur_p:.2f} 达到目标 ${target_p:.2f}。\n建议：阶梯止盈锁定利润。"
            triggered_alerts.append(msg)
            send_pushover_alert(msg, title=f"🎯 止盈达成: {sym}", priority=0, sound="cashregister")
        elif cur_p > vwap_p and (cur_p - vwap_p)/vwap_p < 0.015:
            msg = f"【异动提醒】{sym} 站上做市商 VWAP！\n现价: ${cur_p:.2f} 突破 ${vwap_p:.2f}，主力主动护盘。"
            triggered_alerts.append(msg)
            send_pushover_alert(msg, title=f"⚡ 突破提醒: {sym}", priority=0, sound="intermission")

    return triggered_alerts

# ----------------------------------------------------
# 6. UI 侧边栏配置
# ----------------------------------------------------
st.sidebar.header("⚙️ 账户与连接配置")
connect_mode = st.sidebar.radio("持仓数据模式", ["Moomoo OpenD 本地直连", "手动维护 / 模拟持仓"])
portfolio_list = []

if connect_mode == "Moomoo OpenD 本地直连":
    opend_host = st.sidebar.text_input("OpenD IP", value="127.0.0.1")
    opend_port = st.sidebar.number_input("OpenD 端口", value=11111)
    
    if st.sidebar.button("🔄 同步 Moomoo 真实持仓", use_container_width=True):
        with st.spinner("正在直连 Moomoo OpenD 获取持仓..."):
            moo_df, moo_err = fetch_moomoo_positions(host=opend_host, port=opend_port)
            if moo_err:
                st.sidebar.error(moo_err)
            elif moo_df is not None and not moo_df.empty:
                st.session_state.moo_positions = moo_df
                st.sidebar.success(f"成功同步 {len(moo_df)} 只持仓标的！")
            else:
                st.sidebar.info("Moomoo 账户当前无持仓股票。")
    
    if "moo_positions" in st.session_state and not st.session_state.moo_positions.empty:
        df_p = st.session_state.moo_positions
        for _, row in df_p.iterrows():
            portfolio_list.append({
                "symbol": row['symbol'],
                "cost": float(row['cost_price']),
                "qty": int(row['qty']),
                "pl_ratio": float(row['pl_ratio'])
            })
else:
    st.sidebar.write("**手动维护持仓与关注列表:**")
    m_sym = st.sidebar.text_input("股票代码", value="USAR").upper().strip()
    m_cost = st.sidebar.number_input("持仓成本 ($)", value=16.80, step=0.1, format="%.2f")
    m_qty = st.sidebar.number_input("持仓股数", value=100, step=10)

    col_btn1, col_btn2 = st.sidebar.columns(2)
    if col_btn1.button("➕ 保存/更新", use_container_width=True):
        if "manual_positions" not in st.session_state:
            st.session_state.manual_positions = []
        st.session_state.manual_positions = [p for p in st.session_state.manual_positions if p['symbol'] != m_sym]
        st.session_state.manual_positions.append({"symbol": m_sym, "cost": m_cost, "qty": m_qty})
        st.sidebar.success(f"已更新 {m_sym}！")
        st.rerun()

    if col_btn2.button("🗑️ 一键清空", use_container_width=True):
        st.session_state.manual_positions = []
        st.sidebar.warning("持仓列表已全部清空！")
        st.rerun()

    portfolio_list = st.session_state.get("manual_positions", [])

st.sidebar.divider()
st.sidebar.subheader("📲 Pushover 推送状态")
push_status = "🟢 已就绪" if PUSHOVER_USER_KEY and PUSHOVER_API_TOKEN else "⚪ 未配置 (仅面板提示)"
st.sidebar.caption(f"Pushover 状态: {push_status}")

if st.sidebar.button("⚡ 立即全维扫描持仓风控", type="primary", use_container_width=True):
    with st.spinner("量化引擎正在全局扫描持仓攻防点位..."):
        alerts = run_portfolio_monitor_task(portfolio_list)
        if alerts:
            st.session_state.alert_logs.extend(alerts)
            st.sidebar.success(f"扫描完成，触发 {len(alerts)} 条异动预警并推送至手机！")
        else:
            st.sidebar.info("持仓运行平稳，暂未触及止损或止盈触发线。")

# ----------------------------------------------------
# 7. 主界面 (Tab 结构)
# ----------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 私人持仓全景透视", "🔍 单股深度全维诊断", "🚨 实时预警日志"])

# --- TAB 1: 持仓透视 ---
with tab1:
    st.subheader("💼 当前账户实时持仓诊断")
    if portfolio_list:
        p_df = pd.DataFrame(portfolio_list)
        st.dataframe(p_df, use_container_width=True)

        selected_pos_sym = st.selectbox("选择需要投顾深度复盘的持仓:", [p['symbol'] for p in portfolio_list])
        selected_item = next(item for item in portfolio_list if item['symbol'] == selected_pos_sym)

        if st.button(f"生成 {selected_pos_sym} 专属投顾报告", type="primary", key="btn_pos_diag"):
            with st.spinner("专属 AI 投顾正在结合全维研报与操盘纪律推演..."):
                diag_data, d_err = fetch_and_analyze(selected_pos_sym, user_cost=selected_item['cost'], user_qty=selected_item['qty'])
                if d_err:
                    st.error(d_err)
                else:
                    col1, col2, col3, col4, col5 = st.columns(5)
                    col1.metric("现价", f"${diag_data['cur_price']:.2f}")
                    col2.metric("做市商 VWAP", f"${diag_data['vwap_price']:.2f}")
                    col3.metric("筹码中心 (POC)", f"${diag_data['vp_data']['poc']:.2f}")
                    col4.metric("卖空比例", f"{diag_data['funda_data']['short_ratio_float']}")
                    col5.metric("建议防守止损", f"${diag_data['dynamic_stop_loss']:.2f}")

                    st.markdown("---")
                    st.subheader(f"🤖 投顾执行报告 ({selected_pos_sym})")
                    safe_render_markdown(diag_data['ai_analysis_text'])
    else:
        st.info("暂无持仓数据，请在左侧侧边栏同步 Moomoo 或手动录入持仓。")

# --- TAB 2: 单股诊断 ---
with tab2:
    if "history_tickers" not in st.session_state:
        st.session_state.history_tickers = ["NVDA", "USAR", "TSLA", "AAPL"]

    if "selected_ticker" not in st.session_state:
        st.session_state.selected_ticker = "NVDA"

    st.write("**🔥 快速自选与最近查询:**")
    cols = st.columns(len(st.session_state.history_tickers))
    for i, ticker in enumerate(st.session_state.history_tickers):
        if cols[i].button(ticker, key=f"quick_{ticker}", use_container_width=True):
            st.session_state.selected_ticker = ticker

    ticker_input = st.text_input("美股代码", value=st.session_state.selected_ticker).strip().upper()

    if st.button("开始全维实战闭环诊断", type="primary", use_container_width=True):
        if ticker_input and ticker_input in st.session_state.history_tickers:
            st.session_state.history_tickers.remove(ticker_input)
        if ticker_input:
            st.session_state.history_tickers.insert(0, ticker_input)
            if len(st.session_state.history_tickers) > 5:
                st.session_state.history_tickers.pop()

        with st.spinner(f"正在全维运算均线、筹码分布、基本面与衍生品博弈 ({ticker_input})..."):
            data, err = fetch_and_analyze(ticker_input)
            if err:
                st.error(f"❌ {err}")
            else:
                st.session_state.current_data = data
                st.session_state.current_ticker = ticker_input
                st.session_state.chat_history = []

    if "current_data" in st.session_state and st.session_state.current_data:
        data = st.session_state.current_data
        curr_ticker = st.session_state.get("current_ticker", ticker_input)
        fd = data['funda_data']

        st.caption(f"⚡ 数据已智能缓存 (刷新时间: {data['cache_display_time']}) ｜ 5分钟内共享无消耗")
        
        if "🔴" in data['market_status']:
            st.error(f"**大盘风控:** {data['market_status']} ｜ {data['spy_info_str']} ｜ {data['qqq_info_str']}")
        elif "⚠️" in data['market_status']:
            st.warning(f"**大盘风控:** {data['market_status']} ｜ {data['spy_info_str']} ｜ {data['qqq_info_str']}")
        else:
            st.success(f"**大盘风控:** {data['market_status']} ｜ {data['spy_info_str']} ｜ {data['qqq_info_str']}")
        
        st.info(f"🌐 **市场环境：** 【{data['macro_sentiment_tag']}】 ｜ {data['vix_status_str']} ｜ 🏛️ {data['tnx_status_str']} ｜ 均线形态: **{data['cross_status']}**")

        # 核心技术指标栏
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric(label=f"{curr_ticker} 现价", value=f"${data['cur_price']:.2f}")
        col_m2.metric(label="做市商成本 (VWAP)", value=f"${data['vwap_price']:.2f}")
        rr_delta = "🟢 优秀" if data['rr_ratio'] >= 2.0 else "⚠️ 偏低/一般"
        col_m3.metric(label="动态盈亏比", value=f"{data['rr_ratio']:.2f} : 1", delta=rr_delta)
        col_m4.metric(label="动态保护止损 (1.5x ATR)", value=f"${data['dynamic_stop_loss']:.2f}")

        # 基本面与做空全景栏 (对标 Moomoo)
        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
        col_f1.metric(label="总现金储备 / 市值", value=f"{fd['total_cash']} / {fd['market_cap']}")
        col_f2.metric(label="做空比例 (Short Float)", value=f"{fd['short_ratio_float']}")
        col_f3.metric(label="分析师共识评级", value=f"{fd['recommendation_key']} ({fd['num_analysts']}人)")
        target_str = f"${fd['target_mean']:.2f}" if fd['target_mean'] else "N/A"
        col_f4.metric(label="分析师目标均价", value=target_str)

        st.subheader("🤖 操盘手分层实战手册 (研报 + 操盘执行指令)")
        safe_render_markdown(data['ai_analysis_text'])

        st.subheader("🛡️ 全景关键阶梯防线 (均线/年线/缺口/筹码共振)")
        col1, col2 = st.columns(2)
        with col1:
            st.info("**【🟢 阶梯支撑与吸筹带（由近及远）】**\n\n" + "\n\n".join(data['support_list_fmt']))
        with col2:
            st.warning("**【🔴 阶梯阻力与出清目标（由近及远）】**\n\n" + "\n\n".join(data['resistance_list_fmt']))

        st.divider()
        st.subheader("💬 操盘手智能追问助理")
        clicked_faq = None
        if "top_faqs" in data and data["top_faqs"]:
            for idx, faq_text in enumerate(data["top_faqs"]):
                if st.button(faq_text, key=f"faq_tab2_{idx}", use_container_width=True):
                    clicked_faq = faq_text

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                safe_render_markdown(msg["content"])

        user_input = st.chat_input("自由提问（如：有做空逼空可能吗？年线压力怎么看？跌破VWAP怎么办？）...")
        prompt_to_process = user_input or clicked_faq

        if prompt_to_process:
            st.session_state.chat_history.append({"role": "user", "content": prompt_to_process})
            with st.chat_message("user"):
                safe_render_markdown(prompt_to_process)

            with st.chat_message("assistant"):
                with st.spinner("操盘智脑正在结合基本面、年线均线与订单流推演..."):
                    chat_context_prompt = f"""
你是一名顶级美股操盘手兼量化投研导师。
结合以下基本面、微观筹码及均线数据回答用户问题：

【当前标的】: {curr_ticker} ｜ 现价: **${data['cur_price']:.2f}**
【基本面与做空】: 市值: {fd['market_cap']} ｜ 现金: {fd['total_cash']} ｜ 卖空比例: {fd['short_ratio_float']} ｜ 分析师评级: {fd['recommendation_key']} ｜ 目标均价: {fd['target_mean']}
【均线与年线】: EMA5: **${data['ema5']:.2f}** ｜ EMA20: **${data['ema20']:.2f}** ｜ MA50: **${data['ma50']:.2f}** ｜ 年线MA200: {data['ma200_str']} ｜ 形态: {data['cross_status']}
【微观筹码与期权】: 做市商VWAP: **${data['vwap_price']:.2f}** ｜ POC密集峰: **${data['vp_data']['poc']:.2f}** ｜ 期权Max Pain: **${data['opt_data']['max_pain']:.2f}**
【风控基准】: 14日ATR: **${data['atr_d']:.2f}** ｜ 建议动态止损: **${data['dynamic_stop_loss']:.2f}** ｜ 动态盈亏比: **{data['rr_ratio']:.2f} : 1**

用户问题: "{prompt_to_process}"

【回答原则】：
1. 直接输出中文结论，严禁输出任何自检标签或草稿。
2. 算清数学账（百分比空间、止损距离），若涉及基本面与做空比例直接点出其实质影响。
3. 所有涉及的价格数字统一紧跟美元符号加粗（如 **$18.50**）。
"""
                    reply_text = call_gemini_smart(chat_context_prompt)
                    safe_render_markdown(reply_text)
                    st.session_state.chat_history.append({"role": "assistant", "content": reply_text})

# --- TAB 3: 预警日志 ---
with tab3:
    st.subheader("🚨 盘中实时风控与买卖预警日志")
    if st.session_state.alert_logs:
        for log in reversed(st.session_state.alert_logs[-10:]):
            st.warning(log)
    else:
        st.info("暂无最新告警记录。点击左侧【立即全维扫描持仓风控】即可执行一次全局巡检。")
