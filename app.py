import json
import re
import threading
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD, SMAIndicator
from ta.volatility import AverageTrueRange
import yfinance as yf

# ----------------------------------------------------
# 0. Moomoo API 兼容性与安全垫片导入
# ----------------------------------------------------
try:
    from moomoo import (
        OpenSecTradeContext, OpenQuoteContext,
        TrdEnv, TrdMarket, SecurityFirm, TrdSide, OrderType,
        KLType, SubType
    )
    MOOMOO_AVAILABLE = True
except Exception:
    MOOMOO_AVAILABLE = False
    class TrdEnv:
        SIMULATE = "SIMULATE"
        REAL = "REAL"
    class TrdMarket:
        US = "US"
    class SecurityFirm:
        FUTUSECURITIES = "FUTUSECURITIES"
    class TrdSide:
        BUY = "BUY"
        SELL = "SELL"
    class OrderType:
        NORMAL = "NORMAL"
        MARKET = "MARKET"

# 页面基础配置
st.set_page_config(page_title="Moomoo 智能量化交易 & AI投顾终端 Pro Max", page_icon="⚡", layout="wide")
st.title("🛡️ Moomoo 智能量化交易 & AI投顾终端 Pro Max")
st.caption("⚡ Moomoo 自动交易/风控 ｜ 📊 筹码与微观结构 ｜ 🛰️ 机会自动挖掘 ｜ 🎯 阶梯止盈止损 ｜ 📲 Pushover 实时告警")

# ----------------------------------------------------
# 1. 基础配置与最新 Gemini 模型适配引擎
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

raw_api_key = st.secrets.get("GEMINI_API_KEY", "") if "GEMINI_API_KEY" in st.secrets else ""
api_key = str(raw_api_key).strip().replace("\n", "").replace("\r", "").replace(" ", "").replace('"', '').replace("'", "")

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
    
    # 官方要求的最新标准模型队列
    models = [
        "gemini-3.6-flash",
        "gemini-3-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash-latest"
    ]
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key
    }
    payload = {
        "contents": [{
            "parts": [{"text": prompt_text}]
        }]
    }

    last_error = ""
    for m in models:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent"
            resp = requests.post(url, headers=headers, json=payload, timeout=25)
            if resp.status_code == 200:
                res_json = resp.json()
                candidates = res_json.get("candidates", [])
                if candidates and "content" in candidates[0]:
                    parts = candidates[0]["content"].get("parts", [])
                    if parts and "text" in parts[0]:
                        return parts[0]["text"]
            else:
                err_data = resp.json().get("error", {})
                last_error = f"{resp.status_code} - {err_data.get('message', resp.text)}"
        except Exception as e:
            last_error = str(e)
            continue

    return f"⚠️ 智脑调用异常: `{last_error}`"

# ----------------------------------------------------
# 2. Moomoo 账户与交易核心连接模块
# ----------------------------------------------------
@st.cache_resource
def get_moomoo_contexts(host="127.0.0.1", port=11111):
    if not MOOMOO_AVAILABLE:
        return None, None
    try:
        trd = OpenSecTradeContext(
            filter_trdmarket=TrdMarket.US,
            host=host,
            port=port,
            security_firm=SecurityFirm.FUTUSECURITIES
        )
        quote = OpenQuoteContext(host=host, port=port)
        return trd, quote
    except Exception:
        return None, None

def record_trade_log(action, symbol, detail):
    if "trade_audit_logs" not in st.session_state:
        st.session_state.trade_audit_logs = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.trade_audit_logs.insert(0, {"时间": now, "操作": action, "标的": symbol, "详细信息": detail})
    if len(st.session_state.trade_audit_logs) > 50:
        st.session_state.trade_audit_logs.pop()

# ----------------------------------------------------
# 3. 机构微观结构与基本面计算 (Volume Profile, Options, Fundamentals)
# ----------------------------------------------------
def calculate_institutional_volume_profile(df_daily, bins=40):
    if df_daily.empty or 'Close' not in df_daily.columns or 'Volume' not in df_daily.columns:
        return {"poc": 0.0, "vah": 0.0, "val": 0.0, "resistances": [], "supports": []}
    price_min, price_max = df_daily['Low'].min(), df_daily['High'].max()
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

    val_price = float(bin_centers[min(va_indices)]) if va_indices else poc_price
    vah_price = float(bin_centers[max(va_indices)]) if va_indices else poc_price
    cur_price = df_daily['Close'].iloc[-1]
    top_indices = sorted_indices[:8]
    res_bins = sorted([bin_centers[i] for i in top_indices if bin_centers[i] > cur_price * 1.01])
    sup_bins = sorted([bin_centers[i] for i in top_indices if bin_centers[i] < cur_price * 0.99], reverse=True)

    return {
        "poc": poc_price, "vah": vah_price, "val": val_price,
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
        return {"max_pain": float(max_pain), "pcr": float(pcr), "major_call_wall": call_wall, "major_put_wall": put_wall}
    except Exception:
        return {"max_pain": 0.0, "pcr": 1.0, "major_call_wall": 0.0, "major_put_wall": 0.0}

def fetch_fundamental_and_analyst_data(ticker_obj):
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

@st.cache_data(ttl=300, show_spinner=False)
def fetch_and_analyze(ticker_input, user_cost=0.0, user_qty=0):
    ticker_input = ticker_input.strip().upper()
    macro_tickers = ["SPY", "QQQ", "^VIX", "^TNX"]
    macro_data = yf.download(macro_tickers, period="3mo", interval="1d", auto_adjust=True, progress=False)
    
    market_status = "🟢 多头顺风：标普与纳指稳居 EMA20 上方。"
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

            if (spy_close < spy_ema20 and qqq_close < qqq_ema20) or vix_close >= 25:
                market_status = "🔴 极度预警：标普与纳指双双破位EMA20，全市场防守！"
            elif spy_close < spy_ema20: market_status = "⚠️ 警示：标普(SPY) 跌破生命线！"
            elif qqq_close < qqq_ema20: market_status = "⚠️ 警示：纳指(QQQ) 跌破生命线！"
    except Exception:
        pass

    ticker_obj = yf.Ticker(ticker_input)
    df_daily = yf.download(ticker_input, period="2y", interval="1d", auto_adjust=True, progress=False)
    if df_daily.empty:
        return None, f"未找到股票 [{ticker_input}] 的行情数据。"
    if isinstance(df_daily.columns, pd.MultiIndex):
        df_daily.columns = df_daily.columns.get_level_values(0)

    close_d, high_d, low_d = df_daily['Close'].dropna(), df_daily['High'].dropna(), df_daily['Low'].dropna()
    cur_price = close_d.iloc[-1]
    total_days = len(close_d)

    low_30d = low_d.iloc[-min(30, total_days):].min()
    high_30d = high_d.iloc[-min(30, total_days):].max()

    vwap_price = cur_price
    try:
        df_intraday = yf.download(ticker_input, period="1d", interval="5m", auto_adjust=True, progress=False)
        if not df_intraday.empty:
            if isinstance(df_intraday.columns, pd.MultiIndex):
                df_intraday.columns = df_intraday.columns.get_level_values(0)
            typical_p = (df_intraday['High'] + df_intraday['Low'] + df_intraday['Close']) / 3.0
            if df_intraday['Volume'].sum() > 0:
                vwap_price = (typical_p * df_intraday['Volume']).sum() / df_intraday['Volume'].sum()
    except Exception:
        vwap_price = cur_price

    vwap_status_desc = "多头主导(高于日内成本)" if cur_price > vwap_price * 1.002 else "空头压制(低于日内成本)" if cur_price < vwap_price * 0.998 else "多空平衡"

    ema5 = EMAIndicator(close_d, min(5, total_days)).ema_indicator().iloc[-1]
    ema10 = EMAIndicator(close_d, min(10, total_days)).ema_indicator().iloc[-1]
    ema20 = EMAIndicator(close_d, min(20, total_days)).ema_indicator().iloc[-1]
    ma50 = SMAIndicator(close_d, min(50, total_days)).sma_indicator().iloc[-1]
    ma200 = SMAIndicator(close_d, 200).sma_indicator().iloc[-1] if total_days >= 200 else None

    cross_status = "中性排列"
    if ma50 and ma200:
        cross_status = "🔴 50日与200日呈现死亡交叉" if ma50 < ma200 else "🟢 50日与200日呈现黄金交叉"

    atr_d = AverageTrueRange(high_d, low_d, close_d, min(14, total_days)).average_true_range().iloc[-1]
    vp_data = calculate_institutional_volume_profile(df_daily.iloc[-min(252, total_days):])
    opt_data = fetch_options_microstructure(ticker_obj, cur_price)
    funda_data = fetch_fundamental_and_analyst_data(ticker_obj)

    target1_p = vp_data["resistances"][0] if vp_data["resistances"] else cur_price * 1.08
    dynamic_stop_loss = max(low_30d, cur_price - (1.5 * atr_d))
    rr_ratio = max(0.01, target1_p - cur_price) / max(0.01, cur_price - dynamic_stop_loss)

    position_context = f"【持仓数据】成本: ${user_cost:.2f} | 股数: {user_qty}" if user_qty > 0 else "【当前状态】空仓观望中"

    layered_prompt = f"""
你是一名华尔街顶级量化操盘手兼首席投研导师。
标的: {ticker_input} ｜ 最新价: **${cur_price:.2f}** ｜ 大盘: {market_status}
均线: EMA5: **${ema5:.2f}** ｜ EMA20: **${ema20:.2f}** ｜ MA50: **${ma50:.2f}** ｜ MA200: {f"${ma200:.2f}" if ma200 else '无'}
微观与期权: 做市商VWAP: **${vwap_price:.2f}** ｜ POC筹码峰: **${vp_data['poc']:.2f}** ｜ 期权Max Pain: **${opt_data['max_pain']:.2f}**
基本面: 市值: {funda_data['market_cap']} ｜ 做空比例: {funda_data['short_ratio_float']} ｜ 盈亏比: **{rr_ratio:.2f}:1**
{position_context}

请严格按以下 3 个大白话模块输出专业操盘手册：
### 🚦 1. 操盘手 3 秒极简决策灯
核心操作定性与一句话理由。
### 🛡️ 2. 攻防阶梯价格阵地
- 建议防守止损价（结合 **${dynamic_stop_loss:.2f}**）
- 第一止盈目标价（结合 **${target1_p:.2f}**）
- 关键阻力与支撑区间
### 🧠 3. 交易算账与操盘指令
做多与减仓条件，并以一句话收尾。
"""
    ai_analysis_text = call_gemini_smart(layered_prompt)
    now_display = (datetime.now(timezone.utc) + timedelta(hours=8)).strftime("%H:%M:%S")

    return {
        "symbol": ticker_input, "cur_price": cur_price, "market_status": market_status,
        "spy_info_str": spy_info_str, "qqq_info_str": qqq_info_str, "macro_sentiment_tag": macro_sentiment_tag,
        "vix_status_str": vix_status_str, "tnx_status_str": tnx_status_str, "vwap_price": vwap_price,
        "ema5": ema5, "ema20": ema20, "ma50": ma50, "ma200_str": f"${ma200:.2f}" if ma200 else "无",
        "cross_status": cross_status, "atr_d": atr_d, "dynamic_stop_loss": dynamic_stop_loss,
        "vp_data": vp_data, "opt_data": opt_data, "funda_data": funda_data,
        "rr_ratio": rr_ratio, "target1_p": target1_p, "ai_analysis_text": ai_analysis_text,
        "cache_display_time": now_display
    }, None

# ----------------------------------------------------
# 4. 侧边栏：环境与策略总控
# ----------------------------------------------------
st.sidebar.header("⚙️ 交易环境与网关配置")
trd_env_choice = st.sidebar.radio("交易账户环境", ["模拟盘 (Paper Trading - 推荐)", "实盘 (REAL - 谨慎)"])
active_trd_env = TrdEnv.SIMULATE if "模拟" in trd_env_choice else TrdEnv.REAL

opend_host = st.sidebar.text_input("OpenD IP", value="127.0.0.1")
opend_port = st.sidebar.number_input("OpenD 端口", value=11111)

trd_ctx, quote_ctx = get_moomoo_contexts(host=opend_host, port=opend_port)

st.sidebar.divider()
st.sidebar.header("🎯 自动量化策略引擎")
auto_trade_enabled = st.sidebar.toggle("⚡ 开启全自动交易/风控引擎", value=False)
tp_ratio = st.sidebar.slider("📈 自动止盈目标 (%)", min_value=1.0, max_value=30.0, value=8.0, step=0.5) / 100.0
sl_ratio = st.sidebar.slider("📉 自动止损阈值 (%)", min_value=1.0, max_value=20.0, value=4.0, step=0.5) / 100.0
trade_qty_auto = st.sidebar.number_input("每次自动建仓股数", min_value=1, value=10, step=1)
momentum_thresh = st.sidebar.slider("🚀 突破建仓涨幅阈值 (%)", min_value=0.5, max_value=5.0, value=1.5, step=0.1)

st.sidebar.subheader("🔍 自动扫描标的池")
watchlist_raw = st.sidebar.text_area("候选池代码 (英文逗号隔开)", "US.AAPL, US.NVDA, US.TSLA, US.MSFT, US.AMZN, US.PLTR")
auto_candidates = [s.strip().upper() for s in watchlist_raw.split(",") if s.strip()]

# ----------------------------------------------------
# 5. 主界面 (Tab 架构)
# ----------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "⚡ 量化交易 & 自动风控控制台", 
    "📊 私人持仓全景研报", 
    "🔍 单股全维深度诊断", 
    "📋 交易与预警审计流"
])

# ==================== TAB 1: 量化交易执行与看板 ====================
with tab1:
    if not trd_ctx:
        st.warning("⚠️ 未连接到本地 Moomoo OpenD 网关。若在云端运行仅供研报分析；若在本地，请确保 OpenD.exe 处于运行状态。")
    else:
        ret_acc, acc_df = trd_ctx.accinfo_query(trd_env=active_trd_env)
        total_assets, cash_val, mkt_val = 0.0, 0.0, 0.0
        if ret_acc == 0 and not acc_df.empty:
            total_assets = acc_df['total_assets'].iloc[0]
            cash_val = acc_df['cash'].iloc[0]
            mkt_val = acc_df['market_val'].iloc[0]

        env_tag = "🟢 模拟盘" if active_trd_env == TrdEnv.SIMULATE else "🔴 实盘"
        st.subheader(f"💼 账户资产看板 ({env_tag})")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("总资产", f"${total_assets:,.2f}")
        c2.metric("可用现金", f"${cash_val:,.2f}")
        c3.metric("证券市值", f"${mkt_val:,.2f}")
        c4.metric("仓位使用率", f"{(mkt_val/total_assets*100) if total_assets > 0 else 0:.1f}%")

        st.divider()

        st.subheader("📦 当前持仓 & 动态止盈止损矩阵")
        ret_pos, pos_df = trd_ctx.position_list_query(trd_env=active_trd_env)
        held_codes = []

        if ret_pos == 0 and not pos_df.empty:
            active_p = pos_df[pos_df['qty'] > 0].copy()
            if not active_p.empty:
                active_p['盈亏率(%)'] = ((active_p['nominal_price'] - active_p['cost_price']) / active_p['cost_price']) * 100
                held_codes = active_p['code'].tolist()

                disp_p = active_p[['code', 'stock_name', 'qty', 'cost_price', 'nominal_price', 'pl_val', '盈亏率(%)']]
                disp_p.columns = ['代码', '名称', '持仓股数', '成本价', '现价', '浮动盈亏($)', '盈亏率(%)']
                st.dataframe(
                    disp_p.style.format({
                        '成本价': '${:.2f}', '现价': '${:.2f}', '浮动盈亏($)': '${:.2f}', '盈亏率(%)': '{:+.2f}%'
                    }),
                    use_container_width=True
                )

                if auto_trade_enabled:
                    for _, r in active_p.iterrows():
                        code = r['code']
                        qty = int(r['qty'])
                        cost = r['cost_price']
                        curr = r['nominal_price']
                        p_ratio = (curr - cost) / cost

                        if p_ratio >= tp_ratio:
                            st.warning(f"🎯 触发止盈: {code} (+{p_ratio*100:.2f}%)，市价自动平仓！")
                            trd_ctx.place_order(price=curr, qty=qty, code=code, trd_side=TrdSide.SELL, order_type=OrderType.MARKET, trd_env=active_trd_env)
                            record_trade_log("止盈卖出", code, f"收益率: +{p_ratio*100:.2f}%, 数量: {qty}")
                            send_pushover_alert(f"【止盈达成】{code} 自动卖出 {qty} 股，收益率: +{p_ratio*100:.2f}%")
                        elif p_ratio <= -sl_ratio:
                            st.error(f"🛑 触发止损: {code} ({p_ratio*100:.2f}%)，市价自动离场！")
                            trd_ctx.place_order(price=curr, qty=qty, code=code, trd_side=TrdSide.SELL, order_type=OrderType.MARKET, trd_env=active_trd_env)
                            record_trade_log("止损平仓", code, f"亏损率: {p_ratio*100:.2f}%, 数量: {qty}")
                            send_pushover_alert(f"【止损执行】{code} 自动卖出 {qty} 股，亏损率: {p_ratio*100:.2f}%")
            else:
                st.info("当前无有效多头持仓。")
        else:
            st.info("持仓查询为空。")

        st.divider()

        st.subheader("📡 机会挖掘与自动建仓")
        if auto_trade_enabled and quote_ctx:
            st.caption("🔍 量化引擎正在轮询扫描候选池...")
            for sym in auto_candidates:
                formatted_sym = sym if sym.startswith("US.") else f"US.{sym}"
                if formatted_sym in held_codes:
                    continue

                ret_q, q_df = quote_ctx.get_market_snapshot([formatted_sym])
                if ret_q == 0 and not q_df.empty:
                    last_p = q_df['last_price'].iloc[0]
                    prev_c = q_df['prev_close_price'].iloc[0]
                    chg_pct = ((last_p - prev_c) / prev_c) * 100

                    if chg_pct >= momentum_thresh and cash_val > (last_p * trade_qty_auto):
                        st.info(f"💡 发现突破标的: {formatted_sym} (涨幅: {chg_pct:+.2f}%)，自动建仓中...")
                        ret_b, _ = trd_ctx.place_order(price=last_p, qty=trade_qty_auto, code=formatted_sym, trd_side=TrdSide.BUY, order_type=OrderType.MARKET, trd_env=active_trd_env)
                        if ret_b == 0:
                            record_trade_log("动量买入", formatted_sym, f"买入价: ${last_p:.2f}, 数量: {trade_qty_auto}")
                            send_pushover_alert(f"【自动建仓】{formatted_sym} 突破买入 {trade_qty_auto} 股，价格: ${last_p:.2f}")
                            held_codes.append(formatted_sym)
        else:
            st.caption("⏸️ 自动交易引擎暂停中。开启侧边栏开关以启动全自动盯盘与买卖。")

        st.divider()

        st.subheader("⚡ 手动快捷下单")
        mc1, mc2, mc3, mc4 = st.columns(4)
        m_code = mc1.text_input("股票代码 (如 US.AAPL)", value="US.AAPL")
        m_side = mc2.selectbox("交易方向", ["买入 (BUY)", "卖出 (SELL)"])
        m_price = mc3.number_input("价格 ($)", value=200.0, step=0.5)
        m_qty = mc4.number_input("股数", value=10, min_value=1, step=1)

        if st.button("🚀 提交手动订单", use_container_width=True):
            side = TrdSide.BUY if "BUY" in m_side else TrdSide.SELL
            ret_o, res_o = trd_ctx.place_order(price=m_price, qty=m_qty, code=m_code, trd_side=side, order_type=OrderType.NORMAL, trd_env=active_trd_env)
            if ret_o == 0:
                st.success(f"✅ 订单提交成功！订单号: {res_o['order_id'].iloc[0]}")
                record_trade_log("手动下单", m_code, f"{m_side} 数量: {m_qty}, 价格: ${m_price:.2f}")
                st.rerun()
            else:
                st.error(f"❌ 下单失败: {res_o}")

# ==================== TAB 2: 私人持仓全景研报 ====================
with tab2:
    st.subheader("💼 当前账户实时持仓 AI 深度复盘")
    if trd_ctx:
        ret_p2, pos_df2 = trd_ctx.position_list_query(trd_env=active_trd_env)
        if ret_p2 == 0 and not pos_df2.empty and not pos_df2[pos_df2['qty'] > 0].empty:
            valid_pos = pos_df2[pos_df2['qty'] > 0]
            clean_symbols = [c.replace("US.", "").replace(".US", "") for c in valid_pos['code'].tolist()]
            selected_sym = st.selectbox("选择需要投顾复盘的持仓标的:", clean_symbols)

            target_row = valid_pos[valid_pos['code'].str.contains(selected_sym)].iloc[0]
            if st.button(f"生成 {selected_sym} 专属投顾研报", type="primary"):
                with st.spinner("AI 智脑正在深度解析持仓攻防位置..."):
                    diag_data, d_err = fetch_and_analyze(selected_sym, user_cost=target_row['cost_price'], user_qty=int(target_row['qty']))
                    if d_err:
                        st.error(d_err)
                    else:
                        safe_render_markdown(diag_data['ai_analysis_text'])
        else:
            st.info("当前账户无持仓股票。")
    else:
        st.info("连接到本地 OpenD 后可自动读取持仓生成研报。")

# ==================== TAB 3: 单股深度全维诊断 ====================
with tab3:
    t_in = st.text_input("输入待分析美股代码", value="NVDA").strip().upper()
    if st.button("开始全维实战研报诊断", type="primary"):
        with st.spinner(f"正在全维运算均线、筹码分布与衍生品博弈 ({t_in})..."):
            data, err = fetch_and_analyze(t_in)
            if err:
                st.error(err)
            else:
                st.session_state.current_diag = data

    if "current_diag" in st.session_state:
        d = st.session_state.current_diag
        st.caption(f"⚡ 数据已缓存 (刷新时间: {d['cache_display_time']})")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"{d['symbol']} 现价", f"${d['cur_price']:.2f}")
        c2.metric("做市商 VWAP", f"${d['vwap_price']:.2f}")
        c3.metric("动态盈亏比", f"{d['rr_ratio']:.2f}:1")
        c4.metric("动态保护止损", f"${d['dynamic_stop_loss']:.2f}")

        safe_render_markdown(d['ai_analysis_text'])

# ==================== TAB 4: 审计与预警日志 ====================
with tab4:
    st.subheader("📋 交易执行与策略审计流水 (Audit Logs)")
    if "trade_audit_logs" in st.session_state and st.session_state.trade_audit_logs:
        st.dataframe(pd.DataFrame(st.session_state.trade_audit_logs), use_container_width=True)
    else:
        st.caption("暂无交易日志记录。")

# 自动轮询机制
if auto_trade_enabled:
    time.sleep(10)
    st.rerun()
