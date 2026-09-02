import json
import re
import threading
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD, SMAIndicator
from ta.volatility import AverageTrueRange
import yfinance as yf

# ----------------------------------------------------
# 0. Moomoo API 兼容性与安全导入
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
        FUTUINC = "FUTUINC"
        FUTUSECURITIES = "FUTUSECURITIES"
    class TrdSide:
        BUY = "BUY"
        SELL = "SELL"
    class OrderType:
        NORMAL = "NORMAL"
        MARKET = "MARKET"

# 页面基础配置（移动端优先）
st.set_page_config(
    page_title="Moomoo 智能量化控制台",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 移动端样式
st.markdown("""
<style>
    .metric-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-bottom: 15px;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 12px 16px;
        border-left: 4px solid #1E88E5;
        flex: 1 1 calc(33% - 10px);
        min-width: 140px;
    }
    .metric-title { font-size: 13px; color: #6c757d; margin-bottom: 2px; }
    .metric-val { font-size: 20px; font-weight: bold; color: #212529; }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ Moomoo 智能量化控制台")

# ----------------------------------------------------
# 1. 基础配置与通信引擎
# ----------------------------------------------------
raw_api_key = st.secrets.get("GEMINI_API_KEY", "") if "GEMINI_API_KEY" in st.secrets else ""
api_key = str(raw_api_key).strip().replace("\n", "").replace("\r", "").replace(" ", "").replace('"', '').replace("'", "")

PUSHOVER_USER_KEY = st.secrets.get("PUSHOVER_USER_KEY", "") if "PUSHOVER_USER_KEY" in st.secrets else ""
PUSHOVER_API_TOKEN = st.secrets.get("PUSHOVER_API_TOKEN", "") if "PUSHOVER_API_TOKEN" in st.secrets else ""

def safe_render_markdown(text):
    if text:
        st.markdown(text.replace("$", "\\$"))

def send_pushover_alert(message, title="🛡️ 美股量化预警", priority=0, sound="cashregister"):
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

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key
    }
    payload = {
        "contents": [{"parts": [{"text": prompt_text}]}]
    }

    usable_models = []
    try:
        list_url = "https://generativelanguage.googleapis.com/v1beta/models"
        resp_list = requests.get(list_url, headers=headers, timeout=8)
        if resp_list.status_code == 200:
            all_models = resp_list.json().get("models", [])
            for item in all_models:
                m_name = item.get("name", "")
                methods = item.get("supportedGenerationMethods", [])
                if "generateContent" in methods:
                    usable_models.append(m_name.replace("models/", ""))
    except Exception:
        pass

    if not usable_models:
        usable_models = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]

    last_error = ""
    for m in usable_models:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent"
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
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
# 2. Moomoo 数据获取核心
# ----------------------------------------------------
def get_moomoo_account_data(host="127.0.0.1", port=11111, trd_env=TrdEnv.SIMULATE):
    if not MOOMOO_AVAILABLE:
        return False, None, pd.DataFrame(), "Moomoo SDK 未安装"
    try:
        trd = OpenSecTradeContext(
            filter_trdmarket=TrdMarket.US,
            host=host,
            port=int(port),
            security_firm=SecurityFirm.FUTUINC
        )
        ret_acc, acc_df = trd.accinfo_query(trd_env=trd_env, currency=1)
        ret_pos, pos_df = trd.position_list_query(trd_env=trd_env)
        trd.close()

        acc_info = acc_df.iloc[0] if ret_acc == 0 and not acc_df.empty else None
        positions = pos_df if ret_pos == 0 and not pos_df.empty else pd.DataFrame()
        return True, acc_info, positions, "OK"
    except Exception as e:
        return False, None, pd.DataFrame(), str(e)

# ----------------------------------------------------
# 3. 筹码与量化算法模块
# ----------------------------------------------------
def calculate_institutional_volume_profile(df_daily, bins=40):
    if df_daily.empty or 'Close' not in df_daily.columns or 'Volume' not in df_daily.columns:
        return {"poc": 0.0, "vah": 0.0, "val": 0.0, "resistances": [], "supports": [], "bin_centers": [], "vol_profile": []}
    price_min, price_max = df_daily['Low'].min(), df_daily['High'].max()
    if price_max <= price_min or np.isnan(price_min) or np.isnan(price_max):
        return {"poc": 0.0, "vah": 0.0, "val": 0.0, "resistances": [], "supports": [], "bin_centers": [], "vol_profile": []}

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
        "supports": [round(p, 2) for p in sup_bins[:2]],
        "bin_centers": bin_centers.tolist(),
        "vol_profile": vol_profile.tolist()
    }

def plot_microstructure_chart(df_daily, symbol, cur_price, vp_data, sl_price, tp_price):
    try:
        recent_df = df_daily.iloc[-60:].copy()
        fig = make_subplots(
            rows=1, cols=2, shared_yaxes=True,
            column_widths=[0.8, 0.2], horizontal_spacing=0.03,
            subplot_titles=(f"📈 {symbol} 攻防矩阵", "📊 筹码峰")
        )

        fig.add_trace(go.Candlestick(
            x=recent_df.index,
            open=recent_df['Open'], high=recent_df['High'],
            low=recent_df['Low'], close=recent_df['Close'],
            name="K线"
        ), row=1, col=1)

        fig.add_hline(y=sl_price, line_dash="dash", line_color="red", annotation_text=f"止损: ${sl_price:.2f}", row=1, col=1)
        fig.add_hline(y=tp_price, line_dash="dash", line_color="green", annotation_text=f"止盈: ${tp_price:.2f}", row=1, col=1)
        fig.add_hline(y=vp_data["poc"], line_color="orange", annotation_text=f"POC: ${vp_data['poc']:.2f}", row=1, col=1)

        if vp_data["bin_centers"] and vp_data["vol_profile"]:
            fig.add_trace(go.Bar(
                x=vp_data["vol_profile"],
                y=vp_data["bin_centers"],
                orientation='h',
                marker_color='rgba(100, 149, 237, 0.6)',
                name="筹码柱"
            ), row=1, col=2)

        fig.update_layout(
            height=380, margin=dict(l=5, r=5, t=30, b=5),
            showlegend=False, xaxis_rangeslider_visible=False,
            template="plotly_dark"
        )
        return fig
    except Exception:
        return None

def fetch_options_microstructure(ticker_obj, cur_price):
    try:
        expirations = ticker_obj.options
        if not expirations:
            return {"max_pain": 0.0, "pcr": 1.0, "major_call_wall": 0.0, "major_put_wall": 0.0, "nearest_expiry": "无"}
        nearest_exp = expirations[0]
        opt_chain = ticker_obj.option_chain(nearest_exp)
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
            "max_pain": float(max_pain), "pcr": float(pcr), 
            "major_call_wall": call_wall, "major_put_wall": put_wall,
            "nearest_expiry": nearest_exp
        }
    except Exception:
        return {"max_pain": 0.0, "pcr": 1.0, "major_call_wall": 0.0, "major_put_wall": 0.0, "nearest_expiry": "无"}

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
            "pe_ttm": f"{info.get('trailingPE', 0):.2f}" if info.get('trailingPE') else "N/A",
            "short_ratio_float": f"{info.get('shortPercentOfFloat', 0) * 100:.2f}%" if info.get('shortPercentOfFloat') else "N/A",
            "short_days_to_cover": f"{info.get('shortRatio', 0):.2f}" if info.get('shortRatio') else "N/A",
            "target_mean": info.get("targetMeanPrice"),
            "recommendation_key": info.get("recommendationKey", "N/A").upper().replace("_", " ")
        }
    except Exception:
        return {
            "market_cap": "N/A", "pe_ttm": "N/A", "short_ratio_float": "N/A",
            "short_days_to_cover": "N/A", "target_mean": None, "recommendation_key": "N/A"
        }

@st.cache_data(ttl=300, show_spinner=False)
def fetch_and_analyze(ticker_input, user_cost=0.0, user_qty=0):
    ticker_input = ticker_input.strip().upper()
    macro_tickers = ["SPY", "QQQ", "^VIX"]
    macro_data = yf.download(macro_tickers, period="1mo", interval="1d", auto_adjust=True, progress=False)
    
    market_status = "🟢 多头顺风"
    macro_score = 30
    try:
        if not macro_data.empty:
            close_data = macro_data['Close']
            spy_c = close_data['SPY'].dropna()
            spy_close = spy_c.iloc[-1]
            spy_ema20 = EMAIndicator(spy_c, 20).ema_indicator().iloc[-1]
            vix_close = close_data['^VIX'].dropna().iloc[-1]

            if spy_close < spy_ema20 or vix_close >= 25:
                market_status = "🔴 逆风防守"
                macro_score = 10
    except Exception:
        pass

    ticker_obj = yf.Ticker(ticker_input)
    df_daily = yf.download(ticker_input, period="1y", interval="1d", auto_adjust=True, progress=False)
    if df_daily.empty:
        return None, f"未找到股票 [{ticker_input}] 的行情数据。"
    if isinstance(df_daily.columns, pd.MultiIndex):
        df_daily.columns = df_daily.columns.get_level_values(0)

    close_d, high_d, low_d = df_daily['Close'].dropna(), df_daily['High'].dropna(), df_daily['Low'].dropna()
    cur_price = close_d.iloc[-1]
    total_days = len(close_d)

    low_30d = low_d.iloc[-min(30, total_days):].min()
    ema20 = EMAIndicator(close_d, min(20, total_days)).ema_indicator().iloc[-1]
    atr_d = AverageTrueRange(high_d, low_d, close_d, min(14, total_days)).average_true_range().iloc[-1]
    
    vp_data = calculate_institutional_volume_profile(df_daily.iloc[-min(120, total_days):])
    opt_data = fetch_options_microstructure(ticker_obj, cur_price)
    funda_data = fetch_fundamental_and_analyst_data(ticker_obj)

    target1_p = vp_data["resistances"][0] if vp_data["resistances"] else cur_price * 1.08
    dynamic_stop_loss = max(low_30d, cur_price - (1.5 * atr_d))
    rr_ratio = max(0.01, target1_p - cur_price) / max(0.01, cur_price - dynamic_stop_loss)

    total_quant_score = macro_score + (35 if cur_price > ema20 else 15) + (35 if opt_data['pcr'] < 0.9 else 15)
    pos_desc = f"持仓成本: ${user_cost:.2f} | 数量: {user_qty} 股" if user_qty > 0 else "空仓观望"

    layered_prompt = f"""
你是一名华尔街首席宏观量化操盘手，请针对标的 {ticker_input} 提供一份极简有力的实战研报：
- 标的: {ticker_input} ｜ 现价: ${cur_price:.2f} ｜ 评分: {total_quant_score}/100 ｜ 宏观: {market_status}
- 均线与ATR: EMA20: ${ema20:.2f} ｜ 14日ATR: ${atr_d:.2f} ｜ 动态止损线: ${dynamic_stop_loss:.2f}
- 筹码与期权: POC筹码峰: ${vp_data['poc']:.2f} ｜ Max Pain: ${opt_data['max_pain']:.2f} ｜ PCR: {opt_data['pcr']:.2f}
- 基本面与评级: 市值: {funda_data['market_cap']} ｜ 投行评级: {funda_data['recommendation_key']} ｜ 空头比例: {funda_data['short_ratio_float']}
- 用户持仓状态: {pos_desc} ｜ 第一目标价: ${target1_p:.2f} ｜ 盈亏比: {rr_ratio:.2f}:1

请按 3 个要点输出直接指导：
1. **操盘决策灯**：明确给出定性（看多/防守/离场）与一句话核心理由。
2. **攻防阵地**：明确写出防守止损价（${dynamic_stop_loss:.2f}）与第一目标止盈价（${target1_p:.2f}）。
3. **执行指令**：给出现价情况下的买卖或持股建议。
"""
    ai_analysis_text = call_gemini_smart(layered_prompt)
    now_display = (datetime.now(timezone.utc) + timedelta(hours=8)).strftime("%H:%M:%S")
    chart_fig = plot_microstructure_chart(df_daily, ticker_input, cur_price, vp_data, dynamic_stop_loss, target1_p)

    return {
        "symbol": ticker_input, "cur_price": cur_price, "dynamic_stop_loss": dynamic_stop_loss,
        "target1_p": target1_p, "rr_ratio": rr_ratio, "total_quant_score": total_quant_score,
        "chart_fig": chart_fig, "ai_analysis_text": ai_analysis_text, "cache_display_time": now_display
    }, None

# ----------------------------------------------------
# 4. 侧边栏设置
# ----------------------------------------------------
with st.sidebar:
    st.header("⚙️ 系统与风控总控")
    if st.button("🔄 立即刷新数据", use_container_width=True):
        st.rerun()

    trd_env_choice = st.radio("交易环境", ["模拟盘 (SIMULATE)", "实盘 (REAL)"])
    active_trd_env = TrdEnv.SIMULATE if "模拟" in trd_env_choice else TrdEnv.REAL

    opend_host = st.text_input("OpenD Host", value="127.0.0.1")
    opend_port = st.number_input("OpenD Port", value=11111, step=1)

# ----------------------------------------------------
# 5. 单页核心监控与量化分析流
# ----------------------------------------------------
success, acc_info, pos_df, err_msg = get_moomoo_account_data(host=opend_host, port=int(opend_port), trd_env=active_trd_env)

if not success or acc_info is None:
    st.warning(f"⚠️ 无法连接到本地 Moomoo OpenD 网关 ({err_msg})，请确保电脑端 OpenD 客户端处于登录运行状态。")
else:
    # 1. 资产看板卡片
    total_assets = acc_info.get("total_assets", 0.0)
    cash_val = acc_info.get("cash", 0.0)
    mkt_val = acc_info.get("market_val", 0.0)

    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-card"><div class="metric-title">💰 模拟总资产</div><div class="metric-val">${total_assets:,.2f}</div></div>
        <div class="metric-card"><div class="metric-title">💵 可用现金</div><div class="metric-val">${cash_val:,.2f}</div></div>
        <div class="metric-card"><div class="metric-title">📈 持股市值</div><div class="metric-val">${mkt_val:,.2f}</div></div>
    </div>
    """, unsafe_allow_html=True)

    # 2. 持仓监控
    st.subheader("📦 当前持仓监控")
    held_codes = []

    if not pos_df.empty:
        active_p = pos_df[pos_df['qty'] > 0].copy()
        if not active_p.empty:
            active_p['盈亏率'] = ((active_p['nominal_price'] - active_p['cost_price']) / active_p['cost_price']) * 100
            held_codes = active_p['code'].tolist()

            disp_p = active_p[['code', 'stock_name', 'qty', 'cost_price', 'nominal_price', 'pl_val', '盈亏率']]
            disp_p.columns = ['代码', '名称', '数量', '成本价', '现价', '浮动盈亏', '盈亏率(%)']
            
            st.dataframe(
                disp_p.style.format({
                    '成本价': '${:.2f}', '现价': '${:.2f}', '浮动盈亏': '${:.2f}', '盈亏率(%)': '{:+.2f}%'
                }),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("当前账户暂无持仓。")
    else:
        st.info("持仓查询为空。")

    st.markdown("---")

    # 3. 实时 AI 投顾深度诊断 & 筹码量化分析
    with st.expander("🧠 AI 投顾研报诊断 & 筹码量化分析", expanded=True):
        default_sym = held_codes[0].replace("US.", "") if held_codes else "NVDA"
        diag_symbol = st.text_input("输入待诊断美股代码", value=default_sym).strip().upper()
        if st.button("🚀 生成深度量化研报", type="primary", use_container_width=True):
            with st.spinner("AI 正在分析微观筹码与市场结构..."):
                diag_res, err_msg = fetch_and_analyze(diag_symbol)
                if err_msg:
                    st.error(err_msg)
                else:
                    st.session_state.diag_res = diag_res

        if "diag_res" in st.session_state:
            d = st.session_state.diag_res
            c1, c2, c3 = st.columns(3)
            c1.metric("现价", f"${d['cur_price']:.2f}")
            c2.metric("止损点", f"${d['dynamic_stop_loss']:.2f}")
            c3.metric("量化得分", f"{d['total_quant_score']}/100")
            if d['chart_fig']:
                st.plotly_chart(d['chart_fig'], use_container_width=True)
            safe_render_markdown(d['ai_analysis_text'])

    # 4. 模拟盘快捷交易面板
    with st.expander("⚡ 模拟快捷下单", expanded=False):
        c_code, c_side = st.columns(2)
        m_code = c_code.text_input("代码", value="US.SOXS")
        m_side = c_side.selectbox("方向", ["买入 (BUY)", "卖出 (SELL)"])
        
        c_p, c_q = st.columns(2)
        m_price = c_p.number_input("价格 ($)", value=50.0, step=0.1)
        m_qty = c_q.number_input("数量", value=100, min_value=1, step=10)

        if st.button("提交订单", use_container_width=True):
            try:
                trd_ctx = OpenSecTradeContext(filter_trdmarket=TrdMarket.US, host=opend_host, port=int(opend_port), security_firm=SecurityFirm.FUTUINC)
                side = TrdSide.BUY if "买入" in m_side else TrdSide.SELL
                ret_o, res_o = trd_ctx.place_order(
                    price=m_price, qty=m_qty, code=m_code.strip().upper(),
                    trd_side=side, order_type=OrderType.NORMAL, trd_env=active_trd_env
                )
                trd_ctx.close()
                if ret_o == 0:
                    st.success(f"下单成功！订单号: {res_o['order_id'].iloc[0]}")
                    st.rerun()
                else:
                    st.error(f"下单失败: {res_o}")
            except Exception as e:
                st.error(f"下单异常: {e}")
