from datetime import datetime, timedelta, timezone
import json
import re
import threading
import time
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

st.set_page_config(
    page_title="专属私人 AI 投顾 & 智能盯盘系统", layout="wide"
)
st.title("🛡️ 专属私人 AI 量化投顾系统 (Moomoo 实时联动版)")
st.caption(
    "⚡ Moomoo 持仓直连 ｜ 🛰️ 盘中量化盯盘 ｜ 🎯 阶梯止盈与移动止损 ｜ 📲 手机实时告警推送"
)

# ----------------------------------------------------
# 1. 基础配置与通知模块
# ----------------------------------------------------
raw_api_key = (
    st.secrets.get("GEMINI_API_KEY", "") if "GEMINI_API_KEY" in st.secrets else ""
)
api_key = raw_api_key.strip().replace("\n", "").replace("\r", "").replace(" ", "")

TG_BOT_TOKEN = st.secrets.get("TG_BOT_TOKEN", "")
TG_CHAT_ID = st.secrets.get("TG_CHAT_ID", "")


def send_telegram_alert(message):
    """向手机 Telegram 推送实时投顾预警"""
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        return False
    try:
        url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TG_CHAT_ID,
            "text": message,
            "parse_mode": "Markdown",
        }
        res = requests.post(url, json=payload, timeout=5)
        return res.status_code == 200
    except Exception:
        return False


def safe_render_markdown(text):
    if not text:
        return
    st.markdown(text.replace("$", "\\$"))


def call_gemini_advisor(prompt_text):
    if not api_key:
        return "⚠️ 未配置 GEMINI_API_KEY。"
    try:
        genai.configure(api_key=api_key)
        for m_name in [
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-1.5-flash-latest",
        ]:
            try:
                model = genai.GenerativeModel(m_name)
                res = model.generate_content(prompt_text)
                if res and res.text:
                    return res.text
            except Exception:
                continue
        return "⚠️ Gemini 智脑响应超时，请检查模型权限。"
    except Exception as e:
        return f"⚠️ 智脑异常: `{e}`"


# ----------------------------------------------------
# 2. Moomoo 账户直连模块
# ----------------------------------------------------
def fetch_moomoo_positions(host="127.0.0.1", port=11111):
    """从本地 OpenD 网关获取真实持仓"""
    if not MOOMOO_AVAILABLE:
        return (
            None,
            "未检测到 moomoo-api 库，请在终端执行 `pip install moomoo-api`。",
        )

    try:
        trd_ctx = OpenSecTradeContext(
            filter_trdmarket=TrdMarket.US,
            host=host,
            port=port,
            is_acc_need_decrypt=False,
        )
        ret, data = trd_ctx.position_list_query(trd_env=TrdEnv.REAL)
        trd_ctx.close()

        if ret != 0:
            return None, f"Moomoo 网关返回错误: {data}"

        if data.empty:
            return pd.DataFrame(), None

        # 标准化提取列
        clean_df = pd.DataFrame(
            {
                "symbol": (
                    data["code"]
                    .str.replace("US.", "", regex=False)
                    .str.replace(".US", "", regex=False)
                ),
                "qty": data["qty"],
                "can_sell_qty": data["can_sell_qty"],
                "cost_price": data["cost_price"],
                "nominal_price": data["nominal_price"],
                "pl_val": data["pl_val"],
                "pl_ratio": data["pl_ratio"],
            }
        )
        return clean_df, None
    except Exception as e:
        return (
            None,
            f"连接 Moomoo OpenD 失败: {e} (请确认本地 OpenD 客户端已启动并处于登录状态)",
        )


# ----------------------------------------------------
# 3. 量化指标与筹码计算
# ----------------------------------------------------
def calculate_institutional_volume_profile(df_daily, bins=40):
    if (
        df_daily.empty
        or "Close" not in df_daily.columns
        or "Volume" not in df_daily.columns
    ):
        return {
            "poc": 0.0,
            "vah": 0.0,
            "val": 0.0,
            "resistances": [],
            "supports": [],
        }

    price_min, price_max = df_daily["Low"].min(), df_daily["High"].max()
    if price_max <= price_min or np.isnan(price_min) or np.isnan(price_max):
        return {
            "poc": 0.0,
            "vah": 0.0,
            "val": 0.0,
            "resistances": [],
            "supports": [],
        }

    bin_edges = np.linspace(price_min, price_max, bins + 1)
    vol_profile = np.zeros(bins)

    for _, row in df_daily.iterrows():
        mid_p = (row["High"] + row["Low"] + row["Close"]) / 3.0
        b_idx = max(0, min(bins - 1, int(np.digitize(mid_p, bin_edges) - 1)))
        vol_profile[b_idx] += row["Volume"]

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
    cur_price = df_daily["Close"].iloc[-1]

    top_indices = sorted_indices[:8]
    res_bins = sorted(
        [bin_centers[i] for i in top_indices if bin_centers[i] > cur_price * 1.01]
    )
    sup_bins = sorted(
        [bin_centers[i] for i in top_indices if bin_centers[i] < cur_price * 0.99],
        reverse=True,
    )

    return {
        "poc": poc_price,
        "vah": vah_price,
        "val": val_price,
        "resistances": [round(p, 2) for p in res_bins[:3]],
        "supports": [round(p, 2) for p in sup_bins[:2]],
    }


def analyze_stock_full(symbol, user_cost=0.0, user_qty=0):
    symbol = symbol.strip().upper()
    df_daily = yf.download(
        symbol, period="1y", interval="1d", auto_adjust=True, progress=False
    )
    if df_daily.empty:
        return None, f"未获取到 {symbol} 数据"
    if isinstance(df_daily.columns, pd.MultiIndex):
        df_daily.columns = df_daily.columns.get_level_values(0)

    close_d, high_d, low_d = (
        df_daily["Close"].dropna(),
        df_daily["High"].dropna(),
        df_daily["Low"].dropna(),
    )
    cur_price = close_d.iloc[-1]
    total_days = len(close_d)

    ema5 = EMAIndicator(close_d, min(5, total_days)).ema_indicator().iloc[-1]
    ema20 = EMAIndicator(close_d, min(20, total_days)).ema_indicator().iloc[-1]
    ma60 = (
        SMAIndicator(close_d, 60).sma_indicator().iloc[-1]
        if total_days >= 60
        else None
    )
    atr = (
        AverageTrueRange(high_d, low_d, close_d, min(14, total_days))
        .average_true_range()
        .iloc[-1]
    )

    # 日内 VWAP
    vwap_price = cur_price
    try:
        df_intraday = yf.download(
            symbol,
            period="1d",
            interval="5m",
            auto_adjust=True,
            progress=False,
        )
        if not df_intraday.empty:
            if isinstance(df_intraday.columns, pd.MultiIndex):
                df_intraday.columns = df_intraday.columns.get_level_values(0)
            typ = (
                df_intraday["High"]
                + df_intraday["Low"]
                + df_intraday["Close"]
            ) / 3.0
            vol = df_intraday["Volume"]
            if vol.sum() > 0:
                vwap_price = (typ * vol).sum() / vol.sum()
    except Exception:
        pass

    vp_data = calculate_institutional_volume_profile(df_daily)
    dynamic_stop = max(vp_data["poc"] * 0.98, cur_price - (1.5 * atr))
    target1 = (
        vp_data["resistances"][0]
        if vp_data["resistances"]
        else (ma60 if ma60 and ma60 > cur_price else cur_price * 1.06)
    )

    reward = max(0.01, target1 - cur_price)
    risk = max(0.01, cur_price - dynamic_stop)
    rr_ratio = reward / risk

    # 私人专属持仓诊断 Prompt
    position_context = ""
    if user_qty > 0:
        pnl_pct = ((cur_price - user_cost) / user_cost) * 100
        position_context = f"""
【用户专属持仓数据】：
- 持仓均价: **${user_cost:.2f}** ｜ 持仓股数: **{user_qty} 股**
- 当前浮动盈亏: **{pnl_pct:+.2f}%**
请以【私人专属投顾】身份，直接给出：
1. **持仓处理动作**：是继续持有、上移移动止损、还是在第一目标 **${target1:.2f}** 阶梯止盈部分？
2. **移动止损保护线**：为保护利润，建议将止损抬升至哪个具体价格？
"""
    else:
        position_context = "【用户状态】：当前空仓观望中。请给出严格的开仓条件与盈亏比评估。"

    advisor_prompt = f"""
你是一名顶级对冲基金私人专属量化投资顾问。针对用户标的给出具备极高纪律性的实战指令。

【标的】: {symbol} ｜ 最新现价: **${cur_price:.2f}**
【微观筹码】: 做市商VWAP: **${vwap_price:.2f}** ｜ POC筹码密集峰: **${vp_data['poc']:.2f}** ｜ 价值区上沿(VAH): **${vp_data['vah']:.2f}**
【均线防线】: EMA5: **${ema5:.2f}** ｜ EMA20生命线: **${ema20:.2f}** ｜ 季线(MA60): {f"${ma60:.2f}" if ma60 else '无'}
【风控量化】: 14日ATR: **${atr:.2f}** ｜ 建议防守止损: **${dynamic_stop:.2f}** ｜ 第一阻力目标: **${target1:.2f}** ｜ 盈亏比: **{rr_ratio:.2f}:1**
{position_context}

【要求】：
1. 彻底说人话，严禁只有黑话。价格必须带美元符号并加粗（如 **$18.50**）。
2. 给出 3 秒极简决策：【🟢 建议加仓/买入】 / 【🟡 锁利持有/观望】 / 【🔴 破位减仓/硬止损】。
3. 给出清晰的阶梯止盈与防守点位逻辑。
"""
    ai_report = call_gemini_advisor(advisor_prompt)

    return {
        "symbol": symbol,
        "cur_price": cur_price,
        "vwap_price": vwap_price,
        "poc": vp_data["poc"],
        "vah": vp_data["vah"],
        "val": vp_data["val"],
        "ema20": ema20,
        "dynamic_stop": dynamic_stop,
        "target1": target1,
        "rr_ratio": rr_ratio,
        "atr": atr,
        "ai_report": ai_report,
    }, None


# ----------------------------------------------------
# 4. 后台自动盯盘监控引擎
# ----------------------------------------------------
if "alert_logs" not in st.session_state:
    st.session_state.alert_logs = []


def run_portfolio_monitor_task(watchlist):
    """遍历持仓与关注列表进行风控判定"""
    triggered_alerts = []
    for item in watchlist:
        sym = item["symbol"]
        cost = item.get("cost", 0.0)
        qty = item.get("qty", 0)

        data, err = analyze_stock_full(sym, user_cost=cost, user_qty=qty)
        if not data:
            continue

        cur_p = data["cur_price"]
        vwap_p = data["vwap_price"]
        poc_p = data["poc"]
        stop_p = data["dynamic_stop"]
        target_p = data["target1"]

        # 触发硬止损
        if cur_p < stop_p:
            msg = f"🚨 *【止损预警】{sym} 跌破量化防线！*\n现价: `${cur_p:.2f}` 已跌破硬止损位 `${stop_p:.2f}` (POC/ATR防线)。\n建议：立即执行止损防守。"
            triggered_alerts.append(msg)
            send_telegram_alert(msg)

        # 触发第一目标止盈
        elif cur_p >= target_p:
            msg = f"🎯 *【止盈提示】{sym} 触及第一阻力目标！*\n现价: `${cur_p:.2f}` 达到目标 `${target_p:.2f}`。\n建议：阶梯锁定 1/3 ~ 1/2 利润，底仓上移止损。"
            triggered_alerts.append(msg)
            send_telegram_alert(msg)

        # 向上突破做市商 VWAP
        elif cur_p > vwap_p and (cur_p - vwap_p) / vwap_p < 0.015:
            msg = f"⚡ *【异动提醒】{sym} 站上做市商 VWAP 成本线！*\n现价: `${cur_p:.2f}` 突破 `${vwap_p:.2f}`，短线主力由被动转为护盘。"
            triggered_alerts.append(msg)

    return triggered_alerts


# ----------------------------------------------------
# 5. UI 侧边栏与主控制面板
# ----------------------------------------------------
st.sidebar.header("⚙️ 投顾与 Moomoo 连接配置")

# Moomoo 选项
connect_mode = st.sidebar.radio(
    "持仓数据来源", ["Moomoo OpenD 本地直连", "手动输入 / 模拟持仓"]
)

portfolio_list = []

if connect_mode == "Moomoo OpenD 本地直连":
    opend_host = st.sidebar.text_input("OpenD IP", value="127.0.0.1")
    opend_port = st.sidebar.number_input("OpenD 端口", value=11111)

    if st.sidebar.button("🔄 同步 Moomoo 真实持仓", use_container_width=True):
        with st.spinner("正在直连 Moomoo OpenD 读取持仓..."):
            moo_df, moo_err = fetch_moomoo_positions(
                host=opend_host, port=opend_port
            )
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
            portfolio_list.append(
                {
                    "symbol": row["symbol"],
                    "cost": float(row["cost_price"]),
                    "qty": int(row["qty"]),
                    "pl_ratio": float(row["pl_ratio"]),
                }
            )
else:
    st.sidebar.write("**手动维护持仓与关注列表:**")
    m_sym = st.sidebar.text_input("股票代码", value="USAR").upper()
    m_cost = st.sidebar.number_input("持仓成本 ($)", value=16.80)
    m_qty = st.sidebar.number_input("持仓股数", value=100)

    if st.sidebar.button("➕ 加入持仓列表"):
        if "manual_positions" not in st.session_state:
            st.session_state.manual_positions = []
        st.session_state.manual_positions.append(
            {"symbol": m_sym, "cost": m_cost, "qty": m_qty}
        )

    portfolio_list = st.session_state.get(
        "manual_positions", [{"symbol": m_sym, "cost": m_cost, "qty": m_qty}]
    )

# Telegram 推送设置
st.sidebar.divider()
st.sidebar.subheader("📲 实时预警推送")
tg_status = "🟢 已启用" if TG_BOT_TOKEN and TG_CHAT_ID else "⚪ 未配置 (仅面板提示)"
st.sidebar.caption(f"Telegram 状态: {tg_status}")

if st.sidebar.button("⚡ 立即执行一次持仓风控扫描", type="primary"):
    with st.spinner("量化引擎正在全局扫描持仓与攻防点位..."):
        alerts = run_portfolio_monitor_task(portfolio_list)
        if alerts:
            st.session_state.alert_logs.extend(alerts)
            st.success(f"扫描完成，触发 {len(alerts)} 条异动预警！")
        else:
            st.info("持仓运行平稳，暂未触及止损或止盈触发线。")

# ----------------------------------------------------
# 6. 主界面：专属持仓全局看板与诊断
# ----------------------------------------------------
tab1, tab2, tab3 = st.tabs(
    ["📊 私人持仓全景透视", "🔍 单股深度全维诊断", "🚨 实时预警日志"]
)

with tab1:
    st.subheader("💼 当前账户实时持仓诊断")
    if portfolio_list:
        p_df = pd.DataFrame(portfolio_list)
        st.dataframe(p_df, use_container_width=True)

        selected_pos_sym = st.selectbox(
            "选择需要投顾深度复盘的持仓:", [p["symbol"] for p in portfolio_list]
        )
        selected_item = next(
            item
            for item in portfolio_list
            if item["symbol"] == selected_pos_sym
        )

        if st.button(f"生成 {selected_pos_sym} 专属投顾报告", type="primary"):
            with st.spinner("专属 AI 投顾正在根据你的持仓计算执行指令..."):
                diag_data, d_err = analyze_stock_full(
                    selected_pos_sym,
                    user_cost=selected_item["cost"],
                    user_qty=selected_item["qty"],
                )
                if d_err:
                    st.error(d_err)
                else:
                    # 渲染数据指标
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("现价", f"${diag_data['cur_price']:.2f}")
                    col2.metric("做市商 VWAP", f"${diag_data['vwap_price']:.2f}")
                    col3.metric("筹码中心 (POC)", f"${diag_data['poc']:.2f}")
                    col4.metric(
                        "移动防守止损", f"${diag_data['dynamic_stop']:.2f}"
                    )

                    st.markdown("---")
                    st.subheader(f"🤖 投顾执行报告 ({selected_pos_sym})")
                    safe_render_markdown(diag_data["ai_report"])
    else:
        st.info("暂无持仓数据，请在左侧侧边栏同步 Moomoo 或手动添加标的。")

with tab2:
    st.subheader("🔍 任意美股自由诊断与对比")
    custom_sym = (
        st.text_input("输入查询美股代码", value="NVDA").strip().upper()
    )
    if st.button("开始深度诊断", key="btn_single_diag"):
        with st.spinner(f"正在全维分析 {custom_sym}..."):
            c_data, c_err = analyze_stock_full(custom_sym)
            if c_err:
                st.error(c_err)
            else:
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("现价", f"${c_data['cur_price']:.2f}")
                col_b.metric("做市商成本 (VWAP)", f"${c_data['vwap_price']:.2f}")
                col_c.metric("动态盈亏比", f"{c_data['rr_ratio']:.2f} : 1")

                st.subheader("🤖 操盘手指令报告")
                safe_render_markdown(c_data["ai_report"])

with tab3:
    st.subheader("🚨 盘中实时风控与买卖预警日志")
    if st.session_state.alert_logs:
        for log in reversed(st.session_state.alert_logs[-10:]):
            st.warning(log)
    else:
        st.info("暂无最新告警记录。")
