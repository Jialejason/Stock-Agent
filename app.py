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
    page_title="专属私人 AI 投顾 Pro (Pushover联动版)", layout="wide"
)
st.title("🛡️ 专属私人 AI 量化投顾系统 Pro")
st.caption(
    "⚡ Moomoo 持仓直连 ｜ 🛰️ 盘中量化盯盘 ｜ 🎯 阶梯止盈与移动止损 ｜ 📲 Pushover 实时告警推送"
)

# ----------------------------------------------------
# 1. 基础配置与 Pushover 推送模块
# ----------------------------------------------------
TICKER_ALIASES = {
    "TESLA": "TSLA",
    "特斯拉": "TSLA",
    "APPLE": "AAPL",
    "苹果": "AAPL",
    "NVIDIA": "NVDA",
    "英伟达": "NVDA",
    "GOOGLE": "GOOGL",
    "谷歌": "GOOGL",
    "AMAZON": "AMZN",
    "亚马逊": "AMZN",
    "MICROSOFT": "MSFT",
    "微软": "MSFT",
    "META": "META",
    "脸书": "META",
    "AMD": "AMD",
    "超微": "AMD",
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
    words = re.findall(r"\b[A-Z]{2,5}\b", text_upper)
    for w in words:
        found_symbols.add(w)
    return found_symbols


# 读取 Secrets 配置
raw_api_key = (
    st.secrets.get("GEMINI_API_KEY", "") if "GEMINI_API_KEY" in st.secrets else ""
)
api_key = raw_api_key.strip().replace("\n", "").replace("\r", "").replace(" ", "")

PUSHOVER_USER_KEY = (
    st.secrets.get("PUSHOVER_USER_KEY", "")
    if "PUSHOVER_USER_KEY" in st.secrets
    else ""
)
PUSHOVER_API_TOKEN = (
    st.secrets.get("PUSHOVER_API_TOKEN", "")
    if "PUSHOVER_API_TOKEN" in st.secrets
    else ""
)


def send_pushover_alert(
    message, title="🛡️ 美股投顾量化预警", priority=0, sound="cashregister"
):
    """向 Pushover 发送实时投顾预警
    priority: -1 (低), 0 (普通), 1 (高优先级震动加声音)
    """
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
            "sound": sound,
        }
        res = requests.post(url, data=payload, timeout=5)
        return res.status_code == 200
    except Exception:
        return False


def call_gemini_smart(prompt_text):
    """全自动容灾调用，确保 100% 连通可用模型"""
    if not api_key:
        return "⚠️ 未检测到 API Key，请在 Streamlit Secrets 中配置 `GEMINI_API_KEY`。"

    try:
        genai.configure(api_key=api_key)

        candidate_models = [
            "gemini-2.0-flash",
            "gemini-1.5-flash",
            "gemini-1.5-flash-latest",
            "gemini-1.5-pro",
            "gemini-pro",
        ]
        for m_name in candidate_models:
            try:
                model = genai.GenerativeModel(m_name)
                res = model.generate_content(prompt_text)
                if res and res.text:
                    return res.text
            except Exception:
                continue

        for m in genai.list_models():
            if "generateContent" in m.supported_generation_methods:
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
    """从本地 OpenD 网关获取真实持仓数据"""
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
# 3. 机构微观结构计算 (Volume Profile & Options)
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


def fetch_options_microstructure(ticker_obj, cur_price):
    try:
        expirations = ticker_obj.options
        if not expirations:
            return {
                "max_pain": 0.0,
                "pcr": 1.0,
                "major_call_wall": 0.0,
                "major_put_wall": 0.0,
            }

        opt_chain = ticker_obj.option_chain(expirations[0])
        calls, puts = opt_chain.calls, opt_chain.puts

        total_call_oi = calls["openInterest"].fillna(0).sum()
        total_put_oi = puts["openInterest"].fillna(0).sum()
        pcr = (
            float(total_put_oi / total_call_oi) if total_call_oi > 0 else 1.0
        )

        call_wall = (
            float(calls.loc[calls["openInterest"].idxmax()]["strike"])
            if not calls.empty and calls["openInterest"].sum() > 0
            else 0.0
        )
        put_wall = (
            float(puts.loc[puts["openInterest"].idxmax()]["strike"])
            if not puts.empty and puts["openInterest"].sum() > 0
            else 0.0
        )

        strikes = sorted(
            list(set(calls["strike"].tolist() + puts["strike"].tolist()))
        )
        loss_dict = {}
        for s in strikes:
            call_loss = (
                np.maximum(0, s - calls["strike"])
                * calls["openInterest"].fillna(0)
            ).sum()
            put_loss = (
                np.maximum(0, puts["strike"] - s)
                * puts["openInterest"].fillna(0)
            ).sum()
            loss_dict[s] = call_loss + put_loss

        max_pain = (
            min(loss_dict, key=loss_dict.get) if loss_dict else cur_price
        )
        return {
            "max_pain": float(max_pain),
            "pcr": float(pcr),
            "major_call_wall": call_wall,
            "major_put_wall": put_wall,
        }
    except Exception:
        return {
            "max_pain": 0.0,
            "pcr": 1.0,
            "major_call_wall": 0.0,
            "major_put_wall": 0.0,
        }


# ----------------------------------------------------
# 4. 全维量化与 AI 专属投顾诊断
# ----------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_and_analyze(ticker_input, user_cost=0.0, user_qty=0):
    ticker_input = ticker_input.strip().upper()

    macro_tickers = ["SPY", "QQQ", "^VIX", "^TNX"]
    macro_data = yf.download(
        macro_tickers,
        period="3mo",
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    market_status = "🟢 多头顺风：标普(SPY) 与 纳指(QQQ) 均稳居生命线上方。"
    spy_info_str, qqq_info_str = "SPY: 正常", "QQQ: 正常"
    vix_status_str, tnx_status_str = "正常", "正常"
    macro_sentiment_tag = "🟢 情绪向好"

    try:
        if not macro_data.empty:
            close_data = macro_data["Close"]
            spy_c = close_data["SPY"].dropna()
            spy_close = spy_c.iloc[-1]
            spy_prev = spy_c.iloc[-2] if len(spy_c) >= 2 else spy_close
            spy_chg = (spy_close - spy_prev) / spy_prev
            spy_ema20 = EMAIndicator(spy_c, 20).ema_indicator().iloc[-1]
            spy_info_str = f"SPY: ${spy_close:.2f} ({spy_chg*100:+.2f}%)"

            qqq_c = close_data["QQQ"].dropna()
            qqq_close = qqq_c.iloc[-1]
            qqq_prev = qqq_c.iloc[-2] if len(qqq_c) >= 2 else qqq_close
            qqq_chg = (qqq_close - qqq_prev) / qqq_prev
            qqq_ema20 = EMAIndicator(qqq_c, 20).ema_indicator().iloc[-1]
            qqq_info_str = f"QQQ: ${qqq_close:.2f} ({qqq_chg*100:+.2f}%)"

            vix_close = close_data["^VIX"].dropna().iloc[-1]
            vix_status_str = (
                f"⚠️ 恐慌高企 (VIX={vix_close:.2f})"
                if vix_close > 22
                else f"🟢 恐慌平稳 (VIX={vix_close:.2f})"
            )
            tnx_close = close_data["^TNX"].dropna().iloc[-1]
            tnx_status_str = f"10Y美债收益率: {tnx_close:.2f}%"

            if vix_close >= 25:
                macro_sentiment_tag = "🔴 极端恐慌避险"
            elif vix_close <= 15:
                macro_sentiment_tag = "🔥 极度贪婪活跃"
            else:
                macro_sentiment_tag = "🟢 平稳健康"

            if (spy_close < spy_ema20 and qqq_close < qqq_ema20) or (
                vix_close >= 25
            ):
                market_status = (
                    "🔴 极度预警：标普(SPY) 与 纳指(QQQ)"
                    " 双双跌破EMA20，全市场防守！"
                )
            elif spy_close < spy_ema20:
                market_status = (
                    "⚠️ 警示：标普(SPY) 跌破均线生命线，大盘传统权重股走弱！"
                )
            elif qqq_close < qqq_ema20:
                market_status = (
                    "⚠️ 警示：纳指(QQQ) 跌破均线生命线，科技成长股短线承压！"
                )
    except Exception:
        pass

    ticker_obj = yf.Ticker(ticker_input)
    df_daily = yf.download(
        ticker_input,
        period="2y",
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if df_daily.empty:
        return (
            None,
            f"未找到股票 [{ticker_input}] 的数据，请检查代码是否正确。",
        )
    if isinstance(df_daily.columns, pd.MultiIndex):
        df_daily.columns = df_daily.columns.get_level_values(0)

    close_d = df_daily["Close"].dropna()
    high_d = df_daily["High"].dropna()
    low_d = df_daily["Low"].dropna()
    cur_price = close_d.iloc[-1]
    total_days = len(close_d)

    low_30d = low_d.iloc[-min(30, total_days) :].min()
    high_30d = high_d.iloc[-min(30, total_days) :].max()

    # 日内 VWAP
    vwap_price = cur_price
    try:
        df_intraday = yf.download(
            ticker_input,
            period="1d",
            interval="5m",
            auto_adjust=True,
            progress=False,
        )
        if not df_intraday.empty:
            if isinstance(df_intraday.columns, pd.MultiIndex):
                df_intraday.columns = df_intraday.columns.get_level_values(0)
            typical_p = (
                df_intraday["High"]
                + df_intraday["Low"]
                + df_intraday["Close"]
            ) / 3.0
            valid_vol = df_intraday["Volume"]
            if valid_vol.sum() > 0:
                vwap_price = (typical_p * valid_vol).sum() / valid_vol.sum()
    except Exception:
        vwap_price = cur_price

    vwap_status_desc = (
        "多头主导(高于日内成本)"
        if cur_price > vwap_price * 1.002
        else (
            "空头压制(低于日内成本)"
            if cur_price < vwap_price * 0.998
            else "多空平衡(紧贴成本)"
        )
    )

    ema5 = EMAIndicator(close_d, min(5, total_days)).ema_indicator().iloc[-1]
    ema20 = EMAIndicator(close_d, min(20, total_days)).ema_indicator().iloc[-1]
    ma30 = SMAIndicator(close_d, min(30, total_days)).sma_indicator().iloc[-1]
    ma60 = (
        SMAIndicator(close_d, 60).sma_indicator().iloc[-1]
        if total_days >= 60
        else None
    )
    ma250 = (
        SMAIndicator(close_d, 250).sma_indicator().iloc[-1]
        if total_days >= 250
        else None
    )

    gap_support = None
    prev_close_p = close_d.iloc[-2] if total_days >= 2 else cur_price
    if total_days >= 2:
        recent_low = low_d.iloc[-1]
        prev_high = high_d.iloc[-2]
        if recent_low > prev_high:
            gap_support = round(recent_low, 2)
        elif recent_low > prev_close_p:
            gap_support = round(prev_close_p, 2)

    atr_d = (
        AverageTrueRange(high_d, low_d, close_d, min(14, total_days))
        .average_true_range()
        .iloc[-1]
    )
    vp_data = calculate_institutional_volume_profile(
        df_daily.iloc[-min(252, total_days) :]
    )
    opt_data = fetch_options_microstructure(ticker_obj, cur_price)

    target1_p = (
        vp_data["resistances"][0]
        if vp_data["resistances"]
        else (
            ma60
            if ma60 and ma60 > cur_price
            else (high_30d if high_30d > cur_price else cur_price * 1.05)
        )
    )
    dynamic_stop_loss = max(low_30d, cur_price - (1.5 * atr_d))
    reward_space = max(0.01, target1_p - cur_price)
    risk_space = max(0.01, cur_price - dynamic_stop_loss)
    rr_ratio = reward_space / risk_space

    support_dict, resistance_dict = {}, {}

    def add_level(name, val):
        if val and val > 0:
            if val < cur_price:
                support_dict[name] = val
            else:
                resistance_dict[name] = val

    add_level("日内成本线 (VWAP)", vwap_price)
    add_level("短线均线 (EMA5)", ema5)
    add_level("生命线 (EMA20)", ema20)
    add_level("中线均线 (MA30)", ma30)
    if ma60:
        add_level("季线 (MA60)", ma60)
    if ma250:
        add_level("年线 (MA250)", ma250)
    if gap_support:
        add_level("短线跳空缺口", gap_support)
    if vp_data["poc"] > 0:
        add_level("筹码密集峰 (POC)", vp_data["poc"])
    if vp_data["vah"] > 0:
        add_level("价值区上沿 (VAH)", vp_data["vah"])
    if vp_data["val"] > 0:
        add_level("价值区下沿 (VAL)", vp_data["val"])
    if opt_data["max_pain"] > 0:
        add_level("期权最大痛点 (Max Pain)", opt_data["max_pain"])
    if opt_data["major_call_wall"] > 0:
        add_level("期权Call大单阻力墙", opt_data["major_call_wall"])
    if opt_data["major_put_wall"] > 0:
        add_level("期权Put大单支撑墙", opt_data["major_put_wall"])

    sorted_supports = sorted(
        support_dict.items(), key=lambda x: x[1], reverse=True
    )
    sorted_resistances = sorted(resistance_dict.items(), key=lambda x: x[1])
    support_list_fmt = [f"{k}: **${v:.2f}**" for k, v in sorted_supports[:5]]
    resistance_list_fmt = [
        f"{k}: **${v:.2f}**" for k, v in sorted_resistances[:5]
    ]

    position_context = ""
    if user_qty > 0:
        pnl_pct = ((cur_price - user_cost) / user_cost) * 100
        position_context = f"""
【用户真实持仓专属数据】：
- 持仓均价: **${user_cost:.2f}** ｜ 持仓股数: **{user_qty} 股**
- 当前浮动盈亏: **{pnl_pct:+.2f}%**
请以【私人专属投资顾问】身份直接指导：
1. **持仓处理动作**：继续持有 / 阶梯止盈减仓（在哪个阻力位减多少） / 上移移动止损。
2. **移动止损保护线**：为保护利润或本金，止损建议调整到哪个具体价格？
"""
    else:
        position_context = (
            "【用户当前状态】：空仓观望中。请根据严格盈亏比给出最佳开仓条件。"
        )

    layered_prompt = f"""
你是一名身经百战的华尔街资深量化操盘手兼新手实战导师。
请根据以下【大盘联动 + 经典均线/缺口 + 机构筹码分布(Volume Profile) + 期权链与ATR风控】，为用户输出一份全通透的私人操盘指南。

【标的】: {ticker_input} ｜ 最新现价: **${cur_price:.2f}**
【大盘指数环境】: {market_status} ｜ {spy_info_str} ｜ {qqq_info_str} ｜ 情绪度: {macro_sentiment_tag} (VIX: {vix_status_str})
【经典均线与缺口】: EMA5: **${ema5:.2f}** ｜ EMA20生命线: **${ema20:.2f}** ｜ MA30: **${ma30:.2f}** ｜ 季线(MA60): {f"${ma60:.2f}" if ma60 else '无'} ｜ 缺口支撑: {f"${gap_support:.2f}" if gap_support else '无'}
【机构微观结构】: 做市商日内VWAP: **${vwap_price:.2f}** ({vwap_status_desc}) ｜ 筹码中心(POC): **${vp_data['poc']:.2f}** ｜ 价值区(VAL~VAH): **${vp_data['val']:.2f} ~ ${vp_data['vah']:.2f}**
【期权博弈与波动率】: 期权Max Pain: **${opt_data['max_pain']:.2f}** ｜ PCR: **{opt_data['pcr']:.2f}** ｜ Call阻力墙: **${opt_data['major_call_wall']:.2f}** ｜ 14日ATR: **${atr_d:.2f}**
【动态盈亏比】: **{rr_ratio:.2f} : 1** ｜ 建议保护止损: **${dynamic_stop_loss:.2f}** ｜ 上方第一阻力目标: **${target1_p:.2f}**
{position_context}

【核心输出原则】：
1. **彻底说人话，严禁只有黑话！** 遇到专业术语必须紧跟括号大白话说明（例如：VAH(价值区上沿阻力)、POC(最密集主力持仓价)）。
2. 所有价格数字统一紧跟美元符号加粗（如 **$220.50**，**+4.39%**）。
3. 严格按照以下 4 个板块输出：

---
### 🚦 1. 操盘手 3 秒极简决策灯 (新手直接看这里)
- **核心操作定性**：直接给大白话动作（【🟢 顺势轻仓试探】 / 【🟡 观望等回踩】 / 【🔴 风险过大坚决不追/执行止损】）。
- **一句话大白话理由**：结合大盘走势、日内VWAP位置与当前 **{rr_ratio:.2f}:1** 的盈亏比，讲透为什么现在该买、该卖还是该等。

### 🛡️ 2. 跌势与吸筹指南（跌了怎么买，阶梯防守）
- **第 1 关（短线浅回调加仓点）**：明确指出回踩哪个具体价格（如 VWAP / EMA20 / 缺口）可以分批建底仓，为什么？
- **第 2 关（波段深度吸筹大底）**：万一大盘回调，主力筹码峰(POC)或中线均线在哪个价格可以安全补仓？
- **飞刀熔断防线（硬止损）**：跌破哪个价格（结合 **${dynamic_stop_loss:.2f}**）说明趋势彻底走坏，必须无条件止损？

### 🎯 3. 涨势与止盈指南（涨了怎么卖，阶梯撤退）
- **第一目标位（近端阻力锁定利润）**：反弹到哪个阻力位/VAH建议减仓 1/3 ~ 1/2？距离现价还有多少百分比？
- **顺势爆发加速位**：带量突破哪个价格（结合期权 Call Wall）可判定进入主升浪，允许顺势推仓？

### 🧠 4. 机构微观与衍生品深度透视 (进阶与专业交易员精读)
- 结合大盘（SPY/QQQ）、期权 PCR 与筹码真空区，用 2-3 句话拆解主力与做市商目前的博弈意图（是在借势逼空、高位派发还是震荡洗盘）。
"""
    ai_analysis_text = call_gemini_smart(layered_prompt)

    top_faqs = [
        f"🚀 {ticker_input} 距离上方第一目标还有多少%？突破难度如何？",
        f"🛡️ 结合均线与筹码，{ticker_input} 最安全的吸筹买点在哪个价位？",
        (
            f"⚖️ 当前位置的盈亏比 ({rr_ratio:.2f}:1) 是否划算？操盘手怎么看？"
        ),
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
        "ema20": ema20,
        "ma30": ma30,
        "ma60_str": f"${ma60:.2f}" if ma60 else "无",
        "ma250_str": f"${ma250:.2f}" if ma250 else "无",
        "gap_support": gap_support,
        "atr_d": atr_d,
        "dynamic_stop_loss": dynamic_stop_loss,
        "vp_data": vp_data,
        "opt_data": opt_data,
        "support_list_fmt": support_list_fmt,
        "resistance_list_fmt": resistance_list_fmt,
        "rr_ratio": rr_ratio,
        "target1_p": target1_p,
        "top_faqs": top_faqs,
        "ai_analysis_text": ai_analysis_text,
        "cache_display_time": cache_display_time,
    }, None


# ----------------------------------------------------
# 5. 后台盯盘与风控判定引擎 (联动 Pushover)
# ----------------------------------------------------
if "alert_logs" not in st.session_state:
    st.session_state.alert_logs = []


def run_portfolio_monitor_task(watchlist):
    """遍历持仓与关注列表进行实时风控扫描并推送 Pushover"""
    triggered_alerts = []
    for item in watchlist:
        sym = item["symbol"]
        cost = item.get("cost", 0.0)
        qty = item.get("qty", 0)

        data, err = fetch_and_analyze(sym, user_cost=cost, user_qty=qty)
        if not data:
            continue

        cur_p = data["cur_price"]
        vwap_p = data["vwap_price"]
        stop_p = data["dynamic_stop_loss"]
        target_p = data["target1_p"]

        # 1. 触发硬止损 (高优先级报警声)
        if cur_p < stop_p:
            msg = f"【止损预警】{sym} 跌破量化防线！\n现价: ${cur_p:.2f} 已跌破硬止损位 ${stop_p:.2f} (POC/ATR防线)。\n建议：立即执行防守避险。"
            triggered_alerts.append(msg)
            send_pushover_alert(
                msg,
                title=f"🚨 止损预警: {sym}",
                priority=1,
                sound="falling",
            )

        # 2. 触达第一目标止盈 (收银机提示音)
        elif cur_p >= target_p:
            msg = f"【阶梯止盈】{sym} 触及第一阻力目标！\n现价: ${cur_p:.2f} 达到目标 ${target_p:.2f}。\n建议：阶梯止盈 1/3 ~ 1/2 锁定利润，底仓上移止损。"
            triggered_alerts.append(msg)
            send_pushover_alert(
                msg,
                title=f"🎯 止盈达成: {sym}",
                priority=0,
                sound="cashregister",
            )

        # 3. 突破做市商 VWAP
        elif cur_p > vwap_p and (cur_p - vwap_p) / vwap_p < 0.015:
            msg = f"【异动提醒】{sym} 站上做市商 VWAP 成本线！\n现价: ${cur_p:.2f} 突破 ${vwap_p:.2f}，主力由被动转为主动护盘。"
            triggered_alerts.append(msg)
            send_pushover_alert(
                msg,
                title=f"⚡ 突破提醒: {sym}",
                priority=0,
                sound="intermission",
            )

    return triggered_alerts


# ----------------------------------------------------
# 6. UI 侧边栏配置
# ----------------------------------------------------
st.sidebar.header("⚙️ 账户与连接配置")

connect_mode = st.sidebar.radio(
    "持仓数据模式", ["Moomoo OpenD 本地直连", "手动维护 / 模拟持仓"]
)

portfolio_list = []

if connect_mode == "Moomoo OpenD 本地直连":
    opend_host = st.sidebar.text_input("OpenD IP", value="127.0.0.1")
    opend_port = st.sidebar.number_input("OpenD 端口", value=11111)

    if st.sidebar.button("🔄 同步 Moomoo 真实持仓", use_container_width=True):
        with st.spinner("正在直连 Moomoo OpenD 获取持仓..."):
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
    st.sidebar.write("**手动录入持仓:**")
    m_sym = st.sidebar.text_input("股票代码", value="USAR").upper()
    m_cost = st.sidebar.number_input("持仓成本 ($)", value=16.80)
    m_qty = st.sidebar.number_input("持仓股数", value=100)

    if st.sidebar.button("➕ 加入监控持仓"):
        if "manual_positions" not in st.session_state:
            st.session_state.manual_positions = []
        st.session_state.manual_positions.append(
            {"symbol": m_sym, "cost": m_cost, "qty": m_qty}
        )

    portfolio_list = st.session_state.get(
        "manual_positions", [{"symbol": m_sym, "cost": m_cost, "qty": m_qty}]
    )

st.sidebar.divider()
st.sidebar.subheader("📲 Pushover 推送状态")
push_status = (
    "🟢 已就绪"
    if PUSHOVER_USER_KEY and PUSHOVER_API_TOKEN
    else "⚪ 未配置 (仅面板提示)"
)
st.sidebar.caption(f"Pushover 状态: {push_status}")

if st.sidebar.button("⚡ 立即全维扫描持仓风控", type="primary", use_container_width=True):
    with st.spinner("量化引擎正在全局扫描持仓攻防点位..."):
        alerts = run_portfolio_monitor_task(portfolio_list)
        if alerts:
            st.session_state.alert_logs.extend(alerts)
            st.sidebar.success(
                f"扫描完成，触发 {len(alerts)} 条异动预警并推送至手机！"
            )
        else:
            st.sidebar.info("持仓运行平稳，暂未触及止损或止盈触发线。")

# ----------------------------------------------------
# 7. 主界面
# ----------------------------------------------------
tab1, tab2, tab3 = st.tabs(
    ["📊 私人持仓全景透视", "🔍 单股深度全维诊断", "🚨 实时预警日志"]
)

# --- TAB 1: 持仓透视 ---
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

        if st.button(
            f"生成 {selected_pos_sym} 专属投顾报告",
            type="primary",
            key="btn_pos_diag",
        ):
            with st.spinner("专属 AI 投顾正在根据你的持仓计算执行指令..."):
                diag_data, d_err = fetch_and_analyze(
                    selected_pos_sym,
                    user_cost=selected_item["cost"],
                    user_qty=selected_item["qty"],
                )
                if d_err:
                    st.error(d_err)
                else:
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("现价", f"${diag_data['cur_price']:.2f}")
                    col2.metric("做市商 VWAP", f"${diag_data['vwap_price']:.2f}")
                    col3.metric(
                        "筹码中心 (POC)", f"${diag_data['vp_data']['poc']:.2f}"
                    )
                    col4.metric(
                        "建议防守止损", f"${diag_data['dynamic_stop_loss']:.2f}"
                    )

                    st.markdown("---")
                    st.subheader(f"🤖 投顾执行报告 ({selected_pos_sym})")
                    safe_render_markdown(diag_data["ai_analysis_text"])
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

    ticker_input = st.text_input(
        "美股代码", value=st.session_state.selected_ticker
    ).strip().upper()

    if st.button("开始全维实战闭环诊断", type="primary", use_container_width=True):
        if ticker_input and ticker_input in st.session_state.history_tickers:
            st.session_state.history_tickers.remove(ticker_input)
        if ticker_input:
            st.session_state.history_tickers.insert(0, ticker_input)
            if len(st.session_state.history_tickers) > 5:
                st.session_state.history_tickers.pop()

        with st.spinner(
            f"正在全维运算均线、筹码分布与衍生品博弈 ({ticker_input})..."
        ):
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

        st.caption(
            f"⚡ 数据已智能缓存 (刷新时间: {data['cache_display_time']}) ｜"
            " 5分钟内共享无消耗"
        )

        if "🔴" in data["market_status"]:
            st.error(
                f"**大盘风控:** {data['market_status']} ｜ {data['spy_info_str']}"
                f" ｜ {data['qqq_info_str']}"
            )
        elif "⚠️" in data["market_status"]:
            st.warning(
                f"**大盘风控:** {data['market_status']} ｜ {data['spy_info_str']}"
                f" ｜ {data['qqq_info_str']}"
            )
        else:
            st.success(
                f"**大盘风控:** {data['market_status']} ｜ {data['spy_info_str']}"
                f" ｜ {data['qqq_info_str']}"
            )

        st.info(
            f"🌐 **市场流动性环境：** 【{data['macro_sentiment_tag']}】 ｜"
            f" {data['vix_status_str']} ｜ 🏛️ {data['tnx_status_str']}"
        )

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric(
            label=f"{curr_ticker} 现价", value=f"${data['cur_price']:.2f}"
        )
        col_m2.metric(
            label="做市商成本 (VWAP)", value=f"${data['vwap_price']:.2f}"
        )
        rr_delta = "🟢 优秀" if data["rr_ratio"] >= 2.0 else "⚠️ 偏低/一般"
        col_m3.metric(
            label="动态盈亏比",
            value=f"{data['rr_ratio']:.2f} : 1",
            delta=rr_delta,
        )

        col_q1, col_q2, col_q3 = st.columns(3)
        col_q1.metric(
            label="筹码密集区 (POC)", value=f"${data['vp_data']['poc']:.2f}"
        )
        col_q2.metric(
            label="期权痛点 (Max Pain)",
            value=f"${data['opt_data']['max_pain']:.2f}",
        )
        col_q3.metric(
            label="动态保护止损 (1.5x ATR)",
            value=f"${data['dynamic_stop_loss']:.2f}",
        )

        st.subheader("🤖 操盘手分层实战手册 (全维落地指令)")
        safe_render_markdown(data["ai_analysis_text"])

        st.subheader("🛡️ 全景关键阶梯防线 (均线/缺口/筹码共振)")
        col1, col2 = st.columns(2)
        with col1:
            st.info(
                "**【🟢 阶梯支撑与吸筹带（由近及远）】**\n\n"
                + "\n\n".join(data["support_list_fmt"])
            )
        with col2:
            st.warning(
                "**【🔴 阶梯阻力与出清目标（由近及远）】**\n\n"
                + "\n\n".join(data["resistance_list_fmt"])
            )

        st.divider()
        st.subheader("💬 操盘手智能追问助理")
        clicked_faq = None
        if "top_faqs" in data and data["top_faqs"]:
            for idx, faq_text in enumerate(data["top_faqs"]):
                if st.button(
                    faq_text, key=f"faq_tab2_{idx}", use_container_width=True
                ):
                    clicked_faq = faq_text

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                safe_render_markdown(msg["content"])

        user_input = st.chat_input(
            "自由提问（如：到230有多少%？跌破EMA20怎么看？做市商防守位在哪？）..."
        )
        prompt_to_process = user_input or clicked_faq

        if prompt_to_process:
            st.session_state.chat_history.append(
                {"role": "user", "content": prompt_to_process}
            )
            with st.chat_message("user"):
                safe_render_markdown(prompt_to_process)

            with st.chat_message("assistant"):
                with st.spinner(
                    "操盘智脑正在结合全景均线与微观订单流推演..."
                ):
                    extracted_symbols = extract_tickers_from_text(
                        prompt_to_process
                    )
                    extra_data_text = ""
                    for sym in extracted_symbols:
                        if sym != curr_ticker:
                            try:
                                other_data, _ = fetch_and_analyze(sym)
                                if other_data:
                                    extra_data_text += f"""
                                    【联动标的 {sym} 关键数据】:
                                    现价: ${other_data['cur_price']:.2f} | 盈亏比: {other_data['rr_ratio']:.2f}:1 | VWAP: ${other_data['vwap_price']:.2f} | EMA20: ${other_data['ema20']:.2f}
                                    """
                            except Exception:
                                pass

                    chat_context_prompt = f"""
你是一名顶级美股操盘手兼量化导师。你说话干练、接地气，既精通微观期权筹码，也能用最通俗的大白话把交易点位讲透。

【当前标的】: {curr_ticker} ｜ 现价: **${data['cur_price']:.2f}**
【大盘环境】: {data['market_status']} ｜ {data['spy_info_str']} ｜ {data['qqq_info_str']}
【均线体系】: EMA5: **${data['ema5']:.2f}** ｜ EMA20: **${data['ema20']:.2f}** ｜ MA30: **${data['ma30']:.2f}** ｜ 季线: {data['ma60_str']} ｜ 缺口: {f"${data['gap_support']:.2f}" if data['gap_support'] else '无'}
【微观筹码与期权】: 做市商VWAP: **${data['vwap_price']:.2f}** ｜ POC密集峰: **${data['vp_data']['poc']:.2f}** ｜ 价值区: **${data['vp_data']['val']:.2f} ~ ${data['vp_data']['vah']:.2f}** ｜ 期权Max Pain: **${data['opt_data']['max_pain']:.2f}**
【风控基准】: 14日ATR: **${data['atr_d']:.2f}** ｜ 建议动态止损: **${data['dynamic_stop_loss']:.2f}** ｜ 动态盈亏比: **{data['rr_ratio']:.2f} : 1**
{extra_data_text}

用户的真实提问是: "{prompt_to_process}"

【回答规范】：
1. **直奔主题给结论**：问涨跌空间必做数学计算并给出精确百分比；问买点必给具体价格区间与均线/筹码理由；问盈亏比直接定性（划算/不划算）。
2. **通俗与专业兼备**：如果用到微观指标，必须用大白话解释其对散户交易的实际含义。
3. **数字严格加粗**：所有涉及的价格数字紧跟美元符号加粗（如 **$230.47**，**+4.39%**）。
"""
                    reply_text = call_gemini_smart(chat_context_prompt)
                    safe_render_markdown(reply_text)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": reply_text}
                    )

# --- TAB 3: 预警日志 ---
with tab3:
    st.subheader("🚨 盘中实时风控与买卖预警日志")
    if st.session_state.alert_logs:
        for log in reversed(st.session_state.alert_logs[-10:]):
            st.warning(log)
    else:
        st.info("暂无最新告警记录。点击左侧【立即全维扫描持仓风控】即可执行一次全局巡检。")
