# 初始化历史搜索/自选列表
if "history_tickers" not in st.session_state:
    st.session_state.history_tickers = ["SPCX", "NVDA", "TSLA", "AAPL"]

if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = "SPCX"

# 动态展示自选/历史按钮
st.write("**🔥 快速自选与最近查询:**")
cols = st.columns(len(st.session_state.history_tickers))
for i, ticker in enumerate(st.session_state.history_tickers):
    if cols[i].button(ticker, use_container_width=True):
        st.session_state.selected_ticker = ticker

ticker_input = st.text_input("美股代码", value=st.session_state.selected_ticker).strip().upper()

# 点击诊断时，自动将新查询的代码加入快捷栏顶部（去重，保留最新6个）
if ticker_input and ticker_input not in st.session_state.history_tickers:
    st.session_state.history_tickers.insert(0, ticker_input)
    if len(st.session_state.history_tickers) > 6:
        st.session_state.history_tickers.pop()
