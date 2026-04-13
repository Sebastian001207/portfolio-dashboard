import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from datetime import date, timedelta

# ══════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════
st.set_page_config(
    page_title="Portfolio Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Portfolio vs SPY Dashboard")

# ══════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════
BENCHMARK  = "SPY"
START_DATE = "2026-01-01"

TICKERS = [
    "NVDA", "MSFT", "VHT",  "PEP",  "BRK-B", "GOOG",  "META",  "S",
    "UNH",  "V",    "DPZ",  "ADBE", "AMZN",  "NKE",   "CRM",   "PLTR",
    "GRAB", "CDNS", "TLT",  "TSM",  "NBIS",  "NET",   "HII",   "SNOW",
    "EWJ",  "ORCL", "CAT",  "OXY",  "SNPS",  "TEAM",  "BTC-USD", "AMD",
    "COHR", "LITE", "MU",   "DFEN", "AVGO",  "EQIX",  "ETH-USD", "CRDO",
    "SOFI", "MDB",  "PYPL", "COIN", "INTC",  "IREN",  "ALAB",
]

def get_last_trading_day():
    today = date.today()
    offset = {0: 3, 6: 2, 5: 1}
    days_back = offset.get(today.weekday(), 1)
    return (today - timedelta(days=days_back)).strftime("%Y-%m-%d")

END_DATE = get_last_trading_day()

# ══════════════════════════════════════════
#  DATA FETCHING (cached)
# ══════════════════════════════════════════
@st.cache_data(show_spinner=False)
def fetch_ticker(ticker, start, end):
    try:
        raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if raw.empty:
            return None
        raw.columns = [col[0] for col in raw.columns]
        df = raw[["Close"]].reset_index()
        df.columns = ["Date", "Close"]
        df["% Change"] = df["Close"].pct_change()
        df["Date"] = pd.to_datetime(df["Date"])
        return df.dropna().reset_index(drop=True)
    except:
        return None

def compute_indicators(portfolio_data, bench_df):
    results = {}
    for ticker, df in portfolio_data.items():
        try:
            merged = pd.merge(
                bench_df[["Date", "Close", "% Change"]],
                df[["Date", "Close", "% Change"]],
                on="Date", suffixes=("_bench", "_stock")
            ).dropna().reset_index(drop=True)

            if len(merged) < 30:
                continue

            merged["Cum_bench"] = (1 + merged["% Change_bench"]).cumprod()
            merged["Cum_stock"] = (1 + merged["% Change_stock"]).cumprod()
            merged["RS_Line"]   = merged["Cum_stock"] / merged["Cum_bench"]
            merged["RS_Norm"]   = (merged["RS_Line"] / merged["RS_Line"].iloc[0]) * 100

            merged["RS_Rolling"] = (
                merged["% Change_stock"].rolling(30).apply(lambda x: (1+x).prod(), raw=True) /
                merged["% Change_bench"].rolling(30).apply(lambda x: (1+x).prod(), raw=True)
            )

            cov  = merged["% Change_stock"].rolling(30).cov(merged["% Change_bench"])
            var  = merged["% Change_bench"].rolling(30).var()
            merged["Beta"] = cov / var
            merged["Corr"] = merged["% Change_stock"].rolling(30).corr(merged["% Change_bench"])

            delta    = merged["Close_stock"].diff()
            gain     = delta.clip(lower=0)
            loss     = -delta.clip(upper=0)
            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()
            rs       = avg_gain / avg_loss
            merged["RSI"] = 100 - (100 / (1 + rs))

            results[ticker] = merged
        except Exception as e:
            st.warning(f"Skipping {ticker}: {e}")
            continue
    return results

# ══════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════
with st.spinner("📡 Fetching market data... (this may take a minute on first load)"):
    bench = fetch_ticker(BENCHMARK, START_DATE, END_DATE)

    if bench is None or bench.empty:
        st.error("❌ Failed to fetch benchmark (SPY). Please refresh the page.")
        st.stop()

    portfolio_data = {}
    for ticker in TICKERS:
        df = fetch_ticker(ticker, START_DATE, END_DATE)
        if df is not None and len(df) > 30:
            portfolio_data[ticker] = df

    indicators = compute_indicators(portfolio_data, bench)

st.success(f"✅ Loaded {len(indicators)} tickers | 📅 {START_DATE} → {END_DATE}")

# ══════════════════════════════════════════
#  BUILD SUMMARY TABLE
# ══════════════════════════════════════════
def period_return(df, col, days):
    if len(df) < days:
        return np.nan
    return (df[col].iloc[-1] / df[col].iloc[-days] - 1) * 100

rows = []
for ticker, df in indicators.items():
    rows.append({
        "Ticker":      ticker,
        "Last Close":  round(df["Close_stock"].iloc[-1], 2),
        "1M %":        round(period_return(df, "Close_stock", 21),  2),
        "3M %":        round(period_return(df, "Close_stock", 63),  2),
        "6M %":        round(period_return(df, "Close_stock", 126), 2),
        "YTD %":       round((df["Cum_stock"].iloc[-1] - 1) * 100,  2),
        "RSI":         round(df["RSI"].iloc[-1],         2),
        "Beta":        round(df["Beta"].iloc[-1],        2),
        "Corr":        round(df["Corr"].iloc[-1],        2),
        "RS Norm":     round(df["RS_Norm"].iloc[-1],     2),
        "RS Roll 60d": round(df["RS_Rolling"].iloc[-1],  2) if not df["RS_Rolling"].isna().all() else np.nan,
    })

rows.insert(0, {
    "Ticker":      f"{BENCHMARK} (bench)",
    "Last Close":  round(bench["Close"].iloc[-1], 2),
    "1M %":        round(period_return(bench, "Close", 21),  2),
    "3M %":        round(period_return(bench, "Close", 63),  2),
    "6M %":        round(period_return(bench, "Close", 126), 2),
    "YTD %":       round((bench["Close"].iloc[-1] / bench["Close"].iloc[0] - 1) * 100, 2),
    "RSI":         np.nan,
    "Beta":        1.0,
    "Corr":        1.0,
    "RS Norm":     100.0,
    "RS Roll 60d": 1.0,
})

summary = pd.DataFrame(rows)

# ══════════════════════════════════════════
#  TAB LAYOUT
# ══════════════════════════════════════════
tab1, tab2 = st.tabs(["📋 Summary Heatmap", "📈 Deep Dive"])

# ── TAB 1: Heatmap Table ─────────────────
with tab1:
    st.subheader("Portfolio Heatmap Table")

    sort_col = st.selectbox(
        "Sort by:",
        ["YTD %", "1M %", "3M %", "6M %", "RSI", "Beta", "RS Norm", "RS Roll 60d"],
        index=0
    )

    df_display = summary[summary["Ticker"] != f"{BENCHMARK} (bench)"].copy()
    df_display = df_display.sort_values(sort_col, ascending=False).reset_index(drop=True)

    def color_pct(val):
        if pd.isna(val):
            return ""
        intensity = min(abs(val) / 30, 1)
        if val > 0:
            g = int(200 + intensity * 55)
            return f"background-color: rgb(200, {g}, 200); color: black"
        else:
            r = int(200 + intensity * 55)
            return f"background-color: rgb({r}, 200, 200); color: black"

    def color_rs(val):
        if pd.isna(val):
            return ""
        return "background-color: #d4edda" if val > 100 else "background-color: #f8d7da"

    def color_rsi(val):
        if pd.isna(val):
            return ""
        if val > 70:   return "background-color: #f8d7da"
        elif val < 30: return "background-color: #d4edda"
        else:          return "background-color: #fff3cd"

    styled = df_display.style \
        .applymap(color_pct,  subset=["1M %", "3M %", "6M %", "YTD %"]) \
        .applymap(color_rs,   subset=["RS Norm", "RS Roll 60d"]) \
        .applymap(color_rsi,  subset=["RSI"])

    st.dataframe(styled, use_container_width=True, height=600)

# ── TAB 2: Deep Dive ─────────────────────
with tab2:
    st.subheader("Individual Stock Deep Dive")

    ticker = st.selectbox("Select ticker:", sorted(indicators.keys()))

    if ticker and ticker in indicators:
        df = indicators[ticker]

        fig = plt.figure(figsize=(14, 24))
        fig.suptitle(f"{ticker} vs {BENCHMARK} — Full Indicator Dashboard",
                     fontsize=13, fontweight="bold")
        gs = gridspec.GridSpec(6, 1, hspace=0.55)

        # Price
        ax0 = fig.add_subplot(gs[0])
        ax0.plot(df["Date"], df["Close_stock"], color="royalblue", linewidth=1.8, label=ticker)
        ax0_ = ax0.twinx()
        ax0_.plot(df["Date"], df["Close_bench"], color="gray", linewidth=1, linestyle="--", label=BENCHMARK)
        ax0.set_title(f"Price — {ticker} (blue) vs {BENCHMARK} (gray dashed)")
        ax0.set_ylabel(f"{ticker} Price")
        ax0_.set_ylabel(f"{BENCHMARK} Price", color="gray")
        ax0.grid(True, alpha=0.3)

        # Cumulative Return
        ax1 = fig.add_subplot(gs[1])
        ax1.plot(df["Date"], (df["Cum_stock"] - 1) * 100, color="royalblue", linewidth=1.8, label=ticker)
        ax1.plot(df["Date"], (df["Cum_bench"] - 1) * 100, color="gray", linewidth=1.5, linestyle="--", label=BENCHMARK)
        ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax1.set_title("① Cumulative Return (%)")
        ax1.set_ylabel("Return (%)")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # RS Normalized
        ax2 = fig.add_subplot(gs[2])
        ax2.plot(df["Date"], df["RS_Norm"], color="teal", linewidth=1.5)
        ax2.axhline(100, color="gray", linestyle="--", linewidth=1)
        ax2.fill_between(df["Date"], df["RS_Norm"], 100, where=(df["RS_Norm"] > 100), alpha=0.15, color="green")
        ax2.fill_between(df["Date"], df["RS_Norm"], 100, where=(df["RS_Norm"] < 100), alpha=0.15, color="red")
        ax2.set_title("② RS Line Normalized to 100")
        ax2.set_ylabel("RS Index (base=100)")
        ax2.grid(True, alpha=0.3)

        # RSI
        ax3 = fig.add_subplot(gs[3])
        ax3.plot(df["Date"], df["RSI"], color="royalblue", linewidth=1.5)
        ax3.axhline(70, color="red",   linestyle="--", linewidth=1, label="Overbought (70)")
        ax3.axhline(50, color="gray",  linestyle=":",  linewidth=1)
        ax3.axhline(30, color="green", linestyle="--", linewidth=1, label="Oversold (30)")
        ax3.fill_between(df["Date"], df["RSI"], 70, where=(df["RSI"] >= 70), alpha=0.2, color="red")
        ax3.fill_between(df["Date"], df["RSI"], 30, where=(df["RSI"] <= 30), alpha=0.2, color="green")
        ax3.set_title("③ RSI (14)")
        ax3.set_ylabel("RSI")
        ax3.set_ylim(0, 100)
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

        # Beta
        ax4 = fig.add_subplot(gs[4])
        ax4.plot(df["Date"], df["Beta"], color="darkorange", linewidth=1.5)
        ax4.axhline(1, color="gray", linestyle="--", linewidth=1, label="Beta = 1")
        ax4.fill_between(df["Date"], df["Beta"], 1, where=(df["Beta"] > 1), alpha=0.15, color="red")
        ax4.fill_between(df["Date"], df["Beta"], 1, where=(df["Beta"] < 1), alpha=0.15, color="green")
        ax4.set_title("④ Rolling 30-Day Beta")
        ax4.set_ylabel("Beta")
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)

        # Correlation
        ax5 = fig.add_subplot(gs[5])
        ax5.plot(df["Date"], df["Corr"], color="steelblue", linewidth=1.5)
        ax5.axhline(0.8, color="gray", linestyle=":", linewidth=0.8, label="0.8 threshold")
        ax5.fill_between(df["Date"], df["Corr"], 0.8, where=(df["Corr"] < 0.8), alpha=0.2, color="orange")
        ax5.set_title("⑤ Rolling 30-Day Correlation")
        ax5.set_ylabel("Correlation")
        ax5.set_ylim(0, 1.1)
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
