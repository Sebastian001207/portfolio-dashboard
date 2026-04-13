import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import numpy as np
from datetime import date, timedelta
from IPython.display import display
import ipywidgets as widgets

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
# Note: BTCUSD→BTC-USD, ETHUSD→ETH-USD, RHM.DE skipped (limited yfinance support)
# ══════════════════════════════════════════

def get_last_trading_day():
    today = date.today()
    offset = {0: 3, 6: 2, 5: 1}
    days_back = offset.get(today.weekday(), 1)
    return (today - timedelta(days=days_back)).strftime("%Y-%m-%d")

END_DATE = get_last_trading_day()
print(f"📅 Period: {START_DATE} → {END_DATE}")
print(f"📊 Benchmark: {BENCHMARK}")
print(f"🔍 Fetching {len(TICKERS)+1} tickers...\n")

# ── Fetch all tickers ─────────────────────────────────────────────────
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

# Fetch benchmark
bench = fetch_ticker(BENCHMARK, START_DATE, END_DATE)
print(f"✅ {BENCHMARK} (benchmark): {len(bench)} rows")

# Fetch all portfolio tickers
portfolio_data = {}
failed = []
for ticker in TICKERS:
    df = fetch_ticker(ticker, START_DATE, END_DATE)
    if df is not None and len(df) > 30:
        portfolio_data[ticker] = df
        print(f"  ✅ {ticker}: {len(df)} rows")
    else:
        failed.append(ticker)
        print(f"  ❌ {ticker}: failed or insufficient data")

print(f"\n✅ Loaded: {len(portfolio_data)} | ❌ Failed: {len(failed)}")
if failed:
    print(f"   Failed tickers: {failed}")

# ── Compute indicators for all tickers ───────────────────────────────
def compute_indicators(df, bench_df):
    merged = pd.merge(
        bench_df[["Date", "Close", "% Change"]],
        df[["Date", "Close", "% Change"]],
        on="Date", suffixes=("_bench", "_stock")
    ).dropna().reset_index(drop=True)

    if len(merged) < 30:
        return None

    # Returns
    merged["Cum_bench"] = (1 + merged["% Change_bench"]).cumprod()
    merged["Cum_stock"] = (1 + merged["% Change_stock"]).cumprod()
    merged["RS_Line"]   = merged["Cum_stock"] / merged["Cum_bench"]
    merged["RS_Norm"]   = (merged["RS_Line"] / merged["RS_Line"].iloc[0]) * 100

    # Rolling RS 60d
    if len(merged) >= 30:
        merged["RS_Rolling"] = (
            merged["% Change_stock"].rolling(30).apply(lambda x: (1+x).prod(), raw=True) /
            merged["% Change_bench"].rolling(30).apply(lambda x: (1+x).prod(), raw=True)
        )
    else:
        merged["RS_Rolling"] = np.nan

    # Beta & Correlation
    cov  = merged["% Change_stock"].rolling(30).cov(merged["% Change_bench"])
    var  = merged["% Change_bench"].rolling(30).var()
    merged["Beta"] = cov / var
    merged["Corr"] = merged["% Change_stock"].rolling(30).corr(merged["% Change_bench"])

    # RSI (14-period)
    delta    = merged["Close_stock"].diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs       = avg_gain / avg_loss
    merged["RSI"] = 100 - (100 / (1 + rs))

    return merged

print("\n⚙️  Computing indicators...")
indicators = {}
for ticker, df in portfolio_data.items():
    result = compute_indicators(df, bench)
    if result is not None:
        indicators[ticker] = result

print(f"✅ Indicators computed for {len(indicators)} tickers")

# ── Build Summary Table ───────────────────────────────────────────────
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

# Benchmark row
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

# ══════════════════════════════════════════════════════════════════════
#  INTERACTIVE DASHBOARD
# ══════════════════════════════════════════════════════════════════════

# ── Widget 1: Summary Heatmap Table ──────────────────────────────────
def plot_heatmap(sort_col="YTD %"):
    df = summary[summary["Ticker"] != f"{BENCHMARK} (bench)"].copy()
    df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(16, max(8, len(df) * 0.35)))
    ax.axis("off")

    cols    = ["Ticker", "Last Close", "1M %", "3M %", "6M %", "YTD %", "RSI", "Beta", "Corr", "RS Norm", "RS Roll 60d"]
    cell_text = df[cols].values.tolist()
    col_labels = cols

    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    # Color cells
    pct_cols  = [cols.index(c) for c in ["1M %", "3M %", "6M %", "YTD %"]]
    rs_cols   = [cols.index(c) for c in ["RS Norm", "RS Roll 60d"]]
    rsi_col   = cols.index("RSI")

    for i, row in df.iterrows():
        for j, col in enumerate(cols):
            cell = table[i+1, j]
            val  = df.iloc[i][col]

            if j in pct_cols and not pd.isna(val):
                intensity = min(abs(val) / 30, 1)
                color = (1 - intensity * 0.6, 1, 1 - intensity * 0.6) if val > 0 else (1, 1 - intensity * 0.6, 1 - intensity * 0.6)
                cell.set_facecolor(color)
            elif j in rs_cols and not pd.isna(val):
                color = "#d4edda" if val > 100 else "#f8d7da"
                cell.set_facecolor(color)
            elif j == rsi_col and not pd.isna(val):
                if val > 70:   cell.set_facecolor("#f8d7da")
                elif val < 30: cell.set_facecolor("#d4edda")
                else:          cell.set_facecolor("#fff3cd")

    # Header style
    for j in range(len(cols)):
        table[0, j].set_facecolor("#2c3e50")
        table[0, j].set_text_props(color="white", fontweight="bold")

    plt.title(f"Portfolio vs {BENCHMARK} — Sorted by {sort_col} | {START_DATE} → {END_DATE}",
              fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.show()

# ── Widget 2: Individual Stock Deep Dive ─────────────────────────────
def plot_stock(ticker):
    if ticker not in indicators:
        print(f"No data for {ticker}")
        return

    df  = indicators[ticker]
    fig = plt.figure(figsize=(16, 26))
    fig.suptitle(f"{ticker} vs {BENCHMARK} — Full Indicator Dashboard",
                 fontsize=14, fontweight="bold")
    gs  = gridspec.GridSpec(7, 1, hspace=0.55)

    # ── Row 0: Price ──────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(df["Date"], df["Close_stock"], color="royalblue", linewidth=1.8, label=ticker)
    ax0_ = ax0.twinx()
    ax0_.plot(df["Date"], df["Close_bench"], color="gray", linewidth=1, linestyle="--", label=BENCHMARK)
    ax0.set_title(f"Price — {ticker} (blue) vs {BENCHMARK} (gray dashed)")
    ax0.set_ylabel(f"{ticker} Price")
    ax0_.set_ylabel(f"{BENCHMARK} Price", color="gray")
    ax0.grid(True, alpha=0.3)

    # ── Row 1: Cumulative Return ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[1])
    ax1.plot(df["Date"], (df["Cum_stock"] - 1) * 100, color="royalblue", linewidth=1.8, label=ticker)
    ax1.plot(df["Date"], (df["Cum_bench"] - 1) * 100, color="gray",      linewidth=1.5, linestyle="--", label=BENCHMARK)
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax1.set_title("① Cumulative Return (%) — Path dependent, reference only")
    ax1.set_ylabel("Return (%)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ── Row 2: RS Normalized to 100 ──────────────────────────────────
    ax2 = fig.add_subplot(gs[2])
    ax2.plot(df["Date"], df["RS_Norm"], color="teal", linewidth=1.5)
    ax2.axhline(100, color="gray", linestyle="--", linewidth=1, label="Starting point (100)")
    ax2.fill_between(df["Date"], df["RS_Norm"], 100,
                     where=(df["RS_Norm"] > 100), alpha=0.15, color="green", label=f"{ticker} Outperforming")
    ax2.fill_between(df["Date"], df["RS_Norm"], 100,
                     where=(df["RS_Norm"] < 100), alpha=0.15, color="red",   label=f"{BENCHMARK} Outperforming")
    ax2.set_title("② RS Line Normalized to 100 — Easier to compare across periods")
    ax2.set_ylabel("RS Index (base=100)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ── Row 3: Rolling 30-Day RS ──────────────────────────────────────
    ax3 = fig.add_subplot(gs[3])
    ax3.plot(df["Date"], df["RS_Rolling"], color="purple", linewidth=1.5)
    ax3.axhline(1, color="gray", linestyle="--", linewidth=1, label="Equal 30-day return")
    ax3.fill_between(df["Date"], df["RS_Rolling"], 1,
                     where=(df["RS_Rolling"] > 1), alpha=0.15, color="green", label=f"{ticker} stronger (30d)")
    ax3.fill_between(df["Date"], df["RS_Rolling"], 1,
                     where=(df["RS_Rolling"] < 1), alpha=0.15, color="red",   label=f"{BENCHMARK} stronger (30d)")
    ax3.set_title("③ Rolling 30-Day RS — Start date neutral, shows recent momentum")
    ax3.set_ylabel("Rolling RS Ratio")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ── Row 4: RSI ────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[4])
    ax4.plot(df["Date"], df["RSI"], color="royalblue", linewidth=1.5)
    ax4.axhline(70, color="red",   linestyle="--", linewidth=1, label="Overbought (70)")
    ax4.axhline(50, color="gray",  linestyle=":",  linewidth=1, label="Midline (50)")
    ax4.axhline(30, color="green", linestyle="--", linewidth=1, label="Oversold (30)")
    ax4.fill_between(df["Date"], df["RSI"], 70, where=(df["RSI"] >= 70), alpha=0.2, color="red")
    ax4.fill_between(df["Date"], df["RSI"], 30, where=(df["RSI"] <= 30), alpha=0.2, color="green")
    ax4.set_title("④ RSI (14)")
    ax4.set_ylabel("RSI")
    ax4.set_ylim(0, 100)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # ── Row 5: Rolling Beta ───────────────────────────────────────────
    ax5 = fig.add_subplot(gs[5])
    ax5.plot(df["Date"], df["Beta"], color="darkorange", linewidth=1.5)
    ax5.axhline(1,   color="gray", linestyle="--", linewidth=1,   label="Beta = 1")
    ax5.axhline(1.5, color="red",  linestyle=":",  linewidth=0.8, label="Beta = 1.5")
    ax5.fill_between(df["Date"], df["Beta"], 1,
                     where=(df["Beta"] > 1), alpha=0.15, color="red",   label=f"{ticker} more volatile")
    ax5.fill_between(df["Date"], df["Beta"], 1,
                     where=(df["Beta"] < 1), alpha=0.15, color="green", label=f"{ticker} less volatile")
    ax5.set_title("⑤ Rolling 30-Day Beta — Not start date sensitive ✅")
    ax5.set_ylabel("Beta")
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # ── Row 6: Rolling Correlation ────────────────────────────────────
    ax6 = fig.add_subplot(gs[6])
    ax6.plot(df["Date"], df["Corr"], color="steelblue", linewidth=1.5)
    ax6.axhline(1.0, color="green", linestyle="--", linewidth=0.8, label="Perfect correlation (1.0)")
    ax6.axhline(0.8, color="gray",  linestyle=":",  linewidth=0.8, label="0.8 threshold")
    ax6.fill_between(df["Date"], df["Corr"], 0.8,
                     where=(df["Corr"] < 0.8), alpha=0.2, color="orange", label="Diverging from SPY")
    ax6.set_title("⑥ Rolling 30-Day Correlation — Not start date sensitive ✅")
    ax6.set_ylabel("Correlation")
    ax6.set_ylim(0, 1.1)
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# ── Launch Interactive Widgets ────────────────────────────────────────
sort_dropdown = widgets.Dropdown(
    options=["YTD %", "1M %", "3M %", "6M %", "RSI", "Beta", "RS Norm", "RS Roll 60d"],
    value="YTD %",
    description="Sort by:",
    style={"description_width": "initial"}
)

ticker_dropdown = widgets.Dropdown(
    options=sorted(indicators.keys()),
    description="Deep dive:",
    style={"description_width": "initial"}
)

heatmap_btn = widgets.Button(description="🔄 Refresh Table",  button_style="primary")
deepdive_btn = widgets.Button(description="📈 Show Deep Dive", button_style="success")

def on_heatmap(b):
    plot_heatmap(sort_dropdown.value)

def on_deepdive(b):
    plot_stock(ticker_dropdown.value)

heatmap_btn.on_click(on_heatmap)
deepdive_btn.on_click(on_deepdive)

display(widgets.HBox([sort_dropdown, heatmap_btn]))
display(widgets.HBox([ticker_dropdown, deepdive_btn]))

# Auto-render on load
plot_heatmap("YTD %")