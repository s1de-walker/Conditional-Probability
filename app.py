import re
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

# ----------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------
st.set_page_config(page_title="Factor & Sector Lab", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px;}
      h1 {letter-spacing: -0.02em;}
      [data-testid="stMetricValue"] {font-size: 1.6rem;}
      div[data-testid="stTabs"] button {font-size: 1rem; padding: 0.5rem 1.1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

DEFAULT_FACTORS = [
    "EQUAL50ADD", "MOVALUE", "LOWVOLIETF", "HDFCQUAL",
    "MOMOMENTUM", "HDFCGROWTH", "DIVOPPBEES", "SELECTIPO",
]
DEFAULT_SECTORS = [
    "CHEMICAL", "CPSEETF", "GROWWRAIL", "MAKEINDIA", "MNC", "COMMOIETF",
    "INFRA", "DEFENCE", "INTERNET", "EVIETF", "CONSUMBEES", "NIFTYBEES",
    "GOLDBEES", "BANKBEES", "HEALTHIETF", "GROWWPOWER", "ENERGY", "OILIETF",
    "GROWWRLTY", "METAL", "MASPTOP50", "MON100", "MONQ50", "HNGSNGBEES",
]
HORIZONS = {"Weekly (5d)": 5, "15 days": 15, "Monthly (21d)": 21}
OPS = {
    ">": np.greater, ">=": np.greater_equal,
    "<": np.less, "<=": np.less_equal,
}


def parse_tickers(text: str) -> list[str]:
    items = [t.strip().upper() for t in re.split(r"[,\n\s]+", text) if t.strip()]
    return list(dict.fromkeys(items))  # de-dupe, keep order


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="Fetching prices from Yahoo Finance…")
def fetch_prices(tickers: tuple, start: str, suffix: str) -> pd.DataFrame:
    syms = [t + suffix for t in tickers]
    raw = yf.download(syms, start=start, auto_adjust=True, progress=False, threads=True)
    close = raw["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(syms[0])
    close.columns = [c[: -len(suffix)] if suffix and c.endswith(suffix) else c for c in close.columns]
    return close.dropna(axis=1, how="all").sort_index()


def compute_returns(prices: pd.DataFrame, h: int, overlapping: bool) -> pd.DataFrame:
    if overlapping:
        return prices.pct_change(h, fill_method=None)
    return prices.iloc[::h].pct_change(fill_method=None)


def slice_window(df: pd.DataFrame, mode: str, n: int, since: date) -> pd.DataFrame:
    if mode == "Last N trading days":
        return df.iloc[-n:]
    return df.loc[str(since):]


def conditional_stats(R, conds, targets, mode, lag, thr, min_obs):
    masks = [OPS[op](R[t], v / 100) & R[t].notna() for t, op, v in conds]
    m = pd.concat(masks, axis=1)
    signal = m.all(axis=1) if mode == "ALL (AND)" else m.any(axis=1)

    tgt = R[targets].shift(-lag) if lag else R[targets]
    rows = []
    for t in targets:
        valid = tgt[t].notna()
        sig = signal & valid
        n = int(sig.sum())
        if n < min_obs:
            continue
        r_sig, r_all = tgt.loc[sig, t], tgt.loc[valid, t]
        p_c, p_b = (r_sig > thr).mean(), (r_all > thr).mean()
        rows.append({
            "Target": t,
            "Signals": n,
            "P(target > thr | cond) %": p_c * 100,
            "Base P %": p_b * 100,
            "Lift (pp)": (p_c - p_b) * 100,
            "Avg ret | cond %": r_sig.mean() * 100,
            "Median ret | cond %": r_sig.median() * 100,
            "Avg ret base %": r_all.mean() * 100,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("Lift (pp)", ascending=False).reset_index(drop=True)
    return out, signal


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("Controls")

    st.subheader("Timeline")
    tl_mode = st.radio(
        "Window", ["Last N trading days", "Since date"], index=1, label_visibility="collapsed"
    )
    if tl_mode == "Last N trading days":
        n_days = st.number_input("N trading days", 30, 2000, 300, step=10)
        since = date(2021, 1, 1)
    else:
        since = st.date_input("Start date", date(2021, 1, 1), min_value=date(2015, 1, 1))
        n_days = 300

    st.subheader("Returns")
    hz_label = st.selectbox("Horizon", list(HORIZONS), index=2)
    overlapping = st.toggle(
        "Overlapping windows", value=True,
        help="On: rolling returns every day (more observations, but autocorrelated). "
             "Off: independent non-overlapping blocks.",
    )

    with st.expander("Universe & data source"):
        f_text = st.text_area("Factor ETFs", ", ".join(DEFAULT_FACTORS), height=90)
        s_text = st.text_area("Sector ETFs", ", ".join(DEFAULT_SECTORS), height=160)
        suffix = st.selectbox("Exchange suffix", [".NS", ".BO", ""], index=0)
        dl_start = st.date_input("Download history from", date(2020, 6, 1))
        if st.button("Refresh data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

horizon = HORIZONS[hz_label]
factor_list = parse_tickers(f_text)
sector_list = [t for t in parse_tickers(s_text) if t not in factor_list]

# ----------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------
st.title("Factor & Sector Lab")
st.caption("Returns, relative performance and conditional probabilities across Indian factor and sector ETFs.")

all_tickers = tuple(factor_list + sector_list)
if not all_tickers:
    st.warning("Add at least one ticker in the sidebar.")
    st.stop()

try:
    prices = fetch_prices(all_tickers, str(dl_start), suffix)
except Exception as e:
    st.error(f"Could not fetch data: {e}")
    st.stop()

if prices.empty:
    st.error("No price data came back. Check the tickers and exchange suffix.")
    st.stop()

factors = prices[[c for c in factor_list if c in prices.columns]]
sectors = prices[[c for c in sector_list if c in prices.columns]]
missing = [t for t in all_tickers if t not in prices.columns]
if missing:
    st.warning(f"No data for: {', '.join(missing)} (check the ticker symbols).")

ret_full = compute_returns(prices, horizon, overlapping)
R = slice_window(ret_full, tl_mode, n_days, since)
px_win = slice_window(prices, tl_mode, n_days, since)

# ----------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------
tab_ov, tab_ex, tab_cp = st.tabs(["Overview", "Explore returns", "Conditional probability"])

# ---- Overview --------------------------------------------------------
with tab_ov:
    k1, k2, k3, k4 = st.columns(4)
    for col, label, val in [
        (k1, "Factors", len(factors.columns)),
        (k2, "Sectors", len(sectors.columns)),
        (k3, "Window start", px_win.index[0].strftime("%d %b %Y") if len(px_win) else "—"),
        (k4, "Trading days in window", len(px_win)),
    ]:
        with col, st.container(border=True):
            st.metric(label, val)

    st.write("")
    st.subheader("Latest returns snapshot")
    st.caption("Trailing returns as of the most recent close, across all three horizons.")
    snap = pd.DataFrame({
        name: prices.pct_change(h, fill_method=None).iloc[-1] * 100 for name, h in HORIZONS.items()
    })

    def heat(df, title):
        fig = px.imshow(
            df, text_auto=".1f", aspect="auto",
            color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
            zmin=-10, zmax=10,
        )
        fig.update_layout(
            title=title, coloraxis_showscale=False, margin=dict(l=0, r=0, t=40, b=0),
            height=max(160, 30 * len(df) + 90), xaxis_title=None, yaxis_title=None,
        )
        fig.update_xaxes(side="top")
        return fig

    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.plotly_chart(heat(snap.loc[factors.columns], "Factors (%)"), use_container_width=True)
    with c2:
        st.plotly_chart(heat(snap.loc[sectors.columns], "Sectors (%)"), use_container_width=True)

# ---- Explore ---------------------------------------------------------
with tab_ex:
    default_sel = list(prices.columns[:4])
    sel = st.multiselect("Tickers", list(prices.columns), default=default_sel)
    if not sel:
        st.info("Pick at least one ticker.")
    else:
        left, right = st.columns(2)
        with left:
            rebased = px_win[sel].ffill()
            rebased = rebased / rebased.bfill().iloc[0] * 100
            fig = px.line(rebased, title="Rebased performance (start = 100)")
            fig.update_layout(legend_title=None, xaxis_title=None, yaxis_title=None,
                              margin=dict(l=0, r=0, t=40, b=0), height=400)
            st.plotly_chart(fig, use_container_width=True)
        with right:
            long = R[sel].dropna(how="all").melt(var_name="Ticker", value_name="Return").dropna()
            long["Return"] *= 100
            fig = px.box(long, x="Ticker", y="Return", points=False,
                         title=f"Distribution of {hz_label} returns (%)")
            fig.add_hline(y=0, line_dash="dot", line_color="grey")
            fig.update_layout(xaxis_title=None, yaxis_title=None,
                              margin=dict(l=0, r=0, t=40, b=0), height=400)
            st.plotly_chart(fig, use_container_width=True)

    st.subheader(f"{hz_label} return summary")
    summ = R.describe().T[["count", "mean", "std", "min", "50%", "max"]]
    summ[["mean", "std", "min", "50%", "max"]] *= 100
    summ["% positive"] = (R > 0).sum() / R.notna().sum() * 100
    summ.insert(0, "Group", ["Factor" if t in factors.columns else "Sector" for t in summ.index])
    summ = summ.rename(columns={"50%": "median", "count": "obs"})
    st.dataframe(
        summ, use_container_width=True,
        column_config={
            "obs": st.column_config.NumberColumn(format="%d"),
            "mean": st.column_config.NumberColumn("mean %", format="%.2f"),
            "std": st.column_config.NumberColumn("std %", format="%.2f"),
            "min": st.column_config.NumberColumn("min %", format="%.2f"),
            "median": st.column_config.NumberColumn("median %", format="%.2f"),
            "max": st.column_config.NumberColumn("max %", format="%.2f"),
            "% positive": st.column_config.ProgressColumn("% positive", min_value=0, max_value=100, format="%.0f%%"),
        },
    )

# ---- Conditional probability ----------------------------------------
with tab_cp:
    st.subheader("When these move, what do the others do?")
    st.caption("Define conditions on any ETFs. Rows are combined with AND/OR, then each target is compared with its unconditional base rate.")

    cols_avail = list(prices.columns)
    seed = [c for c in ["MOMOMENTUM", "MOVALUE"] if c in cols_avail] or cols_avail[:2]
    cond_df = pd.DataFrame({
        "Ticker": seed, "Operator": [">"] * len(seed), "Threshold %": [5.0] * len(seed),
    })

    left, right = st.columns([1.3, 1])
    with left:
        cond_edit = st.data_editor(
            cond_df, num_rows="dynamic", use_container_width=True, key="conds",
            column_config={
                "Ticker": st.column_config.SelectboxColumn(options=cols_avail, required=True),
                "Operator": st.column_config.SelectboxColumn(options=list(OPS), required=True),
                "Threshold %": st.column_config.NumberColumn(step=0.5, format="%.1f", required=True),
            },
        )
    with right:
        mode = st.radio("Combine conditions", ["ALL (AND)", "ANY (OR)"], horizontal=True)
        tgt_group = st.radio("Targets", ["Sectors", "Factors", "All others"], horizontal=True)
        o1, o2, o3 = st.columns(3)
        thr_pct = o1.number_input("Target thr %", value=0.0, step=0.5)
        lag = o2.number_input("Lag (rows)", 0, 250, 0,
                              help="0 = same period. k = target's return k rows later (forward-looking).")
        min_obs = o3.number_input("Min signals", 1, 500, 10)

    conds = [
        (r["Ticker"], r["Operator"], float(r["Threshold %"]))
        for _, r in cond_edit.dropna().iterrows()
        if r["Ticker"] in cols_avail
    ]
    if not conds:
        st.info("Add at least one condition above.")
        st.stop()

    cond_set = {c[0] for c in conds}
    pool = {"Sectors": sectors.columns, "Factors": factors.columns, "All others": prices.columns}[tgt_group]
    targets = [t for t in pool if t not in cond_set]

    res, signal = conditional_stats(R, conds, targets, mode, int(lag), thr_pct / 100, int(min_obs))

    m1, m2, m3 = st.columns(3)
    with m1, st.container(border=True):
        st.metric("Signal periods", int(signal.sum()))
    with m2, st.container(border=True):
        st.metric("Share of window", f"{signal.mean() * 100:.1f}%")
    with m3, st.container(border=True):
        st.metric("Horizon / lag", f"{horizon}d / {int(lag)}")

    if overlapping and signal.sum() > 0:
        st.caption("⚠️ Overlapping windows are autocorrelated, so the effective sample is smaller than the signal count. "
                   "Switch off overlap in the sidebar for independent observations.")

    if res.empty:
        st.warning("No target has enough signal observations. Lower the thresholds, shorten the lag, or reduce Min signals.")
    else:
        res["Direction"] = np.where(res["Lift (pp)"] >= 0, "Above base", "Below base")
        fig = px.bar(
            res.sort_values("Lift (pp)"), x="Lift (pp)", y="Target", orientation="h",
            color="Direction", color_discrete_map={"Above base": "#2e9e5b", "Below base": "#c8473d"},
            hover_data={"Signals": True, "Direction": False},
            title="Lift in P(target return > threshold) vs base rate",
        )
        fig.update_layout(showlegend=False, yaxis_title=None, xaxis_title="percentage points",
                          margin=dict(l=0, r=0, t=40, b=0), height=max(280, 26 * len(res) + 90))
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            res.drop(columns="Direction"), use_container_width=True, hide_index=True,
            column_config={
                "Signals": st.column_config.NumberColumn(format="%d"),
                "P(target > thr | cond) %": st.column_config.ProgressColumn(
                    "P(tgt > thr | cond)", min_value=0, max_value=100, format="%.0f%%"),
                "Base P %": st.column_config.NumberColumn("Base P", format="%.0f%%"),
                "Lift (pp)": st.column_config.NumberColumn("Lift", format="%+.1f pp"),
                "Avg ret | cond %": st.column_config.NumberColumn("Avg | cond", format="%+.2f%%"),
                "Median ret | cond %": st.column_config.NumberColumn("Median | cond", format="%+.2f%%"),
                "Avg ret base %": st.column_config.NumberColumn("Avg base", format="%+.2f%%"),
            },
        )

    with st.expander("Dates when the condition fired"):
        fired = R.loc[signal, [c[0] for c in conds]] * 100
        st.dataframe(fired.round(2), use_container_width=True)
