import math
from math import erf
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# =========================================================
# Config
# =========================================================
st.set_page_config(
    page_title="Market-Linked Product Analytics Lab",
    layout="wide"
)

# =========================================================
# Quant helpers
# =========================================================

def norm_cdf(x):
    """Standard normal CDF without SciPy."""
    return 0.5 * (1.0 + erf(x / math.sqrt(2.0)))


def norm_pdf(x):
    """Standard normal PDF."""
    return (1.0 / math.sqrt(2.0 * math.pi)) * np.exp(-0.5 * x * x)


def simulate_gbm_paths(S0, mu, sigma, r, T, steps, n_paths, seed=42):
    """
    Simulate GBM equity paths.

    S0: initial price
    mu: drift (real-world)
    sigma: volatility
    r: risk-free rate
    T: horizon in years
    steps: time steps
    n_paths: number of scenarios
    """
    dt = T / steps
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(size=(steps, n_paths))

    paths = np.zeros((steps + 1, n_paths))
    paths[0, :] = S0

    for t in range(1, steps + 1):
        paths[t, :] = paths[t - 1, :] * np.exp(
            (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z[t - 1, :]
        )

    time_index = np.linspace(0, T, steps + 1)
    df = pd.DataFrame(paths, index=time_index)
    df.index.name = "Time (years)"
    return df


def bs_call_price(S, K, T, r, sigma):
    """Black-Scholes price for a European call."""
    if T <= 0:
        return max(S - K, 0.0)

    if S <= 0 or sigma <= 0:
        # Very edge cases; fallback to intrinsic
        return max(S - K, 0.0)

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)


def bs_call_delta(S, K, T, r, sigma):
    """Black-Scholes delta for a European call."""
    if T <= 0 or S <= 0 or sigma <= 0:
        return 0.0

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return norm_cdf(d1)


def monte_carlo_call_price(S0, K, r, sigma, T, steps, n_paths, seed=123):
    """Monte Carlo pricing for sanity-check vs BS."""
    dt = T / steps
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(size=(steps, n_paths))

    S = np.zeros((steps + 1, n_paths))
    S[0, :] = S0
    for t in range(1, steps + 1):
        S[t, :] = S[t - 1, :] * np.exp(
            (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z[t - 1, :]
        )

    payoffs = np.maximum(S[-1, :] - K, 0.0)
    price = math.exp(-r * T) * payoffs.mean()
    return price


def delta_hedge_pnl(S_paths, K, r, sigma, T):
    """
    Discrete-time delta hedging P&L for a short call using many simulated paths.

    S_paths: array shape (N+1, n_paths)
    Returns: array of P&L for each path (final wealth, starting from 0)
    """
    times = S_paths.shape[0] - 1
    n_paths = S_paths.shape[1]
    dt = T / times

    pnls = np.zeros(n_paths)

    for j in range(n_paths):
        S_path = S_paths[:, j]

        # t = 0: price & delta
        C0 = bs_call_price(S_path[0], K, T, r, sigma)
        delta = bs_call_delta(S_path[0], K, T, r, sigma)

        # Short call (receive C0), buy delta shares, invest remaining in cash.
        cash = C0 - delta * S_path[0]

        for t in range(1, times + 1):
            # Cash grows at risk-free rate
            cash *= math.exp(r * dt)

            # Recompute delta for remaining time
            remaining_T = max(T - t * dt, 0.0)
            new_delta = bs_call_delta(S_path[t], K, remaining_T, r, sigma)

            # Rebalance shares
            d_delta = new_delta - delta
            cash -= d_delta * S_path[t]
            delta = new_delta

        # Final cash accrual
        cash *= math.exp(r * dt)
        payoff = max(S_path[-1] - K, 0.0)  # we are short call
        portfolio_value = cash + delta * S_path[-1] - payoff

        pnls[j] = portfolio_value

    return pnls


def var_es(pnl_array, alpha=0.95):
    """
    Compute VaR / ES on loss distribution (loss = -PnL).
    """
    losses = -np.array(pnl_array)
    var = np.quantile(losses, alpha)
    es = losses[losses >= var].mean() if np.any(losses >= var) else var
    return var, es


# =========================================================
# Streamlit UI
# =========================================================

st.title("📈 Market-Linked Product Analytics & Hedging Lab")
st.caption(
    "Prototype analytics stack for variable / fixed indexed annuity-style guarantees: "
    "derivatives valuation, stochastic modeling, scenario analysis, and hedge effectiveness."
)

# ------------- Sidebar: Inputs -----------------
st.sidebar.header("Market & Product Setup")

data_source = st.sidebar.selectbox(
    "Market data source",
    ["Synthetic GBM (recommended)", "Upload equity index CSV"],
)

T_years = st.sidebar.slider("Horizon (years)", min_value=1.0, max_value=10.0, value=5.0, step=0.5)
steps = st.sidebar.slider("Time steps per horizon", 50, 260, 120, step=10)
n_paths = st.sidebar.slider("Simulation paths", 100, 2000, 500, step=100)

if data_source == "Synthetic GBM (recommended)":
    S0 = st.sidebar.number_input("Initial index level (S0)", value=100.0, step=1.0)
    mu = st.sidebar.number_input("Real-world drift (μ)", value=0.05, step=0.01, format="%.3f")
    sigma = st.sidebar.number_input("Volatility (σ)", value=0.20, step=0.01, format="%.3f")
    r = st.sidebar.number_input("Risk-free rate (r)", value=0.03, step=0.005, format="%.3f")

    base_paths = simulate_gbm_paths(
        S0=S0,
        mu=mu,
        sigma=sigma,
        r=r,
        T=T_years,
        steps=steps,
        n_paths=n_paths,
    )

else:
    uploaded = st.sidebar.file_uploader("Upload CSV with Date & Close columns", type=["csv"])
    S0 = None
    mu = st.sidebar.number_input("Assumed drift (μ)", value=0.05, step=0.01, format="%.3f")
    sigma = st.sidebar.number_input("Assumed volatility (σ)", value=0.20, step=0.01, format="%.3f")
    r = st.sidebar.number_input("Risk-free rate (r)", value=0.03, step=0.005, format="%.3f")

    if uploaded is not None:
        hist = pd.read_csv(uploaded)
        # Try to infer price column
        price_col = None
        for c in hist.columns:
            if c.lower() in ["close", "adj close", "price"]:
                price_col = c
                break
        if price_col is None:
            st.sidebar.error("Could not find 'Close'/'Adj Close'/'Price' column in CSV.")
            st.stop()

        hist = hist.dropna(subset=[price_col])
        hist_prices = hist[price_col].values
        S0 = float(hist_prices[-1])
        # Simple realized vol estimate
        log_ret = np.diff(np.log(hist_prices))
        realized_vol = np.std(log_ret) * math.sqrt(252)
        st.sidebar.write(f"Implied annualized vol from history: ~{realized_vol:.2%}")

        base_paths = simulate_gbm_paths(
            S0=S0,
            mu=mu,
            sigma=sigma,
            r=r,
            T=T_years,
            steps=steps,
            n_paths=n_paths,
        )
    else:
        st.warning("Upload a CSV to proceed or switch to Synthetic GBM.")
        st.stop()

# Product / option proxy
st.sidebar.header("Market-Linked Payoff Proxy")
K = st.sidebar.number_input("Option strike (K)", value=float(S0), step=1.0)
notional = st.sidebar.number_input("Notional (units of guarantee)", value=1_000_000.0, step=100_000.0)

hedge_freq = st.sidebar.selectbox(
    "Hedging frequency (for intuition; tied to time steps)",
    ["Re-hedge every step (max)", "Re-hedge every 2 steps", "Re-hedge every 5 steps"],
)

if hedge_freq == "Re-hedge every step (max)":
    hedge_step = 1
elif hedge_freq == "Re-hedge every 2 steps":
    hedge_step = 2
else:
    hedge_step = 5  # For now unused in function; included for extension

# =========================================================
# Section 1: Simulated Market Paths
# =========================================================
st.subheader("1️⃣ Market Scenarios")

with st.expander("View simulated market paths", expanded=True):
    sample_paths = base_paths.iloc[:, : min(20, n_paths)]

    fig_paths = go.Figure()
    for col in sample_paths.columns:
        fig_paths.add_trace(
            go.Scatter(
                x=sample_paths.index,
                y=sample_paths[col],
                mode="lines",
                line=dict(width=1),
                showlegend=False,
            )
        )
    fig_paths.update_layout(
        xaxis_title="Time (years)",
        yaxis_title="Index level",
        title="Sample simulated equity index paths",
        height=400,
    )
    st.plotly_chart(fig_paths, use_container_width=True)

    st.markdown(
        "- **Purpose**: These paths represent possible equity index trajectories driving the market-linked annuity guarantees.\n"
        "- Under the hood, we use **GBM stochastic modeling** to simulate forward scenarios."
    )

# =========================================================
# Section 2: Derivatives Valuation (Base)
# =========================================================
st.subheader("2️⃣ Derivatives Valuation & Greeks (Base Scenario)")

S0_used = float(base_paths.iloc[0, 0])
T_used = T_years
sigma_used = sigma
r_used = r

bs_price = bs_call_price(S0_used, K, T_used, r_used, sigma_used)
mc_price = monte_carlo_call_price(S0_used, K, r_used, sigma_used, T_used, steps, n_paths)

# Simple Greeks (delta, gamma, vega) via BS
if T_used > 0 and sigma_used > 0 and S0_used > 0:
    d1 = (math.log(S0_used / K) + (r_used + 0.5 * sigma_used ** 2) * T_used) / (sigma_used * math.sqrt(T_used))
    d2 = d1 - sigma_used * math.sqrt(T_used)
    delta = norm_cdf(d1)
    gamma = norm_pdf(d1) / (S0_used * sigma_used * math.sqrt(T_used))
    vega = S0_used * norm_pdf(d1) * math.sqrt(T_used)
    theta = -(
        S0_used * norm_pdf(d1) * sigma_used / (2 * math.sqrt(T_used))
    ) - r_used * K * math.exp(-r_used * T_used) * norm_cdf(d2)
else:
    delta = gamma = vega = theta = 0.0

col1, col2, col3, col4 = st.columns(4)
col1.metric("BS Call Price", f"${bs_price:,.2f}")
col2.metric("MC Call Price", f"${mc_price:,.2f}")
col3.metric("Delta (∂C/∂S)", f"{delta:.3f}")
col4.metric("Gamma", f"{gamma:.6f}")

col5, col6 = st.columns(2)
col5.metric("Vega", f"{vega:,.2f}")
col6.metric("Theta (per year)", f"{theta:,.2f}")

st.markdown(
    f"- Interpreting above as a **guarantee unit price**, the portfolio PV is roughly "
    f"**${bs_price * (notional / 100.0):,.0f}** if each 100 notional embeds one option.\n"
    "- The **Greeks** (delta, gamma, vega, theta) feed into hedge ratios, stress testing and capital frameworks."
)

# =========================================================
# Section 3: Delta-Hedging Backtest (Base)
# =========================================================
st.subheader("3️⃣ Delta-Hedging Backtest & Hedge Effectiveness")

S_array = base_paths.values  # (time, paths)
pnl_hedged = delta_hedge_pnl(S_array, K, r_used, sigma_used, T_used)

# Unhedged "short call" PnL: we just short the option at BS price, buy-and-hold
terminal_prices = S_array[-1, :]
unhedged_payoffs = bs_price - np.maximum(terminal_prices - K, 0.0)
# This is PnL of a static short vs final payoff
pnl_unhedged = unhedged_payoffs

var_hedged, es_hedged = var_es(pnl_hedged, alpha=0.95)
var_unhedged, es_unhedged = var_es(pnl_unhedged, alpha=0.95)

var_reduction = 1.0 - (var_hedged / var_unhedged) if var_unhedged > 0 else np.nan

c1, c2, c3 = st.columns(3)
c1.metric("95% VaR (unhedged, loss)", f"${var_unhedged:,.2f}")
c2.metric("95% VaR (delta-hedged, loss)", f"${var_hedged:,.2f}")
c3.metric("VaR reduction", f"{var_reduction*100:,.1f}%" if not np.isnan(var_reduction) else "N/A")

with st.expander("View PnL distributions"):
    fig_pnl = go.Figure()
    fig_pnl.add_trace(
        go.Histogram(
            x=pnl_unhedged,
            name="Unhedged short call",
            opacity=0.6,
            nbinsx=40,
        )
    )
    fig_pnl.add_trace(
        go.Histogram(
            x=pnl_hedged,
            name="Delta-hedged short call",
            opacity=0.6,
            nbinsx=40,
        )
    )
    fig_pnl.update_layout(
        barmode="overlay",
        title="Distribution of P&L (per unit) – Unhedged vs Delta-Hedged",
        xaxis_title="P&L at horizon",
        yaxis_title="Frequency",
        height=400,
    )
    st.plotly_chart(fig_pnl, use_container_width=True)

    st.markdown(
        "- This back-test mimics a **hedging program** for market-linked annuity guarantees.\n"
        "- The tighter P&L distribution for the hedged strategy illustrates **hedge effectiveness**."
    )

# =========================================================
# Section 4: Scenario & Stress Testing
# =========================================================
st.subheader("4️⃣ Scenario & Stress Testing")

st.markdown(
    "We run a small scenario suite on the same guarantee:\n"
    "- **Base**: current parameters\n"
    "- **Equity -20% shock** (instantaneous drop in S0)\n"
    "- **Volatility +10% pts**\n"
    "- **Rates +100 bps**\n"
)

scenarios = []

# Helper to run scenario: price + hedging VaR/ES
def run_scenario(name, S0_scn, r_scn, sigma_scn):
    paths_scn = simulate_gbm_paths(
        S0=S0_scn, mu=mu, sigma=sigma_scn, r=r_scn,
        T=T_years, steps=steps, n_paths=n_paths, seed=999
    )
    S_arr_scn = paths_scn.values
    price_scn = bs_call_price(S0_scn, K, T_years, r_scn, sigma_scn)
    pnl_hedged_scn = delta_hedge_pnl(S_arr_scn, K, r_scn, sigma_scn, T_years)
    var_scn, es_scn = var_es(pnl_hedged_scn, alpha=0.95)
    return {
        "Scenario": name,
        "S0": S0_scn,
        "Rate r": r_scn,
        "Vol σ": sigma_scn,
        "BS Price": price_scn,
        "95% VaR (loss)": var_scn,
        "95% ES (loss)": es_scn,
    }

scenarios.append(run_scenario("Base", S0_used, r_used, sigma_used))
scenarios.append(run_scenario("Equity -20%", S0_used * 0.8, r_used, sigma_used))
scenarios.append(run_scenario("Vol +10% pts", S0_used, r_used, sigma_used + 0.10))
scenarios.append(run_scenario("Rate +100 bps", S0_used, r_used + 0.01, sigma_used))

df_scen = pd.DataFrame(scenarios)
st.dataframe(df_scen.style.format({
    "S0": "{:,.2f}",
    "Rate r": "{:.3f}",
    "Vol σ": "{:.3f}",
    "BS Price": "{:,.2f}",
    "95% VaR (loss)": "{:,.2f}",
    "95% ES (loss)": "{:,.2f}",
}))

st.markdown(
    "- This table is a simple **ALM-style stress test**: how does the liability PV and hedged loss profile move under market shocks?\n"
    "- In a production environment you’d expand this to **full balance-sheet analytics**, feeding into capital frameworks (GAAP, statutory, Bermuda/EBS)."
)

# =========================================================
# Section 5: Management Summary
# =========================================================
st.subheader("5️⃣ Management-Style Summary")

base_row = df_scen[df_scen["Scenario"] == "Base"].iloc[0]
equity_row = df_scen[df_scen["Scenario"] == "Equity -20%"].iloc[0]
vol_row = df_scen[df_scen["Scenario"] == "Vol +10% pts"].iloc[0]
rate_row = df_scen[df_scen["Scenario"] == "Rate +100 bps"].iloc[0]

st.markdown(
    f"""
**Key takeaways (per-unit guarantee):**

- Base PV of market-linked guarantee: **${base_row['BS Price']:,.2f}**  
- Under a **-20% equity shock**, liability value moves to **${equity_row['BS Price']:,.2f}** and 95% hedged loss (VaR) is **${equity_row['95% VaR (loss)']:,.2f}**.  
- A **+10% pt volatility shock** materially increases both liability PV (**${vol_row['BS Price']:,.2f}**) and tail risk (ES: **${vol_row['95% ES (loss)']:,.2f}**).  
- A **+100 bps rate shock** lowers the option PV to **${rate_row['BS Price']:,.2f}**, reflecting discounting impact and forward curve shift.

This prototype shows how we can:

1. Connect **capital markets** (equity, vol, rates) to **insurance liabilities** (market-linked guarantees).  
2. Use **stochastic modeling and derivatives pricing** to quantify exposures.  
3. Evaluate **hedging effectiveness** and **tail risk** via VaR / ES on hedged P&L.  
4. Present outputs in a way that can be consumed by **Risk, ALM, and senior management**.
"""
)
