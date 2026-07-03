# 📈 Market-Linked Products Analytics Lab

> A Python + Streamlit lab for the analytics behind **market-sensitive insurance products** — variable annuities (VA) and fixed indexed annuities (FIA) — covering scenario generation, derivatives valuation, hedging simulation, and stress testing on fully synthetic data.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python) ![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit) ![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Overview

Insurers hedging VA/FIA blocks need to value embedded equity guarantees, measure sensitivity, and test hedge programs under stress. This lab implements that workflow end to end:

- **Equity scenario generation** — Geometric Brownian Motion paths (real-world and risk-neutral)
- **Valuation** — Black–Scholes closed form and Monte Carlo pricing (no SciPy dependency; custom normal CDF/PDF)
- **Greeks** — Delta, Gamma, Vega, Theta
- **Delta-hedging backtests** — hedge P&L distributions across simulated paths
- **Hedge effectiveness** — VaR and Expected Shortfall of hedged vs. unhedged P&L
- **Stress testing** — equity, volatility, and rate shock scenarios
- **Synthetic policy block** — optional VA block generator for portfolio-level views

Everything runs on **synthetic, non-proprietary data** — no real insurer, reinsurer, employer, or client data is used.

## 🗂️ Project Structure

```
market-linked-products-analytics-lab/
├── Market_Linked_Analytics.py            # Streamlit app — valuation, hedging, stress testing
├── requirements.txt                      # Dependencies
├── synthetic_data/
│   ├── generate_synthetic_block.py       # Synthetic VA policy block generator
│   └── synthetic_va_block.csv            # Sample generated block
└── data/README_DATA.md                   # Data documentation
```

## 🚀 Getting Started

```bash
git clone https://github.com/nmadagi/market-linked-products-analytics-lab.git
cd market-linked-products-analytics-lab
pip install -r requirements.txt
streamlit run Market_Linked_Analytics.py
```

## 📦 Tech Stack

| Layer | Technology |
|---|---|
| Quant engine | NumPy, custom Black–Scholes / Monte Carlo |
| Data | Pandas, synthetic generators |
| UI / viz | Streamlit, Plotly |

## ⚠️ Disclaimer

All data is fully synthetic. This lab is for educational and demonstration purposes only.

## 👤 Author

**Nitin Madagi** | [GitHub](https://github.com/nmadagi) | [Portfolio](https://nmadagi.github.io/portfolio)

## 📄 License

This project is licensed under the [MIT License](LICENSE).
