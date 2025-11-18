"""
Synthetic VA Block Generator
Generates a synthetic policy dataset for market-linked insurance analytics.
No real insurer or client data is used.
"""

import os
import numpy as np
import pandas as pd

N_POLICIES = 5000
RANDOM_SEED = 2025

def generate_synthetic_va_block(n_policies: int = N_POLICIES, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    policy_id = np.arange(1, n_policies + 1)

    issue_age = rng.integers(45, 75, size=n_policies)
    duration_years = rng.integers(1, 15, size=n_policies)

    base = rng.lognormal(mean=11.0, sigma=0.5, size=n_policies)
    account_value = np.clip(base, 50_000, 2_000_000)

    guarantee_factor = rng.normal(loc=1.05, scale=0.15, size=n_policies)
    guarantee_factor = np.clip(guarantee_factor, 0.8, 1.5)
    guarantee_base = account_value * guarantee_factor

    product_types = rng.choice(
        ["VA_GMAB", "VA_GMWB", "FIA_CAPPED", "FIA_UNCAPPED"],
        size=n_policies,
        p=[0.3, 0.3, 0.2, 0.2]
    )

    equity_allocation = rng.uniform(0.2, 1.0, size=n_policies)

    guarantee_moneyness = account_value / guarantee_base

    df = pd.DataFrame({
        "policy_id": policy_id,
        "issue_age": issue_age,
        "duration_years": duration_years,
        "account_value": account_value.round(2),
        "guarantee_base": guarantee_base.round(2),
        "product_type": product_types,
        "equity_allocation": equity_allocation.round(3),
        "guarantee_moneyness": guarantee_moneyness.round(3),
    })

    return df

if __name__ == "__main__":
    df_block = generate_synthetic_va_block()
    os.makedirs("data", exist_ok=True)
    out_path = os.path.join("data", "synthetic_va_block.csv")
    df_block.to_csv(out_path, index=False)
    print(f"Saved synthetic VA block to: {out_path}")
