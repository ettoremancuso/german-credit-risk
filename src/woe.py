"""
Weight of Evidence (WoE) encoding and Information Value (IV).

Standard credit-scorecard toolkit (see Siddiqi, 2006, ch. 6). Convention used
throughout the project: target = 1 means default ("bad"), and

    WoE_i = ln( %good_i / %bad_i )
    IV    = sum_i ( %good_i - %bad_i ) * WoE_i

so a positive WoE marks a bin that is safer than the portfolio average and,
in a logit on WoE variables, every coefficient should come out negative.

Rule of thumb for IV: < 0.02 useless, 0.02-0.1 weak, 0.1-0.3 medium, > 0.3 strong.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Laplace-style smoothing: avoids ln(0) when a bin has no goods or no bads.
SMOOTHING = 0.5


def _is_continuous(x: pd.Series, n_bins: int) -> bool:
    return pd.api.types.is_numeric_dtype(x) and x.nunique() > n_bins


def woe_table(x: pd.Series, y: pd.Series, n_bins: int = 5) -> pd.DataFrame:
    """Bin-level WoE/IV table for a single risk driver.

    Continuous drivers (amount, duration, age...) are cut into quantile bins;
    categorical ones keep their own categories.
    """
    if _is_continuous(x, n_bins):
        x = pd.qcut(x, q=n_bins, duplicates="drop").astype(str)

    tab = (pd.DataFrame({"bin": x.astype(str), "bad": y.to_numpy()})
             .groupby("bin", observed=True)["bad"]
             .agg(n="count", bad="sum"))
    tab["good"] = tab["n"] - tab["bad"]

    k = len(tab)
    tab["pct_good"] = (tab["good"] + SMOOTHING) / (tab["good"].sum() + SMOOTHING * k)
    tab["pct_bad"] = (tab["bad"] + SMOOTHING) / (tab["bad"].sum() + SMOOTHING * k)
    tab["bad_rate"] = tab["bad"] / tab["n"]
    tab["woe"] = np.log(tab["pct_good"] / tab["pct_bad"])
    tab["iv"] = (tab["pct_good"] - tab["pct_bad"]) * tab["woe"]
    return tab.reset_index()


class WoEEncoder:
    """Learns WoE mappings on a training sample and applies them out of sample.

    Same fit/transform interface as scikit-learn transformers, so it can be
    refitted inside every cross-validation fold (otherwise the validation fold
    leaks into the encoding and the scorecard looks better than it is).
    """

    def __init__(self, n_bins: int = 5):
        self.n_bins = n_bins
        self.mappings_: dict[str, dict[str, float]] = {}
        self.edges_: dict[str, np.ndarray] = {}

    def _to_bins(self, col: str, x: pd.Series) -> pd.Series:
        if col in self.edges_:
            return pd.cut(x, bins=self.edges_[col]).astype(str)
        return x.astype(str)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WoEEncoder":
        for col in X.columns:
            x = X[col]
            if _is_continuous(x, self.n_bins):
                _, edges = pd.qcut(x, q=self.n_bins, retbins=True, duplicates="drop")
                # open-ended outer bins: a test value outside the training range
                # still lands in a bin instead of becoming NaN
                edges[0], edges[-1] = -np.inf, np.inf
                self.edges_[col] = edges
            tab = woe_table(self._to_bins(col, x), y, self.n_bins)
            self.mappings_[col] = dict(zip(tab["bin"], tab["woe"]))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for col, mapping in self.mappings_.items():
            if col not in X.columns:  # lets us encode a subset of the drivers
                continue
            # a category never seen in training gets WoE = 0 (average risk)
            out[col] = self._to_bins(col, X[col]).map(mapping).fillna(0.0)
        return out

    def information_values(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        iv = {col: woe_table(self._to_bins(col, X[col]), y, self.n_bins)["iv"].sum()
              for col in X.columns}
        return pd.Series(iv, name="IV").sort_values(ascending=False)
