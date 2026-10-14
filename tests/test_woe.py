import numpy as np
import pandas as pd

from woe import WoEEncoder, woe_table


def test_safer_bin_gets_positive_woe():
    x = pd.Series(["safe"] * 100 + ["risky"] * 100)
    y = pd.Series([0] * 90 + [1] * 10 + [0] * 50 + [1] * 50)
    tab = woe_table(x, y).set_index("bin")
    assert tab.loc["safe", "woe"] > 0 > tab.loc["risky", "woe"]


def test_pure_noise_has_negligible_information_value():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"noise": rng.normal(size=20_000)})
    y = pd.Series(rng.binomial(1, 0.3, 20_000))
    iv = WoEEncoder().fit(X, y).information_values(X, y)
    assert iv["noise"] < 0.02


def test_unseen_values_are_handled():
    X_train = pd.DataFrame({"cat": ["a", "b"] * 50, "num": np.arange(100.0)})
    y = pd.Series([0, 1] * 50)
    enc = WoEEncoder().fit(X_train, y)
    X_new = pd.DataFrame({"cat": ["never_seen"], "num": [1e6]})   # new category, out-of-range number
    out = enc.transform(X_new)
    assert out.loc[0, "cat"] == 0.0
    assert np.isfinite(out.loc[0, "num"])
