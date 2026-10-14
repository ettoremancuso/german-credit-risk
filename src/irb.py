"""
IRB capital for retail exposures and model-validation metrics.

Capital follows the Vasicek one-factor model as written in CRR Art. 154
("other retail" exposures):

    R = 0.03 * w + 0.16 * (1 - w),   w = (1 - e^(-35 PD)) / (1 - e^(-35))
    K = LGD * N[ (N^-1(PD) + sqrt(R) * N^-1(0.999)) / sqrt(1 - R) ] - PD * LGD
    RW = 12.5 * K

Retail has no maturity adjustment, and the old 1.06 scaling factor is dropped
under Basel III final / CRR3.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, roc_curve

PD_FLOOR_RETAIL = 0.0005   # CRR3 input floor: 5 bp
CONFIDENCE = 0.999         # regulatory 1-in-1,000-year loss


def other_retail_correlation(pd_: np.ndarray) -> np.ndarray:
    """Asset correlation: 16% for the safest borrowers, falling to 3% as PD rises."""
    w = (1 - np.exp(-35 * pd_)) / (1 - np.exp(-35))
    return 0.03 * w + 0.16 * (1 - w)


def capital_k(pd_, lgd: float) -> np.ndarray:
    """Unexpected-loss capital per unit of EAD (downturn loss minus expected loss)."""
    pd_ = np.maximum(np.asarray(pd_, dtype=float), PD_FLOOR_RETAIL)
    rho = other_retail_correlation(pd_)
    stressed_pd = norm.cdf((norm.ppf(pd_) + np.sqrt(rho) * norm.ppf(CONFIDENCE))
                           / np.sqrt(1 - rho))
    return lgd * stressed_pd - lgd * pd_


def risk_weight(pd_, lgd: float) -> np.ndarray:
    """Risk weight as a share of EAD, e.g. 0.75 = 75%."""
    return 12.5 * capital_k(pd_, lgd)


def recalibrate(pd_, target_rate: float) -> np.ndarray:
    """Move the average PD of a sample to `target_rate` (calibration to central tendency).

    The German Credit sample over-represents defaults (30% bad), so raw PDs are
    far too high for capital purposes. Multiplying every borrower's odds by the
    same constant c is an intercept shift in the logit: ranking and AUC stay
    the same, only the level moves. c is solved numerically with Brent's method.
    """
    pd_ = np.clip(np.asarray(pd_, dtype=float), 1e-6, 1 - 1e-6)
    odds = pd_ / (1 - pd_)

    def gap(c: float) -> float:
        return np.mean(odds * c / (1 + odds * c)) - target_rate

    c = brentq(gap, 1e-8, 1e4)
    return odds * c / (1 + odds * c)


def ks_statistic(y_true, score) -> float:
    """Kolmogorov-Smirnov: max distance between the score distributions of bads and goods."""
    fpr, tpr, _ = roc_curve(y_true, score)
    return float(np.max(tpr - fpr))


def discrimination(y_true, score) -> dict[str, float]:
    """AUC, Gini (= 2*AUC - 1, the usual banking convention) and KS."""
    auc = roc_auc_score(y_true, score)
    return {"AUC": auc, "Gini": 2 * auc - 1, "KS": ks_statistic(y_true, score)}
