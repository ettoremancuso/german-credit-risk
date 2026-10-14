# Credit Risk in Germany: from system-wide signals to a borrower-level PD model

Independent research project by **Ettore Mancuso** (BSc Economics and Finance, Università degli Studi di Palermo).

Paper: [`paper/german_credit_risk.pdf`](paper/german_credit_risk.pdf)
Notebook: [`notebooks/german_credit_risk.ipynb`](notebooks/german_credit_risk.ipynb)
Technical companion (how the code works, step by step): [`docs/technical_companion.pdf`](docs/technical_companion.pdf)

## What this project does

1. **System level.** Tracks the turn in Germany's credit cycle with official data: corporate insolvencies (Destatis), NPL ratios and Stage 2 shares against the EU (EBA), corporate NPLs (BaFin), profitability and capital (ECB), and the Bundesbank's Financial Stability Review.
2. **Borrower level.** Estimates Probability of Default (PD) models on the German Credit data (1,000 consumer loans):
   - a Weight-of-Evidence logistic **scorecard**,
   - a **random forest** and **gradient boosting**,
   - validation with cross-validated AUC, Gini, KS and Brier score.
3. **Regulation.** Builds a seven-grade rating master scale, recalibrates PDs to a realistic portfolio level and computes **IRB risk weights** with the CRR formula for retail exposures, comparing them with the standardised approach and the **CRR3 output floor**.

## Main results

| | |
|---|---|
| Corporate insolvencies 2025 | 24,064 (+72% vs the 2021 trough) |
| German NPL stock 2021 → 2025 | EUR 29.7bn → 48.6bn |
| Stage 2 share, Germany vs EU | 15.3% vs 8.9% |
| Scorecard test AUC / Gini | 0.801 / 0.601 (random forest 0.794, gradient boosting 0.796) |
| IRB RWA as % of standardised (3% PD, 45% LGD) | 71.4%: the output floor binds once it reaches 72.5% (2030) |

## Repository structure

```
german-credit-risk/
├── data/
│   ├── raw/germancredit.csv            # German Credit data (UCI Statlog), 1,000 loans
│   └── macro/                          # hand-collected system data, one source per row
│       ├── insolvencies_de.csv
│       ├── npl_indicators.csv
│       └── regulation_parameters.csv
├── notebooks/
│   └── german_credit_risk.ipynb        # full analysis, executed
├── src/
│   ├── woe.py                          # WoE binning and Information Value
│   └── irb.py                          # CRR retail risk-weight function, PD recalibration, metrics
├── sql/segments.sql                    # default rates by segment (run on SQLite from the notebook)
├── R/pd_logit.R                        # cross-check: logistic regression in base R
├── tests/                              # pytest unit tests for src/ (run by GitHub Actions)
├── outputs/                            # CSV exports used for the Power BI report
├── figures/                            # PDF and PNG charts
├── docs/                               # technical companion: code walkthrough (LaTeX + PDF)
└── paper/                              # LaTeX source and PDF
```

## How to run

```bash
pip install -r requirements.txt pytest
pytest -q                     # unit tests for the WoE encoder and the capital formula
jupyter nbconvert --to notebook --execute --inplace notebooks/german_credit_risk.ipynb
Rscript R/pd_logit.R          # optional, after the notebook (uses outputs/test_index.csv)
```

The paper compiles with XeLaTeX (font: TeX Gyre Termes): `cd paper && xelatex german_credit_risk.tex` (twice).

## Data notes

* **German Credit data:** Hofmann, H. (1994), *Statlog (German Credit Data)*, UCI Machine Learning Repository. The copy in `data/raw/` comes from the `scorecardpy` package mirror. Grömping (2019) documents coding errors in this version; category labels are therefore not interpreted causally. Defaults are over-sampled (30% bad), so PDs are recalibrated before computing capital.
* **Excluded attributes:** `personal_status_and_sex` and `foreign_worker` are not used, for fairness reasons (creditworthiness scoring is a high-risk use under the EU AI Act).
* **System data:** see the `source` column in each file in `data/macro/`.

## Tools and AI use

Python (pandas, scikit-learn, statsmodels, SciPy, matplotlib), SQL (SQLite), R, Jupyter, LaTeX, Git.
I used AI coding assistants while writing the code. The research question, modelling choices, validation and interpretation of the results are my own, and I can walk through every part of the code.

## License

All rights reserved, © 2026 Ettore Mancuso. You are welcome to read the code and reproduce the results; reuse or redistribution requires my written permission. See [LICENSE](LICENSE). Data remain subject to the terms of their original providers.
