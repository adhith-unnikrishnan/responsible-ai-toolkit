# Responsible AI Toolkit: risk tiering, fairness audit, mitigation and model cards

**Live report → https://adhith-unnikrishnan.github.io/responsible-ai-toolkit/**

How to **embed Responsible AI into an AI strategy** rather than bolting it on at the end. Three tools, one worked
example (the same fictional UK retailer as the [AI value case](https://github.com/adhith-unnikrishnan/ai-value-case-roadmap)):

### 1. Use-case risk screening (`src/rai/risk_screen.py`)
Triage each proposed AI use case against the **EU AI Act** (Art. 5 prohibited practices, Annex III high-risk areas,
the Art. 6(3) derogation, Art. 50 transparency duties) and map it to the **UK's five AI regulatory principles**. The output
is a tier, the reasons, and the controls required before build. In the example, CV screening and workforce scheduling are
high-risk, the GenAI chatbot needs transparency, and a vendor's staff-emotion-monitoring proposal is **prohibited**.

### 2. Fairness audit (`src/rai/fairness.py`)
A CV-shortlisting model trained on biased historic hiring decisions (synthetic data):
- Gender is excluded from the features ("fairness through unawareness"), but **career-gap months is a proxy** (r = 0.42).
- The baseline **passes the four-fifths rule (DI 0.82)** but suitable women are **19 percentage points less likely to be
  shortlisted** (equal-opportunity gap). This is why one metric is never enough.
- Metrics by group: selection rate, TPR, FPR, precision, disparate impact, demographic parity and equal opportunity.
  Permutation importance exposes the proxy.

### 3. Mitigation with visible trade-offs
| Approach | Accuracy | Disparate impact | Equal-opportunity gap |
|---|---:|---:|---:|
| A. Baseline (unaware) | 0.808 | 0.82 | −0.193 |
| B. Reweighing (Kamiran & Calders) | 0.811 | 0.85 | −0.169 |
| C. Reweighing + group thresholds | **0.817** | **1.00** | **−0.093** |

Option C uses the protected attribute at decision time, so the generated **model card** flags it for legal review
(e.g. UK Equality Act positive-action rules) instead of hiding the trade-off.

### 4. Generated documentation
[`docs/model_card.md`](docs/model_card.md) (intended use, EU AI Act classification, data issues, fairness results,
limitations, human oversight, monitoring thresholds and redress) is regenerated on every run, so it never drifts from
the model. The report also includes the high-risk governance checklist.

## Run it
```bash
pip install -r requirements.txt
python src/build_report.py   # docs/index.html + docs/model_card.md
python -m pytest -q          # 7 tests: tier logic, Art. 6(3), metric maths, reweighing, mitigation
```

## Part of a Data & AI strategy portfolio
1. [AI Maturity Assessment](https://github.com/adhith-unnikrishnan/ai-maturity-assessment): where is the organisation today?
2. [AI Value Case & Roadmap](https://github.com/adhith-unnikrishnan/ai-value-case-roadmap): what to invest in, and in what order
3. **Responsible AI Toolkit** (this repo): how to deploy it safely

**Author:** Adhith Unnikrishnan, MSc Artificial Intelligence (Brunel University London).
Synthetic data; risk screening is indicative and not legal advice.
