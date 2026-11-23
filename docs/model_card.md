# Model card: CV shortlisting assistant (store roles)

*Generated 23 November 2026 by `src/build_report.py`. Fictional use case on synthetic data.*

## Intended use
- **Purpose:** rank applicants for store-assistant roles so recruiters review the most suitable first.
- **Users:** trained recruiters. **The model recommends; a human decides.** No automatic rejections.
- **Out of scope:** management roles, promotion, termination, or any use without human review.

## Regulatory classification
- **EU AI Act:** high-risk (Annex III, employment: recruitment and CV screening). It profiles natural persons, so the Art. 6(3) derogation does not apply.
- **UK AI principles in focus:** Safety, security & robustness, Appropriate transparency & explainability, Fairness, Accountability & governance, Contestability & redress.

## Data
- 8,000 synthetic applications. The features are experience, skills, customer-service assessment, degree, career-gap months and distance.
- The protected attribute (gender) is **excluded from features** but used for auditing.
- **Known issue:** the historic hiring labels are biased (disparate impact 0.50), and career-gap months is a proxy for gender (correlation 0.42).

## Model
Logistic regression (standardised features), trained with **reweighing**. Post-processed with **group-specific thresholds** to equalise selection rates.

## Performance & fairness (hold-out, against job-relevant suitability)
| Metric | Baseline | Released model |
|---|---:|---:|
| Accuracy | 0.808 | 0.817 |
| Disparate impact ratio (≥ 0.80) | 0.82 | 1.00 |
| Demographic parity difference | -0.084 | -0.000 |
| Equal opportunity difference (TPR gap) | -0.193 | -0.093 |

## Limitations & risks
- Group-specific thresholds use the protected attribute at decision time. Their lawfulness depends on jurisdiction (e.g. UK Equality Act positive-action rules), so **legal review is required before use**.
- Only binary gender is audited. Audit other protected characteristics (age, disability, ethnicity) before deployment.
- Synthetic data: real performance must be re-validated on the client's data.

## Human oversight, monitoring & redress
- Recruiters see the top factors for each score and can override them. Overrides are logged.
- Monthly fairness and drift monitoring, with an alert if the disparate impact ratio falls below 0.85 or the TPR gap exceeds 5 pts.
- Candidates can request human review of any decision (contestability & redress).
- Owner: Head of Talent Acquisition. Model risk sign-off: AI governance board.
