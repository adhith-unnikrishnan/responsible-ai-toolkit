"""Fairness audit + mitigation for a binary decision model (CV shortlisting example).

The synthetic data reproduces a common real-world failure: historic hiring decisions were biased, and a
"neutral" feature (career-gap months) acts as a proxy for the protected attribute. Removing the protected
attribute from the features is therefore NOT enough, and the audit shows that.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["years_experience", "skills_score", "customer_service_score", "has_degree", "career_gap_months",
            "distance_km"]


def make_data(n: int = 8000, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    gender = rng.choice(["Female", "Male"], n)
    f = gender == "Female"
    df = pd.DataFrame({
        "gender": gender,
        "years_experience": rng.gamma(2.2, 2.0, n).round(1),
        "skills_score": rng.normal(60, 12, n).clip(0, 100).round(),
        "customer_service_score": rng.normal(65, 10, n).clip(0, 100).round(),
        "has_degree": rng.binomial(1, 0.35, n),
        # proxy: career gaps are more common for one group (e.g. caregiving)
        "career_gap_months": np.where(f, rng.exponential(10, n), rng.exponential(3, n)).round(),
        "distance_km": rng.gamma(2, 4, n).round(1),
    })
    merit = (0.25 * df.years_experience + 0.06 * df.skills_score + 0.05 * df.customer_service_score
             + 0.4 * df.has_degree - 0.05 * df.distance_km)
    # historic decisions penalised career gaps and, directly, the female group
    logit = merit - 7.2 - 0.06 * df.career_gap_months - 0.7 * f + rng.normal(0, 0.6, n)
    df["true_suitable"] = (merit - 7.65 + rng.normal(0, 0.6, n) > 0).astype(int)   # job-relevant ground truth
    df["hired_historic"] = (logit > 0).astype(int)                                     # biased label
    return df


@dataclass
class GroupMetrics:
    table: pd.DataFrame
    demographic_parity_diff: float
    disparate_impact_ratio: float
    equal_opportunity_diff: float
    accuracy: float


def audit(y_true, y_pred, group, privileged: str = "Male") -> GroupMetrics:
    d = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_pred), "g": np.asarray(group)})
    rows = {}
    for g, x in d.groupby("g"):
        pos = x[x.y == 1]
        neg = x[x.y == 0]
        rows[g] = {"n": len(x), "selection_rate": x.p.mean(),
                   "true_positive_rate": pos.p.mean() if len(pos) else np.nan,
                   "false_positive_rate": neg.p.mean() if len(neg) else np.nan,
                   "precision": x[x.p == 1].y.mean() if x.p.sum() else np.nan}
    t = pd.DataFrame(rows).T
    unpriv = [g for g in t.index if g != privileged][0]
    sr, tpr = t["selection_rate"], t["true_positive_rate"]
    return GroupMetrics(t, float(sr[unpriv] - sr[privileged]), float(sr[unpriv] / sr[privileged]),
                        float(tpr[unpriv] - tpr[privileged]), float((d.y == d.p).mean()))


def reweighing(y, group) -> np.ndarray:
    """Kamiran & Calders reweighing: make label and group statistically independent in training."""
    d = pd.DataFrame({"y": np.asarray(y), "g": np.asarray(group)})
    p_g = d.g.value_counts(normalize=True)
    p_y = d.y.value_counts(normalize=True)
    p_gy = d.groupby(["g", "y"]).size() / len(d)
    return d.apply(lambda r: p_g[r.g] * p_y[r.y] / p_gy[(r.g, r.y)], axis=1).values


def group_thresholds(scores, group, target_rate: float) -> dict:
    """Post-processing: per-group thresholds giving each group the same selection rate."""
    s = pd.Series(np.asarray(scores))
    g = pd.Series(np.asarray(group))
    return {k: float(np.quantile(s[g == k], 1 - target_rate)) for k in g.unique()}


def run(seed: int = 11) -> dict:
    df = make_data(seed=seed)
    tr, te = train_test_split(df, test_size=0.3, random_state=seed, stratify=df.gender)
    model = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))  # noqa: E731

    # A. naive: learn from biased historic hiring labels, protected attribute excluded ("fairness through unawareness")
    m_a = model().fit(tr[FEATURES], tr.hired_historic)
    p_a = m_a.predict(te[FEATURES])
    # B. reweighing on the historic labels
    w = reweighing(tr.hired_historic, tr.gender)
    m_b = model().fit(tr[FEATURES], tr.hired_historic, logisticregression__sample_weight=w)
    p_b = m_b.predict(te[FEATURES])
    # C. B + equal selection rate per group (post-processing), at the overall rate of B
    s_b = m_b.predict_proba(te[FEATURES])[:, 1]
    th = group_thresholds(s_b, te.gender, target_rate=p_b.mean())
    p_c = (s_b >= te.gender.map(th).values).astype(int)

    results = {name: audit(te.true_suitable, p, te.gender) for name, p in
               [("A. Baseline (unaware)", p_a), ("B. Reweighing", p_b), ("C. Reweighing + group thresholds", p_c)]}
    imp = permutation_importance(m_a, te[FEATURES], te.hired_historic, n_repeats=8, random_state=seed)
    importance = pd.Series(imp.importances_mean, FEATURES).sort_values(ascending=False)
    proxy_corr = float(np.corrcoef(df.career_gap_months, (df.gender == "Female").astype(int))[0, 1])
    hist = audit(df.true_suitable, df.hired_historic, df.gender)
    return {"data": df, "test": te, "results": results, "importance": importance, "proxy_corr": proxy_corr,
            "historic": hist, "thresholds": th}
