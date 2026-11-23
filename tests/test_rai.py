import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rai import fairness as F  # noqa: E402
from rai.risk_screen import PORTFOLIO, UseCase, screen  # noqa: E402


def test_portfolio_tiers():
    tiers = {u.name: screen(u).tier for u in PORTFOLIO}
    assert tiers["AI-assisted CV screening for store hiring"] == "high"
    assert tiers["GenAI customer service assistant"] == "limited"
    assert tiers["Demand forecasting & replenishment"] == "minimal"
    assert tiers["Staff emotion monitoring via store cameras"] == "prohibited"


def test_prohibited_beats_everything():
    uc = UseCase("x", "", prohibited_flags=["social_scoring"], high_risk_areas=["employment"],
                 transparency_flags=["interacts_with_people"])
    assert screen(uc).tier == "prohibited"


def test_article_6_3_derogation_only_without_profiling():
    narrow = UseCase("x", "", high_risk_areas=["employment"], narrow_procedural_only=True)
    profiling = UseCase("y", "", high_risk_areas=["employment"], narrow_procedural_only=True, profiles_people=True)
    assert screen(narrow).tier == "minimal"
    assert screen(profiling).tier == "high"


def test_unknown_flag_rejected():
    with pytest.raises(ValueError):
        screen(UseCase("x", "", high_risk_areas=["astrology"]))


def test_audit_metrics_on_known_example():
    y = [1, 1, 0, 0, 1, 1, 0, 0]
    p = [1, 1, 1, 0, 1, 0, 0, 0]
    g = ["Male"] * 4 + ["Female"] * 4
    m = F.audit(y, p, g)
    assert m.disparate_impact_ratio == pytest.approx(0.25 / 0.75)
    assert m.equal_opportunity_diff == pytest.approx(0.5 - 1.0)


def test_reweighing_makes_label_independent_of_group():
    df = F.make_data(n=3000)
    w = F.reweighing(df.hired_historic, df.gender)
    rates = {g: np.average(df.hired_historic[df.gender == g], weights=w[df.gender == g]) for g in ["Male", "Female"]}
    assert abs(rates["Male"] - rates["Female"]) < 1e-9


def test_mitigation_improves_fairness_without_hurting_accuracy():
    r = F.run()
    a, c = r["results"]["A. Baseline (unaware)"], r["results"]["C. Reweighing + group thresholds"]
    assert r["proxy_corr"] > 0.3
    assert c.disparate_impact_ratio > a.disparate_impact_ratio
    assert abs(c.equal_opportunity_diff) < abs(a.equal_opportunity_diff)
    assert c.accuracy >= a.accuracy - 0.02
