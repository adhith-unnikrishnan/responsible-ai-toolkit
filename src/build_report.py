"""Build the Responsible AI report (GitHub Pages) and a model card (Markdown)."""
from __future__ import annotations

import html
import sys
from datetime import date
from pathlib import Path

import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rai import fairness as F  # noqa: E402
from rai.risk_screen import PORTFOLIO, UK_PRINCIPLES, screen  # noqa: E402

DOCS = Path(__file__).resolve().parents[1] / "docs"
TIER_C = {"prohibited": "#7A1F2B", "high": "#B34A6B", "limited": "#E0A12F", "minimal": "#3A9D7A"}
LAYOUT = dict(font=dict(family="Inter, Segoe UI, Arial", size=12, color="#1F2933"), paper_bgcolor="white",
              plot_bgcolor="white", margin=dict(l=60, r=20, t=30, b=40))


def fig_html(fig, first=False, h=330):
    fig.update_layout(height=h, **LAYOUT)
    fig.update_xaxes(gridcolor="#E6E9ED")
    fig.update_yaxes(gridcolor="#E6E9ED")
    return fig.to_html(full_html=False, include_plotlyjs="cdn" if first else False, config={"displaylogo": False})


def model_card(r: dict) -> str:
    c = r["results"]["C. Reweighing + group thresholds"]
    a = r["results"]["A. Baseline (unaware)"]
    return f"""# Model card: CV shortlisting assistant (store roles)

*Generated {date.today():%d %B %Y} by `src/build_report.py`. Fictional use case on synthetic data.*

## Intended use
- **Purpose:** rank applicants for store-assistant roles so recruiters review the most suitable first.
- **Users:** trained recruiters. **The model recommends; a human decides.** No automatic rejections.
- **Out of scope:** management roles, promotion, termination, or any use without human review.

## Regulatory classification
- **EU AI Act:** high-risk (Annex III, employment: recruitment and CV screening). It profiles natural persons, so the Art. 6(3) derogation does not apply.
- **UK AI principles in focus:** {", ".join(UK_PRINCIPLES)}.

## Data
- {len(r['data']):,} synthetic applications. The features are experience, skills, customer-service assessment, degree, career-gap months and distance.
- The protected attribute (gender) is **excluded from features** but used for auditing.
- **Known issue:** the historic hiring labels are biased (disparate impact {r['historic'].disparate_impact_ratio:.2f}), and career-gap months is a proxy for gender (correlation {r['proxy_corr']:.2f}).

## Model
Logistic regression (standardised features), trained with **reweighing**. Post-processed with **group-specific thresholds** to equalise selection rates.

## Performance & fairness (hold-out, against job-relevant suitability)
| Metric | Baseline | Released model |
|---|---:|---:|
| Accuracy | {a.accuracy:.3f} | {c.accuracy:.3f} |
| Disparate impact ratio (≥ 0.80) | {a.disparate_impact_ratio:.2f} | {c.disparate_impact_ratio:.2f} |
| Demographic parity difference | {a.demographic_parity_diff:+.3f} | {c.demographic_parity_diff:+.3f} |
| Equal opportunity difference (TPR gap) | {a.equal_opportunity_diff:+.3f} | {c.equal_opportunity_diff:+.3f} |

## Limitations & risks
- Group-specific thresholds use the protected attribute at decision time. Their lawfulness depends on jurisdiction (e.g. UK Equality Act positive-action rules), so **legal review is required before use**.
- Only binary gender is audited. Audit other protected characteristics (age, disability, ethnicity) before deployment.
- Synthetic data: real performance must be re-validated on the client's data.

## Human oversight, monitoring & redress
- Recruiters see the top factors for each score and can override them. Overrides are logged.
- Monthly fairness and drift monitoring, with an alert if the disparate impact ratio falls below 0.85 or the TPR gap exceeds 5 pts.
- Candidates can request human review of any decision (contestability & redress).
- Owner: Head of Talent Acquisition. Model risk sign-off: AI governance board.
"""


def main() -> None:
    r = F.run()
    res = r["results"]
    screens = [screen(u) for u in PORTFOLIO]

    # risk table
    rows = "".join(
        f"<tr><td><b>{html.escape(s.use_case)}</b></td><td><span class='tag' style='background:{TIER_C[s.tier]}'>{s.tier}</span></td>"
        f"<td>{'<br>'.join(html.escape(x) for x in s.reasons)}</td><td>{len(s.controls)} control{'s' if len(s.controls) != 1 else ''}</td></tr>" for s in screens)
    hi = next(s for s in screens if s.tier == "high")
    checklist = "".join(f"<li>☐ {html.escape(c)}</li>" for c in hi.controls)

    # fairness charts
    names = list(res)
    f1 = go.Figure()
    for g, col in [("Female", "#B34A6B"), ("Male", "#2E5EAA")]:
        t = res[names[0]].table
        f1.add_trace(go.Bar(x=["Selection rate", "True positive rate", "False positive rate", "Precision"],
                            y=t.loc[g, ["selection_rate", "true_positive_rate", "false_positive_rate", "precision"]],
                            name=g, marker_color=col, hovertemplate=g + " %{x}: %{y:.1%}<extra></extra>"))
    f1.update_layout(barmode="group", yaxis=dict(tickformat=".0%"), legend=dict(orientation="h", y=-0.15))

    ok = lambda v, good: f"<td style='color:{'#3A9D7A' if good else '#B34A6B'};font-weight:600'>{v}</td>"  # noqa: E731
    mit_rows = "".join(
        f"<tr><td><b>{n}</b></td><td>{res[n].accuracy:.3f}</td>"
        + ok(f"{res[n].disparate_impact_ratio:.2f}", res[n].disparate_impact_ratio >= 0.8)
        + ok(f"{res[n].demographic_parity_diff:+.3f}", abs(res[n].demographic_parity_diff) <= 0.05)
        + ok(f"{res[n].equal_opportunity_diff:+.3f}", abs(res[n].equal_opportunity_diff) <= 0.1) + "</tr>" for n in names)
    mit_table = ("<table><thead><tr><th>Approach</th><th>Accuracy</th><th>Disparate impact (≥ 0.80)</th>"
                 "<th>Demographic parity diff (±0.05)</th><th>Equal opportunity diff (±0.10)</th></tr></thead>"
                 f"<tbody>{mit_rows}</tbody></table>")

    imp = r["importance"].sort_values()
    f3 = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h",
                          marker_color=["#B34A6B" if i == "career_gap_months" else "#9DB3D8" for i in imp.index]))
    f3.update_layout(margin=dict(l=170, r=20, t=20, b=40), xaxis=dict(title="Permutation importance (baseline model)"))

    a, c = res[names[0]], res[names[2]]
    card = model_card(r)
    (DOCS / "model_card.md").write_text(card, encoding="utf-8")

    import markdown as md  # optional dependency for HTML rendering of the card
    card_html = md.markdown(card, extensions=["tables"])

    sections = [
        ("1. Risk screening: the AI use-case portfolio",
         "Each proposed use case is triaged against the EU AI Act tiers (Art. 5 prohibited practices, Annex III high-risk "
         "areas, Art. 50 transparency duties) and mapped to the UK's five AI principles. The tier sets the controls needed "
         "before build. One vendor proposal (emotion monitoring of staff) is stopped outright.",
         f"<table><thead><tr><th>Use case</th><th>Tier</th><th>Why</th><th>Required</th></tr></thead><tbody>{rows}</tbody></table>"
         f"<p class='note'>Indicative screening, not legal advice.</p>"),
        ("2. Fairness audit: CV shortlisting (high-risk)",
         f"Historic hiring decisions show a disparate impact of <b>{r['historic'].disparate_impact_ratio:.2f}</b> against women. "
         f"The baseline model excludes gender, but it still learns the bias through a proxy: career-gap months correlate "
         f"<b>{r['proxy_corr']:.2f}</b> with gender. It passes the four-fifths rule (DI {a.disparate_impact_ratio:.2f}), yet "
         f"suitable women are <b>{abs(a.equal_opportunity_diff) * 100:.0f} percentage points</b> less likely to be shortlisted than suitable men. "
         "A single fairness metric would have missed this.",
         '<div class="two">' + fig_html(f1, first=True) + fig_html(f3) + "</div>"),
        ("3. Mitigation: trade-offs made visible",
         f"Reweighing (pre-processing) and group-specific thresholds (post-processing) bring the disparate impact to "
         f"<b>{c.disparate_impact_ratio:.2f}</b> and cut the opportunity gap to <b>{c.equal_opportunity_diff:+.3f}</b>, while "
         f"accuracy against job-relevant suitability holds at <b>{c.accuracy:.3f}</b> (baseline {a.accuracy:.3f}). "
         "Group thresholds use the protected attribute at decision time, so the model card flags them for legal review.",
         mit_table),
        ("4. Governance checklist for high-risk deployment",
         f"Controls required before go-live for <b>{html.escape(hi.use_case)}</b>:",
         f"<ul class='check'>{checklist}</ul>"),
        ("5. Model card (generated)", "Documentation produced automatically from the audit run, so it can't drift from the model.",
         f"<div class='card'>{card_html}</div>"),
    ]
    body = "".join(f"<section><h2>{h}</h2><p>{t}</p>{c_}</section>" for h, t, c_ in sections)
    page = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Responsible AI Toolkit | Adhith Unnikrishnan</title><style>
body{{margin:0;background:#F4F6F8;font-family:Inter,Segoe UI,Arial,sans-serif;color:#1F2933}}
header{{background:#14213D;color:#fff;padding:28px 40px}} header h1{{margin:0 0 6px;font-size:26px}}
header p{{margin:0;color:#C9D3E3;font-size:14px;max-width:950px;line-height:1.5}} header a{{color:#9CC3FF}}
main{{max-width:1200px;margin:0 auto;padding:24px}} section{{background:#fff;border-radius:10px;padding:20px 22px;margin-bottom:20px;box-shadow:0 1px 3px rgba(0,0,0,.06)}}
h2{{font-size:18px;margin:0 0 6px}} section>p{{font-size:14px;line-height:1.55;color:#3E4C59;max-width:1000px}}
table{{border-collapse:collapse;font-size:13px;width:100%}} th,td{{padding:8px 10px;border-bottom:1px solid #E6E9ED;text-align:left;vertical-align:top}}
th{{color:#6B7785}} .tag{{color:#fff;border-radius:10px;padding:2px 9px;font-size:11px;text-transform:uppercase}}
.two{{display:grid;grid-template-columns:1fr 1fr;gap:12px}} @media(max-width:900px){{.two{{grid-template-columns:1fr}}}}
.check{{columns:2;font-size:13.5px;line-height:1.8;list-style:none;padding-left:4px}} .note{{font-size:12px!important;color:#6B7785!important}}
.card{{border:1px solid #E6E9ED;border-radius:8px;padding:6px 22px;font-size:13.5px;line-height:1.55}} .card h1{{font-size:20px}} .card h2{{font-size:15px;margin-top:16px}}
</style></head><body><header><h1>Responsible AI Toolkit</h1>
<p>Embedding Responsible AI into an AI strategy: risk-tier every use case (EU AI Act and UK principles), audit high-risk
models for bias, mitigate with visible trade-offs, and generate the documentation governance boards need. The worked example
uses the same fictional UK retailer as the <a href="https://adhith-unnikrishnan.github.io/ai-value-case-roadmap/">AI value case</a>.
Source: <a href="https://github.com/adhith-unnikrishnan/responsible-ai-toolkit">GitHub</a></p></header><main>{body}</main></body></html>"""
    (DOCS / "index.html").write_text(page, encoding="utf-8")
    print("wrote docs/index.html and docs/model_card.md")


if __name__ == "__main__":
    main()
