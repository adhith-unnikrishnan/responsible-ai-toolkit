"""AI use-case risk screening against the EU AI Act risk tiers and the UK's five AI regulatory principles.

An indicative triage tool for consultants and governance boards. It is NOT legal advice: a "high" or
"prohibited" result means "involve legal/compliance before going further".
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Article 5 prohibited practices (summarised)
PROHIBITED = {
    "manipulation": "Subliminal, manipulative or deceptive techniques that materially distort behaviour and cause significant harm",
    "exploit_vulnerability": "Exploiting vulnerabilities due to age, disability or social/economic situation",
    "social_scoring": "Social scoring leading to detrimental or unjustified treatment",
    "predictive_policing_profiling": "Predicting criminal offences based solely on profiling or personality traits",
    "untargeted_face_scraping": "Building facial-recognition databases by untargeted scraping",
    "emotion_recognition_work_education": "Emotion recognition in the workplace or education (except medical/safety)",
    "biometric_categorisation_sensitive": "Biometric categorisation to infer sensitive traits (e.g. race, religion, sexual orientation)",
    "realtime_remote_biometric_id_law_enforcement": "Real-time remote biometric identification in public spaces for law enforcement (narrow exceptions)",
}

# Annex III high-risk areas (summarised)
HIGH_RISK_AREAS = {
    "biometrics": "Remote biometric identification, biometric categorisation, emotion recognition",
    "critical_infrastructure": "Safety components in critical infrastructure (utilities, traffic, digital infrastructure)",
    "education": "Admission, assessment, steering learning, proctoring",
    "employment": "Recruitment, CV screening, promotion/termination, task allocation, monitoring of workers",
    "essential_services": "Creditworthiness/credit scoring, life & health insurance pricing, public benefits, emergency triage",
    "law_enforcement": "Risk assessment, evidence reliability, profiling in investigations",
    "migration_border": "Visa/asylum assessment, border risk assessment",
    "justice_democracy": "Assisting judicial decisions, influencing elections",
}

TRANSPARENCY = {
    "interacts_with_people": "Chatbots and agents must disclose that users are interacting with AI",
    "generates_synthetic_content": "AI-generated audio, image, video or text must be marked as such",
    "deepfake": "Deepfakes must be disclosed",
    "emotion_or_biometric_categorisation": "People must be informed when emotion recognition / biometric categorisation is used",
}

UK_PRINCIPLES = [
    "Safety, security & robustness",
    "Appropriate transparency & explainability",
    "Fairness",
    "Accountability & governance",
    "Contestability & redress",
]

CONTROLS = {
    "prohibited": ["Stop: do not develop or deploy; escalate to legal and the AI ethics board"],
    "high": [
        "Risk management system across the lifecycle",
        "Data governance: representative, relevant, bias-examined training data",
        "Technical documentation and model card",
        "Automatic logging / record-keeping",
        "Transparency and instructions for deployers",
        "Effective human oversight (ability to override / stop)",
        "Accuracy, robustness and cybersecurity testing",
        "Conformity assessment, EU database registration, post-market monitoring",
        "Fundamental-rights impact assessment where required (e.g. public bodies, credit, insurance)",
    ],
    "limited": ["Clear AI disclosure to users", "Label AI-generated content", "Model card and monitoring as good practice"],
    "minimal": ["Voluntary code of conduct", "Standard model risk management and monitoring"],
}


@dataclass
class UseCase:
    name: str
    description: str
    prohibited_flags: list[str] = field(default_factory=list)
    high_risk_areas: list[str] = field(default_factory=list)
    transparency_flags: list[str] = field(default_factory=list)
    narrow_procedural_only: bool = False  # Art. 6(3) derogation: narrow procedural / preparatory task, no profiling
    profiles_people: bool = False
    gpai_based: bool = False


@dataclass
class Screening:
    use_case: str
    tier: str
    reasons: list[str]
    controls: list[str]
    uk_principles_focus: list[str]


def screen(uc: UseCase) -> Screening:
    bad = [k for k in uc.prohibited_flags if k not in PROHIBITED]
    bad += [k for k in uc.high_risk_areas if k not in HIGH_RISK_AREAS]
    bad += [k for k in uc.transparency_flags if k not in TRANSPARENCY]
    if bad:
        raise ValueError(f"unknown flags: {bad}")

    reasons: list[str] = []
    if uc.prohibited_flags:
        tier = "prohibited"
        reasons += [f"Art. 5: {PROHIBITED[k]}" for k in uc.prohibited_flags]
    elif uc.high_risk_areas and not (uc.narrow_procedural_only and not uc.profiles_people):
        tier = "high"
        reasons += [f"Annex III: {HIGH_RISK_AREAS[k]}" for k in uc.high_risk_areas]
        if uc.profiles_people:
            reasons.append("Profiles natural persons, so the Art. 6(3) derogation is not available")
    elif uc.transparency_flags:
        tier = "limited"
        reasons += [f"Art. 50: {TRANSPARENCY[k]}" for k in uc.transparency_flags]
        if uc.high_risk_areas:
            reasons.append("Annex III area, but narrow procedural task without profiling (Art. 6(3)): document the assessment")
    else:
        tier = "minimal"
        reasons.append("No prohibited practice, Annex III area or transparency trigger identified")
        if uc.high_risk_areas:
            reasons.append("Annex III area, but narrow procedural task without profiling (Art. 6(3)): document the assessment")

    controls = list(CONTROLS[tier])
    if tier in ("high", "minimal") and uc.transparency_flags:
        controls += ["Clear AI disclosure to users / label AI-generated content"]
    if uc.gpai_based:
        controls.append("Check the GPAI model provider's documentation and usage policy; evaluate outputs for hallucination")

    focus = {"prohibited": UK_PRINCIPLES, "high": UK_PRINCIPLES,
             "limited": [UK_PRINCIPLES[1], UK_PRINCIPLES[3]],
             "minimal": [UK_PRINCIPLES[0], UK_PRINCIPLES[3]]}[tier]
    if uc.profiles_people and UK_PRINCIPLES[2] not in focus:
        focus = focus + [UK_PRINCIPLES[2], UK_PRINCIPLES[4]]
    return Screening(uc.name, tier, reasons, controls, list(focus))


PORTFOLIO = [  # the same fictional retailer's use cases as the AI value case repo
    UseCase("AI-assisted CV screening for store hiring", "Ranks applicants for store roles",
            high_risk_areas=["employment"], profiles_people=True),
    UseCase("Workforce scheduling optimisation", "Allocates shifts using individual performance and availability",
            high_risk_areas=["employment"], profiles_people=True),
    UseCase("GenAI customer service assistant", "Answers customer queries via chat",
            transparency_flags=["interacts_with_people"], gpai_based=True),
    UseCase("GenAI product content generation", "Drafts product descriptions for the website",
            transparency_flags=["generates_synthetic_content"], gpai_based=True),
    UseCase("Demand forecasting & replenishment", "Forecasts SKU-store demand"),
    UseCase("Payment & returns fraud detection", "Flags suspicious payments and returns for review"),
    UseCase("Staff emotion monitoring via store cameras", "Proposed by a vendor to track staff mood",
            prohibited_flags=["emotion_recognition_work_education"]),
]
