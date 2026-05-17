import json
from dataclasses import asdict

from probus.rules.base import Finding

_SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}

_SEVERITY_WEIGHTS = {"critical": 30, "high": 20, "medium": 10, "low": 5}

_CATEGORIES = {
    "CS": "Conceptual Soundness",
    "OA": "Outcomes Analysis",
    "RT": "Robustness Testing",
    "PV": "Process Verification",
}


def _category_score(findings: list[Finding], prefix: str) -> int:
    score = 100
    for f in findings:
        if f.rule_id.startswith(prefix):
            score -= _SEVERITY_WEIGHTS.get(f.severity, 0)
    return max(0, score)


def _overall_score(findings: list[Finding]) -> int:
    score = 100
    for f in findings:
        score -= _SEVERITY_WEIGHTS.get(f.severity, 0)
    return max(0, score)


def _risk_label(score: int) -> str:
    if score >= 85:
        return "Low Risk"
    if score >= 65:
        return "Moderate Risk"
    if score >= 40:
        return "High Risk"
    return "Critical Risk"


def to_markdown(findings: list[Finding], source_path: str) -> str:
    lines: list[str] = []
    lines.append("# Probus Model Risk Report\n")
    lines.append(f"**Source:** `{source_path}`\n")

    overall = _overall_score(findings)
    lines.append(f"## Overall Score\n\n{overall}/100 — {_risk_label(overall)}\n")

    lines.append("## Category Scores\n")
    for prefix, name in _CATEGORIES.items():
        lines.append(f"- {name}: {_category_score(findings, prefix)}/100")
    lines.append("")

    lines.append("## Findings\n")
    if not findings:
        lines.append("No issues detected.\n")
    else:
        sorted_findings = sorted(
            findings, key=lambda f: _SEVERITY_ORDER.get(f.severity, 99)
        )
        for f in sorted_findings:
            lines.append(f"### [{f.severity.upper()}] {f.rule_id}")
            lines.append(f"\n**File:** `{f.file}` — Line {f.line}\n")
            lines.append(f"{f.message}\n")
            lines.append(f"**Recommendation:** {f.recommendation}\n")
            lines.append(f"**Reference:** {f.reference}\n")

    lines.append("## Limitations\n")
    lines.append(
        "Probus is not a replacement for institutional model validation. "
        "It automates selected static checks and does not guarantee absence of "
        "methodological issues. Human judgment remains essential.\n"
    )

    lines.append("## References\n")
    for ref in [
        "Federal Reserve SR 11-7",
        "BCB Resolução 4557",
        "López de Prado (2018) — Advances in Financial Machine Learning",
        "Bailey & López de Prado (2014) — The Probability of Backtest Overfitting",
    ]:
        lines.append(f"- {ref}")

    return "\n".join(lines)


def to_json(findings: list[Finding]) -> str:
    return json.dumps([asdict(f) for f in findings], indent=2)
