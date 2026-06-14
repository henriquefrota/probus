"""Regression guard for the Markdown report the end user sees.

These snapshot tests pin the exact text of report.to_markdown so that any
unintended change to the report layout, scoring, or wording is caught. To
intentionally update the snapshots after a deliberate change, run:

    pytest tests/test_report_snapshot.py --snapshot-update
"""

from probus.report import to_markdown
from probus.rules.base import Finding


def _sample_findings() -> list[Finding]:
    """Findings spanning categories and severities, in non-sorted order.

    Listing them out of severity order also exercises the report's sorting.
    """
    return [
        Finding(
            rule_id="OA001",
            severity="medium",
            message="No mention of transaction costs found anywhere in this file.",
            line=1,
            file="backtest.py",
            recommendation="Add realistic transaction cost assumptions.",
            reference="SR 11-7 §Outcomes Analysis",
        ),
        Finding(
            rule_id="CS001",
            severity="critical",
            message="shift(-1) at line 6 reads future observations.",
            line=6,
            file="backtest.py",
            recommendation="Remove the negative shift from signal construction.",
            reference="SR 11-7 §Conceptual Soundness",
        ),
        Finding(
            rule_id="CS002",
            severity="high",
            message="A signal is multiplied with returns without shift(1).",
            line=12,
            file="backtest.py",
            recommendation="Apply signal.shift(1) before multiplying with returns.",
            reference="SR 11-7 §Conceptual Soundness",
        ),
    ]


def test_report_with_findings(snapshot):
    output = to_markdown(_sample_findings(), "backtest.py")
    snapshot.assert_match(output, "report_with_findings.md")


def test_report_no_findings(snapshot):
    output = to_markdown([], "clean.py")
    snapshot.assert_match(output, "report_no_findings.md")
