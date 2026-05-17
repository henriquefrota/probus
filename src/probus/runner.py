from pathlib import Path

from probus.rules import get_all_rules
from probus.rules.base import Finding


class Runner:
    def __init__(self) -> None:
        self.rules = get_all_rules()

    def run(self, path: str | Path) -> list[Finding]:
        path = Path(path)
        if path.is_file():
            return self._run_file(path)
        if path.is_dir():
            findings: list[Finding] = []
            for py_file in sorted(path.rglob("*.py")):
                findings.extend(self._run_file(py_file))
            return findings
        return []

    def _run_file(self, filepath: Path) -> list[Finding]:
        try:
            source = filepath.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return []
        findings: list[Finding] = []
        for rule in self.rules:
            findings.extend(rule.check(source, str(filepath)))
        return findings
