from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar


@dataclass
class Finding:
    rule_id: str
    severity: str       # "critical" | "high" | "medium" | "low"
    message: str
    line: int
    file: str
    recommendation: str
    reference: str


class Rule(ABC):
    rule_id: ClassVar[str]
    severity: ClassVar[str]

    @abstractmethod
    def check(self, source: str, filepath: str) -> list[Finding]:
        """Run the rule against source code and return any findings."""
        ...
