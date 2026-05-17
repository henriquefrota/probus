from probus.rules.conceptual_soundness.cs001_lookahead_shift_negative import (
    CS001LookaheadShiftNegative,
)
from probus.rules.conceptual_soundness.cs002_same_period_signal_execution import (
    CS002SamePeriodSignalExecution,
)
from probus.rules.conceptual_soundness.cs003_scaler_fit_full_dataset import (
    CS003ScalerFitFullDataset,
)

# Rule registry: add new rule classes here to make them discoverable by the runner.
_RULE_CLASSES = [
    CS001LookaheadShiftNegative,
    CS002SamePeriodSignalExecution,
    CS003ScalerFitFullDataset,
]


def get_all_rules() -> list:
    return [cls() for cls in _RULE_CLASSES]
