from probus.rules.conceptual_soundness.cs001_lookahead_shift_negative import (
    CS001LookaheadShiftNegative,
)
from probus.rules.conceptual_soundness.cs002_same_period_signal_execution import (
    CS002SamePeriodSignalExecution,
)
from probus.rules.conceptual_soundness.cs003_scaler_fit_full_dataset import (
    CS003ScalerFitFullDataset,
)
from probus.rules.outcomes_analysis.oa001_no_transaction_costs import (
    OA001NoTransactionCosts,
)
from probus.rules.process_verification.pv001_missing_random_seed import (
    PV001MissingRandomSeed,
)
from probus.rules.robustness_testing.rt001_random_split_time_series import (
    RT001RandomSplitTimeSeries,
)

# Rule registry: add new rule classes here to make them discoverable by the runner.
_RULE_CLASSES = [
    CS001LookaheadShiftNegative,
    CS002SamePeriodSignalExecution,
    CS003ScalerFitFullDataset,
    OA001NoTransactionCosts,
    PV001MissingRandomSeed,
    RT001RandomSplitTimeSeries,
]


def get_all_rules() -> list:
    return [cls() for cls in _RULE_CLASSES]
