from probus.rules.conceptual_soundness.cs003_scaler_fit_full_dataset import (
    CS003ScalerFitFullDataset,
)

# Rule registry: add new rule classes here to make them discoverable by the runner.
_RULE_CLASSES = [
    CS003ScalerFitFullDataset,
]


def get_all_rules() -> list:
    return [cls() for cls in _RULE_CLASSES]
