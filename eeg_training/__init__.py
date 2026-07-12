"""Training helpers for independent EEG segments and subject-level reporting."""

from .data import DatasetInfo, SubjectRecord, load_dataset_info
from .runner import check_configuration, run_experiment, run_sanity_overfit

__all__ = [
    "DatasetInfo",
    "SubjectRecord",
    "check_configuration",
    "load_dataset_info",
    "run_experiment",
    "run_sanity_overfit",
]
