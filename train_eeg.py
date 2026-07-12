#!/usr/bin/env python3
"""Command-line entry point for reproducible HC/AD EEG experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eeg_training.runner import (
    check_configuration,
    run_experiment,
    run_sanity_overfit,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the independent-segment HC/AD EEG model and archive every result "
            "under a unique exp/ subdirectory."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "eeg_hc_ad.json",
        help="JSON configuration file (default: configs/eeg_hc_ad.json)",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="optional experiment-name override used in the result directory name",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="optional device override, for example cpu, cuda:0, or auto",
    )
    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument(
        "--check",
        action="store_true",
        help="validate data, split, graph, and model construction without training/writes",
    )
    action_group.add_argument(
        "--sanity-overfit",
        action="store_true",
        help="try to memorize eight balanced real subjects before a full run",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_file = args.config
    if not config_file.is_absolute():
        config_file = (PROJECT_ROOT / config_file).resolve()
    if args.check:
        report = check_configuration(
            project_root=PROJECT_ROOT,
            config_file=config_file,
            run_name=args.run_name,
            device_override=args.device,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.sanity_overfit:
        summary = run_sanity_overfit(
            project_root=PROJECT_ROOT,
            config_file=config_file,
            run_name=args.run_name,
            device_override=args.device,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if summary["status"] == "sanity_overfit_passed" else 2

    summary = run_experiment(
        project_root=PROJECT_ROOT,
        config_file=config_file,
        run_name=args.run_name,
        device_override=args.device,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "run_directory": summary["run_directory"],
                "best_epoch": summary["best_epoch"],
                "test_fixed_threshold_metrics": {
                    name: summary["final_metrics"]["test"][name]
                    for name in (
                        "segment",
                        "subject_majority_vote",
                        "subject_logit_mean",
                    )
                },
                "test_validation_tuned_metrics": summary["final_metrics"][
                    "test_tuned_threshold"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
