from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from eeg_training.data import (
    IndependentSegmentDataset,
    SubjectRecord,
    collate_independent_segments,
    load_dataset_info,
    segment_class_weights,
    split_manifest,
    stratified_subject_split,
)
from eeg_training.engine import (
    aggregate_subject_logit_mean,
    aggregate_subject_majority_vote,
    make_grad_scaler,
    make_segment_loader,
    run_epoch,
)
from eeg_training.experiment import create_experiment
from eeg_training.metrics import (
    binary_classification_metrics,
    select_balanced_accuracy_threshold,
)
import eeg_training.metrics as metrics_module
import eeg_training.runner as runner_module
from eeg_training.modeling import load_model_module
from eeg_training.runner import (
    DEFAULTS,
    _aggregate_numeric_metrics,
    _physiology_weight,
    _split_seed_sequence,
    run_experiment,
)
from model_design.eeg_hc_ad_model import EEGModelConfig
from train_eeg import parse_args as parse_training_args


class _TinySegmentModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.offset = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(sampling_rate=128.0, bands=())

    def forward(self, eeg):
        # This signature intentionally accepts no subject/bag metadata.
        score = eeg.mean(dim=(1, 2)) + self.offset
        logits = torch.stack((-score, score), dim=-1)
        return {
            "segment_logits": logits,
            "learned_quality_scores": torch.full_like(score, 0.5),
        }


class _TinySegmentCriterion(torch.nn.Module):
    segment_weight = 1.0
    physiology_weight = 0.0
    quality_weight = 0.0

    def forward(self, outputs, labels, physiology_targets=None, class_weight=None):
        del physiology_targets
        loss = F.cross_entropy(
            outputs["segment_logits"], labels, weight=class_weight
        )
        zero = loss.detach() * 0.0
        return {
            "loss": loss,
            "segment_loss": loss,
            "physiology_loss": zero,
            "quality_loss": zero,
        }


def _record(
    directory: Path,
    subject_index: int,
    label: int,
    n_segments: int,
    values: np.ndarray | None = None,
) -> SubjectRecord:
    data_file = directory / f"subject_{subject_index}.npy"
    if values is None:
        value = -1.0 if label == 0 else 1.0
        values = np.full((n_segments, 2, 8), value, dtype=np.float32)
    np.save(data_file, values)
    return SubjectRecord(
        subject_id=f"subject_{subject_index}",
        diagnosis="AD" if label else "HC",
        label=label,
        institution="AR",
        data_file=data_file,
        n_segments=n_segments,
        n_channels=2,
        n_samples=8,
        sampling_rate=128.0,
        channel_names=("A", "B"),
    )


class TrainingFrameworkTests(unittest.TestCase):
    def test_model_normalization_cli_is_a_tristate_override(self) -> None:
        with patch("sys.argv", ["train_eeg.py"]):
            self.assertIsNone(parse_training_args().normalize_per_channel)
        with patch("sys.argv", ["train_eeg.py", "--normalize"]):
            self.assertIs(parse_training_args().normalize_per_channel, True)
        with patch("sys.argv", ["train_eeg.py", "--no-normalize"]):
            self.assertIs(parse_training_args().normalize_per_channel, False)

    def test_model_normalization_override_updates_resolved_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project = Path(temporary_directory)
            config_file = project / "config.json"
            config_file.write_text(
                json.dumps(
                    {
                        "model": {
                            "parameters": {"normalize_per_channel": True},
                        }
                    }
                ),
                encoding="utf-8",
            )
            model_source = project / "model_design" / "eeg_hc_ad_model.py"
            model_source.parent.mkdir()
            model_source.write_text("# test model source\n", encoding="utf-8")
            data_info = SimpleNamespace(
                dataset_name="toy",
                records=[],
                n_channels=2,
                n_samples=8,
                sampling_rate=128.0,
            )
            splits = {"train": [], "validation": [], "test": []}
            with (
                patch.object(runner_module, "load_dataset_info", return_value=data_info),
                patch.object(
                    runner_module,
                    "stratified_subject_split",
                    return_value=splits,
                ),
                patch.object(runner_module, "split_manifest", return_value={}),
            ):
                configured = runner_module._prepare_configuration(
                    project_root=project,
                    config_file=config_file,
                    run_name=None,
                    device_override=None,
                )
                overridden = runner_module._prepare_configuration(
                    project_root=project,
                    config_file=config_file,
                    run_name=None,
                    device_override=None,
                    normalize_per_channel_override=False,
                )

            self.assertIs(
                configured.config["model"]["parameters"][
                    "normalize_per_channel"
                ],
                True,
            )
            self.assertIs(
                overridden.config["model"]["parameters"][
                    "normalize_per_channel"
                ],
                False,
            )

    def test_main_model_rejects_non_boolean_normalization_config(self) -> None:
        with self.assertRaisesRegex(ValueError, "normalize_per_channel"):
            EEGModelConfig(
                n_channels=2,
                sampling_rate=128.0,
                normalize_per_channel="false",  # type: ignore[arg-type]
            )

    def test_multi_split_defaults_and_metric_aggregation(self) -> None:
        self.assertEqual(DEFAULTS["data"]["split_seed_count"], 10)
        self.assertEqual(_split_seed_sequence(DEFAULTS), list(range(2026, 2036)))
        aggregate = _aggregate_numeric_metrics(
            [
                {"final_metrics": {"test": {"balanced_accuracy": 0.6}}},
                {"final_metrics": {"test": {"balanced_accuracy": 0.8}}},
            ],
            "final_metrics",
        )
        metric = aggregate["test"]["balanced_accuracy"]
        self.assertEqual(metric["n"], 2)
        self.assertAlmostEqual(metric["mean"], 0.7)
        self.assertAlmostEqual(metric["std"], 2**0.5 / 10)
        self.assertEqual(metric["min"], 0.6)
        self.assertEqual(metric["max"], 0.8)

    def test_binary_metrics(self) -> None:
        metrics = binary_classification_metrics(
            [0, 0, 1, 1], [0.1, 0.7, 0.8, 0.9]
        )
        self.assertEqual(metrics["n_samples"], 4)
        self.assertEqual(metrics["true_positive"], 2)
        self.assertEqual(metrics["false_positive"], 1)
        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["roc_auc_ad"], 1.0)
        threshold, tuned = select_balanced_accuracy_threshold(
            [0, 0, 1, 1], [0.4, 0.45, 0.46, 0.47]
        )
        self.assertGreater(threshold, 0.45)
        self.assertEqual(tuned["balanced_accuracy"], 1.0)

    def test_roc_auc_handles_tied_scores(self) -> None:
        metrics = binary_classification_metrics(
            [0, 0, 1, 1], [0.1, 0.5, 0.5, 0.9]
        )
        self.assertAlmostEqual(metrics["roc_auc_ad"], 0.875)

    def test_threshold_selection_matches_candidate_search_and_computes_auc_once(
        self,
    ) -> None:
        labels = [0, 1, 0, 1, 0, 1, 0, 1]
        scores = [0.1, 0.1, 0.2, 0.4, 0.4, 0.7, 0.8, 0.9]
        unique_scores = sorted(set(scores))
        candidates = {0.0, 0.5, 1.0, *unique_scores}
        candidates.update(
            (left + right) / 2.0
            for left, right in zip(unique_scores, unique_scores[1:])
        )
        expected_threshold = 0.5
        expected_metrics = None
        expected_key = None
        for threshold in sorted(candidates):
            candidate_metrics = binary_classification_metrics(
                labels, scores, threshold=threshold
            )
            sensitivity = candidate_metrics["sensitivity_ad"]
            specificity = candidate_metrics["specificity_hc"]
            self.assertIsNotNone(sensitivity)
            self.assertIsNotNone(specificity)
            key = (
                candidate_metrics["balanced_accuracy"],
                min(sensitivity, specificity),
                -abs(threshold - 0.5),
            )
            if expected_key is None or key > expected_key:
                expected_key = key
                expected_threshold = threshold
                expected_metrics = candidate_metrics

        with patch.object(metrics_module, "_roc_auc", wraps=metrics_module._roc_auc) as auc:
            threshold, tuned = select_balanced_accuracy_threshold(labels, scores)

        self.assertEqual(threshold, expected_threshold)
        self.assertEqual(tuned, expected_metrics)
        self.assertEqual(auc.call_count, 1)

    def test_subject_split_has_no_overlap(self) -> None:
        records = []
        for label, diagnosis in ((0, "HC"), (1, "AD")):
            for institution in ("AR", "CL"):
                for index in range(10):
                    records.append(
                        SubjectRecord(
                            subject_id=f"{diagnosis}_{institution}_{index}",
                            diagnosis=diagnosis,
                            label=label,
                            institution=institution,
                            data_file=Path(f"unused_{index}.npy"),
                            n_segments=3,
                            n_channels=2,
                            n_samples=8,
                            sampling_rate=128.0,
                            channel_names=("A", "B"),
                        )
                    )
        splits = stratified_subject_split(records, (0.6, 0.2, 0.2), seed=7)
        ids = [record.subject_id for split in splits.values() for record in split]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(ids), len(records))
        manifest = split_manifest(splits, (0.6, 0.2, 0.2), seed=7)
        self.assertEqual(manifest["split_unit"], "subject")
        for name in ("train", "validation", "test"):
            expected_per_label = 12 if name == "train" else 4
            self.assertEqual(
                manifest["splits"][name]["label_counts"],
                {"AD": expected_per_label, "HC": expected_per_label},
            )

    def test_flat_data_loading_collation_and_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project = Path(temporary_directory)
            data_directory = project / "data" / "eeg" / "toy"
            model_directory = project / "model_design"
            config_directory = project / "configs"
            framework_directory = project / "eeg_training"
            for directory in (
                data_directory,
                model_directory,
                config_directory,
                framework_directory,
            ):
                directory.mkdir(parents=True)

            subjects = []
            for index, (diagnosis, label) in enumerate((("HC", 0), ("AD", 1))):
                subject_id = f"{diagnosis}__AR__{index}"
                filename = f"{subject_id}.npy"
                np.save(
                    data_directory / filename,
                    np.ones((3, 2, 8), dtype=np.float32) * index,
                )
                subjects.append(
                    {
                        "subject_id": subject_id,
                        "labels": {
                            "diagnosis": diagnosis,
                            "diagnosis_id": label,
                            "institution": "AR",
                        },
                        "data_file": filename,
                        "array_shape": [3, 2, 8],
                        "sampling_frequency_hz": 128.0,
                        "channel_names": ["A", "B"],
                    }
                )
            metadata = {
                "dataset": "toy",
                "data_summary": {
                    "array_axis_order": ["segment", "channel", "time"]
                },
                "subjects": subjects,
            }
            metadata_file = data_directory / "dataset_description.json"
            metadata_file.write_text(json.dumps(metadata), encoding="utf-8")
            extra_json = data_directory / "nested" / "extra.json"
            extra_json.parent.mkdir()
            extra_json.write_text('{"ok": true}', encoding="utf-8")
            model_source = model_directory / "toy_model.py"
            model_source.write_text("VALUE = 1\n", encoding="utf-8")
            loaded_model_module = load_model_module(model_source)
            self.assertEqual(loaded_model_module.VALUE, 1)
            self.assertFalse((model_directory / "__pycache__").exists())
            config_file = config_directory / "toy.json"
            config_file.write_text('{"name": "toy"}', encoding="utf-8")
            training_source = framework_directory / "runner.py"
            training_source.write_text("# toy\n", encoding="utf-8")

            info = load_dataset_info(data_directory, metadata_file)
            self.assertEqual(info.n_channels, 2)
            dataset = IndependentSegmentDataset(
                info.records, training=False, seed=1
            )
            self.assertEqual(len(dataset), 6)
            batch = collate_independent_segments([dataset[0], dataset[1]])
            self.assertEqual(tuple(batch["eeg"].shape), (2, 2, 8))
            self.assertNotIn("segment_mask", batch)
            self.assertNotIn("bag_weights", batch)

            paths = create_experiment(
                project_root=project,
                experiment_name="toy",
                original_config=config_file,
                resolved_config={"name": "toy"},
                data_directory=data_directory,
                model_source=model_source,
                training_sources=[training_source],
            )
            self.assertTrue(paths.model_snapshot.is_file())
            self.assertTrue(
                (
                    paths.root
                    / "snapshots"
                    / "data_json"
                    / "dataset_description.json"
                ).is_file()
            )
            self.assertTrue(
                (
                    paths.root / "snapshots" / "data_json" / "nested" / "extra.json"
                ).is_file()
            )
            manifest = json.loads(
                (paths.root / "snapshots" / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            categories = {entry["category"] for entry in manifest}
            self.assertIn("model_source", categories)
            self.assertIn("preprocessing_json", categories)

    def test_independent_dataset_full_coverage_and_epoch_permutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            records = [
                _record(
                    directory,
                    0,
                    0,
                    5,
                    np.arange(5 * 2 * 8, dtype=np.float32).reshape(5, 2, 8),
                ),
                _record(
                    directory,
                    1,
                    1,
                    9,
                    np.arange(9 * 2 * 8, dtype=np.float32).reshape(9, 2, 8),
                ),
            ]
            dataset = IndependentSegmentDataset(records, training=True, seed=11)
            dataset.set_epoch(1)
            first_report = dataset.coverage_report()
            first_pairs = {
                (dataset[index]["subject_id"], dataset[index]["segment_index"])
                for index in range(len(dataset))
            }
            self.assertEqual(len(dataset), 14)
            self.assertEqual(len(first_pairs), 14)
            self.assertEqual(first_report["segments_used"], 14)
            self.assertEqual(first_report["unique_segments"], 14)
            self.assertEqual(first_report["duplicate_segments"], 0)
            self.assertEqual(first_report["missing_segments"], 0)
            self.assertEqual(first_report["coverage_ratio"], 1.0)
            self.assertFalse(first_report["subject_grouped_batches"])
            dataset.set_epoch(2)
            self.assertNotEqual(
                first_report["assignment_sha256"],
                dataset.coverage_report()["assignment_sha256"],
            )
            second_pairs = {
                (dataset[index]["subject_id"], dataset[index]["segment_index"])
                for index in range(len(dataset))
            }
            self.assertEqual(first_pairs, second_pairs)

            weights = segment_class_weights(records)
            self.assertAlmostEqual(float(weights[0]), 14 / 10)
            self.assertAlmostEqual(float(weights[1]), 14 / 18)

    def test_majority_vote_and_logit_mean_are_distinct(self) -> None:
        # Two weak AD votes beat one strong HC vote under majority voting, while
        # averaging logits produces HC. This verifies the two requested outputs.
        rows = []
        for index, (logit_hc, logit_ad) in enumerate(
            ((0.0, 0.2), (0.0, 0.2), (5.0, 0.0))
        ):
            probability_ad = float(
                torch.softmax(torch.tensor([logit_hc, logit_ad]), dim=0)[1]
            )
            predicted_label = int(probability_ad >= 0.5)
            rows.append(
                {
                    "subject_id": "S1",
                    "institution": "AR",
                    "true_label": 1,
                    "true_diagnosis": "AD",
                    "segment_index": index,
                    "segments_available": 3,
                    "predicted_label": predicted_label,
                    "probability_ad": probability_ad,
                    "logit_hc": logit_hc,
                    "logit_ad": logit_ad,
                }
            )
        majority = aggregate_subject_majority_vote(rows)[0]
        logit_mean = aggregate_subject_logit_mean(rows)[0]
        self.assertEqual(majority["predicted_label"], 1)
        self.assertEqual(majority["ad_votes"], 2)
        self.assertEqual(logit_mean["predicted_label"], 0)
        self.assertAlmostEqual(logit_mean["logit_hc"], 5 / 3)
        self.assertAlmostEqual(logit_mean["logit_ad"], 0.4 / 3)

    def test_tiny_training_and_evaluation_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            records = [
                _record(directory, index, label, 2)
                for index, label in enumerate((0, 0, 1, 1))
            ]
            dataset, loader = make_segment_loader(
                records,
                training=True,
                seed=1,
                batch_size=4,
                num_workers=0,
                pin_memory=False,
            )
            dataset.set_epoch(1)
            model = _TinySegmentModel()
            criterion = _TinySegmentCriterion()
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            device = torch.device("cpu")
            class_weight = torch.ones(2)
            model_module = SimpleNamespace(compute_neurophysiological_targets=None)
            training_result = run_epoch(
                model,
                criterion,
                loader,
                device,
                model_module,
                class_weight,
                amp_enabled=False,
                amp_dtype="float16",
                optimizer=optimizer,
                scaler=make_grad_scaler(device, enabled=False),
                max_grad_norm=1.0,
                gradient_accumulation_steps=2,
            )
            self.assertEqual(len(training_result.segment_predictions), 8)
            self.assertEqual(len(training_result.subject_majority_predictions), 4)
            self.assertEqual(len(training_result.subject_logit_mean_predictions), 4)
            self.assertTrue(np.isfinite(training_result.metrics["loss"]))
            self.assertEqual(training_result.metrics["n_optimizer_steps"], 1)
            evaluation_result = run_epoch(
                model,
                criterion,
                loader,
                device,
                model_module,
                class_weight,
                amp_enabled=False,
                amp_dtype="float16",
            )
            self.assertEqual(evaluation_result.metrics["segment_accuracy"], 1.0)
            self.assertEqual(
                evaluation_result.metrics["subject_majority_accuracy"], 1.0
            )
            self.assertEqual(
                evaluation_result.metrics["subject_logit_mean_accuracy"], 1.0
            )
            self.assertEqual(evaluation_result.metrics["n_segments"], 8)
            self.assertEqual(
                evaluation_result.metrics["coverage_coverage_ratio"], 1.0
            )

    def test_physiology_loss_schedule(self) -> None:
        schedule = {
            "classification_warmup_epochs": 5,
            "physiology_ramp_epochs": 5,
            "target_physiology_weight": 0.1,
        }
        self.assertEqual(_physiology_weight(1, schedule), 0.0)
        self.assertEqual(_physiology_weight(5, schedule), 0.0)
        self.assertAlmostEqual(_physiology_weight(6, schedule), 0.02)
        self.assertAlmostEqual(_physiology_weight(10, schedule), 0.1)
        self.assertAlmostEqual(_physiology_weight(20, schedule), 0.1)

    def test_end_to_end_independent_segment_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project = Path(temporary_directory)
            data_directory = project / "data" / "toy"
            model_directory = project / "model_design"
            config_directory = project / "configs"
            for directory in (data_directory, model_directory, config_directory):
                directory.mkdir(parents=True)

            subjects = []
            for diagnosis, label in (("HC", 0), ("AD", 1)):
                for institution in ("AR", "CL"):
                    for replicate in range(3):
                        subject_id = f"{diagnosis}__{institution}__{replicate}"
                        data_file = f"{subject_id}.npy"
                        value = -1.0 if label == 0 else 1.0
                        np.save(
                            data_directory / data_file,
                            np.full((3, 2, 8), value, dtype=np.float32),
                        )
                        subjects.append(
                            {
                                "subject_id": subject_id,
                                "labels": {
                                    "diagnosis": diagnosis,
                                    "diagnosis_id": label,
                                    "institution": institution,
                                },
                                "data_file": data_file,
                                "array_shape": [3, 2, 8],
                                "sampling_frequency_hz": 128.0,
                                "channel_names": ["A", "B"],
                            }
                        )
            metadata = {
                "dataset": "toy",
                "data_summary": {
                    "array_axis_order": ["segment", "channel", "time"]
                },
                "subjects": subjects,
            }
            (data_directory / "dataset_description.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            model_source = model_directory / "toy_model.py"
            model_source.write_text(
                '''
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F

@dataclass(frozen=True)
class EEGModelConfig:
    n_channels: int
    sampling_rate: float
    bands: tuple = ()

def make_ring_edge_index(n_channels, hops=1):
    del hops
    return torch.tensor([[0], [1]], dtype=torch.long)

class EEGSegmentClassifier(nn.Module):
    def __init__(self, config, edge_index, edge_weight=None):
        super().__init__()
        del edge_index, edge_weight
        self.config = config
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, eeg):
        score = self.scale * eeg.mean(dim=(1, 2))
        logits = torch.stack((-score, score), dim=-1)
        return {
            "segment_logits": logits,
            "learned_quality_scores": torch.full_like(score, 0.5),
        }

class EEGMultiTaskLoss(nn.Module):
    def __init__(self, segment_weight=1.0, physiology_weight=0.0,
                 quality_weight=0.0):
        super().__init__()
        self.segment_weight = segment_weight
        self.physiology_weight = physiology_weight
        self.quality_weight = quality_weight

    def forward(self, outputs, labels, physiology_targets=None,
                quality_targets=None, class_weight=None):
        del physiology_targets, quality_targets
        loss = F.cross_entropy(outputs["segment_logits"], labels,
                               weight=class_weight)
        zero = loss.detach() * 0.0
        return {
            "loss": loss,
            "segment_loss": loss,
            "physiology_loss": zero,
            "quality_loss": zero,
        }
'''.strip()
                + "\n",
                encoding="utf-8",
            )
            config = {
                "schema_version": "3.0",
                "experiment": {
                    "name": "toy_segments",
                    "device": "cpu",
                    "amp": False,
                },
                "data": {
                    "directory": str(data_directory),
                    "metadata_file": "dataset_description.json",
                    "train_fraction": 1 / 3,
                    "validation_fraction": 1 / 3,
                    "test_fraction": 1 / 3,
                    "split_seed_start": 3100,
                    "split_seed_count": 2,
                    "batch_size": 4,
                    "eval_batch_size": 4,
                    "num_workers": 0,
                    "pin_memory": False,
                },
                "model": {
                    "source_file": str(model_source),
                    "class_name": "EEGSegmentClassifier",
                    "parameters": {},
                    "graph": {"type": "ring", "hops": 1},
                },
                "loss": {
                    "segment_weight": 1.0,
                    "physiology_weight": 0.0,
                    "quality_weight": 0.0,
                },
                "training": {
                    "epochs": 2,
                    "gradient_accumulation_steps": 1,
                    "loss_schedule": {
                        "classification_warmup_epochs": 0,
                        "physiology_ramp_epochs": 1,
                        "target_physiology_weight": 0.0,
                    },
                    "early_stopping": {
                        "monitor": "subject_logit_mean_balanced_accuracy",
                        "mode": "max",
                        "tiebreaker": "segment_loss",
                        "start_epoch": 1,
                        "patience": 2,
                        "min_delta": 0.0,
                    },
                },
            }
            config_file = config_directory / "toy.json"
            config_file.write_text(json.dumps(config), encoding="utf-8")
            training_source = project / "frozen_training_source.py"
            training_source.write_text("ORIGINAL = True\n", encoding="utf-8")
            original_train = runner_module._train
            train_calls = 0

            def train_then_mutate_live_inputs(prepared, paths, logger):
                nonlocal train_calls
                result = original_train(prepared, paths, logger)
                train_calls += 1
                if train_calls == 1:
                    config_file.write_text("{broken json", encoding="utf-8")
                    model_source.write_text(
                        "raise RuntimeError('live model source was reused')\n",
                        encoding="utf-8",
                    )
                    training_source.write_text("ORIGINAL = False\n", encoding="utf-8")
                return result

            with (
                patch.object(
                    runner_module,
                    "_training_source_files",
                    return_value=[training_source],
                ),
                patch.object(
                    runner_module,
                    "_train",
                    side_effect=train_then_mutate_live_inputs,
                ),
            ):
                summary = run_experiment(project, config_file)
            aggregate_directory = Path(summary["run_directory"])
            self.assertIn(summary["status"], {"completed", "completed_with_warning"})
            self.assertEqual(summary["split_seed_start"], 3100)
            self.assertEqual(summary["split_seed_count"], 2)
            self.assertEqual(summary["split_seeds"], [3100, 3101])
            self.assertEqual(train_calls, 2)
            self.assertTrue(
                (aggregate_directory / "metrics" / "aggregate_final_metrics.json").is_file()
            )

            prediction_files = [
                "validation_segments.csv",
                "validation_subject_majority_vote.csv",
                "validation_subject_logit_mean.csv",
                "test_segments.csv",
                "test_subject_majority_vote.csv",
                "test_subject_logit_mean.csv",
            ]
            split_subjects = []
            for expected_seed, split_run in zip(
                summary["split_seeds"], summary["split_runs"]
            ):
                run_directory = Path(split_run["run_directory"])
                child_summary = json.loads(
                    (run_directory / "summary.json").read_text(encoding="utf-8")
                )
                self.assertEqual(child_summary["split_seed"], expected_seed)
                self.assertEqual(
                    child_summary["data"]["training_unit"], "independent_segment"
                )
                self.assertEqual(
                    child_summary["data"]["train_segments_per_epoch"], 12
                )
                self.assertTrue((run_directory / "checkpoints" / "best.pt").is_file())
                self.assertTrue(
                    (run_directory / "metrics" / "coverage.jsonl").is_file()
                )
                for filename in prediction_files:
                    self.assertTrue(
                        (run_directory / "predictions" / filename).is_file()
                    )
                self.assertIn(
                    "class EEGSegmentClassifier",
                    (
                        run_directory / "snapshots" / "model_source" / "toy_model.py"
                    ).read_text(encoding="utf-8"),
                )
                self.assertEqual(
                    (
                        run_directory
                        / "snapshots"
                        / "training_code"
                        / training_source.name
                    ).read_text(encoding="utf-8"),
                    "ORIGINAL = True\n",
                )
                split_manifest_value = json.loads(
                    (run_directory / "splits.json").read_text(encoding="utf-8")
                )
                self.assertEqual(split_manifest_value["seed"], expected_seed)
                split_subjects.append(
                    tuple(
                        subject["subject_id"]
                        for subject in split_manifest_value["splits"]["train"][
                            "subjects"
                        ]
                    )
                )
            self.assertNotEqual(split_subjects[0], split_subjects[1])

            run_directory = Path(summary["split_runs"][0]["run_directory"])
            with (
                run_directory / "predictions" / "test_segments.csv"
            ).open(encoding="utf-8", newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 12)
            with (
                run_directory / "predictions" / "test_subject_majority_vote.csv"
            ).open(encoding="utf-8", newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 4)

            coverage_lines = (
                run_directory / "metrics" / "coverage.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            self.assertGreaterEqual(len(coverage_lines), 6)
            self.assertTrue(
                all(json.loads(line)["coverage_ratio"] == 1.0 for line in coverage_lines)
            )
            final_metrics = summary["final_metrics"]
            self.assertEqual(
                set(final_metrics["thresholds_selected_on_validation"]),
                {"segments", "subject_majority_vote", "subject_logit_mean"},
            )
            aggregate_balanced_accuracy = final_metrics["test"]["segment"][
                "balanced_accuracy"
            ]
            self.assertEqual(aggregate_balanced_accuracy["n"], 2)
            self.assertEqual(aggregate_balanced_accuracy["mean"], 1.0)
            self.assertEqual(aggregate_balanced_accuracy["std"], 0.0)


if __name__ == "__main__":
    unittest.main()
