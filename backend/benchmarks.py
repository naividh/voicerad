"""
VoiceRad Clinical Benchmarks
Real clinical validation against CheXpert/NIH-style labels.

Measures:
- Per-pathology detection accuracy (sensitivity/specificity)
- Confidence calibration (are high-confidence outputs actually correct?)
- Safety rail effectiveness (are critical findings caught?)
- Latency per interpretation
- Triage accuracy (does the system correctly escalate?)

Usage:
    from benchmarks import BenchmarkRunner
    runner = BenchmarkRunner(model)
    results = runner.run_suite(test_cases)
    runner.print_report(results)
"""

import time
import json
import logging
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)


# -- CheXpert-style pathology labels -------------------------

CHEXPERT_PATHOLOGIES = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

# Mapping from pathology to keywords in model output
PATHOLOGY_KEYWORDS = {
    "No Finding": [
        "no acute", "normal", "unremarkable", "no significant",
        "no abnormality", "within normal limits",
    ],
    "Cardiomegaly": [
        "cardiomegaly", "enlarged heart", "cardiac enlargement",
        "heart size.{0,10}(enlarged|increased)",
    ],
    "Lung Opacity": [
        "opacity", "opacification", "opacities", "ground.glass",
        "hazy", "haziness",
    ],
    "Consolidation": [
        "consolidation", "consolidative", "air.?bronchogram",
        "dense opacity",
    ],
    "Pneumonia": [
        "pneumonia", "infectious", "infection",
        "consolidation.{0,30}(infect|pneumon)",
    ],
    "Atelectasis": [
        "atelectasis", "atelectatic", "volume loss",
        "collapse.{0,10}(lung|lobe)",
    ],
    "Pneumothorax": [
        "pneumothorax", "collapsed lung",
        "absence of.{0,10}lung marking",
    ],
    "Pleural Effusion": [
        "pleural effusion", "fluid.{0,10}pleural",
        "costophrenic.{0,10}blunt",
    ],
    "Edema": [
        "edema", "oedema", "pulmonary congestion",
        "cephalization", "kerley",
    ],
    "Fracture": [
        "fracture", "fractured", "broken",
        "discontinuity.{0,10}(cortex|bone)",
    ],
    "Enlarged Cardiomediastinum": [
        "mediastinal widening", "wide mediastinum",
        "enlarged mediastinum",
    ],
    "Lung Lesion": [
        "lesion", "nodule", "mass", "tumor", "tumour",
        "neoplasm",
    ],
    "Support Devices": [
        "endotracheal", "central line", "pacemaker",
        "support device", "catheter", "tube",
    ],
    "Pleural Other": [
        "pleural thickening", "pleural plaque",
        "pleural calcification",
    ],
}


# -- Test case structure -------------------------------------

@dataclass
class BenchmarkCase:
    """A single benchmark test case."""
    case_id: str
    image_path: str = ""
    image_url: str = ""
    question: str = "Describe findings in this chest X-ray."
    ground_truth_labels: list = field(default_factory=list)
    expected_triage: str = "ROUTINE"
    description: str = ""


@dataclass
class BenchmarkResult:
    """Result of running a single benchmark case."""
    case_id: str
    ground_truth: list = field(default_factory=list)
    predicted_labels: list = field(default_factory=list)
    interpretation: str = ""
    confidence_score: float = 0.0
    triage_predicted: str = ""
    triage_expected: str = ""
    triage_correct: bool = False
    latency_ms: float = 0.0
    safety_blocked: bool = False
    critical_findings: list = field(default_factory=list)
    true_positives: list = field(default_factory=list)
    false_positives: list = field(default_factory=list)
    false_negatives: list = field(default_factory=list)


@dataclass
class BenchmarkSummary:
    """Aggregate benchmark results."""
    total_cases: int = 0
    mean_latency_ms: float = 0.0
    median_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0

    # Per-pathology metrics
    pathology_metrics: dict = field(default_factory=dict)

    # Overall metrics
    macro_sensitivity: float = 0.0
    macro_specificity: float = 0.0
    macro_f1: float = 0.0

    # Triage accuracy
    triage_accuracy: float = 0.0
    triage_over_escalation_rate: float = 0.0
    triage_under_escalation_rate: float = 0.0

    # Safety rail metrics
    safety_block_rate: float = 0.0
    critical_detection_rate: float = 0.0

    # Confidence calibration
    mean_confidence: float = 0.0
    confidence_when_correct: float = 0.0
    confidence_when_wrong: float = 0.0

    def to_dict(self) -> dict:
        return {
            "total_cases": self.total_cases,
            "latency": {
                "mean_ms": round(self.mean_latency_ms, 1),
                "median_ms": round(self.median_latency_ms, 1),
                "p95_ms": round(self.p95_latency_ms, 1),
            },
            "detection": {
                "macro_sensitivity": round(self.macro_sensitivity, 4),
                "macro_specificity": round(self.macro_specificity, 4),
                "macro_f1": round(self.macro_f1, 4),
                "per_pathology": {
                    k: {kk: round(vv, 4) for kk, vv in v.items()}
                    for k, v in self.pathology_metrics.items()
                },
            },
            "triage": {
                "accuracy": round(self.triage_accuracy, 4),
                "over_escalation_rate": round(self.triage_over_escalation_rate, 4),
                "under_escalation_rate": round(self.triage_under_escalation_rate, 4),
            },
            "safety": {
                "block_rate": round(self.safety_block_rate, 4),
                "critical_detection_rate": round(self.critical_detection_rate, 4),
            },
            "confidence": {
                "mean": round(self.mean_confidence, 4),
                "when_correct": round(self.confidence_when_correct, 4),
                "when_wrong": round(self.confidence_when_wrong, 4),
            },
        }


# -- Benchmark Runner ----------------------------------------

class BenchmarkRunner:
    """Runs clinical validation benchmarks against MedGemma."""

    def __init__(self, model=None, safety_engine=None):
        self.model = model
        self.safety_engine = safety_engine

    def run_single(self, case: BenchmarkCase) -> BenchmarkResult:
        """Run a single benchmark case."""
        import re

        result = BenchmarkResult(
            case_id=case.case_id,
            ground_truth=case.ground_truth_labels,
            triage_expected=case.expected_triage,
        )

        # Run model inference with timing
        start = time.perf_counter()
        try:
            if self.model and case.image_path:
                from PIL import Image
                img = Image.open(case.image_path).convert("RGB")
                result.interpretation = self.model.interpret(
                    img, case.question
                )
            elif self.model and case.image_url:
                # For remote images
                result.interpretation = "(remote image not loaded)"
            else:
                result.interpretation = "(no model or image available)"
        except Exception as exc:
            result.interpretation = f"(inference error: {exc})"
            logger.error("Benchmark case %s failed: %s", case.case_id, exc)

        result.latency_ms = (time.perf_counter() - start) * 1000

        # Run safety assessment
        if self.safety_engine:
            safety = self.safety_engine.assess(result.interpretation)
            result.confidence_score = safety.confidence_score
            result.triage_predicted = safety.triage_level.value
            result.safety_blocked = safety.is_blocked
            result.critical_findings = safety.critical_findings

        # Extract predicted pathologies from interpretation text
        result.predicted_labels = self._extract_pathologies(
            result.interpretation
        )

        # Calculate per-case TP/FP/FN
        gt_set = set(case.ground_truth_labels)
        pred_set = set(result.predicted_labels)
        result.true_positives = list(gt_set & pred_set)
        result.false_positives = list(pred_set - gt_set)
        result.false_negatives = list(gt_set - pred_set)

        # Triage accuracy
        result.triage_correct = (
            result.triage_predicted == case.expected_triage
        )

        return result

    def run_suite(self, cases: list) -> BenchmarkSummary:
        """Run a full benchmark suite and compute aggregate metrics."""
        results = [self.run_single(c) for c in cases]
        return self._compute_summary(results)

    def _extract_pathologies(self, text: str) -> list:
        """Extract CheXpert-style pathology labels from free text."""
        import re
        detected = []
        for pathology, keywords in PATHOLOGY_KEYWORDS.items():
            for kw in keywords:
                if re.search(kw, text, re.IGNORECASE):
                    detected.append(pathology)
                    break
        return detected

    def _compute_summary(self, results: list) -> BenchmarkSummary:
        """Compute aggregate metrics from individual results."""
        summary = BenchmarkSummary()
        summary.total_cases = len(results)

        if not results:
            return summary

        # Latency
        latencies = [r.latency_ms for r in results]
        latencies.sort()
        summary.mean_latency_ms = sum(latencies) / len(latencies)
        summary.median_latency_ms = latencies[len(latencies) // 2]
        summary.p95_latency_ms = latencies[int(len(latencies) * 0.95)]

        # Per-pathology metrics
        all_pathologies = set()
        for r in results:
            all_pathologies.update(r.ground_truth)
            all_pathologies.update(r.predicted_labels)

        sensitivities = []
        specificities = []
        f1_scores = []

        for pathology in all_pathologies:
            tp = sum(1 for r in results if pathology in r.true_positives)
            fp = sum(1 for r in results if pathology in r.false_positives)
            fn = sum(1 for r in results if pathology in r.false_negatives)
            tn = len(results) - tp - fp - fn

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            f1 = (
                2 * precision * sensitivity / (precision + sensitivity)
                if (precision + sensitivity) > 0 else 0.0
            )

            summary.pathology_metrics[pathology] = {
                "sensitivity": sensitivity,
                "specificity": specificity,
                "precision": precision,
                "f1": f1,
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            }
            sensitivities.append(sensitivity)
            specificities.append(specificity)
            f1_scores.append(f1)

        summary.macro_sensitivity = (
            sum(sensitivities) / len(sensitivities) if sensitivities else 0.0
        )
        summary.macro_specificity = (
            sum(specificities) / len(specificities) if specificities else 0.0
        )
        summary.macro_f1 = (
            sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        )

        # Triage metrics
        triage_correct = sum(1 for r in results if r.triage_correct)
        summary.triage_accuracy = triage_correct / len(results)

        triage_levels = {"NORMAL": 0, "ROUTINE": 1, "URGENT": 2, "CRITICAL": 3}
        over_escalated = sum(
            1 for r in results
            if triage_levels.get(r.triage_predicted, 0) > triage_levels.get(r.triage_expected, 0)
        )
        under_escalated = sum(
            1 for r in results
            if triage_levels.get(r.triage_predicted, 0) < triage_levels.get(r.triage_expected, 0)
        )
        summary.triage_over_escalation_rate = over_escalated / len(results)
        summary.triage_under_escalation_rate = under_escalated / len(results)

        # Safety metrics
        summary.safety_block_rate = (
            sum(1 for r in results if r.safety_blocked) / len(results)
        )

        critical_cases = [r for r in results if "CRITICAL" in r.triage_expected]
        if critical_cases:
            summary.critical_detection_rate = (
                sum(1 for r in critical_cases if r.critical_findings)
                / len(critical_cases)
            )

        # Confidence calibration
        confidences = [r.confidence_score for r in results]
        summary.mean_confidence = sum(confidences) / len(confidences)

        correct = [r for r in results if not r.false_negatives and not r.false_positives]
        wrong = [r for r in results if r.false_negatives or r.false_positives]

        if correct:
            summary.confidence_when_correct = (
                sum(r.confidence_score for r in correct) / len(correct)
            )
        if wrong:
            summary.confidence_when_wrong = (
                sum(r.confidence_score for r in wrong) / len(wrong)
            )

        return summary

    @staticmethod
    def load_cases_from_json(path: str) -> list:
        """Load benchmark cases from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        return [
            BenchmarkCase(**case_data)
            for case_data in data.get("cases", data)
        ]

    @staticmethod
    def print_report(summary: BenchmarkSummary):
        """Print a human-readable benchmark report."""
        d = summary.to_dict()
        print("=" * 60)
        print("VOICERAD CLINICAL BENCHMARK REPORT")
        print("=" * 60)
        print(f"Total cases:  {d['total_cases']}")
        print(f"Mean latency: {d['latency']['mean_ms']:.0f} ms")
        print(f"P95 latency:  {d['latency']['p95_ms']:.0f} ms")
        print()
        print("-- Detection Metrics --")
        print(f"Macro Sensitivity: {d['detection']['macro_sensitivity']:.3f}")
        print(f"Macro Specificity: {d['detection']['macro_specificity']:.3f}")
        print(f"Macro F1:          {d['detection']['macro_f1']:.3f}")
        print()
        print("-- Triage --")
        print(f"Accuracy:            {d['triage']['accuracy']:.3f}")
        print(f"Over-escalation:     {d['triage']['over_escalation_rate']:.3f}")
        print(f"Under-escalation:    {d['triage']['under_escalation_rate']:.3f}")
        print()
        print("-- Safety --")
        print(f"Block rate:          {d['safety']['block_rate']:.3f}")
        print(f"Critical detection:  {d['safety']['critical_detection_rate']:.3f}")
        print()
        print("-- Confidence Calibration --")
        print(f"Mean confidence:     {d['confidence']['mean']:.3f}")
        print(f"When correct:        {d['confidence']['when_correct']:.3f}")
        print(f"When wrong:          {d['confidence']['when_wrong']:.3f}")
        print()
        if d['detection']['per_pathology']:
            print("-- Per-Pathology Breakdown --")
            for path, metrics in sorted(d['detection']['per_pathology'].items()):
                print(f"  {path:30s} sens={metrics['sensitivity']:.3f} "
                      f"spec={metrics['specificity']:.3f} "
                      f"f1={metrics['f1']:.3f}")
        print("=" * 60)
