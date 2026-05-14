"""Regression tests for Phase 6 human-memory behavior benchmarks."""

from __future__ import annotations

import argparse

from tests.human_memory_benchmark import (
    benchmark_contradiction_versioning,
    benchmark_one_shot_recall,
    benchmark_selective_forgetting,
    build_report,
)


def test_one_shot_recall_behavior():
    metrics = benchmark_one_shot_recall()
    assert metrics["accuracy"] >= 1.0
    assert metrics["name_hit"] is True
    assert metrics["location_hit"] is True


def test_selective_forgetting_behavior():
    metrics = benchmark_selective_forgetting(key_count=10, noise_count=20)
    assert metrics["key_retention"] >= 0.80
    assert metrics["noise_retention"] <= 0.35


def test_contradiction_versioning_behavior():
    metrics = benchmark_contradiction_versioning()
    assert metrics["accuracy"] >= 0.90
    assert metrics["chain_ok"] is True


def test_phase6_report_status_smoke():
    args = argparse.Namespace(
        seed=1234,
        pair_count=16,
        key_count=10,
        noise_count=20,
    )
    report = build_report(args)
    assert report["benchmark"] == "phase6_human_memory_behavior"
    assert "status" in report
    assert report["status"]["overall_pass"] is True
