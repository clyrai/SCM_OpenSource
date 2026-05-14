"""Smoke tests for the SCM baseline comparison report."""

from __future__ import annotations

import argparse

from tests.baseline_comparison import build_report, render_markdown


def test_baseline_comparison_report_smoke():
    args = argparse.Namespace(
        seed_start=7001,
        seed_count=2,
        pair_count=12,
        phase6_pair_count=12,
        key_count=6,
        noise_count=12,
    )

    report = build_report(args)

    assert report["benchmark"] == "scm_baseline_comparison"
    assert report["status"]["overall_pass"] is True

    methods = {row["method"] for row in report["behavioral_comparison"]["rows"]}
    assert "Lexical retrieval baseline" in methods
    assert "Vector retrieval baseline" in methods
    assert "SCM + DeepSleep" in methods
    assert len(report["reference_systems"]["rows"]) >= 5

    markdown = render_markdown(report)
    assert "SCM Baseline Comparison" in markdown
    assert "Human-Memory Suite" in markdown
