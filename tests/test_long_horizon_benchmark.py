from __future__ import annotations

import argparse

from tests.long_horizon_benchmark import build_report, render_markdown


def test_long_horizon_benchmark_smoke():
    args = argparse.Namespace(
        seed_start=7,
        seed_count=1,
        horizon_days=3,
        daily_pair_count=4,
        extra_noise_count=2,
        review_count=2,
        anchor_update_day=2,
    )

    report = build_report(args)

    assert report["benchmark"] == "scm_long_horizon_memory"
    assert report["status"]["overall_pass"] is True
    assert report["modes"]["sleep_enabled"]["final_disambiguation_recall"] >= report["modes"]["awake_only"]["final_disambiguation_recall"]
    assert report["modes"]["sleep_enabled"]["final_noise_retention"] <= report["modes"]["awake_only"]["final_noise_retention"]

    markdown = render_markdown(report)
    assert "# SCM Long-Horizon Memory" in markdown
    assert "## Summary" in markdown
    assert "## Day-by-Day Curve" in markdown
