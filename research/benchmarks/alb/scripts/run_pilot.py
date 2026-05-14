"""
ALB v0.1 pilot runner — runs every available adapter against every available
persona under both idle conditions, scores all metrics, computes statistics
where there are enough seeds, and writes everything to results/.

Usage:
    python scripts/run_pilot.py [--seeds N]

Output:
    results/raw/<system>__<persona>__<seed>__<idle>.json   — one per run
    results/scored.csv                                      — flat table
    results/summary.md                                      — human-readable
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).parent.parent
sys.path.insert(0, str(HERE))

from adapters.scm_adapter import SCMAdapter
from runner import load_persona, run_persona, score_run
from stats import bootstrap_ci, paired_t_test, cohens_d_paired


SYSTEMS = [
    ("SCM", lambda: SCMAdapter(version_pin="phase7-pilot-v0.1")),
]


def collect_personas() -> List[Path]:
    persona_dir = HERE / "personas"
    return sorted(persona_dir.glob("persona_*.json"))


def write_run_artifact(scored, run, outpath: Path) -> None:
    payload = {
        "system_name": scored.system_name,
        "system_version": scored.system_version,
        "persona_id": scored.persona_id,
        "seed": scored.seed,
        "idle_on": scored.idle_on,
        "scores": {
            "pdr": scored.pdr,
            "css": scored.css,
            "cgc_id": scored.cgc_id,
            "cgc_fill": scored.cgc_fill,
            "crai_current": scored.crai_current,
            "crai_old": scored.crai_old,
            "wsi_precision": scored.wsi_precision,
            "wsi_recall": scored.wsi_recall,
            "wsi_f1": scored.wsi_f1,
            "imc_wall_seconds": scored.imc_wall_seconds,
            "imc_cpu_seconds": scored.imc_cpu_seconds,
            "imc_total_idle_periods": scored.imc_total_idle_periods,
        },
        "raw_subscores": scored.raw_subscores,
        "capabilities": run.capabilities,
        "wall_time_seconds": run.ended_at_wall - run.started_at_wall,
        "turns_ingested": run.turns_ingested,
        "schemas_count": len(run.final_schemas),
        "gaps_count": len(run.final_gaps),
    }
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with outpath.open("w") as f:
        json.dump(payload, f, indent=2, default=str)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=1,
                    help="number of seeds per (system, persona, idle_on) cell")
    args = ap.parse_args()

    results_dir = HERE / "results"
    raw_dir = results_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    personas = collect_personas()
    print(f"ALB v0.1 pilot — {len(personas)} personas, "
          f"{len(SYSTEMS)} systems, {args.seeds} seeds, "
          f"{2} idle conditions = {len(personas) * len(SYSTEMS) * args.seeds * 2} runs")

    rows: List[Dict[str, Any]] = []

    for persona_path in personas:
        persona = load_persona(persona_path)
        pid = persona["persona_id"]

        for sys_name, factory in SYSTEMS:
            for seed in range(args.seeds):
                for idle_on in [True, False]:
                    label = f"{sys_name} | {pid} | seed={seed} | idle_on={idle_on}"
                    print(f"\n→ {label}")
                    t0 = time.perf_counter()
                    try:
                        adapter = factory()
                        run = run_persona(persona, adapter, seed=seed, idle_on=idle_on)
                        scored = score_run(run, persona, adapter=adapter)
                    except Exception:
                        print(f"  CRASHED:")
                        traceback.print_exc()
                        continue

                    elapsed = time.perf_counter() - t0
                    print(f"  done in {elapsed:.1f}s")
                    print(f"  PDR={scored.pdr:.3f} CSS={scored.css:.3f} "
                          f"CGC_id={scored.cgc_id:.3f} CGC_fill={scored.cgc_fill:.3f} "
                          f"CRAI_cur={scored.crai_current:.3f} CRAI_old={scored.crai_old:.3f} "
                          f"WSI_F1={scored.wsi_f1:.3f}")

                    artifact_path = raw_dir / (
                        f"{sys_name}__{pid}__seed{seed}__idle_{'on' if idle_on else 'off'}.json"
                    )
                    write_run_artifact(scored, run, artifact_path)

                    rows.append({
                        "system": scored.system_name,
                        "version": scored.system_version,
                        "persona": scored.persona_id,
                        "seed": scored.seed,
                        "idle_on": scored.idle_on,
                        "pdr": scored.pdr,
                        "css": scored.css,
                        "cgc_id": scored.cgc_id,
                        "cgc_fill": scored.cgc_fill,
                        "crai_current": scored.crai_current,
                        "crai_old": scored.crai_old,
                        "wsi_f1": scored.wsi_f1,
                        "wsi_precision": scored.wsi_precision,
                        "wsi_recall": scored.wsi_recall,
                        "imc_wall_seconds": scored.imc_wall_seconds,
                        "wall_total_s": elapsed,
                    })

    # CSV
    if rows:
        csv_path = results_dir / "scored.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nwrote {csv_path} ({len(rows)} rows)")

    # Summary
    write_summary(rows, results_dir / "summary.md")
    return 0


def write_summary(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("# ALB pilot — no rows produced.\n")
        return

    metrics = ["pdr", "css", "cgc_id", "cgc_fill", "crai_current", "crai_old", "wsi_f1"]
    systems = sorted({r["system"] for r in rows})

    lines: List[str] = ["# ALB v0.1 pilot results\n"]

    for system in systems:
        lines.append(f"## {system}\n")
        lines.append("### idle_on = True\n")
        lines.append("| persona | " + " | ".join(metrics) + " |")
        lines.append("|---|" + "|".join(["---"] * len(metrics)) + "|")
        for r in rows:
            if r["system"] == system and r["idle_on"]:
                cells = [f"{r[m]:.3f}" for m in metrics]
                lines.append(f"| {r['persona']} | " + " | ".join(cells) + " |")
        lines.append("")

        lines.append("### idle_on = False (NIAL ablation)\n")
        lines.append("| persona | " + " | ".join(metrics) + " |")
        lines.append("|---|" + "|".join(["---"] * len(metrics)) + "|")
        for r in rows:
            if r["system"] == system and not r["idle_on"]:
                cells = [f"{r[m]:.3f}" for m in metrics]
                lines.append(f"| {r['persona']} | " + " | ".join(cells) + " |")
        lines.append("")

        # NIAL lift (paired)
        on_rows = [r for r in rows if r["system"] == system and r["idle_on"]]
        off_rows = [r for r in rows if r["system"] == system and not r["idle_on"]]
        if on_rows and off_rows:
            lines.append("### NIAL lift (idle_on - idle_off)\n")
            lines.append("| metric | mean lift | n |")
            lines.append("|---|---|---|")
            for m in metrics:
                pairs = []
                for on in on_rows:
                    for off in off_rows:
                        if on["persona"] == off["persona"] and on["seed"] == off["seed"]:
                            pairs.append((on[m], off[m]))
                            break
                if pairs:
                    diffs = [a - b for a, b in pairs]
                    mean_lift = sum(diffs) / len(diffs)
                    lines.append(f"| {m} | {mean_lift:+.3f} | {len(pairs)} |")
            lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    print(f"wrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())
