#!/usr/bin/env python3
"""Summarize pass^1 scores for retail, airline, and telecom text runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from tau2.data_model.simulation import Results
from tau2.metrics.agent_metrics import compute_metrics


def load_pass1(path: Path) -> float:
    results = Results.load(path)
    metrics = compute_metrics(results)
    score = metrics.pass_hat_ks.get(1)
    if score is None:
        raise ValueError(f"pass^1 missing in {path}")
    return score * 100


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-prefix",
        required=True,
        help="Prefix used by scripts/run_text_core_benchmark.sh",
    )
    parser.add_argument(
        "--sim-dir",
        default="data/simulations",
        help="Directory containing simulation outputs",
    )
    args = parser.parse_args()

    sim_dir = Path(args.sim_dir)
    scores: dict[str, float] = {}
    for domain in ("retail", "airline", "telecom"):
        result_path = sim_dir / f"{args.run_prefix}_{domain}" / "results.json"
        scores[domain] = load_pass1(result_path)

    avg = sum(scores.values()) / len(scores)

    print(f"Retail (%):  {scores['retail']:.2f}")
    print(f"Airline (%): {scores['airline']:.2f}")
    print(f"Telecom (%): {scores['telecom']:.2f}")
    print(f"Avg. (%):    {avg:.2f}")


if __name__ == "__main__":
    main()
