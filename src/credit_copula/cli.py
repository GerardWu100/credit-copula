"""Command-line entrypoint for the notebook-derived pipeline."""

from __future__ import annotations

import argparse
from pprint import pprint

from credit_copula.pipeline import run_pipeline

# Smoke runs keep full logic but use fewer paths for a quicker sanity check.
SMOKE_OVERRIDES = {"n_simulations": 50_000}


def main() -> None:
    """Run the project pipeline from the command line."""
    parser = argparse.ArgumentParser(description="Run the notebook-derived pipeline.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run with smaller verification settings when supported.",
    )
    parser.add_argument(
        "--print-keys",
        action="store_true",
        help="Print the final context keys after execution.",
    )
    args = parser.parse_args()

    overrides = dict(SMOKE_OVERRIDES) if args.smoke else {}
    context = run_pipeline(context_overrides=overrides)

    if args.print_keys:
        pprint(sorted(context.keys()))


if __name__ == "__main__":
    main()
