# GUIDE_scripts.md

## Part 1: Conceptual Explanation

This folder holds thin command-line wrappers. Reusable pipeline logic lives in `src/credit_copula/`; scripts here only parse inputs and call the package entrypoints.

## Part 2: Code Reference

- `run_pipeline.py`: Thin wrapper that calls `credit_copula.cli.main()`.

## Part 3: Short Journal

- 2026-05-20: Moved the root `run_pipeline.py` CLI wrapper here to match the standard project layout.
