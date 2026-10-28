"""
run_pipeline.py
----------------
Runs the whole thing end to end:

    1. catalog.py    -> download + clean the KOI label table (Stage D)
    2. dataset.py     -> per-star download/clean (Stage A/B) + per-candidate
                         views (Stage B#6, Stage C) -> data/transit_dataset.npz

This just calls dataset.main(), which itself calls catalog.load_or_fetch_catalog()
first - it's a single entry point so you don't have to remember the order.

    python run_pipeline.py                    # full run, budget-capped at
                                               # config.MAX_DOWNLOAD_BYTES (12GB)
    python run_pipeline.py --max-targets 300  # small test run first (recommended)
"""

import dataset

if __name__ == "__main__":
    dataset.main()
