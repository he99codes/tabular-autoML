#!/usr/bin/env python3
"""
main.py - CLI entry point for the Tabular AutoML framework.

Usage:
    python main.py --data dataset.csv --target churn --task classification
    python main.py --data housing.csv --target price --task regression --time_budget 120
"""

import argparse
import sys
import os
import pandas as pd

# Allow running from root directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from automl import AutoML


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tabular AutoML Framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", required=True, help="Path to CSV dataset")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument(
        "--task",
        required=True,
        choices=["classification", "regression"],
        help="Task type",
    )
    parser.add_argument(
        "--time_budget",
        type=float,
        default=None,
        help="Training time budget in seconds (optional)",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=15,
        help="Number of Optuna HPO trials per model",
    )
    parser.add_argument(
        "--output_dir",
        default="./automl_output",
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--no_feature_engineering",
        action="store_true",
        help="Disable feature engineering",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    # Load data
    print(f"\nLoading dataset: {args.data}")
    try:
        df = pd.read_csv(args.data)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        sys.exit(1)

    if args.target not in df.columns:
        print(f"Error: target column '{args.target}' not found.")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    # Run AutoML
    automl = AutoML(
        task_type=args.task,
        time_budget=args.time_budget,
        n_optuna_trials=args.n_trials,
        output_dir=args.output_dir,
        seed=args.seed,
        use_feature_engineering=not args.no_feature_engineering,
    )

    automl.fit(df, target_col=args.target)
    automl.report()


if __name__ == "__main__":
    main()
