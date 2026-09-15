"""
Phase 1 - Primary Dataset Inspection
Purpose: Understand actual dataset structure before any modeling decisions.
All outputs are OBSERVED facts only.
"""

import pandas as pd
import numpy as np
import os
import sys

# ── CONFIG ──────────────────────────────────────────────────────────────────
# UPDATE THIS PATH to wherever you downloaded the dataset
DATA_PATH = r"C:\Users\abdul\Downloads\archive extract\twcs\twcs.csv"   # change if filename differs
SAMPLE_ROWS = 5
# ────────────────────────────────────────────────────────────────────────────


def section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def inspect(path):

    # ── 1. FILE EXISTS? ──────────────────────────────────────────
    section("1. FILE CHECK")
    if not os.path.exists(path):
        print(f"FILE NOT FOUND: {path}")
        print("Update DATA_PATH in this script to the correct location.")
        sys.exit(1)

    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"File found: {path}")
    print(f"File size: {size_mb:.1f} MB")

    # ── 2. LOAD ──────────────────────────────────────────────────
    section("2. LOADING (first 100k rows for speed)")
    df = pd.read_csv(path, nrows=100_000, low_memory=False)
    print(f"Rows loaded: {len(df):,}")
    print(f"Columns: {list(df.columns)}")

    # ── 3. DTYPES ────────────────────────────────────────────────
    section("3. COLUMN TYPES")
    print(df.dtypes)

    # ── 4. SAMPLE ROWS ───────────────────────────────────────────
    section("4. FIRST 5 ROWS (raw)")
    pd.set_option('display.max_colwidth', 80)
    pd.set_option('display.max_columns', None)
    print(df.head(SAMPLE_ROWS).to_string())

    # ── 5. MISSING VALUES ────────────────────────────────────────
    section("5. MISSING VALUES")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        'missing_count': missing,
        'missing_pct': missing_pct
    })
    print(missing_df[missing_df['missing_count'] > 0])
    if missing_df['missing_count'].sum() == 0:
        print("No missing values in first 100k rows.")

    # ── 6. UNIQUE VALUES PER COLUMN ──────────────────────────────
    section("6. UNIQUE VALUES PER COLUMN")
    for col in df.columns:
        n_unique = df[col].nunique()
        print(f"  {col}: {n_unique:,} unique values")

    # ── 7. BOOLEAN/FLAG COLUMNS ──────────────────────────────────
    section("7. VALUE COUNTS FOR SMALL-CARDINALITY COLUMNS")
    for col in df.columns:
        n_unique = df[col].nunique()
        if n_unique <= 10:
            print(f"\n  {col}:")
            print(df[col].value_counts(dropna=False).to_string())

    # ── 8. SAMPLE TEXT VALUES ────────────────────────────────────
    section("8. SAMPLE TEXT FROM LIKELY TEXT COLUMNS")
    for col in df.columns:
        if df[col].dtype == object:
            print(f"\n  Column: '{col}' — 3 random samples:")
            samples = df[col].dropna().sample(
                min(3, df[col].dropna().shape[0]),
                random_state=42
            )
            for val in samples:
                print(f"    → {str(val)[:120]}")

    # ── 9. AUTHOR/BRAND COLUMN CANDIDATE ─────────────────────────
    section("9. TOP 30 VALUES IN HIGH-CARDINALITY STRING COLUMNS")
    for col in df.columns:
        if df[col].dtype == object:
            n_unique = df[col].nunique()
            if 10 < n_unique < 100_000:
                print(f"\n  Top 30 in '{col}':")
                print(df[col].value_counts().head(30).to_string())

    # ── 10. NUMERIC COLUMN STATS ─────────────────────────────────
    section("10. NUMERIC COLUMN STATISTICS")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(df[numeric_cols].describe().to_string())
    else:
        print("No numeric columns found.")

    # ── 11. FULL ROW COUNT ───────────────────────────────────────
    section("11. FULL DATASET ROW COUNT (may take 1–2 minutes)")
    print("Counting all rows...")
    full_df = pd.read_csv(path, usecols=[0], low_memory=False)
    print(f"Total rows in full dataset: {len(full_df):,}")

    section("INSPECTION COMPLETE")
    print("Copy everything above and share it.")
    print("Do not interpret the output yourself — share it raw.")


if __name__ == "__main__":
    inspect(DATA_PATH)