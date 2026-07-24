#!/usr/bin/env python3
"""
add_calvados_feature.py

Takes the CSV produced by feature_comp.py (which must contain a 'protein_seq'
column) and adds a 'calvados_ah_pairs' column computed via CALVADOS's
SeqFeatures, writing the result to a new CSV.

Intended to run in a SEPARATE environment from feature_comp.py (CALVADOS pins
numpy==1.24, which conflicts with finches' numpy 2+ requirement used elsewhere
in the pipeline) — run this as a second pass and merge is automatic since it
just adds one column to the existing table.

Usage:
    python add_calvados_feature.py input.csv output.csv
"""

from __future__ import annotations

import os
import sys
import argparse
import importlib.resources
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from tqdm import tqdm

from calvados.sequence import SeqFeatures

def load_calvados_residues() -> pd.DataFrame:
    with importlib.resources.files("calvados.data").joinpath("residues.csv").open("r") as f:
        residues = pd.read_csv(f).set_index("one")
    return residues

def compute_AH_pairs(seq: str, residues: pd.DataFrame, charge_termini: bool = True) -> float:
    feats = SeqFeatures(seq, residues=residues, charge_termini=charge_termini)
    return float(feats.ah_ij)

_worker_residues = None
_worker_charge_termini = True


def _init_calvados_worker(residues: pd.DataFrame, charge_termini: bool):
    # Load residues once per worker process instead of once per sequence.
    global _worker_residues, _worker_charge_termini
    _worker_residues = residues
    _worker_charge_termini = charge_termini


def _compute_ah_pairs_one(seq: str):
    try:
        return compute_AH_pairs(seq, _worker_residues, charge_termini=_worker_charge_termini), None
    except Exception as e:
        return np.nan, str(e)


def compute_calvados_ah_pairs(sequences, charge_termini: bool = True, n_workers=None) -> list[float]:
    sequences = list(sequences)
    if not sequences:
        return []

    residues = load_calvados_residues()

    n_workers = n_workers or os.cpu_count() or 1
    n_workers = max(1, min(n_workers, len(sequences)))

    vals = []
    failed = []

    if n_workers == 1:
        _init_calvados_worker(residues, charge_termini)
        pairs = [_compute_ah_pairs_one(seq) for seq in tqdm(sequences, desc="calvados ah_pairs")]
    else:
        with ProcessPoolExecutor(
            max_workers=n_workers, initializer=_init_calvados_worker, initargs=(residues, charge_termini)
        ) as ex:
            pairs = list(
                tqdm(ex.map(_compute_ah_pairs_one, sequences), total=len(sequences), desc="calvados ah_pairs")
            )

    for i, (val, err) in enumerate(pairs):
        vals.append(val)
        if err is not None:
            print(f"calvados_ah_pairs failed on row {i}: {err}")
            failed.append(i)

    if failed:
        print(f"Done. {len(failed)} / {len(sequences)} sequences failed (set to NaN).")
    return vals

def main():
    parser = argparse.ArgumentParser(
        description="Add calvados_ah_pairs to a features CSV produced by feature_comp.py."
    )
    parser.add_argument("input_csv", help="Path to the features CSV from feature_comp.py")
    parser.add_argument("output_csv", help="Path to write the CSV with calvados_ah_pairs added")
    parser.add_argument("--seq-column", default="protein_seq",
                         help="Name of the sequence column in the input CSV (default: protein_seq)")
    parser.add_argument("--no-charge-termini", action="store_true",
                         help="Disable N/C-terminal charge assignment (default: enabled, matching feature_comp.py)")
    parser.add_argument("--workers", type=int, default=None,
                         help="Number of worker processes (default: os.cpu_count())")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv, low_memory=False)

    if args.seq_column not in df.columns:
        print(f"Error: column '{args.seq_column}' not found in {args.input_csv}. "
              f"Available columns: {list(df.columns)}")
        sys.exit(1)

    if "calvados_ah_pairs" in df.columns:
        print("Warning: 'calvados_ah_pairs' column already exists in the input CSV; it will be overwritten.")

    sequences = df[args.seq_column].astype(str).tolist()

    ah_pairs = compute_calvados_ah_pairs(
        sequences, charge_termini=not args.no_charge_termini, n_workers=args.workers
    )
    df["calvados_ah_pairs"] = ah_pairs

    df.to_csv(args.output_csv, index=False)
    print(f"Saved {len(df)} rows x {len(df.columns)} columns to {args.output_csv}")
    
if __name__ == "__main__":
    main()