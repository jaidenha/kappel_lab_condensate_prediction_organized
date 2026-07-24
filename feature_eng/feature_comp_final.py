#!/usr/bin/env python3
"""
feature_comp.py

Reads protein sequences from a FASTA file and computes the full biophysical
feature set (CondenSeq composition/patterning features, external structural
predictors, handcrafted tier 2-5 features, metapredict disorder, 
writing the result to a CSV file.

Usage in terminal:
    python feature_comp.py fasta_file.fasta output.csv
"""

from __future__ import annotations

import os
import re
import sys
import shutil
import argparse
import subprocess
import tempfile
import importlib.resources
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# path setup
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_STANDALONE_DIR = os.path.join(_THIS_DIR, "standalone_files")
_PSLAB_DIR = os.path.join(_STANDALONE_DIR, "pslab")
for _path in (_PSLAB_DIR, _STANDALONE_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

HUMAN_IDR_SEQUENCES_CSV = os.path.join(_THIS_DIR, "standalone_files", "human_IDR_sequences.csv")

# third-party imports
from Bio import SeqIO
from localcider.sequenceParameters import SequenceParameters
from iupred3.iupred3_lib import iupred
import nardini
from sparrow.predictors import batch_predict
from sparrow.patterning import scd as _sp_scd
from finches import Mpipi_frontend
import metapredict as meta

segmasker_cmd = shutil.which("segmasker") or "segmasker"

try:
    from pslab_predict import predict_pslab_batch
except ImportError:
    predict_pslab_batch = None

# default values
DEFAULT_FINCHES_HETEROTYPIC_MAX_PARTNERS = 256
DEFAULT_FINCHES_HETEROTYPIC_SEED = 42
DEFAULT_NARDINI_SCRAMBLES = 200

# FASTA loading
def load_sequences_from_fasta(fasta_path: str) -> list[str]:
    seqs = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        seq = str(record.seq).replace("*", "").replace(" ", "").upper()
        if seq:
            seqs.append(seq)
    return seqs

# CondenSeq composition / patterning features
GROUPS = {
    "ILMV": list("ILMV"), "RK": list("RK"), "ED": list("ED"), "DE": list("DE"),
    "GS": list("GS"), "YFW": list("YFW"), "FYW": list("FYW"), "FWY": list("FWY"),
    "STNQCH": list("STNQCH"), "A": ["A"], "P": ["P"], "G": ["G"],
}

condenseq_feature_cols = [
    'fraction_A', 'fraction_C', 'fraction_D', 'fraction_E', 'fraction_F', 'fraction_G', 'fraction_H',
    'fraction_I', 'fraction_K', 'fraction_L', 'fraction_M', 'fraction_N', 'fraction_P', 'fraction_Q',
    'fraction_R', 'fraction_S', 'fraction_T', 'fraction_V', 'fraction_W', 'fraction_Y',
    'ratio_R_K', 'ratio_D_E', 'ratio_S_G', 'ratio_N_Q', 'ratio_Y_F', 'ratio_F_W', 'ratio_Y_W', 'ratio_R_Q', 'ratio_K_Q',
    'fraction_group_ILMV', 'fraction_group_RK', 'fraction_group_DE', 'fraction_group_GS', 'fraction_group_YFW',
    'ratio_group_FYW_ILV', 'ratio_group_FYW_R', 'NCPR', 'FCR', 'fraction_disorder_promoting', 'mean_hydropathy',
    'frac_disorder', 'omega_STNQCH', 'kappa_STNQCH_ILMV', 'kappa_STNQCH_RK', 'kappa_STNQCH_ED',
    'kappa_STNQCH_FWY', 'kappa_STNQCH_A', 'kappa_STNQCH_P', 'kappa_STNQCH_G', 'omega_ILMV', 'kappa_ILMV_RK',
    'kappa_ILMV_ED', 'kappa_ILMV_FWY', 'kappa_ILMV_A', 'kappa_ILMV_P', 'kappa_ILMV_G', 'omega_RK', 'kappa_RK_ED',
    'kappa_RK_FWY', 'kappa_RK_A', 'kappa_RK_P', 'kappa_RK_G', 'omega_ED', 'kappa_ED_FWY', 'kappa_ED_A',
    'kappa_ED_P', 'kappa_ED_G', 'omega_FWY', 'kappa_FWY_A', 'kappa_FWY_P', 'kappa_FWY_G', 'omega_A',
    'kappa_A_P', 'kappa_A_G', 'omega_P', 'kappa_P_G', 'omega_G',
    'fraction_R_of_RK', 'fraction_D_of_DE', 'fraction_S_of_SG', 'fraction_N_of_NQ', 'fraction_Y_of_YF',
    'fraction_F_of_FW', 'fraction_Y_of_YW', 'fraction_R_of_RQ', 'fraction_K_of_KQ', 'fraction_group_ILV',
    'fraction_FYW_of_FYWILV', 'fraction_FYW_of_FYWR',
]

external_feature_cols = [
    'lcr_residues', 'radius_of_gyration', 'scaling_exponent', 'asphericity',
    'pslab_delta_g', 'pslab_saturation_mgml', 'finches_heterotypic_epsilon', 'finches_homotypic_epsilon',
]

def safe_div(x, y):
    return x / y if y else 0

def get_fraction(seq, residues):
    return sum(seq.count(r) for r in residues) / len(seq)

def get_ratio(seq, group1, group2):
    return safe_div(sum(seq.count(r) for r in group1), sum(seq.count(r) for r in group2))

def get_fraction_within_group(seq, target, group):
    total = sum(seq.count(r) for r in group)
    return safe_div(seq.count(target), total)

def run_iupred(sequence, pred_type="long", threshold=0.5):
    seq = str(sequence).upper()
    scores = np.array(iupred(seq, pred_type)[0]).tolist()
    return sum(1 for s in scores if s > threshold) / len(scores)

def _extract_condenseq_features_one(seq: str, nardini_num_scrambles: int) -> dict:
    sp = SequenceParameters(seq)
    f = {}

    aa_frac = sp.get_amino_acid_fractions()
    for aa in "ACDEFGHIKLMNPQRSTVWY":
        f[f"fraction_{aa}"] = aa_frac.get(aa, 0)

    f['ratio_R_K'] = get_ratio(seq, "R", "K")
    f['ratio_D_E'] = get_ratio(seq, "D", "E")
    f['ratio_S_G'] = get_ratio(seq, "S", "G")
    f['ratio_N_Q'] = get_ratio(seq, "N", "Q")
    f['ratio_Y_F'] = get_ratio(seq, "Y", "F")
    f['ratio_F_W'] = get_ratio(seq, "F", "W")
    f['ratio_Y_W'] = get_ratio(seq, "Y", "W")
    f['ratio_R_Q'] = get_ratio(seq, "R", "Q")
    f['ratio_K_Q'] = get_ratio(seq, "K", "Q")

    for g in ["ILMV", "RK", "DE", "GS", "YFW"]:
        f[f'fraction_group_{g}'] = get_fraction(seq, GROUPS[g])
    f['fraction_group_ILV'] = get_fraction(seq, list("ILV"))
    f['fraction_FYW_of_FYWILV'] = safe_div(get_fraction(seq, list("FYW")), get_fraction(seq, list("FYWILV")))
    f['fraction_FYW_of_FYWR'] = safe_div(get_fraction(seq, list("FYW")), get_fraction(seq, list("FYWR")))

    f['ratio_group_FYW_ILV'] = get_ratio(seq, "FYW", "ILV")
    f['ratio_group_FYW_R'] = get_ratio(seq, "FYW", "R")

    f['fraction_R_of_RK'] = get_fraction_within_group(seq, 'R', GROUPS["RK"])
    f['fraction_K_of_KQ'] = get_fraction_within_group(seq, 'K', ['K', 'Q'])
    f['fraction_R_of_RQ'] = get_fraction_within_group(seq, 'R', ['R', 'Q'])
    f['fraction_D_of_DE'] = get_fraction_within_group(seq, 'D', GROUPS["ED"])
    f['fraction_S_of_SG'] = get_fraction_within_group(seq, 'S', GROUPS["GS"])
    f['fraction_N_of_NQ'] = get_fraction_within_group(seq, 'N', ['N', 'Q'])
    f['fraction_Y_of_YF'] = get_fraction_within_group(seq, 'Y', ['Y', 'F'])
    f['fraction_F_of_FW'] = get_fraction_within_group(seq, 'F', ['F', 'W'])
    f['fraction_Y_of_YW'] = get_fraction_within_group(seq, 'Y', ['Y', 'W'])

    f['NCPR'] = sp.get_NCPR()
    f['FCR'] = sp.get_FCR()
    f['fraction_disorder_promoting'] = sp.get_fraction_disorder_promoting()
    f['mean_hydropathy'] = sp.get_mean_hydropathy()

    f['frac_disorder'] = run_iupred(seq)

    for g in GROUPS:
        if f'omega_{g}' in condenseq_feature_cols:
            try:
                f[f'omega_{g}'] = nardini.get_omega_zscore(seq, GROUPS[g], num_scrambles=nardini_num_scrambles)
            except Exception as e:
                print(f"omega_{g} failed: {e}")
                f[f'omega_{g}'] = 0.0

    for g1 in GROUPS:
        for g2 in GROUPS:
            if f'kappa_{g1}_{g2}' in condenseq_feature_cols:
                try:
                    f[f'kappa_{g1}_{g2}'] = nardini.get_kappa_zscore(
                        seq, GROUPS[g1], GROUPS[g2], num_scrambles=nardini_num_scrambles
                    )
                except Exception as e:
                    print(f"kappa_{g1}_{g2} failed: {e}")
                    f[f'kappa_{g1}_{g2}'] = 0.0

    return {key: f.get(key, 0.0) for key in condenseq_feature_cols}


def extract_features_condenseq(sequences, *, nardini_num_scrambles=1000, n_workers=None):
    """Compute CondenSeq features for all sequences.

    This is dominated by NARDINI's omega/kappa z-score scrambling, which is
    CPU-bound and completely independent per sequence, so it is parallelized
    across processes.
    """
    sequences = list(sequences)
    if not sequences:
        return []

    n_workers = n_workers or os.cpu_count() or 1
    n_workers = max(1, min(n_workers, len(sequences)))
    worker = partial(_extract_condenseq_features_one, nardini_num_scrambles=nardini_num_scrambles)

    if n_workers == 1:
        return [worker(seq) for seq in tqdm(sequences, desc="condenseq")]

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        return list(tqdm(ex.map(worker, sequences), total=len(sequences), desc="condenseq"))

_human_idr_partner_seqs_cache = None

def _get_human_idr_partner_sequences():
    global _human_idr_partner_seqs_cache
    if _human_idr_partner_seqs_cache is None:
        if not os.path.exists(HUMAN_IDR_SEQUENCES_CSV):
            print(f"Warning: {HUMAN_IDR_SEQUENCES_CSV} not found; "
                  f"finches_heterotypic_epsilon will be NaN.")
            _human_idr_partner_seqs_cache = []
        else:
            df = pd.read_csv(HUMAN_IDR_SEQUENCES_CSV, low_memory=False)
            if "IDR_sequence" not in df.columns:
                raise ValueError(f"Expected column 'IDR_sequence' in {HUMAN_IDR_SEQUENCES_CSV}")
            _human_idr_partner_seqs_cache = [
                str(s).replace(" ", "").upper() for s in df["IDR_sequence"] if pd.notna(s) and str(s).strip()
            ]
    return _human_idr_partner_seqs_cache

def _partners_for_heterotypic_epsilon(all_partners, max_partners, seed):
    n = len(all_partners)
    if n == 0:
        return []
    if max_partners is None or max_partners >= n:
        return list(all_partners)
    if max_partners <= 0:
        return []
    k = min(int(max_partners), n)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=k, replace=False)
    return [all_partners[i] for i in idx]

def _finches_bottom5_percent_mean(scores):
    arr = np.asarray(scores, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    arr.sort()
    k = max(1, int(np.ceil(0.05 * arr.size)))
    return float(arr[:k].mean())

def _finches_heterotypic_epsilon_bottom5(seq, mf, partner_seqs):
    if not partner_seqs:
        return np.nan
    seq = str(seq).replace(" ", "").upper()
    scores = []
    for p in partner_seqs:
        p = str(p).replace(" ", "").upper()
        if not p:
            continue
        try:
            scores.append(mf.epsilon(seq, p))
        except Exception:
            continue
    return _finches_bottom5_percent_mean(scores)

def _batch_predict_values_in_order(seq_list, pred_map):
    if not isinstance(pred_map, dict):
        out = list(pred_map)
        if len(out) != len(seq_list):
            raise ValueError(f"batch_predict output length {len(out)} does not match sequences ({len(seq_list)}).")
        return out
    try:
        return [pred_map[s] for s in seq_list]
    except KeyError:
        vals = list(pred_map.values())
        if len(vals) != len(seq_list):
            raise ValueError(
                f"batch_predict dict keys do not match input sequences: {len(seq_list)} vs {len(vals)}."
            ) from None
        return vals

_worker_mf = None
_worker_partners_hetero = None


def _init_external_worker(partners_hetero):
    # Build the (potentially expensive-to-construct) finches frontend once per
    # worker process rather than once per sequence.
    global _worker_mf, _worker_partners_hetero
    _worker_mf = Mpipi_frontend()
    _worker_partners_hetero = partners_hetero


def _extract_external_features_one(seq: str) -> dict:
    mf = _worker_mf
    partners_hetero = _worker_partners_hetero
    f = {}

    # lcr via segmasker
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
        tmp.write(f">{seq}\n{seq}")
        tmp_path = tmp.name
    try:
        out = subprocess.Popen([segmasker_cmd, "-in", tmp_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        stdout_lcr, _ = out.communicate()
        if stdout_lcr:
            stdout_lcr = stdout_lcr.split()[1:]
            lcr_start_values, lcr_end_values = [], []
            for i in range(0, len(stdout_lcr) // 3):
                try:
                    start = int(stdout_lcr[3 * i].decode("utf-8"))
                    end = int(stdout_lcr[3 * i + 2].decode("utf-8"))
                    lcr_start_values.append(start)
                    lcr_end_values.append(end)
                except (IndexError, ValueError):
                    continue
            lcr_residues = []
            for start, end in zip(lcr_start_values, lcr_end_values):
                lcr_residues.extend(range(start, end + 1))
            f["lcr_residues"] = len(sorted(set(lcr_residues)))
        else:
            f["lcr_residues"] = 0
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    f["finches_homotypic_epsilon"] = mf.epsilon(seq, seq)
    f["finches_heterotypic_epsilon"] = _finches_heterotypic_epsilon_bottom5(seq, mf, partners_hetero)

    return f


def extract_features_external(
    sequences,
    *,
    pslab_inference_batch_size=None,
    finches_heterotypic_max_partners: Optional[int] = DEFAULT_FINCHES_HETEROTYPIC_MAX_PARTNERS,
    finches_heterotypic_seed: Optional[int] = DEFAULT_FINCHES_HETEROTYPIC_SEED,
    n_workers=None,
):
    seq_list = [str(s).replace(" ", "").upper() for s in sequences]
    if not seq_list:
        return []

    all_idr_partners = _get_human_idr_partner_sequences()
    partners_hetero = _partners_for_heterotypic_epsilon(
        all_idr_partners, finches_heterotypic_max_partners, finches_heterotypic_seed
    )

    # segmasker (subprocess) + finches epsilon (up to ~256 pairwise scores per
    # sequence) are independent per sequence and CPU-bound, so run them across
    # worker processes. Each worker builds its own Mpipi_frontend once (via
    # the pool initializer) instead of paying that cost per sequence.
    n_workers = n_workers or os.cpu_count() or 1
    n_workers = max(1, min(n_workers, len(seq_list)))

    if n_workers == 1:
        _init_external_worker(partners_hetero)
        results = [_extract_external_features_one(seq) for seq in tqdm(seq_list, desc="external features")]
    else:
        with ProcessPoolExecutor(
            max_workers=n_workers, initializer=_init_external_worker, initargs=(partners_hetero,)
        ) as ex:
            results = list(
                tqdm(ex.map(_extract_external_features_one, seq_list), total=len(seq_list), desc="external features")
            )

    if predict_pslab_batch is not None:
        dg_arr, sat_arr = predict_pslab_batch(seq_list, charge_termini=True, inference_batch_size=pslab_inference_batch_size)
    else:
        print("Warning: pslab_predict not importable; pslab_delta_g/pslab_saturation_mgml will be NaN.")
        dg_arr, sat_arr = [np.nan] * len(seq_list), [np.nan] * len(seq_list)
    for row, dg, sat in zip(results, dg_arr, sat_arr):
        row["pslab_delta_g"] = dg
        row["pslab_saturation_mgml"] = sat

    rg_map = batch_predict.batch_predict(seq_list, network="scaled_rg", return_seq2prediction=True)
    nu_map = batch_predict.batch_predict(seq_list, network="scaling_exponent", return_seq2prediction=True)
    asp_map = batch_predict.batch_predict(seq_list, network="asphericity", return_seq2prediction=True)
    radius_of_gyration_list = _batch_predict_values_in_order(seq_list, rg_map)
    scaling_exponent_list = _batch_predict_values_in_order(seq_list, nu_map)
    asphericity_list = _batch_predict_values_in_order(seq_list, asp_map)

    for row, rg, nu, asp in zip(results, radius_of_gyration_list, scaling_exponent_list, asphericity_list):
        row["radius_of_gyration"] = rg
        row["scaling_exponent"] = nu
        row["asphericity"] = asp

    return [{key: row.get(key, np.nan) for key in external_feature_cols} for row in results]

# handcrafted features

STANDARD_AAS = "ACDEFGHIKLMNPQRSTVWY"
_AA_INDEX = {aa: i for i, aa in enumerate(STANDARD_AAS)}

_AA_MW = {
    "G": 57.05, "A": 71.08, "V": 99.13, "L": 113.16, "I": 113.16,
    "P": 97.12, "F": 147.18, "W": 186.21, "M": 131.20, "S": 87.08,
    "T": 101.10, "C": 103.14, "Y": 163.18, "H": 137.14, "D": 115.09,
    "E": 129.12, "N": 114.10, "Q": 128.13, "K": 128.17, "R": 156.19,
}
_EISENBERG_HYDROPHOBICITY = {
    "A": 0.62, "R": -2.53, "N": -0.78, "D": -0.90, "C": 0.29,
    "Q": -0.85, "E": -0.74, "G": 0.48, "H": -0.40, "I": 1.38,
    "L": 1.06, "K": -1.50, "M": 0.64, "F": 1.19, "P": 0.12,
    "S": -0.18, "T": -0.05, "W": 0.81, "Y": 0.26, "V": 1.08,
}
_PKA_NTERM = 9.69
_PKA_CTERM = 2.34
_PKA_SIDE = {"D": 3.65, "E": 4.25, "C": 8.18, "Y": 10.07, "H": 6.00, "K": 10.53, "R": 12.48}
_WATER_MW = 18.015

_STICKER_AAS = "FWYR"
_SPACER_AAS = "GSTNQ"
_STICKER_SET = set(_STICKER_AAS)
_AROMATIC_SET = set("FWY")
_CHARGE_MAP = {"K": 1.0, "R": 1.0, "D": -1.0, "E": -1.0}

_GROUPS_TIER2 = {
    "frac_charged": "DEKR", "frac_positive": "KR", "frac_negative": "DE",
    "frac_polar": "STNQ", "frac_hydrophobic": "AVILMFYW", "frac_aromatic": "FWY",
    "frac_tiny": "GAS", "frac_small": "GASDNTPC", "frac_aliphatic": "AVIL",
    "frac_proline": "P", "frac_glycine": "G", "frac_disorder_promoting": "AEGRQSKPD",
}

def _group_indices(letters):
    return np.array([_AA_INDEX[aa] for aa in letters], dtype=np.intp)

def aa_composition(seq: str) -> np.ndarray:
    counts = np.zeros(20, dtype=np.int32)
    for aa in seq:
        idx = _AA_INDEX.get(aa)
        if idx is not None:
            counts[idx] += 1
    return counts

def compute_tier1_aa_fractions(sequences):
    n = len(sequences)
    counts = np.zeros((n, 20), dtype=np.int32)
    lengths = np.empty(n, dtype=np.int64)
    for i, seq in enumerate(sequences):
        counts[i] = aa_composition(seq)
        lengths[i] = len(seq)
    fracs = counts / np.maximum(lengths, 1)[:, None]
    return pd.DataFrame(fracs, columns=[f"frac_{aa}" for aa in STANDARD_AAS])

def compute_tier2_grouped(frac_df):
    result = {}
    for name, aas in _GROUPS_TIER2.items():
        cols = [f"frac_{aa}" for aa in aas]
        result[name] = frac_df[cols].sum(axis=1).values

    charged, positive = result["frac_charged"], result["frac_positive"]
    result["charge_ratio"] = np.divide(positive, charged, out=np.zeros_like(positive), where=charged > 0)

    polar, hydrophobic = result["frac_polar"], result["frac_hydrophobic"]
    result["polar_nonpolar_ratio"] = np.divide(polar, hydrophobic, out=np.zeros_like(polar), where=hydrophobic > 0)

    return pd.DataFrame(result)

def _isoelectric_point_batch(counts_matrix: np.ndarray) -> np.ndarray:
    """Bisection-search pI for every sequence at once (vectorized over rows).

    Equivalent to running the old scalar bisection separately per sequence,
    but does all 100 bisection iterations across the whole matrix in one
    pass instead of 100 * n_sequences individual Python-level iterations.
    """
    n = counts_matrix.shape[0]
    counts_f = counts_matrix.astype(np.float64)
    lo = np.zeros(n, dtype=np.float64)
    hi = np.full(n, 14.0, dtype=np.float64)

    for _ in range(100):
        mid = (lo + hi) / 2.0
        charge = 1.0 / (1.0 + 10.0 ** (mid - _PKA_NTERM))
        charge -= 1.0 / (1.0 + 10.0 ** (_PKA_CTERM - mid))
        for aa, pka in _PKA_SIDE.items():
            count = counts_f[:, _AA_INDEX[aa]]
            if aa in ("D", "E", "C", "Y"):
                charge -= count / (1.0 + 10.0 ** (pka - mid))
            else:
                charge += count / (1.0 + 10.0 ** (mid - pka))
        positive = charge > 0
        lo = np.where(positive, mid, lo)
        hi = np.where(positive, hi, mid)

    return (lo + hi) / 2.0

def compute_tier3_physicochemical(sequences):
    n = len(sequences)
    counts_matrix = np.zeros((n, 20), dtype=np.int32)
    lengths = np.empty(n, dtype=np.int64)
    for i, seq in enumerate(sequences):
        counts_matrix[i] = aa_composition(seq)
        lengths[i] = len(seq)

    mw_array = np.array([_AA_MW[aa] for aa in STANDARD_AAS])
    mol_weight = (counts_matrix * mw_array).sum(axis=1) + np.where(lengths > 0, _WATER_MW, 0.0)
    pi_values = _isoelectric_point_batch(counts_matrix)

    return pd.DataFrame({"molecular_weight": mol_weight, "isoelectric_point": pi_values})

def _scd(seq: str) -> float:
    return float(_sp_scd.compute_scd_x(seq))

def _charge_segregation(seq: str) -> float:
    n = len(seq)
    mid = n // 2
    first, second = seq[:mid], seq[mid:]
    l1, l2 = len(first), len(second)
    if l1 == 0 or l2 == 0:
        return 0.0
    pos_first = sum(1 for aa in first if aa in ("K", "R")) / l1
    pos_second = sum(1 for aa in second if aa in ("K", "R")) / l2
    neg_first = sum(1 for aa in first if aa in ("D", "E")) / l1
    neg_second = sum(1 for aa in second if aa in ("D", "E")) / l2
    return max(abs(pos_first - pos_second), abs(neg_first - neg_second))

def _hydrophobic_moment(seq: str, angle: float = 100.0, window: int = 11) -> float:
    n = len(seq)
    if n < window:
        window = n
    if window == 0:
        return 0.0
    angle_rad = np.radians(angle)
    h_values = np.array([_EISENBERG_HYDROPHOBICITY.get(aa, 0.0) for aa in seq])
    moments = []
    for start in range(n - window + 1):
        h_win = h_values[start:start + window]
        angles = np.arange(window) * angle_rad
        sin_sum = np.sum(h_win * np.sin(angles))
        cos_sum = np.sum(h_win * np.cos(angles))
        moments.append(np.sqrt(sin_sum ** 2 + cos_sum ** 2) / window)
    return float(np.mean(moments)) if moments else 0.0

def _aromatic_clustering(seq: str) -> float:
    positions = [i for i, aa in enumerate(seq) if aa in _AROMATIC_SET]
    n_aro = len(positions)
    if n_aro < 2:
        return 0.0
    total_dist, n_pairs = 0.0, 0
    for i in range(n_aro):
        for j in range(i + 1, n_aro):
            total_dist += positions[j] - positions[i]
            n_pairs += 1
    return (total_dist / n_pairs) / len(seq)

def compute_tier4_patterning(sequences):
    scd_vals = np.array([_scd(seq) for seq in sequences])
    seg_vals = np.array([_charge_segregation(seq) for seq in sequences])
    hm_vals = np.array([_hydrophobic_moment(seq) for seq in sequences])
    ac_vals = np.array([_aromatic_clustering(seq) for seq in sequences])
    return pd.DataFrame({
        "scd": scd_vals,
        "charge_segregation": seg_vals,
        "hydrophobic_moment": hm_vals,
        "aromatic_clustering": ac_vals,
    })

def _max_sticker_spacing(seq: str) -> int:
    positions = [i for i, aa in enumerate(seq) if aa in _STICKER_SET]
    if len(positions) <= 1:
        return len(seq)
    max_gap = 0
    for j in range(1, len(positions)):
        gap = positions[j] - positions[j - 1]
        if gap > max_gap:
            max_gap = gap
    return max_gap

def compute_tier5_sticker_spacer(sequences):
    n = len(sequences)
    sticker_idx = _group_indices(_STICKER_AAS)
    spacer_idx = _group_indices(_SPACER_AAS)

    counts = np.zeros((n, 20), dtype=np.int32)
    lengths = np.empty(n, dtype=np.int64)
    for i, seq in enumerate(sequences):
        counts[i] = aa_composition(seq)
        lengths[i] = len(seq)

    sticker_count = counts[:, sticker_idx].sum(axis=1)
    spacer_count = counts[:, spacer_idx].sum(axis=1)
    safe_lengths = np.maximum(lengths, 1)
    frac_sticker = sticker_count / safe_lengths
    frac_spacer = spacer_count / safe_lengths
    ratio = np.divide(
        sticker_count.astype(float), spacer_count.astype(float),
        out=np.zeros(n, dtype=float), where=spacer_count > 0,
    )
    spacing = np.array([_max_sticker_spacing(seq) for seq in sequences])

    return pd.DataFrame({
        "frac_sticker": frac_sticker,
        "frac_spacer": frac_spacer,
        "sticker_spacer_ratio": ratio,
        "max_sticker_spacing": spacing,
    })


def compute_metapredict_disorder(sequences):
    sequences = list(sequences)

    predictions = meta.predict_disorder(sequences)
    scores_by_seq = {seq: np.asarray(scores, dtype=np.float64) for seq, scores in predictions}

    means, maxes = [], []
    for seq in sequences:
        scores = scores_by_seq.get(seq)
        if scores is None or scores.size == 0:
            means.append(float("nan"))
            maxes.append(float("nan"))
        else:
            means.append(float(scores.mean()))
            maxes.append(float(scores.max()))
    return pd.DataFrame({"meta_disorder_mean": means, "meta_disorder_max": maxes})

# build final dataframe

OUTPUT_COLUMNS = [
    'protein_seq',
    'fraction_A', 'fraction_C', 'fraction_D', 'fraction_E', 'fraction_F', 'fraction_G', 'fraction_H',
    'fraction_I', 'fraction_K', 'fraction_L', 'fraction_M', 'fraction_N', 'fraction_P', 'fraction_Q',
    'fraction_R', 'fraction_S', 'fraction_T', 'fraction_V', 'fraction_W', 'fraction_Y',
    'fraction_group_ILMV', 'fraction_group_RK', 'fraction_group_DE', 'fraction_group_GS', 'fraction_group_YFW',
    'NCPR', 'FCR', 'fraction_disorder_promoting', 'mean_hydropathy', 'frac_disorder',
    'omega_STNQCH', 'kappa_STNQCH_ILMV', 'kappa_STNQCH_RK', 'kappa_STNQCH_ED', 'kappa_STNQCH_FWY',
    'kappa_STNQCH_A', 'kappa_STNQCH_P', 'kappa_STNQCH_G',
    'omega_ILMV', 'kappa_ILMV_RK', 'kappa_ILMV_ED', 'kappa_ILMV_FWY', 'kappa_ILMV_A', 'kappa_ILMV_P', 'kappa_ILMV_G',
    'omega_RK', 'kappa_RK_ED', 'kappa_RK_FWY', 'kappa_RK_A', 'kappa_RK_P', 'kappa_RK_G',
    'omega_ED', 'kappa_ED_FWY', 'kappa_ED_A', 'kappa_ED_P', 'kappa_ED_G',
    'omega_FWY', 'kappa_FWY_A', 'kappa_FWY_P', 'kappa_FWY_G',
    'omega_A', 'kappa_A_P', 'kappa_A_G',
    'omega_P', 'kappa_P_G',
    'omega_G',
    'fraction_R_of_RK', 'fraction_D_of_DE', 'fraction_S_of_SG', 'fraction_N_of_NQ', 'fraction_Y_of_YF',
    'fraction_F_of_FW', 'fraction_Y_of_YW', 'fraction_R_of_RQ', 'fraction_K_of_KQ',
    'fraction_group_ILV', 'fraction_FYW_of_FYWILV', 'fraction_FYW_of_FYWR',
    'lcr_residues', 'radius_of_gyration', 'scaling_exponent', 'asphericity',
    'pslab_delta_g', 'pslab_saturation_mgml', 'finches_heterotypic_epsilon', 'finches_homotypic_epsilon',
    'frac_polar', 'frac_hydrophobic', 'frac_tiny', 'frac_small', 'frac_aliphatic',
    'charge_ratio', 'polar_nonpolar_ratio',
    'molecular_weight', 'isoelectric_point',
    'scd', 'charge_segregation', 'hydrophobic_moment', 'aromatic_clustering',
    'frac_sticker', 'frac_spacer', 'sticker_spacer_ratio', 'max_sticker_spacing',
    'meta_disorder_mean', 'meta_disorder_max'
]

def build_feature_dataframe(
    sequences: list[str],
    *,
    nardini_num_scrambles: int = DEFAULT_NARDINI_SCRAMBLES,
    pslab_inference_batch_size=None,
    finches_heterotypic_max_partners: Optional[int] = DEFAULT_FINCHES_HETEROTYPIC_MAX_PARTNERS,
    finches_heterotypic_seed: Optional[int] = DEFAULT_FINCHES_HETEROTYPIC_SEED,
    n_workers: Optional[int] = None,
) -> pd.DataFrame:
    print(f"Computing features for {len(sequences)} sequences using {n_workers or os.cpu_count() or 1} worker(s)...")

    print("[1/5] CondenSeq composition/patterning features...")
    condenseq_rows = extract_features_condenseq(
        sequences, nardini_num_scrambles=nardini_num_scrambles, n_workers=n_workers
    )
    condenseq_df = pd.DataFrame(condenseq_rows)

    print("[2/5] External structural predictors (lcr/finches/pslab/sparrow)...")
    external_rows = extract_features_external(
        sequences,
        pslab_inference_batch_size=pslab_inference_batch_size,
        finches_heterotypic_max_partners=finches_heterotypic_max_partners,
        finches_heterotypic_seed=finches_heterotypic_seed,
        n_workers=n_workers,
    )
    external_df = pd.DataFrame(external_rows)

    print("[3/5] Handcrafted tier 2/3 features (composition ratios, MW, pI)...")
    tier1_df = compute_tier1_aa_fractions(sequences)
    tier2_df = compute_tier2_grouped(tier1_df)[["frac_polar", "frac_hydrophobic", "frac_tiny", "frac_small",
                                                 "frac_aliphatic", "charge_ratio", "polar_nonpolar_ratio"]]
    tier3_df = compute_tier3_physicochemical(sequences)

    print("[4/5] Handcrafted tier 4/5 features (patterning, sticker-spacer)...")
    tier4_df = compute_tier4_patterning(sequences)
    tier5_df = compute_tier5_sticker_spacer(sequences)

    print("[5/5] Metapredict disorder scores...")
    meta_df = compute_metapredict_disorder(sequences)

    df = pd.concat(
        [
            pd.DataFrame({"protein_seq": sequences}),
            condenseq_df,
            external_df,
            tier2_df.reset_index(drop=True),
            tier3_df.reset_index(drop=True),
            tier4_df.reset_index(drop=True),
            tier5_df.reset_index(drop=True),
            meta_df.reset_index(drop=True)
        ],
        axis=1,
    )

    return df[OUTPUT_COLUMNS]

def main():
    parser = argparse.ArgumentParser(description="Compute biophysical sequence features from a FASTA file.")
    parser.add_argument("fasta_file", help="Path to input FASTA file")
    parser.add_argument("output_csv", help="Path to output CSV file")
    parser.add_argument("--nardini-scrambles", type=int, default=DEFAULT_NARDINI_SCRAMBLES,
                         help=f"Number of NARDINI scrambles (default: {DEFAULT_NARDINI_SCRAMBLES})")
    parser.add_argument("--pslab-batch-size", type=int, default=None,
                         help="pslab inference batch size (default: library default)")
    parser.add_argument("--finches-max-partners", type=int, default=DEFAULT_FINCHES_HETEROTYPIC_MAX_PARTNERS,
                         help=f"Max human IDR partners for heterotypic epsilon (default: {DEFAULT_FINCHES_HETEROTYPIC_MAX_PARTNERS})")
    parser.add_argument("--finches-seed", type=int, default=DEFAULT_FINCHES_HETEROTYPIC_SEED,
                         help=f"Random seed for partner subsampling (default: {DEFAULT_FINCHES_HETEROTYPIC_SEED})")
    parser.add_argument("--workers", type=int, default=None,
                         help="Number of worker processes for the CondenSeq/NARDINI and external "
                              "(segmasker/finches) feature stages (default: os.cpu_count())")
    args = parser.parse_args()

    sequences = load_sequences_from_fasta(args.fasta_file)
    if not sequences:
        print(f"No sequences found in {args.fasta_file}.")
        sys.exit(1)

    df = build_feature_dataframe(
        sequences,
        nardini_num_scrambles=args.nardini_scrambles,
        pslab_inference_batch_size=args.pslab_batch_size,
        finches_heterotypic_max_partners=args.finches_max_partners,
        finches_heterotypic_seed=args.finches_seed,
        n_workers=args.workers,
    )

    df.to_csv(args.output_csv, index=False)
    print(f"Saved {len(df)} rows x {len(df.columns)} columns to {args.output_csv}")

if __name__ == "__main__":
    main()