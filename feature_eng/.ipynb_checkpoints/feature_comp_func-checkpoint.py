from __future__ import annotations

import os
import sys
from typing import Optional
import subprocess
import tempfile

import numpy as np
import pandas as pd

_FEATURE_ENG_DIR = os.path.dirname(os.path.abspath(__file__))
_STANDALONE_DIR = os.path.join(_FEATURE_ENG_DIR, "standalone_files")
_PSLAB_DIR = os.path.join(_STANDALONE_DIR, "pslab")
for _path in (_PSLAB_DIR, _STANDALONE_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# import specific packages
from Bio import SeqIO
from localcider.sequenceParameters import SequenceParameters
from iupred3.iupred3_lib import iupred
import nardini
import shutil
segmasker_cmd = shutil.which('segmasker') or 'segmasker'
from sparrow.predictors import batch_predict
from finches import Mpipi_frontend
from pslab_predict import predict_pslab_batch

import re

_HUMAN_IDR_SEQUENCES_CSV = os.path.join(_FEATURE_ENG_DIR, "datasets", "human_IDR_sequences.csv")
_human_idr_partner_seqs_cache = None

# Defaults tuned for interactive / large-batch use (override per call as needed).
DEFAULT_FINCHES_HETEROTYPIC_MAX_PARTNERS = 256
DEFAULT_FINCHES_HETEROTYPIC_SEED = 42
DEFAULT_RUN_FEATURE_NARDINI_SCRAMBLES = 200


def _get_human_idr_partner_sequences():
    """Load human IDR partners once per process (column IDR_sequence)."""
    global _human_idr_partner_seqs_cache
    if _human_idr_partner_seqs_cache is None:
        df = pd.read_csv(_HUMAN_IDR_SEQUENCES_CSV, low_memory=False)
        if "IDR_sequence" not in df.columns:
            raise ValueError(
                f"Expected column 'IDR_sequence' in {_HUMAN_IDR_SEQUENCES_CSV}"
            )
        _human_idr_partner_seqs_cache = [
            str(s).replace(" ", "").upper()
            for s in df["IDR_sequence"]
            if pd.notna(s) and str(s).strip()
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
    
    # none → OS entropy (non-reproducible); int → fixed partner subset
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=k, replace=False)
    return [all_partners[i] for i in idx]


def _finches_bottom5_percent_mean(scores):
    # mean of the lowest 5% of values (sorted ascending)
    arr = np.asarray(scores, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    arr.sort()
    k = max(1, int(np.ceil(0.05 * arr.size)))
    return float(arr[:k].mean())


def _batch_predict_values_in_order(seq_list, pred_map):
    
    if not isinstance(pred_map, dict):
        out = list(pred_map)
        if len(out) != len(seq_list):
            raise ValueError(
                f"batch_predict output length {len(out)} does not match "
                f"sequences ({len(seq_list)})."
            )
        return out
    try:
        return [pred_map[s] for s in seq_list]
    except KeyError:
        vals = list(pred_map.values())
        if len(vals) != len(seq_list):
            raise ValueError(
                "batch_predict dict keys do not match input sequences: "
                f"{len(seq_list)} sequences vs {len(vals)} predictions."
            ) from None
        return vals


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


def parse_label_from_header(header):
    match = re.search(r'label=(\d+)', header)
    return int(match.group(1)) if match else None

# get sequences from fasta as a list
def load_records_from_fasta(fasta_file_path, concentration_value = "medium"):
    records = []
    for record in SeqIO.parse(fasta_file_path, "fasta"):
        header = str(record.description)
        seq = str(record.seq).replace("*", "").replace(" ", "").upper()
        label = parse_label_from_header(header)
        if seq:
            records.append({
                "sequence_id": header,
                "protein_seq": seq,
                "label": label,
                "concentration": concentration_value
            })
    return records

# initialize list of feature cols
condenseq_feature_cols = [
    'fraction_A','fraction_C','fraction_D','fraction_E','fraction_F','fraction_G','fraction_H','fraction_I','fraction_K','fraction_L','fraction_M',
    'fraction_N','fraction_P','fraction_Q','fraction_R','fraction_S','fraction_T','fraction_V','fraction_W','fraction_Y',
    'ratio_R_K','ratio_D_E','ratio_S_G','ratio_N_Q','ratio_Y_F','ratio_F_W','ratio_Y_W','ratio_R_Q','ratio_K_Q',
    'fraction_group_ILMV','fraction_group_RK','fraction_group_DE','fraction_group_GS','fraction_group_YFW',
    'ratio_group_FYW_ILV','ratio_group_FYW_R','NCPR','FCR','fraction_disorder_promoting','mean_hydropathy',
    'frac_disorder','omega_STNQCH','kappa_STNQCH_ILMV','kappa_STNQCH_RK','kappa_STNQCH_ED',
    'kappa_STNQCH_FWY','kappa_STNQCH_A','kappa_STNQCH_P','kappa_STNQCH_G','omega_ILMV','kappa_ILMV_RK','kappa_ILMV_ED',
    'kappa_ILMV_FWY','kappa_ILMV_A','kappa_ILMV_P','kappa_ILMV_G','omega_RK','kappa_RK_ED','kappa_RK_FWY','kappa_RK_A',
    'kappa_RK_P','kappa_RK_G','omega_ED','kappa_ED_FWY','kappa_ED_A','kappa_ED_P','kappa_ED_G','omega_FWY',
    'kappa_FWY_A','kappa_FWY_P','kappa_FWY_G','omega_A','kappa_A_P','kappa_A_G','omega_P','kappa_P_G','omega_G',
    'fraction_R_of_RK','fraction_D_of_DE','fraction_S_of_SG','fraction_N_of_NQ','fraction_Y_of_YF','fraction_F_of_FW',
    'fraction_Y_of_YW','fraction_R_of_RQ','fraction_K_of_KQ','fraction_group_ILV','fraction_FYW_of_FYWILV','fraction_FYW_of_FYWR'
]

external_feature_cols = [
    'lcr_residues',
    'radius_of_gyration', 'scaling_exponent', 'asphericity',
    'pslab_delta_g', 'pslab_saturation_mgml', 'finches_heterotypic_epsilon',
    'finches_homotypic_epsilon'
]

feature_cols = condenseq_feature_cols + external_feature_cols

# define amino acid groups
GROUPS = {
    "ILMV": list("ILMV"), "RK": list("RK"), "ED": list("ED"), "DE": list("DE"), 
    "GS": list("GS"), "YFW": list("YFW"), "FYW": list("FYW"),"FWY": list("FWY"),
    "STNQCH": list("STNQCH"), "A": ["A"], "P": ["P"], "G": ["G"]
}

# define helper functions
def safe_div(x, y): return x / y if y else 0
def get_fraction(seq, residues): return sum(seq.count(r) for r in residues) / len(seq)
def get_ratio(seq, group1, group2): return safe_div(sum(seq.count(r) for r in group1), sum(seq.count(r) for r in group2))
def get_fraction_within_group(seq, target, group): total = sum(seq.count(r) for r in group); return safe_div(seq.count(target), total)

# iupred extraction
def run_iupred(sequence, pred_type = 'long', threshold = 0.5):
    seq = str(sequence).upper()
    scores = np.array(iupred(seq, pred_type)[0]).tolist()

    frac_disorder = sum(1 for s in scores if s > threshold) / len(scores)
    
    return frac_disorder

# function for computing all features in the original condenseq dataset
def extract_features_condenseq(sequences, *, nardini_num_scrambles: int = 1000):

    results = []

    for seq in sequences:
        sp = SequenceParameters(seq)
        f = {}
        
        # residue fractions
        aa_frac = sp.get_amino_acid_fractions()
        
        for aa in 'ACDEFGHIKLMNPQRSTVWY':
            f[f'fraction_{aa}'] = aa_frac.get(aa, 0)

        # ratios
        f['ratio_R_K'] = get_ratio(seq, "R", "K")
        f['ratio_D_E'] = get_ratio(seq, "D", "E")
        f['ratio_S_G'] = get_ratio(seq, "S", "G")
        f['ratio_N_Q'] = get_ratio(seq, "N", "Q")
        f['ratio_Y_F'] = get_ratio(seq, "Y", "F")
        f['ratio_F_W'] = get_ratio(seq, "F", "W")
        f['ratio_Y_W'] = get_ratio(seq, "Y", "W")
        f['ratio_R_Q'] = get_ratio(seq, "R", "Q")
        f['ratio_K_Q'] = get_ratio(seq, "K", "Q")

        # group fractions
        for g in ["ILMV", "RK", "DE", "GS", "YFW"]:
            f[f'fraction_group_{g}'] = get_fraction(seq, GROUPS[g])
        f['fraction_group_ILV'] = get_fraction(seq, list("ILV"))
        f['fraction_FYW_of_FYWILV'] = safe_div(get_fraction(seq, list("FYW")), get_fraction(seq, list("FYWILV")))
        f['fraction_FYW_of_FYWR'] = safe_div(get_fraction(seq, list("FYW")), get_fraction(seq, list("FYWR")))

        # group ratios
        f['ratio_group_FYW_ILV'] = get_ratio(seq, "FYW", "ILV")
        f['ratio_group_FYW_R'] = get_ratio(seq, "FYW", "R")

        # within-group residue ratios
        f['fraction_R_of_RK'] = get_fraction_within_group(seq, 'R', GROUPS["RK"])
        f['fraction_K_of_KQ'] = get_fraction_within_group(seq, 'K', ['K', 'Q'])
        f['fraction_R_of_RQ'] = get_fraction_within_group(seq, 'R', ['R', 'Q'])
        f['fraction_D_of_DE'] = get_fraction_within_group(seq, 'D', GROUPS["ED"])
        f['fraction_S_of_SG'] = get_fraction_within_group(seq, 'S', GROUPS["GS"])
        f['fraction_N_of_NQ'] = get_fraction_within_group(seq, 'N', ['N', 'Q'])
        f['fraction_Y_of_YF'] = get_fraction_within_group(seq, 'Y', ['Y', 'F'])
        f['fraction_F_of_FW'] = get_fraction_within_group(seq, 'F', ['F', 'W'])
        f['fraction_Y_of_YW'] = get_fraction_within_group(seq, 'Y', ['Y', 'W'])

        # charge, disorder, hydropathy
        f['NCPR'] = sp.get_NCPR()
        f['FCR'] = sp.get_FCR()
        f['fraction_disorder_promoting'] = sp.get_fraction_disorder_promoting()
        f['mean_hydropathy'] = sp.get_mean_hydropathy()

        # iupred
        frac_disorder = run_iupred(seq)
        f["frac_disorder"] = frac_disorder

        # omega and kappa values with nardini
        for g in GROUPS:
            if f'omega_{g}' in feature_cols:
                try:
                    f[f'omega_{g}'] = nardini.get_omega_zscore(
                        seq, GROUPS[g], num_scrambles=nardini_num_scrambles
                    )
                except Exception as e:
                    print(f"omega_{g} failed: {e}")
                    f[f'omega_{g}'] = np.nan

        for g1 in GROUPS:
            for g2 in GROUPS:
                if f'kappa_{g1}_{g2}' in feature_cols:
                    try:
                        f[f'kappa_{g1}_{g2}'] = nardini.get_kappa_zscore(
                            seq,
                            GROUPS[g1],
                            GROUPS[g2],
                            num_scrambles=nardini_num_scrambles,
                        )
                    except Exception as e:
                        print(f"kappa_{g1}_{g2} failed: {e}")
                        f[f'kappa_{g1}_{g2}'] = np.nan
    
        # return dict in exact order of feature_cols
        row = {key: f.get(key, 0.0) for key in condenseq_feature_cols}
        results.append(row)

    return results

def extract_features_external(
    sequences,
    *,
    pslab_inference_batch_size=None,
    finches_heterotypic_max_partners: Optional[int] = DEFAULT_FINCHES_HETEROTYPIC_MAX_PARTNERS,
    finches_heterotypic_seed: Optional[int] = DEFAULT_FINCHES_HETEROTYPIC_SEED,
):

    seq_list = [str(s).replace(" ", "").upper() for s in (
        sequences.tolist() if hasattr(sequences, "tolist") else list(sequences)
    )]
    if not seq_list:
        return []

    results = []
    all_idr_partners = _get_human_idr_partner_sequences()
    partners_hetero = _partners_for_heterotypic_epsilon(
        all_idr_partners,
        finches_heterotypic_max_partners,
        finches_heterotypic_seed,
    )
    mf = Mpipi_frontend()

    for seq in seq_list:
        f = {}

        # 1. lcr
        
        # create temp file for sequence
        with tempfile.NamedTemporaryFile(mode = 'w+', delete = False) as tmp:
            tmp.write(f">{seq}\n{seq}")
            tmp_path = tmp.name
        
        try:
            # run segmasker
            out = subprocess.Popen([segmasker_cmd, '-in', tmp_path],
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            stdout_lcr, _ = out.communicate()
            
            if stdout_lcr:
                stdout_lcr = stdout_lcr.split()[1:]
                
                # Parse LCR regions
                lcr_start_values = []
                lcr_end_values = []
                for i in range(0, len(stdout_lcr)//3):
                    try:
                        start = int(stdout_lcr[3*i].decode('utf-8'))
                        end = int(stdout_lcr[3*i+2].decode('utf-8'))
                        lcr_start_values.append(start)
                        lcr_end_values.append(end)
                    except (IndexError, ValueError):
                        continue
                
                # Calculate LCR residues
                lcr_residues = []
                for start, end in zip(lcr_start_values, lcr_end_values):
                    lcr_residues.extend(range(start, end + 1))
                
                lcr_residues = sorted(set(lcr_residues))
                f['lcr_residues'] = len(lcr_residues)
            else:
                f['lcr_residues'] = 0
        
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        # finches (homotypic + heterotypic vs human IDR library, bottom 5% mean)
        f['finches_homotypic_epsilon'] = mf.epsilon(seq, seq)
        f['finches_heterotypic_epsilon'] = _finches_heterotypic_epsilon_bottom5(
            seq, mf, partners_hetero
        )

        results.append(f)

    # pslab (using pslab_predict.py in standalone files; aligned by index with seq_list)
    dg_arr, sat_arr = predict_pslab_batch(
        seq_list,
        charge_termini=True,
        inference_batch_size=pslab_inference_batch_size,
    )
    for row, dg, sat in zip(results, dg_arr, sat_arr):
        row["pslab_delta_g"] = dg
        row["pslab_saturation_mgml"] = sat

    # sparrow/albatross — align predictions to seq_list order (do not rely on dict.values() order)
    rg_map = batch_predict.batch_predict(
        seq_list, network="scaled_rg", return_seq2prediction=True
    )
    nu_map = batch_predict.batch_predict(
        seq_list, network="scaling_exponent", return_seq2prediction=True
    )
    asp_map = batch_predict.batch_predict(
        seq_list, network="asphericity", return_seq2prediction=True
    )
    radius_of_gyration_list = _batch_predict_values_in_order(seq_list, rg_map)
    scaling_exponent_list = _batch_predict_values_in_order(seq_list, nu_map)
    asphericity_list = _batch_predict_values_in_order(seq_list, asp_map)

    for row, rg, nu, asp in zip(results, radius_of_gyration_list, scaling_exponent_list, asphericity_list):
        row["radius_of_gyration"] = rg
        row["scaling_exponent"] = nu
        row["asphericity"] = asp

    finalized = []
    for row in results:
        finalized.append(
            {key: row.get(key, np.nan) for key in external_feature_cols}
        )
    return finalized


def run_feature_extraction(
    sequences,
    *,
    seq_column: str = "seq",
    pslab_inference_batch_size=None,
    finches_heterotypic_max_partners: Optional[int] = DEFAULT_FINCHES_HETEROTYPIC_MAX_PARTNERS,
    finches_heterotypic_seed: Optional[int] = DEFAULT_FINCHES_HETEROTYPIC_SEED,
    nardini_num_scrambles: int = DEFAULT_RUN_FEATURE_NARDINI_SCRAMBLES,
):

    seq_list = sequences.tolist() if hasattr(sequences, "tolist") else list(sequences)
    seq_list = [str(s).replace(" ", "").upper() for s in seq_list]

    if not seq_list:
        return pd.DataFrame(columns=[seq_column] + list(feature_cols))

    inner = extract_features_condenseq(seq_list, nardini_num_scrambles=nardini_num_scrambles)
    outer = extract_features_external(
        seq_list,
        pslab_inference_batch_size=pslab_inference_batch_size,
        finches_heterotypic_max_partners=finches_heterotypic_max_partners,
        finches_heterotypic_seed=finches_heterotypic_seed,
    )
    if len(inner) != len(outer):
        raise ValueError(
            f"Feature row count mismatch: condenseq {len(inner)} vs external {len(outer)}"
        )

    rows = []
    for seq, cdict, edict in zip(seq_list, inner, outer):
        merged = {**cdict, **edict}
        row = {seq_column: seq}
        for name in feature_cols:
            row[name] = merged.get(name, np.nan)
        rows.append(row)

    column_order = [seq_column] + list(feature_cols)
    return pd.DataFrame(rows, columns=column_order)

build_feature_dataset = run_feature_extraction

__all__ = [
    "GROUPS",
    "DEFAULT_FINCHES_HETEROTYPIC_MAX_PARTNERS",
    "DEFAULT_FINCHES_HETEROTYPIC_SEED",
    "DEFAULT_RUN_FEATURE_NARDINI_SCRAMBLES",
    "run_feature_extraction",
    "build_feature_dataset",
    "condenseq_feature_cols",
    "external_feature_cols",
    "feature_cols",
    "extract_features_condenseq",
    "extract_features_external",
    "get_fraction",
    "get_fraction_within_group",
    "get_ratio",
    "load_records_from_fasta",
    "parse_label_from_header",
    "run_iupred",
    "safe_div",
    "segmasker_cmd",
]
