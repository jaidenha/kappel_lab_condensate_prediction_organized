"""
PSLab phase-separation predictions from Bülow et al. (KULL-Centre PSpred).

Matches the public Colab notebook (PSLab.ipynb): same input features, MLP
ensemble models, and post-processing (exp for saturation in mg/mL).

Source: https://github.com/KULL-Centre/_2024_buelow_PSpred
"""

from __future__ import annotations

import importlib.util
import sys
import threading
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import load

_PSLAB_ROOT = Path(__file__).resolve().parent
_GH = "https://raw.githubusercontent.com/KULL-Centre/_2024_buelow_PSpred/main"

# PSLab.ipynb
PSLAB_FEATURES: list[str] = [
    "mean_lambda",
    "faro",
    "shd",
    "ncpr",
    "fcr",
    "scd",
    "ah_ij",
    "nu_svr",
]

_ASSETS: tuple[tuple[str, str], ...] = (
    ("model_dG.joblib", f"{_GH}/models/idrome90/mlp/dG/model.joblib"),
    ("model_logcdil_mgml.joblib", f"{_GH}/models/idrome90/mlp/logcdil_mgml/model.joblib"),
    ("svr_model_nu.joblib", f"{_GH}/models/svr_model_nu.joblib"),
    ("residues.csv", f"{_GH}/data/residues.csv"),
)

_lock = threading.Lock()
_state: dict[str, Any] = {}

# Feature matrix rows per inference chunk (sklearn + ensemble predict once per chunk).
DEFAULT_INFERENCE_BATCH_SIZE = 2048


def _download_if_missing(rel_name: str, url: str) -> Path:
    dest = _PSLAB_ROOT / rel_name
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    return dest


def ensure_pslab_assets() -> None:
    """Download pretrained models and residue table from the official repo if absent."""
    for rel, url in _ASSETS:
        _download_if_missing(rel, url)


def _patch_main_for_joblib(pc: Any) -> None:
    import __main__

    setattr(__main__, "Model", pc.Model)
    setattr(__main__, "AttrSetter", pc.AttrSetter)


def _load_predictor_module() -> Any:
    root = str(_PSLAB_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    spec = importlib.util.spec_from_file_location(
        "pslab_predictor_colab", _PSLAB_ROOT / "predictor_colab.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load predictor_colab.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _get_runtime():
    global _state
    with _lock:
        if _state.get("ready"):
            return _state
        ensure_pslab_assets()
        pc = _load_predictor_module()
        _patch_main_for_joblib(pc)
        residues = pd.read_csv(_PSLAB_ROOT / "residues.csv").set_index("one")
        nu_path = str(_PSLAB_ROOT / "svr_model_nu.joblib")
        model_dg = load(_PSLAB_ROOT / "model_dG.joblib")
        model_log = load(_PSLAB_ROOT / "model_logcdil_mgml.joblib")
        _state.update(
            ready=True,
            pc=pc,
            residues=residues,
            nu_path=nu_path,
            model_dg=model_dg,
            model_logcdil_mgml=model_log,
        )
        return _state


def _feature_row(
    pc: Any,
    seq: str,
    residues: pd.DataFrame,
    nu_path: str,
    *,
    charge_termini: bool,
) -> np.ndarray:
    """Single sample feature vector (1, n_features) -> (n_features,)."""
    X = pc.X_from_seq(
        seq,
        PSLAB_FEATURES,
        residues=residues,
        charge_termini=charge_termini,
        nu_file=nu_path,
    )
    return np.asarray(X, dtype=np.float64).ravel()


def predict_pslab_batch(
    sequences: list[str] | np.ndarray,
    *,
    charge_termini: bool = True,
    inference_batch_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (delta_g_kT, saturation_mgml) per sequence, aligned with PSLab.ipynb.

    delta_g: mean of 50-fold cross-validated MLP predictions (kT).
    saturation_mgml: exp(mean log prediction) for dilute-phase saturation (mg/mL).

    For large jobs (e.g. 14k+ sequences), features are built in chunks and each
    chunk is passed through the ensembles in one ``predict`` call per model
    (instead of one predict per sequence).     Set ``inference_batch_size`` to tune
    memory vs. chunk count (default ``DEFAULT_INFERENCE_BATCH_SIZE``, 2048).
    """
    rt = _get_runtime()
    pc = rt["pc"]
    residues = rt["residues"]
    nu_path = rt["nu_path"]
    model_dg = rt["model_dg"]
    model_log = rt["model_logcdil_mgml"]

    seq_list = [str(s).replace(" ", "").upper() for s in sequences]
    n = len(seq_list)
    dg_out = np.full(n, np.nan, dtype=np.float64)
    sat_out = np.full(n, np.nan, dtype=np.float64)

    bs = inference_batch_size if inference_batch_size is not None else DEFAULT_INFERENCE_BATCH_SIZE
    if bs < 1:
        raise ValueError("inference_batch_size must be >= 1")

    try:
        from tqdm import tqdm as _tqdm
    except ImportError:
        def _tqdm(it, **_):
            return it

    n_batches = (n + bs - 1) // bs
    for start in _tqdm(
        range(0, n, bs),
        desc="PSLab",
        unit="batch",
        total=n_batches,
    ):
        stop = min(start + bs, n)
        idx_ok: list[int] = []
        rows: list[np.ndarray] = []
        for i in range(start, stop):
            seq = seq_list[i]
            if not seq:
                continue
            try:
                rows.append(
                    _feature_row(
                        pc,
                        seq,
                        residues,
                        nu_path,
                        charge_termini=charge_termini,
                    )
                )
                idx_ok.append(i)
            except Exception:
                continue
        if not rows:
            continue
        X = np.stack(rows, axis=0)
        # (ncrossval, n_chunk) — one ensemble forward per chunk, not per sequence
        ys_dg = model_dg.predict(X)
        ys_log = model_log.predict(X)
        dg_vals = np.mean(ys_dg, axis=0)
        log_vals = np.mean(ys_log, axis=0)
        sat_vals = np.exp(log_vals)
        for j, gi in enumerate(idx_ok):
            dg_out[gi] = float(dg_vals[j])
            sat_out[gi] = float(sat_vals[j])

    return dg_out, sat_out
