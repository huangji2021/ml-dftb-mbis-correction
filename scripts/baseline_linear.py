#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np


# ---------------- I/O ----------------
def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows

def safe_get(d, key):
    if key not in d:
        raise KeyError(f"Missing key '{key}' in record with name={d.get('name')}")
    return d[key]


# ---------------- Merge ----------------
def merge_by_name(dftb_rows, dft_rows):
    dftb = {r["name"]: r for r in dftb_rows if "name" in r}
    dft  = {r["name"]: r for r in dft_rows  if "name" in r}
    names = sorted(set(dftb.keys()) & set(dft.keys()))
    merged = []
    for name in names:
        a = dftb[name]
        b = dft[name]
        z_a = safe_get(a, "z")
        z_b = safe_get(b, "z")
        if z_a != z_b:
            raise ValueError(f"Z mismatch for {name}")
        merged.append({
            "name": name,
            "z": z_a,
            "q_dftb": safe_get(a, "mulliken_dftb3"),
            "q_mbis": safe_get(b, "mbis_dft"),
        })
    return merged


# ---------------- Linear fit per element ----------------
def fit_line(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan"), float("nan"), 0, float("nan")

    A = np.vstack([x, np.ones_like(x)]).T
    sol, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)
    a, b = sol[0], sol[1]

    yhat = a * x + b
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    n = len(x)
    dof = n - 2
    if dof > 0:
        sigma2 = ss_res / dof
        cov = sigma2 * np.linalg.inv(A.T @ A)
        se_a = float(np.sqrt(cov[0, 0]))
        se_b = float(np.sqrt(cov[1, 1]))
    else:
        se_a = float("nan")
        se_b = float("nan")

    return float(a), float(b), se_a, se_b, n, float(r2)


def predict_elementwise(z_list, q_dftb_list, coeffs):
    z = np.asarray(z_list, int)
    q = np.asarray(q_dftb_list, float)
    out = np.empty_like(q)
    for i, Zi in enumerate(z):
        sym = Z_TO_SYM.get(int(Zi))
        if sym not in coeffs:
            raise ValueError(f"No coefficient for element {sym}")
        a, b = coeffs[sym]["a"], coeffs[sym]["b"]
        out[i] = a * q[i] + b
    return out


# ---------------- Metrics ----------------
def mae(x):
    x = np.asarray(x, float)
    return float(np.mean(np.abs(x))) if len(x) else float("nan")

def rmse(x):
    x = np.asarray(x, float)
    return float(np.sqrt(np.mean(x**2))) if len(x) else float("nan")


Z_TO_SYM = {1:"H",6:"C",7:"N",8:"O",9:"F"}


def load_split(path):
    if not path.exists():
        raise RuntimeError(f"Missing split file: {path}")
    return set(path.read_text(encoding="utf-8").split())


# ================================
# MAIN
# ================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dftb", required=True)
    ap.add_argument("--dft", required=True)
    ap.add_argument("--splits_dir", required=True)
    ap.add_argument("--outdir", default="baseline_results")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---------- Load data ----------
    dftb_rows = read_jsonl(args.dftb)
    dft_rows  = read_jsonl(args.dft)
    data = merge_by_name(dftb_rows, dft_rows)

    clean = []
    for r in data:
        if len(r["q_dftb"]) != len(r["q_mbis"]):
            continue
        clean.append(r)

    # ---------- Load split ----------
    splits_dir = Path(args.splits_dir)
    train_set = load_split(splits_dir/"train.txt")
    val_set   = load_split(splits_dir/"val.txt")
    test_set  = load_split(splits_dir/"test.txt")

    print(f"[INFO] Loaded split:")
    print(f"  Train molecules: {len(train_set)}")
    print(f"  Val molecules  : {len(val_set)}")
    print(f"  Test molecules : {len(test_set)}")

    train = [r for r in clean if r["name"] in train_set]
    val   = [r for r in clean if r["name"] in val_set]
    test  = [r for r in clean if r["name"] in test_set]

    # ---------- Fit coefficients ----------
    x_by_el = defaultdict(list)
    y_by_el = defaultdict(list)

    for r in train:
        for Zi, xi, yi in zip(r["z"], r["q_dftb"], r["q_mbis"]):
            sym = Z_TO_SYM.get(int(Zi))
            if sym:
                x_by_el[sym].append(float(xi))
                y_by_el[sym].append(float(yi))

    coeffs = {}
    for sym in sorted(x_by_el.keys()):
        a, b, se_a, se_b, n, r2 = fit_line(x_by_el[sym], y_by_el[sym])
        coeffs[sym] = {
            "a": a, "b": b,
            "se_a": se_a, "se_b": se_b,
            "n_train_atoms": n,
            "r2_train": r2
        }

    (outdir/"coefficients.json").write_text(
        json.dumps(coeffs, indent=2), encoding="utf-8"
    )

    # ---------- Evaluate ----------
    err_all = []
    mol_charge_err = []

    for r in test:
        q_pred = predict_elementwise(r["z"], r["q_dftb"], coeffs)
        q_true = np.asarray(r["q_mbis"], float)
        err_all.extend((q_pred - q_true).tolist())
        mol_charge_err.append(float(np.sum(q_pred) - np.sum(q_true)))

    metrics = {
        "atomwise_MAE_test": mae(err_all),
        "atomwise_RMSE_test": rmse(err_all),
        "mol_charge_MAE_test": mae(mol_charge_err),
        "mol_charge_RMSE_test": rmse(mol_charge_err),
    }

    (outdir/"metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print("Saved to:", outdir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
