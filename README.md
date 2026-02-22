# ML Correction of DFTB Mulliken Charges to DFT-MBIS

This repository contains the data splits and training scripts used in:

> A minimal machine-learning correction of DFTB Mulliken charges towards DFT-level MBIS populations

The goal of this work is to construct a lightweight atom-wise neural correction mapping DFTB Mulliken charges to DFT MBIS charges.

---

## 1. Data

The dataset consists of QM9 molecules for which both DFT and DFTB calculations were performed.

### DFT reference

DFT calculations were performed using Psi4 at the B3LYP/def2-SVP level.
MBIS charges were extracted from the converged electron density and used as reference values.

### DFTB calculations

DFTB calculations were carried out using DFTB+ (DFTB3) with the 3ob Slater–Koster parameter set.
Mulliken charges were extracted from the self-consistent results.

Only this parameter set was used.

---

## 2. Train / Validation / Test split

To ensure exact reproducibility, the molecule IDs for each split are provided in:

splits/
    train.txt
    val.txt
    test.txt

These splits were used for all reported results.

---

## 3. Running the scripts

### Full geometric model

python scripts/train_mlp_geom.py \
--dftb data/dftb_qm9.jsonl \
--dft data/psi4_qm9.jsonl \
--split_dir splits \
--outdir results_geom


### No-geometry ablation model

python scripts/train_train_mlp_nogeom.py \
--dftb data/dftb_qm9.jsonl \
--dft data/psi4_qm9.jsonl \
--split_dir splits \
--outdir results_nogeom

### Baseline

python scripts/baseline_linear.py \
    --dftb data/dftb_qm9.jsonl \
    --dft data/psi4_qm9.jsonl \
    --splits_dir splits \
    --outdir baseline_results

---

## 4. Expected outputs

The scripts generate:

- metrics.json
- per_element_metrics.json
- predictions_test.jsonl
- model.pt

---

## 5. Reproducibility

All results in the manuscript correspond to the provided split files.
Using the same split files and default hyperparameters should reproduce the reported test-set metrics (within floating point variation).

