# Reproducibility workflow

## 1. Prepare data

Create the three dataset directories described in [DATA.md](DATA.md). Validate
the loader and hierarchy contract first:

```bash
python -m pytest -q tests/test_data_loader_contract.py tests/test_target_timestamp_split.py
```

## 2. Model hyperparameter selection

Build the validation-only tuning manifest:

```bash
python scripts/build_ae_final_manifest.py \
  --matrix tuning \
  --manifest-dir results/raw_manifests
```

Select one learning-rate/hidden-size pair for each model and dataset using only
validation WAPE, with validation sMASE as the exact-tie breaker:

```bash
python scripts/select_ae_final_hparams.py \
  --output configs/selected_hparams.generated.json
```

`configs/selected_hparams.json` records the frozen selections used in the paper.

## 3. Graph-control selection

Generate all validation-only S/A/D candidates:

```bash
python scripts/build_ae_graph_tuning_manifest.py \
  --selected-hparams configs/selected_hparams.json \
  --manifest-dir results/raw_manifests
```

Then select the controls:

```bash
python scripts/select_ae_graph_hparams.py \
  --output configs/selected_graph_hparams.generated.json
```

The frozen choices are in `configs/selected_graph_hparams.json`.

## 4. Final matrix

Build the final manifests from both frozen configuration files. Use a clean Git
commit and retain the commit, branch, experiment ID, and dirty-state provenance
written by the training entry point. Long jobs should use `--resume auto` and
epoch checkpoints.

The paper protocol uses 168 input hours, a 24-hour horizon, three seeds, ten
LAGTCN graph configurations, and the external baseline matrix. The generator
performs completeness checks before writing the manifest.

## 5. Reconciliation and evaluation

Use `scripts/postprocess_ae_phase1.py` and
`scripts/postprocess_ae_mint_shrink.py` to build BU, TD-FP, and MinT-SHR results
from common base trajectories. Use `scripts/benchmark_ae_final.py` to replay
selected checkpoints under a unified inference benchmark.

All formal checkpoint selection and reported accuracy metrics operate in the
inverse-transformed load scale. Graph controls are selected on validation data
only. Do not use the synthetic smoke-test data for scientific comparison.

