# LAGTCN

Official code for **LAGTCN: Level-Aware Graph-Temporal Co-Evolution with
Multiple Graph Sources for Hierarchical Electric Load Forecasting**.

The repository is deliberately limited to code required to run LAGTCN and
reproduce the paper. It includes the six reported baselines (DLinear, PatchTST,
N-HiTS, iTransformer, DCRNN, and MTGNN) and the three reconciliation methods
(BU, TD-FP, and MinT-SHR). Raw data, checkpoints, server launchers, exploratory
models, and unrelated experiments are not included.

[中文说明](README_zh-CN.md)

## Repository structure

```text
lagtcn/                       Source code
├── train.py                  Training, validation, and test entry point
├── core/                     Data loading, graph construction, metrics, training
├── models/                   LAGTCN and the six paper baselines
└── reconciliation/           BU, TD-FP, and MinT-SHR primitives
reproduction/                 Paper reproduction commands
├── data/                     Feature, structural-graph, and synthetic-data builders
├── manifests/                Model-matrix and graph-search manifest builders
├── selection/                Validation-only hyperparameter selection
└── evaluation/               Reconciliation, final validation, and benchmarking
configs/                      Frozen model and graph selections used in the paper
data_preparation/             Preprocessing notebooks for the three datasets
docs/                         Data contract and end-to-end reproduction guide
tests/                        Model, protocol, and reconciliation tests
results/                      Destination for small paper-facing artifacts
```

The model source code is under [`lagtcn/`](lagtcn/). Paper-level orchestration
is under [`reproduction/`](reproduction/).

## Environment

The frozen environment uses Python 3.10, PyTorch 2.3, PyTorch Geometric 2.5.3,
and CUDA 12.1:

```bash
conda create -n lagtcn python=3.10 -y
conda activate lagtcn
pip install -r requirements-cuda121.txt
```

For CPU or another CUDA version, install matching PyTorch/PyG wheels first and
then install `requirements.txt`.

## Formal paper configuration

The formal experiments use a 168-hour input window, a 24-hour forecast horizon,
three random seeds (42, 43, and 44), and effective batch size 128. These are the
defaults of the training entry point; the frozen per-dataset learning rates,
hidden sizes, and graph controls are stored in [`configs/`](configs/).

```bash
python -m lagtcn.train \
  --dataset GEFCom2012_2level \
  --model-name LAGTCN \
  --graph-mode H \
  --gnn-type gcn \
  --temporal-type patch_transformer \
  --num-timesteps-in 168 \
  --num-timesteps-out 24 \
  --batch-size 128
```

Use the generated manifests for reported experiments rather than manually
guessing the remaining hyperparameters.

## Quick CPU smoke test

This deliberately reduced `24→1`, batch-32 configuration is only a fast
software check; it is not a paper experiment.

```bash
python -m reproduction.data.make_synthetic

python -m lagtcn.train \
  --data-root Data \
  --dataset GEFCom2012_2level \
  --model-name LAGTCN \
  --graph-mode H \
  --gnn-type gcn \
  --temporal-type patch_transformer \
  --num-timesteps-in 24 \
  --num-timesteps-out 1 \
  --hidden-dim 16 \
  --num-layers 1 \
  --batch-size 32 \
  --epochs 1 \
  --patience 1 \
  --device cpu \
  --no-plots \
  --experiment-stage smoke \
  --experiment-id synthetic_smoke \
  --output-namespace ae/smoke
```

## Tests and full reproduction

```bash
python -m unittest discover -s tests -p 'test_*.py' -q
```

Prepare the three GEFCom datasets according to [`docs/DATA.md`](docs/DATA.md),
then follow [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Citation and license

Please use [`CITATION.cff`](CITATION.cff). Code is released under the
[MIT License](LICENSE); datasets retain their original providers' terms and are
not redistributed here.
