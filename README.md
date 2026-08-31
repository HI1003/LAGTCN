# LAGTCN

Official research code for **LAGTCN: Level-Aware Graph-Temporal Co-Evolution
with Multiple Graph Sources for Hierarchical Electric Load Forecasting**.

LAGTCN combines level-aware temporal encoding, relation-specific graph message
passing, and graph-temporal co-evolution for hierarchical load forecasting. The
repository also contains the baselines, hierarchical reconciliation methods,
and frozen experiment protocol used in the accompanying Applied Energy study.

## Repository layout

```text
code/                Model, data loader, training, evaluation, reconciliation
configs/             Frozen model and graph-control selections
data_preparation/    Output-free notebooks and graph-building scripts
docs/                Data and full-reproduction instructions
scripts/             Manifest generation, selection, and post-processing
tests/               Unit and protocol tests
results/             Location for shareable, paper-facing result artifacts
```

Raw GEFCom files, processed arrays, checkpoints, and experiment output are not
stored in Git. See [docs/DATA.md](docs/DATA.md) for the expected data contract.

## Environment

The frozen experiment environment uses Python 3.10, PyTorch 2.3, PyTorch
Geometric 2.5.3, and CUDA 12.1:

```bash
conda create -n lagtcn python=3.10 -y
conda activate lagtcn
pip install -r requirements-cuda121.txt
pip install pytest
```

For CPU or another CUDA version, install the matching PyTorch and PyG wheels
first, then install the version-pinned scientific packages in `requirements.txt`.

## Quick CPU smoke test

Generate a deterministic, coherent 21-node hierarchy with the same loader
contract as the two-level dataset:

```bash
python scripts/make_synthetic_dataset.py
```

Then train a deliberately small LAGTCN instance for one epoch:

```bash
python code/main.py \
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

The synthetic dataset is only a software check; it must not be used to support
scientific claims.

## Tests

```bash
python -m pytest -q
```

## Full reproduction

After preparing the three formal GEFCom datasets, follow
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md). The protocol separates
model tuning, validation-only graph selection, final training, post-hoc
reconciliation, and checkpoint replay.

## Citation

Please use [CITATION.cff](CITATION.cff). The manuscript DOI will be added after
publication.

## License

Code is released under the [MIT License](LICENSE). Dataset files retain their
original providers' terms and are not redistributed here.

