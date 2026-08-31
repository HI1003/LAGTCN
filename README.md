# LAGTCN

Official source code for **LAGTCN: Level-Aware Graph-Temporal Co-Evolution
with Multiple Graph Sources for Hierarchical Electric Load Forecasting**.

This repository provides the implementation of the LAGTCN model, including
the model architecture, data and graph pipeline, training and evaluation
entry point, and the reported BU, TD-FP, and MinT-SHR reconciliation methods.

[中文说明](README_zh-CN.md)

## Source layout

```text
lagtcn/
├── model.py              LAGTCN architecture and forward pass
├── data.py               GEFCom loading and leakage-safe 80/10/10 windows
├── graphs.py             H/HG construction and S/A/D graph sources
├── train.py              Training, validation, checkpointing, and evaluation
├── metrics.py            Forecast and coherence metrics
└── reconciliation.py     BU, TD-FP, and MinT-SHR
data_preparation/          One preprocessing notebook per paper dataset
examples/forward_pass.py  LAGTCN forward-pass example
```

The model implementation is in [`lagtcn/model.py`](lagtcn/model.py). The three
post-hoc methods are all in
[`lagtcn/reconciliation.py`](lagtcn/reconciliation.py).

## Installation

The code targets Python 3.10 and PyTorch 2.3.

```bash
conda create -n lagtcn python=3.10 -y
conda activate lagtcn
pip install -r requirements.txt
```

For the frozen CUDA 12.1 PyTorch wheel, use
`requirements-cuda121.txt` instead.

## Data

Place each preprocessed dataset under `Data/`:

```text
Data/<dataset>/
├── node_values.npy
├── normalization_params.npy
├── sum_matrix.csv
├── hierarchy_info.json
├── hierarchy.csv
├── adj_hierarchy.npy
├── adj_HGNN.npy
└── <timestamped source CSV>
```

The three dataset names are `GEFCom2012_2level`,
`GEFCom2017QualifyingMatch_3level`, and `GEFCom2017FinalMatch_4level`.
Run the matching notebook in [`data_preparation/`](data_preparation/), then
build the fixed H/HG graphs with:

```bash
python -m lagtcn.graphs Data/<dataset>
```

Raw datasets are not redistributed and remain subject to their providers'
terms.

## Training

The training defaults are the paper protocol: a 168-hour input window, a
24-hour direct forecast horizon, and physical batch size 128.

```bash
python -m lagtcn.train \
  --data-root Data \
  --dataset GEFCom2012_2level \
  --graph-mode H \
  --num-timesteps-in 168 \
  --num-timesteps-out 24 \
  --batch-size 128
```

Modes containing S, A, or D additionally require their validation-selected
controls: `--static-threshold`, `--adaptive-top-k`, or
`--dynamic-threshold`, respectively. The command writes the best checkpoint,
configuration, metrics, validation predictions, and test base predictions to
a timestamped directory below `Data/<dataset>/output/`.

A 168→24 forward-pass example is also available:

```bash
python -m examples.forward_pass
```

## Reconciliation

Set `RUN` to a completed training directory. BU and TD-FP use the test base
forecast directly. MinT-SHR estimates its shrinkage covariance from the saved
validation residuals and then applies it to the test forecast.

```bash
RUN=Data/GEFCom2012_2level/output/<run>

python -m lagtcn.reconciliation \
  --method bu \
  --base-archive "$RUN/base_predictions.npz" \
  --sum-matrix Data/GEFCom2012_2level/sum_matrix.csv \
  --output "$RUN/reconciled_bu.npz"

python -m lagtcn.reconciliation \
  --method td_fp \
  --base-archive "$RUN/base_predictions.npz" \
  --sum-matrix Data/GEFCom2012_2level/sum_matrix.csv \
  --output "$RUN/reconciled_td_fp.npz"

python -m lagtcn.reconciliation \
  --method mint_shrink \
  --base-archive "$RUN/base_predictions.npz" \
  --validation-archive "$RUN/validation_predictions.npz" \
  --sum-matrix Data/GEFCom2012_2level/sum_matrix.csv \
  --output "$RUN/reconciled_mint_shrink.npz"
```

Every method clips or solves in bottom space and reconstructs all levels with
`y = S b`, so its output is nonnegative and coherent by construction.

## Citation and license

Please use [`CITATION.cff`](CITATION.cff). Code is released under the
[MIT License](LICENSE).
