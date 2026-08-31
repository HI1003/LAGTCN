# Paper reproduction workflow

Run every command from the repository root. Formal experiments use the three
datasets in [`DATA.md`](DATA.md), input length 168, horizon 24, seeds 42/43/44,
and effective batch size 128. The manifest builders freeze these values.

## 1. Verify data and code

Prepare the datasets, then run the relevant contract tests:

```bash
python -m unittest tests.test_data_loader_contract tests.test_target_timestamp_split -q
```

## 2. Select model hyperparameters on validation data

```bash
python -m reproduction.manifests.build_model_matrix \
  --matrix tuning \
  --manifest-dir results/raw_manifests
```

Each JSONL row contains a `cmd` array. Execute those arrays with the scheduler
available on your machine; all tuning rows include `--validation-only` and do
not evaluate the test partition. After all candidates terminate, select the
winner for every dataset/model group:

```bash
python -m reproduction.selection.select_model_hparams \
  --runs-root Data \
  --output configs/selected_hparams.generated.json
```

Selection minimizes validation Mean-1:24 all-level WAPE, using validation sMASE
only to break exact WAPE ties. The paper's frozen choices are in
`configs/selected_hparams.json`.

## 3. Select graph controls on validation data

```bash
python -m reproduction.manifests.build_graph_search \
  --selected-hparams configs/selected_hparams.json \
  --manifest-dir results/raw_manifests
```

Execute every generated command, then freeze the S threshold, adaptive top-k,
and D threshold:

```bash
python -m reproduction.selection.select_graph_hparams \
  --runs-root Data \
  --output configs/selected_graph_hparams.generated.json
```

The paper's frozen controls are in `configs/selected_graph_hparams.json`.

## 4. Train the final matrix

```bash
python -m reproduction.manifests.build_model_matrix \
  --matrix main \
  --selected-hparams configs/selected_hparams.json \
  --selected-graph-hparams configs/selected_graph_hparams.json \
  --manifest-dir results/raw_manifests
```

The complete manifest contains 144 runs: ten LAGTCN graph configurations plus
six baselines, across three datasets and three seeds. Execute every `cmd` array
from a clean Git commit. Commands use resumable epoch checkpoints and record
the source revision and experiment ID in each run.

## 5. Reconcile and validate forecasts

After all final runs finish, generate BU, TD-FP, and horizon-wise MinT-SHR from
the same saved Base trajectories:

```bash
python -m reproduction.evaluation.validate_final_matrix \
  --runs-root Data \
  --data-root Data \
  --summary results/final_postprocess_summary.json
```

This step does not retrain forecasting models. MinT-SHR covariance estimates
use validation residuals only; all reported forecasts are checked for finite
values, nonnegativity, node order, timestamp alignment, and hierarchy
coherence.

For an independent checkpoint-replay audit of the MinT-SHR inputs, use:

```bash
python -m reproduction.evaluation.mint_shrink \
  --manifest results/source_data/formal_run_manifest.csv
```

## 6. Benchmark frozen checkpoints

Run the unified batch-1 GPU benchmark after postprocessing:

```bash
python -m reproduction.evaluation.benchmark_checkpoints \
  --runs-root Data \
  --summary results/final_checkpoint_benchmark.csv
```

Formal checkpoint selection and accuracy metrics always operate on the
inverse-transformed load scale. The synthetic smoke dataset is never eligible
for paper comparisons.
