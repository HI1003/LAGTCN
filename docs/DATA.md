# Data contract

The code expects datasets below `Data/` by default. You may use another root
with `--data-root`.

```text
Data/
├── GEFCom2012_2level/
├── GEFCom2017QualifyingMatch_3level/
└── GEFCom2017FinalMatch_4level/
```

Each processed directory contains:

| File | Contract |
| --- | --- |
| `node_values.npy` | Float array `[T, N, 1]`, normalized using training data only |
| `normalization_params.npy` | Pickled dict describing normalization and `train_T` |
| `sum_matrix.csv` | Summing matrix `S`, shape `[N, N_bottom]` |
| `hierarchy_info.json` | Node order, level partition, and hierarchy counts |
| `adj_hierarchy.npy` | Structural hierarchy adjacency `[N, N]` |
| `adj_HGNN.npy` | Expanded fixed hierarchy graph `[N, N]` |
| raw target CSV | Timestamp in column 1, target nodes in hierarchy order |

Optional feature sets are `node_values_calendar.npy` and
`node_values_calendar_weather.npy`; build them with
`scripts/build_dataset_features.py` after the target-only arrays are ready.

The raw target CSV filenames are fixed by `code/main.py`:

| Dataset | Raw target CSV |
| --- | --- |
| `GEFCom2012_2level` | `Load_GEFCom2012_hourly.csv` |
| `GEFCom2017QualifyingMatch_3level` | `GEFCom2017QualifyingMatchDemand.csv` |
| `GEFCom2017FinalMatch_4level` | `load_final_filled.csv` |

The output-free notebooks and adjacency builders under `data_preparation/`
show the transformations used for the paper. Obtain the source datasets from
their original providers and comply with their terms. This repository does not
grant rights to redistribute those files.

For a loader-compatible software check without external data, run:

```bash
python scripts/make_synthetic_dataset.py
```

