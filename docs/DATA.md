# Data contract

The training entry reads datasets from `Data/` by default; use `--data-root`
to select another location.

```text
Data/
├── GEFCom2012_2level/
├── GEFCom2017QualifyingMatch_3level/
└── GEFCom2017FinalMatch_4level/
```

Each processed directory must contain:

| File | Contract |
| --- | --- |
| `node_values.npy` | Float array `[T, N, 1]`, normalized from training data only |
| `normalization_params.npy` | Normalization dictionary including the frozen `train_T` |
| `sum_matrix.csv` | Summing matrix `S`, shape `[N, N_bottom]` |
| `hierarchy_info.json` | Node order, level partition, and hierarchy counts |
| `hierarchy.csv` | Root-to-leaf paths used to rebuild the structural graphs |
| `adj_hierarchy.npy` | Structural hierarchy adjacency `[N, N]` |
| `adj_HGNN.npy` | Expanded fixed hierarchy graph `[N, N]` |
| raw target CSV | Timestamp first, followed by targets in hierarchy node order |

The expected raw target files are:

| Dataset | Raw target CSV |
| --- | --- |
| `GEFCom2012_2level` | `Load_GEFCom2012_hourly.csv` |
| `GEFCom2017QualifyingMatch_3level` | `GEFCom2017QualifyingMatchDemand.csv` |
| `GEFCom2017FinalMatch_4level` | `load_final_filled.csv` |

Use the matching `DataProcessing.ipynb` under
[`data_preparation/`](../data_preparation/) after obtaining the original data.
The notebook establishes the node order, summing matrix, normalization, and
time split. Then generate the fixed structural graphs with the shared command:

```bash
python -m reproduction.data.build_structural_graphs Data/<dataset>
```

Optional calendar/weather arrays can be rebuilt after the target-only arrays:

```bash
python -m reproduction.data.build_features --help
```

Raw datasets and generated arrays are intentionally excluded from Git. Obtain
the source data from its original provider and follow its redistribution terms.

For a loader-compatible software check that does not reproduce paper results:

```bash
python -m reproduction.data.make_synthetic
```
