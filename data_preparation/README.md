# Data preparation

This directory contains one preprocessing notebook for each of the three
GEFCom hierarchies. Notebook outputs and execution counters have been removed.
Raw data and derived arrays are intentionally absent.

Run the appropriate `DataProcessing.ipynb` after obtaining the original source
files and ``hierarchy.csv``. Each notebook writes ``node_values.npy``,
``normalization_params.npy``, ``sum_matrix.csv``, and ``hierarchy_info.json``.
Then build the two structural graphs from the repository root:

```bash
python -m lagtcn.graphs Data/<dataset>
```

The resulting directory must also retain the timestamped CSV used by
``lagtcn.data.HierarchicalLoadDataset``. See the data layout in the repository
README before training.
