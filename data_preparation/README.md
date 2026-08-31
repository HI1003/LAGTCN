# Data preparation

This directory contains the preprocessing notebooks and fixed-graph builders
used for the three GEFCom hierarchies. Notebook outputs and execution counters
have been removed. Raw data and derived arrays are intentionally absent.

Run the appropriate `DataProcessing.ipynb` after obtaining the original source
files, then build the two structural graphs from the repository root:

```bash
python -m reproduction.data.build_structural_graphs Data/<dataset>
```

Compare the generated directory with the contract in `docs/DATA.md` before
training.
