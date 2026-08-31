#!/usr/bin/env python3
"""Regenerate only the fixed H and HG graph sources for this dataset."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_structural_graphs import write_structural_graphs


if __name__ == "__main__":
    result = write_structural_graphs(Path(__file__).resolve().parent)
    print(f"Saved H and HG with {result['num_nodes']} nodes.")
