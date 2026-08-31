"""Minimal LAGTCN forward pass without an external dataset."""

import numpy as np
import torch

from lagtcn import LAGTCN


node_count = 5
sum_matrix = np.vstack(
    [np.ones((1, node_count - 1)), np.eye(node_count - 1)]
).astype(np.float32)
hierarchy_adjacency = np.eye(node_count, dtype=np.float32)
hierarchy_adjacency[0, 1:] = 1.0
hierarchy_adjacency[1:, 0] = 1.0

model = LAGTCN(
    node_num=node_count,
    input_dim=1,
    hidden_dim=64,
    output_dim=24,
    num_layers=2,
    global_min=0.0,
    global_max=1.0,
    num_timesteps_in=168,
)
model.set_norm_params(
    {"norm_method": "zscore", "use_log": False, "mean": 0.0, "std": 1.0}
)
model.set_graph_config({"graph_mode": "H"})
model.set_static_graph_sources(hierarchy_adj=hierarchy_adjacency)
model.set_hierarchy_metadata(sum_matrix, middle_levels=[], bottom_start_idx=1)

features = torch.randn(2, node_count, 1, 168)
with torch.no_grad():
    prediction = model(features)

print(prediction.shape)  # torch.Size([2, 5, 24])
