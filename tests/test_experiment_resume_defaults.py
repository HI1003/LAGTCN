from __future__ import annotations

import sys
import unittest
from argparse import Namespace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_experiment_matrix import _build_command
from graph_sparsity import FINAL_GRAPH_SOURCE_POLICY


def _args(*, resume: str = "auto", checkpoint_every_epochs: int = 1) -> Namespace:
    return Namespace(
        stage="stgnn",
        batch_size=16,
        num_timesteps_in=168,
        hidden_dim=128,
        num_layers=2,
        lr=1e-3,
        epochs=150,
        patience=20,
        resume=resume,
        checkpoint_every_epochs=checkpoint_every_epochs,
        sim_type="cosine",
        graph_sparsity_policy=FINAL_GRAPH_SOURCE_POLICY,
        device="cuda:0",
        paper_scope="journal_applied_energy",
        experiment_id="resume-default-test",
        output_namespace="ae/test",
        static_threshold=None,
        adaptive_top_k=None,
        dynamic_threshold=None,
        feature_set="target",
        no_plots=True,
        plot_node_limit=None,
    )


def _option_value(command: list[str], option: str) -> str:
    index = command.index(option)
    return command[index + 1]


class ExperimentResumeDefaultTest(unittest.TestCase):
    def test_training_command_enables_auto_resume_and_epoch_checkpoints(self) -> None:
        command = _build_command(
            _args(),
            dataset="GEFCom2012_2level",
            seed=42,
            graph_mode="H",
            model_name="DCRNN",
            horizon=1,
            gnn_type="gcn",
            temporal_type="gru",
        )

        self.assertEqual(_option_value(command, "--resume"), "auto")
        self.assertEqual(_option_value(command, "--checkpoint-every-epochs"), "1")

    def test_clean_restart_can_be_requested_explicitly(self) -> None:
        command = _build_command(
            _args(resume="none", checkpoint_every_epochs=0),
            dataset="GEFCom2012_2level",
            seed=42,
            graph_mode="H",
            model_name="DCRNN",
            horizon=1,
            gnn_type="gcn",
            temporal_type="gru",
        )

        self.assertEqual(_option_value(command, "--resume"), "none")
        self.assertEqual(_option_value(command, "--checkpoint-every-epochs"), "0")


if __name__ == "__main__":
    unittest.main()
