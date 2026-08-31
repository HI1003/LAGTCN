from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from lagtcn.core.naming import lagtcn_graph_sources


class LAGTCNGraphSourceMappingTest(unittest.TestCase):
    def test_supported_modes_map_to_independent_sources(self) -> None:
        expected = {
            "I": ("identity",),
            "H": ("hierarchy",),
            "HG": ("hierarchy",),
            "S": ("similarity",),
            "A": ("adaptive",),
            "D": ("dynamic",),
            "H+S": ("hierarchy", "similarity"),
            "H+A": ("hierarchy", "adaptive"),
            "H+D": ("hierarchy", "dynamic"),
            "S+A+D": ("similarity", "adaptive", "dynamic"),
            "H+S+A+D": ("hierarchy", "similarity", "adaptive", "dynamic"),
        }
        for graph_mode, sources in expected.items():
            with self.subTest(graph_mode=graph_mode):
                self.assertEqual(lagtcn_graph_sources(graph_mode), sources)

    def test_identity_is_not_mixed_into_informative_modes(self) -> None:
        for graph_mode in ("A", "D", "H+A", "S+A+D", "H+S+A+D"):
            with self.subTest(graph_mode=graph_mode):
                self.assertNotIn("identity", lagtcn_graph_sources(graph_mode))

    def test_aliases_are_normalized(self) -> None:
        self.assertEqual(lagtcn_graph_sources("identity"), ("identity",))
        self.assertEqual(
            lagtcn_graph_sources("dynamic + hierarchy + similarity + adaptive"),
            ("hierarchy", "similarity", "adaptive", "dynamic"),
        )


if __name__ == "__main__":
    unittest.main()
