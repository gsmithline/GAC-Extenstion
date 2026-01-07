"""
Solution Concepts for Cooperative Game-Theoretic Analysis of Attention Heads.

This module implements four cooperative game theory solution concepts for analyzing
attention head interactions in transformers:

1. Harsanyi Dividend: Measures synergistic effects of head coalitions
2. Shapley Value: Computes fair credit allocation to individual heads
3. Core: Identifies stable coalition configurations
4. Nucleolus: Finds allocations minimizing worst-case dissatisfaction

Reference:
    - Original GAC paper: Qu et al. (ACL 2025) "Cooperative or Competitive?
      Understanding the Interaction between Attention Heads From A Game Theory Perspective"
    - This extension: Comparing multiple solution concepts for attention head analysis
"""

from .harsanyi import compute_harsanyi_dividends, get_subsets
from .shapley import compute_shapley_values
from .core import compute_core, core_exists
from .nucleolus import compute_nucleolus
from .utils import (
    coalition_to_players,
    players_to_coalition,
    load_coalition_values,
    compare_solution_concepts,
)

__all__ = [
    # Harsanyi (from original GAC)
    "compute_harsanyi_dividends",
    "get_subsets",
    # Shapley (new)
    "compute_shapley_values",
    # Core (new)
    "compute_core",
    "core_exists",
    # Nucleolus (new)
    "compute_nucleolus",
    # Utilities
    "coalition_to_players",
    "players_to_coalition",
    "load_coalition_values",
    "compare_solution_concepts",
]

__version__ = "0.1.0"
