"""
Utility functions for cooperative game-theoretic analysis of attention heads.

This module provides helper functions for:
- Converting between coalition representations
- Loading and saving coalition values
- Comparing different solution concepts
- Visualization helpers
"""

import os
import json
from typing import Dict, List, Set, Tuple, Optional, Any
from pathlib import Path

# Type aliases
Coalition = int
CoalitionValues = Dict[Coalition, float]


def coalition_to_players(coalition: Coalition, num_players: int = 8) -> Set[int]:
    """
    Convert a coalition bitmask to a set of player indices.

    Args:
        coalition: Bitmask representation (e.g., 5 = 0b101)
        num_players: Total number of players

    Returns:
        Set of player indices in the coalition

    Example:
        >>> coalition_to_players(5)  # 0b101
        {0, 2}
    """
    return {i for i in range(num_players) if (coalition >> i) & 1}


def players_to_coalition(players: Set[int]) -> Coalition:
    """
    Convert a set of player indices to a coalition bitmask.

    Args:
        players: Set of player indices

    Returns:
        Coalition bitmask

    Example:
        >>> players_to_coalition({0, 2})
        5  # 0b101
    """
    return sum(1 << i for i in players)


def coalition_to_string(coalition: Coalition, num_players: int = 8) -> str:
    """
    Convert coalition bitmask to human-readable string.

    Args:
        coalition: Coalition bitmask
        num_players: Total number of players

    Returns:
        String representation like "{0, 2, 5}"

    Example:
        >>> coalition_to_string(5)
        "{0, 2}"
    """
    players = coalition_to_players(coalition, num_players)
    if not players:
        return "{}"
    return "{" + ", ".join(str(p) for p in sorted(players)) + "}"


def load_coalition_values(
    file_path: str,
    num_players: int = 8
) -> CoalitionValues:
    """
    Load coalition values from a log file in GAC format.

    Expected format: each line contains "coalition_bitmask value"

    Args:
        file_path: Path to the log file
        num_players: Number of players (for validation)

    Returns:
        Dictionary mapping coalition bitmask to value

    Example file format:
        0 0.0
        1 0.123456
        2 0.234567
        ...
        255 0.987654
    """
    v_values: CoalitionValues = {}
    max_coalition = (1 << num_players) - 1

    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            coalition = int(parts[0])
            value = float(parts[1])
            if 0 <= coalition <= max_coalition:
                v_values[coalition] = value

    return v_values


def save_coalition_values(
    v_values: CoalitionValues,
    file_path: str
) -> None:
    """
    Save coalition values to a log file.

    Args:
        v_values: Dictionary mapping coalition bitmask to value
        file_path: Output file path
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'w') as f:
        for coalition in sorted(v_values.keys()):
            f.write(f"{coalition} {v_values[coalition]:.6f}\n")


def load_results_json(file_path: str) -> Dict[str, Any]:
    """
    Load results from JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        Loaded dictionary
    """
    with open(file_path, 'r') as f:
        return json.load(f)


def save_results_json(
    results: Dict[str, Any],
    file_path: str,
    indent: int = 2
) -> None:
    """
    Save results to JSON file.

    Args:
        results: Dictionary to save
        file_path: Output file path
        indent: JSON indentation level
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'w') as f:
        json.dump(results, f, indent=indent)


def compare_solution_concepts(
    v_values: CoalitionValues,
    num_players: int = 8
) -> Dict[str, Any]:
    """
    Compute and compare all four solution concepts for a given game.

    Args:
        v_values: Dictionary mapping coalition bitmasks to values
        num_players: Number of players

    Returns:
        Dictionary containing:
            - 'harsanyi': Harsanyi dividends for all coalitions
            - 'shapley': Shapley values for all players
            - 'core': Core existence and a point if exists
            - 'nucleolus': Nucleolus allocation
            - 'comparison': Analysis comparing the concepts
    """
    from .harsanyi import compute_harsanyi_dividends, get_top_k_coalitions
    from .shapley import compute_shapley_values, rank_players_by_shapley
    from .core import compute_core
    from .nucleolus import compute_nucleolus, get_excess_distribution

    results = {}

    # Compute Harsanyi dividends
    harsanyi = compute_harsanyi_dividends(v_values, num_players)
    results['harsanyi'] = {
        'dividends': harsanyi,
        'top_positive': get_top_k_coalitions(harsanyi, k=10, positive_only=True),
        'top_negative': get_top_k_coalitions(harsanyi, k=10, positive_only=False),
        'num_positive': sum(1 for w in harsanyi.values() if w > 0),
        'num_negative': sum(1 for w in harsanyi.values() if w < 0),
    }

    # Compute Shapley values
    shapley = compute_shapley_values(v_values, num_players)
    results['shapley'] = {
        'values': shapley,
        'ranking': rank_players_by_shapley(shapley),
    }

    # Compute Core
    core_result = compute_core(v_values, num_players)
    results['core'] = core_result

    # Compute Nucleolus
    nucleolus = compute_nucleolus(v_values, num_players)
    results['nucleolus'] = {
        'allocation': nucleolus,
        'excess_distribution': get_excess_distribution(nucleolus, v_values, num_players)
        if nucleolus else None
    }

    # Comparison analysis
    results['comparison'] = analyze_concept_differences(
        shapley, nucleolus, harsanyi, core_result, num_players
    )

    return results


def analyze_concept_differences(
    shapley: Dict[int, float],
    nucleolus: Optional[Dict[int, float]],
    harsanyi: Dict[int, float],
    core_result: Dict,
    num_players: int = 8
) -> Dict[str, Any]:
    """
    Analyze differences between solution concepts.

    Args:
        shapley: Shapley values
        nucleolus: Nucleolus allocation
        harsanyi: Harsanyi dividends
        core_result: Core computation result
        num_players: Number of players

    Returns:
        Dictionary with comparison metrics
    """
    analysis = {}

    # Ranking correlation between Shapley and Nucleolus
    if nucleolus:
        shapley_ranking = sorted(range(num_players), key=lambda i: shapley[i], reverse=True)
        nucleolus_ranking = sorted(range(num_players), key=lambda i: nucleolus[i], reverse=True)

        # Spearman correlation (simplified)
        rank_diff_squared = sum(
            (shapley_ranking.index(i) - nucleolus_ranking.index(i)) ** 2
            for i in range(num_players)
        )
        n = num_players
        spearman = 1 - (6 * rank_diff_squared) / (n * (n**2 - 1)) if n > 1 else 1.0

        analysis['shapley_nucleolus_correlation'] = spearman
        analysis['shapley_ranking'] = shapley_ranking
        analysis['nucleolus_ranking'] = nucleolus_ranking

    # Harsanyi statistics
    positive_dividends = [w for w in harsanyi.values() if w > 0]
    negative_dividends = [w for w in harsanyi.values() if w < 0]

    analysis['harsanyi_stats'] = {
        'num_positive': len(positive_dividends),
        'num_negative': len(negative_dividends),
        'sum_positive': sum(positive_dividends) if positive_dividends else 0,
        'sum_negative': sum(negative_dividends) if negative_dividends else 0,
    }

    # Core analysis
    analysis['core_exists'] = core_result['exists']
    analysis['num_binding_coalitions'] = len(core_result.get('binding_coalitions', []))

    return analysis


def compute_rank_correlation(
    ranking1: List[int],
    ranking2: List[int]
) -> float:
    """
    Compute Spearman rank correlation between two rankings.

    Args:
        ranking1: First ranking (list of player indices in rank order)
        ranking2: Second ranking

    Returns:
        Spearman correlation coefficient (-1 to 1)
    """
    n = len(ranking1)
    if n != len(ranking2):
        raise ValueError("Rankings must have same length")
    if n <= 1:
        return 1.0

    rank_diff_squared = sum(
        (ranking1.index(i) - ranking2.index(i)) ** 2
        for i in range(n)
    )

    return 1 - (6 * rank_diff_squared) / (n * (n**2 - 1))
