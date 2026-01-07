"""
Nucleolus computation for attention head fairness analysis.

The Nucleolus finds the allocation that lexicographically minimizes the sorted
vector of excesses (complaints) across all coalitions.

Mathematical Definition:
    Excess of coalition S at allocation x: e(S, x) = v(S) - Σᵢ∈S xᵢ
    Nucleolus: x* = argmin_x [lexicographically minimize sorted excess vector]

Interpretation for Attention Heads:
    - High excess: Coalition S is "undervalued" relative to what it could achieve alone
    - The Nucleolus minimizes the loudest "complaint" first, then the second-loudest, etc.
    - Heads with high excess under the Nucleolus are underutilized (potential pruning targets)

Properties:
    - Always exists and is unique
    - Always in the Core (if Core is non-empty)
    - Satisfies individual rationality
    - Computed via sequential linear programming

Reference:
    Schmeidler, D. (1969). "The nucleolus of a characteristic function game"
"""

from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np

# Type aliases
Coalition = int
CoalitionValues = Dict[Coalition, float]
Allocation = Dict[int, float]


def compute_excess(
    allocation: Allocation,
    v_values: CoalitionValues,
    num_players: int = 8
) -> Dict[Coalition, float]:
    """
    Compute the excess for each coalition at a given allocation.

    Excess e(S, x) = v(S) - Σᵢ∈S xᵢ

    Positive excess means coalition S is "unsatisfied" - they could do better alone.
    Negative excess means coalition S is "satisfied" - they get more than they could alone.

    Args:
        allocation: Current allocation to each player
        v_values: Coalition values
        num_players: Number of players

    Returns:
        Dictionary mapping coalition to its excess
    """
    excesses = {}
    grand_coalition = (1 << num_players) - 1

    for coalition in range(1, grand_coalition):  # Exclude empty and grand coalition
        players = [i for i in range(num_players) if (coalition >> i) & 1]
        coalition_allocation = sum(allocation.get(i, 0.0) for i in players)
        coalition_value = v_values.get(coalition, 0.0)
        excesses[coalition] = coalition_value - coalition_allocation

    return excesses


def compute_nucleolus(
    v_values: CoalitionValues,
    num_players: int = 8,
    tol: float = 1e-9,
    max_iterations: int = 100
) -> Optional[Allocation]:
    """
    Compute the Nucleolus using Maschler's sequential LP algorithm.

    Algorithm:
    1. Start with all coalitions active
    2. Find allocation minimizing maximum excess
    3. Fix coalitions achieving that maximum excess
    4. Repeat with remaining coalitions until unique

    Args:
        v_values: Dictionary mapping coalition bitmasks to values
        num_players: Number of players
        tol: Numerical tolerance
        max_iterations: Maximum number of LP iterations

    Returns:
        Allocation dictionary mapping player index to value, or None if infeasible

    Example:
        >>> v = {0: 0, 1: 0, 2: 0, 4: 0, 3: 1, 5: 1, 6: 0, 7: 1}  # Majority game
        >>> nuc = compute_nucleolus(v, num_players=3)
        >>> nuc
        {0: 0.333, 1: 0.333, 2: 0.333}
    """
    try:
        from scipy.optimize import linprog
    except ImportError:
        raise ImportError("scipy is required for Nucleolus computation")

    n = num_players
    grand_coalition = (1 << n) - 1
    v_N = v_values.get(grand_coalition, 0.0)

    # Build list of all proper coalitions (excluding empty and grand)
    all_coalitions = list(range(1, grand_coalition))
    m = len(all_coalitions)

    # Track fixed excesses from previous iterations
    fixed_constraints = []  # List of (coalition_indices, excess_value)
    active_coalitions = list(range(m))  # Indices into all_coalitions

    x = None

    for iteration in range(max_iterations):
        if not active_coalitions:
            break

        num_active = len(active_coalitions)

        # Variables: [x_1, ..., x_n, epsilon]
        # Minimize epsilon (the maximum excess)
        c = [0] * n + [1]

        # Constraints for active coalitions:
        # v(S) - Σᵢ∈S xᵢ ≤ epsilon
        # Rearranged: -Σᵢ∈S xᵢ - epsilon ≤ -v(S)
        A_ub = []
        b_ub = []

        for idx in active_coalitions:
            coalition = all_coalitions[idx]
            row = [-1 if (coalition >> i) & 1 else 0 for i in range(n)] + [-1]
            A_ub.append(row)
            b_ub.append(-v_values.get(coalition, 0.0))

        # Constraints for fixed coalitions from previous iterations
        for fixed_indices, eps_val in fixed_constraints:
            for idx in fixed_indices:
                coalition = all_coalitions[idx]
                row = [-1 if (coalition >> i) & 1 else 0 for i in range(n)] + [0]
                A_ub.append(row)
                b_ub.append(-v_values.get(coalition, 0.0) + eps_val + tol)

        # Equality: sum(x) = v(N)
        A_eq = [[1] * n + [0]]
        b_eq = [v_N]

        # Bounds: xᵢ ≥ v({i}), epsilon unbounded
        bounds = [(v_values.get(1 << i, 0.0), None) for i in range(n)]
        bounds.append((None, None))  # epsilon

        result = linprog(
            c,
            A_ub=A_ub if A_ub else None,
            b_ub=b_ub if b_ub else None,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method='highs'
        )

        if not result.success:
            break

        x = result.x[:n]
        eps_star = result.x[n]

        # Find coalitions achieving this maximum excess (within tolerance)
        newly_fixed = []
        still_active = []

        for idx in active_coalitions:
            coalition = all_coalitions[idx]
            players = [i for i in range(n) if (coalition >> i) & 1]
            excess = v_values.get(coalition, 0.0) - sum(x[i] for i in players)

            if abs(excess - eps_star) < tol:
                newly_fixed.append(idx)
            else:
                still_active.append(idx)

        if not newly_fixed:
            # No progress - should not happen in exact arithmetic
            break

        fixed_constraints.append((newly_fixed, eps_star))
        active_coalitions = still_active

    if x is not None:
        return {i: float(x[i]) for i in range(n)}
    else:
        return None


def get_excess_distribution(
    nucleolus: Allocation,
    v_values: CoalitionValues,
    num_players: int = 8
) -> List[Tuple[Coalition, float]]:
    """
    Get the sorted excess distribution at the Nucleolus allocation.

    Args:
        nucleolus: The Nucleolus allocation
        v_values: Coalition values
        num_players: Number of players

    Returns:
        List of (coalition, excess) tuples sorted by excess (descending)
    """
    excesses = compute_excess(nucleolus, v_values, num_players)
    return sorted(excesses.items(), key=lambda x: x[1], reverse=True)


def identify_undervalued_players(
    nucleolus: Allocation,
    v_values: CoalitionValues,
    num_players: int = 8,
    threshold: float = 0.0
) -> List[int]:
    """
    Identify players who appear in many high-excess coalitions.

    These players are "undervalued" - they contribute to coalitions that
    could do better, suggesting they may be underutilized.

    Args:
        nucleolus: The Nucleolus allocation
        v_values: Coalition values
        num_players: Number of players
        threshold: Only consider coalitions with excess above this threshold

    Returns:
        List of player indices sorted by how often they appear in high-excess coalitions
    """
    excesses = compute_excess(nucleolus, v_values, num_players)
    high_excess = {c: e for c, e in excesses.items() if e > threshold}

    # Count how often each player appears in high-excess coalitions
    player_counts = {i: 0 for i in range(num_players)}
    player_total_excess = {i: 0.0 for i in range(num_players)}

    for coalition, excess in high_excess.items():
        for i in range(num_players):
            if (coalition >> i) & 1:
                player_counts[i] += 1
                player_total_excess[i] += excess

    # Sort by total excess contribution
    sorted_players = sorted(
        range(num_players),
        key=lambda i: player_total_excess[i],
        reverse=True
    )

    return sorted_players
