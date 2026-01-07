"""
Core computation for attention head coalition stability analysis.

The Core is the set of all payoff allocations where no coalition has an incentive
to "deviate" and form their own sub-coalition.

Mathematical Definition:
    Core = {x ∈ ℝⁿ : Σᵢxᵢ = v(N) and Σᵢ∈S xᵢ ≥ v(S) for all S⊆N}

Interpretation for Attention Heads:
    - Non-empty Core: There exists a stable configuration where all heads
      can "coexist" without tension
    - Empty Core: Inherent instability/competition; some heads will always
      "resent" the arrangement (connects to negative Harsanyi dividends)
    - Core points: Specific allocations that are stable against all deviations

Properties:
    - If Core is non-empty, it is convex and closed
    - Core ⊆ Imputation set (individually rational allocations)
    - Subadditive games always have non-empty Core
    - Superadditive games may have empty Core

Reference:
    Gillies, D.B. (1959). "Solutions to general non-zero-sum games"
"""

from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np

# Type aliases
Coalition = int
CoalitionValues = Dict[Coalition, float]
Allocation = Dict[int, float]  # Maps player index -> allocated value


def core_exists(
    v_values: CoalitionValues,
    num_players: int = 8
) -> bool:
    """
    Check if the Core is non-empty.

    Uses linear programming feasibility: the Core is non-empty iff
    there exists x such that:
        - Σᵢ xᵢ = v(N)
        - Σᵢ∈S xᵢ ≥ v(S) for all S ⊆ N

    Args:
        v_values: Dictionary mapping coalition bitmasks to values
        num_players: Number of players

    Returns:
        True if Core is non-empty, False otherwise
    """
    try:
        from scipy.optimize import linprog
    except ImportError:
        raise ImportError("scipy is required for Core computation. Install with: pip install scipy")

    n = num_players
    grand_coalition = (1 << n) - 1

    # Variables: x_1, ..., x_n
    # Objective: find any feasible point (minimize 0)
    c = [0] * n

    # Inequality constraints: -Σᵢ∈S xᵢ ≤ -v(S) for all proper subsets S
    A_ub = []
    b_ub = []

    for coalition in range(1, grand_coalition):  # Exclude empty and grand coalition
        # Get players in this coalition
        row = [-1 if (coalition >> i) & 1 else 0 for i in range(n)]
        A_ub.append(row)
        b_ub.append(-v_values.get(coalition, 0.0))

    # Equality constraint: Σᵢ xᵢ = v(N)
    A_eq = [[1] * n]
    b_eq = [v_values.get(grand_coalition, 0.0)]

    # Individual rationality: xᵢ ≥ v({i}) for all i
    bounds = [(v_values.get(1 << i, 0.0), None) for i in range(n)]

    result = linprog(
        c,
        A_ub=A_ub if A_ub else None,
        b_ub=b_ub if b_ub else None,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method='highs'
    )

    return result.success


def compute_core(
    v_values: CoalitionValues,
    num_players: int = 8
) -> Dict:
    """
    Compute the Core: check existence and find a point in the Core if it exists.

    Args:
        v_values: Dictionary mapping coalition bitmasks to values
        num_players: Number of players

    Returns:
        Dictionary with:
            - 'exists': bool indicating if Core is non-empty
            - 'point': Allocation dict if Core exists, None otherwise
            - 'binding_coalitions': List of coalitions where constraint is tight
    """
    try:
        from scipy.optimize import linprog
    except ImportError:
        raise ImportError("scipy is required for Core computation. Install with: pip install scipy")

    n = num_players
    grand_coalition = (1 << n) - 1

    # Variables: x_1, ..., x_n
    c = [0] * n

    # Build constraint matrices
    A_ub = []
    b_ub = []
    coalition_list = []

    for coalition in range(1, grand_coalition):
        row = [-1 if (coalition >> i) & 1 else 0 for i in range(n)]
        A_ub.append(row)
        b_ub.append(-v_values.get(coalition, 0.0))
        coalition_list.append(coalition)

    # Equality: sum = v(N)
    A_eq = [[1] * n]
    b_eq = [v_values.get(grand_coalition, 0.0)]

    # Bounds: xᵢ ≥ v({i})
    bounds = [(v_values.get(1 << i, 0.0), None) for i in range(n)]

    result = linprog(
        c,
        A_ub=A_ub if A_ub else None,
        b_ub=b_ub if b_ub else None,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method='highs'
    )

    if result.success:
        allocation = {i: result.x[i] for i in range(n)}

        # Find binding coalitions (where constraint is tight)
        binding = []
        tol = 1e-6
        for idx, coalition in enumerate(coalition_list):
            players_in_coalition = [i for i in range(n) if (coalition >> i) & 1]
            coalition_allocation = sum(allocation[i] for i in players_in_coalition)
            coalition_value = v_values.get(coalition, 0.0)
            if abs(coalition_allocation - coalition_value) < tol:
                binding.append(coalition)

        return {
            'exists': True,
            'point': allocation,
            'binding_coalitions': binding
        }
    else:
        return {
            'exists': False,
            'point': None,
            'binding_coalitions': []
        }


def find_core_center(
    v_values: CoalitionValues,
    num_players: int = 8
) -> Optional[Allocation]:
    """
    Find the analytic center of the Core (if it exists).

    The analytic center maximizes the product of slack variables,
    giving a point that is "most interior" to the Core.

    Args:
        v_values: Coalition values
        num_players: Number of players

    Returns:
        Allocation at the analytic center, or None if Core is empty
    """
    try:
        from scipy.optimize import minimize
    except ImportError:
        raise ImportError("scipy is required. Install with: pip install scipy")

    # First check if Core exists
    core_result = compute_core(v_values, num_players)
    if not core_result['exists']:
        return None

    n = num_players
    grand_coalition = (1 << n) - 1
    v_N = v_values.get(grand_coalition, 0.0)

    # Get all coalition constraints
    coalitions = list(range(1, grand_coalition))

    def negative_log_barrier(x):
        """Negative log of product of slacks (for minimization)."""
        total = 0.0
        for coalition in coalitions:
            players = [i for i in range(n) if (coalition >> i) & 1]
            slack = sum(x[i] for i in players) - v_values.get(coalition, 0.0)
            if slack <= 0:
                return 1e10  # Infeasible
            total -= np.log(slack)
        return total

    # Start from a feasible point
    x0 = np.array(list(core_result['point'].values()))

    # Equality constraint: sum = v(N)
    constraints = [{'type': 'eq', 'fun': lambda x: sum(x) - v_N}]

    # Bounds
    bounds = [(v_values.get(1 << i, 0.0), None) for i in range(n)]

    result = minimize(
        negative_log_barrier,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    if result.success:
        return {i: result.x[i] for i in range(n)}
    else:
        return core_result['point']  # Fall back to any Core point


def core_emptiness_by_layer(
    layer_v_values: Dict[int, CoalitionValues],
    num_players: int = 8
) -> Dict[int, bool]:
    """
    Check Core existence for each layer.

    Args:
        layer_v_values: Dictionary mapping layer index to coalition values
        num_players: Number of players per layer

    Returns:
        Dictionary mapping layer index to Core existence (True/False)

    Example:
        >>> results = core_emptiness_by_layer(layer_values)
        >>> empty_layers = [l for l, exists in results.items() if not exists]
    """
    results = {}
    for layer, v_values in layer_v_values.items():
        results[layer] = core_exists(v_values, num_players)
    return results
