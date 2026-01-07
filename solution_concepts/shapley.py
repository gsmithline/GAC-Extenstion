"""
Shapley Value computation for attention head importance.

The Shapley value measures each player's average marginal contribution across
all possible orderings of players joining the coalition.

Mathematical Definition:
    For player i in game v with player set N:
    phi_i = sum over S in N minus i of [|S|!(|N|-|S|-1)!/|N|!] * [v(S union i) - v(S)]

Interpretation for Attention Heads:
    - High Shapley value: Head is consistently useful regardless of which other heads are active
    - Low Shapley value: Head is redundant or only useful in specific combinations
    - The Shapley value provides a "fair" attribution of credit to each head

Properties:
    - Efficiency: Σᵢ φᵢ = v(N)
    - Symmetry: Symmetric players receive equal values
    - Dummy: Dummy players receive zero
    - Additivity: φᵢ(v + w) = φᵢ(v) + φᵢ(w)

Reference:
    Shapley, L.S. (1953). "A value for n-person games"
"""

from itertools import combinations
from math import factorial
from typing import Dict, List

# Type aliases
Coalition = int
CoalitionValues = Dict[Coalition, float]
ShapleyValues = Dict[int, float]  # Maps player index -> Shapley value


def compute_shapley_values(
    v_values: CoalitionValues,
    num_players: int = 8
) -> ShapleyValues:
    """
    Compute the Shapley value for each player.

    The Shapley value φᵢ is the weighted average of player i's marginal contributions
    across all possible coalition orderings.

    Args:
        v_values: Dictionary mapping coalition bitmasks to their values v(S).
                  Coalition bitmask uses bit i to indicate player i is in the coalition.
        num_players: Number of players in the game (default: 8 for attention heads)

    Returns:
        Dictionary mapping player index (0 to num_players-1) to their Shapley value

    Example:
        >>> v = {0: 0, 1: 10, 2: 15, 3: 30}  # 2-player game
        >>> phi = compute_shapley_values(v, num_players=2)
        >>> phi[0]  # Player 0's Shapley value
        12.5  # = 0.5 * (v({0}) - v({})) + 0.5 * (v({0,1}) - v({1}))
              # = 0.5 * (10 - 0) + 0.5 * (30 - 15) = 5 + 7.5 = 12.5
    """
    n = num_players
    n_factorial = factorial(n)
    shapley = {i: 0.0 for i in range(n)}

    for i in range(n):
        # Get all other players
        other_players = [j for j in range(n) if j != i]

        # Iterate over all subsets S ⊆ N \ {i}
        for size in range(n):  # |S| from 0 to n-1
            for combo in combinations(other_players, size):
                # Convert combo to coalition bitmask
                S = sum(1 << j for j in combo)
                S_with_i = S | (1 << i)

                # Get coalition values
                v_S = v_values.get(S, 0.0)
                v_S_with_i = v_values.get(S_with_i, 0.0)

                # Marginal contribution
                marginal = v_S_with_i - v_S

                # Shapley weight: |S|!(n-|S|-1)!/n!
                weight = factorial(size) * factorial(n - size - 1) / n_factorial

                shapley[i] += weight * marginal

    return shapley


def compute_shapley_from_harsanyi(
    harsanyi_dividends: Dict[Coalition, float],
    num_players: int = 8
) -> ShapleyValues:
    """
    Compute Shapley values directly from Harsanyi dividends.

    The Shapley value can be expressed as:
        φᵢ = Σ_{S: i∈S} w(S) / |S|

    where w(S) is the Harsanyi dividend of coalition S.

    This is often more efficient if Harsanyi dividends are already computed.

    Args:
        harsanyi_dividends: Pre-computed Harsanyi dividends for all coalitions
        num_players: Number of players

    Returns:
        Dictionary mapping player index to Shapley value
    """
    shapley = {i: 0.0 for i in range(num_players)}

    for coalition, dividend in harsanyi_dividends.items():
        if coalition == 0:  # Skip empty coalition
            continue

        coalition_size = bin(coalition).count('1')

        # Distribute dividend equally among coalition members
        for i in range(num_players):
            if (coalition >> i) & 1:  # Player i is in coalition
                shapley[i] += dividend / coalition_size

    return shapley


def rank_players_by_shapley(
    shapley_values: ShapleyValues,
    descending: bool = True
) -> List[tuple]:
    """
    Rank players by their Shapley values.

    Args:
        shapley_values: Shapley value for each player
        descending: If True, rank from highest to lowest

    Returns:
        List of (player_index, shapley_value) tuples, sorted by value
    """
    return sorted(
        shapley_values.items(),
        key=lambda x: x[1],
        reverse=descending
    )


def shapley_efficiency_check(
    shapley_values: ShapleyValues,
    v_values: CoalitionValues,
    num_players: int = 8,
    tolerance: float = 1e-6
) -> bool:
    """
    Verify that Shapley values satisfy the efficiency property.

    Efficiency: Σᵢ φᵢ = v(N)

    Args:
        shapley_values: Computed Shapley values
        v_values: Coalition values
        num_players: Number of players
        tolerance: Numerical tolerance for comparison

    Returns:
        True if efficiency property holds within tolerance
    """
    grand_coalition = (1 << num_players) - 1  # All players
    v_N = v_values.get(grand_coalition, 0.0)
    sum_shapley = sum(shapley_values.values())

    return abs(sum_shapley - v_N) < tolerance
