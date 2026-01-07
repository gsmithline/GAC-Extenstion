"""
Harsanyi Dividend computation for attention head coalitions.

The Harsanyi dividend measures the pure interaction effect of a coalition,
capturing the synergistic value that emerges only when all members cooperate.

Mathematical Definition:
    For a coalition S and characteristic function v:
    w(S) = Σ_{T⊆S} (-1)^{|T|-|S|} · v(T)

Interpretation for Attention Heads:
    - Positive dividend: Heads in coalition create emergent capabilities together
    - Negative dividend: Heads interfere or duplicate (competition/redundancy)
    - Zero dividend: Heads are independent (no interaction)

Reference:
    Harsanyi, J.C. (1982). "A simplified bargaining model for the n-person cooperative game"
    Qu et al. (ACL 2025). "Cooperative or Competitive? Understanding the Interaction
    between Attention Heads From A Game Theory Perspective"
"""

from itertools import combinations
from typing import Dict, List, Set, Union

# Type aliases for clarity
Coalition = int  # Bitmask representation (e.g., 0b00000101 = players {0, 2})
CoalitionValues = Dict[Coalition, float]  # Maps coalition -> v(S)
HarsanyiDividends = Dict[Coalition, float]  # Maps coalition -> w(S)


def get_subsets(coalition: Coalition, num_players: int = 8) -> List[Coalition]:
    """
    Generate all subsets of a given coalition.

    Args:
        coalition: Bitmask representing the coalition (e.g., 5 = 0b101 = {0, 2})
        num_players: Total number of players in the game (default: 8 for attention heads)

    Returns:
        List of all subset coalitions as bitmasks

    Example:
        >>> get_subsets(0b101)  # Coalition {0, 2}
        [0, 1, 4, 5]  # {}, {0}, {2}, {0,2}
    """
    # Extract player indices from bitmask
    players = [i for i in range(num_players) if (coalition >> i) & 1]

    subsets = []
    for r in range(len(players) + 1):
        for combo in combinations(players, r):
            subset = sum(1 << i for i in combo)
            subsets.append(subset)

    return subsets


def compute_harsanyi_dividends(
    v_values: CoalitionValues,
    num_players: int = 8
) -> HarsanyiDividends:
    """
    Compute the Harsanyi dividend for all coalitions.

    The Harsanyi dividend w(S) captures the pure interaction effect of coalition S,
    which is the portion of value that can only be attributed to the cooperation
    of ALL members of S together.

    Args:
        v_values: Dictionary mapping coalition bitmasks to their values v(S).
                  Should contain entries for all 2^n coalitions (0 to 2^n - 1).
        num_players: Number of players in the game (default: 8)

    Returns:
        Dictionary mapping coalition bitmasks to their Harsanyi dividends w(S)

    Mathematical Properties:
        - Efficiency: Σ_{S⊆N} w(S) = v(N)  (dividends sum to grand coalition value)
        - Linearity: Dividends are linear in the characteristic function
        - Dummy: If player i is dummy, w(S) = 0 for all S containing i (|S| > 1)
        - Symmetry: Symmetric players receive same dividends in symmetric coalitions

    Example:
        >>> v = {0: 0, 1: 10, 2: 15, 3: 30}  # 2-player game
        >>> w = compute_harsanyi_dividends(v, num_players=2)
        >>> w[3]  # Dividend for grand coalition {0,1}
        5  # = v({0,1}) - v({0}) - v({1}) + v({}) = 30 - 10 - 15 + 0
    """
    w_values: HarsanyiDividends = {}

    # Iterate over all possible coalitions
    for coalition in range(1 << num_players):
        subsets = get_subsets(coalition, num_players)
        coalition_size = bin(coalition).count('1')

        w = 0.0
        for subset in subsets:
            subset_size = bin(subset).count('1')
            # Inclusion-exclusion: (-1)^{|T| - |S|}
            sign = (-1) ** (subset_size - coalition_size)
            v_subset = v_values.get(subset, 0.0)
            w += sign * v_subset

        w_values[coalition] = w

    return w_values


def get_top_k_coalitions(
    dividends: HarsanyiDividends,
    k: int = 10,
    positive_only: bool = True
) -> List[tuple]:
    """
    Get the top-k coalitions by Harsanyi dividend magnitude.

    Args:
        dividends: Harsanyi dividends for all coalitions
        k: Number of top coalitions to return
        positive_only: If True, only return positive dividends (cooperative coalitions)

    Returns:
        List of (coalition, dividend) tuples sorted by dividend (descending)
    """
    if positive_only:
        filtered = {c: w for c, w in dividends.items() if w > 0}
    else:
        filtered = dividends

    sorted_coalitions = sorted(filtered.items(), key=lambda x: abs(x[1]), reverse=True)
    return sorted_coalitions[:k]


def get_negative_coalitions(dividends: HarsanyiDividends) -> List[tuple]:
    """
    Get all coalitions with negative Harsanyi dividends (competitive relationships).

    Args:
        dividends: Harsanyi dividends for all coalitions

    Returns:
        List of (coalition, dividend) tuples for negative dividends, sorted by magnitude
    """
    negative = {c: w for c, w in dividends.items() if w < 0}
    return sorted(negative.items(), key=lambda x: x[1])
