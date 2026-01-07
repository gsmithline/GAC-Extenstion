#!/usr/bin/env python3
"""
Minimal test script for GAC Extension.
Tests all four solution concepts with synthetic coalition values.

Usage:
    python minimal_test.py

For Google Colab:
    !git clone https://github.com/YOUR_USERNAME/GAC-Extension.git
    %cd GAC-Extension
    !pip install -r requirements.txt
    !python minimal_test.py
"""

import sys
import time

# Add project root to path (for Colab compatibility)
sys.path.insert(0, '.')

from solution_concepts.harsanyi import compute_harsanyi_dividends
from solution_concepts.shapley import compute_shapley_values
from solution_concepts.core import compute_core
from solution_concepts.nucleolus import compute_nucleolus
from solution_concepts.utils import coalition_to_players


def create_majority_game(num_players=3):
    """Create a simple majority voting game for testing.

    In this game, a coalition wins (value=1) if it has majority of players.
    """
    coalition_values = {}
    majority = num_players // 2 + 1

    for mask in range(2**num_players):
        players = coalition_to_players(mask, num_players)
        coalition_values[mask] = 1.0 if len(players) >= majority else 0.0

    return coalition_values


def create_additive_game(weights):
    """Create an additive game where v(S) = sum of weights for players in S.

    Additive games have zero Harsanyi dividends for all coalitions of size > 1.
    """
    n_players = len(weights)
    coalition_values = {}

    for mask in range(2**n_players):
        players = coalition_to_players(mask, n_players)
        coalition_values[mask] = sum(weights[p] for p in players)

    return coalition_values


def test_harsanyi():
    """Test Harsanyi dividend computation."""
    print("\n[1/4] Testing Harsanyi Dividends...")

    # Additive game should have zero dividends for |S| > 1
    weights = [1.0, 2.0, 3.0]
    coalition_values = create_additive_game(weights)
    dividends = compute_harsanyi_dividends(coalition_values, num_players=3)

    # Check individual player dividends match weights
    assert abs(dividends[0b001] - 1.0) < 1e-9, "Player 0 dividend should be 1.0"
    assert abs(dividends[0b010] - 2.0) < 1e-9, "Player 1 dividend should be 2.0"
    assert abs(dividends[0b100] - 3.0) < 1e-9, "Player 2 dividend should be 3.0"

    # Check multi-player dividends are zero
    assert abs(dividends[0b011]) < 1e-9, "Coalition {0,1} dividend should be 0"
    assert abs(dividends[0b111]) < 1e-9, "Grand coalition dividend should be 0"

    print("    Harsanyi dividends: PASSED")
    return True


def test_shapley():
    """Test Shapley value computation."""
    print("\n[2/4] Testing Shapley Values...")

    coalition_values = create_majority_game(num_players=3)
    shapley = compute_shapley_values(coalition_values, num_players=3)

    # In symmetric majority game, all players have equal Shapley value
    assert abs(shapley[0] - shapley[1]) < 1e-9, "Players 0,1 should have equal value"
    assert abs(shapley[1] - shapley[2]) < 1e-9, "Players 1,2 should have equal value"

    # Efficiency: sum of Shapley values = v(grand coalition)
    total = sum(shapley.values())
    grand_coalition_value = coalition_values[0b111]
    assert abs(total - grand_coalition_value) < 1e-9, "Shapley values should sum to v(N)"

    print(f"    Shapley values: {dict(shapley)}")
    print("    Efficiency check: PASSED")
    return True


def test_core():
    """Test Core computation."""
    print("\n[3/4] Testing Core...")

    coalition_values = create_majority_game(num_players=3)
    core_result = compute_core(coalition_values, num_players=3)

    # Core should return a result (may be empty for some games)
    assert core_result is not None, "Core computation should return a result"

    if core_result.get('is_empty', True):
        print("    Core is empty (expected for majority game)")
    else:
        print(f"    Core point: {core_result.get('core_point', 'N/A')}")

    print("    Core computation: PASSED")
    return True


def test_nucleolus():
    """Test Nucleolus computation."""
    print("\n[4/4] Testing Nucleolus...")

    coalition_values = create_majority_game(num_players=3)
    nucleolus = compute_nucleolus(coalition_values, num_players=3)

    # Nucleolus should return allocation for each player
    assert len(nucleolus) == 3, "Nucleolus should have 3 player allocations"

    # Efficiency: allocations sum to v(N)
    total = sum(nucleolus.values())
    grand_coalition_value = coalition_values[0b111]
    assert abs(total - grand_coalition_value) < 1e-9, "Nucleolus should sum to v(N)"

    # Symmetry: equal players get equal allocation
    assert abs(nucleolus[0] - nucleolus[1]) < 1e-9, "Symmetric players should be equal"

    print(f"    Nucleolus: {dict(nucleolus)}")
    print("    Nucleolus computation: PASSED")
    return True


def run_all_tests():
    """Run all minimal tests."""
    print("=" * 60)
    print("GAC Extension - Minimal Test Suite")
    print("=" * 60)

    start_time = time.time()

    tests = [
        ("Harsanyi", test_harsanyi),
        ("Shapley", test_shapley),
        ("Core", test_core),
        ("Nucleolus", test_nucleolus),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            if test_fn():
                passed += 1
        except Exception as e:
            print(f"    FAILED: {e}")
            failed += 1

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print(f"Time: {elapsed:.2f}s")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
