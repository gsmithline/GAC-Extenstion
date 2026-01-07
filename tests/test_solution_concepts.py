"""
Unit tests for cooperative game theory solution concepts.

These tests use well-known games with analytically verified solutions
to validate the correctness of each implementation.

Test Games:
1. Majority Game (3 players): v(S) = 1 if |S| >= 2, else 0
2. Glove Game (3 players): 1 left-hand glove, 2 right-hand gloves
3. Simple Additive Game: v(S) = sum of player weights

Each test verifies the known analytical solution.
"""

import pytest
from math import isclose

# Import solution concepts
from solution_concepts.harsanyi import compute_harsanyi_dividends
from solution_concepts.shapley import compute_shapley_values, shapley_efficiency_check
from solution_concepts.core import compute_core, core_exists
from solution_concepts.nucleolus import compute_nucleolus, compute_excess
from solution_concepts.utils import coalition_to_players, players_to_coalition


class TestCoalitionConversions:
    """Test coalition bitmask conversion utilities."""

    def test_coalition_to_players(self):
        """Test converting bitmask to player set."""
        assert coalition_to_players(0, 3) == set()
        assert coalition_to_players(1, 3) == {0}
        assert coalition_to_players(5, 3) == {0, 2}  # 0b101
        assert coalition_to_players(7, 3) == {0, 1, 2}  # 0b111

    def test_players_to_coalition(self):
        """Test converting player set to bitmask."""
        assert players_to_coalition(set()) == 0
        assert players_to_coalition({0}) == 1
        assert players_to_coalition({0, 2}) == 5
        assert players_to_coalition({0, 1, 2}) == 7

    def test_roundtrip(self):
        """Test roundtrip conversion."""
        for coalition in range(8):  # 0 to 7 for 3 players
            players = coalition_to_players(coalition, 3)
            assert players_to_coalition(players) == coalition


class TestMajorityGame:
    """
    Test with 3-player majority game.

    v(S) = 1 if |S| >= 2, else 0

    Known solutions:
    - Shapley: (1/3, 1/3, 1/3)
    - Core: Empty
    - Nucleolus: (1/3, 1/3, 1/3)
    """

    @pytest.fixture
    def majority_game(self):
        """Create 3-player majority game."""
        return {
            0: 0,  # {}
            1: 0, 2: 0, 4: 0,  # {0}, {1}, {2}
            3: 1, 5: 1, 6: 1,  # {0,1}, {0,2}, {1,2}
            7: 1  # {0,1,2}
        }

    def test_harsanyi_efficiency(self, majority_game):
        """Test that Harsanyi dividends sum to v(N)."""
        dividends = compute_harsanyi_dividends(majority_game, num_players=3)
        total = sum(dividends.values())
        assert isclose(total, majority_game[7], abs_tol=1e-9)

    def test_shapley_values(self, majority_game):
        """Test Shapley values are (1/3, 1/3, 1/3)."""
        shapley = compute_shapley_values(majority_game, num_players=3)
        expected = 1/3
        for i in range(3):
            assert isclose(shapley[i], expected, abs_tol=1e-9), \
                f"Player {i}: expected {expected}, got {shapley[i]}"

    def test_shapley_efficiency(self, majority_game):
        """Test Shapley efficiency property."""
        shapley = compute_shapley_values(majority_game, num_players=3)
        assert shapley_efficiency_check(shapley, majority_game, num_players=3)

    def test_core_empty(self, majority_game):
        """Test that Core is empty for majority game."""
        assert not core_exists(majority_game, num_players=3)

    def test_nucleolus(self, majority_game):
        """Test Nucleolus is (1/3, 1/3, 1/3)."""
        nucleolus = compute_nucleolus(majority_game, num_players=3)
        assert nucleolus is not None
        expected = 1/3
        for i in range(3):
            assert isclose(nucleolus[i], expected, abs_tol=1e-6), \
                f"Player {i}: expected {expected}, got {nucleolus[i]}"


class TestGloveGame:
    """
    Test with glove game.

    Players: L (left glove), R1, R2 (right gloves)
    v(S) = min(# left gloves in S, # right gloves in S)

    With 1 left and 2 right:
    - v({L}) = v({R1}) = v({R2}) = 0
    - v({L, R1}) = v({L, R2}) = 1
    - v({R1, R2}) = 0
    - v({L, R1, R2}) = 1

    Known solutions:
    - Shapley: L = 2/3, R1 = R2 = 1/6
    - Core: {(1, 0, 0)} (only allocation in Core)
    - Nucleolus: (1, 0, 0)
    """

    @pytest.fixture
    def glove_game(self):
        """Create glove game: L=player0, R1=player1, R2=player2."""
        return {
            0: 0,  # {}
            1: 0, 2: 0, 4: 0,  # {L}, {R1}, {R2}
            3: 1, 5: 1,  # {L,R1}, {L,R2}
            6: 0,  # {R1,R2}
            7: 1  # {L,R1,R2}
        }

    def test_shapley_values(self, glove_game):
        """Test Shapley values: L gets 2/3, each R gets 1/6."""
        shapley = compute_shapley_values(glove_game, num_players=3)

        assert isclose(shapley[0], 2/3, abs_tol=1e-9), \
            f"L: expected {2/3}, got {shapley[0]}"
        assert isclose(shapley[1], 1/6, abs_tol=1e-9), \
            f"R1: expected {1/6}, got {shapley[1]}"
        assert isclose(shapley[2], 1/6, abs_tol=1e-9), \
            f"R2: expected {1/6}, got {shapley[2]}"

    def test_core_exists(self, glove_game):
        """Test that Core is non-empty for glove game."""
        assert core_exists(glove_game, num_players=3)

    def test_core_point(self, glove_game):
        """Test that (1, 0, 0) is in the Core."""
        result = compute_core(glove_game, num_players=3)
        assert result['exists']
        # The unique Core point is (1, 0, 0)
        # Due to LP, we might get any Core point, but (1,0,0) is the only one
        point = result['point']
        assert isclose(point[0], 1.0, abs_tol=1e-6)
        assert isclose(point[1], 0.0, abs_tol=1e-6)
        assert isclose(point[2], 0.0, abs_tol=1e-6)

    def test_nucleolus(self, glove_game):
        """Test Nucleolus is (1, 0, 0)."""
        nucleolus = compute_nucleolus(glove_game, num_players=3)
        assert nucleolus is not None
        assert isclose(nucleolus[0], 1.0, abs_tol=1e-6)
        assert isclose(nucleolus[1], 0.0, abs_tol=1e-6)
        assert isclose(nucleolus[2], 0.0, abs_tol=1e-6)


class TestAdditiveGame:
    """
    Test with simple additive game.

    v(S) = sum of weights for players in S
    Weights: w = [3, 2, 1]

    For additive games:
    - All interaction (Harsanyi dividends for |S| > 1) should be 0
    - Shapley value = player weight
    - Core is non-empty
    """

    @pytest.fixture
    def additive_game(self):
        """Create additive game with weights [3, 2, 1]."""
        weights = [3, 2, 1]
        v = {}
        for coalition in range(8):
            v[coalition] = sum(weights[i] for i in range(3) if (coalition >> i) & 1)
        return v

    def test_harsanyi_no_interaction(self, additive_game):
        """Test that multi-player coalitions have zero dividend."""
        dividends = compute_harsanyi_dividends(additive_game, num_players=3)

        # Single player dividends should equal their weight
        assert isclose(dividends[1], 3, abs_tol=1e-9)  # {0}
        assert isclose(dividends[2], 2, abs_tol=1e-9)  # {1}
        assert isclose(dividends[4], 1, abs_tol=1e-9)  # {2}

        # Multi-player coalitions should have zero dividend (no synergy)
        assert isclose(dividends[3], 0, abs_tol=1e-9)  # {0,1}
        assert isclose(dividends[5], 0, abs_tol=1e-9)  # {0,2}
        assert isclose(dividends[6], 0, abs_tol=1e-9)  # {1,2}
        assert isclose(dividends[7], 0, abs_tol=1e-9)  # {0,1,2}

    def test_shapley_equals_weight(self, additive_game):
        """Test Shapley value equals player weight for additive game."""
        shapley = compute_shapley_values(additive_game, num_players=3)
        assert isclose(shapley[0], 3, abs_tol=1e-9)
        assert isclose(shapley[1], 2, abs_tol=1e-9)
        assert isclose(shapley[2], 1, abs_tol=1e-9)

    def test_core_exists(self, additive_game):
        """Test Core is non-empty for additive (hence subadditive) game."""
        assert core_exists(additive_game, num_players=3)


class TestExcessComputation:
    """Test excess computation for Nucleolus."""

    def test_excess_values(self):
        """Test excess computation."""
        allocation = {0: 2, 1: 2, 2: 2}
        v_values = {
            0: 0,
            1: 1, 2: 1, 4: 1,  # Each singleton worth 1
            3: 3, 5: 3, 6: 3,  # Each pair worth 3
            7: 6  # Grand coalition worth 6
        }

        excesses = compute_excess(allocation, v_values, num_players=3)

        # e({0}, x) = v({0}) - x_0 = 1 - 2 = -1
        assert isclose(excesses[1], -1, abs_tol=1e-9)

        # e({0,1}, x) = v({0,1}) - x_0 - x_1 = 3 - 2 - 2 = -1
        assert isclose(excesses[3], -1, abs_tol=1e-9)


class TestEightPlayerGame:
    """Test with 8 players (typical attention head setup)."""

    @pytest.fixture
    def eight_player_game(self):
        """Create a simple 8-player game."""
        # Simple additive game for 8 players
        weights = [8, 7, 6, 5, 4, 3, 2, 1]
        v = {}
        for coalition in range(256):
            v[coalition] = sum(weights[i] for i in range(8) if (coalition >> i) & 1)
        return v

    def test_shapley_eight_players(self, eight_player_game):
        """Test Shapley for 8 players (additive game)."""
        shapley = compute_shapley_values(eight_player_game, num_players=8)

        # In additive game, Shapley = weight
        expected = [8, 7, 6, 5, 4, 3, 2, 1]
        for i in range(8):
            assert isclose(shapley[i], expected[i], abs_tol=1e-6), \
                f"Player {i}: expected {expected[i]}, got {shapley[i]}"

    def test_harsanyi_eight_players(self, eight_player_game):
        """Test Harsanyi efficiency for 8 players."""
        dividends = compute_harsanyi_dividends(eight_player_game, num_players=8)

        # Sum of dividends should equal v(N)
        total = sum(dividends.values())
        v_N = eight_player_game[255]  # Grand coalition
        assert isclose(total, v_N, abs_tol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
