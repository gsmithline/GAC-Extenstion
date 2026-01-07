# GAC Extension: Comparing Cooperative Game-Theoretic Solution Concepts for Attention Head Analysis

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This repository extends the GAC framework (Qu et al., ACL 2025) to systematically compare multiple cooperative game-theoretic solution concepts for understanding attention head interactions in transformers.

## Overview

While Qu et al. applied the **Harsanyi dividend** to analyze attention head coalitions, this extension introduces three additional solution concepts, each answering a fundamentally different question:

| Concept | Question | Attention Head Interpretation |
|---------|----------|-------------------------------|
| **Harsanyi Dividend** | What synergy emerges from this coalition? | Do these heads create emergent capabilities together? |
| **Shapley Value** | What is each player's fair share of credit? | How much does each head contribute on average? |
| **Core** | Which allocations are stable against deviation? | Can heads coexist without "wanting to leave"? |
| **Nucleolus** | What minimizes worst-case dissatisfaction? | Which heads are undervalued relative to their potential? |

## Key Findings (Expected)

- **Ranking differences**: Shapley and Harsanyi identify different heads as "important"
- **Core emptiness**: Many layers have empty Core, confirming inherent competition
- **Pruning guidance**: Nucleolus excess identifies underutilized heads safe to prune

## Installation

### Using uv (Recommended)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
cd GAC-Extenstion
uv sync
```

### Using pip

```bash
pip install -e .
```

### Development Installation

```bash
uv sync --extra dev
```

## Project Structure

```
GAC-Extenstion/
├── solution_concepts/          # NEW: Solution concept implementations
│   ├── __init__.py
│   ├── harsanyi.py            # Harsanyi dividend (refactored from original)
│   ├── shapley.py             # Shapley value computation
│   ├── core.py                # Core existence and computation
│   ├── nucleolus.py           # Nucleolus via sequential LP
│   └── utils.py               # Utilities for comparison
├── tests/                      # NEW: Unit tests
│   └── test_solution_concepts.py
├── tools/                      # Original GAC tools
│   └── compute_harsanyi_dividends.py
├── model_aug/                  # Original GAC attention augmentation
├── utils/                      # Original GAC data utilities
├── game_theory.py              # Original GAC main script
├── pyproject.toml              # NEW: Project configuration for uv
└── README.md                   # This file
```

## Quick Start

### 1. Compute Coalition Values (Original GAC)

First, run the original GAC pipeline to compute v(S) for all coalitions:

```bash
# For classification tasks
bash scripts/compute_game_theory_cf.sh

# For multiple choice tasks
bash scripts/compute_game_theory_mc.sh
```

### 2. Compare Solution Concepts (New)

```python
from solution_concepts import compare_solution_concepts
from solution_concepts.utils import load_coalition_values

# Load coalition values from GAC output
v_values = load_coalition_values("path/to/harsanyi_dividend_headComb.log")

# Compute all four solution concepts
results = compare_solution_concepts(v_values, num_players=8)

# Access results
print("Shapley values:", results['shapley']['values'])
print("Core exists:", results['core']['exists'])
print("Nucleolus:", results['nucleolus']['allocation'])
print("Ranking correlation:", results['comparison']['shapley_nucleolus_correlation'])
```

### 3. Individual Concept Usage

```python
from solution_concepts import (
    compute_harsanyi_dividends,
    compute_shapley_values,
    compute_core,
    compute_nucleolus
)

# Harsanyi dividends (synergy analysis)
dividends = compute_harsanyi_dividends(v_values, num_players=8)
print(f"Positive dividends: {sum(1 for w in dividends.values() if w > 0)}")
print(f"Negative dividends: {sum(1 for w in dividends.values() if w < 0)}")

# Shapley values (individual importance)
shapley = compute_shapley_values(v_values, num_players=8)
top_heads = sorted(shapley.items(), key=lambda x: x[1], reverse=True)[:3]
print(f"Top 3 heads by Shapley: {top_heads}")

# Core (stability analysis)
core_result = compute_core(v_values, num_players=8)
if core_result['exists']:
    print("Core is non-empty - stable configuration exists")
else:
    print("Core is empty - inherent competition between heads")

# Nucleolus (fairness analysis)
nucleolus = compute_nucleolus(v_values, num_players=8)
print(f"Nucleolus allocation: {nucleolus}")
```

## Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=solution_concepts --cov-report=html
```

## Solution Concepts Explained

### Harsanyi Dividend

The Harsanyi dividend captures the **pure interaction effect** of a coalition:

```
w(S) = Σ_{T⊆S} (-1)^{|T|-|S|} · v(T)
```

- **Positive dividend**: Heads in S create emergent capability together
- **Negative dividend**: Heads interfere or compete (redundancy)
- **Zero dividend**: Heads are independent

### Shapley Value

The Shapley value measures **average marginal contribution**:

```
φᵢ = Σ_{S⊆N\{i}} [|S|!(|N|-|S|-1)!/|N|!] · [v(S∪{i}) - v(S)]
```

- High value: Head is consistently useful regardless of context
- Satisfies efficiency: Σφᵢ = v(N)

### Core

The Core is the set of **stable allocations**:

```
Core = {x : Σxᵢ = v(N), Σᵢ∈S xᵢ ≥ v(S) ∀S}
```

- Non-empty Core: Stable configuration exists
- Empty Core: Inherent instability (connects to negative Harsanyi dividends)

### Nucleolus

The Nucleolus **minimizes worst-case dissatisfaction**:

```
x* = argmin_x [lexicographically minimize sorted excess vector]
excess(S, x) = v(S) - Σᵢ∈S xᵢ
```

- High excess: Coalition S is "undervalued"
- Always exists and is unique
- Always in Core (if Core is non-empty)

## Mathematical Properties

| Property | Harsanyi | Shapley | Core | Nucleolus |
|----------|----------|---------|------|-----------|
| Always exists | ✓ | ✓ | ✗ | ✓ |
| Unique | ✓ | ✓ | ✗ | ✓ |
| Efficiency | ✓ | ✓ | ✓ | ✓ |
| Individual rationality | - | ✓ | ✓ | ✓ |
| Computational complexity | O(2ⁿ) | O(2ⁿ) | LP | Sequential LP |

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{qu2025cooperative,
  title={Cooperative or Competitive? Understanding the Interaction between
         Attention Heads From A Game Theory Perspective},
  author={Qu, Xiaoye and Yu, Zengqi and Liu, Dongrui and Wei, Wei and
          Liu, Daizong and Dong, Jianfeng and Cheng, Yu},
  booktitle={Proceedings of the 63rd Annual Meeting of the Association
             for Computational Linguistics (ACL)},
  year={2025}
}
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Original GAC implementation: [Qu et al.](https://github.com/queng12322/GAC)
- Game theory foundations: Shapley (1953), Harsanyi (1982), Schmeidler (1969)
