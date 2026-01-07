#!/usr/bin/env python3
"""
Compute All Solution Concepts for Attention Head Analysis.

This script integrates with the original GAC pipeline to compute all four
cooperative game-theoretic solution concepts:
1. Harsanyi Dividend (original GAC)
2. Shapley Value
3. Core
4. Nucleolus

Usage:
    python compute_all_concepts.py --input_dir <path_to_gac_output> --output_dir <path_to_results>

Example:
    python compute_all_concepts.py \
        --input_dir ./game_theory_results/classification \
        --output_dir ./solution_concept_results \
        --num_layers 32 \
        --task_type classification
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm

from solution_concepts import (
    compute_harsanyi_dividends,
    compute_shapley_values,
    compute_core,
    compute_nucleolus,
)
from solution_concepts.utils import (
    load_coalition_values,
    save_results_json,
    coalition_to_string,
    compare_solution_concepts,
)


def load_layer_coalition_values(
    input_dir: str,
    layer_idx: int,
    sample_idx: Optional[int] = None,
    num_players: int = 8
) -> Dict[int, float]:
    """
    Load coalition values for a specific layer from GAC output.

    Args:
        input_dir: Directory containing GAC output files
        layer_idx: Layer index
        sample_idx: Sample index (if per-sample files)
        num_players: Number of players (heads grouped)

    Returns:
        Dictionary mapping coalition bitmask to value
    """
    if sample_idx is not None:
        file_path = os.path.join(
            input_dir,
            f"numLayer_{layer_idx}",
            f"harsanyi_dividend_headComb_sampleIdx_{sample_idx}.log"
        )
    else:
        file_path = os.path.join(
            input_dir,
            f"numLayer_{layer_idx}",
            "harsanyi_dividend_headComb.log"
        )

    if not os.path.exists(file_path):
        return {}

    return load_coalition_values(file_path, num_players)


def compute_concepts_for_layer(
    v_values: Dict[int, float],
    num_players: int = 8
) -> Dict[str, Any]:
    """
    Compute all solution concepts for a single layer.

    Args:
        v_values: Coalition values for the layer
        num_players: Number of players

    Returns:
        Dictionary with all solution concept results
    """
    if not v_values:
        return {"error": "No coalition values provided"}

    results = {}

    # 1. Harsanyi Dividend
    harsanyi = compute_harsanyi_dividends(v_values, num_players)
    positive_count = sum(1 for w in harsanyi.values() if w > 1e-9)
    negative_count = sum(1 for w in harsanyi.values() if w < -1e-9)

    results['harsanyi'] = {
        'dividends': {str(k): v for k, v in harsanyi.items()},  # JSON-safe keys
        'num_positive': positive_count,
        'num_negative': negative_count,
        'sum_positive': sum(w for w in harsanyi.values() if w > 0),
        'sum_negative': sum(w for w in harsanyi.values() if w < 0),
    }

    # 2. Shapley Value
    shapley = compute_shapley_values(v_values, num_players)
    shapley_ranking = sorted(shapley.items(), key=lambda x: x[1], reverse=True)

    results['shapley'] = {
        'values': shapley,
        'ranking': [p for p, _ in shapley_ranking],
    }

    # 3. Core
    core_result = compute_core(v_values, num_players)
    results['core'] = {
        'exists': core_result['exists'],
        'point': core_result['point'],
        'num_binding_coalitions': len(core_result.get('binding_coalitions', [])),
    }

    # 4. Nucleolus
    try:
        nucleolus = compute_nucleolus(v_values, num_players)
        if nucleolus:
            nucleolus_ranking = sorted(nucleolus.items(), key=lambda x: x[1], reverse=True)
            results['nucleolus'] = {
                'allocation': nucleolus,
                'ranking': [p for p, _ in nucleolus_ranking],
            }
        else:
            results['nucleolus'] = {'allocation': None, 'ranking': None}
    except Exception as e:
        results['nucleolus'] = {'error': str(e)}

    # 5. Comparison metrics
    if results['nucleolus'].get('allocation'):
        shapley_rank = results['shapley']['ranking']
        nucleolus_rank = results['nucleolus']['ranking']

        # Spearman correlation
        n = num_players
        rank_diff_sq = sum(
            (shapley_rank.index(i) - nucleolus_rank.index(i)) ** 2
            for i in range(n)
        )
        spearman = 1 - (6 * rank_diff_sq) / (n * (n**2 - 1)) if n > 1 else 1.0

        results['comparison'] = {
            'shapley_nucleolus_rank_correlation': spearman,
            'core_exists': core_result['exists'],
            'harsanyi_negative_count': negative_count,
        }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compute all cooperative game-theoretic solution concepts"
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help="Directory containing GAC output (coalition values)"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help="Directory to save solution concept results"
    )
    parser.add_argument(
        '--num_layers',
        type=int,
        default=32,
        help="Number of transformer layers (default: 32)"
    )
    parser.add_argument(
        '--num_players',
        type=int,
        default=8,
        help="Number of players per layer (default: 8)"
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=None,
        help="Number of samples to process (if per-sample files)"
    )
    parser.add_argument(
        '--task_type',
        type=str,
        default='classification',
        choices=['classification', 'multiple_choice', 'question_answer'],
        help="Task type"
    )
    parser.add_argument(
        '--aggregate',
        action='store_true',
        help="Aggregate results across samples"
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process each layer
    all_results = {}
    layer_summary = {
        'core_exists': [],
        'negative_dividend_counts': [],
        'shapley_nucleolus_correlations': [],
    }

    print(f"Processing {args.num_layers} layers...")

    for layer_idx in tqdm(range(args.num_layers)):
        if args.num_samples:
            # Aggregate across samples
            layer_v_values: Dict[int, float] = {}
            sample_count = 0

            for sample_idx in range(args.num_samples):
                sample_v = load_layer_coalition_values(
                    args.input_dir, layer_idx, sample_idx, args.num_players
                )
                if sample_v:
                    for coalition, value in sample_v.items():
                        layer_v_values[coalition] = layer_v_values.get(coalition, 0) + value
                    sample_count += 1

            # Average across samples
            if sample_count > 0:
                layer_v_values = {k: v / sample_count for k, v in layer_v_values.items()}
        else:
            layer_v_values = load_layer_coalition_values(
                args.input_dir, layer_idx, None, args.num_players
            )

        if layer_v_values:
            layer_results = compute_concepts_for_layer(layer_v_values, args.num_players)
            all_results[f"layer_{layer_idx}"] = layer_results

            # Update summary
            layer_summary['core_exists'].append(
                layer_results.get('core', {}).get('exists', False)
            )
            layer_summary['negative_dividend_counts'].append(
                layer_results.get('harsanyi', {}).get('num_negative', 0)
            )
            if 'comparison' in layer_results:
                layer_summary['shapley_nucleolus_correlations'].append(
                    layer_results['comparison'].get('shapley_nucleolus_rank_correlation', None)
                )

            # Save per-layer results
            layer_output_path = os.path.join(args.output_dir, f"layer_{layer_idx}.json")
            save_results_json(layer_results, layer_output_path)

    # Compute and save summary statistics
    summary = {
        'num_layers_processed': len(all_results),
        'num_layers_with_nonempty_core': sum(layer_summary['core_exists']),
        'avg_negative_dividend_count': (
            sum(layer_summary['negative_dividend_counts']) /
            len(layer_summary['negative_dividend_counts'])
            if layer_summary['negative_dividend_counts'] else 0
        ),
        'avg_shapley_nucleolus_correlation': (
            sum(c for c in layer_summary['shapley_nucleolus_correlations'] if c is not None) /
            len([c for c in layer_summary['shapley_nucleolus_correlations'] if c is not None])
            if any(c is not None for c in layer_summary['shapley_nucleolus_correlations']) else None
        ),
        'core_exists_by_layer': layer_summary['core_exists'],
        'negative_counts_by_layer': layer_summary['negative_dividend_counts'],
    }

    summary_path = os.path.join(args.output_dir, "summary.json")
    save_results_json(summary, summary_path)

    print(f"\nResults saved to {args.output_dir}")
    print(f"Summary:")
    print(f"  - Layers with non-empty Core: {summary['num_layers_with_nonempty_core']}/{len(all_results)}")
    print(f"  - Avg negative dividends per layer: {summary['avg_negative_dividend_count']:.2f}")
    if summary['avg_shapley_nucleolus_correlation'] is not None:
        print(f"  - Avg Shapley-Nucleolus correlation: {summary['avg_shapley_nucleolus_correlation']:.3f}")


if __name__ == "__main__":
    main()
