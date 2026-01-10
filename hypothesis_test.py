#!/usr/bin/env python3
"""
Hypothesis Validation Test for GAC Extension.

This script validates the core hypotheses from the GAC paper (Qu et al., ACL 2025):

1. SYNERGY HYPOTHESIS: Some attention heads cooperate synergistically
   → Validated by finding positive Harsanyi dividends for multi-head coalitions

2. IMPORTANCE HYPOTHESIS: Some heads contribute more than others
   → Validated by computing Shapley values and checking variance

3. STABILITY HYPOTHESIS: The grand coalition may or may not be stable
   → Validated by computing the Core

4. FAIRNESS HYPOTHESIS: Nucleolus provides a robust allocation
   → Validated by computing Nucleolus and checking efficiency

Usage (Local - uses synthetic data):
    python hypothesis_test.py

Usage (Google Colab - uses real model):
    !git clone https://github.com/YOUR_USERNAME/GAC-Extension.git
    %cd GAC-Extension
    !pip install -r requirements.txt
    !python hypothesis_test.py --use_model --use_gpu
"""

import sys
import time
import argparse
import random
import math
from typing import Dict, List, Tuple

sys.path.insert(0, '.')

from solution_concepts.harsanyi import compute_harsanyi_dividends, get_top_k_coalitions, get_negative_coalitions
from solution_concepts.shapley import compute_shapley_values, rank_players_by_shapley
from solution_concepts.core import compute_core
from solution_concepts.nucleolus import compute_nucleolus
from solution_concepts.utils import coalition_to_players


def create_realistic_coalition_values(num_heads: int = 8, seed: int = 42) -> Dict[int, float]:
    """
    Create realistic coalition values that mimic attention head behavior.

    Based on observations from the GAC paper:
    - Individual heads have varying importance
    - Some head pairs are synergistic (work better together)
    - Some head pairs are redundant (overlapping function)
    - Grand coalition is typically most valuable

    Returns:
        Dictionary mapping coalition bitmask -> value
    """
    random.seed(seed)

    # Individual head contributions (varying importance)
    head_values = [random.uniform(0.05, 0.2) for _ in range(num_heads)]

    # Pairwise interactions: some synergistic (+), some redundant (-)
    interactions = {}
    for i in range(num_heads):
        for j in range(i + 1, num_heads):
            # ~40% synergistic, ~30% redundant, ~30% neutral
            r = random.random()
            if r < 0.4:
                interactions[(i, j)] = random.uniform(0.02, 0.08)  # synergy
            elif r < 0.7:
                interactions[(i, j)] = random.uniform(-0.05, -0.01)  # redundancy
            else:
                interactions[(i, j)] = 0.0  # neutral

    coalition_values = {}

    for coalition in range(2 ** num_heads):
        if coalition == 0:
            coalition_values[coalition] = 0.0
            continue

        players = [i for i in range(num_heads) if (coalition >> i) & 1]

        # Base value: sum of individual contributions
        value = sum(head_values[p] for p in players)

        # Add pairwise interactions
        for i in range(len(players)):
            for j in range(i + 1, len(players)):
                pair = (min(players[i], players[j]), max(players[i], players[j]))
                value += interactions.get(pair, 0.0)

        # Higher-order synergy for larger coalitions (diminishing returns)
        if len(players) >= 3:
            bonus = 0.01 * math.log(len(players))
            value += bonus

        # Ensure non-negative and normalize
        coalition_values[coalition] = max(0.0, value)

    # Normalize so grand coalition = 1.0
    grand = coalition_values[(1 << num_heads) - 1]
    if grand > 0:
        for k in coalition_values:
            coalition_values[k] /= grand

    return coalition_values


def validate_synergy_hypothesis(dividends: Dict[int, float], num_players: int = 8) -> dict:
    """
    Hypothesis 1: Some attention heads cooperate synergistically.

    Evidence: Positive Harsanyi dividends for coalitions of size > 1
    """
    # Get multi-head coalitions with positive dividends
    synergistic = []
    for coalition, dividend in dividends.items():
        size = bin(coalition).count('1')
        if size > 1 and dividend > 0.001:  # Threshold for significance
            players = coalition_to_players(coalition, num_players)
            synergistic.append((players, dividend))

    synergistic.sort(key=lambda x: x[1], reverse=True)

    # Get competitive coalitions (negative dividends)
    competitive = get_negative_coalitions(dividends)
    competitive_parsed = []
    for coalition, dividend in competitive[:5]:
        players = coalition_to_players(coalition, num_players)
        competitive_parsed.append((players, dividend))

    return {
        "validated": len(synergistic) > 0,
        "num_synergistic_coalitions": len(synergistic),
        "top_synergistic": synergistic[:5],
        "num_competitive_coalitions": len(competitive),
        "top_competitive": competitive_parsed,
    }


def validate_importance_hypothesis(shapley: Dict[int, float]) -> dict:
    """
    Hypothesis 2: Some heads contribute more than others.

    Evidence: High variance in Shapley values across heads
    """
    values = list(shapley.values())
    mean_val = sum(values) / len(values)
    variance = sum((v - mean_val) ** 2 for v in values) / len(values)
    std_dev = variance ** 0.5

    # Coefficient of variation (normalized variance)
    cv = std_dev / mean_val if mean_val > 0 else 0

    ranking = rank_players_by_shapley(shapley)

    return {
        "validated": cv > 0.1,  # At least 10% variation
        "coefficient_of_variation": cv,
        "mean_shapley": mean_val,
        "std_dev": std_dev,
        "ranking": ranking,
        "most_important_head": ranking[0][0] if ranking else None,
        "least_important_head": ranking[-1][0] if ranking else None,
    }


def validate_stability_hypothesis(core_result: dict) -> dict:
    """
    Hypothesis 3: Analyze coalition stability.

    Evidence: Core analysis reveals whether grand coalition is stable
    """
    is_empty = core_result.get('is_empty', True)

    return {
        "grand_coalition_stable": not is_empty,
        "core_is_empty": is_empty,
        "core_point": core_result.get('core_point', None),
        "interpretation": (
            "Grand coalition is STABLE - heads work best together"
            if not is_empty else
            "Grand coalition is UNSTABLE - subgroups may defect"
        )
    }


def validate_fairness_hypothesis(
    nucleolus: Dict[int, float],
    v_values: Dict[int, float],
    num_players: int = 8
) -> dict:
    """
    Hypothesis 4: Nucleolus provides fair credit allocation.

    Evidence: Nucleolus satisfies efficiency and minimizes worst-case dissatisfaction
    """
    grand_coalition = (1 << num_players) - 1
    grand_value = v_values.get(grand_coalition, 0)

    # Handle case where nucleolus computation failed
    if nucleolus is None:
        return {
            "validated": False,
            "efficiency_satisfied": False,
            "total_allocation": None,
            "grand_coalition_value": grand_value,
            "allocation": None,
            "error": "Nucleolus computation failed (LP infeasible)",
        }

    # Check efficiency
    total_allocation = sum(nucleolus.values())
    efficiency_satisfied = abs(total_allocation - grand_value) < 1e-6

    return {
        "validated": efficiency_satisfied,
        "efficiency_satisfied": efficiency_satisfied,
        "total_allocation": total_allocation,
        "grand_coalition_value": grand_value,
        "allocation": dict(nucleolus),
    }


def run_hypothesis_validation(use_model: bool = False, model_name: str = "gpt2", use_gpu: bool = False):
    """Run full hypothesis validation pipeline."""

    print("=" * 70)
    print("GAC Extension - Hypothesis Validation Test")
    print("Based on: Qu et al. (ACL 2025) - Cooperative Game Theory for Attention")
    print("=" * 70)

    num_heads = 8

    if use_model:
        # Use real model (for Colab with GPU)
        import torch
        import torch.nn.functional as F
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        print(f"\nDevice: {device}")

        print(f"\n[1/5] Loading model: {model_name}...")

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model with appropriate settings
        is_llama = "llama" in model_name.lower()
        if is_llama:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto" if use_gpu else None,
                trust_remote_code=True,
            )
            num_heads = 8  # Analyze 8 heads per layer (2^8 = 256 coalitions)
            print(f"    Model: LLaMA ({model.config.num_hidden_layers} layers, {model.config.num_attention_heads} heads/layer)")
            print(f"    Analyzing first {num_heads} heads for tractability")
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
            num_heads = min(8, model.config.n_head)
            print(f"    Model: {model_name} ({model.config.n_layer} layers, {model.config.n_head} heads)")

        model.eval()

        # Compute coalition values from model with actual head masking
        print("\n[2/5] Computing coalition values from model...")
        text = "This movie was absolutely fantastic!"
        inputs = tokenizer(text, return_tensors="pt")
        if not is_llama or not use_gpu:
            inputs = {k: v.to(device) for k, v in inputs.items()}

        pos_token = tokenizer.encode(" positive", add_special_tokens=False)[0]

        # Get model config
        if is_llama:
            n_layers = model.config.num_hidden_layers
            total_heads = model.config.num_attention_heads
            head_dim = model.config.hidden_size // total_heads
        else:
            n_layers = model.config.n_layer
            total_heads = model.config.n_head
            head_dim = model.config.n_embd // total_heads

        # We'll analyze heads in one layer (layer 6 for GPT-2, middle layer)
        target_layer = n_layers // 2
        print(f"    Analyzing {num_heads} heads in layer {target_layer}")

        # Create head mask hook
        active_heads_mask = None

        def create_head_mask_hook(layer_idx):
            def hook(module, input, output):
                nonlocal active_heads_mask
                if layer_idx != target_layer or active_heads_mask is None:
                    return output

                # output shape: (batch, seq_len, hidden_size) - it's a tensor, not tuple
                batch_size, seq_len, hidden_size = output.shape
                reshaped = output.view(batch_size, seq_len, total_heads, head_dim)

                # Apply mask to first num_heads heads
                for head_idx in range(num_heads):
                    if not active_heads_mask[head_idx]:
                        reshaped[:, :, head_idx, :] = 0.0

                # Reshape back
                masked_output = reshaped.view(batch_size, seq_len, hidden_size)
                return masked_output
            return hook

        # Register hooks on attention output projections
        hooks = []
        for layer_idx in range(n_layers):
            if is_llama:
                attn_module = model.model.layers[layer_idx].self_attn.o_proj
            else:
                attn_module = model.transformer.h[layer_idx].attn.c_proj
            hook = attn_module.register_forward_hook(create_head_mask_hook(layer_idx))
            hooks.append(hook)

        # Get baseline (all heads active)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]
            probs = F.softmax(logits.float(), dim=-1)
            baseline_prob = probs[pos_token].item()

        print(f"    Baseline probability for 'positive': {baseline_prob:.4f}")

        v_values = {}
        print(f"    Computing {2**num_heads} coalition values...")
        for coalition in range(2 ** num_heads):
            if coalition == 0:
                v_values[coalition] = 0.0
                continue

            # Create mask for this coalition
            active_heads_mask = [(coalition >> i) & 1 for i in range(num_heads)]

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits[0, -1, :]
                probs = F.softmax(logits.float(), dim=-1)
                v_values[coalition] = probs[pos_token].item()

        # Remove hooks
        for hook in hooks:
            hook.remove()
        active_heads_mask = None

        # Normalize by baseline
        max_val = max(v_values.values()) if v_values else 1.0
        if max_val > 0:
            for k in v_values:
                v_values[k] /= max_val

        print(f"    Computed {len(v_values)} coalition values")
    else:
        # Use synthetic data (fast, no dependencies)
        print("\n[1/5] Generating synthetic coalition values...")
        print("    (Simulating realistic attention head behavior)")
        v_values = create_realistic_coalition_values(num_heads=num_heads, seed=42)
        print(f"    Generated {len(v_values)} coalition values")

    # Compute all solution concepts
    print("\n[2/5] Computing Harsanyi dividends (synergy analysis)...")
    start = time.time()
    dividends = compute_harsanyi_dividends(v_values, num_players=num_heads)
    synergy_result = validate_synergy_hypothesis(dividends, num_heads)
    print(f"    Done in {time.time() - start:.2f}s")

    print("\n[3/5] Computing Shapley values (importance ranking)...")
    start = time.time()
    shapley = compute_shapley_values(v_values, num_players=num_heads)
    importance_result = validate_importance_hypothesis(shapley)
    print(f"    Done in {time.time() - start:.2f}s")

    print("\n[4/5] Computing Core (stability analysis)...")
    start = time.time()
    core_result = compute_core(v_values, num_players=num_heads)
    stability_result = validate_stability_hypothesis(core_result)
    print(f"    Done in {time.time() - start:.2f}s")

    print("\n[5/5] Computing Nucleolus (fairness analysis)...")
    start = time.time()
    nucleolus = compute_nucleolus(v_values, num_players=num_heads)
    fairness_result = validate_fairness_hypothesis(nucleolus, v_values, num_heads)
    print(f"    Done in {time.time() - start:.2f}s")

    # Print results
    print("\n" + "=" * 70)
    print("HYPOTHESIS VALIDATION RESULTS")
    print("=" * 70)

    print("\n[H1] SYNERGY HYPOTHESIS: Some heads cooperate synergistically")
    print(f"     Status: {'VALIDATED' if synergy_result['validated'] else 'NOT VALIDATED'}")
    print(f"     Synergistic coalitions: {synergy_result['num_synergistic_coalitions']}")
    print(f"     Competitive coalitions: {synergy_result['num_competitive_coalitions']}")
    if synergy_result['top_synergistic']:
        top = synergy_result['top_synergistic'][0]
        print(f"     Top synergy: heads {set(top[0])} -> dividend {top[1]:.4f}")

    print("\n[H2] IMPORTANCE HYPOTHESIS: Some heads matter more than others")
    print(f"     Status: {'VALIDATED' if importance_result['validated'] else 'NOT VALIDATED'}")
    print(f"     Coefficient of Variation: {importance_result['coefficient_of_variation']:.4f}")
    print(f"     Most important: head {importance_result['most_important_head']}")
    print(f"     Least important: head {importance_result['least_important_head']}")
    print(f"     Full ranking: {[h for h, _ in importance_result['ranking']]}")

    print("\n[H3] STABILITY HYPOTHESIS: Grand coalition stability")
    print(f"     Core is empty: {stability_result['core_is_empty']}")
    print(f"     -> {stability_result['interpretation']}")

    print("\n[H4] FAIRNESS HYPOTHESIS: Nucleolus provides fair allocation")
    print(f"     Status: {'VALIDATED' if fairness_result['validated'] else 'NOT VALIDATED'}")
    print(f"     Efficiency: {fairness_result['efficiency_satisfied']}")
    total_alloc = fairness_result['total_allocation']
    grand_val = fairness_result['grand_coalition_value']
    print(f"     Total allocated: {total_alloc:.4f}" if total_alloc is not None else "     Total allocated: N/A (computation failed)")
    print(f"     Grand coalition value: {grand_val:.4f}" if grand_val is not None else "     Grand coalition value: N/A")

    print("\n" + "=" * 70)
    validated_count = sum([
        synergy_result['validated'],
        importance_result['validated'],
        fairness_result['validated']
    ])
    print(f"Summary: {validated_count}/3 hypotheses validated")
    if validated_count == 3:
        print("All core hypotheses from the GAC paper are supported!")
    print("=" * 70)

    return {
        "synergy": synergy_result,
        "importance": importance_result,
        "stability": stability_result,
        "fairness": fairness_result,
        "coalition_values": v_values,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAC Hypothesis Validation")
    parser.add_argument("--use_model", action="store_true",
                        help="Use a real transformer model")
    parser.add_argument("--model", type=str, default="gpt2",
                        help="Model to use: 'gpt2' or a LLaMA path/HF id (e.g., 'meta-llama/Llama-2-7b-hf')")
    parser.add_argument("--use_gpu", action="store_true",
                        help="Use GPU if available (only with --use_model)")
    args = parser.parse_args()

    results = run_hypothesis_validation(
        use_model=args.use_model,
        model_name=args.model,
        use_gpu=args.use_gpu
    )

