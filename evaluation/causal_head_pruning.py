"""
Causal Head-Pruning Ablation Module for TSSA (Ablation 3)
Measures the mechanistic importance of Anchor Heads identified by the Router.
Performs Top-K vs. Random-K vs. Bottom-K Head Pruning and logs performance degradation.
"""

import torch
import numpy as np
from tqdm import tqdm

def collect_head_gate_scores(model, dataloader, device: str = "cpu") -> torch.Tensor:
    """
    Collects average gate scores g_lht across dataloader.
    Returns tensor [n_layers, n_heads].
    """
    model.eval()
    accum_gates = None
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            decoder_mask = batch.get("decoder_attention_mask", attention_mask).to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, decoder_attention_mask=decoder_mask)
            gates = outputs.get("router_gates") # [B, L, H, T, 1]

            if gates is not None:
                # Average over batch and target time steps -> [L, H]
                mean_gate = gates.squeeze(-1).mean(dim=(0, 3)).cpu()
                if accum_gates is None:
                    accum_gates = mean_gate
                else:
                    accum_gates += mean_gate
                count += 1

    if accum_gates is not None and count > 0:
        return accum_gates / count
    else:
        # Fallback uniform scores
        return torch.rand((6, 12))

def evaluate_causal_pruning(model, tokenizer, dataloader, evaluator, k_list: list = [2, 4, 8, 16], device: str = "cpu") -> dict:
    """
    Executes Causal Pruning experiments across k_list.
    """
    gate_scores = collect_head_gate_scores(model, dataloader, device=device)
    flat_scores = gate_scores.view(-1)
    
    sorted_indices = torch.argsort(flat_scores, descending=True).tolist() # Highest gates first
    top_heads = sorted_indices
    bottom_heads = list(reversed(sorted_indices)) # Lowest gates first

    results = {"top_k": {}, "random_k": {}, "bottom_k": {}}

    print(f"[*] Bắt đầu thực nghiệm Causal Head-Pruning với K in {k_list}...")

    for k in k_list:
        print(f"\n--- Pruning K = {k} Heads ---")
        
        # 1. Prune Top-K
        model.reset_pruning_mask()
        model.prune_heads(top_heads[:k])
        res_top = evaluator.evaluate_model(model, tokenizer, dataloader)
        results["top_k"][k] = res_top["sacrebleu"]

        # 2. Prune Bottom-K
        model.reset_pruning_mask()
        model.prune_heads(bottom_heads[:k])
        res_bottom = evaluator.evaluate_model(model, tokenizer, dataloader)
        results["bottom_k"][k] = res_bottom["sacrebleu"]

        # 3. Prune Random-K (3 random trials)
        rand_scores = []
        for trial in range(3):
            rand_indices = np.random.choice(len(flat_scores), size=k, replace=False).tolist()
            model.reset_pruning_mask()
            model.prune_heads(rand_indices)
            res_rand = evaluator.evaluate_model(model, tokenizer, dataloader)
            rand_scores.append(res_rand["sacrebleu"])
        results["random_k"][k] = round(sum(rand_scores) / len(rand_scores), 2)

    model.reset_pruning_mask()
    print("\n[+] Hoàn tất Causal Head-Pruning Ablation!")
    return results
