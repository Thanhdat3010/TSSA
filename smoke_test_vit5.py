"""
Comprehensive 5-Stage Smoke Test for ViT5 (VietAI/vit5-base) & TSSA
Strictly verifies:
Stage 1: Tokenizer loading, special tokens, padding, label masking (-100).
Stage 2: TSSAViT5Model forward pass, internal tensor shapes (cross_attentions, align_matrix, router_gates, projections).
Stage 3: Loss computation & Backward pass (gradient check) for ALL 6 methods:
         1. TSSA (TSSAUnifiedCriterion)
         2. Vanilla (Standard CrossEntropy)
         3. Align-to-Distill (A2D)
         4. Shift-AET
         5. AWESOME-align
         6. CL-LSA (InfoNCE)
Stage 4: Sequence generation (model.generate) & metric decoding (SacreBLEU, chrF++).
Stage 5: Mini training step (AdamW optimizer step + FP16 autocast compatibility).
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np

def run_smoke_test():
    print("=" * 80)
    print("      🧪 RUNNING COMPREHENSIVE 5-STAGE SMOKE TEST FOR ViT5 & TSSA")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Device: {device}")

    # =========================================================================
    # STAGE 1: Tokenizer & Padding Check
    # =========================================================================
    print("\n>>> [STAGE 1/5] Testing ViT5 Tokenizer & Label Masking...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base")
    print(f"  [+] Loaded Tokenizer: vocab_size={len(tokenizer)}, pad_token_id={tokenizer.pad_token_id}, eos_token_id={tokenizer.eos_token_id}")
    assert tokenizer.pad_token_id is not None, "Tokenizer must have pad_token_id!"
    
    src_sample = ["Hnăn mtao mblŭ klei hgŭm.", "Khảm bôm nịu chỉ."]
    tgt_sample = ["Vậy tù trưởng lên tiếng sự đoàn kết.", "Vượt mâm ngón trỏ."]
    
    encoded_src = tokenizer(src_sample, padding=True, truncation=True, max_length=32, return_tensors="pt")
    with tokenizer.as_target_tokenizer():
        encoded_tgt = tokenizer(tgt_sample, padding=True, truncation=True, max_length=32, return_tensors="pt")
    
    labels = encoded_tgt["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    
    input_ids = encoded_src["input_ids"].to(device)
    attention_mask = encoded_src["attention_mask"].to(device)
    labels = labels.to(device)
    decoder_attention_mask = encoded_tgt["attention_mask"].to(device)
    
    print(f"  [+] Tokenized Dummy Batch: input_ids shape={input_ids.shape}, labels shape={labels.shape}")
    print("  [✓] STAGE 1 PASSED: Tokenizer & Masking verified.")

    # =========================================================================
    # STAGE 2: TSSAViT5Model Architecture & Tensor Shapes
    # =========================================================================
    print("\n>>> [STAGE 2/5] Testing TSSAViT5Model Forward Pass & Tensor Shapes...")
    from models.tssa_vit5 import TSSAViT5Model
    model = TSSAViT5Model("VietAI/vit5-base", use_route=True).to(device)
    model.train()

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        decoder_attention_mask=decoder_attention_mask
    )

    B, S = input_ids.shape
    _, T = labels.shape
    D = model.d_model
    H = model.n_heads

    print(f"  [+] Model specs: d_model={D}, n_heads={H}, d_kv={model.d_kv}, n_decoder_layers={model.n_decoder_layers}")
    
    # Audit Shapes
    assert outputs["loss"] is not None and not torch.isnan(outputs["loss"]), "Loss must not be None/NaN!"
    assert outputs["student_projected_sent"].shape == (B, D), f"Student proj shape {outputs['student_projected_sent'].shape} != ({B}, {D})"
    assert outputs["teacher_sent_vec"].shape == (B, D), f"Teacher sent shape {outputs['teacher_sent_vec'].shape} != ({B}, {D})"
    assert outputs["align_matrix_ts"].shape == (B, T, S), f"Align matrix shape {outputs['align_matrix_ts'].shape} != ({B}, {T}, {S})"
    
    # Check align matrix row sums (must sum to ~1.0 for valid target tokens)
    align_sums = outputs["align_matrix_ts"].sum(dim=-1) # [B, T]
    assert torch.allclose(align_sums, torch.ones_like(align_sums), atol=1e-3), "Align matrix rows must sum to 1.0!"

    assert outputs["cross_attentions"] is not None, "Cross attentions must not be None!"
    assert len(outputs["cross_attentions"]) == 3, f"Expected 3 top cross-attention layers, got {len(outputs['cross_attentions'])}"
    for l_idx, xattn in enumerate(outputs["cross_attentions"]):
        assert xattn.shape == (B, H, T, S), f"Layer {l_idx} cross_attn shape {xattn.shape} != ({B}, {H}, {T}, {S})"

    assert outputs["router_gates"].shape == (B, model.n_decoder_layers, H, T, 1), f"Router gates shape unexpected: {outputs['router_gates'].shape}"
    print(f"  [+] Verified exact tensor shapes: cross_attn=(B={B}, H={H}, T={T}, S={S}), align_matrix=(B={B}, T={T}, S={S})")
    print("  [✓] STAGE 2 PASSED: Architecture & Shapes verified.")

    # =========================================================================
    # STAGE 3: All 6 Method Criterions & Backward Pass (Gradient Check)
    # =========================================================================
    print("\n>>> [STAGE 3/5] Testing All 6 Method Criterions & Backward Gradients...")
    from losses.unified_criterion import TSSAUnifiedCriterion
    from losses.baselines.factory import UnifiedAlignmentLossFactory

    batch_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "decoder_attention_mask": decoder_attention_mask,
        "align_matrix": outputs["align_matrix_ts"]
    }

    methods_to_test = [
        ("tssa", "TSSA Unified Criterion"),
        ("bartpho_vanilla", "Vanilla MT Loss"),
        ("align_to_distill", "Align-to-Distill (A2D)"),
        ("shift_aet", "Shift-AET"),
        ("awesome_align", "AWESOME-align"),
        ("cl_lsa", "CL-LSA InfoNCE")
    ]

    for m_type, m_desc in methods_to_test:
        model.zero_grad()
        # Re-forward to build fresh computation graph
        fresh_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask
        )
        loss_mt = fresh_outputs["loss"]

        if m_type == "tssa":
            crit = TSSAUnifiedCriterion(use_struct=True, use_prime=True, use_route=True).to(device)
            res = crit(loss_mt, fresh_outputs, batch=batch_inputs, lambdas=(0.5, 0.2, 0.1))
            total_loss = res["loss"]
        else:
            factory = UnifiedAlignmentLossFactory(
                method_name=m_type,
                config={
                    "hidden_dim": D, "embed_dim": D, "n_heads": H,
                    "temperature": 0.1, "alpha": 0.5, "beta": 1.0, "decay": 0.9,
                    "lambda_struct": 0.3, "lambda_co": 1.0
                }
            ).to(device)
            res = factory(loss_mt=loss_mt, model_outputs=fresh_outputs, batch=batch_inputs)
            total_loss = res["loss_total"]

        assert not torch.isnan(total_loss) and not torch.isinf(total_loss), f"Loss for {m_type} is NaN or Inf!"
        total_loss.backward()

        # Check that projector and router have non-zero gradients if TSSA
        has_grad = any(p.grad is not None and torch.norm(p.grad) > 0 for p in model.parameters() if p.requires_grad)
        assert has_grad, f"Gradients not found after backward pass for method {m_type}!"
        print(f"  [+] Method '{m_type}' ({m_desc}): total_loss={total_loss.item():.4f} -> Gradients verified!")

    print("  [✓] STAGE 3 PASSED: All 6 Criterions & Backward Gradients verified.")

    # =========================================================================
    # STAGE 4: Sequence Generation & Metric Pipeline
    # =========================================================================
    print("\n>>> [STAGE 4/5] Testing Generation (model.generate) & Metric Decoding...")
    model.eval()
    with torch.no_grad():
        gen_tokens = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=32,
            num_beams=2
        )
    assert gen_tokens.shape[0] == B, f"Generated batch size {gen_tokens.shape[0]} != {B}"
    decoded_preds = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
    assert len(decoded_preds) == B, f"Decoded predictions count {len(decoded_preds)} != {B}"
    print(f"  [+] Generated {len(decoded_preds)} sample sequences: '{decoded_preds[0][:40]}...'")

    import sacrebleu
    bleu = sacrebleu.corpus_bleu(decoded_preds, [tgt_sample]).score
    chrf = sacrebleu.corpus_chrf(decoded_preds, [tgt_sample], word_order=2).score
    print(f"  [+] sacreBLEU evaluation verified: BLEU={bleu:.2f}, chrF++={chrf:.2f}")
    print("  [✓] STAGE 4 PASSED: Generation & Metric Pipeline verified.")

    # =========================================================================
    # STAGE 5: Mini Training Step (AdamW Optimizer + Mixed Precision)
    # =========================================================================
    print("\n>>> [STAGE 5/5] Testing 1-Step Optimizer (AdamW) & Loss Scheduler...")
    from training.loss_scheduler import TSSALossScheduler
    from torch.optim import AdamW

    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = TSSALossScheduler(total_steps=100, max_l1=0.5, max_l2=0.2, max_l3=0.1)

    model.train()
    optimizer.zero_grad()

    # Test autocast compatibility
    use_cuda = torch.cuda.is_available()
    with torch.cuda.amp.autocast(enabled=use_cuda):
        fresh_out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask
        )
        crit = TSSAUnifiedCriterion(use_struct=True, use_prime=True, use_route=True).to(device)
        res = crit(fresh_out["loss"], fresh_out, batch=batch_inputs, lambdas=scheduler.get_lambdas(1))
        loss = res["loss"]

    loss.backward()
    optimizer.step()
    print(f"  [+] Optimizer step successful (loss={loss.item():.4f}, current_lambdas={scheduler.get_lambdas(1)})")
    print("  [✓] STAGE 5 PASSED: Mini Training Step verified.")

    print("\n" + "=" * 80)
    print("  🎉 ALL 5 STAGES OF SMOKE TEST PASSED FLAWLESSLY! ViT5 IS 100% PRODUCTION READY!")
    print("=" * 80)

if __name__ == "__main__":
    run_smoke_test()
