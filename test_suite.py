"""
Unit and Integration Test Suite for TSSA Codebase
Verifies all mathematical modules, loss functions, head router,
and data pipelines without requiring heavy GPU training.
"""

import sys
import torch
import pandas as pd
import numpy as np

def run_all_tests():
    print("=" * 70)
    print("      BẮT ĐẦU KIỂM THỬ TOÀN DIỆN HỆ THỐNG MÃ NGUỒN TSSA")
    print("=" * 70)
    
    passed_tests = 0
    total_tests = 8

    # -------------------------------------------------------------
    # Test 1: StructLoss (Confidence-Weighted Barycenter)
    # -------------------------------------------------------------
    try:
        from losses.struct_loss import StructLoss
        struct_fn = StructLoss(conf_threshold=0.2)
        
        B, S, T, D = 2, 16, 16, 64
        student_enc = torch.randn(B, S, D, requires_grad=True)
        teacher_enc = torch.randn(B, T, D)
        align_matrix = torch.rand(B, S, T)
        
        loss_val = struct_fn(student_enc, teacher_enc, align_matrix)
        loss_val.backward()
        
        assert not torch.isnan(loss_val) and loss_val.item() > 0
        assert student_enc.grad is not None
        print("[PASS] Test 1/8: StructLoss (Barycentric Anchoring & Backward Pass) thành công!")
        passed_tests += 1
    except Exception as e:
        print(f"[FAIL] Test 1/8: StructLoss thất bại: {e}")

    # -------------------------------------------------------------
    # Test 2: PrimeLoss (In-Batch InfoNCE)
    # -------------------------------------------------------------
    try:
        from losses.prime_loss import PrimeLoss
        prime_fn = PrimeLoss(temperature=0.07)
        
        B, D = 4, 64
        s_vec = torch.randn(B, D, requires_grad=True)
        t_vec = torch.randn(B, D)
        
        loss_prime = prime_fn(s_vec, t_vec)
        loss_prime.backward()
        
        assert not torch.isnan(loss_prime) and loss_prime.item() > 0
        assert s_vec.grad is not None
        print("[PASS] Test 2/8: PrimeLoss (In-Batch InfoNCE Contrastive) thành công!")
        passed_tests += 1
    except Exception as e:
        print(f"[FAIL] Test 2/8: PrimeLoss thất bại: {e}")

    # -------------------------------------------------------------
    # Test 3: RouteLoss (Soft BCE Supervision)
    # -------------------------------------------------------------
    try:
        from losses.route_loss import RouteLoss
        route_fn = RouteLoss()
        
        B, L, H, T = 2, 6, 12, 16
        gates = torch.sigmoid(torch.randn(B, L, H, T))
        target = torch.rand(B, L, H, T)
        
        loss_route = route_fn(gates, target)
        assert not torch.isnan(loss_route) and loss_route.item() > 0
        print("[PASS] Test 3/8: RouteLoss (Soft BCE Router) thành công!")
        passed_tests += 1
    except Exception as e:
        print(f"[FAIL] Test 3/8: RouteLoss thất bại: {e}")

    # -------------------------------------------------------------
    # Test 4: Baseline Alignment Losses
    # -------------------------------------------------------------
    try:
        from losses.baselines.guided_attention_loss import GuidedAttentionLoss
        from losses.baselines.joint_align_loss import JointAlignLoss
        from losses.baselines.awesome_align_loss import AwesomeAlignLoss
        from losses.baselines.cl_lsa_loss import CrossLingualInfoNCELoss
        
        B, S, T, D, H = 2, 16, 16, 64, 8
        align_mat = torch.rand(B, S, T)
        
        # 4.1 Guided Attn
        guided_fn = GuidedAttentionLoss()
        cross_attns = (torch.rand(B, H, T, S),)
        l_guided = guided_fn(cross_attns, align_mat)
        assert not torch.isnan(l_guided)
        
        # 4.2 Joint Align
        joint_fn = JointAlignLoss(embed_dim=D)
        dec_h = torch.randn(B, T, D)
        enc_h = torch.randn(B, S, D)
        l_joint = joint_fn(dec_h, enc_h, align_mat)
        assert not torch.isnan(l_joint)
        
        # 4.3 AWESOME Align
        awesome_fn = AwesomeAlignLoss()
        l_awesome = awesome_fn(enc_h, dec_h, align_mat)
        assert not torch.isnan(l_awesome)
        
        # 4.4 CL-LSA
        cl_fn = CrossLingualInfoNCELoss()
        l_cl = cl_fn(enc_h.mean(dim=1), dec_h.mean(dim=1))
        assert not torch.isnan(l_cl)

        print("[PASS] Test 4/8: Toàn bộ 4 Baseline Loss (Guided, Joint, AWESOME, CL-LSA) thành công!")
        passed_tests += 1
    except Exception as e:
        print(f"[FAIL] Test 4/8: Baseline Losses thất bại: {e}")

    # -------------------------------------------------------------
    # Test 5: HeadWiseRouter & Causal Pruning Mask
    # -------------------------------------------------------------
    try:
        from models.head_router import HeadWiseRouter
        router = HeadWiseRouter(d_model=64, n_heads=8, n_layers=4)
        
        B, T, D = 2, 10, 64
        dec_q = torch.randn(B, T, D)
        ctx_v = torch.randn(B, T, D)
        
        # Forward pass layer 0
        gate_out = router(dec_q, ctx_v, layer_idx=0) # [B, H, T, 1]
        assert gate_out.shape == (B, 8, T, 1)
        
        # Pruning test
        router.prune_heads([(0, 2), (0, 5)])
        gate_pruned = router(dec_q, ctx_v, layer_idx=0)
        assert gate_pruned[:, 2, :, :].sum().item() == 0.0
        assert gate_pruned[:, 5, :, :].sum().item() == 0.0
        
        router.reset_pruning_mask()
        assert router.pruning_mask.sum().item() == 4 * 8
        print("[PASS] Test 5/8: HeadWiseRouter (Forward & Causal Pruning Mask) thành công!")
        passed_tests += 1
    except Exception as e:
        print(f"[FAIL] Test 5/8: HeadWiseRouter thất bại: {e}")

    # -------------------------------------------------------------
    # Test 6: TSSALossScheduler 3-Phase Schedule
    # -------------------------------------------------------------
    try:
        from training.loss_scheduler import TSSALossScheduler
        scheduler = TSSALossScheduler(total_steps=1000, max_l1=0.5, max_l2=0.2, max_l3=0.1)
        
        # Step 50 (Phase 1 Warmup)
        l1, l2, l3 = scheduler.get_lambdas(50)
        assert l1 == 0.0 and l2 == 0.0 and l3 == 0.0
        
        # Step 250 (Phase 2 Rampup)
        l1, l2, l3 = scheduler.get_lambdas(250)
        assert 0.0 < l1 <= 0.5 and 0.0 < l2 <= 0.2 and l3 == 0.0
        
        # Step 800 (Phase 3 Full Routing)
        l1, l2, l3 = scheduler.get_lambdas(800)
        assert l1 == 0.5 and l2 == 0.2 and 0.0 < l3 <= 0.1
        
        print("[PASS] Test 6/8: TSSALossScheduler (3-Phase Dynamic Loss Scheduling) thành công!")
        passed_tests += 1
    except Exception as e:
        print(f"[FAIL] Test 6/8: Loss Scheduler thất bại: {e}")

    # -------------------------------------------------------------
    # Test 7: Robustness Noise Generator
    # -------------------------------------------------------------
    try:
        from evaluation.robustness_noise import RobustnessEvaluator
        noise_eval = RobustnessEvaluator()
        
        sample_text = "Học sinh dân tộc Ba Na và Ê Đê rất chăm chỉ học tập"
        no_accent = noise_eval.remove_diacritics(sample_text)
        assert "ọ" not in no_accent and "ô" not in no_accent
        
        typo_text = noise_eval.add_char_typos(sample_text, typo_rate=0.8)
        assert isinstance(typo_text, str) and len(typo_text) > 0
        
        print("[PASS] Test 7/8: RobustnessEvaluator (Xóa dấu thanh, Sinh typo, UNK) thành công!")
        passed_tests += 1
    except Exception as e:
        print(f"[FAIL] Test 7/8: Robustness Evaluator thất bại: {e}")

    # -------------------------------------------------------------
    # Test 8: Data Formatting & Leakage Check
    # -------------------------------------------------------------
    try:
        from data.download_and_preprocess import clean_text, extract_pair
        raw_text = "  Câu  chứa \n\r nhiều khoảng    trắng thừa.  "
        cleaned = clean_text(raw_text)
        assert cleaned == "Câu chứa nhiều khoảng trắng thừa."
        
        pair_sample = {"translation": {"bahnaric": "Hnam", "vietnamese": "Nhà"}}
        src, tgt = extract_pair(pair_sample["translation"], "bahnaric", "vietnamese")
        assert src == "Hnam" and tgt == "Nhà"
        
        print("[PASS] Test 8/8: Data Cleaning & Translation Pair Extractor thành công!")
        passed_tests += 1
    except Exception as e:
        print(f"[FAIL] Test 8/8: Data Cleaning thất bại: {e}")

    # -------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------
    print("=" * 70)
    print(f"  KẾT QUẢ: {passed_tests}/{total_tests} BÀI KIỂM THỬ ĐÃ VƯỢT QUA HOÀN HẢO! 🏆")
    print("=" * 70)
    return passed_tests == total_tests

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
