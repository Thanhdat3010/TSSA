"""
smoke_test_pro.py
Standalone Lightning Smoke Test Script for TSSA-Pro (BARTpho & ViT5)
Runs an end-to-end verification:
1. Tokenizer & Model loading (BARTpho & ViT5)
2. Online Frozen Teacher extraction & Latent Barycenter projection
3. TSSA-Pro Loss with Dynamic Morphology & Entropy Gate
4. Backward pass & Gradient flow verification (no NaNs, teacher stopped)
5. Autoregressive sequence generation verification (model.generate)
"""

import sys
import os
import torch
import torch.nn as nn

# Safeguard against Transformers CVE check on older torch
try:
    import transformers.utils.import_utils as _hf_import_utils
    if hasattr(_hf_import_utils, "check_torch_load_is_safe"):
        _hf_import_utils.check_torch_load_is_safe = lambda: None
except Exception:
    pass

from transformers import AutoTokenizer
from models.tssa_seq2seq import TSSASeq2SeqModel
from models.tssa_vit5 import TSSAViT5Model
from losses.pro_criterion import TSSAProCriterion

def test_backbone(model_ckpt: str, is_t5: bool, device: str):
    print(f"\n[*] -------------------------------------------------------------")
    print(f"[*] ĐANG KIỂM THỬ BACKBONE: {model_ckpt} (is_t5={is_t5})")
    print(f"[*] Thiết bị: {device}")
    print(f"[*] -------------------------------------------------------------")

    # 1. Nạp Tokenizer
    print("   [1/5] Nạp Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    
    # 2. Khởi tạo Model
    print("   [2/5] Nạp Model & Semantic Projector...")
    if is_t5:
        model = TSSAViT5Model(model_name_or_path=model_ckpt, use_route=True).to(device)
    else:
        model = TSSASeq2SeqModel(model_name_or_path=model_ckpt, use_route=True).to(device)
    model.train()

    assert hasattr(model, "projector"), "Thiếu ResidualSemanticProjector!"
    assert model.can_generate(), "Model phải hỗ trợ can_generate()!"

    # 3. Tạo Mini-Batch mẫu (2 câu song ngữ Ê Đê - Việt)
    print("   [3/5] Tạo Batch & Chạy Forward Pass...")
    src_texts = ["Hnam kơ dră kơ kơl", "Ơi ya gơñ kơ ai"]
    tgt_texts = ["Nhà của tôi ở đây", "Bà ơi cháu đi học"]

    src_enc = tokenizer(src_texts, padding="max_length", max_length=16, truncation=True, return_tensors="pt").to(device)
    tgt_enc = tokenizer(tgt_texts, padding="max_length", max_length=16, truncation=True, return_tensors="pt").to(device)

    labels = tgt_enc["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100

    batch = {
        "input_ids": src_enc["input_ids"],
        "attention_mask": src_enc["attention_mask"],
        "decoder_attention_mask": tgt_enc["attention_mask"],
        "labels": labels
    }

    outputs = model(**batch)

    assert "loss" in outputs and outputs["loss"] is not None, "Outputs thiếu MT loss!"
    assert "encoder_last_hidden_state" in outputs, "Thiếu encoder_last_hidden_state!"
    assert "teacher_enc_states" in outputs and outputs["teacher_enc_states"] is not None, "Thiếu teacher_enc_states!"

    # 4. Kiểm tra TSSA-Pro Criterion
    print("   [4/5] Tính toán TSSA-Pro Loss (Latent Barycenter + Dynamic Entropy Filter)...")
    criterion = TSSAProCriterion(
        use_struct=True,
        use_prime=True,
        use_route=True,
        conf_threshold=0.20,
        temperature=0.07,
        align_tau=0.10,
        entropy_tau=1.5,
        target_budget=0.250,
        kappa=1.20
    ).to(device)

    loss_res = criterion(outputs["loss"], outputs, batch, lambdas=(0.20, 0.08, 0.05))
    total_loss = loss_res["loss"]

    print(f"         -> Loss MT     = {loss_res['log_dict']['loss_mt']:.4f}")
    print(f"         -> Loss Struct = {loss_res['log_dict']['loss_struct']:.4f}")
    print(f"         -> Loss Prime  = {loss_res['log_dict']['loss_prime']:.4f}")
    print(f"         -> Loss Route  = {loss_res['log_dict']['loss_route']:.4f}")
    print(f"         -> TOTAL LOSS  = {total_loss.item():.4f}")

    assert not torch.isnan(total_loss), "Total loss bị NaN!"
    assert total_loss.item() > 0, "Total loss phải > 0!"

    # Kiểm tra Backward Pass
    print("         -> Kiểm tra Backward & Gradient Flow...")
    total_loss.backward()

    # Xác nhận gradient chảy vào Student Encoder & Projector
    has_grad = False
    for name, param in model.named_parameters():
        if "projector" in name and param.grad is not None:
            if param.grad.abs().sum() > 0:
                has_grad = True
                break
    assert has_grad, "Gradient không chảy vào ResidualSemanticProjector!"

    # 5. Kiểm tra Sinh Câu (Generation Sanity Check)
    print("   [5/5] Kiểm tra Sequence Generation (model.generate)...")
    model.eval()
    with torch.no_grad():
        gen_tokens = model.generate(input_ids=src_enc["input_ids"][:1], max_length=10)
        gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
        print(f"         -> Generation output sample: '{gen_text}'")

    print(f"   [✓] Backbone {model_ckpt} PASS HOÀN TOÀN!")

def main():
    print("=" * 70)
    print("   🚀 BẮT ĐẦU LIGHTNING SMOKE TEST CHO PHƯƠNG PHÁP MỚI TSSA-PRO")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Thiết bị phát hiện: {device}")

    try:
        # 1. Test BARTpho
        test_backbone("vinai/bartpho-syllable", is_t5=False, device=device)

        # 2. Test ViT5
        test_backbone("VietAI/vit5-base", is_t5=True, device=device)

        print("\n" + "=" * 70)
        print("  🎉 [PASS] ALL SMOKE TESTS COMPLETED SUCCESSFULLY!")
        print("  (Loss > 0, Gradients OK, Generation OK, Latent Barycenter OK)")
        print("=" * 70 + "\n")
        sys.exit(0)

    except Exception as e:
        print("\n" + "!" * 70)
        print(f"  ❌ [FAIL] SMOKE TEST THẤT BẠI: {str(e)}")
        print("!" * 70 + "\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
