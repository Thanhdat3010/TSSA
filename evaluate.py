"""
Standalone Evaluation Script for TSSA & Ablations
Supports:
- Standard Benchmark Evaluation (BLEU, chrF++, METEOR, COMET)
- Ablation 3: Causal Head-Pruning Evaluation
- Ablation 4: Robustness against 4 types of Noise
"""

import os
import argparse
import torch
import pandas as pd
from transformers import AutoTokenizer

from data.dataloader import TSSADataset, get_dataloaders
from models.tssa_seq2seq import TSSASeq2SeqModel
from evaluation.evaluator import TranslationEvaluator
from evaluation.causal_head_pruning import evaluate_causal_pruning
from evaluation.robustness_noise import RobustnessEvaluator

def parse_args():
    parser = argparse.ArgumentParser(description="TSSA Comprehensive Evaluation Suite")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Đường dẫn thư mục checkpoint mô hình")
    parser.add_argument("--lang", type=str, default="bahnaric", choices=["rhade", "tay", "bahnaric"])
    parser.add_argument("--data_dir", type=str, default="data_processed")
    parser.add_argument("--max_target_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    
    # Flags cho các bài Ablation
    parser.add_argument("--run_causal_pruning", action="store_true", help="Chạy thí nghiệm Causal Head-Pruning")
    parser.add_argument("--run_robustness", action="store_true", help="Chạy thí nghiệm Robustness trước nhiễu")
    parser.add_argument("--output_file", type=str, default="evaluation_results.csv")

    return parser.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print(f"[*] Bắt đầu đánh giá mô hình: {args.checkpoint_dir} trên {args.lang}")
    print("=" * 60)

    # 1. Nạp Tokenizer và Model
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir)
    model = TSSASeq2SeqModel(model_name_or_path=args.checkpoint_dir).to(device)

    # 2. Chuẩn bị DataLoader Test
    test_csv = os.path.join(args.data_dir, args.lang, "test.csv")
    test_dataset = TSSADataset(test_csv, tokenizer, max_tgt_len=args.max_target_length)
    from torch.utils.data import DataLoader
    from data.dataloader import tssa_collate_fn
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=tssa_collate_fn)

    evaluator = TranslationEvaluator(device=device, use_comet=True)

    # 3. Main Benchmark Eval
    print("\n--- 1. Đánh Giá Điểm Chuẩn (Main Benchmark) ---")
    base_results = evaluator.evaluate_model(
        model, tokenizer, test_loader,
        max_target_len=args.max_target_length,
        output_save_path=os.path.join(args.checkpoint_dir, "test_predictions.csv")
    )
    print(f"Base Results: {base_results}")

    # 4. Ablation 3: Causal Head Pruning (nếu được bật)
    if args.run_causal_pruning:
        print("\n--- 2. Chạy Thí Nghiệm Causal Head-Pruning (Ablation 3) ---")
        pruning_results = evaluate_causal_pruning(model, tokenizer, test_loader, evaluator, k_list=[2, 4, 8, 16], device=device)
        print("Pruning Results:", pruning_results)

    # 5. Ablation 4: Robustness Test (nếu được bật)
    if args.run_robustness:
        print("\n--- 3. Chạy Thí Nghiệm Độ Bền Vững Trước Nhiễu (Ablation 4) ---")
        noise_eval = RobustnessEvaluator()
        test_df = pd.read_csv(test_csv)
        
        noise_results = {}
        for n_type in ["diacritics", "typo", "word_swap", "unk"]:
            noisy_df = noise_eval.generate_noisy_testset(test_df, noise_type=n_type)
            noisy_csv = os.path.join(args.checkpoint_dir, f"test_noisy_{n_type}.csv")
            noisy_df.to_csv(noisy_csv, index=False, encoding="utf-8")
            
            noisy_ds = TSSADataset(noisy_csv, tokenizer, max_tgt_len=args.max_target_length)
            noisy_loader = DataLoader(noisy_ds, batch_size=args.batch_size, shuffle=False, collate_fn=tssa_collate_fn)
            
            res = evaluator.evaluate_model(model, tokenizer, noisy_loader, max_target_len=args.max_target_length)
            noise_results[n_type] = res["sacrebleu"]
            
        print("Robustness Results (BLEU):", noise_results)

    print("\n[V] Toàn bộ quá trình đánh giá đã hoàn tất thành công!")

if __name__ == "__main__":
    main()
