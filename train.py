"""
Main Unified Training Script for TSSA & All 8 Competitor Alignment Baselines
Supports:
- Proposed TSSA (Full & Ablations)
- Baselines: bartpho_vanilla, align_to_distill, structural_supervision, shift_aet,
             cross_init, awesome_align, dm_bli, cl_lsa, dpo_align
"""

import os
import argparse
import torch
import shutil
import numpy as np
import sacrebleu
from transformers import AutoTokenizer, Seq2SeqTrainingArguments, EarlyStoppingCallback

# Safeguard against Transformers v4.49+ CVE-2025-32434 check when torch < 2.6
try:
    import transformers.utils.import_utils as _hf_import_utils
    if hasattr(_hf_import_utils, "check_torch_load_is_safe"):
        _hf_import_utils.check_torch_load_is_safe = lambda: None
except Exception:
    pass

from data.dataloader import get_dataloaders
from models.tssa_seq2seq import TSSASeq2SeqModel
from models.tssa_vit5 import TSSAViT5Model
from losses.unified_criterion import TSSAUnifiedCriterion
from losses.baselines.factory import UnifiedAlignmentLossFactory
from training.loss_scheduler import TSSALossScheduler
from training.trainer import TSSASeq2SeqTrainer
from evaluation.evaluator import TranslationEvaluator

def parse_args():
    parser = argparse.ArgumentParser(description="TSSA & All 8 Baselines Unified Training Suite")

    # 1. Dữ liệu & Ngôn ngữ
    parser.add_argument("--lang", type=str, default="bahnaric", choices=["rhade", "tay", "bahnaric"],
                        help="Ngôn ngữ cần huấn luyện")
    parser.add_argument("--data_dir", type=str, default="data_processed", help="Thư mục chứa dữ liệu")
    parser.add_argument("--max_source_length", type=int, default=256, help="Độ dài tối đa câu nguồn")
    parser.add_argument("--max_target_length", type=int, default=256, help="Độ dài tối đa câu đích")

    # 2. Backbone Mô hình & Phương pháp
    parser.add_argument("--model_ckpt", type=str, default="vinai/bartpho-syllable", help="Pretrained Backbone checkpoint")
    parser.add_argument("--model_type", type=str, default="tssa",
                        choices=[
                            "tssa", "bartpho_vanilla", "vanilla", "vit5_vanilla",
                            "align_to_distill", "structural_supervision", "shift_aet",
                            "cross_init", "awesome_align", "dm_bli",
                            "cl_lsa", "dpo_align"
                        ],
                        help="Loại phương pháp cần chạy")

    # 3. Cấu hình TSSA Loss & Ablation
    parser.add_argument("--use_struct", action="store_true", default=True, help="Bật L_struct (Token Barycenter)")
    parser.add_argument("--no_struct", dest="use_struct", action="store_false")
    parser.add_argument("--use_prime", action="store_true", default=True, help="Bật L_prime (Sentence InfoNCE)")
    parser.add_argument("--no_prime", dest="use_prime", action="store_false")
    parser.add_argument("--use_route", action="store_true", default=True, help="Bật L_route (Decoder Head Router)")
    parser.add_argument("--no_route", dest="use_route", action="store_false")

    parser.add_argument("--lambda_struct", type=float, default=0.5)
    parser.add_argument("--lambda_prime", type=float, default=0.2)
    parser.add_argument("--lambda_route", type=float, default=0.1)

    # 4. Tham số Huấn luyện
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size trên mỗi GPU")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate chuẩn")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Tùy chỉnh tên thí nghiệm/thư mục lưu (nếu None sẽ tự động suy luận theo ablation)")

    return parser.parse_args()

def main():
    args = parse_args()
    print("=" * 65)
    print(f"[*] Bắt đầu quy trình huấn luyện: Model={args.model_type}, Lang={args.lang}")
    print(f"[*] Max Source Len={args.max_source_length}, Max Target Len={args.max_target_length}")
    print("=" * 65)

    # 1. Chuẩn bị đường dẫn dữ liệu
    lang_data_dir = os.path.join(args.data_dir, args.lang)
    if not os.path.exists(lang_data_dir):
        from data.download_and_preprocess import process_all_datasets
        print(f"[*] Chưa tìm thấy {lang_data_dir}. Đang tự động tải dữ liệu...")
        process_all_datasets(args.data_dir)

    # 2. Nạp Tokenizer
    print(f"[*] Đang nạp Tokenizer từ: {args.model_ckpt}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_ckpt)

    # 3. Tạo DataLoaders
    train_loader, test_loader, train_dataset, test_dataset = get_dataloaders(
        lang_data_dir, tokenizer, batch_size=args.batch_size,
        max_src_len=args.max_source_length, max_tgt_len=args.max_target_length
    )
    print(f"[+] Dữ liệu đã sẵn sàng: Train={len(train_dataset)} mẫu, Test={len(test_dataset)} mẫu")

    # 4. Khởi tạo Mô hình
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Đang khởi tạo mô hình trên thiết bị: {device}")
    is_t5 = "t5" in args.model_ckpt.lower()
    if is_t5:
        print(f"[+] Khởi tạo kiến trúc ViT5 (TSSAViT5Model) với checkpoint: {args.model_ckpt}")
        model = TSSAViT5Model(
            model_name_or_path=args.model_ckpt,
            use_route=args.use_route and (args.model_type == "tssa")
        ).to(device)
    else:
        print(f"[+] Khởi tạo kiến trúc BARTpho (TSSASeq2SeqModel) với checkpoint: {args.model_ckpt}")
        model = TSSASeq2SeqModel(
            model_name_or_path=args.model_ckpt,
            use_route=args.use_route and (args.model_type == "tssa")
        ).to(device)

    # 5. Khởi tạo Hàm Loss
    criterion = None
    baseline_loss_factory = None
    total_steps = len(train_loader) * args.num_epochs
    loss_scheduler = None

    if args.model_type == "tssa":
        criterion = TSSAUnifiedCriterion(
            use_struct=args.use_struct,
            use_prime=args.use_prime,
            use_route=args.use_route
        ).to(device)
        loss_scheduler = TSSALossScheduler(
            total_steps=total_steps,
            max_l1=args.lambda_struct,
            max_l2=args.lambda_prime,
            max_l3=args.lambda_route
        )
    elif args.model_type not in ["bartpho_vanilla", "vanilla", "vit5_vanilla"]:
        d_model = getattr(model, "d_model", getattr(model.config, "d_model", getattr(model.config, "hidden_size", 1024)))
        n_heads = getattr(model, "n_heads", getattr(model.config, "num_heads", getattr(model.config, "decoder_attention_heads", 16)))
        config = {
            "hidden_dim": d_model,
            "embed_dim": d_model,
            "n_heads": n_heads,
            "subspace_dim": 64,
            "temperature": 0.1,
            "alpha": 0.5,
            "beta": 1.0,
            "decay": 0.9,
            "lambda_struct": 0.3,
            "lambda_orth": 0.01,
            "lambda_co": 1.0,
            "mu": 0.5
        }
        baseline_loss_factory = UnifiedAlignmentLossFactory(
            method_name=args.model_type,
            config=config
        ).to(device)

    # 6. Thiết lập Training Arguments
    if args.exp_name is not None and args.exp_name.strip():
        exp_name = args.exp_name.strip()
    else:
        prefix = "vit5_" if is_t5 else ""
        if args.model_type == "tssa":
            parts = [f"{prefix}tssa"]
            if not args.use_route:
                parts.append("no_route")
            if not args.use_struct:
                parts.append("no_struct")
            if not args.use_prime:
                parts.append("no_prime")
            parts.append(args.lang)
            exp_name = "_".join(parts)
        elif args.model_type in ["vanilla", "bartpho_vanilla", "vit5_vanilla"]:
            if is_t5:
                exp_name = f"vit5_vanilla_{args.lang}"
            else:
                exp_name = f"bartpho_vanilla_{args.lang}"
        else:
            if is_t5:
                exp_name = f"{prefix}{args.model_type}_{args.lang}"
            else:
                exp_name = f"{args.model_type}_{args.lang}"
    save_dir = os.path.join(args.output_dir, exp_name)

    training_args = Seq2SeqTrainingArguments(
        output_dir=save_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        logging_dir=os.path.join(save_dir, "logs"),
        logging_steps=50,
        predict_with_generate=True,
        load_best_model_at_end=True,
        metric_for_best_model="sacrebleu",
        greater_is_better=True,
        save_total_limit=1,
        fp16=args.fp16 and torch.cuda.is_available(),
        report_to="tensorboard"
    )

    # 7. Định nghĩa hàm tính SacreBLEU cho validation
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = [p.strip() for p in tokenizer.batch_decode(preds, skip_special_tokens=True)]
        decoded_labels = [[l.strip() for l in tokenizer.batch_decode(labels, skip_special_tokens=True)]]
        bleu_res = sacrebleu.corpus_bleu(decoded_preds, decoded_labels, smooth_method="exp")
        return {"sacrebleu": round(bleu_res.score, 2)}

    # 8. Khởi tạo Trainer
    trainer = TSSASeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        criterion=criterion,
        loss_scheduler=loss_scheduler,
        model_type=args.model_type,
        baseline_loss_factory=baseline_loss_factory,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # 9. Bắt đầu Huấn Luyện
    print("\n🚀 Bắt đầu quá trình huấn luyện ...")
    trainer.train()

    # 10. Lưu mô hình tốt nhất và dọn dẹp checkpoint trung gian để tiết kiệm 90% dung lượng
    print("\n[*] Đang lưu mô hình tốt nhất (Best Model) và Tokenizer...")
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)

    for item in os.listdir(save_dir):
        item_path = os.path.join(save_dir, item)
        if os.path.isdir(item_path) and item.startswith("checkpoint-"):
            try:
                shutil.rmtree(item_path)
            except Exception:
                pass
    print(f"[+] Đã tối ưu hóa dung lượng lưu trữ cho {save_dir} (chỉ giữ lại Best Model ~1.6GB)")

    # 11. Đánh giá cuối cùng trên Test set
    print("\n📊 Đang tiến hành đánh giá toàn diện trên tập Test ...")
    evaluator = TranslationEvaluator(device=device, use_comet=True)
    results = evaluator.evaluate_model(
        model, tokenizer, test_loader,
        max_target_len=args.max_target_length,
        output_save_path=os.path.join(save_dir, "test_predictions.csv")
    )
    print(f"\n[V] Hoàn tất thí nghiệm {exp_name}! Kết quả: {results}")

if __name__ == "__main__":
    main()
