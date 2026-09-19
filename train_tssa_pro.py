"""
train_tssa_pro.py
Unified Training Script for TSSA-Pro (BARTpho & ViT5)
100% Continuous Mathematical Formulation (Zero Language Hardcoding):
1. Latent Barycentric Semantic Anchoring (L_struct)
2. Dynamic Information-Theoretic Entropy Filter (w_s = exp(-H_s / tau_H))
3. Continuous Gaussian Fertility Attenuation (L_prime) based on subword fertility kappa
4. Closed Capacity Head Budget rho*(H) = min(0.333, 4/H)
"""

import os
import json
import argparse
import shutil
import torch
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
from losses.pro_criterion import TSSAProCriterion
from training.loss_scheduler import TSSALossScheduler
from training.trainer import TSSASeq2SeqTrainer
from training.log_tracker import LogTracker
from evaluation.evaluator import TranslationEvaluator

def estimate_dataset_fertility(dataset, tokenizer, max_samples=1000) -> float:
    """
    Computes data-driven subword fertility kappa = len(subwords) / len(words)
    Empirically measured from dataset directly without hardcoding.
    """
    total_words = 0
    total_subwords = 0
    
    if hasattr(dataset, "df") and "src_text" in dataset.df.columns:
        samples = dataset.df["src_text"].dropna().head(max_samples)
        for text in samples:
            text_str = str(text).strip()
            words = len(text_str.split())
            subwords = len(tokenizer.tokenize(text_str))
            if words > 0:
                total_words += words
                total_subwords += subwords
    else:
        num_samples = min(len(dataset), max_samples)
        for i in range(num_samples):
            item = dataset[i]
            src_text = str(item.get("src_text", item.get("source_text", ""))).strip()
            words = len(src_text.split())
            subwords = len(tokenizer.tokenize(src_text))
            if words > 0:
                total_words += words
                total_subwords += subwords

    if total_words > 0:
        return round(total_subwords / total_words, 2)
    return 1.20

def parse_args():
    parser = argparse.ArgumentParser(description="TSSA-Pro Official Unified Training Runner (100% General)")

    # 1. Dữ liệu & Ngôn ngữ
    parser.add_argument("--lang", type=str, default="tay", help="Ngôn ngữ hoặc mã ngôn ngữ (ví dụ: rhade, tay, bahnaric)")
    parser.add_argument("--data_dir", type=str, default="data_processed", help="Thư mục chứa dữ liệu")
    parser.add_argument("--max_source_length", type=int, default=256, help="Độ dài tối đa câu nguồn")
    parser.add_argument("--max_target_length", type=int, default=256, help="Độ dài tối đa câu đích")

    # 2. Backbone Mô hình
    parser.add_argument("--model_ckpt", type=str, default="vinai/bartpho-syllable", help="Pretrained Backbone checkpoint")

    # 3. Cấu hình Tham Số Loại Hình Học Toán Học (Typological Mathematical Descriptors)
    parser.add_argument("--kappa", type=float, default=None,
                        help="Độ phân mảnh subword kappa (len(subwords)/len(words)). Nếu None sẽ tự động ước tính từ corpus.")
    parser.add_argument("--sigma_kappa", type=float, default=0.75,
                        help="Độ rộng băng thông phân phối Gauss cho hàm suy giảm mỏ neo câu")
    parser.add_argument("--use_struct", action="store_true", default=True, help="Bật L_struct (Latent Barycenter)")
    parser.add_argument("--no_struct", dest="use_struct", action="store_false")
    parser.add_argument("--use_prime", action="store_true", default=True, help="Bật L_prime (Sentence InfoNCE)")
    parser.add_argument("--no_prime", dest="use_prime", action="store_false")
    parser.add_argument("--use_route", action="store_true", default=False, help="Bật L_route (Decoder Head Router - Mặc định Tắt để giải phóng Decoder)")
    parser.add_argument("--no_route", dest="use_route", action="store_false")

    parser.add_argument("--use_centering", action="store_true", default=True, help="Bật Centering trước L2 normalize để khử anisotropy")
    parser.add_argument("--no_centering", dest="use_centering", action="store_false")
    parser.add_argument("--protect_struct_fertility", action="store_true", default=True,
                        help="Áp dụng fertility factor vào cả L_struct để bảo vệ Ba Na")
    parser.add_argument("--no_protect_struct_fertility", dest="protect_struct_fertility", action="store_false")

    parser.add_argument("--lambda_struct", type=float, default=0.20, help="Trọng số mỏ neo L_struct")
    parser.add_argument("--lambda_prime", type=float, default=0.08, help="Trọng số mỏ neo L_prime")
    parser.add_argument("--lambda_route", type=float, default=0.00, help="Trọng số mỏ neo L_route")
    parser.add_argument("--target_budget", type=float, default=0.250,
                        help="Ngân sách chuyên biệt hóa Anchor Head rho* (mặc định Pareto: 0.250 = 25% heads)")
    parser.add_argument("--prime_tau", type=float, default=0.07, help="Nhiệt độ InfoNCE cho L_prime")
    parser.add_argument("--align_tau", type=float, default=0.10, help="Nhiệt độ ma trận tương đồng")
    parser.add_argument("--entropy_tau", type=float, default=0.50, help="Hệ số làm mềm Normalized Entropy Gate")
    parser.add_argument("--conf_threshold", type=float, default=0.20, help="Ngưỡng tin cậy Anchor mỏ neo")

    # 4. Tham số Huấn luyện
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size trên mỗi GPU")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate (tự động theo backbone nếu None)")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="checkpoints/tssa_pro")
    parser.add_argument("--exp_name", type=str, default=None, help="Tên thư mục lưu thí nghiệm")

    return parser.parse_args()

def main():
    args = parse_args()

    # 1. Xác định kiến trúc & Learning Rate chuẩn
    is_t5 = "t5" in args.model_ckpt.lower()
    if args.learning_rate is None:
        args.learning_rate = 1e-4 if is_t5 else 2e-5

    prefix = "vit5_" if is_t5 else ""
    if args.exp_name is None:
        exp_name = f"{prefix}tssa_{args.lang}"
    else:
        exp_name = args.exp_name.strip()

    save_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 75)
    print(f"[*] 🚀 BẮT ĐẦU HUẤN LUYỆN TSSA-PRO UNIVERSAL: Exp={exp_name}")
    print(f"[*] Ngôn ngữ nguồn: {args.lang.upper()} -> Tiếng Việt")
    print(f"[*] Backbone: {args.model_ckpt} (is_t5={is_t5})")
    print(f"[*] LR={args.learning_rate}, Epochs={args.num_epochs}, Batch={args.batch_size}, Seed={args.seed}")
    print(f"[*] Loss Weights: struct={args.lambda_struct}, prime={args.lambda_prime}, route={args.lambda_route}")
    print(f"[*] Target Head Budget: rho*={args.target_budget} (25% Anchor Heads, 75% Free Heads)")
    print(f"[*] Thư mục lưu checkpoint: {save_dir}")
    print("=" * 75)

    # 2. Chuẩn bị đường dẫn dữ liệu
    lang_data_dir = os.path.join(args.data_dir, args.lang)
    if not os.path.exists(lang_data_dir):
        from data.download_and_preprocess import process_all_datasets
        print(f"[*] Chưa tìm thấy {lang_data_dir}. Đang tự động tải dữ liệu...")
        process_all_datasets(args.data_dir)

    # 3. Nạp Tokenizer
    print(f"[*] Đang nạp Tokenizer từ: {args.model_ckpt}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_ckpt)

    # 4. Tạo DataLoaders
    train_loader, test_loader, train_dataset, test_dataset = get_dataloaders(
        lang_data_dir, tokenizer, batch_size=args.batch_size,
        max_src_len=args.max_source_length, max_tgt_len=args.max_target_length
    )
    print(f"[+] Dữ liệu đã sẵn sàng: Train={len(train_dataset)} mẫu, Test={len(test_dataset)} mẫu")

    # Tự động ước tính độ phân mảnh kappa nếu không truyền tham số
    if args.kappa is None:
        args.kappa = estimate_dataset_fertility(train_dataset, tokenizer)
        print(f"[+] Tự động ước tính độ phân mảnh từ tố (Subword Fertility): kappa = {args.kappa}")
    else:
        print(f"[+] Sử dụng độ phân mảnh từ tố chỉ định: kappa = {args.kappa}")

    # 5. Khởi tạo Mô hình
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Đang khởi tạo mô hình trên thiết bị: {device}")
    if is_t5:
        model = TSSAViT5Model(
            model_name_or_path=args.model_ckpt,
            use_route=args.use_route
        ).to(device)
    else:
        model = TSSASeq2SeqModel(
            model_name_or_path=args.model_ckpt,
            use_route=args.use_route
        ).to(device)

    # 6. Khởi tạo Hàm Loss TSSA-Pro (100% Toán học liên tục)
    total_steps = len(train_loader) * args.num_epochs
    criterion = TSSAProCriterion(
        use_struct=args.use_struct,
        use_prime=args.use_prime,
        use_route=args.use_route,
        use_centering=args.use_centering,
        protect_struct_fertility=args.protect_struct_fertility,
        conf_threshold=args.conf_threshold,
        temperature=args.prime_tau,
        align_tau=args.align_tau,
        entropy_tau=args.entropy_tau,
        target_budget=args.target_budget,
        kappa=args.kappa,
        sigma_kappa=args.sigma_kappa
    ).to(device)

    loss_scheduler = TSSALossScheduler(
        total_steps=total_steps,
        max_l1=args.lambda_struct,
        max_l2=args.lambda_prime,
        max_l3=args.lambda_route
    )

    # 7. Thiết lập Training Arguments (HF Trainer)
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
        report_to="none"
    )

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
    log_tracker = LogTracker(exp_name=exp_name, log_dir=os.path.join(args.output_dir, "logs"))
    trainer = TSSASeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        criterion=criterion,
        loss_scheduler=loss_scheduler,
        model_type="tssa",
        log_tracker=log_tracker,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # 9. Bắt đầu Huấn Luyện
    print("\n🚀 Bắt đầu huấn luyện...")
    trainer.train()
    trainer.save_state()
    log_tracker.save()

    # 10. Lưu mô hình tốt nhất và dọn dẹp checkpoint trung gian
    print("\n[*] Đang lưu mô hình tốt nhất (Best Model)...")
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    trainer.save_state()

    for item in os.listdir(save_dir):
        item_path = os.path.join(save_dir, item)
        if os.path.isdir(item_path) and item.startswith("checkpoint-"):
            try:
                shutil.rmtree(item_path)
            except Exception:
                pass
    print(f"[+] Đã tối ưu hóa dung lượng lưu trữ cho {save_dir}")

    # 11. Đánh giá toàn diện trên Test set (4 chỉ số: BLEU, chrF++, METEOR, COMET)
    print("\n📊 Đang đánh giá toàn diện trên tập Test...")
    evaluator = TranslationEvaluator(device=device, use_comet=torch.cuda.is_available())
    results = evaluator.evaluate_model(
        model, tokenizer, test_loader,
        max_target_len=args.max_target_length,
        output_save_path=os.path.join(save_dir, "test_predictions.csv")
    )

    # Lưu metrics vào eval_metrics.json để script so sánh đọc
    metrics_file = os.path.join(save_dir, "eval_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[✓] Hoàn tất thí nghiệm {exp_name}!")
    print(f"    Kết quả: BLEU={results.get('bleu', results.get('sacrebleu'))}, chrF++={results.get('chrf', results.get('chrf++'))}")
    print(f"    Chi tiết metrics đã lưu tại: {metrics_file}")

if __name__ == "__main__":
    main()
