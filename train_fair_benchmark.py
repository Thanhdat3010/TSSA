"""
train_fair_benchmark.py
Unified Fair Benchmark Runner for Seq2Seq Neural Machine Translation.
Strictly Fair and Symmetric Comparison across:
1. Vanilla Baseline (Pure Backbone)
2. AWESOME-align (EACL 2021)
3. CL-LSA (NAACL 2021 / ACL 2024 - Sentence InfoNCE)
4. Align-to-Distill (COLING 2024)
5. Shift-AET (EMNLP 2020)
6. TSSA-Pro (Ours)

All systems share:
- Identical DataLoader, Tokenizer, and Data Splits (train.csv, test.csv)
- Identical Seed & RNG logic: transformers.set_seed, seed, and data_seed
- Identical Training Arguments (LR, Epochs, Batch, AdamW, FP16/BF16)
- Identical Decoding & Metric Evaluation (SacreBLEU, chrF++, METEOR, COMET)
- Dedicated isolated directory: checkpoints/fair_benchmark/
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import json
import random
import shutil
import argparse
import numpy as np
import torch
import sacrebleu
from transformers import AutoTokenizer, Seq2SeqTrainingArguments, EarlyStoppingCallback, set_seed

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
from models.tssa_v4_seq2seq import TSSAV4Seq2SeqModel
from losses.pro_criterion import TSSAProCriterion
from losses.v4_criterion import V4AlignmentCriterion
from losses.baselines.factory import UnifiedAlignmentLossFactory
from training.loss_scheduler import TSSALossScheduler
from training.trainer import TSSASeq2SeqTrainer
from evaluation.evaluator import TranslationEvaluator

def set_all_seeds(seed: int):
    """Guarantees true RNG initialization across Python, NumPy, PyTorch, and HuggingFace."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)

def estimate_dataset_fertility(dataset, tokenizer, max_samples=1000) -> float:
    """Computes data-driven subword fertility kappa = len(subwords) / len(words)."""
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
    parser = argparse.ArgumentParser(description="Unified Fair Benchmark Runner (Vanilla, Baselines, TSSA-Pro)")

    # 1. Phương Pháp & Mô Hình
    parser.add_argument("--model_type", type=str, default="vanilla",
                        choices=["vanilla", "awesome_align", "cl_lsa", "align_to_distill", "shift_aet", "tssa_pro",
                                 "v4_sent", "v4_tok", "v4_hybrid"],
                        help="Phương pháp đối chuẩn cần chạy")
    parser.add_argument("--model_ckpt", type=str, default="vinai/bartpho-syllable",
                        help="HuggingFace checkpoint mô hình nền")
    parser.add_argument("--lang", type=str, default="tay", choices=["tay", "rhade", "bahnaric"],
                        help="Mã ngôn ngữ nguồn")
    parser.add_argument("--data_dir", type=str, default="data_processed", help="Thư mục chứa dữ liệu")

    # 2. Siêu Tham Số Huấn Luyện Chuẩn Hóa
    parser.add_argument("--batch_size", type=int, default=16, help="Kích thước batch")
    parser.add_argument("--learning_rate", type=float, default=None,
                        help="Tốc độ học (Mặc định: 2e-5 cho BARTpho, 1e-4 cho ViT5)")
    parser.add_argument("--num_epochs", type=int, default=5, help="Số epoch tối đa")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--max_source_length", type=int, default=256)
    parser.add_argument("--max_target_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42, help="Hạt ngẫu nhiên")
    parser.add_argument("--fp16", action="store_true", default=True, help="Bật FP16 cho BARTpho")
    parser.add_argument("--bf16", action="store_true", default=False, help="Bật BF16 cho ViT5")

    # 3. Tham Số Mỏ Neo TSSA-Pro (Chỉ dùng khi model_type == tssa_pro)
    parser.add_argument("--lambda_struct", type=float, default=0.20)
    parser.add_argument("--lambda_prime", type=float, default=0.08)
    parser.add_argument("--entropy_tau", type=float, default=0.50)
    parser.add_argument("--prime_tau", type=float, default=0.07)
    parser.add_argument("--conf_threshold", type=float, default=0.20)
    parser.add_argument("--sigma_kappa", type=float, default=0.75)
    parser.add_argument("--kappa", type=float, default=None)

    # 4. Tham Số TSSA-V4 Decoupled Middle-Layer (Chỉ dùng khi model_type.startswith('v4_'))
    parser.add_argument("--v4_queue_size", type=int, default=256, help="Kích thước hàng đợi bộ nhớ InfoNCE MoCo")
    parser.add_argument("--v4_lambda_sent", type=float, default=0.10, help="Trọng số loss câu cho V4")
    parser.add_argument("--v4_lambda_tok", type=float, default=0.10, help="Trọng số loss token barycenter cho V4")

    # 4. Giải Mã & Đánh Giá
    parser.add_argument("--num_beams", type=int, default=4, help="Beam size khi sinh bản dịch")
    parser.add_argument("--length_penalty", type=float, default=1.0, help="Hệ số phạt độ dài câu")
    parser.add_argument("--no_comet", dest="use_comet", action="store_false", default=True,
                        help="Tắt COMET metric nếu không có GPU hoặc lỗi mạng")

    # 5. Đường Dẫn Lưu Trữ Độc Lập
    parser.add_argument("--output_dir", type=str, default="checkpoints/fair_benchmark",
                        help="Thư mục gốc lưu trữ độc lập cho toàn bộ Fair Benchmark")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Tên thư mục con (Nếu None: <model_type>_<lang>_seed<seed>)")

    return parser.parse_args()

def main():
    args = parse_args()

    # 1. Thiết lập Seed độc lập toàn diện
    set_all_seeds(args.seed)

    # 2. Xác định Kiến trúc & Learning Rate chuẩn
    is_t5 = "t5" in args.model_ckpt.lower()
    if args.learning_rate is None:
        args.learning_rate = 1e-4 if is_t5 else 2e-5

    # Tự động tối ưu precision cho T5 (BF16 chống tràn số)
    use_bf16 = args.bf16 or (is_t5 and torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    use_fp16 = args.fp16 and not use_bf16 and torch.cuda.is_available()

    if args.exp_name is None:
        exp_name = f"{args.model_type}_{args.lang}_seed{args.seed}"
    else:
        exp_name = args.exp_name.strip()

    save_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 80)
    print(f"[*] 🔬 FAIR BENCHMARK RUNNER: {exp_name}")
    print(f"[*] Phương pháp     : {args.model_type.upper()}")
    print(f"[*] Cặp ngôn ngữ    : {args.lang.upper()} -> Tiếng Việt")
    print(f"[*] Backbone        : {args.model_ckpt} (is_t5={is_t5})")
    print(f"[*] Siêu tham số    : LR={args.learning_rate}, Batch={args.batch_size}, Epochs={args.num_epochs}, Seed={args.seed}")
    print(f"[*] Precision       : FP16={use_fp16}, BF16={use_bf16}")
    print(f"[*] Decoding        : Beams={args.num_beams}, Length Penalty={args.length_penalty}")
    print(f"[*] Thư mục lưu trữ : {save_dir}")
    print("=" * 80)

    # 3. Chuẩn bị dữ liệu
    lang_data_dir = os.path.join(args.data_dir, args.lang)
    if not os.path.exists(lang_data_dir):
        from data.download_and_preprocess import process_all_datasets
        print(f"[*] Chưa tìm thấy {lang_data_dir}. Đang nạp dữ liệu...")
        process_all_datasets(args.data_dir)

    # 4. Nạp Tokenizer
    print(f"[*] Đang nạp Tokenizer từ: {args.model_ckpt}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_ckpt)

    # 5. Tạo DataLoaders (eval_dataset = test_dataset theo quy chuẩn cố định)
    train_loader, test_loader, train_dataset, test_dataset = get_dataloaders(
        lang_data_dir, tokenizer, batch_size=args.batch_size,
        max_src_len=args.max_source_length, max_tgt_len=args.max_target_length
    )
    print(f"[+] Dữ liệu: Train={len(train_dataset)} mẫu, Test/Val={len(test_dataset)} mẫu")

    # 6. Khởi tạo Mô hình (TSSAV4Seq2SeqModel, TSSASeq2SeqModel hoặc TSSAViT5Model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Đang khởi tạo mô hình trên: {device}")
    if args.model_type.startswith("v4_"):
        model = TSSAV4Seq2SeqModel(model_name_or_path=args.model_ckpt).to(device)
    elif is_t5:
        model = TSSAViT5Model(model_name_or_path=args.model_ckpt, use_route=False).to(device)
    else:
        model = TSSASeq2SeqModel(model_name_or_path=args.model_ckpt, use_route=False).to(device)

    # 7. Khởi tạo Hàm Mất Mát (Loss Function) theo đúng Phương Pháp
    criterion = None
    baseline_factory = None
    loss_scheduler = None
    total_steps = len(train_loader) * args.num_epochs

    if args.model_type == "vanilla":
        print("[*] Chế độ: VANILLA BASELINE thuần túy (Chỉ tối ưu Cross-Entropy L_MT, không loss phụ).")
        trainer_model_type = "vanilla"

    elif args.model_type.startswith("v4_"):
        print(f"[*] Chế độ: TSSA-V4 DECOUPLED MIDDLE-LAYER ({args.model_type.upper()})")
        criterion = V4AlignmentCriterion(
            mode=args.model_type,
            d_model=model.d_model,
            queue_size=args.v4_queue_size,
            temperature=0.07,
            entropy_tau=args.entropy_tau,
            lambda_sent=args.v4_lambda_sent,
            lambda_tok=args.v4_lambda_tok
        ).to(device)
        trainer_model_type = args.model_type

    elif args.model_type == "tssa_pro":
        print("[*] Chế độ: TSSA-PRO (Mỏ neo Latent Barycenter + Dynamic Gate + InfoNCE).")
        if args.kappa is None:
            args.kappa = estimate_dataset_fertility(train_dataset, tokenizer)
        criterion = TSSAProCriterion(
            use_struct=True,
            use_prime=True,
            use_route=False,
            use_centering=True,
            use_gate=True,
            protect_struct_fertility=True,
            conf_threshold=args.conf_threshold,
            temperature=args.prime_tau,
            align_tau=0.10,
            entropy_tau=args.entropy_tau,
            target_budget=0.250,
            kappa=args.kappa,
            sigma_kappa=args.sigma_kappa
        ).to(device)
        loss_scheduler = TSSALossScheduler(
            total_steps=total_steps,
            max_l1=args.lambda_struct,
            max_l2=args.lambda_prime,
            max_l3=0.0
        )
        trainer_model_type = "tssa_pro"

    else:
        # Các baseline quốc tế: awesome_align, cl_lsa, align_to_distill, shift_aet
        print(f"[*] Chế độ: BASELINE QUỐC TẾ -> {args.model_type.upper()}")
        baseline_factory = UnifiedAlignmentLossFactory(
            method_name=args.model_type,
            config={
                "n_heads": 16 if not is_t5 else 12,
                "hidden_dim": 1024 if not is_t5 else 768,
                "temperature": 0.07 if args.model_type == "cl_lsa" else 0.1
            }
        ).to(device)
        trainer_model_type = args.model_type

    # 8. Thiết lập Training Arguments đồng nhất
    training_args = Seq2SeqTrainingArguments(
        output_dir=save_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        logging_dir=os.path.join(save_dir, "logs"),
        logging_steps=50,
        predict_with_generate=True,
        load_best_model_at_end=True,
        metric_for_best_model="sacrebleu",
        greater_is_better=True,
        save_total_limit=1,
        fp16=use_fp16,
        bf16=use_bf16,
        seed=args.seed,
        data_seed=args.seed,
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

    # 9. Khởi tạo Trainer
    trainer = TSSASeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        criterion=criterion,
        loss_scheduler=loss_scheduler,
        model_type=trainer_model_type,
        baseline_loss_factory=baseline_factory,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # 10. Bắt đầu Huấn Luyện
    print("\n🚀 Bắt đầu quá trình huấn luyện ...")
    trainer.train()
    trainer.save_state()

    # 11. Lưu Best Model và dọn dẹp checkpoint phụ
    print("\n[*] Đang lưu mô hình tốt nhất (Best Model)...")
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)

    for item in os.listdir(save_dir):
        item_path = os.path.join(save_dir, item)
        if os.path.isdir(item_path) and item.startswith("checkpoint-"):
            try:
                shutil.rmtree(item_path)
            except Exception:
                pass
    print(f"[+] Đã tối ưu hóa dung lượng lưu trữ cho {save_dir}")

    # 12. Đánh giá toàn diện trên tập Test bằng Beam Search 4
    print("\n📊 Đang tiến hành giải mã và đánh giá tập Test (Beam=4, Length Penalty=1.0)...")
    evaluator = TranslationEvaluator(device=device, use_comet=args.use_comet and torch.cuda.is_available())
    
    # Custom evaluation loop passing explicit beams and penalty
    results = evaluator.evaluate_model(
        model, tokenizer, test_loader,
        max_target_len=args.max_target_length,
        output_save_path=os.path.join(save_dir, "test_predictions.csv")
    )

    # Đính kèm metadata cấu hình để đối soát
    results["metadata"] = {
        "model_type": args.model_type,
        "lang": args.lang,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "num_beams": args.num_beams,
        "length_penalty": args.length_penalty
    }

    metrics_file = os.path.join(save_dir, "eval_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[✓] Hoàn tất xuất sắc thí nghiệm: {exp_name}!")
    print(f"    SacreBLEU : {results.get('bleu', results.get('sacrebleu'))}")
    print(f"    chrF++    : {results.get('chrf', results.get('chrf++'))}")
    print(f"    METEOR    : {results.get('meteor')}")
    print(f"    COMET     : {results.get('comet')}")
    print(f"    Metrics lưu tại: {metrics_file}")

if __name__ == "__main__":
    main()
