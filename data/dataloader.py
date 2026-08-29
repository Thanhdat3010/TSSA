"""
DataLoader and Dataset Module for TSSA
Integrates tokenization (MAX_SOURCE_LENGTH=256, MAX_TARGET_LENGTH=256),
alignment matrices, and teacher cached features.
"""

import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class TSSADataset(Dataset):
    def __init__(self, csv_path: str, tokenizer, max_src_len: int = 256, max_tgt_len: int = 256,
                 align_cache_path: str = None, teacher_cache_path: str = None):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        
        # Load optional cached alignment matrices
        self.align_matrices = None
        if align_cache_path and os.path.exists(align_cache_path):
            print(f"[*] Nạp ma trận căn chỉnh từ: {align_cache_path}")
            self.align_matrices = torch.load(align_cache_path)
            
        # Load optional cached teacher features
        self.teacher_features = None
        if teacher_cache_path and os.path.exists(teacher_cache_path):
            print(f"[*] Nạp đặc trưng Teacher: {teacher_cache_path}")
            self.teacher_features = torch.load(teacher_cache_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        src_text = str(row["src_text"])
        tgt_text = str(row["tgt_text"])

        # Tokenize source
        src_encoded = self.tokenizer(
            src_text,
            max_length=self.max_src_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize target
        with self.tokenizer.as_target_tokenizer():
            tgt_encoded = self.tokenizer(
                tgt_text,
                max_length=self.max_tgt_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

        labels = tgt_encoded["input_ids"].squeeze(0)
        # Replace padding token id with -100 for PyTorch CrossEntropyLoss
        labels[labels == self.tokenizer.pad_token_id] = -100

        item = {
            "input_ids": src_encoded["input_ids"].squeeze(0),
            "attention_mask": src_encoded["attention_mask"].squeeze(0),
            "labels": labels,
            "decoder_attention_mask": tgt_encoded["attention_mask"].squeeze(0),
            "src_text": src_text,
            "tgt_text": tgt_text
        }

        # Attach alignment matrix if available
        if self.align_matrices is not None and idx < len(self.align_matrices):
            mat = self.align_matrices[idx]
            item["align_matrix"] = mat.to_dense() if mat.is_sparse else mat
        else:
            item["align_matrix"] = torch.zeros((self.max_src_len, self.max_tgt_len), dtype=torch.float32)

        # Attach teacher cached features if available
        if self.teacher_features is not None:
            item["teacher_enc_states"] = self.teacher_features["token_states"][idx].float()
            item["teacher_sent_vec"] = self.teacher_features["sent_vectors"][idx].float()

        return item

def tssa_collate_fn(batch):
    """Custom batch collator for TSSA."""
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    decoder_attention_mask = torch.stack([b["decoder_attention_mask"] for b in batch])
    align_matrix = torch.stack([b["align_matrix"] for b in batch])

    collated = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "decoder_attention_mask": decoder_attention_mask,
        "align_matrix": align_matrix,
        "src_texts": [b["src_text"] for b in batch],
        "tgt_texts": [b["tgt_text"] for b in batch]
    }

    if "teacher_enc_states" in batch[0]:
        collated["teacher_enc_states"] = torch.stack([b["teacher_enc_states"] for b in batch])
        collated["teacher_sent_vec"] = torch.stack([b["teacher_sent_vec"] for b in batch])

    return collated

def get_dataloaders(data_dir: str, tokenizer, batch_size: int = 16, max_src_len: int = 256, max_tgt_len: int = 256):
    """Constructs train and test dataloaders for a given language data directory."""
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Không tìm thấy train.csv hoặc test.csv tại {data_dir}. Vui lòng chạy download_and_preprocess.py trước.")

    align_cache = os.path.join(data_dir, "train_alignments.pt")
    teacher_cache = os.path.join(data_dir, "train_teacher_features.pt")

    train_dataset = TSSADataset(
        train_path, tokenizer, max_src_len, max_tgt_len,
        align_cache_path=align_cache if os.path.exists(align_cache) else None,
        teacher_cache_path=teacher_cache if os.path.exists(teacher_cache) else None
    )

    test_dataset = TSSADataset(test_path, tokenizer, max_src_len, max_tgt_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=tssa_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=tssa_collate_fn)

    return train_loader, test_loader, train_dataset, test_dataset
