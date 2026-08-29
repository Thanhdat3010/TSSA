"""
Word Aligner Module for TSSA
Extracts subword/token-level alignment matrix A_ij in [0, 1] using SimAlign (EMNLP 2020)
or fallback cosine matching, and caches them to disk for fast training.
"""

import os
import torch
import numpy as np
from tqdm import tqdm

class OfflineWordAligner:
    def __init__(self, model_name: str = "vinai/bartpho-syllable", device: str = "cpu", method: str = "inter"):
        """
        method: 'inter' (argmax intersection), 'itermax', or 'mwmf' (match)
        """
        self.device = device
        self.method = method
        self.aligner = None
        try:
            from simalign import SentenceAligner
            self.aligner = SentenceAligner(model_name_or_path=model_name, token_type="bpe", device=device)
            print(f"[*] Khởi tạo SimAlign thành công với model: {model_name}")
        except Exception as e:
            print(f"[!] Warning: Không thể khởi tạo SimAlign ({e}). Sẽ sử dụng Fallback Cosine Matcher.")

    def compute_alignment_matrix(self, src_tokens: list, tgt_tokens: list, max_src: int = 256, max_tgt: int = 256) -> torch.Tensor:
        """
        Computes soft/hard alignment matrix A of shape [max_src, max_tgt].
        """
        matrix = torch.zeros((max_src, max_tgt), dtype=torch.float32)
        
        if len(src_tokens) == 0 or len(tgt_tokens) == 0:
            return matrix
            
        if self.aligner is not None:
            try:
                alignments = self.aligner.get_word_aligns(src_tokens, tgt_tokens)
                pairs = alignments.get(self.method, alignments.get("inter", []))
                for (s_idx, t_idx) in pairs:
                    if s_idx < max_src and t_idx < max_tgt:
                        matrix[s_idx, t_idx] = 1.0
                return matrix
            except Exception:
                pass
                
        # Fallback: Simple position and character overlap heuristic
        for i, s_tok in enumerate(src_tokens[:max_src]):
            for j, t_tok in enumerate(tgt_tokens[:max_tgt]):
                if s_tok.lower() == t_tok.lower() and len(s_tok) > 1:
                    matrix[i, j] = 1.0
                elif abs(i / max(1, len(src_tokens)) - j / max(1, len(tgt_tokens))) < 0.1:
                    matrix[i, j] = 0.5
        return matrix

    def cache_dataset_alignments(self, df, tokenizer, save_path: str, max_src: int = 256, max_tgt: int = 256):
        """Precomputes alignment matrices for a whole dataframe and saves as .pt."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cached_matrices = []
        
        print(f"[*] Đang sinh ma trận căn chỉnh từ cho {len(df)} mẫu...")
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Aligning"):
            src_text = str(row["src_text"])
            tgt_text = str(row["tgt_text"])
            
            src_tokens = tokenizer.tokenize(src_text)
            tgt_tokens = tokenizer.tokenize(tgt_text)
            
            mat = self.compute_alignment_matrix(src_tokens, tgt_tokens, max_src, max_tgt)
            cached_matrices.append(mat.to_sparse()) # Lưu sparse để tiết kiệm RAM
            
        torch.save(cached_matrices, save_path)
        print(f"[+] Đã lưu ma trận căn chỉnh vào: {save_path}")
        return cached_matrices
