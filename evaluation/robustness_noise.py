"""
Robustness and Orthographic Noise Evaluation Module (Ablation 4)
Generates 4 realistic noise perturbations on test data:
1. Diacritic Removal (Xóa dấu)
2. Character Typo / Swap (Lỗi chính tả hoán đổi ký tự)
3. Word Order Swap (Đảo trật tự từ)
4. Out-of-Vocabulary / UNK Dropout (Rơi rụng từ)
"""

import re
import random
import unicodedata
import pandas as pd

class RobustnessEvaluator:
    def __init__(self, seed: int = 42):
        random.seed(seed)

    @staticmethod
    def remove_diacritics(text: str) -> str:
        """Strips accents and tone marks (NFD -> ASCII-like)."""
        normalized = unicodedata.normalize('NFD', text)
        no_accent = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')
        return unicodedata.normalize('NFC', no_accent)

    @staticmethod
    def add_char_typos(text: str, typo_rate: float = 0.1) -> str:
        """Randomly swaps adjacent characters or introduces common typo substitutions."""
        words = text.split()
        corrupted = []
        for w in words:
            if len(w) > 3 and random.random() < typo_rate:
                idx = random.randint(0, len(w) - 2)
                chars = list(w)
                chars[idx], chars[idx+1] = chars[idx+1], chars[idx]
                corrupted.append(''.join(chars))
            else:
                corrupted.append(w)
        return ' '.join(corrupted)

    @staticmethod
    def swap_word_order(text: str, swap_rate: float = 0.1) -> str:
        """Randomly swaps adjacent word pairs."""
        words = text.split()
        if len(words) < 3:
            return text
        for i in range(len(words) - 1):
            if random.random() < swap_rate:
                words[i], words[i+1] = words[i+1], words[i]
        return ' '.join(words)

    @staticmethod
    def inject_unk_tokens(text: str, unk_rate: float = 0.05) -> str:
        """Randomly replaces words with <unk>."""
        words = text.split()
        corrupted = [w if random.random() > unk_rate else "<unk>" for w in words]
        return ' '.join(corrupted)

    def generate_noisy_testset(self, test_df: pd.DataFrame, noise_type: str = "diacritics") -> pd.DataFrame:
        """Creates a noisy copy of test_df with specified noise_type."""
        noisy_df = test_df.copy()
        
        if noise_type == "diacritics":
            noisy_df["src_text"] = noisy_df["src_text"].apply(self.remove_diacritics)
        elif noise_type == "typo":
            noisy_df["src_text"] = noisy_df["src_text"].apply(self.add_char_typos)
        elif noise_type == "word_swap":
            noisy_df["src_text"] = noisy_df["src_text"].apply(self.swap_word_order)
        elif noise_type == "unk":
            noisy_df["src_text"] = noisy_df["src_text"].apply(self.inject_unk_tokens)
        else:
            raise ValueError(f"Không nhận diện noise_type: {noise_type}")
            
        return noisy_df
