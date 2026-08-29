"""
Teacher Feature Caching Module for TSSA
Extracts representations from frozen Vietnamese Teacher (BARTpho)
to avoid re-computing teacher forward pass repeatedly.
"""

import os
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

class TeacherFeatureCacher:
    def __init__(self, teacher_ckpt: str = "vinai/bartpho-syllable", device: str = "cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(teacher_ckpt)
        self.model = AutoModel.from_pretrained(teacher_ckpt).to(device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def extract_sentence_and_token_features(self, tgt_texts: list, max_len: int = 256):
        """
        Extracts token-level hidden states [B, T, D] and sentence pooled vector [B, D].
        """
        encoded = self.tokenizer(
            tgt_texts,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.model.encoder(**encoded, output_hidden_states=True)
        token_states = outputs.last_hidden_state # [B, T, D]
        
        # Sentence vector: mean pooling over non-padded tokens
        mask = encoded["attention_mask"].unsqueeze(-1) # [B, T, 1]
        sent_vector = (token_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1) # [B, D]

        return token_states.cpu(), sent_vector.cpu()

    def cache_dataset_teacher_features(self, df, save_path: str, batch_size: int = 32, max_len: int = 256):
        """Caches teacher features for an entire dataframe."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        tgt_texts = df["tgt_text"].astype(str).tolist()
        
        all_token_states = []
        all_sent_vectors = []
        
        print(f"[*] Đang trích xuất đặc trưng Teacher cho {len(tgt_texts)} câu tiếng Việt...")
        for i in tqdm(range(0, len(tgt_texts), batch_size), desc="Teacher Caching"):
            batch_texts = tgt_texts[i:i+batch_size]
            tok_states, sent_vec = self.extract_sentence_and_token_features(batch_texts, max_len=max_len)
            all_token_states.append(tok_states.half()) # FP16 to save space
            all_sent_vectors.append(sent_vec.half())
            
        all_token_states = torch.cat(all_token_states, dim=0)
        all_sent_vectors = torch.cat(all_sent_vectors, dim=0)
        
        torch.save({"token_states": all_token_states, "sent_vectors": all_sent_vectors}, save_path)
        print(f"[+] Đã lưu đặc trưng Teacher vào: {save_path}")
        return all_token_states, all_sent_vectors
