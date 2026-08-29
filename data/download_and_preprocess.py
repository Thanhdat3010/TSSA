"""
Download and Preprocessing Script for Ethnic Minority Datasets (TSSA)
Downloads datasets directly from Hugging Face, cleans text with Regex + Unicode NFC,
unpacks translation dicts, and saves clean train.csv and test.csv without leaks.
"""

import os
import re
import unicodedata
import pandas as pd
from datasets import load_dataset

DATASETS_CONFIG = {
    "bahnaric": {
        "hf_path": "FiveC/bahnaric_vietnamese",
        "src_key": "bahnaric",
        "tgt_key": "vietnamese",
        "train_split": "train",
        "test_split": "test"
    },
    "rhade": {
        "hf_path": "NIRVLab/rhade-vietnamese-mt",
        "src_key": ["ede", "cdc"],
        "tgt_key": "vi",
        "train_split": "train",
        "test_split": "test"
    },
    "tay": {
        "hf_path": "HeyDunaX/tay-vietnamese-nmt",
        "src_key": "tay",
        "tgt_key": ["viet", "vietnamese", "vi"],
        "train_split": "train",
        "test_split": "val"
    }
}

def clean_text(text: str) -> str:
    """
    Standard cleaning based on bartbana_final.py:
    - Normalizes Unicode to NFC.
    - Cleans newline, tab, and excessive whitespace.
    """
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'\\n|\|\\|[\n\r\t]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_pair(entry, src_keys, tgt_keys):
    """Extracts (src_text, tgt_text) from translation dictionary."""
    if not isinstance(entry, dict):
        return None, None
    
    src_text = None
    if isinstance(src_keys, list):
        for k in src_keys:
            if k in entry and entry[k]:
                src_text = entry[k]
                break
    else:
        src_text = entry.get(src_keys)
        
    tgt_text = None
    if isinstance(tgt_keys, list):
        for k in tgt_keys:
            if k in entry and entry[k]:
                tgt_text = entry[k]
                break
    else:
        tgt_text = entry.get(tgt_keys)
        
    return clean_text(src_text), clean_text(tgt_text)

def parse_split_to_df(split_data, src_key, tgt_key):
    """Iterates through split items and builds a deduplicated DataFrame."""
    rows = []
    for item in split_data:
        trans = item.get("translation", item)
        src, tgt = extract_pair(trans, src_key, tgt_key)
        if src and tgt and len(src) > 0 and len(tgt) > 0:
            rows.append({"src_text": src, "tgt_text": tgt})
    return pd.DataFrame(rows).drop_duplicates()

def process_single_dataset(lang_name: str, config: dict, output_base: str = "data_processed"):
    """Downloads and processes a single language dataset."""
    print(f"\n=======================================================")
    print(f"[*] Đang tải và trích xuất: {lang_name} ({config['hf_path']})")
    print(f"=======================================================")
    
    save_dir = os.path.join(output_base, lang_name)
    os.makedirs(save_dir, exist_ok=True)
    
    ds = load_dataset(config['hf_path'])
    splits = list(ds.keys())
    print(f"    -> Splits có sẵn trên HF: {splits}")
    
    # 1. Trích xuất tập Train
    train_split_name = config.get("train_split", "train")
    if train_split_name not in splits and "train" in splits:
        train_split_name = "train"
    train_df = parse_split_to_df(ds[train_split_name], config['src_key'], config['tgt_key'])
    
    # 2. Trích xuất tập Test
    test_split_name = config.get("test_split", "test")
    if test_split_name not in splits:
        for candidate in ["test", "val", "validation", "dev"]:
            if candidate in splits and candidate != train_split_name:
                test_split_name = candidate
                break
    test_df = parse_split_to_df(ds[test_split_name], config['src_key'], config['tgt_key'])
    
    # 3. Khử rò rỉ (Prevent data leakage)
    train_df = train_df[~train_df["src_text"].isin(test_df["src_text"])].reset_index(drop=True)
    
    # 4. Lưu ra CSV
    train_path = os.path.join(save_dir, "train.csv")
    test_path = os.path.join(save_dir, "test.csv")
    
    train_df.to_csv(train_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")
    
    print(f"    [+] Hoàn tất: train.csv ({len(train_df)} dòng), test.csv ({len(test_df)} dòng)")
    return train_df, test_df

def process_all_datasets(output_base: str = "data_processed"):
    """Processes all 3 language datasets."""
    results = {}
    for lang, config in DATASETS_CONFIG.items():
        try:
            train_df, test_df = process_single_dataset(lang, config, output_base)
            results[lang] = {"train_len": len(train_df), "test_len": len(test_df)}
        except Exception as e:
            print(f"[!] Lỗi khi xử lý {lang}: {e}")
    return results

if __name__ == "__main__":
    process_all_datasets()
