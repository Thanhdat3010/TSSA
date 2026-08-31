import os
import requests

papers = {
    "awesome_align_eacl2021.pdf": "https://aclanthology.org/2021.eacl-main.181.pdf",
    "shift_aet_emnlp2020.pdf": "https://aclanthology.org/2020.emnlp-main.42.pdf",
    "align_to_distill_coling2024.pdf": "https://aclanthology.org/2024.lrec-main.64.pdf",
    "cross_init_acl2024.pdf": "https://aclanthology.org/2024.findings-acl.358.pdf",
    "structural_supervision_acl2022.pdf": "https://aclanthology.org/2022.findings-acl.322.pdf",
    "dm_bli_acl2024.pdf": "https://aclanthology.org/2024.acl-long.112.pdf",
    "dpo_align_emnlp2024.pdf": "https://aclanthology.org/2024.emnlp-main.188.pdf"
}

save_dir = "docs/baseline_papers"
os.makedirs(save_dir, exist_ok=True)

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

print("=" * 65)
print("🚀 ĐANG TẢI TOÀN BỘ CÁC BÀI BÁO BASELINE VÀO docs/baseline_papers/")
print("=" * 65)

for filename, url in papers.items():
    dest_path = os.path.join(save_dir, filename)
    print(f"[*] Đang tải: {filename} từ {url} ...")
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code == 200 and len(resp.content) > 10000:
            with open(dest_path, "wb") as f:
                f.write(resp.content)
            size_kb = len(resp.content) / 1024
            print(f"    -> [+] Thành công: {filename} ({size_kb:.1f} KB)")
        else:
            print(f"    -> [!] Thất bại: HTTP {resp.status_code}, size={len(resp.content)}")
    except Exception as e:
        print(f"    -> [!] Lỗi: {e}")

print("\n" + "=" * 65)
print("🎉 ĐÃ HOÀN TẤT TẢI TOÀN BỘ CÁC BÀI BÁO!")
print("=" * 65)
