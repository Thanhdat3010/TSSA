# 📊 KẾT QUẢ ĐỐI CHUẨN CÔNG BẰNG CHÍNH THỨC (FAIR BENCHMARK OFFICIAL ARCHIVE)

> **Mục tiêu:** Lưu trữ vĩnh viễn kết quả thực nghiệm đo đạc thực tế trên GPU NVIDIA A100 tại Seed 42 cho Tiếng Tày (`tay` $\to$ `vi`, 20,554 câu train, 2,295 câu test), được chạy đồng thời qua cùng một script huấn luyện `train_fair_benchmark.py` và quy chuẩn [FAIR_BENCHMARK_SETTINGS.md](file:///d:/Code/Mapping/docs/FAIR_BENCHMARK_SETTINGS.md).
> **Ngày đo đạc:** 22/09/2026.

---

## 1. BẢNG KẾT QUẢ ĐỐI CHUẨN ĐỒNG BỘ 6 PHƯƠNG PHÁP (BARTpho - Tiếng Tày)

| Hệ Thống / Phương Pháp | Phân Loại Căn Chỉnh | Seed | SacreBLEU ↑ | chrF++ ↑ | METEOR ↑ | COMET (wmt22) ↑ | Δ BLEU vs Vanilla | Δ chrF++ vs Vanilla | Trạng Thái So Với Vanilla |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Vanilla Baseline** | Không can thiệp (Chỉ $L_{\text{MT}}$) | 42 | **25.34** | **36.05** | **26.12** | **0.6502** | **Ref (0.00)** | **Ref (0.00)** | **Mốc sàn chuẩn cơ sở** |
| **AWESOME-align (EACL 2021)** | Căn chỉnh từ tố hai chiều | 42 | **25.39** | **36.30** | **26.41** | **0.6579** | **+0.05** | **+0.25** | Tương đương (nằm trong dải nhiễu) |
| **TSSA-Pro (Bản Hiện Tại)** | Mỏ neo Latent Barycenter | 42 | **25.36** | **36.00** | **25.88** | **0.6539** | **+0.02** | **-0.05** | Tương đương (nằm trong dải nhiễu) |
| **Shift-AET (EMNLP 2020)** | Giám sát tịnh tiến tự hồi quy | 42 | **24.52** | **35.60** | **25.95** | **0.6525** | **-0.82** | **-0.45** | Suy giảm rõ rệt |
| **CL-LSA (NAACL 2021)** | InfoNCE từ tố qua argmax | 42 | **24.45** | **34.92** | **25.34** | **0.6458** | **-0.89** | **-1.13** | Suy giảm rõ rệt |
| **Align-to-Distill (COLING 2024)** | Chưng cất Cross-Attention | 42 | **23.34** | **33.93** | **23.93** | **0.6410** | **-2.00** | **-2.12** | Suy giảm nghiêm trọng |

---

## 2. NHỮNG PHÁT HIỆN THỰC NGHIỆM CỐT LÕI (KEY EMPIRICAL FINDINGS)

1. **Mốc Vanilla chuẩn là 25.34 (chứ không phải 24.67 cũ):**  
   Khi được tối ưu hóa chuẩn với AdamW, FP16, learning rate `2e-5`, Vanilla tự đạt **25.34 BLEU**. Mọi so sánh trước đây với mốc 24.67 là so sánh bất đối xứng.
2. **Hiện tượng bão hòa của các phương pháp căn chỉnh hiện tại:**  
   Cả AWESOME-align (25.39) và TSSA-Pro (25.36) chỉ nhỉnh hơn Vanilla $\approx 0.02 - 0.05$ BLEU. Điều này chứng minh các phương pháp neo trực tiếp vào output của Encoder đều gặp trần hiệu năng quanh mốc ~25.35.
3. **Hiện tượng phản tác dụng của việc ép loss căn chỉnh quá mạnh:**  
   Align-to-Distill (Loss ban đầu ~12.15) và CL-LSA (Loss ban đầu ~9.01) do không có tầng đệm chiếu (projection head) và trọng số bị lấn át hàm MT đã làm giảm chất lượng dịch từ $-0.89$ đến $-2.00$ BLEU.
4. **Mục tiêu cho phương pháp cải tiến:**  
   Bản cải tiến cần vượt qua mốc **25.34** của Vanilla và **25.39** của AWESOME-align với mức tăng có ý nghĩa thống kê ($\ge +0.50$ BLEU).
