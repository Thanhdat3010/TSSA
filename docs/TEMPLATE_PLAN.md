# MẪU KẾ HOẠCH THỰC NGHIỆM TỔNG QUÁT (STANDARD EXPERIMENTAL PLAN TEMPLATE)

> [!NOTE]
**Mục đích:** Dùng chung cho kế hoạch phương pháp, đối chuẩn và mở rộng thực nghiệm ở các dự án ML/NLP. Giữ cấu trúc 9 phần; điền số liệu và hạ tầng theo dự án thực tế. Mục không áp dụng phải ghi rõ `Không áp dụng` cùng lý do, không sao chép lệnh hoặc chỉ số của dự án khác.

**Nơi lưu kế hoạch:** Cập nhật artifact/file kế hoạch mà người dùng chỉ định (ví dụ `implementation_plan.md` nếu dự án có file này). Không tự tạo nhiều bản plan trùng nhau. Template này là dữ liệu tham khảo về định dạng, không cấp quyền tự động chạy, upload, xóa hay tắt tài nguyên.

---

## CẤU TRÚC 9 PHẦN BẤT BIẾN CỦA MỘT PLAN THỰC NGHIỆM CHUẨN

Mỗi khi soạn thảo hoặc cập nhật kế hoạch trong `implementation_plan`, nội dung phải tuân thủ nghiêm ngặt cấu trúc 9 phần sau:

---

### PHẦN 1: CALLOUT NGUYÊN TẮC BẤT BIẾN & MỤC TIÊU HỌC THUẬT
* **Thẻ định dạng:** Sử dụng GitHub alert `> [!IMPORTANT]`.
* **Nội dung bắt buộc:**
  1. **Câu hỏi nghiên cứu và novelty ứng viên:** Nêu đóng góp cần chứng minh, paper gần nhất và ranh giới claim; không hứa chấp nhận hội nghị.
  2. **Fairness:** Đóng băng backbone, dữ liệu/split, tokenizer, seed, ngân sách huấn luyện và evaluator giữa phương pháp so sánh.
  3. **Faithfulness:** Với baseline từ bài báo, dẫn đúng nguồn, mục/trang/công thức sau khi kiểm tra; phân biệt tái hiện trung thành với bản thích nghi trong dự án.
  4. **Bảo tồn:** Liệt kê chính xác file, dữ liệu, checkpoint và kết quả lịch sử không được sửa/ghi đè.
  5. **Cách ly output:** Phương pháp/run mới có đường dẫn artifact riêng, hỗ trợ resume an toàn.
  6. **Preflight:** Smoke test tối thiểu một batch, một optimizer step và một lần suy luận trước full run.
  7. **Hạ tầng:** Ghi rõ local/server, môi trường, cách chạy và chính sách lưu checkpoint thực tế; không mặc định Colab, SSH hay notebook.
  8. **Thao tác bên ngoài:** Upload, push, dừng máy hoặc xóa dữ liệu chỉ nằm trong plan khi người dùng đã yêu cầu và đích thao tác đã xác định.

---

### PHẦN 2: CÂY THƯ MỤC DỰ ÁN (PROJECT DIRECTORY TREE)
* **Thẻ định dạng:** Khối mã ASCII tree dạng `bash` hoặc `text`.
* **Nội dung bắt buộc:**
  * Thể hiện rõ ranh giới giữa phần cũ được đóng băng (`[FROZEN]`) và phần module mới (`[NEW]`).
  * Chỉ rõ file mô hình, trainer, runner và thư mục artifacts theo đúng cây thư mục thực tế; không đặt tên file giả như thể đã tồn tại.

---

### PHẦN 3: DANH SÁCH FILE SẼ CHỈNH SỬA, TẠO MỚI & BẢO TỒN (FILES TO BE MODIFIED / CREATED)
* **Thẻ định dạng:** Bảng Markdown (`| Loại Thao Tác | Đường Dẫn File | Mục Đích & Phạm Vi Thay Đổi Cụ Thể |`).
* **Nội dung bắt buộc:**
  1. **Modified:** Ghi rõ file hiện có và phạm vi chỉnh sửa dự kiến; nếu chưa kiểm tra thì ghi `Cần xác minh`.
  2. **New:** Liệt kê module, smoke test, runner và báo cáo dự kiến; đánh dấu rõ `CHƯA TẠO` trước khi triển khai.
  3. **Frozen:** Ghi file/dữ liệu/checkpoint cũ không thay đổi.

---

### PHẦN 4: ĐẶC TẢ KỸ THUẬT CHI TIẾT (TECHNICAL SPECIFICATIONS)
* **Nội dung bắt buộc cho từng phương pháp / backbone:**
  * **Bối cảnh và novelty:** Một câu hỏi nghiên cứu, giả thuyết phản chứng, đóng góp mới ở cấp cơ chế hay bằng chứng; bảng đối chiếu với công trình gần nhất.
  * **Trích dẫn nguồn/BibTeX đã xác minh** cho các bài báo tham chiếu; không bịa trang, mục hoặc phương trình.
  * **Công thức toán học chuẩn LaTeX:** Diễn giải rõ các biến, chỉ số tầng, hàm loss, hệ số phạt, nhiệt độ $\tau$.
  * **Cơ chế hoạt động:** Sự khác biệt về mặt cấu trúc so với baseline và mô hình đề xuất của chúng ta.
  * **Training vs. inference:** Nêu module chỉ dùng khi train, số tham số tăng thêm và chi phí suy luận.
  * **Ablation và negative control:** Mỗi claim novelty phải có phép thử phân biệt được với giải thích thay thế.

---

### PHẦN 5: BẢNG QUY CHUẨN SIÊU THAM SỐ CÔNG BẰNG TUYỆT ĐỐI (FAIR COMPARISON PROTOCOL)
* **Thẻ định dạng:** Bảng Markdown (`| Cột 1 | Cột 2 | ... |`).
* **Các tham số bắt buộc phải liệt kê so sánh:**
  * Model Backbone
  * Tokenizer, kích thước từ vựng và số tầng encoder/decoder (nếu đã xác minh từ config; không hardcode theo trí nhớ)
  * Số tham số huấn luyện bổ sung ($\Delta\theta$)
  * Dataset, kích thước và split Train / Val / Test; nếu không có Val mà dùng Test để chọn checkpoint/hyperparameter theo giao thức lịch sử, ghi rõ `selected-on-test`, ngân sách tìm kiếm và giới hạn suy luận; không trộn với bảng giao thức khác
  * Random Seed
  * Epoch cố định hoặc early stopping; nếu có thì nêu metric và patience, nếu không thì ghi `Không áp dụng`
  * Effective Batch Size (Per-device batch size $\times$ Gradient accumulation steps)
  * Learning Rate & Warmup Steps/Ratio (phân biệt hai loại, không thay thế ngầm)
  * Precision (BF16 / FP16)
  * Generation Beam Size cho Test set
  * Phiên bản và cấu hình metric phù hợp tác vụ (ví dụ SacreBLEU signature, chrF++ word order, model COMET)
  * Quy tắc so sánh thống kê, chi phí train/inference và giới hạn khái quát hóa

---

### PHẦN 6: QUY TRÌNH PREFLIGHT SMOKE TEST BẮT BUỘC
* **Nội dung bắt buộc:**
  * Mục tiêu: Nạp mô hình trên batch nhỏ (ví dụ $B=2$), forward, backward, optimizer step, suy luận và save/load nếu có checkpoint.
  * Test cơ chế riêng: gradient/masking/precision và các bất biến của phương pháp; kiểm tra quyền ghi kho lưu trữ chỉ khi có upload được người dùng yêu cầu.
  * Giới hạn thời gian phải đo trên môi trường đích; không hứa thời lượng khi chưa benchmark.
  * Câu lệnh thực thi chỉ được ghi là chạy được khi file script thực sự đã tồn tại; trong plan trước triển khai phải đánh dấu `DỰ KIẾN`.

---

### PHẦN 7: CƠ CHẾ THỰC THI TRÊN MÔI TRƯỜNG ĐÍCH
* **Nội dung bắt buộc:** Sơ đồ luồng xử lý dạng khối (ASCII Flowchart):
  * **Bước 1:** Kiểm tra Git/worktree, môi trường Python, thiết bị GPU và dung lượng lưu trữ.
  * **Bước 2:** Pull/cài dependencies nếu phù hợp, chạy smoke; nếu fail thì dừng.
  * **Bước 3:** Chạy pilot hoặc screening đúng giao thức đã khai báo; nếu dùng Test để chọn cấu hình/checkpoint thì đánh dấu exploratory và công bố toàn bộ ứng viên, không gọi điểm đó là đánh giá độc lập.
  * **Bước 4:** Tổng hợp kết quả, đối chứng, thời gian/VRAM và quyết định GO/NO-GO.
  * Chỉ thêm lệnh SSH, upload hoặc tắt máy khi đúng hạ tầng và được người dùng yêu cầu; lệnh phải có guard để không chạy bước kế khi bước trước thất bại.

---

### PHẦN 8: DỰ TOÁN THỜI GIAN & TÀI NGUYÊN
* **Nội dung bắt buộc:**
  * Ước lượng từ benchmark pilot: thời gian/step, steps/run, số run, overhead eval; ghi rõ giả định và khoảng bất định.
  * VRAM và dung lượng checkpoint/log dự kiến; đo dung lượng trống trên máy đích.
  * Chi phí cloud/Compute Units chỉ tính khi biết đơn giá và số dư thực; nếu dùng máy chủ sẵn có, ghi `Không áp dụng/Chưa biết`.

---

### PHẦN 9: MA TRẬN KẾT QUẢ VÀ CỔNG QUYẾT ĐỊNH CHO BÀI BÁO
* **Thẻ định dạng:** Bảng Markdown hoàn chỉnh với đầy đủ các cột:
  * Backbone Kiến trúc
  * Tên Phương pháp
  * Bài báo tham chiếu
  * Số tham số thêm ($\Delta\theta$)
  * Dataset/split, seed và metric chính/phụ phù hợp tác vụ; ô chưa chạy để `TBD`, tuyệt đối không điền điểm kỳ vọng như kết quả thật
  * Vai trò học thuật, ablation/negative control và chi phí
* **Decision gate:** Ngưỡng GO/NO-GO định trước, cách tổng hợp nhiều seed/dataset, xử lý kết quả âm và giới hạn do tái sử dụng test.

---

## BỘ KHUNG MẪU DÙNG ĐỂ KHỞI TẠO ARTIFACT (STARTER SNIPPET)

```markdown
# [Tên kế hoạch thực nghiệm cụ thể]

> [!IMPORTANT]
> **NGUYÊN TẮC BẤT BIẾN:**
> 1. **Câu hỏi nghiên cứu và novelty ứng viên:** ...
> 2. **Bảo tồn:** [đường dẫn cụ thể] và kết quả đã chốt.
> 3. **Fairness:** [backbone, split, seed, optimizer, metric] cố định giữa đối chứng.
> 4. **Preflight:** Smoke test trước pilot/full run.
> 5. **Tác động bên ngoài:** [không áp dụng / đích upload / lệnh dừng máy đã được duyệt].

## 1. Nguyên tắc bất biến và mục tiêu học thuật
...

## 2. Cây thư mục dự án (Project Directory Tree)
...

## 3. Danh Sách File Sẽ Chỉnh Sửa, Tạo Mới & Bảo Tồn (Files to be Modified / Created)
| Loại thao tác | Đường dẫn file | Mục đích & Phạm vi thay đổi cụ thể |
| :--- | :--- | :--- |
| **Chỉnh sửa (Modified)** | `src/...` | ... |
| **Tạo mới (New/Created)** | `scripts/...` | ... |
| **Bảo tồn (Frozen)** | `[đường dẫn thực tế]` | Không sửa/ghi đè |

## 4. Đặc Tả Kỹ Thuật Chi Tiết (Technical Specifications)
...

## 5. Bảng Quy Chuẩn Siêu Tham Số Công Bằng Tuyệt Đối (Fair Comparison Protocol)
...

## 6. Quy Trình Preflight Smoke Test Bắt Buộc
...

## 7. Cơ Chế Thực Thi Trên Môi Trường Đích
...

## 8. Dự Toán Thời Gian & Tài Nguyên
...

## 9. Ma Trận Kết Quả và Cổng Quyết Định Cho Bài Báo
...
```
