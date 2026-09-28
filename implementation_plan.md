# Kế hoạch thực nghiệm CF-TSSA v2: Semantic-Specific Counterfactual Anchoring cho dịch máy ít dữ liệu

> [!IMPORTANT]
> **MỤC TIÊU VÀ NGUYÊN TẮC BẤT BIẾN**
> 1. **Câu hỏi nghiên cứu:** Một target-side semantic anchor chỉ nên được dùng khi can thiệp của nó cải thiện loss dịch **hơn một can thiệp sai ngữ nghĩa nhưng cùng độ lớn** hay không?
> 2. **Novelty là giả thuyết, không phải chứng nhận ACL:** Đóng góp ứng viên là *semantic-specific finite-intervention gate* tại encoder–decoder interface. Chỉ claim sau rà soát công trình gần nhất và ablation xác nhận; không claim frozen teacher, barycenter, gradient scaling hoặc activation patching là mới.
> 3. **Fairness theo đúng giao thức:** Bảng lịch sử dùng `legacy_best`: tối đa 5 epoch, test mỗi epoch để chọn best checkpoint/early stopping; CF phải chạy y hệt nếu muốn so trực tiếp với bảng đó. Bảng `fixed_final` dùng đúng 5 epoch và test sau train; chỉ so với những phương pháp đã chạy lại bằng `fixed_final`. Hai bảng không trộn điểm.
> 4. **Bảo tồn:** Không sửa dữ liệu gốc, docs kết quả cũ hoặc checkpoint Vanilla/GIRA/CA-TSSA. Mọi output CF-TSSA dùng đường dẫn riêng.
> 5. **Tài nguyên:** Smoke → pilot train-only → decision gate → full benchmark. Không tự động chạy full khi pilot fail. Teacher/probe chỉ dùng khi train; không tăng inference cost.
> 6. **Hạ tầng thực tế:** Local test bằng conda `TSSA`; GPU chạy trong môi trường TSSA trên server `/workspace/long/TSSA`. Không giả định Colab, Hugging Face Hub hoặc quyền tắt máy.

## 1. Nguyên tắc bất biến và mục tiêu học thuật

**Tình trạng hiện tại.** Trên Tay → Việt, BARTpho, seed 42, Vanilla = 25.34 BLEU; TSSA-Pro = 25.36, V4 token barycenter = 24.83, V4 hybrid = 22.18, GIRA A1 = 24.82, CA-TSSA token = 25.53, CA global = 24.06. Bảng lịch sử sử dụng `legacy_best` và không phải bảng fixed-final. Fixed-final 3 seed cho CA token so với Vanilla là `+0.19, −1.27, −0.57` BLEU (mean `−0.55`), CA detach mean khoảng `−0.48`. Vì vậy **không** xem seed 42 là thành công hay xem conflict rate ~45% là bằng chứng gradient routing giải quyết được bài toán. Không gộp `legacy_best` với `fixed_final` để tính hiệu ứng.

**Sửa lỗi fairness trong bản plan trước:** `train_fair_benchmark.py` mặc định `legacy_best`, với `eval_strategy="epoch"`, `load_best_model_at_end=True`, `EarlyStoppingCallback(patience=3)` và `test_dataset` làm eval set. [Thiết lập benchmark cũ](/D:/Code/TSSA/docs/FAIR_BENCHMARK_SETTINGS.md) ghi rõ dùng test để chọn checkpoint; [TSSA-Pro](/D:/Code/TSSA/docs/TSSA_PRO_EXPERIMENTS.md) từng thử nhiều giá trị \(\lambda\) trên test. Đây là **hai việc khác nhau**: chọn checkpoint là một phần giao thức `legacy_best` cần khớp khi so bảng cũ; thử nhiều siêu tham số rồi chọn giá trị tốt nhất không phải yêu cầu bắt buộc của fairness và **không áp dụng cho CF-TSSA trong plan này**. Bảng lịch sử vẫn có selection bias vì dùng test chọn checkpoint, nên không gọi là đánh giá độc lập.

**Giả thuyết có thể bác bỏ:** Các phương pháp trước quyết định neo dựa vào hình học biểu diễn/entropy hoặc chỉnh gradient *sau khi* đã tạo anchor loss. Tín hiệu neo sai về mặt tác vụ vẫn có thể đi qua. CF-TSSA v2 đo xem mỏ neo thật mang lợi ích dịch đặc thù hơn một mỏ neo sai được chuẩn hóa cùng độ lớn hay không, rồi mới bật loss.

**Ba đóng góp cần chứng minh để nhắm ACL:** (C1) thuật toán chọn anchor theo hiệu ứng hữu hạn và đối chứng âm cùng độ lớn, không chỉ theo cosine/gradient sign; (C2) phân tích cơ chế về tỷ lệ được chọn, hiệu ứng thật–giả, fertility/độ dài và liên hệ với BLEU; (C3) hiệu quả nhất quán qua ít nhất BARTpho và ViT5, ba bộ dữ liệu, nhiều seed, không thêm tham số học được hay inference overhead. Nếu thiếu (C1) hoặc (C3), không claim đây là method paper mạnh cho ACL Main.

**Rủi ro novelty quan trọng:** Với can thiệp vô cùng nhỏ, thay đổi NLL xấp xỉ tích vô hướng giữa gradient MT và hướng neo. Đó là một dạng gradient agreement, gần [Xu et al., Findings ACL 2025](https://aclanthology.org/2025.findings-acl.261/) hơn dự thảo trước đã nhận ra. Do đó phải đối chiếu trực tiếp với gradient-sign gate và chứng minh **finite intervention + semantic-specific control** tạo giá trị tăng thêm. Không tuyên bố “đầu tiên” trước khi hoàn thành literature matrix.

## 2. Cây thư mục dự án (Project Directory Tree)

```text
TSSA/
├── data_processed/{tay,rhade,bahnaric}/{train,test}.csv  [FROZEN]
├── docs/                                              [FROZEN: kết quả cũ]
├── models/backbone_adapter.py                         [GIỮ API; chỉ sửa nếu thật cần]
├── models/ca_tssa_seq2seq.py                          [FROZEN]
├── losses/ca_tssa_criterion.py                        [FROZEN]
├── train_fair_benchmark.py                            [MODIFIED: đăng ký model_type mới]
├── models/cf_tssa_seq2seq.py                          [IMPLEMENTED]
├── losses/cf_tssa_criterion.py                         [IMPLEMENTED]
├── training/cf_tssa_trainer.py                         [IMPLEMENTED]
├── scripts/smoke_test_cf_tssa.py                      [IMPLEMENTED]
├── scripts/run_cf_tssa_stage1.py/.sh                  [IMPLEMENTED]
├── scripts/run_cf_tssa_legacy_parity.sh               [IMPLEMENTED]
├── scripts/run_cf_tssa_full_fixed_final.sh            [IMPLEMENTED]
├── scripts/report_cf_tssa.py                          [IMPLEMENTED]
└── checkpoints/fair_benchmark/cf_tssa_*/               [NEW outputs; không ghi đè]
```

Không đổi `TSSADataset`, `tssa_collate_fn`, `get_dataloaders`; dữ liệu mới chỉ cần cùng schema `src_text,tgt_text`. Hỗ trợ backbone encoder–decoder qua adapter, không hardcode hidden size/layer; claim tổng quát chỉ tới các backbone thực sự được test.

## 3. Danh sách file sẽ chỉnh sửa, tạo mới và bảo tồn

| Loại thao tác | Đường dẫn file | Mục đích và phạm vi |
| :--- | :--- | :--- |
| Modified | `train_fair_benchmark.py` | Thêm lựa chọn `cf_tssa`, cấu hình đã khóa, kết nối trainer; không thay đổi nhánh Vanilla/CA và evaluator |
| Modified nếu cần | `models/backbone_adapter.py` | Chỉ bổ sung API decode từ encoder hidden nếu thiếu; giữ nguyên hành vi cũ |
| New | `models/cf_tssa_seq2seq.py`, `losses/cf_tssa_criterion.py`, `training/cf_tssa_trainer.py` | Frozen teacher, đối chứng âm, probe và loss; không thêm tham số học được |
| New | `scripts/smoke_test_cf_tssa.py`, `scripts/run_cf_tssa_stage1.sh`, `scripts/run_cf_tssa_legacy_parity.sh`, `scripts/run_cf_tssa_full_fixed_final.sh`, `scripts/report_cf_tssa.py` | Kiểm thử, pilot, đối sánh lịch sử, fixed-final, hai báo cáo không trộn giao thức |
| New outputs | `checkpoints/fair_benchmark/cf_tssa_*` | Checkpoint, predictions, metrics, diagnostics, timing; tên run riêng |
| Frozen | `data_processed/**`, `models/ca_tssa_seq2seq.py`, `losses/ca_tssa_criterion.py`, checkpoint và báo cáo cũ | Không sửa hoặc ghi đè |

Trước khi code, kiểm tra lại status/diff. Khi hoàn tất code và CPU tests trong conda `TSSA`, mới commit và push `master` theo yêu cầu trước đó; **không push trong giai đoạn lập kế hoạch** và không force-push.

## 4. Đặc tả kỹ thuật chi tiết và novelty

### 4.1. Mỏ neo ứng viên không học tham số

Với câu nguồn `x`, câu đích train `y`, backbone đang học tạo hidden cuối `H=E_θ(x)`. Bản copy pretrained encoder frozen `E₀` tạo `F=E₀(x)` và `T=E₀(y)`. Với token không phải PAD/special:

\[
s_{ij}=\cos(F_i,T_j)/\tau,\quad P_{ij}=\operatorname{softmax}_j(s_{ij}),\quad Q_{ji}=\operatorname{softmax}_i(s_{ij}),\quad R_{ij}=P_{ij}Q_{ji}.
\]

Chọn tối đa một source token `i*` mỗi câu theo `argmax_i Σ_j R_ij`. Mỏ neo `a_i` là trung bình có trọng số `R_ij` của các target states, chuẩn hóa về norm tương thích với `H_i`. Khóa trước `τ=0.10`, xử lý norm gần 0 và phép tính similarity/probe FP32. Alignment là **ứng viên**, không được diễn giải như gold word alignment.

### 4.2. Semantic-specific finite-intervention gate

Tạo `b_i` từ target của **câu khác trong batch**: trong các ứng viên sai, chọn hard negative có cosine với `F_i` gần cosine của `a_i` với `F_i` nhất; tie-break bằng RNG seeded. Không chọn trùng câu/target nếu phát hiện duplicate. Chuẩn hóa hướng can thiệp sai để có cùng norm với hướng thật. Như vậy control không chỉ khớp độ lớn mà còn cố gắng khớp độ dễ theo hình học; lưu sai lệch cosine còn lại để audit. Với `η=0.10`:

\[
d_a=\eta\|H_i\|\frac{a_i-H_i}{\|a_i-H_i\|+\epsilon},\quad
d_b=\eta\|H_i\|\frac{b_i-H_i}{\|b_i-H_i\|+\epsilon}.
\]

Trong `no_grad`, tắt dropout **chỉ cho ba lượt probe decoder** để bảo đảm so sánh cùng điều kiện; chạy teacher forcing và NLL chuẩn hóa theo độ dài câu đích:

\[
u_a=\ell(y\mid H)-\ell(y\mid H+d_a),\qquad
u_b=\ell(y\mid H)-\ell(y\mid H+d_b).
\]

Nhận anchor nếu `u_a > 0` **và** `u_a-u_b > 10^{-4}`. Câu cuối batch nếu chỉ còn một phần tử, hoặc không có hard negative hợp lệ, thì bỏ auxiliary loss và vẫn train MT. Control `b` không phải gold âm tuyệt đối; nó chỉ giúp kiểm tra hiệu ứng có đặc thù hơn một perturbation sai được khớp norm/cosine gần đúng.

Sau probe, trả lại training mode/dropout như cũ, tính `L_MT` bình thường. Decoder luôn dùng `H`, **không dùng** `H+d_a` cho update hay inference. Loss phụ trên câu được nhận:

\[
L_A=\operatorname{mean}_{i\in\mathcal A}\frac{\|H_i-\operatorname{sg}(a_i)\|_2^2}{\operatorname{sg}(\|H_i\|_2^2)+\epsilon},\quad
L=L_{MT}+r(t)\lambda_t L_A.
\]

`r(t)` ramp tuyến tính 10% số bước. `λ_t` là hệ số `stop-gradient` để gradient norm `λ_t∂L_A/∂H` bằng tối đa 5% gradient MT tại encoder interface. Nếu `A` rỗng, `L=L_MT`. Không có projector, router hoặc lớp mới; `Δθ_trainable=0`. Teacher/probe bị loại khỏi `generate()`, nên suy luận giống Vanilla về kiến trúc.

**Cách đọc novelty chính xác:** Phép patch/probe hữu hạn không tự nó mới; phần cần chứng minh là *lọc mỏ neo target-side bằng lợi ích dịch vượt đối chứng semantic mismatch cùng độ lớn*, và lợi ích này không được giải thích đầy đủ bởi gradient sign hay việc chỉ giảm cường độ loss. Tính bậc một cho thấy `u_a≈−⟨∇_Hℓ,d_a⟩`; vì vậy phải có ablation `gradient-sign gate` và `true-only gate` để bác bỏ việc chỉ đổi tên gradient alignment.

### 4.3. Related work bắt buộc đối chiếu

| Công trình | Cơ chế đã có | Ranh giới claim của CF-TSSA v2 |
| :--- | :--- | :--- |
| [AWESOME-align](https://aclanthology.org/2021.eacl-main.181/) | Căn chỉnh token bằng biểu diễn đa ngữ | Không claim mutual alignment là mới |
| [Selective KD for NMT](https://aclanthology.org/2021.acl-long.504/) §3–4 | Chọn tri thức teacher theo đặc tính mẫu/word, distill output | Đối chiếu criterion chọn và nơi can thiệp; không claim selective supervision là mới |
| [Align-to-Distill](https://aclanthology.org/2024.lrec-main.64/) §4 | Distill attention qua module alignment | CF kiểm tra utility tại encoder interface, không distill attention map |
| [Self-Distillation Recipe](https://aclanthology.org/2025.findings-acl.261/) | Warmup và điều chỉnh/triệt gradient KD ngược MT | Bắt buộc so với gradient-sign gate; không claim gradient control là mới |

BibTeX tối thiểu, đối chiếu theo trang chính thức của ACL Anthology:

```bibtex
@inproceedings{dou-neubig-2021-word,
  title={Word Alignment by Fine-tuning Embeddings on Parallel Corpora},
  author={Dou, Zi-Yi and Neubig, Graham}, year={2021},
  url={https://aclanthology.org/2021.eacl-main.181/},
  doi={10.18653/v1/2021.eacl-main.181}
}
@inproceedings{wang-etal-2021-selective,
  title={Selective Knowledge Distillation for Neural Machine Translation},
  author={Wang, Fusheng and Yan, Jianhao and Meng, Fandong and Zhou, Jie}, year={2021},
  url={https://aclanthology.org/2021.acl-long.504/},
  doi={10.18653/v1/2021.acl-long.504}
}
@inproceedings{jin-etal-2024-align,
  title={Align-to-Distill: Trainable Attention Alignment for Knowledge Distillation in Neural Machine Translation},
  author={Jin, Heegon and Son, Seonil and Park, Jemin and Kim, Youngseok and Noh, Hyungjong and Lee, Yeonsoo}, year={2024},
  url={https://aclanthology.org/2024.lrec-main.64/}
}
@inproceedings{xu-etal-2025-self-distillation,
  title={A Self-Distillation Recipe for Neural Machine Translation},
  author={Xu, Hongfei and Liang, Zhuofei and Liu, Qiuhui and Mu, Lingling}, year={2025},
  url={https://aclanthology.org/2025.findings-acl.261/},
  doi={10.18653/v1/2025.findings-acl.261}
}
```

**Novelty gate trước code:** lập literature matrix thêm các bài intervention/activation-patching và selective representation KD gần nhất. Nếu thấy đúng cơ chế so sánh anchor thật với hard negative khớp độ lớn/hình học để chọn auxiliary loss trong NMT, bỏ claim phương pháp mới; chuyển sang câu hỏi thực nghiệm/negative-result paper. Không ép novelty bằng tên gọi. Teacher pretrained trên tiếng Việt có thể mã hóa nguồn ít tài nguyên kém; pilot phải ghi phân bố cosine/alignment và tỷ lệ anchor bị từ chối theo ngôn ngữ, không giả định teacher là gold semantic space.

## 5. Bảng quy chuẩn siêu tham số công bằng

| Thuộc tính | Vanilla | CF-TSSA v2 | Quy tắc |
| :--- | :--- | :--- | :--- |
| Backbone | `vinai/bartpho-syllable`; `VietAI/vit5-base` | Cùng backbone từng cell | Không so trực tiếp điểm tuyệt đối khác backbone |
| Tokenizer/vocab/layers | Nạp từ checkpoint/config tương ứng | Y hệt | Ghi lại config thực khi chạy, không hardcode |
| Tham số học được thêm | 0 | 0 | Teacher frozen chỉ tăng chi phí train/VRAM |
| Dữ liệu | `data_processed/{tay,rhade,bahnaric}/{train,test}.csv` | Cùng file, không đổi split | Không tạo Val mới; thời điểm dùng test tùy giao thức ở bảng dưới |
| Seed | 42, 43, 44 | Cùng seed và data seed | Ghép cặp theo backbone/dataset/seed |
| Epoch / chọn checkpoint | `legacy_best` **hoặc** `fixed_final` | Đúng cùng giao thức trong từng bảng | Không so chéo hai giao thức |
| Batch / accumulation | 16 / 1 | 16 / 1 | Effective batch 16 mỗi device |
| Optimizer | AdamW, weight decay 0.01 | Giống | Các tham số còn lại cùng runner |
| LR | BARTpho `2e-5`; ViT5 `1e-4` | Giống trong từng backbone | Giữ nguyên LR cũ; không thêm sweep LR |
| Warmup | 500 optimizer steps | 500 optimizer steps | Phân biệt với CF ramp 10% |
| Precision | BARTpho FP16; ViT5 BF16 | Giống | Probe/similarity FP32 |
| Length | Source/target tối đa 256 | Giống | Cùng tokenizer |
| Decoding | Beam 4, length penalty 1.0 | Giống | `generate()` không teacher/probe |
| Metrics | SacreBLEU exp smoothing, chrF++ word_order=2, METEOR, COMET wmt22 | Giống | Lưu SacreBLEU signature và evaluator version |

| Giao thức | Các phương pháp được so trực tiếp | Chọn checkpoint | Chọn hyperparameter | Vai trò |
| :--- | :--- | :--- | :--- | :--- |
| **H — historical parity** | Vanilla, TSSA-Pro/V4/GIRA/CA và CF **cùng `legacy_best`** | Test BLEU sau mỗi epoch, best model, patience 3; tối đa 5 epoch | CF dùng đúng **một cấu hình cố định**, không sweep/chọn hyperparameter bằng test | Đối chiếu trực tiếp bảng cũ; checkpoint vẫn selected-on-test |
| **F — fixed-final** | Vanilla, CA và các baseline được **chạy lại bằng `fixed_final`** cùng CF | Epoch 5 cuối, không early stopping, test sau train | Dùng **chính cấu hình CF của H**, không thay đổi theo kết quả H | Kiểm tra độ bền đa seed/backbone; không trộn với điểm H |

**Cấu hình CF cố định trước khi chạy:** `τ=0.10`, `η=0.10`, margin `1e-4`, gradient budget 5%, ramp 10%, tối đa một token/câu và một hard negative/câu. Giá trị `λ_t` thích ứng theo gradient norm **trong một run** là thành phần thuật toán, không phải sweep. Chạy một CF và một Vanilla phù hợp giao thức cho mỗi cell/seed; các ablation được báo cáo riêng để kiểm tra cơ chế, không chọn ablation thắng test làm phương pháp chính. Những phương pháp cũ có ngân sách tune khác nhau thì ghi nhận như một giới hạn của bảng lịch sử; cùng training/evaluation protocol không tự động làm bằng nhau cả ngân sách phát triển. Chi phí train không đối xứng với Vanilla, nên báo cáo wall-time, GPU hours, VRAM; fairness ở dữ liệu/optimizer steps, không claim bằng compute.

## 6. Quy trình preflight smoke test bắt buộc

**Script smoke:** `python scripts/smoke_test_cf_tssa.py`. Chạy trên server trong môi trường TSSA, bắt đầu B=2 với tiny BART/T5 rồi BARTpho/ViT5 một batch; forward/backward/generate/save-load. Kiểm tra teacher frozen, `Δθ=0`, PAD/special không được chọn, không lấy negative cùng target sentence, norm hai patch bằng nhau, mode/dropout được phục hồi, gradient MT tới encoder, anchor loss bằng 0 khi không accepted, gradient budget 5%, không NaN/Inf FP16/BF16, generation không gọi teacher. Test gradient-sign và full-size optimizer step vẫn cần bổ sung trước khi claim cơ chế hoàn chỉnh; smoke hiện tại chưa chứng nhận novelty.

**Cổng:** mọi assert PASS mới chạy pilot. Thời lượng chỉ báo sau khi đo trên server; không cam kết 30–45 giây cho hai backbone full-size.

## 7. Cơ chế thực thi bằng Bash trên server TSSA

```text
Code + CPU tests trong conda TSSA local
  → kiểm tra diff/status → commit + push master (không force)
  → server: pull + smoke
  → pilot 500 steps, train-only diagnostics và shuffled control
  → PASS mới chạy historical parity: legacy_best Tay × BARTpho × seed 42,
    một cấu hình CF cố định; test chọn checkpoint như bảng cũ
  → report H + decision gate; nếu GO mới chạy fixed-final 2 backbones × 3 datasets × 3 seeds
  → report F + ablations + claim gate; không trộn điểm H/F, không tự động nộp paper
```

**Lệnh chạy sau khi code được push:**

```bash
git status --short; if git pull origin master; then python scripts/smoke_test_cf_tssa.py; fi
```

Nếu smoke PASS, chạy pilot kỹ thuật (500 bước CF và 500 bước Vanilla để đo cost; chỉ dữ liệu train, không đọc test):

```bash
bash scripts/run_cf_tssa_stage1.sh
```

Nếu pilot PASS, chạy đối sánh đúng giao thức cũ (`legacy_best`) với **một cấu hình CF cố định**; script không sweep và không tự khởi chạy multi-seed:

```bash
bash scripts/run_cf_tssa_legacy_parity.sh tay 42
```

Chỉ sau khi xem báo cáo H và người dùng quyết định GO, chạy xác nhận theo giao thức mới (`fixed_final`) cho **cả Vanilla/baseline được so và CF**:

```bash
bash scripts/run_cf_tssa_full_fixed_final.sh
```

Runner phải `set -euo pipefail`, ghi manifest gồm `selection_protocol` và cấu hình CF cố định, chỉ skip run khi đủ checkpoint, metrics, predictions/diagnostics đã xác nhận; không ghi đè run dở. Báo cáo H và F lưu riêng. Không tự động upload Hugging Face hoặc tắt server. Lệnh dùng `;` thay vì `&&` theo yêu cầu trước; `if` ngăn smoke chạy nếu pull thất bại.

## 8. Dự toán thời gian và tài nguyên

Không có số đo wall-time/step, VRAM trống, dung lượng server hay đơn giá máy hiện tại đủ tin cậy để điền số giờ/CU. Trước full run, pilot ghi `seconds/optimizer_step`, peak VRAM và checkpoint size cho từng backbone. Dự toán:

\[
T_{full}\approx\sum_{b,d,s}(N_{steps}(b,d,s)\times t_{step}^{CF}(b,d)+t_{eval}(b,d,s))
\]

Track H có 1 CF và 1 Vanilla/đối chứng trên Tay × BARTpho × seed 42; kiểm tra cache Vanilla đúng `legacy_best` trước khi bỏ qua. Track F có tối đa 18 CF runs (2 backbone × 3 dataset × 3 seed), cộng các Vanilla và baseline fixed-final còn thiếu; **không tự chạy tất cả** trước decision gate. Vì có thêm track H và có thể cần chạy lại baseline cho F, dự toán phải cộng từng run thực tế, không dùng con số cũ. Mục tiêu chi phí `t_step^CF/t_step^Vanilla ≤ 3`; nếu vượt, tối ưu batching probe một lần; nếu vẫn vượt, dừng và báo lại. Compute Units/số dư Colab và SSD `/content`: **không áp dụng** cho server TSSA hiện tại; đo `nvidia-smi`, `df -h` trên server trước khi chạy.

## 9. Ma trận kết quả và cổng quyết định cho bài báo

| Backbone | Dataset | Phương pháp | Tham chiếu | Δθ học được | BLEU | chrF++ | Vai trò |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | :--- |
| BARTpho | Tay | Vanilla và các phương pháp cũ, seed 42, **H** | lịch sử | tùy method | điểm trong bảng cũ | điểm trong bảng cũ | `legacy_best`; tham chiếu cho đối sánh trực tiếp H |
| BARTpho | Tay | CF-TSSA v2, một cấu hình, seed 42, **H** | đề xuất | 0 | TBD | TBD | Cùng `legacy_best`; không chọn hyperparameter từ test |
| BARTpho | Tay | Vanilla, 3 seed, **F** | nội bộ | 0 | `25.34/25.72/24.53` | TBD theo seed | Baseline fixed-final đã có |
| BARTpho | Tay | CA-TSSA token, 3 seed, **F** | nội bộ | >0 (projector) | `25.53/24.45/23.96` | TBD theo seed | Kết quả âm fixed-final; mean ΔBLEU `−0.55` |
| BARTpho | Tay | CF-TSSA v2, 3 seed, **F** | đề xuất | 0 | TBD | TBD | Đối sánh fixed-final, không dùng điểm H để tính delta |
| BARTpho | Rhade/Bahnaric | Vanilla và CF-TSSA v2, 3 seed, **F** | nội bộ/đề xuất | 0 | TBD | TBD | Generalization cùng fixed-final |
| ViT5 | Tay/Rhade/Bahnaric | Vanilla và CF-TSSA v2, 3 seed, **F** | nội bộ/đề xuất | 0 | TBD | TBD | Backbone generalization cùng fixed-final |
| BARTpho + ViT5 | Các cell GO | CF không gate; true-only gate; gradient-sign gate; shuffled-target control | ablations | 0 | TBD | TBD | Kiểm tra novelty và giải thích thay thế |

**Pilot GO:** accepted rate hữu dụng không bằng 0, hiệu ứng `u_true−u_control` có phân bố dương ổn định trên train và vượt shuffled-target control; thời gian ≤3× Vanilla. Đây chỉ là cổng khả thi, không phải bằng chứng BLEU.

**Historical-parity GO (H):** cấu hình CF đã khóa, với checkpoint được chọn bằng `legacy_best` như Vanilla cũ, phải vượt Vanilla cùng giao thức và đáng để tốn compute cho F. Chỉ một cấu hình CF được chạy; không đổi nó sau khi xem điểm H. Đây là cổng thăm dò, **không** phải bằng chứng chất lượng độc lập.

**Fixed-final GO (F) cho method paper:** macro paired ΔBLEU ≥ `+0.30`; cả hai backbone có mean dương; ≥4/6 backbone–dataset cells dương; không cell nào giảm quá `−0.20`; chrF++/COMET không mâu thuẫn lớn; ablation chứng minh specificity gate vượt true-only và gradient-sign gate với cùng budget. Baseline nào được so trong F phải có run F của chính nó; kết quả H chỉ đặt ở bảng lịch sử riêng. Báo cáo paired bootstrap theo câu **và** biến thiên theo seed, không gọi 3 seed là chứng minh thống kê mạnh.

**ACL claim gate:** Chỉ viết method contribution mạnh nếu literature matrix xác nhận khoảng trống, F full-run GO, ablation GO và chi phí được báo cáo. [ARR review form](https://aclrollingreview.org/reviewform) đánh giá tính soundness, reproducibility, novelty/impact; plan không thể bảo đảm acceptance. H cố ý tái hiện cách chạy cũ nên có selection bias do test chọn checkpoint; F cũng không trở thành độc lập hoàn toàn vì quyết định mở rộng nghiên cứu dựa vào H và các test hiện tại đã được xem nhiều lần. Người dùng hiện không muốn thêm dataset/test mới: ghi hạn chế này công khai, không gọi kết quả là xác nhận độc lập. Nếu gate fail, dừng thí nghiệm mở rộng, viết phân tích negative results hoặc tái định vị bài báo, không đổi tên phương pháp để cứu claim.
