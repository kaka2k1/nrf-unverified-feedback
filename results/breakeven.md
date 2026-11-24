# Phân tích "tỷ giá" (break-even) — feedback chưa verify đáng giá bao nhiêu mẫu verify? (2026-11-18, CPU)

**Đã chạy thật** (CPU ≤2 core, không Kaggle). Script: `breakeven_cpu.py` (tổng hợp), `bfcl_breakeven_cpu.py` (BFCL feedback thật),
`scripts/analyze_breakeven.py` (tính tỷ giá), `scripts/make_breakeven_outputs.py` (bảng/hình paper).
Số liệu đầy đủ: `results/cpu_breakeven/exchange_rates.md`, `results/cpu_bfcl_breakeven/exchange_rates.md`.

## Thiết kế
- Với mỗi seed, mặt nạ verify lồng nhau theo ρ (cùng RNG) → đường cong `verified_only` a_V(ρ) trên lưới
  ρ ∈ {0, .02, .05, .1, .15, .25, .35, .5, .75, 1}. Tỷ giá của phương pháp M tại ρ: ρ_eq/ρ với a_V(ρ_eq) = a_M(ρ)
  (nghịch đảo nội suy đường cong trung bình; CI bootstrap theo seed). >1: pool chưa verify đáng giá thêm mẫu verify.
- Learner CPU 3 mức mạnh: TF-IDF+LR (seed-only 59.8%), bge-small đóng băng+LR (74.2%), bge-base đóng băng+LR (77.8%).
  20 seed; 4 điều kiện: context 40%, source 40%, uniform 20%, context 20%.
- Baseline mới **self_train**: train trên seed+verified, gán nhãn giả cho input chưa verify, BỎ QUA nhãn feedback
  (trả lời câu hỏi "nhãn feedback có thêm gì ngoài input không?"). Có test rò rỉ `test_self_train_no_leakage`.
- BFCL thật: 7 nguồn LLM, cùng giao thức split như `real_feedback.py`, learner = bge-small đóng băng, scorer cặp
  (query, tên function) LR C=1000 (view "names"), 20 seed. Nhiễu thực 12.9%.

## Kết quả chính
1. **Tỷ giá giảm mạnh theo ρ** ở mọi điều kiện: CA-EB (bge-base, context 40%) 5.3 [3.6,7.5] ở ρ=0.02 (+4.0±1.4 điểm so
   với verified_only) → 2.7 (ρ=.05) → 1.8 (ρ=.1) → 1.3 (ρ=.25, +0.9±1.1, không có ý nghĩa). Source 40%: 4.6 (+3.6±1.2)
   → 0.90 (−0.5±0.9). → "không thắng verified_only" ở GPU (ρ=0.1/0.25, learner mạnh hơn) là hệ quả của chế độ
   ngân sách verify, không phải phương pháp vô dụng: với verify rất thưa (2–5%), CA-EB thắng verified_only có ý nghĩa.
2. **Learner càng mạnh, tỷ giá càng thấp** với phương pháp không dùng confidence: source_prior ở ρ=.02 context 40%:
   6.5 (TF-IDF) → 2.6 (bge-small) → 1.7 (bge-base); naive ở nhiễu 40% từ có lợi (TF-IDF, 4.4) thành có hại (embedding,
   thua cả seed-only). CA-EB giữ ~4.8–5.3 ở cả 3 learner.
3. **Confidence filter gần như không mang thông tin mới**: tỷ giá 0.8–2.2 với learner embedding (≈1 khi ρ≥0.05), <1 với TF-IDF — nó chỉ giữ
   những gì model đã tin (khớp phát hiện "từ chối dạy điều mới").
4. **Self-training (bỏ nhãn feedback)**: tỷ giá 1.0–1.6 (embedding), <1 (TF-IDF). Nhãn feedback có trọng số (CA-EB/Ctx-EB)
   đáng giá gấp nhiều lần chỉ input ở ρ thấp → nhãn feedback thực sự mang thông tin.
5. **BFCL thật (CPU)**: đường cong verified_only rất phẳng (84.8% seed-only → 88.2% verify hết). ρ=.02: source_prior 4.1
   [2.3,8.1] (+1.2±1.0), ctx_eb 3.9; điểm hoà vốn nằm giữa ρ=.05 và .1; ở ρ=.25 MỌI phương pháp thua verified_only có ý
   nghĩa (source_prior −1.0±0.6, Ctx-EB −1.2±0.6, CA-EB −1.5±0.5). Khớp kết quả GPU real1/real2 (ρ=.1/.25 không lợi).
5b. **BFCL view "full" (thấy cả mô tả function)** `results/cpu_bfcl_breakeven_full/`: similarity đóng băng đã 91.3%,
   thêm dữ liệu (kể cả verify hết) không giúp (90.6% ở ρ=1); mọi phương pháp thêm feedback chưa verify thấp hơn
   verified_only 0.7–2.0 điểm ở mọi ρ (kể cả naive) (learner bão hoà — đầu cực của kết quả 2).
6. **Dự đoán heuristic thất bại** (trung thực): công thức ρ + W·κ(π)/N (thông tin Fisher của nhiễu đối xứng) đánh giá
   quá cao tỷ giá, nhất là với conf_filter (dự đoán ~12 vs đo 1.6 ở ρ=.02): mẫu được chọn theo confidence là mẫu "dễ",
   ít thông tin — số lượng × độ chính xác nhãn không đủ để dự đoán giá trị.
7. Lưu ý: `oracle_clean` loại cả mẫu ĐÃ verify có feedback sai, nên không phải cận trên khi ρ lớn (BFCL ρ=.25: oracle
   −0.8±0.6 so với verified_only).

## Giới hạn
Learner CPU (embedding đóng băng) chứ không phải LLM fine-tune; benchmark tổng hợp do mình thiết kế; BFCL chỉ chọn
function. Liên hệ với GPU là "nhất quán", không phải chứng minh.
