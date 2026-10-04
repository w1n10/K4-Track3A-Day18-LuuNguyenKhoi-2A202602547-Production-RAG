# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Lưu Nguyễn Khôi — 2A202602547  
**Khóa:** K4 - Track 3A  

**Cấu hình chạy:** LLM sinh câu trả lời + enrichment + RAGAS judge: `gemini-3.1-flash-lite` (OpenAI-compatible endpoint) · RAGAS embeddings: `gemini-embedding-001` · Dense: `BAAI/bge-m3` · Reranker: `BAAI/bge-reranker-v2-m3` (CPU) · 20 câu hỏi trong `test_set.json`.

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.9000 | **0.9267** | +0.0267 |
| Answer Relevancy | 0.7731 | **0.8843** | +0.1113 |
| Context Precision | 0.7500 | **0.7750** | +0.0250 |
| Context Recall | 0.8000 | **0.9250** | +0.1250 |

- Naive = paragraph chunking (51 chunks) + dense-only top-3, không rerank, không enrichment.
- Production = hierarchical chunking (104 child / 26 parent) + combined enrichment (1 call/chunk) + BM25 + Dense + RRF → cross-encoder rerank → trả về **parent** của top child (tối đa 3 parent khác nhau) → LLM.
- Cải thiện lớn nhất ở **context_recall (+0.125)**: retrieve child (nhỏ, chính xác) nhưng trả parent (cả tài liệu chính sách) nên LLM thấy đủ ngữ cảnh. **answer_relevancy (+0.111)** tăng nhờ prompt yêu cầu trả lời thẳng vào câu hỏi + xử lý phiên bản chính sách.
- **context_precision tăng ít nhất (+0.025)** — đây là điểm yếu còn lại (xem #3, #4, #5).

> Lưu ý trung thực: trong bước enrichment có 3/104 chunk gọi API thất bại (2 lần `429 RESOURCE_EXHAUSTED` do free tier giới hạn 15 request/phút, 1 lần `503 UNAVAILABLE`) → 3 chunk đó dùng text gốc không có context prepend. Context precision = 0 ở các câu #3–#5 cũng có thể bị ảnh hưởng một phần bởi judge `flash-lite` (model nhỏ) — xem phần Case Study.

## Bottom-5 Failures

### #1 — Multi-hop: phép năm + lương (avg 0.375)
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** 15 + 3 = 18 ngày phép (v2024). Lương Senior (P3–P4): 20–35 triệu VNĐ/tháng.
- **Got:** "18 ngày phép năm (15 + 3 thâm niên, v2024) … Context không cung cấp thông tin về mức lương."
- **Scores:** faithfulness 1.0 · answer_relevancy **0.0** · context_precision **0.0** · context_recall 0.5
- **Worst metric:** answer_relevancy (0.0) — RAGAS đánh câu trả lời "không có thông tin" là *noncommittal* → 0.
- **Error Tree:** Output sai (thiếu nửa câu trả lời) → Context đúng? **KHÔNG** — top-3 parent là `nghi_phep_nam_v2024`, `nghi_phep_nam_v2023`, `nghi_phep_khong_luong`; thiếu `bang_luong_2024.md` → Query OK? **KHÔNG** — 1 query chứa 2 ý (phép + lương), phần "nghỉ phép" áp đảo cả BM25 lẫn reranker nên cả 3 slot đều thuộc chủ đề nghỉ phép → **Fix ở bước Retrieval/Query**.
- **Root cause:** Câu hỏi multi-hop nhưng pipeline chỉ chạy 1 lượt retrieval, top-3 bị chiếm hết bởi 1 chủ đề. LLM làm đúng khi không bịa lương (faithfulness 1.0), nhưng câu trả lời bị thiếu.
- **Suggested fix:** (1) Query decomposition: LLM tách thành 2 sub-query ("ngày phép 9 năm thâm niên", "lương Senior") → retrieve riêng → gộp; (2) đa dạng hoá context — giới hạn tối đa 2 parent cùng `category`/chủ đề; (3) tăng `RERANK_TOP_K` lên 4–5 cho câu hỏi dài.

### #2 — Numeric: phạt tạm ứng quá hạn (avg 0.656)
- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Hạn 15 ngày → quá 5 ngày; phí 2%/tháng × 15 triệu = 300.000 VNĐ/tháng (~50.000 VNĐ pro-rata 5 ngày).
- **Got:** "Số tiền phạt là 100.000 VNĐ" … rồi tính ra 50.000 VNĐ … rồi kết luận 300.000 VNĐ — **3 con số mâu thuẫn nhau**.
- **Scores:** faithfulness **0.2** · answer_relevancy 0.92 · context_precision 1.0 · context_recall 0.5
- **Worst metric:** faithfulness (0.2)
- **Error Tree:** Output sai → Context đúng? **CÓ** — `tam_ung.md` rank 1 (rerank score 0.968), precision 1.0 → Query OK? **CÓ** → **Fix ở bước Generation**.
- **Root cause:** LLM nhỏ (`flash-lite`, thinking ở mức minimal) tính toán nhiều bước kém: đưa ra con số 100.000 không có trong context rồi tự sửa giữa chừng. Prompt yêu cầu "nêu phép tính" nhưng không ép thứ tự "tính trước → kết luận sau".
- **Suggested fix:** Prompt dạng "liệt kê dữ kiện → tính từng bước → 1 dòng KẾT LUẬN duy nhất ở cuối"; bật `reasoning_effort="low"` cho câu hỏi numeric (router theo loại câu hỏi), hoặc dùng model lớn hơn cho bước generation.

### #3 — Version: thâm niên cộng phép (avg 0.722)
- **Question:** Thâm niên bao nhiêu năm thì được cộng thêm ngày phép?
- **Expected:** v2024: từ 3 năm, +1 ngày mỗi 3 năm (v2023 cũ: 5 năm).
- **Got:** "Theo phiên bản 2024, từ 3 năm trở lên … Phiên bản 2023 (5 năm) đã bị thay thế." — **đúng**.
- **Scores:** faithfulness 1.0 · answer_relevancy 0.89 · context_precision **0.0** · context_recall 1.0
- **Worst metric:** context_precision (0.0)
- **Error Tree:** Output đúng → Context đúng? **Một phần** — tài liệu **cũ** `nghi_phep_nam_v2023.md` xếp **rank 1** (0.993), bản hiện hành v2024 chỉ rank 2 (0.986), rank 3 là `nghi_phep_khong_luong` không liên quan → Query OK? CÓ → **Fix ở bước Retrieval (ranking/metadata)**.
- **Root cause:** Hai phiên bản gần như giống hệt về từ ngữ → BM25, dense và cross-encoder đều không phân biệt được "hiện hành" vs "đã thay thế". Ngữ cảnh lỗi thời đứng đầu bị judge coi là không hữu ích.
- **Suggested fix:** Thêm metadata `version` / `effective_date` / `superseded: true` khi ingest; filter bỏ tài liệu đã bị thay thế (hoặc phạt điểm), hoặc boost bản mới nhất khi 2 parent cùng chủ đề.

### #4 — Cross-doc: phân loại thông tin lương (avg 0.725)
- **Question:** Thông tin lương thuộc cấp độ phân loại dữ liệu nào?
- **Expected:** "Bí mật" (cấp 3) — theo quy chế lương + chính sách phân loại dữ liệu (mã hoá khi truyền, need-to-know).
- **Got:** "Thông tin lương thuộc cấp độ 3: **Bí mật**." — đúng nhưng ngắn.
- **Scores:** faithfulness 1.0 · answer_relevancy 0.90 · context_precision **0.0** · context_recall 1.0
- **Worst metric:** context_precision (0.0)
- **Error Tree:** Output đúng → Context đúng? **CÓ nhưng thứ tự chưa tốt** — rank 1 `ky_luong.md` (rerank chỉ **0.242**), rank 2 `phan_loai_du_lieu.md` (0.113), rank 3 `danh_gia_hieu_suat.md` (0.005, nhiễu) → Query OK? CÓ → **Fix ở Reranking + kiểm chứng judge**.
- **Root cause:** Reranker không chắc chắn (score thấp), luôn lấy cứng 3 parent nên kéo theo context rác. Context recall = 1.0 nhưng precision = 0.0 là mâu thuẫn — khả năng cao judge `flash-lite` đánh giá sai trên context dài (cả tài liệu) → điểm metric này có độ nhiễu.
- **Suggested fix:** Ngưỡng rerank (VD bỏ context có `rerank_score < 0.05`) thay vì luôn lấy 3; chạy lại RAGAS với judge mạnh hơn hoặc `strictness=3` để kiểm chứng metric.

### #5 — Version: chu kỳ đổi mật khẩu (avg 0.735)
- **Question:** Bao lâu phải đổi mật khẩu một lần?
- **Expected:** v2.0: 120 ngày (v1.0 cũ: 90 ngày).
- **Got:** "Mỗi 120 ngày (theo phiên bản 2.0, đã thay thế phiên bản 1.0)." — **đúng**.
- **Scores:** faithfulness 1.0 · answer_relevancy 0.94 · context_precision **0.0** · context_recall 1.0
- **Worst metric:** context_precision (0.0)
- **Error Tree:** Output đúng → Context đúng? **Một phần** — `mat_khau_v1.md` (cũ) rank 1 (0.948) trên `mat_khau_v2.md` (0.935); rank 3 `bao_mat_su_co.md` không liên quan → Query OK? CÓ → **Fix ở Retrieval (metadata version)**.
- **Root cause:** Giống #3 — conflict giữa các phiên bản tài liệu. Prompt "ưu tiên phiên bản mới nhất" cứu được câu trả lời (faithfulness 1.0), nhưng retrieval vẫn đưa bản cũ lên đầu.
- **Suggested fix:** Như #3 — metadata `superseded` + filter. Đây là pattern lặp lại ở 3/20 câu (#3, #5, và câu "nghỉ phép năm" có precision 0.5) nên fix 1 lần được nhiều câu.

**Ngoài bottom-5 —** câu "khoá học 25 triệu, nghỉ sau 8 tháng" (faithfulness 0.33): câu trả lời mở đầu bằng **"Có."** cho một câu hỏi không phải Có/Không — do quy tắc trong prompt *"Với câu hỏi Có/Không, mở đầu bằng Có/Không"* bị LLM áp dụng sai. Câu "laptop 30 triệu" cũng bị tương tự. Fix: bỏ quy tắc này hoặc để LLM tự phân loại câu hỏi trước.

## Tổng hợp theo Diagnostic Tree

| Nhóm lỗi | Số câu (trong bottom-10) | Bước cần fix | Fix ưu tiên |
|---|---|---|---|
| Conflict phiên bản tài liệu (v1/v2, 2023/2024) | 3 | Retrieval | Metadata `superseded` + filter |
| Multi-hop thiếu tài liệu thứ 2 | 1 | Query / Retrieval | Query decomposition |
| Tính toán nhiều bước sai | 1 | Generation | Prompt step-by-step / model lớn hơn |
| Prompt rule áp sai ("Có." cho câu không phải Có/Không) | 2 | Generation | Sửa prompt template |
| Context nhiễu ở rank 3 | 2 | Reranking | Ngưỡng rerank score |

## Latency Breakdown (bonus)

Đo trên laptop CPU (không GPU), 20 queries — nguồn: `reports/latency_report.json`.

| Bước (build, 1 lần) | Thời gian |
|---|---|
| Chunking (26 docs → 104 child) | 0.18 s |
| Enrichment (104 LLM calls, 2 luồng, free tier) | 355.76 s |
| Indexing (BM25 + bge-m3 + Qdrant) | 68.76 s |
| Load reranker | 13.33 s |

| Bước (mỗi query) | avg | p50 | p95 | max |
|---|---|---|---|---|
| Hybrid search | 348 ms | 319 ms | 748 ms | 748 ms |
| Rerank (20 candidates, CPU) | **8 782 ms** | 8 801 ms | 10 033 ms | 10 033 ms |
| LLM generation | 5 739 ms | 5 353 ms | 14 103 ms | 14 103 ms |
| **Tổng** | **14 869 ms** | 14 078 ms | 23 148 ms | 23 148 ms |

→ Nút cổ chai là **reranker trên CPU (~59% latency)**: `bge-reranker-v2-m3` (568M params) chấm 20 cặp × ~512 tokens. Hướng giảm: rerank top-10 thay vì 20, dùng GPU, hoặc `FlashrankReranker` (đã implement trong M3) cho môi trường latency-sensitive. Spike LLM ở p95 (14 s) do retry khi chạm rate limit.

## Case Study (cho presentation)

**Question chọn phân tích:** "Thâm niên bao nhiêu năm thì được cộng thêm ngày phép?" (#3) — đại diện cho lỗi phổ biến nhất (conflict phiên bản, 3/20 câu).

**Error Tree walkthrough:**
1. **Output đúng?** → CÓ. LLM trả lời đúng v2024 (3 năm) và nêu v2023 đã bị thay thế → faithfulness 1.0, answer_relevancy 0.89.
2. **Context đúng?** → CÓ nhưng **sai thứ tự**: tài liệu lỗi thời `nghi_phep_nam_v2023.md` đứng rank 1, v2024 rank 2 → context_precision 0.0. Recall = 1.0 chứng tỏ thông tin cần thiết *có mặt*, vấn đề là *ranking* và *nhiễu*.
3. **Query rewrite OK?** → Không cần rewrite: query rõ ràng, lỗi nằm ở chỗ 2 phiên bản tài liệu gần như trùng từ ngữ — mọi retriever dựa trên độ tương đồng đều không phân biệt được.
4. **Fix ở bước:** **Indexing/Retrieval** — đưa thông tin phiên bản thành metadata có cấu trúc (`version`, `effective_date`, `superseded_by`) khi ingest, filter tài liệu đã bị thay thế trước khi rerank. Hiện tại prompt đang "chữa cháy" ở bước generation; nếu LLM yếu hơn hoặc prompt khác, câu trả lời sẽ sai theo bản cũ.

**Nếu có thêm 1 giờ, sẽ optimize:**
- Metadata `superseded` cho 2 cặp tài liệu v1/v2 + filter trong `HybridSearch.search()` → kỳ vọng sửa #3, #5 và câu "nghỉ phép năm".
- Query decomposition cho câu hỏi multi-hop (#1).
- Sửa prompt: bỏ rule "Có/Không", thêm format "tính từng bước → kết luận 1 dòng" (#2 và câu khoá học).
- Retry có backoff theo `retryDelay` của lỗi 429 trong M5 để không còn chunk nào mất enrichment.
- Chạy lại RAGAS với judge mạnh hơn để kiểm chứng các giá trị context_precision = 0 đáng ngờ.
