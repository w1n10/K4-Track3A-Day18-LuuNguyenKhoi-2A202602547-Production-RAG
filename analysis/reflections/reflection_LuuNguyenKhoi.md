# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Lưu Nguyễn Khôi — 2A202602547  
**Khóa:** K4 - Track 3A  
**Ngày hoàn thành:** 04/10/2026

---

## Phần 1: Mapping bài giảng (Lecture Mapping)

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|--------|-------------|--------------------------|
| Semantic chunking | M1 | `chunk_semantic()` | Threshold 0.85 với `all-MiniLM-L6-v2` tạo **208 chunks (avg 99 ký tự, min 6)** so với basic **51 chunks (avg 410)** trên toàn corpus. Chunk quá vụn vì MiniLM là model tiếng Anh: cosine giữa 2 câu tiếng Việt liền kề thường < 0.85 → tách gần như từng câu. Muốn dùng thật cần embedding đa ngôn ngữ (bge-m3) hoặc hạ threshold. |
| Hierarchical (parent-child) | M1 | `chunk_hierarchical()` | Pipeline: **104 child (≤256 ký tự) / 26 parent**. Mỗi file chính sách < 2048 ký tự nên 1 parent ≈ 1 tài liệu. Retrieve child → trả parent giúp **context_recall 0.80 → 0.925**, mức tăng lớn nhất trong 4 metric. |
| Structure-aware chunking | M1 | `chunk_structure_aware()` | 106 chunks, avg 196, max 788 — giữ nguyên section theo header `##`, mỗi chunk có `section` metadata. Section dài không bị cắt nên max cao hơn hẳn các strategy khác. |
| BM25 + Dense fusion | M2 | `reciprocal_rank_fusion()` | RRF (k=60) chỉ dùng **thứ hạng**, không dùng score → không cần chuẩn hoá thang điểm BM25 (0–20+) với cosine (0–1). BM25 bắt từ khoá chính xác ("MFA", "PVI", số tiền), dense bắt diễn đạt khác. Phải `replace("_", " ")` sau `underthesea` để "nghỉ phép" khớp token. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | Latency **~8.8 s/query** (20 candidates, CPU) — chiếm ~59% tổng latency. Đổi lại phân tách relevance rất rõ: VD câu tạm ứng `tam_ung.md` 0.968 vs tài liệu kế tiếp 0.005. Điểm yếu: không phân biệt được 2 phiên bản tài liệu gần giống nhau (v2023 0.993 vs v2024 0.986). |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | Production: faithfulness 0.927, answer_relevancy 0.884, context_precision **0.775 (thấp nhất)**, context_recall 0.925. Precision thấp vì luôn lấy cứng 3 parent → context nhiễu ở rank 2–3, và tài liệu phiên bản cũ đứng rank 1. Phát hiện thêm: answer_relevancy = 0 khi câu trả lời là "không có thông tin" (noncommittal). |
| Contextual embeddings | M5 | `_enrich_single_call()` | Combined mode: **1 call/chunk × 104 chunks = 355.8 s** (thay vì 416 calls nếu gọi 4 technique riêng). Câu context prepend (VD "Đoạn này thuộc chính sách nghỉ phép năm v2024…") giúp chunk nhỏ không mất ngữ cảnh tài liệu. Dùng JSON mode (`response_format`) để tránh lỗi parse khi LLM bọc JSON trong ```` ```json ````. |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

### Lỗi 1 — Không cài được dependencies trên Python 3.13
- **Exact error:**
  ```
  Downloading numpy-1.26.4.tar.gz (15.8 MB)
  Preparing metadata (pyproject.toml): finished with status 'error'
  error: subprocess-exited-with-error
  ```
- **Nguyên nhân & debug:** Đọc log thấy pip đang build numpy **từ source** (.tar.gz) thay vì tải wheel. Truy ngược: `ragas<0.2` → `langchain-community<0.3` → ép `numpy<2`, mà numpy 1.26.4 không có wheel cho Python 3.13 → phải compile C và thất bại. File `.python-version` của repo ghi `3.11` nhưng máy chỉ cài 3.13.
- **Cách sửa:** Tạo venv bằng `uv venv --python 3.11 .venv` rồi `uv pip install -r requirements.txt`.

### Lỗi 2 — Windows chặn DLL của torch / pandas / tiktoken / pyarrow
- **Exact error:**
  ```
  OSError: [WinError 4551] An Application Control policy has blocked this file.
  Error loading "...\.venv\Lib\site-packages\torch\lib\torch.dll" or one of its dependencies.
  ```
  sau đó tiếp tục: `ImportError: DLL load failed while importing internals: An Application Control policy has blocked this file.` (pandas), tương tự với `_tiktoken` và `pyarrow._dataset`.
- **Nguyên nhân & debug:** Máy đang bật **Smart App Control** (kiểm tra bằng `Get-MpComputerStatus` → `SmartAppControlState: On`). Tính năng này chặn binary chưa có "reputation" — các bản phát hành rất mới (torch 2.14.1, pandas 3.0.6, tiktoken 0.14.0, pyarrow 25) bị chặn. Không tắt Smart App Control vì tắt rồi không bật lại được nếu không cài lại Windows. Viết vòng lặp `import` từng package có phần native để tìm hết package bị chặn thay vì sửa từng lỗi một.
- **Cách sửa:** Hạ về các bản phổ biến hơn: `torch==2.6.0+cpu`, `pandas==2.2.3`, `tiktoken==0.8.0`, `pyarrow==21.0.0` → import thành công, 37/37 tests pass.

### Lỗi 3 — Gemini free tier bị rate limit khi enrichment
- **Exact error:**
  ```
  Error code: 429 - ... Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests,
  limit: 15, model: gemini-3.1-flash-lite ... Please retry in 20.452434556s.
  ```
  và `Error code: 503 - This model is currently experiencing high demand.`
- **Nguyên nhân & debug:** Free tier giới hạn 15 request/phút/model; enrichment gọi 104 lần và RAGAS mặc định chạy 16 luồng song song. SDK `openai` retry với backoff ngắn hơn `retryDelay` (20 s) mà server yêu cầu nên 3 chunk vẫn thất bại → rơi vào fallback (dùng text gốc).
- **Cách sửa:** Giảm số luồng enrichment xuống 2 (`LLM_MAX_WORKERS`), RAGAS dùng `RunConfig(max_workers=4, max_retries=10, max_wait=90)`. Pipeline vẫn chạy hết nhờ fallback; cải tiến tiếp theo là retry đọc đúng `retryDelay` từ response 429.

### Kiến thức còn thiếu & cách bổ sung
- **Cách RAGAS tính từng metric:** chỉ hiểu khi đọc kết quả per-question — context_precision là *average precision theo thứ hạng* (context đúng mà nằm rank 2 vẫn bị trừ), answer_relevancy = 0 với câu trả lời kiểu "không có thông tin". → Đọc source `ragas/metrics/_answer_relevance.py` và docs RAGAS.
- **Judge LLM cũng có sai số:** có câu context_recall = 1.0 nhưng context_precision = 0.0 — mâu thuẫn, nhiều khả năng do judge `flash-lite` yếu. → Khi so sánh pipeline cần cố định judge và nên kiểm chứng bằng judge mạnh hơn.
- **Ràng buộc môi trường thực tế** (phiên bản Python, chính sách bảo mật Windows, quota API) tốn thời gian hơn code RAG. → Đọc kỹ `.python-version`/requirements trước, cài model và kiểm tra quota trước khi lab.

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

### Project: PC Builder Assistant — trợ lý tư vấn build PC

#### 1. Hiện trạng
- **Pipeline hiện tại:** Chatbot nhận nhu cầu (ngân sách, mục đích: gaming / đồ hoạ / văn phòng) → tìm linh kiện trong dữ liệu sản phẩm (CPU, mainboard, RAM, GPU, PSU, case) → LLM gợi ý cấu hình. Retrieval chủ yếu dense search đơn giản trên mô tả sản phẩm.
- **Vấn đề / Bottlenecks đang gặp:**
  - Tìm theo **mã model chính xác** ("RTX 4060", "B760M", "DDR5-6000") kém — dense embedding coi các mã gần giống nhau là tương tự.
  - LLM đôi khi gợi ý linh kiện **không tương thích** (sai socket CPU–mainboard, RAM DDR4 vs DDR5, PSU thiếu công suất) hoặc bịa giá.
  - Dữ liệu giá / tồn kho thay đổi liên tục → dễ trả lời theo **thông tin cũ** (giống lỗi conflict phiên bản v2023/v2024 trong lab).

#### 2. Kế hoạch cải tiến
1. **Chunking strategy:** **Structure-aware** — mỗi linh kiện là 1 chunk theo bảng thông số (tên, socket, chuẩn RAM, TDP, giá), không cắt giữa bảng. Bài viết hướng dẫn build dài dùng **hierarchical** (child để tìm, parent để trả ngữ cảnh).
2. **Search retrieval:** **Hybrid BM25 + Dense + RRF** — BM25 bắt mã model chính xác, dense bắt nhu cầu mô tả tự nhiên ("PC chơi game 20 triệu"). Thêm **metadata filter** cứng: `category`, `socket`, `ram_type`, khoảng giá, `in_stock`, `updated_at`.
3. **Reranking:** Có — `bge-reranker-v2-m3`, nhưng chỉ rerank top-10 để giữ latency < 2–3 s/query; cân nhắc Flashrank nếu chạy CPU.
4. **Evaluation:** RAGAS 4 metrics trên test set ~30 câu (tra cứu thông số, so sánh 2 linh kiện, build theo ngân sách) + **metric tự viết: tỉ lệ cấu hình tương thích** (check socket/RAM/PSU bằng rule) và sai lệch giá so với dữ liệu.
5. **Enrichment:** **Auto metadata extraction** (trích socket, chuẩn RAM, TDP từ mô tả thô) và **HyQA** (sinh câu hỏi kiểu "CPU này đi với main nào?") để khớp cách người dùng hỏi.

#### 3. Timeline triển khai
- **Tuần 1:** Chuẩn hoá dữ liệu linh kiện + structure-aware chunking + metadata (socket, RAM, giá, ngày cập nhật).
- **Tuần 2:** Hybrid search + metadata filter; viết bộ test ~30 câu và chạy RAGAS baseline.
- **Tuần 3:** Thêm reranker + rule kiểm tra tương thích sau khi LLM gợi ý cấu hình; đo latency.
- **Tuần 4:** Enrichment (auto metadata, HyQA), chạy lại RAGAS so sánh với baseline, phân tích bottom-5 failure.
