# Reproduction Log
## Baseline: <tên_baseline>
- **Paper:** <citation đầy đủ với venue, năm, link>
- **GitHub repo:** <link đến repo chính thức hoặc thay thế> - **Ngày verify:** <YYYY-MM-DD>
- **NCS thực hiện:** <tên>
### 1. Dataset neo (Bước 1)
- Dataset: <tên + phiên bản, ví dụ “Yoochoose 1/64”>
- Split: <time-based 80/10/10>
- Metric: <full-ranking, không sampled negatives>
- Hyperparameter: <embedding_dim=100, lr=0.001, bs=100, epochs=30>
### 2. Môi trường (Bước 2)
- Python: 3.10.12
- PyTorch: 2.1.0
- CUDA: 11.8
- GPU: NVIDIA A100 (40GB) - Commit hash: <abc123def>
### 3. Verify trên dataset neo (Bước 3-4)
| Metric | Paper báo | NCS đo | Sai lệch | Trạng thái | 
|---------|-----------|--------|----------|------------|
| HR@20 | 70.57 | 67.21 | 4.76% | PASS |
| MRR@20 | 30.94 | 29.82 | 3.62% | PASS |
**Kết luận Bước 4:** PASS reproduce. Chuyển sang Bước 6.
### 4. Diagnostic nếu cần (Bước 5)
<Để trống nếu PASS ở Bước 4>
<Nếu FAIL: ghi nguyên nhân đã thử, nguyên nhân thực sự, kết quả sau fix>
### 5. Extend sang dataset bài hiện tại (Bước 6)
| Dataset | HR@20 | NDCG@20 | HR@20_cold | Note | 
|--------------|--------|---------|------------|--------------|
| RetailRocket | 32.45 | 15.20 | 0.85 | Sanity OK |
 
   
| Diginetica | 30.10 | 14.85 | 0.78 | Sanity OK | |Yoochoose |67.21 |29.82 |0.92 |=Bước3 |
### 6. Vấn đề và cách giải quyết
- Lỗi: Chạy quá chậm, mỗi lần chạy cần quá nhiều thời gian, vượt quá giới hạn 24h của colap, bài toán cần sử dụng CPU để tính toán ban đầu quá nhiều dẫn đến tốc độ luôn chậm
- Giải quyết: chạy trên máy chủ hiện tại không bị giới hạn thời gian, chuyển đổi việc tính toán qua GPU không làm CPU quá tải nữa tuy nhiên vẫn cần chiếm 1 core CPU để tính toán (không chiếm thêm để chừa tài nguyên thử nghiệm các bài toán khác)
- <ví dụ: “Code gốc dùng PyTorch 1.x, em port sang 2.x;
verify pass nhưng phải điều chỉnh deprecated APIs”> ### 7. Supervisor review
- **Reviewed by:** <tên supervisor>
- **Ngày:** <YYYY-MM-DD>
- **Nhận xét:** <pass / cần điều chỉnh / câu hỏi>