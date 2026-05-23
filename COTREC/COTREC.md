# Reproduction Log
## Baseline: COTREC
- **Paper:** <citation đầy đủ với venue, năm, link>
- **GitHub repo:** https://github.com/xiaxin1998/COTREC
- **Ngày verify:** <YYYY-MM-DD>
- **NCS thực hiện:** Phạm Nguyên Hồng (ĐH Hạ Long)
### 1. Dataset neo (Bước 1)
- Dataset: retailrocket
- Split: 7/2/1
- Metric: <full-ranking, không sampled negatives>
- Hyperparameter: epoch=30, batchSize=100, embSize=100, l2=1e-05, lr=0.001, layer=2, beta=0.005, lam=0.005, eps=0.2
### 2. Môi trường (Bước 2)
- Python: 3.13.13
- PyTorch: 2.5.1+cu121
- CUDA: 12.1
- CPU: Intel(R) Core(TM) i5-14600KF 
### 3. Verify trên dataset neo (Bước 3-4)
| Metric | Paper báo | NCS đo | Sai lệch | Trạng thái |
|---------|-----------|--------|----------|------------|
| HR@20 | 54.12% | 57.4276 | 6.11% | PASS |
| MRR@20 | 32.84% | 29.6605 | 9.68% | PASS |
**Kết luận Bước 4:** PASS reproduce. Chuyển sang Bước 6.
### 4. Diagnostic nếu cần (Bước 5)
### 5. Extend sang dataset bài hiện tại (Bước 6)
### 6. Vấn đề và cách giải quyết
- không điều chỉnh
### 7. Supervisor review
- **Reviewed by:** <tên supervisor>
- **Ngày:** <YYYY-MM-DD>
- **Nhận xét:** <pass / cần điều chỉnh / câu hỏi>