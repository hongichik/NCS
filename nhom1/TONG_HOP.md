# Tổng hợp kiến thức — nhom1 (GNN session nền tảng)

Tài liệu ghi lại các nội dung đã tìm hiểu về **SR-GNN** và **GCE-GNN** trong repo `NCS_serve`.

---

## 1. Vai trò nhom1 trong repo


| Dự án       | Paper      | Vai trò                                                        |
| ----------- | ---------- | -------------------------------------------------------------- |
| **SR-GNN**  | AAAI 2019  | Baseline GNN session — chỉ **local graph**                     |
| **GCE-GNN** | SIGIR 2020 | Mở rộng SR-GNN — thêm **global graph** (ngữ cảnh toàn dataset) |


**Quan hệ:** GCE-GNN kế thừa pipeline preprocess của SR-GNN, cùng format `train.txt` / `test.txt`, thêm `all_train_seq.txt` và cache global graph.

**Data / log (quy ước NCS):**

- Data: `Data/<tên bài>/` (ví dụ `Data/SR-GNN/`, `Data/GCE-GNN/`)
- Test: `Data/Test/SR-GNN/`, `Data/Test/GCE-GNN/`
- Log quá trình: `Log/<tên bài>/`
- Log kết quả: `LogMins/<tên bài>/`

---

## 2. Bài toán chung: Session-based recommendation

- **Input:** Một session = chuỗi item user đã tương tác, ví dụ `[A, B, C, D]`
- **Output:** Dự đoán item tiếp theo
- **Metric:** Recall@K, MRR@K

---

## 3. SR-GNN vs GCE-GNN


| Tiêu chí     | SR-GNN                        | GCE-GNN                                         |
| ------------ | ----------------------------- | ----------------------------------------------- |
| Graph        | Chỉ **local** (trong session) | **Local + global**                              |
| Nguồn global | Không                         | Thống kê từ **mọi session train**               |
| File thêm    | —                             | `all_train_seq.txt`, `adj_12.pkl`, `num_12.pkl` |
| Độ phức tạp  | Nhẹ hơn                       | Nặng hơn (build + lưu global graph)             |
| Kết hợp      | —                             | `output = h_local + h_global`                   |


### Minh họa

```
Session hiện tại:  A → B → C → D

LOCAL (cả SR-GNN & GCE-GNN):
  A—B—C—D   (cạnh trong session này)

GLOBAL (chỉ GCE-GNN, tra từ graph train sẵn):
  A ↔ X, Y, Z   (top neighbor của A trên toàn data)
  B ↔ P, Q
  ...
```

---

## 4. Local vs Global — bảng so sánh


| Tiêu chí           | **Local**                   | **Global**                                          |
| ------------------ | --------------------------- | --------------------------------------------------- |
| Phạm vi            | **1 session** đang predict  | **Nhiều session train** gộp lại                     |
| Câu hỏi            | User **đang** làm gì?       | Trên hệ thống, item này **thường** đi với item nào? |
| Build              | Mỗi lần từ session hiện tại | **1 lần** từ `all_train_seq.txt`                    |
| Thay đổi theo user | Có                          | Không (graph cố định)                               |
| Rủi ro             | Session ngắn → ít context   | Item hot, co-click ngẫu nhiên → **nhiễu**           |
| Trong code GCE-GNN | `LocalAggregator`           | `GlobalAggregator`                                  |


**Trả lời ngắn:** Global **không** tính trong 1 session — tính trên **nhiều session khác nhau** trong tập train, rồi dùng lại khi predict.

---

