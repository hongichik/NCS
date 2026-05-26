# Tổng hợp kiến thức — nhom3 (Sequential + contrastive / framework)

Tài liệu ghi lại nội dung về **DuoRec** và **SCL** trong repo `NCS_serve`.

| Mục | Nội dung |
|-----|----------|
| [Bài toán & vị trí nhom3](#bài-toán--vị-trí-nhom3-trong-repo) | Khác nhom1/2 thế nào |
| [CL / SCL là gì](#cl-và-scl-trong-ngữ-cảnh-này) | Contrastive learning |
| [DuoRec](#duorec--transformer--contrastive) | Transformer + CL (RecBole) |
| [SCL](#scl--self-contrastive-learning) | Gắn CL gọn lên GCE / COTREC / DHCN |
| [Tóm tắt & so sánh](#tóm-tắt-nhom3) | Bảng 3 nhóm, chạy thử |

**Data / log (quy ước NCS):** `Data/DuoRec/`, `Data/Test/...`, `Log/DuoRec/`, `LogMins/DuoRec/`.  
**Code:** `nhom3/DuoRec/`, `nhom3/SCL/` (paper: SelfContrastiveLearningRecSys).

---

## Bài toán & vị trí nhom3 trong repo

### Bài toán (giống nhom1, nhom2)

- **Input:** chuỗi tương tác (session / sequence) — các item user đã xem
- **Output:** **một** item tiếp theo
- **Metric:** Recall@K, MRR@K hoặc NDCG@K (DuoRec/RecBole thường báo NDCG)

**Mẫu:** `Điện thoại → Ốp lưng → Sạc dự phòng` → đoán `Tai nghe`.

### Ba nhóm trong NCS

| Nhóm | Hướng | Ví dụ |
|------|--------|-------|
| **nhom1** | GNN session nền | SR-GNN, GCE-GNN (`train.txt`) |
| **nhom2** | GNN + SSL hai view (graph) | DHCN, COTREC, CSGNN |
| **nhom3** | **Sequential / framework** | DuoRec (Transformer+CL), SCL (tích hợp CL lên baseline) |

**nhom3 không thay nhom2** — bổ sung **kiến trúc chuỗi (Transformer)** và **gói thí nghiệm SCL**.

---

## CL và SCL trong ngữ cảnh này

### CL (Contrastive Learning) — cột “CL” trên bảng baseline

**Không đổi bài toán.** CL giải quyết thêm khi **train**:

| Vấn đề | CL làm gì |
|--------|-----------|
| Embedding session **phẳng** (degeneration) | Ép hai “view” cùng hành vi **gần**, khác batch **xa** |
| Dữ liệu **thưa** | Thêm tín hiệu tự giám sát (InfoNCE, …) |
| Chỉ học đúng label | Representation **ổn định** hơn trước khi ranking |

```
Loss = Predict đúng item tiếp theo  (luôn có)
     + λ × Contrastive(view A, view B)  (lúc train)
```

**Lúc gợi ý:** thường **một** forward / một nhánh — không trộn hai bảng điểm 50–50.

### SCL (Self Contrastive Learning)

Paper ECIR 2024 — **không phải model graph mới**, mà **cách thay/simplify CL** trên **GCE-GNN, COTREC, DHCN** (bản copy trong `nhom3/SCL/`).

- Bỏ/giảm positive-negative phức tạp, augment nặng
- Thúc đẩy **không gian embedding item** đồng đều hơn (uniformity)
- Code chạy hàng ngày cho 3 baseline đó vẫn nên ưu tiên **nhom1/** và **nhom2/**; **SCL** dùng để **reproduce paper SCL**

---

# DuoRec — Transformer + contrastive

Paper: WSDM 2022 — *Contrastive Learning for Representation Degeneration Problem in Sequential Recommendation*.  
Repo: `nhom3/DuoRec/` — framework **RecBole**.

**Một câu:** Đọc chuỗi bằng **Transformer** (kiểu SASRec), predict item tiếp theo, thêm **contrastive** để vector cuối chuỗi không bị “phẳng”.

> **Không có** graph session / hypergraph / nhánh I–S như COTREC hay HG–LG như DHCN. Trên bảng related work gần **CL4SRec** (Sequential + CL), không phải Dual-view GNN.

---

## 1. Nhánh chính — đọc chuỗi (SASRec-style)

| Bước | Việc làm |
|------|----------|
| 1 | Embedding **item** + embedding **vị trí** trong chuỗi |
| 2 | **Transformer** attention **một chiều** (chỉ nhìn quá khứ) |
| 3 | Lấy vector tại **ô cuối** (sau click gần nhất) → `seq_output` |
| 4 | `seq_output` × mọi item → **CrossEntropy** với item kế tiếp |

> *“Thứ tự click quan trọng; vector cuối chuỗi = ý nghĩa phiên đến hiện tại.”*

---

## 2. Contrastive — hai view (lúc train)

Config repo thường dùng `contrast: us_x` (xem `configs/duorec_retailrocket_tuned.yaml`).

### View A — cùng chuỗi, forward 2 lần

- Cùng `item_seq`, `forward` hai lần (dropout khác → `seq_output` vs `aug_seq_output`).
- Cùng một session thật → hai vector nên **gần**.

### View B — semantic augmentation (`sem_aug`)

- Loader tạo chuỗi **biến thể**: thay một số item bằng item **gần nghĩa** (hay đi cùng trên data).
- `sem_aug_seq_output` = forward trên chuỗi đó.

### Chế độ `us_x` (trong code)

- Contrastive giữa **`aug_seq_output`** và **`sem_aug_seq_output`** (InfoNCE trong batch).
- Cộng loss: `lmd_sem ×` contrastive (+ loss CE chính).

```
Chuỗi gốc ──► Transformer ──► predict item t+1     (loss chính)
     │
     ├─► forward lần 2 (dropout) ──► aug ──┐
     └─► sem_aug (chuỗi gần nghĩa) ──► sem ──┴─► InfoNCE (us_x)
```

**Test:** chỉ chuỗi gốc, **một** forward → ranking.

---

## 3. Tham số hay gặp

| Param | Ý |
|-------|---|
| `lmd` | Trọng số CL unsupervised (mode `us` / `un`) |
| `lmd_sem` | Trọng số CL semantic (`us_x`, `su`) |
| `tau` | Nhiệt độ InfoNCE |
| `contrast` | `us`, `su`, `us_x`, … |
| `sim` | `dot` hoặc `cos` |

---

## 4. Data & chạy

| | DuoRec |
|---|--------|
| **Format** | RecBole: `*.inter` trong `Data/DuoRec/<dataset>/` |
| **Khác nhom1/2** | Không dùng `train.txt` pickle mặc định |

```bash
cd nhom3/DuoRec
python run_seq.py --dataset=ml-100k --model=DuoRec

# retailrocket (config tune):
python run_seq.py --dataset=retailrocket \
  --config_files "seq.yaml configs/duorec_retailrocket_tuned.yaml"
```

---

## 5. DuoRec vs nhom2 (tóm tắt)

| | DHCN / COTREC | DuoRec |
|---|---------------|--------|
| Cấu trúc | **Graph** (HG/LG hoặc I/S) | **Chuỗi + Transformer** |
| CL | Hai view **graph** | Augment **sequence** + semantic |
| Framework | `main.py` | **RecBole** |
| Predict | Một nhánh graph | Một forward sequence |

---

# SCL — Self Contrastive Learning

Paper: ECIR 2024. Repo: `nhom3/SCL/`.

**Một câu:** Giữ kiến trúc **GCE-GNN / COTREC / DHCN**, thay phần contrastive phức tạp bằng **SCL** — objective gọn, cải thiện embedding item.

---

## 1. Cấu trúc folder SCL

| Thư mục | Baseline gốc (nhom1/2) |
|---------|-------------------------|
| `SCL/GCE-GNN/` | `nhom1/GCE-GNN` |
| `SCL/COTREC/` | `nhom2/COTREC` |
| `SCL/DHCN/` | `nhom2/DHCN` |

Mỗi folder có README riêng; flag `--remove_original_cl_loss` để tắt CL gốc của baseline khi ablation.

---

## 2. Logic SCL (nguyên lý)

| | CL gốc (COTREC, DHCN, …) | SCL |
|---|---------------------------|-----|
| Positive/negative | Thường phức tạp, dual-view, augment | **Đơn giản hóa** |
| Mục tiêu thêm | Đồng thuận view + predict | **Uniformity** không gian item + gắn lên SOTA CL models |
| Bài toán | Session → next item | **Giống** |

**Train:** vẫn loss predict + SCL thay/thêm vào phần contrastive cũ.  
**So sánh metric:** xem `SCL/SCL.md` (reproduce log).

---

## 3. Khi nào dùng SCL vs code nhom1/2?

| Mục đích | Gợi ý |
|----------|--------|
| Học / sửa baseline GNN | **nhom1**, **nhom2** |
| Reproduce paper SCL, ablation `--remove_original_cl_loss` | **nhom3/SCL** |
| So metric SCL vs CL gốc trên cùng data | Chạy cả hai, cùng split |

---

# Tóm tắt nhom3

## Bảng một dòng

| Dự án | Kiến trúc | CL | Predict lúc dùng | Data |
|-------|-----------|-----|------------------|------|
| **DuoRec** | Transformer sequential | Aug + semantic (`us_x`) | Một forward chuỗi gốc | `.inter` RecBole |
| **SCL** | GCE / COTREC / DHCN + SCL loss | Self-CL (thay CL cũ) | Theo từng baseline | `train.txt` từng bài |

## So 3 nhóm repo

| | nhom1 | nhom2 | nhom3 |
|---|-------|-------|-------|
| **Cốt lõi** | GNN local/global | GNN + 2 view SSL | Transformer (DuoRec) hoặc SCL trên GNN |
| **Ví dụ** | SR-GNN, GCE | DHCN, COTREC, CSGNN | DuoRec, SCL |
| **Bảng paper (ảnh baseline)** | Tier 1 GNN | Tier 2 CL+GNN (COTREC) | CL4SRec / DuoRec (Sequential+CL); SCL không phải Cat-GNN |

## CL / TCL / SCL — phân biệt nhanh

| Tên | Ý trong repo / paper |
|-----|---------------------|
| **CL** | Contrastive Learning — kỹ thuật chung (COTREC, DHCN, DuoRec) |
| **SCL** | Self Contrastive Learning — paper + code `nhom3/SCL/` |
| **TCL** | **Không có** trong repo; nếu gặp ngoài paper thường là Temporal(-aware) CL — cùng họ sequential + contrastive |

## Train chậm (DuoRec)

| Nguyên nhân | Ghi chú |
|-------------|---------|
| Nhiều batch, full catalog scoring | Giống áp lực nhom2 (n_node lớn) |
| RecBole train + eval mỗi epoch | `epochs`, early stop |
| CPU vs GPU | Bật CUDA, kiểm tra log device |

**Debug:** `ml-100k` / `sample`, `--epoch 1`, config `duorec_retailrocket_tuned.yaml` khi đã có data.

## Luồng nhớ 5 ý

1. **nhom3** = DuoRec (chuỗi+CL) + SCL (CL gọn trên GCE/COTREC/DHCN).  
2. **Bài toán** vẫn: sequence/session → item tiếp theo.  
3. **DuoRec** không dùng graph như nhom2.  
4. **SCL** không thay folder nhom1/2 cho dev hàng ngày.  
5. **CL** = cột trên bảng baseline; **SCL** = biến thể tích hợp trong nhom3.

---

*Tham chiếu thêm: `nhom1/TONG_HOP.md`, `nhom2/TONG_HOP.md`.*
