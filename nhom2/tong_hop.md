# Tổng hợp kiến thức — nhom2 (GNN session + SSL)

Tài liệu ghi lại nội dung đã tìm hiểu về **DHCN**, **COTREC**, **CSGNN** trong repo `NCS_serve`.

| Mục | Nội dung |
|-----|----------|
| [Bài toán chung](#bài-toán-chung-nhom2) | Session → item tiếp theo |
| [DHCN](#dhcn--dual-hypergraph--line-graph--ssl) | HG + LG + SSL |
| [COTREC](#cotrec--hiểu-nhánh-i-nhánh-s-và-quy-tắc-đồng-thuận) | I + S + co-training |
| [CSGNN](#csgnn--category-aware--self-supervised-hai-nhánh-hg--lg) | DHCN + category |
| [Tóm tắt & so sánh](#tóm-tắt-nhom2) | Bảng chung, train chậm |

**Data / log (quy ước NCS):** `Data/<tên bài>/`, `Data/Test/<tên bài>/`, `Log/`, `LogMins/`. Code: `nhom2/DHCN/`, `nhom2/COTREC/`, `nhom2/CSGNN/`.

---

## Bài toán chung (nhom2)

- **Input:** một phiên — chuỗi item user đã xem (CSGNN thêm **category** mỗi click)
- **Output:** dự đoán **một** item tiếp theo
- **Cách chấm:** xếp hạng toàn catalog → Recall@K, MRR@K

**Mẫu dùng xuyên suốt:** phiên `Điện thoại → Ốp lưng → Sạc dự phòng` → đáp án `Tai nghe`.

**Khung chung nhom2:** **hai cách hiểu** cùng phiên khi **học**, **một cách** khi **gợi ý** (không trộn 50–50 hai điểm số).

---

# DHCN — Dual Hypergraph + Line graph + SSL

Paper: *Dual Channel Hypergraph Convolutional Network*. Repo: `nhom2/DHCN/`.

**Một câu:** Mỗi session train = **một túi item** (hypergraph) → nhánh **HG** dự đoán; nhánh **LG** (session giống nhau trong batch) + **SSL** chỉ giúp học — **predict chỉ HG**.

> **Quan hệ:** CSGNN = DHCN + category. COTREC **khác kiến trúc** (item graph I + session S, không hypergraph).

---

## 1. Hypergraph (HG) — nhánh chính

### Hyperedge là gì?

Mỗi **session train** tạo một **túi** gồm các **item khác nhau** trong phiên (bỏ trùng).

| Session train | Hyperedge (túi) |
|---------------|------------------|
| U1: Điện thoại → Ốp → Sạc | {Điện thoại, Ốp, Sạc} |
| U2: Ốp → Sạc → Tai nghe | {Ốp, Sạc, Tai nghe} |

**Khác SR-GNN:** SR-GNN chỉ nối **cạnh liền kề** trong một phiên.  
**Khác COTREC (I):** I học **cặp hay đi sau nhau** (A→B). HG học **cùng nằm trong một giỏ** — Sạc và Tai nghe có thể gần nhau dù không liền kề trong mọi chuỗi.

### Ba bước HG

1. **Build hypergraph** từ toàn train → ma trận liên quan item×item (`DHBH_T` trong `util.py`).
2. **HyperConv** lan truyền → embedding item “biết hàng xóm theo túi”.
3. **Đọc phiên hiện tại:** position + attention (GLU) → `sess_emb_hgnn` → chấm **mọi item** → item kế tiếp.

> **Một câu HG:** *“Trên cả shop, item nào hay cùng giỏ — và phiên này đang nhấn item nào?”*

---

## 2. Line graph (LG) — nhánh phụ

**Giống nhánh S (COTREC) / LG (CSGNN):** trong **một batch**, session trùng nhiều item → Jaccard cao → lan truyền giữa session.

| User | Session | Ghi chú |
|------|---------|---------|
| Bạn | Điện thoại → Ốp → Sạc | Cần đoán |
| User 2 | Ốp → Sạc → Tai nghe | Rất giống |
| User 4 | Quần → Áo | Không giống |

→ `session_emb_lg` hút tín hiệu từ session tương tự trong lô. **Không dùng khi gợi ý.**

---

## 3. SSL — đồng thuận HG và LG

| | Ý nghĩa |
|---|---------|
| **Positive** | `sess_emb_hgnn` và `session_emb_lg` cùng session → **gần** |
| **Negative** | Xáo trộn HG, ghép với LG → **xa** |
| **Trọng số** | `beta` |

**Loss:** CrossEntropy trên **scores từ HG** + `beta × SSL`.

**Không** nhân đôi file train.

---

## 4. Luồng nhớ DHCN

```
Session (chỉ item)
    ├─ HG: hypergraph toàn train → attention → sess_emb_hgnn → PREDICT
    └─ LG: session giống nhau trong batch → session_emb_lg
              LÚC HỌC: SSL
```

---

## 5. DHCN vs COTREC (tóm tắt)

| | **DHCN (HG)** | **COTREC (I)** |
|---|---------------|----------------|
| Graph toàn train | **Túi** item trong session | **Cặp / transition** item |
| Nhánh batch | LG | S |
| SSL | 2 vector (đơn giản) | Chéo ranking + KL + adversarial |
| Predict | HG | I |

**Cái nào tốt hơn:** phụ thuộc dataset — so Recall/MRR cùng split trên data của bạn.

---

## 6. Vì sao train DHCN lâu?

| Nguyên nhân | Ghi chú |
|-------------|---------|
| Rất nhiều batch/epoch | Retailrocket ~10k batch/epoch, `batchSize=100` |
| Full-ranking mỗi batch | `scores = session × n_node` (~37k item) |
| HyperConv mỗi forward | 3 layer sparse trên graph lớn |
| HG + LG + SSL mỗi batch | Nặng hơn một nhánh |
| Mỗi epoch = train + test hết | Gấp đôi thời gian/epoch |
| Build `n_node×n_node` lúc load | CPU, có thể chậm/OOM |
| Chạy CPU (Mac) | Chậm hàng chục lần so GPU |

**Debug nhanh:** `--epoch 1`, dataset `sample`, GPU, tăng `batchSize` nếu đủ RAM.

---

# COTREC — Hiểu nhánh I, nhánh S và quy tắc đồng thuận

Cùng **bài toán chung** ở trên. Khác DHCN: **không hypergraph** — dùng **item graph global (I)** + **session graph trong batch (S)**.

---

## 1. Tình huống mẫu (COTREC)


|                     |                                     |
| ------------------- | ----------------------------------- |
| **Phiên cần đoán**  | Điện thoại → Ốp lưng → Sạc dự phòng |
| **Đáp án mong đợi** | Tai nghe                            |


---

## 2. Ba cách nhìn: GNN (SR-GNN) · I · S

### 3.1 GNN session thuần (SR-GNN) — graph **trong phiên**

```
Điện thoại — Ốp lưng — Sạc dự phòng
     (chỉ cạnh liền kề trong session này)
```


| Ý                | Giải thích                                                            |
| ---------------- | --------------------------------------------------------------------- |
| **Nhìn gì**      | Đồ thị nhỏ: các item **trong giỏ**, nối theo thứ tự click             |
| **Câu hỏi ngầm** | *“Chuỗi liền nhau này nói gì?”*                                       |
| **Tín hiệu**     | Ốp–Sạc liền nhau → nhóm phụ kiện điện thoại                           |
| **Gợi ý**        | Tai nghe / cáp / giá đỡ (theo pattern **cấu trúc chuỗi** đã học)      |
| **Giới hạn**     | Không mở trực tiếp “trên cả site, Sạc hay đi với gì” trong bước graph |


---

### 3.2 Nhánh **I** (Item view) — catalog **toàn site** + thứ tự phiên

**Bước A — Bản đồ item (từ mọi session train)**  
Trên toàn website đã quan sát: sau `Điện thoại` hay có `Ốp`, `Tai nghe`; `Sạc` hay đi với `Cáp`, `Tai nghe`; `Ốp` và `Sạc` hay xuất hiện cùng nhau.

**Bước B — Đọc phiên hiện tại**  
Chú ý mạnh item vừa click (`Sạc`), thứ tự `Điện thoại → Ốp → Sạc` → một **ý nghĩa phiên**: *ngữ cảnh phụ kiện điện thoại, nhấn vào sạc*.

**Bước C — Dự đoán**  
So với **toàn catalog** → `Tai nghe` cao vì khớp **lịch sử site** + **phiên hiện tại**.

> **Một câu I:** *“Trên cả shop, sau kiểu Điện thoại–Ốp–Sạc người ta thường mua gì — và trong phiên này đang nhấn Sạc.”*


| So với GNN           | I khác ở đâu                                              |
| -------------------- | --------------------------------------------------------- |
| Phạm vi graph        | **Item–item toàn train**, không chỉ 3 nút trong một phiên |
| Vai trò trong COTREC | **Nhánh chính** — kết quả gợi ý cuối cùng lấy từ I        |


**So SR-GNN / GCE (nhom1):** I gần ý **global item** (như GCE) + **đọc chuỗi có chú ý** (như SR-GNN), nhưng **không** gộp `local + global` thành một vector như GCE; và trong COTREC còn có nhánh S song song.

---

### 3.3 Nhánh **S** (Session view) — phiên **giống** phiên khác **trong cùng lô học**

Trong **một batch** train đang chạy:


| User    | Session (đã xem)          | Ghi chú                          |
| ------- | ------------------------- | -------------------------------- |
| **Bạn** | Điện thoại → Ốp → Sạc     | Cần đoán tiếp                    |
| User 2  | Ốp → Sạc → **Tai nghe**   | Rất giống bạn (chung Ốp, Sạc)    |
| User 3  | Điện thoại → Ốp → **Cáp** | Khá giống (chung Điện thoại, Ốp) |
| User 4  | Quần → Áo                 | Không giống                      |


**Độ giống (Jaccard):** `|item chung| / |item hợp|` — session càng trùng item thì càng “nối” mạnh trong graph session của batch.

**Logic S:** vector phiên của bạn **hút thêm** từ User 2, 3; User 2 vừa chọn `Tai nghe` → tín hiệu mạnh; User 4 **không** lan sang bạn.

> **Một câu S:** *“Trong lô học này, có người xem gần giống bạn và họ chọn Tai nghe — phiên bạn được kéo về phía hành vi đó.”*


| So với I         | S khác ở đâu                                                            |
| ---------------- | ----------------------------------------------------------------------- |
| Nguồn tri thức   | **Batch hiện tại** (session tương tự), không phải bản đồ item toàn site |
| Khi predict thật | **Không dùng** — S chỉ có ích lúc **học**                               |


---

### 3.4 Bảng so nhanh


|                | **GNN (SR-GNN)**        | **I**                                | **S**                                 |
| -------------- | ----------------------- | ------------------------------------ | ------------------------------------- |
| Nhìn gì        | Graph **trong phiên**   | Item–item **cả site** + thứ tự       | Session **giống** session trong batch |
| Câu hỏi ngầm   | Chuỗi liền nhau nói gì? | Toàn site sau pattern này hay là gì? | User giống mình trong lô vừa làm gì?  |
| Ví dụ tín hiệu | Ốp–Sạc liền nhau        | Train: Sạc → Tai nghe nhiều          | User 2: Ốp–Sạc → Tai nghe             |
| Lúc gợi ý thật | (model độc lập)         | **Có — chính**                       | **Không**                             |


---

## 3. I và S: bổ trợ hay đối đầu?

**Bổ trợ (complementary), không phải bỏ phiếu 50–50.**

Hai nhánh trả lời **cùng một câu hỏi** (“item tiếp theo là gì?”) nhưng bằng **hai nguồn bằng chứng**:


| Nhánh | Kiểu bằng chứng                                                 | Ví dụ trong tình huống mẫu           |
| ----- | --------------------------------------------------------------- | ------------------------------------ |
| **I** | **Thống kê dài hạn** — catalog, co-occurrence toàn train        | “Sạc trên site hay dẫn tới Tai nghe” |
| **S** | **Tín hiệu ngắn hạn trong lô** — ai giống mình *ngay lúc train* | “User 2 vừa Ốp–Sạc rồi mua Tai nghe” |


- **I** ổn định, giống “luật chung của shop”.
- **S** nhạy theo **ngữ cảnh batch** (đổi khi shuffle) — bổ sung khi có nhiều session tương tự trong lô.

**Không phải:** trung bình hai danh sách gợi ý khi deploy.  
**Là:** S **dạy thêm** cho I lúc train; lúc dùng chỉ **I** phát biểu.

---

## 4. Quy tắc đồng thuận (co-training)

Nguyên lý: **hai chuyên gia cùng xem một phiên, phải đồng ý; khi gợi ý thật chỉ nghe chuyên gia I.**

### 5.1 Ba lớp ràng buộc (khi học)


| Lớp                                     | Ý nghĩa                                                                                                                      | Hình dung                                                       |
| --------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| **① Học đúng đáp án**                   | Chỉ **I** bị chấm trực tiếp với label (vd. `Tai nghe`)                                                                       | Thi chính thức: I phải chọn đúng                                |
| **② Đồng thuận embedding (SSL)**        | I và S phải **cùng coi** item liên quan là gần, không liên quan là xa — **chéo view** (I tin gợi ý của S, S tin gợi ý của I) | Hai người không mô tả cùng phiên theo hai “ngôn ngữ” trái ngược |
| **③ Đồng thuận xếp hạng (consistency)** | Top item từ I và S **không lệch quá xa**                                                                                     | Không chỉ vector gần — danh sách gợi ý cũng phải giống nhau     |


**Tại sao cần đồng thuận?**

- Chỉ I → dễ học một chiều (catalog + thứ tự).
- Chỉ S → phụ thuộc batch, không ổn khi test từng phiên.
- Ép đồng thuận → I hấp thụ tín hiệu từ S mà **không đổi** pipeline gợi ý cuối.

### 5.2 Ai bổ trợ cho ai?

```
        ┌──────────────────────────────────────┐
        │  Cùng phiên: Điện thoại→Ốp→Sạc       │
        └─────────────────┬────────────────────┘
                          │
           ┌──────────────┴──────────────┐
           ▼                             ▼
      Nhánh I                         Nhánh S
   (luật shop + thứ tự)          (user giống trong lô)
           │                             │
           │    LÚC HỌC: phải gần nhau    │
           └──────────────┬──────────────┘
                          ▼
              I học representation tốt hơn
                          │
           LÚC DÙNG: chỉ I → gợi ý Tai nghe
```


| Vai trò                            | I                                | S                                 |
| ---------------------------------- | -------------------------------- | --------------------------------- |
| Khi **học**                        | Học sinh chính — phải đúng label | Gia sư phụ — kéo I không lệch quá |
| Khi **gợi ý**                      | Phát ngôn                        | Im lặng                           |
| Batch không ai giống (vd. Quần–Áo) | Vẫn dựa catalog                  | S yếu; I vẫn đủ                   |


### 5.3 Một câu tổng kết

**I và S bổ trợ:** I mang **tri thức site**, S mang **tín hiệu người giống bạn trong lô**; **đồng thuận** = bắt hai cách hiểu **không mâu thuẫn** khi học, để khi chỉ dùng **I** thì gợi ý vừa đúng catalog vừa không bỏ qua hành vi tương tự gần đây.

---

# CSGNN — Category-aware + Self-supervised (hai nhánh HG · LG)

Paper: *Category-aware Self-supervised Graph Neural Network for Session-based Recommendation*.

## 1. Bài toán

Giống SR-GNN / COTREC / DHCN:

- **Input:** session (chuỗi item) + **category** từng click (`category_train.txt` / `category_test.txt`)
- **Output:** item tiếp theo
- **Metric:** Recall@K, MRR@K

**Một câu:** CSGNN = **hypergraph (item + category)** làm nhánh chính + **line graph (session trong batch)** làm nhánh phụ + **SSL** ép hai nhánh đồng thuận lúc học.

---

## 2. Tình huống mẫu


|              |                                     |
| ------------ | ----------------------------------- |
| **Session**  | Điện thoại → Ốp lưng → Sạc dự phòng |
| **Category** | Điện tử → Phụ kiện → Phụ kiện       |
| **Đáp án**   | Tai nghe                            |


---

## 3. Hai nhánh

### 3.1 Nhánh **HG** (Hypergraph) — nhánh chính

**Hypergraph (toàn train):** mỗi session train = một siêu cạnh chứa **item khác nhau + category khác nhau** trong phiên đó.

Ví dụ hyperedge: `{Điện thoại, Ốp, Sạc, Điện tử, Phụ kiện}`.


| Ý               | Giải thích                                                                                                         |
| --------------- | ------------------------------------------------------------------------------------------------------------------ |
| **Nguyên lý**   | Item và category **cùng xuất hiện trong một phiên** → embedding được kéo gần (ngữ cảnh cả sản phẩm lẫn ngành hàng) |
| **Đọc session** | Self-attention trên chuỗi **item** (ISA) + trên chuỗi **category** (CSA) + vị trí → `sess_emb_hgnn`                |
| **Một câu HG**  | *“Trên cả shop, item/category hay đi cùng nhau thế nào — phiên này đang nhấn gì?”*                                 |
| **Lúc gợi ý**   | **Chỉ nhánh này** — `sess_emb_hgnn` × embedding item (sau hypergraph conv)                                         |


### 3.2 Nhánh **LG** (Line graph) — nhánh phụ

**Giống ý nhánh S của COTREC:** graph **session–session trong batch** (Jaccard: trùng item → gần).


| Ý             | Giải thích                                                                     |
| ------------- | ------------------------------------------------------------------------------ |
| **Khác HG**   | Không dùng hypergraph toàn train; dùng **ai giống mình trong lô học hiện tại** |
| **Category**  | Line graph cũng gộp chỉ số **item + category** khi pool session                |
| **Vector**    | `session_emb_lg`                                                               |
| **Lúc gợi ý** | **Không dùng** — chỉ hỗ trợ lúc train                                          |


> **Một câu LG:** *“Trong lô học này, user xem gần giống bạn đang dẫn tới hành vi nào?”*

---

## 4. SSL — đồng thuận hai nhánh


|              | Ý nghĩa                                                               |
| ------------ | --------------------------------------------------------------------- |
| **Positive** | `sess_emb_hgnn` và `session_emb_lg` **cùng một session** phải **gần** |
| **Negative** | Ghép LG với embedding HG **bị xáo trộn** (session khác) → phải **xa** |
| **Trọng số** | `beta` (0 = tắt SSL)                                                  |


**Không** nhân đôi dữ liệu trong file — chỉ ràng buộc **representation** khi train.

**Khác COTREC:** COTREC SSL chéo ranking + KL + adversarial; CSGNN SSL **đơn giản hơn** — so **hai vector phiên** HG vs LG.

**Loss train:** CrossEntropy (đúng item tiếp) + `beta × SSL`.

---

## 5. Luồng nhớ CSGNN

```
Session + category
    ├─ HG → sess_emb_hgnn → PREDICT
    └─ LG → session_emb_lg (+ SSL lúc học)
```

---

## 6. CSGNN vs DHCN

| | **DHCN** | **CSGNN** |
|---|----------|-----------|
| Hyperedge | Chỉ item | Item + **category** |
| Đọc session | Attention item | + attention **category** (ISA, CSA) |
| LG / SSL / predict | Giống | Giống |

**CSGNN ≈ DHCN + category.**

---

# Tóm tắt nhom2

## Bảng một dòng

| Bài | Nhánh chính (predict) | Nhánh phụ (train) | Đặc điểm |
|-----|------------------------|-------------------|----------|
| **DHCN** | HG (hypergraph item) | LG (session batch) | SSL 2 vector |
| **COTREC** | I (item global) | S (session batch) | Co-training + KL + adv |
| **CSGNN** | HG (item+category) | LG | Mở rộng DHCN |

## So nhom1

| | nhom1 (SR-GNN, GCE) | nhom2 |
|---|---------------------|-------|
| Graph | Local; GCE gộp local+global **một vector** | Hypergraph hoặc item-global + nhánh batch |
| SSL | Không | Có |
| Deploy | Một pipeline | **Một nhánh** (HG hoặc I) |

## Luồng nhớ 5 ý

1. Session → item tiếp theo.  
2. Học: hai view; dùng: một view chính.  
3. **DHCN:** túi (HG) + lô giống nhau (LG).  
4. **COTREC:** catalog cặp (I) + lô (S).  
5. **CSGNN:** DHCN + category.