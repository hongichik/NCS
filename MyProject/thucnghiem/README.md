# CatSA Module 1

Mã nguồn cho Module 1 của CatSA: `Category-Enhanced Session Graph` cho bài toán Session-Based Recommendation bằng `PyTorch` và `PyTorch Geometric`.

## Cấu trúc thư mục

- `preprocessing/`: Chứa code tiền xử lý RetailRocket và bộ dựng đồ thị `HeteroData` cho từng session.
- `experiments/`: Chứa backbone heterogeneous GNN, logic huấn luyện thử nghiệm và script chạy.

## Thứ tự chạy đúng

Luồng chuẩn là:

1. Tiền xử lý dữ liệu thô RetailRocket.
2. Chia session theo thời gian thành `train/valid/test`.
3. Sinh artifact đã xử lý ra file.
4. Script thực nghiệm đọc đúng split để huấn luyện hoặc đánh giá.

Không nên để script huấn luyện luôn gánh cả bước tiền xử lý trong quy trình chính.

## Chức năng hiện có

- Xây dựng đồ thị không đồng nhất cho mỗi session với 3 loại node: `item`, `leaf_cat`, `parent_cat`.
- Tạo các cạnh:
	- `(item, sequential, item)`
	- `(item, rev_sequential, item)`
	- `(item, belongs_to, leaf_cat)` và `(leaf_cat, contains, item)`
	- `(leaf_cat, child_of, parent_cat)` và `(parent_cat, parent_of, leaf_cat)`
- Backbone `CategoryEnhancedGNN` sử dụng `HeteroConv` với `SAGEConv` hoặc `GATConv`, kèm residual connection và `LayerNorm` giữa các lớp.
- Readout theo kiểu SR-GNN với soft-attention để tạo biểu diễn session.
- Dự đoán item kế tiếp bằng phép nhân với ma trận embedding toàn bộ item.

## Cài đặt môi trường

```bash
pip install -r requirements.txt
```

Lưu ý: `torch_geometric` đôi khi cần cài đúng phiên bản theo hệ điều hành và phiên bản PyTorch. Nếu môi trường của bạn chưa có PyG, hãy cài bản tương thích trước khi chạy.

## Chạy ví dụ toy

```bash
python -m experiments.run_catsa_module1
```

Script sẽ tự dùng CUDA nếu môi trường có GPU khả dụng. Nếu muốn ép thiết bị chạy, dùng `--device cpu` hoặc `--device cuda`.

Mặc định script chạy `--version 1` (giữ nguyên hành vi cũ).
Nếu chạy `--version 2`, các item không có danh mục sẽ được gán vào danh mục ảo `UNK_CAT=0` và được nối cạnh membership `(item, belongs_to, leaf_cat)` tới node danh mục ảo này.

## Bước 1: Tiền xử lý dữ liệu RetailRocket

Khi bạn thêm dữ liệu vào thư mục `DATA`, hãy chạy tiền xử lý trước:

```bash
python -m preprocessing.retailrocket_preprocess \
	--data-root DATA \
	--output-path outputs/processed/retailrocket_module1.json \
	--allowed-events view \
	--min-item-clicks 5
```

Script hiện hỗ trợ cả hai cách bố trí file:

- `DATA/events.csv`, `DATA/category_tree.csv`, ...
- `DATA/retailrocket/events.csv`, `DATA/retailrocket/category_tree.csv`, ...

Nếu muốn lưu log ra thư mục riêng từ lệnh bên ngoài:

```bash
python -m preprocessing.retailrocket_preprocess \
	--data-root DATA \
	--output-path outputs/processed/retailrocket_module1.json \
	--allowed-events view \
	--min-item-clicks 5 \
	--log-dir outputs/logs \
	--log-file-name preprocess.log
```

Mặc định hiện tại của preprocess:

- Chỉ dùng sự kiện click (`view`).
- Loại bỏ item hiếm có số click toàn cục `<= 5`.
- Sau khi bỏ item hiếm, session nào còn dưới `--min-session-length` sẽ bị loại.

Artifact đầu ra sẽ chứa:

- `session_sequences`
- `splits.train`
- `splits.valid`
- `splits.test`
- `item2leaf_dict`
- `leaf2parent_dict`

Việc chia split hiện dùng cách chia theo thứ tự thời gian kết thúc session, đây là cách phù hợp hơn cho recommender so với random split.

## Bước 2: Chạy thực nghiệm từ dữ liệu đã tiền xử lý

Sau khi đã có artifact tiền xử lý, chạy huấn luyện như sau:

```bash
python -m experiments.run_catsa_module1 \
	--processed-path outputs/processed/retailrocket_module1.json \
	--split train
```

Bạn vẫn có thể chạy trực tiếp từ dữ liệu thô nếu cần debug nhanh:

```bash
python -m experiments.run_catsa_module1 --data-root DATA
```

Các file được script sử dụng gồm:

- `events.csv`
- `category_tree.csv`
- `item_properties_part1.csv`
- `item_properties_part2.csv`

## Cấu hình vị trí lưu log từ bên ngoài

Cả script tiền xử lý và script thực nghiệm đều hỗ trợ truyền vị trí lưu log bằng tham số dòng lệnh. Đây là cách phù hợp khi bạn gọi từ shell script, VS Code task, hoặc job runner bên ngoài.

Ví dụ:

```bash
python -m experiments.run_catsa_module1 \
	--processed-path outputs/processed/retailrocket_module1.json \
	--log-dir outputs/logs \
	--log-file-name run1.log
```

Ý nghĩa:

- `--log-dir`: Thư mục chứa file log. Script sẽ tự tạo thư mục nếu chưa tồn tại.
- `--log-file-name`: Tên file log bên trong thư mục log.
- `--split`: Chọn split cần dùng trong artifact đã tiền xử lý.

Nếu không truyền `--log-dir`, log chỉ được in ra màn hình.

## Một số tham số quan trọng khi huấn luyện

```bash
python -m experiments.run_catsa_module1 \
	--processed-path outputs/processed/retailrocket_module1.json \
	--hidden-dim 128 \
	--num-layers 2 \
	--batch-size 32 \
	--epochs 10 \
	--lr 1e-3 \
	--conv-type sage \
	--device auto \
	--log-dir outputs/logs
```

Chạy phiên bản 2 với danh mục ảo:

```bash
python -m experiments.run_catsa_module1 \
	--processed-path outputs/processed/retailrocket_module1.json \
	--hidden-dim 128 \
	--num-layers 2 \
	--batch-size 32 \
	--epochs 10 \
	--lr 1e-3 \
	--conv-type sage \
	--device auto \
	--version 2 \
	--log-dir outputs/logs
```

Mặc định script sẽ log `Recall@20`, `MRR@20` theo epoch.
Khi train với `--split train` từ artifact đã tiền xử lý, split đánh giá mặc định là `valid`.
`--epochs` là số epoch tối đa; script sẽ dừng sớm nếu `MRR@K` không cải thiện trong 5 epoch liên tiếp (Early Stopping).

Trong đó:

- `--hidden-dim`: Kích thước embedding và hidden state.
- `--num-layers`: Số lớp message passing trên heterogeneous graph.
- `--batch-size`: Batch size cho `torch_geometric.loader.DataLoader`.
- `--epochs`: Số epoch huấn luyện tối đa (có thể dừng sớm theo Early Stopping).
- `--lr`: Learning rate cho Adam.
- `--conv-type`: Chọn `sage` hoặc `gat`.
- `--device`: Chọn `auto`, `cpu`, hoặc `cuda`. Mặc định `auto` sẽ ưu tiên CUDA nếu có.
- `--version`: Chọn `1` hoặc `2`. `2` sẽ gán item thiếu danh mục vào `UNK_CAT=0`.
- `--metric-k`: Giá trị K dùng để tính `Recall@K` và `MRR@K`. Mặc định `20`.
- `--eval-split`: Chọn split đánh giá theo epoch: `auto`, `none`, `train`, `valid`, `test`. Mặc định `auto`.
- `--early-stop-patience`: Patience cho Early Stopping. Mặc định `5`, đặt `0` để tắt.
- `--early-stop-min-delta`: Ngưỡng cải thiện tối thiểu của `MRR@K` (đơn vị: điểm %). Mặc định `1e-4`.
- `--log-every-steps`: Tần suất log tiến độ trong mỗi epoch (đơn vị mini-batch). Mặc định `500`.

## Gợi ý tổ chức đầu ra thí nghiệm

Bạn có thể tổ chức thư mục như sau để tiện quản lý:

```text
outputs/
	logs/
		run1.log
		run2.log
```

Sau này nếu cần, tôi có thể bổ sung tiếp:

- Lưu checkpoint model theo từng epoch.
- Ghi thêm file cấu hình thí nghiệm dạng JSON.
- Tách train/validation/test cho RetailRocket.
- Tính metric `Recall@K`, `MRR@K`, `NDCG@K`.