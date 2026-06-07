# COTREC2

Chạy **COTREC gốc** trên **cùng dữ liệu CatSA** (`Data/CatSA/retailrocket/retailrocket.json`) để so sánh metric với CatSA — kiểm tra chênh lệch do model hay do preprocess/split.

## Bước 1 — Export CatSA → format COTREC

```bash
cd nhom2/COTREC2
python export_data.py --dataset retailrocket --split-mode catsa
```

Output: `Data/COTREC2/retailrocket/`

| File | Nội dung |
|------|----------|
| `train.txt` | prefix/target từ **CatSA train split** |
| `test.txt` | prefix/target từ **CatSA test split** |
| `all_train_seq.txt` | session đầy đủ (train+valid) cho global graph |
| `meta.json` | `num_items`, số session/sample |

`--split-mode full`: train COTREC trên train+valid (gần COTREC paper hơn).

## Bước 2 — Train COTREC

```bash
python -u main.py --dataset retailrocket --epoch 30 \
  --log-dir /content/drive/MyDrive/NCS_NEW/LOG/COTREC2 \
  --log-mins-dir /content/drive/MyDrive/NCS_NEW/LOGMins/COTREC2
```

Colab + nohup (full eval, không `--smoke`):

```bash
mkdir -p /content/drive/MyDrive/NCS_NEW/LOG/COTREC2/retailrocket
mkdir -p /content/drive/MyDrive/NCS_NEW/LOGMins/COTREC2/retailrocket

nohup python -u main.py \
  --dataset retailrocket \
  --epoch 30 \
  --log-dir /content/drive/MyDrive/NCS_NEW/LOG/COTREC2 \
  --log-mins-dir /content/drive/MyDrive/NCS_NEW/LOGMins/COTREC2 \
  > /content/drive/MyDrive/NCS_NEW/LOG/COTREC2/retailrocket/nohup-$(date +%d-%m-%Y).log 2>&1 &

echo $!
```

Smoke test nhanh:

```bash
python main.py --dataset retailrocket --smoke
```

Log:

- Quá trình: `Log/COTREC2/retailrocket/DD-MM-YYYY.log`
- Kết quả: `LogMins/COTREC2/retailrocket/DD-MM-YYYY.log`

## So sánh với CatSA

| | CatSA | COTREC2 |
|---|-------|---------|
| Train data | `splits.train` | `splits.train` (mode `catsa`) |
| Test data | `splits.test` | `splits.test` |
| Metric | HR@20, MRR@20 | Recall@20 (=HR), MRR@20 |
| `n_node` | `num_items` từ artifact | cùng giá trị từ `meta.json` |

Nếu COTREC2 ~55% mà CatSA ~40% → bottleneck ở **model CatSA**.  
Nếu cả hai ~40% → kiểm tra **preprocess / metric / indexing**.

Hyperparameter RetailRocket (README COTREC): `beta=0.01`, `lam=0.005`, `eps=0.2`.
